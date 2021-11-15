import jax
import jax.numpy as jnp
import random
from tqdm import tqdm

from jaxtorch import Module, PRNG, Context
from jaxtorch import nn
from einops import rearrange

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = jnp.tril(jnp.ones([config.block_size, config.block_size]))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, cx, x):
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(cx, x).rearrange('b t (h c) -> b h t c', h=self.n_head)
        q = self.query(cx, x).rearrange('b t (h c) -> b h t c', h=self.n_head)
        v = self.value(cx, x).rearrange('b t (h c) -> b h t c', h=self.n_head)

        scale = 1.0 / jnp.sqrt(C // self.n_head)
        att = (q @ k.rearrange('b h t c -> b h c t')) * scale # b h t t
        att = jnp.where(self.mask[None,None,:T,:T]==1, att, float('-inf'))
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_drop(cx, att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.rearrange('b h t c -> b t h c').reshape((B, T, C))

        # output projection
        y = self.resid_drop(cx, self.proj(cx, y))
        return y

    def exec_x(self, cx, x, k1, v1):
        B, T2, C = x.shape
        _, _, T1, _ = k1.shape

        HS = C // self.n_head
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k2 = self.key(cx, x).rearrange('b t (h c) -> b h t c', h=self.n_head)
        q = self.query(cx, x).rearrange('b t (h c) -> b h t c', h=self.n_head)
        v2 = self.value(cx, x).rearrange('b t (h c) -> b h t c', h=self.n_head)

        scale = 1.0 / jnp.sqrt(HS)
        att2 = (q @ k2.rearrange('b h t c -> b h c t')) * scale # (B, nh, T2, T2)
        att2 = jnp.where(self.mask[None,None,:T2,:T2]==1, att2, float('-inf'))
        att1 = (q @ k1.rearrange('b h t c -> b h c t')) * scale # (B, nh, T2, T1)
        att = jnp.concatenate([att1, att2], axis=-1) # (B, nh, T2, T)
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_drop(cx, att)
        att1 = att[:, :, :, :T1] # (B, nh, T2, T1)
        att2 = att[:, :, :, T1:] # (B, nh, T2, T2)
        y1 = att1 @ v1 # (B, nh, T2, hs)
        y2 = att2 @ v2 # (B, nh, T2, hs)
        y = y1 + y2
        y = y.rearrange('b h t c -> b t h c').reshape(B, T2, C)

        # output projection
        y = self.resid_drop(cx, self.proj(cx, y))
        return (y, k2, v2)

class Block(Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, cx, x):
        x = x + self.attn(cx, self.ln1(cx, x))
        x = x + self.mlp(cx, self.ln2(cx, x))
        return x

    def exec_x(self, cx, x, k1, v1):
        (y, k2, v2) = self.attn.exec_x(cx, self.ln1(cx, x), k1, v1)
        x = x + y
        x = x + self.mlp(cx, self.ln2(cx, x))
        return (x, k2, v2)

class History(object):
    def __init__(self, n_layer, b, n_head, seq_len, head_size):
        self.ks = [jnp.zeros([b, n_head, seq_len, head_size]) for _ in range(n_layer)]
        self.vs = [jnp.zeros([b, n_head, seq_len, head_size]) for _ in range(n_layer)]
        self.h = 0
    def to_(self, *args, **kwargs):
        self.ks = [k.to(*args, **kwargs) for k in self.ks]
        self.vs = [v.to(*args, **kwargs) for v in self.vs]
        return self

class GPT(nn.Module):
    """A series of transformer blocks."""
    def __init__(self, config):
        super().__init__()

        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.block_size = config.block_size

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer

        assert self.n_embd % self.n_head == 0

    def forward(self, cx, x):
        x = self.drop(cx, x)
        x = self.blocks(cx, x)
        x = self.ln_f(cx, x)
        return x

    def history(self, batch_size):
        return History(self.n_layer, batch_size, self.n_head, self.block_size, self.n_embd // self.n_head)

    def exec_x(self, cx, x, history):
        #x : [b, len, n_embd]
        [b, l, _] = x.shape
        x = self.drop(cx, x)
        h = history.h
        for (i, block) in enumerate(self.blocks.modules):
            k1 = history.ks[i][:, :, :history.h, :]
            v1 = history.vs[i][:, :, :history.h, :]
            (x, k2, v2) = block.exec_x(cx, x, k1, v1)
            history.ks[i] = jax.ops.index_update(history.ks[i], jax.ops.index[:, :, h : h+l, :], k2)
            history.vs[i] = jax.ops.index_update(history.vs[i], jax.ops.index[:, :, h : h+l, :], v2)
        history.h = h + l
        return self.ln_f(cx, x)


class GPTLM(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.gpt = GPT(config)
        self.seq_len = config.block_size
        self.n_vocab = config.vocab_size

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size+1, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.class_emb = nn.Linear(512, config.n_embd)

        # decoder head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def self_init_weights(self, cx):
        super().self_init_weights(cx)
        for module in self.gen_postorder_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                cx[module.weight] = 0.02 * cx.random.normal(module.weight.shape)
            if isinstance(module, nn.Linear) and module.bias is not None:
                cx[module.bias] = jnp.zeros_like(cx[module.bias])

    def forward(self, cx, idx, add_emb=None):
        [b, t] = idx.shape
        assert t <= self.seq_len, f"Cannot forward more than {self.seq_len} tokens."
        input_idx = jnp.concatenate([jnp.full([b, 1], self.n_vocab, dtype=jnp.int32),
                                     idx[:, :-1]], axis=1)
        # forward the GPT model
        token_embeddings = self.tok_emb(cx, input_idx) # each index maps to a (learnable) vector
        position_embeddings = cx[self.pos_emb.weight][:t, :] # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        if add_emb is not None:
            add_emb = self.class_emb(cx, add_emb.reshape(b, 1, 512))
            x += add_emb
        x = self.gpt(cx, x)
        logits = self.head(cx, x)
        return logits

    def loss(self, cx, idx, add_emb=None):
        [b, t] = idx.shape
        logits = self.forward(cx, idx, add_emb)
        def ix(logits, idx):
            return logits[idx]
        ix = jax.vmap(jax.vmap(ix))
        lps = jax.nn.log_softmax(logits)
        return -ix(lps, idx).mean()

    # TODO: it would be nice to jit this for efficiency
    def generate(self, cx, idx, add_emb=None, temp=1.0):
        """Fills idx with generated tokens.

        If any tokens are already >=0 in idx they will be kept for
        conditional sampling.

        """
        B, T = idx.shape
        assert T <= self.seq_len, f"Cannot generate more than {self.seq_len} tokens."

        if add_emb is not None:
            add_emb = self.class_emb(cx, add_emb.reshape(B, 1, 512))

        # forward the GPT model
        history = self.gpt.history(B)
        for i in tqdm(range(T)):
            if i == 0:
                input_idx = jnp.full([B, 1], self.n_vocab, dtype=jnp.int32)
            else:
                input_idx = idx[:, i-1].reshape(B, 1)
            token_embeddings = self.tok_emb(cx, input_idx)
            position_embeddings = cx[self.pos_emb.weight][i:i+1, :]
            x = token_embeddings + position_embeddings
            if add_emb is not None:
                x += add_emb
            x = self.gpt.exec_x(cx, x, history)
            logits = self.head(cx, x) / temp
            toks = cx.random.categorical(logits=logits, axis=-1)
            idx = jax.ops.index_update(idx, jax.ops.index[:, i], jnp.where(idx[:,i] < 0, toks[:,0], idx[:,i]))

        return idx

    # def optimizer(self, train_config, scaled=False):
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
    #             if pn.endswith('bias'):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #                 p.scale = 1.0
    #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #                 p.scale = np.sqrt(2.0 / (p.shape[0] + p.shape[1]))
    #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)
    #                 p.scale = 1.0

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     no_decay.add('pos_emb')
    #     self.pos_emb.scale = 1.0

    #     # validate that we considered every parameter
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     missing_params = param_dict.keys() - union_params
    #     assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    #     assert len(missing_params) == 0, f"parameters {missing_params} were not separated into either decay/no_decay set!"


    #     # param_dict = {pn:p for (pn,p) in param_dict.items() if 'head' in pn}
    #     decay = decay & param_dict.keys()
    #     no_decay = no_decay & param_dict.keys()
    #     # create the pytorch optimizer object
    #     optim_groups = [
    #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
    #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    #     ]
    #     if scaled:
    #         optim_groups = [{'params': [p],
    #                          'weight_decay': train_config.weight_decay if pn in decay else 0.0,
    #                          'lr': train_config.learning_rate * p.scale,
    #                          '_scale': p.scale}
    #                          for (pn, p) in param_dict.items()]

    #     Opt = torch.optim.AdamW if train_config.weight_decay > 0 else torch.optim.Adam

    #     optimizer = Opt(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
    #     def update_lr(optimizer, lr):
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr * param_group['_scale']
    #     optimizer.update_lr = functools.partial(update_lr, optimizer)
    #     return optimizer
