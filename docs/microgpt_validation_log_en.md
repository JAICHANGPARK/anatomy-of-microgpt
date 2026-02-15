# microgpt — verification log (claims cross-checked against primary sources)

This log is meant to support “tutorial correctness.” It lists the main technical claims used in the Medium article, the evidence used to confirm them, and what was checked.

> Scope: algorithmic claims about (1) Transformer decoder masking/causality, (2) GPT‑2 architectural notes (pre-norm + final norm; BOS/EOS config; activation), (3) RMSNorm definition, (4) Adam definition, (5) standard GPT‑2 reference implementation details (position embeddings, causal mask, final ln_f, dropout, Conv1D bias).

---

## 1) microgpt’s stated goal and included components

- **Claim:** microgpt is a single-file, dependency-free script containing dataset, tokenizer, autograd, GPT-2-like architecture, Adam, training loop, inference loop; “everything else is efficiency.”
- **Evidence:** Karpathy’s microgpt blog post (Feb 12, 2026).

## 2) Decoder causality / masking in the Transformer

- **Claim:** Decoder self-attention must be causal: position *t* cannot attend to positions > *t*.
- **Evidence:** *Attention Is All You Need* (Vaswani et al., 2017) describes masking “subsequent positions” in decoder self-attention.
- **What was checked:** The paper’s decoder description; specifically that the mask enforces unidirectional conditioning.

## 3) microgpt’s causality mechanism (implementation-level)

- **Claim:** microgpt enforces causality without an explicit triangular mask by only attending over the “past” KV cache built sequentially.
- **Evidence:** direct inspection of the provided microgpt code.
- **What was checked:** At position `pos_id`, keys/values are appended and attention iterates over `keys[li]` which only contains previously appended entries.

## 4) GPT‑2 pre-norm shift and additional final normalization

- **Claim:** GPT‑2 moved LayerNorm to the input of each sub-block (pre-norm) and adds an additional LayerNorm after the final self-attention block (often called `ln_f`).
- **Evidence:** OpenAI GPT‑2 report (*Language Models are Unsupervised Multitask Learners*, 2019) + Hugging Face GPT‑2 reference implementation (source view).
- **What was checked:** The GPT‑2 report’s architectural notes; the HF code’s presence of `self.ln_f = nn.LayerNorm(...)` and application `hidden_states = self.ln_f(hidden_states)`.

## 5) Causal mask in a standard GPT‑2 reference implementation

- **Claim:** Reference GPT‑2 implementations typically construct a lower-triangular causal mask and apply it to attention scores.
- **Evidence:** Hugging Face GPT‑2 source view (`GPT2Attention` registers `bias` as `torch.tril(...)` and uses it to mask attention weights).
- **What was checked:** `register_buffer("bias", torch.tril(...))` and the masking logic in `_attn(...)`.

## 6) Learned absolute position embeddings in GPT‑2

- **Claim:** GPT‑2 uses learned absolute position embeddings (`wpe`) and combines them by addition with token embeddings.
- **Evidence:** Hugging Face GPT‑2 docs and HF source view.
- **What was checked:** `self.wpe = nn.Embedding(config.max_position_embeddings, ...)` and `hidden_states = inputs_embeds + position_embeds`.

## 7) BOS/EOS convention in GPT‑2 configs

- **Claim:** GPT‑2 commonly uses the same token ID for `bos_token_id` and `eos_token_id`.
- **Evidence:** Hugging Face GPT‑2 docs (configuration snippet).
- **What was checked:** Config shows `bos_token_id == eos_token_id` in the documentation.

## 8) Activation function in GPT‑2 configs

- **Claim:** GPT‑2 commonly uses a GELU-family activation (often `gelu_new` in many HF GPT‑2 configs).
- **Evidence:** Hugging Face GPT‑2 docs (configuration snippet).
- **What was checked:** `activation_function = 'gelu_new'`.

## 9) Adam algorithm

- **Claim:** The optimizer in microgpt is Adam with first/second moment estimates and bias correction.
- **Evidence:** Kingma & Ba (2014), arXiv:1412.6980.
- **What was checked:** Update equations in the paper; correspondence to `m_hat`, `v_hat`, and the bias-correction denominators.

## 10) RMSNorm definition (and microgpt’s simplification)

- **Claim:** RMSNorm normalizes by RMS (mean of squared activations) and can omit re-centering; microgpt uses an RMS-based normalization without learned affine parameters.
- **Evidence:** Zhang & Sennrich (2019), arXiv:1910.07467.
- **What was checked:** RMSNorm definition and motivation in the paper; microgpt’s `rmsnorm` matches the RMS scaling form but omits the common learned gain.

## 11) Bias terms and dropout in “standard” GPT‑2 references

- **Claim:** Reference GPT‑2 implementations include bias parameters (e.g., Conv1D) and dropout modules.
- **Evidence:** HF GPT‑2 source view + HF modeling utilities (Conv1D definition) + docs.
- **What was checked:** `Conv1D` contains an explicit bias parameter; GPT‑2 uses dropout modules in attention/residual/embedding paths.

---

## Notes on limits

- microgpt’s educational simplifications mean it is not a drop-in replica of production GPT‑2. The verification above aims to ensure (a) the post’s statements about microgpt match the code, and (b) statements about canonical Transformer/GPT‑2/Adam/RMSNorm match primary sources.
