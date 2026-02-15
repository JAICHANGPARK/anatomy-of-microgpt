# microgpt Appendix: Line-numbered code + line-by-line commentary

This appendix reproduces the user-provided *microgpt* script **verbatim** as (A) line-numbered code and (B) commentary keyed to the same line numbers.

---

## How to use this appendix

- The **Transformer block** (a GPT decoder block) is inside `gpt()` within `for li in range(n_layer):` (roughly **L114–L142**).
- The **attention** sublayer (QKV, softmax, weighted sum) is concentrated in **L118–L135**.
- The **MLP/FFN** sublayer (FC1 → ReLU → FC2) is **L136–L141**.
- The **training loop** is **L151–L184**, and **inference/sampling** is **L186–L200**.

---

## Appendix A. Line-numbered code (L001–L200)

```python
001: """
002: The most atomic way to train and inference a GPT in pure, dependency-free Python.
003: This file is the complete algorithm.
004: Everything else is just efficiency.
005: 
006: @karpathy
007: """
008: 
009: import os       # os.path.exists
010: import math     # math.log, math.exp
011: import random   # random.seed, random.choices, random.gauss, random.shuffle
012: random.seed(42) # Let there be order among chaos
013: 
014: # Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
015: if not os.path.exists('input.txt'):
016:     import urllib.request
017:     names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
018:     urllib.request.urlretrieve(names_url, 'input.txt')
019: docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
020: random.shuffle(docs)
021: print(f"num docs: {len(docs)}")
022: 
023: # Let there be a Tokenizer to translate strings to discrete symbols and back
024: uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
025: BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
026: vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
027: print(f"vocab size: {vocab_size}")
028: 
029: # Let there be Autograd, to recursively apply the chain rule through a computation graph
030: class Value:
031:     __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage
032: 
033:     def __init__(self, data, children=(), local_grads=()):
034:         self.data = data                # scalar value of this node calculated during forward pass
035:         self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
036:         self._children = children       # children of this node in the computation graph
037:         self._local_grads = local_grads # local derivative of this node w.r.t. its children
038: 
039:     def __add__(self, other):
040:         other = other if isinstance(other, Value) else Value(other)
041:         return Value(self.data + other.data, (self, other), (1, 1))
042: 
043:     def __mul__(self, other):
044:         other = other if isinstance(other, Value) else Value(other)
045:         return Value(self.data * other.data, (self, other), (other.data, self.data))
046: 
047:     def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
048:     def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
049:     def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
050:     def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
051:     def __neg__(self): return self * -1
052:     def __radd__(self, other): return self + other
053:     def __sub__(self, other): return self + (-other)
054:     def __rsub__(self, other): return other + (-self)
055:     def __rmul__(self, other): return self * other
056:     def __truediv__(self, other): return self * other**-1
057:     def __rtruediv__(self, other): return other * self**-1
058: 
059:     def backward(self):
060:         topo = []
061:         visited = set()
062:         def build_topo(v):
063:             if v not in visited:
064:                 visited.add(v)
065:                 for child in v._children:
066:                     build_topo(child)
067:                 topo.append(v)
068:         build_topo(self)
069:         self.grad = 1
070:         for v in reversed(topo):
071:             for child, local_grad in zip(v._children, v._local_grads):
072:                 child.grad += local_grad * v.grad
073: 
074: # Initialize the parameters, to store the knowledge of the model.
075: n_embd = 16     # embedding dimension
076: n_head = 4      # number of attention heads
077: n_layer = 1     # number of layers
078: block_size = 16 # maximum sequence length
079: head_dim = n_embd // n_head # dimension of each head
080: matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
081: state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
082: for i in range(n_layer):
083:     state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
084:     state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
085:     state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
086:     state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
087:     state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
088:     state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
089: params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
090: print(f"num params: {len(params)}")
091: 
092: # Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
093: # Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
094: def linear(x, w):
095:     return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
096: 
097: def softmax(logits):
098:     max_val = max(val.data for val in logits)
099:     exps = [(val - max_val).exp() for val in logits]
100:     total = sum(exps)
101:     return [e / total for e in exps]
102: 
103: def rmsnorm(x):
104:     ms = sum(xi * xi for xi in x) / len(x)
105:     scale = (ms + 1e-5) ** -0.5
106:     return [xi * scale for xi in x]
107: 
108: def gpt(token_id, pos_id, keys, values):
109:     tok_emb = state_dict['wte'][token_id] # token embedding
110:     pos_emb = state_dict['wpe'][pos_id] # position embedding
111:     x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
112:     x = rmsnorm(x)
113: 
114:     for li in range(n_layer):
115:         # 1) Multi-head attention block
116:         x_residual = x
117:         x = rmsnorm(x)
118:         q = linear(x, state_dict[f'layer{li}.attn_wq'])
119:         k = linear(x, state_dict[f'layer{li}.attn_wk'])
120:         v = linear(x, state_dict[f'layer{li}.attn_wv'])
121:         keys[li].append(k)
122:         values[li].append(v)
123:         x_attn = []
124:         for h in range(n_head):
125:             hs = h * head_dim
126:             q_h = q[hs:hs+head_dim]
127:             k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
128:             v_h = [vi[hs:hs+head_dim] for vi in values[li]]
129:             attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
130:             attn_weights = softmax(attn_logits)
131:             head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
132:             x_attn.extend(head_out)
133:         x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
134:         x = [a + b for a, b in zip(x, x_residual)]
135:         # 2) MLP block
136:         x_residual = x
137:         x = rmsnorm(x)
138:         x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
139:         x = [xi.relu() for xi in x]
140:         x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
141:         x = [a + b for a, b in zip(x, x_residual)]
142: 
143:     logits = linear(x, state_dict['lm_head'])
144:     return logits
145: 
146: # Let there be Adam, the blessed optimizer and its buffers
147: learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
148: m = [0.0] * len(params) # first moment buffer
149: v = [0.0] * len(params) # second moment buffer
150: 
151: # Repeat in sequence
152: num_steps = 1000 # number of training steps
153: for step in range(num_steps):
154: 
155:     # Take single document, tokenize it, surround it with BOS special token on both sides
156:     doc = docs[step % len(docs)]
157:     tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
158:     n = min(block_size, len(tokens) - 1)
159: 
160:     # Forward the token sequence through the model, building up the computation graph all the way to the loss.
161:     keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
162:     losses = []
163:     for pos_id in range(n):
164:         token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
165:         logits = gpt(token_id, pos_id, keys, values)
166:         probs = softmax(logits)
167:         loss_t = -probs[target_id].log()
168:         losses.append(loss_t)
169:     loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.
170: 
171:     # Backward the loss, calculating the gradients with respect to all model parameters.
172:     loss.backward()
173: 
174:     # Adam optimizer update: update the model parameters based on the corresponding gradients.
175:     lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
176:     for i, p in enumerate(params):
177:         m[i] = beta1 * m[i] + (1 - beta1) * p.grad
178:         v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
179:         m_hat = m[i] / (1 - beta1 ** (step + 1))
180:         v_hat = v[i] / (1 - beta2 ** (step + 1))
181:         p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
182:         p.grad = 0
183: 
184:     print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
185: 
186: # Inference: may the model babble back to us
187: temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
188: print("\n--- inference (new, hallucinated names) ---")
189: for sample_idx in range(20):
190:     keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
191:     token_id = BOS
192:     sample = []
193:     for pos_id in range(block_size):
194:         logits = gpt(token_id, pos_id, keys, values)
195:         probs = softmax([l / temperature for l in logits])
196:         token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
197:         if token_id == BOS:
198:             break
199:         sample.append(uchars[token_id])
200:     print(f"sample {sample_idx+1:2d}: {''.join(sample)}") 
```

---

## Appendix B. Line-by-line commentary (L001–L200)

- **L001**: Start of the module docstring.
- **L002**: Declares the project goal: an 'atomic' (minimal) GPT training + inference script in pure, dependency-free Python.
- **L003**: States the claim that the script contains the complete algorithmic content required for training/inference.
- **L004**: Emphasizes the separation between algorithm (here) and efficiency engineering (vectorization, kernels, distributed systems, etc.).
- **L005**: Blank line inside the docstring for readability.
- **L006**: Attribution/handle.
- **L007**: End of the module docstring.
- **L008**: Blank line used to separate sections.
- **L009**: Import `os` (standard library); used for file existence checks (`os.path.exists`).
- **L010**: Import `math` (standard library); used for scalar `log`/`exp` in the autograd Value operations.
- **L011**: Import `random` (standard library); used for seeding, shuffling, Gaussian initialization, and sampling.
- **L012**: Fix the RNG seed for reproducibility.
- **L013**: Blank line separator.
- **L014**: Comment: introduce the dataset abstraction `docs: list[str]` (a list of documents).
- **L015**: If the dataset file does not exist locally, download it.
- **L016**: Import `urllib.request` only when needed (lazy import).
- **L017**: URL of the example dataset (names) used by the script.
- **L018**: Download the dataset and save it as `input.txt`.
- **L019**: Read `input.txt`, split by lines, strip whitespace, and build `docs` (dropping empty lines).
- **L020**: Shuffle documents to reduce order effects during training.
- **L021**: Print the dataset size for sanity checking.
- **L022**: Blank line separator.
- **L023**: Comment: introduce the tokenizer that maps between strings and integer token ids.
- **L024**: Collect the set of unique characters appearing in the dataset and sort them; indices become token ids 0..n-1.
- **L025**: Define a special BOS token id as the next id after the last character token.
- **L026**: Set `vocab_size` = (#unique chars) + 1 (for BOS).
- **L027**: Print vocabulary size for sanity checking.
- **L028**: Blank line separator.
- **L029**: Comment: introduce a tiny autograd engine implementing reverse-mode AD (backprop) via the chain rule.
- **L030**: Define the scalar Value node type used to build a computation graph during the forward pass.
- **L031**: Use `__slots__` to reduce per-object memory overhead (important when many Value objects are created).
- **L032**: Blank line separator.
- **L033**: Constructor: initialize a Value with data and optional graph metadata (children + local gradients).
- **L034**: Store the forward value (a Python float).
- **L035**: Initialize the accumulated gradient dL/d(this) to 0 (will be filled by `backward()`).
- **L036**: Store references to upstream nodes (children in the computation graph).
- **L037**: Store local partial derivatives of this node with respect to each child (for chain rule propagation).
- **L038**: Blank line separator.
- **L039**: Define addition for Value nodes; creates a new Value and records local gradients (1, 1).
- **L040**: Coerce Python scalars to Value so mixed arithmetic works (`Value + float`).
- **L041**: Return a new Value representing the sum; record graph edges and local derivatives.
- **L042**: Blank line separator.
- **L043**: Define multiplication for Value nodes; creates a new Value and records local gradients (∂/∂a=b, ∂/∂b=a).
- **L044**: Coerce Python scalars to Value for mixed arithmetic.
- **L045**: Return a new Value representing the product; record graph edges and local derivatives.
- **L046**: Blank line separator.
- **L047**: Power operation: z = x**n with local derivative n*x**(n-1).
- **L048**: Natural log with local derivative 1/x.
- **L049**: Exponential with local derivative exp(x).
- **L050**: ReLU activation: max(0, x) with derivative 1 if x>0 else 0.
- **L051**: Unary negation implemented as multiplication by -1.
- **L052**: Right-add: allow `float + Value` by redirecting to `__add__`.
- **L053**: Subtraction implemented as addition with negation.
- **L054**: Right-subtraction: allow `float - Value`.
- **L055**: Right-multiplication: allow `float * Value`.
- **L056**: True division implemented via multiplication by reciprocal power (x / y = x * y**-1).
- **L057**: Right-division: allow `float / Value`.
- **L058**: Blank line separator.
- **L059**: Begin reverse-mode autodiff: compute gradients for all nodes reachable from this Value (typically the loss).
- **L060**: Initialize a list to hold nodes in topological order.
- **L061**: Track visited nodes to avoid revisiting in DFS.
- **L062**: Define a DFS that builds a topological ordering of the computation graph.
- **L063**: If node has not been visited, mark it visited.
- **L064**: Record the visit to prevent cycles/repeats.
- **L065**: Recursively process all children first (post-order).
- **L066**: DFS continues down the graph.
- **L067**: Append the node after its children → yields a valid topological ordering.
- **L068**: Build the topological ordering starting from `self` (the output node).
- **L069**: Seed the backward pass with d(self)/d(self)=1 (i.e., dL/dL = 1 at the loss).
- **L070**: Traverse nodes in reverse topological order (from outputs back to inputs).
- **L071**: For each edge (v → child), propagate gradients using the stored local derivative.
- **L072**: Accumulate child.grad += (∂v/∂child) * v.grad (note the `+=` for shared subgraphs).
- **L073**: Blank line separator.
- **L074**: Comment: initialize model parameters (the trainable weights).
- **L075**: Set embedding/model width `n_embd` (dimension of the residual stream).
- **L076**: Set number of attention heads.
- **L077**: Set number of Transformer layers (blocks).
- **L078**: Set maximum context length (`block_size`).
- **L079**: Compute per-head dimension (`head_dim = n_embd // n_head`).
- **L080**: Define a helper to initialize a weight matrix (nout×nin) with small Gaussian values wrapped as Value nodes.
- **L081**: Create `state_dict` with token embeddings (wte), position embeddings (wpe), and output projection (lm_head).
- **L082**: Loop over layers to create per-layer attention and MLP weight matrices.
- **L083**: Initialize Wq for the layer (n_embd×n_embd).
- **L084**: Initialize Wk for the layer.
- **L085**: Initialize Wv for the layer.
- **L086**: Initialize Wo for the layer (output projection of concatenated heads).
- **L087**: Initialize first MLP matrix (expand: n_embd → 4*n_embd).
- **L088**: Initialize second MLP matrix (project back: 4*n_embd → n_embd).
- **L089**: Flatten all matrices into a single parameter list for the optimizer.
- **L090**: Print the total parameter count for sanity checking.
- **L091**: Blank line separator.
- **L092**: Comment: define the model as a pure function mapping (token, position, cached KV) → logits over next token.
- **L093**: Comment: notes deviations from GPT-2 for simplicity (RMSNorm instead of LayerNorm, no bias, ReLU instead of GeLU).
- **L094**: Define a linear layer: matrix-vector multiply y = W x (no bias).
- **L095**: Compute one dot product per output row; returns a list[Value] of length nout.
- **L096**: Blank line separator.
- **L097**: Define softmax to convert logits to a probability distribution.
- **L098**: Compute max(logits) in `.data` space for numerical stability (stable softmax).
- **L099**: Compute exp(logit - max) for each logit.
- **L100**: Sum exponentials to obtain the normalization constant.
- **L101**: Normalize to probabilities (each in (0,1) and sum to 1).
- **L102**: Blank line separator.
- **L103**: Define RMSNorm (Root-Mean-Square normalization).
- **L104**: Compute mean square of the vector entries.
- **L105**: Compute scaling factor 1/sqrt(ms + eps).
- **L106**: Scale each component of x; this variant has no learned affine gain/bias.
- **L107**: Blank line separator.
- **L108**: Define the GPT forward step: process one token at one position, using KV cache for prior context, and return logits.
- **L109**: Lookup the token embedding vector for `token_id`.
- **L110**: Lookup the position embedding vector for `pos_id`.
- **L111**: Combine token + position embeddings by elementwise addition to form the initial residual stream vector x.
- **L112**: Apply RMSNorm to stabilize activation scale.
- **L113**: Blank line separator.
- **L114**: Loop over Transformer layers (blocks).
- **L115**: Comment: start the multi-head self-attention sublayer.
- **L116**: Save residual connection input for the attention sublayer.
- **L117**: Pre-norm: apply RMSNorm before computing QKV (pre-norm block style).
- **L118**: Project x to queries: q = Wq x.
- **L119**: Project x to keys: k = Wk x.
- **L120**: Project x to values: v = Wv x.
- **L121**: Append the current key to the layer's KV cache (enforces causality structurally in this sequential implementation).
- **L122**: Append the current value to the layer's KV cache.
- **L123**: Initialize the concatenated attention output across heads.
- **L124**: Iterate over attention heads.
- **L125**: Compute the head's slice offset into the full q/k/v vectors.
- **L126**: Slice query for the current head (dimension head_dim).
- **L127**: Slice cached keys for the current head across all prior positions.
- **L128**: Slice cached values for the current head across all prior positions.
- **L129**: Compute scaled dot-product attention logits against all cached keys.
- **L130**: Apply softmax over time to obtain attention weights.
- **L131**: Compute the weighted sum of cached values (the head output).
- **L132**: Concatenate this head output onto the full attention output vector.
- **L133**: Apply Wo to the concatenated head outputs (output projection).
- **L134**: Residual add: add attention output back to the sublayer input.
- **L135**: Comment: start the MLP (feed-forward) sublayer.
- **L136**: Save residual connection input for the MLP sublayer.
- **L137**: Pre-norm: apply RMSNorm before the MLP.
- **L138**: First MLP linear: expand dimension (n_embd → 4*n_embd).
- **L139**: Apply elementwise ReLU nonlinearity.
- **L140**: Second MLP linear: project back (4*n_embd → n_embd).
- **L141**: Residual add: add MLP output back to the sublayer input.
- **L142**: Blank line separator.
- **L143**: Final projection to vocabulary logits via `lm_head` (one logit per token).
- **L144**: Return logits to be converted to probabilities/loss downstream.
- **L145**: Blank line separator.
- **L146**: Comment: initialize Adam optimizer hyperparameters and state buffers.
- **L147**: Set learning rate and Adam coefficients (β1, β2) and epsilon for numerical stability.
- **L148**: Initialize first-moment (mean gradient) buffer m for each parameter.
- **L149**: Initialize second-moment (mean squared gradient) buffer v for each parameter.
- **L150**: Blank line separator.
- **L151**: Comment: training loop (repeat for `num_steps`).
- **L152**: Set number of training steps (iterations).
- **L153**: Iterate over training steps.
- **L154**: Blank line (indent block start).
- **L155**: Comment: pick a document, tokenize it, and wrap with BOS on both ends.
- **L156**: Select the current training document (cycling through docs).
- **L157**: Convert characters to token ids and wrap with BOS at start and end.
- **L158**: Set sequence length n (truncate to block_size, and ensure targets exist).
- **L159**: Blank line separator.
- **L160**: Comment: forward pass over positions; build the computation graph up to the loss.
- **L161**: Initialize per-layer KV caches for this sequence (empty lists that will grow with pos_id).
- **L162**: Initialize list to hold per-position losses.
- **L163**: Loop over positions (teacher forcing): predict token_{t+1} from token_t and past context.
- **L164**: Set current input token id and the target (next) token id.
- **L165**: Run one forward step of the model to obtain logits (KV cache is updated inside gpt).
- **L166**: Convert logits to probabilities via softmax.
- **L167**: Per-position negative log-likelihood loss: -log p(target).
- **L168**: Accumulate per-position losses.
- **L169**: Average the losses across positions to produce a single scalar loss for the sequence.
- **L170**: Blank line separator.
- **L171**: Comment: backward pass—compute gradients of the loss with respect to all parameters.
- **L172**: Run backpropagation through the entire computation graph.
- **L173**: Blank line separator.
- **L174**: Comment: Adam optimizer step—update parameters using their gradients and Adam state.
- **L175**: Compute a linearly decayed learning rate for this step.
- **L176**: Iterate over all parameters and update each one.
- **L177**: Update first moment estimate m (EMA of gradients).
- **L178**: Update second moment estimate v (EMA of squared gradients).
- **L179**: Bias-correct m to obtain m_hat.
- **L180**: Bias-correct v to obtain v_hat.
- **L181**: Parameter update: p -= lr * m_hat / (sqrt(v_hat) + eps).
- **L182**: Reset p.grad to 0 to avoid gradient accumulation across steps.
- **L183**: Blank line separator.
- **L184**: Print training progress (step index and current loss).
- **L185**: Blank line separator.
- **L186**: Comment: inference/sampling—generate new sequences from the trained model.
- **L187**: Set sampling temperature (controls distribution sharpness / randomness).
- **L188**: Print inference header.
- **L189**: Generate 20 samples (documents).
- **L190**: Initialize fresh KV caches for each generated sample.
- **L191**: Start generation with BOS to signal a new document.
- **L192**: Collect generated character tokens here.
- **L193**: Autoregressively generate up to block_size tokens.
- **L194**: Run the model on the current token to get next-token logits.
- **L195**: Scale logits by temperature and softmax to obtain a sampling distribution.
- **L196**: Sample the next token id according to the model probabilities.
- **L197**: If BOS is sampled, treat it as end-of-document and stop generation.
- **L198**: Break out of the generation loop.
- **L199**: Otherwise append the sampled character to the output string.
- **L200**: Print the completed sample (a generated name).
