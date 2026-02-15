# microgpt 부록: 라인 번호 코드 + 라인별 해석

본 부록은 사용자가 제공한 microgpt 스크립트를 **그대로** 옮겨, (A) 라인 번호를 부여한 코드와 (B) 동일 라인 번호 기준의 해설을 제공한다.

---

## 부록 사용 가이드

- **트랜스포머 블록(=GPT 디코더 블록)**은 `gpt()` 함수 내부의 `for li in range(n_layer):` 구간(대략 **L114-L142**)이다.
- Attention(QKV, softmax, weighted sum)은 **L118-L135**에 집중되어 있고, MLP(FC1-ReLU-FC2)는 **L136-L141**에 위치한다.
- 훈련 루프는 **L151-L184**, 추론(샘플링)은 **L186-L200**이다.


## 부록 A. 라인 번호 포함 코드 (L001-L200)

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

## 부록 B. 라인별 해석 (L001-L200)

- **L001**: 모듈 최상단 도큐스트링 시작. 파일의 목적/컨셉(‘최소 알고리즘’)을 선언.
- **L002**: 설명 문장: 외부 의존성 없이 GPT를 학습/추론하는 ‘가장 원자적(atomic)’ 구현이라는 문제의식.
- **L003**: 설명 문장: 본 파일이 필요한 알고리즘 내용을 모두 담고 있음을 강조.
- **L004**: 설명 문장: 나머지는 효율(벡터화/커널/분산 등) 문제라는 관점.
- **L005**: 도큐스트링 내부 빈 줄(가독성).
- **L006**: 저자 표기(karpathy).
- **L007**: 모듈 도큐스트링 종료.
- **L008**: 빈 줄(구획/가독성).
- **L009**: 표준 라이브러리 os import. 파일 존재 여부 확인에 사용.
- **L010**: 표준 라이브러리 math import. log/exp 등 스칼라 함수에 사용.
- **L011**: 표준 라이브러리 random import. 초기화/샘플링/셔플 등에 사용.
- **L012**: 난수 시드를 고정해 재현성(reproducibility)을 확보.
- **L013**: 빈 줄(구획/가독성).
- **L014**: 주석: 입력 데이터셋 docs(문서 리스트)를 준비한다는 ‘창세기’식 내러티브.
- **L015**: input.txt가 없으면 다운로드한다(데이터 준비의 최소 구현).
- **L016**: 필요할 때만 urllib.request를 import(조건부 import로 의존성 최소화).
- **L017**: 데이터 URL 지정(karpathy/makemore의 names.txt).
- **L018**: URL에서 파일을 내려받아 input.txt로 저장.
- **L019**: 파일을 읽어 줄 단위로 문서 리스트(docs) 구성. 공백 줄은 제거.
- **L020**: 문서 순서를 섞어(셔플) 학습 순서 편향을 줄임.
- **L021**: 문서 개수 출력(데이터 로딩 확인용).
- **L022**: 빈 줄(구획/가독성).
- **L023**: 주석: 문자열↔토큰 ID 변환기(tokenizer)를 정의.
- **L024**: 데이터 전체에서 등장하는 유니크 문자를 수집/정렬해 어휘를 만든다.
- **L025**: 특수 토큰 BOS의 ID를 어휘 끝에 배정(문서 경계 토큰).
- **L026**: 총 어휘 크기(문자 수 + BOS) 산출.
- **L027**: 어휘 크기 출력(토크나이저 확인용).
- **L028**: 빈 줄(구획/가독성).
- **L029**: 주석: 오토그라드(자동미분) 엔진(Value)를 정의.
- **L030**: Value 클래스 정의. 각 노드는 스칼라 값과 그래프 연결/국소 미분을 저장.
- **L031**: __slots__로 속성 고정 → 파이썬 객체 메모리/속도 최적화.
- **L032**: 빈 줄(구획/가독성).
- **L033**: Value 생성자. data(값)와 children(입력 노드들), local_grads(국소 도함수)를 저장.
- **L034**: 전방향(forward) 계산 결과 스칼라 값.
- **L035**: 역전파(backward)에서 누적될 기울기(∂L/∂self). 초기값 0.
- **L036**: 현재 노드가 의존하는 입력 노드(연산 그래프의 부모/자식 관계) 저장.
- **L037**: 현재 노드의 출력이 각 child에 대해 어떻게 변하는지(국소 미분) 저장.
- **L038**: 빈 줄(구획/가독성).
- **L039**: 덧셈 연산 정의: Value + Value → 새 Value 노드 생성.
- **L040**: other가 스칼라면 Value로 감싸 동일 인터페이스로 처리.
- **L041**: 출력 값은 합, 국소 미분은 (1,1). 그래프 연결(children)도 기록.
- **L042**: 빈 줄(구획/가독성).
- **L043**: 곱셈 연산 정의: Value * Value → 새 Value 노드 생성.
- **L044**: other가 스칼라면 Value로 감싸 처리.
- **L045**: 출력 값은 곱, 국소 미분은 (∂/∂a=b, ∂/∂b=a)로 저장.
- **L046**: 빈 줄(구획/가독성).
- **L047**: 거듭제곱 연산(스칼라 지수). 미분: n*a^(n-1).
- **L048**: 자연로그 연산. 미분: 1/a.
- **L049**: 지수(exp) 연산. 미분: exp(a).
- **L050**: ReLU. 미분: a>0이면 1, 아니면 0.
- **L051**: 단항 음수(-a) 구현: a * (-1).
- **L052**: 오른쪽 덧셈(other + self) 지원.
- **L053**: 뺄셈 구현: a - b = a + (-b).
- **L054**: 오른쪽 뺄셈(other - self) 지원.
- **L055**: 오른쪽 곱셈(other * self) 지원.
- **L056**: 나눗셈 구현: a/b = a * b^(-1).
- **L057**: 오른쪽 나눗셈(other / self) 지원.
- **L058**: 빈 줄(구획/가독성).
- **L059**: 역전파 시작. loss 노드에서 모든 입력(파라미터)까지 기울기를 전파.
- **L060**: topo: 위상 정렬된 노드 목록을 만들기 위한 리스트.
- **L061**: visited: DFS 중복 방문 방지용 집합.
- **L062**: 내부 함수 build_topo: 그래프를 DFS로 순회하며 위상 정렬 구축.
- **L063**: 노드 v를 아직 방문하지 않았다면 처리 시작.
- **L064**: 방문 처리.
- **L065**: v가 의존하는 child들로 재귀적으로 내려감.
- **L066**: child에 대해 build_topo 재귀 호출.
- **L067**: 모든 child 처리가 끝난 뒤 v를 topo에 추가(후위 순회).
- **L068**: loss(=self)에서 위상 정렬 구축 시작.
- **L069**: ∂L/∂L = 1로 초기화(역전파 시작점).
- **L070**: 위상 정렬 역순으로 순회하며 체인룰 적용(자식→부모 방향으로 gradient 전파).
- **L071**: 각 child와 해당 local_grad(∂v/∂child)를 짝지어 처리.
- **L072**: child.grad += local_grad * v.grad (다변수 체인룰 + 분기 시 gradient 누적).
- **L073**: 빈 줄(구획/가독성).
- **L074**: 주석: 모델 파라미터(지식)를 초기화.
- **L075**: n_embd: 임베딩/잔차 스트림 차원(d_model).
- **L076**: n_head: 어텐션 헤드 수.
- **L077**: n_layer: Transformer 레이어 수(여기서는 1).
- **L078**: block_size: 최대 시퀀스 길이(컨텍스트 윈도우).
- **L079**: head_dim: 각 헤드의 차원 = n_embd / n_head.
- **L080**: matrix 헬퍼: (nout x nin) 가중치 행렬을 작은 가우시안으로 초기화(Value로 래핑).
- **L081**: state_dict: 토큰 임베딩(wte), 포지션 임베딩(wpe), 출력 헤드(lm_head) 저장.
- **L082**: 각 레이어별 파라미터 행렬들을 state_dict에 등록.
- **L083**: W_Q: query 프로젝션 행렬.
- **L084**: W_K: key 프로젝션 행렬.
- **L085**: W_V: value 프로젝션 행렬.
- **L086**: W_O: 어텐션 출력 프로젝션 행렬.
- **L087**: MLP 1층(확장) 가중치.
- **L088**: MLP 2층(축소) 가중치.
- **L089**: state_dict의 모든 행렬을 1차 리스트 params로 평탄화(옵티마이저 루프 단순화).
- **L090**: 총 파라미터 수 출력(모델 규모 확인).
- **L091**: 빈 줄(구획/가독성).
- **L092**: 주석: 모델 아키텍처는 ‘상태 없는 함수’로 정의한다는 선언.
- **L093**: 주석: GPT-2 계열을 따르되 LN→RMSNorm, bias 제거, GeLU→ReLU로 단순화.
- **L094**: linear: 벡터 x에 대해 w의 각 row와 내적(=행렬-벡터 곱) 수행.
- **L095**: 파이썬 리스트/제너레이터로 dot product를 직접 합산(텐서 라이브러리 미사용).
- **L096**: 빈 줄(구획/가독성).
- **L097**: softmax 정의: 로짓을 확률로 변환.
- **L098**: 수치 안정성을 위해 로짓의 최대값을 뺌(softmax 불변 성질 이용).
- **L099**: exp(logits - max) 계산. 여기서 exp는 Value.exp()라 그래프에 포함.
- **L100**: 분모(합) 계산.
- **L101**: 정규화해 확률 분포 반환(합이 1).
- **L102**: 빈 줄(구획/가독성).
- **L103**: rmsnorm 정의: RMS 기반 정규화.
- **L104**: ms = mean(x_i^2).
- **L105**: scale = (ms + eps)^(-1/2). eps는 0으로 나누는 문제 방지.
- **L106**: 각 성분에 scale을 곱해 출력. (학습 가능한 gain 없음).
- **L107**: 빈 줄(구획/가독성).
- **L108**: gpt 정의: (현재 토큰, 위치, KV cache) → 다음 토큰 로짓. GPT형 디코더의 핵심.
- **L109**: 토큰 임베딩 lookup: wte[token_id] → 길이 n_embd 벡터.
- **L110**: 포지션 임베딩 lookup: wpe[pos_id] → 길이 n_embd 벡터.
- **L111**: 두 임베딩을 더해 입력 표현을 구성(절대 위치 + 토큰).
- **L112**: 입력에 RMSNorm 적용(안정화).
- **L113**: 빈 줄(구획/가독성).
- **L114**: Transformer 레이어 반복(여기서는 n_layer=1이라 1회).
- **L115**: 주석: 1) 멀티헤드 self-attention 블록 시작.
- **L116**: 잔차(residual) 연결을 위해 현재 x를 저장.
- **L117**: pre-norm: 어텐션 입력을 RMSNorm으로 정규화.
- **L118**: q = W_Q x (선형 변환).
- **L119**: k = W_K x.
- **L120**: v = W_V x.
- **L121**: 현재 위치의 k를 레이어별 KV cache에 append(과거 키 저장).
- **L122**: 현재 위치의 v를 레이어별 KV cache에 append(과거 값 저장).
- **L123**: 모든 헤드 출력을 이어 붙일 리스트 준비.
- **L124**: 각 head별로 attention을 계산.
- **L125**: 헤드 시작 인덱스 hs 계산(슬라이싱 오프셋).
- **L126**: q에서 해당 head 구간(q_h)만 추출.
- **L127**: cache에 쌓인 모든 k에 대해 head 구간만 추출(k_h). 길이 t+1.
- **L128**: cache에 쌓인 모든 v에 대해 head 구간만 추출(v_h).
- **L129**: scaled dot-product logits: q_h · k_h[t] / sqrt(head_dim). (미래 항 없음)
- **L130**: softmax로 attention 가중치(확률) 산출.
- **L131**: 가중합으로 head 출력 계산: Σ_t a_t * v_t.
- **L132**: 해당 head 출력(head_dim)을 x_attn에 이어 붙임(concat).
- **L133**: 헤드 concat 결과를 W_O로 선형 사상(출력 프로젝션).
- **L134**: 잔차 연결: 어텐션 출력 + 입력(x_residual).
- **L135**: 주석: 2) MLP(FFN) 블록 시작.
- **L136**: 잔차 연결을 위해 현재 x 저장.
- **L137**: pre-norm: MLP 입력 정규화.
- **L138**: 확장 선형층: n_embd → 4*n_embd.
- **L139**: ReLU 비선형 적용(원소별).
- **L140**: 축소 선형층: 4*n_embd → n_embd.
- **L141**: 잔차 연결: MLP 출력 + 입력.
- **L142**: 빈 줄(구획/가독성).
- **L143**: 최종 로짓 산출: lm_head로 n_embd → vocab_size 프로젝션.
- **L144**: 로짓 반환.
- **L145**: 빈 줄(구획/가독성).
- **L146**: 주석: Adam 옵티마이저 하이퍼파라미터/버퍼 준비.
- **L147**: learning_rate, beta1, beta2, eps 설정. (단순 선형 decay는 아래에서 적용)
- **L148**: m: 1차 모멘트(평균 gradient) 버퍼 초기화.
- **L149**: v: 2차 모멘트(평균 squared gradient) 버퍼 초기화.
- **L150**: 빈 줄(구획/가독성).
- **L151**: 주석: 학습 루프 시작.
- **L152**: 총 학습 스텝 수(num_steps).
- **L153**: 각 step마다 1개 문서를 사용(배치 없음).
- **L154**: 빈 줄(구획/가독성).
- **L155**: 주석: 현재 step에서 사용할 문서를 선택하고 토큰화한다.
- **L156**: docs를 순환하며 하나 선택(step % len(docs)).
- **L157**: 문서를 [BOS] + 문자토큰 + [BOS]로 래핑(문서 경계 학습).
- **L158**: 최대 길이(block_size)를 초과하지 않도록 학습 길이 n을 설정.
- **L159**: 빈 줄(구획/가독성).
- **L160**: 주석: 순차적으로 gpt를 호출하며 연산 그래프를 loss까지 구축.
- **L161**: 레이어별 KV cache(keys/values) 초기화. (t가 증가하며 append)
- **L162**: 시점별 loss를 모을 리스트.
- **L163**: pos_id를 0..n-1로 순회(teacher forcing).
- **L164**: 현재 입력 토큰과 정답(target=다음 토큰) 설정.
- **L165**: 현재 토큰을 gpt에 통과시켜 로짓 계산(이때 KV cache가 갱신됨).
- **L166**: 로짓을 softmax로 확률화.
- **L167**: 정답 토큰의 음의 로그확률을 loss로 정의(크로스엔트로피의 1항).
- **L168**: 시점 loss를 누적.
- **L169**: 시점 평균으로 문서 loss를 얻음(스칼라 1개).
- **L170**: 빈 줄(구획/가독성).
- **L171**: 주석: 역전파로 모든 파라미터 grad 계산.
- **L172**: loss.backward() 호출(연산 그래프 전체에 대해 체인룰 적용).
- **L173**: 빈 줄(구획/가독성).
- **L174**: 주석: Adam 업데이트(파라미터 갱신) 수행.
- **L175**: 학습률 선형 decay: step이 증가할수록 lr 감소.
- **L176**: 모든 파라미터에 대해 Adam 업데이트 적용.
- **L177**: m 갱신: 지수이동평균(gradient).
- **L178**: v 갱신: 지수이동평균(gradient^2).
- **L179**: 편향 보정(bias correction)된 m_hat 계산.
- **L180**: 편향 보정된 v_hat 계산.
- **L181**: 파라미터 업데이트: p -= lr * m_hat / (sqrt(v_hat) + eps).
- **L182**: 다음 step을 위해 gradient를 0으로 리셋(누적 방지).
- **L183**: 빈 줄(구획/가독성).
- **L184**: 현재 step과 loss를 출력(학습 진행 모니터링).
- **L185**: 빈 줄(구획/가독성).
- **L186**: 주석: 추론(샘플링) 단계 시작.
- **L187**: temperature 설정. 분포의 날카로움/다양성을 조절.
- **L188**: 추론 시작 메시지 출력.
- **L189**: 샘플 20개 생성 루프.
- **L190**: 각 샘플마다 KV cache를 초기화(새 문서 시작).
- **L191**: 초기 입력 토큰을 BOS로 설정(‘시작’ 트리거).
- **L192**: 생성된 문자들을 모을 리스트.
- **L193**: 최대 block_size 길이까지 autoregressive 생성.
- **L194**: 현재 token_id를 gpt에 넣어 다음 로짓을 계산.
- **L195**: temperature로 로짓을 스케일한 뒤 softmax → 샘플링 분포 생성.
- **L196**: random.choices로 확률에 비례해 다음 토큰을 샘플링.
- **L197**: BOS가 나오면 문서 종료로 간주하고 루프 종료.
- **L198**: 종료 조건 처리(반복 탈출).
- **L199**: BOS가 아니면 해당 문자를 결과에 추가.
- **L200**: 완성된 샘플(이름)을 출력.