// atomic_gpt.dart
//
// "The most atomic way to train and inference a GPT in pure, dependency-free Python."
// -> Ported to dependency-free Dart (standard library only).
//
// Run: dart run atomic_gpt.dart

import 'dart:collection';
import 'dart:io';
import 'dart:math' as math;

// -----------------------------
// Autograd scalar Value
// -----------------------------
class Value {
  double data; // scalar forward value
  double grad = 0.0; // d(loss)/d(this)
  final List<Value> _children;
  final List<double> _localGrads;

  Value(this.data, [List<Value>? children, List<double>? localGrads])
      : _children = children ?? <Value>[],
        _localGrads = localGrads ?? <double>[];

  static Value _ensureValue(Object other) {
    if (other is Value) return other;
    if (other is num) return Value(other.toDouble());
    throw ArgumentError('Expected Value or num, got ${other.runtimeType}');
  }

  Value operator +(Object other) {
    final o = _ensureValue(other);
    return Value(data + o.data, <Value>[this, o], <double>[1.0, 1.0]);
    // local grads: d/dself=1, d/dother=1
  }

  Value operator -() => this * -1.0;

  Value operator -(Object other) => this + (-_ensureValue(other));

  Value operator *(Object other) {
    final o = _ensureValue(other);
    // out = self * other
    // d(out)/d(self)=other.data, d(out)/d(other)=self.data
    return Value(
      data * o.data,
      <Value>[this, o],
      <double>[o.data, data],
    );
  }

  Value pow(double exponent) {
    final outData = math.pow(data, exponent).toDouble();
    final localGrad = exponent * math.pow(data, exponent - 1.0).toDouble();
    return Value(outData, <Value>[this], <double>[localGrad]);
  }

  Value operator /(Object other) {
    final o = _ensureValue(other);
    // self / other = self * other**-1
    return this * o.pow(-1.0);
  }

  Value log() {
    return Value(math.log(data), <Value>[this], <double>[1.0 / data]);
  }

  Value exp() {
    final e = math.exp(data);
    return Value(e, <Value>[this], <double>[e]);
  }

  Value relu() {
    final out = data > 0.0 ? data : 0.0;
    final lg = data > 0.0 ? 1.0 : 0.0;
    return Value(out, <Value>[this], <double>[lg]);
  }

  void backward() {
    final topo = <Value>[];
    final visited = HashSet<Value>.identity();

    void buildTopo(Value v) {
      if (visited.add(v)) {
        for (final child in v._children) {
          buildTopo(child);
        }
        topo.add(v);
      }
    }

    buildTopo(this);
    grad = 1.0; // d(loss)/d(loss)=1

    for (final v in topo.reversed) {
      final g = v.grad;
      for (var i = 0; i < v._children.length; i++) {
        v._children[i].grad += v._localGrads[i] * g;
      }
    }
  }
}

// -----------------------------
// Helpers (random, math, IO)
// -----------------------------
double gaussian(math.Random rng, double mean, double std) {
  // Box-Muller transform
  double u1 = 0.0;
  while (u1 == 0.0) {
    u1 = rng.nextDouble();
  }
  final u2 = rng.nextDouble();
  final z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  return mean + std * z0;
}

int weightedChoice(List<double> weights, math.Random rng) {
  double total = 0.0;
  for (final w in weights) {
    total += w;
  }
  final r = rng.nextDouble() * total;
  double c = 0.0;
  for (var i = 0; i < weights.length; i++) {
    c += weights[i];
    if (r < c) return i;
  }
  return weights.length - 1; // fallback
}

int minInt(int a, int b) => a < b ? a : b;

Future<void> downloadTextFile(String url, String path) async {
  final client = HttpClient();
  try {
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();
    if (response.statusCode != 200) {
      throw HttpException('HTTP ${response.statusCode}', uri: Uri.parse(url));
    }
    final file = File(path);
    await response.pipe(file.openWrite());
  } finally {
    client.close(force: true);
  }
}

// -----------------------------
// Model building blocks
// -----------------------------
List<List<Value>> matrix(int nOut, int nIn, math.Random rng, {double std = 0.08}) {
  return List<List<Value>>.generate(
    nOut,
    (_) => List<Value>.generate(nIn, (_) => Value(gaussian(rng, 0.0, std))),
  );
}

List<Value> linear(List<Value> x, List<List<Value>> w) {
  // w: [nOut][nIn], x: [nIn] => out: [nOut]
  final out = <Value>[];
  for (final wo in w) {
    var s = Value(0.0);
    for (var i = 0; i < wo.length; i++) {
      s = s + (wo[i] * x[i]);
    }
    out.add(s);
  }
  return out;
}

List<Value> softmax(List<Value> logits) {
  double maxVal = logits[0].data;
  for (var i = 1; i < logits.length; i++) {
    final d = logits[i].data;
    if (d > maxVal) maxVal = d;
  }

  final exps = <Value>[];
  for (final v in logits) {
    exps.add((v - maxVal).exp());
  }

  var total = Value(0.0);
  for (final e in exps) {
    total = total + e;
  }

  final probs = <Value>[];
  for (final e in exps) {
    probs.add(e / total);
  }
  return probs;
}

List<Value> rmsnorm(List<Value> x) {
  var ms = Value(0.0);
  for (final xi in x) {
    ms = ms + (xi * xi);
  }
  ms = ms / x.length;
  final scale = (ms + 1e-5).pow(-0.5);
  return x.map((xi) => xi * scale).toList();
}

// -----------------------------
// Main
// -----------------------------
Future<void> main() async {
  final rng = math.Random(42);

  // Let there be an input dataset `docs`
  const inputPath = 'input.txt';
  if (!File(inputPath).existsSync()) {
    // 원본 파이썬 코드의 URL (refs/heads/master)
    const url1 =
        'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt';
    // 혹시 위 URL이 막히는 환경을 대비한 일반 raw URL fallback
    const url2 = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt';

    try {
      await downloadTextFile(url1, inputPath);
    } catch (_) {
      await downloadTextFile(url2, inputPath);
    }
  }

  final raw = await File(inputPath).readAsString();
  final docs = raw
      .trim()
      .split(RegExp(r'\r?\n'))
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toList();
  docs.shuffle(rng);
  print('num docs: ${docs.length}');

  // Tokenizer: unique characters -> token ids
  final charSet = <String>{};
  for (final doc in docs) {
    for (final r in doc.runes) {
      charSet.add(String.fromCharCode(r));
    }
  }
  final uchars = charSet.toList()..sort();
  final BOS = uchars.length;
  final vocabSize = uchars.length + 1;
  print('vocab size: $vocabSize');

  final stoi = <String, int>{};
  for (var i = 0; i < uchars.length; i++) {
    stoi[uchars[i]] = i;
  }

  List<int> tokenize(String doc) {
    final tokens = <int>[BOS];
    for (final r in doc.runes) {
      final ch = String.fromCharCode(r);
      final id = stoi[ch];
      if (id == null) {
        throw StateError('Unknown char "$ch" in "$doc"');
      }
      tokens.add(id);
    }
    tokens.add(BOS);
    return tokens;
  }

  // Hyperparams
  const nEmb = 16;
  const nHead = 4;
  const nLayer = 1;
  const blockSize = 16;
  const headDim = nEmb ~/ nHead;

  // Parameters (state_dict)
  final stateDict = <String, List<List<Value>>>{};
  stateDict['wte'] = matrix(vocabSize, nEmb, rng);
  stateDict['wpe'] = matrix(blockSize, nEmb, rng);
  stateDict['lm_head'] = matrix(vocabSize, nEmb, rng);

  for (var i = 0; i < nLayer; i++) {
    stateDict['layer$i.attn_wq'] = matrix(nEmb, nEmb, rng);
    stateDict['layer$i.attn_wk'] = matrix(nEmb, nEmb, rng);
    stateDict['layer$i.attn_wv'] = matrix(nEmb, nEmb, rng);
    stateDict['layer$i.attn_wo'] = matrix(nEmb, nEmb, rng);
    stateDict['layer$i.mlp_fc1'] = matrix(4 * nEmb, nEmb, rng);
    stateDict['layer$i.mlp_fc2'] = matrix(nEmb, 4 * nEmb, rng);
  }

  // Flatten params
  final params = <Value>[];
  for (final mat in stateDict.values) {
    for (final row in mat) {
      params.addAll(row);
    }
  }
  print('num params: ${params.length}');

  // Model: stateless function mapping (token, pos, kv-cache) -> logits
  List<Value> gpt(
    int tokenId,
    int posId,
    List<List<List<Value>>> keys,
    List<List<List<Value>>> values,
  ) {
    final tokEmb = stateDict['wte']![tokenId];
    final posEmb = stateDict['wpe']![posId];

    // x = tok_emb + pos_emb
    var x = List<Value>.generate(nEmb, (i) => tokEmb[i] + posEmb[i]);
    x = rmsnorm(x);

    for (var li = 0; li < nLayer; li++) {
      // 1) Multi-head attention block
      var xResidual = x;
      x = rmsnorm(x);

      final q = linear(x, stateDict['layer$li.attn_wq']!);
      final k = linear(x, stateDict['layer$li.attn_wk']!);
      final vVec = linear(x, stateDict['layer$li.attn_wv']!);

      keys[li].add(k);
      values[li].add(vVec);

      final xAttn = <Value>[];

      for (var h = 0; h < nHead; h++) {
        final hs = h * headDim;
        final qh = q.sublist(hs, hs + headDim);

        final kh = keys[li].map((kv) => kv.sublist(hs, hs + headDim)).toList();
        final vh = values[li].map((vv) => vv.sublist(hs, hs + headDim)).toList();

        final attnLogits = <Value>[];
        final scale = math.sqrt(headDim.toDouble());

        for (var t = 0; t < kh.length; t++) {
          var dot = Value(0.0);
          for (var j = 0; j < headDim; j++) {
            dot = dot + (qh[j] * kh[t][j]);
          }
          attnLogits.add(dot / scale);
        }

        final attnWeights = softmax(attnLogits);

        final headOut = List<Value>.generate(headDim, (j) {
          var s = Value(0.0);
          for (var t = 0; t < vh.length; t++) {
            s = s + (attnWeights[t] * vh[t][j]);
          }
          return s;
        });

        xAttn.addAll(headOut);
      }

      x = linear(xAttn, stateDict['layer$li.attn_wo']!);
      x = List<Value>.generate(nEmb, (i) => x[i] + xResidual[i]);

      // 2) MLP block
      xResidual = x;
      x = rmsnorm(x);
      x = linear(x, stateDict['layer$li.mlp_fc1']!);
      x = x.map((xi) => xi.relu()).toList();
      x = linear(x, stateDict['layer$li.mlp_fc2']!);
      x = List<Value>.generate(nEmb, (i) => x[i] + xResidual[i]);
    }

    final logits = linear(x, stateDict['lm_head']!);
    return logits;
  }

  // Adam optimizer buffers
  const learningRate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const epsAdam = 1e-8;

  final m = List<double>.filled(params.length, 0.0);
  final v = List<double>.filled(params.length, 0.0);

  // Training
  const numSteps = 1000;
  for (var step = 0; step < numSteps; step++) {
    final doc = docs[step % docs.length];
    final tokens = tokenize(doc);
    final n = minInt(blockSize, tokens.length - 1);

    // kv cache (per step)
    final keys = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);
    final values = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);

    final losses = <Value>[];

    for (var posId = 0; posId < n; posId++) {
      final tokenId = tokens[posId];
      final targetId = tokens[posId + 1];

      final logits = gpt(tokenId, posId, keys, values);
      final probs = softmax(logits);

      final lossT = -(probs[targetId].log());
      losses.add(lossT);
    }

    var loss = Value(0.0);
    for (final lt in losses) {
      loss = loss + lt;
    }
    loss = loss / n; // average loss

    // Backprop
    loss.backward();

    // Adam update
    final lrT = learningRate * (1.0 - (step / numSteps));
    final b1Pow = math.pow(beta1, step + 1).toDouble();
    final b2Pow = math.pow(beta2, step + 1).toDouble();

    for (var i = 0; i < params.length; i++) {
      final p = params[i];

      m[i] = beta1 * m[i] + (1.0 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1.0 - beta2) * (p.grad * p.grad);

      final mHat = m[i] / (1.0 - b1Pow);
      final vHat = v[i] / (1.0 - b2Pow);

      p.data -= lrT * mHat / (math.sqrt(vHat) + epsAdam);
      p.grad = 0.0;
    }

    final stepStr = (step + 1).toString().padLeft(4);
    final totalStr = numSteps.toString().padLeft(4);
    print('step $stepStr / $totalStr | loss ${loss.data.toStringAsFixed(4)}');
  }

  // Inference
  const temperature = 0.5;
  print('\n--- inference (new, hallucinated names) ---');

  for (var sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    final keys = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);
    final values = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);

    var tokenId = BOS;
    final sb = StringBuffer();

    for (var posId = 0; posId < blockSize; posId++) {
      final logits = gpt(tokenId, posId, keys, values);
      final scaled = logits.map((l) => l / temperature).toList();
      final probs = softmax(scaled);

      final weights = probs.map((p) => p.data).toList();
      tokenId = weightedChoice(weights, rng);

      if (tokenId == BOS) break;
      sb.write(uchars[tokenId]);
    }

    final idxStr = (sampleIdx + 1).toString().padLeft(2);
    print('sample $idxStr: ${sb.toString()}');
  }
}
