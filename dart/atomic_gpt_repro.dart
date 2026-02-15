// atomic_gpt_repro.dart
//
// Repro refactor: keep scalar autograd(Value) like original,
// but make randomness match Python random as closely as possible:
// - MT19937
// - seed/init_by_array (Python-style)
// - random() 53-bit
// - gauss() with caching same formula as Python
// - shuffle() using randbelow/getrandbits rejection
// - choices(weights) using cum_weights + bisect_right
//
// Run:
//   dart run atomic_gpt_repro.dart
//
// NOTE:
// - For closest match, ensure input.txt is identical to Python run.

import 'dart:collection';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

// -----------------------------
// Python-compatible random (MT19937)
// -----------------------------
class PyRandom {
  static const int _n = 624;
  static const int _m = 397;
  static const int _matrixA = 0x9908b0df;
  static const int _upperMask = 0x80000000;
  static const int _lowerMask = 0x7fffffff;

  final Uint32List _mt = Uint32List(_n);
  int _mti = _n + 1;

  double? _gaussNext;

  PyRandom([int seed = 0]) {
    this.seed(seed);
  }

  void seed(int a) {
    // Python's _random seeds by building a 32-bit key array and init_by_array.
    // For small int seeds (like 42) this effectively uses key=[42].
    var x = a;
    if (x < 0) x = -x;

    final key = <int>[];
    if (x == 0) {
      key.add(0);
    } else {
      while (x != 0) {
        key.add(x & 0xffffffff);
        x = x >>> 32;
      }
    }
    _initByArray(key);
    _gaussNext = null;
  }

  void _initGenrand(int s) {
    _mt[0] = s & 0xffffffff;
    for (var i = 1; i < _n; i++) {
      final prev = _mt[i - 1];
      final v = 1812433253 * (prev ^ (prev >>> 30)) + i;
      _mt[i] = v & 0xffffffff;
    }
    _mti = _n;
  }

  void _initByArray(List<int> initKey) {
    _initGenrand(19650218);
    var i = 1;
    var j = 0;
    var k = _n > initKey.length ? _n : initKey.length;

    for (; k > 0; k--) {
      final prev = _mt[i - 1];
      final v = (_mt[i] ^
              ((prev ^ (prev >>> 30)) * 1664525)) +
          (initKey[j] & 0xffffffff) +
          j;
      _mt[i] = v & 0xffffffff;
      i++;
      j++;
      if (i >= _n) {
        _mt[0] = _mt[_n - 1];
        i = 1;
      }
      if (j >= initKey.length) j = 0;
    }

    for (k = _n - 1; k > 0; k--) {
      final prev = _mt[i - 1];
      final v = (_mt[i] ^
              ((prev ^ (prev >>> 30)) * 1566083941)) -
          i;
      _mt[i] = v & 0xffffffff;
      i++;
      if (i >= _n) {
        _mt[0] = _mt[_n - 1];
        i = 1;
      }
    }
    _mt[0] = 0x80000000; // MSB is 1; assuring non-zero initial array
  }

  int _genrandUint32() {
    int y;
    if (_mti >= _n) {
      // twist
      for (var kk = 0; kk < _n - _m; kk++) {
        y = (_mt[kk] & _upperMask) | (_mt[kk + 1] & _lowerMask);
        _mt[kk] = _mt[kk + _m] ^
            (y >>> 1) ^
            ((y & 1) != 0 ? _matrixA : 0);
      }
      for (var kk = _n - _m; kk < _n - 1; kk++) {
        y = (_mt[kk] & _upperMask) | (_mt[kk + 1] & _lowerMask);
        _mt[kk] = _mt[kk + (_m - _n)] ^
            (y >>> 1) ^
            ((y & 1) != 0 ? _matrixA : 0);
      }
      y = (_mt[_n - 1] & _upperMask) | (_mt[0] & _lowerMask);
      _mt[_n - 1] = _mt[_m - 1] ^
          (y >>> 1) ^
          ((y & 1) != 0 ? _matrixA : 0);

      _mti = 0;
    }

    y = _mt[_mti++];

    // tempering
    y ^= (y >>> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >>> 18);
    return y & 0xffffffff;
  }

  double random() {
    // Python: (a*2^26 + b)/2^53 with a=gen>>5 (27 bits), b=gen>>6 (26 bits)
    final a = _genrandUint32() >>> 5;
    final b = _genrandUint32() >>> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  int getrandbits(int k) {
    if (k <= 0) return 0;
    if (k <= 32) {
      final r = _genrandUint32();
      return (r >>> (32 - k)) & ((1 << k) - 1);
    }
    // generic (not needed here usually), build via BigInt then cast down if possible
    var remaining = k;
    BigInt acc = BigInt.zero;
    while (remaining > 0) {
      final take = remaining >= 32 ? 32 : remaining;
      final chunk = getrandbits(take);
      acc = (acc << take) | BigInt.from(chunk);
      remaining -= take;
    }
    // if k > 63 this may not fit int; but our uses are small
    return acc.toInt();
  }

  int _randbelow(int n) {
    if (n <= 0) throw ArgumentError('n must be > 0');
    final k = n.bitLength;
    while (true) {
      final r = getrandbits(k);
      if (r < n) return r;
    }
  }

  void shuffle<T>(List<T> x) {
    // Python shuffle: for i in reversed(range(1, len(x))): j=randbelow(i+1); swap
    for (var i = x.length - 1; i > 0; i--) {
      final j = _randbelow(i + 1);
      final tmp = x[i];
      x[i] = x[j];
      x[j] = tmp;
    }
  }

  // Python gauss(): uses cached second sample and the exact transform:
  // x2pi = random() * 2*pi
  // g2rad = sqrt(-2*log(1.0 - random()))
  // z = cos(x2pi) * g2rad; cache = sin(x2pi) * g2rad
  double gauss(double mu, double sigma) {
    final z = _gaussNext;
    if (z != null) {
      _gaussNext = null;
      return mu + z * sigma;
    }
    final x2pi = random() * 2.0 * math.pi;
    final g2rad = math.sqrt(-2.0 * math.log(1.0 - random()));
    final z0 = math.cos(x2pi) * g2rad;
    final z1 = math.sin(x2pi) * g2rad;
    _gaussNext = z1;
    return mu + z0 * sigma;
    // (matches Python's caching behavior)
  }

  int choicesIndexFromWeights(List<double> weights) {
    // Python choices(population=range(n), weights=weights, k=1)[0]
    // cum_weights = list(accumulate(weights))
    // r = random()*total
    // return bisect_right(cum_weights, r)
    double total = 0.0;
    final cum = List<double>.filled(weights.length, 0.0);
    for (var i = 0; i < weights.length; i++) {
      total += weights[i];
      cum[i] = total;
    }
    final r = random() * total;

    // bisect_right: first index where cum[i] > r
    var lo = 0;
    var hi = cum.length;
    while (lo < hi) {
      final mid = (lo + hi) >> 1;
      if (r < cum[mid]) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    // lo in [0..n]
    if (lo >= weights.length) return weights.length - 1;
    return lo;
  }
}

// -----------------------------
// Autograd scalar Value (same spirit as original)
// -----------------------------
class Value {
  double data;
  double grad = 0.0;
  final List<Value> _children;
  final List<double> _localGrads;

  Value(this.data, [List<Value>? children, List<double>? localGrads])
      : _children = children ?? <Value>[],
        _localGrads = localGrads ?? <double>[];

  static Value _ensure(Object other) {
    if (other is Value) return other;
    if (other is num) return Value(other.toDouble());
    throw ArgumentError('Expected Value or num, got ${other.runtimeType}');
  }

  Value operator +(Object other) {
    final o = _ensure(other);
    return Value(data + o.data, <Value>[this, o], <double>[1.0, 1.0]);
  }

  Value operator -() => this * -1.0;

  Value operator -(Object other) => this + (-_ensure(other));

  Value operator *(Object other) {
    final o = _ensure(other);
    return Value(
      data * o.data,
      <Value>[this, o],
      <double>[o.data, data],
    );
  }

  Value pow(double exponent) {
    final out = math.pow(data, exponent).toDouble();
    final lg = exponent * math.pow(data, exponent - 1.0).toDouble();
    return Value(out, <Value>[this], <double>[lg]);
  }

  Value operator /(Object other) {
    final o = _ensure(other);
    return this * o.pow(-1.0);
  }

  Value log() => Value(math.log(data), <Value>[this], <double>[1.0 / data]);

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

    void build(Value v) {
      if (visited.add(v)) {
        for (final c in v._children) build(c);
        topo.add(v);
      }
    }

    build(this);
    grad = 1.0;
    for (final v in topo.reversed) {
      final g = v.grad;
      for (var i = 0; i < v._children.length; i++) {
        v._children[i].grad += v._localGrads[i] * g;
      }
    }
  }
}

// -----------------------------
// Model helpers (Value-based)
// -----------------------------
List<List<Value>> matrix(int nOut, int nIn, PyRandom rng, {double std = 0.08}) {
  return List<List<Value>>.generate(
    nOut,
    (_) => List<Value>.generate(nIn, (_) => Value(rng.gauss(0.0, std))),
  );
}

List<Value> linear(List<Value> x, List<List<Value>> w) {
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
  final exps = logits.map((v) => (v - maxVal).exp()).toList();
  var total = Value(0.0);
  for (final e in exps) {
    total = total + e;
  }
  return exps.map((e) => e / total).toList();
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
// Main (repro)
// -----------------------------
Future<void> main() async {
  // Python random.seed(42) equivalent
  final rng = PyRandom(42);

  const inputPath = 'input.txt';
  if (!File(inputPath).existsSync()) {
    const url1 =
        'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt';
    const url2 =
        'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt';
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

  // Python random.shuffle(docs)
  rng.shuffle(docs);
  print('num docs: ${docs.length}');

  // Tokenizer
  final joined = docs.join();
  final setChars = <String>{};
  for (final r in joined.runes) {
    setChars.add(String.fromCharCode(r));
  }
  final uchars = setChars.toList()..sort();
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
      tokens.add(stoi[String.fromCharCode(r)]!);
    }
    tokens.add(BOS);
    return tokens;
  }

  // Hyperparams (same as original)
  const nEmb = 16;
  const nHead = 4;
  const nLayer = 1;
  const blockSize = 16;
  const headDim = nEmb ~/ nHead;

  // Params
  final state = <String, List<List<Value>>>{};
  state['wte'] = matrix(vocabSize, nEmb, rng);
  state['wpe'] = matrix(blockSize, nEmb, rng);
  state['lm_head'] = matrix(vocabSize, nEmb, rng);

  for (var li = 0; li < nLayer; li++) {
    state['layer$li.attn_wq'] = matrix(nEmb, nEmb, rng);
    state['layer$li.attn_wk'] = matrix(nEmb, nEmb, rng);
    state['layer$li.attn_wv'] = matrix(nEmb, nEmb, rng);
    state['layer$li.attn_wo'] = matrix(nEmb, nEmb, rng);
    state['layer$li.mlp_fc1'] = matrix(4 * nEmb, nEmb, rng);
    state['layer$li.mlp_fc2'] = matrix(nEmb, 4 * nEmb, rng);
  }

  final params = <Value>[];
  for (final mat in state.values) {
    for (final row in mat) {
      params.addAll(row);
    }
  }
  print('num params: ${params.length}');

  List<Value> gpt(
    int tokenId,
    int posId,
    List<List<List<Value>>> keys,
    List<List<List<Value>>> values,
  ) {
    final tokEmb = state['wte']![tokenId];
    final posEmb = state['wpe']![posId];
    var x = List<Value>.generate(nEmb, (i) => tokEmb[i] + posEmb[i]);
    x = rmsnorm(x);

    for (var li = 0; li < nLayer; li++) {
      // attn
      final xRes = x;
      x = rmsnorm(x);

      final q = linear(x, state['layer$li.attn_wq']!);
      final k = linear(x, state['layer$li.attn_wk']!);
      final v = linear(x, state['layer$li.attn_wv']!);

      keys[li].add(k);
      values[li].add(v);

      final xAttn = <Value>[];
      for (var h = 0; h < nHead; h++) {
        final hs = h * headDim;
        final qh = q.sublist(hs, hs + headDim);
        final kh = keys[li].map((kv) => kv.sublist(hs, hs + headDim)).toList();
        final vh = values[li].map((vv) => vv.sublist(hs, hs + headDim)).toList();

        final attnLogits = <Value>[];
        for (var t = 0; t < kh.length; t++) {
          var dot = Value(0.0);
          for (var j = 0; j < headDim; j++) {
            dot = dot + (qh[j] * kh[t][j]);
          }
          attnLogits.add(dot / math.sqrt(headDim.toDouble()));
        }
        final attnW = softmax(attnLogits);

        final headOut = List<Value>.generate(headDim, (j) {
          var s = Value(0.0);
          for (var t = 0; t < vh.length; t++) {
            s = s + (attnW[t] * vh[t][j]);
          }
          return s;
        });

        xAttn.addAll(headOut);
      }

      x = linear(xAttn, state['layer$li.attn_wo']!);
      x = List<Value>.generate(nEmb, (i) => x[i] + xRes[i]);

      // mlp
      final xRes2 = x;
      x = rmsnorm(x);
      x = linear(x, state['layer$li.mlp_fc1']!);
      x = x.map((xi) => xi.relu()).toList();
      x = linear(x, state['layer$li.mlp_fc2']!);
      x = List<Value>.generate(nEmb, (i) => x[i] + xRes2[i]);
    }

    return linear(x, state['lm_head']!);
  }

  // Adam
  const learningRate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const epsAdam = 1e-8;

  final m = List<double>.filled(params.length, 0.0);
  final v = List<double>.filled(params.length, 0.0);

  const numSteps = 1000;
  for (var step = 0; step < numSteps; step++) {
    final doc = docs[step % docs.length];
    final tokens = tokenize(doc);
    final n = math.min(blockSize, tokens.length - 1);

    final keys = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);
    final values = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);

    final losses = <Value>[];
    for (var pos = 0; pos < n; pos++) {
      final tokenId = tokens[pos];
      final targetId = tokens[pos + 1];
      final logits = gpt(tokenId, pos, keys, values);
      final probs = softmax(logits);
      losses.add(-(probs[targetId].log()));
    }

    var loss = Value(0.0);
    for (final lt in losses) loss = loss + lt;
    loss = loss / n;

    loss.backward();

    final lrT = learningRate * (1.0 - step / numSteps);
    final b1Pow = math.pow(beta1, step + 1).toDouble();
    final b2Pow = math.pow(beta2, step + 1).toDouble();

    for (var i = 0; i < params.length; i++) {
      final p = params[i];
      m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1 - beta2) * (p.grad * p.grad);
      final mHat = m[i] / (1 - b1Pow);
      final vHat = v[i] / (1 - b2Pow);
      p.data -= lrT * mHat / (math.sqrt(vHat) + epsAdam);
      p.grad = 0.0;
    }

    final stepStr = (step + 1).toString().padLeft(4);
    final totalStr = numSteps.toString().padLeft(4);
    print('step $stepStr / $totalStr | loss ${loss.data.toStringAsFixed(4)}');
  }

  // Inference (Python random.choices equivalent)
  const temperature = 0.5;
  print('\n--- inference (new, hallucinated names) ---');
  for (var sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    final keys = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);
    final values = List<List<List<Value>>>.generate(nLayer, (_) => <List<Value>>[]);

    var tokenId = BOS;
    final out = StringBuffer();

    for (var pos = 0; pos < blockSize; pos++) {
      final logits = gpt(tokenId, pos, keys, values);
      final scaled = logits.map((l) => l / temperature).toList();
      final probs = softmax(scaled);

      final weights = probs.map((p) => p.data).toList();
      tokenId = rng.choicesIndexFromWeights(weights);

      if (tokenId == BOS) break;
      out.write(uchars[tokenId]);
    }

    final idxStr = (sampleIdx + 1).toString().padLeft(2);
    print('sample $idxStr: ${out.toString()}');
  }
}

// same downloader as fast version
Future<void> downloadTextFile(String url, String path) async {
  final client = HttpClient();
  try {
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();
    if (response.statusCode != 200) {
      throw HttpException('HTTP ${response.statusCode}', uri: Uri.parse(url));
    }
    await response.pipe(File(path).openWrite());
  } finally {
    client.close(force: true);
  }
}
