// atomic_gpt_fast.dart
//
// Fast refactor: Float64List + manual backprop + buffer reuse (no scalar autograd).
// Dependency-free (Dart standard library only).
//
// Run:
//   dart run atomic_gpt_fast.dart
//
// Tip for speed:
//   dart compile exe atomic_gpt_fast.dart -O2 -o gpt_fast && ./gpt_fast

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

// -----------------------------
// IO: download input if missing
// -----------------------------
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

// -----------------------------
// Random helpers
// -----------------------------
double gaussian(math.Random rng, double mean, double std) {
  // Box-Muller (no caching; fast enough here)
  double u1 = 0.0;
  while (u1 == 0.0) {
    u1 = rng.nextDouble();
  }
  final u2 = rng.nextDouble();
  final z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  return mean + std * z0;
}

int weightedChoiceFromProbs(Float64List probs, int off, int len, math.Random rng) {
  // probs sum to 1
  final r = rng.nextDouble();
  double c = 0.0;
  for (var i = 0; i < len; i++) {
    c += probs[off + i];
    if (r < c) return i;
  }
  return len - 1;
}

// -----------------------------
// Math kernels (typed, no alloc)
// -----------------------------
void matVecMul(
  Float64List w, // row-major [nOut, nIn]
  int nOut,
  int nIn,
  Float64List x,
  int xOff,
  Float64List y,
  int yOff,
) {
  for (var o = 0; o < nOut; o++) {
    final wOff = o * nIn;
    double s = 0.0;
    for (var i = 0; i < nIn; i++) {
      s += w[wOff + i] * x[xOff + i];
    }
    y[yOff + o] = s;
  }
}

void outerAccum(
  Float64List dW, // row-major [nOut, nIn]
  int nOut,
  int nIn,
  Float64List dy,
  int dyOff,
  Float64List x,
  int xOff,
) {
  for (var o = 0; o < nOut; o++) {
    final g = dy[dyOff + o];
    if (g == 0.0) continue;
    final wOff = o * nIn;
    for (var i = 0; i < nIn; i++) {
      dW[wOff + i] += g * x[xOff + i];
    }
  }
}

void matVecMulTAccum(
  Float64List w, // row-major [nOut, nIn]
  int nOut,
  int nIn,
  Float64List dy,
  int dyOff,
  Float64List dx,
  int dxOff,
) {
  // dx += W^T * dy
  for (var i = 0; i < nIn; i++) {
    double s = 0.0;
    for (var o = 0; o < nOut; o++) {
      s += w[o * nIn + i] * dy[dyOff + o];
    }
    dx[dxOff + i] += s;
  }
}

double rmsnormForward(
  Float64List x,
  int xOff,
  Float64List y,
  int yOff,
  int len,
) {
  double ms = 0.0;
  for (var i = 0; i < len; i++) {
    final v = x[xOff + i];
    ms += v * v;
  }
  ms /= len;
  final s = 1.0 / math.sqrt(ms + 1e-5);
  for (var i = 0; i < len; i++) {
    y[yOff + i] = x[xOff + i] * s;
  }
  return s;
}

void rmsnormBackwardAccum(
  Float64List x,
  int xOff,
  Float64List dy,
  int dyOff,
  double s,
  Float64List dx,
  int dxOff,
  int len,
) {
  // y = x*s, s = (mean(x^2)+eps)^(-1/2)
  // dx_i += dy_i*s - x_i*(s^3/len)*dot(dy, x)
  double dot = 0.0;
  for (var i = 0; i < len; i++) {
    dot += dy[dyOff + i] * x[xOff + i];
  }
  final s3 = s * s * s;
  final coeff = (s3 / len) * dot;
  for (var i = 0; i < len; i++) {
    dx[dxOff + i] += dy[dyOff + i] * s - x[xOff + i] * coeff;
  }
}

void softmaxInto(
  Float64List logits,
  int logOff,
  int len,
  Float64List probs,
  int probOff,
) {
  double maxv = logits[logOff];
  for (var i = 1; i < len; i++) {
    final v = logits[logOff + i];
    if (v > maxv) maxv = v;
  }
  double sumExp = 0.0;
  for (var i = 0; i < len; i++) {
    final e = math.exp(logits[logOff + i] - maxv);
    probs[probOff + i] = e;
    sumExp += e;
  }
  final inv = 1.0 / sumExp;
  for (var i = 0; i < len; i++) {
    probs[probOff + i] *= inv;
  }
}

// -----------------------------
// Adam update
// -----------------------------
void adamUpdate(
  Float64List w,
  Float64List dw,
  Float64List m,
  Float64List v,
  double lrT,
  double beta1,
  double beta2,
  double eps,
  double b1Pow,
  double b2Pow,
) {
  final oneMinusB1 = 1.0 - beta1;
  final oneMinusB2 = 1.0 - beta2;
  final invB1 = 1.0 / (1.0 - b1Pow);
  final invB2 = 1.0 / (1.0 - b2Pow);
  for (var i = 0; i < w.length; i++) {
    final g = dw[i];
    final mi = m[i] = beta1 * m[i] + oneMinusB1 * g;
    final vi = v[i] = beta2 * v[i] + oneMinusB2 * (g * g);
    final mHat = mi * invB1;
    final vHat = vi * invB2;
    w[i] -= lrT * mHat / (math.sqrt(vHat) + eps);
    dw[i] = 0.0;
  }
}

// -----------------------------
// Buffers (reused each step)
// -----------------------------
class Buffers {
  final int blockSize, nEmb, hidden, vocabSize, nHead, headDim;

  // activations (max blockSize)
  final Float64List x0; // [T, nEmb]
  final Float64List x1; // rmsnorm(x0)
  final Float64List x2; // rmsnorm(x1) (attn input)
  final Float64List q;  // [T, nEmb]
  final Float64List k;  // [T, nEmb]
  final Float64List v;  // [T, nEmb]
  final Float64List attnConcat; // [T, nEmb]
  final Float64List x3; // attn residual out
  final Float64List x4; // rmsnorm(x3) (mlp input)
  final Float64List h1; // [T, hidden]
  final Float64List relu; // [T, hidden]
  final Float64List fc2; // [T, nEmb]
  final Float64List x5; // [T, nEmb]
  final Float64List logits; // [T, vocab]
  final Float64List probs;  // [T, vocab]

  // rmsnorm scales
  final Float64List s0; // [T] for x0->x1
  final Float64List s1; // [T] for x1->x2
  final Float64List s2; // [T] for x3->x4

  // attention weights: [head, t, tau] stored in [nHead*blockSize*blockSize]
  final Float64List attnW;

  // grads (reused)
  final Float64List dLogits; // [T, vocab]
  final Float64List dX5;     // [T, nEmb]
  final Float64List dX3;     // [T, nEmb]
  final Float64List dAttnConcat; // [T, nEmb]
  final Float64List dRelu;   // [T, hidden] (also used as dH1 after masking)
  final Float64List dX4;     // [T, nEmb]
  final Float64List dQ;      // [T, nEmb]
  final Float64List dK;      // [T, nEmb]
  final Float64List dV;      // [T, nEmb]
  final Float64List dX2;     // [T, nEmb]
  final Float64List dX1;     // [T, nEmb]
  final Float64List dX0;     // [T, nEmb]

  // small temp
  final Float64List tmp; // [blockSize]

  Buffers({
    required this.blockSize,
    required this.nEmb,
    required this.hidden,
    required this.vocabSize,
    required this.nHead,
    required this.headDim,
  })  : x0 = Float64List(blockSize * nEmb),
        x1 = Float64List(blockSize * nEmb),
        x2 = Float64List(blockSize * nEmb),
        q = Float64List(blockSize * nEmb),
        k = Float64List(blockSize * nEmb),
        v = Float64List(blockSize * nEmb),
        attnConcat = Float64List(blockSize * nEmb),
        x3 = Float64List(blockSize * nEmb),
        x4 = Float64List(blockSize * nEmb),
        h1 = Float64List(blockSize * hidden),
        relu = Float64List(blockSize * hidden),
        fc2 = Float64List(blockSize * nEmb),
        x5 = Float64List(blockSize * nEmb),
        logits = Float64List(blockSize * vocabSize),
        probs = Float64List(blockSize * vocabSize),
        s0 = Float64List(blockSize),
        s1 = Float64List(blockSize),
        s2 = Float64List(blockSize),
        attnW = Float64List(nHead * blockSize * blockSize),
        dLogits = Float64List(blockSize * vocabSize),
        dX5 = Float64List(blockSize * nEmb),
        dX3 = Float64List(blockSize * nEmb),
        dAttnConcat = Float64List(blockSize * nEmb),
        dRelu = Float64List(blockSize * hidden),
        dX4 = Float64List(blockSize * nEmb),
        dQ = Float64List(blockSize * nEmb),
        dK = Float64List(blockSize * nEmb),
        dV = Float64List(blockSize * nEmb),
        dX2 = Float64List(blockSize * nEmb),
        dX1 = Float64List(blockSize * nEmb),
        dX0 = Float64List(blockSize * nEmb),
        tmp = Float64List(blockSize);

  void clearStep() {
    dLogits.fillRange(0, dLogits.length, 0.0);
    dX5.fillRange(0, dX5.length, 0.0);
    dX3.fillRange(0, dX3.length, 0.0);
    dAttnConcat.fillRange(0, dAttnConcat.length, 0.0);
    dRelu.fillRange(0, dRelu.length, 0.0);
    dX4.fillRange(0, dX4.length, 0.0);
    dQ.fillRange(0, dQ.length, 0.0);
    dK.fillRange(0, dK.length, 0.0);
    dV.fillRange(0, dV.length, 0.0);
    dX2.fillRange(0, dX2.length, 0.0);
    dX1.fillRange(0, dX1.length, 0.0);
    dX0.fillRange(0, dX0.length, 0.0);
  }
}

// -----------------------------
// Main
// -----------------------------
Future<void> main() async {
  // Same dataset behavior
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

  // Read docs
  final raw = await File(inputPath).readAsString();
  final docs = raw
      .trim()
      .split(RegExp(r'\r?\n'))
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toList();

  // RNG for this fast version (not Python-identical; reproducibility version below)
  final rng = math.Random(42);
  docs.shuffle(rng);
  print('num docs: ${docs.length}');

  // Tokenizer (char-level)
  final charSet = <String>{};
  for (final d in docs) {
    for (final r in d.runes) {
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

  // Pretokenize all docs for speed
  final docsTokens = <Int32List>[];
  for (final d in docs) {
    final runes = d.runes.toList();
    final t = Int32List(runes.length + 2);
    t[0] = BOS;
    for (var i = 0; i < runes.length; i++) {
      final ch = String.fromCharCode(runes[i]);
      t[i + 1] = stoi[ch]!;
    }
    t[t.length - 1] = BOS;
    docsTokens.add(t);
  }

  // Hyperparams (same as original)
  const nEmb = 16;
  const nHead = 4;
  const nLayer = 1; // this fast refactor implements nLayer=1 exactly
  const blockSize = 16;
  const headDim = nEmb ~/ nHead;
  const hidden = 4 * nEmb;

  if (nLayer != 1) {
    throw StateError('This fast refactor currently supports nLayer=1 only.');
  }

  // Params (Float64List row-major)
  Float64List initMat(int rows, int cols, {double std = 0.08}) {
    final w = Float64List(rows * cols);
    for (var i = 0; i < w.length; i++) {
      w[i] = gaussian(rng, 0.0, std);
    }
    return w;
  }

  // Embeddings & head
  final wte = initMat(vocabSize, nEmb);
  final wpe = initMat(blockSize, nEmb);
  final lm = initMat(vocabSize, nEmb);

  // Layer 0
  final wq = initMat(nEmb, nEmb);
  final wk = initMat(nEmb, nEmb);
  final wv = initMat(nEmb, nEmb);
  final wo = initMat(nEmb, nEmb);
  final w1 = initMat(hidden, nEmb);
  final w2 = initMat(nEmb, hidden);

  // Grad buffers for params
  final dwte = Float64List(wte.length);
  final dwpe = Float64List(wpe.length);
  final dlm = Float64List(lm.length);
  final dwq = Float64List(wq.length);
  final dwk = Float64List(wk.length);
  final dwv = Float64List(wv.length);
  final dwo = Float64List(wo.length);
  final dw1 = Float64List(w1.length);
  final dw2 = Float64List(w2.length);

  // Adam moments
  final mwte = Float64List(wte.length);
  final vwte = Float64List(wte.length);
  final mwpe = Float64List(wpe.length);
  final vwpe = Float64List(wpe.length);
  final mlm = Float64List(lm.length);
  final vlm = Float64List(lm.length);
  final mwq = Float64List(wq.length);
  final vwq = Float64List(wq.length);
  final mwk = Float64List(wk.length);
  final vwk = Float64List(wk.length);
  final mwv = Float64List(wv.length);
  final vwv = Float64List(wv.length);
  final mwo = Float64List(wo.length);
  final vwo = Float64List(wo.length);
  final mw1 = Float64List(w1.length);
  final vw1 = Float64List(w1.length);
  final mw2 = Float64List(w2.length);
  final vw2 = Float64List(w2.length);

  // Buffers
  final buf = Buffers(
    blockSize: blockSize,
    nEmb: nEmb,
    hidden: hidden,
    vocabSize: vocabSize,
    nHead: nHead,
    headDim: headDim,
  );

  // Adam hyperparams (same as original)
  const learningRate = 0.01;
  const beta1 = 0.85;
  const beta2 = 0.99;
  const epsAdam = 1e-8;

  // Training
  const numSteps = 1000;
  final invSqrtHead = 1.0 / math.sqrt(headDim.toDouble());

  for (var step = 0; step < numSteps; step++) {
    // zero param grads
    dwte.fillRange(0, dwte.length, 0.0);
    dwpe.fillRange(0, dwpe.length, 0.0);
    dlm.fillRange(0, dlm.length, 0.0);
    dwq.fillRange(0, dwq.length, 0.0);
    dwk.fillRange(0, dwk.length, 0.0);
    dwv.fillRange(0, dwv.length, 0.0);
    dwo.fillRange(0, dwo.length, 0.0);
    dw1.fillRange(0, dw1.length, 0.0);
    dw2.fillRange(0, dw2.length, 0.0);

    buf.clearStep();

    // pick doc
    final tokens = docsTokens[step % docsTokens.length];
    final n = math.min(blockSize, tokens.length - 1);

    // -----------------
    // Forward
    // -----------------
    // x0[t] = wte[token] + wpe[t]
    for (var t = 0; t < n; t++) {
      final tok = tokens[t];
      final wteOff = tok * nEmb;
      final wpeOff = t * nEmb;
      final x0Off = t * nEmb;
      for (var i = 0; i < nEmb; i++) {
        buf.x0[x0Off + i] = wte[wteOff + i] + wpe[wpeOff + i];
      }
      buf.s0[t] = rmsnormForward(buf.x0, x0Off, buf.x1, x0Off, nEmb); // x1 stored at same offsets
      buf.s1[t] = rmsnormForward(buf.x1, x0Off, buf.x2, x0Off, nEmb); // x2 at same offsets
    }

    // Q,K,V for all t
    for (var t = 0; t < n; t++) {
      final x2Off = t * nEmb;
      matVecMul(wq, nEmb, nEmb, buf.x2, x2Off, buf.q, x2Off);
      matVecMul(wk, nEmb, nEmb, buf.x2, x2Off, buf.k, x2Off);
      matVecMul(wv, nEmb, nEmb, buf.x2, x2Off, buf.v, x2Off);
    }

    // Attention (causal), store weights in buf.attnW and outputs in buf.attnConcat
    for (var t = 0; t < n; t++) {
      final qOffBase = t * nEmb;

      // for each head
      for (var h = 0; h < nHead; h++) {
        final hs = h * headDim;
        // compute logits tau=0..t into tmp
        double maxLogit = -1e30;
        for (var tau = 0; tau <= t; tau++) {
          final kOffBase = tau * nEmb;
          double dot = 0.0;
          for (var j = 0; j < headDim; j++) {
            dot += buf.q[qOffBase + hs + j] * buf.k[kOffBase + hs + j];
          }
          final logit = dot * invSqrtHead;
          buf.tmp[tau] = logit;
          if (logit > maxLogit) maxLogit = logit;
        }
        // softmax over tau=0..t
        double sumExp = 0.0;
        final wBase = (h * blockSize + t) * blockSize;
        for (var tau = 0; tau <= t; tau++) {
          final e = math.exp(buf.tmp[tau] - maxLogit);
          buf.attnW[wBase + tau] = e;
          sumExp += e;
        }
        final invSum = 1.0 / sumExp;
        for (var tau = 0; tau <= t; tau++) {
          buf.attnW[wBase + tau] *= invSum;
        }
        // head output: sum_tau weight * v[tau]
        final outOff = t * nEmb + hs;
        for (var j = 0; j < headDim; j++) {
          double s = 0.0;
          for (var tau = 0; tau <= t; tau++) {
            final vOff = tau * nEmb + hs + j;
            s += buf.attnW[wBase + tau] * buf.v[vOff];
          }
          buf.attnConcat[outOff + j] = s;
        }
      }

      // attn proj: wo * attnConcat
      matVecMul(wo, nEmb, nEmb, buf.attnConcat, t * nEmb, buf.x3, t * nEmb);
      // residual: x3 = attnProj + x1
      final x3Off = t * nEmb;
      final x1Off = t * nEmb;
      for (var i = 0; i < nEmb; i++) {
        buf.x3[x3Off + i] = buf.x3[x3Off + i] + buf.x1[x1Off + i];
      }

      // mlp norm
      buf.s2[t] = rmsnormForward(buf.x3, x3Off, buf.x4, x3Off, nEmb);

      // fc1 -> relu
      matVecMul(w1, hidden, nEmb, buf.x4, x3Off, buf.h1, t * hidden);
      final h1Off = t * hidden;
      for (var j = 0; j < hidden; j++) {
        final a = buf.h1[h1Off + j];
        buf.relu[h1Off + j] = a > 0.0 ? a : 0.0;
      }

      // fc2
      matVecMul(w2, nEmb, hidden, buf.relu, h1Off, buf.fc2, x3Off);

      // residual: x5 = fc2 + x3
      for (var i = 0; i < nEmb; i++) {
        buf.x5[x3Off + i] = buf.fc2[x3Off + i] + buf.x3[x3Off + i];
      }

      // logits
      matVecMul(lm, vocabSize, nEmb, buf.x5, x3Off, buf.logits, t * vocabSize);

      // probs
      softmaxInto(buf.logits, t * vocabSize, vocabSize, buf.probs, t * vocabSize);
    }

    // loss
    double loss = 0.0;
    for (var t = 0; t < n; t++) {
      final target = tokens[t + 1];
      final p = buf.probs[t * vocabSize + target];
      loss += -math.log(p);
    }
    loss /= n;

    // -----------------
    // Backward
    // -----------------
    // dLogits = (probs - onehot(target)) / n
    final invN = 1.0 / n;
    for (var t = 0; t < n; t++) {
      final target = tokens[t + 1];
      final base = t * vocabSize;
      for (var i = 0; i < vocabSize; i++) {
        double g = buf.probs[base + i];
        if (i == target) g -= 1.0;
        buf.dLogits[base + i] = g * invN;
      }
    }

    // logits = lm * x5
    // dlm += outer(dLogits, x5), dX5 += lm^T * dLogits
    for (var t = 0; t < n; t++) {
      final xOff = t * nEmb;
      final dlogOff = t * vocabSize;

      // dLM
      outerAccum(dlm, vocabSize, nEmb, buf.dLogits, dlogOff, buf.x5, xOff);

      // dX5
      matVecMulTAccum(lm, vocabSize, nEmb, buf.dLogits, dlogOff, buf.dX5, xOff);
    }

    // x5 = fc2 + x3  => dFc2 += dX5, dX3 += dX5
    for (var t = 0; t < n; t++) {
      final off = t * nEmb;
      for (var i = 0; i < nEmb; i++) {
        final g = buf.dX5[off + i];
        buf.dX3[off + i] += g; // residual to x3
        // fc2 grad uses same g; we just read buf.dX5 later as dfc2
      }
    }

    // fc2 = w2 * relu
    // dw2 += outer(dFc2, relu), dRelu += w2^T * dFc2
    for (var t = 0; t < n; t++) {
      final dfc2Off = t * nEmb;
      final reluOff = t * hidden;

      outerAccum(dw2, nEmb, hidden, buf.dX5, dfc2Off, buf.relu, reluOff);
      matVecMulTAccum(w2, nEmb, hidden, buf.dX5, dfc2Off, buf.dRelu, reluOff);

      // ReLU backward in-place: dRelu -> dH1
      final h1Off = t * hidden;
      for (var j = 0; j < hidden; j++) {
        if (buf.h1[h1Off + j] <= 0.0) {
          buf.dRelu[reluOff + j] = 0.0;
        }
      }
    }

    // fc1 = w1 * x4
    // dw1 += outer(dH1, x4), dX4 += w1^T * dH1
    for (var t = 0; t < n; t++) {
      final dh1Off = t * hidden;
      final x4Off = t * nEmb;

      outerAccum(dw1, hidden, nEmb, buf.dRelu, dh1Off, buf.x4, x4Off);
      matVecMulTAccum(w1, hidden, nEmb, buf.dRelu, dh1Off, buf.dX4, x4Off);
    }

    // x4 = rmsnorm(x3): add rmsnorm backward into dX3
    for (var t = 0; t < n; t++) {
      final off = t * nEmb;
      rmsnormBackwardAccum(buf.x3, off, buf.dX4, off, buf.s2[t], buf.dX3, off, nEmb);
    }

    // x3 = wo*attnConcat + x1
    // so dX1 += dX3
    for (var t = 0; t < n; t++) {
      final off = t * nEmb;
      for (var i = 0; i < nEmb; i++) {
        buf.dX1[off + i] += buf.dX3[off + i];
      }
    }

    // attnProj = wo * attnConcat
    // dwo += outer(dX3, attnConcat), dAttnConcat += wo^T * dX3
    for (var t = 0; t < n; t++) {
      final off = t * nEmb;
      outerAccum(dwo, nEmb, nEmb, buf.dX3, off, buf.attnConcat, off);
      matVecMulTAccum(wo, nEmb, nEmb, buf.dX3, off, buf.dAttnConcat, off);
    }

    // Attention backward -> dQ, dK, dV
    for (var t = 0; t < n; t++) {
      for (var h = 0; h < nHead; h++) {
        final hs = h * headDim;
        final dHeadOff = t * nEmb + hs;
        final wBase = (h * blockSize + t) * blockSize;

        // dWeight[tau] = dot(dHead, V[tau])
        double dotDwW = 0.0;
        for (var tau = 0; tau <= t; tau++) {
          double dw = 0.0;
          final vOff = tau * nEmb + hs;
          for (var j = 0; j < headDim; j++) {
            dw += buf.dAttnConcat[dHeadOff + j] * buf.v[vOff + j];
          }
          buf.tmp[tau] = dw;
          final w = buf.attnW[wBase + tau];
          dotDwW += dw * w;

          // dV[tau] += w * dHead
          for (var j = 0; j < headDim; j++) {
            buf.dV[vOff + j] += w * buf.dAttnConcat[dHeadOff + j];
          }
        }

        // dLogit[tau] = w * (dW - dot(dW,w))
        final qOff = t * nEmb + hs;
        for (var tau = 0; tau <= t; tau++) {
          final w = buf.attnW[wBase + tau];
          final dlogit = w * (buf.tmp[tau] - dotDwW);
          final kOff = tau * nEmb + hs;

          for (var j = 0; j < headDim; j++) {
            buf.dQ[qOff + j] += dlogit * buf.k[kOff + j] * invSqrtHead;
            buf.dK[kOff + j] += dlogit * buf.q[qOff + j] * invSqrtHead;
          }
        }
      }
    }

    // Q,K,V = Wq/Wk/Wv * x2
    // dw? += outer(d?, x2), dX2 += W?^T * d?
    for (var t = 0; t < n; t++) {
      final x2Off = t * nEmb;
      outerAccum(dwq, nEmb, nEmb, buf.dQ, x2Off, buf.x2, x2Off);
      outerAccum(dwk, nEmb, nEmb, buf.dK, x2Off, buf.x2, x2Off);
      outerAccum(dwv, nEmb, nEmb, buf.dV, x2Off, buf.x2, x2Off);

      matVecMulTAccum(wq, nEmb, nEmb, buf.dQ, x2Off, buf.dX2, x2Off);
      matVecMulTAccum(wk, nEmb, nEmb, buf.dK, x2Off, buf.dX2, x2Off);
      matVecMulTAccum(wv, nEmb, nEmb, buf.dV, x2Off, buf.dX2, x2Off);
    }

    // x2 = rmsnorm(x1) -> add into dX1
    for (var t = 0; t < n; t++) {
      final off = t * nEmb;
      rmsnormBackwardAccum(buf.x1, off, buf.dX2, off, buf.s1[t], buf.dX1, off, nEmb);
    }

    // x1 = rmsnorm(x0) -> into dX0
    for (var t = 0; t < n; t++) {
      final off = t * nEmb;
      rmsnormBackwardAccum(buf.x0, off, buf.dX1, off, buf.s0[t], buf.dX0, off, nEmb);
    }

    // x0 = wte[token] + wpe[pos]
    for (var t = 0; t < n; t++) {
      final tok = tokens[t];
      final wteOff = tok * nEmb;
      final wpeOff = t * nEmb;
      final dx0Off = t * nEmb;
      for (var i = 0; i < nEmb; i++) {
        final g = buf.dX0[dx0Off + i];
        dwte[wteOff + i] += g;
        dwpe[wpeOff + i] += g;
      }
    }

    // -----------------
    // Adam update
    // -----------------
    final lrT = learningRate * (1.0 - step / numSteps);
    final b1Pow = math.pow(beta1, step + 1).toDouble();
    final b2Pow = math.pow(beta2, step + 1).toDouble();

    adamUpdate(wte, dwte, mwte, vwte, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);
    adamUpdate(wpe, dwpe, mwpe, vwpe, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);
    adamUpdate(lm, dlm, mlm, vlm, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);

    adamUpdate(wq, dwq, mwq, vwq, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);
    adamUpdate(wk, dwk, mwk, vwk, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);
    adamUpdate(wv, dwv, mwv, vwv, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);
    adamUpdate(wo, dwo, mwo, vwo, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);

    adamUpdate(w1, dw1, mw1, vw1, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);
    adamUpdate(w2, dw2, mw2, vw2, lrT, beta1, beta2, epsAdam, b1Pow, b2Pow);

    final stepStr = (step + 1).toString().padLeft(4);
    final totalStr = numSteps.toString().padLeft(4);
    print('step $stepStr / $totalStr | loss ${loss.toStringAsFixed(4)}');
  }

  // -----------------
  // Inference (KV-cache incremental, still Float64List)
  // -----------------
  const temperature = 0.5;
  print('\n--- inference (new, hallucinated names) ---');

  // small temp vectors reused
  final x0v = Float64List(nEmb);
  final x1v = Float64List(nEmb);
  final x2v = Float64List(nEmb);
  final qv = Float64List(nEmb);
  final kv = Float64List(nEmb);
  final vv = Float64List(nEmb);
  final attnCv = Float64List(nEmb);
  final x3v = Float64List(nEmb);
  final x4v = Float64List(nEmb);
  final h1v = Float64List(hidden);
  final reluv = Float64List(hidden);
  final x5v = Float64List(nEmb);
  final logv = Float64List(vocabSize);
  final probv = Float64List(vocabSize);

  final kCache = Float64List(blockSize * nEmb);
  final vCache = Float64List(blockSize * nEmb);
  final attnTmp = Float64List(blockSize);

  double rmsnormForwardVec(Float64List x, Float64List y) {
    double ms = 0.0;
    for (var i = 0; i < x.length; i++) ms += x[i] * x[i];
    ms /= x.length;
    final s = 1.0 / math.sqrt(ms + 1e-5);
    for (var i = 0; i < x.length; i++) y[i] = x[i] * s;
    return s;
  }

  for (var sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    // reset cache
    kCache.fillRange(0, kCache.length, 0.0);
    vCache.fillRange(0, vCache.length, 0.0);

    var tokenId = BOS;
    final sb = StringBuffer();

    for (var pos = 0; pos < blockSize; pos++) {
      // x0 = wte[token] + wpe[pos]
      final wteOff = tokenId * nEmb;
      final wpeOff = pos * nEmb;
      for (var i = 0; i < nEmb; i++) {
        x0v[i] = wte[wteOff + i] + wpe[wpeOff + i];
      }
      rmsnormForwardVec(x0v, x1v);
      rmsnormForwardVec(x1v, x2v);

      // q,k,v
      // (reuse matVecMul with offsets by wrapping x arrays into a larger list is annoying,
      //  so do small manual multiplies here)
      void matVecSmall(Float64List W, int nOut, int nIn, Float64List x, Float64List y) {
        for (var o = 0; o < nOut; o++) {
          double s = 0.0;
          final wOff = o * nIn;
          for (var i = 0; i < nIn; i++) {
            s += W[wOff + i] * x[i];
          }
          y[o] = s;
        }
      }

      matVecSmall(wq, nEmb, nEmb, x2v, qv);
      matVecSmall(wk, nEmb, nEmb, x2v, kv);
      matVecSmall(wv, nEmb, nEmb, x2v, vv);

      // write cache at pos
      final kcOff = pos * nEmb;
      final vcOff = pos * nEmb;
      for (var i = 0; i < nEmb; i++) {
        kCache[kcOff + i] = kv[i];
        vCache[vcOff + i] = vv[i];
      }

      // attention over tau=0..pos
      for (var i = 0; i < nEmb; i++) attnCv[i] = 0.0;

      for (var h = 0; h < nHead; h++) {
        final hs = h * headDim;

        // logits
        double maxLogit = -1e30;
        for (var tau = 0; tau <= pos; tau++) {
          double dot = 0.0;
          final kOff = tau * nEmb + hs;
          for (var j = 0; j < headDim; j++) {
            dot += qv[hs + j] * kCache[kOff + j];
          }
          final logit = dot * invSqrtHead;
          attnTmp[tau] = logit;
          if (logit > maxLogit) maxLogit = logit;
        }

        double sumExp = 0.0;
        for (var tau = 0; tau <= pos; tau++) {
          final e = math.exp(attnTmp[tau] - maxLogit);
          attnTmp[tau] = e;
          sumExp += e;
        }
        final invSum = 1.0 / sumExp;
        for (var tau = 0; tau <= pos; tau++) {
          attnTmp[tau] *= invSum;
        }

        // head out
        for (var j = 0; j < headDim; j++) {
          double s = 0.0;
          for (var tau = 0; tau <= pos; tau++) {
            final vOff = tau * nEmb + hs + j;
            s += attnTmp[tau] * vCache[vOff];
          }
          attnCv[hs + j] = s;
        }
      }

      // wo proj + residual
      void matVecSmallOut(Float64List W, Float64List x, Float64List y) {
        for (var o = 0; o < nEmb; o++) {
          double s = 0.0;
          final wOff = o * nEmb;
          for (var i = 0; i < nEmb; i++) s += W[wOff + i] * x[i];
          y[o] = s;
        }
      }

      matVecSmallOut(wo, attnCv, x3v);
      for (var i = 0; i < nEmb; i++) {
        x3v[i] += x1v[i];
      }

      rmsnormForwardVec(x3v, x4v);

      // mlp
      // fc1
      for (var o = 0; o < hidden; o++) {
        double s = 0.0;
        final wOff = o * nEmb;
        for (var i = 0; i < nEmb; i++) s += w1[wOff + i] * x4v[i];
        h1v[o] = s;
        reluv[o] = s > 0.0 ? s : 0.0;
      }
      // fc2
      for (var o = 0; o < nEmb; o++) {
        double s = 0.0;
        final wOff = o * hidden;
        for (var i = 0; i < hidden; i++) s += w2[wOff + i] * reluv[i];
        x5v[o] = s + x3v[o];
      }

      // logits
      for (var o = 0; o < vocabSize; o++) {
        double s = 0.0;
        final wOff = o * nEmb;
        for (var i = 0; i < nEmb; i++) s += lm[wOff + i] * x5v[i];
        logv[o] = s / temperature;
      }
      // softmax probv
      double maxv = logv[0];
      for (var i = 1; i < vocabSize; i++) {
        if (logv[i] > maxv) maxv = logv[i];
      }
      double sumExp = 0.0;
      for (var i = 0; i < vocabSize; i++) {
        final e = math.exp(logv[i] - maxv);
        probv[i] = e;
        sumExp += e;
      }
      final invSum = 1.0 / sumExp;
      for (var i = 0; i < vocabSize; i++) probv[i] *= invSum;

      tokenId = weightedChoiceFromProbs(probv, 0, vocabSize, rng);
      if (tokenId == BOS) break;
      sb.write(uchars[tokenId]);
    }

    final idxStr = (sampleIdx + 1).toString().padLeft(2);
    print('sample $idxStr: ${sb.toString()}');
  }
}
