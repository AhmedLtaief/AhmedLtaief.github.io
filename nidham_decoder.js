/* nidham_decoder.js - the command decoder in the browser (12 Nov 2026).
   Line-by-line port of nidham_phd_fw/cmd_decoder.h (= nidham_desktop/nidham/rig_decoder.py = the Commands tab in
   Run mode). The model is the same bytes the desktop app writes (rig_decoder.export, file *.ndm).
   Checked against the Python spec on real runs: nidham_desktop/tests/test_js_decoder.py (runs this file in Node).
   Use:  const m = NidhamDecoder.loadModel(arrayBuffer);  const f = new NidhamDecoder.Front(m);
         events = f.push2k([emg1, emg2, emg3], [gx, gy, gz])   (2 kHz rows; or f.push1k(...) for 1 kHz rows)
         each event: {name:'PINCH_UP', key:'PAGE_UP', k, mod, conf}  */
(function (root) {
  'use strict';
  const KEYNAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'ENTER', 'BACK_SPACE', 'PAGE_UP', 'PAGE_DOWN', 'HOME', 'END', 'SPACE',
                    'ESCAPE', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'TAB', 'DELETE'];
  const NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'TWIST_CW', 'TWIST_CCW', 'C7', 'C8'];
  const EPS = 2.220446049250313e-16;

  function loadModel(buf) {
    const dv = new DataView(buf instanceof ArrayBuffer ? buf : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    let p = 0;
    const u8 = () => dv.getUint8(p++), u16 = () => { const v = dv.getUint16(p, true); p += 2; return v; };
    const i16 = () => { const v = dv.getInt16(p, true); p += 2; return v; }, f64 = () => { const v = dv.getFloat64(p, true); p += 8; return v; };
    if (String.fromCharCode(u8(), u8(), u8(), u8()) !== 'NDM1') throw new Error('not a Nidham model file');
    const m = { fs: u16(), nf: u16(), ncls: u16(), nm: u16(), nc: u16() };
    m.onThr = f64();
    for (const k of ['PRE_MS', 'FIST_PRE_MS', 'QUIET_AVG_MS', 'OFF_FRAC', 'QUIET_MS', 'RETURN_FRAC', 'MIN_ROT', 'MAX_MS', 'MIN_MS',
                     'REFRACT_MS', 'REJECT_CONF', 'MAX_ANGLE', 'D2MAX', 'SQ_MIN_MS', 'SQ_MAX_MS', 'SQ_OFF_FRAC', 'SQ_OFF_MS']) m[k] = f64();
    m.cls = []; for (let c = 0; c < m.ncls; c++) m.cls.push(i16());
    m.mu = []; for (let c = 0; c < m.ncls; c++) { const r = []; for (let j = 0; j < m.nf; j++) r.push(f64()); m.mu.push(r); }
    m.si = []; for (let i = 0; i < m.nf; i++) { const r = []; for (let j = 0; j < m.nf; j++) r.push(f64()); m.si.push(r); }
    m.ML = []; for (let r = 0; r < 3; r++) { const a = []; for (let k = 0; k < m.nm; k++) a.push(f64()); m.ML.push(a); }
    m.modC = []; for (let c = 0; c < m.nc; c++) { const b = []; for (let r = 0; r < 3; r++) { const a = []; for (let k = 0; k < m.nm; k++) a.push(f64()); b.push(a); } m.modC.push(b); }
    m.modMask = u8(); m.squeeze = u8();
    m.keys = []; for (let r = 0; r < 3; r++) { const a = []; for (let c = 0; c < m.nc; c++) a.push(u8()); m.keys.push(a); }
    m.sqKey = u8();
    if (p !== dv.byteLength) throw new Error('model file has the wrong size');
    // MS = Mu Si and the constant 0.5 sum(MS * Mu): fixed for a static model
    m.MS = m.mu.map(row => { const o = []; for (let j = 0; j < m.nf; j++) { let s = 0; for (let i = 0; i < m.nf; i++) s += row[i] * m.si[i][j]; o.push(s); } return o; });
    return m;
  }

  function pyround(x) { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); }
  const lgv = v => Number.isNaN(v) ? NaN : Math.log(v > 1 ? v : 1);
  const norm3 = a => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function median(v) {                   // numpy median
    const s = Float64Array.from(v).sort(), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
  }

  class Decoder {
    constructor(m) {
      this.M = m; this.ms = 1000 / m.fs; this.Dropped = 0;
      this.pre = Math.max(1, pyround(m.PRE_MS / 1000 * m.fs));
      this.preEN = Math.max(1, pyround(Math.max(m.PRE_MS, m.FIST_PRE_MS) / 1000 * m.fs));
      this.wN = Math.max(1, pyround(m.QUIET_AVG_MS / 1000 * m.fs));
      this.wHist = new Float64Array(this.wN); this.wSum = 0; this.wPos = 0;
      this.resetSeg(); this.newSq(false);
    }
    resetSeg() { this.active = false; this.G = []; this.E = []; this.quiet = 0; this.refract = 0; this.preG = []; this.preE = [];
                 this.E0 = 0; this.nPre = 0; this.theta = [0, 0, 0]; this.thetaMax = 0; }
    keepPre(g, e) {
      this.preG.push(g); if (this.preG.length > this.pre) this.preG.shift();
      this.preE.push(e); if (this.preE.length > this.preEN) this.preE.shift();
    }
    // Segmenter.push for one sample -> a cut motion {G, E, E0} or null
    pushOne(g, e) {
      const M = this.M, w = norm3(g), off = M.OFF_FRAC * M.onThr;
      if (this.refract > 0) { this.refract--; this.keepPre(g, e); return null; }
      if (!this.active) {
        if (w > M.onThr) {
          this.active = true; this.quiet = 0;
          this.G = this.preG.concat([g]); this.E = this.preE.concat([e]);
          this.E0 = this.preE.length; this.nPre = this.preG.length;
          this.theta = [0, 1, 2].map(j => this.G.reduce((s, x) => s + x[j], 0) / M.fs);
          this.thetaMax = norm3(this.theta);
        } else this.keepPre(g, e);
        return null;
      }
      this.G.push(g); this.E.push(e);
      for (let j = 0; j < 3; j++) this.theta[j] += g[j] / M.fs;
      this.thetaMax = Math.max(this.thetaMax, norm3(this.theta));
      const old = this.wHist[this.wPos]; this.wHist[this.wPos] = w; this.wPos = (this.wPos + 1) % this.wN; this.wSum += w - old;
      this.quiet = this.wSum / this.wN < off ? this.quiet + 1 : 0;
      const ms = this.ms, lenMs = (this.G.length - this.nPre) * ms, back = norm3(this.theta) <= M.RETURN_FRAC * this.thetaMax;
      if (this.quiet * ms >= M.QUIET_MS && this.thetaMax < M.MIN_ROT) { this.active = false; this.preG = []; this.preE = []; return null; }
      if ((this.quiet * ms >= M.QUIET_MS && back) || lenMs > M.MAX_MS) {
        const dur = lenMs - this.quiet * ms; let seg = null;
        if (dur >= M.MIN_MS && lenMs <= M.MAX_MS && this.thetaMax >= M.MIN_ROT)
          seg = { G: this.G.slice(0, this.G.length - this.quiet), E: this.E.slice(0, this.E.length - this.quiet), E0: this.E0 };
        else this.Dropped++;
        this.active = false; this.refract = pyround(M.REFRACT_MS / ms); this.preG = []; this.preE = [];
        return seg;
      }
      return null;
    }
    features(s) {
      const M = this.M, G = s.G, n = G.length, w = G.map(norm3), wmax = Math.max(...w);
      let a = 0; for (let i = 0; i < n; i++) if (w[i] > 0.5 * wmax) { a = i; break; }
      const d0 = G[a]; let b = n - 1;
      for (let i = a; i < n; i++) if (dot3(G[i], d0) < 0) { b = i - 1; break; }
      const lobe = [0, 0, 0], intg = [0, 0, 0]; let pk = 0;
      for (let i = a; i <= b; i++) { for (let j = 0; j < 3; j++) lobe[j] += G[i][j]; if (w[i] > pk) pk = w[i]; }
      for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) intg[j] += G[i][j];
      for (let j = 0; j < 3; j++) { lobe[j] /= M.fs; intg[j] /= M.fs; }
      const ln = Math.max(norm3(lobe), EPS);
      const x = [lobe[0] / ln, lobe[1] / ln, lobe[2] / ln, Math.log(Math.max(pk, EPS)), n / M.fs, ...intg, ...lobe];
      if (M.nf > 11) {
        const e0 = Math.max(1, s.E0);
        for (let m = 0; m < M.nm; m++) {
          let sum = 0; for (let i = e0 - 1; i < s.E.length; i++) sum += s.E[i][m];
          x.push(lgv(sum / (s.E.length - e0 + 1)) - lgv(median(s.E.slice(0, e0).map(r => r[m]))));
        }
      }
      return x;
    }
    level(s) {
      const a = Math.max(0, s.E0 - pyround(this.M.FIST_PRE_MS / 1000 * this.M.fs)), out = [];
      for (let m = 0; m < this.M.nm; m++) { const v = s.E.slice(a).map(r => r[m]); out.push(v.length ? median(v) : 0); }
      return out;
    }
    modOf(k, lv) {
      const M = this.M, z = lv.map(lgv); let best = Infinity, out = 0;
      for (let mm = 0; mm < 3; mm++) {
        if (!(M.modMask & (1 << mm))) continue;
        let c = M.modC[k - 1][mm];
        if (c.some(Number.isNaN)) c = M.ML[mm].map(lgv);
        if (c.some(Number.isNaN)) continue;
        let d = 0; for (let m = 0; m < M.nm; m++) d += (z[m] - c[m]) ** 2;
        d = Math.sqrt(d);
        if (d < best) { best = d; out = mm; }
      }
      return out;
    }
    decide(s) {
      const M = this.M, x = this.features(s), C = M.ncls, nf = M.nf, g = [];
      for (let c = 0; c < C; c++) {
        let s1 = 0, s2 = 0;
        for (let j = 0; j < nf; j++) { s1 += M.MS[c][j] * x[j]; s2 += M.MS[c][j] * M.mu[c][j]; }
        g.push(s1 - 0.5 * s2);
      }
      const gmax = Math.max(...g); let sum = 0; const post = g.map(v => { const e = Math.exp(v - gmax); sum += e; return e; });
      let cb = 0; for (let c = 0; c < C; c++) { post[c] /= sum; if (post[c] > post[cb]) cb = c; }
      const pmax = post[cb], e = x.map((v, j) => v - M.mu[cb][j]);
      let d2 = 0; for (let i = 0; i < nf; i++) { let t = 0; for (let j = 0; j < nf; j++) t += M.si[i][j] * e[j]; d2 += e[i] * t; }
      const mu3 = M.mu[cb].slice(0, 3), cs = Math.min(1, Math.max(-1, dot3(x, mu3) / Math.max(norm3(mu3), EPS)));
      const ang = Math.acos(cs) * 57.29577951308232, k = M.cls[cb];
      let why = 'ok';
      if (pmax < M.REJECT_CONF) why = 'not sure';
      else if (d2 > M.D2MAX) why = 'unlike any calibrated motion';
      else if (ang > M.MAX_ANGLE) why = 'direction off';
      if (why !== 'ok') return { k: 0, mod: 0, conf: pmax, why };
      const ff = (M.modMask & ~1) && k > 0 ? this.modOf(k, this.level(s)) : 0;
      return { k, mod: ff, conf: pmax, why };
    }
    newSq(on) { this.sq = { on, n: 0, low: 0, moved: false, pending: false, buf: [] }; }
    usedFist() { this.newSq(true); this.sq.moved = true; }
    activity(e) { let a = -Infinity; for (let m = 0; m < this.M.nm; m++) { const v = e[m] / Math.max(this.M.ML[0][m], 1); if (v > a || Number.isNaN(v)) a = v; } return a; }
    actThr() { const f = [this.activity(this.M.ML[1]), this.activity(this.M.ML[2])].filter(Number.isFinite); return Math.sqrt(Math.min(...f)); }
    staticMod(lv) {
      const d = [1, 2].map(r => { let s = 0; for (let m = 0; m < this.M.nm; m++) s += (lgv(lv[m]) - lgv(this.M.ML[r][m])) ** 2; return Number.isFinite(s) ? Math.sqrt(s) : Infinity; });
      return d[1] < d[0] ? 2 : 1;
    }
    sqStep(E, real, act, on) {
      const M = this.M;
      if (!on || M.ML[2].some(Number.isNaN)) { this.newSq(false); return false; }
      const thr = this.actThr(); let sq = this.sq, fired = false;
      if (real) sq.moved = true;
      if (sq.pending) { if (real) sq.pending = false; else if (!act) { sq.pending = false; return true; } }
      for (const e of E) {
        const a = this.activity(e);
        if (!sq.on) { if (a > thr) { this.newSq(true); sq = this.sq; sq.moved = real; } continue; }
        sq.n++; sq.buf.push(e);
        if (sq.buf.length > 6 * M.fs) sq.buf = sq.buf.slice(-3 * M.fs);
        sq.low = a < M.SQ_OFF_FRAC * thr ? sq.low + 1 : 0;
        if (sq.low * this.ms >= M.SQ_OFF_MS) {
          const dur = (sq.n - sq.low) * this.ms; sq.on = false;
          const nh = Math.max(1, sq.buf.length - sq.low), held = [];
          for (let m = 0; m < M.nm; m++) held.push(median(sq.buf.slice(0, nh).map(r => r[m])));
          if (sq.moved) {} else if (dur < M.SQ_MIN_MS || dur > M.SQ_MAX_MS) {} else if (this.staticMod(held) !== 2) {}
          else if (act) sq.pending = true; else fired = true;
        }
      }
      return fired;
    }
    // one chunk: G [[gx,gy,gz]...] offset removed, E [[e1,e2..]...] envelopes -> events
    process(G, E) {
      const out = [], busy0 = this.active, d0 = this.Dropped; let nseg = 0;
      for (let i = 0; i < G.length; i++) {
        const s = this.pushOne(G[i], E[i]);
        if (!s) continue;
        nseg++;
        const ev = this.decide(s); out.push(ev);
        if (ev.k > 0) this.usedFist();
      }
      if (this.sqStep(E, nseg > 0 || this.Dropped > d0, busy0 || this.active, this.M.squeeze)) {
        out.push({ k: this.M.nc + 1, mod: 0, conf: -1, why: 'ok' }); this.usedFist();
      }
      return out.map(ev => this.named(ev));
    }
    named(ev) {
      const M = this.M;
      if (ev.k === 0) return Object.assign({ name: 'reject', key: null }, ev);
      if (ev.k > M.nc) return Object.assign({ name: 'SQUEEZE', key: KEYNAMES[M.sqKey] }, ev);
      return Object.assign({ name: ['', 'PINCH_', 'FIST_'][ev.mod] + NAMES[ev.k - 1], key: KEYNAMES[M.keys[ev.mod][ev.k - 1]] }, ev);
    }
  }

  class Envelope {                        // dsp.py CmdEmg
    constructor(fs, tau = 0.05) {
      const wc = Math.tan(Math.PI * 20 / fs), k = 1 + Math.SQRT2 * wc + wc * wc;
      this.b0 = 1 / k; this.b1 = -2 / k; this.b2 = 1 / k; this.a1 = 2 * (wc * wc - 1) / k; this.a2 = (1 - Math.SQRT2 * wc + wc * wc) / k;
      this.al = 1 - Math.exp(-1 / (tau * fs)); this.z0 = this.z1 = this.msq = 0; this.first = true;
    }
    push(x) {
      if (this.first) { this.p2 = this.p1 = x; this.first = false; }
      const lo = this.p2, mid = this.p1, hi = x, r = mid - (lo + hi) / 2;
      const z = Math.abs(r) > 60 && Math.abs(hi - lo) < Math.abs(r) / 3 ? (lo + hi) / 2 : mid;
      this.p2 = this.p1; this.p1 = x;
      const h = this.b0 * z + this.z0; this.z0 = this.b1 * z - this.a1 * h + this.z1; this.z1 = this.b2 * z - this.a2 * h;
      this.msq = this.al * h * h + (1 - this.al) * this.msq;
      return Math.sqrt(Math.max(this.msq, 0));
    }
  }

  class Front {                           // 2 kHz rows -> 1 kHz pairs -> envelopes, gyro offset -> 70-row chunks -> Decoder
    constructor(m, chunk = 70) {
      this.M = m; this.dec = new Decoder(m); this.chunk = chunk; this.bias = [0, 0, 0];
      this.env = []; for (let i = 0; i < m.nm; i++) this.env.push(new Envelope(m.fs));
      this.half = null; this.G = []; this.E = [];
    }
    push2k(emg, gyro) {
      if (!this.half) { this.half = emg.slice(0, this.M.nm); return []; }
      const a = this.half; this.half = null;
      return this.push1k(a.map((v, m) => (v + emg[m]) / 2), gyro);
    }
    push1k(emg, gyro) {
      this.G.push([gyro[0] - this.bias[0], gyro[1] - this.bias[1], gyro[2] - this.bias[2]]);
      this.E.push(this.env.map((en, m) => en.push(emg[m])));
      if (this.G.length < this.chunk) return [];
      const out = this.dec.process(this.G, this.E);
      let mx = 0; const s = [0, 0, 0];
      for (const g of this.G) { mx = Math.max(mx, norm3(g)); for (let j = 0; j < 3; j++) s[j] += g[j]; }
      if (mx <= 0.4 * this.M.onThr) { const a = Math.min(1, this.G.length / (10 * this.M.fs)); for (let j = 0; j < 3; j++) this.bias[j] += a * s[j] / this.G.length; }
      this.G = []; this.E = [];
      return out;
    }
  }

  const api = { loadModel, Decoder, Envelope, Front, KEYNAMES };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.NidhamDecoder = api;
})(typeof window !== 'undefined' ? window : globalThis);
