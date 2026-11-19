/* Nidham live captions: microphone -> Whisper (in a worker, in this browser) -> text.
   Audio never leaves the device. Model files come once from huggingface.co and stay in the browser cache.
   Used by the website (index.html, Conversation tab) and the phone app's web build (lib/screens/conversation_screen.dart).

   How "near live" works: the mic is cut into speech pieces by loudness (a pause of ~0.7 s ends a piece,
   a piece is never longer than 12 s). While someone speaks, the current piece is re-read every ~1 s and shown
   grey ("partial"); when the piece ends it is read once more and becomes a fixed line ("line").
   This avoids the doubled words that fixed 3-5 s chunks with overlap give.

   window.NidhamCaptions:
     LANGS, MODELS
     plan(model)          -> Promise<{device, mb, cached}>   size BEFORE downloading
     load(model)          -> Promise<{device, ms}>            downloads (progress events) + warms up
     start(lang)          -> Promise<''|error>                opens the mic
     pause() / resume() / stop()
     setLang(lang)
     transcribe(buf, lang)-> Promise<{text, ms, sec}>         one file / Float32Array at 16 kHz (tests)
     feed(buf, lang, speed) -> Promise                        play a file through the live path (tests)
     on(fn)  / onJson(fn)                                     events: {type, ...}
     READ, setTarget(code)                                    language the READER wants ('same' = no translation)
     planMt(from, to)     -> Promise<{mb, cached, viaEnglish, ready}>
     loadMt(from, to)     -> Promise<{ms}>                    downloads the translation model(s) for that pair
     translate(text, from, to) -> Promise<{text, ms}>
   Events: state {state, msg} | progress {what:'speech'|'translation', pct, loaded, total} | level {rms, speaking}
           partial {text} | line {id, text, ms, sec, lag, translating, to} | translated {id, text, ms, lag, to, error?} | info {msg}                                          */
(() => {
  const SRC = document.currentScript && document.currentScript.src;
  const SR = 16000, FRAME = 1600;                         // 100 ms frames at 16 kHz
  const PAUSE_MS = 700, MAX_SEG_S = 12, MIN_SPEECH_MS = 300, PARTIAL_MS = 1000, PREROLL = 3;

  const LANGS = [
    { code: 'en', whisper: 'english', label: 'English' },
    { code: 'ar', whisper: 'arabic', label: 'Arabic' },
    { code: 'aeb', whisper: 'arabic', label: 'Tunisian (darija)',
      note: 'Whisper has no Tunisian: it is read as Arabic. Darija, and darija mixed with French, comes out weaker than the other languages.' },
    { code: 'fr', whisper: 'french', label: 'French' },
  ];
  // file sizes on huggingface.co/onnx-community/whisper-* (MB, checked 12 Nov 2026)
  const MB = {
    tiny:  { q8: [10.1, 30.7], fp32: [32.9, 0], fp16: [16.5, 0], q4: [0, 86.7] },
    base:  { q8: [23.2, 53.7], fp32: [82.5, 0], fp16: [41.3, 0], q4: [0, 123.6] },
    small: { q8: [92.3, 156.8], fp32: [352.8, 0], fp16: [176.6, 0], q4: [0, 233.1] },
    'large-v3-turbo': { q8: [644.8, 439.9], q4: [424.9, 334.1], q4f16: [370.0, 193.5] },   // checked 13 Nov 2026
  };
  const MODELS = [
    { id: 'tiny', label: 'tiny - fastest, weakest' },
    { id: 'base', label: 'base - good balance' },
    { id: 'small', label: 'small - better, big download, slow without a GPU' },
    { id: 'large-v3-turbo', label: 'large-v3-turbo - best offline (accents, slang), ~560 MB, needs a GPU' },
    { id: 'online', label: 'Online - best quality (slang, darija), needs internet, costs money' },
  ];
  const RUNTIME_MB = 10;                                  // onnxruntime wasm (5.5 MB) + tokenizer (2.8 MB) + library, first time only

  let listeners = [], worker = null, device = '', model = '', ready = false, loading = null;
  let lang = LANGS[0], state = 'idle';
  let ctx = null, stream = null, node = null, src = null, wake = null, paused = false;
  const emit = ev => listeners.forEach(f => { try { f(ev); } catch (e) { console.error(e); } });
  const setState = (s, msg) => { state = s; emit({ type: 'state', state: s, msg: msg || '' }); };

  /* ---------- device + size ---------- */
  let gpuInfo = null;
  async function gpu() {
    if (gpuInfo) return gpuInfo;
    gpuInfo = { ok: false, f16: false };
    try {
      const a = navigator.gpu && await navigator.gpu.requestAdapter();
      if (a) gpuInfo = { ok: true, f16: a.features.has('shader-f16') };
    } catch (e) {}
    return gpuInfo;
  }
  function dtypeFor(dev, f16, m) {
    if (m === 'large-v3-turbo' && dev === 'webgpu')       // fp16 encoder alone would be 1.3 GB: 4-bit both halves
      return f16 ? { encoder_model: 'q4f16', decoder_model_merged: 'q4f16' } : { encoder_model: 'q4', decoder_model_merged: 'q4' };
    return dev === 'webgpu' ? { encoder_model: f16 ? 'fp16' : 'fp32', decoder_model_merged: 'q4' }
                            : { encoder_model: 'q8', decoder_model_merged: 'q8' };
  }
  const SUFFIX = { q8: '_quantized', fp32: '', fp16: '_fp16', q4: '_q4', q4f16: '_q4f16' };
  async function isCached(m, dt) {
    try {
      const c = await caches.open('transformers-cache'), keys = (await c.keys()).map(r => r.url);
      const has = f => keys.some(u => u.includes('/onnx-community/whisper-' + m + '/') && u.endsWith('/onnx/' + f + '.onnx'));
      return has('encoder_model' + SUFFIX[dt.encoder_model]) && has('decoder_model_merged' + SUFFIX[dt.decoder_model_merged]);
    } catch (e) { return false; }
  }
  async function plan(m, forceDev) {
    if (m === 'online') return { device: 'cloud', online: true, mb: 0, runtimeMb: 0, cached: true, ready: !!cloud };
    const g = await gpu(), dev = forceDev || (g.ok ? 'webgpu' : 'wasm'), dt = dtypeFor(dev, g.f16, m), t = MB[m];
    const mb = Math.round(t[dt.encoder_model][0] + t[dt.decoder_model_merged][1]);
    return { device: dev, dtype: dt, mb, runtimeMb: RUNTIME_MB, cached: await isCached(m, dt),
             warn: m === 'large-v3-turbo' && dev !== 'webgpu' ? 'No GPU here: this model is too big and too slow on the CPU. Pick base or small.' : '' };
  }

  /* ---------- worker ---------- */
  let pending = new Map(), nextId = 1;
  function newWorker() {
    const w = new Worker(new URL('nidham_captions_worker.js?v=5', SRC || location.href), { type: 'module' });
    w.onmessage = e => onWorker(e.data);
    w.onerror = e => emit({ type: 'info', msg: 'worker error: ' + (e.message || e) });
    return w;
  }
  function onWorker(m) {
    if (m.type === 'progress') emit({ type: 'progress', what: 'speech', pct: m.pct, loaded: m.loaded, total: m.total });
    else if (m.type === 'ready' || (m.type === 'error' && m.id === undefined)) {
      const p = pending.get('load'); pending.delete('load'); p && p(m);
    } else if (m.type === 'text' || m.type === 'error') {
      const p = pending.get(m.id); pending.delete(m.id); p && p(m);
    }
  }
  function ask(audio, final) {
    return new Promise(res => {
      const id = nextId++; pending.set(id, res);
      worker.postMessage({ type: 'run', id, audio, lang: lang.whisper, final });
    });
  }

  async function load(m) {
    if (ready && m === model) return { device, ms: 0 };
    if (m === 'online') {                                 // nothing to download: each speech piece goes to the server
      if (!cloud) throw new Error('online mode needs you to be logged in');
      if (worker) { worker.terminate(); worker = null; }
      model = m; device = 'cloud'; ready = true; setState(stream ? (paused ? 'paused' : 'listening') : 'ready');
      return { device, ms: 0 };
    }
    if (loading) return loading;
    loading = (async () => {
      if (worker) { worker.terminate(); worker = null; }
      ready = false; model = m;
      let p = await plan(m);
      for (;;) {
        setState('loading', `whisper-${m} on ${p.device === 'webgpu' ? 'GPU (WebGPU)' : 'CPU (wasm)'}`);
        worker = newWorker();
        const r = await new Promise(res => { pending.set('load', res); worker.postMessage({ type: 'load', model: m, device: p.device, dtype: p.dtype }); });
        if (r.type === 'ready') { device = p.device; ready = true; setState(stream ? (paused ? 'paused' : 'listening') : 'ready'); return { device, ms: r.ms }; }
        worker.terminate(); worker = null;
        // CPU retry only for a real GPU problem: not after a network error, and never for turbo (1.1 GB on the CPU)
        if (p.device === 'webgpu' && m !== 'large-v3-turbo' && !/fetch|network/i.test(r.msg)) { emit({ type: 'info', msg: 'GPU failed (' + r.msg + ') - trying CPU' }); p = await plan(m, 'wasm'); continue; }
        setState('error', r.msg); throw new Error(r.msg);
      }
    })();
    try { return await loading; } finally { loading = null; }
  }

  /* ---------- translation: the reader picks the language they read (opus-mt models, own worker, CPU) ----------
     en<->ar and en<->fr are one model each (~113 MB). ar<->fr has no browser model: it goes through English (2 models).
     Tunisian speech is read as Arabic by Whisper, so it is translated from Arabic. Nothing writes darija: a Tunisian reader gets Arabic. */
  const READ = [
    { code: 'same', label: 'Same as the speaker (no translation)' },
    { code: 'en', label: 'English' },
    { code: 'ar', label: 'Arabic', note: 'Translations are in standard Arabic (no model writes Tunisian darija).' },
    { code: 'fr', label: 'French' },
    { code: 'aeb', label: 'Tunisian darija (online mode only)', note: 'Only the Online mode writes darija. Offline modes give standard Arabic.' },
  ];
  const MT_MB = 113;                                      // encoder 51.8 + decoder 59.1 + tokenizer, per language pair (q8)
  let target = 'same', mtWorker = null, mtLoaded = new Set(), mtPending = new Map(), mtLoading = null;
  const baseLang = c => c === 'aeb' ? 'ar' : c;
  function route(from, to) {
    from = baseLang(from); to = to && baseLang(to);
    if (!to || to === 'same' || from === to) return [];
    // (tested 13 Nov: the >>ara<< target token is NOT understood by Xenova/opus-mt-en-ar - it came out as junk text)
    const one = (a, b) => ({ model: `Xenova/opus-mt-${a}-${b}`, prefix: '' });
    return from === 'en' || to === 'en' ? [one(from, to)] : [one(from, 'en'), one('en', to)];
  }
  async function mtCached(model) {
    try {
      const c = await caches.open('transformers-cache'), keys = (await c.keys()).map(r => r.url);
      return keys.some(u => u.includes('/' + model + '/') && u.endsWith('/onnx/encoder_model_quantized.onnx')) &&
             keys.some(u => u.includes('/' + model + '/') && u.endsWith('/onnx/decoder_model_merged_quantized.onnx'));
    } catch (e) { return false; }
  }
  async function planMt(from, to) {                       // size BEFORE downloading
    const steps = route(from, to), miss = [];
    for (const st of steps) if (!mtLoaded.has(st.model) && !(await mtCached(st.model))) miss.push(st.model);
    return { steps: steps.length, viaEnglish: steps.length === 2, mb: miss.length * MT_MB, cached: miss.length === 0,
             ready: steps.every(st => mtLoaded.has(st.model)) };
  }
  function mtNewWorker() {
    const w = new Worker(new URL('nidham_captions_worker.js?v=5', SRC || location.href), { type: 'module' });
    w.onmessage = e => {
      const m = e.data;
      if (m.type === 'progress') emit({ type: 'progress', what: 'translation', pct: m.pct, loaded: m.loaded, total: m.total });
      else if (m.type === 'mtReady' || m.type === 'mtText') { const k = m.type === 'mtReady' ? 'L' + m.model : m.id, p = mtPending.get(k); mtPending.delete(k); p && p(m); }
    };
    w.onerror = e => emit({ type: 'info', msg: 'translation worker error: ' + (e.message || e) });
    return w;
  }
  async function loadMt(from, to) {                       // downloads (progress events) the models this pair needs
    while (mtLoading) await mtLoading.catch(() => {});
    mtLoading = (async () => {
      const t0 = performance.now();
      for (const st of route(from, to)) {
        if (mtLoaded.has(st.model)) continue;
        if (!mtWorker) mtWorker = mtNewWorker();
        emit({ type: 'info', msg: 'loading translation ' + st.model.split('/')[1] });
        const r = await new Promise(res => { mtPending.set('L' + st.model, res); mtWorker.postMessage({ type: 'loadMt', model: st.model }); });
        if (r.error) throw new Error(r.error);
        mtLoaded.add(st.model);
      }
      return { ms: Math.round(performance.now() - t0) };
    })();
    try { return await mtLoading; } finally { mtLoading = null; }
  }
  let mtNext = 1;
  async function translate(text, from, to) {              // -> {text, ms} ; text unchanged when no translation is needed
    const steps = route(from, to);
    if (!steps.length) return { text, ms: 0 };
    if (!steps.every(st => mtLoaded.has(st.model))) throw new Error('translation model not loaded');
    const r = await new Promise(res => { const id = 'T' + mtNext++; mtPending.set(id, res); mtWorker.postMessage({ type: 'mt', id, steps, text }); });
    if (r.error) throw new Error(r.error);
    return { text: r.text, ms: r.ms };
  }
  function setTarget(code) { target = READ.some(r => r.code === code) ? code : 'same'; }

  /* ---------- speech pieces from the mic ---------- */
  let lineId = 1, seg = null, pre = [], hist = [], noise = 0.003, lastPartial = 0, finals = [], running = false, partialText = '';
  function frameIn(f) {                                   // f: Float32Array(FRAME)
    let s = 0; for (let i = 0; i < f.length; i++) s += f[i] * f[i];
    const rms = Math.sqrt(s / f.length);
    // room noise = quietest 100 ms of the last 5 s (speech always has small gaps; steady hiss / fan does not)
    hist.push(rms); if (hist.length > 50) hist.shift();
    noise = Math.min(...hist);
    const thr = Math.max(0.008, noise * 3);
    const loud = rms > thr;
    emit({ type: 'level', rms, speaking: !!seg });
    if (!seg) {
      pre.push(f); if (pre.length > PREROLL) pre.shift();
      if (loud) { seg = { frames: pre.slice(), speech: 1, quiet: 0, t0: performance.now() }; pre = []; }
      return;
    }
    seg.frames.push(f);
    if (loud) { seg.speech++; seg.quiet = 0; } else seg.quiet++;
    if (seg.quiet * 100 >= PAUSE_MS) endSeg(seg.frames.length - seg.quiet + 2);
    else if (seg.frames.length * FRAME / SR >= MAX_SEG_S) {
      // too long: cut at the quietest 100 ms of the last 2 s, keep the rest for the next piece
      let best = seg.frames.length - 1, bestE = Infinity;
      for (let i = seg.frames.length - 20; i < seg.frames.length; i++) {
        let e = 0; const g = seg.frames[i]; for (let j = 0; j < g.length; j += 4) e += g[j] * g[j];
        if (e < bestE) { bestE = e; best = i; }
      }
      const rest = seg.frames.slice(best + 1);
      endSeg(best + 1);
      seg = { frames: rest, speech: rest.length, quiet: 0, t0: performance.now() };
    } else if (model !== 'online' && performance.now() - lastPartial > PARTIAL_MS && seg.frames.length >= 10 && seg.quiet < 2) pump(true);   // no partial once a pause starts: the final line would wait for it
  }
  function join(frames) {
    const a = new Float32Array(frames.length * FRAME);
    frames.forEach((f, i) => a.set(f, i * FRAME)); return a;
  }
  function endSeg(n) {
    const s = seg; seg = null;
    if (s.speech * 100 >= MIN_SPEECH_MS) finals.push({ audio: join(s.frames.slice(0, n)), t0: s.t0, tEnd: performance.now() - s.quiet * 100 });   // lag counts from the last loud 100 ms
    else if (partialText) { partialText = ''; emit({ type: 'partial', text: '' }); }
    pump(false);
  }
  async function pump(wantPartial) {
    if (running || !ready) return;
    running = true;
    try {
      while (finals.length) {
        const f = finals.shift();
        if (model === 'online') { await cloudLine(f); continue; }
        const r = await ask(f.audio, true);
        const text = r.type === 'text' ? clean(r.text) : '';
        if (r.type === 'error') emit({ type: 'info', msg: 'whisper: ' + r.msg });
        partialText = ''; emit({ type: 'partial', text: '' });
        if (text) {
          const id = lineId++, steps = route(lang.code, target), can = steps.length > 0 && steps.every(st => mtLoaded.has(st.model));
          emit({ type: 'line', id, text, ms: r.ms, sec: +(r.sec || 0).toFixed(1), lag: Math.round(performance.now() - f.tEnd), lang: lang.code,
                 translating: can, to: steps.length ? target : '' });
          if (can) {
            const from = lang.code, to = target, tEnd = f.tEnd;
            translate(text, from, to).then(
              t => emit({ type: 'translated', id, text: t.text, ms: t.ms, to, lag: Math.round(performance.now() - tEnd) }),
              e => emit({ type: 'translated', id, text: '', error: e.message, to }));
          }
        }
      }
      if (wantPartial && seg && !paused) {
        lastPartial = performance.now();
        const r = await ask(join(seg.frames), false);
        const text = r.type === 'text' ? clean(r.text) : '';
        if (seg && text && !finals.length) { partialText = text; emit({ type: 'partial', text }); }
      }
    } finally { running = false; }
    if (finals.length) pump(false);
  }
  /* ---------- online mode: speech piece -> Nidham server (Supabase Edge Function 'captions') ----------
     The server sends the audio to a speech service and the text to Claude, which translates the MEANING
     (slang, idioms, Tunisian darija). Needs internet and a logged-in user. */
  let cloud = null, recent = [];
  function setCloud(c) { cloud = c && c.url ? c : null; }   // {url, key, token: string | () => Promise<string>}
  function wav(a) {                                       // Float32 16 kHz -> 16-bit PCM WAV
    const b = new ArrayBuffer(44 + a.length * 2), v = new DataView(b);
    const w = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + a.length * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, a.length * 2, true);
    for (let i = 0; i < a.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, a[i])) * 32767, true);
    return new Blob([b], { type: 'audio/wav' });
  }
  async function cloudLine(f) {
    const t0 = performance.now(), to = target, from = lang.code, wantMt = to !== 'same' && to !== from;
    try {
      const tok = typeof cloud.token === 'function' ? await cloud.token() : cloud.token;
      if (!tok) throw new Error('not logged in');
      const fd = new FormData();
      fd.append('audio', wav(f.audio), 'speech.wav'); fd.append('lang', from); fd.append('to', to);
      fd.append('prev', recent.slice(-3).join(' | '));
      const res = await fetch(cloud.url, { method: 'POST', body: fd, headers: { Authorization: 'Bearer ' + tok, apikey: cloud.key || '' } });
      const j = await res.json().catch(() => ({}));
      if (!res.ok || j.error) throw new Error(j.error || ('server ' + res.status));
      const text = clean(j.text);
      if (!text) return;
      recent.push(text); if (recent.length > 5) recent.shift();
      const id = lineId++, sec = +(f.audio.length / SR).toFixed(1);
      emit({ type: 'line', id, text, ms: j.sttMs || 0, sec, lag: Math.round(performance.now() - f.tEnd), lang: from,
             translating: false, to: wantMt ? to : '' });
      if (wantMt) emit({ type: 'translated', id, text: j.translation || '', ms: j.mtMs || 0, to, lag: Math.round(performance.now() - f.tEnd),
                         error: j.translation ? undefined : 'no translation' });
    } catch (e) {
      emit({ type: 'info', msg: 'online: ' + e.message + ` (${Math.round(performance.now() - t0)} ms)` });
    }
  }

  // Whisper invents these on noise / silence (from film subtitles): drop them
  const JUNK = [/^\W*$/, /^[\[(].*[\])]$/, /^(thank you|thanks)( (so much|very much|for watching))?[.!]*$/i, /^you\.?$/i,
    /^(merci|merci beaucoup)[.!]*$/i, /sous-titr/i, /amara\.org/i, /^(شكرا|شكراً)( لكم)?( على المشاهدة)?[.!]*$/,
    /ترجمة/, /اشتركوا? في القناة/, /^موسيقى$/, /^\.{2,}$/];
  function clean(t) {
    t = (t || '').replace(/\s+/g, ' ').trim();
    if (!t || JUNK.some(r => r.test(t))) return '';
    const w = t.split(' ');                               // "no no no no no no" style loops
    if (w.length >= 6 && new Set(w).size <= 2) return '';
    return t;
  }

  /* ---------- microphone ---------- */
  const WORKLET = `class P extends AudioWorkletProcessor{constructor(){super();this.b=[];}
    process(i){const c=i[0]&&i[0][0];if(c){this.b.push(new Float32Array(c));if(this.b.length>=8){this.port.postMessage(this.b);this.b=[];}}return true;}}
    registerProcessor('nidham-cap',P);`;
  let carry = new Float32Array(0), rate = SR;
  function audioIn(blocks) {
    if (paused) return;
    let n = carry.length; for (const b of blocks) n += b.length;
    let a = new Float32Array(n); a.set(carry); let o = carry.length;
    for (const b of blocks) { a.set(b, o); o += b.length; }
    if (rate !== SR) a = resample(a, rate);
    let i = 0;
    for (; i + FRAME <= a.length; i += FRAME) frameIn(a.slice(i, i + FRAME));
    carry = a.slice(i);
  }
  function resample(a, from) {                            // only if the browser refused a 16 kHz context
    const r = from / SR, out = new Float32Array(Math.floor(a.length / r));
    for (let i = 0; i < out.length; i++) { const x = i * r, k = Math.floor(x), t = x - k; out[i] = a[k] * (1 - t) + (a[k + 1] || 0) * t; }
    return out;
  }
  async function start(code) {
    setLang(code);
    if (stream) { resume(); return ''; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch (e) { setState(ready ? 'ready' : 'idle', 'microphone refused: ' + e.message); return 'microphone: ' + e.message; }
    try { ctx = new AudioContext({ sampleRate: SR }); } catch (e) { ctx = new AudioContext(); }
    rate = ctx.sampleRate;
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' })));
    src = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, 'nidham-cap');
    node.port.onmessage = e => audioIn(e.data);
    src.connect(node);                                    // not to the speakers
    paused = false; seg = null; pre = []; carry = new Float32Array(0);
    try { wake = await navigator.wakeLock.request('screen'); } catch (e) { wake = null; }
    setState('listening');
    return '';
  }
  function pause() { if (!stream) return; paused = true; seg = null; pre = []; setState('paused'); }
  function resume() { if (!stream) return; paused = false; setState('listening'); if (ctx.state === 'suspended') ctx.resume(); }
  function stop() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    try { src && src.disconnect(); node && node.disconnect(); } catch (e) {}
    try { ctx && ctx.close(); } catch (e) {}
    try { wake && wake.release(); } catch (e) {}
    stream = ctx = src = node = wake = null; seg = null; paused = false;
    setState(ready ? 'ready' : 'idle');
  }
  function setLang(code) { lang = LANGS.find(l => l.code === code) || lang; }

  /* ---------- tests: a recording through Whisper, or through the whole live path ---------- */
  async function decode(buf) {
    if (buf instanceof Float32Array) return buf;
    const c = new OfflineAudioContext(1, 1, SR);
    const ab = await c.decodeAudioData(buf.slice ? buf.slice(0) : buf);
    return ab.getChannelData(0);
  }
  async function transcribe(buf, code) {
    if (!ready) throw new Error('model not loaded');
    if (code) setLang(code);
    const a = await decode(buf);
    while (running) await new Promise(r => setTimeout(r, 50));
    running = true;
    try { const r = await ask(a, true); return { text: r.text, ms: r.ms, sec: +(a.length / SR).toFixed(1) }; }
    finally { running = false; }
  }
  async function feed(buf, code, speed) {                 // real-time playback of a file into the segmenter
    if (code) setLang(code);
    const a = await decode(buf), dt = 100 / (speed || 1);
    seg = null; pre = [];
    for (let i = 0; i + FRAME <= a.length; i += FRAME) { frameIn(a.slice(i, i + FRAME)); await new Promise(r => setTimeout(r, dt)); }
    for (let i = 0; i < 10; i++) { frameIn(new Float32Array(FRAME)); await new Promise(r => setTimeout(r, dt)); }
    while (running || finals.length) await new Promise(r => setTimeout(r, 50));
  }

  window.NidhamCaptions = {
    LANGS, MODELS, READ, setCloud, plan, load, planMt, loadMt, translate, setTarget, route, start, pause, resume, stop, setLang, transcribe, feed,
    get state() { return state; }, get target() { return target; }, get device() { return device; }, get model() { return model; }, get ready() { return ready; },
    supported: () => !!(window.Worker && window.AudioWorkletNode && navigator.mediaDevices && window.caches),
    on: fn => { listeners.push(fn); return () => { listeners = listeners.filter(f => f !== fn); }; },
    onJson: fn => { const g = ev => fn(JSON.stringify(ev)); listeners.push(g); },
  };
})();
