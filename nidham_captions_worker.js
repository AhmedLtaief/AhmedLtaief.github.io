// Nidham live captions - Whisper runs HERE, in a worker inside the browser.
// Audio never leaves the device: only the model files are downloaded (once, then cached by the browser).
// Messages in:  {type:'load', model, device, dtype} | {type:'run', id, audio(Float32Array 16 kHz), lang, final}
// Translation (opus-mt, CPU): {type:'loadMt', model} -> {type:'mtReady', model, ms, error?}; {type:'mt', id, steps, text} -> {type:'mtText', id, text, ms, error?}
// Messages out: {type:'progress', loaded, total, pct} | {type:'ready', device, ms} | {type:'text', id, text, ms, final}
//               {type:'error', msg, id?}
const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
let asr = null, busy = false;
const mts = {};                                  // translation pipelines by model id

self.onmessage = async e => {
  const m = e.data;
  if (m.type === 'load') {
    try {
      const { pipeline, env } = await import(TJS);
      env.allowLocalModels = false;            // models only from the Hugging Face hub, kept in the browser cache
      const t0 = performance.now();
      asr = await pipeline('automatic-speech-recognition', 'onnx-community/whisper-' + m.model, {
        device: m.device, dtype: m.dtype,
        progress_callback: p => {
          if (p.status === 'progress_total')
            self.postMessage({ type: 'progress', loaded: p.loaded, total: p.total, pct: p.progress });
        }
      });
      // first run compiles the GPU shaders / wasm: do it now on 1 s of silence, not on the first sentence
      await asr(new Float32Array(16000), { language: 'english', task: 'transcribe' });
      self.postMessage({ type: 'ready', device: m.device, ms: Math.round(performance.now() - t0) });
    } catch (err) {
      asr = null;
      self.postMessage({ type: 'error', msg: String(err && err.message || err) });
    }
  } else if (m.type === 'loadMt') {             // translation model (second worker): {type:'loadMt', model:'Xenova/opus-mt-en-ar'}
    try {
      const { pipeline, env } = await import(TJS);
      env.allowLocalModels = false;
      const t0 = performance.now();
      if (!mts[m.model]) mts[m.model] = await pipeline('translation', m.model, {
        device: 'wasm', dtype: 'q8',
        progress_callback: p => {
          if (p.status === 'progress_total')
            self.postMessage({ type: 'progress', what: 'translation', model: m.model, loaded: p.loaded, total: p.total, pct: p.progress });
        }
      });
      self.postMessage({ type: 'mtReady', model: m.model, ms: Math.round(performance.now() - t0) });
    } catch (err) {
      self.postMessage({ type: 'mtReady', model: m.model, error: String(err && err.message || err) });
    }
  } else if (m.type === 'mt') {                 // {type:'mt', id, steps:[{model, prefix}], text}: one step, or two through English
    try {
      const t0 = performance.now();
      // one sentence at a time: with two sentences in one call the model often drops one (tested 13 Nov)
      const parts = m.text.match(/[^.!?؟]+[.!?؟]*\s*/g) || [m.text], done = [];
      for (const part of parts) {
        let text = part.trim();
        if (!text) continue;
        for (const st of m.steps) {
          const out = await mts[st.model]((st.prefix || '') + text, { max_new_tokens: 200 });
          text = (out && out[0] && out[0].translation_text || '').trim();
        }
        done.push(text);
      }
      const text = done.join(' ');
      self.postMessage({ type: 'mtText', id: m.id, text, ms: Math.round(performance.now() - t0) });
    } catch (err) {
      self.postMessage({ type: 'mtText', id: m.id, error: String(err && err.message || err) });
    }
  } else if (m.type === 'run') {
    if (!asr || busy) { self.postMessage({ type: 'text', id: m.id, text: null, final: m.final, skipped: true }); return; }
    busy = true;
    try {
      const t0 = performance.now();
      const out = await asr(m.audio, {
        language: m.lang, task: 'transcribe',
        no_repeat_ngram_size: 4,               // stops "the the the the" loops on noise
        max_new_tokens: 120
      });
      self.postMessage({ type: 'text', id: m.id, text: (out && out.text || '').trim(), final: m.final,
                         ms: Math.round(performance.now() - t0), sec: m.audio.length / 16000 });
    } catch (err) {
      self.postMessage({ type: 'error', id: m.id, msg: String(err && err.message || err) });
    }
    busy = false;
  }
};
