// USB rig for the Nidham web app: read the Teensy plugged into THIS computer (Web Serial,
// Chrome / Edge on a computer - phones and Firefox / Safari cannot). Same logic as the
// website's "USB rig" button (ahmedltaief.github.io, index.html).
//   nidham_phd_fw 1.x: 25 columns at 2 kHz  t_us, IMU1 acc/mag/gyro, IMU2 acc/mag/gyro, emg1..3, imu1_t_us, imu2_t_us, drop_count
//   older firmware:    23 columns at 1 kHz  (no imu*_t_us).  '#' lines = header / stat / gyro bias.
// Rows go to Dart at 100 Hz like the desktop app's broadcast:
//   [t_ms, emg1..3 (largest swing of each block), IMU1 acc xyz, gyro xyz, IMU2 acc xyz, gyro xyz (block means)]
// Dart gets JSON strings: {type:'rows', rows, fs, fw, drops, dropTotal, gaps, bytes, raw}
//                         {type:'log', msg}   {type:'closed', why}
(function () {
  'use strict';
  const IMU_IDX = [1, 2, 3, 7, 8, 9, 10, 11, 12, 16, 17, 18], EMG_IDX = [19, 20, 21];
  let u = null;

  function emit(cb, o) { try { cb(JSON.stringify(o)); } catch (e) { console.error(e); } }

  function block(B) {
    const k = B.length, row = [Math.round(B[0][0] / 1000)];
    for (const j of EMG_IDX) {
      let m = 0; for (const r of B) m += r[j]; m /= k;
      let best = B[0][j], bd = -1;
      for (const r of B) { const d = Math.abs(r[j] - m); if (d > bd) { bd = d; best = r[j]; } }
      row.push(Math.round(best));
    }
    IMU_IDX.forEach((j, i) => {
      let m = 0; for (const r of B) m += r[j]; m /= k;
      const s = i % 6 < 3 ? 1e3 : 1e4;
      row.push(Math.round(m * s) / s);
    });
    return row;
  }

  function onHash(ln) {
    const ver = ln.match(/^#\s*(\S+)\s+version=(\S+)/);
    if (ver) {                           // (re)start of the stream: the time jump before it is not lost data
      const fw = `${ver[1]} ${ver[2]}`, fs = (ln.match(/fs_emg=(\d+)/) || [])[1];
      if (fw !== u.fw) emit(u.cb, { type: 'log', msg: `firmware ${fw}` + (fs ? `, EMG ${fs} Hz` : '') });
      u.fw = fw; u.lastT = NaN;
    } else if (/gyro_bias/.test(ln)) emit(u.cb, { type: 'log', msg: 'gyro calibrated (keep the IMU still ~2 s after the Teensy starts)' });
  }

  function onRows(R) {
    const nc = R[0].length;
    if (nc !== u.nc) { u.nc = nc; u.fs = nc === 25 ? 2000 : 1000; u.lastT = NaN; u.blk = []; }
    const per = 1e6 / u.fs, D = u.fs / 100, out = [];
    for (const r of R) {
      if (r.length !== nc) continue;
      const t = r[0];
      if (isFinite(u.lastT) && t - u.lastT > 2.5 * per) {
        u.gaps++;
        if (Date.now() - u.gapLogAt > 2000) {
          u.gapLogAt = Date.now();
          emit(u.cb, { type: 'log', msg: `GAP - ~${Math.round((t - u.lastT) / per) - 1} samples missing (Teensy drop_count ${r[nc - 1]})` });
        }
      }
      u.lastT = t;
      const drop = r[nc - 1];
      if (u.dropBase === null || drop < u.dropBase) u.dropBase = drop;   // counter restarts with the Teensy
      u.dropTotal = drop; u.drops = drop - u.dropBase;
      u.raw++;
      u.blk.push(r);
      if (u.blk.length >= D) { out.push(block(u.blk)); u.blk = []; }
    }
    if (out.length) emit(u.cb, { type: 'rows', rows: out, fs: u.fs, fw: u.fw, drops: u.drops, dropTotal: u.dropTotal,
                                 gaps: u.gaps, bytes: u.bytes, raw: u.raw });
  }

  function onText(s) {
    const parts = (u.carry + s).split('\n');
    u.carry = parts.pop();
    if (u.carry.length > 4096) u.carry = '';
    const ok = [];
    for (let ln of parts) {
      ln = ln.trim();
      if (!ln) continue;
      if (ln[0] === '#') { onHash(ln); continue; }
      const v = ln.split(',');
      if (v.length !== 25 && v.length !== 23) continue;
      const n = v.map(Number);
      if (n.every(Number.isFinite)) ok.push(n);
    }
    if (ok.length) onRows(ok);
  }

  async function readLoop(me) {
    const dec = new TextDecoder();
    while (me.port.readable && !me.closing) {
      const reader = me.port.readable.getReader(); me.reader = reader;
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          me.bytes += value.length;
          onText(dec.decode(value, { stream: true }));
        }
      } catch (e) {
        if (!me.closing) emit(me.cb, { type: 'log', msg: 'read problem - ' + e.message });
      } finally { reader.releaseLock(); }
    }
    if (!me.closing) close('the Teensy was unplugged or restarted - press USB rig again');
  }

  async function close(why) {
    const me = u;
    if (!me) return;
    me.closing = true; u = null;
    try { if (me.reader) await me.reader.cancel(); } catch (e) {}
    try { await me.done; } catch (e) {}
    try { await me.port.close(); } catch (e) {}
    emit(me.cb, { type: 'closed', why: why || 'disconnected' });
  }

  // resolves with '' when open, or with the reason it could not open
  async function open(cb) {
    if (!('serial' in navigator)) return 'this browser cannot read USB - use Chrome or Edge on a computer';
    if (u) return '';
    let port;
    try { port = await navigator.serial.requestPort({ filters: [{ usbVendorId: 0x16C0 }] }); }   // PJRC (Teensy)
    catch (e) { return 'no Teensy chosen'; }
    try { await port.open({ baudRate: 115200, bufferSize: 1 << 20 }); }
    catch (e) { return 'could not open the Teensy - is the desktop app, MATLAB, the website or the Arduino serial monitor using it? (' + e.message + ')'; }
    u = { port, cb, reader: null, closing: false, fw: '', fs: 0, nc: 0, lastT: NaN, gaps: 0, dropBase: null, drops: 0,
          dropTotal: 0, raw: 0, bytes: 0, carry: '', blk: [], gapLogAt: 0 };
    try { await port.setSignals({ dataTerminalReady: true }); } catch (e) {}
    try { const w = port.writable.getWriter(); await w.write(new TextEncoder().encode('H')); w.releaseLock(); } catch (e) {}
    u.done = readLoop(u);
    return '';
  }

  if ('serial' in navigator) navigator.serial.addEventListener('disconnect', e => { if (u && e.target === u.port) close('the Teensy was unplugged'); });

  window.nidhamUsb = {
    supported: () => 'serial' in navigator,
    open,
    close: () => close('disconnected'),
  };
})();
