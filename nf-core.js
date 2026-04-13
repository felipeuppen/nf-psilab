 
 
  // ===================== Configuración general =====================
  var muse;
  let readEEGDataInterval;
  let voltageGraphUpdateInterval;
  let frequencyGraphUpdateInterval;

  const ELECTRODES = { TP9:0, FP1:1, FP2:2, TP10:3, AUX:4 };
  let activeElectrodes = { TP9:true, FP1:true, FP2:true, TP10:true, AUX:false };

  const FFT_SIZE = 256;   // Muse = 256 Hz
  const MAX_POINTS = 26;  // ~ 26 muestras suavizadas en la línea

  let eegBuffers = { TP9:[], FP1:[], FP2:[], TP10:[] };
  let eegFrequencyBuffer = { TP9:[], FP1:[], FP2:[], TP10:[] };

// ===================== Estado estable (sin baseline manual) =====================

// ventana de suavizado (segundos) para el ESTADO
const STATE_WINDOW_SEC = 8; // prueba 6–10
// tu tick actual de frecuenciaGraphUpdateInterval es 1000ms => 1 Hz
const STATE_TICK_HZ = 1;

// ring buffer por banda (guardamos log-powers)
let bandRing = { delta: [], theta: [], alpha: [], smr: [] };

// EMA para normalización (baseline automático)
const EMA_ALPHA = 0.06; // 0.04–0.10 (más bajo = más estable)
let ema = {
  attMean: 0, attVar: 1,
  calmMean: 0, calmVar: 1,
  init: false
};

function pushRing(arr, x, maxLen){
  arr.push(x);
  while (arr.length > maxLen) arr.shift();
}

function mean(arr){
  if (!arr.length) return 0;
  let s=0; for (let i=0;i<arr.length;i++) s += arr[i];
  return s / arr.length;
}

function emaUpdate(meanVarObj, x){
  // actualiza mean y var (var aproximada) tipo EWMA
  // var <- (1-a)*var + a*(x-mean)^2
  const a = EMA_ALPHA;
  const mPrev = meanVarObj.mean;
  const m = (1-a)*mPrev + a*x;
  const v = (1-a)*meanVarObj.var + a*(x - mPrev)*(x - mPrev);
  meanVarObj.mean = m;
  meanVarObj.var = Math.max(v, 1e-6);
}

function zFromEma(m, v, x){
  return (x - m) / Math.sqrt(v);
}

function clamp01(x){ return Math.max(0, Math.min(1, x)); }


const EPS = 1e-9;
const DELTA_WEIGHT = 0.7;     // mismo valor que en Python
const ATT_RATIO_TARGET = 0.35; // umbral: 1.0 = SMR igual a (Theta + w*Delta)
const ATT_FULL_SCALE  = 0.25;  // cuánto sobre el umbral “llena” el volumen (p.ej. 1.5 => volumen ~1)

const hpState = { TP9:{lp:0}, FP1:{lp:0}, FP2:{lp:0}, TP10:{lp:0} };

  let batteryIntervalId, statusIntervalId;

  // Montaje por celda (Bubble)
  let currentMount = null;
  function setMount(mountId){ currentMount = mountId ? String(mountId) : null; }
  function NF_getCanvas(baseId){
    if (!currentMount) return null;
    const id = `${baseId}-${currentMount}`;
    return document.getElementById(id) || document.getElementById(id + 'UI');
  }

  // Audio de NF
  let audioElement = new Audio('https://c86405ffad19a6265b95230b2818f733.cdn.bubble.io/f1694387578210x547236694268484700/Her%20Beautiful%20Hairs.mp3');
  audioElement.loop = true;

try {
  if (Chart?.register) {
    if (window.ChartAnnotation) {
      Chart.register(window.ChartAnnotation);
    } else if (window['chartjs-plugin-annotation']?.default) {
      Chart.register(window['chartjs-plugin-annotation'].default);
    } else if (window['chartjs-plugin-annotation']) {
      Chart.register(window['chartjs-plugin-annotation']);
    }
  }
} catch (e) {
  console.warn('No se pudo registrar chartjs-plugin-annotation:', e);
}

  let voltageChart, frequencyChart, stateChart;
  let neurofeedbackProtocol = 'none';
  let recording = false;
  let recordedData = [];

  // Sonidos de eventos
  let connectedSound = new Audio('https://c86405ffad19a6265b95230b2818f733.cdn.bubble.io/f1726653640774x758044899730232300/connected.wav');
  let lowBatterySoundPlayed = false;
  let lowBatterySound = new Audio('https://c86405ffad19a6265b95230b2818f733.cdn.bubble.io/f1726653628974x377035599482211460/lowBattery.wav');
  // Mejor usar un sonido distinto para desconexión súbita
  let suddenDisconnectSound = new Audio('https://c86405ffad19a6265b95230b2818f733.cdn.bubble.io/f1726653633045x802424070320823300/disconnected.wav');

  // ===================== FFT (arreglada) =====================
  class FFT {
    constructor(bufferSize, sampleRate) {
      this.bufferSize = bufferSize;
      this.sampleRate = sampleRate;
      this.spectrum = new Float32Array(bufferSize / 2);
      this.real = new Float32Array(bufferSize);
      this.imag = new Float32Array(bufferSize);
      this.reverseTable = new Uint32Array(bufferSize);
      this.sinTable = new Float32Array(bufferSize);
      this.cosTable = new Float32Array(bufferSize);

      // Tabla de bit-reversed
      let limit = 1;
      let bit = bufferSize >> 1;
      this.reverseTable[0] = 0;
      while (limit < bufferSize) {
        for (let i = 0; i < limit; i++) {
          this.reverseTable[i + limit] = this.reverseTable[i] + bit;
        }
        limit = limit << 1;
        bit = bit >> 1;
      }

      // Twiddle para cada halfSize: e^{-i * pi / halfSize}
      for (let i = 0; i < bufferSize; i++) {
        if (i === 0) { this.sinTable[0] = 0; this.cosTable[0] = 1; continue; }
        this.sinTable[i] = Math.sin(-Math.PI / i);
        this.cosTable[i] = Math.cos(-Math.PI / i);
      }
    }

    forward(buffer) {
      const { real, imag, reverseTable, sinTable, cosTable, spectrum, bufferSize } = this;
      if (buffer.length !== bufferSize) {
        throw new Error('Supplied buffer is not the same size as defined FFT. FFT Size: ' + bufferSize + ' Buffer Size: ' + buffer.length);
      }

      // Reordenar por bit-reversal
      for (let i = 0; i < bufferSize; i++) { real[i] = buffer[reverseTable[i]]; imag[i] = 0; }

      // Cooley–Tukey radix-2
      let halfSize = 1;
      while (halfSize < bufferSize) {
        let phaseShiftStepReal = cosTable[halfSize];
        let phaseShiftStepImag = sinTable[halfSize];
        let currentPhaseShiftReal = 1.0;
        let currentPhaseShiftImag = 0.0;

        for (let fftStep = 0; fftStep < halfSize; fftStep++) {
          for (let i = fftStep; i < bufferSize; i += (halfSize << 1)) {
            const off = i + halfSize;
            const tr = (currentPhaseShiftReal * real[off]) - (currentPhaseShiftImag * imag[off]);
            const ti = (currentPhaseShiftReal * imag[off]) + (currentPhaseShiftImag * real[off]);

            real[off] = real[i] - tr;
            imag[off] = imag[i] - ti;
            real[i] += tr;
            imag[i] += ti;
          }
          const tmpReal = currentPhaseShiftReal;
          // CORRECCIÓN: actualización compleja correcta
          currentPhaseShiftReal = (tmpReal * phaseShiftStepReal) - (currentPhaseShiftImag * phaseShiftStepImag);
          currentPhaseShiftImag = (tmpReal * phaseShiftStepImag) + (currentPhaseShiftImag * phaseShiftStepReal);
        }
        halfSize = halfSize << 1;
      }

      // Magnitud (amplitud normalizada)
      for (let i = 0; i < bufferSize / 2; i++) {
        spectrum[i] = 2 * Math.hypot(real[i], imag[i]) / bufferSize;
      }
    }
  }

  // ===================== Gráficos =====================
  function initializeCharts(){
    if (!currentMount) return;
const vEl = NF_getCanvas('voltageGraph');
const fEl = NF_getCanvas('frequencyGraph');
const sEl = NF_getCanvas('stateGraph');
if (!vEl && !fEl && !sEl) return;


// ===================== VOLTAJE (línea) =====================
if (vEl && document.body.contains(vEl)) {
  const prev = Chart.getChart ? Chart.getChart(vEl) : null; 
  if (prev) prev.destroy();
  const vCtx = vEl.getContext('2d');

        voltageChart = new Chart(vCtx, {
    type: 'line',
    data: {
      labels: Array(MAX_POINTS).fill(0).map((_, i) => i * 120),
      datasets: Object.keys(activeElectrodes)
        .filter(e => activeElectrodes[e])
        .map(e => ({
          label: e,
          data: Array(MAX_POINTS).fill(0),
          borderColor: getFixedColor(e),
          borderWidth: 1.5,
          pointRadius: 0
        }))
    },
    options: {
      animation: false,
      responsive: false,
      scales: {
        x: { min: 0, max: 3000, ticks: { stepSize: 120 } },
        y: { min: -400, max: 400 }
      },
      plugins: {
        legend: { display: true },
        annotation: {
          annotations: {
            safeBand: {
              type: 'box',
              yMin: -200,
              yMax: 200,
              xMin: 0,
              xMax: 3000,
              backgroundColor: 'rgba(190,255,203,0.25)',
              borderWidth: 0
            }
          }
        }
      }
    }
  });
} else { 
  if (voltageChart) { voltageChart.destroy(); voltageChart = null; } 
}

// ===================== FRECUENCIAS (barras en %) =====================
if (fEl && document.body.contains(fEl)) {
  const prevF = Chart.getChart ? Chart.getChart(fEl) : null; 
  if (prevF) prevF.destroy();
  const fCtx = fEl.getContext('2d');

  frequencyChart = new Chart(fCtx, {
    type: 'bar',
    data: {
      labels: ['Delta','Theta','Alpha','SMR'],
      datasets: [{
        backgroundColor:['#00042E','#001745','#002B5D','#4B7FD1'],
        borderColor:    ['#00042E','#001745','#002B5D','#4B7FD1'],
        data:[0,0,0,0]
      }]
    },
    options: {
      responsive:false,
      scales:{ y:{ min:0, max:100 } },
      plugins:{ legend:{ display:false } }
    }
  });
} else { 
  if (frequencyChart) { frequencyChart.destroy(); frequencyChart = null; } 
}

// ===================== ESTADO (línea % Calma y % Atención) =====================

if (sEl && document.body.contains(sEl)) {
  const prevS = Chart.getChart ? Chart.getChart(sEl) : null;
  if (prevS) prevS.destroy();
  const sCtx = sEl.getContext('2d');

stateChart = new Chart(sCtx, {
  type: 'line',
  data: {
    labels: Array(MAX_POINTS).fill(0).map((_, i) => i * 1),
    datasets: [
      {
        label: 'Calma %',
        data: Array(MAX_POINTS).fill(0),
        borderWidth: 2,
        pointRadius: 0,
        borderColor: '#C2B2DE',
        backgroundColor: 'rgba(194,178,222,0.20)', // opcional
        tension: 0.25
      },
      {
        label: 'Atención %',
        data: Array(MAX_POINTS).fill(0),
        borderWidth: 2,
        pointRadius: 0,
        borderColor: '#FCE2A8',
        backgroundColor: 'rgba(252,226,168,0.20)', // opcional
        tension: 0.25
      }
    ]
  },
  options: {
    animation: false,
    responsive: false,
    scales: { y: { min: 0, max: 100 } },
    plugins: { legend: { display: true } }
  }
});

} else {
  if (stateChart) { stateChart.destroy(); stateChart = null; }
}
}

function setElectrodesActive(map) {
  Object.keys(activeElectrodes).forEach(e => {
    const shouldBeActive = !!map[e];

    if (activeElectrodes[e] !== shouldBeActive) {
      activeElectrodes[e] = shouldBeActive;
    }
  });

  destroyCharts();
  initializeCharts();
}

function showStateGraph() {
  if (!currentMount) return;
  const v = document.getElementById(`voltageGraphContainer-${currentMount}`);
  const f = document.getElementById(`frequencyGraphContainer-${currentMount}`);
  const s = document.getElementById(`stateGraphContainer-${currentMount}`);
  if (v) v.style.display = 'none';
  if (f) f.style.display = 'none';
  if (s) s.style.display = 'block';
  initializeCharts();
}



  function showVoltageGraph() {
    if (!currentMount) return;
    const v = document.getElementById(`voltageGraphContainer-${currentMount}`);
    const f = document.getElementById(`frequencyGraphContainer-${currentMount}`);
    if (v) v.style.display = 'block';
    if (f) f.style.display = 'none';
  }
  function showFrequencyGraph() {
    if (!currentMount) return;
    const v = document.getElementById(`voltageGraphContainer-${currentMount}`);
    const f = document.getElementById(`frequencyGraphContainer-${currentMount}`);
    if (v) v.style.display = 'none';
    if (f) f.style.display = 'block';
    initializeCharts();
  }

  function updateVoltageGraph(electrode, newValue) {
    if (!voltageChart) return;
    const dataset = voltageChart.data.datasets.find(ds => ds.label === electrode);
    if (dataset) {
      dataset.data.shift();
      dataset.data.push(newValue);
      voltageChart.update();
    }
  }

  function sumPower(data, startFreq, endFreq) {
    const freqResolution = (256 / FFT_SIZE);
    const startIndex = Math.max(0, Math.floor(startFreq / freqResolution));
    const endIndex   = Math.min(data.length-1, Math.floor(endFreq / freqResolution));
let acc = 0;
for (let i = startIndex; i <= endIndex; i++) acc += data[i] * data[i];
return acc;
  }

function applyAudioFeedback_v2({ attentionPct, calmPct }) {
  if (!audioElement || audioElement.muted) return;

  const smooth = 0.12;
  let target = 0;

  if (neurofeedbackProtocol === 'attention') {
    const start = 60, full = 85;
    target = clamp01((attentionPct - start) / (full - start));
  } else {
    const start = 60, full = 85;
    target = clamp01((calmPct - start) / (full - start));
  }

  const nextVol = audioElement.volume + (target - audioElement.volume) * smooth;

  if (nextVol <= 0.01) {
    audioElement.volume = 0;
    if (!audioElement.paused) audioElement.pause();
  } else {
    if (audioElement.paused) audioElement.play();
    audioElement.volume = nextVol;
  }
}


function updateNeurofeedback(frequencies) {

  // ===== Band power =====
  const band = (lo, hi) => sumPower(frequencies, lo, hi);

  const delta = band(0.5, 4);
  const theta = band(4, 8);
  const alpha = band(8, 12);
  const smr   = band(12, 18);

  // ===== % SOLO para gráfico =====
  const total = Math.max(1e-12, delta + theta + alpha + smr);
  const deltaPct = (delta / total) * 100;
  const thetaPct = (theta / total) * 100;
  const alphaPct = (alpha / total) * 100;
  const smrPct   = (smr   / total) * 100;

  if (frequencyChart) {
    frequencyChart.data.datasets[0].data = [
      deltaPct, thetaPct, alphaPct, smrPct
    ];
    frequencyChart.update();
  }

  // ===== Estado estable (log power) =====
  const ld = Math.log(delta + EPS);
  const lt = Math.log(theta + EPS);
  const la = Math.log(alpha + EPS);
  const ls = Math.log(smr   + EPS);

  const maxLen = Math.max(2, Math.round(STATE_WINDOW_SEC * STATE_TICK_HZ));
  pushRing(bandRing.delta, ld, maxLen);
  pushRing(bandRing.theta, lt, maxLen);
  pushRing(bandRing.alpha, la, maxLen);
  pushRing(bandRing.smr,   ls, maxLen);

  const D = mean(bandRing.delta);
  const T = mean(bandRing.theta);
  const A = mean(bandRing.alpha);
  const S = mean(bandRing.smr);

  // ===== Índices =====
  const attIdx  = S - Math.log(Math.exp(T) + DELTA_WEIGHT * Math.exp(D) + EPS);
  const calmIdx = Math.log(Math.exp(A) + Math.exp(T) + EPS) - S;

  // ===== EMA adaptativa =====
  if (!ema.init) {
    ema.attMean = attIdx;
    ema.attVar  = 1;
    ema.calmMean = calmIdx;
    ema.calmVar  = 1;
    ema.init = true;
  } else {
    const a = EMA_ALPHA;

    // Atención
    const mA = ema.attMean;
    ema.attMean = (1 - a) * mA + a * attIdx;
    ema.attVar  = (1 - a) * ema.attVar + a * (attIdx - mA) ** 2;

    // Calma
    const mC = ema.calmMean;
    ema.calmMean = (1 - a) * mC + a * calmIdx;
    ema.calmVar  = (1 - a) * ema.calmVar + a * (calmIdx - mC) ** 2;

    ema.attVar  = Math.max(ema.attVar, 1e-6);
    ema.calmVar = Math.max(ema.calmVar, 1e-6);
  }

  const attZ  = (attIdx  - ema.attMean)  / Math.sqrt(ema.attVar);
  const calmZ = (calmIdx - ema.calmMean) / Math.sqrt(ema.calmVar);

  const sigmoid = z => 1 / (1 + Math.exp(-z));

  const attentionPct = 100 * sigmoid(attZ);
  const calmPct      = 100 * sigmoid(calmZ);

  // ===== Estado chart =====
  if (stateChart) {
    stateChart.data.datasets[0].data.shift();
    stateChart.data.datasets[0].data.push(calmPct);

    stateChart.data.datasets[1].data.shift();
    stateChart.data.datasets[1].data.push(attentionPct);

    stateChart.update();
  }

  // ===== Audio =====
if (neurofeedbackProtocol === 'attention' || neurofeedbackProtocol === 'calm') {
  applyAudioFeedback_v2({ attentionPct, calmPct });
}
}


const fft256 = new FFT(FFT_SIZE, 256);

function computeSpectrum256(arr) {
  const windowed = arr.slice(-FFT_SIZE);

  for (let n = 0; n < FFT_SIZE; n++) {
    windowed[n] *= 0.5 * (1 - Math.cos(2 * Math.PI * n / (FFT_SIZE - 1)));
  }

  fft256.forward(windowed);
  return Array.from(fft256.spectrum);
}



function updateFrequencyGraphData() {
  const spectra = [];

  Object.keys(activeElectrodes).forEach(e => {
    if (!activeElectrodes[e]) return;

    const buf = eegFrequencyBuffer[e];
    if (buf.length >= FFT_SIZE) {
      spectra.push(computeSpectrum256(buf));
      eegFrequencyBuffer[e] = buf.slice(-FFT_SIZE);
    }
  });

  if (!spectra.length) return;

  const L = spectra[0].length;
  const avg = new Array(L).fill(0);

  for (let s = 0; s < spectra.length; s++) {
    for (let i = 0; i < L; i++) {
      avg[i] += spectra[s][i];
    }
  }

  for (let i = 0; i < L; i++) avg[i] /= spectra.length;

  updateNeurofeedback(avg);
}


function processEEGData(electrode, eegData) {
  try {
    if (eegData.length < FFT_SIZE) return;
    const spec = computeSpectrum256(eegData);
    updateNeurofeedback(spec);
  } catch (e) {
    console.error('FFT error:', e);
  }
}


function stopAudioFeedback() {
  if (!audioElement) return;
  audioElement.muted = true;   // bloquea applyAudioFeedback
  audioElement.pause();
  audioElement.currentTime = 0;
  audioElement.volume = 0;
  console.log('Feedback de audio detenido.');
}


  // ===================== Conexión / lectura =====================
// Pasa-altos 1er orden por sustracción de componente lenta (sin libs)
function applyFilters(value, key) {
  if (!hpState[key]) return value;      // safety

  const FS = 256;                       // Muse 2
  const FC = 1.0;                       // cutoff ~1 Hz (ajustable 0.5–2)
  const dt = 1/FS;
  const rc = 1/(2*Math.PI*FC);
  const alpha = dt/(rc+dt);             // coef del low-pass

  const s = hpState[key];
  s.lp = s.lp + alpha*(value - s.lp);   // low-pass de la línea base
  return value - s.lp;                  // high-pass: quita DC y muy baja f
}

  function readEEGData() {
    const timestamp = new Date().toISOString();
    const tp9Value  = muse.eeg[ELECTRODES.TP9].read();
    const fp1Value  = muse.eeg[ELECTRODES.FP1].read();
    const fp2Value  = muse.eeg[ELECTRODES.FP2].read();
    const tp10Value = muse.eeg[ELECTRODES.TP10].read();

    if (tp9Value !== null && fp1Value !== null && fp2Value !== null && tp10Value !== null) {
const tp9F  = applyFilters(tp9Value,  'TP9');
const fp1F  = applyFilters(fp1Value,  'FP1');
const fp2F  = applyFilters(fp2Value,  'FP2');
const tp10F = applyFilters(tp10Value, 'TP10');


      eegBuffers.TP9.push(tp9F);   eegFrequencyBuffer.TP9.push(tp9F);
      eegBuffers.FP1.push(fp1F);   eegFrequencyBuffer.FP1.push(fp1F);
      eegBuffers.FP2.push(fp2F);   eegFrequencyBuffer.FP2.push(fp2F);
      eegBuffers.TP10.push(tp10F); eegFrequencyBuffer.TP10.push(tp10F);

if (recording) {
  const tp9Rec  = activeElectrodes.TP9  ? tp9Value  : '';
  const fp1Rec  = activeElectrodes.FP1  ? fp1Value  : '';
  const fp2Rec  = activeElectrodes.FP2  ? fp2Value  : '';
  const tp10Rec = activeElectrodes.TP10 ? tp10Value : '';

  recordedData.push(`${timestamp},${tp9Rec},${fp1Rec},${fp2Rec},${tp10Rec}`);
}

    }
  }

  function updateVoltageGraphData() {
    ['TP9','FP1','FP2','TP10'].forEach(electrode => {
      if (!activeElectrodes[electrode]) return;
      const buffer = eegBuffers[electrode];
      if (buffer.length > 0) {
        const avg = buffer.reduce((s,v)=>s+v,0) / buffer.length;
        updateVoltageGraph(electrode, avg);
        eegBuffers[electrode] = [];
      } else { updateVoltageGraph(electrode, 0); }
    });
  }

  function getFixedColor(electrode) {
    const colors = ['#00042E','#001745','#002B5D','#004176','#005890'];
    const index = Object.keys(ELECTRODES).indexOf(electrode);
    return colors[index % colors.length];
  }

function destroyCharts(){
  if (voltageChart)  { voltageChart.destroy();  voltageChart=null; }
  if (frequencyChart){ frequencyChart.destroy(); frequencyChart=null; }
  if (stateChart)    { stateChart.destroy();    stateChart=null; }
}

  function activateElectrode(e){ if (ELECTRODES.hasOwnProperty(e)) { activeElectrodes[e]=true; destroyCharts(); initializeCharts(); console.log(`${e} activado y añadido al gráfico.`);} }
  function deactivateElectrode(e){ if (ELECTRODES.hasOwnProperty(e)) { activeElectrodes[e]=false; destroyCharts(); initializeCharts(); console.log(`${e} desactivado y removido del gráfico.`);} }

  function connectAndReadData(mountId){
    setMount(mountId);
    destroyCharts();
    try {
      if (muse) muse.disconnect();
      muse = new Muse();

      muse.batteryData = function(event){
        let data = event.target.value; data = data.buffer ? data : new DataView(data);
        const fuelGauge = data.getUint16(4, true);
        this.batteryLevel = Math.max(0, Math.min(1, fuelGauge / 65535));
      };

      muse.onDisconnected = function(){
        this.dev = null; this.state = 0;
        suddenDisconnectSound.play();
        if (typeof bubble_fn_MuseStatus === 'function') bubble_fn_MuseStatus('Sudden Disconnection');
        clearInterval(readEEGDataInterval);
        clearInterval(voltageGraphUpdateInterval);
        clearInterval(frequencyGraphUpdateInterval);
        clearInterval(batteryIntervalId);
        clearInterval(statusIntervalId);
        batteryIntervalId = null; statusIntervalId = null;
        destroyCharts();
      };

muse.connect().then(()=>{
bandRing = { delta: [], theta: [], alpha: [], smr: [] };
ema = { attMean: 0, attVar: 1, calmMean: 0, calmVar: 1, init: false };
  connectedSound.play(); 
  lowBatterySoundPlayed = false;
  initializeCharts();

  recordedData = [];
  eegBuffers = { TP9:[], FP1:[], FP2:[], TP10:[] };
  eegFrequencyBuffer = { TP9:[], FP1:[], FP2:[], TP10:[] };

  // Reset del estado del filtro pasa-altos por canal
  hpState.TP9.lp = 0;
  hpState.FP1.lp = 0;
  hpState.FP2.lp = 0;
  hpState.TP10.lp = 0;

  readEEGDataInterval          = setInterval(readEEGData, 4);
  voltageGraphUpdateInterval   = setInterval(updateVoltageGraphData, 120);
  frequencyGraphUpdateInterval = setInterval(updateFrequencyGraphData, 1000);
  batteryIntervalId            = setInterval(showMuseBattery, 5000);
  statusIntervalId             = setInterval(showMuseStatus, 5000);
      }).catch(err => console.error('Error en muse.connect():', err));
    } catch (error) { console.error('Error in connectAndReadData:', error); }
  }

  function disconnect(){
    try {
      if (!muse) return;
      muse.disconnect();
      clearInterval(readEEGDataInterval);
      clearInterval(voltageGraphUpdateInterval);
      clearInterval(frequencyGraphUpdateInterval);
      clearInterval(batteryIntervalId);
      clearInterval(statusIntervalId);
      batteryIntervalId=null; statusIntervalId=null;
      destroyCharts();
      console.log('Desconectado exitosamente.');
      if (typeof bubble_fn_MuseStatus === 'function') bubble_fn_MuseStatus('Disconnected');
    } catch (error) { console.error('Error en la desconexión:', error); }
  }

  // ===================== Grabación =====================
  function startRecording(){ recordedData=[]; recording=true; console.log('Grabación iniciada.'); }
function stopRecording(){
  recording = false;
  console.log('Grabación detenida. Datos:', recordedData);

  if (typeof bubble_fn_js_to_bubble !== 'function') {
    console.error('bubble_fn_js_to_bubble no está definido.');
    return;
  }

  if (!recordedData.length) {
    console.warn('No hay datos grabados.');
    bubble_fn_js_to_bubble('');   // o ni llames si quieres
    return;
  }

  const payload = recordedData.join('@');
  console.log('Enviando a Bubble. Filas:', recordedData.length, 'len=', payload.length);
  bubble_fn_js_to_bubble(payload);
}


  // ===================== Audio (controlado por señal) =====================
function enableAudioFeedback(mode) {
  if (!audioElement) return;

  if (mode === 'attention') setAttentionProtocol();
  else if (mode === 'calm') setCalmProtocol();

  audioElement.muted  = false;
  audioElement.volume = 0;
  audioElement.pause();
  console.log('Feedback de audio listo (controlado por señal).');
}

  function setCalmProtocol(){ neurofeedbackProtocol='calm'; console.log('Protocolo de calma activado.'); }
  function setAttentionProtocol(){  neurofeedbackProtocol='attention';  console.log('Protocolo de atención activado.'); }

  function fadeInAudio(){ let v=0; audioElement.volume=v; audioElement.play(); const id=setInterval(()=>{ v=Math.min(1, v+0.05); audioElement.volume=v; if(v>=1) clearInterval(id); },100); }
  function fadeOutAudio(){ let v=audioElement.volume; const id=setInterval(()=>{ v=Math.max(0, v-0.05); audioElement.volume=v; if(v<=0){ audioElement.pause(); clearInterval(id);} },100); }

  // ===================== Estado / batería =====================
  function showMuseBattery(){
    if (muse && muse.batteryLevel != null) {
      const batteryPercentage = Math.round(muse.batteryLevel * 100);
      if (batteryPercentage <= 20 && !lowBatterySoundPlayed) { lowBatterySound.play(); lowBatterySoundPlayed = true; }
      if (typeof bubble_fn_MuseBattery === 'function') bubble_fn_MuseBattery(batteryPercentage);
    } else if (typeof bubble_fn_MuseBattery === 'function') bubble_fn_MuseBattery(0);
  }

  function showMuseStatus(){
    if (muse) {
      let status = 'Unknown';
      switch (muse.state) { case 0: status='Disconnected'; break; case 1: status='Connecting'; break; case 2: status='Connected'; break; }
      if (typeof bubble_fn_MuseStatus === 'function') bubble_fn_MuseStatus(status);
    } else if (typeof bubble_fn_MuseStatus === 'function') bubble_fn_MuseStatus('Not initialized');
  }

  // ===================== Exponer a window =====================
window.NF = window.NF || {};

window.NF.connectAndReadData = connectAndReadData;
window.NF.disconnect = disconnect;
window.NF.activateElectrode = activateElectrode;
window.NF.deactivateElectrode = deactivateElectrode;
window.NF.setElectrodesActive = setElectrodesActive;
window.NF.startRecording = startRecording;
window.NF.stopRecording = stopRecording;
window.NF.enableAudioFeedback = enableAudioFeedback;
window.NF.showVoltageGraph = showVoltageGraph;
window.NF.showFrequencyGraph = showFrequencyGraph;
window.NF.showStateGraph = showStateGraph;

console.log("NF listo:", typeof window.NF.connectAndReadData);
