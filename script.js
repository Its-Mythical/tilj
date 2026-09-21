(function(){
  const state = {
    stream:null, running:false,
    modes:{ faces:true, hands:true, motion:true, objects:true },
    prevGray:null, faceDetector:null, faceSupported:false, lastFaces:[],
    analysisW:160, analysisH:120, filter:'none',
    particles:[], lastBoxes:{ motion:[], objects:[], faces:[], hands:[] },
  };
  const FILTER_EMOJI = { glasses:'😎', hat:'🎩', bunny:'🐰', crown:'👑' };

  // Feature centroids for a lightweight object-ID classifier: 64-bin color histogram +
  // 8-bin edge-orientation histogram, L2-normalized, averaged from ~2-4 real photos per
  // category fetched from a public ImageNet sample-image set on GitHub. This is classical
  // nearest-centroid matching, not a neural network — expect rough, sometimes-wrong guesses.
  const OBJECT_CENTROIDS = {cat:[0.2631,0.0,0.0,0.0,0.0011,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.163,0.0005,0.0,0.0,0.239,0.6588,0.0015,0.0,0.0,0.0031,0.0092,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0019,0.0774,0.0005,0.0,0.0,0.0496,0.1967,0.002,0.0,0.0,0.0,0.0005,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0005,0.0,0.0,0.0,0.0,0.0015,0.2718,0.1832,0.2061,0.1231,0.19,0.2779,0.2061,0.212],dog:[0.3616,0.002,0.0,0.0,0.0026,0.0047,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0972,0.0,0.0,0.0,0.1971,0.2445,0.0046,0.0,0.187,0.0433,0.0099,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0054,0.0385,0.0,0.0,0.0348,0.1533,0.1167,0.0069,0.0,0.0,0.0,0.0005,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0247,0.2059,0.0,0.0,0.0,0.3017,0.0288,0.2427,0.2279,0.2063,0.2333,0.3281,0.2553,0.2595,0.3185],car:[0.4387,0.0147,0.0,0.0,0.0045,0.0212,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1241,0.0023,0.0,0.0,0.0607,0.459,0.0726,0.0,0.0005,0.0147,0.0607,0.0,0.0,0.0,0.0,0.0,0.0381,0.0009,0.0,0.0,0.001,0.0654,0.0122,0.0,0.0005,0.0433,0.2107,0.0237,0.0,0.0,0.0036,0.0233,0.0088,0.0005,0.0,0.0,0.0,0.026,0.0,0.0,0.0,0.0023,0.0363,0.0,0.0,0.0,0.0239,0.1383,0.1714,0.2608,0.2928,0.1883,0.1874,0.3097,0.3198,0.2024],bottle:[0.4284,0.0072,0.0,0.0,0.0007,0.003,0.0006,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0982,0.0098,0.0,0.0,0.1502,0.3536,0.032,0.0,0.0,0.0007,0.0108,0.0,0.0,0.0,0.0,0.0,0.021,0.0,0.0,0.0,0.0321,0.0645,0.0097,0.0,0.0,0.1118,0.3509,0.0132,0.0,0.0,0.0021,0.0144,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0217,0.1001,0.0006,0.0,0.0,0.0673,0.0493,0.1911,0.1897,0.2425,0.2373,0.2365,0.2209,0.2716,0.3642],cup:[0.3509,0.0363,0.0,0.0,0.0,0.0162,0.0163,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.267,0.0,0.0,0.0,0.0298,0.2826,0.023,0.0,0.0,0.0,0.0147,0.0,0.0,0.0,0.0,0.0026,0.0,0.0,0.0,0.0,0.0063,0.0227,0.0096,0.0,0.0,0.0044,0.3147,0.0112,0.0,0.0,0.0052,0.0354,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0125,0.0048,0.0,0.0,0.0138,0.3861,0.1892,0.2127,0.1977,0.1875,0.3462,0.2058,0.1761,0.3509],chair:[0.7201,0.0075,0.0,0.0,0.0009,0.0031,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1419,0.0079,0.0,0.0,0.1182,0.2769,0.0009,0.0,0.0009,0.0004,0.0013,0.0,0.0,0.0,0.0,0.0,0.0355,0.0,0.0,0.0,0.0138,0.1346,0.0,0.0,0.0031,0.0707,0.051,0.0,0.0,0.0,0.0,0.0,0.0049,0.0,0.0,0.0,0.0049,0.0,0.0,0.0,0.0039,0.001,0.0138,0.0,0.0,0.0,0.0053,0.0046,0.1611,0.212,0.2104,0.1614,0.1918,0.2871,0.2273,0.1759],plant:[0.3118,0.0312,0.0,0.0,0.3669,0.0142,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0321,0.0151,0.0,0.0,0.2487,0.345,0.0132,0.0,0.0048,0.0457,0.0104,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0265,0.0094,0.0,0.0029,0.0076,0.1051,0.0028,0.001,0.001,0.0095,0.0191,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0019,0.0009,0.0181,0.0133,0.0162,0.2686,0.2794,0.2006,0.1872,0.3144,0.3391,0.1904,0.2012,0.2307],phone:[0.3134,0.0144,0.0,0.0,0.0012,0.0183,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.136,0.0,0.0,0.0,0.029,0.4955,0.0717,0.0,0.0,0.0122,0.038,0.0104,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0488,0.0135,0.0016,0.0,0.0,0.0091,0.3705,0.0261,0.0,0.0,0.0006,0.0128,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0076,0.0,0.0,0.0,0.0052,0.2075,0.2362,0.2059,0.1849,0.1533,0.299,0.2453,0.2071,0.312],book:[0.4869,0.0595,0.0,0.0,0.0144,0.181,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1725,0.0,0.0,0.0,0.0843,0.3128,0.0009,0.0,0.0,0.0075,0.0115,0.0,0.0,0.0,0.0,0.0,0.1821,0.0,0.0,0.0,0.053,0.0975,0.0,0.0,0.0142,0.12,0.1154,0.0,0.0,0.0,0.0009,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0132,0.0443,0.0,0.0,0.0,0.0226,0.0075,0.2017,0.2418,0.2289,0.2339,0.2063,0.3291,0.3044,0.2562]};

  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const octx = overlay.getContext('2d');
  const small = document.createElement('canvas');
  small.width = state.analysisW; small.height = state.analysisH;
  const sctx = small.getContext('2d', { willReadFrequently:true });
  const sample = document.createElement('canvas');
  const spctx = sample.getContext('2d', { willReadFrequently:true });
  const mirrorFrame = document.createElement('canvas');
  const mfctx = mirrorFrame.getContext('2d', { willReadFrequently:true });
  const classCanvas = document.createElement('canvas');
  classCanvas.width = 32; classCanvas.height = 32;
  const ccctx = classCanvas.getContext('2d', { willReadFrequently:true });

  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const snapBtn = document.getElementById('snapBtn');
  const snapFxBtn = document.getElementById('snapFxBtn');
  const filterSelect = document.getElementById('faceFilter');
  const statusEl = document.getElementById('status');
  const countsEl = document.getElementById('counts');
  const viewfinderEl = document.getElementById('viewfinder');
  const reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  let baseStatus = '';

  function setBaseStatus(msg){ baseStatus = msg; statusEl.textContent = msg; }
  function flashStatus(msg, ms=2200){
    statusEl.textContent = msg;
    clearTimeout(flashStatus._t);
    flashStatus._t = setTimeout(()=>{ statusEl.textContent = baseStatus; }, ms);
  }

  ['faces','hands','motion','objects'].forEach(m=>{
    document.getElementById('chk_'+m).addEventListener('change', e=>{ state.modes[m] = e.target.checked; });
  });
  filterSelect.addEventListener('change', e=>{ state.filter = e.target.value; });
  snapBtn.addEventListener('click', takeSnapshot);
  snapFxBtn.addEventListener('click', castSnap);

  if ('FaceDetector' in window) {
    try { state.faceDetector = new FaceDetector({ maxDetectedFaces:6, fastMode:true }); state.faceSupported = true; }
    catch(e){ state.faceSupported = false; }
  }
  setBaseStatus(state.faceSupported
    ? 'Face detection: supported in this browser (on-device AI).'
    : 'Face detection: not supported in this browser — Motion & Objects still work. Try Chrome/Edge.');

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);

  async function startCamera(){
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:'user' }, audio:false });
      video.srcObject = state.stream;
      await video.play();
      resizeOverlay();
      state.running = true;
      startBtn.disabled = true; stopBtn.disabled = false; snapBtn.disabled = false; snapFxBtn.disabled = false;
      viewfinderEl.classList.add('is-live');
      requestAnimationFrame(loop);
      if (state.faceSupported) faceLoop();
    } catch(err){
      statusEl.textContent = 'Camera error: ' + err.message;
    }
  }

  function stopCamera(){
    state.running = false;
    if (state.stream) state.stream.getTracks().forEach(t=>t.stop());
    octx.clearRect(0,0,overlay.width,overlay.height);
    countsEl.textContent = '';
    state.particles = [];
    viewfinderEl.classList.remove('is-live');
    startBtn.disabled = false; stopBtn.disabled = true; snapBtn.disabled = true; snapFxBtn.disabled = true;
  }

  function resizeOverlay(){ overlay.width = video.clientWidth; overlay.height = video.clientHeight; }
  window.addEventListener('resize', ()=>{ if(state.running) resizeOverlay(); });

  function toGray(imgData){
    const { data, width, height } = imgData;
    const gray = new Uint8ClampedArray(width*height);
    for (let i=0,p=0; i<data.length; i+=4,p++){
      gray[p] = (data[i]*0.299 + data[i+1]*0.587 + data[i+2]*0.114) | 0;
    }
    return gray;
  }

  function sobelMask(gray,w,h,thresh){
    const mask = new Uint8Array(w*h);
    for (let y=1;y<h-1;y++){
      for (let x=1;x<w-1;x++){
        const i=y*w+x;
        const gx = gray[i+1]-gray[i-1];
        const gy = gray[i+w]-gray[i-w];
        if (Math.abs(gx)+Math.abs(gy) > thresh) mask[i]=1;
      }
    }
    return mask;
  }

  function diffMask(a,b,w,h,thresh){
    const mask = new Uint8Array(w*h);
    for (let i=0;i<w*h;i++){ if (Math.abs(a[i]-b[i]) > thresh) mask[i]=1; }
    return mask;
  }

  function dilate(mask,w,h,iterations){
    let m = mask;
    for (let it=0; it<iterations; it++){
      const out = new Uint8Array(w*h);
      for (let y=0;y<h;y++){
        for (let x=0;x<w;x++){
          const i=y*w+x;
          if (m[i]) { out[i]=1; continue; }
          if ((x>0&&m[i-1])||(x<w-1&&m[i+1])||(y>0&&m[i-w])||(y<h-1&&m[i+w])) out[i]=1;
        }
      }
      m = out;
    }
    return m;
  }

  function connectedComponents(mask,w,h,minPixels,maxBoxes){
    const visited = new Uint8Array(w*h);
    const boxes = [];
    const stack = [];
    for (let y=0;y<h;y++){
      for (let x=0;x<w;x++){
        const idx=y*w+x;
        if (!mask[idx] || visited[idx]) continue;
        let minX=x,maxX=x,minY=y,maxY=y,count=0;
        stack.push(idx); visited[idx]=1;
        while (stack.length){
          const cur = stack.pop();
          const cx = cur % w, cy = (cur / w) | 0;
          count++;
          if (cx<minX) minX=cx; if (cx>maxX) maxX=cx;
          if (cy<minY) minY=cy; if (cy>maxY) maxY=cy;
          const nbrs = [cur-1,cur+1,cur-w,cur+w];
          for (const n of nbrs){
            if (n>=0 && n<w*h && mask[n] && !visited[n]){ visited[n]=1; stack.push(n); }
          }
        }
        if (count >= minPixels) boxes.push({ x:minX, y:minY, w:maxX-minX+1, h:maxY-minY+1, area:count });
      }
    }
    boxes.sort((a,b)=>b.area-a.area);
    return boxes.slice(0,maxBoxes);
  }

  // AR-style HUD reticle: corner brackets + pulsing lock-on dot + floating tag
  // --- Lightweight "object ID" classifier: same 64-bin color + 8-bin edge histogram used
  // to build OBJECT_CENTROIDS, run on a live 32x32 crop and compared by cosine similarity.
  function colorHist32(data){
    const hist = new Array(64).fill(0);
    for (let i=0;i<data.length;i+=4){
      const ri=data[i]>>6, gi=data[i+1]>>6, bi=data[i+2]>>6;
      hist[ri*16+gi*4+bi]++;
    }
    const total = data.length/4;
    for (let i=0;i<64;i++) hist[i] /= total;
    return hist;
  }

  function edgeHist32(data){
    const gray = new Float32Array(32*32);
    for (let i=0,p=0; i<data.length; i+=4,p++){
      gray[p] = data[i]*0.299 + data[i+1]*0.587 + data[i+2]*0.114;
    }
    const bins = new Array(8).fill(0);
    let totalMag = 0;
    for (let y=1;y<31;y++){
      for (let x=1;x<31;x++){
        const i = y*32+x;
        const gx = gray[i-31] - gray[i-33] + 2*(gray[i+1]-gray[i-1]) + gray[i+33] - gray[i+31];
        const gy = gray[i+31] - gray[i-33] + 2*(gray[i+32]-gray[i-32]) + gray[i+33] - gray[i-31];
        const mag = Math.sqrt(gx*gx+gy*gy);
        if (mag < 1e-6) continue;
        let angle = Math.atan2(gy,gx);
        if (angle < 0) angle += Math.PI*2;
        const b = Math.min(7, Math.floor(angle/(Math.PI*2/8)));
        bins[b] += mag; totalMag += mag;
      }
    }
    if (totalMag > 0) for (let i=0;i<8;i++) bins[i] /= totalMag;
    return bins;
  }

  function l2Normalize(v){
    let n = 0; for (const x of v) n += x*x;
    n = Math.sqrt(n);
    return n < 1e-9 ? v : v.map(x=>x/n);
  }

  function classifyCrop(sourceCanvas, x, y, w, h){
    ccctx.clearRect(0,0,32,32);
    ccctx.drawImage(sourceCanvas, x, y, w, h, 0, 0, 32, 32);
    const data = ccctx.getImageData(0,0,32,32).data;
    const vec = l2Normalize(colorHist32(data).concat(edgeHist32(data)));
    let bestLabel = null, bestScore = -2;
    for (const label in OBJECT_CENTROIDS){
      const c = OBJECT_CENTROIDS[label];
      let dot = 0; for (let i=0;i<vec.length;i++) dot += vec[i]*c[i];
      if (dot > bestScore){ bestScore = dot; bestLabel = label; }
    }
    return { label:bestLabel, score:bestScore };
  }

  // --- Hand/finger detection: YCbCr skin-color segmentation + radial silhouette profiling ---
  function skinMask(data, w, h){
    const mask = new Uint8Array(w*h);
    for (let i=0, p=0; i<data.length; i+=4, p++){
      const r=data[i], g=data[i+1], b=data[i+2];
      const cb = 128 - 0.168736*r - 0.331264*g + 0.5*b;
      const cr = 128 + 0.5*r - 0.418688*g - 0.081312*b;
      if (cb>=77 && cb<=127 && cr>=133 && cr<=173) mask[p] = 1;
    }
    return mask;
  }

  function handComponents(mask, w, h, minPixels, maxBoxes){
    const visited = new Uint8Array(w*h);
    const comps = [];
    const stack = [];
    for (let y=0;y<h;y++){
      for (let x=0;x<w;x++){
        const idx = y*w+x;
        if (!mask[idx] || visited[idx]) continue;
        let minX=x,maxX=x,minY=y,maxY=y;
        const pixels = [];
        stack.push(idx); visited[idx]=1;
        while (stack.length){
          const cur = stack.pop();
          const cx = cur % w, cy = (cur / w) | 0;
          pixels.push(cur);
          if (cx<minX) minX=cx; if (cx>maxX) maxX=cx;
          if (cy<minY) minY=cy; if (cy>maxY) maxY=cy;
          const nbrs = [cur-1, cur+1, cur-w, cur+w];
          for (const n of nbrs){
            if (n>=0 && n<w*h && mask[n] && !visited[n]){ visited[n]=1; stack.push(n); }
          }
        }
        if (pixels.length >= minPixels) comps.push({ x:minX, y:minY, w:maxX-minX+1, h:maxY-minY+1, pixels });
      }
    }
    comps.sort((a,b)=>b.pixels.length-a.pixels.length);
    return comps.slice(0, maxBoxes);
  }

  // Counts finger-like protrusions on a blob by sampling the max boundary distance from the
  // centroid at each angle (a radial silhouette profile) and finding peaks well above the
  // blob's core "palm" radius — a simpler stand-in for convex-hull convexity-defect counting.
  function analyzeHandShape(comp, w){
    const member = new Set(comp.pixels);
    let sumX=0, sumY=0;
    const boundary = [];
    for (const idx of comp.pixels){
      const x = idx % w, y = (idx / w) | 0;
      sumX += x; sumY += y;
      const nbrs = [idx-1, idx+1, idx-w, idx+w];
      let isBoundary = false;
      for (const n of nbrs){ if (!member.has(n)) { isBoundary = true; break; } }
      if (isBoundary) boundary.push({ x, y });
    }
    const cx = sumX/comp.pixels.length, cy = sumY/comp.pixels.length;
    if (boundary.length < 8) return { fingers:0, tips:[] };

    const polar = boundary.map(p=>{
      const dx=p.x-cx, dy=p.y-cy;
      return { x:p.x, y:p.y, dist:Math.hypot(dx,dy), angle:(Math.atan2(dy,dx)+Math.PI*2)%(Math.PI*2) };
    });
    const dists = polar.map(p=>p.dist).sort((a,b)=>a-b);
    const palmRadius = dists[Math.floor(dists.length*0.4)] || 1;

    const BINS = 72;
    const profile = new Array(BINS).fill(0);
    const profilePoint = new Array(BINS).fill(null);
    for (const p of polar){
      const bin = Math.min(BINS-1, Math.floor(p.angle/(Math.PI*2/BINS)));
      if (p.dist > profile[bin]){ profile[bin] = p.dist; profilePoint[bin] = p; }
    }
    for (let i=0;i<BINS;i++){
      if (profile[i] === 0){
        let j=i, steps=0;
        while (profile[j] === 0 && steps < BINS){ j = (j-1+BINS)%BINS; steps++; }
        profile[i] = profile[j]; profilePoint[i] = profilePoint[j];
      }
    }

    const threshold = palmRadius * 1.25;
    let peakIdx = [];
    for (let i=0;i<BINS;i++){
      if (profile[i] <= threshold) continue;
      const prev = profile[(i-1+BINS)%BINS], next = profile[(i+1)%BINS];
      if (profile[i] >= prev && profile[i] >= next) peakIdx.push(i);
    }
    const minSep = Math.round(BINS*20/360);
    peakIdx.sort((a,b)=>profile[b]-profile[a]);
    const chosen = [];
    for (const i of peakIdx){
      if (chosen.every(c => Math.min(Math.abs(c-i), BINS-Math.abs(c-i)) >= minSep)) chosen.push(i);
      if (chosen.length >= 5) break;
    }
    return { fingers: chosen.length, tips: chosen.map(i=>({ x:profilePoint[i].x, y:profilePoint[i].y })) };
  }

  function overlapsAny(box, others, fracThreshold){
    for (const o of others){
      const ix = Math.max(box.x,o.x), iy = Math.max(box.y,o.y);
      const ax = Math.min(box.x+box.w,o.x+o.w), ay = Math.min(box.y+box.h,o.y+o.h);
      const inter = Math.max(0,ax-ix) * Math.max(0,ay-iy);
      if (inter / (box.w*box.h) > fracThreshold) return true;
    }
    return false;
  }

  function drawARTag(x,y,w,h,color,label,extra){
    const pulse = reduceMotion ? 0.6 : 0.5 + 0.5*Math.sin(performance.now()/300);
    const cl = Math.max(6, Math.min(w,h)*0.28);
    octx.save();
    octx.strokeStyle = color; octx.lineWidth = 2;
    octx.shadowColor = color; octx.shadowBlur = 5 + pulse*6;
    [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]].forEach(([cx,cy,dx,dy])=>{
      octx.beginPath();
      octx.moveTo(cx, cy+cl*dy);
      octx.lineTo(cx, cy);
      octx.lineTo(cx+cl*dx, cy);
      octx.stroke();
    });
    octx.beginPath();
    octx.arc(x+w/2, y+h/2, 2+pulse*2, 0, Math.PI*2);
    octx.fillStyle = color; octx.fill();
    octx.shadowBlur = 0;

    const tagX = Math.min(x+w+14, overlay.width-90);
    const tagY = Math.max(y-6, 14);
    octx.beginPath();
    octx.moveTo(x+w, y); octx.lineTo(tagX, tagY);
    octx.lineWidth = 1; octx.stroke();

    octx.font = '11px sans-serif';
    const text = extra ? `${label} · ${extra}` : label;
    const textW = octx.measureText(text).width + 10;
    octx.fillStyle = color;
    octx.fillRect(tagX, tagY-14, textW, 18);
    octx.fillStyle = '#0b0d10';
    octx.save();
    octx.translate(tagX+textW, tagY-14);
    octx.scale(-1,1);
    octx.fillText(text, 5, 12);
    octx.restore();
    octx.restore();
  }

  // AR face filter: emoji anchored proportionally on the tracked face box
  function drawFaceFilter(x,y,w,h,type){
    const emoji = FILTER_EMOJI[type];
    if (!emoji) return;
    octx.save();
    octx.textAlign = 'center'; octx.textBaseline = 'middle';
    let cy, size;
    if (type === 'glasses'){ cy = y + h*0.38; size = w*1.05; }
    else { cy = y - h*0.12; size = w*1.15; } // hat / bunny ears / crown sit above the head
    octx.font = `${size}px sans-serif`;
    octx.fillText(emoji, x+w/2, cy);
    octx.restore();
  }

  function loop(){
    if (!state.running) return;
    sctx.drawImage(video, 0, 0, state.analysisW, state.analysisH);
    const imgData = sctx.getImageData(0,0,state.analysisW,state.analysisH);
    const gray = toGray(imgData);

    octx.clearRect(0,0,overlay.width,overlay.height);
    const sx = overlay.width/state.analysisW, sy = overlay.height/state.analysisH;
    let motionCount=0, objectCount=0, handCount=0;

    const frameArea = overlay.width*overlay.height;

    if (state.modes.motion && state.prevGray){
      const mask = diffMask(gray, state.prevGray, state.analysisW, state.analysisH, 25);
      const dilated = dilate(mask, state.analysisW, state.analysisH, 1);
      const boxes = connectedComponents(dilated, state.analysisW, state.analysisH, 15, 6);
      motionCount = boxes.length;
      state.lastBoxes.motion = boxes.map(b=>({ x:b.x*sx, y:b.y*sy, w:b.w*sx, h:b.h*sy }));
      state.lastBoxes.motion.forEach(({x,y,w,h})=>{
        const pct = Math.round((w*h/frameArea)*100)+'% frame';
        drawARTag(x,y,w,h, getVar('--motion'), 'Motion', pct);
      });
    } else { state.lastBoxes.motion = []; }

    if (state.modes.objects){
      const edges = sobelMask(gray, state.analysisW, state.analysisH, 45);
      const dilated = dilate(edges, state.analysisW, state.analysisH, 2);
      const boxes = connectedComponents(dilated, state.analysisW, state.analysisH, 40, 6);
      objectCount = boxes.length;
      state.lastBoxes.objects = boxes.map(b=>({ x:b.x*sx, y:b.y*sy, w:b.w*sx, h:b.h*sy }));

      if (state.lastBoxes.objects.length && video.videoWidth){
        mirrorFrame.width = overlay.width; mirrorFrame.height = overlay.height;
        mfctx.save();
        mfctx.translate(mirrorFrame.width,0); mfctx.scale(-1,1);
        mfctx.drawImage(video, 0, 0, mirrorFrame.width, mirrorFrame.height);
        mfctx.restore();
      }

      state.lastBoxes.objects.forEach(({x,y,w,h})=>{
        const pct = Math.round((w*h/frameArea)*100)+'% frame';
        if (w >= 20 && h >= 20 && video.videoWidth){
          const { label, score } = classifyCrop(mirrorFrame, x, y, w, h);
          if (score >= 0.3){
            drawARTag(x,y,w,h, getVar('--object'), label.charAt(0).toUpperCase()+label.slice(1)+'?', Math.round(score*100)+'%');
            return;
          }
        }
        drawARTag(x,y,w,h, getVar('--object'), 'Object', pct);
      });
    } else { state.lastBoxes.objects = []; }

    if (state.modes.faces && state.faceSupported && video.videoWidth){
      const fsx = overlay.width/video.videoWidth, fsy = overlay.height/video.videoHeight;
      state.lastBoxes.faces = state.lastFaces.map(f=>({
        x:f.boundingBox.x*fsx, y:f.boundingBox.y*fsy, w:f.boundingBox.width*fsx, h:f.boundingBox.height*fsy,
      }));
      state.lastBoxes.faces.forEach(({x,y,w,h})=>{
        if (state.filter !== 'none') drawFaceFilter(x,y,w,h,state.filter);
        else drawARTag(x,y,w,h, getVar('--face'), 'Face');
      });
    } else { state.lastBoxes.faces = []; }

    if (state.modes.hands){
      const skin = skinMask(imgData.data, state.analysisW, state.analysisH);
      const candidates = handComponents(skin, state.analysisW, state.analysisH, 250, 3);
      const hands = [];
      candidates.forEach(comp=>{
        const bx=comp.x*sx, by=comp.y*sy, bw=comp.w*sx, bh=comp.h*sy;
        if (overlapsAny({x:bx,y:by,w:bw,h:bh}, state.lastBoxes.faces, 0.4)) return;
        const shape = analyzeHandShape(comp, state.analysisW);
        hands.push({
          x:bx, y:by, w:bw, h:bh,
          fingers: shape.fingers,
          tips: shape.tips.map(t=>({ x:t.x*sx, y:t.y*sy })),
        });
      });
      handCount = hands.length;
      state.lastBoxes.hands = hands;
      hands.forEach(hd=>{
        drawARTag(hd.x, hd.y, hd.w, hd.h, getVar('--hand'), 'Hand', `${hd.fingers} finger${hd.fingers===1?'':'s'}`);
        hd.tips.forEach(t=>{
          octx.beginPath();
          octx.arc(t.x, t.y, 4, 0, Math.PI*2);
          octx.fillStyle = getVar('--hand');
          octx.fill();
        });
      });
    } else { state.lastBoxes.hands = []; }

    drawParticles();
    state.prevGray = gray;
    countsEl.textContent = `Motion: ${motionCount}   Objects: ${objectCount}   Faces: ${state.modes.faces && state.faceSupported ? state.lastFaces.length : '—'}   Hands: ${state.modes.hands ? handCount : '—'}`;
    requestAnimationFrame(loop);
  }

  async function faceLoop(){
    if (!state.running || !state.faceSupported) return;
    try { state.lastFaces = state.modes.faces ? await state.faceDetector.detect(video) : []; }
    catch(e){ state.lastFaces = []; }
    setTimeout(faceLoop, 200);
  }

  function getVar(name){ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function captureFrameBlob(){
    return new Promise(resolve=>{
      const out = document.createElement('canvas');
      out.width = overlay.width; out.height = overlay.height;
      const c = out.getContext('2d');
      c.save();
      c.translate(out.width, 0); c.scale(-1,1); // match the mirrored on-screen view
      c.drawImage(video, 0, 0, out.width, out.height);
      c.drawImage(overlay, 0, 0);
      c.restore();
      out.toBlob(blob=>resolve(blob), 'image/png');
    });
  }

  async function takeSnapshot(){
    if (!state.running) return;
    const blob = await captureFrameBlob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'ar-camera-snapshot.png';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(()=>URL.revokeObjectURL(url), 2000);
    flashStatus('Snapshot saved.');
  }

  // --- Gauntlet-style snap: disintegrate whatever's currently detected into drifting particles ---
  function drawParticles(){
    if (!state.particles.length) return;
    const now = performance.now();
    const alive = [];
    for (const p of state.particles){
      const t = (now - p.spawn) / p.duration;
      if (t >= 1) continue;
      const x = p.x0 + p.vx*t, y = p.y0 + p.vy*t;
      const alpha = (1-t) * 0.9;
      const size = Math.max(0.4, p.size * (1 - t*0.6));
      octx.fillStyle = `rgba(${p.r},${p.g},${p.b},${alpha})`;
      octx.beginPath();
      octx.arc(x, y, size, 0, Math.PI*2);
      octx.fill();
      alive.push(p);
    }
    state.particles = alive;
  }

  function flashSnap(){
    if (reduceMotion) return;
    viewfinderEl.classList.remove('snap-flash');
    void viewfinderEl.offsetWidth; // restart the animation if triggered again quickly
    viewfinderEl.classList.add('snap-flash');
    setTimeout(()=>viewfinderEl.classList.remove('snap-flash'), 750);
  }

  function playSnapSound(){
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const bufferSize = ctx.sampleRate * 0.06;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i=0;i<bufferSize;i++) data[i] = (Math.random()*2-1) * (1 - i/bufferSize);
      const noise = ctx.createBufferSource();
      noise.buffer = buffer;
      const filter = ctx.createBiquadFilter();
      filter.type = 'highpass'; filter.frequency.value = 1500;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.9, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.09);
      noise.connect(filter).connect(gain).connect(ctx.destination);
      noise.start();
      noise.stop(ctx.currentTime + 0.1);
      setTimeout(()=>ctx.close(), 300);
    } catch(e){ /* audio unavailable, silently skip */ }
  }

  function castSnap(){
    if (!state.running) return;
    const boxes = [...state.lastBoxes.motion, ...state.lastBoxes.objects, ...state.lastBoxes.faces, ...state.lastBoxes.hands];
    if (!boxes.length){ flashStatus('Nothing detected to snap.'); return; }

    sample.width = overlay.width; sample.height = overlay.height;
    spctx.save();
    spctx.translate(sample.width, 0); spctx.scale(-1,1); // match the mirrored on-screen view
    spctx.drawImage(video, 0, 0, sample.width, sample.height);
    spctx.restore();
    const img = spctx.getImageData(0, 0, sample.width, sample.height);
    const now = performance.now();

    boxes.forEach(({x,y,w,h})=>{
      const count = Math.max(16, Math.min(70, Math.round((w*h)/70)));
      for (let i=0; i<count && state.particles.length<280; i++){
        const px = Math.min(img.width-1, Math.max(0, Math.round(x + Math.random()*w)));
        const py = Math.min(img.height-1, Math.max(0, Math.round(y + Math.random()*h)));
        const idx = (py*img.width + px) * 4;
        const angle = Math.random()*Math.PI*2;
        const speed = 40 + Math.random()*100;
        state.particles.push({
          x0:px, y0:py,
          vx: Math.cos(angle)*speed*0.5,
          vy: -Math.abs(Math.sin(angle))*speed - 30,
          size: 1.4 + Math.random()*2.6,
          r:img.data[idx], g:img.data[idx+1], b:img.data[idx+2],
          spawn: now, duration: 850 + Math.random()*650,
        });
      }
    });

    flashSnap();
    playSnapSound();
    flashStatus(`🫰 Snapped ${boxes.length} ${boxes.length===1?'thing':'things'} into dust.`);
  }
})();
