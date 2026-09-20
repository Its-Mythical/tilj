(function(){
  const state = {
    stream:null, running:false,
    modes:{ faces:true, motion:true, objects:true },
    prevGray:null, faceDetector:null, faceSupported:false, lastFaces:[],
    analysisW:160, analysisH:120, filter:'none',
  };
  const FILTER_EMOJI = { glasses:'😎', hat:'🎩', bunny:'🐰', crown:'👑' };

  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const octx = overlay.getContext('2d');
  const small = document.createElement('canvas');
  small.width = state.analysisW; small.height = state.analysisH;
  const sctx = small.getContext('2d', { willReadFrequently:true });

  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const snapBtn = document.getElementById('snapBtn');
  const filterSelect = document.getElementById('faceFilter');
  const statusEl = document.getElementById('status');
  const countsEl = document.getElementById('counts');
  const viewfinderEl = document.getElementById('viewfinder');
  const reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  ['faces','motion','objects'].forEach(m=>{
    document.getElementById('chk_'+m).addEventListener('change', e=>{ state.modes[m] = e.target.checked; });
  });
  filterSelect.addEventListener('change', e=>{ state.filter = e.target.value; });
  snapBtn.addEventListener('click', takeSnapshot);

  if ('FaceDetector' in window) {
    try { state.faceDetector = new FaceDetector({ maxDetectedFaces:6, fastMode:true }); state.faceSupported = true; }
    catch(e){ state.faceSupported = false; }
  }
  statusEl.textContent = state.faceSupported
    ? 'Face detection: supported in this browser (on-device AI).'
    : 'Face detection: not supported in this browser — Motion & Objects still work. Try Chrome/Edge.';

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);

  async function startCamera(){
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:'user' }, audio:false });
      video.srcObject = state.stream;
      await video.play();
      resizeOverlay();
      state.running = true;
      startBtn.disabled = true; stopBtn.disabled = false; snapBtn.disabled = false;
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
    viewfinderEl.classList.remove('is-live');
    startBtn.disabled = false; stopBtn.disabled = true; snapBtn.disabled = true;
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
    let motionCount=0, objectCount=0;

    const frameArea = overlay.width*overlay.height;

    if (state.modes.motion && state.prevGray){
      const mask = diffMask(gray, state.prevGray, state.analysisW, state.analysisH, 25);
      const dilated = dilate(mask, state.analysisW, state.analysisH, 1);
      const boxes = connectedComponents(dilated, state.analysisW, state.analysisH, 15, 6);
      motionCount = boxes.length;
      boxes.forEach(b=>{
        const bx=b.x*sx, by=b.y*sy, bw=b.w*sx, bh=b.h*sy;
        const pct = Math.round((bw*bh/frameArea)*100)+'% frame';
        drawARTag(bx,by,bw,bh, getVar('--motion'), 'Motion', pct);
      });
    }

    if (state.modes.objects){
      const edges = sobelMask(gray, state.analysisW, state.analysisH, 45);
      const dilated = dilate(edges, state.analysisW, state.analysisH, 2);
      const boxes = connectedComponents(dilated, state.analysisW, state.analysisH, 40, 6);
      objectCount = boxes.length;
      boxes.forEach(b=>{
        const bx=b.x*sx, by=b.y*sy, bw=b.w*sx, bh=b.h*sy;
        const pct = Math.round((bw*bh/frameArea)*100)+'% frame';
        drawARTag(bx,by,bw,bh, getVar('--object'), 'Object', pct);
      });
    }

    if (state.modes.faces && state.faceSupported && video.videoWidth){
      const fsx = overlay.width/video.videoWidth, fsy = overlay.height/video.videoHeight;
      state.lastFaces.forEach(f=>{
        const bx=f.boundingBox.x*fsx, by=f.boundingBox.y*fsy, bw=f.boundingBox.width*fsx, bh=f.boundingBox.height*fsy;
        if (state.filter !== 'none') drawFaceFilter(bx,by,bw,bh,state.filter);
        else drawARTag(bx,by,bw,bh, getVar('--face'), 'Face');
      });
    }

    state.prevGray = gray;
    countsEl.textContent = `Motion: ${motionCount}   Objects: ${objectCount}   Faces: ${state.modes.faces && state.faceSupported ? state.lastFaces.length : '—'}`;
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
  }
})();
