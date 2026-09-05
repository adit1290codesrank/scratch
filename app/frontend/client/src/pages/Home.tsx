import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment, OrbitControls, PerspectiveCamera, RoundedBox, Text } from "@react-three/drei";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, ArrowUpRight, Layers3, Maximize2, Pause, Play, RotateCcw, SkipBack, UploadCloud } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import * as THREE from "three";


type Layer = { id: number; name: string; stage: string; dataUrl: string; intensity: string; shape: string; channels: number; receivedAt: number };

const pgmToDataURL = (pgmBase64: string) => {
  const binaryString = atob(pgmBase64);
  const bytes = new Uint8Array(binaryString.length);
  for(let i=0; i<binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  let offset=0;
  const nextToken = () => {
    while(offset<bytes.length && bytes[offset]<=32) offset++;
    if(offset>=bytes.length) return null;
    let start=offset;
    while(offset<bytes.length && bytes[offset]>32) offset++;
    return String.fromCharCode(...Array.from(bytes.slice(start,offset)));
  };
  nextToken(); const w = parseInt(nextToken() || "0"); const h = parseInt(nextToken() || "0"); nextToken(); offset++;
  const canvas = document.createElement("canvas"); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d"); if (!ctx) return "";
  const imgData = ctx.createImageData(w,h);
  let pIdx=0;
  for(let i=offset; i<bytes.length; i++) {
    const val=bytes[i]; imgData.data[pIdx++]=val; imgData.data[pIdx++]=val; imgData.data[pIdx++]=val; imgData.data[pIdx++]=255;
  }
  ctx.putImageData(imgData,0,0);
  return canvas.toDataURL("image/png");
};

const svgData = (index: number, label: string, original = false) => {
  const colors = ["#69f7ff", "#4ce1e8", "#b4ff63", "#ffce55", "#ff6e66"];
  const color = colors[index % colors.length];
  const seed = index * 29 + 7;
  const blobs = Array.from({ length: 7 }, (_, i) => {
    const x = 17 + ((seed * (i + 3)) % 66);
    const y = 18 + ((seed * (i + 5)) % 64);
    const r = 7 + ((seed + i * 8) % 15);
    return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" opacity="${0.14 + (i % 3) * 0.08}"/>`;
  }).join("");
  const grid = Array.from({ length: 8 }, (_, i) => `<path d="M${i * 14} 0V100 M0 ${i * 14}H100" stroke="#d4fbff" stroke-opacity=".07" stroke-width=".5"/>`).join("");
  const center = original ? `<path d="M24 67C29 44 40 28 56 33c17 5 21 21 25 36-12 9-42 12-57-2Z" fill="none" stroke="#b9fbff" stroke-width="2" stroke-opacity=".7"/><circle cx="52" cy="47" r="12" fill="#74f6ff" opacity=".3"/>` : blobs;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#091b27"/><stop offset="1" stop-color="#061016"/></linearGradient><filter id="b"><feGaussianBlur stdDeviation="2.6"/></filter></defs><rect width="100" height="100" fill="url(#g)"/>${grid}<g filter="url(#b)">${center}</g>${center}<text x="6" y="93" fill="#b9fbff" opacity=".72" font-size="4" font-family="monospace" letter-spacing=".6">${label}</text></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
};

const stages = ["INPUT / RGB", "STEM / CONV2", "STEM / CONV3", "RES4 / ATTENTION", "RES5 / LOGITS"];

function LayerPlane({ layer, index, selected, onSelect, revealed }: { layer: Layer; index: number; selected: boolean; onSelect: () => void; revealed: boolean }) {
  const texture = useMemo(() => new THREE.TextureLoader().load(layer.dataUrl), [layer.dataUrl]);
  const groupRef = useRef<THREE.Group>(null);
  
  // Selected layer comes forward, unselected layers drop to the bottom and push back
  const targetZ = selected ? 2.5 : -index * 1.1;
  const targetY = selected ? 0.2 : -2.0 + (Math.sin(performance.now() / 1600 + index) * 0.045);
  const targetX = selected ? 0 : index * -0.15;
  
  useFrame((_, delta) => {
    if (!groupRef.current) return;
    groupRef.current.position.x = THREE.MathUtils.damp(groupRef.current.position.x, targetX, 4.5, delta);
    groupRef.current.position.z = THREE.MathUtils.damp(groupRef.current.position.z, targetZ, 4.5, delta);
    groupRef.current.position.y = THREE.MathUtils.damp(groupRef.current.position.y, targetY, 4.0, delta);
    const targetScale = selected ? 1.2 : 0.85;
    groupRef.current.scale.setScalar(THREE.MathUtils.damp(groupRef.current.scale.x, targetScale, 5, delta));
  });
  
  if (!revealed) return null;
  return (
    <group ref={groupRef} onClick={(event) => { event.stopPropagation(); onSelect(); }} onPointerOver={(event) => { event.stopPropagation(); document.body.style.cursor = "pointer"; }} onPointerOut={() => { document.body.style.cursor = "default"; }}>
      <mesh position={[0, 0, 0.06]}>
        <planeGeometry args={[3.7, 2.85]} />
        <meshBasicMaterial map={texture} color={selected ? "#ffffff" : "#8edfeb"} transparent opacity={selected ? 1.0 : 0.4} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[3.72, 2.87]} />
        <meshBasicMaterial color={["#55e8f4", "#49d9ca", "#b7f566", "#ffd45b", "#ff8b71"][index % 5]} transparent opacity={selected ? 0.15 : 0.05} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0, 0.03]}>
        <planeGeometry args={[3.8, 2.95]} />
        <meshBasicMaterial color={selected ? "#8effff" : "#2b829a"} transparent opacity={selected ? 0.4 : 0.1} wireframe />
      </mesh>
      {selected && <pointLight color="#77f7ff" intensity={1.3} distance={4} position={[0, 0, 0.4]} />}
    </group>
  );
}

function Scene({ layers, selected, onSelect, revealed, originalUrl }: { layers: Layer[]; selected: number; onSelect: (i: number) => void; revealed: number; originalUrl: string }) {
  const group = useRef<THREE.Group>(null);
  useFrame((state, delta) => {
    if (!group.current) return;
    group.current.rotation.y = THREE.MathUtils.damp(group.current.rotation.y, Math.sin(state.clock.elapsedTime * 0.34) * 0.065, 3, delta);
    group.current.rotation.x = THREE.MathUtils.damp(group.current.rotation.x, -0.06 + Math.sin(state.clock.elapsedTime * 0.2) * 0.012, 3, delta);
    group.current.position.y = Math.sin(state.clock.elapsedTime * 0.62) * 0.06;
  });
  return (
    <>
      <PerspectiveCamera makeDefault position={[0.2, 0.12, 7.7]} fov={37} />
      <ambientLight intensity={0.55} color="#c9faff" />
      <directionalLight position={[3, 4, 5]} intensity={1.1} color="#9ffcff" />
      <Environment preset="city" environmentIntensity={0.24} />
      <group ref={group} position={[0.05, 0, 0]}>
        <LayerPlane layer={{ id: 0, name: "Original Input", stage: "INPUT / RGB", dataUrl: originalUrl, intensity: "100%", shape: "32 × 32", channels: 3, receivedAt: 0 }} index={0} selected={selected === 0} onSelect={() => onSelect(0)} revealed={revealed >= 1} />
        {layers.map((layer, i) => <LayerPlane key={layer.id} layer={layer} index={i + 1} selected={selected === i + 1} onSelect={() => onSelect(i + 1)} revealed={revealed >= i + 2} />)}
      </group>
      <OrbitControls enablePan={false} enableZoom={false} minPolarAngle={Math.PI / 2.55} maxPolarAngle={Math.PI / 1.75} target={[0, 0, -1.1]} />
    </>
  );
}

function ScanGrid() {
  return <div className="scan-grid absolute inset-0 pointer-events-none opacity-40" />;
}

export default function Home() {
  // The useAuth hook provides authentication state.
  // To implement login/logout, call logout(), or start login from an event
  // handler: onClick={() => startLogin()} (imported from "@/const"). Never call
  // startLogin() during render (no href={startLogin()}) — it mints a one-time
  // nonce cookie and must run only at the moment of navigation.

  const [status, setStatus] = useState<"idle" | "processing" | "ready" | "error">("idle");
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState("");
  const [originalUrl, setOriginalUrl] = useState(svgData(0, "ORIGINAL / 32×32", true));
  const [layers, setLayers] = useState<Layer[]>([]);
  const [selected, setSelected] = useState(0);
  const [revealed, setRevealed] = useState(0);
  const [timelineCursor, setTimelineCursor] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const pausedRef = useRef(false);
  const [prediction, setPrediction] = useState("");
  const [fileName, setFileName] = useState("awaiting specimen");
  const inputRef = useRef<HTMLInputElement>(null);
  const layerCounter = useRef(0);

  const demoLayers = useMemo(() => stages.slice(1).map((stage, i) => ({ id: i + 1, name: `Feature map ${String(i + 1).padStart(2, "0")}`, stage, dataUrl: svgData(i + 1, `${stage} / ${32 >> Math.min(i, 2)}×${32 >> Math.min(i, 2)}`), intensity: `${94 - i * 11}%`, shape: `${32 >> Math.min(i, 2)} × ${32 >> Math.min(i, 2)}`, channels: [64, 128, 256, 512][i], receivedAt: 0 })), []);

  const runInference = async (file?: File) => {
    setStatus("processing"); setPrediction(""); setRevealed(0); setSelected(0); setLayers([]); setTimelineCursor(0); setIsPaused(false); pausedRef.current = false; layerCounter.current = 0;
    if (!file) {
      setFileName("demo specimen / synthetic");
      const demo = svgData(0, "DEMO / 32×32", true); setPreview(demo); setOriginalUrl(demo); setStatus("ready"); setRevealed(1);
      demoLayers.forEach((layer, i) => window.setTimeout(() => { setLayers(prev => [...prev, { ...layer, receivedAt: performance.now() }]); if (!pausedRef.current) setTimelineCursor(i + 1); setRevealed(i + 2); if (i === demoLayers.length - 1) setPrediction("TRUCK"); }, 520 + i * 360));
      return;
    }
    setFileName(file.name);
    const url = await new Promise<string>((resolve) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsDataURL(file); });
    setPreview(url); setOriginalUrl(url);
    try {
      const image = new Image(); image.src = url; await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; });
      const canvas = document.createElement("canvas"); canvas.width = 32; canvas.height = 32;
      const ctx = canvas.getContext("2d"); if (!ctx) throw new Error("Canvas unavailable");
      ctx.drawImage(image, 0, 0, 32, 32); const pixels = ctx.getImageData(0, 0, 32, 32).data; const tensorBuffer = new Float32Array(3072);
      for (let c = 0; c < 3; c++) for (let i = 0; i < 1024; i++) tensorBuffer[c * 1024 + i] = (pixels[i * 4 + c] / 255.0 - 0.5) / 0.5;
      const socket = new WebSocket("ws://localhost:8000/ws");
      socket.binaryType = "arraybuffer";
      socket.onopen = () => { socket.send(tensorBuffer); };
      socket.onmessage = (event) => {
        const msg = JSON.parse(typeof event.data === "string" ? event.data : new TextDecoder().decode(event.data));
        if (msg.type === "layer") {
          const layerIndex = layerCounter.current++;
          const metadata = msg.metadata || {};
          const rawShape = msg.shape ?? msg.tensor_shape ?? metadata.shape ?? metadata.tensor_shape;
          const shape = Array.isArray(rawShape) ? rawShape.join(" × ") : String(rawShape || "streaming tensor").replace(/[x,]/g, " × ");
          const channels = Number(msg.channels ?? msg.channel_count ?? metadata.channels ?? metadata.channel_count ?? 0);
          
          setTimeout(() => {
            const nextLayer: Layer = { id: layerIndex + 1, name: msg.filename || `Feature map ${String(layerIndex + 1).padStart(2, "0")}`, stage: stages[Math.min(layerIndex + 1, stages.length - 1)], dataUrl: pgmToDataURL(msg.data), intensity: `${92 - layerIndex * 10}%`, shape, channels, receivedAt: performance.now() };
            setLayers(prev => [...prev, nextLayer]); 
            if (!pausedRef.current) { setTimelineCursor(layerIndex + 1); setSelected(layerIndex + 1); }
            setRevealed(layerIndex + 2); 
            setStatus("ready");
          }, layerIndex * 600); // Cinematic 600ms stagger!
          
        } else if (msg.type === "prediction") {
          const layerIndex = layerCounter.current;
          setTimeout(() => {
            setPrediction(String(msg.data)); 
            if (!pausedRef.current) setTimelineCursor(layerIndex); 
            setRevealed(layerIndex + 1); 
            socket.close();
          }, layerIndex * 600 + 400);
        }
      };
      socket.onerror = () => { setStatus("ready"); setPrediction("OFFLINE"); };
    } catch { setStatus("ready"); setPrediction("OFFLINE"); }
  };

  const onDrop = (event: React.DragEvent) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files?.[0]; if (file?.type.startsWith("image/")) runInference(file); };
  const reset = () => { setStatus("idle"); setLayers([]); setRevealed(0); setTimelineCursor(0); setIsPaused(false); pausedRef.current = false; setPrediction(""); setPreview(""); setFileName("awaiting specimen"); setOriginalUrl(svgData(0, "ORIGINAL / 32×32", true)); };
  const selectedLayer = selected === 0 ? { name: "Original Input", stage: "INPUT / RGB", intensity: "100%" } : layers[selected - 1] || layers[0];
  const visibleLayerCount = Math.min(timelineCursor, layers.length);
  const totalLocked = Math.min(visibleLayerCount + 1, 5);
  const isAtLiveEdge = timelineCursor >= layers.length;
  const scrubTimeline = (value: number) => { setTimelineCursor(value); setIsPaused(value < layers.length); pausedRef.current = value < layers.length; setRevealed(Math.min(value + 1, 5)); };
  const resumeLive = () => { setTimelineCursor(layers.length); setIsPaused(false); pausedRef.current = false; setRevealed(Math.min(layers.length + 1, 5)); };
  const pauseReplay = () => { setIsPaused(true); pausedRef.current = true; };

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#05090d] text-slate-100 selection:bg-cyan-300 selection:text-slate-950">
      <ScanGrid />
      <div className="noise absolute inset-0 pointer-events-none" />
      <AnimatePresence mode="wait">
        {status === "idle" ? (
          <motion.section key="idle" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.94, filter: "blur(8px)" }} transition={{ duration: 0.55 }} className="idle-dashboard relative z-10 flex min-h-screen flex-col items-center justify-center px-6 pt-20">
            <div className="mb-8 text-center"><h1 className="max-w-3xl font-display text-5xl font-light tracking-[-0.055em] text-white md:text-7xl">See the network<br /><span className="text-cyan-200">think in layers.</span></h1><p className="mx-auto mt-5 max-w-lg text-sm leading-6 text-slate-400">Upload a specimen to project its latent representations into a living 3D activation stack.</p></div>
            <div onDrop={onDrop} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onClick={() => inputRef.current?.click()} className={`group relative w-full max-w-[590px] cursor-pointer rounded-md border border-dashed p-8 text-center transition-all duration-500 md:p-12 ${dragging ? "border-cyan-200 bg-cyan-200/[0.12] shadow-[0_0_80px_rgba(88,238,255,0.18)]" : "border-cyan-100/25 bg-white/[0.025] hover:border-cyan-200/50 hover:bg-white/[0.045]"}`}><input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && runInference(e.target.files[0])} /><div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-md border border-cyan-100/20 bg-cyan-100/[0.05] text-cyan-100 transition group-hover:scale-105 group-hover:bg-cyan-100/10"><UploadCloud size={26} strokeWidth={1.3} /></div><p className="text-sm font-medium text-white">Drop an image to initialize</p><p className="mt-2 font-mono text-[10px] tracking-[0.18em] text-slate-500">PNG / JPG · NORMALIZED TO 32×32 RGB</p><div className="mt-7 flex items-center justify-center gap-3"><span className="h-px w-10 bg-white/10" /><span className="font-mono text-[9px] text-slate-600">OR</span><span className="h-px w-10 bg-white/10" /></div><button onClick={(e) => { e.stopPropagation(); runInference(); }} className="mt-6 inline-flex items-center gap-2 rounded-md border border-cyan-100/25 bg-cyan-100/[0.08] px-4 py-2 text-[11px] font-medium text-cyan-50 transition hover:bg-cyan-100/[0.15]">Load demo scan <ArrowUpRight size={13} /></button></div>
          </motion.section>
        ) : status === "processing" ? (
          <motion.section key="processing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative z-10 flex min-h-screen items-center justify-center px-6"><div className="w-full max-w-[620px] overflow-hidden rounded-[28px] border border-cyan-100/20 bg-[#071219]/80 p-5 shadow-[0_0_100px_rgba(88,238,255,0.12)] backdrop-blur-xl"><div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-[#08151d]"><img src={preview || originalUrl} className="h-full w-full object-cover opacity-70" /><div className="absolute inset-x-0 h-px animate-scan bg-gradient-to-r from-transparent via-cyan-200 to-transparent shadow-[0_0_20px_#6ef3ff]" /><div className="absolute inset-0 flex items-center justify-center"><div className="rounded-full border border-cyan-100/20 bg-[#061016]/70 px-4 py-2 font-mono text-[10px] tracking-[0.24em] text-cyan-100 backdrop-blur">PROCESSING TENSOR…</div></div></div><div className="flex items-end justify-between px-1 pt-5"><div><p className="font-mono text-[10px] tracking-[0.18em] text-cyan-100/70">INGEST / NORMALIZE / PROJECT</p><p className="mt-2 text-sm text-slate-400">Mapping raw pixels into latent space</p></div><p className="font-display text-4xl text-white">32<span className="text-lg text-cyan-200">²</span></p></div></div></motion.section>
        ) : (
          <motion.section key="ready" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 z-10" onWheel={(e) => { if (e.deltaY > 0) setSelected(prev => Math.min(prev + 1, visibleLayerCount)); else setSelected(prev => Math.max(prev - 1, 0)); }}><Canvas dpr={[1, 1.75]} gl={{ antialias: true, alpha: true }}><Scene layers={layers.slice(0, visibleLayerCount)} selected={selected} onSelect={setSelected} revealed={visibleLayerCount + 1} originalUrl={originalUrl} /></Canvas><div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_52%_48%,transparent_0,rgba(5,9,13,0.08)_36%,rgba(5,9,13,0.76)_100%)]" />
            <div className="absolute left-6 top-24 max-w-[240px] md:left-10"><motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }} className="glass-panel p-4"><div className="flex items-center justify-between"><span className="flex items-center gap-2 font-mono text-[9px] tracking-[0.18em] text-cyan-100/70"><Activity size={12} /> LIVE INFERENCE</span><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-lime-300" /></div><div className="mt-4 flex items-center gap-3"><div className="h-10 w-10 overflow-hidden rounded-lg border border-white/10"><img src={originalUrl} className="h-full w-full object-cover" /></div><div className="min-w-0"><p className="truncate text-xs text-white">{fileName}</p><p className="mt-1 font-mono text-[9px] text-slate-500">STREAMING FROM C++</p></div></div><div className="mt-4 h-px bg-white/10" /> <div className="mt-3 font-mono text-[9px] text-slate-500">ENGINE STATUS: <span className="text-cyan-100">{prediction ? "COMPLETE" : "COMPUTING..."}</span></div></motion.div></div>
            <div className="absolute right-6 top-24 text-right md:right-10"><motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: prediction ? 1 : 0.35, y: 0 }} transition={{ delay: 1.4, duration: 0.7 }}><p className="font-mono text-[10px] tracking-[0.26em] text-cyan-100/60">NETWORK PREDICTION</p><p className="mt-1 font-display text-5xl font-light tracking-[-0.06em] text-white drop-shadow-[0_0_24px_rgba(126,244,255,0.65)] md:text-7xl">{prediction ? prediction : "—"}</p></motion.div></div>
            
            <div className="absolute right-6 top-[222px] hidden w-[228px] md:right-10 md:block"><AnimatePresence>{selected !== 0 && selectedLayer && (<motion.div initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 18 }} transition={{ type: "spring", stiffness: 180, damping: 22 }} className="glass-panel p-4"><div className="mb-2 flex items-center justify-between border-b border-white/10 pb-2 font-mono text-[9px] tracking-[0.18em] text-cyan-200"><span>LAYER INSPECTOR</span></div><p className="mt-2 text-sm text-white font-medium">{selectedLayer.name}</p><p className="text-xs text-slate-400 mt-1">{selectedLayer.stage}</p><div className="mt-4 space-y-2"><div className="flex justify-between font-mono text-[9px] text-slate-500"><span>TENSOR SHAPE</span><b className="text-slate-200">{selectedLayer.shape}</b></div><div className="flex justify-between font-mono text-[9px] text-slate-500"><span>CHANNELS</span><b className="text-slate-200">{selectedLayer.channels || "—"}</b></div><div className="flex justify-between font-mono text-[9px] text-slate-500"><span>LATENCY</span><b className="text-slate-200">{selectedLayer.receivedAt ? `+${Math.round(selectedLayer.receivedAt % 1000)}ms` : "—"}</b></div></div></motion.div>)}</AnimatePresence></div>
            <div className="absolute bottom-6 left-6 right-6 flex items-end justify-between md:bottom-9 md:left-10 md:right-10"><div className="glass-panel pointer-events-auto flex max-w-[360px] items-center gap-3 p-3"><div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-200/10 text-cyan-100"><Layers3 size={16} /></div><div><p className="font-mono text-[9px] tracking-[0.16em] text-cyan-100/70">SELECTED LAYER</p><p className="mt-1 text-xs text-white">{selectedLayer?.name || "Awaiting layer"} <span className="mx-1 text-slate-600">/</span> <span className="text-slate-400">{selectedLayer?.stage || "—"}</span></p></div></div><div className="pointer-events-auto flex gap-2"><button onClick={reset} className="glass-button" title="Reset"><RotateCcw size={15} /></button><button onClick={() => alert("Viewport controls: drag to orbit · click a layer to pull it forward")} className="glass-button" title="Viewport help"><Maximize2 size={15} /></button></div></div>
            <div className="pointer-events-none absolute bottom-9 left-1/2 hidden -translate-x-1/2 items-center gap-2 font-mono text-[9px] tracking-[0.16em] text-slate-600 md:flex"><span className="h-1 w-1 rounded-full bg-cyan-200/70" /> SCROLL OR CLICK TO BROWSE LAYERS <span className="h-1 w-1 rounded-full bg-cyan-200/70" /></div>
          </motion.section>
        )}
      </AnimatePresence>
    </main>
  );
}
