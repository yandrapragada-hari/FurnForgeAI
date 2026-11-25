import React, { useState, useRef } from 'react';
import { 
  Download, RefreshCw, Eye, Sparkles, Sliders, Layers, 
  CheckCircle2, Info, ArrowLeftRight, Share2, Maximize2, Box, Camera 
} from 'lucide-react';
import { COLORS } from '../utils/presets';
import Model3DViewer from './Model3DViewer';
import ViewInRoomModal from './ViewInRoomModal';

export default function ResultsViewer({
  resultData,
  isGenerating,
  generationJob,
  onRegenerate,
  onQuickTweak,
  onOpenInspector,
  onModelUpdated,
  rawSketchUrl
}) {
  const [viewMode, setViewMode] = useState('slider'); // 'slider', 'sideBySide', 'edges', '3d'
  const [autoStart3DTrigger, setAutoStart3DTrigger] = useState(0);
  const [isRoomViewOpen, setIsRoomViewOpen] = useState(false);
  const [activeVarIdx, setActiveVarIdx] = useState(0);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const sliderRef = useRef(null);

  if (isGenerating) {
    const progressPct = generationJob ? Math.round((generationJob.progress || 0) * 100) : 15;
    const msg = generationJob?.message || 'Synthesizing Photoreal Furniture...';
    const stepInfo = generationJob?.total_steps ? `Step ${generationJob.step} of ${generationJob.total_steps}` : null;
    const elapsed = generationJob?.elapsed_seconds ? `${generationJob.elapsed_seconds}s` : null;

    return (
      <div className="bg-white rounded-2xl sm:rounded-3xl p-5 sm:p-8 border border-[#E8E2D5] shadow-xs flex flex-col items-center justify-center min-h-[340px] sm:min-h-[460px] text-center space-y-5 sm:space-y-6">
        <div className="relative">
          <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full border-4 border-[#F0ECE1] border-t-[#8B5A2B] animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center text-[#8B5A2B]">
            <Sparkles className="w-7 h-7 sm:w-8 sm:h-8 animate-pulse" />
          </div>
        </div>

        <div className="space-y-2 max-w-md w-full">
          <span className="text-xs uppercase font-bold tracking-widest text-[#8B5A2B]">
            Generative Diffusion Pipeline Active
          </span>
          <h4 className="text-lg sm:text-xl font-bold text-[#1A1A1E]">{msg}</h4>
          
          {/* Real Denoising Progress Bar */}
          <div className="space-y-1.5 pt-2">
            <div className="flex justify-between text-xs text-[#78716C]">
              <span>{stepInfo || 'Latent Denoising'}</span>
              <span className="font-mono font-bold text-[#8B5A2B]">{progressPct}% {elapsed && `(${elapsed})`}</span>
            </div>
            <div className="w-full bg-[#EAE4D7] h-2.5 rounded-full overflow-hidden">
              <div 
                className="bg-gradient-to-r from-[#8B5A2B] to-[#D4A373] h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.max(5, Math.min(100, progressPct))}%` }}
              />
            </div>
          </div>

          <p className="text-xs text-[#78716C] leading-relaxed pt-1">
            Running Stable Diffusion 1.5 + ControlNet Canny reverse-diffusion denoising iterations.
          </p>
        </div>

        {/* 4 Pipeline Stage Badges */}
        <div className="w-full max-w-md grid grid-cols-1 sm:grid-cols-2 gap-2 text-left text-xs">
          <div className="p-2.5 rounded-xl bg-[#FAF8F5] border border-[#ECE6D8] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="font-medium text-[#444]">1. OpenCV Bilateral Clean</span>
          </div>
          <div className="p-2.5 rounded-xl bg-[#FAF8F5] border border-[#ECE6D8] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            <span className="font-medium text-[#444]">2. CNN Topology & Grad-CAM</span>
          </div>
          <div className="p-2.5 rounded-xl bg-[#FAF8F5] border border-[#ECE6D8] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#8B5A2B] animate-ping" />
            <span className="font-medium text-[#444]">3. ControlNet Canny Lock</span>
          </div>
          <div className="p-2.5 rounded-xl bg-[#FAF8F5] border border-[#ECE6D8] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-purple-500 animate-ping" />
            <span className="font-medium text-[#444]">4. Latent Reverse Diffusion</span>
          </div>
        </div>
      </div>
    );
  }

  if (!resultData) {
    return (
      <div className="bg-white rounded-2xl sm:rounded-3xl p-6 sm:p-8 border border-[#E8E2D5] shadow-xs flex flex-col items-center justify-center min-h-[320px] sm:min-h-[460px] text-center space-y-4">
        <div className="w-16 h-16 rounded-2xl bg-[#F5F2EB] flex items-center justify-center text-[#9E978B]">
          <Layers className="w-8 h-8 stroke-[1.5]" />
        </div>
        <div className="space-y-1">
          <h4 className="text-base font-bold text-[#1A1A1E]">Studio Output Display</h4>
          <p className="text-xs text-[#78716C] max-w-sm">
            Upload a furniture sketch and click "Forge Photoreal Furniture" to inspect the rendered piece, compare variations, and export.
          </p>
        </div>
      </div>
    );
  }

  const variations = resultData.variations && resultData.variations.length > 0
    ? resultData.variations
    : [{ image_url: resultData.primary_image_url, style_variant: 'Studio Master Render', seed: 42 }];

  const currentActiveImage = variations[activeVarIdx]?.image_url || resultData.primary_image_url;
  const sketchImg = rawSketchUrl || resultData.original_image_url;

  const handlePointerMove = (e) => {
    if (!sliderRef.current) return;
    const rect = sliderRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = Math.max(5, Math.min(95, (x / rect.width) * 100));
    setSliderPos(percent);
  };

  const handleDownload = (url, name = 'furnforge-render.png') => {
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const currentColorHex = COLORS.find((c) => c.name === resultData.color)?.hex || '#8B5A2B';

  return (
    <div className="bg-white rounded-2xl sm:rounded-3xl p-4 sm:p-6 md:p-7 border border-[#E8E2D5] shadow-xs space-y-5 sm:space-y-6 overflow-hidden">
      {/* Top Bar with Mode Controls & Action Buttons */}
      <div className="flex flex-col gap-3.5 pb-4 border-b border-[#F0ECE1]">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              Photoreal Render Generated
            </span>
            <span className="text-[11px] text-[#78716C]">
              in {resultData.execution_time}s
            </span>
          </div>
          <h3 className="text-lg sm:text-xl font-bold text-[#1A1A1E] mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
            <span>{resultData.category}</span>
            <span className="text-[#8B5A2B]">•</span>
            <span>{resultData.material}</span>
            <span className="text-[#8B5A2B]">•</span>
            <span className="inline-flex items-center gap-1.5 text-sm sm:text-base font-semibold text-[#555]">
              <span className="w-3.5 h-3.5 rounded-full border border-black/20 shadow-xs inline-block shrink-0" style={{ backgroundColor: currentColorHex }} />
              {resultData.color}
            </span>
            <span className="text-xs font-normal text-[#888]">({resultData.style}, {resultData.finish})</span>
          </h3>
        </div>

        {/* View Switcher Pills */}
        <div className="grid grid-cols-2 sm:flex sm:flex-wrap items-center gap-1.5 bg-[#F5F2EB] p-1.5 rounded-xl text-xs w-full">
          <button
            onClick={() => setViewMode('slider')}
            className={`px-2.5 sm:px-3 py-2 sm:py-1.5 rounded-lg font-medium transition-all text-center whitespace-nowrap ${
              viewMode === 'slider'
                ? 'bg-white text-[#1A1A1E] shadow-xs font-semibold'
                : 'text-[#78716C] hover:text-[#1A1A1E]'
            }`}
          >
            Split Slider
          </button>
          <button
            onClick={() => setViewMode('sideBySide')}
            className={`px-2.5 sm:px-3 py-2 sm:py-1.5 rounded-lg font-medium transition-all text-center whitespace-nowrap ${
              viewMode === 'sideBySide'
                ? 'bg-white text-[#1A1A1E] shadow-xs font-semibold'
                : 'text-[#78716C] hover:text-[#1A1A1E]'
            }`}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setViewMode('edges')}
            className={`px-2.5 sm:px-3 py-2 sm:py-1.5 rounded-lg font-medium transition-all text-center whitespace-nowrap ${
              viewMode === 'edges'
                ? 'bg-[#1A1A1E] text-white shadow-xs font-semibold'
                : 'text-[#78716C] hover:text-[#1A1A1E]'
            }`}
          >
            OpenCV Edges
          </button>
          <button
            onClick={() => setViewMode('3d')}
            className={`px-2.5 sm:px-3 py-2 sm:py-1.5 rounded-lg font-medium transition-all flex items-center justify-center gap-1.5 whitespace-nowrap ${
              viewMode === '3d'
                ? 'bg-[#8B5A2B] text-white shadow-xs font-semibold'
                : 'text-[#78716C] hover:text-[#1A1A1E]'
            }`}
          >
            <Box className="w-3.5 h-3.5 shrink-0" />
            <span>3D Model</span>
          </button>
          <button
            onClick={() => setIsRoomViewOpen(true)}
            className="col-span-2 sm:col-span-1 px-3 py-2 sm:py-1.5 rounded-lg font-semibold transition-all flex items-center justify-center gap-1.5 bg-[#1A1A1E] hover:bg-[#2E2E35] text-white shadow-xs whitespace-nowrap"
            title="Open live camera and place 3D furniture in your room"
          >
            <Camera className="w-3.5 h-3.5 text-amber-300 shrink-0" />
            <span>View in Room</span>
          </button>
        </div>
      </div>

      {/* Main Interactive Canvas Area */}
      <div>
        {viewMode === 'slider' && (
          <div
            ref={sliderRef}
            onMouseMove={(e) => isDragging && handlePointerMove(e)}
            onMouseDown={() => setIsDragging(true)}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
            onTouchStart={() => setIsDragging(true)}
            onTouchEnd={() => setIsDragging(false)}
            onTouchMove={(e) => {
              if (e.touches[0] && sliderRef.current) {
                const rect = sliderRef.current.getBoundingClientRect();
                const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
                setSliderPos(Math.max(2, Math.min(98, (x / rect.width) * 100)));
              }
            }}
            className="relative w-full aspect-square max-h-[500px] mx-auto rounded-2xl overflow-hidden cursor-ew-resize select-none touch-none border border-[#E8E2D5] bg-[#FAF8F5] shadow-inner"
          >
            {/* Bottom layer: Generated render (always full size) */}
            <div className="absolute inset-0 w-full h-full">
              <img
                src={currentActiveImage}
                alt="Rendered Furniture"
                className="w-full h-full object-contain"
                draggable={false}
              />
              <span className="absolute bottom-3 right-3 bg-[#1A1A1E]/80 backdrop-blur-md text-white text-[11px] font-semibold px-2.5 py-1 rounded-lg">
                AI Render
              </span>
            </div>

            {/* Top layer: Sketch, clipped to left of slider position */}
            <div
              className="absolute inset-0 h-full overflow-hidden bg-white"
              style={{ width: `${sliderPos}%` }}
            >
              <img
                src={sketchImg}
                alt="Original Sketch"
                className="w-full h-full object-contain filter contrast-125"
                style={{ width: `${sliderRef.current?.getBoundingClientRect().width || 500}px`, maxWidth: 'none' }}
                draggable={false}
              />
              <span className="absolute top-3 left-3 bg-white/90 backdrop-blur-md text-[#1A1A1E] text-[11px] font-semibold px-2.5 py-1 rounded-lg border border-[#E8E2D5] shadow-xs">
                Hand Sketch
              </span>
            </div>

            {/* Divider line and handle */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.5)] z-20"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-white shadow-lg border-2 border-[#8B5A2B] flex items-center justify-center text-[#8B5A2B] cursor-ew-resize">
                <ArrowLeftRight className="w-3.5 h-3.5" />
              </div>
            </div>
          </div>
        )}

        {viewMode === 'sideBySide' && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="aspect-square rounded-2xl overflow-hidden border border-[#E8E2D5] bg-white flex flex-col">
              <div className="p-2 border-b border-[#F0ECE1] text-[11px] font-bold text-[#78716C] bg-[#FAF8F5]">
                Original Input Sketch
              </div>
              <div className="flex-1 flex items-center justify-center p-3">
                <img src={sketchImg} alt="Sketch" className="max-h-full max-w-full object-contain" />
              </div>
            </div>

            <div className="aspect-square rounded-2xl overflow-hidden border border-[#E8E2D5] bg-[#FAF8F5] flex flex-col">
              <div className="p-2 border-b border-[#F0ECE1] text-[11px] font-bold text-[#8B5A2B] bg-[#FAF8F5] flex items-center justify-between">
                <span>Photoreal Render</span>
                <span className="text-[10px] text-[#78716C]">{variations[activeVarIdx]?.style_variant}</span>
              </div>
              <div className="flex-1 flex items-center justify-center p-3">
                <img src={currentActiveImage} alt="Render" className="max-h-full max-w-full object-contain" />
              </div>
            </div>
          </div>
        )}

        {viewMode !== '3d' && viewMode === 'edges' && (
          <div className="aspect-square max-h-[500px] mx-auto rounded-2xl overflow-hidden border border-[#1A1A1E] bg-black flex flex-col items-center justify-center p-4">
            <img
              src={resultData.preprocessed_edge_url}
              alt="OpenCV Canny Edge Conditioning"
              className="max-h-full max-w-full object-contain filter invert"
            />
            <div className="mt-2 text-center text-xs text-white/70">
              ControlNet Canny Matrix • Locks structural joinery without shape drift
            </div>
          </div>
        )}

        <div className={viewMode === '3d' ? 'block' : 'hidden'}>
          <Model3DViewer
            key={`${resultData.id}-${currentActiveImage}`}
            item={{ ...resultData, result_image_url: currentActiveImage }}
            autoStartTrigger={autoStart3DTrigger}
            onModelUpdated={onModelUpdated}
            onViewInRoom={() => setIsRoomViewOpen(true)}
          />
        </div>
      </div>

      {/* Multiple Variations Strip (if > 1) */}
      {variations.length > 1 && (
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-[#555]">
            Render Variations ({variations.length})
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3">
            {variations.map((v, idx) => (
              <button
                key={idx}
                onClick={() => setActiveVarIdx(idx)}
                className={`relative aspect-square rounded-xl overflow-hidden border transition-all p-1 bg-[#FAF8F5] ${
                  activeVarIdx === idx
                    ? 'border-[#8B5A2B] ring-2 ring-[#8B5A2B]/30 scale-105'
                    : 'border-[#E8E2D5] hover:border-[#8B5A2B]/40'
                }`}
              >
                <img src={v.image_url} alt={`Var ${idx + 1}`} className="w-full h-full object-cover rounded-lg" />
                <span className="absolute bottom-1 right-1 bg-black/70 text-white text-[9px] px-1.5 py-0.5 rounded">
                  #{idx + 1}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Quick Action Toolbar */}
      <div className="pt-2 flex flex-col gap-2.5 sm:gap-3">
        <div className="grid grid-cols-2 sm:flex sm:flex-wrap items-center gap-2">
          <button
            onClick={() => handleDownload(currentActiveImage, `furnforge-${resultData.category.toLowerCase()}.png`)}
            className="px-3 sm:px-4 py-2.5 rounded-xl bg-[#1A1A1E] hover:bg-[#2C2C32] text-white text-xs font-semibold flex items-center justify-center gap-1.5 sm:gap-2 shadow-xs transition-all"
          >
            <Download className="w-3.5 h-3.5 shrink-0" />
            <span className="truncate">Download 8K PNG</span>
          </button>

          <button
            onClick={onRegenerate}
            className="px-3 sm:px-4 py-2.5 rounded-xl bg-white hover:bg-[#F2ECE1] text-[#1A1A1E] border border-[#E0D7C6] text-xs font-semibold flex items-center justify-center gap-1.5 sm:gap-2 transition-all shadow-xs"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[#8B5A2B] shrink-0" />
            <span className="truncate">Regenerate</span>
          </button>

          <button
            onClick={() => {
              setViewMode('3d');
              setAutoStart3DTrigger((n) => n + 1);
            }}
            className="px-3 sm:px-4 py-2.5 rounded-xl bg-[#8B5A2B] hover:bg-[#724821] text-white text-xs font-semibold flex items-center justify-center gap-1.5 sm:gap-2 transition-all shadow-xs"
          >
            <Box className="w-3.5 h-3.5 shrink-0" />
            <span className="truncate">Create 3D Model</span>
          </button>

          <button
            onClick={() => setIsRoomViewOpen(true)}
            className="px-3 sm:px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#8B5A2B] to-[#5C3818] hover:from-[#724821] hover:to-[#4A2D12] text-white text-xs font-semibold flex items-center justify-center gap-1.5 sm:gap-2 transition-all shadow-xs"
          >
            <Camera className="w-3.5 h-3.5 text-amber-200 shrink-0" />
            <span className="truncate">View in Room</span>
          </button>
        </div>

        <button
          onClick={onOpenInspector}
          className="w-full sm:w-auto sm:self-start px-3.5 py-2.5 rounded-xl bg-[#FAF8F5] hover:bg-[#F2ECE1] text-[#5C3818] border border-[#D5C7B0] text-xs font-semibold flex items-center justify-center gap-1.5 transition-all"
        >
          <Info className="w-3.5 h-3.5 text-[#8B5A2B] shrink-0" />
          <span>AI Architecture Inspector</span>
        </button>
      </div>

      {/* Amazon / Flipkart style Live Camera AR Room View */}
      <ViewInRoomModal
        isOpen={isRoomViewOpen}
        onClose={() => setIsRoomViewOpen(false)}
        item={{ ...resultData, result_image_url: currentActiveImage }}
        onModelUpdated={onModelUpdated}
      />
    </div>
  );
}

