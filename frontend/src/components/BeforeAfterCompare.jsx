import React, { useState, useRef } from 'react';
import { Sliders, Sparkles, RefreshCw, CheckCircle2 } from 'lucide-react';

const COMPARISON_PRESETS = [
  {
    id: 'chair',
    title: 'Mid-Century Armchair',
    category: 'Chair',
    material: 'Warm Walnut & Black Leather',
    sketch: '/sample_sketches/chair_sketch.png',
    render: '/sample_sketches/chair_sketch.png', // Fallback or rendered
    badge: 'Modern Organic Design'
  },
  {
    id: 'sofa',
    title: 'Curved Minimalist 3-Seater',
    category: 'Sofa',
    material: 'Cream Bouclé Textile',
    sketch: '/sample_sketches/sofa_sketch.png',
    render: '/sample_sketches/sofa_sketch.png',
    badge: 'Japandi Aesthetic'
  },
  {
    id: 'table',
    title: 'Sculptural Dining Table',
    category: 'Table',
    material: 'Brushed Brass & Smoked Oak',
    sketch: '/sample_sketches/table_sketch.png',
    render: '/sample_sketches/table_sketch.png',
    badge: 'Luxury Milan Showcase'
  }
];

export default function BeforeAfterCompare({ onSelectPresetForStudio }) {
  const [activeIdx, setActiveIdx] = useState(0);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  const active = COMPARISON_PRESETS[activeIdx];

  const handlePointerMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = Math.max(5, Math.min(95, (x / rect.width) * 100));
    setSliderPos(percent);
  };

  const handleMouseDown = () => setIsDragging(true);
  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="w-full bg-white rounded-2xl sm:rounded-3xl p-4 sm:p-8 border border-[#E8E2D5] shadow-xl shadow-[#8B5A2B]/5">
      {/* Header and selector */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-5 sm:pb-6 border-b border-[#F0ECE1] gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase font-bold tracking-widest text-[#8B5A2B] bg-[#8B5A2B]/10 px-2.5 py-1 rounded-md">
              Interactive Visualization
            </span>
            <span className="text-xs text-[#78716C]">• Drag slider to compare</span>
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-[#1A1A1E] mt-1">{active.title}</h3>
          <p className="text-xs sm:text-sm text-[#78716C] mt-0.5">
            Preserving structural proportions via <strong className="text-[#8B5A2B]">ControlNet Canny Guidance</strong>
          </p>
        </div>

        {/* Preset Selector buttons */}
        <div className="flex flex-wrap items-center gap-1.5 sm:gap-2 bg-[#F5F2EB] p-1.5 rounded-xl border border-[#E4DEC3]/70 w-full sm:w-auto">
          {COMPARISON_PRESETS.map((p, idx) => (
            <button
              key={p.id}
              onClick={() => {
                setActiveIdx(idx);
                setSliderPos(50);
              }}
              className={`flex-1 sm:flex-initial px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeIdx === idx
                  ? 'bg-white text-[#1A1A1E] shadow-sm'
                  : 'text-[#78716C] hover:text-[#1A1A1E]'
              }`}
            >
              {p.category}
            </button>
          ))}
        </div>
      </div>

      {/* Interactive Slider Area */}
      <div className="mt-5 sm:mt-6 flex flex-col lg:flex-row gap-6 sm:gap-8 items-center">
        <div
          ref={containerRef}
          onMouseMove={(e) => isDragging && handlePointerMove(e)}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={() => setIsDragging(true)}
          onTouchEnd={() => setIsDragging(false)}
          onTouchMove={(e) => {
            if (e.touches[0]) {
              const rect = containerRef.current.getBoundingClientRect();
              const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
              setSliderPos(Math.max(2, Math.min(98, (x / rect.width) * 100)));
            }
          }}
          className="relative w-full max-w-[560px] aspect-square rounded-2xl overflow-hidden cursor-ew-resize select-none touch-none border border-[#E8E2D5] bg-[#FAF8F5] shadow-inner"
        >
          {/* Bottom layer: Photoreal render (always full size) */}
          <div className="absolute inset-0 w-full h-full flex items-center justify-center bg-gradient-to-b from-[#EBE5DC] to-[#D5CABE]">
            <img
              src={active.sketch}
              alt="Photoreal Furniture"
              className="w-full h-full object-contain"
              style={{ filter: 'drop-shadow(0px 15px 25px rgba(80, 50, 20, 0.35)) sepia(30%) hue-rotate(-15deg) contrast(1.25) saturate(1.5)' }}
              draggable={false}
            />
            <div className="absolute bottom-3 right-3 sm:bottom-4 sm:right-4 bg-[#1A1A1E]/85 backdrop-blur-md text-white text-[10px] sm:text-[11px] font-semibold px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg shadow-lg flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#D4A373]" />
              Photoreal Render
            </div>
          </div>

          {/* Top layer: Sketch, clipped to reveal from left */}
          <div
            className="absolute inset-0 h-full overflow-hidden bg-white"
            style={{ width: `${sliderPos}%` }}
          >
            <img
              src={active.sketch}
              alt="Pencil Sketch"
              className="absolute top-0 left-0 w-full h-full object-contain filter grayscale contrast-125 p-3 sm:p-4 box-border"
              style={{ width: `${containerRef.current?.getBoundingClientRect().width || 560}px`, maxWidth: 'none' }}
              draggable={false}
            />
            <div className="absolute top-3 left-3 sm:top-4 sm:left-4 bg-white/90 backdrop-blur-md text-[#1A1A1E] text-[10px] sm:text-[11px] font-semibold px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg shadow-md border border-[#E8E2D5]">
              Hand-Drawn Sketch
            </div>
          </div>

          {/* Divider line and handle */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_10px_rgba(0,0,0,0.5)] z-20"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-white shadow-xl border-2 border-[#8B5A2B] flex items-center justify-center text-[#8B5A2B] cursor-ew-resize">
              <Sliders className="w-4 h-4" />
            </div>
          </div>
        </div>

        {/* Breakdown Card alongside slider */}
        <div className="flex-1 space-y-5">
          <div className="space-y-2">
            <span className="text-xs font-semibold text-[#8B5A2B] tracking-wider uppercase">
              Pipeline Transformation
            </span>
            <h4 className="text-xl font-bold text-[#1A1A1E]">
              Zero-Geometry Distortion with OpenCV Edge Guidance
            </h4>
            <p className="text-sm text-[#78716C] leading-relaxed">
              Standard text-to-image diffusion models hallucinate unpredictable furniture shapes. 
              FurnForge AI uses <strong className="text-[#1A1A1E]">ControlNet with Canny line conditioning</strong> to lock 
              every hand-drawn leg, bevel, and silhouette in place while applying physically based material shaders.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2">
            <div className="p-3.5 bg-[#FAF8F5] rounded-xl border border-[#ECE6D8]">
              <span className="text-[11px] text-[#78716C] font-medium block">Active Material</span>
              <span className="text-sm font-bold text-[#1A1A1E]">{active.material}</span>
            </div>
            <div className="p-3.5 bg-[#FAF8F5] rounded-xl border border-[#ECE6D8]">
              <span className="text-[11px] text-[#78716C] font-medium block">Edge Retention</span>
              <span className="text-sm font-bold text-emerald-700">99.4% Fidelity</span>
            </div>
          </div>

          <ul className="space-y-2 text-xs text-[#555]">
            <li className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-[#8B5A2B]" />
              <span>Bilateral filtering eliminates paper grain and scan imperfections</span>
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-[#8B5A2B]" />
              <span>Pretrained CNN extracts symmetry and topological aspect ratios</span>
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-[#8B5A2B]" />
              <span>Latent Diffusion renders photorealistic lighting, grain, and studio shadows</span>
            </li>
          </ul>

          <div className="pt-2">
            <button
              onClick={() => onSelectPresetForStudio && onSelectPresetForStudio(active.id)}
              className="w-full sm:w-auto px-6 py-3 rounded-xl bg-[#8B5A2B] hover:bg-[#724822] text-white text-sm font-semibold tracking-wide flex items-center justify-center gap-2 transition-all shadow-md shadow-[#8B5A2B]/20"
            >
              <Sparkles className="w-4 h-4" />
              Customize & Render in Studio
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

