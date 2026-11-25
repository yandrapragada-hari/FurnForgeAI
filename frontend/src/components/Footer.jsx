import React from 'react';
import { Armchair, Sparkles } from 'lucide-react';

export default function Footer({ onOpenTechModal, setActiveTab }) {
  const navigateTo = (tab) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="bg-white border-t border-zinc-200/80 mt-16 sm:mt-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-14 space-y-8">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 pb-8 border-b border-zinc-100">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-zinc-950 text-white flex items-center justify-center shrink-0 border border-zinc-800 shadow-2xs">
              <Armchair className="w-5 h-5 text-amber-500" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-base text-zinc-950 tracking-tight">FurnForge AI</span>
                <span className="text-[10px] font-mono uppercase bg-zinc-100 text-zinc-600 px-2 py-0.5 rounded-full border border-zinc-200">v2.0</span>
              </div>
              <p className="text-xs text-zinc-500">Pencil Sketch to Photoreal Furniture & 3D Neural Studio</p>
            </div>
          </div>

          {/* Quick Links */}
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-xs font-medium text-zinc-600">
            <button onClick={() => navigateTo('home')} className="hover:text-zinc-950 transition-colors">
              Overview
            </button>
            <button onClick={() => navigateTo('studio')} className="hover:text-zinc-950 transition-colors">
              AI Studio
            </button>
            <button onClick={() => navigateTo('gallery')} className="hover:text-zinc-950 transition-colors">
              3D Showcase
            </button>
            <button onClick={onOpenTechModal} className="hover:text-amber-700 font-semibold transition-colors flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-500" />
              <span>AI Architecture</span>
            </button>
          </div>
        </div>

        {/* Bottom Credits & Academic Disclaimer */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-zinc-400 font-mono">
          <p>
            University Generative Vision & Geometry Project • OpenCV, ResNet-50, ControlNet & TripoSR.
          </p>
          <div className="flex items-center gap-2">
            <span>Constraint-Guided Spatial Synthesis</span>
            <span>•</span>
            <span className="text-zinc-700 font-bold">FurnForge AI</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
