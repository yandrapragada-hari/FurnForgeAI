import React, { useState } from 'react';
import { Armchair, Sparkles, Image as ImageIcon, BookOpen, Home, Settings } from 'lucide-react';
import BackendConfigModal from './BackendConfigModal';

export default function Navbar({ activeTab, setActiveTab, backendStatus, onOpenTechModal, onBackendChanged }) {
  const [isBackendModalOpen, setIsBackendModalOpen] = useState(false);

  const navigateTo = (tab) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const isOnline = backendStatus && backendStatus.status !== 'offline';
  const engineLabel = isOnline
    ? backendStatus?.pipeline_mode
      ? backendStatus.pipeline_mode.includes('PyTorch')
        ? 'SD1.5 ControlNet'
        : 'Hybrid AI'
      : 'CV/CNN Active'
    : 'Backend Offline';

  return (
    <header className="sticky top-0 z-50 bg-white/85 backdrop-blur-xl border-b border-zinc-200/80 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 md:h-18 flex items-center justify-between gap-3">
        {/* Brand Logo */}
        <div
          onClick={() => navigateTo('home')}
          className="flex items-center gap-3 cursor-pointer group min-w-0"
        >
          <div className="w-10 h-10 shrink-0 rounded-xl bg-zinc-950 flex items-center justify-center text-white shadow-md shadow-zinc-900/10 group-hover:scale-105 transition-all border border-zinc-800">
            <Armchair className="w-5 h-5 text-amber-500 stroke-[2.2]" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="text-lg font-bold tracking-tight text-zinc-950 truncate">FurnForge</span>
              <span className="text-lg font-medium text-amber-700">AI</span>
              <span className="hidden sm:inline-block text-[10px] font-mono uppercase font-bold tracking-wider bg-zinc-100 text-zinc-600 px-2 py-0.5 rounded-full border border-zinc-200 ml-1">
                STUDIO v2
              </span>
            </div>
            <p className="hidden sm:block text-[11px] text-zinc-500 tracking-normal truncate">
              Pencil Sketch to Photoreal & 3D Neural Studio
            </p>
          </div>
        </div>

        {/* Desktop Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1 bg-zinc-100/80 p-1 rounded-full border border-zinc-200/70 shadow-2xs">
          <button
            onClick={() => navigateTo('home')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
              activeTab === 'home'
                ? 'bg-white text-zinc-950 shadow-xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => navigateTo('studio')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'studio'
                ? 'bg-zinc-950 text-white shadow-xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            AI Studio
          </button>
          <button
            onClick={() => navigateTo('gallery')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'gallery'
                ? 'bg-white text-zinc-950 shadow-xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5 text-zinc-500" />
            Gallery
          </button>
        </nav>

        {/* Right Actions: Gallery Shortcut (mobile), Architecture Docs, Engine Status & Launch CTA */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Quick Gallery Shortcut on Mobile */}
          <button
            onClick={() => navigateTo('gallery')}
            className={`md:hidden flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${
              activeTab === 'gallery'
                ? 'bg-zinc-950 text-white border-zinc-950'
                : 'bg-white text-zinc-800 border-zinc-200 hover:bg-zinc-50'
            }`}
            title="Open Furniture Gallery"
          >
            <ImageIcon className="w-3.5 h-3.5" />
            <span>Gallery</span>
          </button>

          <button
            onClick={onOpenTechModal}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold text-zinc-700 bg-zinc-100 hover:bg-zinc-200 border border-zinc-200/80 transition-colors"
            title="Inspect 5 Core AI Techniques"
            aria-label="Inspect AI Architecture"
          >
            <BookOpen className="w-3.5 h-3.5 text-zinc-500" />
            <span className="hidden sm:inline">AI Architecture</span>
          </button>

          {/* Backend Status Pill (Clickable to configure) */}
          <div
            onClick={() => setIsBackendModalOpen(true)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsBackendModalOpen(true); }}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50 cursor-pointer text-xs font-medium text-zinc-700 shadow-2xs transition-all group"
            title="Click to view or switch backend endpoint"
          >
            <span className="relative flex h-2 w-2 shrink-0">
              {isOnline ? (
                <>
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </>
              ) : (
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
              )}
            </span>
            <span className="hidden lg:inline text-zinc-400 font-mono text-[11px]">API:</span>
            <span className="text-zinc-950 font-semibold whitespace-nowrap hidden sm:inline text-[11px]">
              {engineLabel}
            </span>
            <span className="text-zinc-950 font-semibold sm:hidden text-[11px]">
              {isOnline ? 'AI' : 'Offline'}
            </span>
            <Settings className="w-3.5 h-3.5 text-zinc-400 group-hover:text-zinc-900 group-hover:rotate-45 transition-transform" />
          </div>

          <button
            onClick={() => navigateTo('studio')}
            className="hidden sm:flex items-center gap-2 bg-zinc-950 hover:bg-zinc-800 text-white px-4 py-2 rounded-full text-xs font-semibold tracking-wide transition-all shadow-sm hover:shadow"
          >
            Launch Studio
          </button>
        </div>
      </div>

      <BackendConfigModal
        isOpen={isBackendModalOpen}
        onClose={() => setIsBackendModalOpen(false)}
        onBackendChanged={onBackendChanged}
      />

      {/* Mobile Navigation Bar (Overview / AI Studio / Gallery) */}
      <nav
        aria-label="Mobile navigation"
        className="md:hidden border-t border-zinc-200/80 bg-white/95 px-3 py-2"
      >
        <div className="grid grid-cols-3 gap-1 bg-zinc-100 p-1 rounded-xl border border-zinc-200">
          <button
            onClick={() => navigateTo('home')}
            className={`py-2 px-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
              activeTab === 'home'
                ? 'bg-white text-zinc-950 shadow-2xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
          >
            <Home className="w-3.5 h-3.5" />
            <span>Overview</span>
          </button>
          <button
            onClick={() => navigateTo('studio')}
            className={`py-2 px-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
              activeTab === 'studio'
                ? 'bg-zinc-950 text-white shadow-2xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>AI Studio</span>
          </button>
          <button
            onClick={() => navigateTo('gallery')}
            className={`py-2 px-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
              activeTab === 'gallery'
                ? 'bg-white text-zinc-950 shadow-2xs'
                : 'text-zinc-600 hover:text-zinc-950'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5 text-zinc-500" />
            <span>Gallery</span>
          </button>
        </div>
      </nav>
    </header>
  );
}

