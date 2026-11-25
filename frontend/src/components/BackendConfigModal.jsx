import React, { useState, useEffect } from 'react';
import {
  X,
  Server,
  Radio,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Zap,
  Globe,
  ExternalLink,
  Laptop
} from 'lucide-react';
import { getDynamicBackendUrl, setDynamicBackendUrl, checkHealth } from '../services/api';

export default function BackendConfigModal({ isOpen, onClose, onBackendChanged }) {
  const [currentUrl, setCurrentUrl] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    if (isOpen) {
      const active = getDynamicBackendUrl();
      setCurrentUrl(active);
      setInputUrl(active);
      setTestResult(null);
      setSaveSuccess(false);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleTestConnection = async (targetUrl = inputUrl) => {
    setIsTesting(true);
    setTestResult(null);
    const startTime = performance.now();
    try {
      const res = await checkHealth(targetUrl);
      const latencyMs = Math.round(performance.now() - startTime);
      if (res && res.status !== 'offline') {
        const engine = res.pipeline_mode || (res.device ? `Device: ${res.device}` : 'Online');
        setTestResult({
          success: true,
          message: `Connection successful! (${latencyMs}ms)`,
          details: `Service: ${res.app || 'FurnForge'} v${res.version || '2.0'} | ${engine}`
        });
      } else {
        setTestResult({
          success: false,
          message: 'Backend is offline or unreachable.',
          details: res.error || 'Ensure CORS is enabled and port 8005 is forwarded.'
        });
      }
    } catch (err) {
      setTestResult({
        success: false,
        message: 'Connection failed.',
        details: err.message
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveAndApply = async () => {
    const updated = setDynamicBackendUrl(inputUrl);
    setCurrentUrl(updated);
    setSaveSuccess(true);

    if (onBackendChanged) {
      onBackendChanged(updated);
    }

    // Auto test the newly configured backend
    await handleTestConnection(updated);

    setTimeout(() => {
      setSaveSuccess(false);
    }, 2500);
  };

  const handleResetDefault = () => {
    setInputUrl('');
  };

  const handleSetLocalhost = () => {
    setInputUrl('http://localhost:8005');
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl sm:rounded-3xl max-w-xl w-full max-h-[92vh] overflow-y-auto border border-[#E8E2D5] shadow-2xl p-4 sm:p-7 space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-[#F0ECE1]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#8B5A2B]/10 text-[#8B5A2B] flex items-center justify-center">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg sm:text-xl font-bold text-[#1A1A1E]">Configure Backend Connection</h3>
              <p className="text-xs text-[#78716C]">Dynamically switch API & Cloudflare Tunnel endpoints</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-[#F5F2EB] hover:bg-[#EAE4D7] flex items-center justify-center text-[#555] transition-colors"
            title="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Current Active Backend Indicator */}
        <div className="p-3 bg-[#FAF8F5] rounded-xl border border-[#ECE6D8] flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 min-w-0">
            <Radio className="w-4 h-4 text-[#8B5A2B] shrink-0" />
            <span className="text-[#78716C] shrink-0">Active Backend:</span>
            <span className="font-mono font-medium text-[#1A1A1E] truncate">
              {currentUrl || '(Relative /api proxy)'}
            </span>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#E8E2D5] text-[#5C3818] font-semibold shrink-0 ml-2">
            {currentUrl ? 'Custom' : 'Default'}
          </span>
        </div>

        {/* URL Input Form */}
        <div className="space-y-2">
          <label className="block text-xs font-semibold text-[#1A1A1E]">
            Backend URL or Tunnel Endpoint
          </label>
          <div className="relative">
            <input
              type="text"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              placeholder="e.g. https://your-tunnel.trycloudflare.com or http://localhost:8005"
              className="w-full px-3.5 py-2.5 bg-[#FAF8F5] border border-[#D5C7B0] focus:border-[#8B5A2B] rounded-xl text-xs font-mono text-[#1A1A1E] focus:outline-hidden transition-colors"
            />
          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px]">
            <span className="text-[#78716C]">Quick Presets:</span>
            <button
              type="button"
              onClick={handleSetLocalhost}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#F5F2EB] hover:bg-[#EBE5D8] border border-[#E0D7C6] text-[#5C3818] font-medium transition-colors"
            >
              <Laptop className="w-3 h-3" />
              Localhost:8005
            </button>
            <button
              type="button"
              onClick={handleResetDefault}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#F5F2EB] hover:bg-[#EBE5D8] border border-[#E0D7C6] text-[#78716C] font-medium transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              Reset Default
            </button>
          </div>
        </div>

        {/* Test Result Message Box */}
        {testResult && (
          <div
            className={`p-3 rounded-xl border text-xs flex items-start gap-2.5 ${
              testResult.success
                ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                : 'bg-amber-50 border-amber-200 text-amber-900'
            }`}
          >
            {testResult.success ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            )}
            <div className="space-y-0.5">
              <div className="font-semibold">{testResult.message}</div>
              {testResult.details && (
                <div className="text-[11px] opacity-80 font-mono">{testResult.details}</div>
              )}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-between gap-3 pt-2">
          <button
            type="button"
            onClick={() => handleTestConnection(inputUrl)}
            disabled={isTesting}
            className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#EAE2D2] hover:bg-[#E0D5C1] border border-[#D5C7B0] text-[#5C3818] transition-colors flex items-center gap-1.5 disabled:opacity-50"
          >
            <Zap className={`w-3.5 h-3.5 ${isTesting ? 'animate-spin' : ''}`} />
            {isTesting ? 'Testing Ping...' : 'Test Connection'}
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold text-[#78716C] hover:bg-[#F5F2EB] transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSaveAndApply}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#8B5A2B] hover:bg-[#734A23] text-white transition-all shadow-sm flex items-center gap-1.5"
            >
              {saveSuccess ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Applied!
                </>
              ) : (
                'Save & Connect'
              )}
            </button>
          </div>
        </div>

        {/* Dynamic URL Presentation Tip Box */}
        <div className="p-3.5 bg-[#FAF8F5] rounded-xl border border-[#ECE6D8] space-y-2 text-xs text-[#666]">
          <div className="flex items-center gap-1.5 font-bold text-[#1A1A1E]">
            <Globe className="w-3.5 h-3.5 text-[#8B5A2B]" />
            Presentation & Demonstration Tip:
          </div>
          <p className="leading-relaxed">
            You can automatically point this Vercel deployment to any live GPU tunnel simply by sharing a direct link with the query parameter:
          </p>
          <div className="p-2 bg-[#F1ECE1] rounded-lg font-mono text-[11px] text-[#5C3818] select-all break-all">
            https://your-app.vercel.app?backend=https://your-tunnel.trycloudflare.com
          </div>
          <p className="text-[11px] text-[#78716C]">
            When opened, the frontend automatically stores the backend URL in local storage and connects without rebuilding the site.
          </p>
        </div>
      </div>
    </div>
  );
}
