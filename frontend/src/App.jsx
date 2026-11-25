import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Studio from './components/Studio';
import Gallery from './components/Gallery';
import TechStackModal from './components/TechStackModal';
import Footer from './components/Footer';
import { checkHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('home'); // 'home', 'studio', 'gallery'
  const [backendStatus, setBackendStatus] = useState(null);
  const [isTechModalOpen, setIsTechModalOpen] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState(null);

  const refreshStatus = async () => {
    const status = await checkHealth();
    setBackendStatus(status);
  };

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, 15000);
    const handleBackendUpdated = () => {
      refreshStatus();
    };
    window.addEventListener('furnforge-backend-updated', handleBackendUpdated);
    return () => {
      clearInterval(interval);
      window.removeEventListener('furnforge-backend-updated', handleBackendUpdated);
    };
  }, []);

  const handleSelectPresetForStudio = (presetId) => {
    setSelectedPresetId(presetId);
    setActiveTab('studio');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#FAFAFA] text-[#09090B] selection:bg-[#0EA5E9]/15 selection:text-[#0284C7]">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
        onOpenTechModal={() => setIsTechModalOpen(true)}
        onBackendChanged={refreshStatus}
      />

      {/* Main Content Pages */}
      <main className="flex-1">
        {activeTab === 'home' && (
          <Hero
            onLaunchStudio={() => {
              setActiveTab('studio');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            onExploreGallery={() => {
              setActiveTab('gallery');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            onSelectPreset={handleSelectPresetForStudio}
          />
        )}

        {activeTab === 'studio' && (
          <Studio selectedPresetId={selectedPresetId} />
        )}

        {activeTab === 'gallery' && (
          <Gallery onSelectForStudio={handleSelectPresetForStudio} />
        )}
      </main>

      {/* University Tech Documentation Modal */}
      <TechStackModal
        isOpen={isTechModalOpen}
        onClose={() => setIsTechModalOpen(false)}
      />

      {/* Footer */}
      <Footer
        onOpenTechModal={() => setIsTechModalOpen(true)}
        setActiveTab={setActiveTab}
      />
    </div>
  );
}
