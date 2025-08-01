import React, { useState } from 'react';
import {Header, Footer} from './HeaderFooter.jsx';
import { UploadSection, ImagePreview, LanguageSelection, HandwrittenCheckbox, ProcessButton, SpellCheck } from './FirstView.jsx';
import { ResultsView } from './ResultView.jsx';

// Main App Component
function App() {
  const backendURL = 
                      process.env.REACT_APP_API_URL || 
                     'http://localhost:5000';



  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [isHandwritten, setIsHandwritten] = useState(false);
  const [ocrData, setOcrData] = useState([]);
  const [isSpellChecked, setIsSpellChecked] = useState(false);

  const handleProcess = async () => {
    if (!selectedFile || !selectedLanguage) return;

    setIsProcessing(true);

      // Prepare FormData
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('language', selectedLanguage);
      formData.append('is_handwritten', isHandwritten);
      formData.append('enable_spell_check', isSpellChecked);
      try {
        // Send POST request to backend
        console.log("Sending file:", selectedFile.name);
        console.log("Selected language:", selectedLanguage);
        console.log("Is handwritten:", isHandwritten);
        console.log("Form data:", formData);
        await fetch(`${backendURL}/result`, {
          method: 'POST',
          body: formData,
        });
        setShowResults(true);
      } catch (error) {
        alert(error.message || 'An error occurred during processing.');
      } finally {
        setIsProcessing(false);
      }
    
    
    // Simulate OCR processing
    const data = await fetch(`${backendURL}/static/final_data.json`);
    if (!data.ok) {
      throw new Error(`HTTP error: ${data.status}`);
    }
    const jsonData = await data.json();
    setOcrData(jsonData);
    console.log("OCR Data:", jsonData);
    
    setIsProcessing(false);
    setShowResults(true);
  };

  const handleNewImage = () => {
    setShowResults(false);
    setSelectedFile(null);
    setSelectedLanguage('');
  };

  const canProcess = selectedFile && selectedLanguage && !isProcessing;

  if (showResults && ocrData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden">
        {/* Background decoration */}
        <div className="fixed inset-0 bg-gradient-radial from-teal-500/10 via-transparent to-transparent pointer-events-none" />
        
        <div className="relative z-10 max-w-4xl mx-auto px-4 py-8">
          <Header />
          <ResultsView  
            fileName={selectedFile?.name || 'Unknown'}
            onNewImage={handleNewImage}
            ocrData={ocrData}
          />
          <Footer />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden">
      {/* Background decoration */}
      <div className="fixed inset-0 bg-gradient-radial from-teal-500/10 via-transparent to-transparent pointer-events-none" />
      
      <div className="relative z-10 max-w-4xl mx-auto px-4 py-8">
        <Header />
        
        {!selectedFile ? (
          <UploadSection onFileSelect={(file) => setSelectedFile(file)} />
        ) : (
          <ImagePreview 
            file={selectedFile} 
            onRemove={() => setSelectedFile(null)} 
          />
        )}
        
        <LanguageSelection 
          selectedLanguage={selectedLanguage}
          onLanguageChange={(lang) => setSelectedLanguage(lang)}
        />
        
        <HandwrittenCheckbox 
          isHandwritten={isHandwritten}
          onHandwrittenChange={setIsHandwritten}
        />

        <SpellCheck
          isSpellChecked={isSpellChecked}
          onSpellCheckChange={setIsSpellChecked}
        />

        <ProcessButton 
          canProcess={canProcess}
          isProcessing={isProcessing}
          onProcess={handleProcess}
        />
        
        <Footer />
      </div>
    </div>
  );
}

export default App;