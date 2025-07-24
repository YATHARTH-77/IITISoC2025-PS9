import React, { useState, useRef } from 'react';
import { 
  FileText, 
  Upload, 
  Image as ImageIcon, 
  X, 
  Globe, 
  Zap, 
  AlertCircle, 
  Loader2,
  Copy, 
  Download, 
  ArrowLeft, 
  CheckCircle,
  Clock,
  Target
} from 'lucide-react';

// Header Component
const Header = () => {
  return (
    <div className="text-center mb-12">
      <div className="flex items-center justify-center gap-3 mb-4">
        <div className="p-3 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-2xl shadow-lg shadow-teal-500/30">
          <FileText className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-teal-400 to-cyan-400 bg-clip-text text-transparent">
          PolyOCR
        </h1>
      </div>
      <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
        Advanced optical character recognition with multi-language support.
      </p>
    </div>
  );
};

// Upload Section Component
const UploadSection = ({ onFileSelect }) => {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFile = (file) => {
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
      onFileSelect(file);
    } else {
      alert('Please upload a valid image file (JPEG or PNG).');
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    
    if (x < rect.left || x >= rect.right || y < rect.top || y >= rect.bottom) {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  return (
    <div
      className={`
        bg-slate-800/50 backdrop-blur-xl border-2 border-dashed rounded-3xl p-8 mb-8 text-center cursor-pointer
        transition-all duration-300 hover:bg-slate-800/70 hover:-translate-y-1
        ${isDragOver 
          ? 'border-teal-400 bg-teal-500/10 shadow-lg shadow-teal-500/30' 
          : 'border-slate-600 hover:border-slate-500'
        }
      `}
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png"
        onChange={handleFileChange}
        className="hidden"
      />
      
      <div className={`
        inline-flex p-4 rounded-full mb-4 transition-all duration-300
        ${isDragOver 
          ? 'bg-teal-500/20 text-teal-400' 
          : 'bg-slate-700/50 text-slate-400 hover:bg-teal-500/20 hover:text-teal-400'
        }
      `}>
        <Upload className="w-8 h-8" />
      </div>
      
      <h3 className={`text-xl font-semibold mb-2 transition-colors duration-300 ${
        isDragOver ? 'text-teal-400' : 'text-slate-200'
      }`}>
        Upload Image For OCR Processing
      </h3>
      <p className="text-slate-400 mb-1">
        Drag and drop your image here or click to browse
      </p>
      <p className="text-slate-500 text-sm mb-6">
        Supported formats: JPG, PNG (Maximum 1 file)
      </p>
      
      <button className="bg-teal-500 hover:bg-teal-600 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:-translate-y-0.5">
        Choose File
      </button>
    </div>
  );
};

// Image Preview Component
const ImagePreview = ({ file, onRemove }) => {
  const [imageUrl, setImageUrl] = useState('');

  React.useEffect(() => {
    const url = URL.createObjectURL(file);
    setImageUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  return (
    <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-3xl p-6 mb-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-teal-500/20 text-teal-400 rounded-lg">
            <ImageIcon className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-200">Uploaded Image</h3>
            <p className="text-slate-400 text-sm truncate max-w-[200px] md:max-w-none">
              {file.name}
            </p>
          </div>
        </div>
        
        <button
          onClick={onRemove}
          className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all duration-200"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      
      <div className="relative">
        <img
          src={imageUrl}
          alt="Uploaded for OCR processing"
          className="w-full max-h-64 object-contain bg-slate-900/50 rounded-xl border border-slate-700/30 shadow-lg"
        />
      </div>
    </div>
  );
};

// Language Selection Component
const LanguageSelection = ({ selectedLanguage, onLanguageChange }) => {
  const languages = [
    { value: 'AutoDetect', label: 'Auto Detect' },
    { value: 'en', label: 'English' },
    { value: 'hin', label: 'Hindi' },
    { value: 'Chinese', label: 'Chinese' },
    { value: 'French', label: 'French' },
    { value: 'Spanish', label: 'Spanish' },
  ];

  return (
    <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-3xl p-6 mb-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
          <Globe className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-200">Language Settings</h3>
          <p className="text-slate-400 text-sm">Select language to detect</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {languages.map((language) => (
          <label
            key={language.value}
            className={`
              relative flex items-center justify-center p-3 rounded-xl cursor-pointer
              transition-all duration-200 border-2
              ${selectedLanguage === language.value
                ? 'bg-teal-500/20 border-teal-400 text-teal-400'
                : 'bg-slate-700/30 border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500'
              }
            `}
          >
            <input
              type="radio"
              name="language"
              value={language.value}
              checked={selectedLanguage === language.value}
              onChange={(e) => onLanguageChange(e.target.value)}
              className="absolute opacity-0 pointer-events-none"
            />
            <span className="font-medium text-sm">{language.label}</span>
            
            {selectedLanguage === language.value && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-teal-400 rounded-full border-2 border-slate-900" />
            )}
          </label>
        ))}
      </div>
    </div>
  );
};

// Process Button Component
const ProcessButton = ({ canProcess, isProcessing, onProcess }) => {
  return (
    <div className="flex flex-col items-center gap-4 mb-16">
      <button
        onClick={onProcess}
        disabled={!canProcess}
        className={`
          flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold text-lg
          transition-all duration-300 relative overflow-hidden
          ${canProcess && !isProcessing
            ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg shadow-teal-500/40 hover:-translate-y-1 hover:shadow-xl hover:shadow-teal-500/60'
            : 'bg-slate-600 text-slate-400 cursor-not-allowed'
          }
          ${isProcessing ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white' : ''}
        `}
      >
        {isProcessing ? (
          <>
            <Loader2 className="w-6 h-6 animate-spin" />
            <span>Processing Image...</span>
          </>
        ) : (
          <>
            <Zap className="w-6 h-6" />
            <span>Process Image</span>
          </>
        )}
      </button>
      
      {!canProcess && !isProcessing && (
        <div className="flex items-center gap-2 text-amber-400 text-sm">
          <AlertCircle className="w-4 h-4" />
          <span>Please upload an image and select a language</span>
        </div>
      )}
    </div>
  );
};

// Results View Component
const ResultsView = ({ results, fileName, onNewImage }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(results.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([results.text], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `ocr-result-${fileName.split('.')[0]}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'text-green-400';
    if (confidence >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 90) return 'Excellent';
    if (confidence >= 70) return 'Good';
    return 'Fair';
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-2xl shadow-lg shadow-teal-500/30">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-slate-200">OCR Results</h2>
            <p className="text-slate-400">Extracted from: {fileName}</p>
          </div>
        </div>
        
        <button
          onClick={onNewImage}
          className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg transition-colors duration-200"
        >
          <ArrowLeft className="w-4 h-4" />
          New Image
        </button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Confidence</p>
              <p className={`text-lg font-semibold ${getConfidenceColor(results.confidence)}`}>
                {results.confidence}% ({getConfidenceLabel(results.confidence)})
              </p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-500/20 text-green-400 rounded-lg">
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Processing Time</p>
              <p className="text-lg font-semibold text-slate-200">{results.processingTime}s</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/20 text-purple-400 rounded-lg">
              <Globe className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Language</p>
              <p className="text-lg font-semibold text-slate-200">{results.language}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Extracted Text */}
      <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-200">Extracted Text</h3>
          <div className="flex gap-2">
            <button
              onClick={handleCopy}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium
                transition-all duration-200
                ${copied 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                }
              `}
            >
              {copied ? (
                <>
                  <CheckCircle className="w-4 h-4" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  Copy
                </>
              )}
            </button>
            
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>
        </div>
        
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
          <pre className="text-slate-200 whitespace-pre-wrap leading-relaxed text-sm font-mono">
            {results.text}
          </pre>
        </div>
        
        <div className="mt-4 flex items-center gap-2 text-slate-400 text-xs">
          <FileText className="w-3 h-3" />
          <span>{results.text.length} characters extracted</span>
        </div>
      </div>
    </div>
  );
};

// Footer Component
const Footer = () => {
  return (
    <div className="text-center pt-8 border-t border-slate-700/30">
      <p className="text-slate-500 text-sm">
        Powered by advanced OCR technology • Supports multiple languages
      </p>
    </div>
  );
};

// Main App Component
function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
  };

  const handleLanguageChange = (language) => {
    setSelectedLanguage(language);
  };

  const handleProcess = async () => {
    if (!selectedFile || !selectedLanguage) return;

    setIsProcessing(true);
    
    // Simulate OCR processing
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Mock results - in a real app, this would be an API call
    const mockResult = {
      text: `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

This is a sample OCR result showing extracted text from your uploaded image. The actual implementation would connect to your OCR processing backend.`,
      confidence: 94.7,
      language: selectedLanguage === 'AutoDetect' ? 'English' : selectedLanguage,
      processingTime: 2.8
    };

    setResults(mockResult);
    setIsProcessing(false);
    setShowResults(true);
  };

  const handleNewImage = () => {
    setShowResults(false);
    setResults(null);
    setSelectedFile(null);
    setSelectedLanguage('');
  };

  const canProcess = selectedFile && selectedLanguage && !isProcessing;

  if (showResults && results) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden">
        {/* Background decoration */}
        <div className="fixed inset-0 bg-gradient-radial from-teal-500/10 via-transparent to-transparent pointer-events-none" />
        
        <div className="relative z-10 max-w-4xl mx-auto px-4 py-8">
          <Header />
          <ResultsView 
            results={results} 
            fileName={selectedFile?.name || 'Unknown'}
            onNewImage={handleNewImage}
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
          <UploadSection onFileSelect={handleFileSelect} />
        ) : (
          <ImagePreview 
            file={selectedFile} 
            onRemove={handleRemoveImage} 
          />
        )}
        
        <LanguageSelection 
          selectedLanguage={selectedLanguage}
          onLanguageChange={handleLanguageChange}
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