import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, Image as ImageIcon, Globe, PenTool, Loader2, Zap, AlertCircle, ChevronDown, Check, LetterText } from 'lucide-react';

export const UploadSection = ({ onFileSelect }) => {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const handleFile = (file) => {
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png'|| file.type === 'image/jpg')) {
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
    setIsDragOver(false);
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
        accept="image/jpeg,image/png,image/jpg"
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

export const ImagePreview = ({ file, onRemove }) => {
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
      {imageUrl &&  <img
          src={imageUrl}
          alt="Uploaded for OCR processing"
          className="w-full max-h-64 object-contain bg-slate-900/50 rounded-xl border border-slate-700/30 shadow-lg"
          />}
      </div>
    </div>
  );
};

export const LanguageSelection = ({ selectedLanguage, onLanguageChange }) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);
  const isAutoDetect = selectedLanguage === 'AutoDetect';
  
  const languages = [
    { value: 'en', label: 'English' },
    { value: 'hin', label: 'Hindi' },
    { value: 'ru', label: 'Russian' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'ko', label: 'Korean' },
    { value: 'de', label: 'German' },
    { value: 'it', label: 'Italian' },
    { value: 'tr', label: 'Turkish' },
  ];

  const handleAutoDetectClick = () => {
    if (isAutoDetect) {
      onLanguageChange('');
    } else {
      onLanguageChange('AutoDetect');
    }
    setIsDropdownOpen(false);
  };

  const handleLanguageSelect = (languageValue) => {
    onLanguageChange(languageValue);
    setIsDropdownOpen(false);
  };

  const handleDropdownToggle = () => {
    if (!isAutoDetect) {
      setIsDropdownOpen(!isDropdownOpen);
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isDropdownOpen]);

  const selectedLanguageLabel = languages.find(lang => lang.value === selectedLanguage)?.label || 'Select Language';

  return (
    <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-3xl p-6 mb-8 relative z-10">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
          <Globe className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-200">Language Settings</h3>
          <p className="text-slate-400 text-sm">Select language to detect</p>
        </div>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Auto Detect Button */}
        <button
          onClick={handleAutoDetectClick}
          className={`
            flex items-center justify-center p-3 rounded-xl font-medium text-sm
            transition-all duration-200 border-2 min-w-[280px]
            ${isAutoDetect
              ? 'bg-teal-500/20 border-teal-400 text-teal-400'
              : 'bg-slate-700/30 border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500'
            }
          `}
        >
          Auto Detect
          {isAutoDetect && (
            <div className="ml-2 w-2 h-2 bg-teal-400 rounded-full" />
          )}
        </button>

        {/* Language Dropdown Container */}
        <div className="relative flex-1 z-20" ref={dropdownRef}>
          <button
            onClick={handleDropdownToggle}
            className={`
              w-full flex items-center justify-between p-3 rounded-xl font-medium text-sm
              transition-all duration-200 border-2
              ${isAutoDetect 
                ? 'bg-slate-600/30 border-slate-600/50 text-slate-500 cursor-not-allowed' 
                : !selectedLanguage || selectedLanguage === 'AutoDetect'
                  ? 'bg-slate-700/30 border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500 cursor-pointer'
                  : 'bg-teal-500/10 border-teal-400/50 text-teal-300 hover:bg-teal-500/20 cursor-pointer'
              }
            `}
            disabled={isAutoDetect}
          >
            <span className={isAutoDetect ? 'text-slate-500' : ''}>
              {isAutoDetect ? 'Auto Detection Enabled' : selectedLanguageLabel}
            </span>
            <ChevronDown 
              className={`w-4 h-4 transition-transform duration-200 ${
                isDropdownOpen ? 'rotate-180' : ''
              } ${isAutoDetect ? 'text-slate-500' : ''}`} 
            />
          </button>

          {/* Dropdown Menu - Simple relative positioning */}
          {isDropdownOpen && !isAutoDetect && (
            <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-2xl shadow-black/50 max-h-60 overflow-y-auto z-[9999]">
              {languages.map((language) => (
                <button
                  key={language.value}
                  onClick={() => handleLanguageSelect(language.value)}
                  className={`
                    w-full flex items-center justify-between px-4 py-3 text-left text-sm
                    transition-all duration-150 first:rounded-t-xl last:rounded-b-xl
                    ${selectedLanguage === language.value
                      ? 'bg-teal-500/20 text-teal-400 border-l-2 border-teal-400'
                      : 'text-slate-300 hover:bg-slate-700/50 hover:text-slate-200'
                    }
                  `}
                >
                  <span className="font-medium">{language.label}</span>
                  {selectedLanguage === language.value && (
                    <Check className="w-4 h-4 text-teal-400" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* Helper Text */}
      <div className="mt-4 text-xs text-slate-500">
        {isAutoDetect 
          ? 'Language will be automatically detected from the image'
          : selectedLanguage && selectedLanguage !== 'AutoDetect'
            ? `OCR will be optimized for ${selectedLanguageLabel} text`
            : 'Please select a specific language or use auto-detect'
        }
      </div>
    </div>
  );
};

export const HandwrittenCheckbox = ({ isHandwritten, onHandwrittenChange }) => {
  return (
    <div className="z-0 bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-3xl p-6 mb-8">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-purple-500/20 text-purple-400 rounded-lg">
          <PenTool className="w-5 h-5" />
        </div>
        <label className="flex items-center gap-3 cursor-pointer flex-1">
          <input
            type="checkbox"
            checked={isHandwritten}
            onChange={(e) => onHandwrittenChange(e.target.checked)}
            className="w-5 h-5 rounded border-2 border-slate-600 bg-slate-700/50 text-teal-500 focus:ring-2 focus:ring-teal-500/50 focus:border-teal-400 transition-all duration-200"
          />
          <div>
            <span className="text-lg font-semibold text-slate-200">Handwritten Text</span>
            <p className="text-slate-400 text-sm">Enable enhanced recognition for handwritten content</p>
          </div>
        </label>
      </div>
    </div>
);
};

export const SpellCheck = ({ isSpellChecked, onSpellCheckChange }) => {
  return (
    <div className="z-0 bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-3xl p-6 mb-8">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-purple-500/20 text-purple-400 rounded-lg">
          <LetterText className="w-5 h-5" />
        </div>
        <label className="flex items-center gap-3 cursor-pointer flex-1">
          <input
            type="checkbox"
            checked={isSpellChecked}
            onChange={(e) => onSpellCheckChange(e.target.checked)}
            className="w-5 h-5 rounded border-2 border-slate-600 bg-slate-700/50 text-teal-500 focus:ring-2 focus:ring-teal-500/50 focus:border-teal-400 transition-all duration-200"
          />
          <div>
            <span className="text-lg font-semibold text-slate-200">Spell Check</span>
            <p className="text-slate-400 text-sm">Enable enhanced recognition for spelling errors</p>
          </div>
        </label>
      </div>
    </div>
);
};

export const ProcessButton = ({ canProcess, isProcessing, onProcess }) => {
  return (
    <div className="z-0 flex flex-col items-center gap-4 mb-16">
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