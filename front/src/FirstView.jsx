import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, Image as ImageIcon, Globe, PenTool, Loader2, Zap, AlertCircle, ChevronDown, Check, Text as LetterText, Sparkles, Eye } from 'lucide-react';

export const UploadSection = ({ onFileSelect }) => {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const handleFile = (file) => {
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png'|| file.type === 'image/jpg')) {
      if(file.type === 'image/heif'){
        alert('Please upload a valid image file (JPEG or PNG or JPG) not HEIF.');
      }
      onFileSelect(file);
    } else {
      alert('Please upload a valid image file (JPEG or PNG or JPG).');
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
    <div className="relative group">
      {/* Animated background gradient */}
      <div className="absolute -inset-1 bg-gradient-to-r from-teal-500/20 via-cyan-500/20 to-blue-500/20 rounded-3xl opacity-0 group-hover:opacity-100 transition-all duration-500 blur-lg"></div>
      
      <div
        className={`
          relative bg-slate-800/60 backdrop-blur-2xl border-2 border-dashed rounded-3xl p-10 mb-8 text-center cursor-pointer
          transition-all duration-500 ease-out group-hover:bg-slate-800/80 group-hover:-translate-y-2 group-hover:shadow-2xl group-hover:shadow-teal-500/20
          ${isDragOver 
            ? 'border-teal-400 bg-gradient-to-br from-teal-500/20 via-cyan-500/10 to-blue-500/20 shadow-2xl shadow-teal-500/40 scale-[1.02]' 
            : 'border-slate-600/50 hover:border-slate-500/70'
          }
        `}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          onChange={handleFileChange}
          className="hidden"
        />
        
        {/* Floating particles effect */}
        <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className={`absolute w-1 h-1 bg-teal-400/30 rounded-full animate-pulse transition-all duration-1000 ${
                isHovered ? 'opacity-100' : 'opacity-0'
              }`}
              style={{
                left: `${20 + i * 12}%`,
                top: `${15 + (i % 3) * 25}%`,
                animationDelay: `${i * 0.2}s`,
                transform: isHovered ? 'translateY(-10px)' : 'translateY(0px)'
              }}
            />
          ))}
        </div>
        
        <div className={`
          inline-flex p-6 rounded-full mb-6 transition-all duration-500 relative overflow-hidden
          ${isDragOver 
            ? 'bg-gradient-to-br from-teal-500/30 to-cyan-500/30 text-teal-300 scale-110 shadow-lg shadow-teal-500/50' 
            : 'bg-slate-700/40 text-slate-400 hover:bg-gradient-to-br hover:from-teal-500/20 hover:to-cyan-500/20 hover:text-teal-400 hover:scale-105'
          }
        `}>
          {/* Icon background effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-teal-500/10 to-transparent rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <Upload className={`w-10 h-10 transition-all duration-300 ${isDragOver ? 'animate-bounce' : ''} ${isHovered ? 'rotate-12' : ''}`} />
          <Sparkles className={`absolute -top-1 -right-1 w-4 h-4 text-teal-400 transition-all duration-300 ${isHovered ? 'opacity-100 animate-ping' : 'opacity-0'}`} />
        </div>
        
        <h3 className={`text-2xl font-bold mb-3 transition-all duration-300 ${
          isDragOver ? 'text-teal-300 scale-105' : 'text-slate-200 group-hover:text-teal-100'
        }`}>
          Upload Image For OCR Processing
        </h3>
        <p className="text-slate-400 mb-2 text-lg group-hover:text-slate-300 transition-colors duration-300">
          Drag and drop your image here or click to browse
        </p>
        <p className="text-slate-500 text-sm mb-8 group-hover:text-slate-400 transition-colors duration-300">
          Supported formats: JPG, PNG (Maximum 1 file)
        </p>
        
        <button className={`
          relative bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 
          text-white px-8 py-3 rounded-xl font-semibold text-lg
          transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-teal-500/50
          group-hover:scale-105 overflow-hidden
          ${isDragOver ? 'animate-pulse' : ''}
        `}>
          <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300"></div>
          <span className="relative flex items-center gap-2">
            <Eye className="w-5 h-5" />
            Choose File
          </span>
        </button>
      </div>
    </div>
  );
};

export const ImagePreview = ({ file, onRemove }) => {
  const [imageUrl, setImageUrl] = useState('');
  const [isLoaded, setIsLoaded] = useState(false);

  React.useEffect(() => {
    const url = URL.createObjectURL(file);
    setImageUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  return (
    <div className="relative group">
      {/* Glow effect */}
      <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500/20 via-teal-500/20 to-cyan-500/20 rounded-3xl opacity-75 blur-lg group-hover:opacity-100 transition-opacity duration-300"></div>
      
      <div className="relative bg-slate-800/60 backdrop-blur-2xl border border-slate-700/30 rounded-3xl p-8 mb-8 overflow-hidden">
        {/* Animated background pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0 bg-gradient-to-br from-teal-500 via-transparent to-cyan-500"></div>
        </div>
        
        <div className="relative flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="relative p-3 bg-gradient-to-br from-emerald-500/20 to-teal-500/20 text-emerald-400 rounded-xl border border-emerald-500/20">
              <ImageIcon className="w-6 h-6" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h3 className="text-xl font-bold text-slate-100 mb-1">Uploaded Image</h3>
              <p className="text-slate-400 text-sm truncate max-w-[200px] md:max-w-none font-medium">
                {file.name}
              </p>
              <p className="text-emerald-400 text-xs mt-1">Ready for processing</p>
            </div>
          </div>
          
          <button
            onClick={onRemove}
            className="relative p-3 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-xl transition-all duration-300 group/button"
          >
            <div className="absolute inset-0 bg-red-500/10 rounded-xl scale-0 group-hover/button:scale-100 transition-transform duration-200"></div>
            <X className="relative w-5 h-5 group-hover/button:rotate-90 transition-transform duration-200" />
          </button>
        </div>
        
        <div className="relative group/image">
          <div className="absolute -inset-2 bg-gradient-to-br from-teal-500/10 to-cyan-500/10 rounded-2xl opacity-0 group-hover/image:opacity-100 transition-opacity duration-300"></div>
          {imageUrl && (
            <div className="relative">
              <img
                src={imageUrl}
                alt="Uploaded for OCR processing"
                className={`
                  relative w-full max-h-72 object-contain bg-slate-900/60 rounded-xl border border-slate-700/30 
                  shadow-2xl transition-all duration-500 group-hover/image:shadow-teal-500/20
                  ${isLoaded ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}
                `}
                onLoad={() => setIsLoaded(true)}
              />
              {/* Image overlay effect */}
              <div className="absolute inset-0 bg-gradient-to-t from-slate-900/20 via-transparent to-transparent rounded-xl opacity-0 group-hover/image:opacity-100 transition-opacity duration-300"></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const LanguageSelection = ({ selectedLanguage, onLanguageChange }) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);
  const isAutoDetect = selectedLanguage === 'AutoDetect';
  
  const languages = [
    { value: 'en', label: 'English', flag: '🇺🇸' },
    { value: 'hin', label: 'Hindi', flag: '🇮🇳' },
    { value: 'ru', label: 'Russian', flag: '🇷🇺' },
    { value: 'fr', label: 'French', flag: '🇫🇷' },
    { value: 'es', label: 'Spanish', flag: '🇪🇸' },
    { value: 'ko', label: 'Korean', flag: '🇰🇷' },
    { value: 'de', label: 'German', flag: '🇩🇪' },
    { value: 'it', label: 'Italian', flag: '🇮🇹' },
    { value: 'tr', label: 'Turkish', flag: '🇹🇷' },
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

  const selectedLanguageData = languages.find(lang => lang.value === selectedLanguage);
  const selectedLanguageLabel = selectedLanguageData?.label || 'Select Language';

  return (
    <div className="relative group">
      <div className="absolute -inset-1 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-indigo-500/20 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-lg"></div>
      
      <div className="relative bg-slate-800/60 backdrop-blur-2xl border border-slate-700/30 rounded-3xl p-8 mb-8 z-10">
        <div className="flex items-center gap-4 mb-8">
          <div className="relative p-3 bg-gradient-to-br from-blue-500/20 to-purple-500/20 text-blue-400 rounded-xl border border-blue-500/20">
            <Globe className="w-6 h-6" />
            <div className={`absolute -top-1 -right-1 w-3 h-3 bg-blue-400 rounded-full transition-all duration-300 ${isAutoDetect ? 'animate-pulse' : ''}`}></div>
          </div>
          <div>
            <h3 className="text-xl font-bold text-slate-100 mb-1">Language Settings</h3>
            <p className="text-slate-400 text-sm">Select language to detect or use auto-detection</p>
          </div>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-6">
          {/* Auto Detect Button */}
          <button
            onClick={handleAutoDetectClick}
            className={`
              relative flex items-center justify-center p-4 rounded-xl font-semibold text-sm
              transition-all duration-300 border-2 min-w-[300px] overflow-hidden group/auto
              ${isAutoDetect
                ? 'bg-gradient-to-r from-teal-500/20 to-cyan-500/20 border-teal-400 text-teal-300 shadow-lg shadow-teal-500/20'
                : 'bg-slate-700/30 border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500 hover:shadow-lg'
              }
            `}
          >
            <div className={`absolute inset-0 bg-gradient-to-r from-teal-500/10 to-cyan-500/10 transition-opacity duration-300 ${isAutoDetect ? 'opacity-100' : 'opacity-0'}`}></div>
            <span className="relative flex items-center gap-2">
              <Sparkles className={`w-4 h-4 transition-all duration-300 ${isAutoDetect ? 'animate-spin' : ''}`} />
              Auto Detect
            </span>
            {isAutoDetect && (
              <div className="ml-3 w-2 h-2 bg-teal-400 rounded-full animate-pulse" />
            )}
          </button>

          {/* Language Dropdown Container */}
          <div className="relative flex-1 z-20" ref={dropdownRef}>
            <button
              onClick={handleDropdownToggle}
              className={`
                relative w-full flex items-center justify-between p-4 rounded-xl font-semibold text-sm
                transition-all duration-300 border-2 overflow-hidden
                ${isAutoDetect 
                  ? 'bg-slate-600/30 border-slate-600/50 text-slate-500 cursor-not-allowed' 
                  : !selectedLanguage || selectedLanguage === 'AutoDetect'
                    ? 'bg-slate-700/30 border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500 cursor-pointer hover:shadow-lg'
                    : 'bg-gradient-to-r from-teal-500/10 to-cyan-500/10 border-teal-400/50 text-teal-300 hover:from-teal-500/20 hover:to-cyan-500/20 cursor-pointer shadow-lg'
                }
              `}
              disabled={isAutoDetect}
            >
              <span className={`flex items-center gap-2 ${isAutoDetect ? 'text-slate-500' : ''}`}>
                {selectedLanguageData?.flag && <span className="text-lg">{selectedLanguageData.flag}</span>}
                {isAutoDetect ? 'Auto Detection Enabled' : selectedLanguageLabel}
              </span>
              <ChevronDown 
                className={`w-5 h-5 transition-all duration-300 ${
                  isDropdownOpen ? 'rotate-180' : ''
                } ${isAutoDetect ? 'text-slate-500' : ''}`} 
              />
            </button>

            {/* Dropdown Menu */}
            {isDropdownOpen && !isAutoDetect && (
              <div className="absolute top-full left-0 right-0 mt-3 bg-slate-800/95 backdrop-blur-2xl border border-slate-700/50 rounded-xl shadow-2xl shadow-black/50 max-h-64 overflow-y-auto z-[9999]">
                {languages.map((language, index) => (
                  <button
                    key={language.value}
                    onClick={() => handleLanguageSelect(language.value)}
                    className={`
                      relative w-full flex items-center justify-between px-5 py-4 text-left text-sm
                      transition-all duration-200 overflow-hidden
                      ${index === 0 ? 'rounded-t-xl' : ''} 
                      ${index === languages.length - 1 ? 'rounded-b-xl' : ''}
                      ${selectedLanguage === language.value
                        ? 'bg-gradient-to-r from-teal-500/20 to-cyan-500/20 text-teal-300 border-l-4 border-teal-400'
                        : 'text-slate-300 hover:bg-slate-700/50 hover:text-slate-200'
                      }
                    `}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-teal-500/5 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-200"></div>
                    <span className="relative flex items-center gap-3 font-medium">
                      <span className="text-lg">{language.flag}</span>
                      {language.label}
                    </span>
                    {selectedLanguage === language.value && (
                      <Check className="relative w-4 h-4 text-teal-400" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        
        {/* Enhanced Helper Text */}
        <div className={`mt-6 p-4 rounded-xl transition-all duration-300 ${
          isAutoDetect 
            ? 'bg-teal-500/10 border border-teal-500/20' 
            : selectedLanguage && selectedLanguage !== 'AutoDetect'
              ? 'bg-blue-500/10 border border-blue-500/20'
              : 'bg-amber-500/10 border border-amber-500/20'
        }`}>
          <div className="flex items-center gap-2 text-sm">
            {isAutoDetect ? (
              <>
                <Sparkles className="w-4 h-4 text-teal-400" />
                <span className="text-teal-300">Language will be automatically detected from the image</span>
              </>
            ) : selectedLanguage && selectedLanguage !== 'AutoDetect' ? (
              <>
                <Check className="w-4 h-4 text-blue-400" />
                <span className="text-blue-300">OCR will be optimized for {selectedLanguageLabel} text</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-4 h-4 text-amber-400" />
                <span className="text-amber-300">Please select a specific language or use auto-detect</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export const HandwrittenCheckbox = ({ isHandwritten, onHandwrittenChange }) => {
  return (
    <div className="relative group">
      <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/20 via-pink-500/20 to-indigo-500/20 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-lg"></div>
      
      <div className="relative z-0 bg-slate-800/60 backdrop-blur-2xl border border-slate-700/30 rounded-3xl p-8 mb-8 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/10">
        <div className="flex items-center gap-4">
          <div className="relative p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 text-purple-400 rounded-xl border border-purple-500/20">
            <PenTool className="w-6 h-6" />
            <div className={`absolute -top-1 -right-1 w-3 h-3 bg-purple-400 rounded-full transition-all duration-300 ${isHandwritten ? 'animate-pulse' : 'opacity-50'}`}></div>
          </div>
          <label className="flex items-center gap-4 cursor-pointer flex-1 group/checkbox">
            <div className="relative">
              <input
                type="checkbox"
                checked={isHandwritten}
                onChange={(e) => onHandwrittenChange(e.target.checked)}
                className="w-6 h-6 rounded-lg border-2 border-slate-600 bg-slate-700/50 text-purple-500 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-400 transition-all duration-200 cursor-pointer"
              />
              {isHandwritten && (
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg pointer-events-none"></div>
              )}
            </div>
            <div className="transition-all duration-200 group-hover/checkbox:translate-x-1">
              <span className="text-xl font-bold text-slate-100 block mb-1">Handwritten Text</span>
              <p className="text-slate-400 text-sm">Enable enhanced recognition for handwritten content</p>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
};

export const SpellCheck = ({ isSpellChecked, onSpellCheckChange }) => {
  return (
    <div className="relative group">
      <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/20 via-blue-500/20 to-cyan-500/20 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-lg"></div>
      
      <div className="relative z-0 bg-slate-800/60 backdrop-blur-2xl border border-slate-700/30 rounded-3xl p-8 mb-8 transition-all duration-300 hover:shadow-lg hover:shadow-indigo-500/10">
        <div className="flex items-center gap-4">
          <div className="relative p-3 bg-gradient-to-br from-indigo-500/20 to-blue-500/20 text-indigo-400 rounded-xl border border-indigo-500/20">
            <LetterText className="w-6 h-6" />
            <div className={`absolute -top-1 -right-1 w-3 h-3 bg-indigo-400 rounded-full transition-all duration-300 ${isSpellChecked ? 'animate-pulse' : 'opacity-50'}`}></div>
          </div>
          <label className="flex items-center gap-4 cursor-pointer flex-1 group/checkbox">
            <div className="relative">
              <input
                type="checkbox"
                checked={isSpellChecked}
                onChange={(e) => onSpellCheckChange(e.target.checked)}
                className="w-6 h-6 rounded-lg border-2 border-slate-600 bg-slate-700/50 text-indigo-500 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-400 transition-all duration-200 cursor-pointer"
              />
              {isSpellChecked && (
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/20 to-blue-500/20 rounded-lg pointer-events-none"></div>
              )}
            </div>
            <div className="transition-all duration-200 group-hover/checkbox:translate-x-1">
              <span className="text-xl font-bold text-slate-100 block mb-1">Spell Check</span>
              <p className="text-slate-400 text-sm">Enable enhanced recognition for spelling errors</p>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
};

export const ProcessButton = ({ canProcess, isProcessing, onProcess }) => {
  return (
    <div className="z-0 flex flex-col items-center gap-6 mb-20">
      <div className="relative group">
        {/* Animated ring effect */}
        <div className={`absolute -inset-4 rounded-full transition-all duration-1000 ${
          canProcess && !isProcessing 
            ? 'bg-gradient-to-r from-teal-500/30 via-cyan-500/30 to-blue-500/30 animate-pulse blur-lg' 
            : ''
        }`}></div>
        
        <button
          onClick={onProcess}
          disabled={!canProcess}
          className={`
            relative flex items-center gap-4 px-12 py-5 rounded-2xl font-bold text-xl
            transition-all duration-500 overflow-hidden
            ${canProcess && !isProcessing
              ? 'bg-gradient-to-r from-teal-500 via-cyan-500 to-blue-500 text-white shadow-2xl shadow-teal-500/50 hover:-translate-y-2 hover:shadow-3xl hover:shadow-teal-500/70 hover:scale-105'
              : 'bg-slate-600/50 text-slate-400 cursor-not-allowed border border-slate-600/30'
            }
            ${isProcessing ? 'bg-gradient-to-r from-teal-500 via-cyan-500 to-blue-500 text-white animate-pulse' : ''}
          `}
        >
          {/* Button shine effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
          
          {isProcessing ? (
            <>
              <Loader2 className="w-7 h-7 animate-spin relative z-10" />
              <span className="relative z-10">Processing Image...</span>
            </>
          ) : (
            <>
              <div className="relative">
                <Zap className="w-7 h-7 relative z-10" />
                {canProcess && (
                  <div className="absolute inset-0 bg-white/30 rounded-full animate-ping"></div>
                )}
              </div>
              <span className="relative z-10">Process Image</span>
            </>
          )}
        </button>
      </div>
      
      {!canProcess && !isProcessing && (
        <div className="flex items-center gap-3 text-amber-400 text-sm bg-amber-500/10 px-6 py-3 rounded-xl border border-amber-500/20 backdrop-blur-xl">
          <AlertCircle className="w-5 h-5 animate-pulse" />
          <span className="font-medium">Please upload an image and select a language</span>
        </div>
      )}
    </div>
  );
};