import React, { useState, useEffect } from 'react';
import { FileText, Sparkles, Zap, Globe, Eye } from 'lucide-react';

export const Header = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div className="relative text-center mb-16 overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-teal-400/20 rounded-full animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${i * 0.1}s`,
              animationDuration: `${2 + Math.random() * 2}s`
            }}
          />
        ))}
      </div>

      {/* Main header content */}
      <div className={`relative transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
        <div className="flex items-center justify-center gap-4 mb-6">
          {/* Enhanced logo with multiple layers */}
          <div className="relative group">
            {/* Outer glow */}
            <div className="absolute -inset-3 bg-gradient-to-r from-teal-500/30 via-cyan-500/30 to-blue-500/30 rounded-3xl blur-lg group-hover:blur-xl transition-all duration-500"></div>
            
            {/* Icon container */}
            <div className="relative p-4 bg-gradient-to-br from-teal-500 via-cyan-500 to-blue-500 rounded-3xl shadow-2xl shadow-teal-500/50 group-hover:shadow-teal-500/70 transition-all duration-500 group-hover:scale-110">
              {/* Inner shine effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-white/30 to-transparent rounded-3xl"></div>
              
              {/* Icon */}
              <FileText className="relative w-10 h-10 text-white transform group-hover:rotate-12 transition-transform duration-500" />
              
              {/* Floating accent icons */}
              <Sparkles className="absolute -top-1 -right-1 w-4 h-4 text-white/80 animate-pulse" />
              <Zap className="absolute -bottom-1 -left-1 w-3 h-3 text-white/60 animate-bounce" />
            </div>
          </div>
          
          {/* Title with enhanced styling */}
          <h1 className="text-6xl md:text-7xl font-black bg-gradient-to-r from-teal-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent relative">
            PolyOCR
            {/* Text shadow effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-teal-400/20 via-cyan-400/20 to-blue-400/20 bg-clip-text text-transparent blur-sm -z-10">
              PolyOCR
            </div>
          </h1>
        </div>

        {/* Enhanced subtitle */}
        <div className="relative">
          <p className="text-slate-400 text-xl max-w-3xl mx-auto leading-relaxed mb-4 font-medium">
            Advanced optical character recognition with multi-language support and AI-powered accuracy.
          </p>
          
          {/* Feature highlights */}
          <div className="flex flex-wrap items-center justify-center gap-6 text-sm">
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 backdrop-blur-xl rounded-full border border-slate-700/30">
              <Globe className="w-4 h-4 text-teal-400" />
              <span className="text-slate-300 font-medium">Multi-Language</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 backdrop-blur-xl rounded-full border border-slate-700/30">
              <Eye className="w-4 h-4 text-cyan-400" />
              <span className="text-slate-300 font-medium">High Accuracy</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 backdrop-blur-xl rounded-full border border-slate-700/30">
              <Zap className="w-4 h-4 text-blue-400" />
              <span className="text-slate-300 font-medium">Fast Processing</span>
            </div>
          </div>
        </div>
      </div>

      {/* Decorative elements */}
      <div className="absolute left-1/2 transform -translate-x-1/2 -bottom-8 w-32 h-1 bg-gradient-to-r from-transparent via-teal-500/50 to-transparent rounded-full"></div>
    </div>
  );
};

export const Footer = () => {
  return (
    <div className="relative">
      {/* Decorative top border */}
      <div className="w-full h-px bg-gradient-to-r from-transparent via-slate-700/50 to-transparent mb-8"></div>
      
      <div className="text-center pt-8 relative">
        {/* Background decoration */}
        <div className="absolute inset-0 bg-black "></div>
        
        <div className="relative">
          {/* Main footer text */}
          <p className="text-slate-400 text-sm mb-4 font-medium">
            Powered by advanced OCR technology • Supports multiple languages
          </p>
          
          {/* Enhanced feature list */}
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-slate-500 mb-6">
            <span className="flex items-center gap-1">
              <div className="w-1 h-1 bg-teal-400 rounded-full"></div>
              Handwriting Recognition
            </span>
            <span className="flex items-center gap-1">
              <div className="w-1 h-1 bg-cyan-400 rounded-full"></div>
              Spell Check Integration
            </span>
            <span className="flex items-center gap-1">
              <div className="w-1 h-1 bg-blue-400 rounded-full"></div>
              Auto Language Detection
            </span>
          </div>
          
          {/* Copyright and branding */}
          <div className="text-xs text-slate-600">
            <p className="mb-2">© 2024 PolyOCR. Advanced text recognition technology.</p>
            <p className="flex items-center justify-center gap-2">
              <span>Made with</span>
              <div className="w-3 h-3 bg-gradient-to-r from-red-500 to-pink-500 rounded-full animate-pulse"></div>
              <span>for seamless text extraction</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};