import React from 'react';
import { FileText } from 'lucide-react';

export const Header = () => {
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
}

export const Footer = () => {
  return (
    <div className="text-center pt-8 border-t border-slate-700/30">
      <p className="text-slate-500 text-sm">
        Powered by advanced OCR technology • Supports multiple languages
      </p>
    </div>
  );
}
