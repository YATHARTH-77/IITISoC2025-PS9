import React, { useState, useRef, useEffect, useMemo } from 'react';
import { FileText, ArrowLeft, Target, Clock, Globe, CheckCircle, Copy, Download, Volume2, Loader2, Play, BookOpenText} from 'lucide-react';

// Results View Component
export const ResultsView = ({ fileName, onNewImage, ocrData, backendURL, selectedFile }) => {
  const [copied, setCopied] = useState(false);
  const [copiedTranslated, setCopiedTranslated] = useState(false);
  const [scale, setScale] = useState({ x: 1, y: 1 });
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [imageError, setImageError] = useState(false);
  // const [clickedBox, setClickedBox] = useState(false);
  const [languages, setLanguages] = useState([]);
  const [selectedBox, setSelectedBox] = useState(null);
  const [hoveredBox, setHoveredBox] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0, text: '' });
  const [translatedText, setTranslatedText] = useState("");
  const [isProcessingTranslate, setIsProcessingTranslate] = useState(false);
  const [haveAudio, setHaveAudio] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  const [isAllTextTranslated, setIsAllTextTranslated] = useState(false);
  const [isAllTextTranslating, setIsAllTextTranslating] = useState(false);
  const [allTranslatedText, setAllTranslatedText] = useState("");
  const [allTextAudioUrl, setAllTextAudioUrl] = useState(null);
  const [isAllTextAudioProcessing, setIsAllTextAudioProcessing] = useState(false);
  const [haveAllTextAudio, setHaveAllTextAudio] = useState(false);
  const [isProcessingTranslatedAudio, setIsProcessingTranslatedAudio] = useState(false);
  const [haveTranslatedAudio, setHaveTranslatedAudio] = useState(false);
  const [translatedAudioUrl, setTranslatedAudioUrl] = useState(null);
  const [summarizedAllExtractedText,setSummarizedAllExtractedText] = useState("");
  const [imageUrl,setImageUrl] = useState('');
  // Handle both old and new data structure
  const results = ocrData?.results || [];
  const processingTime = ocrData?.processing_time || 0;
  const totalDetections = ocrData?.total_detections || results.length;
  
  const avgConfidence = results.length > 0
    ? Math.floor(
        (results.reduce((acc, item) => acc + (item.confidence || 0), 0) / results.length) * 1000
      ) / 10
    : 0;

  const imgRef = useRef(null);
  const containerRef = useRef(null);
  
  React.useEffect(() => {
      const url = URL.createObjectURL(selectedFile);
      setImageUrl(url);
  
      return () => {
        URL.revokeObjectURL(url);
      };
    }, [selectedFile]);


//   const imageUrl = useMemo(() => {
//   const timestamp = new Date().getTime();
//   return `${backendURL}/static/preprocess.png?ts=${timestamp}`;
// }, [fileName]);
//   const imageUrl = useMemo(() => {
//   const timestamp = new Date().getTime();
//   const url = `${backendURL}/static/preprocess.png?ts=${timestamp}`;
//   console.log('Image URL generated:', url);
//   return url;
// }, [fileName]);

  const playAudio = (URL) => {
    if (URL) {
      const audio = new Audio(URL);
      audio.play().catch(e => console.error('Audio play failed:', e));
    }
  }

  // Function to get audio for selected text
  const getAudioUrl = async (Boolean, IsTranslatedAudioRequest) => {
    const indexselected = selectedBox;
    if (!IsTranslatedAudioRequest){
    try {
      const formData = new FormData();
      if (!Boolean) {
        setIsProcessingAudio(true);
        formData.append('text', results[selectedBox].detected_text || '');
        formData.append('isAllText', Boolean || false);
        formData.append('isTranslatedAudioRequest', isProcessingTranslatedAudio);
      }
      else{
        setIsAllTextAudioProcessing(true);
        formData.append('isAllText', true);
        formData.append('text', allExtractedText);
        formData.append('isTranslatedAudioRequest', isAllTextAudioProcessing);
      }
      const audioFileName = results[selectedBox].audio_file;
      await fetch(`${backendURL}/audio`, {
        method: 'POST',
        body: formData,
      }
      );
      // if(indexselected != selectedBox) return; // Check if still selected
      if (!Boolean) {
        setAudioUrl(`${backendURL}/static/data.mp3?t=${Date.now()}`);
        setHaveAudio(true);
        setIsProcessingAudio(false); 
      }
      else{
        setAllTextAudioUrl(`${backendURL}/static/alldata.mp3?t=${Date.now()}`);
        setHaveAllTextAudio(true);
        setIsAllTextAudioProcessing(false);
      }
    } catch (error) {
      console.error('Audio error:', error);
    }
  }
  else{
    try{
      setIsProcessingTranslatedAudio(true);
      const formData = new FormData();
      formData.append('isAllText', true);
      formData.append('text', allTranslatedText);
      formData.append('isTranslatedAudioRequest', isAllTextAudioProcessing);
      await fetch(`${backendURL}/audio`, {
        method: 'POST',
        body: formData,
      });
      setTranslatedAudioUrl(`${backendURL}/static/translated_data.mp3?t=${Date.now()}`);
      setHaveTranslatedAudio(true);
      setIsProcessingTranslatedAudio(false);
    } catch (error) {
      console.error('Audio error:', error);
    }
  }
}
  const getTranslate = async (Boolean) => {
    
    try {
      const formData = new FormData();
      if(!Boolean) {
        formData.append('text', results[selectedBox].detected_text || '');
        formData.append('isAllText', Boolean || false);
        setIsProcessingTranslate(true);
      }
      else{
        formData.append('isAllText', true);
        formData.append('text', allExtractedText);
        setIsAllTextTranslating(true);
      }
      if(!Boolean) {
      setTranslatedText((await (await fetch(`${backendURL}/translate`, {
                                    method: 'POST',
                                    body: formData,
                                  })).json()).translated_text);
        setIsProcessingTranslate(false);
        setIsAllTextTranslated(true);
      }
      else{
        setAllTranslatedText((await (await fetch(`${backendURL}/translate`, {
                                    method: 'POST',
                                    body: formData,
                                  })).json()).translated_text);
                                  console.log("All Translated Text:", allTranslatedText);
        setIsAllTextTranslating(false);
        setIsAllTextTranslated(true);
      }
    } catch (error) {
      console.error('Translation error:', error);
    }
  };

  useEffect(() => {
    // Filter unique languages from results
    const uniqueLanguages = results.length > 0
      ? [...new Set(results.map(item => item.language).filter(Boolean))]
      : [];
    setLanguages(uniqueLanguages);
  }, [results]);

  const getConfidenceColor = () => {
    if (avgConfidence >= 90) return 'text-green-400';
    if (avgConfidence >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceLabel = () => {
    if (avgConfidence >= 90) return 'Excellent';
    if (avgConfidence >= 70) return 'Good';
    return 'Fair';
  };

  useEffect(() => {
    const updateMetrics = () => {
      if (imgRef.current && !imageError) {
        const img = imgRef.current;
        const rect = img.getBoundingClientRect();
        setScale({
          x: img.clientWidth / img.naturalWidth,
          y: img.clientHeight / img.naturalHeight,
        });
        setOffset({
          x: rect.left + window.scrollX,
          y: rect.top + window.scrollY,
        });
      }
    };

    updateMetrics();
    window.addEventListener('resize', updateMetrics);
    return () => window.removeEventListener('resize', updateMetrics);
  }, [imageError]);

  // Extract all text from results
  const allExtractedText = results.length > 0
    ? results.map(item => item.detected_text || '').filter(text => text.trim()).join(' ')
    : '';

  // Function to create SVG path from 4 points with safe coordinate handling
  const createPolygonPath = (box, scale) => {
    if (!box || !Array.isArray(box)) return '';
    
    try {
      // Handle different box formats
      let points = [];
      
      if (box.length === 4 && Array.isArray(box[0])) {
        // Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        points = box;
      } else if (box.length === 4 && typeof box[0] === 'number') {
        // Format: [x1, y1, x2, y2] - convert to point pairs
        points = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]];
      } else if (box.length === 8) {
        // Format: [x1, y1, x2, y2, x3, y3, x4, y4] - convert to point pairs
        points = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]];
      } else {
        console.warn('Unsupported box format:', box);
        return '';
      }
      
      // Validate and scale points
      const scaledPoints = points.map(([x, y]) => {
        const scaledX = Number(x) * scale.x;
        const scaledY = Number(y) * scale.y;
        
        // Check for valid numbers
        // if (isNaN(scaledX) || isNaN(scaledY)) {
        //   console.warn('Invalid coordinates:', x, y);    
        //   return [0, 0];
        // }
        
        return [scaledX, scaledY];
      });
      
      // Create SVG path
      return scaledPoints.map(([x, y], index) => 
        `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
      ).join(' ') + ' Z';
      
    } catch (error) {
      console.error('Error creating polygon path:', error, box);
      return '';
    }
  };

  // Function to check if a point is inside a polygon with safe coordinate handling
  const isPointInPolygon = (point, polygon) => {
    if (!polygon || !Array.isArray(polygon)) return false;
    
    try {
      // Handle different polygon formats
      let points = [];
      
      if (polygon.length === 4 && Array.isArray(polygon[0])) {
        // Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        points = polygon;
      } else if (polygon.length === 4 && typeof polygon[0] === 'number') {
        // Format: [x1, y1, x2, y2] - convert to point pairs
        points = [[polygon[0], polygon[1]], [polygon[2], polygon[1]], [polygon[2], polygon[3]], [polygon[0], polygon[3]]];
      } else if (polygon.length === 8) {
        // Format: [x1, y1, x2, y2, x3, y3, x4, y4] - convert to point pairs
        points = [[polygon[0], polygon[1]], [polygon[2], polygon[3]], [polygon[4], polygon[5]], [polygon[6], polygon[7]]];
      } else {
        return false;
      }
      
      const [x, y] = point;
      let inside = false;
      
      for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
        const [xi, yi] = points[i];
        const [xj, yj] = points[j];
        
        // Validate coordinates
        if (isNaN(xi) || isNaN(yi) || isNaN(xj) || isNaN(yj)) {
          continue;
        }
        
        if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
          inside = !inside;
        }
      }
      
      return inside;
    } catch (error) {
      console.error('Error checking point in polygon:', error);
      return false;
    }
  };

  // Calculate smart tooltip position that stays within viewport
  const calculateTooltipPosition = (box, scale, mouseEvent) => {
    if (!box || !Array.isArray(box) || !imgRef.current || !containerRef.current) {
      return { left: 0, top: 0 };
    }
    
    try {
      // Handle different box formats to get center point
      let points = [];
      
      if (box.length === 4 && Array.isArray(box[0])) {
        points = box;
      } else if (box.length === 4 && typeof box[0] === 'number') {
        points = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]];
      } else if (box.length === 8) {
        points = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]];
      } else {
        return { left: 0, top: 0 };
      }
      
      const scaledPoints = points.map(([x, y]) => [
        Number(x) * scale.x || 0, 
        Number(y) * scale.y || 0
      ]);
      
      const validPoints = scaledPoints.filter(([x, y]) => !isNaN(x) && !isNaN(y));
      
      if (validPoints.length === 0) {
        return { left: 0, top: 0 };
      }
      
      // Get bounding box of the text region
      const minX = Math.min(...validPoints.map(([x]) => x));
      const maxX = Math.max(...validPoints.map(([x]) => x));
      const minY = Math.min(...validPoints.map(([, y]) => y));
      
      // Get image and container positions
      const imgRect = imgRef.current.getBoundingClientRect();
      const containerRect = containerRef.current.getBoundingClientRect();
      
      // Calculate center of the box relative to the container
      const boxCenterX = minX + (maxX - minX) / 2;
      const boxCenterY = minY;
      
      // Convert to container-relative coordinates
      const containerX = imgRect.left - containerRect.left + boxCenterX;
      const containerY = imgRect.top - containerRect.top + boxCenterY;
      
      // Tooltip dimensions (estimated)
      const tooltipWidth = 200; // Adjust based on your needs
      const tooltipHeight = 32;
      const padding = 8;
      
      // Calculate position, keeping tooltip within container bounds
      let left = containerX - tooltipWidth / 2;
      let top = containerY - tooltipHeight - padding;
      
      // Adjust horizontal position if tooltip would go outside container
      if (left < padding) {
        left = padding;
      } else if (left + tooltipWidth > containerRect.width - padding) {
        left = containerRect.width - tooltipWidth - padding;
      }
      
      // Adjust vertical position if tooltip would go above container
      if (top < padding) {
        top = containerY + padding; // Show below the box instead
      }
      
      return { left, top };
    } catch (error) {
      console.error('Error calculating tooltip position:', error);
      return { left: 0, top: 0 };
    }
  };

  // Handle SVG click
  const handleSvgClick = (e) => {
    // Check which box was clicked
    for (let i = 0; i < results.length; i++) {
      const box = results[i].box;
    if (!imgRef.current) return;
    setIsProcessingAudio(false);
    setHaveAudio(false);
    setIsProcessingTranslate(false);
    setTranslatedText("");
    setHaveAudio(false);
    setAudioUrl(null);
    
    const rect = imgRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale.x;
    const y = (e.clientY - rect.top) / scale.y;
    
      if (box && isPointInPolygon([x, y], box)) {
        setSelectedBox(i);
        setHaveAudio(false);
        setAudioUrl(null);
        return;
      }
    }
    
    // If no box was clicked, deselect
    setSelectedBox(null);
  };

  // Handle SVG mouse move for hover
  const handleSvgMouseMove = (e) => {
    if (!imgRef.current) return;
    
    const rect = imgRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale.x;
    const y = (e.clientY - rect.top) / scale.y;
    
    // Check which box is being hovered
    for (let i = 0; i < results.length; i++) {
      const box = results[i].box;
      if (box && isPointInPolygon([x, y], box)) {
        setHoveredBox(i);
        const tooltipPos = calculateTooltipPosition(box, scale, e);
        setTooltipPosition({
          x: tooltipPos.left,
          y: tooltipPos.top,
          text: results[i].detected_text || ''
        });
        return;
      }
    }
    
    // If no box is hovered, clear hover
    setHoveredBox(null);
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
            <h2 className="text-2xl font-bold text-slate-200">
              OCR Results
            </h2>
            <p className="text-slate-400">Extracted from: {fileName}</p>
            {processingTime > 0 && (
              <p className="text-slate-500 text-sm">Processing time: {processingTime}s</p>
            )}
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

      {/* Image display container with tooltip */}
      <div 
        ref={containerRef}
        className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-3xl p-6 flex justify-center relative overflow-visible"
      >
        <div className="relative inline-block">
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Processed OCR result"
            className="max-h-72 object-contain rounded-xl border border-slate-700/30 shadow-lg bg-slate-900/50"
            onError={(e) => {
              setImageError(true);
              e.target.style.display = 'none';
            }}
            onLoad={() => {
              if (imgRef.current) {
                setScale({
                  x: imgRef.current.clientWidth / imgRef.current.naturalWidth,
                  y: imgRef.current.clientHeight / imgRef.current.naturalHeight,
                });
              }
            }}
          />
          
          {!imageError && imgRef.current && results.length > 0 && (
            <svg
              className="absolute top-0 left-0 pointer-events-auto cursor-pointer"
              width={imgRef.current.clientWidth}
              height={imgRef.current.clientHeight}
              onClick={handleSvgClick}
              onMouseMove={handleSvgMouseMove}
              onMouseLeave={() => {
                setHoveredBox(null);
                setTooltipPosition({ x: 0, y: 0, text: '' });
              }}
              style={{ pointerEvents: 'auto' }}
            >
              {results.map((item, index) => {
                const isSelected = selectedBox === index;
                const isHovered = hoveredBox === index;
                
                // Skip if no valid box data
                if (!item.box || !Array.isArray(item.box) || item.box.length === 0) {
                  console.warn(`Invalid box data for item ${index}:`, item.box);
                  return null;
                }
                
                const pathData = createPolygonPath(item.box, scale);
                if (!pathData) {
                  console.warn(`Could not create path for item ${index}`);
                  return null;
                }
                
                return (
                  <path
                    key={index}
                    d={pathData}
                    fill={isSelected ? "rgba(34, 197, 94, 0.1)" : (isHovered ? "rgba(6, 182, 212, 0.1)" : "transparent")}
                    stroke={isSelected ? "#22c55e" : "#06b6d4"}
                    strokeWidth={isSelected ? "2" : "1"}
                    className="transition-all duration-200"
                  />
                );
              })}
            </svg>
          )}
        </div>
        
        {/* Floating tooltip outside of SVG */}
        {hoveredBox !== null && tooltipPosition.text && (
          <div
            className="absolute z-50 px-3 py-2 bg-slate-900/95 text-white text-sm rounded-lg shadow-lg border border-slate-600/50 backdrop-blur-sm pointer-events-none"
            style={{
              left: `${tooltipPosition.x}px`,
              top: `${tooltipPosition.y}px`,
              maxWidth: '200px',
              wordWrap: 'break-word',
              whiteSpace: 'pre-wrap'
            }}
          >
            {tooltipPosition.text}
            {/* Small arrow pointing to the box */}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-900/95"></div>
          </div>
        )}
      </div>

      {/* Selected box details */}
      {selectedBox !== null && results[selectedBox] && (
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-4 my-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-green-500/20 text-green-400 rounded-lg">
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Detected Text</p>
              <p className="text-lg font-semibold text-slate-200">{results[selectedBox].detected_text || 'N/A'}</p>
            </div>
          </div> 
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-purple-500/20 text-purple-400 rounded-lg">
              <Globe className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Language</p>
              <p className="text-lg font-semibold text-slate-200">{results[selectedBox].language || 'Unknown'}</p>
            </div>
          </div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Confidence</p>
              <p className="text-lg font-semibold text-slate-200">
                {Math.round((results[selectedBox].confidence || 0) * 100)}%
              </p>
            </div>
          </div>
          {translatedText && !isProcessingTranslate && (
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-cyan-500/20 text-cyan-400 rounded-lg">
                <Clock className="w-5 h-5" />
              </div>
           
                <div>
                  <p className="text-slate-400 text-sm">Translated Text</p>
                  <p className="text-lg font-semibold text-slate-200">{translatedText || "N/A"}</p>
                </div>
            
            </div>
          )}
          <div className="flex items-center gap-3 mb-2">
            <button
              className="mt-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg"
              onClick={() => {
                setSelectedBox(null);
                setHaveAudio(false);
                setAudioUrl(null);
              }}
            >
              Close
            </button>
            {(!isProcessingAudio && !haveAudio) && (
                <button
                  className="mt-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg flex items-center gap-2"
                  onClick={() => getAudioUrl(false, false)}
                >
                  <Volume2 className="w-4 h-4" />
                  Play Audio
                </button>
              )}
            {(isProcessingAudio && !haveAudio) && (
                <button
                  className="mt-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg flex items-center gap-2"
                  disabled
                >
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading...
                </button>
              )}
            {(!isProcessingAudio && haveAudio) && (
              <>
                <audio controls hidden>
                  <source src={audioUrl} type="audio/mpeg"/>
                </audio>
                <button
                  className="mt-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg flex items-center gap-2"
                  onClick={() => playAudio(audioUrl)}
                >
                  <Play className="w-4 h-4" />
                  Play
                </button>
              </>
            )}
            {(!isProcessingTranslate && !translatedText) && (
                <button
                  className="mt-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg flex items-center gap-2"
                  onClick={() => getTranslate(false)}
                >
                  <Globe className="w-4 h-4" />
                  Translate
                </button>
              )}
            {(isProcessingTranslate && !translatedText) && (
                <button
                  className="mt-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg flex items-center gap-2"
                  disabled
                >
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Translating...
                </button>
              )}
            
          </div>
        </div>
      )}

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Confidence</p>
              <p className={`text-lg font-semibold ${getConfidenceColor()}`}>
                {avgConfidence}% ({getConfidenceLabel()})
              </p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-500/20 text-green-400 rounded-lg">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Total Detections</p>
              <p className="text-lg font-semibold text-slate-200">{totalDetections}</p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/20 text-purple-400 rounded-lg">
              <Globe className="w-5 h-5" />
            </div>
            <div>
              <p className="text-slate-400 text-sm">Languages</p>
              <p className="text-lg font-semibold text-slate-200">{languages.join(', ') || 'Unknown'}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Extracted Text */}
      <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-200">
            Extracted Text
          </h3>
          <div className="flex gap-2">
            {(!isAllTextAudioProcessing && !haveAllTextAudio) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => getAudioUrl(true, false)}
                >
                  <Volume2 className="w-4 h-4" />
                </button>
              )}
            {(isAllTextAudioProcessing && !haveAllTextAudio) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  disabled
                >
                  <Loader2 className="w-4 h-4 animate-spin" />
                </button>
              )}
            {(!isAllTextAudioProcessing && haveAllTextAudio) && (
              <>
                <audio controls hidden>
                  <source src={allTextAudioUrl} type="audio/mpeg"/>
                </audio>
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => {
                    if (allTextAudioUrl) {
                      const audio = new Audio(allTextAudioUrl);
                      audio.play().catch(e => console.error('Audio play failed:', e));
                    }
                  }}
                >
                  <Play className="w-4 h-4" />
                </button>
              </>
            )}

            {(!isAllTextTranslating && !isAllTextTranslated) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => getTranslate(true)}
                >
                  <Globe className="w-4 h-4" />
                </button>
              )}
            {(isAllTextTranslating && !isAllTextTranslated) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  disabled
                >
                  <Loader2 className="w-4 h-4 animate-spin" />
                </button>
              )}
            <button
              onClick={async () => {
                        try {
                          await navigator.clipboard.writeText(allExtractedText);
                          setCopied(true);
                          setTimeout(() => setCopied(false), 2000);
                        } catch (err) {
                          console.error('Failed to copy text:', err);
                        }
                      }}
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
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                </>
              )}
            </button>
            
            <button
              onClick={() => {
                        const element = document.createElement('a');
                        const file = new Blob([allExtractedText], { type: 'text/plain' });
                        element.href = URL.createObjectURL(file);
                        element.download = `ocr-result-${fileName.split('.')[0]}.txt`;
                        document.body.appendChild(element);
                        element.click();
                        document.body.removeChild(element);
                      }}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
          <pre className="text-slate-200 whitespace-pre-wrap leading-relaxed text-sm font-mono">
            {allExtractedText || 'No text detected'}
          </pre>
        </div>
        <div className='flex items-center justify-between'>
          <div className="mt-4 flex items-center justify-between gap-2 text-slate-400 text-xs">
            <FileText className="w-4 h-4" />
            <span>{allExtractedText.length} characters extracted</span>
          </div>
          {(!summarizedAllExtractedText) &&
          <button 
            className={`mt-4 flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200
                    ${!allExtractedText?"hidden":''}
                      `}
            onClick={async () => {
              try {
                    const formData = new FormData();
                    formData.append('text', allExtractedText);

                    const response = await fetch(`${backendURL}/summarize`, {
                      method: 'POST',
                      body: formData
                    });
                    
                    const summarizedTextjson = await (response.json());
                    console.log("Summarized Text JSON:", summarizedTextjson);
                    const summarizedText = summarizedTextjson.summary;
                    console.log("Summarized Text:", summarizedText);
                    setSummarizedAllExtractedText(summarizedText);
                  } catch (error) {
                    console.log(error);
                  }
            }}
          >
            <BookOpenText className="w-4 h-4" />
            <span>Summarize</span>
          </button>}
        </div>
      </div>
      {/* Translated Text */}
      {(isAllTextTranslated ) &&
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-200">
            Translated Text
          </h3>
          <div className="flex gap-2">
            {(!isProcessingTranslatedAudio && !haveTranslatedAudio) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => getAudioUrl(true, true)}
                >
                  <Volume2 className="w-4 h-4" />
                </button>
              )}
            {(isProcessingTranslatedAudio && !haveTranslatedAudio) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  disabled
                >
                  <Loader2 className="w-4 h-4 animate-spin" />
                </button>
              )}
            {(!isProcessingTranslatedAudio && haveTranslatedAudio) && (
              <>
                <audio controls hidden>
                  <source src={translatedAudioUrl} type="audio/mpeg"/>
                </audio>
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => playAudio(translatedAudioUrl)}
                >
                  <Play className="w-4 h-4" />
                </button>
              </>
            )}
            <button
              onClick={async () => {
                        try {
                          await navigator.clipboard.writeText(allTranslatedText);
                          setCopiedTranslated(true);
                          setTimeout(() => setCopiedTranslated(false), 2000);
                        } catch (err) {
                          console.error('Failed to copy text:', err);
                        }
                      }}
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
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                </>
              )}
            </button>
            
            <button
              onClick={() => {
                        const element = document.createElement('a');
                        const file = new Blob([allTranslatedText], { type: 'text/plain' });
                        element.href = URL.createObjectURL(file);
                        element.download = `ocr-result-${fileName.split('.')[0]}.txt`;
                        document.body.appendChild(element);
                        element.click();
                        document.body.removeChild(element);
                      }}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
          <pre className="text-slate-200 whitespace-pre-wrap leading-relaxed text-sm font-mono">
            {allTranslatedText}
          </pre>
        </div>
        
        </div>
      }
      {/* Summarized Text */}
      {summarizedAllExtractedText && (
        
      <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/30 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-200">
            Summarized Text
          </h3>
          <div className="flex gap-2">
            {(!isAllTextAudioProcessing && !haveAllTextAudio) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => getAudioUrl(true, false)}
                >
                  <Volume2 className="w-4 h-4" />
                </button>
              )}
            {(isAllTextAudioProcessing && !haveAllTextAudio) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  disabled
                >
                  <Loader2 className="w-4 h-4 animate-spin" />
                </button>
              )}
            {(!isAllTextAudioProcessing && haveAllTextAudio) && (
              <>
                <audio controls hidden>
                  <source src={allTextAudioUrl} type="audio/mpeg"/>
                </audio>
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => {
                    if (allTextAudioUrl) {
                      const audio = new Audio(allTextAudioUrl);
                      audio.play().catch(e => console.error('Audio play failed:', e));
                    }
                  }}
                >
                  <Play className="w-4 h-4" />
                </button>
              </>
            )}

            {(!isAllTextTranslating && !isAllTextTranslated) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  onClick={() => getTranslate(true)}
                >
                  <Globe className="w-4 h-4" />
                </button>
              )}
            {(isAllTextTranslating && !isAllTextTranslated) && (
                <button
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
                  disabled
                >
                  <Loader2 className="w-4 h-4 animate-spin" />
                </button>
              )}
            <button
              onClick={async () => {
                        try {
                          await navigator.clipboard.writeText(allExtractedText);
                          setCopied(true);
                          setTimeout(() => setCopied(false), 2000);
                        } catch (err) {
                          console.error('Failed to copy text:', err);
                        }
                      }}
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
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                </>
              )}
            </button>
            
            <button
              onClick={() => {
                        const element = document.createElement('a');
                        const file = new Blob([allExtractedText], { type: 'text/plain' });
                        element.href = URL.createObjectURL(file);
                        element.download = `ocr-result-${fileName.split('.')[0]}.txt`;
                        document.body.appendChild(element);
                        element.click();
                        document.body.removeChild(element);
                      }}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm font-medium transition-colors duration-200"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
          <pre className="text-slate-200 whitespace-pre-wrap leading-relaxed text-sm font-mono">
            {summarizedAllExtractedText || 'No text detected'}
          </pre>
        </div>
      </div>
      )}
    </div>
  );
};