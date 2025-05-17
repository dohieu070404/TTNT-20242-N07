import React from 'react';
import '../styles.css';

const ResultDisplay = ({ result, originalText }) => {
  if (!result) return null;

  // Xác định màu sắc dựa trên nhãn
  const getLabelColor = () => {
    switch (result.label) {
      case 'HATE':
        return { backgroundColor: '#ffebee', color: '#c62828', borderColor: '#ef9a9a' };
      case 'OFFENSIVE':
        return { backgroundColor: '#fff8e1', color: '#ff8f00', borderColor: '#ffe082' };
      case 'CLEAN':
        return { backgroundColor: '#e8f5e9', color: '#2e7d32', borderColor: '#a5d6a7' };
      default:
        return {};
    }
  };

  // Highlight các từ độc hại
  const highlightToxicWords = (text, spans) => {
    if (!spans || spans.length === 0) return text;
    
    const words = text.split(/(\s+)/);
    return words.map((word, index) => {
      const cleanWord = word.trim().toLowerCase();
      const isToxic = spans.some(span => 
        cleanWord.includes(span.toLowerCase()) || 
        span.toLowerCase().includes(cleanWord)
      );
      
      return isToxic ? (
        <mark key={index} className="toxic-word">
          {word}
        </mark>
      ) : (
        <React.Fragment key={index}>{word}</React.Fragment>
      );
    });
  };

  return (
    <div className="result-container">
      <div className="result-header" style={getLabelColor()}>
        <h3>Kết quả phân tích:</h3>
        <div className="result-label">
          {result.label === 'HATE' && 'NỘI DUNG CĂM THÙ'}
          {result.label === 'OFFENSIVE' && 'NỘI DUNG XÚC PHẠM'}
          {result.label === 'CLEAN' && 'NỘI DUNG SẠCH'}
          <span className="confidence-score">
            ({Math.round(result.score * 100)}% độ tin cậy)
          </span>
        </div>
      </div>
      
      <div className="text-analysis">
        <h4>Văn bản đã phân tích:</h4>
        <div className="analyzed-text">
          {highlightToxicWords(originalText, result.spans)}
        </div>
      </div>
      
      {result.spans && result.spans.length > 0 && (
        <div className="toxic-words-section">
          <h4>Từ ngữ độc hại phát hiện:</h4>
          <div className="toxic-words-list">
            {result.spans.map((word, index) => (
              <span key={index} className="toxic-word-tag">
                {word}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;