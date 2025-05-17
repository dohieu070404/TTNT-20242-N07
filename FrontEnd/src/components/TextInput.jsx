import React, { useState } from 'react';
import '../styles.css'

const TextInput = ({ onSubmit, isLoading }) => {
  const [text, setText] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onSubmit(text);
    }
  };

  return (
    <div className="text-input-container">
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Nhập văn bản tiếng Việt cần kiểm tra..."
          rows={6}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          disabled={isLoading || !text.trim()}
          className="analyze-button"
        >
          {isLoading ? 'Đang phân tích...' : 'Phân tích'}
        </button>
      </form>
    </div>
  );
};

export default TextInput;