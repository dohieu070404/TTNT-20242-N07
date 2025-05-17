import React, { useState } from 'react';
import axios from 'axios';
import TextInput from './components/TextInput';
import ResultDisplay from './components/ResultDisplay';
import styles from './TiengViet.module.css';

const TiengViet = () => {
  const [result, setResult] = useState(null);
  const [originalText, setOriginalText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeText = async (text) => {
    setIsLoading(true);
    setError(null);
    setOriginalText(text);
    
    try {
      const response = await axios.post('http://localhost:5000/predict', { text });
      setResult(response.data);
    } catch (err) {
      setError('Có lỗi xảy ra khi phân tích văn bản. Vui lòng thử lại.');
      console.error('API error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <TextInput onSubmit={analyzeText} isLoading={isLoading} />
        
        {isLoading && (
          <div className={styles.loadingIndicator}>
            <div className={styles.spinner}></div>
            <p>Đang phân tích văn bản...</p>
          </div>
        )}
        
        {error && <div className={styles.errorMessage}>{error}</div>}
        
        {result && !isLoading && (
          <ResultDisplay result={result} originalText={originalText} />
        )}
      </main>
    </div>
  );
};

export default TiengViet;