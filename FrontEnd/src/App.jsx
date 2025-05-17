// import { useState } from 'react'
// import './styles/icondesigns.css'
// import Header from './components/layouts/header'
// import './App.css'
// import ContentMain from './components/designs/contentmain'
// import Footer from './components/layouts/footer'
// import From from './components/designs/from'



// const App = () => {

//   return (
//     <>
// <Header/>
//       <From/>
// <ContentMain/>

// <Footer/>
//     </>
//   )
// }

// export default App                                                


import React, { useState } from 'react';
import axios from 'axios';
import TextInput from './components/TextInput';
import ResultDisplay from './components/ResultDisplay';
import './styles.css';

const App = () => {
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
    <div className="app-container">
      <header className="app-header">
        <h1>Hệ thống phát hiện ngôn ngữ độc hại tiếng Việt</h1>
        <p>
          Phân loại văn bản thành 3 loại: Sạch (Clean), Xúc phạm (Offensive), Căm thù (Hate)
        </p>
      </header>
      
      <main className="app-main">
        <TextInput onSubmit={analyzeText} isLoading={isLoading} />
        
        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <p>Đang phân tích văn bản...</p>
          </div>
        )}
        
        {error && <div className="error-message">{error}</div>}
        
        {result && !isLoading && (
          <ResultDisplay result={result} originalText={originalText} />
        )}
      </main>
      
      <footer className="app-footer">
        <p>Hệ thống sử dụng mô hình PhoBERT được huấn luyện trên tập dữ liệu ViHSD</p>
      </footer>
    </div>
  );
};

export default App;