import React, { useState } from "react";
import axios from "axios";
import './from.css';

/**
 * Displays the toxicity analysis results in a formatted list
 */
const ToxicityAnalysisResults = ({ toxicityScores }) => {
  if (!toxicityScores || toxicityScores.length === 0) {
    return null;
  }

  return (
    <div className="results-container">
      <h3 className="results-title">Analysis Results</h3>
      {toxicityScores.map((toxicityScore, index) => (
        <div key={index} className="result-item" style={{
          '--score-bg': toxicityScore.score > 0.5 ? '#fff5f5' : '#f0fff4',
          '--score-color': toxicityScore.score > 0.5 ? '#dc3545' : '#28a745'
        }}>
          <span className="label">{toxicityScore.label}</span>
          <span className="score">
            {(toxicityScore.score * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
};

function From() {
  const [userInput, setUserInput] = useState("");
  const [toxicityScores, setToxicityScores] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyzeText = async () => {
    if (!userInput.trim()) {
      setErrorMessage("Please enter some text.");
      return;
    }

    setErrorMessage("");
    setToxicityScores([]);
    setIsAnalyzing(true);

    try {
      const response = await axios.post("http://127.0.0.1:5000/analyze", { text: userInput });
      console.log('API Response:', response.data);

      if (response.data?.data?.analysis) {
        const analysis = response.data.data.analysis;
        const formattedResults = [{
          label: analysis.label,
          score: analysis.score
        }];
        setToxicityScores(formattedResults);
      } else {
        setErrorMessage("Invalid response format from server");
      }
    } catch (err) {
      console.error('Analysis Error:', err);
      setErrorMessage(err.response?.data?.message || "Failed to analyze the text. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Toxic Comment Detector</h1>
      <textarea
        className="textarea"
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        placeholder="Enter text to analyze..."
        aria-label="Text to analyze"
      />
      <button
        className="button"
        onClick={handleAnalyzeText}
        disabled={isAnalyzing || !userInput.trim()}
        aria-busy={isAnalyzing}
      >
        {isAnalyzing ? "Waiting..." : "Submit"}
      </button>
      
      {errorMessage && <p className="error-message" role="alert">{errorMessage}</p>}

      <div aria-live="polite">
        <ToxicityAnalysisResults toxicityScores={toxicityScores} />
      </div>
    </div>
  );
}

export default From;
