import { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

/**
 * Custom hook for intelligent model selection using Gemini API
 * Analyzes text or CSV content to determine the most appropriate classification model
 * 
 * Available models:
 * - finbert: Financial text classification
 * - berttweet: Twitter/social media text
 * - patentBert: Patent documents
 * - clinicalbert: Clinical/medical text
 * - BioBert: Biomedical/biological text
 * - sciBert: Scientific papers and research
 */
export const useModelSelection = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [analysisDetails, setAnalysisDetails] = useState(null);

  /**
   * Analyze content and select appropriate model
   * @param {string} content - Text content to analyze
   * @param {string} inputType - Type of input: 'text' or 'csv'
   * @returns {Promise<string>} - Selected model name
   */
  const analyzeAndSelectModel = async (content, inputType = 'text') => {
    if (!content || content.trim() === '') {
      setError('Content cannot be empty');
      return null;
    }

    setIsAnalyzing(true);
    setError(null);
    setSelectedModel('');
    setAnalysisDetails(null);

    try {
      const response = await axios.post(
        `${API_BASE_URL}/analyze-content`,
        {
          content: content,
          input_type: inputType
        },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      const { model, confidence, reasoning } = response.data;
      
      setSelectedModel(model);
      setAnalysisDetails({
        confidence,
        reasoning,
        inputType
      });

      return model;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to analyze content';
      setError(errorMessage);
      console.error('Model Selection Error:', err);
      return null;
    } finally {
      setIsAnalyzing(false);
    }
  };

  /**
   * Reset the model selection state
   */
  const resetModelSelection = () => {
    setSelectedModel('');
    setError(null);
    setAnalysisDetails(null);
  };

  /**
   * Manually set a model (if user wants to override)
   */
  const setModelManually = (model) => {
    setSelectedModel(model);
    setAnalysisDetails(null);
  };

  return {
    selectedModel,
    isAnalyzing,
    error,
    analysisDetails,
    analyzeAndSelectModel,
    resetModelSelection,
    setModelManually
  };
};
