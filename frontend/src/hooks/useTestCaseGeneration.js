import { useState, useCallback } from 'react';
import axios from 'axios'; 

const API_BASE_URL = 'http://127.0.0.1:8000';

export const useTestCaseGeneration = () => {
  const [generatedData, setGeneratedData] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);

  const generateTestCases = async (numTestCases) => {
    setIsGenerating(true);
    setError(null);
    setGeneratedData(null);
    
    try {
      const response = await axios.post(
        `${API_BASE_URL}/generate`,
        { num_test_cases: parseInt(numTestCases) },
        { responseType: 'blob' } // Handle the CSV file response
      );

      // Convert blob response to text and then parse CSV
      const csvText = await response.data.text();
      const rows = csvText.split('\n');
      const headers = rows[0].split(',');
      const data = rows.slice(1).filter(row => row.trim() !== '').map(row => {
          const values = row.match(/(".*?"|[^",]+)(?=\s*,|\s*$)/g).map(val => val.replace(/^"|"$/g, ''));
          return headers.reduce((obj, header, i) => {
              obj[header.trim()] = values[i] ? values[i].trim() : '';
              return obj;
          }, {});
      });

      setGeneratedData(data);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to generate test cases.';
      setError(errorMessage);
      console.error("Generation Error:", err);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadCSV = useCallback(() => {
    if (!generatedData || generatedData.length === 0) return;

    const headers = Object.keys(generatedData[0]);
    const csvContent = [
      headers.join(','),
      ...generatedData.map(row => 
        headers.map(h => `"${(row[h] || '').replace(/"/g, '""')}"`).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated_test_cases.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [generatedData]);

  return {
    generatedData,
    isGenerating,
    error,
    generateTestCases,
    downloadCSV
  };
};
