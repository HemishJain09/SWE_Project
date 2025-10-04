import { useState } from 'react';

export const useTestCaseGeneration = () => {
  const [generatedData, setGeneratedData] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const generateTestCases = async (numTestCases) => {
    setIsGenerating(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Generate dummy CSV data
    const dummyData = Array.from({ length: parseInt(numTestCases) }, (_, i) => ({
      id: i + 1,
      text: `Sample financial text ${i + 1}: The company reported strong earnings growth.`,
      expected_sentiment: ['positive', 'negative', 'neutral'][Math.floor(Math.random() * 3)],
      category: ['earnings', 'market', 'policy'][Math.floor(Math.random() * 3)]
    }));

    setGeneratedData(dummyData);
    setIsGenerating(false);
  };

  const downloadCSV = () => {
    if (!generatedData || generatedData.length === 0) return;

    const headers = Object.keys(generatedData[0]);
    const csvContent = [
      headers.join(','),
      ...generatedData.map(row => headers.map(h => row[h]).join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'test_cases.csv';
    a.click();
  };

  return {
    generatedData,
    isGenerating,
    generateTestCases,
    downloadCSV
  };
};