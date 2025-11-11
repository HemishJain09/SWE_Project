import { useState } from 'react';
import axios from 'axios'; // Make sure to install axios: npm install axios

const API_BASE_URL = 'http://127.0.0.1:8000';

export const useTestCaseVerification = () => {
  const [verificationReport, setVerificationReport] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [error, setError] = useState(null);

  const verifyTestCases = async (generatedData) => {
    if (!generatedData || generatedData.length === 0) {
        setError("Cannot verify: No data provided.");
        return;
    }

    setIsVerifying(true);
    setError(null);
    setVerificationReport(null);

    try {
        // Convert the generated JSON data back to a CSV file to send to the backend
        const headers = Object.keys(generatedData[0]);
        const csvContent = [
            headers.join(','),
            ...generatedData.map(row => 
                headers.map(h => `"${(row[h] || '').replace(/"/g, '""')}"`).join(',')
            )
        ].join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const formData = new FormData();
        formData.append('file', blob, 'test_cases_to_verify.csv');

        const response = await axios.post(`${API_BASE_URL}/verify`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        
        // Log the raw response from the backend for debugging
        console.log("Raw backend response:", response.data);

        // Transform the backend data to match expected frontend property names
        const backendReport = response.data;
        const formattedReport = {
          accuracy: backendReport.accuracy,
          precision: backendReport.precision,
          recall: backendReport.recall,
          f1Score: backendReport.f1_score,
          correctPredictions: backendReport.correctly_classified,
          totalPredictions: backendReport.total_cases, 
        };

        console.log("Formatted report for UI:", formattedReport);
        setVerificationReport(formattedReport);

    } catch (err) {
        const errorMessage = err.response?.data?.detail || err.message || 'Failed to verify test cases.';
        setError(errorMessage);
        console.error("Verification Error:", err);
    } finally {
        setIsVerifying(false);
    }
  };

  const resetVerification = () => {
    setVerificationReport(null);
    setError(null);
  };

  return {
    verificationReport,
    isVerifying,
    error,
    verifyTestCases,
    resetVerification
  };
};

