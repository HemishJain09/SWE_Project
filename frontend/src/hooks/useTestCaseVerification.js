import { useState } from 'react';

export const useTestCaseVerification = () => {
  const [verificationReport, setVerificationReport] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);

  const verifyTestCases = async (generatedData) => {
    if (!generatedData || generatedData.length === 0) return;

    setIsVerifying(true);

    // Simulate API call to FinBERT
    await new Promise(resolve => setTimeout(resolve, 2500));

    // Generate dummy accuracy report
    const total = generatedData.length;
    const correct = Math.floor(total * (0.75 + Math.random() * 0.2));
    const accuracy = ((correct / total) * 100).toFixed(2);
    const precision = (70 + Math.random() * 25).toFixed(2);
    const recall = (65 + Math.random() * 30).toFixed(2);

    setVerificationReport({
      accuracy,
      correct,
      total,
      precision,
      recall
    });

    setIsVerifying(false);
  };

  const resetVerification = () => {
    setVerificationReport(null);
  };

  return {
    verificationReport,
    isVerifying,
    verifyTestCases,
    resetVerification
  };
};