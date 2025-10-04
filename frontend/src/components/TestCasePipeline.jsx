import { useState } from 'react';
import ConfigurationSection from './ConfigurationSection';
import GeneratedDataSection from './GeneratedDataSection';
import VerificationSection from './VerificationSection';
import { useTestCaseGeneration } from '../hooks/useTestCaseGeneration';
import { useTestCaseVerification } from '../hooks/useTestCaseVerification';

// Main App Component
const TestCasePipeline = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [numTestCases, setNumTestCases] = useState('');

  // Use custom hooks for business logic
  const { 
    generatedData, 
    isGenerating, 
    generateTestCases, 
    downloadCSV 
  } = useTestCaseGeneration();

  const { 
    verificationReport, 
    isVerifying, 
    verifyTestCases, 
    resetVerification 
  } = useTestCaseVerification();

  const handleGenerateTestCases = async () => {
    resetVerification(); // Reset verification when generating new test cases
    await generateTestCases(numTestCases);
  };

  const handleVerifyTestCases = async () => {
    await verifyTestCases(generatedData);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Test Case Generation Pipeline
          </h1>
          <p className="text-gray-600">
            Generate and verify test cases for classification models
          </p>
        </header>

        {/* Configuration Section */}
        <ConfigurationSection
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          numTestCases={numTestCases}
          setNumTestCases={setNumTestCases}
          onGenerate={handleGenerateTestCases}
          isGenerating={isGenerating}
        />

        {/* Generated Data Section */}
        <GeneratedDataSection
          generatedData={generatedData}
          onDownloadCSV={downloadCSV}
          onVerify={handleVerifyTestCases}
          isVerifying={isVerifying}
        />

        {/* Verification Report Section */}
        <VerificationSection verificationReport={verificationReport} />
      </div>
    </div>
  );
};

export default TestCasePipeline;