import { useState } from 'react';
import DataInputSection from './DataInputSection';
import ModelSelectionSection from './ModelSelectionSection';
import ConfigurationSection from './ConfigurationSection';
import GeneratedDataSection from './GeneratedDataSection';
import VerificationSection from './VerificationSection';
import { useModelSelection } from '../hooks/useModelSelection';
import { useTestCaseGeneration } from '../hooks/useTestCaseGeneration';
import { useTestCaseVerification } from '../hooks/useTestCaseVerification';

/**
 * Main TestCasePipeline Component
 * Orchestrates the complete flow:
 * 1. Data Input (text or CSV)
 * 2. AI Model Selection (via Gemini API)
 * 3. Test Case Generation
 * 4. Verification with selected model
 */
const TestCasePipeline = () => {
  // Data Input State
  const [inputType, setInputType] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [csvFile, setCsvFile] = useState(null);
  const [numTestCases, setNumTestCases] = useState('');

  // Custom Hooks
  const { 
    selectedModel,
    isAnalyzing,
    error: modelError,
    analysisDetails,
    analyzeAndSelectModel
  } = useModelSelection();

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

  /**
   * Handle content analysis and model selection
   */
  const handleAnalyzeContent = async () => {
    let content = '';

    if (inputType === 'text') {
      if (!textInput || textInput.trim() === '') {
        alert('Please enter some text to analyze');
        return;
      }
      content = textInput;
    } else {
      if (!csvFile) {
        alert('Please upload a CSV file');
        return;
      }
      // Read CSV file content
      try {
        const fileContent = await csvFile.text();
        content = fileContent;
      } catch (error) {
        alert('Failed to read CSV file');
        console.error('File reading error:', error);
        return;
      }
    }

    // Reset previous states
    resetVerification();

    // Analyze content and select model
    await analyzeAndSelectModel(content, inputType);
  };

  /**
   * Handle test case generation
   */
  const handleGenerateTestCases = async () => {
    resetVerification(); // Reset verification when generating new test cases
    await generateTestCases(numTestCases, selectedModel);
  };

  /**
   * Handle test case verification
   */
  const handleVerifyTestCases = async () => {
    await verifyTestCases(generatedData);
  };

  /**
   * Check if analysis can be performed
   */
  const canAnalyze = inputType === 'text' 
    ? textInput && textInput.trim() !== '' 
    : csvFile !== null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            AI-Powered Test Case Generation Pipeline
          </h1>
          <p className="text-gray-600">
            Intelligent model selection and automated test case generation
          </p>
        </header>

        {/* Step 1: Data Input Section */}
        <DataInputSection
          inputType={inputType}
          setInputType={setInputType}
          textInput={textInput}
          setTextInput={setTextInput}
          csvFile={csvFile}
          setCsvFile={setCsvFile}
        />

        {/* Step 2: AI Model Selection Section */}
        <ModelSelectionSection
          onAnalyze={handleAnalyzeContent}
          isAnalyzing={isAnalyzing}
          selectedModel={selectedModel}
          analysisDetails={analysisDetails}
          canAnalyze={canAnalyze}
          error={modelError}
        />

        {/* Step 3: Test Case Configuration Section */}
        {selectedModel && (
          <ConfigurationSection
            selectedModel={selectedModel}
            numTestCases={numTestCases}
            setNumTestCases={setNumTestCases}
            onGenerate={handleGenerateTestCases}
            isGenerating={isGenerating}
          />
        )}

        {/* Step 4: Generated Data Section */}
        <GeneratedDataSection
          generatedData={generatedData}
          onDownloadCSV={downloadCSV}
          onVerify={handleVerifyTestCases}
          isVerifying={isVerifying}
        />

        {/* Step 5: Verification Report Section */}
        <VerificationSection verificationReport={verificationReport} />
      </div>
    </div>
  );
};

export default TestCasePipeline;