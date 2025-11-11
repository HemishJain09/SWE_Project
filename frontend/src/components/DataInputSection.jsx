import { useState } from 'react';
import { FileInput, Type } from 'lucide-react';
import RadioGroup from './ui/RadioGroup';
import TextArea from './ui/TextArea';
import FileUpload from './ui/FileUpload';

/**
 * DataInputSection Component
 * Allows users to choose between text input or CSV file upload
 * for content analysis and model selection
 */
const DataInputSection = ({ 
  inputType, 
  setInputType, 
  textInput, 
  setTextInput, 
  csvFile, 
  setCsvFile 
}) => {
  const [fileError, setFileError] = useState(null);
  const [textError, setTextError] = useState(null);

  const inputOptions = [
    { value: 'text', label: 'Paste Text' },
    { value: 'csv', label: 'Upload CSV File' }
  ];

  const handleFileSelect = (file, error) => {
    setFileError(error);
    setCsvFile(file);
    if (file) {
      setTextInput(''); // Clear text input when file is selected
    }
  };

  const handleClearFile = () => {
    setCsvFile(null);
    setFileError(null);
  };

  const handleTextChange = (text) => {
    setTextInput(text);
    setTextError(null);
    if (text) {
      setCsvFile(null); // Clear file when text is entered
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <Type className="text-blue-600" />
        <h2 className="text-2xl font-semibold text-gray-800">
          Data Input
        </h2>
      </div>
      
      <p className="text-gray-600 mb-6">
        Provide your data for analysis. Our AI will automatically select the best classification model.
      </p>

      {/* Input Type Selection */}
      <div className="mb-6">
        <RadioGroup
          label="Choose Input Method"
          options={inputOptions}
          value={inputType}
          onChange={setInputType}
          name="inputType"
        />
      </div>

      {/* Conditional Input Fields */}
      <div className="mt-6">
        {inputType === 'text' ? (
          <TextArea
            label="Enter Text for Analysis"
            value={textInput}
            onChange={handleTextChange}
            placeholder="Paste your text here... (e.g., financial statements, tweets, medical records, scientific papers)"
            rows={8}
            error={textError}
          />
        ) : (
          <FileUpload
            onFileSelect={handleFileSelect}
            selectedFile={csvFile}
            onClear={handleClearFile}
            error={fileError}
          />
        )}
      </div>

      {/* Helper Text */}
      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-800">
          <strong>💡 Tip:</strong> The AI will analyze your content and automatically select from:
        </p>
        <ul className="text-xs text-blue-700 mt-2 ml-4 space-y-1">
          <li>• <strong>FinBERT</strong> - Financial text and documents</li>
          <li>• <strong>BERTweet</strong> - Twitter and social media content</li>
          <li>• <strong>PatentBERT</strong> - Patent documents and legal text</li>
          <li>• <strong>ClinicalBERT</strong> - Medical and clinical records</li>
          <li>• <strong>BioBERT</strong> - Biomedical and biological research</li>
          <li>• <strong>SciBERT</strong> - Scientific papers and research</li>
        </ul>
      </div>
    </div>
  );
};

export default DataInputSection;
