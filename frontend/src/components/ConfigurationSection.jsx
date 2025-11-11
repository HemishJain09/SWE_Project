import { FileText, Sparkles } from 'lucide-react';
import NumberInput from './ui/NumberInput';
import Button from './ui/Button';

/**
 * ConfigurationSection Component
 * Configure test case generation parameters
 * Model is now AI-selected and displayed (read-only)
 */
const ConfigurationSection = ({ 
  selectedModel, 
  numTestCases, 
  setNumTestCases, 
  onGenerate, 
  isGenerating 
}) => {
  // Model display names
  const modelDisplayNames = {
    finbert: 'FinBERT',
    berttweet: 'BERTweet',
    patentbert: 'PatentBERT',
    clinicalbert: 'ClinicalBERT',
    biobert: 'BioBERT',
    scibert: 'SciBERT'
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="text-blue-600" />
        <h2 className="text-2xl font-semibold text-gray-800">
          Test Case Configuration
        </h2>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* AI-Selected Model Display (Read-only) */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Selected Model (AI-Selected)
          </label>
          <div className="px-4 py-3 border border-gray-300 rounded-lg bg-gray-50 flex items-center gap-2">
            <Sparkles className="text-purple-600" size={20} />
            <span className="text-gray-800 font-medium">
              {selectedModel ? modelDisplayNames[selectedModel.toLowerCase()] || selectedModel : 'No model selected'}
            </span>
          </div>
        </div>

        {/* Number of Test Cases Input */}
        <NumberInput
          label="Number of Test Cases"
          value={numTestCases}
          onChange={setNumTestCases}
          min="1"
          max="1000"
          placeholder="Enter number (1-1000)"
        />
      </div>

      {/* Generate Button */}
      <Button
        onClick={onGenerate}
        disabled={!selectedModel || !numTestCases || isGenerating}
        icon={FileText}
      >
        {isGenerating ? 'Generating...' : 'Generate Test Cases'}
      </Button>

      {!selectedModel && (
        <p className="text-sm text-amber-600 mt-3">
          ⚠️ Please analyze your content first to select a model
        </p>
      )}
    </div>
  );
};

export default ConfigurationSection;