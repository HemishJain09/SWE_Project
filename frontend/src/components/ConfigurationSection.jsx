import { FileText } from 'lucide-react';
import Dropdown from './ui/Dropdown';
import NumberInput from './ui/NumberInput';
import Button from './ui/Button';

const ConfigurationSection = ({ 
  selectedModel, 
  setSelectedModel, 
  numTestCases, 
  setNumTestCases, 
  onGenerate, 
  isGenerating 
}) => {
  const modelOptions = [
    { value: 'bert', label: 'BERT' },
    { value: 'roberta', label: 'RoBERTa' },
    { value: 'distilbert', label: 'DistilBERT' },
    { value: 'finbert', label: 'FinBERT' },
    { value: 'xlnet', label: 'XLNet' }
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Configuration
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <Dropdown
          label="Select Classification Model"
          value={selectedModel}
          onChange={setSelectedModel}
          options={modelOptions}
          placeholder="Choose a model..."
        />
        <NumberInput
          label="Number of Test Cases"
          value={numTestCases}
          onChange={setNumTestCases}
          min="1"
          max="1000"
          placeholder="Enter number (1-1000)"
        />
      </div>
      <Button
        onClick={onGenerate}
        disabled={!selectedModel || !numTestCases || isGenerating}
        icon={FileText}
      >
        {isGenerating ? 'Generating...' : 'Generate Test Cases'}
      </Button>
    </div>
  );
};

export default ConfigurationSection;