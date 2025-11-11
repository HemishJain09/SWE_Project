import { Brain, Sparkles, CheckCircle, AlertCircle } from 'lucide-react';
import Button from './ui/Button';

/**
 * ModelSelectionSection Component
 * Displays AI analysis button and selected model information
 */
const ModelSelectionSection = ({ 
  onAnalyze, 
  isAnalyzing, 
  selectedModel, 
  analysisDetails,
  canAnalyze,
  error 
}) => {
  // Model display names and descriptions
  const modelInfo = {
    finbert: {
      name: 'FinBERT',
      description: 'Specialized in financial text and documents',
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200'
    },
    berttweet: {
      name: 'BERTweet',
      description: 'Optimized for Twitter and social media content',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200'
    },
    patentbert: {
      name: 'PatentBERT',
      description: 'Designed for patent documents and legal text',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200'
    },
    clinicalbert: {
      name: 'ClinicalBERT',
      description: 'Specialized in medical and clinical records',
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200'
    },
    biobert: {
      name: 'BioBERT',
      description: 'Expert in biomedical and biological research',
      color: 'text-teal-600',
      bgColor: 'bg-teal-50',
      borderColor: 'border-teal-200'
    },
    scibert: {
      name: 'SciBERT',
      description: 'Tailored for scientific papers and research',
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50',
      borderColor: 'border-indigo-200'
    }
  };

  const currentModelInfo = selectedModel ? modelInfo[selectedModel.toLowerCase()] : null;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="text-purple-600" />
        <h2 className="text-2xl font-semibold text-gray-800">
          AI Model Selection
        </h2>
      </div>

      <p className="text-gray-600 mb-6">
        Let our AI analyze your content and select the most suitable classification model.
      </p>

      {/* Analyze Button */}
      <div className="mb-6">
        <Button
          onClick={onAnalyze}
          disabled={!canAnalyze || isAnalyzing}
          icon={Sparkles}
          variant="primary"
        >
          {isAnalyzing ? 'Analyzing Content...' : 'Analyze & Select Model'}
        </Button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertCircle className="text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-800">Analysis Failed</p>
            <p className="text-sm text-red-600 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Selected Model Display */}
      {selectedModel && currentModelInfo && (
        <div className={`p-6 border-2 rounded-lg ${currentModelInfo.bgColor} ${currentModelInfo.borderColor}`}>
          <div className="flex items-start gap-4">
            <CheckCircle className={`${currentModelInfo.color} flex-shrink-0 mt-1`} size={32} />
            <div className="flex-1">
              <h3 className={`text-xl font-bold ${currentModelInfo.color} mb-2`}>
                {currentModelInfo.name}
              </h3>
              <p className="text-gray-700 mb-3">
                {currentModelInfo.description}
              </p>
              
              {analysisDetails && (
                <div className="space-y-2">
                  {analysisDetails.confidence && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-600">Confidence:</span>
                      <span className={`text-sm font-bold ${currentModelInfo.color}`}>
                        {analysisDetails.confidence}%
                      </span>
                    </div>
                  )}
                  
                  {analysisDetails.reasoning && (
                    <div className="mt-3 p-3 bg-white rounded border border-gray-200">
                      <p className="text-sm font-medium text-gray-600 mb-1">AI Reasoning:</p>
                      <p className="text-sm text-gray-700 italic">
                        "{analysisDetails.reasoning}"
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Placeholder when no model selected */}
      {!selectedModel && !isAnalyzing && (
        <div className="p-6 border-2 border-dashed border-gray-300 rounded-lg text-center">
          <Brain className="mx-auto h-12 w-12 text-gray-400 mb-3" />
          <p className="text-gray-500">
            Click "Analyze & Select Model" to begin
          </p>
        </div>
      )}
    </div>
  );
};

export default ModelSelectionSection;
