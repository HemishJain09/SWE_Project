import { CheckCircle, Download, AlertCircle } from 'lucide-react';
import CSVTable from './ui/CSVTable';
import Button from './ui/Button';

const GeneratedDataSection = ({ 
  generatedData, 
  onDownloadCSV, 
  onVerify, 
  isVerifying 
}) => {
  if (!generatedData) return null;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-gray-800 flex items-center gap-2">
          <CheckCircle className="text-green-600" />
          Generated Test Cases
        </h2>
        <Button onClick={onDownloadCSV} variant="outline" icon={Download}>
          Download CSV
        </Button>
      </div>
      <div className="mb-6">
        <CSVTable data={generatedData.slice(0, 10)} />
        {generatedData.length > 10 && (
          <p className="text-sm text-gray-500 mt-2 text-center">
            Showing 10 of {generatedData.length} test cases
          </p>
        )}
      </div>
      <Button
        onClick={onVerify}
        variant="secondary"
        disabled={isVerifying}
        icon={AlertCircle}
      >
        {isVerifying ? 'Verifying...' : 'Verify with FinBERT'}
      </Button>
    </div>
  );
};

export default GeneratedDataSection;