import { CheckCircle } from 'lucide-react';

// Accuracy Report Component
const AccuracyReport = ({ report }) => {
  if (!report) return null;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
        <CheckCircle className="text-green-600" />
        Verification Report
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Overall Accuracy</p>
          <p className="text-3xl font-bold text-blue-600">{report.accuracy}%</p>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Correct Predictions</p>
          <p className="text-3xl font-bold text-green-600">{report.correctPredictions}/{report.totalPredictions}</p>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Precision</p>
          <p className="text-3xl font-bold text-purple-600">{report.precision}%</p>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Recall</p>
          <p className="text-3xl font-bold text-orange-600">{report.recall}%</p>
        </div>
      </div>
    </div>
  );
};

export default AccuracyReport;