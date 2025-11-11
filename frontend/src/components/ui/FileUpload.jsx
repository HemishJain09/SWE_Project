import { Upload, FileText, X } from 'lucide-react';

// FileUpload Component for CSV file upload
const FileUpload = ({ onFileSelect, selectedFile, onClear, error }) => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      if (!file.name.endsWith('.csv')) {
        onFileSelect(null, 'Please select a CSV file');
        return;
      }
      onFileSelect(file, null);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        onFileSelect(null, 'Please select a CSV file');
        return;
      }
      onFileSelect(file, null);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        Upload CSV File
      </label>
      
      {!selectedFile ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition ${
            error ? 'border-red-500' : 'border-gray-300'
          }`}
        >
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-3" />
            <p className="text-sm text-gray-600 mb-1">
              <span className="font-semibold text-blue-600">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-500">CSV files only</p>
          </label>
        </div>
      ) : (
        <div className="border border-gray-300 rounded-lg p-4 flex items-center justify-between bg-gray-50">
          <div className="flex items-center gap-3">
            <FileText className="h-8 w-8 text-blue-600" />
            <div>
              <p className="text-sm font-medium text-gray-800">{selectedFile.name}</p>
              <p className="text-xs text-gray-500">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
          <button
            onClick={onClear}
            className="p-2 hover:bg-gray-200 rounded-full transition"
            title="Remove file"
          >
            <X className="h-5 w-5 text-gray-600" />
          </button>
        </div>
      )}
      
      {error && (
        <p className="text-sm text-red-600">{error}</p>
      )}
    </div>
  );
};

export default FileUpload;
