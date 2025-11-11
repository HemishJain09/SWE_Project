# Frontend Data Flow Changes - Summary

## 🎯 Overview
The frontend has been restructured to implement an AI-powered model selection workflow using Gemini API.

## 📋 New Data Flow

### Previous Flow:
1. User manually selects model from dropdown
2. User enters number of test cases
3. Generate test cases
4. Verify with FinBERT

### New Flow:
1. **Data Input** → User chooses text or CSV upload
2. **AI Analysis** → Gemini API analyzes content and selects optimal model
3. **Configuration** → User enters number of test cases (model is read-only)
4. **Generation** → Generate test cases with AI-selected model
5. **Verification** → Verify test cases

---

## 📁 New Files Created

### Custom Hook
- **`/hooks/useModelSelection.js`**
  - Handles content analysis via Gemini API
  - Returns selected model, confidence, and reasoning
  - States: `selectedModel`, `isAnalyzing`, `error`, `analysisDetails`

### Components
- **`/components/DataInputSection.jsx`**
  - Radio buttons for text/CSV selection
  - Text area for pasting content
  - File upload for CSV files
  - Helper text showing available models

- **`/components/ModelSelectionSection.jsx`**
  - "Analyze & Select Model" button
  - Displays AI-selected model with color coding
  - Shows confidence level and reasoning
  - Visual indicators for each model type

### UI Components
- **`/components/ui/RadioGroup.jsx`**
  - Reusable radio button group component
  
- **`/components/ui/TextArea.jsx`**
  - Textarea with label and error handling
  
- **`/components/ui/FileUpload.jsx`**
  - Drag-and-drop CSV file upload
  - File validation and preview
  - Clear file functionality

---

## 🔄 Modified Files

### `/components/TestCasePipeline.jsx`
**Changes:**
- Added data input state management (`inputType`, `textInput`, `csvFile`)
- Integrated `useModelSelection` hook
- Added `handleAnalyzeContent()` function
- Updated component rendering order
- Conditional rendering based on model selection
- CSV file reading logic for analysis

**New State Variables:**
```javascript
const [inputType, setInputType] = useState('text');
const [textInput, setTextInput] = useState('');
const [csvFile, setCsvFile] = useState(null);
```

### `/components/ConfigurationSection.jsx`
**Changes:**
- Removed model dropdown (Dropdown component)
- Added read-only model display with AI indicator
- Model is now passed as prop (not editable)
- Added warning message if no model selected
- Updated styling to show AI-selected model

---

## 🎨 Model Information

The system now supports 6 specialized BERT models:

| Model | Name | Use Case | Color Theme |
|-------|------|----------|-------------|
| `finbert` | FinBERT | Financial text & documents | Green |
| `berttweet` | BERTweet | Twitter & social media | Blue |
| `patentbert` | PatentBERT | Patent documents & legal text | Purple |
| `clinicalbert` | ClinicalBERT | Medical & clinical records | Red |
| `biobert` | BioBERT | Biomedical & biological research | Teal |
| `scibert` | SciBERT | Scientific papers & research | Indigo |

---

## 🔌 API Integration

### New Endpoint Required (Backend)
```
POST http://127.0.0.1:8000/analyze-content
```

**Request Body:**
```json
{
  "content": "string (text content or CSV content)",
  "input_type": "text" | "csv"
}
```

**Response:**
```json
{
  "model": "finbert",
  "confidence": 95,
  "reasoning": "The content contains financial terminology..."
}
```

### Existing Endpoints (Still Used)
- `POST /generate` - Generate test cases
- `POST /verify` - Verify test cases

---

## 🎯 User Experience Flow

### Step 1: Data Input
- User sees clean interface with radio buttons
- Choose between "Paste Text" or "Upload CSV File"
- Helper tooltip shows all 6 available models
- Input validation prevents empty submissions

### Step 2: AI Analysis
- Click "Analyze & Select Model" button
- Loading state shows "Analyzing Content..."
- AI analyzes content and selects best model
- Results display:
  - Model name (large, color-coded)
  - Model description
  - Confidence percentage
  - AI reasoning in italics

### Step 3: Configuration
- Model field shows AI-selected model (read-only)
- Sparkles icon indicates AI selection
- User enters number of test cases (1-1000)
- Warning shown if no model selected yet

### Step 4: Generation & Verification
- Same as before (unchanged)

---

## 🧩 Component Hierarchy

```
TestCasePipeline
├─ DataInputSection
│  ├─ RadioGroup (input type selection)
│  ├─ TextArea (if text input)
│  └─ FileUpload (if CSV upload)
│
├─ ModelSelectionSection
│  └─ Button (analyze trigger)
│
├─ ConfigurationSection (conditional: if selectedModel)
│  ├─ Read-only model display
│  ├─ NumberInput (test case count)
│  └─ Button (generate trigger)
│
├─ GeneratedDataSection (conditional: if generatedData)
│  ├─ CSVTable
│  └─ Buttons (download, verify)
│
└─ VerificationSection (conditional: if verificationReport)
   └─ AccuracyReport
```

---

## 🎨 Visual Improvements

### Color-Coded Models
Each model has its own color theme for easy identification:
- Background color (light shade)
- Border color
- Text color
- Icon color

### Progressive Disclosure
- Sections appear as user progresses through the flow
- ConfigurationSection only shows after model selection
- Clear visual feedback at each step

### Loading States
- "Analyzing Content..." during AI analysis
- "Generating..." during test case generation
- "Verifying..." during verification

---

## 🔍 Validation & Error Handling

### Input Validation
- Empty text prevention
- CSV file type validation
- File size display
- Clear error messages

### Error Display
- Red alert boxes for errors
- Specific error messages from API
- Inline validation feedback

---

## 🚀 Next Steps (Backend Required)

To complete this implementation, the backend needs to:

1. **Create `/analyze-content` endpoint**
   - Accept text or CSV content
   - Integrate with Gemini API
   - Analyze content and determine domain
   - Return model name, confidence, and reasoning

2. **Update model configuration**
   - Ensure all 6 models are configured
   - Map model names correctly

3. **Environment Setup**
   - Ensure GEMINI_API_KEY is set
   - Configure model selection logic

---

## 📊 Code Quality

### Modularity
✅ Clean separation of concerns
✅ Reusable UI components
✅ Custom hooks for business logic
✅ Clear component responsibilities

### Maintainability
✅ Comprehensive documentation
✅ Descriptive variable names
✅ JSDoc comments
✅ Consistent code style

### Scalability
✅ Easy to add new models
✅ Flexible input types
✅ Extensible API structure

---

## 🎉 Summary

The frontend has been successfully restructured to support AI-powered model selection. The code is modular, clean, and follows React best practices. All changes are isolated to the frontend folder as requested.

**Key Achievements:**
- ✅ New data flow implemented
- ✅ 6 specialized models supported
- ✅ Clean, modular component structure
- ✅ Comprehensive error handling
- ✅ Beautiful, intuitive UI
- ✅ Ready for backend integration
