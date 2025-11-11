# 📊 System Architecture Diagram

## Overall System Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                 USER                                     │
│                    (Browser - http://localhost:5173)                    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + Vite)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      App.jsx                                     │   │
│  │                         ↓                                        │   │
│  │                  TestCasePipeline.jsx                           │   │
│  │          (Main Orchestrator Component)                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │ DataInputSection │  │ ModelSelection   │  │ Configuration    │     │
│  │                  │  │ Section          │  │ Section          │     │
│  │ - Text Input     │  │ - AI Analysis    │  │ - Test Count     │     │
│  │ - CSV Upload     │  │ - Model Display  │  │ - GenerateBtn   │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐                            │
│  │ GeneratedData    │  │ Verification     │                            │
│  │ Section          │  │ Section          │                            │
│  │ - CSV Table      │  │ - Metrics        │                            │
│  │ - Download       │  │ - Report Card    │                            │
│  └──────────────────┘  └──────────────────┘                            │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Custom Hooks (Business Logic)                │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │  • useModelSelection.js    → Gemini API integration            │    │
│  │  • useTestCaseGeneration.js → Generate endpoint                │    │
│  │  • useTestCaseVerification.js → Verify endpoint                │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ HTTP/JSON
                               │ Axios Requests
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI + Python)                             │
│                    http://127.0.0.1:8000                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         main.py                                  │   │
│  │                    (FastAPI Application)                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      API ENDPOINTS                                │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  GET  /                                                           │  │
│  │  └─→ API info, version, supported models                        │  │
│  │                                                                   │  │
│  │  POST /analyze-content                                           │  │
│  │  ├─→ Input: { content, input_type }                             │  │
│  │  ├─→ Calls: gemini_analyzer.py                                  │  │
│  │  └─→ Output: { model, confidence, reasoning }                   │  │
│  │                                                                   │  │
│  │  POST /generate                                                  │  │
│  │  ├─→ Input: { num_test_cases, model_type }                      │  │
│  │  ├─→ Calls: test_case_generation.py                             │  │
│  │  └─→ Output: CSV file with test cases                           │  │
│  │                                                                   │  │
│  │  POST /verify                                                    │  │
│  │  ├─→ Input: CSV file                                            │  │
│  │  ├─→ Calls: finbert_verification.py                             │  │
│  │  └─→ Output: { accuracy, precision, recall, f1_score }          │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     PYTHON MODULES                                │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │                                                                   │  │
│  │  gemini_analyzer.py                                              │  │
│  │  ├─→ GeminiContentAnalyzer class                                │  │
│  │  ├─→ Analyzes content & recommends model                        │  │
│  │  └─→ Fallback: keyword-based analysis                           │  │
│  │                                                                   │  │
│  │  test_case_generation.py                                         │  │
│  │  ├─→ construct_prompt() - Model-specific prompts                │  │
│  │  ├─→ get_llm_response() - Gemini API calls                      │  │
│  │  └─→ run_generation() - Batch generation logic                  │  │
│  │                                                                   │  │
│  │  finbert_verification.py                                         │  │
│  │  ├─→ Loads FinBERT model                                        │  │
│  │  ├─→ Classifies test cases                                      │  │
│  │  └─→ Calculates metrics                                         │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      GOOGLE GEMINI API                                   │
│                    (Gemini 2.5 Pro Model)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  • Content Analysis & Model Selection                                   │
│  • Test Case Generation                                                 │
│  • Domain-Specific Understanding                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Sequence

### 1. Content Analysis Flow

```
User Input
   │
   ├─→ Paste Text / Upload CSV
   │
   ▼
DataInputSection Component
   │
   ├─→ State: textInput or csvFile
   │
   ▼
User Clicks "Analyze & Select Model"
   │
   ├─→ handleAnalyzeContent()
   │
   ▼
useModelSelection Hook
   │
   ├─→ analyzeAndSelectModel(content, inputType)
   │
   ▼
Axios POST /analyze-content
   │
   ├─→ { content: "...", input_type: "text" }
   │
   ▼
Backend: main.py
   │
   ├─→ ContentAnalysisRequest validation
   │
   ▼
gemini_analyzer.py
   │
   ├─→ GeminiContentAnalyzer.analyze_content()
   ├─→ construct_analysis_prompt()
   ├─→ Gemini API call
   ├─→ parse_gemini_response()
   │
   ▼
Response: { model, confidence, reasoning }
   │
   ├─→ Frontend receives response
   │
   ▼
ModelSelectionSection Component
   │
   ├─→ Displays selected model
   ├─→ Shows confidence score
   └─→ Shows AI reasoning
```

### 2. Test Case Generation Flow

```
User Enters Test Case Count
   │
   ├─→ numTestCases state updated
   │
   ▼
User Clicks "Generate Test Cases"
   │
   ├─→ handleGenerateTestCases()
   │
   ▼
useTestCaseGeneration Hook
   │
   ├─→ generateTestCases(numTestCases, selectedModel)
   │
   ▼
Axios POST /generate
   │
   ├─→ { num_test_cases: 100, model_type: "finbert" }
   │
   ▼
Backend: main.py
   │
   ├─→ GenerationRequest validation
   │
   ▼
test_case_generation.py
   │
   ├─→ run_generation(total_samples, batch_size, model_type)
   ├─→ Loop: while num_generated < total_samples
   │   ├─→ construct_prompt(batch_size, existing_count, model_type)
   │   ├─→ get_llm_response(prompt, api_key)
   │   ├─→ parse_llm_output(response)
   │   └─→ Accumulate test cases
   │
   ▼
Response: CSV Blob
   │
   ├─→ Frontend parses CSV
   ├─→ Converts to JSON array
   │
   ▼
GeneratedDataSection Component
   │
   ├─→ CSVTable displays data
   ├─→ Download button available
   └─→ Verify button available
```

### 3. Verification Flow

```
User Clicks "Verify with FinBERT"
   │
   ├─→ handleVerifyTestCases()
   │
   ▼
useTestCaseVerification Hook
   │
   ├─→ verifyTestCases(generatedData)
   ├─→ Convert JSON to CSV
   ├─→ Create FormData with CSV
   │
   ▼
Axios POST /verify
   │
   ├─→ multipart/form-data with CSV file
   │
   ▼
Backend: main.py
   │
   ├─→ Read uploaded CSV
   ├─→ Parse to DataFrame
   │
   ▼
finbert_verification.py
   │
   ├─→ run_verification(dataframe)
   ├─→ Load FinBERT model
   ├─→ Classify each review
   ├─→ Compare with expected sentiment
   ├─→ Calculate metrics
   │
   ▼
Response: { accuracy, precision, recall, f1_score, ... }
   │
   ├─→ Frontend receives metrics
   ├─→ Transform property names
   │
   ▼
VerificationSection Component
   │
   └─→ AccuracyReport displays metrics
```

---

## Component Hierarchy

```
App
└── TestCasePipeline
    ├── DataInputSection
    │   ├── RadioGroup (text/csv selection)
    │   ├── TextArea (if text input)
    │   └── FileUpload (if csv input)
    │
    ├── ModelSelectionSection
    │   └── Button (analyze trigger)
    │
    ├── ConfigurationSection (conditional: if model selected)
    │   ├── Model Display (read-only)
    │   ├── NumberInput (test count)
    │   └── Button (generate trigger)
    │
    ├── GeneratedDataSection (conditional: if data exists)
    │   ├── CSVTable (first 10 rows)
    │   ├── Button (download)
    │   └── Button (verify)
    │
    └── VerificationSection (conditional: if report exists)
        └── AccuracyReport
            └── Metric Cards (4x grid)
```

---

## State Management Flow

```
TestCasePipeline Component State
├── inputType: 'text' | 'csv'
├── textInput: string
├── csvFile: File | null
├── numTestCases: string
│
├── useModelSelection Hook State
│   ├── selectedModel: string
│   ├── isAnalyzing: boolean
│   ├── error: string | null
│   └── analysisDetails: { confidence, reasoning }
│
├── useTestCaseGeneration Hook State
│   ├── generatedData: Array<Object>
│   ├── isGenerating: boolean
│   └── error: string | null
│
└── useTestCaseVerification Hook State
    ├── verificationReport: Object
    ├── isVerifying: boolean
    └── error: string | null
```

---

## Backend Module Dependencies

```
main.py
├── FastAPI
├── pydantic (BaseModel)
├── pandas
├── test_case_generation
│   └── run_generation()
├── finbert_verification
│   └── run_verification()
└── gemini_analyzer
    └── analyze_content_for_model()

gemini_analyzer.py
├── google.generativeai
├── os, time
└── dotenv

test_case_generation.py
├── google.generativeai
├── pandas
├── tqdm
├── json
└── dotenv

finbert_verification.py
├── transformers
├── torch
├── sklearn
└── pandas
```

---

## API Request/Response Flow

### POST /analyze-content

```
Request:
┌─────────────────────────────┐
│ Headers:                    │
│   Content-Type: application/json
│                             │
│ Body:                       │
│ {                           │
│   "content": "Apple stock...",
│   "input_type": "text"      │
│ }                           │
└─────────────────────────────┘
            │
            ▼
Response:
┌─────────────────────────────┐
│ Status: 200 OK              │
│                             │
│ Body:                       │
│ {                           │
│   "model": "finbert",       │
│   "confidence": 95,         │
│   "reasoning": "Financial..." │
│ }                           │
└─────────────────────────────┘
```

### POST /generate

```
Request:
┌─────────────────────────────┐
│ Headers:                    │
│   Content-Type: application/json
│                             │
│ Body:                       │
│ {                           │
│   "num_test_cases": 100,    │
│   "model_type": "finbert"   │
│ }                           │
└─────────────────────────────┘
            │
            ▼
Response:
┌─────────────────────────────┐
│ Status: 200 OK              │
│ Content-Type: text/csv      │
│                             │
│ Body: CSV File              │
│ review,sentiment            │
│ "Apple earnings...",positive│
│ "Stock declined...",negative│
│ ...                         │
└─────────────────────────────┘
```

### POST /verify

```
Request:
┌─────────────────────────────┐
│ Headers:                    │
│   Content-Type: multipart/form-data
│                             │
│ Body:                       │
│   file: test_cases.csv      │
└─────────────────────────────┘
            │
            ▼
Response:
┌─────────────────────────────┐
│ Status: 200 OK              │
│                             │
│ Body:                       │
│ {                           │
│   "accuracy": 85.5,         │
│   "precision": 87.2,        │
│   "recall": 84.1,           │
│   "f1_score": 85.6,         │
│   "correctly_classified": 171,
│   "total_cases": 200        │
│ }                           │
└─────────────────────────────┘
```

---

## Model Selection Decision Tree

```
Content Input
     │
     ▼
Contains financial keywords?
(earnings, stock, revenue, market)
     │
     ├─ YES → FinBERT
     │
     ├─ NO
     │   │
     │   ▼
     │   Contains social media markers?
     │   (hashtags, @mentions, emojis, "tweet")
     │   │
     │   ├─ YES → BERTweet
     │   │
     │   ├─ NO
     │   │   │
     │   │   ▼
     │   │   Contains legal/patent terms?
     │   │   (patent, claim, invention, IP)
     │   │   │
     │   │   ├─ YES → PatentBERT
     │   │   │
     │   │   ├─ NO
     │   │   │   │
     │   │   │   ▼
     │   │   │   Contains medical terms?
     │   │   │   (patient, diagnosis, treatment)
     │   │   │   │
     │   │   │   ├─ YES → ClinicalBERT
     │   │   │   │
     │   │   │   ├─ NO
     │   │   │   │   │
     │   │   │   │   ▼
     │   │   │   │   Contains biomedical terms?
     │   │   │   │   (protein, gene, DNA, cell)
     │   │   │   │   │
     │   │   │   │   ├─ YES → BioBERT
     │   │   │   │   │
     │   │   │   │   └─ NO → SciBERT (default scientific)
```

---

## Error Handling Flow

```
User Action
     │
     ▼
Try Block
     │
     ├─ Success → Update State → Render Result
     │
     └─ Error Caught
          │
          ▼
     Error Type?
          │
          ├─ Validation Error (400)
          │   └─→ Display: "Please provide valid input"
          │
          ├─ API Error (500)
          │   └─→ Display: "Service temporarily unavailable"
          │
          ├─ Network Error
          │   └─→ Display: "Check your connection"
          │
          └─ Unknown Error
              └─→ Display: Generic error message
                  └─→ Log to console for debugging
```

This architecture provides a robust, scalable foundation for AI-powered test case generation across multiple domains!
