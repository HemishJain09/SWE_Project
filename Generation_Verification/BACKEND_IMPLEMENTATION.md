# Backend Implementation - AI-Powered Model Selection

## 🎯 Overview
The backend has been enhanced with Gemini AI integration to automatically analyze content and recommend the most suitable BERT model for classification tasks.

---

## 📁 New Files Created

### 1. `gemini_analyzer.py`
**Purpose:** Intelligent content analysis and model recommendation using Google Gemini API.

**Key Features:**
- Analyzes text or CSV content
- Returns recommended model with confidence score and reasoning
- Supports 6 specialized BERT models
- Includes fallback keyword-based analysis
- Robust error handling with retry logic

**Main Class:**
```python
class GeminiContentAnalyzer:
    """
    Analyzes content using Gemini API to recommend the best BERT model.
    """
    def analyze_content(content: str, input_type: str) -> Dict
```

**Model Information:**
Each model has:
- Name and description
- Domain-specific keywords
- Use case definitions

**Supported Models:**
| Model | Domain | Keywords |
|-------|--------|----------|
| `finbert` | Financial | stock, earnings, revenue, market, investment |
| `berttweet` | Social Media | tweet, twitter, hashtag, viral, post |
| `patentbert` | Legal/Patent | patent, invention, claim, IP, trademark |
| `clinicalbert` | Medical | patient, diagnosis, clinical, treatment |
| `biobert` | Biomedical | protein, gene, DNA, molecular, cell |
| `scibert` | Scientific | research, study, experiment, hypothesis |

---

## 🔄 Modified Files

### 1. `main.py`

#### New Endpoint: `/analyze-content`
```python
POST /analyze-content
Content-Type: application/json

Request Body:
{
  "content": "string (text or CSV content)",
  "input_type": "text" | "csv"
}

Response:
{
  "model": "finbert",
  "confidence": 95,
  "reasoning": "The content contains financial terminology..."
}
```

**Features:**
- Accepts text or CSV content
- Validates input
- Returns model recommendation with confidence
- Provides AI reasoning for selection

#### Updated Endpoint: `/generate`
**Changes:**
- Added `model_type` parameter (default: "finbert")
- Validates model type against supported models
- Generates domain-specific test cases
- Dynamic filename based on model type

```python
POST /generate
Content-Type: application/json

Request Body:
{
  "num_test_cases": 100,
  "model_type": "finbert"  // NEW PARAMETER
}
```

#### Updated Root Endpoint: `/`
Now returns:
- API version (2.0.0)
- Feature list
- Supported models
- Enhanced welcome message

---

### 2. `test_case_generation.py`

#### Updated: `construct_prompt()`
**Changes:**
- Added `model_type` parameter
- Model-specific prompt templates
- Domain-appropriate examples for each model
- Customized instructions per domain

**Model-Specific Configurations:**
```python
model_configs = {
    'finbert': {
        'domain': 'financial sentiment analysis',
        'content_type': 'financial statements, earnings reports...',
        'examples': [...]
    },
    'berttweet': { ... },
    'patentbert': { ... },
    'clinicalbert': { ... },
    'biobert': { ... },
    'scibert': { ... }
}
```

#### Updated: `run_generation()`
**Changes:**
- Added `model_type` parameter (default: 'finbert')
- Model type validation
- Enhanced logging with model info
- Progress bar shows model type

---

## 🚀 API Workflow

### Complete Data Flow

```
1. Frontend sends content → POST /analyze-content
   ↓
2. Gemini analyzes content → Returns model recommendation
   ↓
3. Frontend sends generation request → POST /generate
   with: { num_test_cases, model_type }
   ↓
4. Backend generates domain-specific test cases
   ↓
5. Returns CSV file with test cases
   ↓
6. Frontend displays and allows verification
```

---

## 📊 Model Selection Logic

### Primary Method: Gemini AI Analysis
1. Constructs detailed prompt with content
2. Asks Gemini to analyze domain
3. Parses JSON response (model, confidence, reasoning)
4. Validates and returns result

### Fallback Method: Keyword Analysis
If Gemini fails:
1. Counts keyword matches for each model
2. Selects model with highest score
3. Calculates confidence based on matches
4. Returns fallback recommendation

### Example Analysis Flow:
```python
# Input
content = "Apple's quarterly earnings exceeded expectations..."

# Gemini Analysis
{
  "model": "finbert",
  "confidence": 95,
  "reasoning": "Content contains financial terminology including 
                earnings, quarterly results, and company performance 
                indicators typical of financial analysis."
}
```

---

## 🔧 Configuration

### Environment Variables
Required in `.env`:
```bash
GEMINI_API_KEY=your_api_key_here
```

### API Settings
- **Gemini Model:** gemini-2.5-pro (fast and accurate)
- **Max Retries:** 3 attempts
- **Retry Delay:** Exponential backoff (2^attempt seconds)
- **Content Truncation:** 5000 characters max
- **Batch Size:** 10 test cases per API call

---

## 📝 Test Case Generation Examples

### FinBERT (Financial)
```json
{
  "review": "Apple's quarterly earnings exceeded Wall Street expectations",
  "sentiment": "positive"
}
```

### BERTweet (Social Media)
```json
{
  "review": "Just tried the new iPhone 15 and it's amazing! 🔥 #Apple",
  "sentiment": "positive"
}
```

### PatentBERT (Legal)
```json
{
  "review": "A novel apparatus for wireless power transmission...",
  "sentiment": "positive"
}
```

### ClinicalBERT (Medical)
```json
{
  "review": "Patient shows improvement after physical therapy",
  "sentiment": "positive"
}
```

### BioBERT (Biomedical)
```json
{
  "review": "Protein binding mechanism shows enhanced affinity",
  "sentiment": "positive"
}
```

### SciBERT (Scientific)
```json
{
  "review": "Results demonstrate significant correlation (p < 0.05)",
  "sentiment": "positive"
}
```

---

## 🛡️ Error Handling

### Validation Errors (400)
- Empty content
- Invalid model type
- Missing required fields

### Processing Errors (500)
- Gemini API failures
- JSON parsing errors
- Test case generation failures

### Retry Logic
- Automatic retries on API failures
- Exponential backoff
- Fallback to keyword analysis

---

## 🧪 Testing the Backend

### Test Content Analysis
```bash
curl -X POST http://127.0.0.1:8000/analyze-content \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Apple stock surged after strong earnings report",
    "input_type": "text"
  }'
```

**Expected Response:**
```json
{
  "model": "finbert",
  "confidence": 92,
  "reasoning": "Content contains financial market terminology..."
}
```

### Test Generation with Model Type
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "num_test_cases": 10,
    "model_type": "berttweet"
  }' \
  --output berttweet_cases.csv
```

### Check API Status
```bash
curl http://127.0.0.1:8000/
```

---

## 📦 Dependencies

All required packages are in `requirements.txt`:
- ✅ `google-generativeai` - Gemini API
- ✅ `fastapi` - Web framework
- ✅ `pandas` - Data processing
- ✅ `python-dotenv` - Environment variables
- ✅ `uvicorn` - ASGI server

---

## 🚀 Running the Backend

### 1. Install Dependencies
```bash
cd Generation_Verification
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create/update `.env` file:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Start the Server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Access API Documentation
Open browser:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## 📊 API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API info and supported models |
| POST | `/analyze-content` | Analyze content and recommend model |
| POST | `/generate` | Generate test cases for specific model |
| POST | `/verify` | Verify test cases (unchanged) |

---

## 🎨 Code Quality

### Design Principles
✅ **Modularity** - Separated concerns (analyzer, generator, API)
✅ **Extensibility** - Easy to add new models
✅ **Robustness** - Comprehensive error handling
✅ **Documentation** - Clear docstrings and comments
✅ **Type Safety** - Pydantic models for validation

### Best Practices
- Environment variable management
- Retry logic with exponential backoff
- Fallback mechanisms
- Detailed logging
- Input validation
- Proper HTTP status codes

---

## 🔍 Model Selection Algorithm

### Step-by-Step Process

1. **Content Preprocessing**
   - Truncate if > 5000 chars
   - Clean and format content

2. **Gemini API Call**
   - Construct domain-aware prompt
   - Send to Gemini 2.5 Pro
   - Parse JSON response

3. **Validation**
   - Check model name validity
   - Validate confidence range (0-100)
   - Ensure reasoning is present

4. **Fallback (if needed)**
   - Count keyword matches
   - Score each model
   - Select highest scorer
   - Calculate confidence

5. **Return Result**
   - Model name (lowercase)
   - Confidence score (0-100)
   - Reasoning (1-2 sentences)

---

## 💡 Integration Notes

### Frontend Integration
The backend is fully integrated with the updated frontend:
- `/analyze-content` endpoint called by `useModelSelection` hook
- `/generate` endpoint receives `model_type` parameter
- Response format matches frontend expectations

### Future Enhancements
- Model caching for repeated content
- Batch content analysis
- Custom model fine-tuning options
- Advanced sentiment analysis
- Multi-language support

---

## 🎉 Summary

### What's New
✅ AI-powered model selection using Gemini API
✅ Support for 6 specialized BERT models
✅ Domain-specific test case generation
✅ Robust fallback mechanisms
✅ Enhanced API documentation
✅ Comprehensive error handling

### Key Achievements
- **Intelligent**: Gemini AI analyzes and recommends models
- **Flexible**: Supports 6 different BERT variants
- **Robust**: Multiple fallback layers
- **Production-Ready**: Comprehensive error handling
- **Well-Documented**: Clear API docs and code comments

### Status
🟢 **READY FOR TESTING**
All backend changes are complete and ready for integration testing with the frontend.
