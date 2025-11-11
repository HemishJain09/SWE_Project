# 🚀 Quick Start Guide - AI-Powered Test Case Pipeline

## Complete System Setup and Testing

---

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Google Gemini API Key

---

## ⚙️ Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend directory
cd Generation_Verification

# Install Python dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

**Verify Backend:**
```bash
# In a new terminal
curl http://127.0.0.1:8000/

# You should see API info with supported models
```

---

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

**Access Frontend:**
Open browser to `http://localhost:5173` (or the port shown in terminal)

---

## 🧪 Testing the Complete Flow

### Test 1: Financial Content Analysis

1. **Open Frontend** in browser
2. **Select "Paste Text"**
3. **Enter this financial text:**
   ```
   Apple Inc. reported record quarterly revenue of $89.5 billion, 
   beating analyst expectations. The iPhone maker saw strong growth 
   in services revenue, which increased 16% year-over-year. Stock 
   prices surged 5% in after-hours trading following the earnings 
   announcement.
   ```
4. **Click "Analyze & Select Model"**
5. **Expected Result:** FinBERT selected with high confidence
6. **Enter number of test cases:** 20
7. **Click "Generate Test Cases"**
8. **Verify:** Should see 20 financial-themed test cases
9. **Click "Download CSV"** to save
10. **Click "Verify with FinBERT"** to check accuracy

---

### Test 2: Social Media Content Analysis

1. **Clear previous data** (refresh page)
2. **Select "Paste Text"**
3. **Enter this social media text:**
   ```
   OMG just got the new iPhone 15 Pro and it's incredible! 🔥
   The camera quality is insane. Best phone ever! #Apple #iPhone15Pro
   Everyone needs to get this ASAP! 📱✨
   Totally worth the upgrade from my old phone 💯
   ```
4. **Click "Analyze & Select Model"**
5. **Expected Result:** BERTweet selected
6. **Enter number of test cases:** 15
7. **Generate and verify**

---

### Test 3: Medical Content Analysis

1. **Select "Paste Text"**
2. **Enter this clinical text:**
   ```
   Patient presents with acute onset chest pain, radiating to left arm.
   Blood pressure 140/90 mmHg, heart rate 110 bpm. ECG shows ST elevation.
   Troponin levels elevated at 2.5 ng/mL. Diagnosis: Acute myocardial infarction.
   Treatment initiated with aspirin, heparin, and urgent cardiology consultation.
   ```
3. **Expected Result:** ClinicalBERT selected
4. **Generate medical test cases**

---

### Test 4: CSV Upload

1. **Select "Upload CSV File"**
2. **Create a test CSV file:**
   ```csv
   review,sentiment
   "The drug trial showed promising results",positive
   "Patient experienced severe side effects",negative
   "Standard blood pressure measurement taken",neutral
   ```
3. **Drag and drop or click to upload**
4. **Analyze and generate**

---

## 🔍 API Testing with cURL

### Test Content Analysis
```bash
curl -X POST http://127.0.0.1:8000/analyze-content \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Bitcoin price reached new all-time high today",
    "input_type": "text"
  }'
```

### Test Generation for Different Models
```bash
# Generate FinBERT test cases
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_test_cases": 10, "model_type": "finbert"}' \
  --output finbert_cases.csv

# Generate BERTweet test cases
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_test_cases": 10, "model_type": "berttweet"}' \
  --output berttweet_cases.csv
```

---

## 🐛 Troubleshooting

### Backend Issues

**Problem:** `GEMINI_API_KEY not found`
- **Solution:** Ensure `.env` file exists in `Generation_Verification/` directory
- **Check:** `cat .env` should show your API key

**Problem:** Port 8000 already in use
- **Solution:** Use different port: `uvicorn main:app --reload --port 8001`
- **Update:** Frontend API_BASE_URL in `useTestCaseGeneration.js`

**Problem:** Module not found errors
- **Solution:** `pip install -r requirements.txt`

---

### Frontend Issues

**Problem:** Cannot connect to backend
- **Check:** Backend is running on port 8000
- **Check:** No CORS errors in browser console
- **Solution:** Restart both frontend and backend

**Problem:** Analysis takes too long
- **Reason:** Gemini API may be rate-limited
- **Wait:** 10-30 seconds for response
- **Check:** Backend terminal for progress logs

**Problem:** Model not being selected
- **Check:** Content has sufficient length (>10 chars)
- **Check:** Backend logs for error messages
- **Try:** Different content or shorter text

---

## 📊 Expected Behavior

### Content Analysis (Fast)
- Analysis time: 3-10 seconds
- Confidence: Usually 70-95%
- Reasoning: Clear, specific explanation

### Test Case Generation (Slower)
- Time per batch (10 cases): 5-15 seconds
- Total for 100 cases: 50-150 seconds
- Progress bar shows status

### Verification
- Time: 5-20 seconds depending on cases
- Returns: Accuracy, precision, recall, F1

---

## 🎯 Success Criteria

✅ **Backend Running:** API docs accessible at http://127.0.0.1:8000/docs
✅ **Frontend Running:** UI loads without errors
✅ **Analysis Works:** Model selected based on content
✅ **Generation Works:** CSV with correct format downloaded
✅ **Different Models:** Each model generates appropriate content
✅ **Verification Works:** Metrics displayed correctly

---

## 📈 Performance Tips

### For Faster Testing
- Use smaller batch sizes (10-20 test cases)
- Keep content under 1000 characters
- Use text input instead of large CSV files

### For Production
- Increase batch size to 50
- Add caching layer for repeated content
- Consider rate limiting on frontend

---

## 🔒 Security Notes

- ✅ API key stored in `.env` (not committed to git)
- ✅ CORS enabled for development
- ⚠️ For production: Restrict CORS origins
- ⚠️ For production: Add authentication
- ⚠️ For production: Rate limiting recommended

---

## 📚 Additional Resources

### Documentation
- **Frontend:** See `frontend/DATAFLOW_CHANGES.md`
- **Backend:** See `Generation_Verification/BACKEND_IMPLEMENTATION.md`
- **API Docs:** http://127.0.0.1:8000/docs (when running)

### Logs Location
- **Backend:** Terminal where uvicorn is running
- **Frontend:** Browser console (F12)

### Common Commands
```bash
# Stop servers
Ctrl + C

# Restart backend
uvicorn main:app --reload --port 8000

# Restart frontend
npm run dev

# Check Python packages
pip list | grep -E "fastapi|generative|pandas"

# Check Node packages
npm list axios react
```

---

## 🎉 Demo Script

### 5-Minute Demo

1. **Start servers** (2 min setup)
   ```bash
   # Terminal 1: Backend
   cd Generation_Verification && uvicorn main:app --reload

   # Terminal 2: Frontend
   cd frontend && npm run dev
   ```

2. **Demo financial analysis** (1 min)
   - Paste financial text
   - Show FinBERT selection
   - Generate 10 cases

3. **Demo social media** (1 min)
   - Paste tweet-like text
   - Show BERTweet selection
   - Show different content style

4. **Show verification** (1 min)
   - Verify generated cases
   - Show accuracy metrics
   - Download CSV

---

## ✨ What Makes This Special

1. **AI-Powered:** Gemini automatically selects the right model
2. **Multi-Domain:** Works for finance, social media, medical, scientific, etc.
3. **End-to-End:** Complete pipeline from input to verified results
4. **User-Friendly:** Intuitive UI with clear feedback
5. **Extensible:** Easy to add new models or features

---

## 🎊 You're Ready!

The system is now fully configured and ready to use. Try different types of content and see how the AI intelligently selects the appropriate model for each domain!

**Questions or issues?** Check the detailed documentation in:
- `frontend/DATAFLOW_CHANGES.md`
- `Generation_Verification/BACKEND_IMPLEMENTATION.md`
