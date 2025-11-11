# 🧪 Testing Guide - AI-Powered Test Case Pipeline

## Test Suite for Complete System Validation

---

## 🎯 Testing Objectives

1. Verify AI model selection accuracy
2. Validate test case generation for all models
3. Ensure data flow correctness
4. Test error handling
5. Validate UI/UX responsiveness

---

## 📋 Test Cases

### Category 1: Content Analysis & Model Selection

#### Test 1.1: Financial Content → FinBERT
**Input:**
```text
Apple Inc. announced record quarterly revenue of $89.5 billion, 
exceeding Wall Street expectations by 8%. The tech giant's strong 
performance was driven by iPhone sales in China and robust services 
revenue growth of 16% year-over-year. Stock prices surged 5.2% in 
after-hours trading, and analysts raised their price targets.
```

**Expected:**
- Model: `finbert`
- Confidence: 85-95%
- Reasoning: Mentions financial terminology

**Test Steps:**
1. Open frontend
2. Select "Paste Text"
3. Paste content
4. Click "Analyze & Select Model"
5. Wait 5-10 seconds
6. Verify model selection

**Pass Criteria:** ✅ FinBERT selected with high confidence

---

#### Test 1.2: Social Media Content → BERTweet
**Input:**
```text
OMG just got the new iPhone 15 Pro and it's absolutely FIRE! 🔥🔥
The camera is insane! Best phone I've ever had!! #Apple #iPhone15Pro
Everyone needs to get this ASAP 📱✨ @Apple killed it this time! 
#TechTwitter #NewPhone
```

**Expected:**
- Model: `berttweet`
- Confidence: 85-95%
- Reasoning: Contains social media markers

**Pass Criteria:** ✅ BERTweet selected

---

#### Test 1.3: Medical Content → ClinicalBERT
**Input:**
```text
Patient presents with acute onset chest pain radiating to left arm, 
accompanied by diaphoresis and shortness of breath. Vital signs: 
BP 140/90 mmHg, HR 110 bpm, O2 sat 94%. ECG reveals ST-segment 
elevation in leads II, III, and aVF. Troponin levels elevated at 
2.5 ng/mL. Diagnosis: Acute myocardial infarction. Treatment 
initiated with aspirin, heparin, and urgent cardiology consultation.
```

**Expected:**
- Model: `clinicalbert`
- Confidence: 85-95%

**Pass Criteria:** ✅ ClinicalBERT selected

---

#### Test 1.4: Scientific Content → SciBERT
**Input:**
```text
The study examined the correlation between ambient temperature and 
chemical reaction rates using spectroscopic analysis. Results 
demonstrated a significant positive correlation (r=0.89, p<0.001) 
between temperature increase and reaction velocity. The methodology 
employed high-resolution NMR spectroscopy and gas chromatography-mass 
spectrometry to quantify product formation.
```

**Expected:**
- Model: `scibert`
- Confidence: 80-95%

**Pass Criteria:** ✅ SciBERT selected

---

#### Test 1.5: Biomedical Content → BioBERT
**Input:**
```text
The novel protein binding mechanism exhibits enhanced affinity for 
the BRCA1 gene product. Molecular dynamics simulations revealed that 
the mutation at position 1654 significantly alters the tertiary 
structure, affecting downstream signaling pathways. Western blot 
analysis confirmed increased phosphorylation of the substrate, 
suggesting potential therapeutic applications in cancer treatment.
```

**Expected:**
- Model: `biobert`
- Confidence: 85-95%

**Pass Criteria:** ✅ BioBERT selected

---

#### Test 1.6: Patent Content → PatentBERT
**Input:**
```text
A novel apparatus for wireless power transmission comprising a 
resonant inductive coupling system. The invention includes a primary 
coil operating at 6.78 MHz and a secondary coil with impedance 
matching circuitry. Claims: 1. An apparatus for wireless energy 
transfer characterized by improved efficiency through adaptive 
frequency tuning. 2. The apparatus of claim 1, wherein the system 
maintains power transfer efficiency above 85% at distances up to 10cm.
```

**Expected:**
- Model: `patentbert`
- Confidence: 85-95%

**Pass Criteria:** ✅ PatentBERT selected

---

### Category 2: CSV Upload

#### Test 2.1: Valid CSV Upload
**Create file: test_financial.csv**
```csv
review,sentiment
"Apple's earnings exceeded expectations",positive
"Stock prices declined sharply today",negative
"The company held its annual meeting",neutral
"Revenue growth accelerated this quarter",positive
```

**Test Steps:**
1. Select "Upload CSV File"
2. Drag/drop or select file
3. Verify file preview shows
4. Click "Analyze & Select Model"
5. Verify model selection

**Pass Criteria:** ✅ File uploaded, FinBERT selected

---

#### Test 2.2: Invalid File Type
**Test Steps:**
1. Try uploading .txt or .json file
2. Verify error message

**Expected Error:** "Please select a CSV file"

**Pass Criteria:** ✅ Error shown, file rejected

---

### Category 3: Test Case Generation

#### Test 3.1: Generate 10 FinBERT Test Cases
**Test Steps:**
1. Complete analysis (FinBERT selected)
2. Enter "10" in test case count
3. Click "Generate Test Cases"
4. Wait for generation
5. Verify CSV table displays

**Pass Criteria:**
- ✅ 10 rows in table
- ✅ Columns: "review", "sentiment"
- ✅ Content is financial-themed
- ✅ Sentiments: mix of positive/negative/neutral

---

#### Test 3.2: Generate 100 Test Cases
**Test Steps:**
1. Enter "100" in count
2. Generate
3. Wait (50-150 seconds)

**Pass Criteria:**
- ✅ 100 test cases generated
- ✅ Progress shown during generation
- ✅ All cases have review and sentiment

---

#### Test 3.3: Generate for Each Model Type
**Test Steps:**
For each model (finbert, berttweet, patentbert, clinicalbert, biobert, scibert):
1. Use appropriate content
2. Analyze and select model
3. Generate 10 test cases
4. Verify content matches domain

**Pass Criteria:**
- ✅ FinBERT: Financial terms (earnings, stock, revenue)
- ✅ BERTweet: Social media style (hashtags, emojis, @mentions)
- ✅ PatentBERT: Legal/technical language
- ✅ ClinicalBERT: Medical terminology
- ✅ BioBERT: Biological/molecular terms
- ✅ SciBERT: Research/scientific language

---

### Category 4: CSV Download

#### Test 4.1: Download Generated Data
**Test Steps:**
1. Generate test cases
2. Click "Download CSV"
3. Open downloaded file

**Pass Criteria:**
- ✅ File downloads successfully
- ✅ Filename format: `[model]_test_cases.csv`
- ✅ Valid CSV structure
- ✅ All data present

---

### Category 5: Verification

#### Test 5.1: Verify Generated Test Cases
**Test Steps:**
1. Generate test cases
2. Click "Verify with FinBERT"
3. Wait for verification
4. Check metrics

**Pass Criteria:**
- ✅ Accuracy between 60-100%
- ✅ Precision shown
- ✅ Recall shown
- ✅ F1 score shown
- ✅ Correct predictions count
- ✅ Total predictions count

---

### Category 6: Error Handling

#### Test 6.1: Empty Text Input
**Test Steps:**
1. Select "Paste Text"
2. Leave textarea empty
3. Click "Analyze & Select Model"

**Expected:** Alert or error message

**Pass Criteria:** ✅ Cannot proceed without content

---

#### Test 6.2: Invalid Test Case Count
**Test Steps:**
1. Complete analysis
2. Enter "0" or "1001" in count
3. Try to generate

**Expected:** Validation error

**Pass Criteria:** ✅ Button disabled or error shown

---

#### Test 6.3: Network Error Simulation
**Test Steps:**
1. Stop backend server
2. Try to analyze content

**Expected:** Clear error message

**Pass Criteria:** ✅ "Service unavailable" or similar error

---

#### Test 6.4: Gemini API Error
**Test Steps:**
1. Use invalid/expired API key
2. Try to analyze

**Expected:** Error message about API

**Pass Criteria:** ✅ Graceful fallback or clear error

---

### Category 7: UI/UX

#### Test 7.1: Responsive Design
**Test Steps:**
1. Resize browser window
2. Test on mobile viewport
3. Check all components

**Pass Criteria:**
- ✅ Layout adapts to screen size
- ✅ All elements accessible
- ✅ No horizontal scroll

---

#### Test 7.2: Loading States
**Test Steps:**
1. Trigger analysis
2. Check for loading indicator
3. Trigger generation
4. Check for loading indicator

**Pass Criteria:**
- ✅ "Analyzing Content..." shown
- ✅ "Generating..." shown
- ✅ Buttons disabled during operations

---

#### Test 7.3: Progressive Disclosure
**Test Steps:**
1. Load app
2. Verify only DataInputSection visible
3. Complete analysis
4. Verify ModelSelectionSection appears
5. Verify ConfigurationSection appears
6. Generate data
7. Verify GeneratedDataSection appears

**Pass Criteria:** ✅ Sections appear in order

---

### Category 8: Integration Tests

#### Test 8.1: Complete Happy Path
**Test Steps:**
1. Open app
2. Paste financial text
3. Analyze (FinBERT selected)
4. Enter 20 test cases
5. Generate
6. Download CSV
7. Verify
8. Check metrics

**Pass Criteria:** ✅ All steps complete without errors

---

#### Test 8.2: Multiple Model Types in Session
**Test Steps:**
1. Analyze financial content → Generate
2. Clear and analyze social media → Generate
3. Clear and analyze medical → Generate

**Pass Criteria:** ✅ Each generates appropriate content

---

### Category 9: Backend API Tests

#### Test 9.1: Root Endpoint
```bash
curl http://127.0.0.1:8000/
```

**Expected:**
```json
{
  "message": "Welcome to...",
  "version": "2.0.0",
  "supported_models": [...]
}
```

**Pass Criteria:** ✅ 200 OK with correct JSON

---

#### Test 9.2: Analyze Endpoint
```bash
curl -X POST http://127.0.0.1:8000/analyze-content \
  -H "Content-Type: application/json" \
  -d '{"content": "Apple stock surged today", "input_type": "text"}'
```

**Expected:**
```json
{
  "model": "finbert",
  "confidence": 90,
  "reasoning": "..."
}
```

**Pass Criteria:** ✅ 200 OK with model recommendation

---

#### Test 9.3: Generate Endpoint
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_test_cases": 5, "model_type": "finbert"}' \
  --output test.csv
```

**Pass Criteria:**
- ✅ 200 OK
- ✅ CSV file downloaded
- ✅ 5 rows + header

---

#### Test 9.4: Invalid Model Type
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_test_cases": 5, "model_type": "invalid_model"}'
```

**Expected:** 400 Bad Request with error message

**Pass Criteria:** ✅ Validation error returned

---

## 🎯 Test Execution Checklist

### Pre-Testing Setup
- [ ] Backend running on port 8000
- [ ] Frontend running on port 5173
- [ ] .env file with valid GEMINI_API_KEY
- [ ] All dependencies installed
- [ ] Browser console open (F12)

### Automated Testing (Future)
- [ ] Unit tests for hooks
- [ ] Component tests with React Testing Library
- [ ] API integration tests
- [ ] E2E tests with Playwright/Cypress

### Manual Testing
- [ ] All Category 1 tests (Model Selection)
- [ ] All Category 2 tests (CSV Upload)
- [ ] All Category 3 tests (Generation)
- [ ] All Category 4 tests (Download)
- [ ] All Category 5 tests (Verification)
- [ ] All Category 6 tests (Error Handling)
- [ ] All Category 7 tests (UI/UX)
- [ ] All Category 8 tests (Integration)
- [ ] All Category 9 tests (Backend API)

---

## 📊 Test Results Template

### Test Execution Summary

**Date:** ___________
**Tester:** ___________
**Environment:** Development

| Category | Total | Passed | Failed | Blocked | Notes |
|----------|-------|--------|--------|---------|-------|
| Model Selection | 6 | ___ | ___ | ___ | |
| CSV Upload | 2 | ___ | ___ | ___ | |
| Generation | 3 | ___ | ___ | ___ | |
| Download | 1 | ___ | ___ | ___ | |
| Verification | 1 | ___ | ___ | ___ | |
| Error Handling | 4 | ___ | ___ | ___ | |
| UI/UX | 3 | ___ | ___ | ___ | |
| Integration | 2 | ___ | ___ | ___ | |
| Backend API | 4 | ___ | ___ | ___ | |
| **TOTAL** | **26** | ___ | ___ | ___ | |

---

## 🐛 Bug Report Template

**Bug ID:** ___
**Severity:** Critical / High / Medium / Low
**Test Case:** ___
**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Result:** ___
**Actual Result:** ___
**Screenshots:** ___
**Console Errors:** ___

---

## ✅ Acceptance Criteria

### Must Pass (Critical)
- ✅ All 6 model types correctly identified
- ✅ Test case generation works for all models
- ✅ CSV download successful
- ✅ Verification returns metrics
- ✅ No console errors during normal flow

### Should Pass (High Priority)
- ✅ Error handling graceful
- ✅ Loading states shown
- ✅ UI responsive
- ✅ API errors handled

### Nice to Have (Medium Priority)
- ✅ Fast response times (<10s for analysis)
- ✅ Smooth animations
- ✅ Helpful error messages

---

## 🎉 Test Completion

Once all tests pass:
1. ✅ Document any issues found
2. ✅ Create bug tickets for failures
3. ✅ Update documentation
4. ✅ Mark system as ready for UAT
5. ✅ Prepare for production deployment

**Status:** Ready for comprehensive testing!
