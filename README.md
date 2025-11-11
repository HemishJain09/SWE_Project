# 🤖 AI-Powered Test Case Generation Pipeline

> Intelligent test case generation using Google Gemini AI and specialized BERT models

[![Status](https://img.shields.io/badge/status-ready-brightgreen)]()
[![Version](https://img.shields.io/badge/version-2.0.0-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🎯 Overview

An end-to-end full-stack application that leverages **Google Gemini AI** to automatically analyze content and select the most appropriate BERT model for test case generation. Supports **6 specialized domains** including financial, social media, medical, scientific, biomedical, and patent classification.

### ✨ Key Features

- 🧠 **AI-Powered Model Selection** - Gemini AI automatically selects the best BERT model
- 📊 **Multi-Domain Support** - 6 specialized BERT models for different domains
- 🎨 **Beautiful UI** - Modern, responsive interface with TailwindCSS
- 📈 **Accuracy Verification** - Built-in verification with detailed metrics
- 💾 **Export Functionality** - Download generated test cases as CSV
- 🔄 **Complete Pipeline** - Input → Analysis → Generation → Verification

---

## 🏗️ Architecture

```
Frontend (React + Vite) ↔ Backend (FastAPI + Python) ↔ Gemini AI
```

**Tech Stack:**
- **Frontend:** React 19, TailwindCSS 4, Axios, Vite
- **Backend:** FastAPI, Google Generative AI, Pandas, Transformers
- **AI:** Google Gemini 2.5 Pro, FinBERT, BERTweet, PatentBERT, ClinicalBERT, BioBERT, SciBERT

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### 1. Clone Repository

```bash
git clone https://github.com/HemishJain09/SWE_Project.git
cd SWE_Project
```

### 2. Backend Setup

```bash
cd Generation_Verification

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Start server
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Open Application

Navigate to `http://localhost:5173` in your browser.

---

## 📖 Usage Guide

### Step 1: Input Your Data
- **Option A:** Paste text content directly
- **Option B:** Upload a CSV file with existing data

### Step 2: AI Analysis
Click "Analyze & Select Model" and wait for Gemini AI to:
- Analyze your content
- Select the most appropriate BERT model
- Provide confidence score and reasoning

### Step 3: Configure Generation
- Review the AI-selected model
- Enter number of test cases (1-500)
- Click "Generate Test Cases"

### Step 4: Download & Verify
- View generated test cases in table
- Download as CSV
- Verify accuracy with FinBERT

---

## 🎯 Supported Models

| Model | Domain | Best For |
|-------|--------|----------|
| **FinBERT** | Finance | Earnings reports, stock analysis, financial news |
| **BERTweet** | Social Media | Tweets, posts, social media discussions |
| **PatentBERT** | Legal/Patents | Patent claims, legal documents, IP text |
| **ClinicalBERT** | Medical | Patient notes, diagnoses, medical reports |
| **BioBERT** | Biomedical | Genetics, molecular biology, proteins |
| **SciBERT** | Scientific | Research papers, academic studies |

---

## 📚 Documentation

Comprehensive documentation is available in the following files:

- **[QUICKSTART.md](QUICKSTART.md)** - Complete setup and testing guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Full implementation details
- **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - Visual architecture diagrams
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive test cases
- **[frontend/DATAFLOW_CHANGES.md](frontend/DATAFLOW_CHANGES.md)** - Frontend architecture
- **[Generation_Verification/BACKEND_IMPLEMENTATION.md](Generation_Verification/BACKEND_IMPLEMENTATION.md)** - Backend details

---

## 🔧 API Endpoints

### `POST /analyze-content`
Analyzes content and recommends the best BERT model.

```json
Request:
{
  "content": "Apple's earnings exceeded expectations",
  "input_type": "text"
}

Response:
{
  "model": "finbert",
  "confidence": 95,
  "reasoning": "Content contains financial terminology..."
}
```

### `POST /generate`
Generates domain-specific test cases.

```json
Request:
{
  "num_test_cases": 100,
  "model_type": "finbert"
}

Response: CSV file
```

### `POST /verify`
Verifies test cases and returns accuracy metrics.

```
Request: CSV file (multipart/form-data)

Response:
{
  "accuracy": 85.5,
  "precision": 87.2,
  "recall": 84.1,
  "f1_score": 85.6
}
```

### `GET /`
Returns API information and supported models.

---

## 🧪 Testing

### Run Manual Tests

Follow the comprehensive testing guide in [TESTING_GUIDE.md](TESTING_GUIDE.md).

### Example Test Cases

**Financial Content:**
```
Apple Inc. reported record quarterly revenue of $89.5 billion, 
beating analyst expectations...
```
Expected: FinBERT selected

**Social Media Content:**
```
OMG just got the new iPhone 15 and it's amazing! 🔥 #Apple
```
Expected: BERTweet selected

---

## 🎨 Screenshots

### Data Input Section
Choose between text input or CSV upload with visual feedback.

### Model Selection
AI analyzes content and displays recommended model with confidence.

### Generated Test Cases
View, download, and verify generated test cases.

### Verification Report
Comprehensive accuracy metrics with beautiful visualizations.

---

## 🔐 Environment Variables

Create a `.env` file in both frontend and backend directories:

```bash
# Generation_Verification/.env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 📦 Project Structure

```
SWE_Project/
├── frontend/                      # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── hooks/                # Custom hooks
│   │   └── assets/               # Static assets
│   ├── package.json
│   └── vite.config.js
│
├── Generation_Verification/      # Python backend
│   ├── main.py                   # FastAPI app
│   ├── gemini_analyzer.py        # AI model selection
│   ├── test_case_generation.py   # Test case generator
│   ├── finbert_verification.py   # Verification logic
│   ├── requirements.txt
│   └── .env
│
└── Documentation files            # Comprehensive docs
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Hemish Jain** - [HemishJain09](https://github.com/HemishJain09)

---

## 🙏 Acknowledgments

- Google Gemini AI for intelligent content analysis
- Hugging Face for BERT model implementations
- FastAPI team for the excellent web framework
- React team for the UI library

---

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the comprehensive documentation
- Review the testing guide

---

## 🎉 Status

✅ **Version 2.0.0 - Production Ready**

All features implemented and tested. Ready for deployment and user testing.

---

## 🔄 Changelog

### Version 2.0.0 (November 2025)
- ✨ Added AI-powered model selection using Gemini
- ✨ Support for 6 specialized BERT models
- ✨ New data input methods (text and CSV)
- ✨ Enhanced UI with progressive disclosure
- ✨ Comprehensive documentation
- 🐛 Improved error handling
- 🎨 Modern responsive design

### Version 1.0.0
- Basic test case generation with FinBERT
- Manual model selection
- CSV verification

---

## 📈 Roadmap

### Future Enhancements
- [ ] User authentication and saved sessions
- [ ] Model performance comparison
- [ ] Custom model fine-tuning
- [ ] Batch processing multiple files
- [ ] Advanced analytics dashboard
- [ ] Export to multiple formats (JSON, Excel)
- [ ] Multi-language support
- [ ] Model-specific verification (not just FinBERT)

---

## 💡 Use Cases

1. **QA Testing** - Generate diverse test cases for NLP models
2. **Data Augmentation** - Expand training datasets
3. **Model Evaluation** - Test model performance across domains
4. **Research** - Study cross-domain classification
5. **Education** - Learn about different BERT variants

---

## ⚡ Performance

- **Analysis Time:** 3-10 seconds
- **Generation (10 cases):** 5-15 seconds
- **Generation (100 cases):** 50-150 seconds
- **Verification:** 5-20 seconds

*Times may vary based on network and API rate limits*

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

<div align="center">

**Made with ❤️ using React, FastAPI, and Google Gemini AI**

[Report Bug](https://github.com/HemishJain09/SWE_Project/issues) · 
[Request Feature](https://github.com/HemishJain09/SWE_Project/issues) · 
[Documentation](./QUICKSTART.md)

</div>
