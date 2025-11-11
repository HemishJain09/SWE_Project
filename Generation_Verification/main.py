import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

# Import the refactored logic from your scripts
from test_case_generation import run_generation
from finbert_verification import run_verification
from gemini_analyzer import analyze_content_for_model

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI-Powered Test Case Generation API",
    description="An API to intelligently select models, generate, and verify test cases using specialized BERT models.",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Bodies ---
class GenerationRequest(BaseModel):
    num_test_cases: int = Field(..., gt=0, le=500, description="The number of test cases to generate (must be between 1 and 500).")
    model_type: str = Field(default="finbert", description="Model type: finbert, berttweet, patentbert, clinicalbert, biobert, or scibert.")

class ContentAnalysisRequest(BaseModel):
    content: str = Field(..., min_length=10, description="Text or CSV content to analyze for model selection.")
    input_type: str = Field(default="text", description="Type of input: 'text' or 'csv'.")

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "Welcome to the AI-Powered Test Case Generation and Verification API",
        "version": "2.0.0",
        "features": [
            "Intelligent model selection using Gemini AI",
            "Support for 6 specialized BERT models",
            "Automated test case generation",
            "Model verification and accuracy reporting"
        ],
        "supported_models": [
            "finbert", "berttweet", "patentbert", 
            "clinicalbert", "biobert", "scibert"
        ]
    }

@app.post("/analyze-content", tags=["Model Selection"])
async def analyze_content(request: ContentAnalysisRequest):
    """
    Analyzes content using Gemini AI and recommends the most suitable BERT model.
    
    Supported models:
    - finbert: Financial text and documents
    - berttweet: Twitter and social media content
    - patentbert: Patent documents and legal text
    - clinicalbert: Medical and clinical records
    - biobert: Biomedical and biological research
    - scibert: Scientific papers and research
    
    Returns:
        - model: Recommended model name
        - confidence: Confidence score (0-100)
        - reasoning: Explanation for the selection
    """
    print(f"Analyzing content of type '{request.input_type}' ({len(request.content)} chars)")
    
    try:
        # Analyze content using Gemini
        result = analyze_content_for_model(request.content, request.input_type)
        
        print(f"Analysis result: {result['model']} (confidence: {result['confidence']}%)")
        
        return {
            "model": result['model'],
            "confidence": result['confidence'],
            "reasoning": result['reasoning']
        }
        
    except ValueError as ve:
        print(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during content analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Content analysis failed: {str(e)}")

@app.post("/generate", tags=["Test Cases"])
async def generate_test_cases(request: GenerationRequest):
    """
    Generates synthetic test cases using a generative AI model for the specified BERT model type.
    
    Supported models:
    - finbert: Financial sentiment analysis
    - berttweet: Social media sentiment
    - patentbert: Patent document classification
    - clinicalbert: Clinical note classification
    - biobert: Biomedical research classification
    - scibert: Scientific paper classification
    """
    print(f"Received request to generate {request.num_test_cases} test cases for model: {request.model_type}")
    
    # Validate model type
    valid_models = ['finbert', 'berttweet', 'patentbert', 'clinicalbert', 'biobert', 'scibert']
    if request.model_type not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Must be one of: {', '.join(valid_models)}"
        )
    
    try:
        # Call the refactored generation logic with model type
        generated_data = run_generation(
            total_samples=request.num_test_cases, 
            batch_size=10,
            model_type=request.model_type
        )

        if not generated_data:
            raise HTTPException(status_code=500, detail="Failed to generate test cases from the language model.")

        # Convert the list of dictionaries to a pandas DataFrame and then to a CSV string
        df = pd.DataFrame(generated_data)
        
        # Ensure 'review' and 'sentiment' are the only columns
        if 'review' not in df.columns or 'sentiment' not in df.columns:
             raise HTTPException(status_code=500, detail="Generated data is missing required 'review' or 'sentiment' columns.")
        
        df = df[['review', 'sentiment']]

        stream = io.StringIO()
        df.to_csv(stream, index=False)
        
        # Return the CSV content as a streaming response
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename={request.model_type}_test_cases.csv"
        return response

    except Exception as e:
        print(f"Error during test case generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify", tags=["Verification"])
async def verify_test_cases(file: UploadFile = File(...)):
    """
    Verifies the provided test cases (in CSV format) using the FinBERT model.
    """
    print(f"Received file '{file.filename}' for verification.")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        # Read the uploaded CSV file directly into a pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Basic validation of the CSV structure
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'review' and 'sentiment' columns.")

        # Call the refactored verification logic
        report = run_verification(df)
        
        if not report:
             raise HTTPException(status_code=500, detail="Failed to generate verification report.")

        return report

    except Exception as e:
        print(f"Error during verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))



