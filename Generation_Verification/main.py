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

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FinBERT Test Case API",
    description="An API to generate and verify financial sentiment test cases.",
    version="1.0.0"
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

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the FinBERT Test Case Generation and Verification API"}

@app.post("/generate", tags=["Test Cases"])
async def generate_test_cases(request: GenerationRequest):
    """
    Generates synthetic financial test cases using a generative AI model.
    """
    print(f"Received request to generate {request.num_test_cases} test cases.")
    try:
        # The API key is now loaded automatically from the .env file within the run_generation function.
        # Call the refactored generation logic without the api_key argument.
        generated_data = run_generation(total_samples=request.num_test_cases, batch_size=10)

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
        response.headers["Content-Disposition"] = "attachment; filename=generated_test_cases.csv"
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



