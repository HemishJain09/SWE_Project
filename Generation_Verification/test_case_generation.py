import os
import json
import time
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

def get_llm_response(prompt, api_key, retries=5, delay=5):
    """
    Calls the Gemini API to get a response for a given prompt with retry logic.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro') # Using a faster model for API responsiveness
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
            else:
                print(f"Warning: Received an empty response on attempt {attempt + 1}")
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Error: Max retries reached. Could not get response from LLM.")
                return None
    return None

def construct_prompt(batch_size, existing_count):
    """
    Constructs the prompt for the LLM to generate test cases.
    """
    return f"""
    You are a test data generator for a machine learning model.
    Your task is to generate synthetic data for a financial sentiment analysis classification task.
    The data schema is: "review" (string) and "sentiment" (string: "positive", "negative", or "neutral").

    Please generate {batch_size} new, unique, and diverse financial statements.

    **Instructions:**
    1.  **Balance:** Ensure a roughly equal distribution of "positive", "negative", and "neutral" sentiments.
    2.  **Financial Focus:** Generate statements about company earnings, revenue, stock market movements, economic indicators, and corporate announcements.
    3.  **Realistic Content:** The statements should resemble real financial news or analyst reports.
    4.  **Output Format:** Provide the output as a valid JSON array of objects. Each object must have a "review" key and a "sentiment" key. Do not include markdown formatting like ```json.

    **Example Output Format:**
    [
        {{"review": "Apple's quarterly earnings exceeded Wall Street expectations, driven by strong iPhone sales in Asia.", "sentiment": "positive"}},
        {{"review": "The company reported a significant loss in Q3, with revenue declining by 15% year-over-year.", "sentiment": "negative"}},
        {{"review": "The Federal Reserve raised interest rates by 75 basis points to combat inflation.", "sentiment": "neutral"}}
    ]

    We have already generated {existing_count} statements. Ensure the new statements are distinct.
    Generate exactly {batch_size} financial statements now.
    """

def parse_llm_output(output_text):
    """
    Parses the JSON output from the LLM into a list of dictionaries.
    """
    try:
        clean_text = output_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()

        parsed_json = json.loads(clean_text)
        
        if not isinstance(parsed_json, list):
            print("Error: Parsed data is not a list.")
            return []
        
        # Filter out malformed items
        valid_items = [item for item in parsed_json if "review" in item and "sentiment" in item]
        return valid_items
        
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing LLM output: {e}")
        print(f"--- Received Text ---\n{output_text}\n---------------------")
        return []

def run_generation(total_samples, batch_size):
    """
    Main function to run the test case generation workflow, adapted for API calls.
    It loads the API key from a .env file.
    Args:
        total_samples (int): Total number of test cases to generate.
        batch_size (int): Number of test cases to generate per API call.

    Returns:
        list: A list of dictionaries, where each dictionary is a test case.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")

    all_test_cases = []
    num_generated = 0
    
    with tqdm(total=total_samples, desc="Generating Test Cases via API") as pbar:
        while num_generated < total_samples:
            current_batch_size = min(batch_size, total_samples - num_generated)
            if current_batch_size <= 0:
                break
            
            prompt = construct_prompt(current_batch_size, num_generated)
            llm_response = get_llm_response(prompt, api_key)

            if llm_response:
                parsed_data = parse_llm_output(llm_response)
                if parsed_data:
                    # Ensure we don't add more samples than requested
                    num_to_add = min(len(parsed_data), total_samples - num_generated)
                    all_test_cases.extend(parsed_data[:num_to_add])
                    
                    added_count = len(parsed_data[:num_to_add])
                    num_generated += added_count
                    pbar.update(added_count)
    
    return all_test_cases

