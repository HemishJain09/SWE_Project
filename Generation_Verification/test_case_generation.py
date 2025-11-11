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
    model = genai.GenerativeModel('gemini-2.0-flash-exp') # Using flash model for faster response and lower quota usage
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

def construct_prompt(batch_size, existing_count, model_type='finbert'):
    """
    Constructs the prompt for the LLM to generate test cases based on model type.
    
    Args:
        batch_size: Number of test cases to generate
        existing_count: Number of already generated test cases
        model_type: Type of BERT model (finbert, berttweet, patentbert, clinicalbert, biobert, scibert)
    """
    
    # Model-specific configurations
    model_configs = {
        'finbert': {
            'domain': 'financial sentiment analysis',
            'content_type': 'financial statements, earnings reports, market news, and economic indicators',
            'examples': [
                {"review": "Apple's quarterly earnings exceeded Wall Street expectations, driven by strong iPhone sales in Asia.", "sentiment": "positive"},
                {"review": "The company reported a significant loss in Q3, with revenue declining by 15% year-over-year.", "sentiment": "negative"},
                {"review": "The Federal Reserve raised interest rates by 75 basis points to combat inflation.", "sentiment": "neutral"}
            ]
        },
        'berttweet': {
            'domain': 'social media sentiment analysis',
            'content_type': 'tweets, social media posts, hashtags, and online discussions',
            'examples': [
                {"review": "Just tried the new iPhone 15 and it's absolutely amazing! 🔥 #Apple #iPhone15", "sentiment": "positive"},
                {"review": "Disappointed with the customer service today. Been waiting for 2 hours and still no response 😤", "sentiment": "negative"},
                {"review": "The weather forecast says it might rain tomorrow. Don't forget your umbrella! ☔", "sentiment": "neutral"}
            ]
        },
        'patentbert': {
            'domain': 'patent document classification',
            'content_type': 'patent claims, inventions, intellectual property descriptions, and legal text',
            'examples': [
                {"review": "A novel apparatus for wireless power transmission comprising a resonant inductive coupling system with improved efficiency.", "sentiment": "positive"},
                {"review": "The prior art clearly discloses the same method of data encryption, rendering this claim obvious.", "sentiment": "negative"},
                {"review": "The invention relates to a method for processing semiconductor wafers using photolithography techniques.", "sentiment": "neutral"}
            ]
        },
        'clinicalbert': {
            'domain': 'clinical note classification',
            'content_type': 'patient records, diagnoses, treatment plans, and medical observations',
            'examples': [
                {"review": "Patient shows significant improvement in mobility after physical therapy. No adverse reactions to medication.", "sentiment": "positive"},
                {"review": "Patient presents with severe abdominal pain and elevated white blood cell count. Suspected acute appendicitis.", "sentiment": "negative"},
                {"review": "Patient attended follow-up appointment. Blood pressure measured at 120/80 mmHg. Current medications reviewed.", "sentiment": "neutral"}
            ]
        },
        'biobert': {
            'domain': 'biomedical research classification',
            'content_type': 'gene sequences, protein interactions, molecular biology findings, and biological research',
            'examples': [
                {"review": "The novel protein binding mechanism shows enhanced affinity, suggesting potential therapeutic applications.", "sentiment": "positive"},
                {"review": "Mutation in the BRCA1 gene significantly increases susceptibility to breast cancer development.", "sentiment": "negative"},
                {"review": "The study examined the expression patterns of cytokine genes in immune cells during inflammatory response.", "sentiment": "neutral"}
            ]
        },
        'scibert': {
            'domain': 'scientific paper classification',
            'content_type': 'research findings, experimental results, hypotheses, and academic studies',
            'examples': [
                {"review": "The experimental results demonstrate a significant correlation between variables, supporting our hypothesis (p < 0.05).", "sentiment": "positive"},
                {"review": "The proposed methodology failed to replicate previous findings, suggesting fundamental flaws in the approach.", "sentiment": "negative"},
                {"review": "This study investigates the relationship between temperature and reaction rate using spectroscopic analysis.", "sentiment": "neutral"}
            ]
        }
    }
    
    config = model_configs.get(model_type, model_configs['finbert'])
    
    examples_str = "\n        ".join([
        f'{{"review": "{ex["review"]}", "sentiment": "{ex["sentiment"]}"}},' 
        for ex in config['examples']
    ])
    
    return f"""
    You are a test data generator for a machine learning model.
    Your task is to generate synthetic data for a {config['domain']} classification task.
    The data schema is: "review" (string) and "sentiment" (string: "positive", "negative", or "neutral").

    Please generate {batch_size} new, unique, and diverse examples related to: {config['content_type']}.

    **Instructions:**
    1.  **Balance:** Ensure a roughly equal distribution of "positive", "negative", and "neutral" sentiments.
    2.  **Domain Focus:** Generate content specifically related to {config['content_type']}.
    3.  **Realistic Content:** The statements should be authentic and representative of the domain.
    4.  **Output Format:** Provide the output as a valid JSON array of objects. Each object must have a "review" key and a "sentiment" key. Do not include markdown formatting like ```json.

    **Example Output Format:**
    [
        {examples_str}
    ]

    We have already generated {existing_count} statements. Ensure the new statements are distinct and diverse.
    Generate exactly {batch_size} statements now for {config['domain']}.
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

def run_generation(total_samples, batch_size, model_type='finbert'):
    """
    Main function to run the test case generation workflow, adapted for API calls.
    It loads the API key from a .env file.
    
    Args:
        total_samples (int): Total number of test cases to generate.
        batch_size (int): Number of test cases to generate per API call.
        model_type (str): Type of BERT model (finbert, berttweet, patentbert, 
                         clinicalbert, biobert, scibert). Default: 'finbert'

    Returns:
        list: A list of dictionaries, where each dictionary is a test case.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")
    
    # Validate model type
    valid_models = ['finbert', 'berttweet', 'patentbert', 'clinicalbert', 'biobert', 'scibert']
    if model_type not in valid_models:
        print(f"Warning: Invalid model type '{model_type}'. Defaulting to 'finbert'.")
        model_type = 'finbert'

    all_test_cases = []
    num_generated = 0
    
    print(f"Generating test cases for model: {model_type}")
    
    with tqdm(total=total_samples, desc=f"Generating {model_type.upper()} Test Cases") as pbar:
        while num_generated < total_samples:
            current_batch_size = min(batch_size, total_samples - num_generated)
            if current_batch_size <= 0:
                break
            
            prompt = construct_prompt(current_batch_size, num_generated, model_type)
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

