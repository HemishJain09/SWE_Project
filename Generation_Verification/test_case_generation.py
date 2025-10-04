import os
import argparse
import pandas as pd
import json
import time
from tqdm import tqdm
import google.generativeai as genai

def get_llm_response(prompt, api_key, retries=5, delay=5):
    """
    Calls the Gemini API to get a response for a given prompt with retry logic.

    Args:
        prompt (str): The prompt to send to the LLM.
        api_key (str): Your Google AI Studio API key.
        retries (int): Number of retries for the API call.
        delay (int): Delay in seconds between retries.

    Returns:
        str: The text response from the LLM, or None if an error occurs.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            # Handling potential empty or malformed responses
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

    Args:
        batch_size (int): The number of test cases to generate in this batch.
        existing_count (int): The number of test cases already generated.

    Returns:
        str: A formatted prompt string.
    """
    prompt_template = f"""
    You are a test data generator for a machine learning model.
    Your task is to generate synthetic data for a financial sentiment analysis classification task.
    The data schema is: "review" (string) and "sentiment" (string: "positive", "negative", or "neutral").

    Please generate {batch_size} new, unique, and diverse financial statements.

    **Instructions:**
    1.  **Balance:** Ensure a roughly equal distribution of "positive", "negative", and "neutral" sentiments in this batch.
    2.  **Financial Focus:** Generate statements about:
        - Company earnings, revenue, profits
        - Stock market movements, IPOs, mergers
        - Economic indicators, interest rates, inflation
        - Corporate announcements, product launches
        - Financial metrics, guidance, forecasts
        - Industry trends, market conditions
        - Credit ratings, debt levels, bankruptcies
    3.  **Realistic Content:** The statements should resemble real financial news or analyst reports. Vary length and complexity.
    4.  **Output Format:** Provide the output as a valid JSON array of objects. Each object must have a "review" key and a "sentiment" key.

    **Example Output Format:**
    [
        {{"review": "Apple's quarterly earnings exceeded Wall Street expectations, driven by strong iPhone sales in Asia.", "sentiment": "positive"}},
        {{"review": "The company reported a significant loss in Q3, with revenue declining by 15% year-over-year.", "sentiment": "negative"}},
        {{"review": "The Federal Reserve raised interest rates by 75 basis points to combat inflation.", "sentiment": "neutral"}}
    ]

    We have already generated {existing_count} statements. Ensure the new statements are distinct and diverse.
    Generate exactly {batch_size} financial statements now.
    """
    return prompt_template

def parse_llm_output(output_text):
    """
    Parses the JSON output from the LLM into a list of dictionaries.

    Args:
        output_text (str): The raw string output from the LLM.

    Returns:
        list: A list of dictionaries, where each dictionary is a test case. Returns an empty list on failure.
    """
    try:
        # The LLM might wrap the JSON in markdown backticks, so we clean it.
        if "```json" in output_text:
            clean_text = output_text.split("```json")[1].split("```")[0].strip()
        else:
            clean_text = output_text.strip()
            
        parsed_json = json.loads(clean_text)
        
        # Basic validation
        if not isinstance(parsed_json, list):
            print("Error: Parsed data is not a list.")
            return []
        for item in parsed_json:
            if "review" not in item or "sentiment" not in item:
                print(f"Warning: Skipping malformed item: {item}")
                continue
        return parsed_json
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing LLM output: {e}")
        print(f"--- Received Text ---\n{output_text}\n---------------------")
        return []

def main():
    """Main function to run the test case generation workflow."""
    parser = argparse.ArgumentParser(description="Generate Test Cases using an LLM.")
    parser.add_argument("--total_samples", type=int, default=100, help="Total number of test cases to generate.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of test cases to generate per API call.")
    args = parser.parse_args()

    # It's recommended to use environment variables for API keys in production.
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyD0Eb-rZlgQgJjUy6iSZtqhVhN-lNZS7c0")
    if api_key == "YOUR_GEMINI_API_KEY":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Using a placeholder API key.            !!!")
        print("!!! Please set the GEMINI_API_KEY environment variable !!!")
        print("!!! or edit the script to provide your actual key.   !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    output_csv_path = 'sample_financial_test_cases.csv'
    all_test_cases = []

    num_generated = 0
    with tqdm(total=args.total_samples, desc="Generating Test Cases") as pbar:
        while num_generated < args.total_samples:
            current_batch_size = min(args.batch_size, args.total_samples - num_generated)
            if current_batch_size <= 0:
                break
            
            prompt = construct_prompt(current_batch_size, num_generated)
            llm_response = get_llm_response(prompt, api_key)

            if llm_response:
                parsed_data = parse_llm_output(llm_response)
                if parsed_data:
                    all_test_cases.extend(parsed_data)
                    num_generated += len(parsed_data)
                    pbar.update(len(parsed_data))

    # Save to CSV
    if all_test_cases:
        df = pd.DataFrame(all_test_cases)
        # Ensure only the required columns are saved
        df = df[['review', 'sentiment']]
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully generated and saved {len(df)} test cases to {output_csv_path}")
    else:
        print("\nNo test cases were generated. Please check the logs for errors.")

if __name__ == "__main__":
    main()
