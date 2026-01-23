import json
import csv
import os
import glob
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai

# 1. Define the Schema
class CaseEvent(BaseModel):
    date: Optional[str] = Field(None, description="The date of the event (e.g., 7-29-18). Years are 1917-1921.")
    action: str = Field(description="Summary of the event, e.g., Warrant issued, Recommendation sent")

class PersonRecord(BaseModel):
    name: str = Field(description="Full name of the individual")
    alias: Optional[str] = Field(None, description="Alias or other names if mentioned")
    location: Optional[str] = Field(None, description="City and State mentioned (e.g., St. Louis, Mo.)")
    nationality: Optional[str] = Field(None, description="Nationality if listed (e.g., Ger, Austrian, gen)")
    final_status: Optional[str] = Field("Unknown", description="Final disposition: e.g., Paroled, Insane, Released, To War")
    final_status_date: Optional[str] = Field(None, description="The date the final status was reached")
    events: List[CaseEvent] = Field(default_factory=list, description="Chronological list of all events for this person")

class ExtractionResponse(BaseModel):
    people: List[PersonRecord]

# 2. Gemini API Configuration
# Use uv add google-genai to install the necessary library
apiKey = os.getenv("GEMINI_API_KEY", "")
MODEL_ID = "gemini-2.5-flash-preview-09-2025" # Or gemini-2.5-flash-preview-09-2025 once generally available in the SDK

# Initialize the Google GenAI client
client = genai.Client(api_key=apiKey)

def extract_structured_data(ocr_text):
    """
    Sends text to Gemini API using the google-genai SDK with native structured outputs.
    """
    if not apiKey:
        raise ValueError("API Key is missing. Please set the GEMINI_API_KEY environment variable.")
    
    prompt = (
        "Extract the primary individuals and their legal chronology from these warrant logs. "
        "Include dates for all events. Ignore administrative staff or officials unless "
        "they are the subject of the warrant.\n\n"
        f"LOG TEXT:\n{ocr_text}"
    )

    # Exponential Backoff Implementation
    for i in range(6):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExtractionResponse.model_json_schema(),
                },
            )
            
            # Use model_validate_json to turn the response string into Pydantic objects
            return ExtractionResponse.model_validate_json(response.text)
            
        except Exception as e:
            if i == 5: 
                print(f"  !! API Error after retries: {e}")
                raise e
            wait_time = 2 ** i
            time.sleep(wait_time) 
            
    return ExtractionResponse(people=[])

# 3. Processing a Directory of JSONL Files
input_folder = './data/test_json/' 
output_file = 'warrant_results.csv'
all_records = []

jsonl_files = sorted(glob.glob(os.path.join(input_folder, "*.jsonl")))

if not jsonl_files:
    print(f"Error: No .jsonl files found in {input_folder}")
else:
    print(f"Starting Gemini extraction from {len(jsonl_files)} files...")
    
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        print(f"\n--- Opening File: {file_name} ---")
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    line_data = json.loads(line)
                    source_pdf = line_data.get('metadata', {}).get('Source-File', 'Unknown')
                    
                    print(f"Processing Entry {i+1} (Source: {source_pdf})...")
                    
                    result = extract_structured_data(line_data['text'])
                    
                    for person in result.people:
                        print(f"  > Found: {person.name}")
                        record_dict = person.model_dump()
                        record_dict['source_file'] = source_pdf
                        record_dict['original_jsonl'] = file_name
                        all_records.append(record_dict)
                        
                except Exception as e:
                    print(f"  !! Error on line {i+1} of {file_name}: {e}")

    # 4. Save to CSV
    if all_records:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'name', 'alias', 'location', 'nationality', 
                'final_status', 'final_status_date', 'source_file', 'original_jsonl', 'chronology'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in all_records:
                events = r.get('events', [])
                event_str = " | ".join([f"{e.get('date') or 'No Date'}: {e.get('action') or 'No Action'}" for e in events])
                
                writer.writerow({
                    'name': r['name'],
                    'alias': r['alias'],
                    'location': r['location'],
                    'nationality': r['nationality'],
                    'final_status': r['final_status'],
                    'final_status_date': r['final_status_date'],
                    'source_file': r['source_file'],
                    'original_jsonl': r['original_jsonl'],
                    'chronology': event_str
                })
        
        print(f"\nFinished! Extracted {len(all_records)} total records from {len(jsonl_files)} files to {output_file}")
    else:
        print("\nNo records were extracted.")