import json
import csv
import os
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai

# 1. Define the Schema
class CaseEvent(BaseModel):
    date: Optional[str] = Field(None, description="The date of the event (e.g., 7-29-18). Years are 1917-1921.")
    action: str = Field(description="Summary of the event, e.g., Warrant issued, Recommendation sent")

class PersonRecord(BaseModel):
    id: str = Field(description="Identifier of the individual, this usually precedes the name and is typically of the format ###-#### or ####")
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
apiKey = os.getenv("GEMINI_API_KEY", "")
MODEL_ID = "gemini-3-flash-preview" 

client = genai.Client(api_key=apiKey)

def extract_structured_data(ocr_text):
    """
    Sends a single text entry to Gemini for extraction.
    """
    if not apiKey:
        raise ValueError("API Key is missing. Please set the GEMINI_API_KEY environment variable.")
    
    prompt = (
        "You are a specialized historical researcher. Extract every individual from the following arrest warrant log text. "
        "Pay attention to case IDs (###-####) and clerk shorthand for nationalities. "
        "Ignore administrative staff unless they are the primary subject of a warrant.\n\n"
        f"LOG TEXT:\n{ocr_text}"
    )

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
            return ExtractionResponse.model_validate_json(response.text)
            
        except Exception as e:
            if i == 5: 
                print(f"  !! API Error after retries: {e}")
                raise e
            wait_time = 2 ** i
            time.sleep(wait_time) 
            
    return ExtractionResponse(people=[])

# 3. Processing the Large JSONL File
input_file = './data/test_json/test_25.jsonl'
output_file = 'warrant_results.csv'
all_records = []

if not os.path.exists(input_file):
    print(f"Error: File not found at {input_file}")
else:
    print(f"Starting extraction from {input_file}...")
    
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            
            try:
                line_data = json.loads(line)
                # Pull source file directly from JSONL metadata (Loop Logic)
                source_pdf = line_data.get('metadata', {}).get('Source-File', 'Unknown')
                raw_text = line_data.get('text', '')
                
                print(f"Processing Entry {i+1} (Source: {source_pdf})...")
                
                # Send strictly this entry's text to the model
                result = extract_structured_data(raw_text)
                
                for person in result.people:
                    print(f"  > Found: {person.name} ({person.id})")
                    
                    record_dict = person.model_dump()
                    # Assign metadata in the loop, ensuring 100% accuracy
                    record_dict['source_file'] = source_pdf
                    # Append raw JSON for troubleshooting as requested
                    record_dict['raw_json_input'] = json.dumps(line_data)
                    all_records.append(record_dict)
                    
            except Exception as e:
                print(f"  !! Error on line {i+1}: {e}")

    # 4. Save to CSV
    if all_records:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'id', 'name', 'alias', 'location', 'nationality', 
                'final_status', 'final_status_date', 'source_file', 
                'chronology', 'raw_json_input'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in all_records:
                events = r.get('events', [])
                event_str = " | ".join([f"{e.get('date') or 'No Date'}: {e.get('action') or 'No Action'}" for e in events])
                
                writer.writerow({
                    'id': r['id'],
                    'name': r['name'],
                    'alias': r['alias'],
                    'location': r['location'],
                    'nationality': r['nationality'],
                    'final_status': r['final_status'],
                    'final_status_date': r['final_status_date'],
                    'source_file': r['source_file'],
                    'chronology': event_str,
                    'raw_json_input': r['raw_json_input']
                })
        
        print(f"\nFinished! Extracted {len(all_records)} total records to {output_file}")