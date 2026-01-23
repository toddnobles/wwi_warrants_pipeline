import json
import csv
import os
import glob
from typing import List, Optional
from pydantic import BaseModel, Field
from ollama import chat

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
    final_status_date: Optional[str] = Field(None, description="The date of the final status")
    events: List[CaseEvent] = Field(default_factory=list, description="Chronological list of all events for this person")

class ExtractionResponse(BaseModel):
    people: List[PersonRecord]

# 2. Extraction Function
def extract_structured_data(ocr_text):
    """
    Sends text to local Ollama  and enforces Pydantic schema.
    """
    response = chat(
        model='gemma3:4b', 
        messages=[{
            'role': 'user', 
            'content': (
                f"Extract the primary individuals and their legal chronology from these warrant logs. "
                f"Include dates for all events. Ignore administrative staff or officials unless "
                f"they are the subject of the warrant. Return valid JSON for the following text:\n\n{ocr_text}"
            )
        }],
        format=ExtractionResponse.model_json_schema(), 
        options={'temperature': 0} 
    )
    return ExtractionResponse.model_validate_json(response.message.content)

# 3. Processing a Directory of JSONL Files
# Change this to the folder where all your 5-page chunks are stored
input_folder = './data/json/' 
output_file = 'warrant_results.csv'
all_records = []

# Get a list of all .jsonl files in the directory
jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))

if not jsonl_files:
    print(f"Error: No .jsonl files found in {input_folder}")
else:
    print(f"Starting extraction from {len(jsonl_files)} files...")
    
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        print(f"\n--- Opening File: {file_name} ---")
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    source_pdf = data.get('metadata', {}).get('Source-File', 'Unknown')
                    
                    print(f"Processing Entry {i+1} (Source: {source_pdf})...")
                    
                    result = extract_structured_data(data['text'])
                    
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
                event_str = " | ".join([f"{e['date']}: {e['action']}" for e in r['events']])
                
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