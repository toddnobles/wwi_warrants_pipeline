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
    # We add this field so the model can link the person back to the specific text block in the batch
    text_block_index: int = Field(description="The index number (0, 1, 2...) of the text block where this individual was found.")
    id: str = Field(description="Identifier of the individual, typically format ###-#### or ####")
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

def extract_from_batch(batch_texts: List[str]):
    """
    Sends a batch of text blocks to Gemini.
    Constructs a prompt where each block is explicitly indexed (0, 1, 2...).
    """
    if not apiKey:
        raise ValueError("API Key is missing. Please set the GEMINI_API_KEY environment variable.")
    
    # Build a single prompt containing all text blocks with clear delimiters
    combined_text = ""
    for idx, text in enumerate(batch_texts):
        combined_text += f"\n--- TEXT BLOCK {idx} ---\n{text}\n"

    prompt = (
        "You are a specialized historical researcher. Extract every individual from the following batch of warrant log text blocks. "
        "Pay attention to case IDs (###-####) and clerk shorthand for nationalities. Nationalities are listed after the name on the same line and are abbreviated where gen, Ger, ger, per, mean German, and Austrian might be Aus, aus, or aust.\n"
        f"BATCH DATA:\n{combined_text}"
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

# 3. Processing the Large JSONL File with Batches
input_file = './data/test_json/test_25.jsonl'
output_file = 'warrant_results_25_batch_context_v2.csv'
BATCH_SIZE = 10 # Adjust this to change how much context the model sees (10-20 is usually good)
all_records = []

if not os.path.exists(input_file):
    print(f"Error: File not found at {input_file}")
else:
    print(f"Starting batch extraction from {input_file}...")
    
    with open(input_file, 'r') as f:
        # Buffer to hold lines until we reach BATCH_SIZE
        batch_buffer = [] 
        
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                line_data = json.loads(line)
                batch_buffer.append(line_data)
                
                # Check if batch is full
                if len(batch_buffer) >= BATCH_SIZE:
                    print(f"Processing Batch (Lines {i+1-BATCH_SIZE} to {i+1})...")
                    
                    # Extract just the text for the model
                    text_batch = [item.get('text', '') for item in batch_buffer]
                    
                    # Call API
                    result = extract_from_batch(text_batch)
                    
                    # Map results back to metadata using the index
                    for person in result.people:
                        idx = person.text_block_index
                        
                        # Safety check: ensure index is valid for this batch
                        if 0 <= idx < len(batch_buffer):
                            source_data = batch_buffer[idx]
                            source_pdf = source_data.get('metadata', {}).get('Source-File', 'Unknown')
                            raw_json = json.dumps(source_data)
                            
                            print(f"  > Found: {person.name} (Block {idx} -> {source_pdf})")
                            
                            record_dict = person.model_dump()
                            # STRICT METADATA ASSIGNMENT HERE
                            record_dict['source_file'] = source_pdf
                            record_dict['raw_json_input'] = raw_json
                            all_records.append(record_dict)
                        else:
                            print(f"  !! Warning: Model returned invalid block index {idx} for {person.name}")

                    # Clear buffer
                    batch_buffer = []

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {i+1}")

        # Process remaining items in buffer (if any)
        if batch_buffer:
            print(f"Processing Final Batch ({len(batch_buffer)} items)...")
            text_batch = [item.get('text', '') for item in batch_buffer]
            result = extract_from_batch(text_batch)
            for person in result.people:
                idx = person.text_block_index
                if 0 <= idx < len(batch_buffer):
                    source_data = batch_buffer[idx]
                    record_dict = person.model_dump()
                    record_dict['source_file'] = source_data.get('metadata', {}).get('Source-File', 'Unknown')
                    record_dict['raw_json_input'] = json.dumps(source_data)
                    all_records.append(record_dict)

    # 4. Save to CSV
    if all_records:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'id', 'name', 'alias', 'location', 'nationality', 
                'final_status', 'final_status_date',
                'chronology', 'raw_json_input', 'text_block_index'
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
                    'chronology': event_str,
                    'raw_json_input': r['raw_json_input'],
                    'text_block_index': r.get('text_block_index')
                })
        
        print(f"\nFinished! Extracted {len(all_records)} total records to {output_file}")