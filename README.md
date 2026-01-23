## This repository contains the code for processing RG 60 Arrest Warrants images into a useable format for matching and analyses. 

## General workflow and scripts used: 
- Split large pdfs downloaded from Dropbox or Genius Cloud using the pdf_split.R script. 
- Then we run these pdfs through Cirrascale's hosted verison of the olmocr2 pipeline in the olmocr_warrants.py script. This outputs jsonls and markdown versions of the pages. 
- The json outputs are in groups of five as that's the batch setting I used for the olmocr pipeline. Within these jsons were the name lists/indices at the beginning of each volume of warrants. Removing these manually was easier than through a script with some rule based exclusion, so I ran the json_combination.py script to combine the jsons with 5 records in each to one large json file. I then extracted the name list pages and stored them in the name_lists.jsonl file. The indivdual "narratives" (the pages we care about) are in the individual_narratives.jsonl file. 
- This individual_narratives.json file is what we then pass into various json_extraction_****.py scripts for testing which model is performing best at getting a useful summary of case for each person including their name, location of arrest, nationality, final status (paroled, to war camp, etc.). 
    - For the local models I've tested the following (none of which provided adequate results). 
        - llama3.1 was okay
        - deepseek-r1:8b missed a full person and didn't catch nationalities for most 
        - gemma3:4b was fast but left most fields blank
        - 
- An alternative route could be to do regex string searches to separate the json entries into individuals. The core difficulty to move from jsons or markdown files to csv or analysis ready dataset is that each page in the documents contained two individuals. Usually they are separated by a double line break, but that is not the only time when double line breaks are present. Other string pattern anchoring problems arise when trying to anchor based on individual ids in the top left of the page or by line length, etc. This is why I switched to using a pass by an LLM to try and process these markdowns or jsons as a human would. 