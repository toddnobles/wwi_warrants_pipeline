library(pdftools)

# this assumes you've already made the output dir
split_pdf_to_pages <- function(input_file, output_dir) {
  
  # Get document info to find total page count
  info <- pdf_info(input_file)
  total_pages <- info$pages
  
  message(paste("Processing:", input_file))
  message(paste("Total pages to split:", total_pages))
  
  input_file_short_name <- sub(".*/(.*)\\.pdf$", "\\1", input_file)
  # 4. Loop through pages and extract them one by one
  for (i in 1:total_pages) {
    filename_pattern <- sprintf("%spage_%03d.pdf", input_file_short_name, i)
    
    output_filename <- file.path(output_dir, filename_pattern)
    
    # pdf_subset creates a new PDF from specific page numbers
    pdf_subset(input_file, pages = i, output = output_filename)
    
    message(paste("Saved:", output_filename))
  }
  
  message("Done! All pages separated.")
}



for (x in 1:11) {
  file_name = paste0("/Volumes/ExtremeSSD/NARA Visit 2 Scans/RG 60 Warrants/RG 60 Warrants Vol ", x, ".pdf")
  print(file_name)
  split_pdf_to_pages(file_name,
                     "/Volumes/ExtremeSSD/NARA Visit 2 Scans/RG 60 Warrants/separated" )
}
# # Run the function
# split_pdf_to_pages("/Volumes/ExtremeSSD/NARA Visit 2 Scans/RG 60 Warrants/RG 60 Warrants Vol 1.pdf", 
#                    "/Volumes/ExtremeSSD/NARA Visit 2 Scans/RG 60 Warrants/separated" )