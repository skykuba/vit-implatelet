# Script to load data from .RData file and save it to CSV format

# Check and install packages if needed
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}

# Path to input .RData file
rdata_path <- "../../data/rdata/countsAndSamples.RData"

# Path to output CSV file
output_raw_path <- "../../data/raw/counts_raw.csv"
output_samples_path <- "../../data/raw/sampleInfo.csv"
# Loading the .RData object
# Assuming the script is run from the main project directory
load(rdata_path)

# Check if 'countsRaw' object exists
if (exists("countsRaw")) {
  # Saving the 'countsRaw' data frame to CSV file
  # Using write.csv from base R to preserve row.names (genes)
  write.csv(as.data.frame(countsRaw), file = output_raw_path, row.names = TRUE)

  print(paste("Data successfully saved to:", output_raw_path))
} else {
  stop("Object 'countsRaw' not found in .RData file.")
}

if (exists("sampleInfo")) {
  write.csv(as.data.frame(sampleInfo), file = output_samples_path, row.names = FALSE)

  print(paste("Data successfully saved to:", output_samples_path))
} else {
  stop("Object 'sampleInfo' not found in .RData file.")
}
