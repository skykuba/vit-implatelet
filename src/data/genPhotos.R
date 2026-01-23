path <- getwd()
if (substr(path, nchar(path), nchar(path)) != "/") {
  path <- paste0(path, "/")
}
includePath <- "include/"

source(paste0(path, includePath, 'generateKeggPathwayImages.R'))

user_rdata_file <- "../../data/rdata/countsAndSamples.RData"
url <- "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz"
dest <- "../../data/raw/gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz"

dir.create("../../data/raw", recursive = TRUE, showWarnings = FALSE)

user_gtf_file <- "../../data/raw/gencode.v19.chr_patch_hapl_scaff.annotation.gtf"
if (!file.exists(user_gtf_file)) {
  if (!file.exists(dest)) {
    download.file(url, dest, mode = "wb")
  }
  R.utils::gunzip(dest, remove = TRUE)
} else {
  message("GTF file already exists, skipping download and extraction.")
}
source(paste0(path, includePath, "statisticalAnalysis.R"))
source(paste0(path, includePath, "annotateData.R"))

message(paste("Loading data from:", user_rdata_file))
if (!file.exists(user_rdata_file)) {
  stop("The .RData file was not found. Check the path in 'user_rdata_file'.")
}
load(user_rdata_file)
if (!exists("countsRaw")) {
  stop("The loaded .RData file does not contain an object named 'countsRaw'.")
}
message("Normalizing data...")
dataFiltered <- normalizeDESeq2NoReport(countsRaw)
message("Normalization completed.")

annotation_result <- annotateData(user_gtf_file, dataFiltered)
dataFiltered_annotated <- annotation_result$dataFiltered
message("Annotation completed.")


sampleGroups <- NULL
if (exists("sampleInfo") && "Group" %in% colnames(sampleInfo)) {
  if (nrow(sampleInfo) == ncol(dataFiltered_annotated)) {
    sampleGroups <- sampleInfo$Group
  } else {
    warning("sampleInfo$Group omitted: number of rows does not match number of samples")
  }
} else {
  message("No sampleInfo$Group – files will be named by sample only")
}


message("Starting generation of KEGG pathway images...")
matrixPath <- generateKeggPathwayImages("../../data/", dataFiltered_annotated, sampleGroups = sampleGroups)

message(paste("Completed. Matrices saved to:", matrixPath))
message(paste("Images saved to:", gsub("Matrices", "Images", matrixPath)))
