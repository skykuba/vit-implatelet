 
#' Annotate transcripts
#' 
#' This function adds information about gene name and position to transcripts
#' returns list with two elements:
#' dataFIltered = normalized expression matrix with gene namesas rownames
#' genePositionInfo - table with position andname information about each transcript
#' input:
#' gtfPath19 - path to Gencode hg19 annotaton 
#' dataFiltered - normalized reads with ENSEMBL rownames
#' 
annotateData = function(gtfPath19, dataFiltered)
{
ann = read.delim(gtfPath19, skip = 5, header = FALSE)
ann_gene = ann[which(ann$V3 == "gene"),] 
ann_gene = ann_gene[grep("gene_status KNOWN", ann_gene$V9),]
# ann_gene = ann_gene[-grep("level 3;", ann_gene$V9),]
# ann_gene = ann_gene[-grep("level 2;", ann_gene$V9),]
splits = strsplit(ann_gene$V9, split = ";")
splitsFull = strsplit(ann$V9, split = ";") 
fsplits2 = rownames(dataFiltered)
fsplits = substr(unlist(lapply(splits, `[[`, 1)), 9, nchar(unlist(lapply(splits, `[[`, 1)))) 
fsplits = gsub(x = fsplits, "\\..*","") 
fsplitsMatches = match(fsplits2, fsplits)
fsplitsNonMatched = which(is.na(fsplitsMatches))
if(length(fsplitsNonMatched) > 0)
{
  fsplitsMatches = fsplitsMatches[-fsplitsNonMatched]
  dataFiltered = dataFiltered[-fsplitsNonMatched,]  
}
ann_gene = ann_gene[fsplitsMatches,]
splits = strsplit(ann_gene$V9, split = ";")
fsplits = substr(unlist(lapply(splits, `[[`, 1)), 9, nchar(unlist(lapply(splits, `[[`, 1)))) 
fsplits = gsub(x = fsplits, "\\..*","") 
gnsplits = substr(unlist(lapply(splits, `[[`, 5)), 12, nchar(unlist(lapply(splits, `[[`, 5))))  
duplicatedId = which(duplicated(gnsplits)==T)
if(length(duplicatedId) > 0)
{
  gnsplits = gnsplits[-duplicatedId]
  fsplits = fsplits[-duplicatedId]
  ann_gene = ann_gene[-duplicatedId, ]
  dataFiltered = dataFiltered[-duplicatedId,]
} 
genePositionInfo =  data.frame(gnsplits, fsplits,
                               ann_gene$V1, ann_gene$V4, ann_gene$V5)
colnames(genePositionInfo) = c( "Gene", "EnsembleID", "Chr", "Start", "End" ) 
rownames(genePositionInfo) = genePositionInfo$Gene
rownames(dataFiltered) = gnsplits 
print("Data annotated")
return(list("dataFiltered" = dataFiltered, "genePositionInfo" = genePositionInfo))
}