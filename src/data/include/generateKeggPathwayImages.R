 
#' Creates directory if it didnt exist
#' 
#' Helper function
#' input:
#' mainDir = path in which directory should be created
#' subDIr == name of the directory
#' 
createDirIfNotExists = function(mainDir, subDir){
  if (file.exists(paste(mainDir,  subDir, sep = "")) == F){
    
    dir.create(paste(mainDir, subDir, sep = ""))
  }  
}  

#' Build KEGG pathway image
#' 
#' This function build images and corresponding matrices based on expression data
#' Input:
#' sigSym - list of symbols in KEGG pathways connected to signalling processes
#' metSym - list of symbols in KEGG pathways connected to metabolic processes
#' cancerSym - list of symbols in KEGG pathways connected to cancerogenesis
#' each symbol list is a list of KEGG pathways, where each KEGG pathway is represented as a list of symbols in the pathway
#' data - normalized expression matrix with gene names as rownames
#' col - column corresponding to the sample used
#' picWidth - width in pixels (calculated from max matched genes across pathways)
#' picHeight - height in pixels (total number of pathways)
#'
buildExpressionPathwayMatrix = function(sigSym, metSym, cancerSym, data, col, picWidth, picHeight)
{
  output = matrix(0, ncol = picWidth, nrow = picHeight)
  k = 1 
  for(i in 1:length(sigSym))
  {
    symbols = unlist(sigSym[i])
    nameMatch = match(symbols, rownames(data))
    nonMissing = which(is.na(nameMatch) == F)
    if (length(nonMissing) > 0) {
      output[k, 1:length(nonMissing)] = data[nameMatch[nonMissing], col]
    }
    k = k+1 
  } 
  for(i in 1:length(metSym))
  {
    symbols = unlist(metSym[i]) 
    nameMatch = match(symbols, rownames(data))
    nonMissing = which(is.na(nameMatch) == F)
    if (length(nonMissing) > 0) {
      output[k, 1:length(nonMissing)] = data[nameMatch[nonMissing], col]
    }
    k = k+1 
  } 
  for(i in 1:length(cancerSym))
  {
    symbols = unlist(cancerSym[i])
    nameMatch = match(symbols, rownames(data))
    nonMissing = which(is.na(nameMatch) == F)
    if (length(nonMissing) > 0) {
      output[k, 1:length(nonMissing)] = data[nameMatch[nonMissing], col]
    }
    k = k+1 
  } 
  return(output)
} 
#' Prepares KEGG pathway panel images
#' returns:
#' matrixPath - path to directory with matrices
#' input:
#' path  - path to the directory in which matrices and images should beheld
#' dataFiltered - normalized reads with ENSEMBL gene names as rownames
#' pathwayPaths - name of the folder in which images and matrices should be stored. If the folder doesnt exist in path, it will be created
#' sampleGroups - optional vector of group labels aligned to columns of dataFiltered; used for naming output files
#'  
generateKeggPathwayImages = function(path, dataFiltered, pathwaysPath = "KeggImages", sampleGroups = NULL)
{
  library(KEGGREST)
  library(gage)
  library(org.Hs.eg.db)
  data(egSymb)
  kg.hsa=kegg.gsets(species = "hsa", id.type = "kegg")
  dis.kg = kg.hsa$kg.sets[kg.hsa$dise.idx]
  met.kg = kg.hsa$kg.sets[kg.hsa$met.idx]
  sig.kg = kg.hsa$kg.sets[kg.hsa$sig.idx]
  cancerKeggId = dis.kg[grep("hsa052", names(dis.kg))] 
  cancerKeggId = cancerKeggId[-grep("hsa05200", names(cancerKeggId))]
  metabolismSpecificKeggId = met.kg[grep("hsa0", names(met.kg))]
  #metabolismSpecificKeggId = met.kg[grep("hsa00", names(met.kg))]
  metabolismGeneralKeggId = met.kg[c(
    grep("hsa011", names(met.kg)),
    grep("hsa012", names(met.kg)))]
  metabolismSpecificKeggId = metabolismSpecificKeggId[-grep("hsa01100", names(metabolismSpecificKeggId))]
  signalKeggId = sig.kg[c(
    grep("hsa030", names(sig.kg)),
    grep("hsa00970", names(sig.kg)),
    grep("hsa030", names(sig.kg)),
    grep("hsa041", names(sig.kg)),
    grep("hsa034", names(sig.kg)),
    grep("hsa040", names(sig.kg)),
    grep("hsa0421", names(sig.kg)),
    grep("hsa0491", names(sig.kg)),
    grep("hsa0492", names(sig.kg)),
    grep("hsa04935", names(sig.kg)),
    grep("hsa046", names(sig.kg)),
    grep("hsa03320", names(sig.kg)),
    grep("hsa045", names(sig.kg)),
    grep("hsa04810", names(sig.kg)),
    grep("hsa040", names(sig.kg)))]
  metabolismGeneralKeggId = metabolismGeneralKeggId[-grep("hsa01100", names(metabolismGeneralKeggId))] 
  cancerSym = lapply(cancerKeggId, eg2sym)
  sigSym = lapply(signalKeggId , eg2sym)
  metSym = lapply(metabolismSpecificKeggId , eg2sym)
  
  # Oblicz maksymalną liczbę genów dopasowanych w dowolnym szlaku
  countMatchedGenes = function(symList, geneNames) {
    sapply(symList, function(symbols) {
      sum(!is.na(match(symbols, geneNames)))
    })
  }

  allCounts = c(
    countMatchedGenes(sigSym, rownames(dataFiltered)),
    countMatchedGenes(metSym, rownames(dataFiltered)),
    countMatchedGenes(cancerSym, rownames(dataFiltered))
  )

  # picWidth = maksymalna liczba dopasowanych genów + margines
  picWidth = max(allCounts, na.rm = TRUE) + 10
  message(paste("Maksymalna liczba genów w szlaku:", max(allCounts), "-> picWidth:", picWidth))

  # Dynamicznie oblicz wysokość na podstawie liczby szlaków
  picHeight = length(sigSym) + length(metSym) + length(cancerSym)
  message(paste("Liczba szlaków KEGG:", picHeight, "(sigSym:", length(sigSym),
                ", metSym:", length(metSym), ", cancerSym:", length(cancerSym), ")"))

  createDirIfNotExists(path, "KEGG_Pathway_Image")
  pathwayPath = paste0(path, "KEGG_Pathway_Image", "/") 
  createDirIfNotExists(pathwayPath, "Matrices")
  matrixPath = paste0(pathwayPath, "Matrices", "/") 
  createDirIfNotExists(pathwayPath, "Images")
  imagePath = paste0(pathwayPath, "Images", "/") 
  
  totalSamples = ncol(dataFiltered)
  message(paste("Generowanie obrazów dla", totalSamples, "próbek..."))

  # przygotuj etykiety plików: jeśli sampleGroups podane i długość zgodna, prefiks grupy
  sampleNames = colnames(dataFiltered)
  if (!is.null(sampleGroups) && length(sampleGroups) == length(sampleNames)) {
    fileLabels = paste0(sampleGroups, "_", sampleNames)
  } else {
    fileLabels = sampleNames
  }

  for(col in 1:ncol(dataFiltered))
  {
    output = buildExpressionPathwayMatrix(sigSym, metSym, cancerSym, data = dataFiltered,col, picWidth, picHeight)  
    write.table(output, file = paste0(matrixPath, fileLabels[col], ".txt"), row.names = F, quote = F, col.names = F) 
    png(paste0(imagePath, fileLabels[col], ".png"), width = picWidth, height = picHeight) 
    par(mar = c(0,0,0,0))
    require(grDevices) # for colours
    paletteFn = colorRampPalette(c("black", "red"))
    posOutput = pmax(output, 0) # ujemne i zera pozostają czarne, dodatnie mapowane na czerwienie
    maxVal = max(posOutput, na.rm = TRUE)
    if (maxVal <= 0 || is.infinite(maxVal) || is.na(maxVal)) {
      maxVal = 1 # unikaj zlim z tymi samymi wartościami przy macierzy zerowej
    }
    image(
      t(posOutput),
      col = paletteFn(50),
      axes = FALSE,
      zlim = c(0, maxVal),
      mar = c(0, 0, 0, 0),
      ann = FALSE
    )
    dev.off()
  }
  
  
  
  print("KEGG pathway images prepared")
  return(matrixPath)
}