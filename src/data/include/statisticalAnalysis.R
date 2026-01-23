options(stringsAsFactors = FALSE)

# Opcjonalnie: włącz paralelizację DESeq2 (odkomentuj poniższe linie)
library(BiocParallel)
register(MulticoreParam(workers = parallel::detectCores() - 1))  # macOS/Linux
# register(SnowParam(workers = parallel::detectCores() - 1))       # Windows

my.t.test.p.value <- function(...) {
  obj<-try(t.test(...), silent=TRUE)
  if (is(obj, "try-error")) return(NA) else return(obj$p.value)
} 
 

#' Normalizes data with VST and saves between group expression comparison
#' returns:
#' normalized expression matrix
#' input:
#' rawCounts
#' beningnId - ids of group A (column indices in count matrix)
#' malignantId - ids of group B (column indices in count matrix)
#' reportPath - path where report should be saved
#' groupNameA - name of group A 
#' groupNameB - name of group B
#' saveReport -  default T, check if report should be written to a .tsv file 
normalizeDESeq2 = function ( rawCounts, benignId, malignantId, reportPath, groupNameA, groupNameB, saveReport = T)
{
  library(DESeq2)
  
  d = rawCounts
  sampleIds = colnames(rawCounts)
  condition <- factor(rep("A",dim(d)[2]))
  
  dds <- DESeqDataSetFromMatrix(countData = d, DataFrame(condition), design = ~ 1)
  dds <- DESeq(dds)
  vsd <- varianceStabilizingTransformation(dds, blind=TRUE)
  
  logs_norm_arr_ord = assay(vsd)
  
  logsBenign = logs_norm_arr_ord[,benignId]
  logsMalignant = logs_norm_arr_ord[,malignantId]
  
  write.table(logsBenign, file = paste(reportPath, groupNameA,"_", Sys.Date(), ".tsv", sep = ""), sep = "\t")
  write.table(logsMalignant, file = paste(reportPath, groupNameB, "_", Sys.Date(), ".tsv", sep = ""), sep = "\t")
  
  logsBenign = t(logsBenign)
  logsMalignant = t(logsMalignant)
  meanBenign <- sapply(1:dim(t(logs_norm_arr_ord))[2], function(i) mean(logsBenign[,i]))
  meanMalignant <- sapply(1:dim(t(logs_norm_arr_ord))[2], function(i) mean(logsMalignant[,i])) 
  
  medianBenign <- sapply(1:dim(t(logs_norm_arr_ord))[2], function(i) median(logsBenign[,i]))
  medianMalignant <- sapply(1:dim(t(logs_norm_arr_ord))[2], function(i) median(logsMalignant[,i])) 
  
  folds_mean_Benign_Malignant = 2^(meanBenign - meanMalignant)  
  ttest_Benign_Malignant_pvalue = sapply(1:dim(t(logs_norm_arr_ord))[2], function(i)
    my.t.test.p.value(logsBenign[,i], logsMalignant[,i]))
  
  fdrt_Benign_Malignant_qvalue = p.adjust(ttest_Benign_Malignant_pvalue, method = "fdr", n = length(ttest_Benign_Malignant_pvalue))
  summary_table_Benign_Malignant = cbind.data.frame(rownames(logs_norm_arr_ord), meanBenign, meanMalignant,
                                                    medianBenign, medianMalignant, folds_mean_Benign_Malignant,
                                                    ttest_Benign_Malignant_pvalue, fdrt_Benign_Malignant_qvalue )
  
  
  
  tableCols_Benign_Malignant=  cbind( "Gene",paste("Mean ", groupNameA, sep = ""), paste("Mean ", groupNameB, sep = ""),
                                      paste("Median ", groupNameA, sep = ""), paste("Median ", groupNameB, sep = ""),
                                      paste("Mean fold change ", groupNameA, "-", groupNameB, sep = ""),
                                      "T-test p-value ",
                                      "FDR q-value " ) 
  colnames(summary_table_Benign_Malignant) = tableCols_Benign_Malignant
  if (saveReport == T)
    write.table(summary_table_Benign_Malignant,file = paste(reportPath,  "Table_I_", groupNameA, "_vs_",
                                                            groupNameB, "_", Sys.Date(), ".tsv", sep = ""), 
                sep = "\t", row.names = TRUE)
  return(logs_norm_arr_ord[, c(benignId, malignantId)])
}

 

#' Normalizes data with VST and saves between group expression comparison
#' returns:
#' normalized expression matrix
#' input:
#' rawCounts
#' fast - jeśli TRUE, używa szybszej metody normalizacji (domyślnie TRUE)
#'
normalizeDESeq2NoReport = function ( rawCounts, fast = TRUE )
{
  library(DESeq2)
  message("Rozpoczynanie normalizacji DESeq2...")
  
  d = rawCounts
  sampleIds = colnames(rawCounts)
  condition <- factor(rep("A",dim(d)[2]))
  
  message("Tworzenie obiektu DESeqDataSet...")
  dds <- DESeqDataSetFromMatrix(countData = d, DataFrame(condition), design = ~ 1)

  if (fast) {
    # SZYBKA METODA: tylko estymacja size factors + VST
    # Pomija pełny model DESeq (estymację dyspersji i testowanie),
    # co znacząco przyspiesza obliczenia
    message("Szacowanie size factors (tryb szybki)...")
    dds <- estimateSizeFactors(dds)

    message("Wykonywanie transformacji VST (tryb szybki)...")
    # nsub=1000 - używa tylko 1000 genów do dopasowania krzywej (zamiast wszystkich)
    # fitType="mean" - szybsza metoda dopasowania
    vsd <- vst(dds, blind=TRUE, nsub=1000, fitType="mean")
  } else {
    # PEŁNA METODA: standardowy workflow DESeq2
    message("Szacowanie parametrów modelu (DESeq - pełny)...")
    dds <- DESeq(dds)
    message("Wykonywanie transformacji VST...")
    vsd <- varianceStabilizingTransformation(dds, blind=TRUE)
  }

  logs_norm_arr_ord = assay(vsd)
  return(logs_norm_arr_ord) 
}
 
 