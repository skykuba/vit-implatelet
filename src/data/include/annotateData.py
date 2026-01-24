import pandas as pd

def annotateData(gtfPath19, dataFiltered):
    """
    Annotate transcripts

    This function adds information about gene name and position to transcripts
    returns dict with two elements:
    dataFiltered = normalized expression matrix with gene names as index
    genePositionInfo - dataframe with position and name information about each transcript
    input:
    gtfPath19 - path to Gencode hg19 annotation
    dataFiltered - normalized reads with ENSEMBL rownames
    """
    # Read GTF file
    ann = pd.read_csv(gtfPath19, sep='\t', header=None, skiprows=5, low_memory=False)

    # Filter for gene entries with gene_status KNOWN
    ann_gene = ann[(ann[2] == "gene") & (ann[8].str.contains('gene_status "KNOWN"'))].copy()

    # Function to extract gene_id without version
    def extract_gene_id(attr):
        parts = attr.split(';')
        gid_part = [p for p in parts if 'gene_id' in p][0].strip()
        gid = gid_part.split('"')[1]
        return gid.split('.')[0]

    # Function to extract gene_name
    def extract_gene_name(attr):
        parts = attr.split(';')
        gname_part_list = [p for p in parts if 'gene_name' in p]
        if gname_part_list:
            gname_part = gname_part_list[0].strip()
            gname = gname_part.split('"')[1]
            return gname
        return None

    # Extract gene_ids from ann_gene
    ann_gene['gene_id_no_version'] = ann_gene[8].apply(extract_gene_id)
    fsplits = ann_gene['gene_id_no_version']

    # Get ENSEMBL IDs from dataFiltered index, remove version
    fsplits2 = dataFiltered.index.str.split('.').str[0]

    # Match fsplits2 to fsplits (R's `match` equivalent)
    fsplits_map = {val: i for i, val in enumerate(fsplits)}
    fsplitsMatches = [fsplits_map.get(val) for val in fsplits2]

    # Find and remove non-matched entries from dataFiltered
    fsplitsNonMatched_indices = [i for i, x in enumerate(fsplitsMatches) if x is None]
    if fsplitsNonMatched_indices:
        dataFiltered = dataFiltered.drop(dataFiltered.index[fsplitsNonMatched_indices])
        fsplitsMatches = [x for x in fsplitsMatches if x is not None]

    # Filter ann_gene to matched entries
    ann_gene = ann_gene.iloc[fsplitsMatches].reset_index(drop=True)

    # Extract gene_name
    ann_gene['gene_name'] = ann_gene[8].apply(extract_gene_name)

    # Remove duplicates based on gene_name, keeping the first occurrence
    duplicated_mask = ann_gene['gene_name'].duplicated(keep='first')
    if duplicated_mask.any():
        # Remove duplicates from both ann_gene and dataFiltered
        dataFiltered = dataFiltered[~duplicated_mask.values]
        ann_gene = ann_gene[~duplicated_mask].reset_index(drop=True)

    # Re-extract gene_id for the final EnsembleID column, and remove version
    ann_gene['gene_id'] = ann_gene[8].apply(lambda x: x.split(';')[0].strip().split('"')[1].split('.')[0])

    # Create genePositionInfo
    genePositionInfo = pd.DataFrame({
        'Gene': ann_gene['gene_name'],
        'EnsembleID': ann_gene['gene_id'],
        'Chr': ann_gene[0],
        'Start': ann_gene[3],
        'End': ann_gene[4]
    })
    genePositionInfo.index = genePositionInfo['Gene']

    # Set index of dataFiltered to gene names
    dataFiltered.index = ann_gene['gene_name']

    print("Data annotated")
    return {"dataFiltered": dataFiltered, "genePositionInfo": genePositionInfo}
