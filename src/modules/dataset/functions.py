import pandas as pd


def enrichConformers( r_cdrna_pairs     : pd.DataFrame
                    , dataset_conformers: pd.DataFrame ) -> pd.DataFrame:
    """
    Extract unique conformer information from pairs and enrich the conformers dataset.

    Parameters:

                * r_cdrna_pairs :      a dataframe. Contains pairs of conformers with _1 and _2 suffixes.
                * dataset_conformers : a dataframe. Contains the target conformers to be enriched.

    Return: an enriched dataframe with unique conformer data.
    """
    # 1. Identify columns belonging to conformer 1 and conformer 2
    cols_1 = [c for c in r_cdrna_pairs.columns if c.endswith('_1')]
    cols_2 = [c for c in r_cdrna_pairs.columns if c.endswith('_2')]

    # 2. Create dictionaries to remove the '_1' and '_2' suffixes
    rename_1 = {c: c.removesuffix('_1') for c in cols_1}
    rename_2 = {c: c.removesuffix('_2') for c in cols_2}

    # 3. Extract and rename data from both positions
    conf_1 = r_cdrna_pairs[cols_1].rename(columns=rename_1)
    conf_2 = r_cdrna_pairs[cols_2].rename(columns=rename_2)

    # 4. Concatenate both sources and drop duplicate rows by 'conformer_id'
    df_conformers = pd.concat([conf_1, conf_2], ignore_index=True)
    df_conformers = df_conformers.drop_duplicates(subset=['conformer_id']).reset_index(drop=True)

    # 5. Merge information using a left join to keep all original records intact
    dataset_conformers_enriched = pd.merge(dataset_conformers
                                        ,  df_conformers
                                        ,  left_on='conformer_id_reduced'
                                        ,  right_on='conformer_id'
                                        ,  how='left' )

    # 6. Remove redundant and unneeded columns from the enriched dataframe
    columns_to_remove = [ 'conformer_id'
                        , 'classification'
                        , 'modifications'
                        , 'neighbors'
                        , 'q_score'
                        , 'nb_inter_clashes'
                        , 'nb_intra_clashes'
                        , 'nb_inter_nt_interactions'
                        , 'nb_inter_aa_interactions'
                        , 'nb_intra_interactions'
                        , 'src_nat_ncbi_taxonomy_id'
                        , 'src_nat_scientific_name'
                        , 'src_gen_gene_src_ncbi_taxonomy_id'
                        , 'src_gen_gene_src_scientific_name'
                        , 'src_syn_ncbi_taxonomy_id'
                        , 'src_syn_scientific_name'
                        , 'src_gen_host_org_ncbi_taxonomy_id'
                        , 'src_gen_host_org_scientific_name' ]
    
    dataset_conformers_enriched = dataset_conformers_enriched.drop(columns=columns_to_remove, errors='ignore')

    return dataset_conformers_enriched


def merge_consensus_to_conformers(dataset_conformers: pd.DataFrame, df_cdhit: pd.DataFrame) -> pd.DataFrame:
    """
    Merges consensus positions from df_cdhit into dataset_conformers.

    Matches the two datasets by aligning the 'query_id' column from df_cdhit with the
    first three underscore-separated fields of the 'conformer_id_reduced' column in
    dataset_conformers.

    Parameters:

        * dataset_conformers : pd.DataFrame
            The primary conformers dataset. Expected to have column: 'conformer_id_reduced'.
        * df_cdhit : pd.DataFrame
            The CD-HIT clustering results. Expected to have columns: 'query_id', 'consensus_start', 'consensus_end'.

    Returns:

        * dataset_conformers : pd.DataFrame
            The original dataset_conformers DataFrame with 'consensus_start' and 'consensus_end' columns added and explicitly typed as 'Int64'.
    """
    # 1. Create a shallow copy of df_cdhit to prevent altering the input DataFrame
    df_cdhit_subset = df_cdhit[["query_id", "consensus_start", "consensus_end"]].copy()

    # 2. Extract the first 3 tokens from conformer_id_reduced to build the match key
    # e.g., "1abc_A_1_additional" becomes "1abc_A_1"
    match_key = dataset_conformers["conformer_id_reduced"].str.split("_").str[:3].str.join("_")

    # 3. Temporarily append this key to a copy of dataset_conformers for merging
    df_conformers_temp = dataset_conformers.copy()
    df_conformers_temp["_temp_join_key"] = match_key

    # 4. Perform a left join to map consensus metrics while keeping original index order
    df_merged = df_conformers_temp.merge( right     = df_cdhit_subset
                                        , left_on   = "_temp_join_key"
                                        , right_on  = "query_id"
                                        , how       = "left" )

    # 5. Assign columns back to original reference and enforce strict Int64 to prevent downcasting bugs
    dataset_conformers["consensus_start"] = df_merged["consensus_start"].astype("Int64")
    dataset_conformers["consensus_end"] = df_merged["consensus_end"].astype("Int64")

    return dataset_conformers


def map_conformers_to_residues(df_prepared_conformers : pd.DataFrame
                            ,  df_residues             : pd.DataFrame ) -> pd.DataFrame:
    """
    Maps expanded conformers to residues to retain only those within
    the consensus SEQRES region of the cluster.

    Parameters:

                    * df_prepared_conformers : pd.DataFrame. Output from `prepare_conformers`.
                    * df_residues : pd.DataFrame. The residues table with multi-scheme, 3D and missings.

    Expected columns:

                    * df_prepared_conformers : ["entry_id", "version", "entity_id", "instance_id", "model_id", "consensus_start", "consensus_end"]
                    * df_residues : ["entry_id", "version", "entity_id", "auth_asym_id", "pdbx_PDB_model_num", "label_seq_id"]

    Returns: pd.DataFrame
    """
    # Merge using author space chain id (instance_id -> auth_asym_id)
    df_mapped = pd.merge(
        df_prepared_conformers,
        df_residues,
        left_on=["entry_id", "version", "entity_id", "instance_id", "model_id"],
        right_on=["entry_id", "version", "entity_id", "auth_asym_id", "pdbx_PDB_model_num"],
        how="inner",
    )
    # Filter residues strictly within the consensus alignment window
    # label_seq_id maps to the theoretical SEQRES sequence index
    df_mapped["label_seq_id"] = df_mapped["label_seq_id"].astype("Int64")
    in_region = (df_mapped["label_seq_id"] >= df_mapped["consensus_start"]) & (
        df_mapped["label_seq_id"] <= df_mapped["consensus_end"]
    )
    return df_mapped.loc[in_region].copy()


def map_one_letter_code(df_filtered     : pd.DataFrame
                        , multifasta    : Dict[str, str]) -> pd.DataFrame:
    """
    Preprocesses the molecular DataFrame and multifasta dictionary, aligns sequence positions,
    and assigns the corresponding one-letter code to the final column.

    Parameters:

                    * df_filtered : pd.DataFrame. The filtered molecular data table.
                    * multifasta : Dict[str, str]. The original dictionary of sequence strings.

    Expected columns:

                    * df_filtered : ["conformer_id_reduced", "label_seq_id"]

    Returns: pd.DataFrame
    """
    # Preprocess and build alignment keys (ensuring upper case for the entry id)
    split_ids = df_filtered['conformer_id_reduced'].str.split('_', expand=True)[[0, 1, 2]]
    df_filtered['conformer_id_reduced_to_align'] = (
        split_ids[0].str.upper() + "_" + split_ids[1].astype(str) + "_" + split_ids[2].astype(str)
    )

    # Filter the multifasta dictionary using only the required keys
    ids_in_df = set(df_filtered['conformer_id_reduced_to_align'].dropna().unique())
    multifasta_shrunk = {key: val for key, val in multifasta.items() if key in ids_in_df}

    # Initialize intermediate column for mapping
    df_filtered['one_letter_code'] = None

    # Map sequences using block index assignment
    for seq_id, sub_df in df_filtered.groupby('conformer_id_reduced_to_align'):
        lookup_key = str(seq_id).strip()
        
        try:
            sequence = multifasta_shrunk[lookup_key]
            
            positions = sub_df['label_seq_id'].astype(int).values - 1
            row_indices = sub_df.index.values
            
            valid_mask = (positions >= 0) & (positions < len(sequence))
            valid_rows = row_indices[valid_mask]
            valid_positions = positions[valid_mask]
            
            letters = [sequence[idx] for idx in valid_positions]
            df_filtered.loc[valid_rows, 'one_letter_code'] = letters
            
        except KeyError:
            continue

    # Cleanup temporary mapping column
    df_filtered = df_filtered.drop(columns=['conformer_id_reduced_to_align'], errors='ignore')

    # Reorder columns to ensure 'one_letter_code' is explicitly at the very end
    cols = [col for col in df_filtered.columns if col != 'one_letter_code'] + ['one_letter_code']
    df_filtered = df_filtered[cols]

    return df_filtered


def integrate_interactions(df_dataset_residues : pd.DataFrame
                        ,  df_interactions   : pd.DataFrame ) -> pd.DataFrame:
    """
    Integrates inter-chain residue interaction information into the dataset_residues.

    Parameters

                * df_dataset_residues : pd.DataFrame. Output from `map_conformers_to_residues`.
                * df_interactions : pd.DataFrame. Table containing individual inter-chain structural interactions.


    Expected columns:

                * df_dataset_residues : ['version', 'conformer_id_reduced', 'auth_seq_id', 'pdb_ins_code']
                * df_interactions : ['version', 'conformer_id_reduced', 'res_num', 'res_ins_code', 'ring_interaction_type']

    Returns: pd.DataFrame
    """
    # Prepare interaction tracking columns to guarantee correct string matching formats
    # Standardize insertion codes to clean empty strings
    df_interactions["res_ins_code"] = (
        df_interactions["res_ins_code"]
        .fillna("")
        .astype(str)
        .replace(["False", "false", "nan", "None", "<NA>"], "")
        .str.strip()
    )

    # Left join ensures positions with 0 interactions are preserved for valid statistical baselines
    df_integrated = pd.merge(
        df_dataset_residues,
        df_interactions,
        left_on=["version", "conformer_id_reduced", "auth_seq_id", "auth_comp_id", "pdb_ins_code"],
        right_on=["version", "conformer_id_reduced", "res_num", "res_id", "res_ins_code"],
        how="left",
    )

    return df_integrated


def addLigandIdsAndIsHoloToConformers(dataset_conformers : pd.DataFrame
                                    , dataset_ligands    : pd.DataFrame ) -> pd.DataFrame:
    """
    Extract ligand identifiers and determine the holo state for each conformer.

    Parameters:

                * dataset_conformers  : a dataframe. Contains the target conformers to be enriched.
                * dataset_ligands     : a dataframe. Contains ligand information per conformer.

    Return: an enriched dataframe with ligand_ids and is_holo columns.
    """
    # Validate required columns in both dataframes
    assert (
        "cluster_id" in dataset_conformers.columns
        and "cluster_id" in dataset_ligands.columns
    ), "Both dataframes must contain the 'cluster_id' column."
    assert (
        "conformer_id_reduced" in dataset_conformers.columns
        and "conformer_id_reduced" in dataset_ligands.columns
    ), "Both dataframes must contain the 'conformer_id_reduced' column."

    # Validate specific columns required for the ligand logic
    required_ligand_cols = [
        "identifier",
        "is_metal",
        "is_crystallization_aid",
    ]
    for col in required_ligand_cols:
        if col not in dataset_ligands.columns:
            raise ValueError(f"The '{col}' column is missing in the 'dataset_ligands' dataframe.")

    # Filter, group and aggregate the ligand information
    df_ligands_grouped = (
        dataset_ligands[[ "cluster_id"
                        , "conformer_id_reduced"
                        , "identifier"
                        , "is_metal"
                        , "is_crystallization_aid" ]]
        .drop_duplicates()
        .loc[
            lambda df:  (df["is_metal"] == False)               &
                        (df["is_crystallization_aid"] == False)
        ]
        .groupby(["cluster_id", "conformer_id_reduced"])
        .agg(ligand_ids=("identifier", lambda x: sorted(x.unique())))
        .reset_index()
    )

    # Drop target columns if they already exist to avoid duplication conflicts
    dataset_conformers.drop(columns=["ligand_ids", "is_holo"], inplace=True, errors="ignore")

    # Merge the calculated ligand identifiers into the conformers dataset
    enriched_df  =  pd.merge( dataset_conformers
                            , df_ligands_grouped
                            , on  = ["cluster_id", "conformer_id_reduced"]
                            , how = "left" )

    # Handle missing values and determine the holo state
    enriched_df["ligand_ids"] = enriched_df["ligand_ids"].fillna("[]")
    enriched_df["is_holo"]    = enriched_df["ligand_ids"] != "[]"

    return enriched_df


def addHoloApoPercsToCluster( dataset_clusters   : pd.DataFrame
                            , dataset_conformers : pd.DataFrame ) -> pd.DataFrame:
    """
    Calculate holo and apo percentages per cluster and enrich the clusters dataset.

    Parameters:

                * dataset_clusters   : a dataframe. Contains the target clusters to be enriched.
                * dataset_conformers : a dataframe. Contains the conformers with their holo/apo state and cluster IDs.

    Return: an enriched dataframe with holo and apo percentage columns.
    """
    assert 'cluster_id' in dataset_clusters.columns and 'cluster_id' in dataset_conformers.columns, "Both dataframes must contain the 'cluster_id' column."

    if 'is_holo' not in dataset_conformers.columns:
        raise ValueError("The 'is_holo' column is missing in the dataset_conformers dataframe.")
    elif dataset_conformers['is_holo'].isnull().any():
        raise ValueError("The 'is_holo' column contains null values in the dataset_conformers dataframe.")
    elif not all(dataset_conformers['is_holo'].isin([True, False])):
        unexpected_values = dataset_conformers.loc[~dataset_conformers['is_holo'].isin([True, False]), 'is_holo'].unique()
        raise ValueError(f"The 'is_holo' column contains unexpected values: {unexpected_values}. Expected values are: True, False.")

    # Group by cluster and calculate the percentage of holo conformers
    df_grouped = (
        dataset_conformers.groupby("cluster_id")
        .agg(perc_holo=("is_holo", lambda x: (x == True).mean() * 100))
        .rename(columns={"is_holo": "perc_holo"})
        .reset_index()
    )

    if 'perc_holo' in dataset_clusters.columns:
        dataset_clusters.drop(columns=['perc_holo'], inplace=True)

    # Merge the calculated holo percentages into the clusters dataset
    enriched_df = pd.merge(
        dataset_clusters, df_grouped, on="cluster_id", how="left"
    )

    # Calculate the apo percentage as the complement of holo
    enriched_df["perc_apo"] = 100 - enriched_df["perc_holo"]

    return enriched_df


def addMethodsPercsToClusters(dataset_clusters   : pd.DataFrame
                            , dataset_conformers : pd.DataFrame ) -> pd.DataFrame:
    """
    Calculate method percentages per cluster and enrich the clusters dataset.

    Parameters:

                * dataset_clusters   : a dataframe. Contains the target clusters to be enriched.
                * dataset_conformers : a dataframe. Contains the conformers with their methods and cluster IDs.

    Return: an enriched dataframe with method percentage columns.
    """
    assert 'cluster_id' in dataset_clusters.columns and 'cluster_id' in dataset_conformers.columns, "Both dataframes must contain the 'cluster_id' column."

    if 'method' not in dataset_conformers.columns:
        raise ValueError("The 'method' column is missing in the dataset_conformers dataframe.")
    elif dataset_conformers['method'].isnull().any():
        raise ValueError("The 'method' column contains null values in the dataset_conformers dataframe.")
    elif not all(dataset_conformers['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY', 'SOLUTION NMR'])):
        unexpected_values = dataset_conformers.loc[~dataset_conformers['method'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY', 'SOLUTION NMR']), 'method'].unique()
        raise ValueError(f"The 'method' column contains unexpected values: {unexpected_values}. Expected values are: 'X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY', 'SOLUTION NMR'.")

    # Group by cluster and calculate percentages for each experimental method
    df_grouped = (
        dataset_conformers.groupby("cluster_id")
        .agg(
            perc_xrd=("method", lambda x: (x == "X-RAY DIFFRACTION").mean()     * 100),
            perc_em =("method", lambda x: (x == "ELECTRON MICROSCOPY").mean()   * 100),
            perc_nmr=("method", lambda x: (x == "SOLUTION NMR").mean()          * 100),
        )
        .reset_index()
    )

    dataset_clusters.drop(columns=['perc_xrd', 'perc_em', 'perc_nmr'], inplace=True, errors='ignore')

    # Merge the calculated percentages into the clusters dataset
    return pd.merge(dataset_clusters, df_grouped, on="cluster_id", how="left")


def computePairForm(row: pd.Series) -> str:
    """
    Determine the 'form' of a pair based on its ligand status.

    Parameters:
        row : pd.Series
            A row from the dataset_pairs DataFrame containing 'nb_ligand_ids_#_cleaned'

    Returns:
        str : The form of the pair, which can be 'holo-holo', 'apo-apo', or 'mixed'.
    """
    if row['nb_ligand_ids_1_cleaned'] > 0 and row['nb_ligand_ids_2_cleaned'] > 0:
        return 'holo-holo'
    elif row['nb_ligand_ids_1_cleaned'] == 0 and row['nb_ligand_ids_2_cleaned'] == 0:
        return 'apo-apo'
    else:
        return 'mixed'


def chooseMinPair(df: pd.DataFrame, group_col: str, metric_col: str) -> list[int]:
    """
    Selects the original row indices with the minimum value of *metric_col* 
    for each group defined by *group_col*.

    If multiple rows share the same minimum value, the first occurrence is retained.
    It prioritizes rows where 'conformer_id_1' and 'conformer_id_2' belong to 
    different entries. If a group lacks such rows, it falls back to the absolute minimum.

    Parameters:

                * df : A pandas DataFrame.
                * group_col : The column name used to define groups.
                * metric_col : The column name whose minimum value is used for selection.

    Returns:

                * list[int] : A list of original indices corresponding to the selected minimum rows.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame.")
    if metric_col not in df.columns:
        raise KeyError(f"Metric column '{metric_col}' not found in DataFrame.")

    # 1. Sort to ensure the first occurrence is retained in case of ties
    sorted_df = df.sort_values(by=metric_col, ascending=True)

    # 2. Extract base entry IDs from conformers
    entry_1 = sorted_df['conformer_id_1'].astype(str).str.split('_').str[0]
    entry_2 = sorted_df['conformer_id_2'].astype(str).str.split('_').str[0]
    
    # Mask for rows where conformers do NOT belong to the same entry
    valid_mask = entry_1 != entry_2

    # 3. MAIN CASE: Groups with at least one valid pair
    valid_df = sorted_df[valid_mask]
    if not valid_df.empty:
        valid_indices = valid_df.groupby(group_col)[metric_col].idxmin().tolist()
    else:
        valid_indices = []

    # 4. FALLBACK CASE: Groups that do not have any valid cross-entry pair
    resolved_groups = valid_df[group_col].unique()
    fallback_df = sorted_df[~sorted_df[group_col].isin(resolved_groups)]
    
    if not fallback_df.empty:
        fallback_indices = fallback_df.groupby(group_col)[metric_col].idxmin().tolist()
    else:
        fallback_indices = []

    # 5. Combine all retrieved original indices
    indices = valid_indices + fallback_indices
    
    print(f"Total min-RMSD pairs found: {len(indices)}")
    return indices


def assignAsMinPair(df: pd.DataFrame, group_col: str, metric_col: str, new_col: str = "is_min_new") -> pd.DataFrame:
    """
    Adds a boolean column to the DataFrame indicating whether each row is the 
    minimum pair within its group based on *metric_col*.

    Parameters:
    * df : A pandas DataFrame.
    * group_col : The column name used to define groups.
    * metric_col : The column name whose minimum value is used for selection.
    * new_col : The name of the new boolean column to be added. Default is "is_min_new".

    Returns:
    * pd.DataFrame : The original DataFrame with the additional boolean column.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame.")
    if metric_col not in df.columns:
        raise KeyError(f"Metric column '{metric_col}' not found in DataFrame.")

    # Get the correct original indices for the minimum rows per group
    min_indices = chooseMinPair(df, group_col, metric_col)

    # Create the boolean mask based on original index positions
    df[new_col] = df.index.isin(min_indices)
    return df


def get_pairs_w_residues(pairs: pd.DataFrame, dataset_residues_w_interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves corresponding residue interaction metadata and calculates differences using aligned metrics across a collective dataframe of aligned pairs.

    Parameters:

                * pairs : pd.DataFrame containing paired data with ['conformer_id_1', 'conformer_id_2'].
                * dataset_residues_w_interactions : Reference dataset containing granular residue measurements.

    Returns: pd.DataFrame containing concatenated and structured results for all pairs.
    """
    # First merge: suffix everything from dataset_residues_w_interactions with '_1'
    residues_1 = dataset_residues_w_interactions.add_suffix("_1").rename(
        columns={"cluster_id_1": "cluster_id"}
    )
    step_1: pd.DataFrame = pairs.merge(
        residues_1,
        left_on=["cluster_id", "conformer_id_1"],
        right_on=["cluster_id", "conformer_id_reduced_1"],
        how="left"
    )

    # Second merge: match conformer_id_2 using the mapped '_1' structural keys
    merged_all: pd.DataFrame = step_1.merge(
        dataset_residues_w_interactions.add_suffix("_2"),
        left_on=[
            "conformer_id_2",
            "cluster_id",
            "label_seq_id_aligned_1",
            "one_letter_code_1",
        ],
        right_on=[
            "conformer_id_reduced_2",
            "cluster_id_2",
            "label_seq_id_aligned_2",
            "one_letter_code_2",
        ],
        how="left"
    )

    # Create uniform primary keys for the final combined dataframe
    merged_all = merged_all.rename(
        columns={
            "label_seq_id_aligned_1": "label_seq_id_aligned",
            "one_letter_code_1": "one_letter_code",
        }
    )

    # Vectorized metric differences computation
    metrics: list[str] = ["_all", "_no_aid", "_no_aid_no_metal"]
    int_cols: list[str] = []

    for metric in metrics:
        col_1: str = f"interaction_count{metric}_1"
        col_2: str = f"interaction_count{metric}_2"
        diff_col: str = f"interaction_count{metric}_diff"

        merged_all[diff_col] = (merged_all[col_1] - merged_all[col_2]).abs()
        int_cols.extend([col_1, col_2, diff_col])

    # Enforce correct nullable Integer datatypes for all metrics
    merged_all[int_cols] = merged_all[int_cols].astype("Int64")

    # Sort structural priorities visually
    columns_head: list[str] = [
        "cluster_id",
        "conformer_id_1",
        "conformer_id_2",
        "label_seq_id_aligned",
        "one_letter_code",
        "interaction_count_all_diff",
        "interaction_count_no_aid_diff",
        "interaction_count_no_aid_no_metal_diff",
    ]

    remaining_cols: list[str] = [
        col for col in merged_all.columns if col not in columns_head
    ]
    return merged_all[columns_head + remaining_cols]