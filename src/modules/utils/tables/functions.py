import json
import os
import numpy as np
import pandas as pd



def split_conformer_id_full(df_conformers: pd.DataFrame) -> pd.DataFrame:
    """
    Split the conformer_id_full column into its components. It use '_' as field separator.

    New columns:

                * prefix
                * accession_code
                * content_type
                * version
                * pol_entity
                * instance
                * model

    Parameters:

                * df_conformers:    a DataFrame. Correspond to a pandas dataframe where are stored all CoDNaS-RNA conformers.

    Returns: a DataFrame.
    """
    # Example: pdb_00003jce_xyz_v1-4_1_A_1 --> 'pdb', '00003jce', 'xyz', 'v1-4', '1', 'A', '1'
    df_conformers[['prefix', 'accession_code', 'content_type', 'version', 'pol_entity', 'instance', 'model']] = df_conformers['conformer_id_full'].apply(lambda x: pd.Series(str(x).split('_')))
    return df_conformers


def split_conformer_id(df_conformers: "pd.DataFrame", suffix: str = None) -> "pd.DataFrame":
    """
    Split 'conformer_id_full' column into:

            * 'entry_id'
            * 'content_type'
            * 'version'
            * 'pol_entity_id'
            * 'instance_id'
            * 'model_id'
            * 'entry_entity_instance_id'

    After split, it drops 'conformer_id_full' column.
    If suffix is not None, it adds it to all new columns like:

            * 'entry_id_suffix' (note it uses '_' as separator).

    Observation: 'df_conformers' may have only these columns:

            * 'cluster_id'
            * 'conformer_id_full'       (pdb_00004v6f_xyz_v2-1_54_CA_1)
            * 'conformer_id_reduced'    (4v6f_54_CA_1)
            * 'conformer_id_simplified' (4v6f_54_CA)

    Parameters:

                * df_conformers:          a dataframe. Correspond to a pandas dataframe where are stored all CoDNaS-RNA conformers.
                * suffix:                 a string. Correspond to a suffix to be added to all new columns.

    Return: a dataframe
    """
    assert "conformer_id_full" in df_conformers.columns, f"[Error]  : Given dataframe does not have 'conformer_id_full' column."

    columns_to_keep = list(df_conformers.columns.difference(['conformer_id_full']))

    # Keep 'model_id' due dbn file use this format: <entry_id>_<instance_id>_<model_id>.(rev).dbn
    new_columns = ['entry_id', 'content_type', 'version', 'pol_entity_id', 'instance_id', 'model_id']
    df_conformers[new_columns] = df_conformers.conformer_id_full.apply(lambda x: pd.Series(str(x).split('pdb_0000')[1].split('_')))
    if 'conformer_id_reduced' in df_conformers.columns:
        df_conformers['entry_entity_instance_id'] = df_conformers.conformer_id_reduced.apply(lambda x: '_'.join(x.split('_')[:-1]))
    else:
        df_conformers['entry_entity_instance_id'] = df_conformers['entry_id'] + '_' + df_conformers['pol_entity_id'] + '_' + df_conformers['instance_id']
    df_conformers.drop(columns=df_conformers.columns.difference(columns_to_keep + new_columns + ['entry_entity_instance_id']), inplace=True)
    df_conformers = df_conformers.drop_duplicates().reset_index(drop=True)
    if suffix is not None:
        df_conformers.columns = [f"{col}_{suffix}" if col in new_columns + ['entry_entity_instance_id'] else col for col in df_conformers.columns]
        df_conformers
    return df_conformers


# Taken from: https://github.com/pandas-dev/pandas/issues/45459
def implode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Implode a column into a list of values.

    Parameters:
    
                * df: a pandas DataFrame.
                * column: a string. Correspond to the column to be imploded.

    Returns: a pandas DataFrame.
    """
    keys = [c for c in df if c != column]
    return df.replace(np.nan, '').groupby(keys, as_index=False).agg({column: list})[df.columns].replace('', np.nan)


def guessExtensionFromSeparator(a_separator: str) -> tuple:
    """
    Takes a table separator and guess to which extension is related.
    If the separator is not recognized, returns a CSV extension and its
    corresponding separator.

    Parameters:

                a_separator:    a string. Correspond to a table separator.

    Return: tuple of two strings
    """
    # Guess separator
    if a_separator == "\t":
        ext = ".tsv"
    elif a_separator == ";":
        ext = ".csv"
    elif a_separator == "|":
        ext = ".psv"
    else:
        ext = ".csv"
        a_separator = ","
    return a_separator, ext


def guessSeparatorFromExtension(filename: str) -> tuple:
    """
    Takes a filename and guess to which separator is related.
    If the extension is not recognized, returns an error.

    Parameters:

                * filename:     a string. A full path filename.


    Observation: CSV separator is a comma (;)


    Return: tuple of two strings (sep, ext)
    """
    filename_orig = filename

    # Get extension
    if filename.endswith(".gz"):
        filename = filename[:-3]

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".tsv":
        a_separator = "\t"
    elif ext == ".csv":
        a_separator = ";"
    elif ext == ".psv":
        a_separator = "|"
    else:
        raise ValueError(f"[Error]  : extension '{ext}' of '{filename_orig}' not recognized.")
    return a_separator, ext


def saveTable( df          : pd.DataFrame
             , outputDir   : str
             , table_name  : str
             , a_separator : str
             , na_rep      : str = "NA"
             , compress    : bool = True) -> None:
    """
    Save dataframe as a gzip table.

    Parameters:

                * df:           a dataframe. Correspond to a dataframe name.
                * outputDir:    a string. Correspond to an output directory.
                * table_name:   a string. Correspond to a table name.
                * a_separator:  a string. Correspond to a table separator.
                * na_rep:       a string. Correspond to a NaN value. Default: "NA".

    Return: nothing
    """
    outputDir = f"{outputDir}/" if not outputDir.endswith("/") else outputDir
    a_separator, ext = guessExtensionFromSeparator(a_separator)
    if table_name.endswith(".gz"):
        table_name = table_name[:-3]
    if ext == os.path.splitext(table_name)[1]:
        table_name = f"{os.path.splitext(table_name)[0]}"
    # Save dataframa as a file.
    filename = f"{outputDir}{table_name}{ext}"
    filename = f"{filename}.gz" if compress else filename
    df.to_csv( filename
             , sep         = a_separator
             , na_rep      = na_rep
             , index       = False
             , header      = True
             , compression = "gzip" if compress else None )
    return


def getDfIfFileExist( df_name     : str
                    , afile       : str
                    , a_separator : str  = None
                    , nan_status  : bool = False ) -> pd.DataFrame:
    """
    Get a dataframe from a table file.

    Observation:

        * Tables are considered that have a header at row number 0.
        * If a_separator is None, it will be guessed from the extension of the file.
        * If nan_status is False, 'NA' is removed from pandas nan list.

    Parameters:

                * df_name:          a string. Correspond to a dataframe name.
                * afile:            a file.   Correspond to a table.
                * a_separator:      a string. Correspond to a table separator (default: None)
                * nan_status:       a bool.   Correspond to keep or not default nan values (default: False).

    Return: a pandas dataframe
    """
    # Taken from: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html
    pandas_nan_list = [ ''
                      , '#N/A'
                      , '#N/A N/A'
                      , '#NA'
                      , '-1.#IND'
                      , '-1.#QNAN'
                      , '-NaN'
                      , '-nan'
                      , '1.#IND'
                      , '1.#QNAN'
                      , '<NA>'
                      , 'N/A'
                      , 'NA'
                      , 'NULL'
                      , 'NaN'
                      , 'n/a'
                      , 'nan'
                      , 'null' ]

    # Check first if file exists.
    if os.path.exists(afile):
        filename = afile
    elif os.path.exists(f"{afile}.gz"):
        filename = f"{afile}.gz"
    else:
        raise FileNotFoundError(f"Error: file does not exist\t'{afile}' or\t'{afile}.gz'")

    return pd.read_table( filename
                        , sep             = guessSeparatorFromExtension(filename)[0] if a_separator is None else a_separator
                        , dtype           = str
                        , compression     = "gzip" if os.path.splitext(filename)[1].endswith('gz') else None
                        , na_values       = pandas_nan_list if nan_status else pandas_nan_list.remove('NA')
                        , keep_default_na = nan_status
                        , header          = 0 )


def load_cdrna_conformers(dirpath: str) -> "pd.DataFrame":
    """
    Load CoDNaS-RNA conformers table.

    Parameters:

                    * dirpath: a string. Correspond to the directory path where the CoDNaS-RNA conformers table is located.

    Returns: a pandas DataFrame.
    """
    try:
        cdrna_conformers = getDfIfFileExist(  df_name     = "cdrna_conformers"
                                            , afile       = f"{dirpath}cdrna_conformers.tsv.gz"
                                            , a_separator = f"\t"
                                            , nan_status  = False )
    except Exception as e:
        raise e

    assert cdrna_conformers.empty == False            , f"Error: {dirpath}cdrna_conformers.tsv.gz is empty."
    assert len(cdrna_conformers.columns) > 0          , f"Error: {dirpath}cdrna_conformers.tsv.gz does not have columns."

    cdrna_conformers.replace({'': np.nan}, inplace=True)

    cdrna_conformers  =  cdrna_conformers.astype({'cluster_id'                          : 'float'
                                                # , 'temperature'                         : 'float' # Some has a "range" as value like: 267-298
                                                # , 'ph'                                  : 'float' # Some has a "range" as value like: 6.0-6.1
                                                , 'resolution'                          : 'float'
                                                , 'q_score'                             : 'float'
                                                , 'nb_inter_clashes'                    : 'float'
                                                , 'nb_intra_clashes'                    : 'float'
                                                , 'nb_intra_interactions'               : 'float'
                                                , 'nb_inter_nt_interactions'            : 'float'
                                                , 'nb_inter_aa_interactions'            : 'float'
                                                , 'nb_nt_modified'                      : 'float'
                                                , 'src_nat_ncbi_taxonomy_id'            : 'float'
                                                , 'src_gen_gene_src_ncbi_taxonomy_id'   : 'float'
                                                , 'src_syn_ncbi_taxonomy_id'            : 'float'
                                                , 'src_gen_host_org_ncbi_taxonomy_id'   : 'float'
                                                , 'seqres_length'                       : 'float'})\
                                    .astype({ 'cluster_id'                          : 'Int64'
                                            , 'nb_inter_clashes'                    : 'Int64'
                                            , 'nb_intra_clashes'                    : 'Int64'
                                            , 'nb_intra_interactions'               : 'Int64'
                                            , 'nb_inter_nt_interactions'            : 'Int64'
                                            , 'nb_inter_aa_interactions'            : 'Int64'
                                            , 'nb_nt_modified'                      : 'Int64'
                                            , 'src_nat_ncbi_taxonomy_id'            : 'Int64'
                                            , 'src_gen_gene_src_ncbi_taxonomy_id'   : 'Int64'
                                            , 'src_syn_ncbi_taxonomy_id'            : 'Int64'
                                            , 'src_gen_host_org_ncbi_taxonomy_id'   : 'Int64'
                                            , 'seqres_length'                       : 'Int64'})

    return cdrna_conformers


def load_cdrna_pairs(dirpath: str) -> "pd.DataFrame":
    """
    Load CoDNaS-RNA pairs table.

    Parameters:

                * dirpath: a string. Correspond to the directory path where the CoDNaS-RNA pairs table is located.

    Returns: a pandas DataFrame.
    """
    try:
        cdrna_pairs = getDfIfFileExist(   df_name     = "cdrna_pairs"
                                        , afile       = f"{dirpath}cdrna_pairs.tsv.gz"
                                        , a_separator = f"\t"
                                        , nan_status  = False ) # This is important to NOT misunderstand the is_max column. With False: NaNs are '' and "NA" are a valid value.
    except Exception as e:
        raise e

    assert cdrna_pairs.empty == False            , f"Error: {dirpath}cdrna_pairs.tsv.gz is empty."
    assert len(cdrna_pairs.columns) > 0          , f"Error: {dirpath}cdrna_pairs.tsv.gz does not have columns."

    cdrna_pairs = cdrna_pairs.astype({'cluster_id'     : 'float'
                                    , 'seq_ident'      : 'float'
                                    , 'tmscore'        : 'float'
                                    , 'rmsd'           : 'float'
                                    , 'aligned_length' : 'float'
                                    , 'is_max_rank'    : 'float'
                                    , 'is_max'         : 'float'})\
                            .astype({ 'cluster_id'     : 'Int64'
                                    , 'aligned_length' : 'Int64'
                                    , 'is_max_rank'    : 'Int64'
                                    , 'is_max'         : 'Int64'})

    return cdrna_pairs


def load_cdrna_rnacentral(dirpath: str) -> "pd.DataFrame":
    """
    Load CoDNaS-RNA RNACentral table.

    Parameters:

                * dirpath: a string. Correspond to the directory path where the CoDNaS-RNA RNACentral table is located.

    Returns: a pandas DataFrame.
    """
    try:
        cdrna_rnacentral = getDfIfFileExist(  df_name     = "cdrna_rnacentral"
                                            , afile       = f"{dirpath}cdrna_rnacentral.tsv.gz"
                                            , a_separator = f"\t"
                                            , nan_status  = True )
    except Exception as e:
        raise e

    assert cdrna_rnacentral.empty == False            , f"Error: {dirpath}cdrna_rnacentral.tsv.gz is empty."
    assert len(cdrna_rnacentral.columns) > 0          , f"Error: {dirpath}cdrna_rnacentral.tsv.gz does not have columns."

    cdrna_rnacentral  = cdrna_rnacentral.astype({ 'cluster_id' : 'float'
                                                , 'taxid'      : 'float'
                                                , 'len'        : 'float'})\
                                        .astype({ 'cluster_id' : 'Int64'
                                                , 'taxid'      : 'Int64'
                                                , 'len'        : 'Int64'})

    return cdrna_rnacentral


def load_cdrna_release_table(filepath: str, metadata: str, name: str = None, relax: bool = False) -> tuple:
    """
    Load CoDNaS-RNA release table from TSV file into pandas DataFrame.

    Parameters:

                    * filepath :      string. Path to the CoDNaS-RNA release TSV table file.
                    * metadata :      string. Path to the CoDNaS-RNA release json metadata file.
                    * name :          string. Table name of a CoDNaS-RNA's release table to indicate which metadata to load (default: takes filename from filepath).
                    * relax:          bool.   If True, it will not raise an error if the columns of the table do not match the expected columns in the metadata. Default: False.

    Returns: a tuple containing the pandas DataFrame and the metadata dictionary.
    """
    try:
        assert filepath.endswith('.tsv') or filepath.endswith('.tsv.gz'), f"Error: '{filepath}' is not a TSV or TSV.GZ file. Must end with '.tsv' or '.tsv.gz'"
    except Exception as e:
        raise e

    if not os.path.isfile(filepath):
        if os.path.isfile(f"{filepath}.gz"):
            filepath = f"{filepath}.gz"
        else:
            raise FileNotFoundError(f"Error: '{filepath}' file not found.")

    try:
        with open(metadata, 'r') as jfile:
            # Load the JSON metadata file as a dictionary
            metadata = json.load(jfile)
    except Exception as e:
        raise e

    try:
        table_name = name if name is not None else os.path.basename(filepath).split('.')[0]
        table_name = f"{table_name.lower()}.tsv"
        # metadata = [mt for mt in metadata['metadata']["tables"] if name.lower() in mt]
        metadata = metadata['metadata']['tables'].get(table_name, None)
        assert metadata != None, f"Error: '{name}' table metadata not found in metadata tables: '{list(metadata['metadata']['tables'].keys())}'."
    except Exception as e:
        raise e

    # Docs: 2.3 (stable) https://pandas.pydata.org/docs/reference/api/pandas.read_table.html#pandas.read_table
    table = pd.read_table(filepath_or_buffer = filepath
                        , delimiter          = "\t"
                        , compression        = "infer"
                        , na_values          = ["", "NA"] if not 'interaction' in table_name else [""]
                        , keep_default_na    = False
                        , low_memory         = False )

    columns = set(metadata['columns'].keys())

    try:
        assert table.empty == False                   , f"Error: '{filepath}' is empty."
        assert len(table.columns) > 0                 , f"Error: '{filepath}' does not have columns."
        assert set(table.columns) == columns or relax , f"Error: '{filepath}' columns do not match expected columns."
    except Exception as e:
        raise e

    for name in metadata['columns'].keys():
        col = metadata['columns'][name]
        if name.startswith('temperature_'): # To avoid ranges like '267-298'
            continue
        atype = col['type']
        if atype == "FLOAT":
            table = table.astype({name: "float"})
        elif atype == "INT":
            table = table.astype({name: "Int64"})
        elif atype == "STRING":
            table = table.astype({name: "object"})

    return table, {table_name: metadata}