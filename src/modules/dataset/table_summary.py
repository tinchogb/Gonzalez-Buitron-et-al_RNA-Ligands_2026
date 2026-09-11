"""
Descriptive statistics module for RNA conformer datasets.

This module provides a collection of typed functions to compute and assemble
descriptive statistics across multiple related DataFrames (clusters, conformers,
pairs, and ligands), stratified by every RNA type present in the dataset.

Consistency checks are enforced across DataFrames to guarantee that cluster
identifiers, RNA type labels, and structural counts align before any aggregation
is performed.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import ast


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_min_max_avg(
    s             : pd.Series
  , fmt           : str = "{min:.1f} / {max:.1f} / {avg:.1f}"
) -> str:
    """
    Format a numeric Series into a single min / max / avg string.

    Parameters:

                * s :   a pandas Series. Must contain numeric data.
                * fmt : a string. The format template accepting {min}, {max}, and {avg}.
                        Default is "{min:.1f} / {max:.1f} / {avg:.1f}".

    Returns:

                * str : The formatted descriptive string.
    """
    return fmt.format(min=s.min(), max=s.max(), avg=s.mean())


def _parse_ligand_ids(
    val           : Union[str, list, None]
) -> List[str]:
    """
    Parse a ligand_ids cell into a flat list of ligand identifier strings.

    Parameters:

                * val : a string, list, or None. The raw cell content from a
                        ligand_ids column (e.g. "[]", "['MG', 'SO4']", or a list).

    Observations:

                * Uses ``ast.literal_eval`` for safe parsing of string representations.
                * Returns an empty list for missing values, empty strings, or the
                  literal string "[]".

    Returns:

                * List[str] : The extracted ligand identifiers.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x is not None and str(x).strip()]
    if isinstance(val, str):
        val = val.strip()
        if val in ("", "[]", "None", "nan"):
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x is not None and str(x).strip()]
            return []
        except (ValueError, SyntaxError):
            return []
    return []


def _validate_rna_column(
    *dfs          : pd.DataFrame
  , rna_col       : str
) -> None:
    """
    Assert that *rna_col* exists in every supplied DataFrame.

    Parameters:

                * dfs :     one or more pandas DataFrames to inspect.
                * rna_col : a string. The RNA type column name that must be present
                            in every DataFrame.

    Raises:

                * KeyError : If *rna_col* is missing from any DataFrame.
    """
    for i, df in enumerate(dfs, start=1):
        if rna_col not in df.columns:
            raise KeyError(
                f"Column '{rna_col}' not found in DataFrame #{i} "
                f"(columns: {list(df.columns)})."
            )


def _validate_cluster_consistency(
    df_clusters   : pd.DataFrame
  , df_conformers : pd.DataFrame
  , df_pairs      : pd.DataFrame
  , cluster_col   : str = "cluster_id"
) -> None:
    """
    Cross-check cluster identifiers across the cluster, conformer, and pair
    DataFrames and emit informative warnings when inconsistencies are detected.

    Parameters:

                * df_clusters :   a pandas DataFrame. The cluster-level metadata.
                * df_conformers : a pandas DataFrame. The conformer-level metadata.
                * df_pairs :      a pandas DataFrame. The pairwise comparisons.
                * cluster_col :   a string. The column name that stores cluster IDs.
                                  Default is "cluster_id".

    Observations:

                * Compares the unique set of *cluster_col* values among the three
                  DataFrames.  Mismatches are reported but **not** corrected, because
                  the caller may intentionally be working with a filtered subset.
                * Also verifies that the number of unique clusters in *df_clusters*
                  equals the number of unique clusters in *df_conformers* when both
                  are expected to describe the same population.
    """
    clusters_in_clusters   = set(df_clusters[cluster_col].unique())
    clusters_in_conformers = set(df_conformers[cluster_col].unique())
    clusters_in_pairs      = set(df_pairs[cluster_col].unique())

    n_clust = len(clusters_in_clusters)
    n_conf  = len(clusters_in_conformers)
    n_pair  = len(clusters_in_pairs)

    if n_clust != n_conf:
        missing_in_conf = clusters_in_clusters - clusters_in_conformers
        extra_in_conf   = clusters_in_conformers - clusters_in_clusters
        print(
            f"[WARNING] Cluster count mismatch: "
            f"df_clusters has {n_clust} unique clusters, "
            f"df_conformers has {n_conf}. "
            f"Missing in conformers: {len(missing_in_conf)} | "
            f"Extra in conformers: {len(extra_in_conf)}"
        )

    if n_clust != n_pair:
        missing_in_pairs = clusters_in_clusters - clusters_in_pairs
        extra_in_pairs   = clusters_in_pairs - clusters_in_clusters
        print(
            f"[WARNING] Cluster count mismatch: "
            f"df_clusters has {n_clust} unique clusters, "
            f"df_pairs has {n_pair}. "
            f"Missing in pairs: {len(missing_in_pairs)} | "
            f"Extra in pairs: {len(extra_in_pairs)}"
        )

    all_clusters = clusters_in_clusters | clusters_in_conformers | clusters_in_pairs
    common_clusters = clusters_in_clusters & clusters_in_conformers & clusters_in_pairs
    if len(common_clusters) != len(all_clusters):
        print(
            f"[WARNING] Only {len(common_clusters)} / {len(all_clusters)} clusters "
            f"are present across all three DataFrames."
        )


def _validate_rna_type_consistency(
    df_clusters   : pd.DataFrame
  , df_conformers : pd.DataFrame
  , df_pairs      : pd.DataFrame
  , df_ligands    : pd.DataFrame
  , rna_col       : str
) -> None:
    """
    Ensure that every RNA type observed in *df_clusters* also appears in the
    other DataFrames, flagging orphan types that could lead to empty cells in
    the final table.

    Parameters:

                * df_clusters :   a pandas DataFrame. The cluster-level metadata.
                * df_conformers : a pandas DataFrame. The conformer-level metadata.
                * df_pairs :      a pandas DataFrame. The pairwise comparisons.
                * df_ligands :    a pandas DataFrame. The ligand-level annotations.
                * rna_col :       a string. The RNA type column name.
    """
    rna_clusters   = set(df_clusters[rna_col].dropna().unique())
    rna_conformers = set(df_conformers[rna_col].dropna().unique())
    rna_pairs      = set(df_pairs[rna_col].dropna().unique())
    rna_ligands    = set(df_ligands[rna_col].dropna().unique())

    for name, rna_set in [
        ("df_conformers", rna_conformers),
        ("df_pairs", rna_pairs),
        ("df_ligands", rna_ligands),
    ]:
        orphan = rna_clusters - rna_set
        if orphan:
            print(
                f"[WARNING] RNA types present in df_clusters but missing "
                f"in {name}: {sorted(orphan)}"
            )


# ---------------------------------------------------------------------------
# RNA type discovery
# ---------------------------------------------------------------------------

def get_all_rna_types(
    df_conformers : pd.DataFrame
  , rna_col       : str = "rna_type_cdrna"
) -> List[str]:
    """
    Return every unique RNA type present in the conformer DataFrame, sorted by
    descending population.

    Parameters:

                * df_conformers : a pandas DataFrame. The conformer-level dataset
                                  used as the population reference.
                * rna_col :       a string. The column that stores RNA type annotations.
                                  Default is "rna_type_cdrna".

    Observations:

                * Drops missing values in *rna_col* before counting.
                * Returns the types sorted in descending order of abundance.
                * No types are hidden or binned into an "Other" category.

    Returns:

                * List[str] : The ordered list of all RNA type labels.
    """
    counts = df_conformers[rna_col].dropna().value_counts()
    return counts.index.tolist()


# ---------------------------------------------------------------------------
# Metric calculators
# ---------------------------------------------------------------------------

def compute_cluster_counts(
    df_clusters   : pd.DataFrame
  , rna_col       : str = "rna_type_cdrna"
) -> pd.Series:
    """
    Compute the number of distinct clusters per RNA type.

    Parameters:

                * df_clusters : a pandas DataFrame. The cluster-level metadata.
                * rna_col :     a string. The column that stores RNA type annotations.
                                Default is "rna_type_cdrna".

    Returns:

                * pd.Series : A Series indexed by RNA type with cluster counts.
    """
    return df_clusters.groupby(rna_col, observed=False)["cluster_id"].nunique()


def compute_conformer_counts(
    df_conformers : pd.DataFrame
  , rna_col       : str = "rna_type_cdrna"
) -> pd.Series:
    """
    Compute the total number of conformers per RNA type.

    Parameters:

                * df_conformers : a pandas DataFrame. The conformer-level dataset.
                * rna_col :       a string. The column that stores RNA type annotations.
                                  Default is "rna_type_cdrna".

    Returns:

                * pd.Series : A Series indexed by RNA type with conformer counts.
    """
    return df_conformers.groupby(rna_col, observed=False).size()


def compute_conformers_per_cluster_stats(
    df_clusters   : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , cluster_size_col : str     = "cluster_size"
  , fmt           : str        = "{min:.0f} / {max:.0f} / {avg:.1f}"
) -> pd.Series:
    """
    Compute min / max / avg cluster size per RNA type.

    Parameters:

                * df_clusters :      a pandas DataFrame. The cluster-level metadata.
                * rna_col :          a string. The column that stores RNA type annotations.
                                     Default is "rna_type_cdrna".
                * cluster_size_col : a string. The column holding the pre-computed cluster size.
                                     Default is "cluster_size".
                * fmt :              a string. The output format template for min/max/avg.
                                     Default is "{min:.0f} / {max:.0f} / {avg:.1f}".

    Observations:

                * Relies on the pre-calculated *cluster_size_col* rather than recomputing
                  from a conformer table, ensuring consistency with the clustering pipeline.

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted size statistics.
    """
    grouped = df_clusters.groupby(rna_col, observed=False)[cluster_size_col]
    return grouped.apply(lambda s: _format_min_max_avg(s, fmt))


def compute_seqres_length_stats(
    df_conformers : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , length_col    : str        = "seqres_length"
  , fmt           : str        = "{min:.0f} / {max:.0f} / {avg:.1f}"
) -> pd.Series:
    """
    Compute min / max / avg SEQRES length per RNA type.

    Parameters:

                * df_conformers : a pandas DataFrame. The conformer-level dataset.
                * rna_col :       a string. The column that stores RNA type annotations.
                                  Default is "rna_type_cdrna".
                * length_col :    a string. The column holding sequence lengths.
                                  Default is "seqres_length".
                * fmt :           a string. The output format template for min/max/avg.
                                  Default is "{min:.0f} / {max:.0f} / {avg:.1f}".

    Observations:

                * Drops missing values in *length_col* to avoid biased averages.

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted length statistics.
    """
    grouped = df_conformers.groupby(rna_col, observed=False)[length_col]
    return grouped.apply(lambda s: _format_min_max_avg(s.dropna(), fmt))


def compute_rmsd_stats(
    df_pairs      : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , rmsd_col      : str        = "rmsd"
  , fmt           : str        = "{min:.2f} / {max:.2f} / {avg:.2f}"
) -> pd.Series:
    """
    Compute min / max / avg RMSD per RNA type from all pairwise comparisons.

    Parameters:

                * df_pairs :  a pandas DataFrame. The pairwise conformer comparisons.
                * rna_col :   a string. The column that stores RNA type annotations.
                              Default is "rna_type_cdrna".
                * rmsd_col :  a string. The column holding RMSD values.
                              Default is "rmsd".
                * fmt :       a string. The output format template for min/max/avg.
                              Default is "{min:.2f} / {max:.2f} / {avg:.2f}".

    Observations:

                * Drops missing RMSD values before aggregation.
                * Operates on the full pairwise matrix, therefore the average reflects
                  the global structural variability within each RNA class.
                * Large clusters are over-represented because they contribute more pairs.

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted RMSD statistics.
    """
    grouped = df_pairs.groupby(rna_col, observed=False)[rmsd_col]
    return grouped.apply(lambda s: _format_min_max_avg(s.dropna(), fmt))


def compute_max_rmsd_per_cluster_stats(
    df_clusters   : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , max_rmsd_col  : str        = "max_rmsd"
  , fmt           : str        = "{min:.2f} / {max:.2f} / {avg:.2f}"
) -> pd.Series:
    """
    Compute min / max / avg of the pre-calculated maximum RMSD per cluster.

    Parameters:

                * df_clusters :  a pandas DataFrame. The cluster-level metadata.
                * rna_col :      a string. The column that stores RNA type annotations.
                                 Default is "rna_type_cdrna".
                * max_rmsd_col : a string. The column holding the maximum RMSD per cluster.
                                 Default is "max_rmsd".
                * fmt :          a string. The output format template for min/max/avg.
                                 Default is "{min:.2f} / {max:.2f} / {avg:.2f}".

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted max-RMSD statistics.
    """
    grouped = df_clusters.groupby(rna_col, observed=False)[max_rmsd_col]
    return grouped.apply(lambda s: _format_min_max_avg(s.dropna(), fmt))


def compute_apo_stats(
    df_clusters   : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , apo_col       : str        = "perc_apo"
  , fmt           : str        = "{min:.1f} / {max:.1f} / {avg:.1f}"
) -> pd.Series:
    """
    Compute min / max / avg percentage of apo conformers per cluster.

    Parameters:

                * df_clusters : a pandas DataFrame. The cluster-level metadata.
                * rna_col :     a string. The column that stores RNA type annotations.
                                Default is "rna_type_cdrna".
                * apo_col :     a string. The column holding the apo percentage per cluster.
                                Default is "perc_apo".
                * fmt :         a string. The output format template for min/max/avg.
                                Default is "{min:.1f} / {max:.1f} / {avg:.1f}".

    Observations:

                * The percentage is expected to be stored as a raw number (e.g. 85.5 for 85.5 %),
                  not as a fraction.

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted apo statistics.
    """
    grouped = df_clusters.groupby(rna_col, observed=False)[apo_col]
    return grouped.apply(lambda s: _format_min_max_avg(s.dropna(), fmt))


def compute_holo_stats(
    df_clusters   : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , holo_col      : str        = "perc_holo"
  , fmt           : str        = "{min:.1f} / {max:.1f} / {avg:.1f}"
) -> pd.Series:
    """
    Compute min / max / avg percentage of holo conformers per cluster.

    Parameters:

                * df_clusters : a pandas DataFrame. The cluster-level metadata.
                * rna_col :     a string. The column that stores RNA type annotations.
                                Default is "rna_type_cdrna".
                * holo_col :    a string. The column holding the holo percentage per cluster.
                                Default is "perc_holo".
                * fmt :         a string. The output format template for min/max/avg.
                                Default is "{min:.1f} / {max:.1f} / {avg:.1f}".

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted holo statistics.
    """
    grouped = df_clusters.groupby(rna_col, observed=False)[holo_col]
    return grouped.apply(lambda s: _format_min_max_avg(s.dropna(), fmt))


def compute_method_distribution(
    df_clusters   : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , xrd_col       : str        = "perc_xrd"
  , em_col        : str        = "perc_em"
  , nmr_col       : str        = "perc_nmr"
  , fmt           : str        = "XRD: {xrd:.1f} / EM: {em:.1f} / NMR: {nmr:.1f}"
) -> pd.Series:
    """
    Compute the average percentage of X-ray crystallography, cryo-EM, and NMR
    structures per RNA type.

    Parameters:

                * df_clusters : a pandas DataFrame. The cluster-level metadata.
                * rna_col :     a string. The column that stores RNA type annotations.
                                Default is "rna_type_cdrna".
                * xrd_col :     a string. The column holding the XRD percentage.
                                Default is "perc_xrd".
                * em_col :      a string. The column holding the EM percentage.
                                Default is "perc_em".
                * nmr_col :     a string. The column holding the NMR percentage.
                                Default is "perc_nmr".
                * fmt :         a string. The output format template accepting {xrd}, {em}, {nmr}.
                                Default is "XRD: {xrd:.1f} / EM: {em:.1f} / NMR: {nmr:.1f}".

    Observations:

                * Averages the method percentages across clusters of each RNA type.
                * Missing values in any method column are ignored during the avg calculation.

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted method strings.
    """
    def _format_methods(g: pd.DataFrame) -> str:
        xrd = g[xrd_col].dropna().mean()
        em  = g[em_col].dropna().mean()
        nmr = g[nmr_col].dropna().mean()
        return fmt.format(xrd=xrd, em=em, nmr=nmr)

    return df_clusters.groupby(rna_col, observed=False).apply(_format_methods, include_groups=False)


def _old_compute_ligand_counts_from_ids(
    df_conformers : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , ligand_ids_col: str        = "ligand_ids"
) -> pd.Series:
    """
    Compute the total number of unique ligands per RNA type by parsing the ligand
    identifier lists stored in *ligand_ids_col*.

    Parameters:

                * df_conformers :  a pandas DataFrame. The conformer-level dataset.
                * rna_col :        a string. The column that stores RNA type annotations.
                                   Default is "rna_type_cdrna".
                * ligand_ids_col : a string. The column holding ligand identifier lists
                                   (e.g. "['MG', 'SO4']" or Python lists). Empty lists are "[]".
                                   Default is "ligand_ids".

    Observations:

                * Uses :func:`_parse_ligand_ids` to safely evaluate string representations
                  of Python lists.
                * Sums the lengths of the parsed lists per RNA type.
                * A conformer with an empty list ("[]") contributes zero ligands.

    Returns:

                * pd.Series : A Series indexed by RNA type with total ligand counts.
    """
    df = df_conformers.copy()
    parsed_ligands = df[ligand_ids_col].apply(_parse_ligand_ids)
    return (
        parsed_ligands.groupby(df[rna_col], observed=False)
        .agg(lambda series: len({ligand for lst in series for ligand in lst}))
        .rename("_ligand_count")
    )


def compute_ligand_counts_from_df(
    df_ligands     : pd.DataFrame
  , rna_col        : str = "rna_type_cdrna"
  , identifier_col : str = "identifier"
) -> pd.Series:
    """
    Compute the total number of unique ligand identifiers per RNA type directly
    from the ligand DataFrame.
    """
    return df_ligands.groupby(rna_col, observed=False)[identifier_col].nunique()


def _old_compute_all_ligand_type_percentages(
    df_ligands    : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , ligand_type_col : str      = "partner_rcsb_type_grouped"
  , fmt           : str        = "{name}: {pct:.1f}%"
) -> pd.Series:
    """
    Compute the percentage distribution of **all** ligand types per RNA type.

    Parameters:

                * df_ligands :      a pandas DataFrame. The ligand-level dataset.
                * rna_col :         a string. The column that stores RNA type annotations.
                                    Default is "rna_type_cdrna".
                * ligand_type_col : a string. The column storing ligand type classifications.
                                    Default is "partner_rcsb_type_grouped".
                * fmt :             a string. The format template for each ligand type line,
                                    accepting {name} and {pct}. Default is "{name}: {pct:.1f}%".

    Observations:

                * Percentages are computed relative to the total ligand entries of each RNA type.
                * The output is a single concatenated string per RNA type, with entries separated
                  by " / ".
                * **No truncation** is applied; every distinct ligand type is reported.
                  Callers should be aware that RNA types with many ligand categories will
                  produce very long cell strings.

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted ligand-type strings.
    """
    def _format_ligand_pct(g: pd.DataFrame) -> str:
        counts = g[ligand_type_col].value_counts()
        total = counts.sum()
        if total == 0:
            return "N/A"
        parts = [fmt.format(name=name, pct=(cnt / total) * 100) for name, cnt in counts.items()]
        return " / ".join(parts)

    return df_ligands.groupby(rna_col, observed=False).apply(_format_ligand_pct, include_groups=False)


def compute_all_ligand_type_percentages(
    df_ligands      : pd.DataFrame
  , rna_col         : str = "rna_type_cdrna"
  , ligand_type_col : str = "partner_rcsb_type_grouped"
  , identifier_col  : str = "identifier"
  , fmt             : str = "{name}: {pct:.1f}%"
) -> pd.Series:
    def _format_ligand_pct(g: pd.DataFrame) -> str:
        # Deduplicate identifiers within the group before calculating percentages
        unique_g = g[[ligand_type_col, identifier_col]].drop_duplicates()
        counts = unique_g[ligand_type_col].value_counts()
        total = counts.sum()
        if total == 0:
            return "N/A"
        parts = [fmt.format(name=name, pct=(cnt / total) * 100) for name, cnt in counts.items()]
        return " / ".join(parts)

    return df_ligands.groupby(rna_col, observed=False).apply(_format_ligand_pct, include_groups=False)

# ---------------------------------------------------------------------------
# Pair-level metric calculators
# ---------------------------------------------------------------------------

def compute_pair_counts(
    df_pairs      : pd.DataFrame
  , rna_col       : str = "rna_type_cdrna"
) -> pd.Series:
    """
    Compute the total number of pairwise comparisons per RNA type.

    Parameters:

                * df_pairs :  a pandas DataFrame. The pairwise conformer comparisons.
                * rna_col :   a string. The column that stores RNA type annotations.
                              Default is "rna_type_cdrna".

    Returns:

                * pd.Series : A Series indexed by RNA type with pair counts.
    """
    return df_pairs.groupby(rna_col, observed=False).size()


def compute_pairs_per_cluster_stats(
    df_pairs      : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , cluster_col   : str        = "cluster_id"
  , fmt           : str        = "{min:.0f} / {max:.0f} / {avg:.1f}"
) -> pd.Series:
    """
    Compute min / max / avg number of pairwise comparisons per cluster per RNA type.

    Parameters:

                * df_pairs :    a pandas DataFrame. The pairwise conformer comparisons.
                * rna_col :     a string. The column that stores RNA type annotations.
                                Default is "rna_type_cdrna".
                * cluster_col : a string. The column holding cluster identifiers.
                                Default is "cluster_id".
                * fmt :         a string. The output format template for min/max/avg.
                                Default is "{min:.0f} / {max:.0f} / {avg:.1f}".

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted pair-count statistics.
    """
    pair_counts = df_pairs.groupby([rna_col, cluster_col], observed=False).size().reset_index(name="n_pairs")
    grouped = pair_counts.groupby(rna_col, observed=False)["n_pairs"]
    return grouped.apply(lambda s: _format_min_max_avg(s, fmt))


def compute_pair_holo_apo_stats(
    df_pairs      : pd.DataFrame
  , df_conformers : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , cluster_col   : str        = "cluster_id"
  , conformer_id_col : str    = "conformer_id_reduced"
  , pair_id_1_col : str        = "conformer_id_1"
  , pair_id_2_col : str        = "conformer_id_2"
  , holo_col      : str        = "is_holo"
  , fmt           : str        = "{min:.1f} / {max:.1f} / {avg:.1f}"
) -> Dict[str, pd.Series]:
    """
    Compute min / max / avg percentages of apo-apo, apo-holo, and holo-holo
    pairwise comparisons per cluster, stratified by RNA type.

    Parameters:

                * df_pairs :         a pandas DataFrame. The pairwise conformer comparisons.
                * df_conformers :    a pandas DataFrame. The conformer-level metadata.
                * rna_col :          a string. The column that stores RNA type annotations.
                                     Default is "rna_type_cdrna".
                * cluster_col :      a string. The column holding cluster identifiers.
                                     Default is "cluster_id".
                * conformer_id_col : a string. The conformer identifier column in *df_conformers*.
                                     Default is "conformer_id_reduced".
                * pair_id_1_col :    a string. The first conformer identifier column in *df_pairs*.
                                     Default is "conformer_id_1".
                * pair_id_2_col :    a string. The second conformer identifier column in *df_pairs*.
                                     Default is "conformer_id_2".
                * holo_col :         a string. The boolean column indicating holo status in
                                     *df_conformers*. Default is "is_holo".
                * fmt :              a string. The output format template for min/max/avg.
                                     Default is "{min:.1f} / {max:.1f} / {avg:.1f}".

    Observations:

                * Merges *df_pairs* with *df_conformers* twice (once per conformer) to obtain
                  the holo/apo status of each member of the pair.
                * Classifies each pair as ``apo-apo``, ``apo-holo``, or ``holo-holo``.
                  ``apo-holo`` and ``holo-apo`` are treated as the same mixed category.
                * Computes the percentage of each category **per cluster**, then aggregates
                  those percentages by RNA type (min / max / avg across clusters).
                * This cluster-centric approach avoids bias from large clusters dominating
                  the global pair count.

    Returns:

                * Dict[str, pd.Series] : A dictionary with keys ``"apo_apo"``, ``"apo_holo"``,
                  and ``"holo_holo"``. Each value is a Series indexed by RNA type.
    """
    # Build a lightweight lookup: conformer_id -> is_holo
    holo_map = df_conformers.set_index(conformer_id_col)[holo_col]

    df = df_pairs.copy()
    df["holo_1"] = df[pair_id_1_col].map(holo_map)
    df["holo_2"] = df[pair_id_2_col].map(holo_map)

    # Classify each pair
    def _classify_pair(row: pd.Series) -> str:
        h1, h2 = row["holo_1"], row["holo_2"]
        if pd.isna(h1) or pd.isna(h2):
            return "unknown"
        if h1 and h2:
            return "holo_holo"
        if not h1 and not h2:
            return "apo_apo"
        return "apo_holo"

    df["pair_type"] = df.apply(_classify_pair, axis=1)

    # Compute percentages per (RNA type, cluster)
    pair_type_counts = (
        df.groupby([rna_col, cluster_col, "pair_type"], observed=False)
          .size()
          .unstack(fill_value=0)
    )
    pair_totals = pair_type_counts.sum(axis=1)
    pair_type_pct = pair_type_counts.div(pair_totals, axis=0) * 100

    # Aggregate min/max/avg across clusters for each RNA type
    def _agg_pct(series: pd.Series) -> str:
        return _format_min_max_avg(series.dropna(), fmt)

    result = {}
    for pt in ["apo_apo", "apo_holo", "holo_holo"]:
        if pt in pair_type_pct.columns:
            grouped = pair_type_pct[pt].groupby(rna_col, observed=False)
            result[pt] = grouped.apply(_agg_pct)
        else:
            result[pt] = pd.Series(dtype=object)

    return result


# ---------------------------------------------------------------------------
# Resolution (optional enrichment)
# ---------------------------------------------------------------------------

def compute_resolution_stats(
    df_conformers : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , resolution_col: str        = "resolution"
  , fmt           : str        = "{min:.2f} / {max:.2f} / {avg:.2f}"
) -> pd.Series:
    """
    Compute min / max / avg crystallographic resolution per RNA type.

    Parameters:

                * df_conformers :  a pandas DataFrame. The conformer-level dataset.
                * rna_col :        a string. The column that stores RNA type annotations.
                                   Default is "rna_type_cdrna".
                * resolution_col : a string. The column holding resolution values in Å.
                                   Default is "resolution".
                * fmt :            a string. The output format template for min/max/avg.
                                   Default is "{min:.2f} / {max:.2f} / {avg:.2f}".

    Observations:

                * Drops missing resolution values before aggregation.
                * Only conformers with an experimental resolution contribute.

    Returns:

                * pd.Series : A Series indexed by RNA type with formatted resolution statistics.
    """
    grouped = df_conformers.groupby(rna_col, observed=False)[resolution_col]
    return grouped.apply(lambda s: _format_min_max_avg(s.dropna(), fmt))


# ---------------------------------------------------------------------------
# Table assembler
# ---------------------------------------------------------------------------

def build_descriptive_statistics_table(
    df_clusters   : pd.DataFrame
  , df_conformers : pd.DataFrame
  , df_pairs      : pd.DataFrame
  , df_ligands    : pd.DataFrame
  , rna_col       : str        = "rna_type_cdrna"
  , cluster_col   : str        = "cluster_id"
  , ligand_ids_col: str        = "identifier"
  , ligand_type_col : str      = "partner_rcsb_type_grouped"
  , include_resolution : bool   = False
) -> pd.DataFrame:
    """
    Assemble the full descriptive-statistics table stratified by **every**
    RNA type present in the dataset.

    Parameters:

                * df_clusters :        a pandas DataFrame. Cluster-level metadata.
                * df_conformers :      a pandas DataFrame. Conformer-level metadata.
                * df_pairs :           a pandas DataFrame. Pairwise RMSD comparisons.
                * df_ligands :         a pandas DataFrame. Ligand-level annotations.
                * rna_col :            a string. The **single** RNA type column name used
                                       across all four DataFrames. Default is "rna_type_cdrna".
                * cluster_col :        a string. The cluster identifier column name used for
                                       consistency checks. Default is "cluster_id".
                * ligand_ids_col :     a string. The column in *df_conformers* that stores
                                       ligand identifier lists. Default is "ligand_ids".
                * ligand_type_col :    a string. The column in *df_ligands* that stores
                                       ligand type classifications. Default is
                                       "partner_rcsb_type_grouped".
                * include_resolution : a boolean. Whether to append a resolution row
                                       (min / max / avg in Å). Default is False.

    Observations:

                * **Validation layer** — Before any computation, the function verifies:
                  1. That *rna_col* exists in every DataFrame (raises `KeyError` if not).
                  2. That cluster identifiers are consistent across *df_clusters*,
                     *df_conformers*, and *df_pairs* (prints warnings but does not halt).
                  3. That every RNA type found in *df_clusters* also appears in the
                     remaining DataFrames (prints warnings for orphan types).
                * **No binning** — All RNA types are displayed as independent columns.
                  No residual "Other" group is created.
                * Population ordering is determined from *df_conformers* via
                  :func:`get_all_rna_types`.
                * Ligand counts are derived by parsing the list-valued *ligand_ids_col*
                  rather than relying on the pre-computed integer *nb_ligands*.
                * Pair apo/holo statistics are computed **per cluster** first, then
                  aggregated by RNA type, preventing large clusters from dominating
                  the global average.
                * A **"Total"** column is appended at the end, summarising the entire
                  dataset in a single vertical slice.

    Returns:

                * pd.DataFrame : The descriptive statistics table. Columns are all RNA types
                  sorted by descending conformer population, followed by a "Total" column.
                  Rows are descriptive metric labels.
    """
    # ------------------------------------------------------------------
    # 1. Validation
    # ------------------------------------------------------------------
    _validate_rna_column(df_clusters, df_conformers, df_pairs, df_ligands, rna_col=rna_col)
    _validate_cluster_consistency(df_clusters, df_conformers, df_pairs, cluster_col=cluster_col)
    _validate_rna_type_consistency(df_clusters, df_conformers, df_pairs, df_ligands, rna_col=rna_col)

    # ------------------------------------------------------------------
    # 2. Discover RNA types (all of them, sorted by conformer population)
    # ------------------------------------------------------------------
    all_rna_types = get_all_rna_types(df_conformers, rna_col)

    # ------------------------------------------------------------------
    # 3. Compute every metric
    # ------------------------------------------------------------------
    row_cluster_counts = compute_cluster_counts(df_clusters, rna_col)
    row_conformer_counts = compute_conformer_counts(df_conformers, rna_col)
    row_cluster_size_stats = compute_conformers_per_cluster_stats(df_clusters, rna_col)
    row_pair_counts = compute_pair_counts(df_pairs, rna_col)
    row_pairs_per_cluster = compute_pairs_per_cluster_stats(df_pairs, rna_col, cluster_col)
    pair_holo_apo = compute_pair_holo_apo_stats(df_pairs, df_conformers, rna_col, cluster_col)
    row_seqres_stats = compute_seqres_length_stats(df_conformers, rna_col)
    row_rmsd_stats = compute_rmsd_stats(df_pairs, rna_col)
    row_max_rmsd_stats = compute_max_rmsd_per_cluster_stats(df_clusters, rna_col)
    row_apo_stats = compute_apo_stats(df_clusters, rna_col)
    row_holo_stats = compute_holo_stats(df_clusters, rna_col)
    row_method_dist = compute_method_distribution(df_clusters, rna_col)
    row_ligand_counts = compute_ligand_counts_from_df(df_ligands, rna_col, ligand_ids_col)
    # row_ligand_counts = _old_compute_ligand_counts_from_ids(df_conformers, rna_col, ligand_ids_col)
    row_ligand_type_pct = compute_all_ligand_type_percentages(df_ligands, rna_col, ligand_type_col)


    # Optional resolution
    rows: Dict[str, pd.Series] = {
        "# clusters"                                      : row_cluster_counts,
        "# conformers"                                    : row_conformer_counts,
        "conformers per cluster (min / max / avg)"        : row_cluster_size_stats,
        "# pairwise comparisons"                          : row_pair_counts,
        "pairs per cluster (min / max / avg)"             : row_pairs_per_cluster,
        "experimental method (% XRD / % EM / % NMR)"      : row_method_dist,
        "max RMSD per cluster (min / max / avg)"          : row_max_rmsd_stats,
        "pairwise RMSD (min / max / avg)"                 : row_rmsd_stats,
        "sequence length (min / max / avg)"               : row_seqres_stats,
        "# ligands"                                       : row_ligand_counts,
        "ligand type distribution"                        : row_ligand_type_pct,
        "% apo conformers per cluster (min / max / avg)"  : row_apo_stats,
        "% holo conformers per cluster (min / max / avg)" : row_holo_stats,
        "% apo-apo pairs per cluster (min / max / avg)"   : pair_holo_apo["apo_apo"],
        "% apo-holo pairs per cluster (min / max / avg)"  : pair_holo_apo["apo_holo"],
        "% holo-holo pairs per cluster (min / max / avg)" : pair_holo_apo["holo_holo"],
    }

    if include_resolution:
        rows["resolution Å (min / max / avg)"] = compute_resolution_stats(df_conformers, rna_col)

    # ------------------------------------------------------------------
    # 4. Build ordered column index (all types)
    # ------------------------------------------------------------------
    ordered_cols = all_rna_types

    # ------------------------------------------------------------------
    # 5. Assemble DataFrame
    # ------------------------------------------------------------------
    out = pd.DataFrame({k: v.reindex(ordered_cols) for k, v in rows.items()}).T
    out.columns.name = "RNA type"
    out.index.name = "observation"
    out = out.fillna("N/A")

    # ------------------------------------------------------------------
    # 6. Append "Total" column
    # ------------------------------------------------------------------
    def _build_total(col_data: pd.Series) -> str:
        """Infer the total representation for a given metric row."""
        label = col_data.name  # type: ignore
        if label in ("# clusters", "# conformers", "# ligands", "# pairwise comparisons"):
            # Simple sum for count metrics
            numeric = pd.to_numeric(col_data.replace("N/A", np.nan), errors="coerce")
            total = numeric.sum()
            return f"{int(total)}" if not pd.isna(total) else "N/A"

        if "(min / max / avg)" in label:
            # Parse min/max/avg from all RNA types and compute global min/max/avg
            mins, maxs, avgs = [], [], []
            for val in col_data:
                if val == "N/A" or pd.isna(val):
                    continue
                parts = str(val).split(" / ")
                if len(parts) == 3:
                    try:
                        mins.append(float(parts[0]))
                        maxs.append(float(parts[1]))
                        avgs.append(float(parts[2]))
                    except ValueError:
                        continue
            if mins:
                return f"{min(mins):.1f} / {max(maxs):.1f} / {np.mean(avgs):.1f}"
            return "N/A"

        if label == "experimental method (% XRD / % EM / % NMR)":
            # Average the percentages across all RNA types
            xrd_vals, em_vals, nmr_vals = [], [], []
            for val in col_data:
                if val == "N/A" or pd.isna(val):
                    continue
                # Parse "XRD: 20.0 / EM: 3.0 / NMR: 77.0"
                try:
                    xrd = float(val.split("XRD:")[1].split("/")[0].strip())
                    em  = float(val.split("EM:")[1].split("/")[0].strip())
                    nmr = float(val.split("NMR:")[1].strip())
                    xrd_vals.append(xrd)
                    em_vals.append(em)
                    nmr_vals.append(nmr)
                except (IndexError, ValueError):
                    continue
            if xrd_vals:
                return f"XRD: {np.mean(xrd_vals):.1f} / EM: {np.mean(em_vals):.1f} / NMR: {np.mean(nmr_vals):.1f}"
            return "N/A"

        if label == "ligand type distribution":
            # Aggregate all ligand type percentages globally
            type_counts: Dict[str, float] = {}
            for val in col_data:
                if val == "N/A" or pd.isna(val):
                    continue
                parts = val.split(" / ")
                for part in parts:
                    if ":" not in part:
                        continue
                    name, pct_str = part.rsplit(":", 1)
                    name = name.strip()
                    try:
                        pct = float(pct_str.replace("%", "").strip())
                    except ValueError:
                        continue
                    # Weight by conformer count of that RNA type (approximate)
                    # For simplicity, we do unweighted average of percentages here
                    type_counts[name] = type_counts.get(name, 0.0) + pct
            if not type_counts:
                return "N/A"
            total_pct = sum(type_counts.values())
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            parts = [f"{name}: {(pct/total_pct)*100:.1f}%" for name, pct in sorted_types]
            return " / ".join(parts)

        return "N/A"

    total_col = out.apply(_build_total, axis=1)
    out.insert(len(out.columns), "Total", total_col)

    return out


def table_to_markdown(
    df            : pd.DataFrame
  , caption       : str        = "Descriptive dataset statistics"
  , dataset_name  : str        = "final_dataset"
) -> str:
    """
    Convert a descriptive-statistics DataFrame into a publication-ready Markdown string.

    Parameters:

                * df :           a pandas DataFrame. The table produced by
                                 :func:`build_descriptive_statistics_table`.
                * caption :      a string. The table caption text. Default is
                                 "Descriptive dataset statistics".
                * dataset_name : a string. The dataset identifier inserted into the caption.
                                 Default is "final_dataset".

    Returns:

                * str : A Markdown-formatted table preceded by a bold caption line.
    """
    header = f"**Table 1. {caption} ({dataset_name}).**\n\n"
    md = df.to_markdown()
    return header + md