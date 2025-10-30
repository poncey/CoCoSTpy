import warnings
import numpy as np
from scipy.sparse import issparse
from anndata import AnnData
from ._spatial_neighbour import spatial_affinity
from ._gcpca import gcpca


def cocost(adb: AnnData, 
           adt: AnnData, 
           spaconn_key: str = "CoCoST_affinity", 
           use_genes=None,
           add_keys: str = "X_CoCoST",
           return_weights: bool = False,
           **kwargs):
    """
    Perform CoCo-ST (Contrastive Cross-omics Spatial Transcriptomics) alignment 
    between two AnnData objects using graph-contrastive PCA (gcpca).

    This function computes shared spatial representations between two datasets 
    (e.g., before and after treatment, or two tissue conditions) by integrating 
    both expression and spatial affinity structures.

    Parameters
    ----------
    adb : AnnData
        Reference AnnData object (e.g., baseline condition). 
        Must contain a spatial connectivity matrix in `obsp[spaconn_key]`.
    adt : AnnData
        Target AnnData object (e.g., treatment condition).
        Must contain a spatial connectivity matrix in `obsp[spaconn_key]`.
    spaconn_key : str, optional (default: "CoCoST_affinity")
        Key in `obsp` of each AnnData object storing the spatial adjacency or affinity matrix.
    use_genes : list or array-like, optional
        Subset of gene names or indices to use for the computation. 
        If None, all genes are used.
    add_keys : str, optional (default: "X_CoCoST")
        Key to store the resulting CoCo-ST embeddings in the `obsm` slot of each AnnData.
    return_weights : bool, optional (default: False)
        If True, return the eigenvector weights computed by gcpca.
    **kwargs : dict
        Additional keyword arguments passed to the `gcpca` function.

    Returns
    -------
    None or np.ndarray
        If `return_weights` is True, returns the eigenvectors (weights) from gcpca.
        Otherwise, stores the embeddings directly in the AnnData objects:
            - `adb.obsm[add_keys]` : background (reference) embedding
            - `adt.obsm[add_keys]` : foreground (target) embedding

    Raises
    ------
    RuntimeError
        If the computed components contain NaN values, indicating data quality or 
        numerical instability issues.

    Notes
    -----
    - The method suppresses runtime warnings from gcpca computations.
    - Supports both dense and sparse input matrices; if sparse, converts to dense for subset selection.
    - The resulting embeddings can be used for downstream visualization, integration, 
      or differential spatial pattern analysis.

    Examples
    --------
    >>> cocost(adb, adt, spaconn_key="spatial_affinity", use_genes=marker_genes)
    >>> eigvecs = cocost(adb, adt, return_weights=True)
    """
    Sb, St = adb.obsp[spaconn_key], adt.obsp[spaconn_key]
    Xb, Xt  = adb.X, adt.X
    if use_genes is not None:
        if issparse(Xb) and issparse(Xt):
            Xb, Xt = adb[...,use_genes].X.todense(), adt[...,use_genes].X.todense()
        else:
            Xb, Xt = adb[...,use_genes].X, adt[...,use_genes].X
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", category=RuntimeWarning)
         eigvecs, fg_comp, bg_comp = gcpca(Xt, Xb, St, Sb, **kwargs)
    if np.isnan(fg_comp).any() or np.isnan(bg_comp).any():
        raise RuntimeError("Results contains NaN. Please check the data.")
    adb.obsm[add_keys] = bg_comp
    adt.obsm[add_keys] = fg_comp
    
    if return_weights:
        return eigvecs
