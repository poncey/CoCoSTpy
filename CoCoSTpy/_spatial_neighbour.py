import time
import numpy as np
from typing import Literal, Optional
from anndata._core.anndata import AnnData
from scipy.sparse import isspmatrix, csr_matrix, coo_matrix
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def spatial_affinity(adata: AnnData,
                     coord_key: str = "spatial",
                     add_key: str = "CoCoST_affinity",
                     scale_factor: Optional[float] = None,
                     **kwargs):
    spa_conn = spatial_neighbours(adata, coord_key=coord_key, key_added=None, **kwargs)
    spa_affi = _to_cocost_affinity(spa_conn, adata.obsm[coord_key], scale_factor, gamma=1.0)
    if add_key is None:
        return spa_affi
    adata.obsp[add_key] = spa_affi


def spatial_neighbours(adata: AnnData,
                       coord_key="spatial",
                       key_added="spatialCONN",
                       **kwargs,
                       ):
    """Compute spatial neighbours for giving a list of Anndata

    Args:
        adatalist (Union[AnnData, List[AnnData]]): (List of) anndata
        coord_key (str, optional): '.obsm' key for saving coordinates. 
                                   All adata should have same keys. Defaults to "spatial".
        key_added (str, optional): Keys added to '.obsp'. Defaults to "spatialCONN".
        verbose (bool, optional): Show verbose. Defaults to True.

    Returns:
        Add the spatial neighbours into adata(s). if 'key_added' is None, retrun list of spa_conn.
        _type_: _description_
    """
    coord = adata.obsm[coord_key]
    spa_conn = _spatial_neighbours(coord, **kwargs)

    if key_added is None:
        return spa_conn
    adata.obsp[key_added] = spa_conn


def _spatial_neighbours(coords,
                        delaunay: bool = False,
                        k_neighs: int = 10,
                        radius: float = None,
                        mode: Literal["connectivity",
                                      "distance"] = "connectivity",
                        delaunay_solver="convex",):
    if delaunay:
        conn = __calculate_delaunay(
            coords, output_mode="adjacent_sparse", strategy_t=delaunay_solver)
    else:
        if radius is not None:
            conn = radius_neighbors_graph(coords, radius, mode=mode)
        else:
            conn = kneighbors_graph(coords, k_neighs, mode=mode)
    return conn


def __calculate_delaunay(coords, output_mode='adjacent_sparse', strategy_t='convex'):
    if strategy_t == 'convex':  # slow but may generate more reasonable delaunay graph
        import networkx as nx
        from libpysal.cg import voronoi_frames
        from libpysal import weights
        cells, _ = voronoi_frames(
            coords, clip="convex_hull", as_gdf=True, return_input=True)
        delaunay_graph = weights.Rook.from_dataframe(
            cells, use_index=True).to_networkx()
    elif strategy_t == 'delaunay':  # fast but may generate long distance edges
        from scipy.spatial import Delaunay
        from collections import defaultdict
        tri = Delaunay(coords)
        delaunay_graph = nx.Graph()
        coords_dict = defaultdict(list)
        for i, coord in enumerate(coords):
            coords_dict[tuple(coord)].append(i)
        for simplex in tri.simplices:
            for i in range(3):
                for node1 in coords_dict[tuple(coords[simplex[i]])]:
                    for node2 in coords_dict[tuple(coords[simplex[(i+1) % 3]])]:
                        if not delaunay_graph.has_edge(node1, node2):
                            delaunay_graph.add_edge(node1, node2)
    if output_mode == 'adjacent':
        return nx.to_scipy_sparse_array(delaunay_graph).todense()
    elif output_mode == 'raw':
        return delaunay_graph
    elif output_mode == 'adjacent_sparse':
        return nx.to_scipy_sparse_array(delaunay_graph)
def _to_cocost_affinity(spaconn: csr_matrix,
                        pos: np.ndarray,
                        scale_factor: Optional[float] = None,
                        **kwargs):
    shape = spaconn.shape
    spaconn = spaconn.tocoo()
    row, col = spaconn.row, spaconn.col

    if scale_factor is not None:
        pos = pos / scale_factor
    pos = np.asarray(pos)

    expression = kwargs.pop("expression", None)
    if expression is None:
        expression = kwargs.pop("expr", None)
    expression_scale = kwargs.pop("expression_scale", None)
    expression_scale = kwargs.pop("expr_scale", expression_scale)

    combine = kwargs.pop("combine", None)
    if combine is None:
        combine = kwargs.pop("combination", "product")

    pos_weight = kwargs.pop("pos_weight", kwargs.pop("spatial_weight", 1.0))
    expr_weight = kwargs.pop("expr_weight", kwargs.pop("expression_weight", 1.0))

    pos_kwargs = kwargs.pop("pos_kwargs", None) or {}
    expr_kwargs = kwargs.pop("expr_kwargs", None) or {}

    gamma = kwargs.pop("gamma", None)
    if gamma is not None:
        pos_kwargs.setdefault("gamma", gamma)
        expr_kwargs.setdefault("gamma", gamma)

    pos_gamma = kwargs.pop("pos_gamma", None)
    if pos_gamma is not None:
        pos_kwargs["gamma"] = pos_gamma

    expr_gamma = kwargs.pop("expr_gamma", kwargs.pop("expression_gamma", None))
    if expr_gamma is not None:
        expr_kwargs["gamma"] = expr_gamma

    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    pos_data = pos_weight * _rbf_kernel_paired(pos[row], pos[col], **pos_kwargs)

    if expression is not None:
        if isspmatrix(expression):
            expr_matrix = expression.tocsr()
            if expression_scale is not None:
                if np.isscalar(expression_scale):
                    expr_matrix = expr_matrix.copy()
                    expr_matrix.data /= expression_scale
                else:
                    raise ValueError(
                        "expression_scale must be a scalar when expression is sparse."
                    )
            expr_row = expr_matrix[row]
            expr_col = expr_matrix[col]
        else:
            expr_matrix = np.asarray(expression)
            if expression_scale is not None:
                expr_matrix = expr_matrix / expression_scale
            expr_row = expr_matrix[row]
            expr_col = expr_matrix[col]

        expr_data = expr_weight * _rbf_kernel_paired(expr_row, expr_col, **expr_kwargs)

        mode = (combine or "product").lower()
        if mode in ("product", "multiply", "hadamard"):
            data = pos_data * expr_data
        elif mode in ("sum", "add"):
            data = pos_data + expr_data
        elif mode in ("mean", "average"):
            data = 0.5 * (pos_data + expr_data)
        elif mode == "max":
            data = np.maximum(pos_data, expr_data)
        else:
            raise ValueError(f"Unsupported combine mode '{combine}'.")
    else:
        data = pos_data

    data = np.asarray(data, dtype=np.float64)
    return coo_matrix((data, (row, col)), shape=shape).tocsr()


def _rbf_kernel_paired(X, Y, gamma=1.0):
    if isspmatrix(X) or isspmatrix(Y):
        if not (isspmatrix(X) and isspmatrix(Y)):
            raise TypeError("X and Y must both be sparse matrices or both be dense.")
        diff = X - Y
        dist_sq = np.asarray(diff.multiply(diff).sum(axis=1)).ravel()
    else:
        diff = X - Y
        dist_sq = (diff * diff).sum(axis=1)
    return np.exp(-gamma * dist_sq)
