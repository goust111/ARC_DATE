import numpy as np

from arc_utils import bg_boundary as _bg_boundary, cc4 as _cc4


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]] | None:
    """
    The legend is a top-left corner region whose bottom edge and right edge are the same color x.
    We detect the smallest size sz such that:
      - g[sz-1, 0:sz] are all x
      - g[0:sz, sz-1] are all x
    Then the legend color set is the set of colors in the interior (0:sz-1, 0:sz-1),
    excluding bg and excluding the border color x.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(x)
            return int(sz), int(x), colors
    return None


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose colors match a legend swatch.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a small square legend region bordered on its bottom and right edges by a single color; the interior holds a set of "target" colors.
    - Elsewhere: multicolored objects (4-connected components) on the bg.

    Rule:
    - Read the target color set from inside the legend border.
    - For each connected component, check whether it contains **all** target colors.
    - Erase (fill with bg) every component that does; leave all others unchanged.
    - Output grid is the same size.

    Key insight: The legend acts as a filter specification — only components carrying the full color palette shown in the legend are removed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    legend = _find_legend(g, bg)
    if legend is None:
        return g.copy()
    _sz, _x, legend_colors = legend
    if not legend_colors:
        return g.copy()

    out = g.copy()
    comps = _cc4(g != bg)
    for pts in comps:
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            for r, c in pts:
                out[r, c] = bg
    return out
