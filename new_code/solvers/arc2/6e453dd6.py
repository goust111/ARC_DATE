import numpy as np

from arc_utils import components_mask

def _components4(mask):
    return components_mask(mask, connectivity=4)
def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _find_separator_col(g: np.ndarray, sep_color: int = 5) -> int:
    h, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    if cols:
        return int(cols[0])
    # Fallback: column with the most sep_color pixels.
    counts = [(int((g[:, c] == sep_color).sum()), c) for c in range(w)]
    counts.sort(reverse=True)
    return int(counts[0][1])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if c != sep_col and g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if c != sep_col and g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    if border:
        return _mode(np.array(border, dtype=int))
    return _mode(g[g != sep_color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to a grey separator, then shoot red rays from enclosed holes.

    Input structure:
    - A vertical grey (5) separator column divides the grid into left and right halves.
    - Black shapes sit on the left side against a background color (bg).

    Rule:
    - Translate each black connected component rightward until its right edge is flush with the separator.
    - After sliding, find enclosed bg regions (holes) within the black shapes that touch the column just left of the separator.
    - For each row spanned by such a hole, emit a red ray across the entire right half of the grid.

    Key insight: Only holes adjacent to the separator trigger rays — the sliding creates enclosures that act as "nozzles" projecting rightward.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
    left_mask = left == black
    comps = _components4(left_mask)

    # Clear old black pixels (leave other colors intact).
    left[left == black] = bg

    target_right = sep - 1  # last column in left side, adjacent to separator
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        if delta <= 0:
            delta = 0
        for r, c in comp:
            nc = c + delta
            if 0 <= nc < sep:
                left[r, nc] = black

    out[:, :sep] = left

    # Step 2: find enclosed background regions (holes) on the left.
    # A hole is a bg component not touching the left subgrid boundary.
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    # Step 3: if a hole touches column sep-2 (one cell left of the black border that touches separator),
    # emit a red ray on the right side for each row containing hole pixels.
    trigger_col = sep - 2
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        rows = sorted(set(r for r, _ in pts))
        for r in rows:
            for c in range(sep + 1, w):
                if out[r, c] == bg:
                    out[r, c] = red

    return out
