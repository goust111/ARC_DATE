import numpy as np

from arc_utils import cc8 as _cc8, get_bg as _bg_color


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the parallelogram-shaped region between parallel anti-diagonal line segments with red.

    Input structure:
    - bg = most common color.
    - One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).
    - Segments come in parallel pairs bounding a region to be filled.

    Rule:
    - For each main-diagonal axis (r−c = const), collect anti-diagonal segments whose span covers that axis.
    - Sort those segments by their anti-diagonal index (r+c).
    - Fill all bg cells lying strictly between consecutive segment pairs along that axis with red (color 2).

    Key insight: The two diagonal directions decouple — segments sit on anti-diagonals, and filling sweeps along main diagonals between them, forming parallelogram regions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    if not non_bg:
        return g.copy()

    # This task uses one diagonal-line color and fills with 2.
    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2

    comps = _cc8(g == line)

    # Each diagonal segment is on a constant anti-diagonal sum k=r+c,
    # and spans a continuous range of main-diagonal indices b=r-c.
    segs: list[tuple[int, int, int]] = []
    for pts in comps:
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))

    out = g.copy()

    # Scan each main diagonal b=r-c. For every b, find the active diagonal segments
    # (those whose b-range covers b), sort them by k=r+c, then fill between adjacent
    # segments for cells on this b with k strictly between.
    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        ks = [t[0] for t in active]

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for k_left, k_right in zip(ks, ks[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill

    return out
