import numpy as np


BLACK = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint line crossings so the color priority from the right-edge legend determines which segment stays visible.

    Input structure:
    - Black background with one straight horizontal or vertical segment for each legend color.
    - Segments may cross, and the visible crossing color in the input can disagree with the intended priority order.
    - The rightmost column contains one singleton legend marker per line color from highest to lowest priority.

    Rule:
    - Read the legend from top to bottom to obtain the correct priority order.
    - For each color, reconstruct its full segment span from the visible same-colored cells outside the legend column.
    - Redraw all segments from lowest priority to highest priority, so higher-priority colors overwrite lower-priority ones at crossings.

    Key insight:
    - The input crossings are not authoritative; the true layering comes from the edge legend, so the hidden full spans must be inferred before repainting.
    """
    out = np.zeros_like(grid)
    legend = [int(color) for color in grid[:, -1] if color != BLACK]
    for r, color in enumerate(grid[:, -1]):
        if color != BLACK:
            out[r, -1] = color
    for color in reversed(legend):
        pts = np.argwhere(grid[:, :-1] == color)
        rows = np.unique(pts[:, 0])
        cols = np.unique(pts[:, 1])
        if len(rows) == 1 or (len(rows) <= len(cols) and pts[:, 1].ptp() >= pts[:, 0].ptp()):
            r = int(round(pts[:, 0].mean()))
            c0 = int(pts[:, 1].min())
            c1 = int(pts[:, 1].max())
            out[r, c0 : c1 + 1] = color
        else:
            c = int(round(pts[:, 1].mean()))
            r0 = int(pts[:, 0].min())
            r1 = int(pts[:, 0].max())
            out[r0 : r1 + 1, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    n = int(rng.integers(4, 6))
    colors = list(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], size=n, replace=False))
    grid = np.zeros((24, 24), dtype=int)
    horiz_rows = rng.choice(np.arange(3, 18, 3), size=(n + 1) // 2, replace=False)
    vert_cols = rng.choice(np.arange(3, 20, 3), size=n // 2, replace=False)
    spans = []
    idx_h = idx_v = 0
    for i, color in enumerate(colors):
        if i % 2 == 0:
            r = int(horiz_rows[idx_h])
            idx_h += 1
            c0 = int(rng.integers(1, 6))
            c1 = int(rng.integers(14, 21))
            spans.append(("h", color, r, c0, c1))
        else:
            c = int(vert_cols[idx_v])
            idx_v += 1
            r0 = int(rng.integers(1, 6))
            r1 = int(rng.integers(14, 21))
            spans.append(("v", color, c, r0, r1))
    # Draw correct layering first.
    for kind, color, a, b, c in spans[::-1]:
        if kind == "h":
            grid[a, b : c + 1] = color
        else:
            grid[b : c + 1, a] = color
    # Corrupt some crossings.
    for i, (kind1, color1, a1, b1, c1) in enumerate(spans):
        if kind1 != "h":
            continue
        for kind2, color2, a2, b2, c2 in spans:
            if kind2 != "v":
                continue
            if b1 <= a2 <= c1 and b2 <= a1 <= c2:
                if rng.random() < 0.7:
                    grid[a1, a2] = rng.choice([color1, color2])
    for idx, color in enumerate(colors):
        grid[2 + 4 * idx, -1] = color
    return grid
