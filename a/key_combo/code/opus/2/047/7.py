import numpy as np


def _panel_slices(grid):
    h, w = grid.shape
    sep = np.where(np.all(grid == 5, axis=0))[0]
    cuts = [0]
    for c in sep:
        cuts.append(c)
        cuts.append(c + 1)
    cuts.append(w)
    cuts = sorted(set(cuts))
    panels = []
    starts = []
    for i in range(len(cuts) - 1):
        if cuts[i] >= cuts[i + 1]:
            continue
        seg = grid[:, cuts[i] : cuts[i + 1]]
        if seg.shape[1] == 0:
            continue
        # skip separator columns
        if np.all(seg == 5):
            continue
        if np.any(seg != 5):
            panels.append(seg)
            starts.append(cuts[i])
    return panels, sep


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    panels, sep_cols = _panel_slices(g)
    if len(panels) == 0:
        return np.zeros((1, 1), dtype=np.int64)

    counts = []
    for p in panels:
        counts.append(int(np.sum(p == 2)))
    idx = int(np.argmax(counts))

    chosen = panels[idx].copy()
    rot = counts[idx] % 4
    out = np.rot90(chosen, -rot)
    out = out.astype(np.int64)
    out[out == 2] = 0
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 7
    w = 5
    total_w = 3 * w + 2
    grid = np.zeros((h, total_w), dtype=np.int64)
    grid[:, w] = 5
    grid[:, 2 * w + 1] = 5

    counts = [1, 2, 3]
    while True:
        rng.shuffle(counts)
        if len(set(counts)) == len(counts):
            break

    starts = [0, w + 1, 2 * w + 2]
    palette = [8]

    for i, s in enumerate(starts):
        panel_w = w
        pts = set()
        start = (int(rng.integers(0, h)), int(rng.integers(0, panel_w)))
        rr, cc = start
        pts.add((rr, cc))
        for _ in range(7):
            rr = int(np.clip(rr + int(rng.choice([-1, 0, 1])), 0, h - 1))
            cc = int(np.clip(cc + int(rng.choice([-1, 0, 1])), 0, panel_w - 1))
            pts.add((rr, cc))
        for r, c in pts:
            grid[r, s + c] = 8
        color_cells = list(pts)
        if len(color_cells) == 0:
            continue
        k = counts[i]
        chosen = color_cells[: int(k)]
        for r, c in chosen:
            grid[r, s + c] = 2
    return grid

