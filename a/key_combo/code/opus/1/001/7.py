import random

import numpy as np


BG = 0
SEP = 5


def _full_sep_col(g: np.ndarray) -> int:
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(cols) == 1, "expected one grey separator column"
    return cols[0]


def _crop_non_bg(panel: np.ndarray) -> np.ndarray:
    pts = np.argwhere(panel != BG)
    assert pts.size > 0, "panel must contain a rectangle"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return panel[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _line_runs(pattern: np.ndarray, axis: int):
    lines = pattern if axis == 0 else pattern.T
    runs = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or not np.array_equal(lines[i], lines[start]):
            runs.append((start, i, tuple(map(int, lines[start]))))
            start = i
    return runs


def _stretch_run_index(runs) -> int:
    color_counts = [len(set(line)) for _, _, line in runs]
    max_colors = max(color_counts)
    candidates = [i for i, count in enumerate(color_counts) if count == max_colors]
    if len(candidates) == 1:
        return candidates[0]
    lengths = [end - start for start, end, _ in runs]
    min_len = min(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == min_len]
    assert len(candidates) == 1, "expected one stretchable internal run"
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    src_len = pattern.shape[axis]
    if src_len == target_length:
        return list(range(src_len))
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed = sum(end - start for i, (start, end, _) in enumerate(runs) if i != stretch_i)
    stretch_len = target_length - fixed
    assert stretch_len > 0, "target is too small for fixed border runs"
    indices = []
    for i, (start, end, _) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch_len)
        else:
            indices.extend(range(start, end))
    assert len(indices) == target_length
    return indices


def _stretch(src: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = _axis_indices(src, 0, shape[0])
    cols = _axis_indices(src, 1, shape[1])
    return src[np.ix_(rows, cols)].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the left exemplar rectangle to the right rectangle size, then output right-side pixel differences.

    Input structure:
    - A black grid has one full-height grey vertical separator column.
    - The left panel contains a smaller bordered multicolor exemplar rectangle.
    - The right panel contains a larger rectangle with the same border color and a few altered pixels.

    Rule:
    - Crop the left exemplar and the right target rectangle.
    - Elastically resize the exemplar to the target shape, preserving first and last rows/columns and stretching interior rows/columns.
    - Compare the stretched exemplar with the target cell by cell.
    - Output black where they match and the target cell color where they differ.

    Key insight:
    - Stretching is an intermediate alignment step; only after alignment can the target differences be extracted.
    """
    g = np.asarray(grid, dtype=int)
    sep = _full_sep_col(g)
    left = _crop_non_bg(g[:, :sep])
    right = _crop_non_bg(g[:, sep + 1 :])
    stretched = _stretch(left, right.shape)
    return np.where(stretched == right, BG, right).astype(int)


def _make_rect(h, w, stretch_axis):
    border = random.choice([1, 2, 3, 4])
    fills = random.sample([c for c in range(1, 10) if c != border and c != SEP], 3)
    rect = np.full((h, w), border, dtype=int)
    if stretch_axis == 1:
        for r in range(1, h - 1):
            rect[r, 1 : w - 1] = fills[(r - 1) % len(fills)]
    else:
        for c in range(1, w - 1):
            rect[1 : h - 1, c] = fills[(c - 1) % len(fills)]
    return rect


def _gen_attempt():
    if random.random() < 0.5:
        h1, w1 = random.randint(4, 6), random.randint(4, 5)
        h2, w2 = h1, random.randint(w1 + 2, min(10, w1 + 5))
        stretch_axis = 1
    else:
        h1, w1 = random.randint(4, 5), random.randint(4, 6)
        h2, w2 = random.randint(h1 + 2, min(10, h1 + 5)), w1
        stretch_axis = 0
    left = _make_rect(h1, w1, stretch_axis)
    right = _stretch(left, (h2, w2))
    colors = [c for c in range(1, 10) if c != SEP]
    for _ in range(random.randint(2, 5)):
        r = random.randint(1, h2 - 2)
        c = random.randint(1, w2 - 2)
        choices = [x for x in colors if x != int(right[r, c])]
        right[r, c] = random.choice(choices)

    H = max(h2, h1) + 4
    W = w1 + w2 + 7
    sep = w1 + 3
    grid = np.zeros((H, W), dtype=int)
    grid[:, sep] = SEP
    lr = random.randint(1, H - h1 - 1)
    rr = random.randint(1, H - h2 - 1)
    grid[lr : lr + h1, 1 : 1 + w1] = left
    grid[rr : rr + h2, sep + 2 : sep + 2 + w2] = right
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
