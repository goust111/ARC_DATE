import random
from collections import deque

import numpy as np


BG = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


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
    candidates = [i for i, n in enumerate(color_counts) if n == max_colors]
    if len(candidates) == 1:
        return candidates[0]
    lengths = [end - start for start, end, _ in runs]
    min_len = min(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == min_len]
    assert len(candidates) == 1
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_len: int) -> list[int]:
    src_len = pattern.shape[axis]
    if src_len == target_len:
        return list(range(src_len))
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed = sum(end - start for i, (start, end, _) in enumerate(runs) if i != stretch_i)
    stretch_len = target_len - fixed
    assert stretch_len > 0
    idx = []
    for i, (start, end, _) in enumerate(runs):
        if i == stretch_i:
            idx.extend([start] * stretch_len)
        else:
            idx.extend(range(start, end))
    assert len(idx) == target_len
    return idx


def _stretch(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = _axis_indices(pattern, 0, shape[0])
    cols = _axis_indices(pattern, 1, shape[1])
    return pattern[np.ix_(rows, cols)].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use same-color corner pairs to instantiate hidden rectangles from a run-
    stretched exemplar; anti-diagonal pairs receive a horizontal mirror first.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    exemplar = None
    exemplar_bbox = None
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if len(comp) == patch.size and len(np.unique(patch)) > 1 and patch.size >= 9:
            exemplar = patch.copy()
            exemplar_bbox = (r0, c0, r1, c1)
            break
    assert exemplar is not None
    border = int(exemplar[0, 0])

    ex_cells = {
        (r, c)
        for r in range(exemplar_bbox[0], exemplar_bbox[2] + 1)
        for c in range(exemplar_bbox[1], exemplar_bbox[3] + 1)
    }
    marker_positions = {}
    for color in sorted(set(map(int, g.ravel())) - {BG}):
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color)) if (int(r), int(c)) not in ex_cells]
        if pts:
            assert len(pts) == 2, "marker colors must occur as point pairs"
            marker_positions[color] = pts

    out = g.copy()
    for color, pts in marker_positions.items():
        (r0, c0), (r1, c1) = sorted(pts)
        top, bottom = min(r0, r1), max(r0, r1)
        left, right = min(c0, c1), max(c0, c1)
        patch = _stretch(exemplar, (bottom - top + 1, right - left + 1))
        anti_diag = set(pts) == {(top, right), (bottom, left)}
        if anti_diag:
            patch = np.fliplr(patch)
        patch = patch.copy()
        patch[patch == border] = color
        out[top : bottom + 1, left : right + 1] = patch
    return out


def _make_exemplar(border: int) -> np.ndarray:
    colors = random.sample([c for c in range(1, 10) if c not in (border,)], 8)
    p = np.full((5, 5), border, dtype=int)
    p[1, 1:4] = [colors[0], colors[1], colors[0]]
    p[2, 1:4] = [colors[2], colors[3], colors[4]]
    p[3, 1:4] = [colors[5], colors[6], colors[5]]
    return p


def _rect_free(used, top, left, h, w):
    cells = {(r, c) for r in range(top, top + h) for c in range(left, left + w)}
    expanded = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    return not (expanded & used)


def _gen_attempt():
    h = w = random.randint(23, 29)
    grid = np.zeros((h, w), dtype=int)
    border = random.choice([1, 2, 3, 4])
    exemplar = _make_exemplar(border)
    er = random.randint(1, 4)
    ec = random.randint(1, 4)
    grid[er : er + 5, ec : ec + 5] = exemplar
    used = {(r, c) for r in range(er, er + 5) for c in range(ec, ec + 5)}

    marker_colors = random.sample([6, 7, 8, 9, 5], random.randint(2, 4))
    for color in marker_colors:
        for _ in range(200):
            rh = random.randint(5, 9)
            rw = random.randint(5, 9)
            top = random.randint(1, h - rh - 1)
            left = random.randint(1, w - rw - 1)
            if not _rect_free(used, top, left, rh, rw):
                continue
            anti = random.random() < 0.5
            pts = [(top, left), (top + rh - 1, left + rw - 1)]
            if anti:
                pts = [(top, left + rw - 1), (top + rh - 1, left)]
            for r, c in pts:
                grid[r, c] = color
            used |= {(r, c) for r in range(top, top + rh) for c in range(left, left + rw)}
            break
        else:
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
