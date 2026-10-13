import random
from collections import Counter

import numpy as np


PATTERN_LIBRARY = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
]

MAX_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    best = None
    best_area = -1
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            pts = np.array(comp, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height < 5 or width < 5:
                continue
            border_area = 2 * (height + width) - 4
            if len(comp) != border_area:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((height, width), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if not np.all(sub[~border] == bg):
                continue
            area = height * width
            if area > best_area:
                best_area = area
                best = (color, int(r0), int(c0), int(r1), int(c1))
    assert best is not None, "expected one rectangular frame"
    return best


def _extract_markers(
    grid: np.ndarray, bg: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> tuple[dict[int, int], dict[int, int], set[tuple[int, int]]]:
    top_markers: dict[int, int] = {}
    left_markers: dict[int, int] = {}
    marker_cells: set[tuple[int, int]] = set()

    assert fr0 > 0 and fc0 > 0, "frame must have room for external markers"

    for c in range(fc0 + 1, fc1):
        value = int(grid[fr0 - 1, c])
        if value == bg:
            continue
        assert value not in top_markers, "top marker colors must be unique"
        top_markers[value] = c
        marker_cells.add((fr0 - 1, c))

    for r in range(fr0 + 1, fr1):
        value = int(grid[r, fc0 - 1])
        if value == bg:
            continue
        assert value not in left_markers, "left marker colors must be unique"
        left_markers[value] = r
        marker_cells.add((r, fc0 - 1))

    assert top_markers and left_markers, "expected both top and left marker sets"
    return top_markers, left_markers, marker_cells


def _extract_patterns(
    grid: np.ndarray,
    bg: int,
    top_colors: set[int],
    frame_bbox: tuple[int, int, int, int],
    marker_cells: set[tuple[int, int]],
) -> dict[int, np.ndarray]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ignore = np.zeros_like(grid, dtype=bool)
    ignore[fr0 : fr1 + 1, fc0 : fc1 + 1] |= False
    ignore[fr0, fc0 : fc1 + 1] = True
    ignore[fr1, fc0 : fc1 + 1] = True
    ignore[fr0 : fr1 + 1, fc0] = True
    ignore[fr0 : fr1 + 1, fc1] = True
    for r, c in marker_cells:
        ignore[r, c] = True

    remaining = (grid != bg) & ~ignore
    patterns: dict[int, np.ndarray] = {}
    for comp in _components(remaining):
        counts = Counter(int(grid[r, c]) for r, c in comp)
        label_candidates = [color for color, n in counts.items() if n == 1 and color in top_colors]
        if len(label_candidates) != 1:
            continue
        label_color = label_candidates[0]
        label_cell = next((r, c) for r, c in comp if int(grid[r, c]) == label_color)
        shape_cells = [(r, c) for r, c in comp if (r, c) != label_cell]
        assert shape_cells, "legend entry must contain a shape"
        rs = [r for r, _ in shape_cells]
        cs = [c for _, c in shape_cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in shape_cells:
            mask[r - r0, c - c0] = True
        patterns[label_color] = mask

    assert set(patterns) == top_colors, "every top color must have one legend entry"
    return patterns


def _stamp_mask(
    out: np.ndarray,
    mask: np.ndarray,
    center_r: int,
    center_c: int,
    color: int,
    frame_bbox: tuple[int, int, int, int],
) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    h, w = mask.shape
    r0 = center_r - h // 2
    c0 = center_c - w // 2
    for rr in range(h):
        for cc in range(w):
            if not mask[rr, cc]:
                continue
            r = r0 + rr
            c = c0 + cc
            assert fr0 < r < fr1 and fc0 < c < fc1, "pattern must stay inside the frame"
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a row/column coordinate system from frame-adjacent color ticks and
    stamp legend-defined shapes at the resulting intersections.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    _frame_color, fr0, fc0, fr1, fc1 = _find_frame(g, bg)
    top_markers, left_markers, marker_cells = _extract_markers(g, bg, fr0, fc0, fr1, fc1)
    patterns = _extract_patterns(g, bg, set(top_markers), (fr0, fc0, fr1, fc1), marker_cells)

    out = g.copy()
    for top_color, center_c in top_markers.items():
        mask = patterns[top_color]
        for left_color, center_r in left_markers.items():
            if top_color == left_color:
                continue
            _stamp_mask(out, mask, center_r, center_c, left_color, (fr0, fc0, fr1, fc1))
    return out


def _pick_spaced(rng: random.Random, candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    for _ in range(200):
        picks = sorted(rng.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _build_legend_entry(
    rng: random.Random,
    grid: np.ndarray,
    bg: int,
    top: int,
    left: int,
    mask: np.ndarray,
    shape_color: int,
    label_color: int,
) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w + 1 > grid.shape[1]:
        return False
    shape_patch = grid[top : top + h, left : left + w]
    if np.any(shape_patch != bg):
        return False
    for rr in range(h):
        for cc in range(w):
            if mask[rr, cc]:
                grid[top + rr, left + cc] = shape_color
    contact_rows = [rr for rr in range(h) if mask[rr, w - 1]]
    if not contact_rows:
        return False
    label_r = top + rng.choice(contact_rows)
    label_c = left + w
    if grid[label_r, label_c] != bg:
        return False
    grid[label_r, label_c] = label_color
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]

    n_cols = rng.randint(2, 3)
    n_rows = rng.randint(2, 3)
    overlap = 1 if min(n_cols, n_rows) == 2 else rng.randint(1, min(n_cols, n_rows) - 1)

    top_colors = rng.sample(palette, n_cols)
    overlap_colors = rng.sample(top_colors, overlap)
    left_extra_pool = [color for color in palette if color not in top_colors]
    left_colors = overlap_colors + rng.sample(left_extra_pool, n_rows - overlap)
    rng.shuffle(left_colors)

    used_marker_colors = set(top_colors) | set(left_colors)
    remaining = [color for color in palette if color not in used_marker_colors]
    if len(remaining) < n_cols + 1:
        return None
    frame_color = rng.choice(remaining)
    remaining = [color for color in remaining if color != frame_color]
    shape_colors = rng.sample(remaining, n_cols)
    pattern_masks = rng.sample(PATTERN_LIBRARY, n_cols)

    legend_gap = 1
    legend_h = n_cols * 3 + (n_cols - 1) * legend_gap
    legend_w = 4
    legend_r0 = rng.randint(1, 2)
    legend_c0 = rng.randint(1, 2)

    inner_h = 3 + (n_rows - 1) * 3 + rng.randint(0, 2)
    inner_w = 3 + (n_cols - 1) * 3 + rng.randint(0, 2)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    fr0 = legend_r0 + legend_h + rng.randint(2, 4)
    fc0 = legend_c0 + legend_w + rng.randint(2, 5)
    height = fr0 + frame_h + rng.randint(1, 3)
    width = fc0 + frame_w + rng.randint(1, 3)

    if not (18 <= height <= 30 and 18 <= width <= 30):
        return None

    grid = np.full((height, width), bg, dtype=int)

    legend_rows = [legend_r0 + i * (3 + legend_gap) for i in range(n_cols)]
    for row, mask, shape_color, label_color in zip(legend_rows, pattern_masks, shape_colors, top_colors):
        if not _build_legend_entry(rng, grid, bg, row, legend_c0, mask, shape_color, label_color):
            return None

    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    grid[fr0, fc0 : fc1 + 1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color

    row_candidates = list(range(fr0 + 2, fr1 - 1))
    col_candidates = list(range(fc0 + 2, fc1 - 1))
    row_centers = _pick_spaced(rng, row_candidates, n_rows, 3)
    col_centers = _pick_spaced(rng, col_candidates, n_cols, 3)
    if row_centers is None or col_centers is None:
        return None

    rng.shuffle(top_colors)
    rng.shuffle(left_colors)
    for color, c in zip(top_colors, col_centers):
        grid[fr0 - 1, c] = color
    for color, r in zip(left_colors, row_centers):
        grid[r, fc0 - 1] = color

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid sample")
