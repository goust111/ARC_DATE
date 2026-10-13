import random
from collections import Counter

import numpy as np


BG = 0
FENCE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
CELL_SIZE = 4
STEP = CELL_SIZE + 1
ROW_ORIENTATIONS = ("h", "v")
GEN_RETRIES = 400

PATTERN_LIBRARY = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _most_common_non_bg(grid: np.ndarray, bg: int) -> int:
    vals = grid[grid != bg]
    assert vals.size > 0
    values, counts = np.unique(vals, return_counts=True)
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_simple_frame(subgrid: np.ndarray, fence: int) -> bool:
    h, w = subgrid.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(subgrid[border] == fence) and np.all(subgrid[~border] != fence))


def _frame_components(
    grid: np.ndarray, fence: int
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    dictionary_bbox: tuple[int, int, int, int] | None = None
    region_bboxes: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == fence):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if _is_simple_frame(sub, fence):
            region_bboxes.append((r0, c0, r1, c1))
            continue
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border] == fence) and np.any(sub[1:-1, 1:-1] == fence):
            assert dictionary_bbox is None, "expected one dictionary component"
            dictionary_bbox = (r0, c0, r1, c1)
    assert dictionary_bbox is not None, "dictionary component not found"
    assert region_bboxes, "seed regions not found"
    region_bboxes.sort()
    return dictionary_bbox, region_bboxes


def _intervals(full_lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(full_lines, full_lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b))
    return spans


def _parse_dictionary(
    grid: np.ndarray, dictionary_bbox: tuple[int, int, int, int], bg: int, fence: int
) -> dict[str, dict[int, np.ndarray]]:
    r0, c0, r1, c1 = dictionary_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    full_rows = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == fence)]
    full_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == fence)]
    row_spans = _intervals(full_rows)
    col_spans = _intervals(full_cols)
    assert len(row_spans) == 2, "dictionary must have two pattern rows"
    assert len(col_spans) >= 2, "dictionary must have at least two colors"

    patterns: dict[str, dict[int, np.ndarray]] = {"h": {}, "v": {}}
    for row_idx, orient in enumerate(ROW_ORIENTATIONS):
        rr0, rr1 = row_spans[row_idx]
        for cc0, cc1 in col_spans:
            cell = sub[rr0:rr1, cc0:cc1]
            assert cell.shape == (CELL_SIZE, CELL_SIZE), "unexpected dictionary cell size"
            color = int(cell[0, 0])
            assert color not in (bg, fence), "sample pixel missing"
            pattern = cell[1:, 1:] == color
            assert np.any(pattern) and not np.all(pattern), "pattern must be non-trivial"
            patterns[orient][color] = pattern.copy()

    assert patterns["h"].keys() == patterns["v"].keys()
    return patterns


def _region_seed(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int, fence: int
) -> tuple[int, str]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(v) for v in np.unique(interior) if int(v) not in (bg, fence)]
    assert len(colors) == 1, "region must contain one seed color"
    color = colors[0]
    pts = np.argwhere(interior == color)
    assert len(pts) == 2, "seed must have two pixels"
    (r_a, c_a), (r_b, c_b) = pts.tolist()
    assert abs(r_a - r_b) + abs(c_a - c_b) == 1, "seed pixels must be adjacent"
    orient = "h" if r_a == r_b else "v"
    return color, orient


def _tile_pattern(
    out: np.ndarray,
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    pattern: np.ndarray,
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    ph, pw = pattern.shape
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if int(grid[rr, cc]) != bg:
                continue
            if pattern[(rr - (r0 + 1)) % ph, (cc - (c0 + 1)) % pw]:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed seed region with the dictionary pattern selected by the seed's orientation.

    Input structure:
    - The dominant background color surrounds several grey rectangular fences.
    - One fenced component is a dictionary at a fixed corner, split into two rows and multiple columns by grey divider lines.
    - Each dictionary cell contains one color sample pixel plus a small two-color pattern for that color.
    - Every other fenced rectangle encloses only background and one 2-pixel same-color seed, oriented horizontally or vertically.

    Rule:
    - Parse the dictionary: the top row stores the pattern for horizontal seeds and the bottom row stores the pattern for vertical seeds.
    - For each ordinary fenced rectangle, read the seed color and whether its two pixels are horizontal or vertical.
    - Select the same-color pattern from the matching dictionary row.
    - Tile that pattern across the rectangle interior, repainting only background cells with the seed color while preserving the seed itself.

    Key insight:
    - Color alone is insufficient because every color has two candidate patterns; the 2-pixel seed orientation is the hidden key that selects the top-row or bottom-row dictionary entry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fence = _most_common_non_bg(g, bg)
    dictionary_bbox, region_bboxes = _frame_components(g, fence)
    patterns = _parse_dictionary(g, dictionary_bbox, bg, fence)

    out = g.copy()
    for bbox in region_bboxes:
        color, orient = _region_seed(g, bbox, bg, fence)
        assert color in patterns[orient], "missing dictionary entry for seed color"
        _tile_pattern(out, g, bbox, color, patterns[orient][color], bg)
    return out


def _dictionary_patch(
    colors: list[int], patterns_h: dict[int, np.ndarray], patterns_v: dict[int, np.ndarray]
) -> np.ndarray:
    width = 1 + STEP * len(colors)
    height = 1 + STEP * 2
    patch = np.full((height, width), BG, dtype=int)
    for r in range(0, height, STEP):
        patch[r, :] = FENCE
    patch[-1, :] = FENCE
    for c in range(0, width, STEP):
        patch[:, c] = FENCE
    patch[:, -1] = FENCE

    for row_idx, orient in enumerate(ROW_ORIENTATIONS):
        row0 = 1 + row_idx * STEP
        for col_idx, color in enumerate(colors):
            col0 = 1 + col_idx * STEP
            patch[row0, col0] = color
            pattern = patterns_h[color] if orient == "h" else patterns_v[color]
            patch[row0 + 1 : row0 + CELL_SIZE, col0 + 1 : col0 + CELL_SIZE] = np.where(
                pattern, color, BG
            )
    return patch


def _choose_patterns(rng: random.Random, colors: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    patterns_h: dict[int, np.ndarray] = {}
    patterns_v: dict[int, np.ndarray] = {}
    for color in colors:
        idx_a, idx_b = rng.sample(range(len(PATTERN_LIBRARY)), 2)
        patterns_h[color] = PATTERN_LIBRARY[idx_a].copy()
        patterns_v[color] = PATTERN_LIBRARY[idx_b].copy()
    return patterns_h, patterns_v


def _grid_bounds_for_colors(num_colors: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if num_colors == 2:
        return (20, 27), (16, 28)
    if num_colors == 3:
        return (23, 29), (20, 30)
    return (26, 30), (24, 30)


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _draw_region(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    color: int,
    orient: str,
    rng: random.Random,
) -> None:
    height = inner_h + 2
    width = inner_w + 2
    grid[top : top + height, left : left + width] = BG
    grid[top, left : left + width] = FENCE
    grid[top + height - 1, left : left + width] = FENCE
    grid[top : top + height, left] = FENCE
    grid[top : top + height, left + width - 1] = FENCE

    seed_r = rng.randrange(inner_h)
    seed_c = rng.randrange(inner_w - 1 if orient == "h" else inner_w)
    if orient == "v":
        seed_r = rng.randrange(inner_h - 1)
    rr = top + 1 + seed_r
    cc = left + 1 + seed_c
    grid[rr, cc] = color
    if orient == "h":
        grid[rr, cc + 1] = color
    else:
        grid[rr + 1, cc] = color


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    num_colors = rng.choices([2, 3, 4], weights=[2, 4, 2], k=1)[0]
    colors = rng.sample(PALETTE, num_colors)
    patterns_h, patterns_v = _choose_patterns(rng, colors)
    dictionary = _dictionary_patch(colors, patterns_h, patterns_v)
    dict_h, dict_w = dictionary.shape

    (h_lo, h_hi), (w_lo, w_hi) = _grid_bounds_for_colors(num_colors)
    height = rng.randint(max(h_lo, dict_h + 4), h_hi)
    width = rng.randint(max(w_lo, dict_w + 3), w_hi)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    grid[:dict_h, :dict_w] = dictionary
    _mark_occupied(occupied, 0, 0, dict_h, dict_w)

    region_specs: list[tuple[int, str, int, int]] = []
    for color in colors:
        region_specs.append((color, "h", rng.randint(3, 6), rng.randint(3, 6)))
        region_specs.append((color, "v", rng.randint(3, 6), rng.randint(3, 6)))
    rng.shuffle(region_specs)

    for color, orient, inner_h, inner_w in region_specs:
        box_h = inner_h + 2
        box_w = inner_w + 2
        if height - box_h - 1 <= 1 or width - box_w - 1 <= 1:
            return None
        placed = False
        for _ in range(300):
            top = rng.randint(1, height - box_h - 1)
            left = rng.randint(1, width - box_w - 1)
            if not _can_place(occupied, top, left, box_h, box_w):
                continue
            _draw_region(grid, top, left, inner_h, inner_w, color, orient, rng)
            _mark_occupied(occupied, top, left, box_h, box_w)
            placed = True
            break
        if not placed:
            return None

    solved = main(grid)
    if solved.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
