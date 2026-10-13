import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MAROON = 9
SWATCH_COLORS = (1, 2, 3, 6, 7, 8)
MAX_GEN_RETRIES = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

HORIZONTAL_PATTERNS = (
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
)

VERTICAL_PATTERNS = (
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=int),
)


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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    if not pts:
        return ()
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _pattern_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_coords(np.argwhere(mask > 0))


def _frame_bbox(mask: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    local = mask[r0 : r1 + 1, c0 : c1 + 1]
    if local.shape[0] < 3 or local.shape[1] < 3:
        return None
    boundary = np.zeros_like(local, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    if not np.all(local[boundary]):
        return None
    if np.any(local[~boundary]):
        return None
    if len(comp) != int(boundary.sum()):
        return None
    return r0, r1, c0, c1


def _legend_mapping(grid: np.ndarray) -> dict[tuple[tuple[int, int], ...], int]:
    legend: dict[tuple[tuple[int, int], ...], int] = {}
    grey_mask = grid == GREY
    for comp in _components(grey_mask):
        bbox = _frame_bbox(grey_mask, comp)
        if bbox is None:
            continue
        r0, r1, c0, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        yellow_pts = np.argwhere(inner == YELLOW)
        if yellow_pts.size == 0:
            continue
        swatch_row = r1 + 1
        if swatch_row >= grid.shape[0]:
            continue
        swatches = {
            int(grid[swatch_row, c])
            for c in range(c0, c1 + 1)
            if int(grid[swatch_row, c]) not in {BG, YELLOW, GREY, MAROON}
        }
        if len(swatches) != 1:
            continue
        legend[_normalize_coords(yellow_pts)] = next(iter(swatches))
    assert legend, "expected at least one legend item"
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each container's inner yellow key pattern to the legend, then recolor the container border with the legend's swatch color.

    Input structure:
    - Background is black.
    - The top region contains several grey-framed legend cells, each holding a small yellow 2x3 or 3x2 pattern and a 1x1 color swatch directly below it.
    - The lower region contains several one-pixel-thick maroon rectangular containers whose interiors include yellow pixels arranged as one of the legend patterns.

    Rule:
    - Read every legend cell, normalize its yellow pattern by translating it to the top-left, and map that normalized pattern to the swatch color beneath the cell.
    - For each maroon container, normalize the yellow pixels inside the frame in the same way and look up the matching legend entry.
    - Repaint only that container's maroon border with the matched swatch color, leaving the legend and the interior yellow pixels unchanged.

    Key insight:
    - The yellow pixels inside a container are a lookup key rather than decoration; the legend defines a visible pattern-to-color dictionary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend = _legend_mapping(g)

    maroon_mask = g == MAROON
    for comp in _components(maroon_mask):
        bbox = _frame_bbox(maroon_mask, comp)
        if bbox is None:
            continue
        r0, r1, c0, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        yellow_pts = np.argwhere(inner == YELLOW)
        if yellow_pts.size == 0:
            continue
        key = _normalize_coords(yellow_pts)
        assert key in legend, "container pattern must exist in legend"
        color = int(legend[key])
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    return out


def _build_legend_item(pattern: np.ndarray, swatch: int) -> np.ndarray:
    item = np.full((6, 5), BG, dtype=int)
    item[:5, 0] = GREY
    item[:5, 4] = GREY
    item[0, :] = GREY
    item[4, :] = GREY
    ph, pw = pattern.shape
    row_off = random.randint(0, 3 - ph)
    col_off = random.randint(0, 3 - pw)
    for r, c in np.argwhere(pattern > 0):
        item[1 + row_off + int(r), 1 + col_off + int(c)] = YELLOW
    item[5, 2] = swatch
    return item


def _build_container(pattern: np.ndarray) -> np.ndarray:
    ph, pw = pattern.shape
    inner_h = ph + random.randint(1, 3)
    inner_w = pw + random.randint(1, 3)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = MAROON
    patch[-1, :] = MAROON
    patch[:, 0] = MAROON
    patch[:, -1] = MAROON
    row_off = random.randint(0, inner_h - ph)
    col_off = random.randint(0, inner_w - pw)
    for r, c in np.argwhere(pattern > 0):
        patch[1 + row_off + int(r), 1 + col_off + int(c)] = YELLOW
    return patch


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _reserve_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - gap)
    r1 = min(h, top + height + gap)
    c0 = max(0, left - gap)
    c1 = min(w, left + width + gap)
    if np.any(occupied[r0:r1, c0:c1]):
        return False
    occupied[top : top + height, left : left + width] = True
    return True


def _choose_patterns(k: int) -> list[np.ndarray]:
    if k == 3:
        h_count = random.choice((1, 2))
        v_count = 3 - h_count
    else:
        h_count = 2
        v_count = 2
    selected = list(random.sample(HORIZONTAL_PATTERNS, h_count))
    selected.extend(random.sample(VERTICAL_PATTERNS, v_count))
    random.shuffle(selected)
    return [np.array(p, dtype=int) for p in selected]


def _legend_strip(patterns: list[np.ndarray], swatches: list[int]) -> np.ndarray:
    gaps = [random.randint(1, 2) for _ in range(max(0, len(patterns) - 1))]
    width = len(patterns) * 5 + sum(gaps)
    strip = np.full((6, width), BG, dtype=int)
    col = 0
    for idx, (pattern, swatch) in enumerate(zip(patterns, swatches)):
        _paste_non_bg(strip, _build_legend_item(pattern, swatch), 0, col)
        col += 5
        if idx < len(gaps):
            col += gaps[idx]
    return strip


def _expected_output(inp: np.ndarray, containers: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for top, left, height, width, color in containers:
        out[top, left : left + width] = color
        out[top + height - 1, left : left + width] = color
        out[top : top + height, left] = color
        out[top : top + height, left + width - 1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.choice((3, 4))
    patterns = _choose_patterns(legend_count)
    swatches = random.sample(SWATCH_COLORS, legend_count)
    key_to_color = {_pattern_key(pattern): swatch for pattern, swatch in zip(patterns, swatches)}
    legend = _legend_strip(patterns, swatches)
    legend_h, legend_w = legend.shape

    container_count = legend_count + random.randint(0, 2)
    chosen_patterns = list(patterns)
    while len(chosen_patterns) < container_count:
        chosen_patterns.append(random.choice(patterns))
    random.shuffle(chosen_patterns)
    container_patches = [_build_container(pattern) for pattern in chosen_patterns]

    min_h = 18 if container_count <= 4 else 22
    min_w = max(legend_w, 20 if container_count <= 4 else 22)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    legend_top = random.randint(0, 1)
    legend_left = random.randint(0, grid_w - legend_w)
    if not _reserve_rect(occupied, legend_top, legend_left, legend_h, legend_w, gap=1):
        return None
    _paste_non_bg(grid, legend, legend_top, legend_left)

    containers_meta: list[tuple[int, int, int, int, int]] = []
    min_top = legend_top + legend_h + 2
    if min_top >= grid_h:
        return None

    for pattern, patch in zip(chosen_patterns, container_patches):
        ph, pw = patch.shape
        if ph > grid_h - min_top or pw > grid_w:
            return None
        placed = False
        for _ in range(200):
            top = random.randint(min_top, grid_h - ph)
            left = random.randint(0, grid_w - pw)
            if not _reserve_rect(occupied, top, left, ph, pw, gap=1):
                continue
            _paste_non_bg(grid, patch, top, left)
            color = key_to_color[_pattern_key(pattern)]
            containers_meta.append((top, left, ph, pw, color))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, containers_meta)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
