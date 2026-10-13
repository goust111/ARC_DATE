import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    c = 0
    while c < row.size:
        color = int(row[c])
        end = c + 1
        while end < row.size and int(row[end]) == color:
            end += 1
        runs.append((color, end - c))
        c = end
    return runs


def _parse_legend_band(band: np.ndarray, bg: int, row0: int) -> dict | None:
    if band.size == 0 or not np.all(np.any(band != bg, axis=1)):
        return None

    cols = np.flatnonzero(np.any(band != bg, axis=0))
    if cols.size == 0 or int(cols[-1]) - int(cols[0]) + 1 != int(cols.size):
        return None

    c0, c1 = int(cols[0]), int(cols[-1])
    sub = band[:, c0 : c1 + 1]
    if np.any(sub == bg) or not np.all(sub == sub[0]):
        return None

    runs = _color_runs(sub[0])
    if len(runs) != 5:
        return None

    widths = {width for _color, width in runs}
    colors = [color for color, _width in runs]
    if len(widths) != 1 or len(set(colors)) != 5:
        return None

    return {
        "colors": colors,
        "row0": row0,
        "row1": row0 + band.shape[0] - 1,
        "col0": c0,
        "col1": c1,
    }


def _extract_legend(grid: np.ndarray, bg: int) -> dict:
    h, _w = grid.shape
    candidates = []

    if np.any(grid[0] != bg):
        end = 0
        while end < h and np.any(grid[end] != bg):
            end += 1
        candidate = _parse_legend_band(grid[:end], bg, 0)
        if candidate is not None:
            candidates.append(candidate)

    if np.any(grid[-1] != bg):
        start = h - 1
        while start >= 0 and np.any(grid[start] != bg):
            start -= 1
        candidate = _parse_legend_band(grid[start + 1 :], bg, start + 1)
        if candidate is not None:
            candidates.append(candidate)

    assert len(candidates) == 1, "expected exactly one horizontal edge legend"
    return candidates[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_patch(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch, (r0, c0, r1, c1)


def _enclosed_cells(border_mask: np.ndarray) -> np.ndarray:
    h, w = border_mask.shape
    open_mask = ~border_mask
    seen = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(open_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(open_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(open_mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))

    return open_mask & ~seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed ring using the legend color indexed by the count of
    singleton hole markers inside that ring, then erase the markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    legend = _extract_legend(g, bg)
    legend_colors = legend["colors"]
    legend_color_set = set(legend_colors)

    non_bg_colors = set(map(int, np.unique(g))) - {bg}
    outline_colors = non_bg_colors - legend_color_set
    assert len(outline_colors) == 1, "expected a single outline color outside the legend"
    outline = next(iter(outline_colors))

    comps = _components(g == outline)
    markers = []
    rings = []
    for comp in comps:
        if len(comp) == 1:
            markers.append(comp[0])
            continue
        patch, bbox = _component_patch(comp)
        interior = _enclosed_cells(patch)
        assert np.any(interior), "ring component must enclose a region"
        rings.append((comp, bbox, interior))

    assert rings, "expected at least one closed ring"

    out = g.copy()
    used_markers = set()
    for comp, (r0, c0, r1, c1), interior in rings:
        contained = []
        for mr, mc in markers:
            if r0 <= mr <= r1 and c0 <= mc <= c1 and bool(interior[mr - r0, mc - c0]):
                contained.append((mr, mc))
        count = len(contained)
        assert 0 <= count <= 4, "marker count must map into the 5-color legend"

        ring_color = int(legend_colors[count])
        for r, c in comp:
            out[r, c] = ring_color
        for r, c in contained:
            out[r, c] = bg
            used_markers.add((r, c))

    assert len(used_markers) == len(markers), "all singleton markers must lie inside a ring"
    return out


BACKGROUND = 0
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (20, 30)
LEGEND_HEIGHT_RANGE = (1, 3)
LEGEND_BLOCK_WIDTH_RANGE = (2, 4)
LEGEND_GAP_RANGE = (1, 3)
OBJECT_COUNT_CHOICES = [1, 2, 3, 4]
OBJECT_COUNT_WEIGHTS = [1, 3, 3, 2]
SAFE_DIM_RANGE = (1, 4)
OUTER_MARGIN = 1
OBJECT_PADDING = 1
MAX_GENERATION_RETRIES = 400


def _sample_ring_spec(rng: random.Random, hole_count: int) -> dict:
    for _ in range(100):
        safe_h = rng.randint(*SAFE_DIM_RANGE)
        safe_w = rng.randint(*SAFE_DIM_RANGE)
        if (safe_h * safe_w + 1) // 2 < hole_count:
            continue

        parity_groups = {0: [], 1: []}
        for r in range(safe_h):
            for c in range(safe_w):
                parity_groups[(r + c) % 2].append((r, c))
        valid_groups = [cells for cells in parity_groups.values() if len(cells) >= hole_count]
        if not valid_groups:
            continue

        markers = []
        if hole_count:
            chosen_group = rng.choice(valid_groups)
            markers = rng.sample(chosen_group, hole_count)
        return {
            "hole_count": hole_count,
            "outer_h": safe_h + 4,
            "outer_w": safe_w + 4,
            "markers": [(r + 2, c + 2) for r, c in markers],
        }
    raise RuntimeError("failed to sample ring geometry")


def _placeable(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - OBJECT_PADDING)
    cc0 = max(0, c0 - OBJECT_PADDING)
    rr1 = min(h, r0 + height + OBJECT_PADDING)
    cc1 = min(w, c0 + width + OBJECT_PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _draw_ring(grid: np.ndarray, top: int, left: int, spec: dict, color: int) -> None:
    height = spec["outer_h"]
    width = spec["outer_w"]
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color
    for r, c in spec["markers"]:
        grid[top + r, left + c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    if rng.random() < 0.7:
        hole_counts = rng.sample(range(5), num_objects)
    else:
        hole_counts = [rng.randint(0, 4) for _ in range(num_objects)]

    specs = [_sample_ring_spec(rng, count) for count in hole_counts]
    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)

    legend_h = rng.randint(*LEGEND_HEIGHT_RANGE)
    block_w = rng.randint(*LEGEND_BLOCK_WIDTH_RANGE)
    legend_w = 5 * block_w
    gap = rng.randint(*LEGEND_GAP_RANGE)
    side = rng.choice(["top", "bottom"])

    min_h = max(GRID_H_RANGE[0], legend_h + gap + 2 * OUTER_MARGIN + max(spec["outer_h"] for spec in specs))
    min_w = max(GRID_W_RANGE[0], legend_w + 2, max(spec["outer_w"] for spec in specs) + 2 * OUTER_MARGIN)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = rng.randint(min_h, GRID_H_RANGE[1])
    width = rng.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((height, width), BACKGROUND, dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    outline_color = palette[0]
    legend_colors = palette[1:6]

    legend_left = rng.randint(0, width - legend_w)
    legend_top = 0 if side == "top" else height - legend_h
    for idx, color in enumerate(legend_colors):
        c0 = legend_left + idx * block_w
        grid[legend_top : legend_top + legend_h, c0 : c0 + block_w] = color

    row_min = OUTER_MARGIN
    row_max = height - OUTER_MARGIN
    if side == "top":
        row_min = legend_h + gap + OUTER_MARGIN
    else:
        row_max = height - legend_h - gap - OUTER_MARGIN

    if row_max - row_min < max(spec["outer_h"] for spec in specs):
        return None

    occupied = np.zeros((height, width), dtype=bool)
    placements = []
    for spec in specs:
        obj_h = spec["outer_h"]
        obj_w = spec["outer_w"]
        min_r = row_min
        max_r = row_max - obj_h
        min_c = OUTER_MARGIN
        max_c = width - OUTER_MARGIN - obj_w
        if max_r < min_r or max_c < min_c:
            return None

        placed = False
        for _ in range(250):
            top = rng.randint(min_r, max_r)
            left = rng.randint(min_c, max_c)
            if not _placeable(occupied, top, left, obj_h, obj_w):
                continue
            placements.append((top, left, spec))
            occupied[top : top + obj_h, left : left + obj_w] = True
            placed = True
            break
        if not placed:
            return None

    rng.shuffle(placements)
    for top, left, spec in placements:
        _draw_ring(grid, top, left, spec, outline_color)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
