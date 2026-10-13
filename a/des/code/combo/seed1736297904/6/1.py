from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _shape_key_from_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    assert pts, "shape must be non-empty"
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _palette_candidate_bbox(grid: np.ndarray, corner: str) -> tuple[int, int, int, int] | None:
    h, w = map(int, grid.shape)
    if corner == "tl":
        if int(grid[0, 0]) != GREY:
            return None
        width = 0
        while width < w and int(grid[0, width]) == GREY:
            width += 1
        height = 0
        while height < h and int(grid[height, 0]) == GREY:
            height += 1
        return 0, 0, height - 1, width - 1
    if corner == "tr":
        if int(grid[0, w - 1]) != GREY:
            return None
        width = 0
        while width < w and int(grid[0, w - 1 - width]) == GREY:
            width += 1
        height = 0
        while height < h and int(grid[height, w - 1]) == GREY:
            height += 1
        return 0, w - width, height - 1, w - 1
    if corner == "bl":
        if int(grid[h - 1, 0]) != GREY:
            return None
        width = 0
        while width < w and int(grid[h - 1, width]) == GREY:
            width += 1
        height = 0
        while height < h and int(grid[h - 1 - height, 0]) == GREY:
            height += 1
        return h - height, 0, h - 1, width - 1
    if corner == "br":
        if int(grid[h - 1, w - 1]) != GREY:
            return None
        width = 0
        while width < w and int(grid[h - 1, w - 1 - width]) == GREY:
            width += 1
        height = 0
        while height < h and int(grid[h - 1 - height, w - 1]) == GREY:
            height += 1
        return h - height, w - width, h - 1, w - 1
    return None


def _parse_palette(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[tuple[tuple[int, int], ...], int] | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    rect = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all(rect[0, :] == GREY) and np.all(rect[-1, :] == GREY)):
        return None
    if not bool(np.all(rect[:, 0] == GREY) and np.all(rect[:, -1] == GREY)):
        return None

    full_rows = [idx for idx in range(rect.shape[0]) if bool(np.all(rect[idx, :] == GREY))]
    full_cols = [idx for idx in range(rect.shape[1]) if bool(np.all(rect[:, idx] == GREY))]
    if full_rows[0] != 0 or full_rows[-1] != rect.shape[0] - 1:
        return None
    if full_cols[0] != 0 or full_cols[-1] != rect.shape[1] - 1:
        return None

    row_spans = [(full_rows[i] + 1, full_rows[i + 1] - 1) for i in range(len(full_rows) - 1)]
    col_spans = [(full_cols[i] + 1, full_cols[i + 1] - 1) for i in range(len(full_cols) - 1)]
    if not row_spans or not col_spans:
        return None
    if any(a > b for a, b in row_spans) or any(a > b for a, b in col_spans):
        return None

    row_sizes = {b - a + 1 for a, b in row_spans}
    col_sizes = {b - a + 1 for a, b in col_spans}
    if len(row_sizes) != 1 or len(col_sizes) != 1:
        return None
    if len(row_spans) * len(col_spans) < 2:
        return None

    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    used_colors: set[int] = set()
    for rs, re in row_spans:
        for cs, ce in col_spans:
            cell = rect[rs : re + 1, cs : ce + 1]
            vals = set(map(int, np.unique(cell)))
            fill_colors = vals - {BLACK, GREY}
            if len(fill_colors) != 1 or BLACK not in vals:
                return None
            fill = int(next(iter(fill_colors)))
            if fill in used_colors:
                return None
            key = _shape_key_from_coords(np.argwhere(cell == BLACK))
            if key in mapping:
                return None
            used_colors.add(fill)
            mapping[key] = fill
    return mapping


def _detect_palette(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[tuple[tuple[int, int], ...], int]]:
    candidates: list[tuple[int, tuple[int, int, int, int], dict[tuple[tuple[int, int], ...], int]]] = []
    for corner in ("tl", "tr", "bl", "br"):
        bbox = _palette_candidate_bbox(grid, corner)
        if bbox is None:
            continue
        mapping = _parse_palette(grid, bbox)
        if mapping is None:
            continue
        r0, c0, r1, c1 = bbox
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        candidates.append((area, bbox, mapping))
    assert candidates, "palette must be detectable in one corner"
    _area, bbox, mapping = max(candidates, key=lambda item: item[0])
    return bbox, mapping


def _find_regions(
    grid: np.ndarray, palette_bbox: tuple[int, int, int, int], require_seed: bool = True
) -> list[list[tuple[int, int]]]:
    mask = grid != GREY
    r0, c0, r1, c1 = palette_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(mask)
    h, w = map(int, grid.shape)
    regions = [comp for comp in comps if not _touches_border(comp, h, w)]
    if require_seed:
        regions = [
            comp
            for comp in regions
            if any(int(grid[r, c]) not in {BLACK, GREY} for r, c in comp)
        ]
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed shape to the palette's black cut-out template, then fill the whole enclosed region
    with the palette cell's color while leaving the palette unchanged.
    """
    g = np.asarray(grid, dtype=int)
    palette_bbox, shape_to_color = _detect_palette(g)
    out = g.copy()
    for region in _find_regions(g, palette_bbox, require_seed=True):
        seed = [(r, c) for r, c in region if int(g[r, c]) not in {BLACK, GREY}]
        key = _shape_key_from_coords(seed)
        fill = int(shape_to_color[key])
        for r, c in region:
            out[r, c] = fill
    return out


def _shape(rows: list[str]) -> np.ndarray:
    arr = np.array([[1 if ch == "1" else 0 for ch in row] for row in rows], dtype=bool)
    assert bool(arr.any())
    return arr


_RAW_SHAPES = [
    _shape(["11", "10"]),
    _shape(["11", "01"]),
    _shape(["111"]),
    _shape(["1", "1", "1"]),
    _shape(["11", "11"]),
    _shape(["111", "100"]),
    _shape(["111", "001"]),
    _shape(["100", "111"]),
    _shape(["001", "111"]),
    _shape(["111", "010"]),
    _shape(["010", "111"]),
    _shape(["10", "11", "10"]),
    _shape(["01", "11", "01"]),
    _shape(["110", "011"]),
    _shape(["011", "110"]),
    _shape(["11", "10", "10"]),
    _shape(["11", "01", "01"]),
    _shape(["1111"]),
    _shape(["1", "1", "1", "1"]),
]


SHAPE_LIBRARY: list[np.ndarray] = []
_seen_shape_keys: set[tuple[tuple[int, int], ...]] = set()
for _candidate in _RAW_SHAPES:
    _key = _shape_key_from_coords(np.argwhere(_candidate))
    if _key not in _seen_shape_keys:
        _seen_shape_keys.add(_key)
        SHAPE_LIBRARY.append(_candidate)


def _layout_options(count: int) -> list[tuple[int, int]]:
    opts = {(1, count), (count, 1)}
    for rows in range(2, count + 1):
        if count % rows == 0:
            opts.add((rows, count // rows))
    return sorted(opts)


def _build_palette(
    shapes: list[np.ndarray], colors: list[int], rows: int, cols: int, rng: random.Random
) -> np.ndarray:
    cell_h = max(int(shape.shape[0]) for shape in shapes) + rng.randint(2, 4)
    cell_w = max(int(shape.shape[1]) for shape in shapes) + rng.randint(2, 4)
    h = rows * cell_h + rows + 1
    w = cols * cell_w + cols + 1
    palette = np.full((h, w), GREY, dtype=int)
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            inner_r0 = 1 + rr * (cell_h + 1)
            inner_c0 = 1 + cc * (cell_w + 1)
            palette[inner_r0 : inner_r0 + cell_h, inner_c0 : inner_c0 + cell_w] = int(colors[idx])
            shape = shapes[idx]
            sh, sw = map(int, shape.shape)
            off_r = rng.randint(0, cell_h - sh)
            off_c = rng.randint(0, cell_w - sw)
            hole_r0 = inner_r0 + off_r
            hole_c0 = inner_c0 + off_c
            palette[hole_r0 : hole_r0 + sh, hole_c0 : hole_c0 + sw][shape] = BLACK
            idx += 1
    return palette


def _grow_labels(rows: int, cols: int, n_regions: int, rng: random.Random) -> np.ndarray | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    if n_regions > len(cells):
        return None
    labels = np.full((rows, cols), -1, dtype=int)
    for idx, (r, c) in enumerate(rng.sample(cells, n_regions)):
        labels[r, c] = idx
    while bool((labels == -1).any()):
        frontier: list[tuple[int, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if int(labels[r, c]) < 0:
                    continue
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and int(labels[nr, nc]) < 0:
                        frontier.append((int(labels[r, c]), int(nr), int(nc)))
        if not frontier:
            return None
        label, r, c = rng.choice(frontier)
        labels[r, c] = int(label)
    return labels


def _render_window(labels: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    rows, cols = map(int, labels.shape)
    h = sum(row_sizes) + rows + 1
    w = sum(col_sizes) + cols + 1
    grid = np.zeros((h, w), dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 0
    for size in row_sizes:
        start = cursor + 1
        end = start + size - 1
        row_spans.append((start, end))
        cursor = end + 1

    col_spans: list[tuple[int, int]] = []
    cursor = 0
    for size in col_sizes:
        start = cursor + 1
        end = start + size - 1
        col_spans.append((start, end))
        cursor = end + 1

    row_lines = [0] + [end + 1 for _start, end in row_spans]
    col_lines = [0] + [end + 1 for _start, end in col_spans]
    grid[row_lines, :] = GREY
    grid[:, col_lines] = GREY

    for r in range(rows):
        for c in range(cols - 1):
            if int(labels[r, c]) == int(labels[r, c + 1]):
                boundary_c = col_spans[c][1] + 1
                rs, re = row_spans[r]
                grid[rs : re + 1, boundary_c] = BLACK

    for r in range(rows - 1):
        for c in range(cols):
            if int(labels[r, c]) == int(labels[r + 1, c]):
                boundary_r = row_spans[r][1] + 1
                cs, ce = col_spans[c]
                grid[boundary_r, cs : ce + 1] = BLACK

    for r in range(rows - 1):
        for c in range(cols - 1):
            block = {
                int(labels[r, c]),
                int(labels[r, c + 1]),
                int(labels[r + 1, c]),
                int(labels[r + 1, c + 1]),
            }
            if len(block) == 1:
                grid[row_spans[r][1] + 1, col_spans[c][1] + 1] = BLACK

    return grid


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, src.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _seed_coords_for_shape(
    region: list[tuple[int, int]], shape: np.ndarray, canvas_shape: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]] | None:
    mask = np.zeros(canvas_shape, dtype=bool)
    for r, c in region:
        mask[r, c] = True

    safe = mask.copy()
    for r, c in region:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < canvas_shape[0] and 0 <= nc < canvas_shape[1] and bool(mask[nr, nc])):
                safe[r, c] = False
                break

    shape_pts = [(int(r), int(c)) for r, c in np.argwhere(shape)]
    sh, sw = map(int, shape.shape)
    r0, c0, r1, c1 = _bbox(region)

    for target_mask in (safe, mask):
        placements: list[list[tuple[int, int]]] = []
        for base_r in range(r0, r1 - sh + 2):
            for base_c in range(c0, c1 - sw + 2):
                placed = [(base_r + dr, base_c + dc) for dr, dc in shape_pts]
                if all(bool(target_mask[rr, cc]) for rr, cc in placed):
                    placements.append(placed)
        if placements:
            return rng.choice(placements)
    return None


def _generate_case(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(3, 6)
    shapes = rng.sample(SHAPE_LIBRARY, template_count)
    colors = rng.sample(PALETTE_COLORS, template_count)
    palette_rows, palette_cols = rng.choice(_layout_options(template_count))
    palette = _build_palette(shapes, colors, palette_rows, palette_cols, rng)

    coarse_rows = rng.randint(3, 5)
    coarse_cols = rng.randint(3, 5)
    max_regions = min(coarse_rows * coarse_cols, template_count + 3)
    region_count = rng.randint(template_count, max_regions)
    labels = _grow_labels(coarse_rows, coarse_cols, region_count, rng)
    if labels is None:
        return None

    row_sizes = [rng.randint(4, 7) for _ in range(coarse_rows)]
    col_sizes = [rng.randint(4, 7) for _ in range(coarse_cols)]
    window = _render_window(labels, row_sizes, col_sizes)

    gap = rng.randint(2, 4)
    canvas_h = palette.shape[0] + window.shape[0] + gap + rng.randint(2, 6)
    canvas_w = palette.shape[1] + window.shape[1] + gap + rng.randint(2, 6)
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        palette_r0, palette_c0 = 0, 0
    elif corner == "tr":
        palette_r0, palette_c0 = 0, canvas_w - palette.shape[1]
    elif corner == "bl":
        palette_r0, palette_c0 = canvas_h - palette.shape[0], 0
    else:
        palette_r0, palette_c0 = canvas_h - palette.shape[0], canvas_w - palette.shape[1]
    _paste(canvas, palette, palette_r0, palette_c0)
    palette_bbox = (
        int(palette_r0),
        int(palette_c0),
        int(palette_r0 + palette.shape[0] - 1),
        int(palette_c0 + palette.shape[1] - 1),
    )

    placed_window = False
    for _ in range(200):
        win_r0 = rng.randint(1, canvas_h - window.shape[0] - 1)
        win_c0 = rng.randint(1, canvas_w - window.shape[1] - 1)
        win_bbox = (
            int(win_r0),
            int(win_c0),
            int(win_r0 + window.shape[0] - 1),
            int(win_c0 + window.shape[1] - 1),
        )
        if _boxes_separated(palette_bbox, win_bbox, gap):
            _paste(canvas, window, win_r0, win_c0)
            placed_window = True
            break
    if not placed_window:
        return None

    regions = _find_regions(canvas, palette_bbox, require_seed=False)
    if len(regions) != region_count:
        return None

    assignment = list(range(template_count))
    rng.shuffle(assignment)
    while len(assignment) < len(regions):
        assignment.append(rng.randrange(template_count))
    rng.shuffle(assignment)

    for region, shape_idx in zip(regions, assignment):
        placed = _seed_coords_for_shape(region, shapes[shape_idx], canvas.shape, rng)
        if placed is None:
            return None
        fill = int(colors[shape_idx])
        if rng.random() < 0.25:
            seed_color = fill
        else:
            seed_choices = [color for color in PALETTE_COLORS if color != fill]
            seed_color = int(rng.choice(seed_choices))
        for r, c in placed:
            canvas[r, c] = seed_color

    expected = canvas.copy()
    for region, shape_idx in zip(regions, assignment):
        fill = int(colors[shape_idx])
        for r, c in region:
            expected[r, c] = fill

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return canvas


def generate_input() -> np.ndarray:
    """
    Random puzzle generator for the shape-template stained-glass task.
    """
    rng = random
    for _ in range(400):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
