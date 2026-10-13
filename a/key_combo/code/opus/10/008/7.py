import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 2
WALL = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one red marker"
    marker_pos = tuple(map(int, marker_coords[0]))

    content_mask = (g != bg) & (g != WALL) & (g != MARKER)
    rect_bbox = bbox(content_mask)
    assert rect_bbox is not None, "expected one colored foreground rectangle"
    rect = crop_bbox(g, rect_bbox)
    assert np.all((rect != bg) & (rect != WALL) & (rect != MARKER)), "foreground bbox must be fully filled"
    mirrored = rect[:, ::-1].copy()

    wall_rows = list(map(int, np.flatnonzero(np.any(g == WALL, axis=1))))
    assert wall_rows, "expected wall rows"
    assert len(wall_rows) == mirrored.shape[0], "wall row count must match rectangle height"
    assert wall_rows == list(range(wall_rows[0], wall_rows[0] + len(wall_rows))), "wall rows must be contiguous"

    marker_col = marker_pos[1]
    intervals = []
    for r in wall_rows:
        wall_cols = np.flatnonzero(g[r] == WALL)
        assert wall_cols.size >= 2, "each wall row must provide both boundaries"
        left_candidates = wall_cols[wall_cols < marker_col]
        right_candidates = wall_cols[wall_cols >= marker_col]
        assert left_candidates.size and right_candidates.size, "marker column must lie between the chosen walls"
        left = int(left_candidates.max())
        right = int(right_candidates.min())
        assert right - left > 1, "wall interval must contain at least one fill cell"
        intervals.append((left, right))

    return {
        "bg": int(bg),
        "marker_pos": marker_pos,
        "rect_bbox": rect_bbox,
        "mirrored": mirrored,
        "wall_rows": wall_rows,
        "intervals": intervals,
    }


def _apply_fill(grid: np.ndarray, mirrored: np.ndarray, wall_rows, intervals, marker_pos) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    width = mirrored.shape[1]
    for idx, r in enumerate(wall_rows):
        left, right = intervals[idx]
        span_width = right - left - 1
        take = min(width, span_width)
        if take > 0:
            out[r, left + 1 : left + 1 + take] = mirrored[idx, :take]
    mr, mc = marker_pos
    out[mr, mc] = MARKER
    out[np.asarray(grid) == WALL] = WALL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the cropped foreground rectangle and pour its rows into wall-bounded scanlines.

    Input structure:
    - A black background contains one solid multicolor rectangle away from the wall area.
    - Gray wall pixels occupy one contiguous band of rows; a single red marker fixes the reference column.
    - The rectangle uses colors distinct from the background, wall, and marker colors.

    Rule:
    - Crop the tight bounding box of all non-background, non-wall, non-marker pixels.
    - Mirror that cropped rectangle horizontally.
    - For each wall row from top to bottom, find the nearest gray wall strictly left of the marker column
      and the nearest gray wall at or right of the marker column; their interior is that row's scanline.
    - Copy the corresponding mirrored row into the scanline left-aligned, truncating if necessary and
      leaving any unused scanline cells as background. Preserve the original rectangle, wall pixels, and
      the red marker pixel.

    Key insight:
    - The colored rectangle supplies the fill content after mirroring, while the wall rows independently
      supply the per-row container; the output is determined only by combining both structures.
    """
    info = _extract_structure(grid)
    return _apply_fill(grid, info["mirrored"], info["wall_rows"], info["intervals"], info["marker_pos"])


def _partition_total(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1)) if parts > 1 else []
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _grow_blob(free_mask: np.ndarray, size: int):
    free_cells = np.argwhere(free_mask)
    if free_cells.shape[0] < size:
        return None

    for _ in range(30):
        seed = tuple(map(int, free_cells[random.randrange(free_cells.shape[0])]))
        chosen = {seed}
        while len(chosen) < size:
            frontier = set()
            for r, c in chosen:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < free_mask.shape[0] and 0 <= nc < free_mask.shape[1] and free_mask[nr, nc] and (nr, nc) not in chosen:
                        frontier.add((nr, nc))
            if not frontier:
                break
            chosen.add(random.choice(tuple(frontier)))
        if len(chosen) == size:
            return np.array(sorted(chosen), dtype=int)
    return None


def _make_patch(height: int, width: int) -> np.ndarray | None:
    area = height * width
    color_count = random.choice([2, 3])
    colors = random.sample(PALETTE, color_count)
    dominant = colors[0]
    accents = colors[1:]

    patch = np.full((height, width), dominant, dtype=int)
    free_mask = np.ones((height, width), dtype=bool)

    min_accent = max(len(accents) + 2, area // 4)
    max_accent = min(area - 2, max(min_accent, (area * 2) // 5))
    accent_total = random.randint(min_accent, max_accent)
    blob_count = random.randint(len(accents), min(len(accents) + 3, accent_total))
    blob_sizes = _partition_total(accent_total, blob_count)

    blob_colors = list(accents)
    while len(blob_colors) < blob_count:
        blob_colors.append(random.choice(accents))
    random.shuffle(blob_colors)

    for size, color in zip(blob_sizes, blob_colors):
        blob = _grow_blob(free_mask, size)
        if blob is None:
            return None
        rr, cc = blob[:, 0], blob[:, 1]
        patch[rr, cc] = color
        free_mask[rr, cc] = False

    if np.array_equal(patch, patch[:, ::-1]):
        return None
    if len({tuple(row.tolist()) for row in patch}) < 2:
        return None
    if len(set(map(int, patch.ravel()))) != color_count:
        return None
    return patch


def _build_expected(grid: np.ndarray, mirrored: np.ndarray, wall_rows, intervals, marker_pos) -> np.ndarray:
    return _apply_fill(grid, mirrored, wall_rows, intervals, marker_pos)


def _validate_construction(grid: np.ndarray, mirrored: np.ndarray, rect_bbox, wall_rows, intervals, marker_pos) -> None:
    solved = main(grid)
    expected = _build_expected(grid, mirrored, wall_rows, intervals, marker_pos)
    assert np.array_equal(solved, expected), "main(grid) must match the construction semantics"

    extracted = _extract_structure(grid)
    assert extracted["rect_bbox"] == rect_bbox
    assert extracted["wall_rows"] == wall_rows
    assert extracted["intervals"] == intervals
    assert extracted["marker_pos"] == marker_pos
    assert np.array_equal(extracted["mirrored"], mirrored)

    rect = crop_bbox(grid, rect_bbox)
    assert not np.array_equal(rect, mirrored), "the rectangle must actually change under horizontal mirroring"

    widths = [right - left - 1 for left, right in intervals]
    lefts = [left for left, _ in intervals]
    assert any(width > mirrored.shape[1] for width in widths), "need at least one padded scanline"
    assert len(set(widths)) > 1 or len(set(lefts)) > 1, "wall geometry must vary across rows"

    marker_row, marker_col = marker_pos
    occluding_rows = [idx for idx, (left, _right) in enumerate(intervals) if left + 1 <= marker_col <= left + mirrored.shape[1]]
    assert occluding_rows, "the marker should visibly preserve itself over at least one pasted row"
    assert wall_rows.index(marker_row) in occluding_rows, "choose a marker row where the marker occludes pasted content"

    for idx, r in enumerate(wall_rows):
        left, right = intervals[idx]
        fill_end = left + 1 + mirrored.shape[1]
        unused_start = min(fill_end, right)
        if unused_start < right:
            unused = solved[r, unused_start:right].copy()
            if r == marker_row and unused_start <= marker_col < right:
                unused[marker_col - unused_start] = BG
            assert np.all(unused == BG), "unused scanline cells must stay background"


def _gen_attempt() -> np.ndarray | None:
    rect_h = random.randint(4, 6)
    rect_w = random.randint(5, 8)
    grid_h = random.randint(max(18, 2 * rect_h + 8), 26)
    grid_w = random.randint(max(20, rect_w + 14), 26)

    mirrored = _make_patch(rect_h, rect_w)
    if mirrored is None:
        return None
    rect = mirrored[:, ::-1].copy()

    wall_top_max = grid_h - 2 * rect_h - 4
    if wall_top_max < 2:
        return None
    wall_top = random.randint(2, wall_top_max)
    wall_rows = list(range(wall_top, wall_top + rect_h))

    extras_max = min(4, grid_w - rect_w - 8)
    if extras_max < 1:
        return None
    extras = [random.randint(0, extras_max) for _ in range(rect_h)]
    extras[random.randrange(rect_h)] = random.randint(1, extras_max)
    if rect_h >= 2:
        extras[random.randrange(rect_h)] = 0
    widths = [rect_w + extra for extra in extras]

    left_cap = min(4, grid_w - max(widths) - 2)
    if left_cap < 1:
        return None
    base_left = random.randint(1, left_cap)
    lefts = []
    for _ in range(rect_h):
        left = base_left + random.choice([-1, 0, 1])
        lefts.append(max(1, min(left_cap, left)))
    rights = [left + width + 1 for left, width in zip(lefts, widths)]

    marker_candidates = [
        c
        for c in range(max(lefts) + 1, min(rights))
        if any(left + 1 <= c <= left + rect_w for left in lefts)
    ]
    if not marker_candidates:
        return None
    marker_col = random.choice(marker_candidates)
    marker_row_options = [
        r
        for r, left in zip(wall_rows, lefts)
        if left + 1 <= marker_col <= left + rect_w
    ]
    if not marker_row_options:
        return None
    marker_row = random.choice(marker_row_options)
    marker_pos = (marker_row, marker_col)

    rect_row_min = wall_rows[-1] + 2
    rect_row_max = grid_h - rect_h - 1
    rect_col_min = max(max(rights) + 3, grid_w - rect_w - 3)
    rect_col_max = grid_w - rect_w - 1
    if rect_row_min > rect_row_max or rect_col_min > rect_col_max:
        return None
    rect_r0 = random.randint(rect_row_min, rect_row_max)
    rect_c0 = random.randint(rect_col_min, rect_col_max)
    rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = rect
    for r, left, right in zip(wall_rows, lefts, rights):
        grid[r, left] = WALL
        grid[r, right] = WALL
    grid[marker_row, marker_col] = MARKER

    intervals = list(zip(lefts, rights))
    if len(set(widths)) == 1 and len(set(lefts)) == 1:
        return None

    try:
        _validate_construction(grid, mirrored, rect_bbox, wall_rows, intervals, marker_pos)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-scan mirror sample")
