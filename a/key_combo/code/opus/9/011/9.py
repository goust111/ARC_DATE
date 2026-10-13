import random
from collections import deque

import numpy as np


BG = 0
RECT = 1
MARK = 2
ARC_MAX = 30
GAP = 1
MAX_GENERATION_ATTEMPTS = 500

MIN_RECTS = 4
MAX_RECTS = 6
MIN_SURVIVORS = 2
MAX_SURVIVORS = 4

CELL_MIN = 6
CELL_GAP = 1
CELL_BORDER = 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    if patch.shape[0] > patch.shape[1]:
        return np.rot90(patch, -1)
    return patch.copy()


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components_mask(g != BG)
    assert comps, "input must contain foreground rectangles"

    rectangles = []
    for coords in comps:
        box = _bbox_coords(coords)
        patch = _crop_bbox(g, box)
        assert np.all(patch != BG), "each foreground component bbox must be solid"
        colors = set(map(int, np.unique(patch)))
        assert colors.issubset({RECT, MARK}), "foreground colors must be blue/red"
        rectangles.append(
            {
                "bbox": box,
                "patch": patch,
                "marker_count": int(np.count_nonzero(patch == MARK)),
            }
        )
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Erase overly marked rectangles, rotate surviving tall ones horizontal, and pack them left-to-right.

    Input structure:
    - The background is black.
    - The foreground consists of separated solid blue rectangles whose interior may contain red marker pixels.
    - Red marker pixels belong to the rectangle they sit inside and do not change that rectangle's bbox.

    Rule:
    - Find each multicolor foreground rectangle and count how many red pixels it contains.
    - Erase every rectangle with at least two red markers.
    - Keep rectangles with zero or one marker; if a kept rectangle is taller than wide, rotate it 90 degrees clockwise.
    - Sort the kept rectangles by their original bbox top-left position, using column first and row as tiebreaker.
    - Place the normalized rectangles on a new black canvas, top-aligned, with one blank column between neighbors.

    Key insight:
    - The red pixels have two roles at once: their count decides which rectangles survive, and their exact positions are preserved inside the surviving rectangles after orientation normalization.
    """
    rectangles = _extract_rectangles(grid)
    survivors = [rect for rect in rectangles if rect["marker_count"] < 2]
    assert survivors, "at least one rectangle must survive"

    survivors.sort(key=lambda rect: (rect["bbox"][1], rect["bbox"][0]))
    normalized = [_normalize_patch(rect["patch"]) for rect in survivors]

    out_h = max(patch.shape[0] for patch in normalized)
    out_w = sum(patch.shape[1] for patch in normalized) + GAP * (len(normalized) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for patch in normalized:
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w + GAP
    return out


def _sample_layout(n_rects: int) -> tuple[int, int]:
    candidates = []
    for rows, cols in ((1, n_rects), (n_rects, 1), (2, 2), (2, 3), (3, 2)):
        if rows * cols < n_rects:
            continue
        base_h = 2 * CELL_BORDER + rows * CELL_MIN + (rows - 1) * CELL_GAP
        base_w = 2 * CELL_BORDER + cols * CELL_MIN + (cols - 1) * CELL_GAP
        if base_h <= ARC_MAX and base_w <= ARC_MAX:
            candidates.append((rows, cols))
    assert candidates
    return random.choice(candidates)


def _split_lengths(total: int, parts: int, min_len: int) -> list[int] | None:
    spare = total - parts * min_len
    if spare < 0:
        return None
    lengths = [min_len] * parts
    for _ in range(spare):
        lengths[random.randrange(parts)] += 1
    return lengths


def _build_cells(rows: int, cols: int) -> tuple[list[tuple[int, int, int, int]], int, int] | None:
    base_h = 2 * CELL_BORDER + rows * CELL_MIN + (rows - 1) * CELL_GAP
    base_w = 2 * CELL_BORDER + cols * CELL_MIN + (cols - 1) * CELL_GAP
    if base_h > ARC_MAX or base_w > ARC_MAX:
        return None

    total_h = random.randint(base_h, min(ARC_MAX, base_h + 6))
    total_w = random.randint(base_w, min(ARC_MAX, base_w + 7))
    row_total = total_h - 2 * CELL_BORDER - (rows - 1) * CELL_GAP
    col_total = total_w - 2 * CELL_BORDER - (cols - 1) * CELL_GAP
    row_lengths = _split_lengths(row_total, rows, CELL_MIN)
    col_lengths = _split_lengths(col_total, cols, CELL_MIN)
    if row_lengths is None or col_lengths is None:
        return None

    row_starts = []
    pos = CELL_BORDER
    for length in row_lengths:
        row_starts.append(pos)
        pos += length + CELL_GAP

    col_starts = []
    pos = CELL_BORDER
    for length in col_lengths:
        col_starts.append(pos)
        pos += length + CELL_GAP

    cells = []
    for ri, r0 in enumerate(row_starts):
        for ci, c0 in enumerate(col_starts):
            cells.append((r0, c0, row_lengths[ri], col_lengths[ci]))
    return cells, total_h, total_w


def _sample_dims(shape_type: str, max_h: int, max_w: int) -> tuple[int, int] | None:
    if shape_type == "square":
        side = random.randint(2, min(4, max_h, max_w))
        return side, side

    short_max = min(4, max_h, max_w)
    long_max = min(6, max(max_h, max_w))
    if short_max < 2:
        return None

    for _ in range(50):
        short = random.randint(2, short_max)
        long_lo = max(3, short + 1)
        long_hi = min(6, max_h if shape_type == "tall" else max_w)
        other_limit = max_w if shape_type == "tall" else max_h
        if short > other_limit or long_lo > long_hi:
            continue
        long = random.randint(long_lo, long_hi)
        if shape_type == "tall":
            h, w = long, short
        else:
            h, w = short, long
        if h <= max_h and w <= max_w:
            return h, w
    return None


def _marker_positions(h: int, w: int, count: int, *, avoid_center_single: bool) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    if count == 0:
        return []
    if count == 1:
        if avoid_center_single and h * w > 1:
            center = ((h - 1) / 2.0, (w - 1) / 2.0)
            choices = [(r, c) for r, c in cells if (r, c) != center]
            if choices:
                return [random.choice(choices)]
        return [random.choice(cells)]
    return random.sample(cells, count)


def _make_patch(h: int, w: int, marker_count: int, *, off_center_single: bool) -> np.ndarray:
    patch = np.full((h, w), RECT, dtype=int)
    for r, c in _marker_positions(h, w, marker_count, avoid_center_single=off_center_single):
        patch[r, c] = MARK
    return patch


def _build_expected(specs: list[dict]) -> np.ndarray:
    survivors = [spec for spec in specs if spec["marker_count"] < 2]
    survivors.sort(key=lambda spec: (spec["c0"], spec["r0"]))
    normalized = [_normalize_patch(spec["patch"]) for spec in survivors]
    out_h = max(patch.shape[0] for patch in normalized)
    out_w = sum(patch.shape[1] for patch in normalized) + GAP * (len(normalized) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for patch in normalized:
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w + GAP
    return out


def _sample_specs(n_rects: int) -> list[dict]:
    n_survivors = random.randint(MIN_SURVIVORS, min(MAX_SURVIVORS, n_rects - 1))
    n_erased = n_rects - n_survivors

    survivor_specs = [
        {"shape_type": "tall", "marker_count": 1},
        {"shape_type": random.choice(["wide", "square"]), "marker_count": 0},
    ]
    while len(survivor_specs) < n_survivors:
        survivor_specs.append(
            {
                "shape_type": random.choice(["tall", "wide", "square"]),
                "marker_count": random.choice([0, 1]),
            }
        )

    erased_specs = []
    for _ in range(n_erased):
        erased_specs.append(
            {
                "shape_type": random.choice(["tall", "wide", "square"]),
                "marker_count": random.choice([2, 3]),
            }
        )

    specs = survivor_specs + erased_specs
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(MIN_RECTS, MAX_RECTS)
    rows, cols = _sample_layout(n_rects)
    built = _build_cells(rows, cols)
    if built is None:
        return None
    cells, grid_h, grid_w = built

    chosen_indices = sorted(random.sample(range(len(cells)), n_rects))
    specs = _sample_specs(n_rects)
    align_columns = rows > 1 and random.random() < 0.45
    col_offsets: dict[int, int] = {}

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    used_specs: list[dict] = []

    for cell_index, spec in zip(chosen_indices, specs):
        r0_cell, c0_cell, cell_h, cell_w = cells[cell_index]
        ri = cell_index // cols
        ci = cell_index % cols
        dims = _sample_dims(spec["shape_type"], cell_h, cell_w)
        if dims is None:
            return None
        h, w = dims

        max_dr = cell_h - h
        max_dc = cell_w - w
        if align_columns:
            if ci not in col_offsets:
                col_offsets[ci] = random.randint(0, max_dc)
            dc = min(col_offsets[ci], max_dc)
        else:
            dc = random.randint(0, max_dc)
        dr = random.randint(0, max_dr)

        patch = _make_patch(
            h,
            w,
            spec["marker_count"],
            off_center_single=(spec["shape_type"] == "tall" and spec["marker_count"] == 1),
        )
        r0 = r0_cell + dr
        c0 = c0_cell + dc
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        used_specs.append(
            {
                "patch": patch,
                "marker_count": spec["marker_count"],
                "shape_type": spec["shape_type"],
                "r0": r0,
                "c0": c0,
                "row_band": ri,
                "col_band": ci,
            }
        )

    counts = [spec["marker_count"] for spec in used_specs]
    survivors = [spec for spec in used_specs if spec["marker_count"] < 2]
    if 0 not in counts or 1 not in counts or not any(count >= 2 for count in counts):
        return None
    if len(survivors) < 2:
        return None
    if not any(spec["patch"].shape[0] > spec["patch"].shape[1] for spec in survivors):
        return None
    if not any(spec["patch"].shape[0] <= spec["patch"].shape[1] for spec in survivors):
        return None

    expected = _build_expected(used_specs)
    if max(grid.shape) > ARC_MAX or max(expected.shape) > ARC_MAX:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    extracted = _extract_rectangles(grid)
    if len(extracted) != n_rects:
        return None
    extracted_counts = sorted(rect["marker_count"] for rect in extracted)
    if extracted_counts != sorted(counts):
        return None

    if align_columns:
        tied_cols = {}
        for spec in used_specs:
            tied_cols.setdefault(spec["c0"], []).append(spec["r0"])
        if not any(len(rows_here) >= 2 and len(set(rows_here)) >= 2 for rows_here in tied_cols.values()):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
