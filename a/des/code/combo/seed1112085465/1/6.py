from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 4  # yellow
PALETTE = [0, 1, 2, 3, 5, 6, 7, 8, 9]
ORTHOGONAL_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONAL_STEPS:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return objects


def _bbox_mask(obj: dict) -> np.ndarray:
    r0, c0, r1, c1 = obj["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = obj["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _infer_step2(center_values: list[int]) -> int:
    vals = sorted(set(int(v) for v in center_values))
    if len(vals) <= 1:
        return 2

    diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    step2 = diffs[0]
    for diff in diffs[1:]:
        step2 = math.gcd(step2, diff)
    assert step2 > 0
    return int(step2)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != bg)
    assert len(nz) > 0
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize_objects(grid: np.ndarray) -> list[dict]:
    objects = _find_components(grid, bg=BG)
    assert objects, "expected at least one foreground object"

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        obj["h"] = int(r1 - r0 + 1)
        obj["w"] = int(c1 - c0 + 1)
        obj["row_center2"] = int(r0 + r1)
        obj["col_center2"] = int(c0 + c1)
        obj["mask"] = _bbox_mask(obj)

    row_step2 = _infer_step2([obj["row_center2"] for obj in objects])
    col_step2 = _infer_step2([obj["col_center2"] for obj in objects])
    min_row_center2 = min(obj["row_center2"] for obj in objects)
    min_col_center2 = min(obj["col_center2"] for obj in objects)

    for obj in objects:
        assert (obj["row_center2"] - min_row_center2) % row_step2 == 0
        assert (obj["col_center2"] - min_col_center2) % col_step2 == 0
        obj["row_index"] = int((obj["row_center2"] - min_row_center2) // row_step2)
        obj["col_index"] = int((obj["col_center2"] - min_col_center2) // col_step2)

    return objects


def _render_row_band(row_objects: list[dict]) -> np.ndarray:
    assert row_objects
    row_objects = sorted(row_objects, key=lambda obj: (obj["col_index"], obj["bbox"][1], obj["bbox"][0]))
    assert len({obj["col_index"] for obj in row_objects}) == len(row_objects)

    band_h = max(int(obj["h"]) for obj in row_objects)
    band_center2 = band_h - 1
    left_anchor2 = max(int(obj["w"]) for obj in row_objects) - 1
    min_col_index = min(int(obj["col_index"]) for obj in row_objects)

    placements = []
    band_w = 0
    for obj in row_objects:
        h = int(obj["h"])
        w = int(obj["w"])
        slot_offset = int(obj["col_index"]) - min_col_index
        top = (band_center2 - (h - 1)) // 2
        center2 = left_anchor2 + 2 * slot_offset
        left = (center2 - (w - 1)) // 2
        assert top >= 0 and left >= 0
        placements.append((obj, top, left))
        band_w = max(band_w, left + w)

    band = np.full((band_h, band_w), BG, dtype=int)
    for obj, top, left in placements:
        h = int(obj["h"])
        w = int(obj["w"])
        window = band[top : top + h, left : left + w]
        window[obj["mask"]] = int(obj["color"])

    return _crop_non_bg(band, BG)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered objects by a hidden regular grid, overlay each hidden row, and stack the row bands.

    Input structure
    - Yellow is the background.
    - The foreground consists of several small 4-connected monochrome objects.
    - Object bbox centers lie on a hidden affine grid with fixed row step dy and column step dx.

    Rule
    - Infer the hidden row/column spacing from the object bbox centers.
    - Group objects by hidden grid row.
    - Inside each hidden row, sort objects by hidden column and overlay them onto one compact band.
      Hidden-grid column offsets are preserved in slot units, so later objects can cover earlier ones.
    - Stack the row bands from top hidden row to bottom hidden row, separated by one background row.

    Key insight
    The raw positions are not arbitrary: they encode a regular hidden grid. Recover that grid first,
    then compress each hidden row into one merged strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _normalize_objects(g)
    by_row: dict[int, list[dict]] = {}
    for obj in objects:
        by_row.setdefault(int(obj["row_index"]), []).append(obj)

    row_ids = sorted(by_row)
    bands = [_render_row_band(by_row[row_id]) for row_id in row_ids]
    out_h = sum(band.shape[0] for band in bands) + max(0, len(bands) - 1)
    out_w = max(band.shape[1] for band in bands)
    out = np.full((out_h, out_w), BG, dtype=int)

    r = 0
    for i, band in enumerate(bands):
        h, w = band.shape
        out[r : r + h, :w] = band
        r += h
        if i + 1 < len(bands):
            r += 1

    return out


def _random_connected_shape(h: int, w: int) -> np.ndarray | None:
    if h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)

    target_min = max(max(h, w), math.ceil(h * w * 0.5))
    target_max = h * w

    for _ in range(120):
        target = random.randint(target_min, target_max)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            rr, cc = cell
            for dr, dc in ORTHOGONAL_STEPS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(start)
        while len(cells) < target and frontier:
            options = sorted(
                frontier,
                key=lambda cell: sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in ORTHOGONAL_STEPS),
                reverse=True,
            )
            top_k = min(len(options), 4)
            pick = random.choice(options[:top_k] if random.random() < 0.7 else options)
            frontier.discard(pick)
            cells.add(pick)
            add_frontier(pick)

        if len(cells) != target:
            continue

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask

    return None


def _sample_slot_layout(n_rows: int, n_cols: int) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    featured_row = random.randrange(n_rows)
    start = random.randrange(n_cols - 1)
    occupied.add((featured_row, start))
    occupied.add((featured_row, start + 1))

    for r in range(n_rows):
        if r == featured_row:
            continue
        for c in random.sample(range(n_cols), random.randint(1, min(2, n_cols))):
            occupied.add((r, c))

    for c in range(n_cols):
        if all(slot_c != c for _, slot_c in occupied):
            occupied.add((random.randrange(n_rows), c))

    min_objects = max(n_rows + 1, n_cols + 1)
    max_objects = min(8, n_rows * n_cols)
    target = random.randint(max(min_objects, len(occupied)), max_objects)
    while len(occupied) < target:
        occupied.add((random.randrange(n_rows), random.randrange(n_cols)))

    row_counts = {r: 0 for r in range(n_rows)}
    for r, _c in occupied:
        row_counts[r] += 1
    assert any(count >= 2 for count in row_counts.values())
    assert all(count >= 1 for count in row_counts.values())
    return occupied


def _render_input_from_specs(specs: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        mask = spec["mask"]
        h, w = mask.shape
        window = grid[top : top + h, left : left + w]
        assert np.all(window[mask] == BG)
        window[mask] = int(spec["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(3, 4)
    occupied_slots = sorted(_sample_slot_layout(n_rows, n_cols))
    object_count = len(occupied_slots)
    if object_count > len(PALETTE):
        return None

    row_even = random.choice([False, True])
    col_even = random.choice([False, True])
    height_choices = [2, 4] if row_even else [1, 3]
    width_choices = [2, 4] if col_even else [3]

    specs: list[dict] = []
    for _slot in occupied_slots:
        h = random.choice(height_choices)
        w = random.choice(width_choices)
        mask = _random_connected_shape(h, w)
        if mask is None:
            return None
        specs.append({"h": h, "w": w, "mask": mask})

    max_h = max(int(spec["h"]) for spec in specs)
    max_w = max(int(spec["w"]) for spec in specs)
    dy = random.randint(max_h + 2, max_h + 4)
    dx = random.randint(max_w + 2, max_w + 4)
    top_margin = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    row_offset2 = (max_h - 1) + 2 * top_margin
    col_offset2 = (max_w - 1) + 2 * left_margin

    colors = random.sample(PALETTE, object_count)
    for spec, (row_index, col_index), color in zip(specs, occupied_slots, colors):
        h = int(spec["h"])
        w = int(spec["w"])
        row_center2 = row_offset2 + 2 * dy * row_index
        col_center2 = col_offset2 + 2 * dx * col_index
        top = (row_center2 - (h - 1)) // 2
        left = (col_center2 - (w - 1)) // 2
        spec.update(
            {
                "row_index": row_index,
                "col_index": col_index,
                "color": color,
                "top": top,
                "left": left,
            }
        )

    grid_h = max(int(spec["top"]) + int(spec["h"]) for spec in specs) + bottom_margin
    grid_w = max(int(spec["left"]) + int(spec["w"]) for spec in specs) + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    grid = _render_input_from_specs(specs, (grid_h, grid_w))
    if np.count_nonzero(grid != BG) * 2 >= grid.size:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0 or np.all(out == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
