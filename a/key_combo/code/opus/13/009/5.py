from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 7, 8)
GRID_HEIGHTS = (14, 15, 16, 17, 18)
GRID_WIDTHS = (14, 16, 18)
MAX_ATTEMPTS = 500
MAX_OVERLAP_CELLS = 8


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPES_BY_AREA: dict[int, list[np.ndarray]] = {
    3: [
        _shape(["111"]),
        _shape(["1", "1", "1"]),
        _shape(["10", "11"]),
        _shape(["01", "11"]),
    ],
    4: [
        _shape(["11", "11"]),
        _shape(["111", "010"]),
        _shape(["110", "011"]),
        _shape(["011", "110"]),
        _shape(["10", "11", "10"]),
        _shape(["01", "11", "01"]),
    ],
    5: [
        _shape(["010", "111", "010"]),
        _shape(["11", "11", "10"]),
        _shape(["11", "11", "01"]),
        _shape(["111", "110"]),
        _shape(["111", "011"]),
        _shape(["101", "111"]),
    ],
    6: [
        _shape(["111", "111"]),
        _shape(["11", "11", "11"]),
        _shape(["111", "110", "100"]),
        _shape(["111", "011", "001"]),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _paste_transparent(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _draw_order_key(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = obj["bbox"]
    return (-int(obj["area"]), -int(r0), int(c0), int(obj["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-side sprite to the right, then grow a downward wedge whose width is set by the sprite's area.

    Input structure:
    - The background is black.
    - The foreground consists of several monochrome 4-connected sprites in the left half and upper half.
    - Each sprite has a bounding box no larger than 3x3 and contains 3-6 pixels.

    Rule:
    - Clear the input and start from an empty black canvas of the same size.
    - For each sprite, mirror its bounding-box position horizontally with `new_c0 = grid_width - sprite_width - old_c0`, but keep the internal sprite layout unchanged.
    - Repaint the mirrored sprite, then fill rows directly below its mirrored bounding box with a left-aligned wedge of the same color: widths `P, P-1, ..., 1`, where `P` is the sprite's pixel count.
    - Draw larger sprites first and smaller sprites later so smaller wedges overwrite larger ones on overlaps; for equal sizes, the higher original sprite is drawn later.

    Key insight:
    - The mirrored position controls where the wedge starts, while the sprite's area controls how far and how wide the wedge extends.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    H, W = map(int, g.shape)
    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert objects, "expected foreground sprites"

    out = np.full((H, W), int(bg), dtype=int)
    for obj in sorted(objects, key=_draw_order_key):
        r0, c0, r1, c1 = obj["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        h, w = patch.shape
        dst_c0 = W - w - c0
        assert 0 <= dst_c0 <= W - w
        _paste_transparent(out, patch, int(r0), int(dst_c0), int(bg))

        area = int(obj["area"])
        color = int(obj["color"])
        for step in range(area):
            rr = int(r1) + 1 + step
            if rr >= H:
                break
            width = area - step
            out[rr, dst_c0 : min(W, dst_c0 + width)] = color

    return out


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _mask_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(mask)
    return {(int(r0 + r), int(c0 + c)) for r, c in zip(rows, cols)}


def _wedge_cells(r0: int, c0: int, h: int, area: int, H: int, W: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for step in range(int(area)):
        rr = int(r0 + h + step)
        if not (0 <= rr < H):
            continue
        width = int(area - step)
        for cc in range(int(c0), min(W, int(c0) + width)):
            cells.add((rr, cc))
    return cells


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in b:
                    return True
    return False


def _coverage_overlap(records: list[dict], field: str) -> set[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for rec in records:
        counts.update(rec[field])
    return {cell for cell, count in counts.items() if count > 1}


def _wedge_pair_matches(records: list[dict], require_same_area: bool) -> bool:
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            overlap = records[i]["wedge_cells"] & records[j]["wedge_cells"]
            if not overlap:
                continue
            same_area = int(records[i]["area"]) == int(records[j]["area"])
            diff_rows = int(records[i]["r0"]) != int(records[j]["r0"])
            if require_same_area:
                if same_area and diff_rows:
                    return True
            else:
                if not same_area:
                    return True
    return False


def _record_from_output_position(mask: np.ndarray, color: int, r0: int, mirror_c0: int, H: int, W: int) -> dict | None:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    half = W // 2
    input_c0 = W - w - int(mirror_c0)

    if not (0 <= int(r0) and int(r0) + h <= 8):
        return None
    if not (0 <= int(input_c0) and int(input_c0) + w <= half):
        return None
    if not (half <= int(mirror_c0) and int(mirror_c0) + w <= W):
        return None
    if int(r0) + h + area > H:
        return None

    patch = _make_patch(mask, color)
    input_cells = _mask_cells(mask, int(r0), int(input_c0))
    object_cells = _mask_cells(mask, int(r0), int(mirror_c0))
    wedge_cells = _wedge_cells(int(r0), int(mirror_c0), h, area, H, W)
    return {
        "color": int(color),
        "mask": mask.copy(),
        "patch": patch,
        "area": area,
        "h": h,
        "w": w,
        "r0": int(r0),
        "input_c0": int(input_c0),
        "mirror_c0": int(mirror_c0),
        "bbox": (int(r0), int(input_c0), int(r0 + h - 1), int(input_c0 + w - 1)),
        "input_cells": input_cells,
        "object_cells": object_cells,
        "wedge_cells": wedge_cells,
        "draw_cells": object_cells | wedge_cells,
    }


def _can_add_record(record: dict | None, records: list[dict], *, separate: bool) -> bool:
    if record is None:
        return False
    for other in records:
        if record["object_cells"] & other["object_cells"]:
            return False
        if record["input_cells"] & other["input_cells"]:
            return False
        if _touching(record["input_cells"], other["input_cells"]):
            return False
    trial = records + [record]
    if len(_coverage_overlap(trial, "draw_cells")) > MAX_OVERLAP_CELLS:
        return False
    if separate and _coverage_overlap(trial, "draw_cells"):
        return False
    return True


def _render_input(records: list[dict], H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for rec in records:
        _paste_transparent(grid, rec["patch"], rec["r0"], rec["input_c0"], BG)
    return grid


def _render_output(records: list[dict], H: int, W: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    ordered = sorted(
        records,
        key=lambda rec: (-int(rec["area"]), -int(rec["r0"]), int(rec["input_c0"]), int(rec["color"])),
    )
    for rec in ordered:
        _paste_transparent(out, rec["patch"], rec["r0"], rec["mirror_c0"], BG)
        for step in range(int(rec["area"])):
            rr = int(rec["r0"] + rec["h"] + step)
            width = int(rec["area"] - step)
            out[rr, rec["mirror_c0"] : min(W, rec["mirror_c0"] + width)] = int(rec["color"])
    return out


def _sample_shape(area: int | None = None) -> np.ndarray:
    if area is None:
        area = random.randint(3, 6)
    return random.choice(SHAPES_BY_AREA[int(area)]).copy()


def _place_overlap_pair(H: int, W: int, colors: list[int]) -> list[dict] | None:
    half = W // 2
    for _ in range(250):
        area_left = random.choice((4, 5, 6))
        area_right = random.choice([a for a in (3, 4, 5) if a != area_left and a < area_left])
        mask_left = _sample_shape(area_left)
        mask_right = _sample_shape(area_right)
        h_left, w_left = mask_left.shape
        h_right, w_right = mask_right.shape

        max_r = min(8 - h_left, 8 - h_right, H - h_left - area_left, H - h_right - area_right)
        if max_r < 0:
            continue
        r0 = random.randint(0, max_r)

        min_delta = max(w_left, area_left - 2)
        max_delta = area_left - 1
        if min_delta > max_delta:
            continue
        delta = random.randint(min_delta, max_delta)
        c_left_max = W - delta - w_right
        if c_left_max < half:
            continue
        c_left = random.randint(half, c_left_max)
        c_right = c_left + delta

        rec_left = _record_from_output_position(mask_left, colors[0], r0, c_left, H, W)
        rec_right = _record_from_output_position(mask_right, colors[1], r0, c_right, H, W)
        if not _can_add_record(rec_left, [], separate=False):
            continue
        if not _can_add_record(rec_right, [rec_left], separate=False):
            continue
        if not (rec_left["wedge_cells"] & rec_right["wedge_cells"]):
            continue
        return [rec_left, rec_right]
    return None


def _place_tie_pair(H: int, W: int, colors: list[int]) -> list[dict] | None:
    half = W // 2
    for _ in range(250):
        area = random.choice((3, 4, 5, 6))
        upper = _sample_shape(area)
        lower = _sample_shape(area)
        h_upper, w_upper = upper.shape
        h_lower, w_lower = lower.shape

        max_upper_r = min(8 - h_upper, H - h_upper - area)
        if max_upper_r < 0:
            continue
        r_upper = random.randint(0, min(2, max_upper_r))

        lower_min = r_upper + h_upper
        lower_max = min(8 - h_lower, H - h_lower - area, r_upper + h_upper + 2)
        lower_overlap_cap = r_upper + h_upper + area - h_lower - 1
        lower_max = min(lower_max, lower_overlap_cap)
        if lower_min > lower_max:
            continue
        r_lower = random.randint(lower_min, lower_max)

        min_delta = max(w_lower, area - 2)
        max_delta = area - 1
        if min_delta > max_delta:
            continue
        delta = random.randint(min_delta, max_delta)
        c_lower_max = W - delta - w_upper
        if c_lower_max < half:
            continue
        c_lower = random.randint(half, c_lower_max)
        c_upper = c_lower + delta

        rec_upper = _record_from_output_position(upper, colors[0], r_upper, c_upper, H, W)
        rec_lower = _record_from_output_position(lower, colors[1], r_lower, c_lower, H, W)
        if not _can_add_record(rec_upper, [], separate=False):
            continue
        if not _can_add_record(rec_lower, [rec_upper], separate=False):
            continue
        if not (rec_upper["wedge_cells"] & rec_lower["wedge_cells"]):
            continue
        return [rec_upper, rec_lower]
    return None


def _place_random_record(H: int, W: int, color: int, records: list[dict], *, separate: bool) -> dict | None:
    half = W // 2
    for _ in range(250):
        mask = _sample_shape()
        area = int(mask.sum())
        h, w = mask.shape
        max_r = min(8 - h, H - h - area)
        if max_r < 0 or W - w < half:
            continue
        r0 = random.randint(0, max_r)
        mirror_c0 = random.randint(half, W - w)
        record = _record_from_output_position(mask, color, r0, mirror_c0, H, W)
        if _can_add_record(record, records, separate=separate):
            return record
    return None


def _choose_dimensions(n_objects: int) -> tuple[int, int]:
    if n_objects == 5:
        return random.choice((16, 17, 18)), random.choice((16, 18))
    if n_objects == 4:
        return random.choice((15, 16, 17, 18)), random.choice((14, 16, 18))
    return random.choice(GRID_HEIGHTS), random.choice(GRID_WIDTHS)


def _valid_input_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    H, W = grid.shape
    if max(H, W) > 30:
        return False
    bg = _get_bg(grid)
    if bg != BG:
        return False
    half = W // 2
    comps = _find_components(grid, bg)
    if not (3 <= len(comps) <= 5):
        return False
    seen_colors: set[int] = set()
    cell_sets = []
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        r0, c0, r1, c1 = comp["bbox"]
        if color not in PALETTE or color in seen_colors:
            return False
        seen_colors.add(color)
        if not (3 <= area <= 6):
            return False
        if r1 >= 8 or c1 >= half:
            return False
        if (r1 - r0 + 1) > 3 or (c1 - c0 + 1) > 3:
            return False
        cell_sets.append({tuple(map(int, cell)) for cell in comp["coords"]})
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            if _touching(cell_sets[i], cell_sets[j]):
                return False
    return True


def _mode_ok(records: list[dict], mode: str) -> bool:
    overlap_cells = _coverage_overlap(records, "draw_cells")
    if mode == "separate":
        return not overlap_cells
    if not overlap_cells:
        return False
    if mode == "overlap":
        return _wedge_pair_matches(records, require_same_area=False)
    return _wedge_pair_matches(records, require_same_area=True)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    mode = random.choices(["separate", "overlap", "tie_overlap"], weights=[2, 4, 3], k=1)[0]
    H, W = _choose_dimensions(n_objects)
    colors = random.sample(PALETTE, n_objects)
    records: list[dict] = []

    if mode == "overlap":
        pair = _place_overlap_pair(H, W, colors[:2])
        if pair is None:
            return None
        records.extend(pair)
        remaining_colors = colors[2:]
    elif mode == "tie_overlap":
        pair = _place_tie_pair(H, W, colors[:2])
        if pair is None:
            return None
        records.extend(pair)
        remaining_colors = colors[2:]
    else:
        remaining_colors = colors

    for color in remaining_colors:
        rec = _place_random_record(H, W, color, records, separate=(mode == "separate"))
        if rec is None:
            return None
        records.append(rec)

    if not _mode_ok(records, mode):
        return None

    grid = _render_input(records, H, W)
    if not _valid_input_grid(grid):
        return None
    expected = _render_output(records, H, W)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
