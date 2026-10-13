from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            sys.path.insert(0, str(candidate.parent))
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, components_mask, crop_bbox


Panel = tuple[int, int]
BBox = tuple[int, int, int, int]


def _triangular_offsets(limit: int) -> list[int]:
    offsets: list[int] = []
    offset = 0
    step = 1
    while offset < limit:
        offsets.append(offset)
        offset += step
        step += 1
    return offsets


def _largest_green_component(grid: np.ndarray) -> tuple[np.ndarray, BBox]:
    comps = components_mask(grid == Color.GREEN, connectivity=8)
    assert comps, "expected a green object"
    areas = [len(comp) for comp in comps]
    largest = max(areas)
    assert areas.count(largest) == 1, "expected a unique largest green object"
    coords = comps[areas.index(largest)]
    mask = np.zeros_like(grid, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask, bbox_coords(coords)


def _panel_slices(grid: np.ndarray, upper_h: int) -> list[Panel]:
    upper = grid[:upper_h]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(upper[:, c] == Color.GREY)]
    assert len(sep_cols) in (1, 2), "expected 1 or 2 separator columns"

    panels: list[Panel] = []
    start = 0
    for c in sep_cols + [grid.shape[1]]:
        if start < c:
            panels.append((start, c))
        start = c + 1

    assert len(panels) in (2, 3), "expected 2 or 3 panels"
    widths = {c1 - c0 for c0, c1 in panels}
    assert len(widths) == 1, "expected equal-width panels"
    return panels


def _collect_rows(
    upper: np.ndarray,
    panels: list[Panel],
    *,
    right_to_left: bool,
    discard_empty: bool,
) -> list[np.ndarray]:
    order = list(reversed(panels)) if right_to_left else list(panels)
    rows: list[np.ndarray] = []
    for c0, c1 in order:
        for r in range(upper.shape[0]):
            row = upper[r, c0:c1]
            if discard_empty and np.all(row == Color.BLACK):
                continue
            rows.append(row.copy())
    return rows


def _find_holes(grid: np.ndarray, green_box: BBox) -> list[BBox]:
    r0, c0, r1, c1 = green_box
    crop = crop_bbox(grid, green_box)
    holes: list[BBox] = []
    for comp in components_mask(crop == Color.BLACK, connectivity=4):
        local_box = bbox_coords(comp)
        hr0, hc0, hr1, hc1 = local_box
        if hr0 == 0 or hc0 == 0 or hr1 == crop.shape[0] - 1 or hc1 == crop.shape[1] - 1:
            continue
        if len(comp) != bbox_h(local_box) * bbox_w(local_box):
            continue

        enclosed = True
        for rr in range(hr0 - 1, hr1 + 2):
            for cc in range(hc0 - 1, hc1 + 2):
                if hr0 <= rr <= hr1 and hc0 <= cc <= hc1:
                    continue
                if crop[rr, cc] != Color.GREEN:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            holes.append((r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1))

    holes.sort()
    assert 1 <= len(holes) <= 2, "expected 1 or 2 enclosed rectangular holes"
    return holes


def _fill_holes(
    grid: np.ndarray,
    holes: list[BBox],
    rows: list[np.ndarray],
    *,
    triangular_mode: str = "hole",
    green_top: int | None = None,
) -> np.ndarray:
    assert rows, "row sequence must be non-empty"
    out = np.asarray(grid, dtype=int).copy()
    row_w = len(rows[0])
    for row in rows:
        assert len(row) == row_w, "tile rows must share one width"

    tri_cache: dict[int, set[int]] = {}
    for hole in holes:
        r0, c0, r1, c1 = hole
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert w == row_w, "hole width must equal tile width"
        for i in range(h):
            out[r0 + i, c0 : c1 + 1] = rows[i % len(rows)]

        if triangular_mode == "none":
            continue

        if triangular_mode == "hole":
            offsets = _triangular_offsets(h)
            for off in offsets:
                segment = out[r0 + off, c0 : c1 + 1]
                segment[segment != Color.BLACK] = Color.YELLOW
            continue

        assert triangular_mode == "object"
        assert green_top is not None
        max_rel = r1 - green_top + 1
        tri_set = tri_cache.setdefault(max_rel, set(_triangular_offsets(max_rel)))
        for rr in range(r0, r1 + 1):
            if rr - green_top in tri_set:
                segment = out[rr, c0 : c1 + 1]
                segment[segment != Color.BLACK] = Color.YELLOW

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each panel's nonempty rows into a cyclic tile, fill every enclosed green-hole rectangle with it, and mark triangular hole rows yellow.

    Input structure:
    - Black background. Above the large green object, 1 or 2 full-height grey separator columns split the upper region into 2 or 3 equal-width panels.
    - Each panel contains blue/red nonempty rows mixed with all-black empty rows.
    - The lower region contains one large 8-connected green object whose interior has 1 or 2 enclosed rectangular black holes.

    Rule:
    - Take the rows above the green object and split them into panels using the grey columns.
    - Scan the panels from right to left; within each panel, collect nonempty rows from top to bottom to form row sequence T.
    - For each enclosed hole, copy rows of T from the hole top downward, cycling back to the first row after the last one until the hole is filled.
    - After filling, recolor every non-black cell on hole-local triangular row offsets 0, 1, 3, 6, ... to yellow.
    - Leave the panel area, separators, and the green shell unchanged.

    Key insight:
    - The top half supplies a compressed row tile, while the lower holes reuse that tile cyclically; a second pass then breaks the cycle by measuring triangular offsets from each hole's own top edge.
    """
    g = np.asarray(grid, dtype=int)
    _green_mask, green_box = _largest_green_component(g)
    upper_h = green_box[0]
    panels = _panel_slices(g, upper_h)
    rows = _collect_rows(g[:upper_h], panels, right_to_left=True, discard_empty=True)
    assert len(rows) >= 2, "expected at least two compressed rows"
    holes = _find_holes(g, green_box)
    return _fill_holes(g, holes, rows, triangular_mode="hole")


def _row_pattern(width: int) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    mode = random.choice(("double", "single_tip", "alternating"))
    if mode == "double":
        left_len = random.randint(1, max(1, width // 2))
        right_len = random.randint(1, max(1, width // 2))
        if left_len + right_len >= width:
            if left_len >= right_len:
                left_len -= 1
            else:
                right_len -= 1
        left_len = max(left_len, 1)
        right_len = max(right_len, 1)
        gap = random.randint(1, width - left_len - right_len)
        left_start = random.randint(0, width - left_len - right_len - gap)
        right_start = left_start + left_len + gap
        row[left_start : left_start + left_len] = Color.BLUE
        row[right_start : right_start + right_len] = Color.RED
    elif mode == "single_tip":
        span_len = random.randint(2, width - 1)
        start = random.randint(0, width - span_len)
        main_color = random.choice((Color.BLUE, Color.RED))
        alt_color = Color.RED if main_color == Color.BLUE else Color.BLUE
        row[start : start + span_len] = main_color
        tips = [idx for idx in (start - 1, start + span_len) if 0 <= idx < width]
        if tips:
            row[random.choice(tips)] = alt_color
    else:
        count = random.randint(2, width - 1)
        positions = sorted(random.sample(range(width), count))
        first = random.choice((Color.BLUE, Color.RED))
        second = Color.RED if first == Color.BLUE else Color.BLUE
        for i, pos in enumerate(positions):
            row[pos] = first if i % 2 == 0 else second

    if not np.any(row == Color.BLACK):
        row[random.randrange(width)] = Color.BLACK
    if not np.any(row != Color.BLACK):
        row[random.randrange(width)] = Color.BLUE
    assert np.any(row != Color.BLACK)
    assert np.any(row == Color.BLACK)
    return row


def _panel_count_pattern(panel_count: int) -> tuple[int, ...]:
    if panel_count == 2:
        return random.choice(((1, 1), (2, 1), (1, 2), (2, 2)))
    return random.choice(
        (
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 2, 1),
            (1, 2, 2),
        )
    )


def _build_panels(panel_count: int, panel_w: int, upper_h: int) -> tuple[np.ndarray, list[np.ndarray], list[Panel]]:
    width = panel_count * panel_w + (panel_count - 1)
    upper = np.zeros((upper_h, width), dtype=int)
    panels: list[Panel] = []
    col = 0
    for i in range(panel_count):
        panels.append((col, col + panel_w))
        col += panel_w
        if i != panel_count - 1:
            upper[:, col] = Color.GREY
            col += 1

    counts = _panel_count_pattern(panel_count)
    seen_patterns: set[tuple[int, ...]] = set()
    for panel_idx, ((c0, c1), count) in enumerate(zip(panels, counts)):
        rows = sorted(random.sample(range(upper_h), count))
        for local_idx, r in enumerate(rows):
            for _ in range(40):
                row = _row_pattern(panel_w)
                key = tuple(map(int, row))
                if key not in seen_patterns or (panel_idx + local_idx) % 2 == 0:
                    seen_patterns.add(key)
                    upper[r, c0:c1] = row
                    break
            else:
                upper[r, c0:c1] = _row_pattern(panel_w)

    compressed = _collect_rows(upper, panels, right_to_left=True, discard_empty=True)
    return upper, compressed, panels


def _choose_hole_heights(tile_len: int, hole_count: int) -> list[int]:
    base = max(7, 2 * tile_len)
    candidates = list(range(base, base + 4))
    heights: list[int] = []
    for i in range(hole_count):
        pool = [h for h in candidates if (i == 0 and h % tile_len != 0) or i > 0]
        if not pool:
            pool = candidates
        choice = random.choice(pool)
        heights.append(choice)
        candidates = [h for h in candidates if h != choice] or [choice]
    return heights


def _carve_notches(obj: np.ndarray, protected: np.ndarray) -> None:
    h, w = obj.shape
    attempts = 0
    placed = 0
    target = random.randint(1, 3)
    while attempts < 20 and placed < target:
        attempts += 1
        side = random.choice(("top", "bottom", "left", "right"))
        depth = random.randint(1, 2)
        if side in ("top", "bottom"):
            notch_w = random.randint(1, max(1, w // 3))
            c0 = random.randint(0, w - notch_w)
            rr = slice(0, depth) if side == "top" else slice(h - depth, h)
            candidate = np.zeros_like(obj, dtype=bool)
            candidate[rr, c0 : c0 + notch_w] = True
        else:
            notch_h = random.randint(2, max(2, h // 3))
            r0 = random.randint(0, h - notch_h)
            cc = slice(0, depth) if side == "left" else slice(w - depth, w)
            candidate = np.zeros_like(obj, dtype=bool)
            candidate[r0 : r0 + notch_h, cc] = True

        if np.any(candidate & protected):
            continue
        if np.all(obj[~candidate] == Color.BLACK):
            continue
        obj[candidate] = Color.BLACK
        placed += 1


def _build_object(grid_w: int, panel_w: int, tile_len: int, hole_count: int) -> tuple[np.ndarray, list[BBox]]:
    heights = _choose_hole_heights(tile_len, hole_count)
    min_obj_h = max(heights) + 4
    obj_h = min_obj_h + random.randint(0, 2)

    if hole_count == 1:
        min_obj_w = panel_w + 4
    else:
        min_obj_w = 2 * panel_w + 5
    assert min_obj_w <= grid_w
    obj_w = grid_w - random.randint(0, min(2, grid_w - min_obj_w))

    obj = np.full((obj_h, obj_w), Color.GREEN, dtype=int)
    holes_local: list[BBox] = []

    if hole_count == 1:
        hole_h = heights[0]
        hr0 = 2 + random.randint(0, obj_h - hole_h - 4)
        hc0 = 2 + random.randint(0, obj_w - panel_w - 4)
        holes_local.append((hr0, hc0, hr0 + hole_h - 1, hc0 + panel_w - 1))
    else:
        remaining = obj_w - 4 - 2 * panel_w
        gap = random.randint(1, remaining)
        extra = remaining - gap
        left_extra = random.randint(0, extra)
        c1 = 2 + left_extra
        c2 = c1 + panel_w + gap
        for hole_h, hc0 in zip(heights, (c1, c2)):
            hr0 = 2 + random.randint(0, obj_h - hole_h - 4)
            holes_local.append((hr0, hc0, hr0 + hole_h - 1, hc0 + panel_w - 1))

    protected = np.zeros_like(obj, dtype=bool)
    for hr0, hc0, hr1, hc1 in holes_local:
        obj[hr0 : hr1 + 1, hc0 : hc1 + 1] = Color.BLACK
        protected[max(0, hr0 - 1) : min(obj_h, hr1 + 2), max(0, hc0 - 1) : min(obj_w, hc1 + 2)] = True

    _carve_notches(obj, protected)
    if np.all(obj[0] == Color.BLACK):
        candidates = np.flatnonzero(~protected[0])
        obj[0, int(candidates[len(candidates) // 2])] = Color.GREEN

    comps = components_mask(obj == Color.GREEN, connectivity=8)
    assert len(comps) == 1, "object must stay connected"
    return obj, holes_local


def _alternate_outputs(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _mask, green_box = _largest_green_component(grid)
    upper_h = green_box[0]
    panels = _panel_slices(grid, upper_h)
    upper = grid[:upper_h]
    holes = _find_holes(grid, green_box)

    rows_lr = _collect_rows(upper, panels, right_to_left=False, discard_empty=True)
    rows_keep_empty = _collect_rows(upper, panels, right_to_left=True, discard_empty=False)
    no_tri = _fill_holes(grid, holes, _collect_rows(upper, panels, right_to_left=True, discard_empty=True), triangular_mode="none")
    wrong_order = _fill_holes(grid, holes, rows_lr, triangular_mode="hole")
    wrong_compress = _fill_holes(grid, holes, rows_keep_empty, triangular_mode="hole")
    wrong_frame = _fill_holes(grid, holes, _collect_rows(upper, panels, right_to_left=True, discard_empty=True), triangular_mode="object", green_top=green_box[0])
    return no_tri, wrong_order, wrong_compress, wrong_frame


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    green_mask, green_box = _largest_green_component(g)
    upper_h = green_box[0]
    assert upper_h >= 5, "upper region should be visibly separate"
    panels = _panel_slices(g, upper_h)
    panel_w = panels[0][1] - panels[0][0]
    upper = g[:upper_h]

    rows = _collect_rows(upper, panels, right_to_left=True, discard_empty=True)
    assert len(rows) >= 2
    assert len({tuple(map(int, row)) for row in rows}) >= 2, "compressed tile must have at least two distinct rows"
    assert any(np.all(upper[:, c0:c1] == Color.BLACK, axis=1).any() for c0, c1 in panels), "need empty rows to discard"

    holes = _find_holes(g, green_box)
    assert 1 <= len(holes) <= 2
    heights = []
    for hole in holes:
        r0, c0, r1, c1 = hole
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        heights.append(h)
        assert w == panel_w
        assert h >= 2 * len(rows)
        assert len(_triangular_offsets(h)) >= 3
    assert any(h % len(rows) != 0 for h in heights), "at least one hole should cut off a partial cycle"

    green_crop = crop_bbox(g, green_box)
    border_black = np.zeros_like(green_crop, dtype=bool)
    border_black[0, :] = green_crop[0, :] == Color.BLACK
    border_black[-1, :] = green_crop[-1, :] == Color.BLACK
    border_black[:, 0] = green_crop[:, 0] == Color.BLACK
    border_black[:, -1] = green_crop[:, -1] == Color.BLACK
    assert border_black.any(), "object should be irregular enough to expose exterior background inside its bbox"
    assert green_mask.sum() > 0

    correct = main(g)
    no_tri, wrong_order, wrong_compress, wrong_frame = _alternate_outputs(g)
    assert not np.array_equal(correct, no_tri), "triangular repaint must matter"
    assert not np.array_equal(correct, wrong_order), "right-to-left panel order must matter"
    assert not np.array_equal(correct, wrong_compress), "discarding empty rows must matter"
    assert not np.array_equal(correct, wrong_frame), "hole-local triangular reference frame must matter"

    for hole in holes:
        r0, c0, r1, c1 = hole
        for off in _triangular_offsets(r1 - r0 + 1):
            row = correct[r0 + off, c0 : c1 + 1]
            assert np.any(row == Color.YELLOW), "triangular row should contain marked pixels"


def _attempt_generate() -> np.ndarray | None:
    panel_count = random.choices((2, 3), weights=(2, 3))[0]
    panel_w = random.randint(4, 6)
    upper_h = random.randint(6, 8)
    upper, tile_rows, panels = _build_panels(panel_count, panel_w, upper_h)
    if len(tile_rows) < 2:
        return None

    grid_w = upper.shape[1]
    can_use_two_holes = panel_count == 3 and grid_w >= 2 * panel_w + 5
    hole_count = random.choices((1, 2), weights=(2, 3))[0] if can_use_two_holes else 1

    obj, holes_local = _build_object(grid_w, panel_w, len(tile_rows), hole_count)
    obj_c0 = random.randint(0, grid_w - obj.shape[1])
    obj_r0 = upper_h + random.randint(0, 2)
    grid_h = obj_r0 + obj.shape[0] + random.randint(0, 1)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[:upper_h, :] = upper
    grid[obj_r0 : obj_r0 + obj.shape[0], obj_c0 : obj_c0 + obj.shape[1]] = obj

    for c in range(grid_w):
        if np.all(upper[:, c] == Color.GREY):
            grid[:obj_r0, c] = Color.GREY

    for hr0, hc0, hr1, hc1 in holes_local:
        grid[obj_r0 + hr0 : obj_r0 + hr1 + 1, obj_c0 + hc0 : obj_c0 + hc1 + 1] = Color.BLACK

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
