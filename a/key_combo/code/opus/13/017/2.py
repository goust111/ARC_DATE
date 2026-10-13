from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _bbox_from_mask(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _draw_border(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _marker_info(grid: np.ndarray) -> tuple[int, BBox]:
    candidates: list[tuple[int, BBox]] = []
    for color in map(int, np.unique(grid)):
        if color == 0:
            continue
        mask = grid == color
        if int(np.count_nonzero(mask)) != 4:
            continue
        bbox = _bbox_from_mask(mask)
        assert bbox is not None
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((color, bbox))
    assert len(candidates) == 1, "expected exactly one solid 2x2 marker block"
    return candidates[0]


def _frame_layers(grid: np.ndarray) -> tuple[list[dict[str, int | BBox]], int]:
    marker_color, _ = _marker_info(grid)
    layers: list[dict[str, int | BBox]] = []
    for color in map(int, np.unique(grid)):
        if color in (0, marker_color):
            continue
        bbox = _bbox_from_mask(grid == color)
        assert bbox is not None
        layers.append({"color": color, "bbox": bbox, "area": _bbox_area(bbox)})
    assert len(layers) >= 2, "expected nested frame colors"

    for layer in layers:
        bbox = layer["bbox"]
        assert isinstance(bbox, tuple)
        depth = sum(
            1
            for other in layers
            if other is not layer and _contains(other["bbox"], bbox) and other["bbox"] != bbox
        )
        layer["depth"] = int(depth)

    deepest = max(int(layer["depth"]) for layer in layers)
    protected = [layer for layer in layers if int(layer["depth"]) == deepest]
    assert len(protected) == 1, "expected a unique innermost frame"
    return layers, int(protected[0]["color"])


def _erase_cross(
    grid: np.ndarray, protected_color: int, marker_color: int, marker_bbox: BBox
) -> np.ndarray:
    r0, c0, r1, c1 = marker_bbox
    cross = np.zeros(grid.shape, dtype=bool)
    cross[r0 : r1 + 1, :] = True
    cross[:, c0 : c1 + 1] = True

    out = np.array(grid, copy=True)
    erase_mask = cross & (grid != 0) & (grid != protected_color)
    out[erase_mask] = 0
    out[marker_bbox[0] : marker_bbox[2] + 1, marker_bbox[1] : marker_bbox[3] + 1] = 0
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a 2x2 marker's cross-erasure to selectively complete damaged nested frames.

    Input structure:
    - The grid has black background and one solid 2x2 marker block.
    - Several non-background colors form nested sparse rectangular frame borders.
    - The frames are distinguished by containment depth, not by any fixed color identity.

    Rule:
    - Determine the innermost frame from bbox containment.
    - Erase every foreground cell in the marker's two rows and two columns, but keep the innermost frame protected; remove the marker itself.
    - Any frame that loses pixels is redrawn as a complete 1-pixel rectangular border using the bbox of its surviving pixels.
    - Draw completed frames from largest bbox to smallest, then redraw every undamaged sparse frame on top.

    Key insight:
    - Protection is assigned by topology: the smallest contained frame survives the cross unchanged, while only the damaged outer layers get completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_color, marker_bbox = _marker_info(g)
    layers, protected_color = _frame_layers(g)
    erased = _erase_cross(g, protected_color, marker_color, marker_bbox)

    completed: list[tuple[int, BBox, int]] = []
    sparse: list[tuple[int, np.ndarray, int]] = []
    for layer in layers:
        color = int(layer["color"])
        before = g == color
        after = erased == color
        lost = int(np.count_nonzero(before & ~after))
        if lost > 0:
            bbox = _bbox_from_mask(after)
            assert bbox is not None, "damaged frame must retain its bbox anchors"
            completed.append((_bbox_area(bbox), bbox, color))
        else:
            sparse.append((int(layer["area"]), after, color))

    out = np.zeros_like(g)
    for _area, bbox, color in sorted(completed, key=lambda item: item[0], reverse=True):
        _draw_border(out, bbox, color)
    for _area, mask, color in sorted(sparse, key=lambda item: item[0], reverse=True):
        out[mask] = color
    return out


_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_ATTEMPTS = 500
_MODES = ("outer_only", "middle_only", "both")


def _pair_starts(lo: int, hi: int) -> list[int]:
    return list(range(lo, hi))


def _band_candidates(outer: BBox, inner: BBox) -> tuple[list[int], list[int]]:
    row_starts = _pair_starts(outer[0] + 1, inner[0] - 1) + _pair_starts(inner[2] + 1, outer[2] - 1)
    col_starts = _pair_starts(outer[1] + 1, inner[1] - 1) + _pair_starts(inner[3] + 1, outer[3] - 1)
    return row_starts, col_starts


def _safe_sample(values: list[int], k: int) -> list[int] | None:
    uniq = sorted(set(values))
    if len(uniq) < k:
        return None
    return random.sample(uniq, k)


def _sparse_frame_points(
    bbox: BBox,
    cross_rows: set[int],
    cross_cols: set[int],
    trigger_dims: set[str],
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    points: set[tuple[int, int]] = set()

    top_safe = [c for c in range(c0, c1 + 1) if c not in cross_cols]
    bottom_safe = [c for c in range(c0, c1 + 1) if c not in cross_cols]
    left_safe = [r for r in range(r0, r1 + 1) if r not in cross_rows]
    right_safe = [r for r in range(r0, r1 + 1) if r not in cross_rows]
    if min(len(top_safe), len(bottom_safe), len(left_safe), len(right_safe)) < 2:
        return None

    forced_top: set[int] = set()
    forced_bottom: set[int] = set()
    forced_left: set[int] = set()
    forced_right: set[int] = set()

    if "col" in trigger_dims:
        hit_cols = [c for c in cross_cols if c0 < c < c1]
        if not hit_cols:
            return None
        hit_col = random.choice(hit_cols)
        side_choice = random.choice(("top", "bottom", "both"))
        if side_choice in ("top", "both"):
            forced_top.add(hit_col)
        if side_choice in ("bottom", "both"):
            forced_bottom.add(hit_col)

    if "row" in trigger_dims:
        hit_rows = [r for r in cross_rows if r0 < r < r1]
        if not hit_rows:
            return None
        hit_row = random.choice(hit_rows)
        side_choice = random.choice(("left", "right", "both"))
        if side_choice in ("left", "both"):
            forced_left.add(hit_row)
        if side_choice in ("right", "both"):
            forced_right.add(hit_row)

    def add_horizontal(row: int, safe_cols: list[int], forced_cols: set[int]) -> bool:
        anchors = _safe_sample(safe_cols, 2)
        if anchors is None:
            return False
        cols = set(anchors) | set(forced_cols)
        extra_pool = [c for c in safe_cols if c not in cols]
        extra_count = 0 if not extra_pool else random.randint(0, min(2, len(extra_pool)))
        if extra_count:
            cols.update(random.sample(extra_pool, extra_count))
        if len(cols) < 2:
            return False
        for col in cols:
            points.add((row, int(col)))
        return True

    def add_vertical(col: int, safe_rows: list[int], forced_rows: set[int]) -> bool:
        anchors = _safe_sample(safe_rows, 2)
        if anchors is None:
            return False
        rows = set(anchors) | set(forced_rows)
        extra_pool = [r for r in safe_rows if r not in rows]
        extra_count = 0 if not extra_pool else random.randint(0, min(2, len(extra_pool)))
        if extra_count:
            rows.update(random.sample(extra_pool, extra_count))
        if len(rows) < 2:
            return False
        for row in rows:
            points.add((int(row), col))
        return True

    ok = (
        add_horizontal(r0, top_safe, forced_top)
        and add_horizontal(r1, bottom_safe, forced_bottom)
        and add_vertical(c0, left_safe, forced_left)
        and add_vertical(c1, right_safe, forced_right)
    )
    return points if ok else None


def _damage_signature(grid: np.ndarray) -> tuple[tuple[int, ...], int]:
    marker_color, marker_bbox = _marker_info(grid)
    layers, protected = _frame_layers(grid)
    erased = _erase_cross(grid, protected, marker_color, marker_bbox)
    damaged: list[int] = []
    for layer in sorted(layers, key=lambda item: int(item["area"])):
        color = int(layer["color"])
        before = grid == color
        after = erased == color
        if int(np.count_nonzero(before & ~after)) > 0:
            damaged.append(color)
            bbox = _bbox_from_mask(after)
            assert bbox == layer["bbox"], "surviving sparse pixels must preserve the original bbox"
        else:
            assert np.array_equal(before, after), "undamaged frames must remain unchanged"
    return tuple(damaged), protected


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(5, 8)
    inner_w = random.randint(5, 8)
    gap_om = random.randint(3, 4)
    gap_mi = random.randint(3, 4)

    middle_h = inner_h + 2 * gap_mi
    middle_w = inner_w + 2 * gap_mi
    outer_h = middle_h + 2 * gap_om
    outer_w = middle_w + 2 * gap_om

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    h = outer_h + top_margin + bottom_margin
    w = outer_w + left_margin + right_margin
    if max(h, w) > 30:
        return None

    outer = (top_margin, left_margin, top_margin + outer_h - 1, left_margin + outer_w - 1)
    middle = (outer[0] + gap_om, outer[1] + gap_om, outer[2] - gap_om, outer[3] - gap_om)
    inner = (middle[0] + gap_mi, middle[1] + gap_mi, middle[2] - gap_mi, middle[3] - gap_mi)

    outer_rows, outer_cols = _band_candidates(outer, middle)
    middle_rows, middle_cols = _band_candidates(middle, inner)
    if not outer_rows or not outer_cols or not middle_rows or not middle_cols:
        return None

    mode = random.choice(_MODES)
    if mode == "outer_only":
        marker_r = random.choice(outer_rows)
        marker_c = random.choice(outer_cols)
        outer_triggers = random.choice(({"row"}, {"col"}, {"row", "col"}))
        middle_triggers = set()
    elif mode == "middle_only":
        marker_r = random.choice(middle_rows)
        marker_c = random.choice(middle_cols)
        outer_triggers = set()
        middle_triggers = random.choice(({"row"}, {"col"}, {"row", "col"}))
    else:
        if random.choice((False, True)):
            marker_r = random.choice(outer_rows)
            marker_c = random.choice(middle_cols)
            outer_triggers = random.choice(({"row"}, {"row", "col"}))
            middle_triggers = {"col"}
        else:
            marker_r = random.choice(middle_rows)
            marker_c = random.choice(outer_cols)
            outer_triggers = random.choice(({"col"}, {"row", "col"}))
            middle_triggers = {"row"}

    cross_rows = {marker_r, marker_r + 1}
    cross_cols = {marker_c, marker_c + 1}

    frame_bboxes = [outer, middle, inner]
    trigger_sets = [outer_triggers, middle_triggers, set()]
    layer_points: list[set[tuple[int, int]]] = []
    for bbox, triggers in zip(frame_bboxes, trigger_sets):
        points = _sparse_frame_points(bbox, cross_rows, cross_cols, triggers)
        if points is None:
            return None
        layer_points.append(points)

    palette = random.sample(_PALETTE, 3)
    random.shuffle(palette)

    grid = np.zeros((h, w), dtype=int)
    for color, points in zip(palette, layer_points):
        for r, c in points:
            if grid[r, c] != 0:
                return None
            grid[r, c] = color

    if np.any(grid[marker_r : marker_r + 2, marker_c : marker_c + 2] != 0):
        return None
    grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = 5

    damaged, protected = _damage_signature(grid)
    if protected != palette[2]:
        return None
    if mode == "outer_only" and len(damaged) != 1:
        return None
    if mode == "middle_only" and len(damaged) != 1:
        return None
    if mode == "both" and len(damaged) != 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame cross-erasure input")
