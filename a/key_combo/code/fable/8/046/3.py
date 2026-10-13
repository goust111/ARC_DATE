from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


GRAY = 5
RED = 2
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.ravel()]
    assert vals
    return int(Counter(vals).most_common(1)[0][0])


def _detect_axis(grid: np.ndarray) -> tuple[str, int]:
    gray_rows = np.flatnonzero(np.all(grid == GRAY, axis=1))
    gray_cols = np.flatnonzero(np.all(grid == GRAY, axis=0))
    if len(gray_rows) == 1 and len(gray_cols) == 0:
        return "h", int(gray_rows[0])
    if len(gray_cols) == 1 and len(gray_rows) == 0:
        return "v", int(gray_cols[0])
    raise AssertionError("expected exactly one full gray symmetry axis")


def _panel_backgrounds(grid: np.ndarray, orient: str, axis: int) -> tuple[int, int]:
    if orient == "h":
        side0 = grid[:axis, :]
        side1 = grid[axis + 1 :, :]
    else:
        side0 = grid[:, :axis]
        side1 = grid[:, axis + 1 :]
    assert side0.size > 0 and side1.size > 0
    bg0 = _dominant_color(side0)
    bg1 = _dominant_color(side1)
    assert bg0 != bg1
    assert bg0 != GRAY and bg1 != GRAY
    return bg0, bg1


def _side_of(r: int, c: int, orient: str, axis: int) -> int:
    if orient == "h":
        assert r != axis
        return 0 if r < axis else 1
    assert c != axis
    return 0 if c < axis else 1


def _coords_single_side(coords: np.ndarray, orient: str, axis: int) -> int:
    sides = {_side_of(int(r), int(c), orient, axis) for r, c in coords}
    assert len(sides) == 1
    return next(iter(sides))


def _mirror_coords(coords: np.ndarray, orient: str, axis: int) -> np.ndarray:
    mirrored = np.array(coords, dtype=int, copy=True)
    if orient == "h":
        mirrored[:, 0] = 2 * axis - mirrored[:, 0]
    else:
        mirrored[:, 1] = 2 * axis - mirrored[:, 1]
    return mirrored


def _copy_mask(grid: np.ndarray, orient: str, axis: int, side_bgs: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (GRAY, RED):
                continue
            side = _side_of(r, c, orient, axis)
            if color != side_bgs[side]:
                mask[r, c] = True
    return mask


def _red_index(red_components: list[np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    idx = np.full(shape, -1, dtype=int)
    sets: list[set[tuple[int, int]]] = []
    for comp_id, coords in enumerate(red_components):
        cell_set = {tuple(map(int, p)) for p in coords}
        sets.append(cell_set)
        for r, c in cell_set:
            idx[r, c] = comp_id
    return idx, sets


def _copy_is_matched(
    coords: np.ndarray,
    orient: str,
    axis: int,
    red_idx: np.ndarray,
    red_sets: list[set[tuple[int, int]]],
) -> bool:
    mirrored = _mirror_coords(coords, orient, axis)
    h, w = red_idx.shape
    if np.any(mirrored[:, 0] < 0) or np.any(mirrored[:, 0] >= h):
        return False
    if np.any(mirrored[:, 1] < 0) or np.any(mirrored[:, 1] >= w):
        return False
    comp_ids = {int(red_idx[r, c]) for r, c in mirrored}
    if -1 in comp_ids or len(comp_ids) != 1:
        return False
    comp_id = next(iter(comp_ids))
    return {tuple(map(int, p)) for p in mirrored} == red_sets[comp_id]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Maintain mirror copies across the gray axis by adding missing copies and erasing orphan copies.

    Input structure:
    - One full gray line spans the entire grid and splits the canvas into two panels with different base colors.
    - Red connected shapes are originals and may appear in either panel.
    - Non-red foreground shapes are candidate copies; a copy uses the base color of the opposite panel, namely the panel that contains its red original.

    Rule:
    - For every red original, reflect its full shape across the gray line; if the mirrored side lacks that copy, draw the mirrored shape there using the original panel's base color.
    - For every non-red foreground copy candidate, reflect it across the gray line; if its mirrored partner is not an exact red original, erase that copy back to the local panel base color.
    - Leave the gray axis and every already-correct original/copy pair unchanged.

    Key insight:
    - Role is determined by color semantics rather than shape alone: red always means source original, while a visible non-red shape is only valid when its color matches the opposite panel identity and its footprint mirrors an exact red original.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orient, axis = _detect_axis(g)
    side_bgs = _panel_backgrounds(g, orient, axis)

    red_components = _components(g == RED)
    assert red_components, "expected at least one red original"
    red_idx, red_sets = _red_index(red_components, g.shape)

    out = g.copy()
    for coords in red_components:
        side = _coords_single_side(coords, orient, axis)
        mirrored = _mirror_coords(coords, orient, axis)
        assert _coords_single_side(mirrored, orient, axis) == 1 - side
        out[mirrored[:, 0], mirrored[:, 1]] = side_bgs[side]

    for coords in _components(_copy_mask(g, orient, axis, side_bgs)):
        side = _coords_single_side(coords, orient, axis)
        if not _copy_is_matched(coords, orient, axis, red_idx, red_sets):
            out[coords[:, 0], coords[:, 1]] = side_bgs[side]

    return out


PANEL_COLORS = [0, 1, 3, 4, 6, 7, 8, 9]
SIDE_DEPTH_RANGE = (7, 12)
SPAN_RANGE = (18, 24)
SHAPE_H_RANGE = (2, 4)
SHAPE_W_RANGE = (2, 4)
SHAPE_FILL_RANGE = (0.45, 0.8)
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
MAX_SHAPE_ATTEMPTS = 80
MAX_PLACE_ATTEMPTS = 200
MAX_GENERATE_ATTEMPTS = 400


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(SHAPE_H_RANGE[0], min(SHAPE_H_RANGE[1], max_h))
        w = random.randint(SHAPE_W_RANGE[0], min(SHAPE_W_RANGE[1], max_w))
        area_min = max(3, int(np.ceil(SHAPE_FILL_RANGE[0] * h * w)))
        area_max = max(area_min, int(np.floor(SHAPE_FILL_RANGE[1] * h * w)))
        area_max = min(area_max, h * w - 1 if h * w > 4 else h * w)
        if area_min > area_max:
            continue
        target = random.randint(area_min, area_max)

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for rr in range(h):
                for cc in range(w):
                    if mask[rr, cc]:
                        continue
                    if any(
                        0 <= rr + dr < h and 0 <= cc + dc < w and mask[rr + dr, cc + dc]
                        for dr, dc in DIR4
                    ):
                        frontier.append((rr, cc))
            if not frontier:
                break
            mask[random.choice(frontier)] = True

        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if cropped.sum() < 3 or cropped.sum() == cropped.size:
            continue
        return cropped
    return None


def _expanded_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - PLACEMENT_GAP),
        max(0, c0 - PLACEMENT_GAP),
        min(h - 1, r1 + PLACEMENT_GAP),
        min(w - 1, c1 + PLACEMENT_GAP),
    )


def _mark_reserved(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _placeable(
    reserved: np.ndarray,
    bbox0: tuple[int, int, int, int],
    bbox1: tuple[int, int, int, int],
) -> bool:
    for bbox in (bbox0, bbox1):
        r0, c0, r1, c1 = bbox
        if np.any(reserved[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _shape_cells(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += top
    coords[:, 1] += left
    return coords.astype(int)


def _sample_pair_placement(
    grid_shape: tuple[int, int],
    orient: str,
    axis: int,
    band: int,
    side: int,
    mask: np.ndarray,
    reserved: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = mask.shape
    H, W = grid_shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        if orient == "h":
            if h > band or w > W - 2 * PLACEMENT_MARGIN:
                return None
            near = random.randint(1, band - h + 1)
            left = random.randint(PLACEMENT_MARGIN, W - w - PLACEMENT_MARGIN)
            if side == 0:
                src_bottom = axis - near
                src_top = src_bottom - h + 1
            else:
                src_top = axis + near
            src = _shape_cells(mask, src_top, left)
        else:
            if w > band or h > H - 2 * PLACEMENT_MARGIN:
                return None
            near = random.randint(1, band - w + 1)
            top = random.randint(PLACEMENT_MARGIN, H - h - PLACEMENT_MARGIN)
            if side == 0:
                src_right = axis - near
                src_left = src_right - w + 1
            else:
                src_left = axis + near
            src = _shape_cells(mask, top, src_left)

        copy = _mirror_coords(src, orient, axis)
        src_bbox = _bbox(src)
        copy_bbox = _bbox(copy)

        if np.any(copy[:, 0] == axis) or np.any(copy[:, 1] == axis):
            continue
        if np.any(src[:, 0] == axis) or np.any(src[:, 1] == axis):
            continue
        if not np.array_equal(_mirror_coords(src, orient, axis), copy):
            continue

        expanded_src = _expanded_bbox(src_bbox, grid_shape)
        expanded_copy = _expanded_bbox(copy_bbox, grid_shape)
        if not _placeable(reserved, expanded_src, expanded_copy):
            continue

        _mark_reserved(reserved, expanded_src)
        _mark_reserved(reserved, expanded_copy)
        return src, copy
    return None


def _build_output_grid(
    orient: str,
    side0_depth: int,
    side1_depth: int,
    span: int,
    side_bgs: tuple[int, int],
) -> tuple[np.ndarray, int]:
    if orient == "h":
        grid = np.full((side0_depth + 1 + side1_depth, span), side_bgs[1], dtype=int)
        grid[:side0_depth, :] = side_bgs[0]
        axis = side0_depth
        grid[axis, :] = GRAY
    else:
        grid = np.full((span, side0_depth + 1 + side1_depth), side_bgs[1], dtype=int)
        grid[:, :side0_depth] = side_bgs[0]
        axis = side0_depth
        grid[:, axis] = GRAY
    return grid, axis


def _real_templates() -> list[tuple[int, str]]:
    items = [(0, "missing"), (1, "missing"), (0, "intact"), (1, "intact")]
    random.shuffle(items)
    return items


def _orphan_source_sides() -> list[int]:
    if random.random() < 0.5:
        return [random.choice([0, 1])]
    return random.sample([0, 1], k=2)


def _classify_copy_components(
    grid: np.ndarray,
    orient: str,
    axis: int,
    side_bgs: tuple[int, int],
) -> tuple[int, int]:
    red_components = _components(grid == RED)
    red_idx, red_sets = _red_index(red_components, grid.shape)
    matched = 0
    orphan = 0
    for comp in _components(_copy_mask(grid, orient, axis, side_bgs)):
        if _copy_is_matched(comp, orient, axis, red_idx, red_sets):
            matched += 1
        else:
            orphan += 1
    return matched, orphan


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    real_records: list[dict],
    orphan_records: list[dict],
    orient: str,
    axis: int,
    side_bgs: tuple[int, int],
) -> bool:
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(inp), out):
        return False

    found_orient, found_axis = _detect_axis(inp)
    if (found_orient, found_axis) != (orient, axis):
        return False
    if _panel_backgrounds(inp, orient, axis) != side_bgs:
        return False

    statuses = {rec["status"] for rec in real_records}
    if statuses != {"missing", "intact"}:
        return False

    red_visible_sides = {rec["side"] for rec in real_records}
    copy_visible_sides = {rec["side"] for rec in real_records if rec["status"] == "intact"}
    if red_visible_sides != {0, 1} or copy_visible_sides != {0, 1}:
        return False
    if not orphan_records:
        return False

    for rec in real_records:
        src = rec["source"]
        cp = rec["copy"]
        side = rec["side"]
        source_bg = side_bgs[side]
        target_bg = side_bgs[1 - side]
        if not np.all(out[src[:, 0], src[:, 1]] == RED):
            return False
        if not np.all(out[cp[:, 0], cp[:, 1]] == source_bg):
            return False

        if rec["status"] == "missing":
            if not np.all(inp[src[:, 0], src[:, 1]] == RED):
                return False
            if not np.all(inp[cp[:, 0], cp[:, 1]] == target_bg):
                return False
        else:
            if not np.all(inp[src[:, 0], src[:, 1]] == RED):
                return False
            if not np.all(inp[cp[:, 0], cp[:, 1]] == source_bg):
                return False

    for rec in orphan_records:
        src = rec["source"]
        cp = rec["copy"]
        side = rec["side"]
        source_bg = side_bgs[side]
        target_bg = side_bgs[1 - side]
        if not np.all(out[src[:, 0], src[:, 1]] == source_bg):
            return False
        if not np.all(out[cp[:, 0], cp[:, 1]] == target_bg):
            return False
        if not np.all(inp[src[:, 0], src[:, 1]] == source_bg):
            return False
        if not np.all(inp[cp[:, 0], cp[:, 1]] == source_bg):
            return False

    input_red_sides = {_coords_single_side(comp, orient, axis) for comp in _components(inp == RED)}
    if input_red_sides != {0, 1}:
        return False

    matched, orphan = _classify_copy_components(inp, orient, axis, side_bgs)
    if matched < 1 or orphan < 1:
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    orient = random.choice(["h", "v"])
    side0_depth = random.randint(*SIDE_DEPTH_RANGE)
    side1_depth = random.randint(*SIDE_DEPTH_RANGE)
    span = random.randint(*SPAN_RANGE)
    side_bgs = tuple(random.sample(PANEL_COLORS, 2))
    output, axis = _build_output_grid(orient, side0_depth, side1_depth, span, side_bgs)
    reserved = np.zeros(output.shape, dtype=bool)
    band = min(side0_depth, side1_depth)

    real_records: list[dict] = []
    for side, status in _real_templates():
        placed = None
        for _ in range(MAX_SHAPE_ATTEMPTS):
            if orient == "h":
                shape = _random_shape(band, output.shape[1] - 2 * PLACEMENT_MARGIN)
            else:
                shape = _random_shape(output.shape[0] - 2 * PLACEMENT_MARGIN, band)
            if shape is None:
                continue
            placed = _sample_pair_placement(output.shape, orient, axis, band, side, shape, reserved)
            if placed is not None:
                break
        if placed is None:
            return None
        src, cp = placed
        output[src[:, 0], src[:, 1]] = RED
        output[cp[:, 0], cp[:, 1]] = side_bgs[side]
        real_records.append({"side": side, "status": status, "source": src, "copy": cp})

    orphan_records: list[dict] = []
    for side in _orphan_source_sides():
        placed = None
        for _ in range(MAX_SHAPE_ATTEMPTS):
            if orient == "h":
                shape = _random_shape(band, output.shape[1] - 2 * PLACEMENT_MARGIN)
            else:
                shape = _random_shape(output.shape[0] - 2 * PLACEMENT_MARGIN, band)
            if shape is None:
                continue
            placed = _sample_pair_placement(output.shape, orient, axis, band, side, shape, reserved)
            if placed is not None:
                break
        if placed is None:
            return None
        src, cp = placed
        orphan_records.append({"side": side, "source": src, "copy": cp})

    inp = output.copy()
    for rec in real_records:
        side = rec["side"]
        target_bg = side_bgs[1 - side]
        cp = rec["copy"]
        if rec["status"] == "missing":
            inp[cp[:, 0], cp[:, 1]] = target_bg
    for rec in orphan_records:
        side = rec["side"]
        source_bg = side_bgs[side]
        cp = rec["copy"]
        inp[cp[:, 0], cp[:, 1]] = source_bg

    if not _validate_case(inp, output, real_records, orphan_records, orient, axis, side_bgs):
        return None
    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid mirror-maintenance puzzle")
