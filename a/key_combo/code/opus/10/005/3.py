from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PATTERN = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MARKER_COLORS = (RED, GREEN, ORANGE)
SIDE_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_cells(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _contiguous(values: list[int]) -> bool:
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def detect_axis_and_control(grid: np.ndarray) -> tuple[str, int, list[tuple[int, int]], tuple[int, int]]:
    yellow = [(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)]
    assert len(yellow) >= 4, "expected a yellow axis and a yellow control pixel"

    row_groups: dict[int, list[int]] = {}
    col_groups: dict[int, list[int]] = {}
    for r, c in yellow:
        row_groups.setdefault(r, []).append(c)
        col_groups.setdefault(c, []).append(r)

    candidates: list[tuple[str, int, list[tuple[int, int]]]] = []
    for row, cols in row_groups.items():
        cols = sorted(cols)
        if len(cols) >= 3 and _contiguous(cols):
            candidates.append(("h", int(row), [(int(row), int(c)) for c in cols]))
    for col, rows in col_groups.items():
        rows = sorted(rows)
        if len(rows) >= 3 and _contiguous(rows):
            candidates.append(("v", int(col), [(int(r), int(col)) for r in rows]))

    assert candidates, "expected a unique contiguous yellow axis"
    max_len = max(len(cells) for _orient, _axis, cells in candidates)
    best = [(orient, axis, cells) for orient, axis, cells in candidates if len(cells) == max_len]
    assert len(best) == 1, "axis must be uniquely determined"

    orient, axis, axis_cells = best[0]
    axis_set = set(axis_cells)
    controls = [cell for cell in yellow if cell not in axis_set]
    assert len(controls) == 1, "expected exactly one off-axis yellow control pixel"
    return orient, int(axis), axis_cells, controls[0]


def complete_pattern(grid: np.ndarray) -> tuple[np.ndarray, str, int, tuple[int, int], list[tuple[int, int]]]:
    out = np.array(grid, dtype=int, copy=True)
    orient, axis, axis_cells, control = detect_axis_and_control(out)
    if orient == "v":
        source_is_negative = control[1] < axis
        blue_coords = np.argwhere(out == PATTERN)
        for r, c in blue_coords:
            if (c < axis) == source_is_negative:
                mc = 2 * axis - int(c)
                assert 0 <= mc < out.shape[1], "mirrored blue cell must stay in bounds"
                out[int(r), mc] = PATTERN
    else:
        source_is_negative = control[0] < axis
        blue_coords = np.argwhere(out == PATTERN)
        for r, c in blue_coords:
            if (r < axis) == source_is_negative:
                mr = 2 * axis - int(r)
                assert 0 <= mr < out.shape[0], "mirrored blue cell must stay in bounds"
                out[mr, int(c)] = PATTERN
    return out, orient, int(axis), control, axis_cells


def _principal_side(vector_r: float, vector_c: float) -> str:
    if abs(vector_r) >= abs(vector_c):
        return "down" if vector_r > 0 else "up"
    return "right" if vector_c > 0 else "left"


def marker_side(coords: np.ndarray, shape_mask: np.ndarray) -> str:
    h, w = shape_mask.shape
    sides: set[str] = set()
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and shape_mask[nr, nc]:
                if nr == r - 1:
                    sides.add("down")
                elif nr == r + 1:
                    sides.add("up")
                elif nc == c - 1:
                    sides.add("right")
                elif nc == c + 1:
                    sides.add("left")
    assert sides, "marker block must touch the completed motif"
    if len(sides) == 1:
        return next(iter(sides))

    block_center = coords.mean(axis=0)
    motif_center = np.argwhere(shape_mask).mean(axis=0)
    return _principal_side(float(block_center[0] - motif_center[0]), float(block_center[1] - motif_center[1]))


def _tile_step(side: str, tile: np.ndarray) -> int:
    return tile.shape[0] if side in {"up", "down"} else tile.shape[1]


def _paint_repetition(
    out: np.ndarray,
    tile: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    protect: np.ndarray,
) -> None:
    r0, c0, r1, c1 = bbox
    dr, dc = SIDE_TO_VEC[side]
    step = _tile_step(side, tile)
    h, w = tile.shape
    shift = 1
    while True:
        rr0 = r0 + dr * step * shift
        cc0 = c0 + dc * step * shift
        rr1 = rr0 + h - 1
        cc1 = cc0 + w - 1
        if rr0 < 0 or cc0 < 0 or rr1 >= out.shape[0] or cc1 >= out.shape[1]:
            break
        for tr in range(h):
            for tc in range(w):
                value = int(tile[tr, tc])
                if value == BG:
                    continue
                gr = rr0 + tr
                gc = cc0 + tc
                if protect[gr, gc]:
                    continue
                current = int(out[gr, gc])
                assert current in {BG, value}, "repeated marker tiles must not conflict"
                out[gr, gc] = value
        shift += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the blue motif by yellow-axis symmetry, then redirect each marker block's repetition using the completed motif.

    Input structure:
    - The background is black.
    - A central blue motif is incomplete on one side of a yellow horizontal or vertical axis.
    - One off-axis yellow control pixel marks the trusted source side, and 2-4 small red/green/orange marker blocks sit 4-adjacent to the motif or axis.

    Rule:
    - Find the contiguous yellow axis and use the side containing the off-axis yellow control pixel as the blue source side.
    - Mirror only the source-side blue pixels across the axis to complete the motif; keep the axis and control pixel unchanged.
    - For each marker block, determine which side of the completed motif plus axis it touches, and repeat the whole block outward in the opposite-from-motif direction with period equal to the block size along that direction, without covering blue or yellow protected pixels.

    Key insight:
    - The symmetry completion changes the motif's outline first, so marker-direction inference must be done after completion rather than on the incomplete input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out, _orient, _axis, _control, axis_cells = complete_pattern(g)
    axis_mask = np.zeros_like(out, dtype=bool)
    for r, c in axis_cells:
        axis_mask[r, c] = True
    motif_mask = (out == PATTERN) | axis_mask
    protect = (out == PATTERN) | (out == YELLOW)

    marker_mask = np.isin(out, MARKER_COLORS)
    blocks = components(marker_mask)
    assert 2 <= len(blocks) <= 4, "expected 2-4 marker blocks"
    for coords in blocks:
        tile, box = crop_cells(out, coords)
        side = marker_side(coords, motif_mask)
        _paint_repetition(out, tile, box, side, protect)
    return out


def _apply_transform(
    cells: list[tuple[int, int]],
    orient: str,
    source_side: str,
) -> list[tuple[int, int]]:
    transformed: list[tuple[int, int]] = []
    for r, c in cells:
        if source_side == "right":
            c = -c
        if orient == "h":
            r, c = c, r
        transformed.append((int(r), int(c)))
    return transformed


def _grow_source(axis_len: int, anchor_row: int, rng: random.Random) -> set[tuple[int, int]]:
    source = {(r, -1) for r in range(anchor_row + 1)}
    source.add((0, -2))
    allowed = {(r, c) for r in range(axis_len) for c in (-1, -2, -3)}
    target = rng.randint(max(5, len(source)), min(9, len(allowed)))
    while len(source) < target:
        frontier = []
        for r, c in sorted(source):
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in source:
                    weight = 5 - max(0, nr - anchor_row + 1)
                    frontier.extend([(nr, nc)] * max(1, weight))
        if not frontier:
            break
        source.add(rng.choice(frontier))
    return source


def _mirror_cells_vertical(source: set[tuple[int, int]]) -> set[tuple[int, int]]:
    mirrored = set(source)
    for r, c in list(source):
        mirrored.add((r, -c))
    return mirrored


def _shape_mask_from_cells(cells: set[tuple[int, int]], axis_len: int) -> np.ndarray:
    rows = [r for r, _ in cells] + list(range(axis_len))
    cols = [c for _, c in cells] + [0]
    r0 = min(rows) - 2
    r1 = max(rows) + 2
    c0 = min(cols) - 3
    c1 = max(cols) + 3
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r in range(axis_len):
        mask[r - r0, -c0] = True
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _canonical_marker_candidates(
    complete_blue: set[tuple[int, int]],
    axis_len: int,
    reserved: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    motif = set(complete_blue) | {(r, 0) for r in range(axis_len)}
    shape_mask = _shape_mask_from_cells(complete_blue, axis_len)
    offset_r = min([r for r, _ in complete_blue] + [0]) - 2
    offset_c = min([c for _, c in complete_blue] + [0]) - 3
    candidates: list[list[tuple[int, int]]] = []
    search_rows = range(-1, axis_len + 2)
    search_cols = range(-5, 6)
    patterns = [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1)],
    ]
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for r in search_rows:
        for c in search_cols:
            for pattern in patterns:
                cells = [(r + dr, c + dc) for dr, dc in pattern]
                if any(cell in motif or cell in reserved for cell in cells):
                    continue
                if any(abs(rr) > axis_len + 3 or abs(cc) > 6 for rr, cc in cells):
                    continue
                if any(any(abs(rr - sr) + abs(cc - sc) == 1 for sr, sc in reserved) for rr, cc in cells):
                    continue
                coords = np.array([(rr - offset_r, cc - offset_c) for rr, cc in cells], dtype=int)
                if not np.all((coords[:, 0] >= 0) & (coords[:, 0] < shape_mask.shape[0])):
                    continue
                if not np.all((coords[:, 1] >= 0) & (coords[:, 1] < shape_mask.shape[1])):
                    continue
                try:
                    side = marker_side(coords, shape_mask)
                except AssertionError:
                    continue
                dr, dc = SIDE_TO_VEC[side]
                if dr != 0:
                    span = max(rr for rr, _ in cells) - min(rr for rr, _ in cells) + 1
                    far = max(rr for rr, _ in cells) if dr > 0 else min(rr for rr, _ in cells)
                    room = (axis_len + 6) - far if dr > 0 else far + 6
                    if room < span:
                        continue
                else:
                    span = max(cc for _, cc in cells) - min(cc for _, cc in cells) + 1
                    far = max(cc for _, cc in cells) if dc > 0 else min(cc for _, cc in cells)
                    room = 6 - far if dc > 0 else far + 6
                    if room < span:
                        continue
                key = tuple(sorted(cells))
                if key in used_shapes:
                    continue
                used_shapes.add(key)
                candidates.append(cells)
    return candidates


def _translation_ok(
    transformed_markers: list[list[tuple[int, int]]],
    transformed_complete_blue: list[tuple[int, int]],
    transformed_axis: list[tuple[int, int]],
    shift_r: int,
    shift_c: int,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    motif_mask = np.zeros(shape, dtype=bool)
    for r, c in transformed_complete_blue + transformed_axis:
        rr = r + shift_r
        cc = c + shift_c
        motif_mask[rr, cc] = True
    for marker in transformed_markers:
        coords = np.array([(r + shift_r, c + shift_c) for r, c in marker], dtype=int)
        side = marker_side(coords, motif_mask)
        dr, dc = SIDE_TO_VEC[side]
        r0, c0, r1, c1 = bbox_coords(coords)
        span = (r1 - r0 + 1) if dr else (c1 - c0 + 1)
        if dr > 0 and h - 1 - r1 < span:
            return False
        if dr < 0 and r0 < span:
            return False
        if dc > 0 and w - 1 - c1 < span:
            return False
        if dc < 0 and c0 < span:
            return False
    return True


def _marker_blocks(grid: np.ndarray) -> list[np.ndarray]:
    return components(np.isin(grid, MARKER_COLORS))


def _count_side_changes(inp: np.ndarray) -> int:
    completed, _orient, _axis, _control, axis_cells = complete_pattern(inp)
    axis_mask = np.zeros_like(inp, dtype=bool)
    for r, c in axis_cells:
        axis_mask[r, c] = True
    before_mask = (inp == PATTERN) | axis_mask
    after_mask = (completed == PATTERN) | axis_mask
    changes = 0
    for block in _marker_blocks(inp):
        if marker_side(block, before_mask) != marker_side(block, after_mask):
            changes += 1
    return changes


def _validate_example(inp: np.ndarray) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30
    orient, axis, axis_cells, control = detect_axis_and_control(inp)
    assert orient in {"h", "v"}
    if orient == "v":
        assert control[1] != axis
    else:
        assert control[0] != axis
    blocks = _marker_blocks(inp)
    assert 2 <= len(blocks) <= 4
    assert _count_side_changes(inp) >= 1, "at least one marker must change side after completion"
    out = main(inp)
    assert not np.array_equal(inp, out), "puzzle must transform the grid"
    assert np.count_nonzero(out == PATTERN) > np.count_nonzero(inp == PATTERN)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        orient = rng.choice(["v", "h"])
        source_side = rng.choice(["left", "right"])
        axis_len = rng.randint(4, 7)
        anchor_row = rng.randint(1, axis_len - 2)

        source_blue = _grow_source(axis_len, anchor_row, rng)
        complete_blue = _mirror_cells_vertical(source_blue)
        special_marker = [(anchor_row + 1, 1)]
        control_options = []
        for dist in (3, 2, 4):
            for row in range(axis_len):
                cell = (row, -dist)
                if cell not in source_blue and cell not in complete_blue and cell not in special_marker:
                    control_options.append(cell)
        if not control_options:
            continue
        control = rng.choice(control_options)

        reserved = set(special_marker)
        reserved.add(control)
        other_candidates = _canonical_marker_candidates(complete_blue, axis_len, reserved)
        rng.shuffle(other_candidates)
        chosen_markers = [special_marker]
        occupied = set(special_marker)
        target_count = rng.randint(2, 4)
        for candidate in other_candidates:
            candidate_set = set(candidate)
            if occupied & candidate_set:
                continue
            if any(any(abs(r1 - r2) + abs(c1 - c2) == 1 for r2, c2 in occupied) for r1, c1 in candidate_set):
                continue
            chosen_markers.append(candidate)
            occupied |= candidate_set
            if len(chosen_markers) == target_count:
                break
        if len(chosen_markers) < 2:
            continue

        axis_cells = [(r, 0) for r in range(axis_len)]
        transformed_blue = _apply_transform(sorted(source_blue), orient, source_side)
        transformed_complete_blue = _apply_transform(sorted(complete_blue), orient, source_side)
        transformed_axis = _apply_transform(axis_cells, orient, source_side)
        transformed_control = _apply_transform([control], orient, source_side)[0]
        transformed_markers = [_apply_transform(marker, orient, source_side) for marker in chosen_markers]

        all_cells = transformed_complete_blue + transformed_axis + [transformed_control]
        for marker in transformed_markers:
            all_cells.extend(marker)
        rows = [r for r, _ in all_cells]
        cols = [c for _, c in all_cells]

        height = rng.randint(18, 25)
        width = rng.randint(18, 25)
        min_shift_r = 1 - min(rows)
        max_shift_r = height - 2 - max(rows)
        min_shift_c = 1 - min(cols)
        max_shift_c = width - 2 - max(cols)
        if min_shift_r > max_shift_r or min_shift_c > max_shift_c:
            continue

        shifts = [
            (sr, sc)
            for sr in range(min_shift_r, max_shift_r + 1)
            for sc in range(min_shift_c, max_shift_c + 1)
        ]
        rng.shuffle(shifts)
        shift = None
        for sr, sc in shifts:
            if _translation_ok(
                transformed_markers,
                transformed_complete_blue,
                transformed_axis,
                sr,
                sc,
                (height, width),
            ):
                shift = (sr, sc)
                break
        if shift is None:
            continue

        sr, sc = shift
        grid = np.zeros((height, width), dtype=int)
        for r, c in transformed_blue:
            grid[r + sr, c + sc] = PATTERN
        for r, c in transformed_axis:
            grid[r + sr, c + sc] = YELLOW
        grid[transformed_control[0] + sr, transformed_control[1] + sc] = YELLOW

        marker_colors = [rng.choice(MARKER_COLORS) for _ in transformed_markers]
        for color, marker in zip(marker_colors, transformed_markers):
            for r, c in marker:
                rr = r + sr
                cc = c + sc
                if grid[rr, cc] != BG:
                    break
                grid[rr, cc] = color

        if np.count_nonzero(grid == PATTERN) < 5:
            continue
        try:
            _validate_example(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid example")
