from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _component_center(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return _round_half_up(float(np.mean(rows))), _round_half_up(float(np.mean(cols)))


def _rotate_offset(dr: int, dc: int, rot: int) -> tuple[int, int]:
    rot %= 4
    if rot == 0:
        return dr, dc
    if rot == 1:
        return dc, -dr
    if rot == 2:
        return -dr, -dc
    return -dc, dr


def _rotation_for_marker(marker: tuple[int, int], center: tuple[int, int]) -> int:
    dr = marker[0] - center[0]
    dc = marker[1] - center[1]
    if dr < 0 and dc >= 0:
        return 1
    if dr < 0 and dc < 0:
        return 2
    if dr >= 0 and dc < 0:
        return 3
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each colored stamp at its same-color markers.

    For each color:
    - find the unique multi-pixel 4-connected component as the source stamp
    - find all same-color singleton components as markers
    - compute the stamp center by averaging pixel coordinates and rounding
    - rotate the stamp by the marker's quadrant relative to that center
    - anchor the rotated stamp so the center lands on the marker
    - paint only previously black cells
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    for color in [int(c) for c in np.unique(g) if int(c) != BG]:
        comps = _components(g == color)
        stamp_comps = [comp for comp in comps if len(comp) > 1]
        marker_comps = [comp for comp in comps if len(comp) == 1]
        assert len(stamp_comps) == 1, f"color {color} must have exactly one stamp"

        stamp = stamp_comps[0]
        center = _component_center(stamp)
        offsets = [(r - center[0], c - center[1]) for r, c in stamp]

        for comp in marker_comps:
            marker = comp[0]
            rot = _rotation_for_marker(marker, center)
            for dr, dc in offsets:
                rr_off, cc_off = _rotate_offset(dr, dc, rot)
                rr = marker[0] + rr_off
                cc = marker[1] + cc_off
                if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == BG:
                    out[rr, cc] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
COLOR_COUNT_RANGE = (2, 4)
MARKER_COUNT_RANGE = (1, 4)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 140


def _coords_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(mask)
    rows = pts[:, 0]
    cols = pts[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _connected_coords(coords: tuple[tuple[int, int], ...]) -> bool:
    mask = _mask_from_coords(coords)
    return len(_components(mask)) == 1


def _orbit_coords(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    orbit: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for transformed in (rot, np.fliplr(rot)):
            coords = _coords_from_mask(transformed)
            if coords not in seen:
                seen.add(coords)
                orbit.append(coords)
    return orbit


def _build_stamp_library() -> list[tuple[tuple[int, int], ...]]:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen_orbits: set[tuple[tuple[int, int], ...]] = set()
    for h in range(2, 5):
        for w in range(2, 5):
            for area in range(3, min(6, h * w) + 1):
                for cells in itertools.combinations(range(h * w), area):
                    mask = np.zeros((h, w), dtype=bool)
                    for idx in cells:
                        mask[idx // w, idx % w] = True
                    if mask.all():
                        continue
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    coords = _coords_from_mask(mask)
                    if not _connected_coords(coords):
                        continue
                    orbit = _orbit_coords(mask)
                    if sum(1 for item in orbit if item == coords) != 1:
                        continue
                    key = min(orbit)
                    if key in seen_orbits:
                        continue
                    seen_orbits.add(key)
                    shapes.append(coords)
    return shapes


STAMP_LIBRARY = _build_stamp_library()


def _random_oriented_shape() -> tuple[tuple[int, int], ...]:
    coords = random.choice(STAMP_LIBRARY)
    mask = _mask_from_coords(coords)
    return _coords_from_mask(np.rot90(mask, random.randrange(4)))


def _translate(coords: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r + r0, c + c0) for r, c in coords}


def _margin_ok(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], margin: int = 1) -> bool:
    if not occupied:
        return True
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _shape_center_local(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return _component_center(list(coords))


def _footprint_for_anchor(
    coords: tuple[tuple[int, int], ...],
    center_local: tuple[int, int],
    rot: int,
    anchor: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = set()
    for r, c in coords:
        dr = r - center_local[0]
        dc = c - center_local[1]
        rr_off, cc_off = _rotate_offset(dr, dc, rot)
        cells.add((anchor[0] + rr_off, anchor[1] + cc_off))
    return cells


def _marker_rotations(count: int) -> list[int]:
    order = [0, 1, 2, 3]
    random.shuffle(order)
    rotations = order[: min(count, 4)]
    while len(rotations) < count:
        rotations.append(random.choice(order))
    random.shuffle(rotations)
    return rotations


def _anchor_ranges(
    center_abs: tuple[int, int],
    coords: tuple[tuple[int, int], ...],
    center_local: tuple[int, int],
    rot: int,
    h: int,
    w: int,
) -> tuple[range, range] | None:
    offsets = [_rotate_offset(r - center_local[0], c - center_local[1], rot) for r, c in coords]
    rows = [dr for dr, _ in offsets]
    cols = [dc for _, dc in offsets]
    row_lo = -min(rows)
    row_hi = h - 1 - max(rows)
    col_lo = -min(cols)
    col_hi = w - 1 - max(cols)

    if rot == 0:
        row_lo = max(row_lo, center_abs[0] + 1)
        col_lo = max(col_lo, center_abs[1] + 1)
    elif rot == 1:
        row_hi = min(row_hi, center_abs[0] - 1)
        col_lo = max(col_lo, center_abs[1] + 1)
    elif rot == 2:
        row_hi = min(row_hi, center_abs[0] - 1)
        col_hi = min(col_hi, center_abs[1] - 1)
    else:
        row_lo = max(row_lo, center_abs[0] + 1)
        col_hi = min(col_hi, center_abs[1] - 1)

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _candidate_anchors(
    center_abs: tuple[int, int],
    coords: tuple[tuple[int, int], ...],
    center_local: tuple[int, int],
    rot: int,
    h: int,
    w: int,
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
    ranges = _anchor_ranges(center_abs, coords, center_local, rot, h, w)
    if ranges is None:
        return []

    row_range, col_range = ranges
    row_values = list(row_range)
    col_values = list(col_range)
    random.shuffle(row_values)
    random.shuffle(col_values)

    anchors: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    for mr in row_values:
        for mc in col_values:
            anchor = (mr, mc)
            marker_cell = {anchor}
            if not _margin_ok(marker_cell, input_occupied):
                continue
            if not _margin_ok(marker_cell, output_occupied):
                continue
            footprint = _footprint_for_anchor(coords, center_local, rot, anchor)
            if not _in_bounds(footprint, h, w):
                continue
            if not _margin_ok(footprint, input_occupied):
                continue
            if not _margin_ok(footprint, output_occupied):
                continue
            anchors.append((anchor, footprint))
    return anchors


def _place_color(
    grid: np.ndarray,
    color: int,
    marker_count: int,
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        coords = _random_oriented_shape()
        center_local = _shape_center_local(coords)
        shape_h = max(r for r, _ in coords) + 1
        shape_w = max(c for _, c in coords) + 1

        top_rows = list(range(0, h - shape_h + 1))
        left_cols = list(range(0, w - shape_w + 1))
        random.shuffle(top_rows)
        random.shuffle(left_cols)

        placed = False
        for r0 in top_rows:
            if placed:
                break
            for c0 in left_cols:
                stamp_cells = _translate(coords, r0, c0)
                if not _margin_ok(stamp_cells, input_occupied):
                    continue
                if not _margin_ok(stamp_cells, output_occupied):
                    continue

                center_abs = (r0 + center_local[0], c0 + center_local[1])
                temp_input = set(input_occupied)
                temp_output = set(output_occupied)
                temp_input.update(stamp_cells)
                temp_output.update(stamp_cells)
                markers: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

                rotations = _marker_rotations(marker_count)
                ok = True
                for rot in rotations:
                    anchors = _candidate_anchors(
                        center_abs,
                        coords,
                        center_local,
                        rot,
                        h,
                        w,
                        temp_input,
                        temp_output,
                    )
                    if not anchors:
                        ok = False
                        break
                    anchor, footprint = random.choice(anchors)
                    markers.append((anchor, footprint))
                    temp_input.add(anchor)
                    temp_output.add(anchor)
                    temp_output.update(footprint)

                if not ok:
                    continue

                for r, c in stamp_cells:
                    grid[r, c] = color
                for anchor, _ in markers:
                    grid[anchor] = color

                input_occupied.update(temp_input - input_occupied)
                output_occupied.update(temp_output - output_occupied)
                placed = True
                break

        if placed:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    color_count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, color_count)

    marker_counts = [random.randint(*MARKER_COUNT_RANGE) for _ in range(color_count)]
    if sum(marker_counts) < color_count + 2:
        marker_counts[random.randrange(color_count)] += 1

    input_occupied: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()

    for color, marker_count in zip(colors, marker_counts, strict=False):
        if not _place_color(grid, color, marker_count, input_occupied, output_occupied):
            return None

    try:
        solved = main(grid)
    except Exception:
        return None

    if np.array_equal(solved, grid):
        return None

    for color in colors:
        comps = _components(grid == color)
        if sum(len(comp) > 1 for comp in comps) != 1:
            return None
        if sum(len(comp) == 1 for comp in comps) < 1:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-quadrant ARC input")
