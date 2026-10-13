from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
MARKER_COLORS = (1, 3, 6, 7)
DIRECTIONS = ("up", "down", "left", "right")
HORIZONTAL_DIRECTIONS = {"left", "right"}
VERTICAL_DIRECTIONS = {"up", "down"}

GEN_GRID_H = (15, 22)
GEN_GRID_W = (15, 22)
GEN_OBJECT_COUNT = (3, 4)
GEN_AXIS_H = (3, 5)
GEN_MARKER_BIAS_OFFSHAPE = 0.75
GEN_ATTEMPTS = 500
GEN_UNIT_ATTEMPTS = 250


def get_bg(grid: np.ndarray) -> int:
    """Return the dominant grid color."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox for an (N,2) coordinate array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Return connected components of a boolean mask as coordinate arrays."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros_like(m, dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not m[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _decode_direction(marker_r: int, marker_c: int, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h > 1 and w > 1, "mirror footprint bbox must have height and width > 1"
    row_frac = (int(marker_r) - r0) / (h - 1)
    col_frac = (int(marker_c) - c0) / (w - 1)
    if abs(row_frac - 0.5) >= abs(col_frac - 0.5):
        return "up" if row_frac < 0.5 else "down"
    return "left" if col_frac < 0.5 else "right"


def _slide_coords(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> np.ndarray:
    moved = np.asarray(coords, dtype=int).copy()
    h, w = map(int, shape)
    r0, c0, r1, c1 = bbox_coords(moved)
    if direction == "up":
        moved[:, 0] -= r0
    elif direction == "down":
        moved[:, 0] += h - 1 - r1
    elif direction == "left":
        moved[:, 1] -= c0
    elif direction == "right":
        moved[:, 1] += w - 1 - c1
    else:
        raise ValueError(f"unknown direction {direction}")
    return moved


def _vertical_axis(segment: np.ndarray) -> tuple[int, set[int]]:
    cols = {int(c) for _r, c in segment}
    assert len(cols) == 1, "red component must be a vertical segment"
    axis_col = next(iter(cols))
    axis_rows = {int(r) for r, _c in segment}
    return axis_col, axis_rows


def _touches_axis_8(coords: np.ndarray, axis_col: int, axis_rows: set[int]) -> bool:
    for r, c in coords:
        if abs(int(c) - int(axis_col)) != 1:
            continue
        rr = int(r)
        if rr - 1 in axis_rows or rr in axis_rows or rr + 1 in axis_rows:
            return True
    return False


def _yellow_for_axis(
    yellow_objects: list[np.ndarray],
    axis_col: int,
    axis_rows: set[int],
    used: set[int],
) -> tuple[int, np.ndarray]:
    matches = []
    for idx, coords in enumerate(yellow_objects):
        if idx in used:
            continue
        cols = coords[:, 1]
        if not (np.all(cols < axis_col) or np.all(cols > axis_col)):
            continue
        if _touches_axis_8(coords, axis_col, axis_rows):
            matches.append((idx, coords))
    assert len(matches) == 1, "each red segment must pair with exactly one yellow object"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each yellow object to a grid edge by decoding direction from the marker inside its invisible mirror footprint, then recolor it with the marker color.

    Input structure:
    - The grid has a dominant background, several red vertical axis segments, and one 4-connected yellow object 8-adjacent to each axis on exactly one side.
    - On the opposite side of each axis, inside the bounding box of that yellow object's horizontal reflection, there is exactly one single-pixel marker.
    - Marker colors come from a small non-yellow, non-red palette and axis segments do not touch the grid border.

    Rule:
    - For each red segment, find its unique adjacent yellow object and reflect that object's pixels across the segment column.
    - Use the reflected pixels only to define the mirror-footprint bounding box; find the unique marker inside that box.
    - Convert the marker's normalized position inside the box into one of four slide directions, move the original yellow object until it becomes edge-aligned in that direction, and recolor it with the marker color.
    - Remove all red segments and markers from the output, keeping only background plus the recolored slid objects.

    Key insight:
    - The mirror is not drawn; it creates the coordinate frame that turns marker position into a direction code, while marker color simultaneously supplies the final paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    red_segments = components_mask(g == RED, connectivity=4)
    yellow_objects = components_mask(g == YELLOW, connectivity=4)
    assert red_segments, "expected at least one red segment"
    assert len(red_segments) == len(yellow_objects), "expected one yellow object per red segment"

    out = np.full_like(g, bg)
    used: set[int] = set()
    for segment in red_segments:
        axis_col, axis_rows = _vertical_axis(segment)
        yellow_idx, yellow = _yellow_for_axis(yellow_objects, axis_col, axis_rows, used)
        used.add(yellow_idx)

        reflected = yellow.copy()
        reflected[:, 1] = 2 * axis_col - reflected[:, 1]
        r0, c0, r1, c1 = bbox_coords(reflected)
        bbox = (r0, c0, r1, c1)

        box = g[r0 : r1 + 1, c0 : c1 + 1]
        marker_cells = np.argwhere(np.isin(box, MARKER_COLORS))
        assert marker_cells.shape[0] == 1, "mirror footprint bbox must contain exactly one marker"
        mr, mc = map(int, marker_cells[0])
        marker_color = int(box[mr, mc])
        assert set(map(int, np.unique(box))).issubset({bg, marker_color}), (
            "mirror footprint bbox must contain only background and its marker"
        )

        marker_r = r0 + mr
        marker_c = c0 + mc
        direction = _decode_direction(marker_r, marker_c, bbox)
        moved = _slide_coords(yellow, direction, g.shape)

        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < g.shape[0]))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < g.shape[1]))
        assert np.all(out[moved[:, 0], moved[:, 1]] == bg), "slid objects must not overlap"
        out[moved[:, 0], moved[:, 1]] = marker_color

    assert len(used) == len(yellow_objects)
    return out


def _marker_candidates(h: int, w: int, direction: str) -> list[tuple[int, int]]:
    candidates = []
    for r in range(h):
        row_frac = r / (h - 1)
        if row_frac == 0.5:
            continue
        for c in range(w):
            col_frac = c / (w - 1)
            if col_frac == 0.5:
                continue
            if _decode_direction(r, c, (0, 0, h - 1, w - 1)) == direction:
                candidates.append((r, c))
    return candidates


def _grow_shape(raw_h: int, raw_w: int, area: int) -> np.ndarray:
    start = (random.randrange(raw_h), random.randrange(raw_w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((raw_h, raw_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _sample_unit(direction: str, marker_color: int) -> dict | None:
    for _ in range(GEN_UNIT_ATTEMPTS):
        if direction in HORIZONTAL_DIRECTIONS:
            axis_h = random.randint(4, GEN_AXIS_H[1])
            raw_h = random.randint(4, axis_h)
            raw_w = random.randint(4, 5)
            min_area = 4
        else:
            axis_h = random.randint(*GEN_AXIS_H)
            raw_h = random.randint(2, axis_h)
            raw_w = random.randint(2, 5)
            min_area = 3

        max_area = min(7, raw_h * raw_w)
        if min_area > max_area:
            continue

        mask = _grow_shape(raw_h, raw_w, random.randint(min_area, max_area))
        if int(mask.sum()) < 3 or int(mask.sum()) > 7:
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue

        h, w = mask.shape
        if h < 2 or w < 2:
            continue
        if direction in HORIZONTAL_DIRECTIONS and h < 4:
            continue

        candidates = _marker_candidates(h, w, direction)
        if not candidates:
            continue

        offset_r = random.randint(0, axis_h - h)
        side = random.choice(["left", "right"])
        reflected_mask = np.fliplr(mask)
        offshape = [(r, c) for r, c in candidates if not reflected_mask[r, c]]
        if offshape and random.random() < GEN_MARKER_BIAS_OFFSHAPE:
            marker_local = random.choice(offshape)
        else:
            marker_local = random.choice(candidates)

        rect_w = 2 * w + 1
        axis_col = w
        patch = np.full((axis_h, rect_w), BG, dtype=int)
        patch[:, axis_col] = RED

        source_top = 0 if side == "left" else w + 1
        mirror_top = w + 1 if side == "left" else 0
        source_coords = np.argwhere(mask).astype(int)
        source_coords[:, 0] += offset_r
        source_coords[:, 1] += source_top
        patch[source_coords[:, 0], source_coords[:, 1]] = YELLOW

        marker_r = offset_r + marker_local[0]
        marker_c = mirror_top + marker_local[1]
        patch[marker_r, marker_c] = int(marker_color)

        return {
            "patch": patch,
            "direction": direction,
            "marker_color": int(marker_color),
            "axis_h": int(axis_h),
            "rect_w": int(rect_w),
            "shape_h": int(h),
            "shape_w": int(w),
            "offset_r": int(offset_r),
            "source_coords": source_coords,
            "marker_on_reflected_shape": bool(reflected_mask[marker_local[0], marker_local[1]]),
        }
    return None


def _placement_options(
    unit: dict,
    blocked: np.ndarray,
    output_occ: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    h, w = blocked.shape
    unit_h = int(unit["axis_h"])
    unit_w = int(unit["rect_w"])
    source_coords = np.asarray(unit["source_coords"], dtype=int)
    direction = str(unit["direction"])

    options: list[tuple[int, int, np.ndarray]] = []
    for r0 in range(1, h - unit_h):
        for c0 in range(1, w - unit_w):
            if blocked[r0 - 1 : r0 + unit_h + 1, c0 - 1 : c0 + unit_w + 1].any():
                continue
            placed = source_coords + np.array([r0, c0], dtype=int)
            moved = _slide_coords(placed, direction, blocked.shape)
            if np.any(output_occ[moved[:, 0], moved[:, 1]]):
                continue
            options.append((r0, c0, moved))
    random.shuffle(options)
    return options


def _place_units_recursive(
    units: list[dict],
    idx: int,
    blocked: np.ndarray,
    output_occ: np.ndarray,
    placements: list[dict],
) -> list[dict] | None:
    if idx == len(units):
        return placements

    unit = units[idx]
    options = _placement_options(unit, blocked, output_occ)
    if not options:
        return None

    for r0, c0, moved in options:
        next_blocked = blocked.copy()
        unit_h = int(unit["axis_h"])
        unit_w = int(unit["rect_w"])
        next_blocked[r0 - 1 : r0 + unit_h + 1, c0 - 1 : c0 + unit_w + 1] = True

        next_output = output_occ.copy()
        next_output[moved[:, 0], moved[:, 1]] = True

        result = _place_units_recursive(
            units,
            idx + 1,
            next_blocked,
            next_output,
            placements + [{"unit": unit, "r0": r0, "c0": c0, "moved": moved}],
        )
        if result is not None:
            return result
    return None


def _build_grids(h: int, w: int, placements: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for placement in placements:
        unit = placement["unit"]
        r0 = int(placement["r0"])
        c0 = int(placement["c0"])
        patch = np.asarray(unit["patch"], dtype=int)
        ph, pw = patch.shape
        view = inp[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BG
        assert not np.any(view[mask] != BG)
        view[mask] = patch[mask]

        moved = np.asarray(placement["moved"], dtype=int)
        color = int(unit["marker_color"])
        assert not np.any(out[moved[:, 0], moved[:, 1]] != BG)
        out[moved[:, 0], moved[:, 1]] = color

    return inp, out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, placements: list[dict]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    directions = {str(item["unit"]["direction"]) for item in placements}
    if len(directions) != len(placements):
        return False
    if not (directions & VERTICAL_DIRECTIONS and directions & HORIZONTAL_DIRECTIONS):
        return False

    uses_shorter_bbox = any(
        int(item["unit"]["shape_h"]) < int(item["unit"]["axis_h"]) for item in placements
    )
    uses_bbox_not_shape = any(
        not bool(item["unit"]["marker_on_reflected_shape"]) for item in placements
    )
    if not (uses_shorter_bbox and uses_bbox_not_shape):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*GEN_OBJECT_COUNT)
    directions = random.sample(DIRECTIONS, n_objects)
    marker_colors = random.sample(list(MARKER_COLORS), n_objects)

    units = []
    for direction, marker_color in zip(directions, marker_colors):
        unit = _sample_unit(direction, marker_color)
        if unit is None:
            return None
        units.append(unit)

    min_h = max(15, max(int(unit["axis_h"]) for unit in units) + 2)
    min_w = max(15, max(int(unit["rect_w"]) for unit in units) + 2)
    max_h = max(min_h, GEN_GRID_H[1])
    max_w = max(min_w, GEN_GRID_W[1])
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    units.sort(key=lambda item: int(item["axis_h"]) * int(item["rect_w"]), reverse=True)
    placements = _place_units_recursive(
        units,
        0,
        np.zeros((h, w), dtype=bool),
        np.zeros((h, w), dtype=bool),
        [],
    )
    if placements is None:
        return None

    inp, expected = _build_grids(h, w, placements)
    if not _validate_semantics(inp, expected, placements):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-slide task")
