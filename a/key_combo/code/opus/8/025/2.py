from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SOURCE_COLOR = 2
MARKER_COLOR = 3
QUADRANT_COLORS = {
    "UL": 1,
    "UR": 4,
    "LL": 7,
    "LR": 6,
}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield int(r + dr), int(c + dc)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(comp, dtype=int))
    return comps


def _find_marker_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in components_mask(grid == MARKER_COLOR, connectivity=4):
        if len(coords) != 4:
            continue
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        if r1 - r0 == 1 and c1 - c0 == 1 and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == MARKER_COLOR):
            candidates.append(box)
    assert len(candidates) == 1, "expected exactly one solid 2x2 green marker square"
    return candidates[0]


def _quadrant_of(r: int, c: int, center_r: int, center_c: int) -> str:
    upper = int(r) <= int(center_r)
    left = int(c) <= int(center_c)
    if upper and left:
        return "UL"
    if upper and not left:
        return "UR"
    if not upper and left:
        return "LL"
    return "LR"


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)
    assert bg == BG, "expected black background"

    marker_box = _find_marker_box(g)
    r0, c0, r1, c1 = marker_box
    center_r = r0 + 1
    center_c = c0 + 1
    marker_cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}

    singleton_components = []
    for color in sorted(set(map(int, np.unique(g))) - {bg, MARKER_COLOR}):
        for coords in components_mask(g == color, connectivity=4):
            if len(coords) == 1:
                rr, cc = map(int, coords[0])
                singleton_components.append((color, rr, cc))

    assert len(singleton_components) == 4, "expected four singleton corner markers"

    remaining = singleton_components[:]
    corner_targets = {
        "UL": (0, 0),
        "UR": (0, w - 1),
        "LL": (h - 1, 0),
        "LR": (h - 1, w - 1),
    }
    quadrant_marker_colors: dict[str, int] = {}
    corner_coords: set[tuple[int, int]] = set()
    for quadrant, corner in corner_targets.items():
        pick = min(
            remaining,
            key=lambda item: abs(item[1] - corner[0]) + abs(item[2] - corner[1]),
        )
        remaining.remove(pick)
        color, rr, cc = pick
        quadrant_marker_colors[quadrant] = int(color)
        corner_coords.add((int(rr), int(cc)))

    assert quadrant_marker_colors == QUADRANT_COLORS, "expected canonical quadrant marker colors"

    source_mask = np.zeros_like(g, dtype=bool)
    for rr, cc in np.argwhere(g != bg):
        rr = int(rr)
        cc = int(cc)
        if (rr, cc) in marker_cells or (rr, cc) in corner_coords:
            continue
        source_mask[rr, cc] = True

    source_coords = np.argwhere(source_mask)
    assert 5 <= len(source_coords) <= 12, "source shape must contain 5..12 pixels"
    assert np.all(g[source_mask] == SOURCE_COLOR), "expected a red source shape"
    source_components = components_mask(source_mask, connectivity=4)
    assert len(source_components) == 1, "source shape must be 4-connected"

    source_quadrants = {
        _quadrant_of(int(rr), int(cc), center_r, center_c)
        for rr, cc in source_coords
    }
    assert len(source_quadrants) == 1, "source shape must stay within one quadrant under the puzzle tie rules"

    for quadrant, corner in corner_targets.items():
        marker_pos = next((rr, cc) for color, rr, cc in singleton_components if color == quadrant_marker_colors[quadrant])
        dr = abs(marker_pos[0] - corner[0])
        dc = abs(marker_pos[1] - corner[1])
        assert dr in (1, 2) and dc in (1, 2), "corner markers must sit 1..2 cells from the corner"

    return {
        "grid": g,
        "shape": (h, w),
        "marker_box": marker_box,
        "center": (center_r, center_c),
        "marker_cells": marker_cells,
        "corner_coords": corner_coords,
        "source_coords": [(int(rr), int(cc)) for rr, cc in source_coords],
        "source_quadrant": next(iter(source_quadrants)),
        "quadrant_marker_colors": quadrant_marker_colors,
    }


def _expected_output(parsed: dict) -> np.ndarray:
    g = parsed["grid"]
    (r0, c0, r1, c1) = parsed["marker_box"]
    center_r, center_c = parsed["center"]
    marker_cells = parsed["marker_cells"]
    corner_coords = parsed["corner_coords"]
    quadrant_marker_colors = parsed["quadrant_marker_colors"]

    row_sum = r0 + r1
    col_sum = c0 + c1
    out = g.copy()
    for r, c in parsed["source_coords"]:
        reflections = {
            (r, c),
            (r, col_sum - c),
            (row_sum - r, c),
            (row_sum - r, col_sum - c),
        }
        for rr, cc in reflections:
            if not in_bounds(out.shape, rr, cc):
                continue
            if (rr, cc) in marker_cells or (rr, cc) in corner_coords:
                raise AssertionError("reflections must not collide with the marker square or corner markers")
            if (rr, cc) == (r, c):
                out[rr, cc] = SOURCE_COLOR
            else:
                quadrant = _quadrant_of(rr, cc, center_r, center_c)
                out[rr, cc] = quadrant_marker_colors[quadrant]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the red shape by four-way mirroring around the 2x2 green marker and recolor each reflected quadrant from its corner marker.

    Input structure:
    - Black background with exactly one solid 2x2 green marker square near the center.
    - A 4-connected irregular red source shape lies entirely in one tie-aware quadrant of that marker.
    - Four singleton corner markers sit 1..2 cells from the corners: blue at upper-left, yellow at upper-right, orange at lower-left, and magenta at lower-right.

    Rule:
    - Use the intersection point at the center of the 2x2 green square as both mirror axes.
    - For every red source pixel, generate its horizontal reflection, vertical reflection, and 180-degree rotation partner.
    - Keep the original source pixels red.
    - Recolor every newly created reflected pixel by the quadrant it lands in: upper-left blue, upper-right yellow, lower-left orange, lower-right magenta, using the visible corner markers as the quadrant-color key.
    - When a reflected pixel lies on a mirror axis, break ties toward the upper quadrant for row ties and toward the left quadrant for column ties.
    - Keep the green marker square and the four corner markers unchanged; discard only reflections that would fall outside the grid.

    Key insight:
    - The green 2x2 square defines the geometric mirror center, while the corner markers independently define the color to assign to each mirrored quadrant, so geometry and coloring both matter.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    return _expected_output(parsed)


def _axis_capacities(h: int, w: int, center_r: int, center_c: int) -> dict[str, int]:
    return {
        "top": min(center_r, h - center_r),
        "bottom": min(h - 1 - center_r, center_r - 1),
        "left": min(center_c, w - center_c),
        "right": min(w - 1 - center_c, center_c - 1),
    }


def _candidate_cells(
    shape: tuple[int, int],
    center_r: int,
    center_c: int,
    quadrant: str,
    max_row_span: int,
    max_col_span: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    caps = _axis_capacities(h, w, center_r, center_c)
    cells: list[tuple[int, int]] = []

    if quadrant in ("UL", "UR"):
        row_offsets = range(0, min(caps["top"], max_row_span) + 1)
    else:
        row_offsets = range(1, min(caps["bottom"], max_row_span) + 1)

    if quadrant in ("UL", "LL"):
        col_offsets = range(0, min(caps["left"], max_col_span) + 1)
    else:
        col_offsets = range(1, min(caps["right"], max_col_span) + 1)

    for dr in row_offsets:
        for dc in col_offsets:
            if quadrant == "UL":
                r, c = center_r - dr, center_c - dc
            elif quadrant == "UR":
                r, c = center_r - dr, center_c + dc
            elif quadrant == "LL":
                r, c = center_r + dr, center_c - dc
            else:
                r, c = center_r + dr, center_c + dc

            if not in_bounds(shape, r, c) or (r, c) in blocked:
                continue

            reflections = {
                (r, c),
                (r, 2 * center_c - 1 - c),
                (2 * center_r - 1 - r, c),
                (2 * center_r - 1 - r, 2 * center_c - 1 - c),
            }
            if any(not in_bounds(shape, rr, cc) or (rr, cc) in blocked for rr, cc in reflections):
                continue
            cells.append((r, c))
    return cells


def _frontier(component: set[tuple[int, int]], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = set()
    for r, c in component:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in allowed and (nr, nc) not in component:
                out.add((nr, nc))
    return sorted(out)


def _is_irregular(component: set[tuple[int, int]]) -> bool:
    coords = np.asarray(sorted(component), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if len(component) == h * w:
        return False
    rows = {r for r, _ in component}
    cols = {c for _, c in component}
    if len(rows) == 1 or len(cols) == 1:
        return False
    mask = np.zeros((h, w), dtype=bool)
    for r, c in component:
        mask[r - r0, c - c0] = True
    if np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask)):
        return False
    return True


def _make_connected_shape(
    allowed_cells: list[tuple[int, int]],
    target_area: int,
    center_r: int,
    center_c: int,
    quadrant: str,
    force_axis_touch: bool,
) -> set[tuple[int, int]] | None:
    allowed = set(allowed_cells)
    axis_cells = {
        (r, c)
        for r, c in allowed
        if r == center_r or c == center_c
    }

    for _ in range(140):
        if force_axis_touch and axis_cells:
            seed = random.choice(sorted(axis_cells))
        else:
            seed = random.choice(sorted(allowed))
        comp = {seed}
        while len(comp) < target_area:
            choices = _frontier(comp, allowed)
            if not choices:
                break
            if force_axis_touch and not (comp & axis_cells):
                axis_frontier = [cell for cell in choices if cell in axis_cells]
                if axis_frontier:
                    choices = axis_frontier
            weights = []
            for r, c in choices:
                weight = 3
                if r == center_r or c == center_c:
                    weight += 2
                if quadrant in ("UL", "UR") and r == center_r:
                    weight += 1
                if quadrant in ("UL", "LL") and c == center_c:
                    weight += 1
                weights.append(weight)
            comp.add(random.choices(choices, weights=weights, k=1)[0])

        if len(comp) != target_area:
            continue
        if force_axis_touch and axis_cells and not (comp & axis_cells):
            continue
        if not _is_irregular(comp):
            continue
        return comp
    return None


def _place_corner_markers(grid: np.ndarray) -> set[tuple[int, int]]:
    h, w = grid.shape
    placements = {
        "UL": (random.randint(1, 2), random.randint(1, 2)),
        "UR": (random.randint(1, 2), w - 1 - random.randint(1, 2)),
        "LL": (h - 1 - random.randint(1, 2), random.randint(1, 2)),
        "LR": (h - 1 - random.randint(1, 2), w - 1 - random.randint(1, 2)),
    }
    coords = set()
    for quadrant, (r, c) in placements.items():
        grid[r, c] = QUADRANT_COLORS[quadrant]
        coords.add((r, c))
    return coords


def _choose_marker_top_left(h: int, w: int) -> tuple[int, int]:
    row_candidates = list(range(max(3, h // 2 - 2), min(h - 5, h // 2 + 1) + 1))
    col_candidates = list(range(max(3, w // 2 - 2), min(w - 5, w // 2 + 1) + 1))
    return random.choice(row_candidates), random.choice(col_candidates)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(14, 19)
    w = random.randint(14, 19)
    grid = np.full((h, w), BG, dtype=int)
    corner_coords = _place_corner_markers(grid)

    marker_r, marker_c = _choose_marker_top_left(h, w)
    grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = MARKER_COLOR
    marker_cells = {
        (marker_r, marker_c),
        (marker_r, marker_c + 1),
        (marker_r + 1, marker_c),
        (marker_r + 1, marker_c + 1),
    }
    blocked = marker_cells | corner_coords
    center_r, center_c = marker_r + 1, marker_c + 1

    quadrant = random.choice(["UL", "UR", "LL", "LR"])
    force_axis_touch = quadrant != "LR" and random.random() < 0.8

    for _ in range(120):
        row_span = random.randint(2, 5)
        col_span = random.randint(2, 5)
        allowed = _candidate_cells(
            grid.shape,
            center_r,
            center_c,
            quadrant,
            row_span,
            col_span,
            blocked,
        )
        if len(allowed) < 5:
            continue
        max_area = min(12, len(allowed))
        area = random.randint(5, max_area)
        shape = _make_connected_shape(
            allowed,
            area,
            center_r,
            center_c,
            quadrant,
            force_axis_touch,
        )
        if shape is None:
            continue

        for r, c in shape:
            grid[r, c] = SOURCE_COLOR

        try:
            parsed = _parse_input(grid)
            expected = _expected_output(parsed)
        except AssertionError:
            for r, c in shape:
                grid[r, c] = BG
            continue

        if np.array_equal(expected, grid):
            for r, c in shape:
                grid[r, c] = BG
            continue

        if not _validate_semantics(grid, expected):
            for r, c in shape:
                grid[r, c] = BG
            continue

        return grid

    return None


def _validate_semantics(inp: np.ndarray, out: np.ndarray | None = None) -> bool:
    try:
        parsed = _parse_input(inp)
        expected = _expected_output(parsed)
        if out is None:
            out = main(inp)
        if not np.array_equal(out, expected):
            return False

        marker_box = parsed["marker_box"]
        r0, c0, r1, c1 = marker_box
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == MARKER_COLOR):
            return False

        for rr, cc in parsed["corner_coords"]:
            quadrant = _quadrant_of(rr, cc, parsed["center"][0], parsed["center"][1])
            if out[rr, cc] != QUADRANT_COLORS[quadrant]:
                return False

        source_quadrant = parsed["source_quadrant"]
        seen_quadrants = {source_quadrant}
        for r, c in parsed["source_coords"]:
            if out[r, c] != SOURCE_COLOR:
                return False
            reflections = {
                (r, c),
                (r, c0 + c1 - c),
                (r0 + r1 - r, c),
                (r0 + r1 - r, c0 + c1 - c),
            }
            for rr, cc in reflections:
                if not in_bounds(out.shape, rr, cc):
                    return False
                seen_quadrants.add(_quadrant_of(rr, cc, parsed["center"][0], parsed["center"][1]))
        if seen_quadrants != {"UL", "UR", "LL", "LR"}:
            return False
        if np.count_nonzero(out != inp) < 3:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is None:
            continue
        if _validate_semantics(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid four-way symmetry recolor input")
