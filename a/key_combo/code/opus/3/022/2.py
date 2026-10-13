from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
TRACK = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = (5, 7, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GENERATION_ATTEMPTS = 400
EXTRA_OBJECT_ATTEMPTS = 220


@dataclass(frozen=True)
class ObjSpec:
    color: int
    side: int
    init_center: tuple[int, int]
    direction: tuple[int, int]
    distance: int

    @property
    def half(self) -> int:
        return self.side // 2

    @property
    def shrunk_side(self) -> int:
        return self.side - 4

    @property
    def final_center(self) -> tuple[int, int]:
        r, c = self.init_center
        dr, dc = self.direction
        return r + dr * self.distance, c + dc * self.distance

    @property
    def marker(self) -> tuple[int, int]:
        r, c = self.final_center
        dr, dc = self.direction
        offset = self.half + 1
        return r + dr * offset, c + dc * offset


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_from_center(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    r, c = center
    h = side // 2
    return r - h, c - h, r + h, c + h


def _cells_in_box(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _square_cells(center: tuple[int, int], side: int) -> set[tuple[int, int]]:
    return _cells_in_box(_box_from_center(center, side))


def _track_cells(spec: ObjSpec) -> list[tuple[int, int]]:
    r0, c0 = spec.init_center
    dr, dc = spec.direction
    return [(r0 + dr * step, c0 + dc * step) for step in range(spec.distance)]


def _sweep_box(spec: ObjSpec) -> tuple[int, int, int, int]:
    a = _box_from_center(spec.init_center, spec.side)
    b = _box_from_center(spec.final_center, spec.side)
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _in_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _paint_square(out: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    r0, c0, r1, c1 = _box_from_center(center, side)
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray) -> list[ObjSpec]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert TRACK not in colors, "input must not contain the track color"
    objects: list[ObjSpec] = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form exactly one square and one marker"
        marker_parts = [comp for comp in comps if comp.shape[0] == 1]
        square_parts = [comp for comp in comps if comp.shape[0] > 1]
        assert len(marker_parts) == 1 and len(square_parts) == 1

        square_coords = square_parts[0]
        marker = tuple(map(int, marker_parts[0][0]))
        r0, c0, r1, c1 = _bbox_coords(square_coords)
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1
        assert side in SIDES
        assert square_coords.shape[0] == side * side
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "square must be solid"

        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        same_row = marker[0] == center[0]
        same_col = marker[1] == center[1]
        assert same_row ^ same_col, "marker must align with its square center on exactly one axis"
        half = side // 2

        if same_col and marker[0] < center[0]:
            direction = DIRS["up"]
            final_center = (marker[0] + half + 1, center[1])
        elif same_col and marker[0] > center[0]:
            direction = DIRS["down"]
            final_center = (marker[0] - half - 1, center[1])
        elif same_row and marker[1] < center[1]:
            direction = DIRS["left"]
            final_center = (center[0], marker[1] + half + 1)
        else:
            assert same_row and marker[1] > center[1]
            direction = DIRS["right"]
            final_center = (center[0], marker[1] - half - 1)

        distance = abs(final_center[0] - center[0]) + abs(final_center[1] - center[1])
        assert distance >= 0
        objects.append(
            ObjSpec(
                color=color,
                side=side,
                init_center=center,
                direction=direction,
                distance=distance,
            )
        )
    return objects


def _validate_core_scene(specs: list[ObjSpec], shape: tuple[int, int]) -> None:
    h, w = shape
    final_boxes: list[tuple[int, int, int, int]] = []
    actual_cells = {spec.color: _square_cells(spec.init_center, spec.side) | {spec.marker} for spec in specs}

    for spec in specs:
        for box in (
            _box_from_center(spec.init_center, spec.side),
            _box_from_center(spec.final_center, spec.side),
            _box_from_center(spec.final_center, spec.shrunk_side),
        ):
            r0, c0, r1, c1 = box
            assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w
        mr, mc = spec.marker
        assert 0 <= mr < h and 0 <= mc < w

    for i, spec in enumerate(specs):
        sweep = _sweep_box(spec)
        for j, other in enumerate(specs):
            if i == j:
                continue
            assert not any(_in_box(cell, sweep) for cell in actual_cells[other.color]), "slide corridor must be clear"

    for i, spec in enumerate(specs):
        box_i = _box_from_center(spec.final_center, spec.shrunk_side)
        cells_i = _cells_in_box(box_i)
        for prev_box in final_boxes:
            assert cells_i.isdisjoint(_cells_in_box(prev_box)), "shrunk squares must not overlap"
        final_boxes.append(box_i)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each solid odd square to its same-color marker, then shrink it twice at the landing point while leaving a gray center trail.

    Input structure:
    - A black background contains 2 or 3 monochrome solid odd-sided squares.
    - Each square has a unique same-color single-pixel marker aligned with the square center on the same row or column.
    - Markers lie outside their squares, and the slide corridor for every square is unobstructed.

    Rule:
    - For each square, find its same-color aligned marker and infer the cardinal slide direction from the square center toward that marker.
    - Translate the whole square until the marker is exactly one cell beyond the nearest edge of the translated square.
    - Color every center position visited during that slide, from the start center up to but not including the landing center, gray.
    - Remove the marker, then repaint only the square shrunk by two layers at the landing center, changing side length n to n - 4.
    - Process squares in row-major order of their initial centers; if a later gray trail reaches a square that has already landed, the trail does not overwrite that square.

    Key insight:
    - The marker determines the translation, the landing point determines where the fixed two-layer shrink happens, and the gray trail records the center path, so the three visible effects depend on one another in sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    specs = _extract_objects(g)
    _validate_core_scene(specs, g.shape)

    out = np.zeros_like(g)
    ordered = sorted(specs, key=lambda spec: (spec.init_center[0], spec.init_center[1]))
    for spec in ordered:
        for r, c in _track_cells(spec):
            if out[r, c] == BG:
                out[r, c] = TRACK
        _paint_square(out, spec.final_center, spec.shrunk_side, spec.color)
    return out


def _render_input(specs: Iterable[ObjSpec], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        _paint_square(grid, spec.init_center, spec.side, spec.color)
        mr, mc = spec.marker
        grid[mr, mc] = spec.color
    return grid


def _render_expected(specs: list[ObjSpec], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in sorted(specs, key=lambda item: (item.init_center[0], item.init_center[1])):
        for r, c in _track_cells(spec):
            if out[r, c] == BG:
                out[r, c] = TRACK
        _paint_square(out, spec.final_center, spec.shrunk_side, spec.color)
    return out


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in b:
                    return True
    return False


def _has_counterexample_pair(specs: list[ObjSpec]) -> bool:
    for i, left in enumerate(specs):
        for right in specs[i + 1 :]:
            if left.distance == right.distance and left.side != right.side:
                return True
            if left.side == right.side and left.distance != right.distance:
                return True
    return False


def _track_over_earlier_square_count(specs: list[ObjSpec]) -> int:
    ordered = sorted(specs, key=lambda spec: (spec.init_center[0], spec.init_center[1]))
    total = 0
    prior_squares: list[set[tuple[int, int]]] = []
    for spec in ordered:
        track = set(_track_cells(spec))
        total += sum(len(track & square) for square in prior_squares)
        prior_squares.append(_square_cells(spec.final_center, spec.shrunk_side))
    return total


def _has_both_axes(specs: list[ObjSpec]) -> bool:
    axes = {0 if spec.direction[0] else 1 for spec in specs}
    return axes == {0, 1}


def _validate_generator_scene(specs: list[ObjSpec], shape: tuple[int, int], require_overlap: bool) -> bool:
    try:
        _validate_core_scene(specs, shape)
    except AssertionError:
        return False

    actual_cells = [_square_cells(spec.init_center, spec.side) | {spec.marker} for spec in specs]
    for i, cells_i in enumerate(actual_cells):
        for j, cells_j in enumerate(actual_cells):
            if i >= j:
                continue
            if _touches(cells_i, cells_j):
                return False

    expected = _render_expected(specs, shape)
    actual = main(_render_input(specs, shape))
    if not np.array_equal(actual, expected):
        return False

    if not _has_both_axes(specs):
        return False
    if not _has_counterexample_pair(specs):
        return False
    if require_overlap and _track_over_earlier_square_count(specs) == 0:
        return False
    return True


def _sample_overlap_pair(shape: tuple[int, int], colors: list[int]) -> list[ObjSpec] | None:
    h, w = shape
    mode = random.choice(("htrack", "vtrack"))
    if mode == "htrack":
        side_a = random.choice((7, 9))
        side_b = random.choice([side for side in SIDES if side != side_a])
        half_a = side_a // 2
        half_b = side_b // 2
        fh_a = (side_a - 4) // 2
        fh_b = (side_b - 4) // 2
        d_min = max(half_a + half_b + fh_a + 1, 2 * fh_a + fh_b + 2, 6)
        d_max = min(11, h - (2 * half_a + 2), w - (2 * half_b + 4))
        if d_min > d_max:
            return None
        distance = random.randint(d_min, d_max)

        r_final_low = half_a + distance
        r_final_high = h - half_a - 2
        if r_final_low > r_final_high:
            return None
        r_final = random.randint(r_final_low, r_final_high)
        r_track = r_final - fh_a
        if not (half_b <= r_track < h - half_b):
            return None

        c_final_a_low = max(half_a, half_b + distance, half_b + 2)
        c_final_a_high = min(w - half_a - 1, w - half_b - 2, w - half_b - distance - 1)
        if c_final_a_low > c_final_a_high:
            return None
        c_final_a = random.randint(c_final_a_low, c_final_a_high)
        left_a = c_final_a - fh_a
        right_a = c_final_a + fh_a

        sign = random.choice((-1, 1))
        if sign == 1:
            c_final_b_low = max(right_a + fh_b + 1, half_b + distance)
            c_final_b_high = min(w - half_b - 2, left_a - 1 + distance)
        else:
            c_final_b_low = max(half_b + 1, right_a + 1 - distance)
            c_final_b_high = min(left_a - fh_b - 1, w - half_b - distance - 1)
        if c_final_b_low > c_final_b_high:
            return None
        c_final_b = random.randint(c_final_b_low, c_final_b_high)
        c_init_b = c_final_b - sign * distance

        spec_a = ObjSpec(colors[0], side_a, (r_final - distance, c_final_a), DIRS["down"], distance)
        spec_b = ObjSpec(colors[1], side_b, (r_track, c_init_b), (0, sign), distance)
        return [spec_a, spec_b]

    side_a = random.choice((7, 9))
    side_b = random.choice([side for side in SIDES if side != side_a])
    half_a = side_a // 2
    half_b = side_b // 2
    fh_a = (side_a - 4) // 2
    fh_b = (side_b - 4) // 2
    d_min = max(half_a + half_b + fh_a + 1, 2 * fh_a + fh_b + 2, 6)
    d_max = min(11, w - (2 * half_a + 2), h - (2 * half_b + 4))
    if d_min > d_max:
        return None
    distance = random.randint(d_min, d_max)

    c_final_low = half_a + distance
    c_final_high = w - half_a - 2
    if c_final_low > c_final_high:
        return None
    c_final = random.randint(c_final_low, c_final_high)
    c_track = c_final - fh_a
    if not (half_b <= c_track < w - half_b):
        return None

    r_final_a_low = max(half_a, half_b + distance, half_b + 2)
    r_final_a_high = min(h - half_a - 1, h - half_b - 2, h - half_b - distance - 1)
    if r_final_a_low > r_final_a_high:
        return None
    r_final_a = random.randint(r_final_a_low, r_final_a_high)
    top_a = r_final_a - fh_a
    bottom_a = r_final_a + fh_a

    sign = random.choice((-1, 1))
    if sign == 1:
        r_final_b_low = max(bottom_a + fh_b + 1, half_b + distance)
        r_final_b_high = min(h - half_b - 2, top_a - 1 + distance)
    else:
        r_final_b_low = max(half_b + 1, bottom_a + 1 - distance)
        r_final_b_high = min(top_a - fh_b - 1, h - half_b - distance - 1)
    if r_final_b_low > r_final_b_high:
        return None
    r_final_b = random.randint(r_final_b_low, r_final_b_high)
    r_init_b = r_final_b - sign * distance

    spec_a = ObjSpec(colors[0], side_a, (r_final_a, c_final - distance), DIRS["right"], distance)
    spec_b = ObjSpec(colors[1], side_b, (r_init_b, c_track), (sign, 0), distance)
    return [spec_a, spec_b]


def _sample_extra_object(shape: tuple[int, int], specs: list[ObjSpec], color: int, side: int | None) -> ObjSpec | None:
    h, w = shape
    existing_tracks = {cell for spec in specs for cell in _track_cells(spec)}
    existing_finals = [_square_cells(spec.final_center, spec.shrunk_side) for spec in specs]
    for _ in range(EXTRA_OBJECT_ATTEMPTS):
        chosen_side = random.choice(SIDES) if side is None else side
        direction = random.choice(tuple(DIRS.values()))
        half = chosen_side // 2
        max_distance = 8
        if direction == DIRS["up"]:
            d_limit = h - half - 2
            distance = random.randint(2, min(max_distance, d_limit)) if d_limit >= 2 else None
            if distance is None:
                continue
            r = random.randint(distance + half + 1, h - half - 1)
            c = random.randint(half, w - half - 1)
        elif direction == DIRS["down"]:
            d_limit = h - half - 2
            distance = random.randint(2, min(max_distance, d_limit)) if d_limit >= 2 else None
            if distance is None:
                continue
            r = random.randint(half, h - distance - half - 2)
            c = random.randint(half, w - half - 1)
        elif direction == DIRS["left"]:
            d_limit = w - half - 2
            distance = random.randint(2, min(max_distance, d_limit)) if d_limit >= 2 else None
            if distance is None:
                continue
            r = random.randint(half, h - half - 1)
            c = random.randint(distance + half + 1, w - half - 1)
        else:
            d_limit = w - half - 2
            distance = random.randint(2, min(max_distance, d_limit)) if d_limit >= 2 else None
            if distance is None:
                continue
            r = random.randint(half, h - half - 1)
            c = random.randint(half, w - distance - half - 2)

        candidate = ObjSpec(color, chosen_side, (r, c), direction, distance)
        if not _validate_generator_scene(specs + [candidate], shape, require_overlap=True):
            continue
        candidate_actual = _square_cells(candidate.init_center, candidate.side) | {candidate.marker}
        candidate_final = _square_cells(candidate.final_center, candidate.shrunk_side)
        if any(candidate_final & final for final in existing_finals):
            continue
        if candidate_final & existing_tracks:
            continue
        if set(_track_cells(candidate)) & set().union(*existing_finals):
            continue
        if any(_touches(candidate_actual, _square_cells(spec.init_center, spec.side) | {spec.marker}) for spec in specs):
            continue
        return candidate
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        shape = (random.randint(21, 29), random.randint(21, 29))
        count = 3 if random.random() < 0.62 else 2
        colors = random.sample(list(INPUT_COLORS), count)
        pair = _sample_overlap_pair(shape, colors[:2])
        if pair is None:
            continue

        specs = pair
        if count == 3:
            used_sides = {spec.side for spec in specs}
            forced_side = 9 if 9 not in used_sides else None
            extra = _sample_extra_object(shape, specs, colors[2], forced_side)
            if extra is None:
                continue
            specs = specs + [extra]

        if not _validate_generator_scene(specs, shape, require_overlap=True):
            continue
        return _render_input(specs, shape)
    raise RuntimeError("failed to generate a valid sliding-square ARC input")
