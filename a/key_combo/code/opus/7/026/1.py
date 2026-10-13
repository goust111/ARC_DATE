from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
MARKER_PALETTE = (1, 2, 3, 4)
DIRECTIONS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Marker:
    row: int
    col: int
    color: int


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    out.sort(key=lambda coords: (_bbox(coords)[0], _bbox(coords)[1]))
    return out


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _dist_sq_marker_to_object(marker: Marker, coords: np.ndarray) -> int:
    diffs = coords - np.array([[marker.row, marker.col]], dtype=int)
    return int(np.min(np.sum(diffs * diffs, axis=1)))


def _direction_from_marker(marker: Marker, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    above = marker.row < r0 and c0 <= marker.col <= c1
    below = marker.row > r1 and c0 <= marker.col <= c1
    left = marker.col < c0 and r0 <= marker.row <= r1
    right = marker.col > c1 and r0 <= marker.row <= r1
    assert sum((above, below, left, right)) == 1, "marker must lie on exactly one bbox side"

    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    if above:
        assert marker.row < center_r
        return "up"
    if below:
        assert marker.row > center_r
        return "down"
    if left:
        assert marker.col < center_c
        return "left"
    assert marker.col > center_c
    return "right"


def _erase_and_mirror(
    coords: np.ndarray,
    direction: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    cells = {tuple(map(int, rc)) for rc in coords}
    if direction == "up":
        erased = {(r, c) for r, c in cells if r == r0}
        residual = cells - erased
        mirror = {(2 * r0 - r, c) for r, c in residual}
    elif direction == "down":
        erased = {(r, c) for r, c in cells if r == r1}
        residual = cells - erased
        mirror = {(2 * r1 - r, c) for r, c in residual}
    elif direction == "left":
        erased = {(r, c) for r, c in cells if c == c0}
        residual = cells - erased
        mirror = {(r, 2 * c0 - c) for r, c in residual}
    else:
        erased = {(r, c) for r, c in cells if c == c1}
        residual = cells - erased
        mirror = {(r, 2 * c1 - c) for r, c in residual}
    assert erased and residual
    return erased, residual, mirror


def _pair_objects_to_markers(
    objects: list[np.ndarray],
    markers: list[Marker],
) -> list[int]:
    n = len(objects)
    assert n == len(markers) >= 1
    scores = np.zeros((n, n), dtype=int)
    for i, coords in enumerate(objects):
        for j, marker in enumerate(markers):
            scores[i, j] = _dist_sq_marker_to_object(marker, coords)

    object_choice: list[int] = []
    for i in range(n):
        row = scores[i]
        best = int(row.min())
        candidates = np.flatnonzero(row == best)
        assert len(candidates) == 1, "each grey object must have a unique closest marker"
        object_choice.append(int(candidates[0]))

    marker_choice: list[int] = []
    for j in range(n):
        col = scores[:, j]
        best = int(col.min())
        candidates = np.flatnonzero(col == best)
        assert len(candidates) == 1, "each marker must have a unique closest grey object"
        marker_choice.append(int(candidates[0]))

    assert len(set(object_choice)) == n, "pairing must be one-to-one"
    for obj_idx, marker_idx in enumerate(object_choice):
        assert marker_choice[marker_idx] == obj_idx, "pairing must be mutual nearest"
    return object_choice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the marker-facing edge of each grey object and mirror the remainder outward in the marker color.

    Input structure:
    - The grid has a black background, 3-4 grey 4-connected objects, and the same number of isolated colored marker pixels.
    - Each marker lies strictly above, below, left, or right of exactly one grey object's bounding box.
    - Grey objects are separated so their local edge erasure and outward mirror can be resolved independently.

    Rule:
    - For each grey object, find the unique marker with the smallest Euclidean distance to any pixel of that object.
    - Use that marker's side relative to the object's bounding-box center to choose which boundary row or column to erase.
    - Keep the remaining grey pixels in place, erase the paired marker, and mirror the remaining grey shape outward across the erased edge; paint the mirrored copy in the marker color.
    - If mirrored copies overlap, later objects in top-to-bottom then left-to-right bbox order overwrite earlier mirrored copies; mirrored copies do not replace surviving grey pixels.

    Key insight:
    - The nearest marker encodes three things at once: which object it controls, which local edge gets erased, and which color the mirrored copy uses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _most_common_color(g) == BG, "expected black background"

    objects = _components(g == GREY)
    assert 3 <= len(objects) <= 4

    marker_components = _components((g != BG) & (g != GREY))
    assert len(marker_components) == len(objects), "expected one isolated marker per object"
    markers: list[Marker] = []
    seen_colors: set[int] = set()
    for comp in marker_components:
        assert comp.shape[0] == 1, "markers must be single pixels"
        r, c = map(int, comp[0])
        color = int(g[r, c])
        assert color in MARKER_PALETTE
        assert color not in seen_colors, "marker colors must be unique"
        seen_colors.add(color)
        markers.append(Marker(r, c, color))

    marker_for_object = _pair_objects_to_markers(objects, markers)
    out = g.copy()
    residual_pixels: set[tuple[int, int]] = set()
    mirror_jobs: list[tuple[tuple[int, int], int, set[tuple[int, int]]]] = []

    for obj_idx, coords in enumerate(objects):
        marker = markers[marker_for_object[obj_idx]]
        bbox = _bbox(coords)
        direction = _direction_from_marker(marker, bbox)
        erased, residual, mirror = _erase_and_mirror(coords, direction)

        out[marker.row, marker.col] = BG
        for r, c in erased:
            out[r, c] = BG
        residual_pixels |= residual
        mirror_jobs.append(((bbox[0], bbox[1]), marker.color, mirror))

    for _anchor, color, mirror in mirror_jobs:
        for r, c in mirror:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                if (r, c) in residual_pixels:
                    continue
                out[r, c] = int(color)

    return out


def _normalize_signature(items: Iterable[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    items = list(items)
    min_r = min(r for r, _c, _tag in items)
    min_c = min(c for _r, c, _tag in items)
    return tuple(sorted((r - min_r, c - min_c, tag) for r, c, tag in items))


def _local_direction_signature(cells: set[tuple[int, int]], h: int, w: int, direction: str) -> tuple[tuple[int, int, int], ...] | None:
    coords = np.array(sorted(cells), dtype=int)
    erased, residual, mirror = _erase_and_mirror(coords, direction)
    if len(residual) < 2:
        return None
    return _normalize_signature(
        [(*cell, 1) for cell in residual] + [(*cell, 2) for cell in mirror]
    )


def _boundary_gap_indices(cells: set[tuple[int, int]], h: int, w: int, direction: str) -> list[int]:
    if direction == "up":
        return [c for c in range(w) if (0, c) not in cells]
    if direction == "down":
        return [c for c in range(w) if (h - 1, c) not in cells]
    if direction == "left":
        return [r for r in range(h) if (r, 0) not in cells]
    return [r for r in range(h) if (r, w - 1) not in cells]


def _has_hole(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    seen = np.zeros_like(mask, dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    for r in range(h):
        for c in range(w):
            if not mask[r, c] and not seen[r, c]:
                return True
    return False


def _manhattan_path(
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    r, c = start
    cells = [(r, c)]
    while (r, c) != end:
        moves: list[tuple[int, int]] = []
        if r < end[0]:
            moves.append((r + 1, c))
        elif r > end[0]:
            moves.append((r - 1, c))
        if c < end[1]:
            moves.append((r, c + 1))
        elif c > end[1]:
            moves.append((r, c - 1))
        nr, nc = moves[int(rng.integers(len(moves)))]
        r, c = nr, nc
        cells.append((r, c))
    return cells


def _random_shape(
    h: int,
    w: int,
    direction: str,
    require_gap_for_marker: bool,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], dict[str, tuple[tuple[int, int, int], ...]]]:
    for _ in range(300):
        hub = (int(rng.integers(h)), int(rng.integers(w)))
        anchors = {
            "up": (0, int(rng.integers(w))),
            "down": (h - 1, int(rng.integers(w))),
            "left": (int(rng.integers(h)), 0),
            "right": (int(rng.integers(h)), w - 1),
        }
        cells = {hub}
        for endpoint in anchors.values():
            cells.update(_manhattan_path(endpoint, hub, rng))
        if not (4 <= len(cells) <= 8):
            continue
        target = int(rng.integers(len(cells), 9))
        while len(cells) < target:
            frontier = {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
            }
            if not frontier:
                break
            cells.add(tuple(frontier)[int(rng.integers(len(frontier)))])
        if len(cells) == h * w:
            continue
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if rows != set(range(h)) or cols != set(range(w)):
            continue
        if _has_hole(cells, h, w):
            continue
        gap_indices = _boundary_gap_indices(cells, h, w, direction)
        if require_gap_for_marker and not gap_indices:
            continue
        signatures = {side: _local_direction_signature(cells, h, w, side) for side in DIRECTIONS}
        if any(sig is None for sig in signatures.values()):
            continue
        if len(set(signatures.values())) != 4:
            continue
        return cells, signatures  # type: ignore[return-value]
    raise RuntimeError("failed to build a non-degenerate shape")


def _slot_rectangles(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    slots = [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 1, mid_c - 2),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]
    assert all(r0 <= r1 and c0 <= c1 for r0, r1, c0, c1 in slots)
    return slots


def _place_object_in_slot(
    slot: tuple[int, int, int, int],
    local_cells: set[tuple[int, int]],
    marker_color: int,
    direction: str,
    force_gap_marker: bool,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], Marker]:
    slot_r0, slot_r1, slot_c0, slot_c1 = slot
    h = max(r for r, _c in local_cells) + 1
    w = max(c for _r, c in local_cells) + 1
    top_space = h - 1 if direction == "up" else 0
    bottom_space = h - 1 if direction == "down" else 0
    left_space = w - 1 if direction == "left" else 0
    right_space = w - 1 if direction == "right" else 0

    row_min = slot_r0 + top_space
    row_max = slot_r1 - (h - 1) - bottom_space
    col_min = slot_c0 + left_space
    col_max = slot_c1 - (w - 1) - right_space
    assert row_min <= row_max and col_min <= col_max

    base_r = int(rng.integers(row_min, row_max + 1))
    base_c = int(rng.integers(col_min, col_max + 1))
    abs_cells = [(base_r + r, base_c + c) for r, c in sorted(local_cells)]

    if direction in ("up", "down"):
        gap = int(rng.integers(1, min(2, h - 1) + 1))
        gap_indices = _boundary_gap_indices(local_cells, h, w, direction)
        cols = gap_indices if force_gap_marker else list(range(w))
        marker_col = base_c + int(cols[int(rng.integers(len(cols)))])
        marker_row = base_r - gap if direction == "up" else base_r + h - 1 + gap
    else:
        gap = int(rng.integers(1, min(2, w - 1) + 1))
        gap_indices = _boundary_gap_indices(local_cells, h, w, direction)
        rows = gap_indices if force_gap_marker else list(range(h))
        marker_row = base_r + int(rows[int(rng.integers(len(rows)))])
        marker_col = base_c - gap if direction == "left" else base_c + w - 1 + gap

    return abs_cells, Marker(marker_row, marker_col, marker_color)


def _min_chebyshev_distance(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> int:
    best = math.inf
    for ar, ac in a:
        for br, bc in b:
            best = min(best, max(abs(ar - br), abs(ac - bc)))
    return int(best)


def _marker_uses_boundary_gap(coords: np.ndarray, marker: Marker, direction: str) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    cells = {tuple(map(int, rc)) for rc in coords}
    if direction == "up":
        return (r0, marker.col) not in cells
    if direction == "down":
        return (r1, marker.col) not in cells
    if direction == "left":
        return (marker.row, c0) not in cells
    return (marker.row, c1) not in cells


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert _most_common_color(grid) == BG

    objects = _components(grid == GREY)
    markers_raw = _components((grid != BG) & (grid != GREY))
    assert 3 <= len(objects) <= 4
    assert len(markers_raw) == len(objects)

    markers: list[Marker] = []
    colors = set()
    for comp in markers_raw:
        assert comp.shape[0] == 1
        r, c = map(int, comp[0])
        color = int(grid[r, c])
        assert color in MARKER_PALETTE and color not in colors
        colors.add(color)
        markers.append(Marker(r, c, color))

    for coords in objects:
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
        assert 4 <= coords.shape[0] <= 8
        local = {(int(r - r0), int(c - c0)) for r, c in coords}
        assert not _has_hole(local, r1 - r0 + 1, c1 - c0 + 1)
        signatures = {_local_direction_signature(local, r1 - r0 + 1, c1 - c0 + 1, d) for d in DIRECTIONS}
        assert len(signatures) == 4

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            assert _min_chebyshev_distance(map(tuple, objects[i]), map(tuple, objects[j])) >= 3

    pairing = _pair_objects_to_markers(objects, markers)
    directions = []
    gap_based_count = 0
    for obj_idx, coords in enumerate(objects):
        marker = markers[pairing[obj_idx]]
        direction = _direction_from_marker(marker, _bbox(coords))
        directions.append(direction)
        erased, residual, mirror = _erase_and_mirror(coords, direction)
        assert erased and len(residual) >= 2
        assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in mirror)
        gap_based_count += int(_marker_uses_boundary_gap(coords, marker, direction))
    assert len(set(directions)) == len(objects)
    assert gap_based_count >= 1

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = int(rng.integers(18, 25))
        w = int(rng.integers(18, 25))
        grid = np.zeros((h, w), dtype=int)
        slots = _slot_rectangles(h, w)

        object_count = int(rng.integers(3, 5))
        chosen_slots = list(rng.choice(len(slots), size=object_count, replace=False))
        rng.shuffle(chosen_slots)
        directions = list(rng.choice(np.array(DIRECTIONS), size=object_count, replace=False))
        rng.shuffle(directions)
        marker_colors = list(rng.choice(np.array(MARKER_PALETTE), size=object_count, replace=False))
        gap_object_idx = int(rng.integers(object_count))

        for idx, slot_idx in enumerate(chosen_slots):
            direction = directions[idx]
            slot = slots[int(slot_idx)]
            slot_h = slot[1] - slot[0] + 1
            slot_w = slot[3] - slot[2] + 1
            feasible_dims = []
            for obj_h in (3, 4):
                for obj_w in (3, 4):
                    if obj_h > slot_h or obj_w > slot_w:
                        continue
                    if direction in ("up", "down") and 2 * obj_h - 1 > slot_h:
                        continue
                    if direction in ("left", "right") and 2 * obj_w - 1 > slot_w:
                        continue
                    feasible_dims.append((obj_h, obj_w))
            if not feasible_dims:
                break
            obj_h, obj_w = feasible_dims[int(rng.integers(len(feasible_dims)))]
            require_gap = True
            local_cells, _signatures = _random_shape(
                obj_h,
                obj_w,
                direction,
                require_gap_for_marker=require_gap,
                rng=rng,
            )
            abs_cells, marker = _place_object_in_slot(
                slot,
                local_cells,
                marker_colors[idx],
                direction,
                force_gap_marker=(idx == gap_object_idx),
                rng=rng,
            )
            for r, c in abs_cells:
                grid[r, c] = GREY
            grid[marker.row, marker.col] = marker.color
        else:
            try:
                _validate_instance(grid)
            except AssertionError:
                continue
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
