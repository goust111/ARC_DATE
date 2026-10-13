from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 7]


class _Object:
    def __init__(self, color: int, coords: np.ndarray):
        self.color = int(color)
        self.coords = np.asarray(coords, dtype=int)
        r0 = int(self.coords[:, 0].min())
        c0 = int(self.coords[:, 1].min())
        r1 = int(self.coords[:, 0].max())
        c1 = int(self.coords[:, 1].max())
        self.bbox: BBox = (r0, c0, r1, c1)
        self.height = r1 - r0 + 1
        self.width = c1 - c0 + 1
        self.patch = np.full((self.height, self.width), BG, dtype=int)
        rel = self.coords - np.array([r0, c0], dtype=int)
        self.patch[rel[:, 0], rel[:, 1]] = self.color
        self.closed = _has_enclosed_hole(self.patch != BG)
        self.area = int(np.count_nonzero(self.patch != BG))


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[_Object]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[_Object] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            objects.append(_Object(color, np.array(coords, dtype=int)))
    return objects


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    stack: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~mask) & (~exterior)))


def _can_place(canvas: np.ndarray, patch: np.ndarray, row: int, col: int, bg: int) -> bool:
    h, w = patch.shape
    region = canvas[row : row + h, col : col + w]
    mask = patch != bg
    return bool(np.all(region[mask] == bg))


def _paste(canvas: np.ndarray, patch: np.ndarray, row: int, col: int, bg: int) -> None:
    h, w = patch.shape
    region = canvas[row : row + h, col : col + w]
    mask = patch != bg
    region[mask] = patch[mask]


def _tail_recolored_patch(patch: np.ndarray, bg: int, upward: bool, distance: int) -> tuple[np.ndarray, int]:
    out = patch.copy()
    mask = patch != bg
    remaining = int(distance)
    rows = range(patch.shape[0] - 1, -1, -1) if upward else range(patch.shape[0])
    painted = 0
    for r in rows:
        for c in np.where(mask[r])[0]:
            if remaining <= 0:
                return out, painted
            out[r, c] = YELLOW
            remaining -= 1
            painted += 1
    return out, painted


def _transform(grid: np.ndarray, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    assert bg == BG

    objects = _find_components(g, bg)
    assert 4 <= len(objects) <= 6
    assert all(4 <= obj.area <= 8 for obj in objects)

    occupied = np.full((h, w), bg, dtype=int)
    placements: list[dict] = []

    closed_objects = sorted([obj for obj in objects if obj.closed], key=lambda obj: (obj.bbox[0], obj.bbox[1]))
    open_objects = sorted([obj for obj in objects if not obj.closed], key=lambda obj: (-obj.bbox[2], obj.bbox[1]))
    assert closed_objects and open_objects

    def settle(obj: _Object, upward: bool) -> None:
        c0 = obj.bbox[1]
        if upward:
            rows = range(0, h - obj.height + 1)
        else:
            rows = range(h - obj.height, -1, -1)
        final_row = None
        for row in rows:
            if _can_place(occupied, obj.patch, row, c0, bg):
                final_row = row
                break
        assert final_row is not None
        _paste(occupied, obj.patch, final_row, c0, bg)
        distance = abs(final_row - obj.bbox[0])
        recolored_patch, yellow_count = _tail_recolored_patch(obj.patch, bg, upward, distance)
        placements.append(
            {
                "obj": obj,
                "upward": upward,
                "final_row": int(final_row),
                "final_col": int(c0),
                "distance": int(distance),
                "yellow_count": int(yellow_count),
                "edge_limited": int(final_row if upward else (h - obj.height - final_row)),
            }
        )
        occupied[final_row : final_row + obj.height, c0 : c0 + obj.width][recolored_patch != bg] = recolored_patch[
            recolored_patch != bg
        ]

    for obj in closed_objects:
        settle(obj, upward=True)
    for obj in open_objects:
        settle(obj, upward=False)

    out = occupied
    if return_details:
        return out, placements
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide hole-objects upward and hole-free objects downward, then recolor a yellow tail whose length equals each object's slide distance.

    Input structure
    - The grid is black background with 4-6 monochromatic 4-connected foreground objects in distinct colors.
    - Some objects are closed hollow loops whose bounding boxes contain an enclosed black hole; the rest are open hole-free shapes.
    - Objects keep their original columns and are separated in the input by at least one black cell.

    Rule
    - Classify each object by topology inside its own bounding box: hole-objects move upward, hole-free objects move downward.
    - Within each direction, process nearer-to-target objects first so vertically overlapping objects stack on collision without crossing.
    - Let D be the absolute change in the object's top bbox row after sliding; from the tail end opposite the motion, recolor D object pixels yellow, sweeping row by row inward.
    - Output the settled objects on the same black canvas.

    Key insight
    - The yellow repaint count is not marked directly: it is encoded by the global slide distance, which itself depends on hole detection and collision-limited stacking.
    """
    return _transform(grid, return_details=False)


def _ring_patch(color: int) -> np.ndarray:
    patch = np.array(
        [
            [color, color, color],
            [color, BG, color],
            [color, color, color],
        ],
        dtype=int,
    )
    return patch


def _line_patch(color: int, length: int) -> np.ndarray:
    patch = np.full((length, 1), BG, dtype=int)
    patch[:, 0] = color
    return patch


def _patch_cells(patch: np.ndarray, row: int, col: int) -> set[Coord]:
    coords: set[Coord] = set()
    for rr, cc in np.argwhere(patch != BG):
        coords.add((int(row + rr), int(col + cc)))
    return coords


def _has_gap_from(existing: list[set[Coord]], new_cells: set[Coord]) -> bool:
    for cells in existing:
        for r, c in new_cells:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in cells:
                        return False
    return True


def _column_overlap_exists(intervals: list[tuple[int, int]]) -> bool:
    for i, (a0, a1) in enumerate(intervals):
        for j, (b0, b1) in enumerate(intervals):
            if i == j:
                continue
            if max(a0, b0) <= min(a1, b1):
                return True
    return False


def _validate_input_layout(grid: np.ndarray) -> None:
    objects = _find_components(grid, BG)
    assert 4 <= len(objects) <= 6
    all_cells: list[set[Coord]] = []
    intervals: list[tuple[int, int]] = []
    colors = set()
    for obj in objects:
        assert obj.color in PALETTE
        assert obj.color not in colors
        colors.add(obj.color)
        cells = {(int(r), int(c)) for r, c in obj.coords}
        assert _has_gap_from(all_cells, cells)
        all_cells.append(cells)
        intervals.append((obj.bbox[1], obj.bbox[3]))
        assert 4 <= obj.area <= 8
    assert _column_overlap_exists(intervals)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(17, 19)
    w = random.randint(13, 17)
    n_open = random.choice([2, 3])
    total = 2 + n_open
    colors = random.sample(PALETTE, total)

    closed_col = random.randint(1, 3)
    open_col = random.randint(max(closed_col + 5, w - 3), w - 2)

    open_lengths: list[int]
    if n_open == 2:
        open_lengths = [random.randint(4, 5), random.randint(4, 5)]
    else:
        candidates = [(a, b, c) for a in (4, 5) for b in (4, 5) for c in (4, 5) if a + b + c <= 13]
        open_lengths = list(random.choice(candidates))

    objects: list[dict] = []
    for idx in range(2):
        objects.append(
            {
                "kind": "closed",
                "color": colors[idx],
                "patch": _ring_patch(colors[idx]),
                "final_row": 3 * idx,
                "col": closed_col,
            }
        )

    cursor = h
    for idx, length in enumerate(open_lengths):
        cursor -= length
        objects.append(
            {
                "kind": "open",
                "color": colors[2 + idx],
                "patch": _line_patch(colors[2 + idx], length),
                "final_row": cursor,
                "col": open_col,
            }
        )

    if cursor <= 7:
        return None

    closed_d1 = random.choice([3, 5])
    closed_d2 = 8
    open_dists: list[int] = []
    current = random.choice([2, 3])
    for idx in range(n_open):
        if idx > 0:
            current += random.choice([1, 2])
        open_dists.append(current)

    input_rows: list[int] = []
    for idx, spec in enumerate(objects):
        patch = spec["patch"]
        final_row = int(spec["final_row"])
        if spec["kind"] == "closed":
            distance = closed_d1 if idx == 0 else closed_d2
            row = final_row + distance
        else:
            distance = open_dists[idx - 2]
            row = final_row - distance
        if row < 0 or row + patch.shape[0] > h:
            return None
        spec["distance"] = int(distance)
        spec["row"] = int(row)
        input_rows.append(int(row))

    grid = np.full((h, w), BG, dtype=int)
    occupied_sets: list[set[Coord]] = []
    for spec in objects:
        cells = _patch_cells(spec["patch"], spec["row"], spec["col"])
        if not _has_gap_from(occupied_sets, cells):
            return None
        occupied_sets.append(cells)
        _paste(grid, spec["patch"], spec["row"], spec["col"], BG)

    try:
        _validate_input_layout(grid)
        out, details = _transform(grid, return_details=True)
    except AssertionError:
        return None

    if not np.array_equal(main(grid), out):
        return None

    detail_by_color = {int(item["obj"].color): item for item in details}
    for spec in objects:
        item = detail_by_color[spec["color"]]
        if int(item["distance"]) != int(spec["distance"]):
            return None
        if int(item["final_row"]) != int(spec["final_row"]):
            return None

    closed_details = [item for item in details if item["obj"].closed]
    open_details = [item for item in details if not item["obj"].closed]
    if len(closed_details) != 2 or len(open_details) != n_open:
        return None

    if min(item["distance"] for item in details) <= 0:
        return None
    if not any(item["yellow_count"] == item["obj"].area for item in details):
        return None
    if not any(0 < item["yellow_count"] < item["obj"].area for item in details):
        return None

    if not any(item["edge_limited"] > 0 for item in details):
        return None

    closed_bottom = max(item["final_row"] + item["obj"].height - 1 for item in closed_details)
    open_top = min(item["final_row"] for item in open_details)
    if open_top - closed_bottom < 2:
        return None

    yellow_total = int(np.count_nonzero(out == YELLOW))
    if yellow_total <= 8:
        return None
    if len(set(int(item["distance"]) for item in details)) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
