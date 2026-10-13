from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
TEAL = 8
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": cells,
                    "coord_set": set(cells),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _is_complete_square(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 2:
        return False
    cells = comp["coord_set"]
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    return True


def _corner_markers(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _line_cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(r, c) for c in range(c0 + 1, c1)]
    c = a[1]
    r0, r1 = sorted((a[0], b[0]))
    return [(r, c) for r in range(r0 + 1, r1)]


def _best_marker_pair(
    markers_a: list[tuple[int, int]], markers_b: list[tuple[int, int]]
) -> tuple[tuple[int, tuple[int, int], tuple[int, int]], list[tuple[int, tuple[int, int], tuple[int, int]]]]:
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for a in sorted(markers_a):
        for b in sorted(markers_b):
            if a[0] == b[0] or a[1] == b[1]:
                dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
                candidates.append((dist, a, b))
    assert candidates, "same-color squares must expose an aligned marker pair"
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1], item[2][0], item[2][1]))
    return candidates[0], candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark complete square objects and connect each same-color square pair through the closest aligned outside markers.

    Input structure:
    - A dominant background contains several monochrome 4-connected foreground objects in 2-3 colors.
    - Some objects are complete squares: their tight bounding box is square, side length is at least 2, and every bbox-edge cell belongs to the object.
    - Other same-color objects may be bars or irregular fragments and act only as distractors.

    Rule:
    - Preserve all input objects and first identify only the complete square objects.
    - For each complete square, paint red on the orthogonally outside corner-adjacent cells that are in-bounds and currently background.
    - For each same-color pair of complete squares, examine the red markers created for the two squares and choose the aligned pair sharing one row or one column with minimum Manhattan distance.
    - Draw a teal straight segment strictly between those two chosen markers, leaving the chosen red endpoints themselves red while overwriting any intervening pixels.

    Key insight:
    - The red pixels are not just annotations: after square detection they become the candidate bridge endpoints, so only true complete squares contribute geometry for the final connection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _components_by_color(g, bg)
    squares = [comp for comp in comps if _is_complete_square(comp)]

    out = g.copy()
    markers_by_square: dict[int, list[tuple[int, int]]] = {}
    for idx, comp in enumerate(squares):
        markers: list[tuple[int, int]] = []
        for r, c in _corner_markers(comp["bbox"]):
            if _in_bounds(g.shape, r, c) and int(g[r, c]) == bg:
                markers.append((r, c))
                out[r, c] = RED
        markers_by_square[idx] = markers

    squares_by_color: dict[int, list[int]] = {}
    for idx, comp in enumerate(squares):
        squares_by_color.setdefault(int(comp["color"]), []).append(idx)

    for color, idxs in squares_by_color.items():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs, key=lambda i: squares[i]["bbox"])
        for i in range(0, len(idxs) - 1, 2):
            a_idx = idxs[i]
            b_idx = idxs[i + 1]
            best, _ = _best_marker_pair(markers_by_square[a_idx], markers_by_square[b_idx])
            _, a, b = best
            for r, c in _line_cells_between(a, b):
                out[r, c] = TEAL

    return out


def _draw_square(grid: np.ndarray, r0: int, c0: int, size: int, color: int, filled: bool) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if filled or r in (r0, r1) or c in (c0, c1):
                grid[r, c] = color
                cells.append((r, c))
    return cells


def _rotate_cells(cells: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in cells]
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return sorted((r - min_r, c - min_c) for r, c in pts)


DISTRACTOR_SHAPES = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (1, 1)],
]


def _shape_bbox(cells: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _halo_cells(shape: tuple[int, int], cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    halo.add((nr, nc))
    return halo


def _square_pair_spec(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    color: int,
    orientation: str,
) -> dict | None:
    h, w = shape
    for _ in range(120):
        sizes = random.sample([2, 3, 4, 5], 2)
        size_a, size_b = sizes
        gap = random.randint(2, 5)
        share = random.choice(["near", "far"])
        filled_a = size_a == 2 or random.random() < 0.45
        filled_b = size_b == 2 or random.random() < 0.45

        if orientation == "H":
            total_w = size_a + gap + size_b
            if total_w > w - 2 or max(size_a, size_b) > h - 2:
                continue
            c0 = random.randint(1, w - total_w - 1)
            c1 = c0 + size_a + gap
            if share == "near":
                r0a = r0b = random.randint(1, h - max(size_a, size_b) - 1)
            else:
                bottom = random.randint(max(size_a, size_b), h - 2)
                r0a = bottom - size_a + 1
                r0b = bottom - size_b + 1
        else:
            total_h = size_a + gap + size_b
            if total_h > h - 2 or max(size_a, size_b) > w - 2:
                continue
            r0 = random.randint(1, h - total_h - 1)
            r0a = r0
            r0b = r0 + size_a + gap
            if share == "near":
                c0 = c1 = random.randint(1, w - max(size_a, size_b) - 1)
            else:
                right = random.randint(max(size_a, size_b), w - 2)
                c0 = right - size_a + 1
                c1 = right - size_b + 1

        bbox_a = (r0a, c0, r0a + size_a - 1, c0 + size_a - 1)
        bbox_b = (r0b, c1, r0b + size_b - 1, c1 + size_b - 1)
        cells_a = [(r, c) for r in range(bbox_a[0], bbox_a[2] + 1) for c in range(bbox_a[1], bbox_a[3] + 1)
                   if filled_a or r in (bbox_a[0], bbox_a[2]) or c in (bbox_a[1], bbox_a[3])]
        cells_b = [(r, c) for r in range(bbox_b[0], bbox_b[2] + 1) for c in range(bbox_b[1], bbox_b[3] + 1)
                   if filled_b or r in (bbox_b[0], bbox_b[2]) or c in (bbox_b[1], bbox_b[3])]
        markers_a = _corner_markers(bbox_a)
        markers_b = _corner_markers(bbox_b)
        if not all(_in_bounds(shape, r, c) for r, c in markers_a + markers_b):
            continue

        obj_cells = cells_a + cells_b
        obj_set = set(obj_cells)
        if len(obj_set) != len(obj_cells):
            continue
        if obj_set & occupied:
            continue
        if obj_set & blocked:
            continue

        marker_set = set(markers_a + markers_b)
        if marker_set & occupied:
            continue
        if marker_set & blocked:
            continue
        if marker_set & obj_set:
            continue

        best, candidates = _best_marker_pair(markers_a, markers_b)
        if len(candidates) < 2:
            continue
        min_dist = candidates[0][0]
        if sum(1 for dist, _, _ in candidates if dist == min_dist) != 1:
            continue
        line_cells = _line_cells_between(best[1], best[2])
        if not line_cells:
            continue
        line_set = set(line_cells)
        if line_set & occupied:
            continue
        if line_set & blocked:
            continue

        return {
            "color": color,
            "cells_a": cells_a,
            "cells_b": cells_b,
            "bbox_a": bbox_a,
            "bbox_b": bbox_b,
            "filled_a": filled_a,
            "filled_b": filled_b,
            "markers": marker_set,
            "line": line_set,
            "halo": _halo_cells(shape, obj_cells),
        }
    return None


def _place_distractor(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    color: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    for _ in range(120):
        base = random.choice(DISTRACTOR_SHAPES)
        cells = _rotate_cells(base, random.randrange(4))
        if random.random() < 0.35 and len(cells) <= 4:
            if all(r == 0 for r, _ in cells):
                extra = (0, max(c for _, c in cells) + 1)
                cells = sorted(cells + [extra])
            elif all(c == 0 for _, c in cells):
                extra = (max(r for r, _ in cells) + 1, 0)
                cells = sorted(cells + [extra])
        ph, pw = _shape_bbox(cells)
        if ph >= h - 1 or pw >= w - 1:
            continue
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        placed = [(r0 + r, c0 + c) for r, c in cells]
        placed_set = set(placed)
        if placed_set & occupied:
            continue
        if placed_set & blocked:
            continue
        halo = _halo_cells(shape, placed)
        if halo & occupied:
            continue
        return placed
    return None


def _validate_generated_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    bg = _get_bg(grid)
    comps = _components_by_color(grid, bg)
    if not (5 <= len(comps) <= 7):
        return False
    colors = sorted({int(comp["color"]) for comp in comps})
    if not (2 <= len(colors) <= 3):
        return False

    squares = [comp for comp in comps if _is_complete_square(comp)]
    by_color: dict[int, list[dict]] = {}
    for comp in squares:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    if not by_color:
        return False
    for color, items in by_color.items():
        if len(items) != 2:
            return False
        marker_lists = []
        for comp in items:
            markers = []
            for r, c in _corner_markers(comp["bbox"]):
                if not _in_bounds(grid.shape, r, c):
                    return False
                if int(grid[r, c]) != bg:
                    return False
                markers.append((r, c))
            marker_lists.append(markers)
        best, candidates = _best_marker_pair(marker_lists[0], marker_lists[1])
        if len(candidates) < 2:
            return False
        if sum(1 for dist, _, _ in candidates if dist == best[0]) != 1:
            return False
        if not _line_cells_between(best[1], best[2]):
            return False

    out = main(grid)
    if not np.any(out == RED):
        return False
    if not np.any(out == TEAL):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    if random.random() < 0.55:
        h = random.randint(15, 18)
        w = random.randint(15, 19)
        pair_count = 1
    else:
        h = random.randint(17, 20)
        w = random.randint(17, 20)
        pair_count = 2

    grid = np.full((h, w), BLACK, dtype=int)
    pair_colors = random.sample(INPUT_COLORS, pair_count)

    use_third_color = pair_count == 1 or random.random() < 0.45
    extra_colors = [c for c in INPUT_COLORS if c not in pair_colors]
    distractor_palette = pair_colors[:]
    if use_third_color:
        distractor_palette.append(random.choice(extra_colors))

    total_objects = random.randint(max(5, pair_count * 2 + 1), min(7, pair_count * 2 + 4))
    distractor_count = total_objects - pair_count * 2

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    orientations = ["H", "V"] if pair_count == 2 and random.random() < 0.75 else []
    while len(orientations) < pair_count:
        orientations.append(random.choice(["H", "V"]))
    random.shuffle(orientations)

    pair_specs = []
    for color, orientation in zip(pair_colors, orientations):
        spec = _square_pair_spec((h, w), occupied, blocked, color, orientation)
        if spec is None:
            return None
        for cells_key, bbox_key, filled_key in (
            ("cells_a", "bbox_a", "filled_a"),
            ("cells_b", "bbox_b", "filled_b"),
        ):
            r0, c0, r1, c1 = spec[bbox_key]
            _draw_square(grid, r0, c0, r1 - r0 + 1, color, spec[filled_key])
            occupied.update(spec[cells_key])
        blocked.update(spec["markers"])
        blocked.update(spec["line"])
        blocked.update(spec["halo"])
        pair_specs.append(spec)

    distractor_colors = [random.choice(distractor_palette) for _ in range(distractor_count)]
    if distractor_count and all(color not in pair_colors for color in distractor_colors):
        distractor_colors[0] = random.choice(pair_colors)
    random.shuffle(distractor_colors)

    for color in distractor_colors:
        placed = _place_distractor((h, w), occupied, blocked, color)
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = color
        occupied.update(placed)
        blocked.update(_halo_cells((h, w), placed))

    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
