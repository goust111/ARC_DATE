from __future__ import annotations

from collections import deque
from itertools import permutations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (RED, BLUE, ORANGE)
RANK_RECOLORS = (RED, BLUE, YELLOW)

GRID_MIN = 18
GRID_MAX = 25
DIAG_LEN_RANGE = (3, 6)
STRAIGHT_LEN_RANGE = (3, 7)
COUNT_PATTERNS = ((2, 4, 6), (2, 3, 5), (2, 4, 5), (3, 4, 6))
TOTAL_GREY_RANGE = (15, 25)
OUTSIDE_GREY_MIN = 4
GENERATION_ATTEMPTS = 500
SHAPE_ATTEMPTS = 300
PICK_ATTEMPTS = 24


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    length = max(abs(dr), abs(dc))
    assert length > 0
    assert abs(dr) in (0, length)
    assert abs(dc) in (0, length)
    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + i * sr, c0 + i * sc) for i in range(length + 1)]


def _decompose_marker_vector(
    p: tuple[int, int], q: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int], str]:
    dr = q[0] - p[0]
    dc = q[1] - p[1]
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    assert abs_dr > 0 and abs_dc > 0 and abs_dr != abs_dc

    diag_len = min(abs_dr, abs_dc)
    straight_len = abs(abs_dr - abs_dc)
    sr = _sign(dr)
    sc = _sign(dc)
    diag_vec = (sr * diag_len, sc * diag_len)
    if abs_dr > abs_dc:
        return (sr * straight_len, 0), diag_vec, "row"
    return (0, sc * straight_len), diag_vec, "col"


def _parallelogram_masks(
    shape: tuple[int, int], p: tuple[int, int], q: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, str]:
    straight_vec, diag_vec, dominant_axis = _decompose_marker_vector(p, q)
    straight_corner = (p[0] + straight_vec[0], p[1] + straight_vec[1])
    diag_corner = (p[0] + diag_vec[0], p[1] + diag_vec[1])

    perimeter = np.zeros(shape, dtype=bool)
    for a, b in ((p, straight_corner), (p, diag_corner), (straight_corner, q), (diag_corner, q)):
        for r, c in _segment_cells(a, b):
            assert _in_bounds(shape, r, c)
            perimeter[r, c] = True

    visited = np.zeros(shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    h, w = shape
    for r in range(h):
        for c in (0, w - 1):
            if not perimeter[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not perimeter[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc) and not perimeter[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    interior = (~perimeter) & (~visited)
    assert np.any(interior), "parallelogram must enclose a non-empty interior"
    return perimeter, interior, dominant_axis


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(mask.shape, nr, nc):
                out[nr, nc] = True
    return out


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(mask)]


def _exclusion_zone(coords: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        blocked.add((r, c))
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, nr, nc):
                blocked.add((nr, nc))
    return blocked


def _pick_isolated_cells(
    coords: list[tuple[int, int]],
    count: int,
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    coords = [tuple(map(int, coord)) for coord in coords]
    base_forbidden = set() if forbidden is None else set(forbidden)
    if len(coords) < count:
        return None

    for _ in range(PICK_ATTEMPTS):
        order = np.random.permutation(len(coords))
        blocked = set(base_forbidden)
        picked: list[tuple[int, int]] = []
        for idx in order:
            cell = coords[int(idx)]
            if cell in blocked:
                continue
            picked.append(cell)
            blocked.add(cell)
            for nr, nc in _neighbors8(cell[0], cell[1]):
                if _in_bounds(shape, nr, nc):
                    blocked.add((nr, nc))
            if len(picked) == count:
                return picked
    return None


def _capacity_up_to_six(coords: list[tuple[int, int]], shape: tuple[int, int]) -> int:
    for target in range(min(6, len(coords)), 1, -1):
        if _pick_isolated_cells(coords, target, shape) is not None:
            return target
    return 1


def _shape_from_markers(
    shape: tuple[int, int], marker_color: int, p: tuple[int, int], q: tuple[int, int]
) -> dict:
    perimeter, interior, dominant_axis = _parallelogram_masks(shape, p, q)
    interior_coords = _coords_from_mask(interior)
    return {
        "marker_color": int(marker_color),
        "markers": (tuple(map(int, p)), tuple(map(int, q))),
        "perimeter": perimeter,
        "interior": interior,
        "interior_coords": interior_coords,
        "mask": perimeter | interior,
        "dominant_axis": dominant_axis,
        "interior_area": int(np.count_nonzero(interior)),
        "capacity": int(_capacity_up_to_six(interior_coords, shape)),
    }


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    allowed = {BLACK, BLUE, RED, GREY, ORANGE}
    assert set(map(int, np.unique(g))) <= allowed

    shapes: list[dict] = []
    occupied = np.zeros(g.shape, dtype=bool)
    for marker_color in MARKER_COLORS:
        marker_coords = np.argwhere(g == marker_color)
        assert marker_coords.shape == (2, 2), f"expected exactly two markers of color {marker_color}"
        p = tuple(map(int, marker_coords[0]))
        q = tuple(map(int, marker_coords[1]))
        info = _shape_from_markers(g.shape, marker_color, p, q)
        assert not np.any(occupied & info["mask"]), "parallelograms must not overlap"
        occupied |= info["mask"]
        info["grey_count"] = int(np.count_nonzero((g == GREY) & info["interior"]))
        shapes.append(info)

    counts = [shape["grey_count"] for shape in shapes]
    assert min(counts) >= 2, "each interior must contain at least two grey pixels"
    assert len(set(counts)) == 3, "grey counts must be pairwise distinct"
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the three green parallelogram baskets, count each basket's interior grey pixels, and recolor those interior greys by density rank.

    Input structure:
    - The grid has black background, exactly three same-color marker pairs, and scattered grey singleton pixels.
    - The marker-pair colors are red, blue, and orange, with exactly two pixels of each color.
    - Each pair lies on different rows and columns, and each pair implies one closed parallelogram.

    Rule:
    - For each same-color marker pair, decompose the marker offset into a 45-degree diagonal vector and the remaining axis-aligned vector.
    - Use those two vectors as adjacent sides of a parallelogram whose opposite corners are the marker pixels, then draw its full perimeter in green, covering the markers.
    - For each drawn parallelogram, treat only its own green perimeter as a wall, find the enclosed interior by 4-connected outside flood fill, count the grey pixels inside, then rank the three counts from largest to smallest.
    - Recolor the interior grey pixels of the highest-count parallelogram to red, the second-highest to blue, and the lowest to yellow; keep green perimeters, outside grey pixels, and black background unchanged.

    Key insight:
    - The output color of each grey cluster is not tied to marker color directly; it is determined only after reconstructing each hidden geometric basket and counting how many grey pixels that basket encloses.
    """
    g = np.asarray(grid, dtype=int)
    shapes = _analyze_grid(g)

    out = g.copy()
    for shape in shapes:
        out[shape["perimeter"]] = GREEN

    ranked = sorted(shapes, key=lambda item: item["grey_count"], reverse=True)
    for rank, shape in enumerate(ranked):
        out[shape["interior"] & (g == GREY)] = RANK_RECOLORS[rank]
    return out


def _sample_shape(h: int, w: int, dominant_axis: str) -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        diag_len = int(np.random.randint(DIAG_LEN_RANGE[0], DIAG_LEN_RANGE[1] + 1))
        straight_len = int(np.random.randint(STRAIGHT_LEN_RANGE[0], STRAIGHT_LEN_RANGE[1] + 1))
        sr = int(np.random.choice(np.array([-1, 1], dtype=int)))
        sc = int(np.random.choice(np.array([-1, 1], dtype=int)))

        diag_vec = (sr * diag_len, sc * diag_len)
        if dominant_axis == "row":
            straight_vec = (sr * straight_len, 0)
        else:
            straight_vec = (0, sc * straight_len)

        rel = np.array(
            [
                (0, 0),
                straight_vec,
                diag_vec,
                (straight_vec[0] + diag_vec[0], straight_vec[1] + diag_vec[1]),
            ],
            dtype=int,
        )
        margin = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
        r_lo = int(margin - np.min(rel[:, 0]))
        r_hi = int((h - 1 - margin) - np.max(rel[:, 0]))
        c_lo = int(margin - np.min(rel[:, 1]))
        c_hi = int((w - 1 - margin) - np.max(rel[:, 1]))
        if r_lo > r_hi or c_lo > c_hi:
            continue

        p = (int(np.random.randint(r_lo, r_hi + 1)), int(np.random.randint(c_lo, c_hi + 1)))
        q = (p[0] + int(straight_vec[0] + diag_vec[0]), p[1] + int(straight_vec[1] + diag_vec[1]))
        info = _shape_from_markers((h, w), -1, p, q)
        if info["dominant_axis"] != dominant_axis:
            continue
        if info["interior_area"] < 8:
            continue
        if info["capacity"] < 6:
            continue
        return info
    return None


def _count_assignment(shapes: list[dict]) -> tuple[int, int, int] | None:
    capacities = [int(shape["capacity"]) for shape in shapes]
    area_order = tuple(
        idx for idx, _ in sorted(enumerate(shapes), key=lambda item: item[1]["interior_area"], reverse=True)
    )

    pattern_order = list(COUNT_PATTERNS)
    np.random.shuffle(pattern_order)
    for pattern in pattern_order:
        candidates = list(set(permutations(pattern)))
        np.random.shuffle(candidates)
        for assignment in candidates:
            if any(assignment[i] > capacities[i] for i in range(3)):
                continue
            count_order = tuple(
                idx for idx, _ in sorted(enumerate(assignment), key=lambda item: item[1], reverse=True)
            )
            if count_order == area_order:
                continue
            return tuple(map(int, assignment))
    return None


def _build_grid_from_shapes(h: int, w: int, shapes: list[dict], counts: tuple[int, int, int]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    for shape in shapes:
        p, q = shape["markers"]
        grid[p] = shape["marker_color"]
        grid[q] = shape["marker_color"]

    chosen_greys: list[tuple[int, int]] = []
    for shape, count in zip(shapes, counts):
        picks = _pick_isolated_cells(shape["interior_coords"], count, (h, w))
        if picks is None:
            return None
        chosen_greys.extend(picks)
        for r, c in picks:
            grid[r, c] = GREY

    union_mask = np.zeros((h, w), dtype=bool)
    for shape in shapes:
        union_mask |= shape["mask"]
    outside_zone = ~_dilate8(union_mask)
    outside_coords = _coords_from_mask(outside_zone)

    inside_total = sum(counts)
    total_min = max(TOTAL_GREY_RANGE[0], inside_total + OUTSIDE_GREY_MIN)
    total_max = min(TOTAL_GREY_RANGE[1], inside_total + 9)
    if total_min > total_max:
        return None
    total_greys = int(np.random.randint(total_min, total_max + 1))
    outside_count = total_greys - inside_total

    outside_picks = _pick_isolated_cells(
        outside_coords,
        outside_count,
        (h, w),
        forbidden=_exclusion_zone(chosen_greys, (h, w)),
    )
    if outside_picks is None:
        return None
    for r, c in outside_picks:
        grid[r, c] = GREY
    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert TOTAL_GREY_RANGE[0] <= int(np.count_nonzero(g == GREY)) <= TOTAL_GREY_RANGE[1]

    shapes = _analyze_grid(g)
    assert {"row", "col"} <= {shape["dominant_axis"] for shape in shapes}

    dilated_occupied = np.zeros(g.shape, dtype=bool)
    for shape in shapes:
        assert not np.any(dilated_occupied & shape["mask"])
        dilated_occupied |= _dilate8(shape["mask"])

    marker_coords = [tuple(map(int, coord)) for color in MARKER_COLORS for coord in np.argwhere(g == color)]
    grey_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == GREY)]
    for i, (r0, c0) in enumerate(grey_coords):
        for r1, c1 in grey_coords[i + 1 :]:
            assert max(abs(r0 - r1), abs(c0 - c1)) > 1, "grey pixels must be isolated"
        for mr, mc in marker_coords:
            assert max(abs(r0 - mr), abs(c0 - mc)) > 1, "grey pixels must not touch markers"

    inside_total = sum(shape["grey_count"] for shape in shapes)
    outside_total = len(grey_coords) - inside_total
    assert outside_total >= OUTSIDE_GREY_MIN

    area_order = [idx for idx, _ in sorted(enumerate(shapes), key=lambda item: item[1]["interior_area"], reverse=True)]
    count_order = [idx for idx, _ in sorted(enumerate(shapes), key=lambda item: item[1]["grey_count"], reverse=True)]
    assert area_order != count_order, "grey ranking must not collapse to area ranking"

    out = main(g)
    for shape in shapes:
        assert np.all(out[shape["perimeter"]] == GREEN)

    ranked = sorted(shapes, key=lambda item: item["grey_count"], reverse=True)
    for rank, shape in enumerate(ranked):
        recolor_mask = shape["interior"] & (g == GREY)
        assert int(np.count_nonzero(recolor_mask)) == shape["grey_count"]
        assert np.all(out[recolor_mask] == RANK_RECOLORS[rank])

    outside_mask = (g == GREY).copy()
    for shape in shapes:
        outside_mask &= ~shape["interior"]
    assert np.all(out[outside_mask] == GREY)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        h = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
        w = int(np.random.randint(GRID_MIN, GRID_MAX + 1))

        branch_plan = ["row", "col", str(np.random.choice(np.array(["row", "col"], dtype=object)))]
        np.random.shuffle(branch_plan)
        shapes: list[dict] = []
        reserved = np.zeros((h, w), dtype=bool)

        for dominant_axis in branch_plan:
            chosen = None
            for _ in range(SHAPE_ATTEMPTS):
                candidate = _sample_shape(h, w, dominant_axis)
                if candidate is None:
                    continue
                if np.any(reserved & candidate["mask"]):
                    continue
                chosen = candidate
                break
            if chosen is None:
                shapes = []
                break
            shapes.append(chosen)
            reserved |= _dilate8(chosen["mask"])

        if len(shapes) != 3:
            continue

        colors = list(MARKER_COLORS)
        np.random.shuffle(colors)
        for shape, color in zip(shapes, colors):
            shape["marker_color"] = int(color)

        counts = _count_assignment(shapes)
        if counts is None:
            continue

        grid = _build_grid_from_shapes(h, w, shapes, counts)
        if grid is None:
            continue

        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
