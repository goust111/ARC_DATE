from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MARKER_COLORS = (1, 2, 3, 6, 7, 8, 9)

GRID_H_RANGE = (15, 21)
GRID_W_RANGE = (17, 24)
SHAPE_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 800


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _trace_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    sr = _sign(dr)
    sc = _sign(dc)
    if dr != 0 and dc != 0:
        assert abs(dr) == abs(dc), "only 45-degree diagonals are supported"
        length = abs(dr)
    else:
        assert dr == 0 or dc == 0, "only axis-aligned segments are supported"
        length = max(abs(dr), abs(dc))
    return [(ar + i * sr, ac + i * sc) for i in range(length + 1)]


def _cross(a: tuple[int, int], b: tuple[int, int], p: tuple[int, int]) -> int:
    return (b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])


def _strictly_inside_convex_quad(point: tuple[int, int], verts: list[tuple[int, int]]) -> bool:
    values = []
    for i in range(4):
        val = _cross(verts[i], verts[(i + 1) % 4], point)
        if val == 0:
            return False
        values.append(val)
    return all(v > 0 for v in values) or all(v < 0 for v in values)


def _make_shape_from_vectors(
    anchor: tuple[int, int],
    diag: tuple[int, int],
    axis: tuple[int, int],
    color: int,
    grid_shape: tuple[int, int],
) -> dict | None:
    h, w = grid_shape
    ar, ac = anchor
    br = ar + diag[0] + axis[0]
    bc = ac + diag[1] + axis[1]
    if diag[0] == 0 or diag[1] == 0 or abs(diag[0]) != abs(diag[1]):
        return None
    if axis == (0, 0) or (axis[0] != 0 and axis[1] != 0):
        return None

    verts = [
        (ar, ac),
        (ar + axis[0], ac + axis[1]),
        (br, bc),
        (ar + diag[0], ac + diag[1]),
    ]
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in verts):
        return None

    boundary: set[tuple[int, int]] = set()
    edge_pairs = list(zip(verts, verts[1:] + verts[:1]))
    for p0, p1 in edge_pairs:
        boundary.update(_trace_segment(p0, p1))

    r0 = min(r for r, _ in verts)
    r1 = max(r for r, _ in verts)
    c0 = min(c for _, c in verts)
    c1 = max(c for _, c in verts)
    interior: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _strictly_inside_convex_quad((r, c), verts):
                interior.add((r, c))

    if len(interior) < 3:
        return None

    return {
        "color": int(color),
        "a": (ar, ac),
        "b": (br, bc),
        "diag": tuple(map(int, diag)),
        "axis": tuple(map(int, axis)),
        "diag_len": abs(diag[0]),
        "axis_len": max(abs(axis[0]), abs(axis[1])),
        "axis_kind": "h" if axis[0] == 0 else "v",
        "sign_r": _sign(diag[0]),
        "sign_c": _sign(diag[1]),
        "verts": verts,
        "boundary": boundary,
        "interior": interior,
        "area": abs(diag[0] * axis[1] - diag[1] * axis[0]),
    }


def _shape_from_pair(
    coord0: tuple[int, int],
    coord1: tuple[int, int],
    color: int,
    grid_shape: tuple[int, int],
) -> dict:
    dr = coord1[0] - coord0[0]
    dc = coord1[1] - coord0[1]
    assert dr != 0 and dc != 0 and abs(dr) != abs(dc)
    step = min(abs(dr), abs(dc))
    diag = (_sign(dr) * step, _sign(dc) * step)
    axis = (dr - diag[0], dc - diag[1])
    shape = _make_shape_from_vectors(coord0, diag, axis, color, grid_shape)
    assert shape is not None
    return shape


def _extract_shapes(grid: np.ndarray) -> list[dict]:
    counts = Counter(map(int, grid.flat))
    marker_colors = [color for color in np.unique(grid) if color not in (BLACK, YELLOW)]
    shapes = []
    for color in sorted(map(int, marker_colors)):
        assert counts[color] == 2, f"marker color {color} must appear exactly twice"
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        shapes.append(_shape_from_pair(coords[0], coords[1], color, grid.shape))
    assert 2 <= len(shapes) <= 3
    return shapes


def _triggered_shape_indices(shapes: list[dict], yellow_points: list[tuple[int, int]]) -> set[int]:
    triggered: set[int] = set()
    for point in yellow_points:
        containers = [idx for idx, shape in enumerate(shapes) if point in shape["interior"]]
        if not containers:
            continue
        min_area = min(shapes[idx]["area"] for idx in containers)
        best = [idx for idx in containers if shapes[idx]["area"] == min_area]
        best.sort(key=lambda idx: (len(shapes[idx]["interior"]), idx))
        triggered.add(best[0])
    return triggered


def _render_output(grid_shape: tuple[int, int], shapes: list[dict], triggered: set[int]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for shape in shapes:
        for r, c in shape["boundary"]:
            out[r, c] = GREY
    for idx in triggered:
        for r, c in shapes[idx]["interior"]:
            out[r, c] = YELLOW
    for shape in shapes:
        out[shape["a"]] = shape["color"]
        out[shape["b"]] = shape["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Construct marker-defined parallelogram outlines and fill only those whose interiors contain yellow trigger pixels.

    Input structure:
    - Black background with 2-3 color-distinct marker pairs, each pair being two single pixels of the same color.
    - Each marker pair has both row and column offset, and those absolute offsets are unequal.
    - Several yellow single pixels are scattered across the grid; some lie inside the yet-to-be-constructed parallelograms and some are outside all of them.

    Rule:
    - For each same-color marker pair, decompose the pair offset into a 45-degree diagonal vector using the smaller absolute offset and a remaining pure horizontal or vertical vector.
    - Treat the two markers as opposite vertices of a parallelogram with those two vectors as adjacent sides, draw its outline in grey, and keep the original marker colors at the two marked vertices.
    - After every outline is constructed, test each yellow pixel against the strict interior of the generated parallelograms; a yellow pixel triggers only the smallest-area containing parallelogram if interiors overlap.
    - Fill the full interior of every triggered parallelogram yellow, keep untriggered interiors black, and erase yellow pixels that are outside all parallelograms.

    Key insight:
    - The relevant containment relation is not present in the input: the yellow pixels must be judged against geometries computed from the marker pairs, and overlap is resolved by the smallest generated parallelogram rather than by the raw input positions alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    shapes = _extract_shapes(g)
    yellow_points = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    triggered = _triggered_shape_indices(shapes, yellow_points)
    return _render_output(g.shape, shapes, triggered)


def _sample_shape(
    grid_shape: tuple[int, int],
    color: int,
    existing: list[dict],
    *,
    must_overlap: dict | None = None,
    forbid_overlap: bool = True,
) -> dict | None:
    h, w = grid_shape
    for _ in range(300):
        if must_overlap is None or random.random() < 0.2:
            axis_kind = random.choice(("h", "v"))
            sign_r = random.choice((-1, 1))
            sign_c = random.choice((-1, 1))
            diag_len = random.randint(3, 6)
            axis_len = random.randint(3, 7)
            if axis_kind == "h":
                diag = (sign_r * diag_len, sign_c * diag_len)
                axis = (0, sign_c * axis_len)
            else:
                diag = (sign_r * diag_len, sign_c * diag_len)
                axis = (sign_r * axis_len, 0)
            rels = [(0, 0), diag, axis, (diag[0] + axis[0], diag[1] + axis[1])]
            min_r = min(r for r, _ in rels)
            max_r = max(r for r, _ in rels)
            min_c = min(c for _, c in rels)
            max_c = max(c for _, c in rels)
            r_lo = 1 - min_r
            r_hi = h - 2 - max_r
            c_lo = 1 - min_c
            c_hi = w - 2 - max_c
            if r_lo > r_hi or c_lo > c_hi:
                continue
            anchor = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        else:
            base = must_overlap
            axis_kind = base["axis_kind"]
            sign_r = base["sign_r"]
            sign_c = base["sign_c"]
            diag_len = min(6, max(3, base["diag_len"] + random.choice((-2, -1, 0, 1, 2))))
            axis_len = min(7, max(3, base["axis_len"] + random.choice((-2, -1, 0, 1, 2))))
            if diag_len * axis_len == base["area"]:
                continue
            if axis_kind == "h":
                diag = (sign_r * diag_len, sign_c * diag_len)
                axis = (0, sign_c * axis_len)
            else:
                diag = (sign_r * diag_len, sign_c * diag_len)
                axis = (sign_r * axis_len, 0)
            anchor = (
                base["a"][0] + random.randint(-3, 3),
                base["a"][1] + random.randint(-3, 3),
            )

        shape = _make_shape_from_vectors(anchor, diag, axis, color, grid_shape)
        if shape is None:
            continue

        if any(shape["boundary"] & other["boundary"] for other in existing):
            continue
        if any(shape["a"] in other["interior"] or shape["b"] in other["interior"] for other in existing):
            continue
        if any(other["a"] in shape["interior"] or other["b"] in shape["interior"] for other in existing):
            continue

        overlaps = [other for other in existing if shape["interior"] & other["interior"]]
        if must_overlap is not None and must_overlap not in overlaps:
            continue
        if forbid_overlap and overlaps:
            continue
        if must_overlap is not None and len(shape["interior"] - must_overlap["interior"]) < 2:
            continue
        return shape
    return None


def _sample_overlapping_shape(
    grid_shape: tuple[int, int],
    color: int,
    base: dict,
    existing: list[dict],
) -> dict | None:
    candidates: list[dict] = []
    h, w = grid_shape

    if base["axis_kind"] == "h":
        main_diag_signs = [(-base["sign_r"], base["sign_c"]), (base["sign_r"], base["sign_c"])]
        anchor_centers = [
            (base["a"][0] + base["sign_r"] * (base["diag_len"] - 1), base["a"][1]),
            (base["a"][0] + base["sign_r"] * (base["diag_len"] - 2), base["a"][1]),
        ]
    else:
        main_diag_signs = [(base["sign_r"], -base["sign_c"]), (base["sign_r"], base["sign_c"])]
        anchor_centers = [
            (base["a"][0], base["a"][1] + base["sign_c"] * (base["diag_len"] - 1)),
            (base["a"][0], base["a"][1] + base["sign_c"] * (base["diag_len"] - 2)),
        ]

    for sign_r, sign_c in main_diag_signs:
        for diag_len in range(3, 7):
            for axis_len in range(3, 8):
                if diag_len * axis_len == base["area"]:
                    continue
                if base["axis_kind"] == "h":
                    diag = (sign_r * diag_len, sign_c * diag_len)
                    axis = (0, base["sign_c"] * axis_len)
                    offsets = [(dr, dc) for dr in range(-1, 2) for dc in range(-2, 3)]
                else:
                    diag = (sign_r * diag_len, sign_c * diag_len)
                    axis = (base["sign_r"] * axis_len, 0)
                    offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(-1, 2)]

                for center_r, center_c in anchor_centers:
                    random.shuffle(offsets)
                    for off_r, off_c in offsets:
                        anchor = (center_r + off_r, center_c + off_c)
                        if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
                            continue
                        shape = _make_shape_from_vectors(anchor, diag, axis, color, grid_shape)
                        if shape is None:
                            continue
                        if shape["boundary"] & base["boundary"]:
                            continue
                        overlap = shape["interior"] & base["interior"]
                        if len(overlap) < 1:
                            continue
                        if len(shape["interior"] - base["interior"]) < 2:
                            continue
                        if len(base["interior"] - shape["interior"]) < 2:
                            continue
                        if any(shape["boundary"] & other["boundary"] for other in existing):
                            continue
                        if any(
                            shape["a"] in other["interior"] or shape["b"] in other["interior"]
                            for other in existing
                        ):
                            continue
                        if any(
                            other["a"] in shape["interior"] or other["b"] in shape["interior"]
                            for other in existing
                        ):
                            continue
                        candidates.append(shape)

    if candidates:
        return random.choice(candidates)

    axis_kinds = ["h", "v"]
    sign_pairs = [(sr, sc) for sr in (-1, 1) for sc in (-1, 1)]
    diag_lengths = list(range(3, 7))
    axis_lengths = list(range(3, 8))
    row_offsets = list(range(-5, 6))
    col_offsets = list(range(-6, 7))

    random.shuffle(axis_kinds)
    random.shuffle(sign_pairs)
    random.shuffle(diag_lengths)
    random.shuffle(axis_lengths)
    random.shuffle(row_offsets)
    random.shuffle(col_offsets)

    for axis_kind in axis_kinds:
        for sign_r, sign_c in sign_pairs:
            for diag_len in diag_lengths:
                for axis_len in axis_lengths:
                    if diag_len * axis_len == base["area"]:
                        continue
                    if axis_kind == "h":
                        diag = (sign_r * diag_len, sign_c * diag_len)
                        axis = (0, sign_c * axis_len)
                    else:
                        diag = (sign_r * diag_len, sign_c * diag_len)
                        axis = (sign_r * axis_len, 0)

                    for shift_r in row_offsets:
                        for shift_c in col_offsets:
                            anchor = (base["a"][0] + shift_r, base["a"][1] + shift_c)
                            if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
                                continue
                            shape = _make_shape_from_vectors(anchor, diag, axis, color, grid_shape)
                            if shape is None:
                                continue
                            if shape["boundary"] & base["boundary"]:
                                continue
                            overlap = shape["interior"] & base["interior"]
                            if len(overlap) < 1:
                                continue
                            if len(shape["interior"] - base["interior"]) < 2:
                                continue
                            if len(base["interior"] - shape["interior"]) < 2:
                                continue
                            if any(shape["boundary"] & other["boundary"] for other in existing):
                                continue
                            if any(
                                shape["a"] in other["interior"] or shape["b"] in other["interior"]
                                for other in existing
                            ):
                                continue
                            if any(
                                other["a"] in shape["interior"] or other["b"] in shape["interior"]
                                for other in existing
                            ):
                                continue
                            candidates.append(shape)

    if not candidates:
        return None
    return random.choice(candidates)


def _pick_random_cells(cells: set[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    choices = list(cells)
    random.shuffle(choices)
    return choices[:k]


def _validate_generated_grid(
    grid: np.ndarray,
    *,
    expect_overlap_source: bool,
) -> None:
    shapes = _extract_shapes(grid)
    yellow_points = [tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)]
    assert 3 <= len(yellow_points) <= 5
    assert len(set(yellow_points)) == len(yellow_points)

    counts = []
    for point in yellow_points:
        assert all(point not in shape["boundary"] for shape in shapes)
        containing = [idx for idx, shape in enumerate(shapes) if point in shape["interior"]]
        counts.append(len(containing))
        if len(containing) > 1:
            areas = sorted(shapes[idx]["area"] for idx in containing)
            assert areas[0] < areas[1], "overlap trigger must have a unique smaller area"

    triggered = _triggered_shape_indices(shapes, yellow_points)
    assert triggered
    assert len(triggered) < len(shapes), "at least one parallelogram should remain hollow"
    assert any(cnt == 0 for cnt in counts), "at least one yellow distractor should be outside all shapes"
    assert any(cnt >= 1 for cnt in counts), "at least one yellow source should be inside a parallelogram"
    if expect_overlap_source:
        assert any(cnt > 1 for cnt in counts), "expected an overlap-case yellow source"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.count_nonzero(out == GREY) > 0
    assert np.count_nonzero(out == YELLOW) > 0


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        grid_shape = (h, w)
        shape_count = random.choice(SHAPE_COUNT_CHOICES)
        overlap_mode = random.random() < 0.8
        if overlap_mode and (h < 18 or w < 20):
            continue
        colors = random.sample(MARKER_COLORS, shape_count)

        shapes: list[dict] = []
        first = _sample_shape(grid_shape, colors[0], [], forbid_overlap=True)
        if first is None:
            continue
        if overlap_mode and (first["diag_len"] < 4 or first["axis_len"] < 4):
            continue
        shapes.append(first)

        if overlap_mode:
            second = _sample_overlapping_shape(grid_shape, colors[1], first, shapes)
            if second is None:
                continue
            shapes.append(second)
            start_idx = 2
        else:
            second = _sample_shape(grid_shape, colors[1], shapes, forbid_overlap=True)
            if second is None:
                continue
            shapes.append(second)
            start_idx = 2

        failed = False
        for idx in range(start_idx, shape_count):
            shape = _sample_shape(grid_shape, colors[idx], shapes, forbid_overlap=True)
            if shape is None:
                failed = True
                break
            shapes.append(shape)
        if failed:
            continue

        shape_by_area = sorted(range(len(shapes)), key=lambda i: (shapes[i]["area"], i))
        smaller_overlap_idx = None
        overlap_cells: set[tuple[int, int]] = set()
        if overlap_mode:
            overlap_cells = shapes[0]["interior"] & shapes[1]["interior"]
            if not overlap_cells:
                continue
            smaller_overlap_idx = min((0, 1), key=lambda i: (shapes[i]["area"], i))

        all_interiors = [shape["interior"] for shape in shapes]
        exclusive_cells = []
        for idx, shape in enumerate(shapes):
            others = set().union(*(all_interiors[j] for j in range(len(shapes)) if j != idx))
            exclusive = shape["interior"] - others
            if len(exclusive) < 2:
                failed = True
                break
            exclusive_cells.append(exclusive)
        if failed:
            continue

        inside_points: list[tuple[int, int]] = []
        triggered_shapes: set[int] = set()
        used: set[tuple[int, int]] = set()

        if overlap_mode:
            overlap_choice = random.choice(list(overlap_cells))
            inside_points.append(overlap_choice)
            used.add(overlap_choice)
            triggered_shapes.add(int(smaller_overlap_idx))

            remaining = [idx for idx in range(len(shapes)) if idx != smaller_overlap_idx]
            random.shuffle(remaining)
            if remaining and len(shapes) == 3 and random.random() < 0.7:
                extra_idx = remaining[0]
                extra_pool = exclusive_cells[extra_idx] - used
                if len(extra_pool) < 1:
                    continue
                extra_choice = random.choice(list(extra_pool))
                inside_points.append(extra_choice)
                used.add(extra_choice)
                triggered_shapes.add(extra_idx)
        else:
            fill_count = random.randint(1, len(shapes) - 1)
            chosen = shape_by_area[:]
            random.shuffle(chosen)
            chosen = chosen[:fill_count]
            for idx in chosen:
                pool = exclusive_cells[idx] - used
                if len(pool) < 1:
                    failed = True
                    break
                choice = random.choice(list(pool))
                inside_points.append(choice)
                used.add(choice)
                triggered_shapes.add(idx)
            if failed:
                continue

        if len(triggered_shapes) == len(shapes):
            continue

        if random.random() < 0.35:
            duplicate_targets = [idx for idx in triggered_shapes if len(exclusive_cells[idx] - used) >= 1]
            if duplicate_targets:
                idx = random.choice(duplicate_targets)
                choice = random.choice(list(exclusive_cells[idx] - used))
                inside_points.append(choice)
                used.add(choice)

        total_sources = random.randint(3, 5)
        if total_sources <= len(inside_points):
            continue
        outside_needed = total_sources - len(inside_points)
        all_busy = set().union(*(shape["boundary"] | shape["interior"] for shape in shapes))
        all_busy.update(shape["a"] for shape in shapes)
        all_busy.update(shape["b"] for shape in shapes)
        outside_pool = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in all_busy
        }
        if len(outside_pool) < outside_needed:
            continue
        outside_points = _pick_random_cells(outside_pool, outside_needed)
        if len(outside_points) != outside_needed:
            continue

        grid = np.zeros(grid_shape, dtype=int)
        for shape in shapes:
            grid[shape["a"]] = shape["color"]
            grid[shape["b"]] = shape["color"]
        for r, c in inside_points + outside_points:
            if grid[r, c] != BLACK:
                failed = True
                break
            grid[r, c] = YELLOW
        if failed:
            continue

        try:
            _validate_generated_grid(grid, expect_overlap_source=overlap_mode)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
