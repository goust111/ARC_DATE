from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SCAFFOLD = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    banned = set(int(x) for x in exclude)
    return sorted(int(x) for x in np.unique(grid) if int(x) not in banned)


def _boundary_lines(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        bounds = [r for r in range(grid.shape[0]) if not np.any(grid[r, :] == SCAFFOLD)]
    else:
        bounds = [c for c in range(grid.shape[1]) if not np.any(grid[:, c] == SCAFFOLD)]

    assert len(bounds) >= 2, "expected at least two scaffold boundary lines"
    gaps = [b - a for a, b in zip(bounds, bounds[1:])]
    assert len(set(gaps)) == 1, "expected evenly spaced scaffold boundaries"
    assert gaps[0] == 4, "expected 3x3 hollow scaffold cells separated by 1-pixel gaps"
    return bounds


def _nearest_unique(x: int, bounds: list[int]) -> int:
    dists = [abs(int(x) - int(b)) for b in bounds]
    best = min(dists)
    hits = [int(b) for b, d in zip(bounds, dists) if d == best]
    assert len(hits) == 1, "marker must map unambiguously to a nearest scaffold boundary"
    return hits[0]


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {
        (gr, gc)
        for gr in range(int(r0), int(r1))
        for gc in range(int(c0), int(c1))
    }


def _paint_from_specs(
    base: np.ndarray,
    row_bounds: list[int],
    col_bounds: list[int],
    specs: list[dict],
    *,
    force_parity: int | None = None,
    fill_all_centers: bool = False,
    reverse_order: bool = False,
) -> tuple[np.ndarray, bool]:
    row_index = {int(r): i for i, r in enumerate(row_bounds)}
    col_index = {int(c): i for i, c in enumerate(col_bounds)}
    boundary_row_set = set(int(x) for x in row_bounds)
    boundary_col_set = set(int(x) for x in col_bounds)

    out = base.copy()
    override_happened = False
    ordered = sorted(specs, key=lambda spec: (len(spec["markers"]), spec["color"]))
    if reverse_order:
        ordered = list(reversed(ordered))

    for spec in ordered:
        color = int(spec["color"])
        ri0, ci0, ri1, ci1 = map(int, spec["rect"])
        parity = int(spec["parity"] if force_parity is None else force_parity)

        r0 = int(row_bounds[ri0])
        r1 = int(row_bounds[ri1])
        c0 = int(col_bounds[ci0])
        c1 = int(col_bounds[ci1])

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in boundary_row_set or c in boundary_col_set:
                    if out[r, c] == SCAFFOLD:
                        continue
                    if out[r, c] not in (BG, color):
                        override_happened = True
                    out[r, c] = color

        for gr in range(ri0, ri1):
            center_r = (row_bounds[gr] + row_bounds[gr + 1]) // 2
            for gc in range(ci0, ci1):
                center_c = (col_bounds[gc] + col_bounds[gc + 1]) // 2
                if fill_all_centers or (gr + gc) % 2 == parity:
                    if out[center_r, center_c] == SCAFFOLD:
                        continue
                    if out[center_r, center_c] not in (BG, color):
                        override_happened = True
                    out[center_r, center_c] = color

    return out, override_happened


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each marker color to a scaffold-aligned border rectangle and parity-filtered cell-center fill.

    Input structure:
    - A black background contains a grey scaffold lattice of repeated 3x3 hollow frames, with black boundary rows and columns between frames.
    - Two or three non-black, non-grey marker colors appear only as a few pixels on scaffold boundary intersections or boundary segments.
    - Each marker color spans a different scaffold-aligned region, and the first pixel of that color fixes the parity pattern used inside its region.

    Rule:
    - Infer the scaffold boundary rows and columns from the regularly spaced rows and columns that contain no grey scaffold pixels.
    - For each marker color, map every marker to its unique nearest scaffold-boundary intersection, then take the smallest boundary-aligned rectangle covering those mapped points.
    - Process colors in ascending marker-count order, breaking ties by color id for determinism; on each rectangle, color every non-grey pixel on the rectangle's boundary rows and boundary columns.
    - For every enclosed scaffold cell center in that rectangle, color it iff the cell-coordinate parity `(gr + gc) % 2` matches the first marker pixel's `(row + col) % 2`; leave the other centers black.

    Key insight:
    - The marker span determines which scaffold rectangle expands, while the first marker's coordinate parity determines which checkerboard half of that rectangle's cell centers is filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_bounds = _boundary_lines(g, axis=0)
    col_bounds = _boundary_lines(g, axis=1)
    row_index = {int(r): i for i, r in enumerate(row_bounds)}
    col_index = {int(c): i for i, c in enumerate(col_bounds)}
    boundary_row_set = set(int(x) for x in row_bounds)
    boundary_col_set = set(int(x) for x in col_bounds)

    colors = _colors_present(g, exclude={BG, SCAFFOLD})
    assert 2 <= len(colors) <= 3, "expected 2-3 marker colors"

    order = []
    color_data: dict[int, dict] = {}
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert 2 <= coords.shape[0] <= 4, "expected 2-4 markers per color"
        first_r, first_c = map(int, coords[0])
        mapped_rows = [_nearest_unique(int(r), row_bounds) for r, _ in coords]
        mapped_cols = [_nearest_unique(int(c), col_bounds) for _, c in coords]
        r0 = min(mapped_rows)
        r1 = max(mapped_rows)
        c0 = min(mapped_cols)
        c1 = max(mapped_cols)
        assert r0 < r1 and c0 < c1, "marker span must define a non-degenerate rectangle"
        color_data[int(color)] = {
            "count": int(coords.shape[0]),
            "parity": (first_r + first_c) % 2,
            "rect": (row_index[r0], col_index[c0], row_index[r1], col_index[c1]),
        }
        order.append((int(coords.shape[0]), int(color)))

    out = g.copy()
    for _count, color in sorted(order):
        ri0, ci0, ri1, ci1 = color_data[color]["rect"]
        parity = int(color_data[color]["parity"])
        r0 = int(row_bounds[ri0])
        r1 = int(row_bounds[ri1])
        c0 = int(col_bounds[ci0])
        c1 = int(col_bounds[ci1])

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r in boundary_row_set or c in boundary_col_set) and out[r, c] != SCAFFOLD:
                    out[r, c] = color

        for gr in range(ri0, ri1):
            center_r = (row_bounds[gr] + row_bounds[gr + 1]) // 2
            for gc in range(ci0, ci1):
                center_c = (col_bounds[gc] + col_bounds[gc + 1]) // 2
                if (gr + gc) % 2 == parity and out[center_r, center_c] != SCAFFOLD:
                    out[center_r, center_c] = color

    return out


def _build_scaffold(nr: int, nc: int) -> tuple[np.ndarray, list[int], list[int]]:
    row_bounds = [4 * i for i in range(int(nr) + 1)]
    col_bounds = [4 * i for i in range(int(nc) + 1)]
    grid = np.zeros((4 * int(nr) + 1, 4 * int(nc) + 1), dtype=int)
    for r0 in row_bounds[:-1]:
        for c0 in col_bounds[:-1]:
            grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = SCAFFOLD
            grid[r0 + 2, c0 + 2] = BG
    return grid, row_bounds, col_bounds


def _rect_candidates(nr: int, nc: int) -> list[tuple[int, int, int, int]]:
    candidates = []
    for h in range(1, min(4, nr) + 1):
        for w in range(1, min(4, nc) + 1):
            area = h * w
            if not (2 <= area <= 8):
                continue
            for r0 in range(nr - h + 1):
                for c0 in range(nc - w + 1):
                    candidates.append((r0, c0, r0 + h, c0 + w))
    return candidates


def _cell_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 >= r1 or c0 >= c1:
        return 0
    return (r1 - r0) * (c1 - c0)


def _contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return int(a[0]) <= int(b[0]) and int(a[1]) <= int(b[1]) and int(a[2]) >= int(b[2]) and int(a[3]) >= int(b[3])


def _pick_rectangles(nr: int, nc: int, n_colors: int) -> list[tuple[int, int, int, int]] | None:
    candidates = _rect_candidates(nr, nc)
    want_overlap = random.random() < 0.55
    for _ in range(300):
        rects = [random.choice(candidates)]
        while len(rects) < n_colors:
            if want_overlap and len(rects) == 1:
                pool = [
                    rect
                    for rect in candidates
                    if rect != rects[0]
                    and _cell_overlap(rects[0], rect) > 0
                    and not _contains(rects[0], rect)
                    and not _contains(rect, rects[0])
                ]
            else:
                pool = [
                    rect
                    for rect in candidates
                    if rect not in rects
                    and all(_cell_overlap(rect, other) == 0 for other in rects)
                ]
            if not pool:
                break
            rects.append(random.choice(pool))
        if len(rects) != n_colors:
            continue
        overlaps = sum(_cell_overlap(rects[i], rects[j]) > 0 for i in range(n_colors) for j in range(i + 1, n_colors))
        if want_overlap and overlaps == 0:
            continue
        if not want_overlap and overlaps != 0:
            continue
        return rects
    return None


def _marker_positions(
    rect: tuple[int, int, int, int],
    row_bounds: list[int],
    col_bounds: list[int],
    parity: int,
    count: int,
) -> list[tuple[int, int]]:
    ri0, ci0, ri1, ci1 = map(int, rect)
    r0 = int(row_bounds[ri0])
    c0 = int(col_bounds[ci0])
    r1 = int(row_bounds[ri1])
    c1 = int(col_bounds[ci1])

    first = (r0, c0) if parity == 0 else (r0, c0 + 1)
    anchors = [
        first,
        (r1 - 1, c1),
        (r0, c1),
        (r1, c0 + 1),
        (r0 + 1, c0),
        (r1 - 1, c0),
        (r0, c1 - 1),
    ]

    chosen: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pos in anchors:
        r, c = pos
        if pos in seen:
            continue
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            continue
        if r == r0 and c < first[1]:
            continue
        if r % 2 == 0 or any(rr % 2 == 0 for rr, _ in chosen):
            pass
        chosen.append(pos)
        seen.add(pos)
        if len(chosen) == count:
            break

    assert len(chosen) == count
    assert any(r % 2 == 0 for r, _ in chosen)
    assert any(r % 2 == 1 for r, _ in chosen)
    assert chosen[0] == first
    return chosen


def _render_input(base: np.ndarray, specs: list[dict]) -> np.ndarray:
    grid = base.copy()
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["markers"]:
            if grid[r, c] != BG:
                raise ValueError("marker collision")
            grid[r, c] = color
    return grid


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    base: np.ndarray,
    row_bounds: list[int],
    col_bounds: list[int],
    specs: list[dict],
) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(inp, expected):
        return False

    full_centers, _ = _paint_from_specs(base, row_bounds, col_bounds, specs, fill_all_centers=True)
    if np.array_equal(full_centers, expected):
        return False

    if any(int(spec["parity"]) == 1 for spec in specs):
        even_parity, _ = _paint_from_specs(base, row_bounds, col_bounds, specs, force_parity=0)
        if np.array_equal(even_parity, expected):
            return False

    _, override_happened = _paint_from_specs(base, row_bounds, col_bounds, specs)
    if override_happened:
        reversed_out, _ = _paint_from_specs(base, row_bounds, col_bounds, specs, reverse_order=True)
        if np.array_equal(reversed_out, expected):
            return False

    for spec in specs:
        ri0, ci0, ri1, ci1 = map(int, spec["rect"])
        colored_centers = 0
        uncolored_centers = 0
        for gr in range(ri0, ri1):
            center_r = (row_bounds[gr] + row_bounds[gr + 1]) // 2
            for gc in range(ci0, ci1):
                center_c = (col_bounds[gc] + col_bounds[gc + 1]) // 2
                if (gr + gc) % 2 == int(spec["parity"]):
                    colored_centers += 1
                else:
                    uncolored_centers += 1
        if colored_centers == 0 or uncolored_centers == 0:
            return False

    sig = (tuple(map(int, expected.shape)), expected.tobytes())
    if sig in _SEEN_OUTPUT_SIGNATURES:
        return False
    _SEEN_OUTPUT_SIGNATURES.add(sig)
    return True


def _attempt_generate() -> np.ndarray | None:
    nr = random.randint(3, 6)
    nc = random.randint(3, 6)
    base, row_bounds, col_bounds = _build_scaffold(nr, nc)

    n_colors = random.choice((2, 2, 3))
    rects = _pick_rectangles(nr, nc, n_colors)
    if rects is None:
        return None

    colors = random.sample(MARKER_COLORS, n_colors)
    count_patterns = {2: random.choice(((2, 3), (2, 4), (3, 4))), 3: (2, 3, 4)}
    counts = list(count_patterns[n_colors])
    random.shuffle(counts)

    parities = [random.randint(0, 1) for _ in range(n_colors)]
    if not any(p == 1 for p in parities):
        parities[random.randrange(n_colors)] = 1

    specs = []
    for rect, color, parity, count in zip(rects, colors, parities, counts):
        markers = _marker_positions(rect, row_bounds, col_bounds, parity, count)
        specs.append(
            {
                "rect": tuple(int(x) for x in rect),
                "color": int(color),
                "parity": int(parity),
                "markers": tuple(markers),
            }
        )

    try:
        inp = _render_input(base, specs)
    except ValueError:
        return None
    expected, _ = _paint_from_specs(base, row_bounds, col_bounds, specs)
    if not _validate_semantics(inp, expected, base, row_bounds, col_bounds, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
