from __future__ import annotations

from itertools import product
from pathlib import Path
import random
import sys

import numpy as np


def _import_arc_utils():
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers"
        if (candidate / "arc_utils.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
    try:
        from arc_utils import Color, colors_present, get_bg

        return Color, colors_present, get_bg
    except Exception:
        class _Color:
            BLACK = 0
            BLUE = 1
            RED = 2
            GREEN = 3
            YELLOW = 4
            GREY = 5
            PINK = 6
            ORANGE = 7
            TEAL = 8
            MAROON = 9
            NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        def _colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
            colors = set(map(int, np.unique(grid)))
            if exclude is not None:
                colors -= {int(v) for v in exclude}
            return sorted(colors)

        def _get_bg(grid: np.ndarray) -> int:
            values, counts = np.unique(grid, return_counts=True)
            return int(values[int(np.argmax(counts))])

        return _Color, _colors_present, _get_bg


Color, colors_present, get_bg = _import_arc_utils()


GEN_ATTEMPTS = 400
GRID_SIZE_RANGE = (14, 22)
TWO_PAIR_PROB = 0.35
PAIR_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]


def _interior_area(r0: int, r1: int, c0: int, c1: int) -> int:
    return int((r1 - r0 - 1) * (c1 - c0 - 1))


def _make_pair(color: int, r0: int, r1: int, c0: int, c1: int, diag: int) -> dict:
    if diag == 0:
        points = [(r0, c0), (r1, c1)]
    else:
        points = [(r0, c1), (r1, c0)]
    return {
        "color": int(color),
        "rows": (int(r0), int(r1)),
        "cols": (int(c0), int(c1)),
        "points": [(int(r), int(c)) for r, c in points],
        "area": _interior_area(r0, r1, c0, c1),
    }


def _pair_sort_key(pair: dict) -> tuple[int, int]:
    return int(pair["area"]), int(pair["color"])


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    pairs: list[dict] = []
    for color in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(color))
        assert pts.shape == (2, 2), "each marker color must appear exactly twice"
        (r0, c0), (r1, c1) = [(int(r), int(c)) for r, c in pts]
        assert r0 != r1 and c0 != c1, "marker pair must occupy different rows and columns"
        rlo, rhi = sorted((r0, r1))
        clo, chi = sorted((c0, c1))
        area = _interior_area(rlo, rhi, clo, chi)
        assert area > 0, "marker rectangle must have a non-empty interior"
        pairs.append(
            {
                "color": int(color),
                "rows": (int(rlo), int(rhi)),
                "cols": (int(clo), int(chi)),
                "points": [(r0, c0), (r1, c1)],
                "area": int(area),
            }
        )

    assert 2 <= len(pairs) <= 3, "expected two or three marker pairs"
    return pairs


def _paint_if_better(out: np.ndarray, best_area: np.ndarray, r: int, c: int, color: int, area: int) -> None:
    current_area = int(best_area[r, c])
    current_color = int(out[r, c])
    if area < current_area or (area == current_area and color < current_color):
        best_area[r, c] = int(area)
        out[r, c] = int(color)


def _line_canvas(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), Color.BLACK, dtype=int)
    best_area = np.full((h, w), 10**9, dtype=int)
    for pair in sorted(pairs, key=_pair_sort_key):
        color = int(pair["color"])
        area = int(pair["area"])
        r0, r1 = pair["rows"]
        c0, c1 = pair["cols"]
        for c in range(w):
            _paint_if_better(out, best_area, r0, c, color, area)
            _paint_if_better(out, best_area, r1, c, color, area)
        for r in range(h):
            _paint_if_better(out, best_area, r, c0, color, area)
            _paint_if_better(out, best_area, r, c1, color, area)
    return out


def _containing_pairs(pairs: list[dict], top: int, bottom: int, left: int, right: int) -> list[dict]:
    return [
        pair
        for pair in pairs
        if pair["rows"][0] <= top <= bottom <= pair["rows"][1]
        and pair["cols"][0] <= left <= right <= pair["cols"][1]
    ]


def _enumerate_regions(pairs: list[dict]) -> list[dict]:
    rows = sorted({int(r) for pair in pairs for r in pair["rows"]})
    cols = sorted({int(c) for pair in pairs for c in pair["cols"]})
    regions: list[dict] = []
    for top, bottom in zip(rows, rows[1:]):
        if bottom - top <= 1:
            continue
        for left, right in zip(cols, cols[1:]):
            if right - left <= 1:
                continue
            owners = _containing_pairs(pairs, top, bottom, left, right)
            top_sources = {pair["color"] for pair in pairs if top in pair["rows"]}
            bottom_sources = {pair["color"] for pair in pairs if bottom in pair["rows"]}
            left_sources = {pair["color"] for pair in pairs if left in pair["cols"]}
            right_sources = {pair["color"] for pair in pairs if right in pair["cols"]}
            winner = min(owners, key=_pair_sort_key) if owners else None
            regions.append(
                {
                    "top": int(top),
                    "bottom": int(bottom),
                    "left": int(left),
                    "right": int(right),
                    "owners": owners,
                    "winner": winner,
                    "edge_colors": top_sources | bottom_sources | left_sources | right_sources,
                }
            )
    return regions


def _solve_from_pairs(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = _line_canvas(shape, pairs)
    for region in _enumerate_regions(pairs):
        winner = region["winner"]
        if winner is None:
            continue
        out[
            region["top"] + 1 : region["bottom"],
            region["left"] + 1 : region["right"],
        ] = int(winner["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw full-span divider lines from each same-color marker pair, then fill each enclosed sub-rectangle by the smallest containing pair.

    Input structure:
    - The grid has a black background and two or three foreground colors.
    - Each foreground color appears exactly twice as isolated marker pixels.
    - The two markers of one color lie on different rows and different columns, defining one axis-aligned rectangle.

    Rule:
    - For every color pair, draw full-width horizontal lines through the two marker rows and full-height vertical lines through the two marker columns, using that pair color.
    - If multiple pairs paint the same line cell, keep the color of the pair whose marker rectangle has the smaller strict-interior area.
    - Consider every positive-area block between adjacent divider rows and adjacent divider columns; if one or more marker rectangles contain that block, fill the block with the color of the smallest containing rectangle.
    - Leave divider lines colored, and leave any enclosed block with no containing rectangle black.

    Key insight:
    - The visible output is determined by the global line arrangement, but both line conflicts and interior fills are resolved by the same hidden quantity: the area of the marker-defined rectangle that contains the cell.
    """
    pairs = _extract_pairs(np.asarray(grid, dtype=int))
    return _solve_from_pairs(tuple(map(int, grid.shape)), pairs)


def _sample_gaps(total: int, parts: int, min_gap: int) -> list[int]:
    slack = total - parts * min_gap
    assert slack >= 0
    gaps = [min_gap] * parts
    for _ in range(slack):
        gaps[random.randrange(parts)] += 1
    random.shuffle(gaps)
    return gaps


def _no_three_collinear(points: list[tuple[int, int]]) -> bool:
    for i in range(len(points)):
        r1, c1 = points[i]
        for j in range(i + 1, len(points)):
            r2, c2 = points[j]
            for k in range(j + 1, len(points)):
                r3, c3 = points[k]
                if (r2 - r1) * (c3 - c1) == (r3 - r1) * (c2 - c1):
                    return False
    return True


def _pair_areas_distinct(pairs: list[dict]) -> bool:
    areas = [int(pair["area"]) for pair in pairs]
    return len(set(areas)) == len(areas)


def _visible_for_all_pairs(out: np.ndarray, pairs: list[dict]) -> bool:
    return all(np.any(out == int(pair["color"])) for pair in pairs)


def _has_priority_intersection(pairs: list[dict]) -> bool:
    all_rows = {}
    all_cols = {}
    for pair in pairs:
        for r in pair["rows"]:
            all_rows.setdefault(int(r), []).append(pair)
        for c in pair["cols"]:
            all_cols.setdefault(int(c), []).append(pair)
    for row_pairs in all_rows.values():
        for col_pairs in all_cols.values():
            contributors = {pair["color"]: pair for pair in row_pairs + col_pairs}
            if len(contributors) < 2:
                continue
            areas = sorted(int(pair["area"]) for pair in contributors.values())
            if len(areas) >= 2 and areas[0] < areas[1]:
                return True
    return False


def _region_checks(pairs: list[dict]) -> tuple[bool, bool, bool]:
    regions = _enumerate_regions(pairs)
    if not regions:
        return False, False, False
    no_ownerless = all(region["winner"] is not None for region in regions)
    mixed_boundary = any(len(region["edge_colors"]) >= 2 for region in regions)
    priority_region = any(len(region["owners"]) >= 2 for region in regions)
    return no_ownerless, mixed_boundary, priority_region


def _pairs_to_grid(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for pair in pairs:
        for r, c in pair["points"]:
            assert grid[r, c] == Color.BLACK
            grid[r, c] = int(pair["color"])
    return grid


def _orient_pairs(rect_specs: list[tuple[int, int, int, int]], colors: list[int]) -> list[dict] | None:
    choices = [(0, 1)] * len(rect_specs)
    for diags in product(*choices):
        pairs = [
            _make_pair(color, r0, r1, c0, c1, diag)
            for (r0, r1, c0, c1), color, diag in zip(rect_specs, colors, diags)
        ]
        points = [point for pair in pairs for point in pair["points"]]
        if len(set(points)) != len(points):
            continue
        if not _no_three_collinear(points):
            continue
        return pairs
    return None


def _two_pair_template(h: int, w: int, colors: list[int]) -> list[dict] | None:
    row_span = random.randint(8, h - 3)
    col_span = random.randint(8, w - 3)
    r0 = random.randint(1, h - row_span - 1)
    c0 = random.randint(1, w - col_span - 1)
    r1 = r0 + row_span
    c1 = c0 + col_span

    row_gaps = _sample_gaps(row_span, 3, 2)
    col_gaps = _sample_gaps(col_span, 3, 2)
    ir0 = r0 + row_gaps[0]
    ir1 = ir0 + row_gaps[1]
    ic0 = c0 + col_gaps[0]
    ic1 = ic0 + col_gaps[1]

    rects = [
        (r0, r1, c0, c1),
        (ir0, ir1, ic0, ic1),
    ]
    pairs = _orient_pairs(rects, colors[:2])
    if pairs is None or not _pair_areas_distinct(pairs):
        return None
    return pairs


def _three_pair_template(h: int, w: int, colors: list[int]) -> list[dict] | None:
    row_span = random.randint(12, h - 2)
    col_span = random.randint(12, w - 2)
    r0 = random.randint(1, h - row_span - 1)
    c0 = random.randint(1, w - col_span - 1)
    r5 = r0 + row_span
    c5 = c0 + col_span

    row_gaps = _sample_gaps(row_span, 5, 2)
    col_gaps = _sample_gaps(col_span, 5, 2)

    r1 = r0 + row_gaps[0]
    r2 = r1 + row_gaps[1]
    r3 = r2 + row_gaps[2]
    r4 = r3 + row_gaps[3]

    c1 = c0 + col_gaps[0]
    c2 = c1 + col_gaps[1]
    c3 = c2 + col_gaps[2]
    c4 = c3 + col_gaps[3]

    rects = [
        (r0, r5, c0, c5),
        (r1, r4, c2, c3),
        (r2, r3, c1, c4),
    ]
    pairs = _orient_pairs(rects, colors[:3])
    if pairs is None or not _pair_areas_distinct(pairs):
        return None
    return pairs


def _semantic_ok(grid: np.ndarray, pairs: list[dict], out: np.ndarray) -> bool:
    if not _visible_for_all_pairs(out, pairs):
        return False
    if not _has_priority_intersection(pairs):
        return False
    no_ownerless, mixed_boundary, priority_region = _region_checks(pairs)
    if not (no_ownerless and mixed_boundary and priority_region):
        return False
    if any(np.count_nonzero(grid == int(pair["color"])) != 2 for pair in pairs):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    pair_count = 2 if random.random() < TWO_PAIR_PROB else 3
    colors = random.sample(PAIR_COLORS, k=pair_count)

    try:
        pairs = (
            _two_pair_template(h, w, colors)
            if pair_count == 2
            else _three_pair_template(h, w, colors)
        )
    except ValueError:
        return None

    if pairs is None:
        return None

    grid = _pairs_to_grid((h, w), pairs)
    out = _solve_from_pairs((h, w), pairs)
    if not np.array_equal(main(grid), out):
        return None
    if not _semantic_ok(grid, pairs, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
