from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _utils_dir = _parent / "new" / "solvers"
    if _utils_dir.is_dir():
        sys.path.append(str(_utils_dir))
        break

from arc_utils import Color, in_bounds, neighbors4


COUNT_TO_FILL = {
    1: Color.BLUE,
    2: Color.YELLOW,
    3: Color.ORANGE,
    4: Color.PINK,
}

GEN_HEIGHT_RANGE = (14, 20)
GEN_WIDTH_RANGE = (14, 20)
GEN_BRANCH_COUNT_RANGE = (4, 7)
GEN_TRUNK_SPAN_MIN = 7
GEN_TRUNK_SPAN_MAX = 12
GEN_MAX_ATTEMPTS = 500


def _find_trunk(red_coords: list[tuple[int, int]]) -> tuple[str, int, int, int]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in red_coords:
        rows[r].append(c)
        cols[c].append(r)

    row_trunks = [(r, sorted(cs)) for r, cs in rows.items() if len(cs) == 2]
    col_trunks = [(c, sorted(rs)) for c, rs in cols.items() if len(rs) == 2]
    assert len(row_trunks) + len(col_trunks) == 1, "expected exactly one aligned red pair"

    if row_trunks:
        r, cols_on_row = row_trunks[0]
        return "row", int(r), int(cols_on_row[0]), int(cols_on_row[1])

    c, rows_on_col = col_trunks[0]
    return "col", int(c), int(rows_on_col[0]), int(rows_on_col[1])


def _draw_structure(grid: np.ndarray, red_coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[str, int, int, int]]:
    axis, fixed, lo, hi = _find_trunk(red_coords)
    out = np.array(grid, dtype=int, copy=True)

    if axis == "row":
        for c in range(lo + 1, hi):
            if int(out[fixed, c]) == Color.BLACK:
                out[fixed, c] = Color.GREEN
        for r, c in red_coords:
            if r == fixed:
                continue
            assert lo <= c <= hi, "off-trunk controls must project onto the trunk"
            r0, r1 = sorted((r, fixed))
            for rr in range(r0 + 1, r1):
                if int(out[rr, c]) == Color.BLACK:
                    out[rr, c] = Color.GREEN
        return out, (axis, fixed, lo, hi)

    for r in range(lo + 1, hi):
        if int(out[r, fixed]) == Color.BLACK:
            out[r, fixed] = Color.GREEN
    for r, c in red_coords:
        if c == fixed:
            continue
        assert lo <= r <= hi, "off-trunk controls must project onto the trunk"
        c0, c1 = sorted((c, fixed))
        for cc in range(c0 + 1, c1):
            if int(out[r, cc]) == Color.BLACK:
                out[r, cc] = Color.GREEN

    return out, (axis, fixed, lo, hi)


def _workspace_bbox(red_coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in red_coords]
    cols = [c for _, c in red_coords]
    return min(rows), min(cols), max(rows), max(cols)


def _interior_regions(walls: np.ndarray, bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox
    seen = np.zeros_like(walls, dtype=bool)
    regions: list[np.ndarray] = []

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if bool(walls[r, c]) or bool(seen[r, c]):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not (r0 < nr < r1 and c0 < nc < c1):
                        continue
                    if bool(walls[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            regions.append(np.array(coords, dtype=int))

    return regions


def _adjacent_reds(grid: np.ndarray, coords: np.ndarray) -> set[tuple[int, int]]:
    reds: set[tuple[int, int]] = set()
    for r0, c0 in coords:
        for r, c in neighbors4(int(r0), int(c0)):
            if in_bounds(grid.shape, r, c) and int(grid[r, c]) == Color.RED:
                reds.add((int(r), int(c)))
    return reds


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED}, "input must contain only black and red"

    red_coords = [(int(r), int(c)) for r, c in np.argwhere(g == Color.RED)]
    assert 5 <= len(red_coords) <= 8, "expected 5-8 red control pixels"

    drawn, trunk = _draw_structure(g, red_coords)
    bbox = _workspace_bbox(red_coords)
    regions = _interior_regions(drawn != Color.BLACK, bbox)

    metadata = {
        "bbox": bbox,
        "trunk": trunk,
        "regions": [],
    }

    out = drawn.copy()
    for coords in regions:
        touching_reds = _adjacent_reds(drawn, coords)
        k = len(touching_reds)
        assert k in COUNT_TO_FILL, "each enclosed region must touch 1-4 red controls"
        fill = COUNT_TO_FILL[k]
        out[coords[:, 0], coords[:, 1]] = fill
        metadata["regions"].append(
            {
                "coords": coords,
                "k": int(k),
                "fill": int(fill),
                "touching_reds": touching_reds,
            }
        )

    return out, metadata


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the unique red trunk tree and color each enclosed chamber by how many red controls bound it.

    Input structure:
    - A black background contains only red control pixels.
    - Exactly one red pair shares a row or a column; that aligned pair defines the trunk.
    - Every other red pixel projects orthogonally onto the trunk, and the control pixels together span a local rectangular workspace.

    Rule:
    - Draw the trunk in green between the aligned red endpoints, keeping the red endpoints unchanged.
    - From every other red control, draw a green segment straight to the trunk along the perpendicular axis, again preserving the red pixel itself.
    - Inside the minimal axis-aligned rectangle spanning all red controls, treat the green/red tree plus the rectangle border as walls and find every 4-connected background chamber.
    - Fill each chamber by the number K of distinct red control pixels touching it orthogonally: K=1 -> blue, K=2 -> yellow, K=3 -> orange, K=4 -> pink.

    Key insight:
    - The same red pixels play two roles at once: they define the tree-shaped partition, and their placement on the chamber boundaries encodes which fill color each chamber must receive.
    """
    out, _metadata = _analyze(grid)
    return out


def _sample_spaced(rng: np.random.Generator, lo: int, hi: int, count: int) -> list[int] | None:
    values = np.arange(lo, hi + 1, dtype=int)
    if len(values) < count:
        return None
    for _ in range(200):
        picks = np.sort(rng.choice(values, size=count, replace=False))
        if np.all(np.diff(picks) >= 1):
            return [int(v) for v in picks]
    return None


def _base_region_count(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    return len(_interior_regions(grid != Color.BLACK, bbox))


def _candidate_grid(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GEN_HEIGHT_RANGE))
    width = int(rng.integers(*GEN_WIDTH_RANGE))
    axis = "col" if float(rng.random()) < 0.5 else "row"
    branch_count = int(rng.integers(*GEN_BRANCH_COUNT_RANGE))

    if axis == "col":
        side = str(rng.choice(np.array(["left", "right"])))
        trunk_cols = [
            c
            for c in range(4, width - 4)
            if (c - 1 >= branch_count if side == "left" else width - 2 - c >= branch_count)
        ]
        if not trunk_cols:
            return None
        trunk_col = int(rng.choice(np.array(trunk_cols, dtype=int)))

        span_lo = max(GEN_TRUNK_SPAN_MIN, branch_count + 3)
        span_hi = min(GEN_TRUNK_SPAN_MAX, height - 4)
        if span_lo > span_hi:
            return None
        span = int(rng.integers(span_lo, span_hi + 1))
        top = int(rng.integers(2, height - span - 1))
        bottom = top + span - 1

        branch_rows = _sample_spaced(rng, top + 1, bottom - 1, branch_count)
        if branch_rows is None:
            return None
        available = np.arange(1, trunk_col, dtype=int) if side == "left" else np.arange(trunk_col + 1, width - 1, dtype=int)
        if len(available) < branch_count:
            return None
        branch_cols = [int(v) for v in rng.choice(available, size=branch_count, replace=False)]
        rng.shuffle(branch_cols)

        red_coords = [(top, trunk_col), (bottom, trunk_col)] + list(zip(branch_rows, branch_cols))
    else:
        side = str(rng.choice(np.array(["top", "bottom"])))
        trunk_rows = [
            r
            for r in range(4, height - 4)
            if (r - 1 >= branch_count if side == "top" else height - 2 - r >= branch_count)
        ]
        if not trunk_rows:
            return None
        trunk_row = int(rng.choice(np.array(trunk_rows, dtype=int)))

        span_lo = max(GEN_TRUNK_SPAN_MIN, branch_count + 3)
        span_hi = min(GEN_TRUNK_SPAN_MAX, width - 4)
        if span_lo > span_hi:
            return None
        span = int(rng.integers(span_lo, span_hi + 1))
        left = int(rng.integers(2, width - span - 1))
        right = left + span - 1

        branch_cols = _sample_spaced(rng, left + 1, right - 1, branch_count)
        if branch_cols is None:
            return None
        available = np.arange(1, trunk_row, dtype=int) if side == "top" else np.arange(trunk_row + 1, height - 1, dtype=int)
        if len(available) < branch_count:
            return None
        branch_rows = [int(v) for v in rng.choice(available, size=branch_count, replace=False)]
        rng.shuffle(branch_rows)

        red_coords = [(trunk_row, left), (trunk_row, right)] + list(zip(branch_rows, branch_cols))

    if len(set(red_coords)) != len(red_coords):
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, c in red_coords:
        grid[r, c] = Color.RED
    return grid


def _valid_generated_example(grid: np.ndarray) -> bool:
    red_coords = [(int(r), int(c)) for r, c in np.argwhere(grid == Color.RED)]
    if not all(0 < r < grid.shape[0] - 1 and 0 < c < grid.shape[1] - 1 for r, c in red_coords):
        return False

    try:
        out, metadata = _analyze(grid)
    except AssertionError:
        return False

    ks = sorted(region["k"] for region in metadata["regions"])
    if ks != [1, 2, 3]:
        return False
    if len(metadata["regions"]) != 3:
        return False
    if len({tuple(sorted(region["touching_reds"])) for region in metadata["regions"]}) != 3:
        return False
    if len({region["fill"] for region in metadata["regions"]}) != 3:
        return False

    base_regions = _base_region_count(grid, metadata["bbox"])
    if base_regions >= len(metadata["regions"]):
        return False
    if np.array_equal(out, grid):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _candidate_grid(rng)
        if grid is not None and _valid_generated_example(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
