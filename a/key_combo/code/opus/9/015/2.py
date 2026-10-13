from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
import random

import numpy as np

try:
    from arc_utils import components_mask, neighbors4
except ImportError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    components_mask = _ARC_UTILS.components_mask
    neighbors4 = _ARC_UTILS.neighbors4


BG = 0
GREY = 5
LINE_COLORS = (1, 2, 3, 4)
REGION_DIMS = (
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (5, 2),
)
CRITICAL_DIMS = (
    (2, 4),
    (2, 5),
    (3, 4),
    (4, 2),
    (5, 2),
)
ROW_PAIRINGS = (((0, 2), (1, 3)), ((0, 3), (1, 2)))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _draw_paired_lines(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    marker_colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)]
    h, w = out.shape

    for r in range(h):
        for color in marker_colors:
            cols = np.flatnonzero(grid[r] == color)
            assert len(cols) <= 2
            if len(cols) != 2:
                continue
            c0, c1 = map(int, sorted(cols))
            for c in range(c0 + 1, c1):
                if out[r, c] == BG:
                    out[r, c] = int(color)

    for c in range(w):
        for color in marker_colors:
            rows = np.flatnonzero(grid[:, c] == color)
            assert len(rows) <= 2
            if len(rows) != 2:
                continue
            r0, r1 = map(int, sorted(rows))
            for r in range(r0 + 1, r1):
                if out[r, c] == BG:
                    out[r, c] = int(color)

    return out


def _enclosed_regions(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    regions: list[np.ndarray] = []
    for coords in components_mask(grid == BG, connectivity=4):
        coords = coords.astype(int)
        if np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        ):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        enclosed = True
        for r_raw, c_raw in coords:
            r, c = int(r_raw), int(c_raw)
            for nr, nc in neighbors4(r, c):
                if not _in_bounds((h, w), nr, nc):
                    enclosed = False
                    break
                if not mask[nr, nc] and int(grid[nr, nc]) != GREY:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            regions.append(mask)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw aligned marker connections, then flood enclosed fence regions with the color of the longest line segment passing through each region.

    Input structure:
    - The grid uses black as background and contains several closed grey fence regions with black interiors.
    - Colored marker pixels (blue/red/green/yellow) lie outside every fenced region.
    - Any non-empty marker row contains exactly two same-colored markers, and any non-empty marker column contains exactly two same-colored markers.

    Rule:
    - First draw all horizontal connections: for every row with two same-colored markers, color the black cells strictly between them with that marker color.
    - Then draw all vertical connections the same way, but only onto still-black cells, so earlier horizontal pixels keep priority at crossings.
    - For each grey-enclosed region, count the colored line pixels now visible inside the region; if any are present, flood that region's remaining black interior cells with the color that has the largest in-region line count.

    Key insight:
    - The fences create vote boxes: lines do not repaint the fences, but their visible in-region lengths determine which color is propagated to the rest of that enclosed area.
    """
    g = np.asarray(grid, dtype=int)
    out = _draw_paired_lines(g)

    for region in _enclosed_regions(g):
        counts = Counter(int(v) for v in out[region] if int(v) not in (BG, GREY))
        if not counts:
            continue
        best = max(counts.values())
        winners = [color for color, count in counts.items() if count == best]
        assert len(winners) == 1
        fill_color = int(winners[0])
        out[region & (out == BG)] = fill_color

    return out


def _sample_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    candidates = list(range(lo, hi + 1))
    for _ in range(300):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _region_ok_with_lines(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    *,
    avoid_rows: set[int],
    avoid_cols: set[int],
) -> bool:
    row_lo, row_hi = r0, r0 + inner_h + 1
    col_lo, col_hi = c0, c0 + inner_w + 1
    if any(row_lo <= row <= row_hi for row in avoid_rows):
        return False
    if any(col_lo <= col <= col_hi for col in avoid_cols):
        return False
    return True


def _place_region(
    occupied: np.ndarray,
    inner_h: int,
    inner_w: int,
    *,
    must_row: int | None = None,
    must_col: int | None = None,
    row_window: tuple[int, int] | None = None,
    col_window: tuple[int, int] | None = None,
    avoid_rows: set[int] | None = None,
    avoid_cols: set[int] | None = None,
) -> dict[str, int] | None:
    h, w = occupied.shape
    avoid_rows = set() if avoid_rows is None else set(map(int, avoid_rows))
    avoid_cols = set() if avoid_cols is None else set(map(int, avoid_cols))
    outer_h, outer_w = inner_h + 2, inner_w + 2

    row_starts = list(range(3, h - outer_h - 2))
    col_starts = list(range(3, w - outer_w - 2))
    if must_row is not None:
        row_starts = [
            r0 for r0 in row_starts if r0 + 1 <= int(must_row) <= r0 + inner_h
        ]
    if must_col is not None:
        col_starts = [
            c0 for c0 in col_starts if c0 + 1 <= int(must_col) <= c0 + inner_w
        ]
    random.shuffle(row_starts)
    random.shuffle(col_starts)

    candidates = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if row_window is not None:
            row_lo, row_hi = map(int, row_window)
            if r0 + 1 < row_lo or r0 + inner_h > row_hi:
                continue
        if col_window is not None:
            col_lo, col_hi = map(int, col_window)
            if c0 + 1 < col_lo or c0 + inner_w > col_hi:
                continue
        if not _region_ok_with_lines(
            r0, c0, inner_h, inner_w, avoid_rows=avoid_rows, avoid_cols=avoid_cols
        ):
            continue
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        rr0, rr1 = max(0, r0 - 1), min(h, r1 + 2)
        cc0, cc1 = max(0, c0 - 1), min(w, c1 + 2)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return {"r0": r0, "c0": c0, "inner_h": inner_h, "inner_w": inner_w}
    return None


def _draw_region(grid: np.ndarray, region: dict[str, int]) -> None:
    r0 = int(region["r0"])
    c0 = int(region["c0"])
    inner_h = int(region["inner_h"])
    inner_w = int(region["inner_w"])
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _marker_grid(
    h: int,
    w: int,
    row_a: tuple[int, int],
    col_a: tuple[int, int],
    color_a: int,
    row_b: tuple[int, int],
    col_b: tuple[int, int],
    color_b: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r in row_a:
        for c in col_a:
            grid[int(r), int(c)] = int(color_a)
    for r in row_b:
        for c in col_b:
            assert grid[int(r), int(c)] == BG
            grid[int(r), int(c)] = int(color_b)
    return grid


def _place_region_in_span(
    occupied: np.ndarray,
    outer_rows: tuple[int, int],
    outer_cols: tuple[int, int],
    *,
    must_row: int | None = None,
    must_col: int | None = None,
    dims: tuple[tuple[int, int], ...] = REGION_DIMS,
) -> dict[str, int] | None:
    h, w = occupied.shape
    row_lo, row_hi = map(int, outer_rows)
    col_lo, col_hi = map(int, outer_cols)
    shuffled_dims = list(dims)
    random.shuffle(shuffled_dims)

    for inner_h, inner_w in shuffled_dims:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h > row_hi - row_lo + 1 or outer_w > col_hi - col_lo + 1:
            continue
        row_starts = list(range(row_lo, row_hi - outer_h + 2))
        col_starts = list(range(col_lo, col_hi - outer_w + 2))
        if must_row is not None:
            row_starts = [
                r0 for r0 in row_starts if r0 + 1 <= int(must_row) <= r0 + inner_h
            ]
        if must_col is not None:
            col_starts = [
                c0 for c0 in col_starts if c0 + 1 <= int(must_col) <= c0 + inner_w
            ]
        candidates = [(r0, c0) for r0 in row_starts for c0 in col_starts]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            r1 = r0 + inner_h + 1
            c1 = c0 + inner_w + 1
            if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
                continue
            rr0, rr1 = r0 - 1, r1 + 2
            cc0, cc1 = c0 - 1, c1 + 2
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            return {"r0": r0, "c0": c0, "inner_h": inner_h, "inner_w": inner_w}
    return None


def _build_regions(
    h: int,
    w: int,
    all_rows: set[int],
    all_cols: set[int],
    row_spans: dict[int, tuple[int, int]],
    col_spans: dict[int, tuple[int, int]],
    critical_row: int,
    critical_col: int,
    h_only_row: int,
    v_only_col: int,
    extra_specs: list[tuple[str, int]],
    region_count: int,
) -> list[dict[str, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    regions: list[dict[str, int]] = []

    critical = _place_region(
        occupied,
        *random.choice(CRITICAL_DIMS),
        must_row=critical_row,
        must_col=critical_col,
        row_window=col_spans[critical_col],
        col_window=row_spans[critical_row],
        avoid_rows=all_rows - {critical_row},
        avoid_cols=all_cols - {critical_col},
    )
    if critical is None:
        return None
    regions.append(critical)

    horizontal = _place_region(
        occupied,
        *random.choice(REGION_DIMS),
        must_row=h_only_row,
        col_window=row_spans[h_only_row],
        avoid_rows=all_rows - {h_only_row},
        avoid_cols=all_cols,
    )
    if horizontal is None:
        return None
    regions.append(horizontal)

    vertical = _place_region(
        occupied,
        *random.choice(REGION_DIMS),
        must_col=v_only_col,
        row_window=col_spans[v_only_col],
        avoid_rows=all_rows,
        avoid_cols=all_cols - {v_only_col},
    )
    if vertical is None:
        return None
    regions.append(vertical)

    desired_extra = max(0, region_count - 4)
    random.shuffle(extra_specs)
    for orient, axis in extra_specs[:desired_extra]:
        if orient == "row":
            region = _place_region(
                occupied,
                *random.choice(REGION_DIMS),
                must_row=axis,
                col_window=row_spans[axis],
                avoid_rows=all_rows - {axis},
                avoid_cols=all_cols,
            )
        else:
            region = _place_region(
                occupied,
                *random.choice(REGION_DIMS),
                must_col=axis,
                row_window=col_spans[axis],
                avoid_rows=all_rows,
                avoid_cols=all_cols - {axis},
            )
        if region is None:
            return None
        regions.append(region)

    while len(regions) < region_count:
        region = _place_region(
            occupied,
            *random.choice(REGION_DIMS),
            avoid_rows=all_rows,
            avoid_cols=all_cols,
        )
        if region is None:
            return None
        regions.append(region)

    return regions


def _validate_generated(grid: np.ndarray) -> None:
    h, w = grid.shape
    assert h <= 30 and w <= 30
    markers = np.argwhere(np.isin(grid, LINE_COLORS))
    assert 6 <= len(markers) <= 10

    region_masks = _enclosed_regions(grid)
    assert 3 <= len(region_masks) <= 5
    union = np.zeros_like(grid, dtype=bool)
    for mask in region_masks:
        union |= mask

    for r_raw, c_raw in markers:
        r, c = int(r_raw), int(c_raw)
        assert not union[r, c]

    for r in range(h):
        cols = np.flatnonzero(np.isin(grid[r], LINE_COLORS))
        assert len(cols) in (0, 2)
        if len(cols) == 2:
            assert int(grid[r, cols[0]]) == int(grid[r, cols[1]])

    for c in range(w):
        rows = np.flatnonzero(np.isin(grid[:, c], LINE_COLORS))
        assert len(rows) in (0, 2)
        if len(rows) == 2:
            assert int(grid[rows[0], c]) == int(grid[rows[1], c])

    lined = _draw_paired_lines(grid)
    multi_color_region = False
    filled_regions = 0
    unchanged_regions = 0
    for mask in region_masks:
        counts = Counter(int(v) for v in lined[mask] if int(v) not in (BG, GREY))
        if not counts:
            unchanged_regions += 1
            continue
        filled_regions += 1
        top = max(counts.values())
        assert list(counts.values()).count(top) == 1
        if len(counts) >= 2:
            multi_color_region = True

    assert filled_regions >= 2
    assert multi_color_region
    assert unchanged_regions >= 1

    out = main(grid)
    assert not np.array_equal(out, grid)
    for mask in region_masks:
        counts = Counter(int(v) for v in lined[mask] if int(v) not in (BG, GREY))
        if not counts:
            assert np.all(out[mask] == BG)
            continue
        winner = max(counts, key=counts.get)
        assert np.all(out[mask & (lined == BG)] == int(winner))


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(22, 28)
        w = random.randint(22, 28)

        rows = _sample_spaced_positions(4, h - 5, 4, 4)
        cols = _sample_spaced_positions(4, w - 5, 4, 4)
        if rows is None or cols is None:
            continue

        row_a = (int(rows[0]), int(rows[2]))
        row_b = (int(rows[1]), int(rows[3]))
        col_a = (int(cols[0]), int(cols[2]))
        col_b = (int(cols[1]), int(cols[3]))

        color_a = random.choice(LINE_COLORS)
        color_b = random.choice([c for c in LINE_COLORS if c != color_a])

        grid = _marker_grid(h, w, row_a, col_a, color_a, row_b, col_b, color_b)
        occupied = np.zeros((h, w), dtype=bool)

        critical = _place_region_in_span(
            occupied,
            (row_b[0] + 1, row_b[1] - 1),
            (col_a[0] + 1, col_a[1] - 1),
            must_row=row_a[1],
            must_col=col_b[0],
            dims=CRITICAL_DIMS,
        )
        if critical is None:
            continue

        horizontal = _place_region_in_span(
            occupied,
            (1, row_b[0] - 1),
            (col_a[0] + 1, col_a[1] - 1),
            must_row=row_a[0],
        )
        if horizontal is None:
            continue

        vertical = _place_region_in_span(
            occupied,
            (row_b[0] + 1, row_b[1] - 1),
            (col_a[1] + 1, w - 3),
            must_col=col_b[1],
        )
        if vertical is None:
            continue

        passive = _place_region_in_span(
            occupied,
            (row_a[1] + 1, h - 2),
            (col_b[1] + 1, w - 2),
        )
        if passive is None:
            continue

        regions = [critical, horizontal, vertical, passive]
        if random.random() < 0.45:
            extra = _place_region_in_span(
                occupied,
                (row_a[0] + 1, row_a[1] - 1),
                (1, col_a[0] - 1),
            )
            if extra is not None:
                regions.append(extra)

        for region in regions:
            _draw_region(grid, region)

        _validate_generated(grid)
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
