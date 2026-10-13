from __future__ import annotations

from collections import deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))

    if set(map(tuple, coords.tolist())) != expected:
        return None
    return r0, c0, r1, c1


def _region_for_fragment(
    coords: np.ndarray,
    regions: list[dict[str, int]],
) -> dict[str, int]:
    matches = []
    for region in regions:
        if np.all(
            (region["r0"] < coords[:, 0])
            & (coords[:, 0] < region["r1"])
            & (region["c0"] < coords[:, 1])
            & (coords[:, 1] < region["c1"])
        ):
            matches.append(region)
    assert len(matches) == 1, "each fragment must lie inside exactly one frame"
    return matches[0]


def _region_color_at(
    regions: list[dict[str, int]],
    r: int,
    c: int,
) -> int | None:
    for region in regions:
        if region["r0"] < r < region["r1"] and region["c0"] < c < region["c1"]:
            return int(region["color"])
    return None


def _integer_centroid(value: float) -> int:
    rounded = int(round(value))
    assert abs(value - rounded) < 1e-6, "bridge axis must pass through an integer grid line"
    return rounded


def _draw_horizontal_bridge(
    out: np.ndarray,
    bg: int,
    regions: list[dict[str, int]],
    left: dict,
    right: dict,
    row: int,
) -> None:
    left_on_row = left["coords"][left["coords"][:, 0] == row]
    right_on_row = right["coords"][right["coords"][:, 0] == row]
    assert left_on_row.size and right_on_row.size, "fragment must touch its bridge row"

    c0 = int(left_on_row[:, 1].max()) + 1
    c1 = int(right_on_row[:, 1].min()) - 1
    for c in range(c0, c1 + 1):
        if out[row, c] != bg:
            continue
        color = _region_color_at(regions, row, c)
        if color is not None:
            out[row, c] = color


def _draw_vertical_bridge(
    out: np.ndarray,
    bg: int,
    regions: list[dict[str, int]],
    top: dict,
    bottom: dict,
    col: int,
) -> None:
    top_on_col = top["coords"][top["coords"][:, 1] == col]
    bottom_on_col = bottom["coords"][bottom["coords"][:, 1] == col]
    assert top_on_col.size and bottom_on_col.size, "fragment must touch its bridge column"

    r0 = int(top_on_col[:, 0].max()) + 1
    r1 = int(bottom_on_col[:, 0].min()) - 1
    for r in range(r0, r1 + 1):
        if out[r, col] != bg:
            continue
        color = _region_color_at(regions, r, col)
        if color is not None:
            out[r, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of equal-colored fragments with bridge segments colored by their enclosing frames.

    Input structure:
    - The background is the dominant black color.
    - Distinct foreground colors form one-pixel-thick rectangular frames whose
      interiors are separate regions.
    - Other colors form small 4-connected monochrome fragments of area 2-4;
      each fragment color appears as exactly two components in two different
      framed regions.
    - The two fragments of each color have aligned centroid rows or aligned
      centroid columns.

    Rule:
    - Keep all frames and fragments unchanged.
    - For each equal-colored fragment pair, use the aligned centroid row or
      column as the bridge axis.
    - Draw the straight bridge only on background cells between the nearest
      edge pixels of the two fragments.
    - Each bridge cell takes the border color of the rectangular region that
      contains that cell, so a bridge crossing from one region to another is
      split into differently colored segments at the region boundary.

    Key insight:
    The equal fragment colors identify what must be paired, while the enclosing
    frame colors, not the fragment colors, determine how each bridge is painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    regions: list[dict[str, int]] = []
    frame_mask = np.zeros_like(g, dtype=bool)
    non_bg_colors = [int(c) for c in np.unique(g) if int(c) != bg]

    for color in non_bg_colors:
        for coords in _components(g == color):
            frame_bbox = _rect_frame_bbox(coords)
            if frame_bbox is None:
                continue
            r0, c0, r1, c1 = frame_bbox
            regions.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
            frame_mask[coords[:, 0], coords[:, 1]] = True

    assert regions, "expected at least one rectangular frame"
    frame_colors = {int(region["color"]) for region in regions}
    assert len(frame_colors) == len(regions), "each frame must have a distinct color"

    fragments_by_color: dict[int, list[dict]] = {}
    for color in non_bg_colors:
        if color in frame_colors:
            continue
        mask = (g == color) & ~frame_mask
        for coords in _components(mask):
            area = int(coords.shape[0])
            assert 2 <= area <= 4, "fragments must have area 2-4"
            region = _region_for_fragment(coords, regions)
            fragments_by_color.setdefault(color, []).append(
                {
                    "color": color,
                    "coords": coords,
                    "region": region,
                    "centroid": coords.mean(axis=0),
                }
            )

    out = g.copy()
    assert fragments_by_color, "expected paired fragments"
    for fragments in fragments_by_color.values():
        assert len(fragments) == 2, "each fragment color must appear exactly twice"
        a, b = fragments
        assert a["region"] is not b["region"], "paired fragments must be in different regions"

        ar, ac = map(float, a["centroid"])
        br, bc = map(float, b["centroid"])
        row_aligned = abs(ar - br) < 1e-6
        col_aligned = abs(ac - bc) < 1e-6
        assert row_aligned or col_aligned, "paired fragments must align by row or column"

        if row_aligned and (not col_aligned or abs(ac - bc) >= abs(ar - br)):
            row = _integer_centroid(ar)
            left, right = (a, b) if ac < bc else (b, a)
            _draw_horizontal_bridge(out, bg, regions, left, right, row)
        else:
            col = _integer_centroid(ac)
            top, bottom = (a, b) if ar < br else (b, a)
            _draw_vertical_bridge(out, bg, regions, top, bottom, col)

    return out


# --------------- generator ---------------

BG = 0
MAX_ATTEMPTS = 300
COLOR_POOL = list(range(1, 10))

H_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((-1, 0), (0, 0), (0, 1), (1, 0)),
)
V_SHAPES = tuple(tuple((dc, dr) for dr, dc in shape) for shape in H_SHAPES)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _shape_bounds(shape: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [dr for dr, _dc in shape]
    cols = [dc for _dr, dc in shape]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor_r: int,
    anchor_c: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> bool:
    cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
    height, width = grid.shape
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if grid[r, c] != BG or (r, c) in occupied:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied and (nr, nc) not in cells:
                    return False
    for r, c in cells:
        grid[r, c] = color
        occupied.add((r, c))
    return True


def _separated_values(values: list[int], count: int, min_gap: int) -> list[int] | None:
    values = values[:]
    random.shuffle(values)
    for first in values:
        chosen = [first]
        for value in values:
            if all(abs(value - old) >= min_gap for old in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return chosen
    return None


def _place_horizontal_pair(
    grid: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    left_i: int,
    right_i: int,
    row: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> bool:
    left_shape = random.choice(H_SHAPES)
    right_shape = random.choice(H_SHAPES)
    lr0, lc0, lr1, lc1 = regions[left_i]
    rr0, rc0, rr1, rc1 = regions[right_i]

    for _ in range(40):
        lmin_r, lmin_c, lmax_r, lmax_c = _shape_bounds(left_shape)
        rmin_r, rmin_c, rmax_r, rmax_c = _shape_bounds(right_shape)
        if not (lr0 < row + lmin_r and row + lmax_r < lr1):
            continue
        if not (rr0 < row + rmin_r and row + rmax_r < rr1):
            continue

        left_lo = lc0 + 1 - lmin_c
        left_hi = lc1 - 2 - lmax_c
        right_lo = rc0 + 3 - rmin_c
        right_hi = rc1 - 1 - rmax_c
        if left_lo > left_hi or right_lo > right_hi:
            continue

        left_c = random.randint(left_lo, left_hi)
        right_c = random.randint(right_lo, right_hi)
        snapshot = set(occupied)
        if _paint_shape(grid, left_shape, row, left_c, color, occupied) and _paint_shape(
            grid, right_shape, row, right_c, color, occupied
        ):
            return True
        for r, c in occupied - snapshot:
            grid[r, c] = BG
        occupied.clear()
        occupied.update(snapshot)
    return False


def _place_vertical_pair(
    grid: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    top_i: int,
    bottom_i: int,
    col: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> bool:
    top_shape = random.choice(V_SHAPES)
    bottom_shape = random.choice(V_SHAPES)
    tr0, tc0, tr1, tc1 = regions[top_i]
    br0, bc0, br1, bc1 = regions[bottom_i]

    for _ in range(40):
        tmin_r, tmin_c, tmax_r, tmax_c = _shape_bounds(top_shape)
        bmin_r, bmin_c, bmax_r, bmax_c = _shape_bounds(bottom_shape)
        if not (tc0 < col + tmin_c and col + tmax_c < tc1):
            continue
        if not (bc0 < col + bmin_c and col + bmax_c < bc1):
            continue

        top_lo = tr0 + 1 - tmin_r
        top_hi = tr1 - 2 - tmax_r
        bottom_lo = br0 + 3 - bmin_r
        bottom_hi = br1 - 1 - bmax_r
        if top_lo > top_hi or bottom_lo > bottom_hi:
            continue

        top_r = random.randint(top_lo, top_hi)
        bottom_r = random.randint(bottom_lo, bottom_hi)
        snapshot = set(occupied)
        if _paint_shape(grid, top_shape, top_r, col, color, occupied) and _paint_shape(
            grid, bottom_shape, bottom_r, col, color, occupied
        ):
            return True
        for r, c in occupied - snapshot:
            grid[r, c] = BG
        occupied.clear()
        occupied.update(snapshot)
    return False


def _gen_attempt() -> np.ndarray | None:
    direction = random.choice(("horizontal", "vertical"))
    region_count = random.choice((2, 3))
    pair_count = region_count - 1
    colors = random.sample(COLOR_POOL, region_count + pair_count)
    frame_colors = colors[:region_count]
    fragment_colors = colors[region_count:]

    if direction == "horizontal":
        outer_h = random.randint(11, 14) if region_count == 3 else random.randint(7, 12)
        outer_ws = [random.randint(7, 8) for _ in range(region_count)]
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        height = top_margin + outer_h + bottom_margin
        width = left_margin + sum(outer_ws) + right_margin
        if height > 30 or width > 30:
            return None

        grid = np.full((height, width), BG, dtype=int)
        regions = []
        c = left_margin
        for i, outer_w in enumerate(outer_ws):
            r0, c0 = top_margin, c
            r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
            _draw_frame(grid, r0, c0, r1, c1, frame_colors[i])
            regions.append((r0, c0, r1, c1))
            c = c1 + 1

        row_candidates = list(range(top_margin + 2, top_margin + outer_h - 2))
        rows = _separated_values(row_candidates, pair_count, 4)
        if rows is None:
            return None

        occupied: set[tuple[int, int]] = set()
        for i, color in enumerate(fragment_colors):
            if not _place_horizontal_pair(grid, regions, i, i + 1, rows[i], color, occupied):
                return None
    else:
        outer_w = random.randint(11, 14) if region_count == 3 else random.randint(7, 12)
        outer_hs = [random.randint(7, 8) for _ in range(region_count)]
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        height = top_margin + sum(outer_hs) + bottom_margin
        width = left_margin + outer_w + right_margin
        if height > 30 or width > 30:
            return None

        grid = np.full((height, width), BG, dtype=int)
        regions = []
        r = top_margin
        for i, outer_h in enumerate(outer_hs):
            r0, c0 = r, left_margin
            r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
            _draw_frame(grid, r0, c0, r1, c1, frame_colors[i])
            regions.append((r0, c0, r1, c1))
            r = r1 + 1

        col_candidates = list(range(left_margin + 2, left_margin + outer_w - 2))
        cols = _separated_values(col_candidates, pair_count, 4)
        if cols is None:
            return None

        occupied = set()
        for i, color in enumerate(fragment_colors):
            if not _place_vertical_pair(grid, regions, i, i + 1, cols[i], color, occupied):
                return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bridge-fragment input")
