import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

CORNER_PALETTE = [BLUE, RED, GREEN, YELLOW]
QUADRANT_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
MAX_GENERATION_ATTEMPTS = 500


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"expected color {color} to be present"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quadrant_slices(height: int, width: int, *, middle_to_tl: bool = True) -> list[tuple[slice, slice]]:
    top_h = (height + 1) // 2 if middle_to_tl else height // 2
    left_w = (width + 1) // 2 if middle_to_tl else width // 2
    return [
        (slice(0, top_h), slice(0, left_w)),
        (slice(0, top_h), slice(left_w, width)),
        (slice(top_h, height), slice(0, left_w)),
        (slice(top_h, height), slice(left_w, width)),
    ]


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox_of_color(grid, GREY)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    expected = np.zeros_like(grid, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(grid == GREY, expected), "grey pixels must form exactly one rectangular frame"
    return r0, c0, r1, c1


def _legend_colors(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = frame_bbox
    corners = [(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)]
    H, W = grid.shape
    for r, c in corners:
        assert 0 <= r < H and 0 <= c < W, "corner markers must sit just outside the frame"
    colors = [int(grid[r, c]) for r, c in corners]
    assert len(set(colors)) == 4, "the four corner markers must be pairwise distinct"
    assert all(color in CORNER_PALETTE for color in colors), "corner markers must use the four legend colors"
    return colors


def _dominant_quadrant(counts: list[int], areas: list[int]) -> int:
    winner = None
    for idx in range(4):
        if all(counts[idx] * areas[j] > counts[j] * areas[idx] for j in range(4) if j != idx):
            assert winner is None, "dominant quadrant must be unique"
            winner = idx
    assert winner is not None, "exactly one quadrant must have the highest mask density"
    return winner


def _transform(grid: np.ndarray, *, middle_to_tl: bool = True) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape

    frame_bbox = _find_frame(g)
    r0, c0, r1, c1 = frame_bbox
    legends = _legend_colors(g, frame_bbox)

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = map(int, interior.shape)
    assert inner_h >= 8 and inner_w >= 8, "framed interior must be large enough to define clear quadrant densities"
    assert np.all((interior == BLACK) | (interior == ORANGE)), "interior may only contain background and orange mask pixels"
    assert np.any(interior == ORANGE), "interior must contain mask pixels"

    quadrant_defs = _quadrant_slices(inner_h, inner_w, middle_to_tl=middle_to_tl)
    recolored = interior.copy()
    counts: list[int] = []
    areas: list[int] = []

    for idx, (rows, cols) in enumerate(quadrant_defs):
        quad = interior[rows, cols]
        mask = quad == ORANGE
        counts.append(int(mask.sum()))
        areas.append(int(quad.size))
        block = recolored[rows, cols]
        block[mask] = legends[idx]

    winner = _dominant_quadrant(counts, areas)
    winner_color = int(legends[winner])

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = recolored

    if winner in (0, 1):
        out[r0 - 1, c0 : c1 + 1] = winner_color
    if winner in (2, 3):
        out[r1 + 1, c0 : c1 + 1] = winner_color
    if winner in (0, 2):
        out[r0 : r1 + 1, c0 - 1] = winner_color
    if winner in (1, 3):
        out[r0 : r1 + 1, c1 + 1] = winner_color

    corner_targets = {
        0: (r0 - 1, c0 - 1),
        1: (r0 - 1, c1 + 1),
        2: (r1 + 1, c0 - 1),
        3: (r1 + 1, c1 + 1),
    }
    rr, cc = corner_targets[winner]
    assert 0 <= rr < H and 0 <= cc < W
    out[rr, cc] = winner_color

    return out, {
        "frame_bbox": frame_bbox,
        "legends": legends,
        "counts": counts,
        "areas": areas,
        "winner": winner,
        "winner_name": QUADRANT_NAMES[winner],
        "winner_color": winner_color,
        "interior_shape": (inner_h, inner_w),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the framed orange mask by corner-guided quadrants, then extend the frame outward only toward the densest quadrant.

    Input structure:
    - The grid has a black background, one grey rectangular frame of thickness 1, and four
      single-pixel corner markers placed diagonally outside the frame corners.
    - The framed interior contains only black background and orange mask pixels.
    - The four corner marker colors are distinct legend colors for the top-left, top-right,
      bottom-left, and bottom-right interior quadrants.

    Rule:
    - Find the grey frame and keep the full canvas size.
    - Split the frame interior into four quadrants; when the interior height or width is odd,
      the middle row and/or middle column belong to the top-left side of the split.
    - Recolor each orange mask pixel with the legend color of its quadrant.
    - Count the orange-mask density inside each quadrant and select the unique densest quadrant.
    - Preserve the frame, corner markers, and recolored interior, then add one outer row and/or
      column beside the frame only in the selected quadrant's direction.

    Key insight:
    - The corner pixels first define a quadrant-wise recoloring, and the same quadrant partition
      also groups the mask statistics that choose the single-sided concentric extension direction.
    """
    return _transform(grid)[0]


def _quadrant_cell_lists(height: int, width: int) -> list[list[tuple[int, int]]]:
    cells: list[list[tuple[int, int]]] = []
    for rows, cols in _quadrant_slices(height, width):
        cells.append([(r, c) for r in range(rows.start, rows.stop) for c in range(cols.start, cols.stop)])
    return cells


def _reserved_boundary_cells(height: int, width: int) -> list[set[tuple[int, int]]]:
    reserved = [set() for _ in range(4)]
    top_h = (height + 1) // 2
    left_w = (width + 1) // 2

    if height % 2 == 1:
        mid_r = top_h - 1
        reserved[0].add((mid_r, random.randrange(0, left_w)))
        if left_w < width:
            reserved[1].add((mid_r, random.randrange(left_w, width)))

    if width % 2 == 1:
        mid_c = left_w - 1
        reserved[0].add((random.randrange(0, top_h), mid_c))
        if top_h < height:
            reserved[2].add((random.randrange(top_h, height), mid_c))

    return reserved


def _sample_counts(areas: list[int], reserved: list[set[tuple[int, int]]], winner: int) -> list[int] | None:
    total_area = sum(areas)
    for _ in range(300):
        counts = [0, 0, 0, 0]
        for idx, area in enumerate(areas):
            minimum = max(1, len(reserved[idx]))
            if idx == winner:
                lo = max(minimum + 2, int(np.ceil(area * 0.50)))
                hi = max(lo, int(np.floor(area * 0.82)))
            else:
                lo = minimum
                hi = max(lo, int(np.floor(area * 0.42)))
            counts[idx] = random.randint(lo, hi)

        ordered = sorted(counts)
        if counts[winner] <= ordered[-2]:
            continue
        if counts[winner] < ordered[-2] + 2:
            continue
        if ordered[-2] < 2:
            continue
        if sum(counts) < max(12, int(np.ceil(total_area * 0.24))):
            continue
        if sum(counts) > int(np.floor(total_area * 0.68)):
            continue
        if sum(value >= 2 for value in counts) < 3:
            continue
        if _dominant_quadrant(counts, areas) != winner:
            continue
        return counts
    return None


def _paint_mask(height: int, width: int, counts: list[int], reserved: list[set[tuple[int, int]]]) -> np.ndarray | None:
    interior = np.full((height, width), BLACK, dtype=int)
    cell_lists = _quadrant_cell_lists(height, width)

    for idx, cells in enumerate(cell_lists):
        chosen = set(reserved[idx])
        if len(chosen) > counts[idx]:
            return None
        remaining = [cell for cell in cells if cell not in chosen]
        need = counts[idx] - len(chosen)
        if need > len(remaining):
            return None
        chosen.update(random.sample(remaining, need))
        for r, c in chosen:
            interior[r, c] = ORANGE

    return interior


def _sample_layout(winner: int) -> tuple[int, int, int, int, int, int] | None:
    inner_h = random.randint(8, 12)
    inner_w = random.randint(8, 12)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    top_min = 3 if winner in (0, 1) else 1
    bottom_min = 3 if winner in (2, 3) else 1
    left_min = 3 if winner in (0, 2) else 1
    right_min = 3 if winner in (1, 3) else 1

    H = outer_h + top_min + bottom_min + random.randint(0, 6)
    W = outer_w + left_min + right_min + random.randint(0, 6)
    if max(H, W) > 30:
        return None

    top_max = H - outer_h - bottom_min
    left_max = W - outer_w - right_min
    if top_max < top_min or left_max < left_min:
        return None

    top = random.randint(top_min, top_max)
    left = random.randint(left_min, left_max)
    return H, W, top, left, inner_h, inner_w


def _assemble_input(
    H: int,
    W: int,
    top: int,
    left: int,
    interior: np.ndarray,
    legends: list[int],
) -> np.ndarray:
    inner_h, inner_w = map(int, interior.shape)
    r0, c0 = top, left
    r1, c1 = top + inner_h + 1, left + inner_w + 1

    grid = np.full((H, W), BLACK, dtype=int)
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0 + 1 : r1, c0 + 1 : c1] = interior

    marker_coords = [(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)]
    for (r, c), color in zip(marker_coords, legends):
        grid[r, c] = int(color)

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    out, meta = _transform(grid)
    r0, c0, r1, c1 = meta["frame_bbox"]
    winner = meta["winner"]
    winner_color = meta["winner_color"]
    inner_h, inner_w = meta["interior_shape"]

    assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1] == GREY, grid[r0 : r1 + 1, c0 : c1 + 1] == GREY)
    assert np.array_equal(grid[r0 + 1 : r1, c0 + 1 : c1] == ORANGE, out[r0 + 1 : r1, c0 + 1 : c1] != BLACK)
    assert np.array_equal(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK, out[r0 + 1 : r1, c0 + 1 : c1] == BLACK)
    assert np.any(out != grid), "output must differ from input"
    ordered_counts = sorted(meta["counts"], reverse=True)
    assert ordered_counts[0] > ordered_counts[1], "generator must give one quadrant a strict count maximum"

    if winner in (0, 1):
        assert np.all(out[r0 - 1, c0 : c1 + 1] == winner_color)
    if winner in (2, 3):
        assert np.all(out[r1 + 1, c0 : c1 + 1] == winner_color)
    if winner in (0, 2):
        assert np.all(out[r0 : r1 + 1, c0 - 1] == winner_color)
    if winner in (1, 3):
        assert np.all(out[r0 : r1 + 1, c1 + 1] == winner_color)

    if inner_h % 2 == 1 or inner_w % 2 == 1:
        try:
            wrong = _transform(grid, middle_to_tl=False)[0]
        except AssertionError:
            wrong = None
        assert wrong is None or not np.array_equal(wrong, out), "odd-sized interiors must make the middle-row/column convention matter"

    runner_up = max(
        (idx for idx in range(4) if idx != winner),
        key=lambda idx: meta["counts"][idx] / meta["areas"][idx],
    )
    alt = out.copy()
    alt[r0 - 1, c0 : c1 + 1] = grid[r0 - 1, c0 : c1 + 1]
    alt[r1 + 1, c0 : c1 + 1] = grid[r1 + 1, c0 : c1 + 1]
    alt[r0 : r1 + 1, c0 - 1] = grid[r0 : r1 + 1, c0 - 1]
    alt[r0 : r1 + 1, c1 + 1] = grid[r0 : r1 + 1, c1 + 1]
    runner_color = int(meta["legends"][runner_up])
    if runner_up in (0, 1):
        alt[r0 - 1, c0 : c1 + 1] = runner_color
    if runner_up in (2, 3):
        alt[r1 + 1, c0 : c1 + 1] = runner_color
    if runner_up in (0, 2):
        alt[r0 : r1 + 1, c0 - 1] = runner_color
    if runner_up in (1, 3):
        alt[r0 : r1 + 1, c1 + 1] = runner_color
    corner_targets = {
        0: (r0 - 1, c0 - 1),
        1: (r0 - 1, c1 + 1),
        2: (r1 + 1, c0 - 1),
        3: (r1 + 1, c1 + 1),
    }
    rr, cc = corner_targets[runner_up]
    alt[rr, cc] = runner_color
    assert not np.array_equal(alt, out), "the selected extension direction must affect the visible output"

    return True


def _gen_attempt() -> np.ndarray | None:
    winner = random.randrange(4)
    layout = _sample_layout(winner)
    if layout is None:
        return None

    H, W, top, left, inner_h, inner_w = layout
    legends = random.sample(CORNER_PALETTE, 4)
    reserved = _reserved_boundary_cells(inner_h, inner_w)
    areas = [len(cells) for cells in _quadrant_cell_lists(inner_h, inner_w)]
    counts = _sample_counts(areas, reserved, winner)
    if counts is None:
        return None

    interior = _paint_mask(inner_h, inner_w, counts, reserved)
    if interior is None:
        return None

    grid = _assemble_input(H, W, top, left, interior, legends)
    out, meta = _transform(grid)
    if meta["winner"] != winner:
        return None
    if meta["counts"] != counts:
        return None
    if not _semantic_checks(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
