import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(x) for x in np.unique(grid))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _gap_step(
    gap: tuple[int, int], rect: tuple[int, int, int, int]
) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = rect
    if r == r0:
        return -1, 0
    if r == r1:
        return 1, 0
    if c == c0:
        return 0, -1
    if c == c1:
        return 0, 1
    raise AssertionError("gap is not on the frame border")


def _inside_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _find_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], tuple[int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int], tuple[int, int]]] = []
    for color in colors_present(grid, exclude={bg}):
        mask = grid == color
        rect = bbox(mask)
        if rect is None:
            continue
        r0, c0, r1, c1 = rect
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _border_cells(rect)
        present = {(int(r), int(c)) for r, c in np.argwhere(mask)}
        border_set = set(border)
        if not present <= border_set:
            continue
        missing = [cell for cell in border if cell not in present]
        if len(missing) != 1:
            continue
        gap = missing[0]
        if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        if len(present) != len(border) - 1:
            continue
        candidates.append((color, rect, gap))
    assert len(candidates) == 1, "expected a unique incomplete rectangular frame"
    return candidates[0]


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    frame_color, rect, gap = _find_frame(g, bg)
    inside = _inside_mask(g.shape, rect)

    other_colors = colors_present(g, exclude={bg, frame_color})
    assert len(other_colors) == 2, "expected exactly two scatter colors"

    leak_candidates: list[int] = []
    mixed_candidates: list[int] = []
    color_stats: dict[int, dict[str, int | bool]] = {}
    for color in other_colors:
        mask = g == color
        in_count = int(np.sum(mask & inside))
        out_count = int(np.sum(mask & ~inside))
        color_stats[color] = {
            "inside": in_count,
            "outside": out_count,
            "total": in_count + out_count,
        }
        if in_count > 0 and out_count == 0:
            leak_candidates.append(color)
        if in_count > 0 and out_count > 0:
            mixed_candidates.append(color)

    assert len(leak_candidates) == 1, "expected a unique fully-inside color"
    assert len(mixed_candidates) == 1, "expected a unique inside/outside distractor color"

    leak_color = leak_candidates[0]
    distractor_color = mixed_candidates[0]
    return {
        "bg": bg,
        "frame_color": frame_color,
        "rect": rect,
        "gap": gap,
        "leak_color": leak_color,
        "distractor_color": distractor_color,
        "inside_mask": inside,
        "color_stats": color_stats,
    }


def _render_output(shape: tuple[int, int], info: dict) -> np.ndarray:
    bg = int(info["bg"])
    frame_color = int(info["frame_color"])
    leak_color = int(info["leak_color"])
    rect = info["rect"]
    gap = info["gap"]
    r0, c0, r1, c1 = rect

    out = np.full(shape, bg, dtype=int)
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color
    out[r0 + 1 : r1, c0 + 1 : c1] = leak_color

    dr, dc = _gap_step(gap, rect)
    r, c = gap
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        out[r, c] = leak_color
        r += dr
        c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the broken rectangle in mind, flood its interior with the fully-inside color, and leak that color outward through the gap.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms an axis-aligned rectangular frame with exactly one missing non-corner border cell.
    - Two other scatter colors are present: one color appears only inside the completed frame, while the other appears both inside and outside it.

    Rule:
    - Treat the frame as the full rectangle implied by its bounding box, even though one border cell is missing.
    - Choose the unique color whose every pixel lies inside that completed rectangle.
    - Fill the entire interior of the rectangle with that color, paint the missing border cell with that color, and extend a straight ray from the gap outward along that side's normal to the grid edge.
    - Keep all other frame cells in the frame color and erase every scatter pixel outside the frame to the background.

    Key insight:
    - The gap is not an opening for connectivity-based inside/outside detection; the rectangle must be mentally repaired first, and the deciding cue is complete containment rather than pixel count.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid).shape, info)


GRID_ROWS = (15, 24)
GRID_COLS = (15, 24)
FRAME_HEIGHTS = (5, 10)
FRAME_WIDTHS = (5, 10)
MIN_MARGIN = 1
MAX_ATTEMPTS = 400
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(1, 10))


def _randint(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _sample_distinct(
    rng: np.random.Generator, cells: list[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    assert 0 <= count <= len(cells)
    if count == 0:
        return []
    indices = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in np.atleast_1d(indices)]


def _gap_from_side(
    rng: np.random.Generator, rect: tuple[int, int, int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return r0, int(rng.integers(c0 + 1, c1))
    if side == "bottom":
        return r1, int(rng.integers(c0 + 1, c1))
    if side == "left":
        return int(rng.integers(r0 + 1, r1)), c0
    if side == "right":
        return int(rng.integers(r0 + 1, r1)), c1
    raise AssertionError("unknown side")


def _outside_cells(
    shape: tuple[int, int], rect: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if not (r0 <= r <= r1 and c0 <= c <= c1):
                cells.append((r, c))
    return cells


def _ray_cells(
    shape: tuple[int, int], rect: tuple[int, int, int, int], gap: tuple[int, int]
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    dr, dc = _gap_step(gap, rect)
    r, c = gap
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.add((r, c))
        r += dr
        c += dc
    return cells


def _count_mode_sizes(
    rng: np.random.Generator, interior_area: int, outside_area: int
) -> tuple[int, int, int]:
    mode = int(rng.integers(0, 3))
    leak_max = min(interior_area - 1, max(2, interior_area // 3))
    if mode == 0:
        leak_count = _randint(rng, (1, max(1, leak_max // 2)))
        distractor_inside = _randint(rng, (1, max(1, min(interior_area - leak_count, leak_count + 2))))
        distractor_outside = _randint(rng, (max(2, leak_count + 1), min(outside_area, leak_count + 8)))
    elif mode == 1:
        leak_count = _randint(rng, (min(3, interior_area - 1), leak_max))
        distractor_inside = _randint(rng, (1, max(1, min(interior_area - leak_count, leak_count // 2 + 1))))
        distractor_outside = _randint(rng, (1, max(1, min(outside_area, leak_count // 2 + 1))))
        if distractor_inside + distractor_outside >= leak_count:
            leak_count = min(interior_area - 1, distractor_inside + distractor_outside + 1)
    else:
        leak_count = _randint(rng, (1, leak_max))
        distractor_inside = _randint(rng, (min(2, interior_area - leak_count), max(1, min(interior_area - leak_count, leak_count + 3))))
        distractor_outside = _randint(rng, (1, min(outside_area, max(2, leak_count + 3))))
    assert leak_count >= 1
    assert distractor_inside >= 1
    assert distractor_outside >= 1
    assert leak_count + distractor_inside <= interior_area
    return leak_count, distractor_inside, distractor_outside


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, GRID_ROWS)
    w = _randint(rng, GRID_COLS)
    frame_h = _randint(rng, FRAME_HEIGHTS)
    frame_w = _randint(rng, FRAME_WIDTHS)
    if frame_h >= h - 2 * MIN_MARGIN or frame_w >= w - 2 * MIN_MARGIN:
        return None

    r0 = int(rng.integers(MIN_MARGIN, h - frame_h - MIN_MARGIN + 1))
    c0 = int(rng.integers(MIN_MARGIN, w - frame_w - MIN_MARGIN + 1))
    rect = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    side = str(rng.choice(SIDES))
    gap = _gap_from_side(rng, rect, side)
    ray = _ray_cells((h, w), rect, gap)
    if len(ray) < 2:
        return None

    bg = 0
    frame_color, leak_color, distractor_color = [int(x) for x in rng.choice(PALETTE, size=3, replace=False)]
    grid = np.full((h, w), bg, dtype=int)

    for r, c in _border_cells(rect):
        if (r, c) != gap:
            grid[r, c] = frame_color

    inside_mask = _inside_mask(grid.shape, rect)
    inside_cells = [(int(r), int(c)) for r, c in np.argwhere(inside_mask)]
    outside_cells = [cell for cell in _outside_cells(grid.shape, rect) if cell not in ray]
    if len(inside_cells) < 3 or len(outside_cells) < 2:
        return None

    leak_count, distractor_inside_count, distractor_outside_count = _count_mode_sizes(
        rng, len(inside_cells), len(outside_cells)
    )

    rng.shuffle(inside_cells)
    leak_cells = inside_cells[:leak_count]
    distractor_inside_cells = inside_cells[leak_count : leak_count + distractor_inside_count]
    distractor_outside_cells = _sample_distinct(rng, outside_cells, distractor_outside_count)

    for r, c in leak_cells:
        grid[r, c] = leak_color
    for r, c in distractor_inside_cells:
        grid[r, c] = distractor_color
    for r, c in distractor_outside_cells:
        grid[r, c] = distractor_color

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    info = _analyze_input(grid)
    out = main(grid)
    expected = _render_output(grid.shape, info)
    assert np.array_equal(out, expected), "main does not match the puzzle semantics"

    bg = int(info["bg"])
    rect = info["rect"]
    gap = info["gap"]
    leak_color = int(info["leak_color"])
    distractor_color = int(info["distractor_color"])
    inside = info["inside_mask"]

    leak_mask = grid == leak_color
    distractor_mask = grid == distractor_color
    assert int(np.sum(leak_mask)) >= 1
    assert int(np.sum(distractor_mask & inside)) >= 1
    assert int(np.sum(distractor_mask & ~inside)) >= 1
    assert np.all(grid[inside & leak_mask] == leak_color)
    assert np.all(grid[inside & distractor_mask] == distractor_color)
    assert np.all(out[rect[0] + 1 : rect[2], rect[1] + 1 : rect[3]] == leak_color)
    assert out[gap] == leak_color
    ray = _ray_cells(grid.shape, rect, gap)
    for cell in ray:
        assert out[cell] == leak_color
    remaining_outside = out[~inside]
    assert distractor_color not in set(int(x) for x in remaining_outside.ravel())
    assert np.sum(grid == bg) > np.sum(grid != bg), "background must dominate input"
    assert not np.array_equal(grid, out), "transformation must be non-trivial"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid broken-frame leak puzzle")
