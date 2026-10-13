import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
LEGEND_WIDTH = 2


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue

        stack = [(start_r, start_c)]
        seen[start_r, start_c] = True
        coords: list[tuple[int, int]] = []

        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        components.append(np.array(coords, dtype=int))

    return components


def _extract_legend(grid: np.ndarray) -> list[int]:
    assert grid.shape[1] >= LEGEND_WIDTH + 2, "expected a left legend plus room for rectangles"

    colors: list[int] = []
    for r in range(grid.shape[0]):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left == right and left != BLACK:
            colors.append(left)
        else:
            break

    assert colors, "expected at least one legend row in the left strip"
    return colors


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []

    for coords in _components4(grid == GREY):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        assert c0 >= LEGEND_WIDTH, "grey frames must not overlap the left legend"
        assert height >= 3 and width >= 3, "each grey component must enclose an interior"

        expected_area = 2 * (height + width) - 4
        assert len(coords) == expected_area, "grey component must be a one-pixel rectangular frame"

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert np.all(sub[border] == GREY), "frame border must be solid grey"
        assert not np.any(sub[1:-1, 1:-1] == GREY), "frame interior cannot contain grey border pixels"

        red_count = int(np.count_nonzero(sub[1:-1, 1:-1] == RED))
        frames.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "red_count": red_count,
            }
        )

    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # deterministic order
    assert frames, "expected at least one grey rectangular frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed grey rectangle border by counting its interior markers and looking up that row in the left legend.

    Input structure:
    - The background is black.
    - The left edge contains a vertical width-2 legend whose rows are solid color bars from top to bottom.
    - Elsewhere there are one-pixel-thick grey rectangular frames, each enclosing black interior cells and some red single-cell markers.

    Rule:
    - Keep the left legend unchanged.
    - For each grey rectangular frame, count the red marker cells strictly inside it.
    - Use that count N as a 1-based row index into the legend, recolor the whole frame border with the legend's Nth color, and replace the interior red markers with black.

    Key insight:
    - The red markers are not copied; only their count matters, because it indexes the legend row that determines the frame's new border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend = _extract_legend(g)
    frames = _find_frames(g)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        red_count = int(frame["red_count"])
        assert 1 <= red_count <= len(legend), "marker count must index an existing legend row"

        draw_color = int(legend[red_count - 1])
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = draw_color

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        red_mask = g[r0 + 1 : r1, c0 + 1 : c1] == RED
        interior[red_mask] = BLACK

    return out


# --------------- generator (gpt-5) ---------------

LEGEND_COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (14, 30)
LEGEND_ROWS_RANGE = (3, 6)
FRAME_COUNT_RANGE = (1, 4)
FRAME_OUTER_HEIGHT_RANGE = (3, 10)
FRAME_OUTER_WIDTH_RANGE = (3, 12)
FRAME_PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 300


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _size_candidates(count: int, height: int, width: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    max_h = min(FRAME_OUTER_HEIGHT_RANGE[1], height)
    max_w = min(FRAME_OUTER_WIDTH_RANGE[1], width - 4)

    for outer_h in range(FRAME_OUTER_HEIGHT_RANGE[0], max_h + 1):
        for outer_w in range(FRAME_OUTER_WIDTH_RANGE[0], max_w + 1):
            if (outer_h - 2) * (outer_w - 2) >= count:
                candidates.append((outer_h, outer_w))

    return candidates


def _pick_counts(legend_rows: int, frame_count: int) -> list[int]:
    values = list(range(1, legend_rows + 1))
    random.shuffle(values)
    counts = values[: min(frame_count, legend_rows)]

    while len(counts) < frame_count:
        counts.append(random.randint(1, legend_rows))

    if frame_count >= 2 and legend_rows >= 2 and random.random() < 0.5:
        counts[0] = 1
        counts[-1] = legend_rows

    random.shuffle(counts)
    return counts


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    legend_rows = random.randint(*LEGEND_ROWS_RANGE)

    min_height = max(GRID_HEIGHT_RANGE[0], legend_rows + 2, 10 + frame_count * 3)
    min_width = max(GRID_WIDTH_RANGE[0], 10 + frame_count * 4)
    if min_height > GRID_HEIGHT_RANGE[1] or min_width > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_height, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])

    grid = np.full((height, width), BLACK, dtype=int)
    legend_colors = random.sample(list(LEGEND_COLOR_POOL), legend_rows)
    for r, color in enumerate(legend_colors):
        grid[r, :LEGEND_WIDTH] = color

    blocked = np.zeros((height, width), dtype=bool)
    blocked[:, :4] = True  # Reserve the legend and a visible gutter.

    frame_specs: list[tuple[int, int, int, int, int]] = []
    counts = _pick_counts(legend_rows, frame_count)
    sizes_by_count = {count: _size_candidates(count, height, width) for count in set(counts)}

    for count in counts:
        candidates = sizes_by_count[count][:]
        random.shuffle(candidates)
        placed = False

        for outer_h, outer_w in candidates:
            for _ in range(FRAME_PLACEMENT_RETRIES):
                r0 = random.randint(0, height - outer_h)
                c0 = random.randint(4, width - outer_w)
                r1 = r0 + outer_h - 1
                c1 = c0 + outer_w - 1

                rr0 = max(0, r0 - 1)
                rr1 = min(height - 1, r1 + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width - 1, c1 + 1)
                if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue

                blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
                frame_specs.append((r0, c0, r1, c1, count))
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    for r0, c0, r1, c1, count in frame_specs:
        _draw_frame(grid, r0, c0, r1, c1)
        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if len(interior_cells) < count:
            return None

        for r, c in random.sample(interior_cells, count):
            grid[r, c] = RED

    try:
        parsed_legend = _extract_legend(grid)
        parsed_frames = _find_frames(grid)
        assert parsed_legend == legend_colors
        assert len(parsed_frames) == frame_count
        main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
