import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SEED_COLORS = (6, 7, 8, 9)

DIRECTIONS = ("top", "bottom", "left", "right")
DIRECTION_TO_COLOR = {
    "top": RED,
    "bottom": BLUE,
    "left": GREEN,
    "right": YELLOW,
}

GEN_RETRIES = 300
OUTER_MARGIN = 2
TILE_GAP = 1
MIN_TILE_HEIGHT = 5
MIN_TILE_WIDTH = 5
MIN_FRAME_HEIGHT = 4
MIN_FRAME_WIDTH = 4
MAX_FRAME_HEIGHT = 10
MAX_FRAME_WIDTH = 10

_GEN_CALL_COUNT = 0


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_mask(height: int, width: int, missing: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if missing == "top":
        mask[0, :] = False
    elif missing == "bottom":
        mask[-1, :] = False
    elif missing == "left":
        mask[:, 0] = False
    elif missing == "right":
        mask[:, -1] = False
    else:
        raise ValueError(missing)
    return mask


def _component_frame_mask(height: int, width: int, missing: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if missing == "top":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif missing == "bottom":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif missing == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    elif missing == "right":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(missing)
    return mask


def _missing_side(submask: np.ndarray) -> str | None:
    height, width = submask.shape
    if height < 2 or width < 2:
        return None
    matches = [
        direction
        for direction in DIRECTIONS
        if np.array_equal(submask, _component_frame_mask(height, width, direction))
    ]
    return matches[0] if len(matches) == 1 else None


def _full_bbox_from_component(r0: int, c0: int, r1: int, c1: int, missing: str) -> tuple[int, int, int, int]:
    if missing == "top":
        return r0 - 1, c0, r1, c1
    if missing == "bottom":
        return r0, c0, r1 + 1, c1
    if missing == "left":
        return r0, c0 - 1, r1, c1
    if missing == "right":
        return r0, c0, r1, c1 + 1
    raise ValueError(missing)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each open rectangular frame interior with the edge-bar color keyed by the missing side.

    Input structure:
    - A black background contains several one-pixel-thick grey rectangular frames, each missing exactly one full side.
    - Each frame encloses exactly one non-background seed pixel somewhere inside its intended rectangle.
    - The grid border carries a one-cell-thick color legend: top, bottom, left, and right edges each show a distinct bar color.

    Rule:
    - Read the four border-bar colors as a direction-to-color dictionary keyed by top, bottom, left, and right.
    - For each grey frame, detect which side of its bounding rectangle is entirely missing.
    - Replace the seed by the mapped border color and fill every cell strictly inside that rectangle with the same color.
    - Keep the grey frame pixels, the outer color bars, and the outside background unchanged.

    Key insight:
    - The absent side provides a directional lookup key, while the frame's bounding box reveals the intended interior even though one border edge is open.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    border_colors = {
        "top": _most_common(g[0, 1:-1]),
        "bottom": _most_common(g[-1, 1:-1]),
        "left": _most_common(g[1:-1, 0]),
        "right": _most_common(g[1:-1, -1]),
    }

    inner = g[1:-1, 1:-1]
    non_bg_inner = inner[inner != BG]
    assert non_bg_inner.size > 0, "expected non-background content inside the border bars"
    frame_color = _most_common(non_bg_inner)

    frame_count = 0
    for comp in _components(g == frame_color):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        submask = g[r0 : r1 + 1, c0 : c1 + 1] == frame_color
        missing = _missing_side(submask)
        if missing is None:
            continue

        full_r0, full_c0, full_r1, full_c1 = _full_bbox_from_component(r0, c0, r1, c1, missing)
        assert 1 <= full_r0 < full_r1 <= g.shape[0] - 2, "frame must stay inside the border bars"
        assert 1 <= full_c0 < full_c1 <= g.shape[1] - 2, "frame must stay inside the border bars"

        inside = g[full_r0 + 1 : full_r1, full_c0 + 1 : full_c1]
        assert inside.size > 0, "frame interior must be non-empty"
        assert int(np.count_nonzero(inside != BG)) == 1, "each frame must contain exactly one interior seed"
        out[full_r0 + 1 : full_r1, full_c0 + 1 : full_c1] = border_colors[missing]
        frame_count += 1

    assert frame_count > 0, "expected at least one valid open frame"
    return out


def _draw_bars(grid: np.ndarray) -> None:
    grid[:, 0] = GREEN
    grid[:, -1] = YELLOW
    grid[0, 1:-1] = RED
    grid[-1, 1:-1] = BLUE


def _split_sizes(total: int, parts: int, minimum: int) -> list[int]:
    assert total >= parts * minimum
    sizes = [minimum] * parts
    for _ in range(total - parts * minimum):
        sizes[random.randrange(parts)] += 1
    return sizes


def _tile_spans(total: int, parts: int, minimum: int) -> list[tuple[int, int]]:
    usable = total - TILE_GAP * (parts - 1)
    sizes = _split_sizes(usable, parts, minimum)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for size in sizes:
        spans.append((cursor, cursor + size))
        cursor += size + TILE_GAP
    return spans


def _layout_options() -> list[tuple[int, int]]:
    return [(1, 1), (1, 2), (2, 1), (2, 2), (2, 2)]


def _choose_layout() -> tuple[int, int]:
    return random.choice(_layout_options())


def _min_grid_size(tile_rows: int, tile_cols: int) -> tuple[int, int]:
    min_rows = 2 * OUTER_MARGIN + tile_rows * MIN_TILE_HEIGHT + (tile_rows - 1) * TILE_GAP
    min_cols = 2 * OUTER_MARGIN + tile_cols * MIN_TILE_WIDTH + (tile_cols - 1) * TILE_GAP
    return min_rows, min_cols


def _frame_count_for_layout(tile_rows: int, tile_cols: int) -> int:
    cells = tile_rows * tile_cols
    if cells == 1:
        return 1
    if cells == 2:
        return random.randint(1, 2)
    return random.randint(2, 4)


def _sample_directions(frame_count: int) -> list[str]:
    global _GEN_CALL_COUNT
    mandatory = DIRECTIONS[_GEN_CALL_COUNT % len(DIRECTIONS)]
    _GEN_CALL_COUNT += 1

    chosen = [mandatory]
    remaining = [direction for direction in DIRECTIONS if direction != mandatory]
    if frame_count > 1:
        chosen.extend(random.sample(remaining, frame_count - 1))
    random.shuffle(chosen)
    return chosen


def _paint_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, missing: str) -> None:
    patch = _frame_mask(height, width, missing)
    region = grid[top : top + height, left : left + width]
    region[patch] = GREY


def _expected_output(inp: np.ndarray, frames: list[tuple[int, int, int, int, str]]) -> np.ndarray:
    out = inp.copy()
    for top, left, height, width, missing in frames:
        out[top + 1 : top + height - 1, left + 1 : left + width - 1] = DIRECTION_TO_COLOR[missing]
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    tile_rows, tile_cols = _choose_layout()
    min_rows, min_cols = _min_grid_size(tile_rows, tile_cols)
    rows = random.randint(min_rows, 30)
    cols = random.randint(min_cols, 30)

    grid = np.full((rows, cols), BG, dtype=int)
    _draw_bars(grid)

    usable_height = rows - 2 * OUTER_MARGIN
    usable_width = cols - 2 * OUTER_MARGIN
    row_spans = _tile_spans(usable_height, tile_rows, MIN_TILE_HEIGHT)
    col_spans = _tile_spans(usable_width, tile_cols, MIN_TILE_WIDTH)

    tile_boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile_boxes.append((OUTER_MARGIN + r0, OUTER_MARGIN + c0, r1 - r0, c1 - c0))

    frame_count = _frame_count_for_layout(tile_rows, tile_cols)
    chosen_tiles = random.sample(tile_boxes, frame_count)
    directions = _sample_directions(frame_count)

    frames: list[tuple[int, int, int, int, str]] = []
    for (tile_top, tile_left, tile_height, tile_width), missing in zip(chosen_tiles, directions, strict=False):
        frame_height = random.randint(MIN_FRAME_HEIGHT, min(MAX_FRAME_HEIGHT, tile_height))
        frame_width = random.randint(MIN_FRAME_WIDTH, min(MAX_FRAME_WIDTH, tile_width))
        top = tile_top + random.randint(0, tile_height - frame_height)
        left = tile_left + random.randint(0, tile_width - frame_width)
        _paint_frame(grid, top, left, frame_height, frame_width, missing)

        seed_r = random.randint(top + 1, top + frame_height - 2)
        seed_c = random.randint(left + 1, left + frame_width - 2)
        grid[seed_r, seed_c] = random.choice(SEED_COLORS)
        frames.append((top, left, frame_height, frame_width, missing))

    expected = _expected_output(grid, frames)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, expected = _generate_case()
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
