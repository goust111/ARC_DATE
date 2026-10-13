from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TILE_SIZE = 5
MIN_RECT_SIDE = 3
GUIDE_COLOR = ORANGE
TARGET_COLOR = YELLOW
NOISE_PALETTE = (BLUE, RED, GREEN, GREY)
RECT_COLORS = (BLUE, RED, GREEN)
ALLOWED_RECT_SIZES = ((3, 3), (3, 4), (4, 3))
MAX_GENERATION_ATTEMPTS = 800
NOISE_ATTEMPTS_PER_LAYOUT = 200

_GEN_CALL_COUNT = 0


GuideInfo = tuple[str, int]
Rect = tuple[int, int, int, int, int]  # (r0, c0, h, w, color)


def _guide_info(grid: np.ndarray) -> GuideInfo:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square tile"
    assert h >= MIN_RECT_SIDE, "tile is too small"
    assert h * h <= 30, "expanded canvas would exceed ARC size limits"

    row_guides = [r for r in range(h) if np.all(g[r, :] == GUIDE_COLOR)]
    col_guides = [c for c in range(w) if np.all(g[:, c] == GUIDE_COLOR)]
    assert len(row_guides) + len(col_guides) == 1, "expected exactly one orange guide line"
    if row_guides:
        return ("row", int(row_guides[0]))
    return ("col", int(col_guides[0]))


def _solid_line_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    count = 0
    for r in range(g.shape[0]):
        if np.all(g[r, :] == int(g[r, 0])):
            count += 1
    for c in range(g.shape[1]):
        if np.all(g[:, c] == int(g[0, c])):
            count += 1
    return count


def _right_runs(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    runs = np.ones((h, w), dtype=int)
    for r in range(h):
        for c in range(w - 2, -1, -1):
            if int(grid[r, c]) == int(grid[r, c + 1]):
                runs[r, c] = runs[r, c + 1] + 1
    return runs


def _large_rectangle_candidates(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    runs = _right_runs(g)
    found: set[tuple[int, int, int, int, int, int]] = set()

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == GUIDE_COLOR:
                continue
            max_w = w - c0
            for r1 in range(r0, h):
                if int(g[r1, c0]) != color:
                    break
                max_w = min(max_w, int(runs[r1, c0]))
                rect_h = r1 - r0 + 1
                if rect_h >= MIN_RECT_SIDE and max_w >= MIN_RECT_SIDE:
                    area = rect_h * max_w
                    found.add((r0, c0, rect_h, max_w, color, area))

    return sorted(found)


def _best_hidden_rectangle(grid: np.ndarray) -> Rect:
    candidates = _large_rectangle_candidates(grid)
    assert candidates, "expected at least one large monochrome rectangle"
    best_area = max(area for *_rest, area in candidates)
    best = [(r0, c0, h, w, color) for r0, c0, h, w, color, area in candidates if area == best_area]
    assert len(best) == 1, "largest rectangle must be unique"
    return best[0]


def _band_origins(n: int, guide: GuideInfo) -> list[tuple[int, int]]:
    orient, idx = guide
    if orient == "row":
        return [(idx * n, block_c * n) for block_c in range(n)]
    return [(block_r * n, idx * n) for block_r in range(n)]


def _tile_band(grid: np.ndarray, guide: GuideInfo) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = g.shape[0]
    out = np.zeros((n * n, n * n), dtype=int)
    for r0, c0 in _band_origins(n, guide):
        out[r0 : r0 + n, c0 : c0 + n] = g
    return out


def _expected_output(grid: np.ndarray, rect: Rect | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    guide = _guide_info(g)
    n = g.shape[0]
    if rect is None:
        rect = _best_hidden_rectangle(g)
    r0, c0, h, w, _color = rect

    out = _tile_band(g, guide)
    for block_r, block_c in _band_origins(n, guide):
        block = out[block_r : block_r + n, block_c : block_c + n]
        block_rect = _best_hidden_rectangle(block)
        assert block_rect == rect, "every copied tile must expose the same unique rectangle"
        out[block_r + r0 : block_r + r0 + h, block_c + c0 : block_c + c0 + w] = TARGET_COLOR

    orient, idx = guide
    if orient == "row":
        out[idx * n + idx, :] = GUIDE_COLOR
    else:
        out[:, idx * n + idx] = GUIDE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the guided band on a larger canvas, then recolor the repeated hidden rectangle to yellow.

    Input structure:
    A square noise tile contains exactly one full orange row or column that acts as a guide.
    Away from that guide, the tile also contains one unique large solid rectangle of a non-orange
    color, while the remaining cells are multicolor noise that may include scattered distractor
    pixels of the same rectangle color.

    Rule:
    Detect the single orange guide line. Create an n-by-n block canvas of black n-by-n blocks and
    paste the whole tile into every block of the guide's block-row or block-column. Inside each
    pasted tile copy, locate the unique largest monochrome rectangle whose height and width are at
    least 3, and recolor that rectangle to yellow. Keep the orange guide cells orange.

    Key insight:
    The orange line decides where the repetition band appears, while the repeated tile copies make
    the same hidden rectangle recur at matching coordinates; the output is obtained by combining
    both facts rather than by a single local recolor.
    """
    g = np.asarray(grid, dtype=int)
    guide = _guide_info(g)
    rect = _best_hidden_rectangle(g)
    return _expected_output(g, rect)


def _placement_options(orientation: str, guide_idx: int, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    opts = []
    for r0 in range(TILE_SIZE - rect_h + 1):
        for c0 in range(TILE_SIZE - rect_w + 1):
            rows = range(r0, r0 + rect_h)
            cols = range(c0, c0 + rect_w)
            if orientation == "row" and guide_idx in rows:
                continue
            if orientation == "col" and guide_idx in cols:
                continue
            opts.append((r0, c0))
    return opts


def _choose_layout(call_idx: int) -> tuple[str, int, int, int, int, int] | None:
    orientations = ["row", "col"]
    preferred_orientation = orientations[call_idx % 2]
    orientations.remove(preferred_orientation)
    orientation_order = [preferred_orientation, *orientations]

    guide_cycle = [0, TILE_SIZE - 1, 1, TILE_SIZE - 2, TILE_SIZE // 2]
    preferred_guide = guide_cycle[(call_idx // 2) % len(guide_cycle)]
    guide_order = [preferred_guide] + [idx for idx in range(TILE_SIZE) if idx != preferred_guide]

    preferred_size = ALLOWED_RECT_SIZES[call_idx % len(ALLOWED_RECT_SIZES)]
    size_order = [preferred_size] + [size for size in ALLOWED_RECT_SIZES if size != preferred_size]

    for orientation in orientation_order:
        for guide_idx in guide_order:
            for rect_h, rect_w in size_order:
                placements = _placement_options(orientation, guide_idx, rect_h, rect_w)
                if placements:
                    rect_r0, rect_c0 = random.choice(placements)
                    return (orientation, guide_idx, rect_h, rect_w, rect_r0, rect_c0)
    return None


def _validate_generated_input(grid: np.ndarray) -> Rect:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (TILE_SIZE, TILE_SIZE), "generator uses a 5x5 tile to stay within ARC limits"
    assert set(map(int, np.unique(g))).issubset({BLUE, RED, GREEN, GREY, ORANGE})
    assert TARGET_COLOR not in g and BLACK not in g

    guide = _guide_info(g)
    assert _solid_line_count(g) == 1, "guide line must be the only solid row/column"

    rect = _best_hidden_rectangle(g)
    r0, c0, h, w, color = rect
    assert (h, w) in ALLOWED_RECT_SIZES, "generator only emits the intended rectangle sizes"
    assert color in RECT_COLORS

    orient, guide_idx = guide
    if orient == "row":
        assert not (r0 <= guide_idx < r0 + h), "hidden rectangle must not overlap a guide row"
    else:
        assert not (c0 <= guide_idx < c0 + w), "hidden rectangle must not overlap a guide column"

    rect_mask = np.zeros_like(g, dtype=bool)
    rect_mask[r0 : r0 + h, c0 : c0 + w] = True
    guide_mask = np.zeros_like(g, dtype=bool)
    if orient == "row":
        guide_mask[guide_idx, :] = True
    else:
        guide_mask[:, guide_idx] = True
    noise_mask = ~(rect_mask | guide_mask)

    noise_colors = {int(v) for v in g[noise_mask]}
    assert len(noise_colors) >= 3, "noise should remain visibly multicolor"
    assert GREY in noise_colors, "grey noise branch must appear"

    rect_color_noise = int(np.sum(g[noise_mask] == color))
    assert rect_color_noise >= 2, "the rectangle color must also appear as distractor noise"

    candidates = _large_rectangle_candidates(g)
    best_area = h * w
    assert sum(area == best_area for *_rest, area in candidates) == 1
    assert all(area < best_area or (rr == r0 and cc == c0 and hh == h and ww == w and col == color)
               for rr, cc, hh, ww, col, area in candidates)

    expected = _expected_output(g, rect)
    actual = main(g)
    assert expected.shape == (TILE_SIZE * TILE_SIZE, TILE_SIZE * TILE_SIZE)
    assert np.array_equal(actual, expected)
    assert int(np.sum(actual == TARGET_COLOR)) == TILE_SIZE * h * w
    return rect


def _build_candidate_grid(
    orientation: str,
    guide_idx: int,
    rect_h: int,
    rect_w: int,
    rect_r0: int,
    rect_c0: int,
) -> np.ndarray | None:
    rect_color = random.choice(RECT_COLORS)
    rect_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    rect_mask[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = True

    guide_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    if orientation == "row":
        guide_mask[guide_idx, :] = True
    else:
        guide_mask[:, guide_idx] = True

    free_cells = [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE) if not rect_mask[r, c] and not guide_mask[r, c]]
    if len(free_cells) < 4:
        return None

    for _ in range(NOISE_ATTEMPTS_PER_LAYOUT):
        grid = np.empty((TILE_SIZE, TILE_SIZE), dtype=int)
        grid[rect_mask] = rect_color
        grid[guide_mask] = GUIDE_COLOR

        distractor_count = random.randint(2, min(4, len(free_cells) - 2))
        distractors = set(random.sample(free_cells, distractor_count))
        for r, c in free_cells:
            if (r, c) in distractors:
                grid[r, c] = rect_color
            else:
                grid[r, c] = random.choice(NOISE_PALETTE)

        if _solid_line_count(grid) != 1:
            continue

        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_idx = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = _choose_layout(call_idx)
        if layout is None:
            break
        orientation, guide_idx, rect_h, rect_w, rect_r0, rect_c0 = layout
        grid = _build_candidate_grid(orientation, guide_idx, rect_h, rect_w, rect_r0, rect_c0)
        if grid is None:
            continue
        _validate_generated_input(grid)
        return grid

    raise RuntimeError("failed to generate a valid guided-tiling rectangle input")
