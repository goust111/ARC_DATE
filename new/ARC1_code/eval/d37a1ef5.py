import numpy as np

from arc_utils import bbox, colors_present, get_bg
import random
from itertools import combinations


def _unique_largest_color(g: np.ndarray, colors: list[int]) -> int:
    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    ordered = sorted(counts, key=counts.get, reverse=True)
    assert len(ordered) >= 1
    assert len(ordered) == 1 or counts[ordered[0]] > counts[ordered[1]]
    return int(ordered[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the frame interior with the frame color except for the rectangle spanned by the marker pixels.

    Input structure
    - A background, a one-cell-thick rectangular foreground frame, and marker
      pixels of the other foreground color inside the frame.
    Rule
    - Use the frame bbox and marker bbox.
    - Recolor background cells strictly inside the frame to the frame color unless
      they lie inside the marker bbox; keep frame and marker cells unchanged.
    Key insight
    - The marker pixels define a protected rectangular window, not paths to draw.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2

    frame_color = _unique_largest_color(g, fg_colors)
    marker_colors = [color for color in fg_colors if color != frame_color]
    assert len(marker_colors) == 1
    marker_color = marker_colors[0]

    frame_bbox = bbox(g == frame_color)
    marker_bbox = bbox(g == marker_color)
    assert frame_bbox is not None
    assert marker_bbox is not None

    r0, c0, r1, c1 = frame_bbox
    mr0, mc0, mr1, mc1 = marker_bbox
    assert r0 < mr0 <= mr1 < r1
    assert c0 < mc0 <= mc1 < c1

    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[r0 : r1 + 1, c0] = True
    frame_mask[r0 : r1 + 1, c1] = True
    frame_mask[r0, c0 : c1 + 1] = True
    frame_mask[r1, c0 : c1 + 1] = True
    assert np.array_equal(g == frame_color, frame_mask)

    protected = np.zeros_like(g, dtype=bool)
    protected[mr0 : mr1 + 1, mc0 : mc1 + 1] = True

    interior = np.zeros_like(g, dtype=bool)
    interior[r0 + 1 : r1, c0 + 1 : c1] = True

    out = g.copy()
    fill_mask = interior & ~protected & (g == bg)
    out[fill_mask] = frame_color
    return out

# --------------- generator (o5x) ---------------

# Colors and retry budget.
BG_COLOR, FRAME_COLOR, MARKER_COLOR = 0, 2, 5
MAX_GENERATION_ATTEMPTS = 300

# Inclusive canvas, frame, and marker bounds.
GRID_H_RANGE, GRID_W_RANGE = (8, 13), (11, 12)
FRAME_H_RANGE, FRAME_W_RANGE = (6, 9), (8, 10)
MARKER_H_RANGE, MARKER_W_RANGE, MARKER_COUNT_RANGE = (2, 4), (2, 4), (2, 4)
MIN_OUTER_MARGIN, MIN_MARKER_GAP = 1, 1


def _gen_marker_cells(mh: int, mw: int, count: int) -> tuple[tuple[int, int], ...]:
    cells = [(r, c) for r in range(mh) for c in range(mw)]
    valid = [
        combo
        for combo in combinations(cells, count)
        if min(r for r, _ in combo) == 0
        and max(r for r, _ in combo) == mh - 1
        and min(c for _, c in combo) == 0
        and max(c for _, c in combo) == mw - 1
    ]
    return random.choice(valid)


def _sample_layout() -> tuple[int, ...]:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    h = random.randint(max(GRID_H_RANGE[0], frame_h + 2 * MIN_OUTER_MARGIN), GRID_H_RANGE[1])
    w = random.randint(max(GRID_W_RANGE[0], frame_w + 2 * MIN_OUTER_MARGIN), GRID_W_RANGE[1])
    r0 = random.randint(MIN_OUTER_MARGIN, h - frame_h - MIN_OUTER_MARGIN)
    c0 = random.randint(MIN_OUTER_MARGIN, w - frame_w - MIN_OUTER_MARGIN)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    marker_h = random.randint(MARKER_H_RANGE[0], min(MARKER_H_RANGE[1], frame_h - 2 * MIN_MARKER_GAP))
    marker_w = random.randint(MARKER_W_RANGE[0], min(MARKER_W_RANGE[1], frame_w - 2 * MIN_MARKER_GAP))
    mr0 = random.randint(r0 + MIN_MARKER_GAP, r1 - marker_h - MIN_MARKER_GAP + 1)
    mc0 = random.randint(c0 + MIN_MARKER_GAP, c1 - marker_w - MIN_MARKER_GAP + 1)
    return h, w, r0, c0, r1, c1, mr0, mc0, marker_h, marker_w


def _gen_attempt() -> np.ndarray | None:
    h, w, r0, c0, r1, c1, mr0, mc0, marker_h, marker_w = _sample_layout()
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    marker_cells = _gen_marker_cells(marker_h, marker_w, marker_count)

    out = np.full((h, w), BG_COLOR, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = FRAME_COLOR
    out[mr0 : mr0 + marker_h, mc0 : mc0 + marker_w] = BG_COLOR
    for dr, dc in marker_cells:
        out[mr0 + dr, mc0 + dc] = MARKER_COLOR

    inp = out.copy()
    interior = inp[r0 + 1 : r1, c0 + 1 : c1]
    interior[interior == FRAME_COLOR] = BG_COLOR

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d37a1ef5 input")
