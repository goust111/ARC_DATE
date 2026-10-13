from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox, get_bg


def _rectangular_border_mask(shape: tuple[int, int], b: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = b
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the frame interior with two marker colors arranged in matching 2x2 diagonal quadrants.

    Input structure:
    - The dominant background contains one solid one-pixel-thick rectangular frame.
    - The frame has an even-height and even-width empty interior.
    - Exactly two one-pixel markers of different colors sit outside the frame.

    Rule:
    - A marker in the northwest or southeast side of the frame center supplies the
      main diagonal quadrants of the interior.
    - A marker in the northeast or southwest side supplies the other two quadrants.
    - Split the frame interior into row and column halves and fill each cell by
      the diagonal class of its half-position.

    Key insight:
    - The marker colors are not copied from their exact rows or columns; their
      coarse position relative to the frame center selects one of the two diagonal
      quadrant masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    counts = Counter(map(int, g.flat))
    foreground_colors = [color for color in counts if color != bg]
    frame_colors = [color for color in foreground_colors if counts[color] > 1]
    assert len(frame_colors) == 1, "expected one non-background frame color"
    frame_color = int(frame_colors[0])

    frame_bbox = bbox(g == frame_color)
    assert frame_bbox is not None, "expected a frame"
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"
    expected_frame = _rectangular_border_mask(g.shape, frame_bbox)
    assert np.array_equal(g == frame_color, expected_frame), "frame must be a rectangular border"

    marker_coords = np.argwhere((g != bg) & (g != frame_color))
    assert marker_coords.shape == (2, 2), "expected exactly two marker pixels"
    marker_colors = [int(g[r, c]) for r, c in marker_coords]
    assert len(set(marker_colors)) == 2, "marker colors must be distinct"

    center_r2 = r0 + r1
    center_c2 = c0 + c1
    diagonal_color: dict[str, int] = {}
    for r, c in marker_coords:
        assert not (r0 <= r <= r1 and c0 <= c <= c1), "markers must be outside the frame"
        assert 2 * int(r) != center_r2 and 2 * int(c) != center_c2, (
            "marker must lie strictly to one row side and one column side of the frame center"
        )

        same_center_side = (2 * int(r) < center_r2) == (2 * int(c) < center_c2)
        diagonal = "main" if same_center_side else "anti"
        assert diagonal not in diagonal_color, "expected one marker for each diagonal"
        diagonal_color[diagonal] = int(g[r, c])

    assert set(diagonal_color) == {"main", "anti"}, "both diagonal colors must be present"

    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    assert interior_h > 0 and interior_w > 0, "frame interior must be nonempty"
    assert interior_h % 2 == 0 and interior_w % 2 == 0, "interior must split into equal halves"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "frame interior must start empty"

    out = g.copy()
    for r in range(r0 + 1, r1):
        bottom_half = (r - (r0 + 1)) >= interior_h // 2
        for c in range(c0 + 1, c1):
            right_half = (c - (c0 + 1)) >= interior_w // 2
            diagonal = "main" if bottom_half == right_half else "anti"
            out[r, c] = diagonal_color[diagonal]

    return out

# --------------- generator (o5x) ---------------

# Canvas, frame, and marker placement.
GRID_SIZE_RANGE = (8, 15)  # np.random.randint high is exclusive.
FRAME_SIZE_RANGE = (4, 10)
FRAME_SIZE_STEP = 2
FRAME_MARGIN = 1
MARKER_OFFSET = 1

# Palette and retry policy.
COLOR_IDS = tuple(range(10))
MARKER_COUNT = 2
GENERATION_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*GRID_SIZE_RANGE))
    w = int(np.random.randint(*GRID_SIZE_RANGE))

    frame_min, frame_max = FRAME_SIZE_RANGE
    valid_frame_h = list(range(frame_min, min(frame_max, h - 2 * FRAME_MARGIN) + 1, FRAME_SIZE_STEP))
    valid_frame_w = list(range(frame_min, min(frame_max, w - 2 * FRAME_MARGIN) + 1, FRAME_SIZE_STEP))
    if not valid_frame_h or not valid_frame_w:
        return None

    fh = int(np.random.choice(valid_frame_h))
    fw = int(np.random.choice(valid_frame_w))
    r0 = int(np.random.randint(FRAME_MARGIN, h - fh - FRAME_MARGIN + 1))
    c0 = int(np.random.randint(FRAME_MARGIN, w - fw - FRAME_MARGIN + 1))
    r1, c1 = r0 + fh - 1, c0 + fw - 1

    bg = int(np.random.randint(0, len(COLOR_IDS)))
    frame_color = int(np.random.choice([x for x in COLOR_IDS if x != bg]))
    marker_colors = np.random.choice(
        [x for x in COLOR_IDS if x not in (bg, frame_color)], size=MARKER_COUNT, replace=False
    )
    main_color, anti_color = map(int, marker_colors)

    main_idx = int(np.random.randint(0, MARKER_COUNT))
    anti_idx = int(np.random.randint(0, MARKER_COUNT))
    d = MARKER_OFFSET
    main_pos = ((r0 - d, c0 - d), (r1 + d, c1 + d))[main_idx]
    anti_pos = ((r0 - d, c1 + d), (r1 + d, c0 - d))[anti_idx]

    out = np.full((h, w), bg, dtype=int)
    out[r0 : r1 + 1, [c0, c1]] = frame_color
    out[[r0, r1], c0 : c1 + 1] = frame_color
    out[main_pos] = main_color
    out[anti_pos] = anti_color

    mid_r = r0 + 1 + (fh - 2) // 2
    mid_c = c0 + 1 + (fw - 2) // 2
    out[r0 + 1 : mid_r, c0 + 1 : mid_c] = main_color
    out[mid_r:r1, mid_c:c1] = main_color
    out[r0 + 1 : mid_r, mid_c:c1] = anti_color
    out[mid_r:r1, c0 + 1 : mid_c] = anti_color

    inp = out.copy()
    inp[r0 + 1 : r1, c0 + 1 : c1] = bg
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 94414823 input")
