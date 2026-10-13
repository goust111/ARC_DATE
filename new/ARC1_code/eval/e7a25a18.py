import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste, scale_sprite
import random


def _find_rectangular_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue

        b = bbox(grid == color)
        assert b is not None
        r0, c0, r1, c1 = b
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(patch[border] == color) and not np.any(patch[~border] == color):
            candidates.append((b, color))

    assert len(candidates) == 1, "expected one single-color rectangular frame"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop to the rectangular frame and replace its interior with the doubled enclosed color pattern.

    Input structure:
    - A background canvas contains one single-color rectangular frame.
    - Inside the frame, a smaller non-background, non-frame color pattern occupies a tight sub-bbox of the interior.

    Rule:
    - Crop the output to the frame bbox.
    - Take the tight bbox of the interior pattern, excluding background and frame color.
    - Scale that pattern by a factor of two and write it into the frame interior.

    Key insight:
    - The frame determines the output size, while the colored seed pattern inside it is a compact half-scale version of the final frame interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frame_bbox, frame_color = _find_rectangular_frame(g, bg)
    out = crop_bbox(g, frame_bbox)
    assert out.shape[0] >= 3 and out.shape[1] >= 3, "expected a frame with an interior"

    interior = out[1:-1, 1:-1]
    pattern_bbox = bbox((interior != bg) & (interior != frame_color))
    assert pattern_bbox is not None, "expected a colored pattern inside the frame"

    pattern = crop_bbox(interior, pattern_bbox)
    scaled = scale_sprite(pattern, 2)
    assert scaled.shape == interior.shape, "expected doubled pattern to fill the frame interior"

    out[1:-1, 1:-1] = scaled
    return out

# --------------- generator (o5x) ---------------
# Canvas/colors.
BG_COLOR, FRAME_COLOR, CANVAS_SIDE = 0, 2, 14
PATTERN_COLORS = tuple(c for c in range(1, 10) if c != FRAME_COLOR)
# Pattern geometry/retries.
QUADRANT_SIDES = (2, 4)
SCALE_FACTOR, PATTERN_COLOR_COUNT, MAX_ATTEMPTS = 2, 4, 300


def _make_output(side: int, colors: list[int]) -> np.ndarray:
    """Return the cropped frame output for one sampled quadrant size and palette."""
    frame_side = SCALE_FACTOR * side + 2
    output = np.full((frame_side, frame_side), FRAME_COLOR, dtype=int)
    interior = output[1:-1, 1:-1]
    interior[:side, :side] = colors[0]
    interior[:side, side:] = colors[1]
    interior[side:, :side] = colors[2]
    interior[side:, side:] = colors[3]
    return output


def _make_input(frame_side: int, seed: np.ndarray) -> np.ndarray:
    """Place the hollow frame and half-scale seed on the fixed canvas."""
    inp = np.full((CANVAS_SIDE, CANVAS_SIDE), BG_COLOR, dtype=int)
    frame = np.full((frame_side, frame_side), BG_COLOR, dtype=int)
    frame[[0, -1], :] = FRAME_COLOR
    frame[:, [0, -1]] = FRAME_COLOR
    frame_r = random.randint(0, CANVAS_SIDE - frame_side)
    frame_c = random.randint(0, CANVAS_SIDE - frame_side)
    seed_span = frame_side - 2 - seed.shape[0]
    seed_r = random.randint(0, seed_span)
    seed_c = random.randint(0, seed_span)
    paste(inp, frame, frame_r, frame_c)
    paste(inp, seed, frame_r + 1 + seed_r, frame_c + 1 + seed_c)
    return inp


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(QUADRANT_SIDES)
    colors = random.sample(PATTERN_COLORS, PATTERN_COLOR_COUNT)
    output = _make_output(side, colors)
    seed = output[1:-1, 1:-1][::SCALE_FACTOR, ::SCALE_FACTOR]
    inp = _make_input(output.shape[0], seed)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e7a25a18 input")
