from __future__ import annotations
import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as dots inside.

    Input structure:
    - bg = most common color.
    - Hollow rectangular frames in various colors.
    - Small monochrome pattern sprites scattered outside frames.

    Rule:
    - Remove all small patterns from the grid.
    - For each frame of color C, count how many patterns share color C.
    - Draw that many dots along the frame's center row, spaced by 2, starting from the right.

    Key insight: Frames act as tally displays — scattered same-color patterns are "absorbed" and their count is shown as interior dots.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    # Remove patterns from output.
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    # For each frame, count matching patterns and draw dots.
    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col

    return out

# --------------- generator (codex-att2) ---------------

# --- layout geometry ---
FRAME_H = 5                    # frame height (rows)
FRAME_W = 11                   # frame width (cols)
FRAME_R0 = 1                   # top-left row of first frame
FRAME_C0 = 1                   # top-left col of all frames
FRAME_GAP = 2                  # bg rows between consecutive frames
GRID_W = 23                    # fixed grid width
RIGHT_C0 = 13                  # first column of the right (sprite) region
RIGHT_W = 9                    # width of the right region

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)        # [lo, hi) for number of frames
N_DISTRACTORS_RANGE = (1, 3)   # [lo, hi) for distractor sprite count
TOTAL_PATTERNS_RANGE = (5, 11) # [lo, hi) for total sprite count
MAX_DOTS_PER_FRAME = 4         # max dot tally inside one frame
BOTTOM_PAD_RANGE = (0, 7)      # [lo, hi) extra bg rows when n_frames==3

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)     # [lo, hi) connected-cell count per sprite
SPRITE_MAX_H = 4               # max bbox height for sprite
SPRITE_MAX_W = 3               # max bbox width for sprite
SPRITE_PADDING = 1             # gap between placed sprites

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with n_dots on the center row."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """Distribute total_match sprites across n_frames, each capped at MAX_DOTS_PER_FRAME."""
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match
    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        lo = max(0, remaining - MAX_DOTS_PER_FRAME * frames_left)
        hi = min(MAX_DOTS_PER_FRAME, remaining)
        val = int(rng.integers(lo, hi + 1))
        counts[int(idx)] = val
        remaining -= val
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w bounding box."""
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    """Generate a small random connected sprite, tight-cropped to its bounding box."""
    area = int(rng.integers(*SPRITE_AREA_RANGE))
    dims = [(h, w) for h in range(1, SPRITE_MAX_H + 1)
            for w in range(1, SPRITE_MAX_W + 1) if h * w >= area]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    # --- choose structure ---
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    # --- build output (frames with dots) ---
    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    # --- derive input (dots removed, sprites placed right) ---
    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_canvas, sprite, transparent=bg,
                padding=SPRITE_PADDING, padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
