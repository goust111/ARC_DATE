# Combo variant of 8f215267 (index 7)
# combo_spec: D3/0+D9/2
# Name: 不同尺寸框架+网格式排列

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Original task: 3-4 identical 5x11 frames sit in one left-side column, while all sprites live in
#   a separate right strip. Solver logic can silently rely on one shared frame width and on a largely
#   one-dimensional scan of frame locations.
# - Combo task: exactly four frames appear in a 2x2 arrangement. Each frame is still a hollow 5-row
#   rectangle, but its width is chosen from {7, 9, 11}. Sprites are not in a dedicated right strip;
#   they are confined to the background corridors between the four frames.
# - Trace example:
#   Top-left frame = width 7, red, with 2 matching red sprites in the corridor.
#   Top-right frame = width 11, blue, with 4 matching blue sprites in the corridor.
#   Output removes every sprite. The red frame gets dots at its right-to-left inner slots [4, 2]
#   relative to its own bbox, while the blue frame uses [8, 6, 4, 2].
#   Compared with the original, both frame discovery and dot-capacity inference must now be local to
#   each detected rectangle rather than borrowed from one fixed template.
#
# b) Complexity calibration
# - Width choices are restricted to three odd sizes {7, 9, 11}; this is enough to make capacity
#   changes visible without introducing too many cases.
# - Every generated puzzle contains all three widths across the four frames, so the D3 signal is
#   always observable.
# - The layout is always 2x2, not a variable family of grids. This keeps the D9 signal strong and
#   visually consistent instead of spreading evidence across too many placements.
#
# c) Implicit-assumption audit of main()
# - `find_components(... monochromatic=True)` assumes each frame and each sprite is monochrome.
#   The generator preserves that.
# - Treating components via `_is_hollow_frame` replaces the old area heuristic. This is necessary
#   because frame size is no longer uniform and sprites now occupy the same broad region as frames.
# - `_frame_dot_cols(width)` assumes odd frame widths with one center row and alternating interior
#   slots. The generator only emits widths 7/9/11, so the assumption is safe.
# - Classifying every non-frame component as a sprite assumes sprites never touch a frame border or
#   another sprite. Corridor-only placement with padding guarantees this.
#
# d) Placement legality
# - Frames are pasted first into the output.
# - Input is derived by erasing dots from each frame center row.
# - Sprites are then placed on a separate placement canvas that only exposes corridor cells:
#   the middle horizontal band between frame rows plus the vertical gaps between the left and right
#   frame of each row. Frame interiors are explicitly excluded.
# - `random_free_location(... padding=1)` prevents sprite overlap and accidental merging.
#
# e) Visual observability
# - Humans can see two orthogonal changes at once:
#   1. frames are spread over two columns and two rows rather than stacked in one column;
#   2. right edges do not align because widths differ.
# - The counting rule remains color equality, which is stable across examples and easy to compare.
#
# f) Visual cleanliness
# - Sprites never enter frames and never overlap each other.
# - Corridor bands are wide enough to fit the original-style 4x3 maximum sprite bbox, so the layout
#   does not devolve into noisy clipping or near-collisions.
#
# g) Change inferability
# - No ordering depends on numeric color ids. Color is used only for direct equality between sprite
#   color and frame color.
# - Width is the only structural cue that changes dot capacity, and it is directly visible from the
#   frame bbox itself.
#
# h) Anti-degeneration signals enforced by the generator
# - Exactly two distinct frame columns are always used, so the sample cannot collapse back to the
#   original single-column arrangement.
# - The multiset of frame widths always contains at least three distinct values, so the sample cannot
#   collapse back to a uniform-width layout.
# - Sprites are restricted to inter-frame corridors, not a detached right-side panel.
#
# i) New solver assumptions re-checked
# - `main()` assumes input-time frames are empty hollow rectangles with no tally dots already drawn.
# - It assumes width alone determines how many legal tally slots exist.
# - It assumes every counted sprite remains a separate monochrome component.
# - `generate_input()` enforces all three conditions.


FRAME_H = 5
FRAME_WIDTH_CHOICES = (7, 9, 11)
N_FRAMES = 4

TOP_R0 = 1
BOTTOM_R0 = 11
LEFT_C0 = 1
RIGHT_C0 = 17
GRID_W = 29
BOTTOM_PAD_RANGE = (0, 5)

N_DISTRACTORS_RANGE = (1, 3)
MATCH_TOTAL_RANGE = (5, 10)

SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 300


def _frame_dot_cols(width: int) -> list[int]:
    """Return in-frame dot columns for a frame of the given width."""
    return [int(width - 3 - 2 * i) for i in range((int(width) - 3) // 2)]


def _is_hollow_frame(comp: dict) -> bool:
    """Recognize the exact input-time frame shape used by this generator."""
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    if h != FRAME_H or w not in FRAME_WIDTH_CHOICES:
        return False

    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False

    r0, c0, r1, c1 = map(int, b)
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    border = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    return coords == border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove scattered sprites and redraw their same-color tallies inside variable-width frames.

    Input structure:
    - The background is the most common color.
    - Four hollow monochrome frames are arranged as a 2x2 grid.
    - Every frame is 5 cells tall, but its width can be 7, 9, or 11.
    - Small monochrome sprites appear only in the background corridors between those frames.

    Rule:
    - Erase every scattered sprite from the grid.
    - For each frame of color C, count how many sprite components also have color C.
    - Draw that many dots on the frame's middle row, starting at the right inner cell and stepping
      left by 2 each time.
    - The valid dot slots must be inferred from the detected frame width, not from one global width.

    Key insight:
    The counting principle is unchanged, but two original shortcuts are gone: frames are no longer
    in one column, and they no longer share one capacity. Detect each hollow rectangle in 2D, then
    use its own bounding-box width to place the tally.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_hollow_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        color = int(f["color"])
        width = bbox_w(f["bbox"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        r_mid = r0 + FRAME_H // 2
        num_patterns = sum(1 for p in patterns if int(p["color"]) == color)
        for dc in _frame_dot_cols(width)[:num_patterns]:
            out[r_mid, c0 + int(dc)] = color

    return out


def _frame_patch(color: int, bg: int, width: int, n_dots: int) -> np.ndarray:
    """Build a 5 x width hollow rectangle with n_dots on the center row."""
    patch = np.full((FRAME_H, width), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[FRAME_H // 2, _frame_dot_cols(width)[:n_dots]] = color
    return patch


def _sample_frame_widths(rng: np.random.Generator) -> list[int]:
    """
    Sample four widths with guaranteed variability.

    All three legal widths appear at least once; the fourth frame repeats one of them.
    """
    widths = list(FRAME_WIDTH_CHOICES)
    widths.append(int(rng.choice(FRAME_WIDTH_CHOICES)))
    rng.shuffle(widths)
    return [int(w) for w in widths]


def _sample_counts(
    rng: np.random.Generator,
    capacities: list[int],
    total_match: int,
) -> list[int]:
    """Distribute total_match across frames while keeping at least three active frames."""
    n_active = int(rng.integers(3, N_FRAMES + 1))
    active = set(int(x) for x in rng.choice(N_FRAMES, size=n_active, replace=False))

    counts = [1 if i in active else 0 for i in range(N_FRAMES)]
    remaining = total_match - sum(counts)
    residual_caps = [cap - counts[i] for i, cap in enumerate(capacities)]
    order = [int(x) for x in rng.permutation(N_FRAMES)]

    for pos, idx in enumerate(order):
        caps_left = sum(residual_caps[j] for j in order[pos + 1 :])
        lo = max(0, remaining - caps_left)
        hi = min(residual_caps[idx], remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        remaining -= add

    if remaining != 0:
        raise ValueError("count sampling failed")
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w box."""
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
    """Generate a small connected monochrome sprite."""
    area = int(rng.integers(*SPRITE_AREA_RANGE))
    dims = [
        (h, w)
        for h in range(1, SPRITE_MAX_H + 1)
        for w in range(1, SPRITE_MAX_W + 1)
        if h * w >= area
    ]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _frame_positions() -> list[tuple[int, int]]:
    """Return top-left positions for frames in reading order: TL, TR, BL, BR."""
    return [
        (TOP_R0, LEFT_C0),
        (TOP_R0, RIGHT_C0),
        (BOTTOM_R0, LEFT_C0),
        (BOTTOM_R0, RIGHT_C0),
    ]


def _build_allowed_mask(
    shape: tuple[int, int],
    positions: list[tuple[int, int]],
    widths: list[int],
) -> np.ndarray:
    """
    Corridor-only sprite placement mask.

    Allowed cells are:
    - the horizontal band between the top and bottom frame rows
    - the vertical gap between left and right frames on the top row
    - the vertical gap between left and right frames on the bottom row
    """
    H, W = map(int, shape)
    allowed = np.zeros((H, W), dtype=bool)

    left_edges = [positions[0][1], positions[2][1]]
    right_ends = [positions[1][1] + widths[1] - 1, positions[3][1] + widths[3] - 1]

    band_r0 = TOP_R0 + FRAME_H
    band_r1 = BOTTOM_R0 - 1
    band_c0 = min(left_edges)
    band_c1 = max(right_ends)
    allowed[band_r0 : band_r1 + 1, band_c0 : band_c1 + 1] = True

    top_gap_c0 = positions[0][1] + widths[0]
    top_gap_c1 = positions[1][1] - 1
    if top_gap_c0 <= top_gap_c1:
        allowed[TOP_R0 : TOP_R0 + FRAME_H, top_gap_c0 : top_gap_c1 + 1] = True

    bottom_gap_c0 = positions[2][1] + widths[2]
    bottom_gap_c1 = positions[3][1] - 1
    if bottom_gap_c0 <= bottom_gap_c1:
        allowed[BOTTOM_R0 : BOTTOM_R0 + FRAME_H, bottom_gap_c0 : bottom_gap_c1 + 1] = True

    return allowed


def _pick_total_match(rng: np.random.Generator, capacities: list[int]) -> int:
    max_match = min(MATCH_TOTAL_RANGE[1] - 1, sum(capacities))
    min_match = min(MATCH_TOTAL_RANGE[0], max_match)
    return int(rng.integers(min_match, max_match + 1))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    widths = _sample_frame_widths(rng)
    capacities = [len(_frame_dot_cols(w)) for w in widths]
    total_match = _pick_total_match(rng, capacities)
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))

    palette = rng.choice(np.arange(10), size=1 + N_FRAMES + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + N_FRAMES]]
    distractor_colors = [int(x) for x in palette[1 + N_FRAMES : 1 + N_FRAMES + n_distractors]]

    try:
        counts = _sample_counts(rng, capacities, total_match)
    except ValueError:
        return None

    height = BOTTOM_R0 + FRAME_H + int(rng.integers(*BOTTOM_PAD_RANGE))
    out = np.full((height, GRID_W), bg, dtype=int)

    positions = _frame_positions()
    for (r0, c0), color, width, count in zip(positions, frame_colors, widths, counts):
        paste(out, _frame_patch(color, bg, width, count), r0, c0)

    inp = out.copy()
    for (r0, c0), width, count in zip(positions, widths, counts):
        r_mid = r0 + FRAME_H // 2
        for dc in _frame_dot_cols(width)[:count]:
            inp[r_mid, c0 + int(dc)] = bg

    allowed_mask = _build_allowed_mask(inp.shape, positions, widths)
    placement_canvas = np.full(inp.shape, -1, dtype=int)
    placement_canvas[allowed_mask] = bg

    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                placement_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(placement_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    sprite_mask = placement_canvas >= 0
    sprite_mask &= placement_canvas != bg
    inp[sprite_mask] = placement_canvas[sprite_mask]

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
