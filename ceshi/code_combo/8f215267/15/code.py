# Combo variant of 8f215267 (index 15)
# combo_spec: D6/1+D11/2
# Name: 干扰保留+背景色打孔

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for combo D6/1 + D11/2
#
# concrete_behavior:
#   input:
#     - The left side still contains 3-4 distinct hollow 5x11 frames.
#     - The right side still contains small monochrome sprites.
#     - Some sprites use frame colors and are absorbable by same-color frames.
#     - 1-2 distractor sprites use colors absent from every frame.
#     - Unlike the D11 output, the input frames keep an intact top border so there
#       is no pre-existing hole signal to confuse the before/after comparison.
#   output:
#     - Same-color sprites disappear.
#     - Distractor sprites remain exactly where they were.
#     - Counts are not shown as interior dots because a background-colored dot would
#       be invisible there; instead the top border of each frame loses cells at the
#       original right-to-left tally positions.
#   small_trace:
#     - Suppose frames are red / blue / yellow and right-side sprites are
#       red, red, blue, green.
#     - Original task:
#       erase all four sprites; add 2 red interior dots and 1 blue interior dot.
#     - D6-only:
#       erase red, red, blue; keep green; still draw colored interior dots.
#     - D11-only:
#       erase all four sprites; keep interiors empty; punch 2 red-frame top holes
#       and 1 blue-frame top hole.
#     - Combo here:
#       erase red, red, blue; keep green in place; keep interiors empty; punch
#       2 holes in the red frame and 1 hole in the blue frame.
#     - Net visual difference:
#       both the left display channel and the right-side cleanup behavior change.
#
# complexity_calibration:
#   - The count alphabet stays at 0-4 per frame, so the hole pattern remains easy
#     to read and still reuses the original right-to-left spacing.
#   - Distractors keep using only 1-2 non-frame colors, which is enough repetition
#     to show "some sprites survive" without creating a cluttered palette.
#   - The combo is not simplified away: every generated sample contains at least one
#     distractor and at least one absorbable sprite, so both fused dimensions are
#     active in normal examples rather than only one of them.
#
# old_assumption_audit:
#   - `find_components(... monochromatic=True)` remains valid because both frames
#     and sprites are still monochrome 4-connected regions.
#   - Frame detection by large sparse bbox still works after top holes are added:
#     a 5x11 ring has area 28, and with up to 4 holes its area is still 24, well
#     below the 0.7 * 55 threshold.
#   - The original "erase every non-frame component" assumption is false now.
#     We must erase only components whose color matches some frame color.
#   - The original "write visible same-color dots inside" assumption is also false.
#     Since the tally color becomes background, the solver must write background
#     into the top border instead of drawing inside the already-background interior.
#   - The counting loop now implicitly assumes distractor colors never equal frame
#     colors; the generator guarantees that by sampling them from different palette
#     entries.
#   - The hole-writing loop assumes count <= 4 so holes avoid corners and fit on
#     the top border. The generator preserves the original cap.
#
# placement_legality:
#   - The generator is output-first.
#   - First it builds output frames with punched top-border holes and places only
#     distractor sprites on the right canvas.
#   - Then it derives the input by restoring those holed border cells back to the
#     frame color and placing the absorbable matching sprites into the remaining
#     right-side free space.
#   - `random_free_location(... padding=1)` prevents overlap and near-touching among
#     right-side sprites, and frames live in a disjoint left region, so no sprite
#     can merge with a frame component.
#
# observability_and_cleanliness:
#   - A human can infer the combo from 2-3 examples because the same two signals
#     repeat: some right-side sprites persist, while frame tops lose cells.
#   - No invisible color ordering is used; the distinction is geometric and spatial.
#   - The output stays tidy because each region has one role:
#       left = holed frames,
#       right = surviving distractors.
#     There is no stacking of dots and holes in the same local area.
#
# new_solver_assumptions_and_generator_guarantees:
#   - Assumption: every absorbable sprite color belongs to exactly one frame color.
#     Guarantee: frame colors are distinct, and matching sprites are sampled only
#     from that frame-color list.
#   - Assumption: any sprite with a non-frame color is a distractor to preserve.
#     Guarantee: distractor colors are sampled outside the frame-color set.
#   - Assumption: the input has intact top borders, so output holes are a genuine
#     transformation rather than ambiguous pre-existing damage.
#     Guarantee: the generator explicitly fills all hole positions back in when
#     constructing the input.
# -----------------------------------------------------------------------------


def _split_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Separate large hollow frames from small sprites using the original heuristic."""
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

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
    return frames, patterns


def _marker_cols_from_right(c0: int, c1: int, n_marks: int) -> list[int]:
    """Return tally positions from right to left while avoiding the corners."""
    cols: list[int] = []
    for i in range(int(n_marks)):
        cc = int(c1 - 2 - 2 * i)
        if cc > int(c0):
            cols.append(cc)
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Absorb same-color sprites into frames, preserve distractors, and encode counts as holes.

    Input structure:
    - The background is the most common color.
    - Several large hollow rectangular frames sit on the left, each with a distinct color.
    - Small monochrome sprites sit to the right; some share a frame color and some are distractors.

    Rule:
    - Sprites whose color matches a frame are absorbed: erase them from the output and count them for that frame.
    - Sprites whose color matches no frame are not absorbed and remain where they are.
    - Because the tally mark color is the background, represent each count by punching background holes into
      the corresponding frame's top border at the usual right-to-left spaced positions.

    Key insight:
    - The task is no longer "clear the right side and draw dots inside". Instead, matching sprites vanish into
      same-color frames, mismatching sprites survive, and counts become missing border cells on the frame tops.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_components(g, bg)
    frame_colors = {int(f["color"]) for f in frames}

    for p in patterns:
        if int(p["color"]) not in frame_colors:
            continue
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, c1 = map(int, f["bbox"])
        for cc in _marker_cols_from_right(c0, c1, num_patterns):
            out[int(r0), int(cc)] = bg

    return out


# --------------- generator (combo D6/1 + D11/2) ---------------

# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(color: int, bg: int, n_holes: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with holes in the top border."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for cc in _marker_cols_from_right(0, FRAME_W - 1, n_holes):
        patch[0, int(cc)] = bg
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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    if max(counts) <= 0:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    right_out = np.full((height, RIGHT_W), bg, dtype=int)
    try:
        for color in distractor_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_out,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_out, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None
    paste(out, right_out, 0, RIGHT_C0, transparent=bg)

    inp = out.copy()
    for r0, count, color in zip(frame_rows, counts, frame_colors):
        for cc in _marker_cols_from_right(FRAME_C0, FRAME_C0 + FRAME_W - 1, count):
            inp[int(r0), int(cc)] = int(color)

    right_in = right_out.copy()
    match_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    rng.shuffle(match_colors)
    try:
        for color in match_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_in,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_in, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_in, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
