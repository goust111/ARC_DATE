# Combo variant of 8f215267 (index 21)
# combo_spec: D5/2+D7/1
# Name: 精灵移入框架+对角线点

from __future__ import annotations

"""
Design analysis for the combo variant D5/2 + D7/1

a) Concrete behavior and trace
- Source task: hollow color frames sit on the left, same-color sprites are scattered on the
  right, and the output deletes every sprite then draws a horizontal tally inside each frame.
- Fused task: matching sprites are moved into their frame and preserved as visible objects,
  while the count is also shown by same-color dots on the frame's descending diagonal.
- Small example with a red frame and 3 red sprites on the right:
  source output = erase the 3 red sprites, then place 3 red pixels on one middle row;
  fused output = erase the 3 red sprites from the right, paste 3 red mini-sprites into
  fixed interior slots near the upper-left of the red frame, and also place 3 red dots at
  `(r1-1, c1-1)`, `(r1-2, c1-2)`, `(r1-3, c1-3)` inside that same frame.
- Difference from simply stacking two independent changes: the moved sprites and the diagonal
  dots must coexist in one frame interior, so the geometry has to be redesigned jointly.

b) Complexity calibration
- The original 5x11 frame interior is only 3x9, which is too cramped for both relocated
  sprites and a readable diagonal tally. The frame is therefore enlarged to 7x11, giving a
  5x9 interior while keeping the left-column layout visually close to the source task.
- Matching sprites are standardized to 1x2 horizontal dominoes. This keeps the relocation
  rule concrete and visible without introducing extra shape taxonomy.
- Capacity is capped at 4 per frame. That fits four domino slots along the top interior row
  and still leaves a non-overlapping four-step diagonal descending on the right side.
- Generation insists that at least one frame has count >= 2, otherwise the diagonal could
  collapse to a single pixel and become hard to infer from a few ARC examples.

c) Implicit-assumption audit of the original `main()`
- `out = g.copy()` still holds, but the semantics change: we erase right-side sprites first,
  then paste them back into frames instead of replacing them with dots only.
- The frame/pattern split still relies on frames being large sparse rectangles and patterns
  being tiny compact components. The generator preserves that size separation.
- The original solver assumed the exact sprite geometry was irrelevant once the count was
  known. That assumption fails here: the relocation step needs each component patch, not just
  its color count.
- The original dot code assumed one display row and width-limited spacing. D5/2 breaks that:
  dot placement now depends on both row and column, stepping by `(-1, -1)` from the lower-
  right interior corner.
- New solver assumption introduced by the combo: moved sprites must fit the destination slots
  without colliding with the diagonal dots. The generator guarantees this by using only 1x2
  horizontal sprites and fixed slots that avoid all diagonal cells for counts up to 4.

d) Placement legality
- Frames remain vertically stacked in the left block with constant width and background gaps,
  so they never overlap.
- Right-side sprites are still placed on a dedicated canvas at column 13 using
  `random_free_location(..., padding=1, padding_connectivity=8)`, so they stay disjoint and
  cannot intrude into frames.
- Frame-local relocation slots are `(1,1)`, `(1,3)`, `(1,5)`, `(1,7)` for 1x2 dominoes.
  The diagonal uses `(5,9)`, `(4,8)`, `(3,7)`, `(2,6)` when four dots are needed. Those
  sets are disjoint, so the two fused representations never overlap.
- Output-first construction builds the full fused output, removes only the interior payloads
  to derive the input, scatters right-side sprites, and keeps samples only if
  `main(input) == output`.

e) Visual observability
- The output visibly shows two synchronized consequences of the same count: concrete sprite
  relocation and abstract dot tally. Both are tied to frame color, so the correspondence is
  human-visible.
- Reused identical domino sprites remove any hidden ordering problem: when several matching
  sprites move into a frame, their left-to-right order does not encode information.
- Requiring at least one count >= 2 guarantees that the diagonal pattern is not inferred from
  a lone ambiguous pixel.

f) Visual cleanliness
- The interior layout is intentionally partitioned: moved sprites occupy the top-left strip,
  diagonal dots occupy the lower-right staircase, and the frame border remains readable.
- The right region still contains only source sprites, so input scenes remain sparse and the
  transformation is easy to compare spatially.

g) Change inferability
- No hidden numeric color rule is used. Color only expresses visible frame-sprite matching.
- Distractor colors never match a frame, so they supply negative evidence: they disappear
  from the right but do not reappear inside any frame and do not create diagonal dots.

Post-change generator guarantees checked after implementation
- Every matching sprite is a 1x2 horizontal domino, so slot fitting is deterministic.
- Matching counts stay in `0..4`, so both the relocation slots and the diagonal capacity are
  valid.
- Every generated sample has at least one frame with count >= 2, making the diagonal active.
"""

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# --- layout geometry ---
FRAME_H = 7
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)         # 3 or 4 frames
N_DISTRACTORS_RANGE = (1, 3)    # 1 or 2 distractor sprites
TOTAL_PATTERNS_RANGE = (5, 11)  # 5..10 total sprites
MAX_SPRITES_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 5)

# --- sprite shape ---
SPRITE_H = 1
SPRITE_W = 2
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 400


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((SPRITE_H, SPRITE_W), bg, dtype=int)
    patch[0, :] = color
    return patch


def _diagonal_dot_positions_from_bbox(
    bbox: tuple[int, int, int, int],
    n_dots: int,
) -> list[tuple[int, int]]:
    """Interior cells from lower-right toward upper-left, clipped to the frame interior."""
    r0, c0, r1, c1 = map(int, bbox)
    cells: list[tuple[int, int]] = []
    for i in range(int(n_dots)):
        rr = int(r1 - 1 - i)
        cc = int(c1 - 1 - i)
        if rr > int(r0) and cc > int(c0):
            cells.append((rr, cc))
    return cells


def _frame_slots_local() -> list[tuple[int, int]]:
    """Top-left positions for moved 1x2 sprites inside a FRAME_H x FRAME_W patch."""
    return [(1, 1 + 2 * i) for i in range(MAX_SPRITES_PER_FRAME)]


def _frame_slots(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = map(int, bbox)
    return [(int(r0 + rr), int(c0 + cc)) for rr, cc in _frame_slots_local()]


def _frame_diagonal_cells(n_dots: int) -> list[tuple[int, int]]:
    return _diagonal_dot_positions_from_bbox((0, 0, FRAME_H - 1, FRAME_W - 1), n_dots)


def _frame_patch(color: int, bg: int, n_sprites: int) -> np.ndarray:
    """Build the fused output patch: hollow frame + moved sprites + diagonal dots."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    sprite = _sprite_patch(color, bg)
    for rr, cc in _frame_slots_local()[:n_sprites]:
        paste(patch, sprite, rr, cc, transparent=bg)
    for rr, cc in _frame_diagonal_cells(n_sprites):
        patch[int(rr), int(cc)] = color
    return patch


def _extract_component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    color = int(comp["color"])
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """Distribute matches across frames with capacity 4 and at least one visible diagonal."""
    counts = [0] * n_frames
    for _ in range(64):
        counts = [0] * n_frames
        order = rng.permutation(n_frames)
        remaining = int(total_match)
        for pos, idx in enumerate(order):
            frames_left = n_frames - pos - 1
            lo = max(0, remaining - MAX_SPRITES_PER_FRAME * frames_left)
            hi = min(MAX_SPRITES_PER_FRAME, remaining)
            val = int(rng.integers(lo, hi + 1))
            counts[int(idx)] = val
            remaining -= val
        if max(counts) >= 2:
            return counts
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each matching sprite into its same-color frame and also show the count as diagonal dots.

    Input structure:
    - The background is the most common color.
    - Several hollow rectangular frames of distinct colors are stacked on the left.
    - Small monochrome sprites are scattered in the right-side region; some share a frame color,
      while distractors do not match any frame.

    Rule:
    - Remove every right-side sprite from its original location.
    - For each frame, collect the sprites whose color matches that frame.
    - Paste those sprites into fixed interior slots from left to right.
    - Also place the same number of single-cell dots on the interior diagonal, starting at the
      lower-right interior corner and stepping one cell up-left per dot.

    Key insight:
    - Each frame now displays the same tally in two coordinated ways at once: concrete moved
      sprites parked inside the frame and abstract diagonal count dots. Color matching still
      decides ownership, but the output is no longer a pure count summary.
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

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    patterns_by_color: dict[int, list[dict]] = {}
    for p in patterns:
        patterns_by_color.setdefault(int(p["color"]), []).append(p)
    for plist in patterns_by_color.values():
        plist.sort(key=lambda comp: tuple(map(int, comp["bbox"][:2])))

    frames.sort(key=lambda comp: tuple(map(int, comp["bbox"][:2])))
    for f in frames:
        color = int(f["color"])
        matches = patterns_by_color.get(color, [])
        if not matches:
            continue

        slots = _frame_slots(tuple(map(int, f["bbox"])))
        for (slot_r, slot_c), comp in zip(slots, matches):
            patch = _extract_component_patch(g, comp, bg)
            paste(out, patch, slot_r, slot_c, transparent=bg)

        for rr, cc in _diagonal_dot_positions_from_bbox(f["bbox"], len(matches)):
            out[int(rr), int(cc)] = color

    return out


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
    if max(counts) < 2:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_boxes: list[tuple[int, int, int, int]] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        c0 = FRAME_C0
        paste(out, _frame_patch(color, bg, count), r0, c0)
        frame_boxes.append((r0, c0, r0 + FRAME_H - 1, c0 + FRAME_W - 1))

    inp = out.copy()
    sprite_bg_patch = np.full((SPRITE_H, SPRITE_W), bg, dtype=int)
    for bbox, count in zip(frame_boxes, counts):
        for slot_r, slot_c in _frame_slots(bbox)[:count]:
            paste(inp, sprite_bg_patch, slot_r, slot_c)
        for rr, cc in _diagonal_dot_positions_from_bbox(bbox, count):
            inp[int(rr), int(cc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _sprite_patch(int(color), bg)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
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
