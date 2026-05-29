# Combo variant of 8f215267 (index 18)
# combo_spec: D9/2+D13/0
# Name: 网格排列+精灵在框架内

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Original task: 3-4 tall frames sit in one left-side column; sprites live in a separate
#   right-side field. The solver can count same-color sprites globally because location does not
#   matter.
# - Combo behavior: frames form a 2x2 or 2x3 grid, and every sprite is placed inside a frame
#   interior. A frame may contain a mixture of matching-color sprites and foreign-color sprites.
# - Small trace:
#   Original: a blue sprite anywhere outside the frames increments the blue frame.
#   Combo: a blue sprite inside the red frame is erased but does not increment any frame, while a
#   blue sprite inside the blue frame does increment the blue frame. The output therefore depends
#   on both color and containment.
#
# b) Complexity calibration
# - The grid layout must be visually obvious, so the generator always uses two frame rows and two
#   or three frame columns.
# - "Sprites inside frames" would be degenerate if we still counted colors globally; that would
#   reproduce the old rule on the same input. To prevent that, each sample includes at least one
#   foreign-color sprite whose color matches a different frame.
# - Frame size is increased to 7x9. This is the minimum clean size that leaves a one-cell moat
#   between same-color sprites and the perimeter, so the frame and the sprite remain separate
#   monochrome components.
#
# c) Implicit-assumption review of main()
# - The original area heuristic for frame detection is too weak here: once sprites move inside
#   frames, large non-frame objects could live near frame interiors. This solver upgrades frame
#   detection to an exact hollow-rectangle test.
# - The old "count all same-color patterns" assumption is invalid. The new code explicitly assigns
#   each non-frame component to the unique frame interior that contains it, then counts only local
#   color matches.
# - Dot placement still uses the center row and spacing 2 from the right, but with width 9 the
#   capacity becomes 2 in this generator to preserve interior space for sprites.
#
# d) Placement legality
# - Output is built first as a clean grid of hollow frames plus tally dots.
# - Input is derived by removing dots and inserting sprite components into fixed interior slots.
# - Slots are pairwise 4-disconnected and stay one cell away from the perimeter, so same-color
#   sprites never merge with their host frame or with one another.
#
# e) Visual observability
# - D9 is observable because the number of distinct frame columns is always 2 or 3, never 1.
# - D13 is observable because sprites are visibly inside frames rather than in a separate strip.
# - The local-counting rule is inferable because foreign-color sprites appear inside frames yet do
#   not produce dots there, and they also do not inflate the matching frame elsewhere.
#
# f) Visual cleanliness
# - Every frame interior has only three small reserved sprite slots, so the grid stays legible.
# - Sprites are always 2-cell connected pieces, while output dots are single cells, making the
#   input/output contrast easy to see.
#
# g) Color inference
# - Colors still support direct equality matching, but only after spatial ownership is resolved.
# - No hidden ordering by numeric color id is used; all frame colors are distinct and foreign
#   sprites are explained purely by visible frame containment.
#
# h) Generator-side anti-degeneration signal
# - For every generated sample, the vector "local matching counts per frame" differs from the
#   vector "global counts of each frame color" because at least one frame-color sprite is placed in
#   the wrong frame. This ensures the combo does not collapse back to the original task.
#
# i) New solver assumptions to audit
# - main() assumes frames are exact monochrome hollow rectangles.
# - main() assumes non-frame components are wholly contained in at most one frame interior.
# - generate_input() guarantees both by using fixed frame geometry and fixed interior slots.


FRAME_H = 7
FRAME_W = 9
DOT_CAPACITY = 2
MAX_TOTAL_SPRITES_PER_FRAME = 3

FRAME_ROWS = [1, 10]
GRID_W = 33
BOTTOM_PAD_RANGE = (0, 4)
LAYOUTS = {
    2: [5, 19],
    3: [1, 12, 23],
}

# Three pairwise separated interior sprite slots.
FRAME_SLOTS: list[list[tuple[int, int]]] = [
    [(2, 2), (2, 3)],
    [(2, 5), (2, 6)],
    [(4, 3), (4, 4)],
]

MAX_ATTEMPTS = 300


def _is_hollow_frame(comp: dict) -> bool:
    """Return True iff the component is an exact hollow rectangle perimeter."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    if h < 5 or w < 5:
        return False

    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    if len(coords) != (2 * (h + w) - 4):
        return False

    for rr in range(r0, r1 + 1):
        if (rr, c0) not in coords or (rr, c1) not in coords:
            return False
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in coords or (r1, cc) not in coords:
            return False
    return True


def _inside_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _component_inside_box(comp: dict, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, box)
    for r, c in comp["coords"]:
        rr = int(r)
        cc = int(c)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            return False
    return True


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    for i in range(min(n_dots, DOT_CAPACITY)):
        patch[mid_r, FRAME_W - 3 - 2 * i] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert frame-internal matching sprites into tally dots.

    Input structure:
    - The background is the most common color.
    - Hollow rectangular frames are arranged as a 2x2 or 2x3 grid.
    - Small monochrome sprites sit inside frame interiors; some match the host frame color and
      some are foreign colors.

    Rule:
    - Remove every non-frame sprite from the grid.
    - For each frame, count only the sprites that lie inside that frame and share the frame color.
    - Draw that many dots on the frame's center row, spaced every 2 cells from the right.

    Key insight:
    - Color matching alone is not enough anymore: each sprite must first be assigned to a specific
      frame interior, and only local same-color matches become dots.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames = [comp for comp in regions if _is_hollow_frame(comp)]
    sprites = [comp for comp in regions if not _is_hollow_frame(comp)]
    frames.sort(key=lambda comp: tuple(map(int, comp["bbox"])))

    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = bg

    for frame in frames:
        frame_color = int(frame["color"])
        inner_box = _inside_bbox(frame["bbox"])
        count = 0
        for sprite in sprites:
            if _component_inside_box(sprite, inner_box) and int(sprite["color"]) == frame_color:
                count += 1

        if count <= 0:
            continue

        r0, c0, _r1, c1 = map(int, frame["bbox"])
        r_mid = r0 + FRAME_H // 2
        for i in range(min(count, DOT_CAPACITY)):
            out[r_mid, c1 - 2 - 2 * i] = frame_color

    return out


def _sample_counts(
    rng: np.random.Generator,
    n_frames: int,
    total_match: int,
    min_active: int,
) -> list[int]:
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match

    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        active_now = sum(v > 0 for v in counts)
        need_positive = max(0, min_active - active_now)
        reserve = max(0, need_positive - 1)
        lo = max(0, remaining - DOT_CAPACITY * frames_left)
        if reserve > 0:
            lo = max(lo, 1)
        hi = min(DOT_CAPACITY, remaining)
        if lo > hi:
            raise ValueError("invalid count interval")
        val = int(rng.integers(lo, hi + 1))
        counts[int(idx)] = val
        remaining -= val

    if remaining != 0:
        raise ValueError("count sampling failed")
    if sum(v > 0 for v in counts) < min_active:
        raise ValueError("not enough active frames")
    return counts


def _pick_total_match(rng: np.random.Generator, n_frames: int) -> int:
    min_active = n_frames // 2
    lo = min_active + 1
    hi = min((DOT_CAPACITY * n_frames) - 1, min_active + n_frames)
    return int(rng.integers(lo, hi + 1))


def _make_foreign_specs(
    rng: np.random.Generator,
    n_frames: int,
    frame_colors: list[int],
) -> list[tuple[int, int]]:
    n_foreign = int(rng.integers(1, min(3, n_frames) + 1))
    targets = [int(x) for x in rng.permutation(n_frames)[:n_foreign]]
    specs: list[tuple[int, int]] = []
    for target in targets:
        sources = [idx for idx in range(n_frames) if idx != target]
        source = int(rng.choice(sources))
        specs.append((target, int(frame_colors[source])))
    return specs


def _paint_slot(grid: np.ndarray, frame_r0: int, frame_c0: int, slot_idx: int, color: int) -> None:
    for dr, dc in FRAME_SLOTS[slot_idx]:
        grid[frame_r0 + dr, frame_c0 + dc] = color


def _global_color_counts(frame_colors: list[int], local_counts: list[int], foreign_specs: list[tuple[int, int]]) -> list[int]:
    counts = {int(color): int(val) for color, val in zip(frame_colors, local_counts)}
    for _target, color in foreign_specs:
        counts[int(color)] += 1
    return [counts[int(color)] for color in frame_colors]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_cols = int(rng.choice([2, 3]))
    frame_cols = list(LAYOUTS[n_cols])
    n_frames = 2 * n_cols
    min_active = n_cols
    total_match = _pick_total_match(rng, n_frames)

    palette = rng.choice(np.arange(10), size=1 + n_frames, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]

    try:
        counts = _sample_counts(rng, n_frames, total_match, min_active=min_active)
    except ValueError:
        return None

    foreign_specs = _make_foreign_specs(rng, n_frames, frame_colors)
    if _global_color_counts(frame_colors, counts, foreign_specs) == counts:
        return None

    height = FRAME_ROWS[-1] + FRAME_H + int(rng.integers(*BOTTOM_PAD_RANGE))
    out = np.full((height, GRID_W), bg, dtype=int)

    frame_positions: list[tuple[int, int]] = []
    idx = 0
    for r0 in FRAME_ROWS:
        for c0 in frame_cols:
            paste(out, _frame_patch(frame_colors[idx], bg, counts[idx]), r0, c0)
            frame_positions.append((r0, c0))
            idx += 1

    inp = out.copy()
    for (r0, c0), count in zip(frame_positions, counts):
        mid_r = r0 + FRAME_H // 2
        for i in range(min(count, DOT_CAPACITY)):
            inp[mid_r, c0 + FRAME_W - 3 - 2 * i] = bg

    frame_slot_orders = [list(rng.permutation(len(FRAME_SLOTS))) for _ in range(n_frames)]

    for frame_idx, ((r0, c0), color, count) in enumerate(zip(frame_positions, frame_colors, counts)):
        order = frame_slot_orders[frame_idx]
        for slot_idx in order[:count]:
            _paint_slot(inp, r0, c0, int(slot_idx), color)

    used_slots = [counts[i] for i in range(n_frames)]
    for target_idx, foreign_color in foreign_specs:
        if used_slots[target_idx] >= MAX_TOTAL_SPRITES_PER_FRAME:
            return None
        slot_idx = frame_slot_orders[target_idx][used_slots[target_idx]]
        r0, c0 = frame_positions[target_idx]
        _paint_slot(inp, r0, c0, int(slot_idx), foreign_color)
        used_slots[target_idx] += 1

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
