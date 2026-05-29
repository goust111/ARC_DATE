from __future__ import annotations

# Variant of 8f215267
# Dimension: D7 — 精灵移除方式 (trajectory)
# Change: 所有精灵在输出中完全移除（设为背景色），无轨迹残留... → 精灵移动到框架内部
#
# Design analysis
# a) Concretization + behavior trace
#    - Original task: same-color sprites on the right are counted, removed, and replaced by
#      isolated dots on the center row of each matching hollow frame.
#    - Variant: the matching sprites are not abstracted into dots. They are moved as visible
#      objects into fixed parking slots inside the corresponding frame.
#    - Necessary geometry adaptation: the original 5x11 frame has only a 3x9 interior, which
#      is enough for isolated dots but too cramped for several non-touching moved sprites.
#      To keep the D7 change visually legible, frames become 7x11 so the interior is 5x9.
#    - Moved sprite shape: every sprite is a monochrome 2x1 vertical domino. This preserves
#      the idea of "small connected sprite" while making exact relocation feasible.
#    - Parking slots inside a frame are fixed at four columns, all away from borders:
#      relative top-left offsets (2,2), (2,4), (2,6), (2,8). Each slot hosts one 2x1 domino.
#    - Example trace:
#      old logic: red frame + 3 red sprites -> erase three right-side sprites -> draw three
#      red dots on the frame's middle row.
#      new logic: red frame + 3 red sprites -> erase three right-side sprites -> paste three
#      red vertical dominoes into the first three interior slots of the red frame.
#
# b) Complexity calibration
#    - The matching rule remains only "same color as frame", so the core concept stays close
#      to the original task.
#    - Sprites are uniform in shape, which avoids hidden ordering rules and keeps the change
#      focused on trajectory rather than shape taxonomy.
#    - Capacity is still capped at 4 per frame, so the puzzle keeps repeated evidence across
#      samples without introducing ambiguous crowding.
#
# c) Implicit-assumption review for the original main()
#    - `out = g.copy()` still works: frames are preserved and patterns are later erased.
#    - Component split into `frames` vs `patterns` still works because frames remain large
#      hollow rectangles and sprites remain tiny monochrome components.
#    - The old pattern-removal loop assumed the final representation did not need source
#      sprite geometry. That is no longer enough; the variant must preserve each sprite patch
#      and paste it into a frame.
#    - The old `num_patterns = sum(...)` only needed counts. The variant needs the actual
#      matching component list, because each component is moved, not summarized.
#    - The old dot-placement loop relied on isolated single cells. In the variant, we must
#      ensure destination slots are legal and non-overlapping, so placement is delegated to a
#      fixed slot list inside each frame.
#
# d) Placement legality
#    - Generator places scattered sprites only in the right region, with padding, so they do
#      not overlap each other or the frames.
#    - Output parking slots are precomputed and all fit strictly inside a 7x11 frame.
#    - At most four matching sprites are generated per frame, exactly the number of slots.
#    - Validation remains output-first: build the intended output, derive the input, and keep
#      only samples where `main(input) == output`.
#
# e) Visual observability
#    - Humans can see that right-side domino sprites disappear from the right and reappear,
#      still as dominoes, inside the same-color frame.
#    - This is visibly different from the original tally-dot rule even when only one sprite is
#      moved, because a 2-cell vertical mark replaces a 1-cell dot.
#
# f) Visual cleanliness
#    - Frames are stacked in one column as before.
#    - Scattered sprites stay confined to the right region.
#    - Moved sprites occupy evenly spaced interior slots, so the output does not become noisy.
#
# g) Color inference
#    - No hidden use of numeric color ids exists; the only correspondence is visible color
#      matching between a sprite and a frame.
#
# 9) New assumptions introduced by the variant and how the generator guarantees them
#    - Every movable sprite must fit into a parking slot: generator uses only 2x1 vertical
#      dominoes.
#    - Each frame must have enough room for all moved sprites: generator caps counts at 4.
#    - Matching sprites of one frame need no extra ordering cue: generator makes them identical,
#      so any deterministic ordering yields the same visible output.

import numpy as np
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
BOTTOM_PAD_RANGE = (0, 4)

# --- sprite shape ---
SPRITE_H = 2
SPRITE_W = 1
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((SPRITE_H, SPRITE_W), bg, dtype=int)
    patch[:, 0] = color
    return patch


def _frame_slots(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = map(int, bbox)
    return [(r0 + 2, c0 + 2 + 2 * i) for i in range(MAX_SPRITES_PER_FRAME)]


def _extract_component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    color = int(comp["color"])
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
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
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each small same-color sprite into the interior of its matching frame.
    Distractor sprites are removed.
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
        for slot, comp in zip(slots, matches):
            patch = _extract_component_patch(g, comp, bg)
            paste(out, patch, slot[0], slot[1], transparent=bg)

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
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_boxes: list[tuple[int, int, int, int]] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        c0 = FRAME_C0
        paste(out, _frame_patch(color, bg), r0, c0)
        bbox = (r0, c0, r0 + FRAME_H - 1, c0 + FRAME_W - 1)
        frame_boxes.append(bbox)
        sprite = _sprite_patch(color, bg)
        for slot_r, slot_c in _frame_slots(bbox)[:count]:
            paste(out, sprite, slot_r, slot_c, transparent=bg)

    inp = out.copy()
    for bbox, color, count in zip(frame_boxes, frame_colors, counts):
        sprite = _sprite_patch(color, bg)
        for slot_r, slot_c in _frame_slots(bbox)[:count]:
            paste(inp, np.full(sprite.shape, bg, dtype=int), slot_r, slot_c)

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
