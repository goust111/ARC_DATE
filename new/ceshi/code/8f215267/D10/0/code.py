# Variant of 8f215267
# Dimension: D10 — 颜色使用约束 (meta)
# Change: 每种颜色最多用于一个框架，干扰精灵使用不与任何框架重复的颜色；总颜色数 = 1(背景) + 3-4(框架) + 1-2(... → 多框架共享颜色

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste

"""
Design analysis for the D10 variant
===================================

source_task:
  none

variant_goal:
  Preserve the original visual language:
  - 5x11 hollow frames stacked on the left
  - small monochrome sprites on the right
  - output shows dots on each frame's center row
  But remove the old "frame color is unique" assumption.
  Now exactly one pair of frames may share a color, and sprites of that color
  are assigned to the spatially nearest same-color frame.

concretized_behavior:
  - Input still has 3-4 frames and 1-2 distractor sprites with non-frame colors.
  - One repeated frame color appears on two different frame positions.
  - Sprites matching that repeated color are placed near one of the two frames,
    so the disambiguation signal is spatial, not chromatic.
  - Output removes all sprites, then each frame shows the number of sprites that
    were assigned to it.

small_trace:
  original:
    two blue sprites anywhere on the right + one blue frame -> blue frame gets 2 dots
  variant:
    two blue frames at different rows + three blue sprites:
      one near the upper blue frame, two near the lower blue frame
    new solver:
      upper blue frame gets 1 dot, lower blue frame gets 2 dots
    old solver would fail:
      both blue frames would incorrectly get 3 dots, because it only counts by color

complexity_calibration:
  - Use exactly one duplicated frame-color pair per sample.
    This keeps the visual change obvious without creating many symmetric cases.
  - Keep distractor colors disjoint from all frame colors so the only new source
    of ambiguity is "which same-color frame is nearest".
  - Require both repeated-color frames to receive at least one sprite, so the new
    behavior is visible in the output instead of being a dormant possibility.

implicit_assumption_audit_for_main:
  - find_components(..., monochromatic=True):
      still valid because frames remain disconnected monochrome components even if
      two frames share a color.
  - frame/pattern split by bbox and sparsity:
      still valid because frame geometry is unchanged and sprites stay much smaller.
  - original per-frame count:
      `sum(1 for p in patterns if p["color"] == col)` depended on color uniqueness.
      This is the exact line that breaks under the variant.
  - drawing dots into each frame:
      still valid once counts are computed per frame instead of per color.

placement_legality:
  - Sprites are still confined to the right-side region.
  - A custom placement search enforces no overlap and 8-neighbor padding.
  - For repeated-color sprites, candidate positions are filtered so the intended
    frame is the unique nearest same-color frame under L1 distance between bbox centers.
    Ties are rejected during generation, so the solver never relies on an invisible
    tie-break to match the training distribution.

visual_observability:
  - The duplicated color is visible directly on two separate frames.
  - Same-color sprites are placed close to one repeated-color frame or the other,
    so a human can see that location, not color uniqueness, decides the tally.
  - The repeated-color pair is forced to be active in every generated sample:
    both frames receive at least one matching sprite.

cleanliness:
  - Left region remains reserved for frames, right region for sprites.
  - No new overlays or multi-color objects are introduced.
  - The output remains a clean "frames with dots" display.

new_main_assumptions_and_generator_guarantees:
  - main assumes each matching sprite has a unique nearest same-color frame under
    the chosen distance metric. generate_input guarantees this by rejecting ties.
  - main assumes frames are sorted consistently before indexing. It sorts them
    top-to-bottom, and the generator constructs the intended counts in the same order.
"""


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
MAX_ATTEMPTS = 500


def _bbox_center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return doubled bbox center coordinates to avoid floats."""
    r0, c0, r1, c1 = map(int, bbox)
    return int(r0 + r1), int(c0 + c1)


def _is_frame_component(comp: dict) -> bool:
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    area = int(comp["area"])
    return (h >= 5) and (w >= 5) and (area < (h * w * 0.7))


def _distance_key(center2: tuple[int, int], frame: dict) -> tuple[int, int, int, int]:
    fr, fc = frame["center2"]
    dr = abs(int(center2[0]) - int(fr))
    dc = abs(int(center2[1]) - int(fc))
    return int(dr + dc), int(dr), int(dc), int(frame["index"])


def _unique_nearest_frame(center2: tuple[int, int], frames: list[dict]) -> dict | None:
    if not frames:
        return None
    dists = sorted(abs(int(center2[0]) - int(f["center2"][0])) +
                   abs(int(center2[1]) - int(f["center2"][1])) for f in frames)
    if len(dists) >= 2 and int(dists[0]) == int(dists[1]):
        return None
    return min(frames, key=lambda f: _distance_key(center2, f))


def _frame_infos_from_grid(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_frame_component(comp):
            frames.append(comp)
        else:
            patterns.append(comp)

    frames.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    frame_infos: list[dict] = []
    for idx, comp in enumerate(frames):
        info = dict(comp)
        info["index"] = int(idx)
        info["center2"] = _bbox_center2(tuple(map(int, comp["bbox"])))
        frame_infos.append(info)
    return frame_infos, patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Remove all small sprites from the right region.
    - For each sprite, consider only frames of the same color.
    - Assign the sprite to the unique nearest same-color frame by L1 distance
      between bounding-box centers.
    - Draw one dot per assigned sprite on that frame's center row, from the right.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _frame_infos_from_grid(g, bg)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    frames_by_color: dict[int, list[dict]] = {}
    for f in frames:
        frames_by_color.setdefault(int(f["color"]), []).append(f)

    counts = [0] * len(frames)
    for p in patterns:
        same_color_frames = frames_by_color.get(int(p["color"]), [])
        target = _unique_nearest_frame(_bbox_center2(tuple(map(int, p["bbox"]))), same_color_frames)
        if target is None:
            continue
        counts[int(target["index"])] += 1

    for f, num_patterns in zip(frames, counts):
        if num_patterns <= 0:
            continue
        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = int(f["color"])

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(
    rng: np.random.Generator,
    n_frames: int,
    total_match: int,
    repeated_pair: tuple[int, int],
) -> list[int]:
    """Distribute counts while forcing both repeated-color frames to be active."""
    if total_match < 2:
        raise ValueError("total_match must be >= 2 for the repeated-color pair")

    counts = [0] * n_frames
    capacities = [MAX_DOTS_PER_FRAME] * n_frames
    for idx in repeated_pair:
        counts[int(idx)] = 1
        capacities[int(idx)] -= 1
    remaining = total_match - 2

    order = [int(x) for x in rng.permutation(n_frames)]
    for pos, idx in enumerate(order):
        future_cap = sum(capacities[j] for j in order[pos + 1 :])
        lo = max(0, remaining - future_cap)
        hi = min(capacities[idx], remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        remaining -= add

    if remaining != 0:
        raise ValueError("failed to allocate all matching counts")
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
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


def _choose_repeated_pair(rng: np.random.Generator, n_frames: int) -> tuple[int, int]:
    if n_frames == 3:
        return 0, 2
    pairs = [(0, 2), (0, 3), (1, 3)]
    return pairs[int(rng.integers(0, len(pairs)))]


def _build_frame_colors(
    rng: np.random.Generator,
    n_frames: int,
    n_distractors: int,
    repeated_pair: tuple[int, int],
) -> tuple[int, list[int], list[int]]:
    unique_frame_colors = n_frames - 1
    palette = rng.choice(np.arange(10), size=1 + unique_frame_colors + n_distractors, replace=False)
    bg = int(palette[0])
    frame_palette = [int(x) for x in palette[1 : 1 + unique_frame_colors]]
    distractor_colors = [int(x) for x in palette[1 + unique_frame_colors :]]

    repeated_color = int(frame_palette[0])
    other_colors = frame_palette[1:]
    frame_colors = [-1] * n_frames
    for idx in repeated_pair:
        frame_colors[int(idx)] = repeated_color
    other_slots = [i for i in range(n_frames) if i not in repeated_pair]
    rng.shuffle(other_colors)
    for idx, color in zip(other_slots, other_colors):
        frame_colors[int(idx)] = int(color)
    return bg, frame_colors, distractor_colors


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int) -> bool:
    h, w = map(int, patch.shape)
    H, W = map(int, canvas.shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    fg = np.argwhere(patch != bg)
    if fg.size == 0:
        return False

    for dr, dc in fg:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        for nr in range(max(0, rr - padding), min(H, rr + padding + 1)):
            for nc in range(max(0, cc - padding), min(W, cc + padding + 1)):
                if int(canvas[nr, nc]) != bg:
                    return False
    return True


def _sprite_center2_global(r0: int, c0: int, patch: np.ndarray) -> tuple[int, int]:
    h, w = map(int, patch.shape)
    return int(2 * r0 + h - 1), int(2 * (RIGHT_C0 + c0) + w - 1)


def _place_targeted_sprite(
    rng: np.random.Generator,
    right_canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    target_frame: dict,
    same_color_frames: list[dict],
) -> bool:
    H, W = map(int, right_canvas.shape)
    h, w = map(int, sprite.shape)

    for row_limit2 in (8, 12, None):
        candidates: list[tuple[int, int]] = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if not _can_place(right_canvas, sprite, r0, c0, bg, SPRITE_PADDING):
                    continue
                center2 = _sprite_center2_global(r0, c0, sprite)
                if row_limit2 is not None and abs(center2[0] - int(target_frame["center2"][0])) > row_limit2:
                    continue
                nearest = _unique_nearest_frame(center2, same_color_frames)
                if nearest is None or int(nearest["index"]) != int(target_frame["index"]):
                    continue
                candidates.append((int(r0), int(c0)))
        if candidates:
            r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
            paste(right_canvas, sprite, r0, c0, transparent=bg)
            return True
    return False


def _place_distractor_sprite(
    rng: np.random.Generator,
    right_canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
) -> bool:
    H, W = map(int, right_canvas.shape)
    h, w = map(int, sprite.shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if _can_place(right_canvas, sprite, r0, c0, bg, SPRITE_PADDING):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return False
    r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
    paste(right_canvas, sprite, r0, c0, transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    repeated_pair = _choose_repeated_pair(rng, n_frames)
    bg, frame_colors, distractor_colors = _build_frame_colors(rng, n_frames, n_distractors, repeated_pair)
    counts = _sample_counts(rng, n_frames, total_match, repeated_pair)

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_infos: list[dict] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        paste(out, _frame_patch(int(color), bg, int(count)), r0, FRAME_C0)
        bbox = (r0, FRAME_C0, r0 + FRAME_H - 1, FRAME_C0 + FRAME_W - 1)
        frame_infos.append(
            {
                "index": int(i),
                "color": int(color),
                "bbox": bbox,
                "center2": _bbox_center2(bbox),
            }
        )

    inp = out.copy()
    for frame, count in zip(frame_infos, counts):
        r0 = int(frame["bbox"][0])
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    frames_by_color: dict[int, list[dict]] = {}
    for frame in frame_infos:
        frames_by_color.setdefault(int(frame["color"]), []).append(frame)

    assignments = [idx for idx, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(assignments)

    for frame_idx in assignments:
        target_frame = frame_infos[int(frame_idx)]
        sprite = _make_sprite(rng, int(target_frame["color"]), bg)
        if not _place_targeted_sprite(
            rng,
            right_canvas,
            sprite,
            bg,
            target_frame,
            frames_by_color[int(target_frame["color"])],
        ):
            return None

    distractor_palette = list(distractor_colors)
    rng.shuffle(distractor_palette)
    for color in distractor_palette:
        sprite = _make_sprite(rng, int(color), bg)
        if not _place_distractor_sprite(rng, right_canvas, sprite, bg):
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
