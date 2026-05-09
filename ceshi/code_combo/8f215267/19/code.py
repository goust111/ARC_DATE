# Combo variant of 8f215267 (index 19)
# combo_spec: D10/0+D15/1
# Name: 共享颜色+图例颜色映射

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste

"""
Design analysis for the combo variant
=====================================

a) concretized behavior and trace
---------------------------------
The original task counted sprites by direct color equality with frames.
This combo breaks that in two separate places:

- First break (legend remapping): rows 0-1 hold a 2xN legend on the top-right.
  Each legend column says sprite-color -> frame-color.
- Second break (shared frame color): exactly one pair of frames shares the same
  border color, so the mapped frame color alone does not identify a unique frame.

Small trace:
- Original behavior:
    one blue frame + three blue sprites -> that frame gets 3 blue dots.
- Combo behavior:
    two blue frames exist, and the legend contains:
      orange -> blue
      red    -> blue
    orange sprites are placed near the upper blue frame.
    red sprites are placed near the lower blue frame.
    Solver must:
      1. use the legend to map orange/red sprites to the blue frame-color class
      2. within the two blue frames, assign each sprite to the unique nearest one
    Old direct-color counting fails because orange/red are not blue.
    A legend-only solver without spatial splitting also fails because it would
    give both blue frames the total combined blue-class count.

b) complexity calibration
-------------------------
- Exactly one duplicated frame-color pair is used in every sample. More repeated
  colors would make the grid noisy and underconstrained.
- The two repeated-color frames are forced to end with different dot counts, so
  the spatial split is visible in the output rather than only latent in input.
- The legend has exactly one column per frame slot. Therefore the legend top row
  uses N distinct sprite colors, while the bottom row may repeat the shared frame
  color twice. This makes the duplicated target color visible without requiring
  hidden bookkeeping.
- Counted sprite colors are disjoint from all frame colors, so legend lookup is
  always necessary rather than occasionally optional.
- Both repeated-color frames are forced to receive at least one sprite, and at
  least one non-repeated frame is also active. This keeps both sub-rules visible.

c) implicit assumption audit for main()
---------------------------------------
- Connected-component parsing still works because frames remain large hollow
  monochrome rectangles and sprites remain much smaller connected components.
- The original "count same-color sprites per frame" assumption is invalid twice:
  sprite color is no longer the target frame color, and a target frame color may
  identify multiple frames.
- The original "every small component is a removable sprite" assumption is no
  longer safe because legend cells are also tiny components. main() must detect
  and preserve the legend before removing patterns.
- Nearest-frame assignment introduces a new assumption: for any counted sprite
  whose mapped frame color appears on two frames, one of those frames must be the
  unique nearest under L1 distance between bounding-box centers. The generator
  enforces this by rejecting tie positions.

d) placement legality
---------------------
- Frames stay on the left in a fixed 5x11 geometry with 2 blank rows between
  them, starting at row 2 so rows 0-1 are reserved for the legend.
- Sprites are placed only in the right region below the legend.
- A custom placement search checks bounds, overlap, and 8-neighbor padding.
- Targeted sprites are only accepted at positions whose mapped-color nearest
  frame is exactly the intended frame.

e) visual observability
-----------------------
- The legend is explicit and persistent in both input and output.
- The repeated frame color is directly visible on two separate frames and also
  repeated in the legend bottom row.
- Because counted sprite colors never equal frame colors, a viewer can see that
  direct chromatic matching is wrong. Because the repeated frame color is active
  in both copies, a viewer can also see that legend lookup alone is insufficient.

f) visual cleanliness
---------------------
- Left area = frames only.
- Top-right strip = legend only.
- Right/below legend = sprites only.
- No overlays, mixed-color symbols, or same-cell reuse are introduced.

g) inferability of color changes
--------------------------------
- The legend gives a stable, low-arity codebook with only 3-4 columns.
- Each column uses visible adjacency (top cell over bottom cell) to encode the
  mapping, rather than relying on color ids or ordering by numeric value.
- Repeated bottom-row colors in the legend tell the solver that one mapped target
  color can correspond to more than one frame, which motivates the second-stage
  spatial split.

new main assumptions guaranteed by generate_input()
---------------------------------------------------
- Rows 0-1 contain only legend cells; sprites start below row 1.
- Legend top-row colors are unique and absent from all frame colors.
- Exactly one frame-color pair is duplicated; both copies are active.
- The duplicated-color frames have different final tallies.
- At least one non-duplicated frame is also active.
- Every counted sprite has a unique nearest frame within its mapped frame-color
  class under the same center-distance metric used by main().
"""


# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 2
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = GRID_W - RIGHT_C0
LEGEND_R0 = 0
LEGEND_H = 2

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 6)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 600


def _bbox_center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return doubled bbox-center coordinates to avoid float arithmetic."""
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
    dists = sorted(
        abs(int(center2[0]) - int(frame["center2"][0])) +
        abs(int(center2[1]) - int(frame["center2"][1]))
        for frame in frames
    )
    if len(dists) >= 2 and int(dists[0]) == int(dists[1]):
        return None
    return min(frames, key=lambda frame: _distance_key(center2, frame))


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    """Read the top-edge 2xN legend as sprite-color -> frame-color."""
    if grid.shape[0] < LEGEND_H:
        return {}, set()

    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    for c in range(grid.shape[1]):
        top = int(grid[LEGEND_R0, c])
        bottom = int(grid[LEGEND_R0 + 1, c])
        if top == bg and bottom == bg:
            continue
        if top == bg or bottom == bg:
            continue
        mapping[top] = bottom
        legend_cells.add((LEGEND_R0, c))
        legend_cells.add((LEGEND_R0 + 1, c))
    return mapping, legend_cells


def _frame_infos_from_grid(
    grid: np.ndarray,
    bg: int,
    legend_cells: set[tuple[int, int]],
) -> tuple[list[dict], list[dict]]:
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if coords and all(cell in legend_cells for cell in coords):
            continue
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


def _draw_dots(out: np.ndarray, frame: dict, n_dots: int) -> None:
    if n_dots <= 0:
        return
    r0, c0, _r1, c1 = map(int, frame["bbox"])
    r_mid = r0 + FRAME_H // 2
    for i in range(int(n_dots)):
        cc = int(c1 - 2 - 2 * i)
        if cc > int(c0):
            out[int(r_mid), int(cc)] = int(frame["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count legend-mapped sprites into frames, splitting shared target colors by proximity.

    Input structure:
    - The background is the most common color.
    - Rows 0-1 contain a 2xN legend on the right: top cells are sprite colors and
      bottom cells are the corresponding frame colors.
    - The left side contains 3-4 hollow 5x11 frames; exactly one color is reused by
      two different frame positions.
    - Small monochrome sprites occupy the right side below the legend, and some may
      be distractors whose colors are absent from the legend.

    Rule:
    - Read the legend first to convert each sprite color into a target frame color.
    - Remove all non-legend small components from the grid.
    - For each counted sprite, consider only frames whose border color matches the
      legend-mapped target color.
    - If that target color appears on multiple frames, assign the sprite to the
      unique nearest one by L1 distance between bounding-box centers.
    - Draw one dot per assigned sprite on the chosen frame's center row, moving left
      from the right interior edge in steps of 2.

    Key insight:
    - Matching is two-stage rather than direct: sprite color -> legend target color ->
      nearest frame within that color class. The duplicated frame color makes both
      the legend and the spatial cue necessary.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    sprite_to_frame, legend_cells = _find_legend(g, bg)
    frames, patterns = _frame_infos_from_grid(g, bg, legend_cells)

    for pattern in patterns:
        for r, c in pattern["coords"]:
            out[int(r), int(c)] = bg

    frames_by_color: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_color.setdefault(int(frame["color"]), []).append(frame)

    counts = [0] * len(frames)
    for pattern in patterns:
        target_color = sprite_to_frame.get(int(pattern["color"]))
        if target_color is None:
            continue
        target = _unique_nearest_frame(
            _bbox_center2(tuple(map(int, pattern["bbox"]))),
            frames_by_color.get(int(target_color), []),
        )
        if target is None:
            continue
        counts[int(target["index"])] += 1

    for frame, count in zip(frames, counts):
        _draw_dots(out, frame, int(count))

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[FRAME_H // 2, dot_cols] = color
    return patch


def _sample_counts(
    rng: np.random.Generator,
    n_frames: int,
    total_match: int,
    repeated_pair: tuple[int, int],
) -> list[int]:
    """
    Distribute counts while forcing both repeated-color frames and one unique-color
    frame to be active, and force the repeated-color frames to have distinct
    tallies so the spatial split is visible in the output.
    """
    unique_slots = [idx for idx in range(n_frames) if idx not in repeated_pair]
    if total_match < 4:
        raise ValueError("total_match must be >= 4 for this combo")

    for _ in range(200):
        counts = [0] * n_frames
        capacities = [MAX_DOTS_PER_FRAME] * n_frames

        forced_extra = int(unique_slots[int(rng.integers(0, len(unique_slots)))])
        forced_slots = [int(repeated_pair[0]), int(repeated_pair[1]), forced_extra]
        for idx in forced_slots:
            counts[idx] += 1
            capacities[idx] -= 1

        higher_rep = int(repeated_pair[int(rng.integers(0, 2))])
        counts[higher_rep] += 1
        capacities[higher_rep] -= 1

        remaining = total_match - 4
        order = [int(x) for x in rng.permutation(n_frames)]
        for pos, idx in enumerate(order):
            future_cap = sum(capacities[j] for j in order[pos + 1 :])
            lo = max(0, remaining - future_cap)
            hi = min(capacities[idx], remaining)
            add = int(rng.integers(lo, hi + 1))
            counts[idx] += add
            remaining -= add

        if remaining != 0:
            continue
        if counts[int(repeated_pair[0])] != counts[int(repeated_pair[1])]:
            return counts

    raise ValueError("failed to allocate distinct repeated-frame tallies")


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


def _legend_patch(sprite_colors: list[int], frame_colors: list[int], bg: int) -> np.ndarray:
    patch = np.full((LEGEND_H, len(sprite_colors)), bg, dtype=int)
    patch[0, :] = np.array(sprite_colors, dtype=int)
    patch[1, :] = np.array(frame_colors, dtype=int)
    return patch


def _choose_repeated_pair(rng: np.random.Generator, n_frames: int) -> tuple[int, int]:
    if n_frames == 3:
        return 0, 2
    pairs = [(0, 2), (0, 3), (1, 3)]
    return pairs[int(rng.integers(0, len(pairs)))]


def _build_frame_colors(
    rng: np.random.Generator,
    n_frames: int,
    repeated_pair: tuple[int, int],
    frame_palette: list[int],
) -> list[int]:
    repeated_color = int(frame_palette[0])
    other_colors = [int(x) for x in frame_palette[1:]]
    rng.shuffle(other_colors)

    frame_colors = [-1] * n_frames
    for idx in repeated_pair:
        frame_colors[int(idx)] = repeated_color

    other_slots = [idx for idx in range(n_frames) if idx not in repeated_pair]
    for idx, color in zip(other_slots, other_colors):
        frame_colors[int(idx)] = int(color)
    return [int(x) for x in frame_colors]


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
    return int(2 * (LEGEND_H + r0) + h - 1), int(2 * (RIGHT_C0 + c0) + w - 1)


def _place_targeted_sprite(
    rng: np.random.Generator,
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    target_frame: dict,
    same_color_frames: list[dict],
) -> bool:
    H, W = map(int, canvas.shape)
    h, w = map(int, sprite.shape)

    for row_limit2 in (8, 12, None):
        candidates: list[tuple[int, int]] = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if not _can_place(canvas, sprite, r0, c0, bg, SPRITE_PADDING):
                    continue
                center2 = _sprite_center2_global(r0, c0, sprite)
                if row_limit2 is not None:
                    if abs(int(center2[0]) - int(target_frame["center2"][0])) > int(row_limit2):
                        continue
                nearest = _unique_nearest_frame(center2, same_color_frames)
                if nearest is None or int(nearest["index"]) != int(target_frame["index"]):
                    continue
                candidates.append((int(r0), int(c0)))
        if candidates:
            r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
            paste(canvas, sprite, r0, c0, transparent=bg)
            return True
    return False


def _place_distractor_sprite(
    rng: np.random.Generator,
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
) -> bool:
    H, W = map(int, canvas.shape)
    h, w = map(int, sprite.shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if _can_place(canvas, sprite, r0, c0, bg, SPRITE_PADDING):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return False
    r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
    paste(canvas, sprite, r0, c0, transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors
    if total_match < 4:
        return None

    repeated_pair = _choose_repeated_pair(rng, n_frames)

    n_unique_frame_colors = n_frames - 1
    n_needed_colors = 1 + n_frames + n_unique_frame_colors + n_distractors
    palette = [int(x) for x in rng.choice(np.arange(10), size=n_needed_colors, replace=False)]
    bg = int(palette[0])
    frame_sprite_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    frame_palette = [int(x) for x in palette[1 + n_frames : 1 + n_frames + n_unique_frame_colors]]
    distractor_colors = [int(x) for x in palette[1 + n_frames + n_unique_frame_colors :]]

    frame_colors = _build_frame_colors(rng, n_frames, repeated_pair, frame_palette)
    counts = _sample_counts(rng, n_frames, total_match, repeated_pair)
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)

    legend_order = [int(x) for x in rng.permutation(n_frames)]
    legend_sprite_colors = [int(frame_sprite_colors[idx]) for idx in legend_order]
    legend_frame_colors = [int(frame_colors[idx]) for idx in legend_order]
    paste(out, _legend_patch(legend_sprite_colors, legend_frame_colors, bg), LEGEND_R0, RIGHT_C0)

    frame_infos: list[dict] = []
    for idx, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + idx * (FRAME_H + FRAME_GAP)
        paste(out, _frame_patch(int(color), bg, int(count)), r0, FRAME_C0)
        bbox = (r0, FRAME_C0, r0 + FRAME_H - 1, FRAME_C0 + FRAME_W - 1)
        frame_infos.append(
            {
                "index": int(idx),
                "color": int(color),
                "bbox": bbox,
                "center2": _bbox_center2(bbox),
            }
        )

    inp = out.copy()
    for frame, count in zip(frame_infos, counts):
        r0 = int(frame["bbox"][0])
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[r0 + FRAME_H // 2, dot_cols] = bg

    frames_by_color: dict[int, list[dict]] = {}
    for frame in frame_infos:
        frames_by_color.setdefault(int(frame["color"]), []).append(frame)

    sprite_canvas = np.full((height - LEGEND_H, RIGHT_W), bg, dtype=int)
    assignments = [idx for idx, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(assignments)

    for frame_idx in assignments:
        target_frame = frame_infos[int(frame_idx)]
        sprite_color = int(frame_sprite_colors[int(frame_idx)])
        sprite = _make_sprite(rng, sprite_color, bg)
        if not _place_targeted_sprite(
            rng,
            sprite_canvas,
            sprite,
            bg,
            target_frame,
            frames_by_color[int(target_frame["color"])],
        ):
            return None

    rng.shuffle(distractor_colors)
    for color in distractor_colors:
        sprite = _make_sprite(rng, int(color), bg)
        if not _place_distractor_sprite(rng, sprite_canvas, sprite, bg):
            return None

    paste(inp, sprite_canvas, LEGEND_H, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
