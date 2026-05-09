# Combo variant of 8f215267 (index 24)
# combo_spec: D3/0+D9/2+D10/0
# Name: 不同尺寸框架+网格排列+共享颜色

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste
except ImportError:  # pragma: no cover - robustness for direct file execution
    ROOT = Path("/home/lixinyu/snap/new")
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste


"""
Design analysis for the combo variant D3/0 + D9/2 + D10/0
==========================================================

a) Concrete behavior and trace
- Original task:
  3-4 equal-width hollow frames form one left-side column, each frame color is unique,
  and all small sprites sit off to the right. Output erases the sprites and writes a
  same-color tally on the frame center row from the right edge inward.
- This combo changes all three parts of that pipeline at once:
  1. D9: frames now occupy a 2x2 or 2x3 grid instead of one column, so sprites live in
     the corridors among those frames rather than in a dedicated right strip.
  2. D3: widths vary per frame among {7, 9, 11}, so dot capacity is no longer fixed and
     must be inferred from each frame's actual bbox width.
  3. D10: exactly one color is reused by two frames, so a same-color sprite is assigned
     to the unique nearest frame of that color instead of to "the only frame of that color".
- Small trace:
  suppose the top-left frame and bottom-right frame are both red, with widths 7 and 11.
  Two red sprites sit near the upper one in the central corridor, and one red sprite sits
  near the lower one. The original color-only logic would give both red frames three dots.
  The fused rule instead gives the narrow upper red frame two dots at its legal width-7
  slots, and the wide lower red frame one dot at its width-11 rightmost slot.
- The solver therefore has to use three visible cues together:
  frame color narrows candidates, spatial proximity resolves shared colors, and frame width
  determines where dots can fit.

b) Complexity calibration
- Widths are limited to three odd sizes {7, 9, 11}. That is enough to make size adaptation
  visible without exploding the hypothesis space.
- Layout is limited to 2 columns or 3 columns, always in exactly two rows. This makes the
  "grid instead of stack" change obvious in only a few examples.
- Generation always includes:
  - at least two distinct frame widths,
  - exactly one repeated-color frame pair,
  - both repeated-color frames active with at least one sprite each,
  - at least `n_cols` active frames overall.
  These constraints prevent the fusion dimensions from becoming dormant.

c) Implicit-assumption audit of `main()`
- `find_components(..., monochromatic=True)` still holds because frames and sprites remain
  monochrome connected components even when two frames share a color.
- The original size/sparsity frame heuristic becomes fragile once widths vary and sprites
  move into the frame corridors. The combo solver switches to an exact hollow-rectangle test:
  height must be 5, width must be in {7,9,11}, area must match a perimeter-only rectangle.
- Original counting logic:
    `sum(1 for p in patterns if p["color"] == col)`
  depended on color uniqueness. That assumption is invalid here. The replacement first groups
  same-color frames, then assigns each sprite to the unique nearest such frame by L1 distance
  between bbox centers.
- Original dot placement silently assumed width 11 and capacity 4. The combo solver computes
  legal dot columns from the observed frame width each time.
- New solver assumption introduced here:
  if a sprite color appears on multiple frames, its nearest same-color frame must be unique.
  The generator enforces that by only placing sprites at candidate cells where the nearest
  same-color frame is unique and equals the intended target.

d) Placement legality
- Frames are pasted first into an output grid at fixed slot columns; widths vary but every
  slot is wide enough for the maximum width 11.
- Sprites are placed only inside the union bbox of the frame arrangement after carving out
  every full frame rectangle, so sprites appear in corridors among frames and never inside
  frame interiors.
- A custom feasibility test checks bounds, overlap, and 8-neighbor padding before accepting
  each sprite placement.
- The generator retries if any sprite cannot be placed while respecting the intended nearest
  same-color assignment.

e) Visual observability
- Humans can see the D9 change immediately because frames occupy multiple columns and sprites
  are mixed into the between-frame corridors instead of isolated on the far right.
- Humans can see the D3 change because right frame edges no longer align and dot capacity
  differs between narrow and wide frames.
- Humans can see the D10 change because one color is visibly reused by two frames, yet their
  tallies differ according to nearby same-color sprites.
- No invisible ordering by numeric color id is used. Every decision comes from visible shape,
  color equality, and distance on the grid.

f) Visual cleanliness
- The scene still has only two object types: hollow frames and small monochrome sprites.
- Sprites are excluded from frame interiors, so there is no ambiguity between "count source"
  objects and "count display" dots.
- The corridor-only placement keeps the frame cluster readable even in the denser 2x3 layout.

g) Generator-side anti-degeneration guarantees
- Every sample has a repeated-color pair, so the color-sharing dimension is always present.
- Every sample has at least two distinct widths, so the width-adaptive dimension is always present.
- Every sample places sprites left of the rightmost frame edge inside the frame-union corridors,
  so the layout does not collapse back to the original right-strip organization.
- Every generated sample is validated with `np.array_equal(main(inp), out)`.
"""


FRAME_H = 5
FRAME_WIDTH_CHOICES = (7, 9, 11)
TOP_R0 = 1
BOTTOM_R0 = 13
FRAME_ROWS = (TOP_R0, BOTTOM_R0)
LAYOUT_COLS = {
    2: (4, 24),
    3: (1, 15, 29),
}
GRID_W = 41
BOTTOM_PAD_RANGE = (0, 4)

N_DISTRACTORS_RANGE = (1, 3)
SPRITE_AREA_RANGE = (2, 8)
SPRITE_MAX_H = 3
SPRITE_MAX_W = 3
SPRITE_PADDING = 1
MAX_ATTEMPTS = 500


def _frame_dot_cols(width: int) -> list[int]:
    """Return legal in-frame dot columns for a frame of the given width."""
    return [int(width - 3 - 2 * i) for i in range((int(width) - 3) // 2)]


def _bbox_center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return doubled bbox-center coordinates to keep distances integral."""
    r0, c0, r1, c1 = map(int, bbox)
    return int(r0 + r1), int(c0 + c1)


def _is_hollow_frame(comp: dict) -> bool:
    """Exact recognizer for input-time frames in this combo."""
    b = tuple(map(int, comp["bbox"]))
    h = bbox_h(b)
    w = bbox_w(b)
    if h != FRAME_H or w not in FRAME_WIDTH_CHOICES:
        return False

    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False

    r0, c0, r1, c1 = b
    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    border = set()
    for rr in range(r0, r1 + 1):
        border.add((rr, c0))
        border.add((rr, c1))
    for cc in range(c0, c1 + 1):
        border.add((r0, cc))
        border.add((r1, cc))
    return coords == border


def _distance_l1(center2: tuple[int, int], frame: dict) -> int:
    fr, fc = frame["center2"]
    return int(abs(int(center2[0]) - int(fr)) + abs(int(center2[1]) - int(fc)))


def _unique_nearest_frame(center2: tuple[int, int], frames: list[dict]) -> dict | None:
    """Return the unique nearest frame by L1 distance, or None on ties / empties."""
    if not frames:
        return None
    dists = sorted(_distance_l1(center2, frame) for frame in frames)
    if len(dists) >= 2 and int(dists[0]) == int(dists[1]):
        return None
    return min(frames, key=lambda frame: (_distance_l1(center2, frame), int(frame["index"])))


def _frame_infos_from_grid(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Split components into exact frames and non-frame sprites/distractors."""
    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_hollow_frame(comp):
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
    Remove corridor sprites and display per-frame tallies inside the frames.

    Input structure:
    - The background is the most common color.
    - Hollow rectangular frames of height 5 are arranged in a 2x2 or 2x3 grid.
    - Frame widths vary among 7, 9, and 11, and one color may be reused by two frames.
    - Small monochrome sprites are scattered in the background corridors around the frames.

    Rule:
    - Remove every small sprite from the grid.
    - For each sprite, look only at frames of the same color and assign it to the unique
      nearest such frame using L1 distance between bounding-box centers.
    - For each frame, draw one dot per assigned sprite on the frame's center row, starting
      from the right interior side and stepping left by 2; the number of available slots is
      inferred from that frame's actual width.

    Key insight:
    - Color alone is not enough because two frames can share a color, and location alone is
      not enough because only same-color frames compete. The intended tally comes from
      combining same-color filtering, nearest-frame disambiguation, and width-adaptive dot layout.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _frame_infos_from_grid(g, bg)

    for pattern in patterns:
        for r, c in pattern["coords"]:
            out[int(r), int(c)] = bg

    frames_by_color: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_color.setdefault(int(frame["color"]), []).append(frame)

    counts = [0] * len(frames)
    for pattern in patterns:
        same_color_frames = frames_by_color.get(int(pattern["color"]), [])
        target = _unique_nearest_frame(_bbox_center2(tuple(map(int, pattern["bbox"]))), same_color_frames)
        if target is None:
            continue
        counts[int(target["index"])] += 1

    for frame, count in zip(frames, counts):
        if count <= 0:
            continue
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        width = bbox_w(frame["bbox"])
        r_mid = r0 + FRAME_H // 2
        for dc in _frame_dot_cols(width)[:count]:
            out[int(r_mid), int(c0 + dc)] = int(frame["color"])

    return out


def _frame_patch(color: int, bg: int, width: int, n_dots: int) -> np.ndarray:
    """Build a hollow frame patch with width-adaptive tally dots."""
    patch = np.full((FRAME_H, int(width)), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    patch[mid_r, _frame_dot_cols(width)[:n_dots]] = color
    return patch


def _choose_repeated_pair(rng: np.random.Generator, n_cols: int) -> tuple[int, int]:
    """Choose one repeated-color pair spanning the two frame rows."""
    top = int(rng.integers(0, n_cols))
    bottom = int(rng.integers(0, n_cols)) + n_cols
    return int(top), int(bottom)


def _sample_widths(
    rng: np.random.Generator,
    n_frames: int,
    repeated_pair: tuple[int, int],
) -> list[int]:
    """Sample widths while guaranteeing visible variation and a mixed repeated-color pair."""
    for _ in range(200):
        widths = [int(rng.choice(FRAME_WIDTH_CHOICES)) for _ in range(n_frames)]
        if len(set(widths)) < 2:
            continue
        if widths[int(repeated_pair[0])] == widths[int(repeated_pair[1])]:
            continue
        return widths
    raise ValueError("failed to sample visible width variation")


def _pick_total_match(rng: np.random.Generator, capacities: list[int], n_cols: int) -> int:
    """Choose a total that is visually rich but still easy to pack into corridors."""
    min_match = 4 if n_cols == 2 else 5
    max_match = min(sum(capacities), min_match + 3)
    return int(rng.integers(min_match, max_match + 1))


def _sample_counts(
    rng: np.random.Generator,
    capacities: list[int],
    total_match: int,
    repeated_pair: tuple[int, int],
    min_active: int,
) -> list[int]:
    """Distribute matches while keeping the repeated-color pair active."""
    n_frames = len(capacities)
    rep_a, rep_b = map(int, repeated_pair)
    if total_match < 2:
        raise ValueError("total_match must cover the repeated-color pair")

    others = [idx for idx in range(n_frames) if idx not in repeated_pair]
    for _ in range(300):
        counts = [0] * n_frames
        residual = [int(cap) for cap in capacities]

        for idx in repeated_pair:
            counts[int(idx)] = 1
            residual[int(idx)] -= 1

        max_active = min(n_frames, total_match)
        n_active = int(rng.integers(min_active, max_active + 1))
        extra_needed = max(0, n_active - 2)
        if extra_needed > len(others):
            continue
        if extra_needed > 0:
            extra_active = [int(x) for x in rng.choice(others, size=extra_needed, replace=False)]
            for idx in extra_active:
                if residual[int(idx)] <= 0:
                    break
                counts[int(idx)] = 1
                residual[int(idx)] -= 1
            else:
                pass
                remaining = total_match - sum(counts)
                order = [int(x) for x in rng.permutation(n_frames)]
                for pos, idx in enumerate(order):
                    future_cap = sum(residual[j] for j in order[pos + 1 :])
                    lo = max(0, remaining - future_cap)
                    hi = min(residual[int(idx)], remaining)
                    add = int(rng.integers(lo, hi + 1))
                    counts[int(idx)] += add
                    residual[int(idx)] -= add
                    remaining -= add
                if remaining != 0:
                    continue
                if sum(v > 0 for v in counts) < min_active:
                    continue
                if counts[rep_a] == capacities[rep_a] and counts[rep_b] == capacities[rep_b]:
                    continue
                return counts
            continue

        remaining = total_match - sum(counts)
        order = [int(x) for x in rng.permutation(n_frames)]
        for pos, idx in enumerate(order):
            future_cap = sum(residual[j] for j in order[pos + 1 :])
            lo = max(0, remaining - future_cap)
            hi = min(residual[int(idx)], remaining)
            add = int(rng.integers(lo, hi + 1))
            counts[int(idx)] += add
            residual[int(idx)] -= add
            remaining -= add
        if remaining == 0 and sum(v > 0 for v in counts) >= min_active:
            if counts[rep_a] == capacities[rep_a] and counts[rep_b] == capacities[rep_b]:
                continue
            return counts

    raise ValueError("failed to sample counts")


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino inside an h x w bounding box."""
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    """Generate one small connected monochrome sprite."""
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
        patch[int(r - r_min), int(c - c_min)] = color
    return patch


def _build_frame_colors(
    rng: np.random.Generator,
    n_frames: int,
    n_distractors: int,
    repeated_pair: tuple[int, int],
) -> tuple[int, list[int], list[int]]:
    """Assign exactly one repeated color pair and distinct distractor colors."""
    unique_frame_colors = n_frames - 1
    palette = rng.choice(np.arange(10), size=1 + unique_frame_colors + n_distractors, replace=False)
    bg = int(palette[0])
    frame_palette = [int(x) for x in palette[1 : 1 + unique_frame_colors]]
    distractor_colors = [int(x) for x in palette[1 + unique_frame_colors :]]

    repeated_color = int(frame_palette[0])
    other_colors = [int(x) for x in frame_palette[1:]]
    rng.shuffle(other_colors)

    frame_colors = [-1] * n_frames
    for idx in repeated_pair:
        frame_colors[int(idx)] = repeated_color
    ptr = 0
    for idx in range(n_frames):
        if frame_colors[idx] == -1:
            frame_colors[idx] = int(other_colors[ptr])
            ptr += 1
    return bg, frame_colors, distractor_colors


def _build_allowed_mask(
    shape: tuple[int, int],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Allow only corridor cells inside the overall frame arrangement."""
    allowed = np.zeros(shape, dtype=bool)
    r0 = min(b[0] for b in frame_bboxes)
    c0 = min(b[1] for b in frame_bboxes)
    r1 = max(b[2] for b in frame_bboxes)
    c1 = max(b[3] for b in frame_bboxes)
    allowed[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    for fr0, fc0, fr1, fc1 in frame_bboxes:
        allowed[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1] = False
    return allowed


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int) -> bool:
    """Check bounds, overlap, and local padding on a placement canvas."""
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
    """Return doubled global bbox-center coordinates for a placed sprite patch."""
    h, w = map(int, patch.shape)
    return int(2 * r0 + h - 1), int(2 * c0 + w - 1)


def _place_targeted_sprite(
    rng: np.random.Generator,
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    target_frame: dict,
    same_color_frames: list[dict],
) -> bool:
    """Place a sprite so that the intended target is its unique nearest same-color frame."""
    H, W = map(int, canvas.shape)
    h, w = map(int, sprite.shape)
    candidates: list[tuple[int, int, int]] = []

    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not _can_place(canvas, sprite, r0, c0, bg, SPRITE_PADDING):
                continue
            center2 = _sprite_center2_global(r0, c0, sprite)
            nearest = _unique_nearest_frame(center2, same_color_frames)
            if nearest is None or int(nearest["index"]) != int(target_frame["index"]):
                continue
            dist = _distance_l1(center2, target_frame)
            candidates.append((int(dist), int(r0), int(c0)))

    if not candidates:
        return False

    candidates.sort(key=lambda item: item[0])
    best = int(candidates[0][0])
    shortlist = [(r0, c0) for dist, r0, c0 in candidates if dist <= best + 8]
    pick = shortlist[int(rng.integers(0, len(shortlist)))]
    paste(canvas, sprite, int(pick[0]), int(pick[1]), transparent=bg)
    return True


def _place_distractor_sprite(
    rng: np.random.Generator,
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
) -> bool:
    """Place a distractor sprite anywhere feasible in the corridor canvas."""
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
    paste(canvas, sprite, int(r0), int(c0), transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_cols = int(rng.choice([2, 3]))
    frame_cols = [int(x) for x in LAYOUT_COLS[n_cols]]
    n_frames = 2 * n_cols
    repeated_pair = _choose_repeated_pair(rng, n_cols)
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))

    try:
        frame_widths = _sample_widths(rng, n_frames, repeated_pair)
    except ValueError:
        return None
    capacities = [len(_frame_dot_cols(width)) for width in frame_widths]
    total_match = _pick_total_match(rng, capacities, n_cols)

    try:
        counts = _sample_counts(
            rng,
            capacities,
            total_match,
            repeated_pair,
            min_active=n_cols,
        )
    except ValueError:
        return None

    bg, frame_colors, distractor_colors = _build_frame_colors(rng, n_frames, n_distractors, repeated_pair)
    height = BOTTOM_R0 + FRAME_H + int(rng.integers(*BOTTOM_PAD_RANGE))

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_infos: list[dict] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []

    idx = 0
    for r0 in FRAME_ROWS:
        for c0 in frame_cols:
            width = int(frame_widths[idx])
            color = int(frame_colors[idx])
            count = int(counts[idx])
            paste(out, _frame_patch(color, bg, width, count), r0, c0)
            bbox = (int(r0), int(c0), int(r0 + FRAME_H - 1), int(c0 + width - 1))
            frame_bboxes.append(bbox)
            frame_infos.append(
                {
                    "index": int(idx),
                    "color": int(color),
                    "bbox": bbox,
                    "center2": _bbox_center2(bbox),
                }
            )
            idx += 1

    inp = out.copy()
    for frame, count, width in zip(frame_infos, counts, frame_widths):
        if int(count) <= 0:
            continue
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        r_mid = r0 + FRAME_H // 2
        for dc in _frame_dot_cols(int(width))[: int(count)]:
            inp[int(r_mid), int(c0 + dc)] = bg

    allowed_mask = _build_allowed_mask(inp.shape, frame_bboxes)
    placement_canvas = np.full(inp.shape, -1, dtype=int)
    placement_canvas[allowed_mask] = bg

    frames_by_color: dict[int, list[dict]] = {}
    for frame in frame_infos:
        frames_by_color.setdefault(int(frame["color"]), []).append(frame)

    assignments = [idx for idx, count in enumerate(counts) for _ in range(int(count))]
    rng.shuffle(assignments)
    for frame_idx in assignments:
        target_frame = frame_infos[int(frame_idx)]
        sprite = _make_sprite(rng, int(target_frame["color"]), bg)
        if not _place_targeted_sprite(
            rng,
            placement_canvas,
            sprite,
            bg,
            target_frame,
            frames_by_color[int(target_frame["color"])],
        ):
            return None

    distractor_palette = [int(color) for color in distractor_colors]
    rng.shuffle(distractor_palette)
    for color in distractor_palette:
        sprite = _make_sprite(rng, int(color), bg)
        if not _place_distractor_sprite(rng, placement_canvas, sprite, bg):
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
