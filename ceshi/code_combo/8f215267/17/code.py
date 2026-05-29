# Combo variant of 8f215267 (index 17)
# combo_spec: D9/2+D10/0
# Name: 网格排列+共享颜色

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste

"""
Design analysis for the combo variant
=====================================

source_task:
  8f215267

variant_goal:
  Merge two changes into one coherent rule:
  - D9/2 changes the frame arrangement from one vertical stack into a 2x2 grid.
  - D10/0 removes color uniqueness, so one pair of frames may share a color and
    matching sprites must be assigned by nearest same-color frame.
  The fusion must make both signals visible at once rather than leaving either as
  a dormant possibility.

a) concretized_behavior_and_trace:
  - Original:
      three or four tall 5x11 frames sit in one column on the left, all frame
      colors are unique, and every same-color sprite anywhere on the right simply
      increments the one frame of that color.
  - Fused variant:
      there are exactly four smaller 5x9 hollow frames at top-left, top-right,
      bottom-left, bottom-right. One diagonal pair shares a color. Sprites live
      only in the corridors between those frame rectangles.
  - Small trace:
      suppose top-left and bottom-right are both blue, top-right is red, and
      bottom-left is yellow. Two blue sprites are placed near the upper-left
      corridor and one blue sprite is placed near the lower-right corridor.
      Output:
        top-left blue frame gets 2 dots
        bottom-right blue frame gets 1 dot
        red and yellow frames show their own same-color counts
      The old solver would give both blue frames 3 dots, so color-only counting
      is no longer correct.

b) complexity_calibration:
  - Keep the layout fixed to 2x2 rather than mixing 2x2 and 2x3. This makes the
    spatial rule easy to observe while still being clearly different from the
    original one-column layout.
  - Use exactly one repeated-color pair per sample, and force both of those frames
    to receive at least one sprite. That keeps the shared-color behavior active in
    every generated example.
  - Frames shrink from width 11 to width 9 so the 2x2 layout plus visible corridor
    space fits cleanly without clutter. Dot capacity becomes 3, which is enough to
    vary counts while staying legible.

c) implicit_assumption_review_of_main:
  - Original frame detection by "large sparse component" becomes fragile after the
    layout change, because more objects live in one compact zone. The fused solver
    switches to an exact hollow-rectangle test with the generator's fixed size.
  - Original counting assumed at most one frame per color. That assumption is now
    false. Counting must be per frame after assigning each sprite to its unique
    nearest same-color frame.
  - Original sprite location was irrelevant because all matching used only color.
    Spatial location is now semantically meaningful, so the solver uses L1 distance
    between component bbox centers and ignores ambiguous ties.

d) placement_legality:
  - Frames are pasted first into the output grid.
  - Input is derived by removing dots.
  - Sprites are then placed only inside the union bbox of all frames, but every
    individual frame rectangle is excluded from the legal mask. This creates clean
    corridors "between frames" and prevents accidental placement inside a hollow
    frame interior.
  - A custom placement scan enforces in-bounds placement, non-overlap, and one-cell
    8-neighbor padding. For shared-color sprites, legal candidates are filtered so
    the intended frame is the unique nearest same-color frame.

e) visual_observability:
  - The D9 signal is immediate: the frames occupy four corners of a rectangle rather
    than one vertical strip.
  - The D10 signal is immediate: one color appears on two separated frames and same-
    color sprites cluster near one or the other corner, so location disambiguates.
  - Because the repeated-color pair is always active, the solver cannot fall back to
    the original "one frame per color" story on any generated sample.

f) visual_cleanliness:
  - The generator keeps all sprites in corridor background, not inside frames and
    not outside the 2x2 arrangement.
  - Every object is monochrome and disconnected, so component parsing stays clean.
  - Distinct non-repeated frame colors remain present, which helps a human isolate
    what changed: layout and shared-color disambiguation, not arbitrary recoloring.

g) anti_degeneration_signal:
  - Quantitative signal 1: the number of distinct frame columns is always 2; in the
    original generator it was always 1.
  - Quantitative signal 2: `main(inp)` differs from a naive color-only tally on every
    valid sample because the repeated-color pair is active.

h) new_solver_assumptions_and_generator_guarantees:
  - main() assumes frames are exact 5x9 hollow rectangles. generate_input() builds
    them that way.
  - main() assumes every counted shared-color sprite has a unique nearest same-color
    frame under L1 center distance. generate_input() enforces that by rejecting tie
    locations during placement.
  - main() assumes non-frame components are removable sprites or distractors. The
    generator guarantees this by keeping sprites small, disconnected, and outside
    the frame rectangles.
"""


FRAME_H = 5
FRAME_W = 9
DOT_CAPACITY = 3

TOP_R0 = 1
BOTTOM_R0 = 11
FRAME_ROWS = [TOP_R0, BOTTOM_R0]
FRAME_COLS = [1, 15]
GRID_W = 25
BOTTOM_PAD_RANGE = (0, 4)

N_FRAMES = 4
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_MATCH_RANGE = (4, 9)

SPRITE_AREA_RANGE = (2, 7)
SPRITE_MAX_H = 3
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 500
REPEATED_PAIRS = [(0, 3), (1, 2)]


def _is_hollow_frame(comp: dict) -> bool:
    """Recognize the generator's exact frame geometry."""
    b = tuple(map(int, comp["bbox"]))
    if bbox_h(b) != FRAME_H or bbox_w(b) != FRAME_W:
        return False

    r0, c0, r1, c1 = b
    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    if len(coords) != (2 * (FRAME_H + FRAME_W) - 4):
        return False

    for rr in range(r0, r1 + 1):
        if (rr, c0) not in coords or (rr, c1) not in coords:
            return False
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in coords or (r1, cc) not in coords:
            return False
    return True


def _bbox_center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return doubled center coordinates so all comparisons stay integral."""
    r0, c0, r1, c1 = map(int, bbox)
    return int(r0 + r1), int(c0 + c1)


def _distance_key(center2: tuple[int, int], frame: dict) -> tuple[int, int, int, int]:
    fr, fc = frame["center2"]
    dr = abs(int(center2[0]) - int(fr))
    dc = abs(int(center2[1]) - int(fc))
    return int(dr + dc), int(dr), int(dc), int(frame["index"])


def _unique_nearest_frame(center2: tuple[int, int], frames: list[dict]) -> dict | None:
    if not frames:
        return None
    dists = sorted(
        abs(int(center2[0]) - int(f["center2"][0])) + abs(int(center2[1]) - int(f["center2"][1]))
        for f in frames
    )
    if len(dists) >= 2 and int(dists[0]) == int(dists[1]):
        return None
    return min(frames, key=lambda f: _distance_key(center2, f))


def _frame_infos_from_grid(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
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
    Tally scattered sprites into the nearest frame of the same color.

    Input structure:
    - The background is the most frequent color.
    - Four hollow rectangular frames form a 2x2 arrangement.
    - Exactly one pair of separated frames may share a color.
    - Small monochrome sprites sit in the corridors between frame rectangles.

    Rule:
    - Remove every small sprite from the grid.
    - For each sprite, look only at frames of the same color.
    - Assign the sprite to the unique nearest same-color frame by L1 distance
      between bounding-box centers.
    - Draw one dot per assigned sprite on that frame's center row, stepping
      leftward from the right interior side with a gap of 2 cells.

    Key insight:
    Color alone no longer identifies a target frame. The repeated-color pair is
    disambiguated by spatial proximity inside the 2D frame layout, so each frame
    becomes a local tally for nearby same-color sprites rather than a global color bin.
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

    for f, count in zip(frames, counts):
        if count <= 0:
            continue
        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(count)):
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
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(min(n_dots, DOT_CAPACITY))]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(
    rng: np.random.Generator,
    total_match: int,
    repeated_pair: tuple[int, int],
) -> list[int]:
    """Allocate counts while keeping the shared-color pair active and visible."""
    if total_match < 3:
        raise ValueError("total_match too small")

    counts = [0] * N_FRAMES
    capacities = [DOT_CAPACITY] * N_FRAMES

    for idx in repeated_pair:
        counts[int(idx)] = 1
        capacities[int(idx)] -= 1

    support_choices = [i for i in range(N_FRAMES) if i not in repeated_pair]
    support_idx = int(rng.choice(support_choices))
    counts[support_idx] = 1
    capacities[support_idx] -= 1

    remaining = total_match - 3
    order = [int(x) for x in rng.permutation(N_FRAMES)]
    for pos, idx in enumerate(order):
        future_cap = sum(capacities[j] for j in order[pos + 1 :])
        lo = max(0, remaining - future_cap)
        hi = min(capacities[idx], remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        capacities[idx] -= add
        remaining -= add

    if remaining != 0:
        raise ValueError("failed to allocate all counts")
    if min(counts[int(repeated_pair[0])], counts[int(repeated_pair[1])]) <= 0:
        raise ValueError("shared-color pair became inactive")
    if counts[int(repeated_pair[0])] == DOT_CAPACITY and counts[int(repeated_pair[1])] == DOT_CAPACITY:
        raise ValueError("shared-color pair saturated both frames and became color-only-degenerate")
    if sum(v > 0 for v in counts) < 3:
        raise ValueError("layout signal too weak")
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


def _build_frame_colors(
    rng: np.random.Generator,
    n_distractors: int,
    repeated_pair: tuple[int, int],
) -> tuple[int, list[int], list[int]]:
    unique_frame_colors = N_FRAMES - 1
    palette = rng.choice(np.arange(10), size=1 + unique_frame_colors + n_distractors, replace=False)
    bg = int(palette[0])
    frame_palette = [int(x) for x in palette[1 : 1 + unique_frame_colors]]
    distractor_colors = [int(x) for x in palette[1 + unique_frame_colors :]]

    repeated_color = int(frame_palette[0])
    other_colors = frame_palette[1:]
    rng.shuffle(other_colors)

    frame_colors = [-1] * N_FRAMES
    for idx in repeated_pair:
        frame_colors[int(idx)] = repeated_color

    other_slots = [i for i in range(N_FRAMES) if i not in repeated_pair]
    for idx, color in zip(other_slots, other_colors):
        frame_colors[int(idx)] = int(color)
    return bg, frame_colors, distractor_colors


def _build_allowed_mask(
    shape: tuple[int, int],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    allowed = np.zeros(shape, dtype=bool)
    r0 = min(b[0] for b in frame_bboxes)
    c0 = min(b[1] for b in frame_bboxes)
    r1 = max(b[2] for b in frame_bboxes)
    c1 = max(b[3] for b in frame_bboxes)
    allowed[r0 : r1 + 1, c0 : c1 + 1] = True
    for fr0, fc0, fr1, fc1 in frame_bboxes:
        allowed[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    return allowed


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
    return int(2 * r0 + h - 1), int(2 * c0 + w - 1)


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

    for distance_limit in (12, 18, None):
        candidates: list[tuple[int, int, int]] = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if not _can_place(canvas, sprite, r0, c0, bg, SPRITE_PADDING):
                    continue
                center2 = _sprite_center2_global(r0, c0, sprite)
                nearest = _unique_nearest_frame(center2, same_color_frames)
                if nearest is None or int(nearest["index"]) != int(target_frame["index"]):
                    continue
                dist = abs(int(center2[0]) - int(target_frame["center2"][0])) + abs(
                    int(center2[1]) - int(target_frame["center2"][1])
                )
                if distance_limit is not None and dist > distance_limit:
                    continue
                candidates.append((int(dist), int(r0), int(c0)))
        if candidates:
            candidates.sort()
            top_k = min(8, len(candidates))
            dist, r0, c0 = candidates[int(rng.integers(0, top_k))]
            _ = dist
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

    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_match = int(rng.integers(*TOTAL_MATCH_RANGE))
    repeated_pair = REPEATED_PAIRS[int(rng.integers(0, len(REPEATED_PAIRS)))]

    try:
        counts = _sample_counts(rng, total_match, repeated_pair)
    except ValueError:
        return None

    bg, frame_colors, distractor_colors = _build_frame_colors(rng, n_distractors, repeated_pair)
    height = BOTTOM_R0 + FRAME_H + int(rng.integers(*BOTTOM_PAD_RANGE))

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_infos: list[dict] = []

    idx = 0
    for r0 in FRAME_ROWS:
        for c0 in FRAME_COLS:
            color = int(frame_colors[idx])
            count = int(counts[idx])
            paste(out, _frame_patch(color, bg, count), r0, c0)
            bbox = (r0, c0, r0 + FRAME_H - 1, c0 + FRAME_W - 1)
            frame_infos.append(
                {
                    "index": int(idx),
                    "color": color,
                    "bbox": bbox,
                    "center2": _bbox_center2(bbox),
                }
            )
            idx += 1

    inp = out.copy()
    for frame, count in zip(frame_infos, counts):
        r0, c0, _r1, c1 = map(int, frame["bbox"])
        mid_r = r0 + FRAME_H // 2
        for i in range(int(count)):
            inp[mid_r, c1 - 2 - 2 * i] = bg

    allowed_mask = _build_allowed_mask(inp.shape, [tuple(map(int, f["bbox"])) for f in frame_infos])
    canvas = np.full(inp.shape, -1, dtype=int)
    canvas[allowed_mask] = bg

    frames_by_color: dict[int, list[dict]] = {}
    for frame in frame_infos:
        frames_by_color.setdefault(int(frame["color"]), []).append(frame)

    assignments = [idx for idx, count in enumerate(counts) for _ in range(count)]
    assignments.sort(
        key=lambda idx: len(frames_by_color[int(frame_infos[int(idx)]["color"])]),
        reverse=True,
    )

    for frame_idx in assignments:
        target_frame = frame_infos[int(frame_idx)]
        sprite = _make_sprite(rng, int(target_frame["color"]), bg)
        if not _place_targeted_sprite(
            rng,
            canvas,
            sprite,
            bg,
            target_frame,
            frames_by_color[int(target_frame["color"])],
        ):
            return None

    for color in distractor_colors:
        sprite = _make_sprite(rng, int(color), bg)
        if not _place_distractor_sprite(rng, canvas, sprite, bg):
            return None

    sprite_mask = canvas >= 0
    sprite_mask &= canvas != bg
    inp[sprite_mask] = canvas[sprite_mask]

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
