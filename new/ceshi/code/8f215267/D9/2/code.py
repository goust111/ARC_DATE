# Variant of 8f215267
# Dimension: D9 — 框架数量与排列 (meta)
# Change: 3-4个框架，垂直堆叠在左半部分，等间距排列（间隔2行）... → 框架呈网格式排列

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the D9 variant
# ----------------------------------
# source_task: None
#
# a) Concrete behavior and trace
# - Original inputs place 3-4 large 5x11 frames in one left-side column and scatter sprites
#   in a separate right-side region. The solver only counts same-color sprites and writes dots
#   back into the matching frame.
# - This variant keeps the color-counting rule, but changes the spatial organization: frames are
#   smaller 5x7 hollow rectangles arranged as a 2x2 or 2x3 grid. Sprites are no longer in an
#   isolated right strip. They are forced into the background corridors inside the overall frame
#   arrangement, especially the wide horizontal band between the top and bottom frame rows.
# - Small trace:
#   Original: frame colors [red, blue, yellow] at one shared x-position; a red sprite on the far
#   right contributes one red dot in the red frame.
#   Variant: frame colors [red, blue, yellow, green] may sit at top-left, top-right, bottom-left,
#   bottom-right; a red sprite placed in the corridor between those frames still contributes one
#   red dot in the red frame. The counting logic is the same; only the object layout and sprite
#   placement zone change.
#
# b) Complexity calibration
# - Layout alternates between 2 columns and 3 columns, yielding 4 or 6 frames. This is visually
#   distinct from the old single-column stack.
# - Frames shrink from width 11 to width 7 so a 2x3 layout fits cleanly. That reduces dot capacity
#   from 4 to 2, so counts are capped at 2. This is a necessary compatibility change, not a new
#   logical dimension.
# - Multiple active frames are required in generated samples so the grid layout affects the task in
#   a visible way.
#
# c) Implicit-assumption review of main()
# - The old solver did not actually use "frames are in one column" anywhere; it only separated
#   components using an area heuristic.
# - That heuristic becomes riskier once frames are smaller and corridors are tighter, so frame
#   detection is upgraded to an explicit hollow-rectangle test. This new code assumes input frames
#   are empty rectangular perimeters with no interior dots yet. The generator guarantees that.
# - Dot drawing still assumes dots belong on the center row and are spaced by 2 from the right.
#   With width 7 this remains valid for at most 2 dots.
#
# d) Placement legality
# - Frames are pasted first into an output grid.
# - Input is derived by removing dots.
# - Sprite placement uses a separate placement canvas where only background cells inside the union
#   bounding box of all frames are free. Cells outside that corridor region are marked occupied.
# - random_free_location plus sprite-to-sprite padding prevents overlap and keeps sprites away from
#   each other. The generator retries if no legal packing exists.
#
# e) Visual observability
# - In the original task, every frame shares one x-position and every sprite lies to the right of
#   all frames. In this variant, frames occupy 2 or 3 distinct columns and sprites sit among them.
# - From 2-3 examples, a human can see the same color-counting rule while also seeing the changed
#   spatial organization.
#
# f) Visual cleanliness
# - Frames keep distinct colors and simple rectangular perimeters.
# - Sprites are confined to corridor background, so no object layering ambiguity occurs.
# - A tall horizontal corridor between the two frame rows gives enough room for sprites without
#   cluttering frame interiors.
#
# g) Color inference
# - Colors are still used only for direct equality matching between sprites and frames.
# - Distractor colors remain present but unmatched, exactly as in the original task.
#
# h) Generator-side anti-degeneration signal
# - Quantitatively, the important D9 signal is the number of distinct frame columns. The original
#   generator always yields 1. This variant always yields 2 or 3.
# - A second signal is that sprites are not all to the right of the frames; they lie inside the
#   frame union bbox instead.
#
# i) New solver assumptions to audit
# - main() assumes any component treated as a frame is a monochrome hollow rectangle perimeter.
# - main() assumes non-frame components are sprites/distractors and can be erased safely.
# - generate_input() enforces both by making frames exact 5x7 perimeters and by keeping all sprites
#   much smaller than frame size.


FRAME_H = 5
FRAME_W = 7
DOT_CAPACITY = 2

TOP_R0 = 1
BOTTOM_R0 = 13
GRID_W = 25
BOTTOM_PAD_RANGE = (0, 4)

LAYOUTS = {
    2: [2, 16],      # 2x2 layout: wide middle corridor
    3: [0, 9, 18],   # 2x3 layout: smaller but still visible vertical corridors
}

N_DISTRACTORS_RANGE = (1, 3)
SPRITE_AREA_RANGE = (2, 7)
SPRITE_MAX_H = 3
SPRITE_MAX_W = 3
SPRITE_PADDING = 1
MAX_ATTEMPTS = 300


def _is_hollow_frame(comp: dict) -> bool:
    """Exact frame test for input-time hollow rectangles."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per frame and show the tally as dots.

    Variant-specific note:
    The solver is layout-agnostic after frame detection. Once hollow rectangular frames are found,
    the same color-counting rule applies whether frames form one column or a 2x2 / 2x3 grid.
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
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, c1 = map(int, f["bbox"])
        r_mid = r0 + FRAME_H // 2
        for i in range(num_patterns):
            cc = c1 - 2 - 2 * i
            if cc > c0:
                out[r_mid, cc] = col

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
    n_frames: int,
    total_match: int,
    min_active: int,
) -> list[int]:
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match

    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        mandatory_left = max(0, min_active - sum(v > 0 for v in counts))
        reserve = max(0, mandatory_left - 1)
        lo = max(0, remaining - DOT_CAPACITY * frames_left)
        if reserve > 0:
            lo = max(lo, 1)
        hi = min(DOT_CAPACITY, remaining)
        counts[int(idx)] = int(rng.integers(lo, hi + 1))
        remaining -= counts[int(idx)]

    if remaining != 0:
        raise ValueError("count sampling failed")
    if sum(v > 0 for v in counts) < min_active:
        raise ValueError("not enough active frames")
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


def _layout_params(rng: np.random.Generator) -> tuple[int, list[int], int]:
    n_cols = int(rng.choice([2, 3]))
    frame_cols = list(LAYOUTS[n_cols])
    height = BOTTOM_R0 + FRAME_H + int(rng.integers(*BOTTOM_PAD_RANGE))
    return n_cols, frame_cols, height


def _build_allowed_mask(grid: np.ndarray, frame_bboxes: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    allowed = np.zeros_like(grid, dtype=bool)
    r0 = min(b[0] for b in frame_bboxes)
    c0 = min(b[1] for b in frame_bboxes)
    r1 = max(b[2] for b in frame_bboxes)
    c1 = max(b[3] for b in frame_bboxes)
    allowed[r0 : r1 + 1, c0 : c1 + 1] = True
    allowed &= (grid == bg)
    return allowed


def _pick_total_match(rng: np.random.Generator, n_frames: int) -> int:
    min_active = n_frames // 2
    lo = min_active + 1
    hi = min(n_frames * DOT_CAPACITY, min_active + n_frames)
    return int(rng.integers(lo, hi + 1))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_cols, frame_cols, height = _layout_params(rng)
    n_frames = 2 * n_cols
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_match = _pick_total_match(rng, n_frames)
    total_patterns = total_match + n_distractors
    min_active = n_cols

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames : 1 + n_frames + n_distractors]]

    try:
        counts = _sample_counts(rng, n_frames, total_match, min_active=min_active)
    except ValueError:
        return None

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_positions: list[tuple[int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    frame_rows = [TOP_R0, BOTTOM_R0]

    idx = 0
    for r0 in frame_rows:
        for c0 in frame_cols:
            color = frame_colors[idx]
            count = counts[idx]
            paste(out, _frame_patch(color, bg, count), r0, c0)
            frame_positions.append((r0, c0))
            frame_bboxes.append((r0, c0, r0 + FRAME_H - 1, c0 + FRAME_W - 1))
            idx += 1

    inp = out.copy()
    for (r0, c0), count in zip(frame_positions, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [c0 + FRAME_W - 3 - 2 * i for i in range(min(count, DOT_CAPACITY))]
        inp[mid_r, dot_cols] = bg

    allowed_mask = _build_allowed_mask(inp, frame_bboxes, bg)
    placement_canvas = np.full(inp.shape, -1, dtype=int)
    placement_canvas[allowed_mask] = bg

    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)
    if len(pattern_colors) != total_patterns:
        return None

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

    sprite_mask = placement_canvas != bg
    sprite_mask &= placement_canvas >= 0
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
