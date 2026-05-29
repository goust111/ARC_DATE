# Variant of 8f215267
# Dimension: D5 — 变换规则：点的布局方式 (transform)
# Change: 在框架中心行从右侧起每隔2列放置N个同色点（列坐标: c1-2, c1-4, c1-6, ...）... → 在对角线方向放置

from __future__ import annotations

"""
Design analysis for the D5 variant

a) Concrete behavior and trace
- Objects stay the same as the source task: several 5x11 hollow color frames on the left,
  and small monochrome sprites on the right. Sprites are still counted by color and then
  removed.
- The only semantic change is how each frame displays its tally in the output.
- Old rule: if a frame has color C and N matching sprites, dots appear on the interior
  middle row at columns (..., c1-6, c1-4, c1-2).
- New rule: dots appear on the frame's interior descending diagonal, starting from the
  lower-right interior corner and stepping one cell up-left each time:
  (r1-1, c1-1), (r1-2, c1-2), (r1-3, c1-3), ...
- Example inside one 5x11 frame:
  N=3 used to draw three points on one horizontal row; now it draws a staircase running
  from the bottom-right interior toward the upper-left. This affects exactly the "draw tally"
  stage in `main()` and the corresponding output-first construction/removal logic in the
  generator.

b) Complexity calibration
- The original 5x11 frame has only a 3x9 interior, so a strict down-right to up-left
  diagonal can display at most 3 cells. Therefore the original cap of 4 dots is no longer
  physically valid.
- To keep the task minimal and avoid changing frame geometry, the per-frame tally cap is
  reduced from 4 to 3. This is the smallest compatibility change required by D5.
- To keep the diagonal visible to humans, generation also insists that at least one frame
  in each sample has count >= 2; otherwise a grid with only single dots would underspecify
  the new layout pattern.

c) Implicit-assumption audit of the source `main()`
- Component finding / frame-vs-pattern classification only assumes large sparse same-color
  rectangles for frames and smaller compact objects for sprites. D5 does not invalidate this.
- Pattern removal assumes every non-frame component should be erased. D5 does not change that.
- The old drawing code assumed:
  1. there is one special display row (`r_mid`);
  2. valid dot columns are `c1-2-2*i`;
  3. interior capacity is controlled only by width.
  These assumptions break under D5. The replacement must instead:
  1. use both row and column coordinates;
  2. step with `(-1, -1)` from the lower-right interior corner;
  3. stop when either the interior height or width is exhausted.
- The new solver therefore computes `(rr, cc)` per dot and guards with `rr > r0` and
  `cc > c0`, which matches the notion of "strictly inside the frame border".

d) Placement legality
- Frames remain vertically stacked with unchanged spacing, so they never overlap.
- Sprites are still generated and packed only in the right region via `random_free_location`,
  with 8-connected padding, so they cannot overlap each other or the frames.
- Output-first generation draws diagonal dots only at interior cells, then derives the input
  by clearing exactly those cells. Validation still requires `main(inp) == out`.

e) Visual observability
- A diagonal staircase is visually distinct from the old horizontal row, especially for
  counts 2 and 3. Enforcing at least one count >= 2 per sample gives the solver examples
  repeated support for the diagonal pattern.
- The rule does not depend on numeric color ids or hidden ordering; only visible geometry
  matters.

f) Visual cleanliness
- The new marks stay inside the frames and use the frame's own color, exactly like the
  source task, so the scene remains sparse and easy to parse.
- The right sprite field is unchanged, preserving clean separation between "counted objects"
  and "display area".

g) Color inference
- No new color mapping is introduced. Color still means "which sprites belong to which
  frame", so the D5 change is purely geometric and remains easy to infer.

Post-change assumptions that the generator must guarantee
- Every frame is at least 3 interior cells wide and high enough for the intended diagonal.
  The fixed 5x11 frame shape guarantees this.
- Matching counts must not exceed diagonal capacity. The generator enforces
  `MAX_DOTS_PER_FRAME = 3`.
- To avoid accidental degeneration toward the original task, each generated puzzle must have
  at least one nontrivial diagonal (count >= 2), and at least one matching sprite overall.
"""

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


def _diagonal_dot_positions_from_bbox(bbox: tuple[int, int, int, int], n_dots: int) -> list[tuple[int, int]]:
    """Interior cells from lower-right toward upper-left, clipped by frame interior."""
    r0, c0, r1, c1 = map(int, bbox)
    cells: list[tuple[int, int]] = []
    for i in range(int(n_dots)):
        rr = int(r1 - 1 - i)
        cc = int(c1 - 1 - i)
        if rr > int(r0) and cc > int(c0):
            cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as diagonal dots inside.

    Input structure:
    - bg = most common color.
    - Hollow rectangular frames in various colors.
    - Small monochrome pattern sprites scattered outside frames.

    Rule:
    - Remove all small patterns from the grid.
    - For each frame of color C, count how many patterns share color C.
    - Draw that many dots along the frame interior diagonal, starting from the
      lower-right interior corner and moving one cell up-left per dot.
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

    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        for rr, cc in _diagonal_dot_positions_from_bbox(f["bbox"], num_patterns):
            out[int(rr), int(cc)] = col

    return out


# --------------- generator (codex-att2, D5 diagonal variant) ---------------

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
MAX_DOTS_PER_FRAME = 3
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_diagonal_cells(n_dots: int) -> list[tuple[int, int]]:
    """Dot positions inside a FRAME_H x FRAME_W patch, in local coordinates."""
    return _diagonal_dot_positions_from_bbox((0, 0, FRAME_H - 1, FRAME_W - 1), n_dots)


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with n_dots on the interior diagonal."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for rr, cc in _frame_diagonal_cells(n_dots):
        patch[int(rr), int(cc)] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """Distribute total_match sprites across n_frames, each capped at MAX_DOTS_PER_FRAME."""
    for _ in range(64):
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
        if max(counts) >= 2:
            return counts
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
    if max(counts) < 2:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        for rr, cc in _frame_diagonal_cells(count):
            inp[int(r0 + rr), int(FRAME_C0 + cc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
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
