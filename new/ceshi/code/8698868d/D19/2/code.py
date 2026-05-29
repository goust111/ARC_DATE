# Variant of 8698868d
# Dimension: D19 — 输出中框架的排列顺序 (meta)
# Change: 输出框架按输入中的网格位置排列（行优先顺序保持不变）... → 行列转置输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the D19 variant
# -----------------------------------
# a) Concrete behavior:
#    The input still contains a tight rows_grid x cols_grid array of large square
#    frames plus the same number of scattered smaller sprites. Matching is still by
#    connected-hole count. The only rule change is where each solved frame lands in
#    the output: a frame seen at input grid slot (ri, ci) is written to output slot
#    (ci, ri). Example with a 2x3 frame grid:
#      original output slots:   (0,0) (0,1) (0,2)
#                               (1,0) (1,1) (1,2)
#      variant output slots:    (0,0) (1,0)
#                               (0,1) (1,1)
#                               (0,2) (1,2)
#    So the solver change happens after frame detection and sprite matching; only the
#    output placement map changes. The generator must therefore build its oracle with
#    transposed slot coordinates instead of row-major-preserved coordinates.
#
# b) Complexity calibration:
#    The task must remain visually nontrivial. We keep unique frame colors and unique
#    sprite colors plus unique hole counts, so transposition is easy to observe and
#    never creates matching ambiguity. To avoid degeneracy, the generator rejects any
#    sample whose transposed output accidentally equals the original row-major output.
#    With distinct frame colors and at least two frames, this is effectively always a
#    genuine D19 sample, but the explicit check guards against hidden regressions.
#
# c) Implicit-assumption audit of the original main():
#    - Grouping components by size still works: D19 does not alter object geometry.
#    - Inferring n_rows / n_cols from frame positions still works: the input layout is
#      unchanged.
#    - The original frame_grid stored out_r=ri*frame_size and out_c=ci*frame_size.
#      That hardcoded the old row-major output order and is the only variant-breaking
#      assumption.
#    - Matching via a dict keyed by output coordinates is fragile once coordinates are
#      no longer the same conceptual slot as the input grid. This variant keys matches
#      by the logical input slot (ri, ci) instead, then writes to transposed output
#      coordinates. That removes accidental dependence on the old ordering.
#
# d) Placement legality:
#    Input construction is unchanged: the frame grid is still placed as one tight block
#    on the larger canvas, and sprites are still scattered onto remaining background via
#    random_free_location(..., padding=1). Output construction only rearranges solved
#    frame blocks on a new canvas of shape (cols_grid*frame_size, rows_grid*frame_size),
#    so there is no overlap risk because transposed slots remain disjoint.
#
# e) Visual observability:
#    Humans can see the transpose because output dimensions flip for rectangular frame
#    grids, and for square 2x2 grids the off-diagonal frames swap positions while frame
#    colors remain distinct. The rule never depends on numeric color ids; it depends only
#    on visible input row/column positions.
#
# f) Visual cleanliness:
#    No new marks are added. The output remains a tight block of solved frames with no
#    background gaps, just re-ordered by transpose, so the scene stays as clean as the
#    original task.
#
# g) Color inference:
#    Colors are not remapped by value. Each sprite still inherits its matched frame color
#    only for former hole pixels, exactly as in the base task. D19 changes location only.


# ── Configuration ──────────────────────────────────────────────────────
BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300


# Precompute valid grid layouts: {N: [(rows, cols), ...]}
_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _slot_to_out_coords(ri: int, ci: int, frame_size: int, *, transpose: bool) -> tuple[int, int]:
    if transpose:
        return int(ci * frame_size), int(ri * frame_size)
    return int(ri * frame_size), int(ci * frame_size)


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` cells inside a (size x size) border, each mutually 4-isolated."""
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    """Create a sprite with a solid border and exactly `hole_count` 4-connected holes."""
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite
    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(comps)
                        if j != ci
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Create a frame square with `hole_count` isolated interior BG pixels."""
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame
    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _, num = label_mask(frame != frame_color, connectivity=4)
    if num != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
    *,
    transpose: bool,
) -> np.ndarray:
    out_h = int((cols_grid if transpose else rows_grid) * frame_size)
    out_w = int((rows_grid if transpose else cols_grid) * frame_size)
    expected = np.full((out_h, out_w), BG, dtype=int)
    sprite_by_holes = {int(k): spatch for k, spatch in sprites}

    for idx, (holes, _) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = _slot_to_out_coords(ri, ci, frame_size, transpose=transpose)
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        content = np.array(sprite_by_holes[int(holes)], dtype=int).copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    Variant rule:
    - Detect the input frame grid exactly as in the original task.
    - Preserve matching by equal hole-count.
    - Output the solved frame grid with transposed slot order: input slot (ri, ci)
      is written to output slot (ci, ri).
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != frame_color
            _lab, num_holes = label_mask(hole_mask, connectivity=4)
            out_r, out_c = _slot_to_out_coords(ri, ci, frame_size, transpose=True)

            frame_grid.append(
                {
                    "slot": (int(ri), int(ci)),
                    "color": frame_color,
                    "holes": int(num_holes),
                    "out_r": out_r,
                    "out_c": out_c,
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for i, sprite in enumerate(sprites):
            if i not in used_sprites and frame["holes"] == sprite["holes"]:
                matches[frame["slot"]] = sprite
                used_sprites.add(i)
                break

    out_h = int(n_cols * frame_size)
    out_w = int(n_rows * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches.get(frame["slot"])
        if sprite is None:
            continue
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _gen_attempt() -> np.ndarray | None:
    N = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, N)
    sprite_colors = random.sample(available, N)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - N)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + N))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, holes in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, holes, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((int(holes), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, holes in enumerate(hole_counts):
        patch = _make_frame(frame_size, holes, frame_colors[i])
        if patch is None:
            return None
        frames.append((int(holes), patch))

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_holes, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _holes, sprite_patch in sprites:
        try:
            rr, cc = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
        transpose=True,
    )
    old_expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
        transpose=False,
    )

    if np.array_equal(expected, old_expected):
        return None
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
