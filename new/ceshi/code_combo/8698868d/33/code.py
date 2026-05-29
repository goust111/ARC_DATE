# Combo variant of 8698868d (index 33)
# combo_spec: D4/2+D6/2
# Name: 颜色互换+精灵位填框架色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the combo variant
# =====================================
#
# a) Concrete behavior and trace
#    The input family stays the same as the original task:
#    - azure background (8)
#    - a tight grid of larger square frames
#    - the same number of smaller square sprites scattered elsewhere
#    - each frame/sprite pair is still identified only by equal counts of
#      4-connected hole components
#    The fused output changes in two linked ways:
#    1. At each frame location, nest the matched sprite and perform the D4 swap:
#       the outer frame border takes the sprite color, the sprite's solid pixels
#       take the frame color, and the sprite holes remain background.
#    2. Keep the full canvas like D6 and repaint the matched sprite's source
#       square as a solid block of the frame color.
#    Small trace:
#    - frame color = 4, holes = 2; matched sprite color = 3, holes = 2.
#    - Original task: compact output only; border stays 4; sprite stays 3;
#      sprite holes become 4.
#    - Combo task: full-canvas output; the frame border becomes 3; the nested
#      sprite body becomes 4; nested holes stay 8; the old sprite position
#      becomes a solid 4 square.
#    The two changes reinforce each other: the matched frame color appears both
#    inside the frame and at the sprite's former location.
#
# b) Complexity calibration
#    The structural matching cue is unchanged, so the puzzle does not gain a
#    second hidden key. To keep the combo visible, the generator ensures every
#    matched pair has different frame and sprite colors; otherwise both the swap
#    and the source repaint could become visually ambiguous for low-hole sprites.
#    Using a derangement over frame colors keeps the palette small and repeated
#    enough for humans to infer the local swap rule from 2-4 examples.
#
# c) Implicit-assumption audit of the original main()
#    - Candidate-object detection by square shape and fill ratio still holds,
#      because the input geometry is unchanged.
#    - Grouping larger squares as frames and smaller squares as sprites still
#      holds, because sprite_size remains frame_size - 2.
#    - The original compact `(out_r, out_c)` output coordinates are no longer
#      sufficient: D6 requires painting frame results at input coordinates and
#      repainting sprite source coordinates too.
#    - The original `out[...] = frame_color` assumption breaks under D4 because
#      the final border color is now the sprite color.
#    - The original `content[content != sprite_color] = frame_color` assumption
#      also breaks: for D4 we recolor the sprite's solid cells instead, while
#      the nested holes must remain background.
#    - The fused variant introduces one more assumption: sprite source squares
#      must be safely overwritable as solid blocks. The generator guarantees this
#      by placing sprites in separated background space with padding.
#
# d) Placement legality
#    Frames are placed as a tight, axis-aligned, non-overlapping grid. Sprites
#    are placed outside that grid using `random_free_location(..., padding=1)`,
#    so they do not overlap frames or touch other sprites. Every later write in
#    the output stays inside either a known frame square or a known sprite square.
#
# e) Visual observability
#    A human can detect the fusion because each matched pair exhibits both:
#    - border/body color swapping inside the frame
#    - a same-colored solid source block at the sprite's old location
#    The repeated frame-color echo between those two regions is the intended cue.
#
# f) Visual cleanliness
#    The nested frame result has only three readable regions: swapped border,
#    recolored sprite body, and preserved background holes. The source square is
#    a single solid block, and the placement padding prevents accidental merges.
#
# g) Change inferability
#    Color changes are not arbitrary: the frame color is recoverable from the
#    matched frame tile itself, and hole-count matching explains which scattered
#    sprite belongs to which frame. The generator keeps the number of active
#    colors low and repeats the transformed pattern several times per sample.
#
# h) New solver assumptions checked against the generator
#    - There is exactly one frame and one sprite per hole count.
#    - Matched frame/sprite colors differ for every pair.
#    - Sprite patches fill the full frame interior exactly.
#    - Sprite source bboxes can be repainted without colliding with other output
#      regions.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each scattered sprite with a grid frame by connected-hole count and
    redraw the full scene using a color-swap plus source-fill transform.

    Input structure:
    - Background is azure (8). Large equal-sized square frames form a tight
      rectangular grid somewhere on the canvas, while smaller square sprites sit
      elsewhere on the background.
    - Frame interiors may contain isolated background cells, and sprites may
      contain background-shaped holes inside a solid border.

    Rule:
    - Count 4-connected hole components in every frame interior and in every
      sprite, then match frame and sprite with the same count.
    - Keep the original canvas size. Inside each frame, nest the matched sprite
      and swap the visible object colors: the frame border becomes the sprite
      color, the sprite's solid pixels become the frame color, and the sprite
      holes stay background.
    - Repaint the matched sprite's original square footprint as a solid block of
      that frame's color.

    Key insight:
    - The only pairing key is the number of connected hole regions. After the
      match, the frame's color reappears both inside the swapped nested sprite
      and across the sprite's old location.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
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
            frame_grid.append(
                {
                    "color": frame_color,
                    "inp_r": inp_r,
                    "inp_c": inp_c,
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
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
                "inp_r": int(b[0]),
                "inp_c": int(b[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for i, sprite in enumerate(sprites):
            if i not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["inp_r"]), int(frame["inp_c"]))] = sprite
                used_sprites.add(i)
                break

    out = np.full((H, W), BG, dtype=int)

    for frame in frame_grid:
        fr = int(frame["inp_r"])
        fc = int(frame["inp_c"])
        frame_color = int(frame["color"])

        sprite = matches.get((fr, fc))
        border_color = int(sprite["color"]) if sprite is not None else frame_color
        out[fr : fr + frame_size, fc : fc + frame_size] = border_color

        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content == sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

        sr = int(sprite["inp_r"])
        sc = int(sprite["inp_c"])
        out[sr : sr + sh, sc : sc + sw] = frame_color

    return out


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
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
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_index, comp in enumerate(components):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(components)
                        if j != comp_index
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_holes.add(cell)
            sprite[cell[0], cell[1]] = BG

    _, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
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

    _, num_holes = label_mask(frame != frame_color, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _build_expected_output(
    H: int,
    W: int,
    frames: list[dict],
    sprites: list[dict],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((H, W), BG, dtype=int)
    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    frame_color_by_holes = {int(frame["holes"]): int(frame["color"]) for frame in frames}

    for frame in frames:
        rr = int(frame["r"])
        cc = int(frame["c"])
        sprite = sprite_by_holes[int(frame["holes"])]
        border_color = int(sprite["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = border_color

        content = np.array(sprite["patch"], dtype=int).copy()
        content[content == int(sprite["color"])] = int(frame["color"])
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    for sprite in sprites:
        rr = int(sprite["r"])
        cc = int(sprite["c"])
        fill_color = int(frame_color_by_holes[int(sprite["holes"])])
        expected[rr : rr + sprite_size, cc : cc + sprite_size] = fill_color

    return expected


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

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - N)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + N))
    random.shuffle(hole_counts)

    frames: list[dict] = []
    for i, holes in enumerate(hole_counts):
        patch = _make_frame(frame_size, int(holes), int(frame_colors[i]))
        if patch is None:
            return None
        frames.append({"holes": int(holes), "color": int(frame_colors[i]), "patch": patch})

    rotated_colors = frame_colors[1:] + frame_colors[:1]
    sprite_color_by_holes = {
        int(frame["holes"]): int(rotated_colors[i]) for i, frame in enumerate(frames)
    }

    sprites: list[dict] = []
    sprite_hole_order = hole_counts[:]
    random.shuffle(sprite_hole_order)
    for holes in sprite_hole_order:
        sprite_color = int(sprite_color_by_holes[int(holes)])
        patch = _make_sprite_interior(sprite_size, int(holes), sprite_color)
        if patch is None:
            return None
        sprites.append({"holes": int(holes), "color": sprite_color, "patch": patch})

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        frame["r"] = rr
        frame["c"] = cc
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    for sprite in sprites:
        try:
            rr, cc = random_free_location(
                canvas,
                np.array(sprite["patch"], dtype=int),
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        sprite["r"] = int(rr)
        sprite["c"] = int(cc)
        paste(canvas, np.array(sprite["patch"], dtype=int), int(rr), int(cc), transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(H, W, frames, sprites, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    for frame in frames:
        if int(frame["color"]) == int(sprite_color_by_holes[int(frame["holes"])]):
            return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
