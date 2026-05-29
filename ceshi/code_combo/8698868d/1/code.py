# Combo variant of 8698868d (index 1)
# combo_spec: D1/0+D4/2
# Name: 颜色匹配后互换着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace
# - Keep the original spatial family: larger square frames form a tight grid, and
#   smaller square sprites are scattered elsewhere on the azure background.
# - Replace the old hole-count match key with a fixed-location color marker. Each
#   frame and sprite has a 1-pixel marker at its own top-left corner, and equal
#   marker colors define the pairing.
# - After nesting, perform a coupled color swap:
#   * frame border color -> matched sprite's main color
#   * sprite solid pixels -> hosting frame's main color
#   * sprite bg-hole pixels -> sprite's original main color
#   * marker pixels stay marker-colored; they are matching cues, not body color
# - Small trace:
#   original task:
#     frame holes=2 matches sprite holes=2; output keeps frame border color and
#     recolors sprite holes to that frame color.
#   fused task:
#     red corner marker on a blue frame matches the sprite with the same red
#     corner marker even if hole counts disagree; output border becomes the
#     sprite body color, sprite solid cells become the frame body color, and the
#     sprite's holes become the sprite body color.
#   key difference:
#     "why these two objects pair" comes from the marker, while "how colors
#     change after pairing" comes from a swap between the matched body colors.
#
# b) Complexity calibration
# - The marker sits at a stable semantic position on both object types, so the
#   equality cue is easy to compare across 2-4 examples.
# - Hole counts are intentionally decoupled across frames and sprites: counts may
#   repeat and the two multisets differ, so the original counting strategy is not
#   a fallback that still silently works.
# - The generator uses globally disjoint marker/frame/sprite color sets, so the
#   pairing cue is not visually polluted by unrelated body pixels sharing the
#   same color.
# - The generator keeps per-pair frame color != sprite color so the swap is
#   visibly meaningful rather than collapsing into an unchanged output tile.
# - Sprite holes are kept present in many samples so the "holes become sprite
#   color" part is observable and not reduced to a border-only swap.
#
# c) Hidden assumptions broken from the original main()
# - `sprite_region != sprite_color` can no longer define sprite holes because the
#   corner marker is also a non-sprite-color pixel. The solver now treats only BG
#   cells as holes and leaves the marker untouched.
# - Matching by `f["holes"] == s["holes"]` is invalid. The solver must read the
#   fixed-position marker color instead.
# - Initializing each output tile with the frame color is no longer correct
#   because the outer border should end up in sprite color.
# - The D4-only swap variant left sprite holes as BG; that would miss the fused
#   rule here. The solver must recolor those hole pixels to the sprite's original
#   color to remove BG from the output.
#
# d) Placement legality
# - Frames still occupy a gapless axis-aligned grid; sprites are placed by
#   `random_free_location` on remaining BG cells with padding, so they do not
#   overlap the grid or each other.
# - The marker reuses the top-left cell already inside each object's bbox, so it
#   does not change object size or make placements illegal.
# - Frame colors remain distinct within the grid, which prevents adjacent frames
#   from merging into one monochromatic component during detection.
#
# e) Visual observability
# - A human can see the same top-left marker color repeated once on a frame and
#   once on a sprite; that makes the pairing rule explicit.
# - The output then shows the matched sprite color moving onto the frame border
#   while the frame color moves into the sprite body. This is a strong visual cue.
# - Non-matching hole-count multisets make it clear over several examples that
#   holes are no longer the reason for pairing.
#
# f) Visual cleanliness
# - Only one extra pixel per object is used as a marker, so the scenes remain
#   legible.
# - The output tile still has a simple three-region structure: outer border,
#   inner sprite body, and recolored former hole cells. The unchanged marker
#   remains a single accent pixel rather than introducing clutter.
#
# g) Color-change inferability
# - No hidden ordering of color ids is used. The only pairing rule is direct
#   visible equality of marker colors.
# - The recoloring rule is local to a matched pair and uses only the two body
#   colors already visible in that pair.
# - Because hole pixels adopt the sprite's original body color, the output uses a
#   consistent "frame color inside solid shape, sprite color on border and holes"
#   structure that can be inferred from few examples.
#
# New solver assumptions checked against the generator
# - Every frame and sprite has exactly one top-left marker, and marker colors are
#   unique across pairs inside a sample.
# - Marker colors are globally disjoint from all frame and sprite body colors, so
#   the matching cue never blends into an unrelated object body.
# - For each pair, marker color differs from both the frame body color and the
#   sprite body color, so the marker survives recoloring unchanged.
# - Sprite holes are BG-colored only; frame holes are BG-colored decoys only.
# - Sprite size is always `frame_size - 2`, so the nested sprite exactly fills
#   the frame interior.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered sprites to grid frames by a shared corner-marker color, then
    crop to the tight frame grid and swap the matched frame/sprite body colors.

    Input structure:
    - bg = azure (8). Large square frames appear in a tight grid; smaller square
      sprites of size `frame_size - 2` are scattered elsewhere.
    - Each frame and each sprite has a single 1-pixel marker at its own top-left
      corner. Matching pairs share that marker color.
    - Frames may contain bg holes in their interiors, and sprites may contain
      bg-hole components inside their colored bodies.

    Rule:
    - Detect the frame grid and the scattered sprites by size, then read the
      fixed top-left marker color of each object.
    - Match each frame to the sprite with the same marker color.
    - Output only the tight frame grid. For each matched pair, recolor the frame
      border to the sprite's body color, recolor the sprite's solid pixels to the
      frame's body color, recolor the sprite's bg holes to the sprite's original
      body color, and keep the marker pixels unchanged.

    Key insight:
    - The same corner-marker color explains the pairing, and the matched pair
      then undergoes a two-way body-color swap; hole counts are visual distractors
      rather than the deciding key.
    """
    g = np.array(grid, dtype=int)
    BG = 8

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        bbox = comp["bbox"]
        h = int(bbox_h(bbox))
        w = int(bbox_w(bbox))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((int(h), comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames and sprites_raw

    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames]
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
            frame_grid.append(
                {
                    "color": frame_color,
                    "marker": int(region[0, 0]),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprites.append(
            {
                "color": int(comp["color"]),
                "marker": int(sprite_region[0, 0]),
                "content": np.array(sprite_region, dtype=int).copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    frames_by_marker = {int(frame["marker"]): frame for frame in frame_grid}
    sprites_by_marker = {int(sprite["marker"]): sprite for sprite in sprites}
    assert len(frames_by_marker) == len(frame_grid)
    assert len(sprites_by_marker) == len(sprites)
    assert set(frames_by_marker) <= set(sprites_by_marker)

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        marker_color = int(frame["marker"])
        sprite = sprites_by_marker[marker_color]
        sprite_color = int(sprite["color"])

        out[fr : fr + frame_size, fc : fc + frame_size] = sprite_color
        out[fr, fc] = marker_color

        content = np.array(sprite["content"], dtype=int).copy()
        content[content == sprite_color] = frame_color
        content[content == BG] = sprite_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 3)  # keep total pair count low enough to give marker colors a clean disjoint palette
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
HOLE_RETRIES = 60
GEN_RETRIES = 400
MAX_VISIBLE_HOLES = 3

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _max_isolated_holes(size: int) -> int:
    inner = int(size - 2)
    return int((inner * inner + 1) // 2)


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((int(r), int(c)))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite(size: int, hole_count: int, sprite_color: int, marker_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), int(sprite_color), dtype=int)
    if hole_count > 0:
        inner_cells = int((size - 2) ** 2)
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
        for comp_idx, comp in enumerate(comps):
            grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
            for _ in range(grow):
                frontier = []
                for r, c in comp:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = int(r + dr)
                        nc = int(c + dc)
                        if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                            continue
                        if (nr, nc) in all_hole:
                            continue
                        if any(
                            (nr + dr2, nc + dc2) in other
                            for other_idx, other in enumerate(comps)
                            if other_idx != comp_idx
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

        _, num_holes = label_mask(sprite == BG, connectivity=4)
        if int(num_holes) != int(hole_count):
            return None

    sprite[0, 0] = int(marker_color)
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int, marker_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    if hole_count > 0:
        inner_cells = int((frame_size - 2) ** 2)
        if inner_cells < hole_count:
            return None

        seeds = _place_isolated_seeds(frame_size, hole_count)
        if seeds is None:
            return None

        for r, c in seeds:
            frame[r, c] = BG

        _, num_holes = label_mask(frame == BG, connectivity=4)
        if int(num_holes) != int(hole_count):
            return None

    frame[0, 0] = int(marker_color)
    return frame


def _pick_color_triples(n: int) -> tuple[list[int], list[int], list[int]]:
    available = [c for c in range(10) if c != BG]
    assert 3 * n <= len(available)
    colors = random.sample(available, 3 * n)
    marker_colors = [int(color) for color in colors[:n]]
    frame_colors = [int(color) for color in colors[n : 2 * n]]
    sprite_colors = [int(color) for color in colors[2 * n : 3 * n]]
    return marker_colors, frame_colors, sprite_colors


def _hole_lists_are_decoupled(frame_holes: list[int], sprite_holes: list[int]) -> bool:
    if sorted(frame_holes) == sorted(sprite_holes):
        return False
    return all(int(frame_holes[i]) != int(sprite_holes[i]) for i in range(len(frame_holes)))


def _build_expected_output(
    frames: list[np.ndarray],
    sprites: list[np.ndarray],
    frame_colors: list[int],
    sprite_colors: list[int],
    marker_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    for idx in range(len(frames)):
        ri, ci = divmod(idx, cols_grid)
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        frame_color = int(frame_colors[idx])
        sprite_color = int(sprite_colors[idx])
        marker_color = int(marker_colors[idx])

        expected[rr : rr + frame_size, cc : cc + frame_size] = sprite_color
        expected[rr, cc] = marker_color

        content = np.array(sprites[idx], dtype=int).copy()
        content[content == sprite_color] = frame_color
        content[content == BG] = sprite_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = int(frame_size - 2)

    grid_pixel_h = int(rows_grid * frame_size)
    grid_pixel_w = int(cols_grid * frame_size)
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(min_h, CANVAS_MAX)
    width = random.randint(min_w, CANVAS_MAX)

    marker_colors, frame_colors, sprite_colors = _pick_color_triples(n)

    frame_hole_max = min(MAX_VISIBLE_HOLES, _max_isolated_holes(frame_size))
    sprite_hole_max = min(MAX_VISIBLE_HOLES, _max_isolated_holes(sprite_size))
    frame_hole_choices = list(range(frame_hole_max + 1))
    sprite_hole_choices = list(range(sprite_hole_max + 1))

    for _ in range(HOLE_RETRIES):
        frame_holes = [int(random.choice(frame_hole_choices)) for _ in range(n)]
        sprite_holes = [int(random.choice(sprite_hole_choices)) for _ in range(n)]
        if not _hole_lists_are_decoupled(frame_holes, sprite_holes):
            continue
        if max(sprite_holes, default=0) == 0:
            continue
        break
    else:
        return None

    frames: list[np.ndarray] = []
    for idx in range(n):
        patch = _make_frame(frame_size, frame_holes[idx], frame_colors[idx], marker_colors[idx])
        if patch is None:
            return None
        frames.append(patch)

    sprites: list[np.ndarray] = []
    for idx in range(n):
        patch = _make_sprite(sprite_size, sprite_holes[idx], sprite_colors[idx], marker_colors[idx])
        if patch is None:
            return None
        sprites.append(patch)

    canvas = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - grid_pixel_h)
    c0 = random.randint(0, width - grid_pixel_w)
    for idx, frame_patch in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    sprite_order = list(range(n))
    random.shuffle(sprite_order)
    for idx in sprite_order:
        sprite_patch = sprites[idx]
        try:
            r, c = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames=frames,
        sprites=sprites,
        frame_colors=frame_colors,
        sprite_colors=sprite_colors,
        marker_colors=marker_colors,
        rows_grid=rows_grid,
        cols_grid=cols_grid,
        frame_size=frame_size,
        sprite_size=sprite_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
