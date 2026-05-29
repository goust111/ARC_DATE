# Combo variant of 8698868d (index 36)
# combo_spec: D1/0+D5/0+D6/0
# Name: 颜色匹配+全画布+保留精灵

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS_DIR = _ROOT / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for combo D1/0 + D5/0 + D6/0
# ----------------------------------------------
# a) Concrete behavior and trace
# - Keep the original object inventory: a tight grid of larger square frames plus the same
#   number of smaller square sprites scattered elsewhere on an azure canvas.
# - Replace the old "same connected-hole count" pairing rule with an explicit visible cue:
#   each frame and each sprite gets a 1-pixel marker at the top-left corner of its square,
#   and equal marker colors define the pair.
# - Change the output carrier from a cropped frame-only image to the full original canvas.
# - Keep the scattered source sprites in place. The frame receives a copied nested sprite,
#   so the output shows both the original source sprite and its in-frame copy.
# - Small trace:
#   original:
#     frame A has 2 holes, sprite B has 2 holes, so B is moved into A; output is cropped.
#   fused variant:
#     frame A has red top-left marker, sprite C has red top-left marker, so C is copied
#     into A even if their hole counts disagree; output stays full-size; sprite C also
#     remains visible at its original position.
#   key deltas:
#     counting -> color equality, cropped canvas -> full canvas, move/disappear -> copy/keep.
#
# b) Complexity calibration
# - Marker colors are unique per pair within a sample, so matching is bijective without
#   introducing large search space.
# - Hole patterns remain present as decoys and are intentionally decoupled from pairing:
#   frame and sprite hole multisets are forced to differ, and at least one side has repeats,
#   so the original hole-count strategy is visibly unreliable.
# - The output difference is large enough to notice: same-size canvas, source sprites still
#   present, and repeated marker colors on paired frame corners / nested sprite corners.
#
# c) Hidden-assumption audit for main()
# - Square-object detection by monochromatic components still works because the marker only
#   changes one corner cell; the main body component still spans the full square bbox.
# - Any code that counts `region != object_color` as holes would now misread the marker as
#   a hole. The fused solver must not use hole counts for matching and must recolor only BG
#   cells when nesting, preserving the marker pixel.
# - The original compact output coordinates `ri * frame_size, ci * frame_size` are invalid
#   under D5/D6. The fused solver must use the frames' original canvas coordinates and start
#   from `out = g.copy()` so source sprites survive.
# - Repainting the whole frame square remains necessary; otherwise the frame's input holes
#   would remain visible underneath the nested sprite.
#
# d) Placement legality
# - The generator places the frame grid with a positive outer margin and puts sprites only in
#   remaining background using `random_free_location(..., padding=1)`, so no overlaps occur.
# - The top-left marker stays inside the existing square bbox, so it does not alter geometry.
# - Sprite size is always `frame_size - 2`, guaranteeing exact fit inside the frame border.
#
# e) Visual observability
# - Across 2-4 examples, a human can spot that paired objects share the same corner marker
#   color, while hole counts vary and often repeat, so counting cannot explain the outputs.
# - Full-canvas outputs with unchanged scattered sprites make the copy-not-move behavior
#   directly visible without any hidden numeric convention.
#
# f) Visual cleanliness
# - Only one extra pixel is added per object, at a stable semantic location.
# - Frame regions are redrawn cleanly; outside the frame grid the output is unchanged input,
#   so there is no clutter from partial erasure or mixed layers.
#
# g) Color-change inferability
# - The only new color rule is direct equality of visible corner markers.
# - Sprite body colors stay unchanged; only sprite BG holes are recolored to the host frame
#   color, which remains locally justified by the nesting relation.
#
# New fused assumptions that the generator must guarantee
# - Every frame and sprite has its marker at `[0, 0]` and marker colors are unique per pair.
# - All frame body colors are disjoint from the full marker-color set, and for every pair index i
#   `sprite_color[i] != marker_color[i]`, so markers stay visible and adjacent frames never merge
#   through a neighbor's corner marker.
# - Frame/sprite hole data are decoupled enough that hole-count matching does not reproduce
#   the intended answer.
# - The area outside the frame grid contains visible source sprites, so keep-source is
#   observable rather than vacuous.

BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
# Cap the visible frame count at 4 so frame body colors can stay disjoint from all
# marker colors; this prevents tight-grid frames from merging through neighbor markers.
N_RANGE = (2, 4)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
FRAME_MARGIN = 1
HOLE_RETRIES = 80
SEED_RETRIES = 50
GEN_RETRIES = 400
MAX_VISIBLE_HOLES = 3
MIN_FILL_RATIO = 0.5


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _find_square_regions(grid: np.ndarray) -> dict[int, list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((h, comp))
    if not square_regions:
        raise AssertionError("no square regions found")

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)
    return sizes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by same-color corner markers, copy each sprite into its paired
    frame, and keep the full original canvas with the scattered source sprites still present.

    Input structure:
    - Background is azure (8). Large square frames form a tight grid somewhere on the canvas.
    - Smaller square sprites of size `frame_size - 2` are scattered elsewhere on background.
    - Every frame and every sprite carries a 1-pixel marker at the top-left corner of its
      square; matched pairs share that marker color.

    Rule:
    - Detect the larger frame squares and the smaller sprite squares from their monochrome
      main-body components.
    - Pair a frame with the sprite whose top-left marker color is the same.
    - Keep the whole input canvas in the output. Rebuild each frame in place, then place a
      copy of its matched sprite inside the border, recoloring only the sprite's BG holes to
      the frame color. Do not erase or move the original scattered sprite.

    Key insight:
    - The reliable cue is the repeated marker color at a fixed semantic location, not the
      number of holes. The output is a full-canvas copy-and-nest transformation, not a crop
      and not a move.
    """
    g = np.array(grid, dtype=int)
    sizes = _find_square_regions(g)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    if not frames or not sprites_raw:
        raise AssertionError("missing frames or sprites")

    frame_grid: list[dict] = []
    for frame_comp in frames:
        bbox = frame_comp["bbox"]
        inp_r = int(bbox[0])
        inp_c = int(bbox[1])
        region = crop_bbox(g, bbox)
        border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
        frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
        frame_grid.append(
            {
                "color": frame_color,
                "marker": int(region[0, 0]),
                "inp_r": int(inp_r),
                "inp_c": int(inp_c),
            }
        )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprites.append(
            {
                "marker": int(sprite_region[0, 0]),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    sprites_by_marker = {int(sprite["marker"]): sprite for sprite in sprites}
    if len(sprites_by_marker) != len(sprites):
        raise AssertionError("non-unique sprite markers")

    out = g.copy()
    for frame in frame_grid:
        fr = int(frame["inp_r"])
        fc = int(frame["inp_c"])
        frame_color = int(frame["color"])
        marker_color = int(frame["marker"])
        sprite = sprites_by_marker.get(marker_color)
        if sprite is None:
            raise AssertionError("missing sprite for marker")

        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)
        out[fr, fc] = int(marker_color)

        content = np.array(sprite["content"], dtype=int).copy()
        content[content == BG] = int(frame_color)
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _max_isolated_holes(size: int) -> int:
    inner = size - 2
    return (inner * inner + 1) // 2


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


def _make_sprite(size: int, hole_count: int, sprite_color: int, marker_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count > 0:
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
        for comp_idx, comp in enumerate(comps):
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

        _lab, num = label_mask(sprite == BG, connectivity=4)
        if int(num) != hole_count:
            return None

    sprite[0, 0] = int(marker_color)
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int, marker_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count > 0:
        inner_cells = (frame_size - 2) ** 2
        if inner_cells < hole_count:
            return None

        seeds = _place_isolated_seeds(frame_size, hole_count)
        if seeds is None:
            return None

        for r, c in seeds:
            frame[r, c] = BG

        _lab, num = label_mask(frame == BG, connectivity=4)
        if int(num) != hole_count:
            return None

    frame[0, 0] = int(marker_color)
    return frame


def _pick_frame_colors(n: int, markers: list[int]) -> list[int]:
    available = [c for c in range(10) if c != BG and c not in markers]
    if len(available) < n:
        raise RuntimeError("_pick_frame_colors: not enough colors disjoint from markers")
    return random.sample(available, n)


def _pick_body_colors(n: int, markers: list[int]) -> list[int]:
    available = [c for c in range(10) if c != BG]
    for _ in range(200):
        colors = random.sample(available, n)
        if all(int(colors[i]) != int(markers[i]) for i in range(n)):
            return colors
    raise RuntimeError("_pick_body_colors: failed to avoid marker collisions")


def _hole_lists_are_decoupled(frame_holes: list[int], sprite_holes: list[int]) -> bool:
    if sorted(frame_holes) == sorted(sprite_holes):
        return False
    if not any(int(frame_holes[i]) != int(sprite_holes[i]) for i in range(len(frame_holes))):
        return False
    return len(set(frame_holes)) < len(frame_holes) or len(set(sprite_holes)) < len(sprite_holes)


def _build_expected_output(
    canvas: np.ndarray,
    frame_records: list[dict],
    sprites: list[np.ndarray],
    frame_colors: list[int],
    marker_colors: list[int],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.array(canvas, dtype=int).copy()
    for idx, frame in enumerate(frame_records):
        rr = int(frame["r"])
        cc = int(frame["c"])
        frame_color = int(frame_colors[idx])
        marker_color = int(marker_colors[idx])

        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        expected[rr, cc] = int(marker_color)

        content = np.array(sprites[idx], dtype=int).copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING + 2 * FRAME_MARGIN)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING + 2 * FRAME_MARGIN)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    marker_colors = random.sample(available, n_frames)
    frame_colors = _pick_frame_colors(n_frames, marker_colors)
    sprite_colors = _pick_body_colors(n_frames, marker_colors)

    frame_hole_max = min(MAX_VISIBLE_HOLES, _max_isolated_holes(frame_size))
    sprite_hole_max = min(MAX_VISIBLE_HOLES, _max_isolated_holes(sprite_size))
    if frame_hole_max < 0 or sprite_hole_max < 0:
        return None

    frame_hole_choices = list(range(frame_hole_max + 1))
    sprite_hole_choices = list(range(sprite_hole_max + 1))
    for _ in range(HOLE_RETRIES):
        frame_holes = [random.choice(frame_hole_choices) for _ in range(n_frames)]
        sprite_holes = [random.choice(sprite_hole_choices) for _ in range(n_frames)]
        if _hole_lists_are_decoupled(frame_holes, sprite_holes):
            break
    else:
        return None

    frames: list[np.ndarray] = []
    for idx in range(n_frames):
        patch = _make_frame(frame_size, frame_holes[idx], frame_colors[idx], marker_colors[idx])
        if patch is None:
            return None
        frames.append(patch)

    sprites: list[np.ndarray] = []
    for idx in range(n_frames):
        patch = _make_sprite(sprite_size, sprite_holes[idx], sprite_colors[idx], marker_colors[idx])
        if patch is None:
            return None
        sprites.append(patch)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(FRAME_MARGIN, canvas_h - grid_pixel_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, canvas_w - grid_pixel_w - FRAME_MARGIN)
    frame_records: list[dict] = []
    for idx, frame_patch in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch
        frame_records.append({"r": int(rr), "c": int(cc)})

    sprite_order = list(range(n_frames))
    random.shuffle(sprite_order)
    for idx in sprite_order:
        sprite_patch = sprites[idx]
        try:
            rr, cc = random_free_location(canvas, sprite_patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        canvas=canvas,
        frame_records=frame_records,
        sprites=sprites,
        frame_colors=frame_colors,
        marker_colors=marker_colors,
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
