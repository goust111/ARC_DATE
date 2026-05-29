# Combo variant of 8698868d (index 17)
# combo_spec: D5/0+D6/2
# Name: 全画布输出精灵位填框架色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - Input stays in the same object family as the base task: a tight grid of larger square frames and the same
#      number of smaller square sprites scattered elsewhere on an azure background.
#    - Matching still depends only on the number of 4-connected hole components. Example:
#      frame A has color 4 and 2 isolated interior holes; sprite B has color 1 and exactly 2 hole components.
#      Original task: output only the compact frame grid, with sprite B inserted into frame A and B's hole pixels
#      recolored to 4.
#      This combo: keep the whole canvas, perform that same in-frame insertion at frame A's original location, and
#      also repaint sprite B's original square bbox as a solid 4 block. The frame and the source block therefore share
#      one color and act as a visible pairing cue.
#    - The changed stage is output construction only; object detection and pairing logic stay structural.
#
# b) Complexity calibration
#    - The variant keeps the original nontrivial reasoning step: infer pairings from connected-hole count rather than
#      position or color.
#    - The generator uses each hole count exactly once on the frame side and once on the sprite side, preventing
#      ambiguous matches.
#    - To make the recolored source blocks visible, matched sprite/frame colors are forced to differ. Otherwise a sprite
#      with zero holes could already look like the solid source block and visually hide the D6 part of the combo.
#
# c) Implicit-assumption audit of the original solver
#    - Square detection by size and fill ratio still holds because the input geometry is unchanged.
#    - "Largest squares are frames, next size down are sprites" still holds because frames remain size 6-8 and sprites
#      remain frame_size-2.
#    - The original compact output coordinates (`ri * frame_size`, `ci * frame_size`) are no longer valid under D5/0:
#      output must preserve full-canvas coordinates.
#    - The original assumption that sprites disappear is broken by D6/2. The solver must retain every sprite bbox and
#      explicitly repaint it after matching.
#    - A new assumption appears: sprite source bboxes can be overwritten as solid squares without colliding with other
#      objects. The generator enforces this with non-overlapping placement and padding.
#
# d) Placement legality
#    - Frames are pasted as a tight grid fully inside the canvas.
#    - Sprites are placed only in remaining background via `random_free_location(..., padding=1)`, so no sprite overlaps
#      the frame grid or another sprite, and turning a source sprite into a solid block stays in-bounds and isolated.
#    - Canvas size is chosen to fit both the frame grid and extra sprite area.
#
# e) Visual observability
#    - Across a few examples, humans can see two simultaneous regularities:
#      1. the output is no longer cropped to the frame grid;
#      2. each removed sprite becomes a same-colored solid block that matches one transformed frame.
#    - Because there are 2-6 matched pairs per sample, the pairing cue repeats enough times to be learnable.
#
# f) Visual cleanliness
#    - The output contains three clear layers only: background, transformed frames, and separated source blocks.
#    - The generator avoids sprite touching, so recolored source blocks do not merge into larger accidental shapes.
#
# g) Change inferability
#    - Color transfer is not arbitrary: the source block and the holes inside the destination frame both take the
#      matched frame color, while the match itself is explained by equal hole-count structure.
#    - No ordering depends on numeric color ids or any non-visual signal.
#
# h) Non-degeneracy check target
#    - A sample is only interesting if at least one sprite source location changes from its own color pattern into a
#      solid block of a different frame color. The generator enforces this for every matched pair by construction.
#
# i) New solver assumptions after the combo
#    - Every frame slot in the detected lattice is a real frame-sized square region whose border majority color is the
#      frame color; the generator builds frames exactly this way.
#    - Hole counts form a bijection between frames and sprites; the generator keeps one unique sprite per hole count.
#    - Sprite source blocks are smaller than frames, so repainting them cannot accidentally overwrite frame borders; the
#      generator preserves the base size relationship `sprite_size = frame_size - 2`.


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
FRAME_MARGIN = 1
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


def _extract_frame_grid(grid: np.ndarray, frame_size: int, frames: list[dict]) -> list[dict]:
    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = grid[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            _lab, num_holes = label_mask(inner != frame_color, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "inp_r": inp_r,
                    "inp_c": inp_c,
                    "holes": int(num_holes),
                }
            )
    return frame_grid


def _extract_sprites(grid: np.ndarray, sprites_raw: list[dict]) -> list[dict]:
    sprites: list[dict] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(grid, b)
        sprite_color = int(comp["color"])
        _lab, num_holes = label_mask(sprite_region != sprite_color, connectivity=4)
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
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair sprites with frames by connected-hole count, embed the sprites into the on-canvas frame grid,
    and preserve the whole input canvas while recoloring each source sprite location.

    Input structure:
    - The background is azure (8). Larger square frames form a tight rectangular grid somewhere on the canvas.
    - Smaller square sprites sit elsewhere on background cells. Frames and sprites each use solid non-background colors,
      while their hole cells appear as background-colored gaps inside the squares.

    Rule:
    - Detect the large square frames and the smaller square sprites.
    - For each frame and sprite, count how many 4-connected hole components it contains.
    - Match each frame to the unique sprite with the same hole count.
    - Output the full canvas. At every frame position, place the matched sprite inside the frame and recolor the
      sprite's hole pixels to the frame color. At every sprite's original position, repaint the entire sprite square as
      a solid block of that matched frame color.

    Key insight:
    - The decisive pairing cue is structural: equal numbers of connected hole regions. The shared color between a
      transformed frame and a recolored source block is an effect of that pairing, not the way to discover it.
    """
    g = np.array(grid, dtype=int)
    sizes = _find_square_regions(g)

    size_list = sorted(sizes.keys(), reverse=True)
    if len(size_list) < 2:
        raise AssertionError("expected both frame and sprite square sizes")

    frame_size = int(size_list[0])
    sprite_size = int(size_list[1])

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    if not frames:
        raise AssertionError("no frames found")
    if not sprites_raw:
        raise AssertionError("no sprites found")

    frame_grid = _extract_frame_grid(g, frame_size, frames)
    sprites = _extract_sprites(g, sprites_raw)

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for idx, sprite in enumerate(sprites):
            if idx not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["inp_r"]), int(frame["inp_c"]))] = sprite
                used_sprites.add(idx)
                break

    if len(matches) != len(frame_grid) or len(used_sprites) != len(sprites):
        raise AssertionError("frames and sprites must match bijectively by hole count")

    out = np.full_like(g, BG)
    for frame in frame_grid:
        fr = int(frame["inp_r"])
        fc = int(frame["inp_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

        sr = int(sprite["inp_r"])
        sc = int(sprite["inp_c"])
        out[sr : sr + sh, sc : sc + sw] = frame_color

    return out


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

    _lab, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
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

    _lab, num = label_mask(frame != frame_color, connectivity=4)
    if num != hole_count:
        return None
    return frame


def _choose_sprite_colors(frame_color_by_k: dict[int, int]) -> dict[int, int]:
    available = [c for c in range(10) if c != BG]
    sprite_color_by_k: dict[int, int] = {}
    used: set[int] = set()
    hole_keys = list(frame_color_by_k.keys())
    random.shuffle(hole_keys)
    for holes in hole_keys:
        choices = [c for c in available if c not in used and c != int(frame_color_by_k[holes])]
        if not choices:
            raise AssertionError("no valid sprite colors")
        color = random.choice(choices)
        sprite_color_by_k[int(holes)] = int(color)
        used.add(int(color))
    return sprite_color_by_k


def _build_expected_output(
    canvas_shape: tuple[int, int],
    frames: list[dict],
    sprites: list[dict],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full(canvas_shape, BG, dtype=int)
    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    frame_color_by_holes = {int(frame["holes"]): int(frame["color"]) for frame in frames}

    for frame in frames:
        rr = int(frame["r"])
        cc = int(frame["c"])
        frame_color = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        sprite = sprite_by_holes[int(frame["holes"])]
        content = np.array(sprite["patch"], dtype=int).copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    for sprite in sprites:
        rr = int(sprite["r"])
        cc = int(sprite["c"])
        frame_color = int(frame_color_by_holes[int(sprite["holes"])])
        expected[rr : rr + sprite_size, cc : cc + sprite_size] = frame_color

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

    available = [color for color in range(10) if color != BG]
    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)

    frame_colors = random.sample(available, n_frames)
    frame_color_by_holes = {int(holes): int(color) for holes, color in zip(hole_counts, frame_colors)}
    sprite_color_by_holes = _choose_sprite_colors(frame_color_by_holes)

    sprite_order = hole_counts[:]
    random.shuffle(sprite_order)
    sprites: list[dict] = []
    for holes in sprite_order:
        patch = _make_sprite_interior(sprite_size, int(holes), int(sprite_color_by_holes[int(holes)]))
        if patch is None:
            return None
        sprites.append(
            {
                "holes": int(holes),
                "color": int(sprite_color_by_holes[int(holes)]),
                "patch": patch,
            }
        )

    frames: list[dict] = []
    for holes in hole_counts:
        patch = _make_frame(frame_size, int(holes), int(frame_color_by_holes[int(holes)]))
        if patch is None:
            return None
        frames.append(
            {
                "holes": int(holes),
                "color": int(frame_color_by_holes[int(holes)]),
                "patch": patch,
            }
        )

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(FRAME_MARGIN, canvas_h - grid_pixel_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, canvas_w - grid_pixel_w - FRAME_MARGIN)
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

    expected = _build_expected_output(canvas.shape, frames, sprites, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
