# Variant of 8698868d
# Dimension: D4 — 变换方式：嵌套后的重着色 (transform)
# Change: recolor — 精灵嵌套入框架后，精灵内部的洞（背景色像素）重着色为框架的边框颜色... → 洞保留背景色，精灵色变为框架色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for the D4 inversion variant
============================================

a) Concrete behavior and execution trace
----------------------------------------
- Input stays in the original family:
  - Azure background (8).
  - A tight grid of larger square frames.
  - Same number of smaller square sprites scattered elsewhere.
  - Frames and sprites are still matched by the number of 4-connected holes.
- Original output:
  - Fill each output frame block with the frame color.
  - Nest the matched sprite inside.
  - Recolor the sprite's bg holes to the frame color, so the output contains no bg.
- Variant output:
  - Fill each output frame block with the frame color.
  - Nest the matched sprite inside.
  - Recolor the sprite's solid pixels to the frame color.
  - Leave the sprite's bg holes unchanged as bg.
  - Visually this is the exact inverse of the old recoloring: the hole pattern is now visible as azure cutouts inside each frame.

Small trace:
- Suppose a matched sprite is
    2 2 2 2
    2 8 8 2
    2 2 8 2
    2 2 2 2
  and the frame color is 4.
- Old logic changed every non-2 cell to 4, yielding a solid 4 block.
- New logic changes every 2 cell to 4 and keeps 8 as 8, yielding
    4 4 4 4
    4 8 8 4
    4 4 8 4
    4 4 4 4
- So the matching signal is still hole-count equality, but the displayed result is the negative of the old fill rule.

b) Complexity calibration
-------------------------
- The variant does not add new object classes or matching keys; hole-count matching remains the only combinatorial step.
- The visual effect is strong because any sprite with at least one hole now produces visible bg pixels in output.
- The generator already uses N in [2, 6] and consecutive hole counts, so every sample has at least one nonzero-hole sprite. That prevents total collapse back to the original "all solid" look.
- Colors remain limited to the same original budget, so the task does not become a color-lookup puzzle.

c) Implicit assumption review for main()
----------------------------------------
- Component finding assumes bg is fixed at 8. This remains true; generator still hardcodes 8.
- Square-region detection assumes frames are the largest square components and sprites the next largest size. The transform change happens only in output construction, so this assumption is unchanged.
- Frame hole counting uses `inner != frame_color`. Still valid because input frames are unchanged.
- Sprite hole counting uses `sprite_region != sprite_color`. Still valid because input sprites are unchanged.
- Matching uses a bijection on hole counts. Still valid because generator still creates a unique sprite per frame hole count.
- The old line `content[content != sprite_color] = frame_color` depended on the old property "holes must disappear".
  In the variant this is wrong because it would overwrite the very bg holes that must stay visible.
- The replacement `content[content == sprite_color] = frame_color` depends on the input sprite patch containing only `sprite_color` and bg.
  The generator guarantees exactly that.

d) Placement legality
---------------------
- Frames are still placed as a tight, non-overlapping grid on the canvas.
- Sprites are still placed only in remaining bg area via `random_free_location(..., transparent=BG, padding=1)`, so they do not overlap frames or each other.
- Output nesting still writes each sprite into the frame interior at offset (+1, +1), and sprite size remains `frame_size - 2`, so it exactly fits.
- Leaving holes as bg introduces no new overlap risk because those cells simply retain the prefilled bg in the nested content patch.

e) Visual observability
-----------------------
- In the original task the output has no bg pixels.
- In this variant, outputs for every nonzero-hole sprite visibly retain azure interior holes.
- Across 2-4 examples, a solver can infer:
  1. match by hole-count as before,
  2. but after insertion, keep the hole mask and recolor the solid sprite mass instead.
- Because at least one matched sprite has holes in every generated sample, the new behavior is directly observable instead of latent.

f) Visual cleanliness
---------------------
- The output still has exactly two layers per frame block: frame-colored mass and bg cutouts.
- No extra markers, annotations, or overlapping symbols are introduced.
- The hole silhouettes remain inside clean square blocks, so the inversion is visually crisp rather than cluttered.

g) Color-change inferability
----------------------------
- The only color rule is local and structural:
  - sprite solid cells -> frame color,
  - sprite hole cells -> background.
- This is easier to infer than arbitrary palette remapping because the preserved bg holes align with the same hole shapes seen in the input sprites.

Post-change assumptions that generator must guarantee
-----------------------------------------------------
- Each sprite patch uses exactly two colors: its own color and bg.
- At least one sprite has `hole_count > 0`, so the inversion is visible.
- Frame colors and sprite colors may overlap globally, but each monochromatic component remains isolated enough for component extraction.
"""


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


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    Variant transform:
    - fill the output block with the frame color;
    - overwrite sprite solid pixels with the frame color;
    - keep sprite holes as background.
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Find square regions; the lower fill ratio keeps holed sprites detectable.
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
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
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

    matches: dict[tuple[int, int], tuple[dict, dict]] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for i, sprite in enumerate(sprites):
            if i not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["out_r"]), int(frame["out_c"]))] = (frame, sprite)
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frame_grid:
        fr, fc = int(frame["out_r"]), int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        if (fr, fc) not in matches:
            continue

        _frame, sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])

        # Variant inversion: solid sprite mass fuses into the frame color,
        # while bg holes remain visible as bg cutouts.
        content[content == sprite_color] = frame_color

        sh, sw = int(sprite["h"]), int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` cells inside a bordered square, mutually 4-isolated."""
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
    """Build a sprite with solid border and exactly `hole_count` 4-connected bg holes."""
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

    _lab, num = label_mask(sprite != sprite_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build an input-style frame with exactly `hole_count` isolated bg holes."""
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
    if int(num) != int(hole_count):
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
) -> np.ndarray:
    """Build the expected output grid for the inversion recoloring variant."""
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_holes = {k: patch for k, patch in sprites}

    for idx, (hole_count, _frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        content = np.array(sprite_by_holes[hole_count], dtype=int).copy()
        content[content != BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    # N >= 2 and consecutive hole counts guarantee at least one k > 0,
    # so the inversion remains visible in every generated sample.
    sprites: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[i])
        if patch is None:
            return None
        frames.append((hole_count, patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_hole_count, frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _hole_count, sprite_patch in sprites:
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
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
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
