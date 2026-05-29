# Variant of 8698868d
# Dimension: D4 — 变换方式：嵌套后的重着色 (transform)
# Change: recolor — 精灵嵌套入框架后，精灵内部的洞（背景色像素）重着色为框架的边框颜色
# ... → 颜色互换：精灵色与框架色交换

"""
Design analysis for the D4 variant
==================================

a) Concrete behavior and trace
The input stays in the original family:
- azure background (8)
- a tight grid of larger square frames
- scattered smaller square sprites
- frames and sprites are still matched by the count of 4-connected hole components

Only the output transform changes. In the original task, after nesting a matched
sprite inside a frame, every sprite hole pixel (BG) is recolored to the frame
color, so the output contains no BG. In this variant, the nesting is followed by
a true color swap between the two objects:
- frame border color becomes the matched sprite color
- sprite solid pixels become the original frame color
- sprite hole pixels stay BG

Small trace:
- Suppose a red frame with hole-count 2 matches a green sprite with hole-count 2.
- Original output: border stays red, green sprite stays green, sprite holes become red.
- Variant output: border becomes green, the sprite's solid pixels become red, and
  the sprite holes remain azure.
The geometry is unchanged; only which region receives which color changes.

b) Complexity calibration
The matching key remains hole-count, so the variant does not add a second hidden
matching rule. The only extra requirement is observability: if a matched frame and
sprite use the same color, swapping would be invisible. The generator therefore
guarantees frame-color != matched-sprite-color for every pair by assigning sprite
colors as a derangement of frame colors.

c) Implicit-assumption audit of the original main()
Relevant original lines and their old assumptions:
- `out[...] = frame_color`
  Old assumption: the whole output tile can start as the frame color because the
  final border color is still the frame color. This is no longer true.
- `content[content != sprite_color] = frame_color`
  Old assumption: only non-solid sprite pixels need recoloring, and they should
  become the frame color. In the variant, the solid sprite pixels must change
  color instead, while holes must remain BG.
- Writing the sprite patch into the interior still works because sprite_size is
  exactly frame_size - 2, so the interior is still fully covered by the nested
  sprite patch.

d) Placement legality
The generator keeps the original placement discipline:
- frame grid is tight and axis-aligned
- sprites are placed with padding in remaining BG area
- no overlaps, no out-of-bounds placements
- sprites remain disconnected from frames, so monochromatic component detection
  still separates them correctly

e) Visual observability
This variant is easy to see in 2-3 examples because the entire border changes to
the sprite color, and the sprite body changes to the frame color. We additionally
keep nonzero hole counts in most samples through the consecutive hole-count range,
so the preserved BG holes visibly distinguish this variant from the original
recolor-to-solid behavior.

f) Visual cleanliness
No new overlays are introduced. Each output tile still has three clearly legible
regions: border, sprite solid pixels, and sprite holes. Since the sprite fills the
full interior exactly, there is no ambiguous uncovered area.

g) Color-change inferability
The color rule is local and binary: each matched pair swaps its two visible object
colors. The generator keeps one frame color and one sprite color per pair, and the
hole-count match supplies the structural cue for deciding which scattered sprite
belongs to which frame.

Post-change assumptions checked against the generator
====================================================
- main() assumes there is a unique sprite for each hole-count; generator still
  constructs a bijection over consecutive hole-counts.
- main() assumes nested sprite size equals frame interior size; generator still
  uses sprite_size = frame_size - 2.
- main() now benefits from frame-color != matched-sprite-color for observability;
  generator explicitly guarantees that, but solver correctness does not depend on
  it.
"""

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, nest each sprite inside its
    frame, then swap the two object colors in the nested result.
    """
    g = np.array(grid, dtype=int)
    BG = 8

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
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if i not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = (f, s)
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for f in frame_grid:
        fr = int(f["out_r"])
        fc = int(f["out_c"])
        frame_color = int(f["color"])

        border_color = frame_color
        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            border_color = int(sprite["color"])

        out[fr : fr + frame_size, fc : fc + frame_size] = int(border_color)

        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content == sprite_color] = int(frame_color)

            sh = int(sprite["h"])
            sw = int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


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

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier = []
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
    if int(num) != int(hole_count):
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

    _, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _build_expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {int(s["holes"]): s for s in sprites}
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        frame_color = int(frame["color"])
        sprite = sprite_by_k[int(frame["holes"])]
        border_color = int(sprite["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = border_color
        content = np.array(sprite["patch"], dtype=int).copy()
        content[content == int(sprite["color"])] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
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

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, N)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - N)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + N))
    random.shuffle(hole_counts)

    frames: list[dict] = []
    for i, k in enumerate(hole_counts):
        patch = _make_frame(frame_size, k, frame_colors[i])
        if patch is None:
            return None
        frames.append({"holes": int(k), "color": int(frame_colors[i]), "patch": patch})

    # Derangement of frame colors by one-step rotation guarantees every matched
    # pair has different colors while keeping sprite colors distinct.
    rotated_colors = frame_colors[1:] + frame_colors[:1]
    sprite_color_by_hole = {int(frame["holes"]): int(rotated_colors[i]) for i, frame in enumerate(frames)}

    sprite_hole_order = hole_counts[:]
    random.shuffle(sprite_hole_order)
    sprites: list[dict] = []
    for k in sprite_hole_order:
        sprite_color = int(sprite_color_by_hole[int(k)])
        patch = _make_sprite_interior(sprite_size, int(k), sprite_color)
        if patch is None:
            return None
        sprites.append({"holes": int(k), "color": sprite_color, "patch": patch})

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    for sprite in sprites:
        try:
            r, c = random_free_location(
                canvas,
                sprite["patch"],
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite["patch"], r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    # Reject invisible swaps: every matched pair must have distinct colors.
    for frame in frames:
        if int(frame["color"]) == int(sprite_color_by_hole[int(frame["holes"])]):
            return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
