# Variant of 8698868d
# Dimension: D4 — 变换方式：嵌套后的重着色 (transform)
# Change: recolor — 精灵嵌套入框架后，精灵内部的洞（背景色像素）重着色为框架的边框颜色... → 洞重着色为第三种颜色（基于图例）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior:
#    - The original puzzle has two object families in the input:
#      1) large square frames arranged in a tight grid,
#      2) smaller square sprites scattered elsewhere.
#    - Matching still uses the original key: connected hole-component count.
#    - The D4 change adds a small legend elsewhere on the canvas. The legend is a
#      compact 2-row strip of 2x2 color chips:
#         top chip  = frame border color
#         bottom chip = hole-fill color to use in the output
#      Chips are vertically aligned by column, so each column is one mapping.
#    - Small trace:
#         original behavior:
#           red frame + sprite with 1 hole -> nested output fills the hole red.
#         variant behavior:
#           legend shows red above yellow -> same matched sprite is nested into the
#           red frame, but its hole becomes yellow instead of red.
#      The only rule change is the recolor target after matching; object detection,
#      hole counting, pairing, and grid compaction remain the same.
#
# b) Complexity calibration:
#    - The legend uses only 2x2 chips so it is visually obvious yet structurally
#      separate from frames/sprites (which are at least 4x4 and 6x6 respectively).
#    - Frame colors are distinct, fill colors are distinct, and every fill color is
#      different from its frame color; this prevents the variant from collapsing
#      back into the original recolor-to-frame behavior.
#    - The number of frames is reduced to 2-4. This is the smallest adaptation
#      needed to keep a one-to-one frame-color -> third-color legend feasible with
#      the 10-color ARC palette while preserving repeated support across examples.
#
# c) Hidden assumptions in the original solver:
#    - `find_components(..., monochromatic=True)` still works because legend chips
#      are disconnected monochromatic components.
#    - The square-region filter (`4 <= h <= 10`) relied on "every relevant object is
#      at least 4x4". That remains true for frames and sprites, while the legend is
#      intentionally 2x2 so it is excluded instead of misclassified.
#    - The original recolor line
#          content[content != sprite_color] = frame_color
#      depended on "hole fill is always the frame color". That is exactly the old
#      assumption that breaks in this variant. It must be replaced by a lookup via
#      the decoded legend mapping.
#    - The solver now assumes every frame color appearing in the frame grid is
#      present in the legend. The generator guarantees this by building one legend
#      entry per frame slot/color.
#
# d) Placement legality:
#    - The legend is generated as its own patch and placed in background using the
#      same non-overlap placement utility as sprites.
#    - A 1-cell padding is kept around legend and sprites so same-color objects do
#      not accidentally merge into one component and break detection.
#    - Frames still form a tight grid; sprites remain outside that grid.
#
# e) Visual observability:
#    - The new rule is inferable from 2-3 examples because the output now contains a
#      third color inside the frames, and the input explicitly exposes the mapping in
#      a repeated "top chip -> bottom chip" legend format.
#    - The generator enforces that at least one hole exists in each sample and that
#      the fill color differs from the matched sprite color, so the recolor remains
#      visible after nesting.
#
# f) Visual cleanliness:
#    - The legend is a single compact block with aligned columns, so it does not add
#      clutter to the input. It is also much smaller than the task objects, making
#      it easy to visually classify as metadata rather than another sprite/frame.
#
# g) Color-change inferability:
#    - The mapping is not hidden in numeric ids or ordering. It is encoded by spatial
#      pairing only: same-column chips form one color-transfer rule.
#    - Fill colors are pairwise distinct, which gives multiple independent examples
#      inside one puzzle instance and supports human inference from few examples.
#
# 9) New solver assumptions checked against the generator:
#    - Legend detection assumes all 2x2 monochromatic components belong to the legend.
#      The generator guarantees this by never creating any other standalone 2x2
#      components.
#    - Legend decoding assumes exactly two chip rows with aligned columns.
#      The generator always builds the legend in that canonical layout.
#    - Matching assumes unique hole counts across frames/sprites, unchanged from the
#      original generator, which still constructs consecutive unique counts.


BG = 8


def _extract_legend_mapping(comps: list[dict]) -> dict[int, int]:
    """Read the 2-row legend: top chips are frame colors, bottom chips are fill colors."""
    legend_blocks = []
    for comp in comps:
        b = comp["bbox"]
        if int(bbox_h(b)) == 2 and int(bbox_w(b)) == 2 and int(comp["area"]) == 4:
            legend_blocks.append(comp)

    assert len(legend_blocks) >= 4 and len(legend_blocks) % 2 == 0

    row_tops = sorted({int(comp["bbox"][0]) for comp in legend_blocks})
    assert len(row_tops) == 2
    top_r, bottom_r = row_tops
    assert bottom_r > top_r + 1

    top_by_c = {int(comp["bbox"][1]): comp for comp in legend_blocks if int(comp["bbox"][0]) == top_r}
    bottom_by_c = {int(comp["bbox"][1]): comp for comp in legend_blocks if int(comp["bbox"][0]) == bottom_r}
    common_cols = sorted(set(top_by_c) & set(bottom_by_c))

    assert common_cols
    assert len(common_cols) == len(top_by_c) == len(bottom_by_c)

    mapping: dict[int, int] = {}
    used_fills: set[int] = set()
    for c0 in common_cols:
        frame_color = int(top_by_c[c0]["color"])
        fill_color = int(bottom_by_c[c0]["color"])
        assert frame_color not in mapping
        assert fill_color not in used_fills
        mapping[frame_color] = fill_color
        used_fills.add(fill_color)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, nest each sprite inside its frame,
    and recolor sprite holes using a legend-defined third color.

    Input structure:
    - bg = azure (8).
    - Large square frames are arranged in a tight grid.
    - Smaller square sprites are scattered elsewhere.
    - A compact 2-row legend of 2x2 chips maps frame colors (top row) to fill colors
      (bottom row).

    Rule:
    - Count connected hole regions inside each frame interior and each sprite.
    - Pair each frame with the sprite sharing the same hole count.
    - Place the matched sprite inside the frame border.
    - Recolor sprite holes to the legend color associated with that frame color.
    """
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    legend_map = _extract_legend_mapping(comps)

    # Find square regions (lower fill threshold to catch sprites with holes).
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

    # Group by size; frames are larger, sprites are smaller.
    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames

    # Determine frame grid structure from visible frame positions.
    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid = []
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

    sprites = []
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
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if i not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = s
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_color = int(f["color"])
        fill_color = int(legend_map[frame_color])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        if (fr, fc) in matches:
            sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != sprite_color] = fill_color

            sh, sw = int(sprite["h"]), int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator (output-first, legend-driven recolor) ---------------

FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 4)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
LEGEND_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 400
LEGEND_BLOCK = 2
LEGEND_GAP = 1

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _legend_dims(n: int) -> tuple[int, int]:
    return 2 * LEGEND_BLOCK + LEGEND_GAP, 3 * n - 1


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` interior cells that are pairwise 4-separated."""
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
    """Build a sprite with exactly `hole_count` 4-connected bg-hole components."""
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
    if num != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build a frame patch with exactly `hole_count` isolated bg holes in the interior."""
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


def _make_legend(frame_colors: list[int], fill_colors: list[int]) -> np.ndarray:
    """Top row = frame colors; bottom row = mapped fill colors."""
    h, w = _legend_dims(len(frame_colors))
    legend = np.full((h, w), BG, dtype=int)
    bottom_r = LEGEND_BLOCK + LEGEND_GAP
    for i, (frame_color, fill_color) in enumerate(zip(frame_colors, fill_colors)):
        c0 = 3 * i
        legend[0:LEGEND_BLOCK, c0 : c0 + LEGEND_BLOCK] = int(frame_color)
        legend[bottom_r : bottom_r + LEGEND_BLOCK, c0 : c0 + LEGEND_BLOCK] = int(fill_color)
    return legend


def _choose_fill_colors(
    available: list[int],
    frame_colors: list[int],
    hole_counts: list[int],
    sprite_color_by_hole: dict[int, int],
) -> list[int] | None:
    """Choose distinct third colors, one per frame slot, avoiding the matched sprite color."""
    pool = [c for c in available if c not in set(frame_colors)]
    if len(pool) < len(frame_colors):
        return None

    for _ in range(120):
        fill_colors = random.sample(pool, len(frame_colors))
        if all(
            int(fill_colors[i]) != int(sprite_color_by_hole[hole_counts[i]])
            for i in range(len(hole_counts))
        ):
            return fill_colors
    return None


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    fill_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    """Build the output grid using legend colors for hole fill."""
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {k: spatch for k, spatch in sprites}
    for idx, (k, _) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_colors[idx])
        content = sprite_by_k[k].copy()
        content[content == BG] = int(fill_colors[idx])
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    legend_h, legend_w = _legend_dims(n)

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if grid_pixel_h + legend_h + 1 <= CANVAS_MAX:
        min_h = max(min_h, grid_pixel_h + legend_h + 1)
    if grid_pixel_w + legend_w + 1 <= CANVAS_MAX:
        min_w = max(min_w, grid_pixel_w + legend_w + 1)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n)
    sprite_colors = random.sample(available, n)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprite_color_by_hole = {
        int(k): int(sprite_colors[i]) for i, k in enumerate(sprite_hole_perm)
    }
    fill_colors = _choose_fill_colors(available, frame_colors, hole_counts, sprite_color_by_hole)
    if fill_colors is None:
        return None

    sprites: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, k, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((int(k), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(hole_counts):
        patch = _make_frame(frame_size, k, frame_colors[i])
        if patch is None:
            return None
        frames.append((int(k), patch))

    canvas = np.full((h, w), BG, dtype=int)
    r0 = random.randint(0, h - grid_pixel_h)
    c0 = random.randint(0, w - grid_pixel_w)
    for idx, (_, fpatch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    legend = _make_legend(frame_colors, fill_colors)
    try:
        legend_r, legend_c = random_free_location(
            canvas,
            legend,
            transparent=BG,
            padding=LEGEND_PLACEMENT_PAD,
        )
    except ValueError:
        return None
    paste(canvas, legend, legend_r, legend_c, transparent=BG)

    for _, spatch in sprites:
        try:
            sr, sc = random_free_location(
                canvas,
                spatch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, spatch, sr, sc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        fill_colors,
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
