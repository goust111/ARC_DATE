# Combo variant of 8698868d (index 15)
# combo_spec: D4/3+D16/0
# Name: 图例定义第三色填洞

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - This combo merges two changes into one shared legend:
#      1) D16/0: sprite/frame pairing is no longer inferred from equal hole count.
#         The legend explicitly says which sprite color belongs to which frame color.
#      2) D4/3: sprite hole pixels are no longer recolored to the frame border color.
#         The same legend also supplies a third fill color for that frame.
#    - Each legend entry is a compact 3-row strip:
#         row 0 = sprite color
#         row 1 = frame color
#         row 2 = third fill color
#      So one vertical entry means:
#         sprite_color -> frame_color, and that frame's holes fill with fill_color.
#
# b) Complexity calibration:
#    - The task now needs three color roles per object family. To keep the mapping
#      visually clean and fully distinct within the 10-color ARC palette, object
#      count is reduced to 2-3.
#    - All frames and all sprites still carry unique hole counts, but the generator
#      forces the legend pairing to be a derangement of the hole-count pairing, so
#      "match by holes" is visibly wrong.
#    - Every sprite has at least one hole and every fill color differs from both the
#      matched sprite color and the destination frame color, so the third-color fill
#      effect is always observable.
#
# c) Hidden-assumption review:
#    - The original solver assumed hole count was the true matching key. That is no
#      longer valid: hole counts remain only as distractor structure.
#    - The original recolor step assumed hole pixels should become frame_color.
#      That is also no longer valid and must be replaced by a legend lookup.
#    - The square-object detector still works because frames and sprites remain
#      square 4x4+ objects, while legend strips are only 1x2 monochrome components.
#
# d) Placement legality and cleanliness:
#    - The legend is fixed to rows 0-2, with the frame grid starting at row >= 4.
#      This prevents overlap and makes the legend visibly metadata.
#    - Sprites are still scattered into background with padding, so the three object
#      families stay visually separate: legend band, frame grid, loose sprites.
#
# e) Observability:
#    - The same 3-row legend motif repeats for every mapping entry in every sample.
#    - Because the correct output differs from both single-variant behaviors
#      (legend-pairing-only and third-color-only), the generator rejects cases that
#      would collapse into one of those simpler tasks.


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 3)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
GEN_RETRIES = 400
LEGEND_H = 3
LEGEND_CELL_W = 2
LEGEND_GAP = 1
LEGEND_RESERVED_ROWS = 4


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _count = _rows * _cols
        if N_RANGE[0] <= _count <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_count, []).append((_rows, _cols))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _legend_width(count: int) -> int:
    return int(count * LEGEND_CELL_W + max(0, count - 1) * LEGEND_GAP)


def _max_isolated_components(size: int) -> int:
    inner_side = size - 2
    return int((inner_side * inner_side + 1) // 2)


def _random_derangement(values: list[int]) -> list[int] | None:
    if len(values) <= 1:
        return None
    if len(values) == 2:
        return [int(values[1]), int(values[0])]
    for _ in range(100):
        perm = values[:]
        random.shuffle(perm)
        if all(int(left) != int(right) for left, right in zip(values, perm)):
            return [int(v) for v in perm]
    return None


def _read_legend_info(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int]]:
    """Read the top-edge legend: frame_color -> sprite_color and fill_color."""
    h, w = grid.shape
    assert h >= LEGEND_H

    band = (grid[:LEGEND_H, :] != int(bg)).any(axis=0)
    assert bool(band.any())

    frame_to_sprite: dict[int, int] = {}
    frame_to_fill: dict[int, int] = {}
    used_sprite_colors: set[int] = set()
    used_fill_colors: set[int] = set()

    c = 0
    while c < w:
        if not bool(band[int(c)]):
            c += 1
            continue
        c0 = int(c)
        while c + 1 < w and bool(band[int(c + 1)]):
            c += 1
        c1 = int(c)
        assert c1 - c0 + 1 == LEGEND_CELL_W

        block = grid[:LEGEND_H, c0 : c1 + 1]
        assert bool((block != int(bg)).all())

        row_colors = [{int(v) for v in block[row, :].tolist()} for row in range(LEGEND_H)]
        assert all(len(colors) == 1 for colors in row_colors)

        sprite_color = next(iter(row_colors[0]))
        frame_color = next(iter(row_colors[1]))
        fill_color = next(iter(row_colors[2]))

        assert len({sprite_color, frame_color, fill_color}) == 3
        assert frame_color not in frame_to_sprite
        assert sprite_color not in used_sprite_colors
        assert fill_color not in used_fill_colors

        frame_to_sprite[int(frame_color)] = int(sprite_color)
        frame_to_fill[int(frame_color)] = int(fill_color)
        used_sprite_colors.add(int(sprite_color))
        used_fill_colors.add(int(fill_color))
        c += 1

    assert frame_to_sprite and frame_to_fill
    assert set(frame_to_sprite) == set(frame_to_fill)
    return frame_to_sprite, frame_to_fill


def _count_holes(region: np.ndarray, solid_color: int) -> int:
    hole_mask = np.array(region, dtype=int) != int(solid_color)
    _labels, num_holes = label_mask(hole_mask, connectivity=4)
    return int(num_holes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair sprites to frames through a legend and fill their holes with a third color.

    Input structure:
    - bg is azure (8).
    - The top edge contains a 3-row legend band:
      row 0 shows sprite colors, row 1 shows frame colors, row 2 shows fill colors.
    - Elsewhere there is a tight grid of larger square frames and a scattered set
      of smaller square sprites.

    Rule:
    - Read the legend to determine which sprite color belongs in each frame color
      and which third color should fill that frame's sprite holes.
    - Detect the large square frames and the smaller square sprites.
    - Nest the legend-matched sprite inside each frame and recolor the sprite's
      hole pixels to the legend-defined third color for that frame.

    Key insight:
    - Hole counts are still visible but are no longer the pairing rule; the legend
      controls both the sprite-to-frame assignment and the final hole-fill color.
    """
    g = np.array(grid, dtype=int)
    frame_to_sprite, frame_to_fill = _read_legend_info(g, BG)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        bbox = comp["bbox"]
        height = int(bbox_h(bbox))
        width = int(bbox_w(bbox))
        area = int(comp["area"])
        if height == width and 4 <= height <= 10 and area >= int(height * height * min_fill_ratio):
            square_regions.append((int(height), comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    assert len(size_list) >= 2
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1])

    frames = sizes.get(int(frame_size), [])
    sprites_raw = sizes.get(int(sprite_size), [])
    assert frames and sprites_raw
    assert len(frames) == len(frame_to_sprite) == len(sprites_raw)

    frame_positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict[str, int]] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites_by_color: dict[int, dict[str, np.ndarray | int]] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        sprites_by_color[int(sprite_color)] = {
            "color": int(sprite_color),
            "content": sprite_region.copy(),
            "h": int(sprite_region.shape[0]),
            "w": int(sprite_region.shape[1]),
        }

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        sprite_color = int(frame_to_sprite[frame_color])
        fill_color = int(frame_to_fill[frame_color])
        assert sprite_color in sprites_by_color

        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite = sprites_by_color[int(sprite_color)]
        content = np.array(sprite["content"], dtype=int).copy()
        content[content != int(sprite_color)] = int(fill_color)

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


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


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), int(sprite_color), dtype=int)
    inner_cells = (size - 2) ** 2
    if hole_count <= 0 or hole_count > _max_isolated_components(size) or inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = int(BG)

    max_growth = max(1, inner_cells // hole_count - 1)
    for comp_index, comp in enumerate(components):
        growth_steps = random.randint(0, min(max_growth, inner_cells - len(all_holes)))
        for _ in range(growth_steps):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for other_index, other in enumerate(components)
                        if other_index != comp_index
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_holes.add(cell)
            sprite[cell[0], cell[1]] = int(BG)

    if _count_holes(sprite, sprite_color) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    inner_cells = (frame_size - 2) ** 2
    if hole_count <= 0 or hole_count > _max_isolated_components(frame_size) or inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = int(BG)

    if _count_holes(frame, frame_color) != int(hole_count):
        return None
    return frame


def _make_legend(sprite_colors: list[int], frame_colors: list[int], fill_colors: list[int]) -> np.ndarray:
    width = _legend_width(len(frame_colors))
    legend = np.full((LEGEND_H, width), int(BG), dtype=int)
    for index, (sprite_color, frame_color, fill_color) in enumerate(
        zip(sprite_colors, frame_colors, fill_colors)
    ):
        c0 = int(index * (LEGEND_CELL_W + LEGEND_GAP))
        legend[0, c0 : c0 + LEGEND_CELL_W] = int(sprite_color)
        legend[1, c0 : c0 + LEGEND_CELL_W] = int(frame_color)
        legend[2, c0 : c0 + LEGEND_CELL_W] = int(fill_color)
    return legend


def _build_expected_output(
    frame_colors: list[int],
    sprite_colors: list[int],
    fill_colors: list[int],
    sprite_patches: dict[int, np.ndarray],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), int(BG), dtype=int)
    for index, frame_color in enumerate(frame_colors):
        rr = int((index // cols_grid) * frame_size)
        cc = int((index % cols_grid) * frame_size)
        sprite_color = int(sprite_colors[index])
        fill_color = int(fill_colors[index])
        content = sprite_patches[int(sprite_color)].copy()
        content[content == int(BG)] = int(fill_color)
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _build_d4_only_output(
    frame_holes: list[int],
    frame_colors: list[int],
    fill_colors: list[int],
    sprite_by_holes: dict[int, np.ndarray],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    baseline = np.full((rows_grid * frame_size, cols_grid * frame_size), int(BG), dtype=int)
    for index, hole_count in enumerate(frame_holes):
        rr = int((index // cols_grid) * frame_size)
        cc = int((index % cols_grid) * frame_size)
        content = sprite_by_holes[int(hole_count)].copy()
        content[content == int(BG)] = int(fill_colors[index])
        baseline[rr : rr + frame_size, cc : cc + frame_size] = int(frame_colors[index])
        baseline[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return baseline


def _build_d16_only_output(
    frame_colors: list[int],
    sprite_colors: list[int],
    sprite_patches: dict[int, np.ndarray],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    baseline = np.full((rows_grid * frame_size, cols_grid * frame_size), int(BG), dtype=int)
    for index, frame_color in enumerate(frame_colors):
        rr = int((index // cols_grid) * frame_size)
        cc = int((index % cols_grid) * frame_size)
        sprite_color = int(sprite_colors[index])
        content = sprite_patches[int(sprite_color)].copy()
        content[content == int(BG)] = int(frame_color)
        baseline[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        baseline[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return baseline


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[count])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    legend_w = _legend_width(count)

    max_common_holes = min(_max_isolated_components(frame_size), _max_isolated_components(sprite_size))
    if max_common_holes < count or max_common_holes < 1:
        return None

    grid_h = rows_grid * frame_size
    grid_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, LEGEND_RESERVED_ROWS + grid_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, max(grid_w, legend_w) + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    if 3 * count > len(available):
        return None
    palette = [int(color) for color in random.sample(available, 3 * count)]
    sprite_colors = palette[:count]
    frame_colors = palette[count : 2 * count]
    fill_colors = palette[2 * count :]

    max_start = max_common_holes - count + 1
    start = random.randint(1, max_start)
    frame_holes = list(range(start, start + count))
    random.shuffle(frame_holes)

    sprite_holes = _random_derangement(frame_holes)
    if sprite_holes is None:
        return None

    sprite_patches: dict[int, np.ndarray] = {}
    sprite_by_holes: dict[int, np.ndarray] = {}
    for index, sprite_color in enumerate(sprite_colors):
        hole_count = int(sprite_holes[index])
        patch = _make_sprite_interior(sprite_size, hole_count, int(sprite_color))
        if patch is None:
            return None
        sprite_patches[int(sprite_color)] = patch
        sprite_by_holes[int(hole_count)] = patch

    frame_patches: list[np.ndarray] = []
    for index, frame_color in enumerate(frame_colors):
        patch = _make_frame(frame_size, int(frame_holes[index]), int(frame_color))
        if patch is None:
            return None
        frame_patches.append(patch)

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)

    legend = _make_legend(sprite_colors, frame_colors, fill_colors)
    legend_c = random.randint(0, canvas_w - legend_w)
    paste(canvas, legend, 0, legend_c, transparent=BG)

    grid_r0 = random.randint(LEGEND_RESERVED_ROWS, canvas_h - grid_h)
    grid_c0 = random.randint(0, canvas_w - grid_w)
    for index, frame_patch in enumerate(frame_patches):
        rr = int(grid_r0 + (index // cols_grid) * frame_size)
        cc = int(grid_c0 + (index % cols_grid) * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for sprite_color in sprite_colors:
        sprite_patch = sprite_patches[int(sprite_color)]
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
        frame_colors,
        sprite_colors,
        fill_colors,
        sprite_patches,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    d4_only = _build_d4_only_output(
        frame_holes,
        frame_colors,
        fill_colors,
        sprite_by_holes,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    if np.array_equal(out, d4_only):
        return None

    d16_only = _build_d16_only_output(
        frame_colors,
        sprite_colors,
        sprite_patches,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    if np.array_equal(out, d16_only):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
