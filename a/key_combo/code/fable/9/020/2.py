from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 600


def _full_color_rows(grid: np.ndarray) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color != BG and np.all(grid[r, :] == color):
            rows.append(r)
    return rows


def _full_color_cols(grid: np.ndarray) -> list[int]:
    cols = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color != BG and np.all(grid[:, c] == color):
            cols.append(c)
    return cols


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    row_candidates = _full_color_rows(grid)
    col_candidates = _full_color_cols(grid)
    assert len(row_candidates) == 1, "expected exactly one solid separator row"
    assert len(col_candidates) == 1, "expected exactly one solid separator column"

    sep_r = row_candidates[0]
    sep_c = col_candidates[0]
    color = int(grid[sep_r, 0])
    assert int(grid[0, sep_c]) == color, "separator row and column must share one color"
    return sep_r, sep_c, color


def _quadrant_colors(blocks: Iterable[np.ndarray]) -> list[int]:
    colors = []
    for block in blocks:
        assert block.size > 0, "quadrants must be non-empty"
        present = sorted(int(v) for v in np.unique(block) if int(v) != BG)
        assert len(present) == 1, "each quadrant must be a monochrome mask on black"
        color = present[0]
        assert np.all((block == BG) | (block == color)), "quadrants must be monochrome"
        colors.append(color)
    assert len(set(colors)) == 4, "quadrant colors must be distinct"
    return colors


def _extract_quadrants(grid: np.ndarray) -> tuple[tuple[np.ndarray, ...], list[int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    sep_r, sep_c, sep_color = _find_separator_cross(g)
    assert sep_color == SEPARATOR, "expected a grey separator cross"
    assert 0 < sep_r < g.shape[0] - 1 and 0 < sep_c < g.shape[1] - 1, "separator must split four quadrants"

    tl = g[:sep_r, :sep_c]
    tr = g[:sep_r, sep_c + 1 :]
    bl = g[sep_r + 1 :, :sep_c]
    br = g[sep_r + 1 :, sep_c + 1 :]
    colors = _quadrant_colors((tl, tr, bl, br))
    return (tl, tr, bl, br), colors, sep_color


def _place_aligned(layer: np.ndarray, out_h: int, out_w: int, align: str) -> tuple[slice, slice]:
    h, w = layer.shape
    if align == "bottom_right":
        return slice(out_h - h, out_h), slice(out_w - w, out_w)
    if align == "top_left":
        return slice(0, h), slice(0, w)
    raise ValueError(f"unknown alignment: {align}")


def _overlay_exactly_one(
    layers: list[np.ndarray],
    colors: list[int],
    *,
    align: str = "bottom_right",
) -> tuple[np.ndarray, np.ndarray]:
    out_h = max(layer.shape[0] for layer in layers)
    out_w = max(layer.shape[1] for layer in layers)

    count = np.zeros((out_h, out_w), dtype=int)
    paint = np.zeros((out_h, out_w), dtype=int)
    for layer, color in zip(layers, colors):
        rr, cc = _place_aligned(layer, out_h, out_w, align)
        fg = layer != BG
        count[rr, cc] += fg.astype(int)
        paint[rr, cc][fg] = color

    out = np.zeros((out_h, out_w), dtype=int)
    out[count == 1] = paint[count == 1]
    return out, count


def _fold_layers(quadrants: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    tl, tr, bl, br = quadrants
    return [tl, np.fliplr(tr), np.flipud(bl), np.flipud(np.fliplr(br))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold four offset quadrants onto one stack and keep only the uniquely visible foreground pixels.

    Input structure:
    - One full grey row and one full grey column form an off-center separator cross.
    - The cross splits the grid into four non-empty quadrants on black background.
    - Each quadrant contains a monochrome mask, and the four mask colors are distinct.

    Rule:
    - Fold the top-right quadrant leftward across the vertical separator, the bottom-left
      quadrant upward across the horizontal separator, and the bottom-right quadrant across
      both separators; the top-left quadrant keeps its orientation.
    - Align the four folded layers by the corner nearest the separator intersection, so the
      output height is the larger of the top and bottom quadrant heights and the output width
      is the larger of the left and right quadrant widths.
    - At each aligned position, output the layer color only when exactly one layer has
      foreground there; if zero or multiple layers have foreground, output black.

    Key insight:
    - The separator cross defines both the fold axes and the local coordinate system, so the
      task is an offset four-layer fold with annihilation on overlap, not a centered mirror tile.
    """
    quadrants, colors, _sep_color = _extract_quadrants(grid)
    folded = _fold_layers(quadrants)
    out, _count = _overlay_exactly_one(folded, colors, align="bottom_right")
    return out


def _randint(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _pick_side_sizes(max_size: int) -> tuple[int, int]:
    assert max_size >= 5
    short = _randint(3, max_size - 1)
    if random.random() < 0.5:
        return max_size, short
    return short, max_size


def _shape_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(h) for c in range(w)]


def _cells_in_support(canvas_h: int, canvas_w: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0 = canvas_h - h
    c0 = canvas_w - w
    return [(r, c) for r in range(r0, canvas_h) for c in range(c0, canvas_w)]


def _local_from_canvas(
    coord: tuple[int, int],
    canvas_h: int,
    canvas_w: int,
    shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    r, c = coord
    return r - (canvas_h - h), c - (canvas_w - w)


def _mirror_coord(coord: tuple[int, int], shape: tuple[int, int], symmetry: str) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    if symmetry == "lr":
        return r, w - 1 - c
    if symmetry == "ud":
        return h - 1 - r, c
    if symmetry == "rot":
        return h - 1 - r, w - 1 - c
    raise ValueError(f"unknown symmetry {symmetry}")


def _add_asymmetry_constraint(
    shape: tuple[int, int],
    symmetry: str | None,
    forced_fg: set[tuple[int, int]],
    forced_bg: set[tuple[int, int]],
) -> bool:
    if symmetry is None:
        return True

    coords = _shape_cells(shape)
    random.shuffle(coords)
    for coord in coords:
        mate = _mirror_coord(coord, shape, symmetry)
        if mate == coord:
            continue
        a_fg = coord in forced_fg
        a_bg = coord in forced_bg
        b_fg = mate in forced_fg
        b_bg = mate in forced_bg
        if (a_fg and a_bg) or (b_fg and b_bg):
            continue
        if a_fg and b_fg:
            continue
        if a_bg and b_bg:
            continue
        if a_fg:
            forced_bg.add(mate)
            return True
        if b_fg:
            forced_bg.add(coord)
            return True
        if a_bg:
            forced_fg.add(mate)
            return True
        if b_bg:
            forced_fg.add(coord)
            return True
        if random.random() < 0.5:
            forced_fg.add(coord)
            forced_bg.add(mate)
        else:
            forced_fg.add(mate)
            forced_bg.add(coord)
        return True
    return False


def _make_mask(
    shape: tuple[int, int],
    forced_fg: set[tuple[int, int]],
    forced_bg: set[tuple[int, int]],
    symmetry: str | None,
) -> np.ndarray | None:
    area = shape[0] * shape[1]
    min_area = max(len(forced_fg), 3, int(np.ceil(area * 0.18)))
    max_area = max(min_area, min(area - 1, int(np.floor(area * 0.48))))
    candidates = [coord for coord in _shape_cells(shape) if coord not in forced_bg]
    if len(candidates) < min_area:
        return None

    for _ in range(120):
        target = _randint(min_area, max_area)
        mask = np.zeros(shape, dtype=bool)
        for coord in forced_fg:
            mask[coord] = True

        rect_attempts = _randint(2, 4)
        for _ in range(rect_attempts):
            rh = _randint(1, max(1, min(shape[0], max(2, shape[0] // 2 + 1))))
            rw = _randint(1, max(1, min(shape[1], max(2, shape[1] // 2 + 1))))
            r0 = _randint(0, shape[0] - rh)
            c0 = _randint(0, shape[1] - rw)
            rect = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
            if any(coord in forced_bg for coord in rect):
                continue
            for coord in rect:
                mask[coord] = True
            if int(mask.sum()) > max_area:
                break

        while int(mask.sum()) < target:
            fg_coords = list(map(tuple, np.argwhere(mask)))
            grow_from_neighbors = fg_coords and random.random() < 0.7
            if grow_from_neighbors:
                neighbor_options = set()
                for r, c in fg_coords:
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                            if (nr, nc) not in forced_bg and not mask[nr, nc]:
                                neighbor_options.add((nr, nc))
                if neighbor_options:
                    choice = random.choice(tuple(neighbor_options))
                    mask[choice] = True
                    continue
            free = [coord for coord in candidates if not mask[coord]]
            if not free:
                break
            mask[random.choice(free)] = True

        if int(mask.sum()) < min_area or int(mask.sum()) > max_area:
            continue
        if any(mask[coord] for coord in forced_bg):
            continue
        if any(not mask[coord] for coord in forced_fg):
            continue
        if symmetry == "lr" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if symmetry == "ud" and np.array_equal(mask, np.flipud(mask)):
            continue
        if symmetry == "rot" and np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask
    return None


def _wrong_fold_variants(quadrants: tuple[np.ndarray, ...], colors: list[int]) -> list[np.ndarray]:
    tl, tr, bl, br = quadrants
    return [
        _overlay_exactly_one([tl, tr, np.flipud(bl), np.flipud(np.fliplr(br))], colors)[0],
        _overlay_exactly_one([tl, np.fliplr(tr), bl, np.flipud(np.fliplr(br))], colors)[0],
        _overlay_exactly_one([tl, np.fliplr(tr), np.flipud(bl), br], colors)[0],
        _overlay_exactly_one([tl, np.fliplr(tr), np.flipud(bl), np.flipud(br)], colors)[0],
        _overlay_exactly_one(_fold_layers(quadrants), colors, align="top_left")[0],
    ]


def _compose_input(
    quadrants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    top_h: int,
    bottom_h: int,
    left_w: int,
    right_w: int,
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((top_h + bottom_h + 1, left_w + right_w + 1), dtype=int)
    grid[top_h, :] = SEPARATOR
    grid[:, left_w] = SEPARATOR

    blocks = [
        quadrants[0],
        quadrants[1],
        quadrants[2],
        quadrants[3],
    ]
    slices = [
        (slice(0, top_h), slice(0, left_w)),
        (slice(0, top_h), slice(left_w + 1, left_w + 1 + right_w)),
        (slice(top_h + 1, top_h + 1 + bottom_h), slice(0, left_w)),
        (slice(top_h + 1, top_h + 1 + bottom_h), slice(left_w + 1, left_w + 1 + right_w)),
    ]
    for block, color, (rr, cc) in zip(blocks, colors, slices):
        grid[rr, cc][block] = color
    return grid


def _choose_reserved_cells(
    canvas_h: int,
    canvas_w: int,
    shapes: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, ...]]], set[tuple[int, int]], np.ndarray]:
    supports = [_cells_in_support(canvas_h, canvas_w, shape) for shape in shapes]
    reach = np.zeros((canvas_h, canvas_w), dtype=int)
    for support in supports:
        for coord in support:
            reach[coord] += 1

    used: set[tuple[int, int]] = set()
    unique_cells: list[tuple[int, int]] = []
    for support in supports:
        candidates = [coord for coord in support if coord not in used]
        reach_levels = sorted({int(reach[coord]) for coord in candidates})
        picked = None
        for level in reach_levels:
            same_level = [coord for coord in candidates if int(reach[coord]) == level]
            random.shuffle(same_level)
            if same_level:
                picked = same_level[0]
                break
        assert picked is not None
        unique_cells.append(picked)
        used.add(picked)

    collision_specs: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    all_cells = [(r, c) for r in range(canvas_h) for c in range(canvas_w)]
    full_overlap = [coord for coord in all_cells if int(reach[coord]) == 4 and coord not in used]
    partial_overlap = [coord for coord in all_cells if 2 <= int(reach[coord]) <= 3 and coord not in used]
    random.shuffle(full_overlap)
    random.shuffle(partial_overlap)

    collision_candidates = []
    if full_overlap:
        collision_candidates.append(full_overlap[0])
        used.add(full_overlap[0])
    if partial_overlap:
        collision_candidates.append(partial_overlap[0])
        used.add(partial_overlap[0])

    extra = [coord for coord in all_cells if int(reach[coord]) >= 2 and coord not in used]
    random.shuffle(extra)
    for coord in extra[: random.randint(0, 2)]:
        collision_candidates.append(coord)
        used.add(coord)

    for coord in collision_candidates:
        available = [i for i, support in enumerate(supports) if coord in support]
        subset_size = _randint(2, len(available))
        chosen = tuple(sorted(random.sample(available, subset_size)))
        collision_specs.append((coord, chosen))

    empty_cells = {coord for coord in all_cells if coord not in used}
    empty_cells = set(random.sample(tuple(empty_cells), _randint(2, min(4, len(empty_cells)))))
    return unique_cells, collision_specs, empty_cells, reach


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    out_h = _randint(5, 9)
    out_w = _randint(5, 9)
    top_h, bottom_h = _pick_side_sizes(out_h)
    left_w, right_w = _pick_side_sizes(out_w)

    shapes = [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, left_w),
        (bottom_h, right_w),
    ]
    colors = random.sample(PALETTE, 4)
    unique_cells, collision_specs, empty_cells, reach = _choose_reserved_cells(out_h, out_w, shapes)

    support_sets = [set(_cells_in_support(out_h, out_w, shape)) for shape in shapes]
    local_masks: list[np.ndarray] = []
    symmetries = [None, "lr", "ud", "rot"]
    for i, (shape, symmetry) in enumerate(zip(shapes, symmetries)):
        forced_fg: set[tuple[int, int]] = set()
        forced_bg: set[tuple[int, int]] = set()

        for j, coord in enumerate(unique_cells):
            if coord in support_sets[i]:
                local = _local_from_canvas(coord, out_h, out_w, shape)
                if i == j:
                    forced_fg.add(local)
                else:
                    forced_bg.add(local)

        for coord, chosen_layers in collision_specs:
            if coord not in support_sets[i]:
                continue
            local = _local_from_canvas(coord, out_h, out_w, shape)
            if i in chosen_layers:
                forced_fg.add(local)
            else:
                forced_bg.add(local)

        for coord in empty_cells:
            if coord in support_sets[i]:
                forced_bg.add(_local_from_canvas(coord, out_h, out_w, shape))

        if forced_fg & forced_bg:
            return None
        if not _add_asymmetry_constraint(shape, symmetry, forced_fg, forced_bg):
            return None
        mask = _make_mask(shape, forced_fg, forced_bg, symmetry)
        if mask is None:
            return None
        local_masks.append(mask)

    folded_layers = [mask.astype(int) for mask in local_masks]
    canvas_masks = []
    for mask in local_masks:
        canvas = np.zeros((out_h, out_w), dtype=bool)
        rr, cc = _place_aligned(mask.astype(int), out_h, out_w, "bottom_right")
        canvas[rr, cc] = mask
        canvas_masks.append(canvas)
    correct_out, fg_count = _overlay_exactly_one(folded_layers, colors, align="bottom_right")

    if int((fg_count == 0).sum()) < 2:
        return None
    if int((fg_count >= 2).sum()) < 2:
        return None
    if int((fg_count >= 2).sum()) > (out_h * out_w) // 2:
        return None
    if not np.any((fg_count == 1) & (reach < 4)):
        return None
    for color, canvas_mask in zip(colors, canvas_masks):
        if not np.any((fg_count == 1) & canvas_mask):
            return None
        if color not in set(map(int, np.unique(correct_out))):
            return None

    raw_quadrants = (
        local_masks[0],
        np.fliplr(local_masks[1]),
        np.flipud(local_masks[2]),
        np.flipud(np.fliplr(local_masks[3])),
    )
    grid = _compose_input(raw_quadrants, top_h, bottom_h, left_w, right_w, colors)

    if not np.array_equal(main(grid), correct_out):
        return None

    for wrong in _wrong_fold_variants(raw_quadrants, colors):
        if np.array_equal(wrong, correct_out):
            return None

    if grid.shape[0] % 2 == 1 and top_h == bottom_h:
        return None
    if grid.shape[1] % 2 == 1 and left_w == right_w:
        return None

    return grid, correct_out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, out = built
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(main(grid), out):
            return grid
    raise RuntimeError("failed to generate a valid input for fold-annihilation puzzle")
