from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MASK_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400

TILE_LIBRARY = (
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    frozenset({(0, 1), (0, 2), (1, 0), (1, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 1), (1, 1), (2, 0), (2, 1)}),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_cells(cells: list[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _bbox_dims(cells: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _bbox_area(cells: frozenset[tuple[int, int]]) -> int:
    h, w = _bbox_dims(cells)
    return h * w


def _translate(cells: frozenset[tuple[int, int]], r0: int, c0: int) -> frozenset[tuple[int, int]]:
    return frozenset((r0 + r, c0 + c) for r, c in cells)


def _touches4(cells: frozenset[tuple[int, int]], occupied: frozenset[tuple[int, int]]) -> bool:
    if not occupied:
        return True
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _touch_count(cells: frozenset[tuple[int, int]], occupied: frozenset[tuple[int, int]]) -> int:
    total = 0
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                total += 1
    return total


def _interlocks(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> bool:
    ar0 = min(r for r, _ in a)
    ac0 = min(c for _, c in a)
    ar1 = max(r for r, _ in a)
    ac1 = max(c for _, c in a)
    br0 = min(r for r, _ in b)
    bc0 = min(c for _, c in b)
    br1 = max(r for r, _ in b)
    bc1 = max(c for _, c in b)
    if ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0:
        return False
    for r, c in a:
        if br0 <= r <= br1 and bc0 <= c <= bc1 and (r, c) not in b:
            return True
    for r, c in b:
        if ar0 <= r <= ar1 and ac0 <= c <= ac1 and (r, c) not in a:
            return True
    return False


def _legend_size(grid: np.ndarray, border_color: int) -> tuple[int, int]:
    h_total, w_total = grid.shape
    width = next((c for c in range(1, w_total) if int(grid[0, c]) != border_color), w_total)
    height = next((r for r in range(1, h_total) if int(grid[r, 0]) != border_color), h_total)
    assert height >= 4 and width >= 4, "legend is too small"
    return height, width


def _find_legend(grid: np.ndarray) -> tuple[int, int, int]:
    border_color = int(grid[0, 0])
    assert border_color != BG, "top-left legend border missing"
    height, width = _legend_size(grid, border_color)
    assert np.all(grid[0, :width] == border_color)
    assert np.all(grid[height - 1, :width] == border_color)
    assert np.all(grid[:height, 0] == border_color)
    assert np.all(grid[:height, width - 1] == border_color)
    assert not np.any(grid[1 : height - 1, 1 : width - 1] == border_color)
    return height, width, border_color


def _extract_legend_tiles(grid: np.ndarray, legend_h: int, legend_w: int, border_color: int) -> list[dict]:
    interior = grid[1 : legend_h - 1, 1 : legend_w - 1]
    mask = (interior != BG) & (interior != border_color)
    comps = _components_mask(mask)
    assert len(comps) >= 2, "legend must contain multiple tiles"
    tiles: list[dict] = []
    shapes_seen: set[frozenset[tuple[int, int]]] = set()
    for comp in comps:
        colors = {int(interior[r, c]) for r, c in comp}
        assert len(colors) == 1, "legend tiles must be monochrome"
        cells = _normalize_cells(comp)
        assert cells not in shapes_seen, "legend shapes must be distinct"
        shapes_seen.add(cells)
        h, w = _bbox_dims(cells)
        tiles.append({"cells": cells, "color": colors.pop(), "h": h, "w": w})
    return tiles


def _build_candidates(
    mask_cells: frozenset[tuple[int, int]],
    shape: tuple[int, int],
    tiles: list[dict],
) -> tuple[list[tuple[int, frozenset[tuple[int, int]]]], dict[tuple[int, int], list[int]]]:
    h, w = shape
    min_r = min(r for r, _ in mask_cells)
    max_r = max(r for r, _ in mask_cells)
    min_c = min(c for _, c in mask_cells)
    max_c = max(c for _, c in mask_cells)
    candidates: list[tuple[int, frozenset[tuple[int, int]]]] = []
    for tile_id, tile in enumerate(tiles):
        r_lo = max(0, min_r - tile["h"] + 1)
        r_hi = min(h - tile["h"], max_r)
        c_lo = max(0, min_c - tile["w"] + 1)
        c_hi = min(w - tile["w"], max_c)
        for r0 in range(r_lo, r_hi + 1):
            for c0 in range(c_lo, c_hi + 1):
                cells = _translate(tile["cells"], r0, c0)
                if cells <= mask_cells:
                    candidates.append((tile_id, cells))
    by_cell = {cell: [] for cell in mask_cells}
    for idx, (_, cells) in enumerate(candidates):
        for cell in cells:
            by_cell[cell].append(idx)
    return candidates, by_cell


def _solve_exact_cover(
    mask_cells: frozenset[tuple[int, int]],
    shape: tuple[int, int],
    tiles: list[dict],
    *,
    stop_after: int = 2,
) -> list[list[tuple[int, frozenset[tuple[int, int]]]]]:
    candidates, by_cell = _build_candidates(mask_cells, shape, tiles)
    if not candidates:
        return []
    if any(not by_cell[cell] for cell in mask_cells):
        return []

    solutions: list[list[int]] = []

    def search(remaining: frozenset[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) >= stop_after:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return
        cell = min(
            remaining,
            key=lambda pos: sum(1 for idx in by_cell[pos] if candidates[idx][1] <= remaining),
        )
        options = [idx for idx in by_cell[cell] if candidates[idx][1] <= remaining]
        if not options:
            return
        options.sort(key=lambda idx: (-len(candidates[idx][1]), candidates[idx][0]))
        for idx in options:
            search(remaining - candidates[idx][1], chosen + [idx])

    search(mask_cells, [])
    decoded: list[list[tuple[int, frozenset[tuple[int, int]]]]] = []
    for solution in solutions:
        decoded.append([candidates[idx] for idx in solution])
    return decoded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey mask by uniquely exact-covering it with translated legend tiles.

    Input structure:
    - The background is black.
    - The top-left corner contains a bordered legend box whose interior holds several separated monochrome tile shapes.
    - Elsewhere, one grey foreground mask marks the area that must be covered.

    Rule:
    - Read the inventory of tile shapes and colors from the legend; only translations are allowed, with unlimited reuse and optional unused legend tiles.
    - Cover every grey cell exactly once using those legend shapes, with the guarantee that the exact cover is unique.
    - Recolor each placed tile with the color of its matching legend shape, while leaving the legend and its border unchanged.

    Key insight:
    - The legend defines the available tile geometries and colors, but the mask must be partitioned globally: interlocking placements and unused decoys mean local cut points are not reliable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BG, "background must be black"

    legend_h, legend_w, border_color = _find_legend(g)
    tiles = _extract_legend_tiles(g, legend_h, legend_w, border_color)

    mask = g == MASK_COLOR
    mask[:legend_h, :legend_w] = False
    mask_cells = frozenset((int(r), int(c)) for r, c in np.argwhere(mask))
    assert mask_cells, "grey mask missing"

    solutions = _solve_exact_cover(mask_cells, g.shape, tiles, stop_after=2)
    assert len(solutions) == 1, "legend tiles must give a unique exact cover"

    out = g.copy()
    for tile_id, cells in solutions[0]:
        color = int(tiles[tile_id]["color"])
        for r, c in cells:
            out[r, c] = color
    return out


def _make_tile_def(cells: frozenset[tuple[int, int]], color: int) -> dict:
    h, w = _bbox_dims(cells)
    return {"cells": cells, "color": color, "h": h, "w": w}


def _render_legend_patch(tiles: list[dict], border_color: int) -> np.ndarray:
    order = tiles.copy()
    random.shuffle(order)
    max_h = max(tile["h"] for tile in order)
    gaps = [random.randint(1, 2) for _ in range(max(0, len(order) - 1))]
    left_inner = 1
    right_inner = 1
    interior_h = max_h + 2
    interior_w = left_inner + sum(tile["w"] for tile in order) + sum(gaps) + right_inner
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    col = 1 + left_inner
    for idx, tile in enumerate(order):
        row = 1 + random.randint(0, interior_h - tile["h"])
        for dr, dc in tile["cells"]:
            patch[row + dr, col + dc] = tile["color"]
        if idx < len(gaps):
            col += tile["w"] + gaps[idx]
    return patch


def _choose_legend_tiles() -> tuple[list[dict], list[dict], bool, int]:
    border_color = random.choice([6, 9])
    available_colors = [color for color in (1, 2, 3, 4, 6, 7, 8, 9) if color != border_color]
    include_decoy = random.random() < 0.7
    used_type_count = random.randint(2, 4)
    legend_count = used_type_count + (1 if include_decoy else 0)
    shapes = random.sample(TILE_LIBRARY, legend_count)
    colors = random.sample(available_colors, legend_count)
    used_tiles = [_make_tile_def(shape, color) for shape, color in zip(shapes[:used_type_count], colors[:used_type_count])]
    all_tiles = used_tiles + [
        _make_tile_def(shape, color) for shape, color in zip(shapes[used_type_count:], colors[used_type_count:])
    ]
    random.shuffle(all_tiles)
    return used_tiles, all_tiles, include_decoy, border_color


def _usage_sequence(tile_count: int) -> list[int]:
    total_tiles = random.choices([4, 5, 6], weights=[4, 4, 2])[0]
    usage = list(range(tile_count))
    while len(usage) < total_tiles:
        usage.append(random.randrange(tile_count))
    if len(set(usage)) == len(usage):
        usage[-1] = random.randrange(tile_count)
    random.shuffle(usage)
    return usage


def _place_mask_tiles(tiles: list[dict], usage: list[int]) -> tuple[int, int, list[tuple[int, frozenset[tuple[int, int]]]]] | None:
    total_area = sum(len(tiles[idx]["cells"]) for idx in usage)
    max_h = max(tiles[idx]["h"] for idx in usage)
    max_w = max(tiles[idx]["w"] for idx in usage)
    min_side = max(max_h + 1, max_w + 1, int(np.ceil(np.sqrt(total_area))) + 1)
    mask_h = random.randint(min_side, min(11, min_side + 3))
    mask_w = random.randint(min_side, min(11, min_side + 4))
    if mask_h * mask_w < total_area + 3:
        return None

    def search(
        i: int,
        occupied: frozenset[tuple[int, int]],
        placed: list[tuple[int, frozenset[tuple[int, int]]]],
        has_interlock: bool,
    ) -> list[tuple[int, frozenset[tuple[int, int]]]] | None:
        if i == len(usage):
            return placed if has_interlock else None
        tile_id = usage[i]
        tile = tiles[tile_id]
        anchors = []
        for r0 in range(mask_h - tile["h"] + 1):
            for c0 in range(mask_w - tile["w"] + 1):
                cells = _translate(tile["cells"], r0, c0)
                if cells & occupied:
                    continue
                if occupied and not _touches4(cells, occupied):
                    continue
                next_interlock = has_interlock or any(_interlocks(cells, prev_cells) for _, prev_cells in placed)
                if not next_interlock and i == len(usage) - 1:
                    continue
                union = occupied | cells
                anchors.append(
                    (
                        0 if next_interlock else 1,
                        _bbox_area(union),
                        -_touch_count(cells, occupied),
                        random.random(),
                        cells,
                        next_interlock,
                    )
                )
        anchors.sort()
        for _, _, _, _, cells, next_interlock in anchors[:24]:
            result = search(i + 1, occupied | cells, placed + [(tile_id, cells)], next_interlock)
            if result is not None:
                return result
        return None

    placements = search(0, frozenset(), [], False)
    if placements is None:
        return None
    return mask_h, mask_w, placements


def _build_full_grid(legend_patch: np.ndarray, placements: list[tuple[int, frozenset[tuple[int, int]]]], tiles: list[dict], mask_h: int, mask_w: int) -> tuple[np.ndarray, np.ndarray]:
    legend_h, legend_w = legend_patch.shape
    row_gap = random.randint(1, 3)
    col_offset = random.randint(1, 6)
    bottom_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    mask_r0 = legend_h + row_gap
    mask_c0 = col_offset
    height = mask_r0 + mask_h + bottom_pad
    width = max(legend_w + 1, mask_c0 + mask_w + right_pad)
    if height > 30 or width > 30:
        raise ValueError("grid exceeds ARC size limit")

    inp = np.full((height, width), BG, dtype=int)
    out = inp.copy()
    inp[:legend_h, :legend_w] = legend_patch
    out[:legend_h, :legend_w] = legend_patch
    for tile_id, cells in placements:
        color = tiles[tile_id]["color"]
        for r, c in cells:
            inp[mask_r0 + r, mask_c0 + c] = MASK_COLOR
            out[mask_r0 + r, mask_c0 + c] = color
    return inp, out


def _semantic_checks(inp: np.ndarray, out: np.ndarray, include_decoy: bool) -> bool:
    try:
        legend_h, legend_w, border_color = _find_legend(inp)
        tiles = _extract_legend_tiles(inp, legend_h, legend_w, border_color)
        mask = inp == MASK_COLOR
        mask[:legend_h, :legend_w] = False
        comps = _components_mask(mask)
        if len(comps) != 1:
            return False
        mask_cells = frozenset((int(r), int(c)) for r, c in np.argwhere(mask))
        solutions = _solve_exact_cover(mask_cells, inp.shape, tiles, stop_after=3)
        if len(solutions) != 1:
            return False
        used_tile_ids = {tile_id for tile_id, _ in solutions[0]}
        if len(used_tile_ids) < 2:
            return False
        if len({int(out[r, c]) for r, c in mask_cells}) < 2:
            return False
        if include_decoy and len(used_tile_ids) >= len(tiles):
            return False
        interlocked = False
        solution_cells = [cells for _, cells in solutions[0]]
        for i in range(len(solution_cells)):
            for j in range(i + 1, len(solution_cells)):
                if _interlocks(solution_cells[i], solution_cells[j]):
                    interlocked = True
                    break
            if interlocked:
                break
        if not interlocked:
            return False
        return np.array_equal(main(inp), out)
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        used_tiles, all_tiles, include_decoy, border_color = _choose_legend_tiles()
        usage = _usage_sequence(len(used_tiles))
        placed = _place_mask_tiles(used_tiles, usage)
        if placed is None:
            continue
        mask_h, mask_w, placements = placed
        legend_patch = _render_legend_patch(all_tiles, border_color)
        try:
            inp, out = _build_full_grid(legend_patch, placements, used_tiles, mask_h, mask_w)
        except ValueError:
            continue
        if _semantic_checks(inp, out, include_decoy):
            return inp
    raise RuntimeError("failed to generate a valid legend exact-cover task")
