from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 7)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(
    grid: np.ndarray,
    *,
    include_mask: np.ndarray | None = None,
    colors: set[int] | None = None,
    monochromatic: bool,
) -> list[dict]:
    h, w = grid.shape
    if include_mask is None:
        include_mask = np.ones((h, w), dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not include_mask[r, c]:
                continue
            color = int(grid[r, c])
            if colors is not None and color not in colors:
                continue
            if color == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not include_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) == BG:
                        continue
                    if colors is not None and int(grid[nr, nc]) not in colors:
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "coords": coords,
                    "color": color if monochromatic else None,
                    "bbox": _bbox(coords),
                    "area": len(cells),
                }
            )
    return comps


def _find_legend_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    fg = (grid != BG) & (grid != WALL)
    comps = _components(grid, include_mask=fg, monochromatic=False)
    legends = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if (
            comp["area"] == 2
            and r0 == r1
            and r0 <= 1
            and (c0 <= 2 or c1 >= w - 3)
        ):
            legends.append(comp)
    legends.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return legends


def _legend_mapping_and_mask(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    legends = _find_legend_components(grid)
    mapping: dict[int, int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for comp in legends:
        r0, c0, _r1, c1 = comp["bbox"]
        assert c1 == c0 + 1
        replacement = int(grid[r0, c0])
        source = int(grid[r0, c1])
        assert replacement != source
        assert source not in mapping
        mapping[source] = replacement
        legend_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True
    return mapping, legend_mask


def _recolor_without_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mapping, legend_mask = _legend_mapping_and_mask(g)
    core_mask = (g != BG) & (g != WALL) & (~legend_mask)
    base = g.copy()
    for source, replacement in mapping.items():
        out[core_mask & (base == source)] = replacement
    out[legend_mask] = BG
    return out


def _settle_key(comp: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return (-r1, c0, c1, r0)


def _gravity_down(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert np.all(g[h - 1, :] == WALL)
    movable_mask = (g != BG) & (g != WALL)
    comps = _components(g, include_mask=movable_mask, monochromatic=True)

    out = np.full_like(g, BG)
    out[h - 1, :] = WALL
    occupied = out != BG

    for comp in sorted(comps, key=_settle_key):
        coords = comp["coords"].copy()
        color = int(g[coords[0, 0], coords[0, 1]])
        while True:
            nxt = coords + np.array([1, 0], dtype=int)
            if np.any(nxt[:, 0] >= h) or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt
        out[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor objects from top-corner legend dominos, merge the new monochrome components, and stack them downward onto the grey floor.

    Input structure:
    - The background is black and the bottom row is a full grey wall.
    - The upper-left or upper-right corner contains one or two isolated horizontal two-pixel legend dominos.
    - Away from the legends are several small monochrome 4-connected foreground objects in blue, red, green, or orange.

    Rule:
    - Read each legend left-to-right as [replacement color, source color].
    - Recolor every non-legend foreground pixel of each source color into its replacement color, while leaving the grey wall unchanged.
    - Remove the legend dominos to black, recompute monochrome 4-connected objects after recoloring, then let those rigid objects fall downward.
    - Settle the objects nearest the grey wall first; each object stops when the next step would hit the wall, the grid boundary, or an already settled object.

    Key insight:
    - The legends do not just relabel pixels: they can fuse previously separate same-shape pieces into a larger rigid body, and gravity acts on that merged body rather than on the original smaller objects.
    """
    recolored = _recolor_without_gravity(np.asarray(grid, dtype=int))
    return _gravity_down(recolored)


def _translate(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _piece(cells: list[tuple[int, int]], input_color: int, output_color: int | None = None) -> dict:
    return {
        "cells": list(cells),
        "input_color": int(input_color),
        "output_color": int(input_color if output_color is None else output_color),
    }


def _group(pieces: list[dict]) -> dict:
    return {"pieces": pieces}


def _paint_groups(shape: tuple[int, int], groups: list[dict], *, use_input_colors: bool) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    h, w = shape
    grid[h - 1, :] = WALL
    for group in groups:
        for piece in group["pieces"]:
            color = piece["input_color"] if use_input_colors else piece["output_color"]
            for r, c in piece["cells"]:
                if not (0 <= r < h - 1 and 0 <= c < w):
                    raise ValueError("piece out of bounds")
                if grid[r, c] != BG:
                    raise ValueError("piece overlap")
                grid[r, c] = color
    return grid


def _group_cells(group: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for piece in group["pieces"]:
        cells.update(piece["cells"])
    return cells


def _shift_group_up(group: dict, amount: int) -> dict:
    return _group(
        [
            _piece(
                _translate(piece["cells"], -amount, 0),
                piece["input_color"],
                piece["output_color"],
            )
            for piece in group["pieces"]
        ]
    )


def _cells_adjacent(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _input_object_components(grid: np.ndarray) -> list[dict]:
    mapping, legend_mask = _legend_mapping_and_mask(grid)
    _ = mapping
    mask = (grid != BG) & (grid != WALL) & (~legend_mask)
    colors = set(int(x) for x in np.unique(grid[mask]))
    return _components(grid, include_mask=mask, colors=colors, monochromatic=True)


def _has_required_merge(grid: np.ndarray) -> bool:
    before = _input_object_components(grid)
    recolored = _recolor_without_gravity(grid)
    after_mask = (recolored != BG) & (recolored != WALL)
    after = _components(recolored, include_mask=after_mask, monochromatic=True)
    return len(after) < len(before)


def _gravity_without_recolor(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    _mapping, legend_mask = _legend_mapping_and_mask(g)
    g[legend_mask] = BG
    return _gravity_down(g)


def _legend_positions(h: int, w: int, corner: str, count: int) -> list[tuple[int, int]]:
    if corner == "left":
        all_pos = [(0, 0), (1, 2)]
    else:
        all_pos = [(0, w - 2), (1, w - 4)]
    return all_pos[:count]


def _random_ground_shape(width_hint: int) -> list[tuple[int, int]]:
    options = [
        [(0, 0), (0, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0)],
    ]
    choices = [shape for shape in options if max(c for _r, c in shape) < width_hint]
    return random.choice(choices)


def _random_stack_shape() -> list[tuple[int, int]]:
    return random.choice(
        [
            [(0, 0), (1, 0)],
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1)],
        ]
    )


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _place_group_cells(
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    anchor_r: int,
    anchor_c: int,
) -> list[tuple[int, int]]:
    h, w = shape
    placed = _translate(cells, anchor_r, anchor_c)
    for r, c in placed:
        if not (0 <= r < h - 1 and 0 <= c < w):
            raise ValueError("placement out of bounds")
        if (r, c) in occupied:
            raise ValueError("placement overlap")
    occupied.update(placed)
    return placed


def _build_output_groups(h: int, w: int, legends: list[tuple[int, int]]) -> list[dict]:
    replacement1, source1 = legends[0]
    use_second = len(legends) == 2
    replacement2 = source2 = None
    if use_second:
        replacement2, source2 = legends[1]

    filler_colors = [c for c in PALETTE if c not in {source1, source2}]
    support_color = random.choice([c for c in filler_colors if c != replacement1])
    left_color = random.choice([c for c in filler_colors if c != support_color])
    right_color = random.choice([c for c in filler_colors if c != left_color])
    top_color = random.choice([c for c in filler_colors if c != right_color])

    occupied: set[tuple[int, int]] = set()
    groups: list[dict] = []

    base_c = max(6, min(w - 5, w // 2))
    blocker_h = random.choice([2, 3])
    blocker_top = h - 1 - blocker_h
    blocker_cells = _place_group_cells(occupied, (h, w), [(i, 0) for i in range(blocker_h)], blocker_top, base_c)
    groups.append(_group([_piece(blocker_cells, support_color)]))

    merged_r = blocker_top - 2
    merged_vertical = _place_group_cells(occupied, (h, w), [(0, 0), (1, 0)], merged_r, base_c)
    merged_bridge = _place_group_cells(occupied, (h, w), [(0, 0), (0, 1)], merged_r + 1, base_c + 1)
    groups.append(
        _group(
            [
                _piece(merged_vertical, replacement1),
                _piece(merged_bridge, source1, replacement1),
            ]
        )
    )

    left_base_shape = _random_ground_shape(3)
    left_base_anchor_c = 1
    left_base_anchor_r = h - 2 - max(r for r, _c in left_base_shape)
    left_base = _place_group_cells(occupied, (h, w), left_base_shape, left_base_anchor_r, left_base_anchor_c)
    groups.append(_group([_piece(left_base, left_color)]))

    right_base_shape = _random_ground_shape(3)
    right_base_w = 1 + max(c for _r, c in right_base_shape)
    right_base_anchor_c = w - 2 - right_base_w
    right_base_anchor_r = h - 2 - max(r for r, _c in right_base_shape)
    right_base = _place_group_cells(occupied, (h, w), right_base_shape, right_base_anchor_r, right_base_anchor_c)
    groups.append(_group([_piece(right_base, right_color)]))

    support_left = random.choice([True, False])
    support_cells = left_base if support_left else right_base
    support_color_for_top = top_color if top_color != (left_color if support_left else right_color) else replacement1
    top_shape = _normalize(_random_stack_shape())
    top_h = 1 + max(r for r, _c in top_shape)
    top_w = 1 + max(c for _r, c in top_shape)
    support_cols = sorted({c for _r, c in support_cells})
    support_top_row = min(r for r, _c in support_cells)
    top_anchor_r = support_top_row - top_h
    candidate_cs = [support_cols[0], support_cols[-1] - top_w + 1]
    random.shuffle(candidate_cs)
    placed_top = None
    for anchor_c in candidate_cs:
        trial = _translate(top_shape, top_anchor_r, anchor_c)
        trial_set = set(trial)
        if any(cell in occupied for cell in trial_set):
            continue
        if not any((r + 1, c) in set(support_cells) for r, c in trial):
            continue
        for r, c in trial:
            if not (0 <= r < h - 1 and 0 <= c < w):
                break
        else:
            occupied.update(trial_set)
            placed_top = trial
            break
    if placed_top is None:
        raise ValueError("failed to place stacked object")
    groups.append(_group([_piece(placed_top, support_color_for_top)]))

    if use_second:
        second_c = 4 if base_c >= 8 else w - 6
        second_top = h - 3
        left_half = _place_group_cells(occupied, (h, w), [(0, 0), (1, 0)], second_top, second_c)
        right_half = _place_group_cells(occupied, (h, w), [(0, 0), (1, 0)], second_top, second_c + 1)
        groups.append(
            _group(
                [
                    _piece(left_half, replacement2),
                    _piece(right_half, source2, replacement2),
                ]
            )
        )

    return groups


def _groups_to_input(
    h: int,
    w: int,
    groups: list[dict],
    legends: list[tuple[int, int]],
    corner: str,
) -> np.ndarray | None:
    shifted: list[dict] = []
    attempts = 120
    legend_occ = set()
    for r, c in _legend_positions(h, w, corner, len(legends)):
        legend_occ.add((r, c))
        legend_occ.add((r, c + 1))

    group_cells_cache = [set(_group_cells(group)) for group in groups]
    tops = [min(r for r, _c in cells) for cells in group_cells_cache]
    order = sorted(range(len(groups)), key=lambda i: tops[i])

    for _ in range(attempts):
        shifted = [None] * len(groups)  # type: ignore[list-item]
        occupied: list[set[tuple[int, int]]] = []
        ok = True
        for idx in order:
            group = groups[idx]
            max_shift = min(r for r, _c in group_cells_cache[idx]) - 3
            if max_shift < 1:
                ok = False
                break
            offsets = list(range(1, max_shift + 1))
            random.shuffle(offsets)
            placed = None
            for amount in offsets:
                candidate = _shift_group_up(group, amount)
                candidate_cells = _group_cells(candidate)
                if candidate_cells & legend_occ:
                    continue
                if any(candidate_cells & other for other in occupied):
                    continue
                if any(
                    _cells_adjacent(candidate_cells, other)
                    for j, other in enumerate(occupied)
                    if order[j] != idx
                ):
                    continue
                placed = candidate
                occupied.append(candidate_cells)
                break
            if placed is None:
                ok = False
                break
            shifted[idx] = placed
        if not ok:
            continue

        grid = _paint_groups((h, w), shifted, use_input_colors=True)  # type: ignore[arg-type]
        for (replacement, source), (r, c) in zip(legends, _legend_positions(h, w, corner, len(legends))):
            grid[r, c] = replacement
            grid[r, c + 1] = source
        return grid
    return None


def _validate_instance(inp: np.ndarray) -> bool:
    out = main(inp)
    static = _recolor_without_gravity(inp)
    gravity_only = _gravity_without_recolor(inp)
    if np.array_equal(out, static):
        return False
    if np.array_equal(out, gravity_only):
        return False
    if not _has_required_merge(inp):
        return False

    comps = _input_object_components(inp)
    if not (6 <= len(comps) <= 8):
        return False
    if any(comp["area"] not in (2, 3) for comp in comps):
        return False

    legends = _find_legend_components(inp)
    if len(legends) not in (1, 2):
        return False

    recolored = _recolor_without_gravity(inp)
    recolored_comps = _components(
        recolored,
        include_mask=(recolored != BG) & (recolored != WALL),
        monochromatic=True,
    )
    if len(recolored_comps) >= len(comps):
        return False

    if not np.any(out[:-1, :] == BG):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 14)
    w = random.randint(14, 16)
    legend_count = random.choice([1, 2])
    corner = random.choice(["left", "right"])

    colors = list(PALETTE)
    random.shuffle(colors)
    if legend_count == 1:
        legends = [(colors[0], colors[1])]
    else:
        legends = [(colors[0], colors[1]), (colors[2], colors[3])]

    try:
        groups = _build_output_groups(h, w, legends)
        out_expected = _paint_groups((h, w), groups, use_input_colors=False)
    except ValueError:
        return None

    inp = _groups_to_input(h, w, groups, legends, corner)
    if inp is None:
        return None

    if not _validate_instance(inp):
        return None
    if not np.array_equal(main(inp), out_expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
