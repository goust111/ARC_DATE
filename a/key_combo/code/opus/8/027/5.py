from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, neighbors4


BG = Color.YELLOW
SEP = Color.GREY
FG_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.PINK)
HALF_SHAPE = (5, 7)
GRID_SHAPE = (HALF_SHAPE[0] * 2 + 1, HALF_SHAPE[1])
GEN_ATTEMPTS = 500


BASE_SHAPES: dict[int, list[tuple[tuple[int, int], ...]]] = {
    2: [((0, 0), (0, 1))],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ],
    5: [
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ],
    6: [
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    ],
}


def _normalize(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _transform_shape(cells: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in cells}
    for _ in range(int(rot) % 4):
        pts = {(int(c), int(-r)) for r, c in pts}
    if flip:
        pts = {(int(r), int(-c)) for r, c in pts}
    return _normalize(pts)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in neighbors4(*cell):
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return seen == cells


def _shape_variants(area: int) -> list[dict]:
    variants: dict[tuple[tuple[int, int], ...], dict] = {}
    for base in BASE_SHAPES[area]:
        for rot in range(4):
            for flip in (False, True):
                cells = _transform_shape(base, rot, flip)
                if cells in variants:
                    continue
                cell_set = {(int(r), int(c)) for r, c in cells}
                safe = []
                for cell in cells:
                    remainder = set(cell_set)
                    remainder.remove(cell)
                    if _is_connected(remainder):
                        safe.append(cell)
                h = max(r for r, _ in cells) + 1
                w = max(c for _, c in cells) + 1
                variants[cells] = {
                    "cells": cells,
                    "cell_set": frozenset(cell_set),
                    "safe_local": frozenset(safe),
                    "h": int(h),
                    "w": int(w),
                    "area": int(area),
                }
    return list(variants.values())


SHAPE_LIBRARY = {area: _shape_variants(area) for area in range(2, 7)}


def _translate(shape: dict, r0: int, c0: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = {(int(r0 + r), int(c0 + c)) for r, c in shape["cells"]}
    safe = {(int(r0 + r), int(c0 + c)) for r, c in shape["safe_local"]}
    return cells, safe


def _fits(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int] = HALF_SHAPE) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _enumerate_candidates(
    area: int,
    occupied: set[tuple[int, int]],
    *,
    allowed_overlap: set[tuple[int, int]] | None = None,
    forbidden_overlap: set[tuple[int, int]] | None = None,
    required_overlap: int = 0,
    overlap_safe_in_candidate: bool = False,
) -> list[dict]:
    allowed = set() if allowed_overlap is None else set(allowed_overlap)
    forbidden = set() if forbidden_overlap is None else set(forbidden_overlap)
    candidates = []
    for shape in SHAPE_LIBRARY[area]:
        for r0 in range(HALF_SHAPE[0] - shape["h"] + 1):
            for c0 in range(HALF_SHAPE[1] - shape["w"] + 1):
                cells, safe = _translate(shape, r0, c0)
                if not _fits(cells, occupied):
                    continue
                overlap = cells & allowed
                if len(overlap) != int(required_overlap):
                    continue
                if cells & forbidden:
                    continue
                if overlap_safe_in_candidate and not overlap <= safe:
                    continue
                candidates.append(
                    {
                        "area": int(area),
                        "cells": cells,
                        "safe_abs": safe,
                    }
                )
    random.shuffle(candidates)
    return candidates


def _render_half(components: list[dict], shape: tuple[int, int] = HALF_SHAPE) -> np.ndarray:
    half = np.full(shape, BG, dtype=int)
    for comp in components:
        for r, c in comp["cells"]:
            half[int(r), int(c)] = int(comp["color"])
    return half


def _split_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    assert h % 2 == 1, "expected two equal halves separated by one row"
    mid = h // 2
    assert np.all(g[mid, :] == SEP), "expected a grey separator row"
    top = g[:mid, :]
    bottom = g[mid + 1 :, :]
    assert top.shape == bottom.shape, "top and bottom halves must match"
    return top, bottom


def _label_half(half: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], tuple[int, int, int]]]:
    components = find_components(
        half,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=FG_COLORS,
        return_format="dict",
    )
    cell_map: dict[tuple[int, int], tuple[int, int, int]] = {}
    for idx, comp in enumerate(components):
        area = int(comp["area"])
        color = int(comp["color"])
        for r, c in comp["coords"]:
            cell_map[(int(r), int(c))] = (idx, area, color)
    return components, cell_map


def _merge_halves(
    top: np.ndarray,
    bottom: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[str, int]], dict[str, int], list[dict], list[dict]]:
    top_components, top_map = _label_half(top)
    bottom_components, bottom_map = _label_half(bottom)

    merged = np.full(top.shape, BG, dtype=int)
    winners: dict[tuple[int, int], tuple[str, int]] = {}
    conflict_wins = {"top": 0, "bottom": 0}

    for r in range(top.shape[0]):
        for c in range(top.shape[1]):
            tv = int(top[r, c])
            bv = int(bottom[r, c])
            if tv == BG and bv == BG:
                continue
            if tv != BG and bv == BG:
                merged[r, c] = tv
                winners[(r, c)] = ("top", int(top_map[(r, c)][0]))
                continue
            if tv == BG and bv != BG:
                merged[r, c] = bv
                winners[(r, c)] = ("bottom", int(bottom_map[(r, c)][0]))
                continue

            top_id, top_area, _ = top_map[(r, c)]
            bottom_id, bottom_area, _ = bottom_map[(r, c)]
            assert top_area != bottom_area, "conflicting source components must not tie in area"
            if top_area > bottom_area:
                merged[r, c] = tv
                winners[(r, c)] = ("top", int(top_id))
                conflict_wins["top"] += 1
            else:
                merged[r, c] = bv
                winners[(r, c)] = ("bottom", int(bottom_id))
                conflict_wins["bottom"] += 1

    return merged, winners, conflict_wins, top_components, bottom_components


def _bucket_color(area: int) -> int:
    if area <= 2:
        return int(Color.GREEN)
    if area <= 4:
        return int(Color.RED)
    return int(Color.BLUE)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge two stacked masks by parent-component area, then recolor the merged components by their new areas.

    Input structure:
    - The grid is split into equal top and bottom halves by one grey separator row.
    - Each half has yellow background and several 4-connected foreground components drawn in blue, red, green, or pink.
    - Foreground components are compared at matching coordinates between the two halves.

    Rule:
    - Remove the separator and compare the two halves cell by cell.
    - Background/background stays yellow; a lone foreground cell survives unchanged.
    - When both cells are foreground, find each cell's 4-connected source component in its own half and keep the color
      from the larger-area source component.
    - In the merged half-grid, find same-color 4-connected components and recolor them by new area:
      size 1-2 -> green, size 3-4 -> red, size 5+ -> blue.

    Key insight:
    - Conflict resolution depends on global source-component area, but the final color depends on a second area
      measurement taken after merging changes which pixels still belong to each output component.
    """
    top, bottom = _split_grid(grid)
    merged, _winners, _conflict_wins, _top_components, _bottom_components = _merge_halves(top, bottom)

    out = merged.copy()
    components = find_components(
        merged,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=FG_COLORS,
        return_format="dict",
    )
    for comp in components:
        color = _bucket_color(int(comp["area"]))
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _count_choices() -> tuple[int, int]:
    options = [
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
        (4, 4),
    ]
    weights = [5, 5, 5, 5, 2, 2, 1, 1, 1]
    return random.choices(options, weights=weights, k=1)[0]


def _choose_area_plan(top_count: int, bottom_count: int) -> tuple[list[int], list[int]]:
    top_small = random.choice([2, 3])
    bottom_medium = random.choice([4, 5])
    top_large = random.choice([area for area in [5, 6] if area > bottom_medium])
    bottom_large = random.choice([area for area in [4, 5, 6] if area != bottom_medium and area > top_small])

    top_areas = [top_large, top_small]
    bottom_areas = [bottom_medium, bottom_large]

    top_remaining = [area for area in range(2, 7) if area not in top_areas]
    bottom_remaining = [area for area in range(2, 7) if area not in bottom_areas]
    random.shuffle(top_remaining)
    random.shuffle(bottom_remaining)

    top_areas.extend(top_remaining[: max(0, top_count - 2)])
    bottom_areas.extend(bottom_remaining[: max(0, bottom_count - 2)])
    return top_areas, bottom_areas


def _assign_colors(count: int, palette: tuple[int, int]) -> list[int]:
    colors = [palette[0], palette[1]]
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return [int(c) for c in colors[:count]]


def _source_area_dict(components: list[dict], prefix: str) -> dict[tuple[str, int], int]:
    return {(prefix, idx): int(comp["area"]) for idx, comp in enumerate(components)}


def _semantic_valid(grid: np.ndarray) -> bool:
    try:
        top, bottom = _split_grid(grid)
    except AssertionError:
        return False

    try:
        merged, winners, conflict_wins, top_components, bottom_components = _merge_halves(top, bottom)
    except AssertionError:
        return False

    if merged.shape != HALF_SHAPE:
        return False
    if np.array_equal(merged, top) or np.array_equal(merged, bottom):
        return False
    if not np.all(grid[HALF_SHAPE[0], :] == SEP):
        return False
    if not set(map(int, np.unique(top))).issubset(set(FG_COLORS) | {BG}):
        return False
    if not set(map(int, np.unique(bottom))).issubset(set(FG_COLORS) | {BG}):
        return False

    top_areas = [int(comp["area"]) for comp in top_components]
    bottom_areas = [int(comp["area"]) for comp in bottom_components]
    if not (2 <= len(top_areas) <= 4 and 2 <= len(bottom_areas) <= 4):
        return False
    if len(set(top_areas)) != len(top_areas) or len(set(bottom_areas)) != len(bottom_areas):
        return False
    if any(area < 2 or area > 6 for area in top_areas + bottom_areas):
        return False

    conflict_total = int(conflict_wins["top"] + conflict_wins["bottom"])
    if conflict_total < 2 or conflict_wins["top"] == 0 or conflict_wins["bottom"] == 0:
        return False

    top_only = int(np.sum((top != BG) & (bottom == BG)))
    bottom_only = int(np.sum((top == BG) & (bottom != BG)))
    if top_only == 0 or bottom_only == 0:
        return False

    output_components = find_components(
        merged,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=FG_COLORS,
        return_format="dict",
    )
    if len(output_components) < 3:
        return False

    buckets = {_bucket_color(int(comp["area"])) for comp in output_components}
    if buckets != {Color.GREEN, Color.RED, Color.BLUE}:
        return False

    source_areas = {}
    source_areas.update(_source_area_dict(top_components, "top"))
    source_areas.update(_source_area_dict(bottom_components, "bottom"))
    changed = False
    for comp in output_components:
        contributors = {
            winners[(int(r), int(c))]
            for r, c in comp["coords"]
        }
        if len(contributors) != 1:
            changed = True
            break
        only = next(iter(contributors))
        if int(comp["area"]) != int(source_areas[only]):
            changed = True
            break
    if not changed:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    if solved.shape != HALF_SHAPE:
        return False
    if max(grid.shape) > 30 or max(solved.shape) > 30:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    top_count, bottom_count = _count_choices()
    top_areas, bottom_areas = _choose_area_plan(top_count, bottom_count)

    palette = list(FG_COLORS)
    random.shuffle(palette)
    top_palette = (int(palette[0]), int(palette[1]))
    bottom_palette = (int(palette[2]), int(palette[3]))

    for _ in range(120):
        top_occ: set[tuple[int, int]] = set()
        t_large_cands = _enumerate_candidates(top_areas[0], top_occ)
        if not t_large_cands:
            continue
        t_large = random.choice(t_large_cands)
        top_occ |= t_large["cells"]

        t_small_cands = _enumerate_candidates(top_areas[1], top_occ)
        if not t_small_cands:
            continue
        t_small = random.choice(t_small_cands)
        top_occ |= t_small["cells"]
        top_fg = set(top_occ)

        bm_cands = _enumerate_candidates(
            bottom_areas[0],
            set(),
            allowed_overlap=set(t_large["cells"]),
            forbidden_overlap=top_fg - set(t_large["cells"]),
            required_overlap=1,
            overlap_safe_in_candidate=True,
        )
        if not bm_cands:
            continue

        random.shuffle(bm_cands)
        for b_medium in bm_cands[:40]:
            bottom_occ = set(b_medium["cells"])
            b_large_cands = _enumerate_candidates(
                bottom_areas[1],
                bottom_occ,
                allowed_overlap=set(t_small["safe_abs"]),
                forbidden_overlap=top_fg - set(t_small["safe_abs"]),
                required_overlap=1,
                overlap_safe_in_candidate=False,
            )
            if not b_large_cands:
                continue
            b_large = random.choice(b_large_cands)
            bottom_occ |= b_large["cells"]

            top_components = [
                {"area": int(top_areas[0]), "cells": set(t_large["cells"])},
                {"area": int(top_areas[1]), "cells": set(t_small["cells"])},
            ]
            bottom_components = [
                {"area": int(bottom_areas[0]), "cells": set(b_medium["cells"])},
                {"area": int(bottom_areas[1]), "cells": set(b_large["cells"])},
            ]

            ok = True
            bottom_fg = set(bottom_occ)
            for area in sorted(top_areas[2:], reverse=True):
                cands = _enumerate_candidates(
                    area,
                    top_occ,
                    allowed_overlap=set(),
                    forbidden_overlap=bottom_fg,
                    required_overlap=0,
                )
                if not cands:
                    ok = False
                    break
                comp = random.choice(cands)
                top_occ |= comp["cells"]
                top_components.append({"area": int(area), "cells": set(comp["cells"])})
            if not ok:
                continue

            top_fg = set(top_occ)
            for area in sorted(bottom_areas[2:], reverse=True):
                cands = _enumerate_candidates(
                    area,
                    bottom_occ,
                    allowed_overlap=set(),
                    forbidden_overlap=top_fg,
                    required_overlap=0,
                )
                if not cands:
                    ok = False
                    break
                comp = random.choice(cands)
                bottom_occ |= comp["cells"]
                bottom_components.append({"area": int(area), "cells": set(comp["cells"])})
            if not ok:
                continue

            top_colors = _assign_colors(len(top_components), top_palette)
            bottom_colors = _assign_colors(len(bottom_components), bottom_palette)
            for comp, color in zip(top_components, top_colors):
                comp["color"] = int(color)
            for comp, color in zip(bottom_components, bottom_colors):
                comp["color"] = int(color)

            grid = np.full(GRID_SHAPE, BG, dtype=int)
            grid[HALF_SHAPE[0], :] = SEP
            grid[: HALF_SHAPE[0], :] = _render_half(top_components)
            grid[HALF_SHAPE[0] + 1 :, :] = _render_half(bottom_components)
            if _semantic_valid(grid):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
