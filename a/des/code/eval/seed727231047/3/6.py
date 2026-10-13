import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
LIGHTHOUSE_SIZE = 2
GEN_RETRIES = 600

RAY_SPECS = (
    ("up", (-1, 0), (-1, 0)),
    ("right", (0, 2), (0, 1)),
    ("down", (2, 1), (1, 0)),
    ("left", (1, -1), (0, -1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mark(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _probe_corner_legend(
    g: np.ndarray, start_row: int, row_step: int, start_col: int
) -> tuple[np.ndarray, dict[int, int], dict[int, int]] | None:
    h, w = g.shape
    if w < 2:
        return None
    cols = [start_col, start_col + 1]
    if not (0 <= cols[0] < w and 0 <= cols[1] < w):
        return None

    rows: list[int] = []
    r = start_row
    while 0 <= r < h and int(g[r, cols[0]]) != BLACK and int(g[r, cols[1]]) != BLACK:
        rows.append(r)
        r += row_step
    if len(rows) < 3:
        return None

    ordered_rows = sorted(rows)
    patch = g[ordered_rows, :][:, cols]
    if np.any(patch == BLACK) or np.any(patch == GREY):
        return None

    inward_col = cols[1] + 1 if start_col == 0 else cols[0] - 1
    if 0 <= inward_col < w and np.any(g[ordered_rows, inward_col] != BLACK):
        return None

    inward_row = ordered_rows[-1] + 1 if row_step == 1 else ordered_rows[0] - 1
    if 0 <= inward_row < h and np.any(g[inward_row, cols] != BLACK):
        return None

    left = [int(x) for x in patch[:, 0]]
    right = [int(x) for x in patch[:, 1]]
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        return None

    mapping = {src: dst for src, dst in zip(left, right, strict=False)}
    if len(mapping) != len(left):
        return None
    if set(mapping) != set(mapping.values()):
        return None

    priority = {color: idx for idx, color in enumerate(left)}
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[np.ix_(ordered_rows, cols)] = True
    return legend_mask, mapping, priority


def _find_legend(g: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    h, w = g.shape
    candidates = []
    probes = (
        _probe_corner_legend(g, 0, 1, 0),
        _probe_corner_legend(g, 0, 1, w - 2),
        _probe_corner_legend(g, h - 1, -1, 0),
        _probe_corner_legend(g, h - 1, -1, w - 2),
    )
    for item in probes:
        if item is not None:
            candidates.append(item)
    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _find_lighthouses(
    g: np.ndarray, legend_mask: np.ndarray, palette: set[int]
) -> list[dict[str, object]]:
    lighthouses: list[dict[str, object]] = []
    for color in sorted(palette):
        mask = (g == int(color)) & (~legend_mask)
        for comp in _components(mask):
            if len(comp) != LIGHTHOUSE_SIZE * LIGHTHOUSE_SIZE:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
                continue
            lighthouses.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    lighthouses.sort(key=lambda item: tuple(item["bbox"]))  # deterministic
    assert lighthouses, "expected at least one lighthouse"
    return lighthouses


def _ray_lines(
    g: np.ndarray, legend_mask: np.ndarray, lighthouses: list[dict[str, object]]
) -> list[dict[str, object]]:
    h, w = g.shape
    rays: list[dict[str, object]] = []
    for lighthouse in lighthouses:
        color = int(lighthouse["color"])
        r0, c0, _r1, _c1 = lighthouse["bbox"]
        for name, (sr_off, sc_off), (dr, dc) in RAY_SPECS:
            cells: list[tuple[int, int]] = []
            r = int(r0 + sr_off)
            c = int(c0 + sc_off)
            while 0 <= r < h and 0 <= c < w:
                if legend_mask[r, c]:
                    break
                value = int(g[r, c])
                if value not in (BLACK, GREY):
                    break
                cells.append((r, c))
                r += dr
                c += dc
            rays.append(
                {
                    "bbox": tuple(lighthouse["bbox"]),
                    "source_color": int(color),
                    "dir": name,
                    "cells": cells,
                }
            )
    return rays


def _claim_details(
    g: np.ndarray,
    rays: list[dict[str, object]],
    mapping: dict[int, int],
) -> dict[tuple[int, int], list[int]]:
    claims: dict[tuple[int, int], list[int]] = {}
    for ray in rays:
        color = int(ray["source_color"])
        for cell in ray["cells"]:
            claims.setdefault(cell, []).append(int(color))
            if int(g[cell]) == GREY:
                color = int(mapping[color])
    return claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate lighthouse beams that recolor prisms and resolve overlaps by legend order.

    The input contains:
    - a black background,
    - one corner legend of adjacent color pairs (left = before prism, right = after prism),
    - several monochrome 2x2 lighthouses,
    - grey single-cell prisms.

    Each lighthouse emits four 1-cell-wide rays. For a 2x2 raster block, the side-center
    ports are resolved by the unique rotationally symmetric clockwise tie-break:
    up uses the exterior cell above the left column, right uses the exterior cell beside the
    top row, down uses the exterior cell below the right column, and left uses the exterior
    cell beside the bottom row. Rays traverse black and grey cells, repaint visited cells,
    change color after passing a grey prism, and stop at any other non-black cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_mask, mapping, priority = _find_legend(g)
    lighthouses = _find_lighthouses(g, legend_mask, set(mapping))
    rays = _ray_lines(g, legend_mask, lighthouses)
    claims = _claim_details(g, rays, mapping)

    out = g.copy()
    for cell, colors in claims.items():
        best = max(colors, key=lambda color: priority[int(color)])
        out[cell] = int(best)
    return out


def _random_derangement(colors: list[int]) -> list[int] | None:
    for _ in range(80):
        out = colors[:]
        random.shuffle(out)
        if all(a != b for a, b in zip(colors, out, strict=False)):
            return out
    return None


def _legend_bbox(corner: str, h: int, w: int, k: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, k - 1, 1
    if corner == "tr":
        return 0, w - 2, k - 1, w - 1
    if corner == "bl":
        return h - k, 0, h - 1, 1
    return h - k, w - 2, h - 1, w - 1


def _paint_legend(
    grid: np.ndarray, corner: str, sources: list[int], targets: list[int]
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = _legend_bbox(corner, h, w, len(sources))
    for i, (src, dst) in enumerate(zip(sources, targets, strict=False)):
        grid[r0 + i, c0] = int(src)
        grid[r0 + i, c0 + 1] = int(dst)
    return r0, c0, r1, c1


def _can_place_lighthouse(blocked: np.ndarray, r: int, c: int) -> bool:
    h, w = blocked.shape
    if not (1 <= r <= h - 4 and 1 <= c <= w - 4):
        return False
    return not bool(blocked[r - 1 : r + 3, c - 1 : c + 3].any())


def _place_lighthouse(
    grid: np.ndarray,
    filled: np.ndarray,
    blocked: np.ndarray,
    r: int,
    c: int,
    color: int,
) -> tuple[int, int, int, int]:
    assert _can_place_lighthouse(blocked, r, c)
    grid[r : r + 2, c : c + 2] = int(color)
    filled[r : r + 2, c : c + 2] = True
    _mark(blocked, r, c, r + 1, c + 1, pad=1)
    return r, c, r + 1, c + 1


def _find_overlap_pair(
    grid: np.ndarray,
    filled: np.ndarray,
    blocked: np.ndarray,
    chosen_colors: list[int],
) -> tuple[dict[str, object], dict[str, object]] | None:
    h, w = grid.shape
    order = [(r, c) for r in range(1, h - 3) for c in range(1, w - 3)]
    random.shuffle(order)
    color_a, color_b = random.sample(chosen_colors, 2)

    for r_a, c_a in order:
        if not _can_place_lighthouse(blocked, r_a, c_a):
            continue
        b_candidates = []
        for r_b in range(1, r_a - 1):
            for c_b in range(c_a + 2, w - 3):
                if not _can_place_lighthouse(blocked, r_b, c_b):
                    continue
                cells_a = [(r_a, cc) for cc in range(c_a + 2, c_b + 2)]
                cells_b = [(rr, c_b + 1) for rr in range(r_b + 2, r_a + 1)]
                if len(cells_a) < 3 or len(cells_b) < 3:
                    continue
                if any(filled[cell] for cell in cells_a) or any(filled[cell] for cell in cells_b):
                    continue
                b_candidates.append((r_b, c_b, cells_a, cells_b))
        if not b_candidates:
            continue

        r_b, c_b, cells_a, cells_b = random.choice(b_candidates)
        bbox_a = _place_lighthouse(grid, filled, blocked, r_a, c_a, color_a)
        bbox_b = _place_lighthouse(grid, filled, blocked, r_b, c_b, color_b)
        for rr, cc in set(cells_a + cells_b):
            blocked[rr, cc] = True
        return (
            {"bbox": bbox_a, "color": int(color_a), "dir": "right", "corridor": cells_a},
            {"bbox": bbox_b, "color": int(color_b), "dir": "down", "corridor": cells_b},
        )
    return None


def _place_random_lighthouse_for_color(
    grid: np.ndarray,
    filled: np.ndarray,
    blocked: np.ndarray,
    color: int,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    candidates = [(r, c) for r in range(1, h - 3) for c in range(1, w - 3)]
    random.shuffle(candidates)
    for r, c in candidates:
        if _can_place_lighthouse(blocked, r, c):
            return _place_lighthouse(grid, filled, blocked, r, c, color)
    return None


def _advance_color(color: int, mapping: dict[int, int], steps: int) -> int:
    out = int(color)
    for _ in range(steps):
        out = int(mapping[out])
    return out


def _ray_index(rays: list[dict[str, object]]) -> dict[tuple[tuple[int, int, int, int], str], dict[str, object]]:
    return {(tuple(ray["bbox"]), str(ray["dir"])): ray for ray in rays}


def _choose_overlap_prisms(
    ray_a: dict[str, object],
    ray_b: dict[str, object],
    intersection: tuple[int, int],
    cell_usage: dict[tuple[int, int], int],
    mapping: dict[int, int],
    forbidden: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[int]] | None:
    cells_a = list(ray_a["cells"])
    cells_b = list(ray_b["cells"])
    if intersection not in cells_a or intersection not in cells_b:
        return None

    idx_a = cells_a.index(intersection)
    idx_b = cells_b.index(intersection)
    exclusive_a = [cell for cell in cells_a[:idx_a] if cell_usage[cell] == 1 and not forbidden[cell]]
    exclusive_b = [cell for cell in cells_b[:idx_b] if cell_usage[cell] == 1 and not forbidden[cell]]
    max_a = min(2, len(exclusive_a))
    max_b = min(2, len(exclusive_b))

    options = []
    for count_a in range(max_a + 1):
        for count_b in range(max_b + 1):
            if count_a + count_b == 0:
                continue
            color_a = _advance_color(int(ray_a["source_color"]), mapping, count_a)
            color_b = _advance_color(int(ray_b["source_color"]), mapping, count_b)
            if color_a != color_b:
                options.append((count_a, count_b))
    if not options and int(ray_a["source_color"]) != int(ray_b["source_color"]):
        options.append((0, 0))
    if not options:
        return None

    count_a, count_b = random.choice(options)
    prisms_a = []
    prisms_b = []
    if count_a:
        indices = sorted(random.sample(range(len(exclusive_a)), count_a))
        prisms_a = [exclusive_a[idx] for idx in indices]
    if count_b:
        indices = sorted(random.sample(range(len(exclusive_b)), count_b))
        prisms_b = [exclusive_b[idx] for idx in indices]

    exposed = set()
    if prisms_a:
        exposed.add(int(ray_a["source_color"]))
    if prisms_b:
        exposed.add(int(ray_b["source_color"]))
    return prisms_a, prisms_b, exposed


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 20)
    w = random.randint(14, 20)
    k = random.choice((3, 4))
    colors = random.sample(COLOR_POOL, k)
    mapped = _random_derangement(colors)
    if mapped is None:
        return None
    mapping = {src: dst for src, dst in zip(colors, mapped, strict=False)}

    grid = np.full((h, w), BLACK, dtype=int)
    filled = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    lr0, lc0, lr1, lc1 = _paint_legend(grid, corner, colors, mapped)
    _mark(filled, lr0, lc0, lr1, lc1)
    _mark(blocked, lr0, lc0, lr1, lc1, pad=1)
    legend_guard = np.zeros((h, w), dtype=bool)
    _mark(legend_guard, lr0, lc0, lr1, lc1, pad=1)

    want_overlap = random.random() < 0.5
    overlap_meta = None
    used_pair_colors: set[int] = set()
    if want_overlap:
        overlap = _find_overlap_pair(grid, filled, blocked, colors)
        if overlap is None:
            return None
        overlap_meta = overlap
        used_pair_colors = {int(overlap[0]["color"]), int(overlap[1]["color"])}

    lighthouse_colors = colors[:]
    if random.random() < 0.4:
        lighthouse_colors.append(random.choice(colors))
    if overlap_meta is not None:
        remaining = lighthouse_colors[:]
        for color in used_pair_colors:
            remaining.remove(int(color))
        lighthouse_colors = remaining
    random.shuffle(lighthouse_colors)

    extra_bboxes: list[tuple[int, int, int, int]] = []
    for color in lighthouse_colors:
        bbox = _place_random_lighthouse_for_color(grid, filled, blocked, int(color))
        if bbox is None:
            return None
        extra_bboxes.append(bbox)

    legend_mask, mapping_check, _priority = _find_legend(grid)
    if mapping_check != mapping:
        return None

    lighthouses = _find_lighthouses(grid, legend_mask, set(mapping))
    rays = _ray_lines(grid, legend_mask, lighthouses)
    ray_by_key = _ray_index(rays)

    cell_usage: dict[tuple[int, int], int] = {}
    for ray in rays:
        for cell in ray["cells"]:
            cell_usage[cell] = cell_usage.get(cell, 0) + 1

    planned_prisms: list[tuple[int, int]] = []
    planned_set: set[tuple[int, int]] = set()
    exposed_colors: set[int] = set()

    if overlap_meta is not None:
        ray_a = ray_by_key[(tuple(overlap_meta[0]["bbox"]), str(overlap_meta[0]["dir"]))]
        ray_b = ray_by_key[(tuple(overlap_meta[1]["bbox"]), str(overlap_meta[1]["dir"]))]
        intersection = (
            overlap_meta[0]["corridor"][-1]
            if overlap_meta[0]["corridor"][-1] == overlap_meta[1]["corridor"][-1]
            else None
        )
        if intersection is None:
            return None
        overlap_prisms = _choose_overlap_prisms(
            ray_a, ray_b, intersection, cell_usage, mapping, legend_guard
        )
        if overlap_prisms is None:
            return None
        prisms_a, prisms_b, overlap_exposed = overlap_prisms
        for cell in prisms_a + prisms_b:
            planned_set.add(cell)
            planned_prisms.append(cell)
        exposed_colors |= overlap_exposed

    for color in colors:
        if int(color) in exposed_colors:
            continue
        candidates = []
        for ray in rays:
            if int(ray["source_color"]) != int(color):
                continue
            cells = list(ray["cells"])
            for idx, cell in enumerate(cells):
                if cell in planned_set or cell_usage[cell] != 1 or legend_guard[cell]:
                    continue
                if any(prev in planned_set for prev in cells[:idx]):
                    continue
                candidates.append((idx, cell))
        if not candidates:
            return None
        best_idx = min(idx for idx, _cell in candidates)
        best_cells = [cell for idx, cell in candidates if idx == best_idx]
        cell = random.choice(best_cells)
        planned_set.add(cell)
        planned_prisms.append(cell)
        exposed_colors.add(int(color))

    if random.random() < 0.5:
        extra_candidates = []
        for ray in rays:
            cells = list(ray["cells"])
            prism_indices = [idx for idx, cell in enumerate(cells) if cell in planned_set]
            if not prism_indices:
                continue
            last_idx = max(prism_indices)
            for idx, cell in enumerate(cells[last_idx + 1 :], start=last_idx + 1):
                if cell in planned_set or cell_usage[cell] != 1 or legend_guard[cell]:
                    continue
                extra_candidates.append(cell)
        if extra_candidates:
            cell = random.choice(extra_candidates)
            planned_set.add(cell)
            planned_prisms.append(cell)

    line_cells = set(cell_usage)
    distractor_prisms: list[tuple[int, int]] = []
    if random.random() < 0.6:
        free_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if grid[r, c] == BLACK and (r, c) not in line_cells and not legend_guard[r, c]
        ]
        random.shuffle(free_cells)
        for cell in free_cells[: random.randint(0, 2)]:
            distractor_prisms.append(cell)

    for cell in planned_prisms + distractor_prisms:
        grid[cell] = GREY

    claims = _claim_details(grid, _ray_lines(grid, legend_mask, lighthouses), mapping)
    multi_claim = any(len(set(colors_here)) >= 2 for colors_here in claims.values())
    if want_overlap != multi_claim:
        return None

    if not planned_set <= set(claims):
        return None
    if len(exposed_colors) != len(colors):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    for cell in distractor_prisms:
        if int(out[cell]) != GREY:
            return None

    changed = int(np.count_nonzero(out != grid))
    if changed < len(planned_prisms) + len(colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate lighthouse-prism task input")
