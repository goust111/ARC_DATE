import random
from collections import deque

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_COLOR = 5
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACE_ATTEMPTS = 220


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _expand8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _legend_candidates(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []

    for r in range(h - 2):
        for c0 in range(w - 3):
            frame = int(grid[r, c0])
            if frame == bg:
                continue
            for c1 in range(c0 + 3, w):
                if not np.all(grid[r, c0 : c1 + 1] == frame):
                    break
                if not np.all(grid[r + 2, c0 : c1 + 1] == frame):
                    continue
                if int(grid[r + 1, c0]) != frame or int(grid[r + 1, c1]) != frame:
                    continue
                seq = grid[r + 1, c0 + 1 : c1]
                if seq.size < 2:
                    continue
                vals = list(map(int, seq))
                if any(v in (bg, frame) for v in vals):
                    continue
                if len(set(vals)) != len(vals):
                    continue
                mask = np.zeros_like(grid, dtype=bool)
                mask[r : r + 3, c0 : c1 + 1] = True
                candidates.append(
                    {
                        "orientation": "h",
                        "frame": frame,
                        "seq": vals,
                        "mask": mask,
                        "area": 3 * (c1 - c0 + 1),
                    }
                )

    for r0 in range(h - 3):
        for c in range(w - 2):
            frame = int(grid[r0, c])
            if frame == bg:
                continue
            for r1 in range(r0 + 3, h):
                if not np.all(grid[r0 : r1 + 1, c] == frame):
                    break
                if not np.all(grid[r0 : r1 + 1, c + 2] == frame):
                    continue
                if int(grid[r0, c + 1]) != frame or int(grid[r1, c + 1]) != frame:
                    continue
                seq = grid[r0 + 1 : r1, c + 1]
                if seq.size < 2:
                    continue
                vals = list(map(int, seq))
                if any(v in (bg, frame) for v in vals):
                    continue
                if len(set(vals)) != len(vals):
                    continue
                mask = np.zeros_like(grid, dtype=bool)
                mask[r0 : r1 + 1, c : c + 3] = True
                candidates.append(
                    {
                        "orientation": "v",
                        "frame": frame,
                        "seq": vals,
                        "mask": mask,
                        "area": (r1 - r0 + 1) * 3,
                    }
                )

    assert candidates, "legend not found"
    candidates.sort(key=lambda d: d["area"], reverse=True)
    best = candidates[0]
    tied = [cand for cand in candidates if cand["area"] == best["area"]]
    assert len(tied) == 1, "legend should be unique"
    return best


def _direction_from_arrow(
    grid: np.ndarray,
    bg: int,
    obj_cells: list[tuple[int, int]],
    arrow: tuple[int, int],
) -> tuple[int, int]:
    h, w = grid.shape
    obj_set = set(obj_cells)
    hits: set[tuple[int, int]] = set()
    ar, ac = arrow
    for r, c in obj_cells:
        for dr, dc in DIRS:
            mr, mc = r + dr, c + dc
            rr, cc = r + 2 * dr, c + 2 * dc
            if (rr, cc) != (ar, ac):
                continue
            if not (0 <= mr < h and 0 <= mc < w):
                continue
            if (mr, mc) in obj_set:
                continue
            if int(grid[mr, mc]) != bg:
                continue
            hits.add((dr, dc))
    assert len(hits) == 1, "arrow must define a unique direction"
    return next(iter(hits))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each compact foreground object into a gradient shadow strip using its nearby arrow cue.

    Input structure:
    - The dominant color is the background.
    - Several different-colored 4-connected objects appear, each with one same-colored singleton arrow head placed two cells away in one cardinal direction.
    - A separate framed legend strip contains one horizontal or vertical ordered sequence of distinct colors that defines the gradient.

    Rule:
    - Read the legend colors in strip order to obtain the repeating gradient palette.
    - For each object, pair it with its same-colored singleton arrow and infer the projection direction from that offset.
    - Let N be the object's area; along that direction, project rays from the object's pixels for N steps, coloring step k with the k-th legend color cyclically.
    - Keep the original object pixels unchanged, erase every arrow singleton to background, and stop each ray once it reaches another object's original pixel.

    Key insight:
    - The answer combines three separate cues: same-color singleton arrows give directions, object area gives projection length, and the framed ordered legend gives the shadow colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    legend = _legend_candidates(g, bg)
    legend_mask = legend["mask"]
    seq = legend["seq"]

    outside = ~legend_mask
    colors = sorted({int(v) for v in np.unique(g[outside]) if int(v) != bg})
    objects = []

    for color in colors:
        comps = _find_components((g == color) & outside)
        big = [comp for comp in comps if len(comp) > 1]
        singles = [comp for comp in comps if len(comp) == 1]
        if len(big) != 1 or len(singles) != 1:
            continue
        obj_cells = sorted(big[0])
        arrow = singles[0][0]
        dr, dc = _direction_from_arrow(g, bg, obj_cells, arrow)
        objects.append(
            {
                "color": color,
                "cells": obj_cells,
                "arrow": arrow,
                "dir": (dr, dc),
                "area": len(obj_cells),
            }
        )

    assert objects, "no valid object-arrow pairs found"

    all_object_mask = np.zeros((h, w), dtype=bool)
    arrow_mask = np.zeros((h, w), dtype=bool)
    for obj in objects:
        for r, c in obj["cells"]:
            all_object_mask[r, c] = True
        ar, ac = obj["arrow"]
        arrow_mask[ar, ac] = True

    out = g.copy()
    out[arrow_mask] = bg

    for obj in objects:
        own_mask = np.zeros((h, w), dtype=bool)
        for r, c in obj["cells"]:
            own_mask[r, c] = True
        stop_mask = (all_object_mask & (~own_mask)) | legend_mask
        step_grid = np.full((h, w), 10**9, dtype=int)
        dr, dc = obj["dir"]
        area = int(obj["area"])

        for sr, sc in obj["cells"]:
            for step in range(1, area + 1):
                nr = sr + dr * step
                nc = sc + dc * step
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if stop_mask[nr, nc]:
                    break
                if own_mask[nr, nc] or arrow_mask[nr, nc]:
                    continue
                if step < step_grid[nr, nc]:
                    step_grid[nr, nc] = step

        coords = np.argwhere(step_grid < 10**9)
        for r, c in coords:
            step = int(step_grid[r, c])
            out[int(r), int(c)] = int(seq[(step - 1) % len(seq)])

    return out


def _grow_shape(area: int) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        frontier = set()
        frontier.update({(-1, 0), (1, 0), (0, -1), (0, 1)})
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in DIRS:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)

        if len(cells) != area:
            continue

        norm = _normalize(list(cells))
        r0, c0, r1, c1 = _bbox(norm)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh > 4 or bw > 4:
            continue
        if area >= 3 and (bh == 1 or bw == 1):
            continue
        if area > 2 and area == bh * bw:
            continue
        if area / (bh * bw) < 0.5:
            continue
        return norm
    return None


def _make_spec(area: int, color: int):
    shape = _grow_shape(area)
    if shape is None:
        return None
    cell_set = set(shape)
    dr, dc = random.choice(DIRS)
    boundary = []
    for r, c in shape:
        gap = (r + dr, c + dc)
        arrow = (r + 2 * dr, c + 2 * dc)
        if gap not in cell_set and arrow not in cell_set:
            boundary.append((r, c, arrow))
    if not boundary:
        return None
    _r, _c, arrow = random.choice(boundary)

    shadow = set()
    for r, c in shape:
        for step in range(1, area + 1):
            shadow.add((r + dr * step, c + dc * step))

    rel_all = set(shape) | {arrow} | shadow
    min_r = min(r for r, _ in rel_all)
    max_r = max(r for r, _ in rel_all)
    min_c = min(c for _, c in rel_all)
    max_c = max(c for _, c in rel_all)
    return {
        "color": color,
        "coords": shape,
        "arrow": arrow,
        "dir": (dr, dc),
        "area": area,
        "min_r": min_r,
        "max_r": max_r,
        "min_c": min_c,
        "max_c": max_c,
    }


def _instantiate(spec: dict, top: int, left: int, h: int, w: int):
    obj = {(top + r, left + c) for r, c in spec["coords"]}
    arrow = (top + spec["arrow"][0], left + spec["arrow"][1])
    dr, dc = spec["dir"]
    shadow = set()
    for r, c in spec["coords"]:
        sr = top + r
        sc = left + c
        for step in range(1, spec["area"] + 1):
            nr = sr + dr * step
            nc = sc + dc * step
            if 0 <= nr < h and 0 <= nc < w:
                shadow.add((nr, nc))
    shadow -= obj
    shadow.discard(arrow)
    return {
        "color": spec["color"],
        "cells": obj,
        "arrow": arrow,
        "shadow": shadow,
        "dir": spec["dir"],
        "area": spec["area"],
    }


def _fit_ranges(spec: dict, h: int, w: int):
    row_lo = -spec["min_r"]
    row_hi = h - 1 - spec["max_r"]
    col_lo = -spec["min_c"]
    col_hi = w - 1 - spec["max_c"]
    return row_lo, row_hi, col_lo, col_hi


def _try_place_free(
    spec: dict,
    h: int,
    w: int,
    occupied_moat: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    reserved_shadow: set[tuple[int, int]],
):
    row_lo, row_hi, col_lo, col_hi = _fit_ranges(spec, h, w)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACE_ATTEMPTS):
        top = random.randint(row_lo, row_hi)
        left = random.randint(col_lo, col_hi)
        placed = _instantiate(spec, top, left, h, w)
        footprint = set(placed["cells"]) | {placed["arrow"]}
        if footprint & occupied_moat:
            continue
        if footprint & reserved_shadow:
            continue
        if placed["shadow"] & occupied_input:
            continue
        if placed["shadow"] & reserved_shadow:
            continue
        return placed
    return None


def _try_place_blocker(
    spec: dict,
    h: int,
    w: int,
    occupied_moat: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    reserved_shadow: set[tuple[int, int]],
    target_shadow: set[tuple[int, int]],
):
    targets = list(target_shadow)
    random.shuffle(targets)
    coords = list(spec["coords"])
    random.shuffle(coords)

    for target in targets[: min(len(targets), PLACE_ATTEMPTS)]:
        tr, tc = target
        for anchor_r, anchor_c in coords:
            top = tr - anchor_r
            left = tc - anchor_c
            row_lo, row_hi, col_lo, col_hi = _fit_ranges(spec, h, w)
            if not (row_lo <= top <= row_hi and col_lo <= left <= col_hi):
                continue
            placed = _instantiate(spec, top, left, h, w)
            footprint = set(placed["cells"]) | {placed["arrow"]}
            if target not in placed["cells"]:
                continue
            if footprint & occupied_moat:
                continue
            if placed["shadow"] & occupied_input:
                continue
            if placed["shadow"] & reserved_shadow:
                continue
            if placed["shadow"] & target_shadow:
                continue
            return placed
    return None


def _paint_feature(grid: np.ndarray, placed: dict):
    color = int(placed["color"])
    for r, c in placed["cells"]:
        grid[r, c] = color
    ar, ac = placed["arrow"]
    grid[ar, ac] = color


def _legend_patch(seq: list[int], orientation: str) -> np.ndarray:
    if orientation == "h":
        patch = np.full((3, len(seq) + 2), FRAME_COLOR, dtype=int)
        patch[1, 1:-1] = np.array(seq, dtype=int)
        return patch
    patch = np.full((len(seq) + 2, 3), FRAME_COLOR, dtype=int)
    patch[1:-1, 1] = np.array(seq, dtype=int)
    return patch


def _place_legend(grid: np.ndarray, patch: np.ndarray):
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACE_ATTEMPTS):
        top = random.randint(1, h - ph - 1)
        left = random.randint(1, w - pw - 1)
        region = grid[top : top + ph, left : left + pw]
        if np.any(region != grid[0, 0]):
            continue
        grid[top : top + ph, left : left + pw] = patch
        cells = {(r, c) for r in range(top, top + ph) for c in range(left, left + pw)}
        return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 26)
    w = random.randint(16, 26)
    bg = random.choice(BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)

    orientation = random.choice(["h", "v"])
    legend_len = random.randint(3, 4)
    object_count = random.randint(2, 4)

    pool = [c for c in range(10) if c not in (bg, FRAME_COLOR)]
    need = legend_len + object_count
    if len(pool) < need:
        return None
    sampled = random.sample(pool, need)
    seq = sampled[:legend_len]
    obj_colors = sampled[legend_len:]

    legend_cells = _place_legend(grid, _legend_patch(seq, orientation))
    if legend_cells is None:
        return None

    occupied_input = set(legend_cells)
    occupied_moat = _expand8(occupied_input, h, w)
    reserved_shadow: set[tuple[int, int]] = set()

    objects = []
    blocked = random.random() < 0.5

    specs = []
    for color in obj_colors:
        area = random.randint(2, 8)
        spec = _make_spec(area, color)
        if spec is None:
            return None
        specs.append(spec)

    if blocked and len(specs) >= 2:
        first = _try_place_free(specs[0], h, w, occupied_moat, occupied_input, reserved_shadow)
        if first is None:
            return None

        target_shadow = set(first["shadow"])
        if not target_shadow:
            return None

        second = _try_place_blocker(
            specs[1],
            h,
            w,
            occupied_moat | _expand8(set(first["cells"]) | {first["arrow"]}, h, w),
            occupied_input | set(first["cells"]) | {first["arrow"]},
            reserved_shadow,
            target_shadow,
        )
        if second is None:
            return None

        for placed in (first, second):
            objects.append(placed)
            footprint = set(placed["cells"]) | {placed["arrow"]}
            occupied_input |= footprint
            occupied_moat |= _expand8(footprint, h, w)
        reserved_shadow |= first["shadow"]
        reserved_shadow |= second["shadow"]
        start_idx = 2
    else:
        start_idx = 0

    for spec in specs[start_idx:]:
        placed = _try_place_free(spec, h, w, occupied_moat, occupied_input, reserved_shadow)
        if placed is None:
            return None
        objects.append(placed)
        footprint = set(placed["cells"]) | {placed["arrow"]}
        occupied_input |= footprint
        occupied_moat |= _expand8(footprint, h, w)
        reserved_shadow |= placed["shadow"]

    for placed in objects:
        _paint_feature(grid, placed)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid shadow-projection input")
