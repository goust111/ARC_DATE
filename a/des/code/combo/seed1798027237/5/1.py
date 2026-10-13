from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        q = deque([(r0, c0)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for rr, cc in _neighbors4(r, c, h, w):
                if mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if bg[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    return bg & ~seen


def _palette_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    grey_mask = grid == GREY
    comps = _components(grey_mask)
    h, w = grid.shape
    candidates: list[tuple[int, dict[int, int], np.ndarray]] = []

    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if r0 != 0 or c1 != w - 1 or bw != 5 or bh < 3 or bh % 2 == 0:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected_grey = np.fromfunction(
            lambda rr, cc: (rr % 2 == 0) | (cc % 2 == 0),
            (bh, bw),
            dtype=int,
        ).astype(bool)
        if not np.array_equal(sub == GREY, expected_grey):
            continue

        mapping: dict[int, int] = {}
        ok = True
        for rr in range(1, bh, 2):
            src = int(sub[rr, 1])
            fill = int(sub[rr, 3])
            if src in (BG, GREY) or fill in (BG, GREY) or src == fill or src in mapping:
                ok = False
                break
            mapping[src] = fill
        if not ok or not mapping:
            continue

        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append((len(mapping), mapping, legend_mask))

    assert candidates, "expected a grey palette legend in the top-right corner"
    _n, mapping, legend_mask = max(candidates, key=lambda x: x[0])
    return mapping, legend_mask


def _mask_from_component(coords: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
    for r, c in coords:
        mask[r - r0 + 1, c - c0 + 1] = True
    return mask, r0, c0


def _find_gap(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c]:
                continue

            neighbor_dirs = []
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if mask[rr, cc]:
                    neighbor_dirs.append((dr, dc))
            if len(neighbor_dirs) != 2:
                continue

            opposite = {(-1, 0), (1, 0)} == set(neighbor_dirs) or {(0, -1), (0, 1)} == set(neighbor_dirs)
            if not opposite:
                continue

            closed_mask = mask.copy()
            closed_mask[r, c] = True
            interior = _enclosed_background(closed_mask)
            if not interior.any():
                continue

            interior_dirs = []
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if interior[rr, cc]:
                    interior_dirs.append((dr, dc))
            if len(interior_dirs) != 1:
                continue

            idir = interior_dirs[0]
            out_dir = (-idir[0], -idir[1])
            interior_size = int(interior.sum())
            candidates.append(((r, c), out_dir, interior_size))

    assert len(candidates) == 1, "open contour must have exactly one resolvable gap"
    gap, direction, _score = candidates[0]
    return gap, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose the transformation by contour closure:
    - closed contours fill their enclosed interior using the top-right palette legend
    - open contours project a same-color ray out of their unique gap
    - the grey palette legend is erased in the output
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mapping, legend_mask = _palette_legend(g)
    work = g.copy()
    work[legend_mask] = BG
    out = work.copy()

    colors = [int(c) for c in np.unique(work) if int(c) != BG]
    closed_colors: set[int] = set()

    for color in colors:
        comps = _components(work == color)
        assert len(comps) == 1, "each contour color must define one object"
        coords = comps[0]
        mask, r0, c0 = _mask_from_component(coords)
        interior = _enclosed_background(mask)

        if interior.any():
            assert color in mapping, "closed contour color must appear in the legend"
            closed_colors.add(color)
            fill_color = mapping[color]
            for rr, cc in np.argwhere(interior):
                out[r0 + int(rr) - 1, c0 + int(cc) - 1] = fill_color
        else:
            gap, direction = _find_gap(mask)
            gr = r0 + gap[0] - 1
            gc = c0 + gap[1] - 1
            dr, dc = direction
            length = len(coords)
            rr, cc = gr, gc
            for _ in range(length):
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = color
                rr += dr
                cc += dc

    assert closed_colors == set(mapping), "legend rows must correspond exactly to closed contours"
    out[legend_mask] = BG
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _boundary_of_filled(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros_like(filled, dtype=bool)
    for r, c in np.argwhere(filled):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w) or not filled[rr, cc]:
                boundary[r, c] = True
                break
    return boundary


def _carve_notch(filled: np.ndarray, side: str) -> None:
    h, w = filled.shape
    if side in ("top", "bottom"):
        notch_w = random.randint(1, w - 4)
        start = random.randint(2, w - notch_w - 2)
        depth = random.randint(1, h - 4)
        if side == "top":
            filled[:depth, start : start + notch_w] = False
        else:
            filled[h - depth :, start : start + notch_w] = False
    else:
        notch_h = random.randint(1, h - 4)
        start = random.randint(2, h - notch_h - 2)
        depth = random.randint(1, w - 4)
        if side == "left":
            filled[start : start + notch_h, :depth] = False
        else:
            filled[start : start + notch_h, w - depth :] = False


def _make_filled_shape() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(300):
        h = random.randint(6, 8)
        w = random.randint(6, 9)
        filled = np.ones((h, w), dtype=bool)
        notch_count = random.choices([0, 1, 2], weights=[3, 5, 2])[0]
        for side in random.sample(["top", "bottom", "left", "right"], k=notch_count):
            _carve_notch(filled, side)

        if not _is_connected(filled):
            continue

        boundary = _boundary_of_filled(filled)
        interior = filled & ~boundary
        if not interior.any():
            continue
        if not _is_connected(interior):
            continue

        area = int(boundary.sum())
        if not (14 <= area <= 24):
            continue
        return filled, boundary, interior

    raise RuntimeError("failed to build a valid contour shape")


def _open_gap_candidates(boundary: np.ndarray, interior: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = boundary.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in np.argwhere(boundary):
        r = int(r)
        c = int(c)
        if not (r in (0, h - 1) or c in (0, w - 1)):
            continue

        neighbor_dirs = []
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and boundary[rr, cc]:
                neighbor_dirs.append((dr, dc))
        if len(neighbor_dirs) != 2:
            continue

        opposite = {(-1, 0), (1, 0)} == set(neighbor_dirs) or {(0, -1), (0, 1)} == set(neighbor_dirs)
        if not opposite:
            continue

        inside_dirs = []
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and interior[rr, cc]:
                inside_dirs.append((dr, dc))
        if len(inside_dirs) != 1:
            continue

        idir = inside_dirs[0]
        candidates.append(((r, c), (-idir[0], -idir[1])))
    return candidates


def _local_ray(gap: tuple[int, int], direction: tuple[int, int], length: int, h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    rr, cc = gap
    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _make_object_spec(color: int, is_open: bool) -> dict:
    for _ in range(300):
        _filled, boundary, interior = _make_filled_shape()
        cells = [(int(r), int(c)) for r, c in np.argwhere(boundary)]

        if not is_open:
            return {
                "color": color,
                "open": False,
                "shape": boundary.shape,
                "cells": cells,
            }

        candidates = _open_gap_candidates(boundary, interior)
        if not candidates:
            continue

        gap, direction = random.choice(candidates)
        open_boundary = boundary.copy()
        open_boundary[gap] = False
        open_cells = [(int(r), int(c)) for r, c in np.argwhere(open_boundary)]
        if _enclosed_background(np.pad(open_boundary, 1)).any():
            continue

        mask = np.pad(open_boundary, 1)
        try:
            found_gap, found_dir = _find_gap(mask)
        except AssertionError:
            continue

        if found_gap != (gap[0] + 1, gap[1] + 1) or found_dir != direction:
            continue

        return {
            "color": color,
            "open": True,
            "shape": open_boundary.shape,
            "cells": open_cells,
            "gap": gap,
            "direction": direction,
        }

    raise RuntimeError("failed to build an open contour")


def _mark_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[max(0, r0) : min(mask.shape[0], r1 + 1), max(0, c0) : min(mask.shape[1], c1 + 1)] = True


def _legend_patch(rows: list[tuple[int, int]]) -> np.ndarray:
    n = len(rows)
    patch = np.full((2 * n + 1, 5), GREY, dtype=int)
    for i, (src, fill) in enumerate(rows):
        patch[2 * i + 1, 1] = src
        patch[2 * i + 1, 3] = fill
    return patch


def _global_ray(spec: dict, top: int, left: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    gap_r, gap_c = spec["gap"]
    dr, dc = spec["direction"]
    rr = top + gap_r
    cc = left + gap_c
    cells: list[tuple[int, int]] = []
    for _ in range(len(spec["cells"])):
        if 0 <= rr < grid_h and 0 <= cc < grid_w:
            cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _try_place(grid_h: int, grid_w: int, specs: list[dict], legend_mask: np.ndarray) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    space_mask = legend_mask.copy()
    ray_mask = legend_mask.copy()

    ordered = sorted(specs, key=lambda s: (s["shape"][0] * s["shape"][1], len(s["cells"])), reverse=True)
    placements: list[tuple[dict, int, int]] = []

    for spec in ordered:
        sh, sw = spec["shape"]
        placed = False
        for _ in range(600):
            top = random.randint(0, grid_h - sh)
            left = random.randint(0, grid_w - sw)
            r0 = top - 1
            c0 = left - 1
            r1 = top + sh
            c1 = left + sw
            if space_mask[max(0, r0) : min(grid_h, r1 + 1), max(0, c0) : min(grid_w, c1 + 1)].any():
                continue

            if spec["open"]:
                ray = _global_ray(spec, top, left, grid_h, grid_w)
                if len(ray) < 2:
                    continue
                if any(ray_mask[r, c] for r, c in ray):
                    continue
            else:
                ray = []

            for r, c in spec["cells"]:
                grid[top + r, left + c] = spec["color"]

            _mark_rect(space_mask, r0, c0, r1, c1)
            for r, c in ray:
                ray_mask[r, c] = True
            placements.append((spec, top, left))
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    closed_count = random.randint(1, 3)
    open_count = random.randint(1, min(3, 5 - closed_count))
    total = closed_count + open_count

    colors = random.sample(PALETTE, total)
    fill_colors = random.sample([c for c in PALETTE if c not in colors], closed_count)
    closed_colors = colors[:closed_count]
    open_colors = colors[closed_count:]

    closed_specs = [_make_object_spec(color, is_open=False) for color in closed_colors]
    open_specs = [_make_object_spec(color, is_open=True) for color in open_colors]
    specs = closed_specs + open_specs
    random.shuffle(specs)

    legend_rows = list(zip(closed_colors, fill_colors, strict=False))
    random.shuffle(legend_rows)
    legend = _legend_patch([(int(src), int(fill)) for src, fill in legend_rows])

    grid_h = random.randint(max(18, legend.shape[0] + 4), 26)
    grid_w = random.randint(22, 30)
    if legend.shape[1] + 2 >= grid_w:
        return None

    legend_mask = np.zeros((grid_h, grid_w), dtype=bool)
    legend_top = 0
    legend_left = grid_w - legend.shape[1]
    legend_mask[legend_top : legend_top + legend.shape[0], legend_left:] = True
    _mark_rect(legend_mask, legend_top - 1, legend_left - 1, legend_top + legend.shape[0], grid_w - 1)

    grid = _try_place(grid_h, grid_w, specs, legend_mask)
    if grid is None:
        return None

    grid[legend_top : legend_top + legend.shape[0], legend_left:] = legend

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if not np.all(out[legend_top : legend_top + legend.shape[0], legend_left:] == BG):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid sample")
