from __future__ import annotations

import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
OPS = ("cw", "ccw", "hflip", "vflip")
INSTR_COLORS = {"cw": 2, "ccw": 3, "hflip": 4, "vflip": 6}
NON_INSTR_COLORS = [0, 1, 5, 7, 8, 9]

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
ARROW_COUNT_RANGE = (3, 5)
LINK_LENGTHS = (4, 5, 6, 7)
LEGEND_BUFFER = 1
GEN_RETRIES = 400
PATH_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _apply_op(direction: tuple[int, int], op: str) -> tuple[int, int]:
    dr, dc = direction
    if op == "cw":
        return _turn_right(direction)
    if op == "ccw":
        return _turn_left(direction)
    if op == "hflip":
        return dr, -dc
    if op == "vflip":
        return -dr, dc
    raise AssertionError(f"unknown op {op}")


def _invert_op(direction: tuple[int, int], op: str) -> tuple[int, int]:
    if op == "cw":
        return _turn_left(direction)
    if op == "ccw":
        return _turn_right(direction)
    return _apply_op(direction, op)


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    dr, dc = direction
    side1 = _turn_left(direction)
    side2 = _turn_right(direction)
    return {
        (r, c),
        (r + dr, c + dc),
        (r + side1[0], c + side1[1]),
        (r + side2[0], c + side2[1]),
    }


def _legend_origin(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 2
    if corner == "bl":
        return h - 2, 0
    if corner == "br":
        return h - 2, w - 2
    raise AssertionError(f"unknown corner {corner}")


def _legend_cells(corner: str, h: int, w: int) -> set[tuple[int, int]]:
    r0, c0 = _legend_origin(corner, h, w)
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _expanded(cells: set[tuple[int, int]], pad: int, h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _placement_forbidden(corner: str, h: int, w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    legend = _legend_cells(corner, h, w)
    other_corners: set[tuple[int, int]] = set()
    for name in ("tl", "tr", "bl", "br"):
        if name == corner:
            continue
        other_corners |= _legend_cells(name, h, w)
    halo = _expanded(legend, LEGEND_BUFFER, h, w) - legend
    return legend, other_corners | halo


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], dict[int, str], set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for corner in ("tl", "tr", "bl", "br"):
        r0, c0 = _legend_origin(corner, h, w)
        patch = g[r0 : r0 + 2, c0 : c0 + 2]
        if patch.shape != (2, 2):
            continue
        vals = [int(v) for v in patch.flat]
        if bg in vals or len(set(vals)) != 4:
            continue
        candidates.append(((r0, c0), patch.copy()))
    assert len(candidates) == 1, "expected exactly one corner legend"
    (r0, c0), patch = candidates[0]
    op_by_color = {
        int(patch[0, 0]): "cw",
        int(patch[0, 1]): "ccw",
        int(patch[1, 0]): "hflip",
        int(patch[1, 1]): "vflip",
    }
    return (r0, c0), op_by_color, set(op_by_color)


def _decode_arrow(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    cells = {tuple(map(int, xy)) for xy in coords}
    assert len(cells) == 4, "arrow must have 4 cells"
    center_candidates = []
    for r, c in cells:
        degree = sum((r + dr, c + dc) in cells for dr, dc in DIRS)
        if degree == 3:
            center_candidates.append((r, c))
    assert len(center_candidates) == 1, "arrow must have a unique center"
    center = center_candidates[0]
    sr = 0
    sc = 0
    cr, cc = center
    for r, c in cells:
        if (r, c) == center:
            continue
        sr += r - cr
        sc += c - cc
    direction = (int(sr), int(sc))
    assert direction in DIRS, "arrow orientation must be cardinal"
    assert cells == _arrow_cells(center, direction), "arrow shape must match template"
    return center, direction


def _parse_arrows(grid: np.ndarray, bg: int, legend_colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    arrows = []
    for color in sorted(int(v) for v in np.unique(g)):
        if color == bg or color in legend_colors:
            continue
        coords = np.argwhere(g == color)
        assert len(coords) == 4, f"color {color} must form exactly one arrow"
        center, direction = _decode_arrow(coords)
        arrows.append(
            {
                "id": len(arrows),
                "color": color,
                "center": center,
                "orig_dir": direction,
                "cells": {tuple(map(int, xy)) for xy in coords},
            }
        )
    assert arrows, "expected at least one arrow"
    arrows.sort(key=lambda item: (item["center"][0], item["center"][1], item["color"]))
    for idx, arrow in enumerate(arrows):
        arrow["id"] = idx
    return arrows


def _target_ids(
    arrows: list[dict],
    new_dirs: dict[int, tuple[int, int]],
    shape: tuple[int, int],
) -> dict[int, int | None]:
    h, w = shape
    cell_to_id: dict[tuple[int, int], int] = {}
    for arrow in arrows:
        aid = int(arrow["id"])
        for cell in arrow["cells"]:
            assert cell not in cell_to_id, "arrows must not overlap"
            cell_to_id[cell] = aid

    targets: dict[int, int | None] = {}
    for arrow in arrows:
        aid = int(arrow["id"])
        r, c = arrow["center"]
        dr, dc = new_dirs[aid]
        rr = r + 2 * dr
        cc = c + 2 * dc
        target = None
        while 0 <= rr < h and 0 <= cc < w:
            other = cell_to_id.get((rr, cc))
            if other is not None and other != aid:
                target = other
                break
            rr += dr
            cc += dc
        targets[aid] = target
    return targets


def _chain_order(targets: dict[int, int | None], count: int) -> list[int]:
    indegree = [0] * count
    for target in targets.values():
        if target is not None:
            indegree[target] += 1
    assert all(v <= 1 for v in indegree), "each arrow must have indegree at most 1"
    starts = [idx for idx, deg in enumerate(indegree) if deg == 0]
    assert len(starts) == 1, "expected exactly one chain start"

    order = []
    seen = set()
    cur = starts[0]
    while cur is not None:
        assert cur not in seen, "chain must not cycle"
        seen.add(cur)
        order.append(cur)
        cur = targets[cur]
    assert len(order) == count, "all arrows must lie on one chain"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each arrow by the command block at its tip, then follow the new
    pointing direction to recover the unique instruction chain.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    _, op_by_color, legend_colors = _find_legend(g, bg)
    arrows = _parse_arrows(g, bg, legend_colors)

    new_dirs: dict[int, tuple[int, int]] = {}
    for arrow in arrows:
        r, c = arrow["center"]
        dr, dc = arrow["orig_dir"]
        ir = r + 2 * dr
        ic = c + 2 * dc
        assert 0 <= ir < h and 0 <= ic < w, "instruction must stay in bounds"
        instr_color = int(g[ir, ic])
        assert instr_color in op_by_color, "every arrow must point to a valid instruction"
        new_dirs[int(arrow["id"])] = _apply_op(arrow["orig_dir"], op_by_color[instr_color])

    targets = _target_ids(arrows, new_dirs, g.shape)
    _chain_order(targets, len(arrows))

    out = np.full_like(g, bg)
    for arrow in arrows:
        color = int(arrow["color"])
        for r, c in _arrow_cells(arrow["center"], new_dirs[int(arrow["id"])]):
            assert 0 <= r < h and 0 <= c < w, "transformed arrow must stay in bounds"
            assert out[r, c] == bg, "transformed arrows must not overlap"
            out[r, c] = color
    return out


def _choose_ops(count: int, rng: random.Random) -> list[str]:
    ops = list(OPS)
    if count >= 4:
        chosen = rng.sample(ops, 4)
        while len(chosen) < count:
            chosen.append(rng.choice(ops))
        rng.shuffle(chosen)
        return chosen
    omitted = rng.randrange(4)
    pool = ops[:omitted] + ops[omitted + 1 :]
    chosen = rng.sample(pool, count)
    rng.shuffle(chosen)
    return chosen


def _center_clear(center: tuple[int, int], h: int, w: int, forbidden: set[tuple[int, int]]) -> bool:
    r, c = center
    if not (2 <= r < h - 2 and 2 <= c < w - 2):
        return False
    influence = {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
        (r - 2, c),
        (r + 2, c),
        (r, c - 2),
        (r, c + 2),
    }
    return not (influence & forbidden)


def _between(a: tuple[int, int], b: tuple[int, int], p: tuple[int, int]) -> bool:
    if a[0] == b[0] == p[0]:
        lo, hi = sorted((a[1], b[1]))
        return lo < p[1] < hi
    if a[1] == b[1] == p[1]:
        lo, hi = sorted((a[0], b[0]))
        return lo < p[0] < hi
    return False


def _build_centers(
    count: int,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    candidates = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if _center_clear((r, c), h, w, forbidden)
    ]
    if not candidates:
        return None

    for _ in range(PATH_RETRIES):
        centers = [rng.choice(candidates)]
        new_dirs: list[tuple[int, int]] = []

        while len(centers) < count:
            cur = centers[-1]
            options = []
            dirs = DIRS[:]
            rng.shuffle(dirs)
            lengths = list(LINK_LENGTHS)
            rng.shuffle(lengths)
            for direction in dirs:
                dr, dc = direction
                for dist in lengths:
                    nxt = (cur[0] + dist * dr, cur[1] + dist * dc)
                    if not _center_clear(nxt, h, w, forbidden):
                        continue
                    if any(max(abs(nxt[0] - r), abs(nxt[1] - c)) < 4 for r, c in centers):
                        continue
                    if any(_between(cur, nxt, prev) for prev in centers):
                        continue
                    options.append((direction, nxt))
            if not options:
                break
            direction, nxt = rng.choice(options)
            new_dirs.append(direction)
            centers.append(nxt)

        if len(centers) != count:
            continue

        tail_options = []
        for direction in rng.sample(DIRS, len(DIRS)):
            dr, dc = direction
            r, c = centers[-1]
            if not any(
                (other[0] - r) * dc == (other[1] - c) * dr
                and (other[0] - r) * dr + (other[1] - c) * dc > 0
                for other in centers[:-1]
            ):
                tail_options.append(direction)
        if not tail_options:
            continue
        new_dirs.append(rng.choice(tail_options))
        return centers, new_dirs

    return None


def _render_expected(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        for r, c in _arrow_cells(spec["center"], spec["new_dir"]):
            assert out[r, c] == bg
            out[r, c] = int(spec["color"])
    return out


def _render_input(
    h: int,
    w: int,
    bg: int,
    legend_corner: str,
    specs: list[dict],
) -> np.ndarray | None:
    legend_cells, forbidden = _placement_forbidden(legend_corner, h, w)
    grid = np.full((h, w), bg, dtype=int)
    r0, c0 = _legend_origin(legend_corner, h, w)
    grid[r0, c0] = INSTR_COLORS["cw"]
    grid[r0, c0 + 1] = INSTR_COLORS["ccw"]
    grid[r0 + 1, c0] = INSTR_COLORS["hflip"]
    grid[r0 + 1, c0 + 1] = INSTR_COLORS["vflip"]

    occupied = set(legend_cells)
    cell_to_id: dict[tuple[int, int], int] = {}
    for spec in specs:
        arrow_cells = _arrow_cells(spec["center"], spec["orig_dir"])
        if any(not (0 <= r < h and 0 <= c < w) for r, c in arrow_cells):
            return None
        if arrow_cells & forbidden or arrow_cells & occupied:
            return None
        for cell in arrow_cells:
            occupied.add(cell)
            cell_to_id[cell] = int(spec["id"])
            grid[cell] = int(spec["color"])

    for spec in specs:
        r, c = spec["center"]
        dr, dc = spec["orig_dir"]
        instr = (r + 2 * dr, c + 2 * dc)
        if instr in occupied or instr in forbidden:
            return None
        if not (0 <= instr[0] < h and 0 <= instr[1] < w):
            return None
        occupied.add(instr)
        grid[instr] = INSTR_COLORS[spec["op"]]

    targets = []
    for spec in specs:
        r, c = spec["center"]
        dr, dc = spec["new_dir"]
        rr = r + 2 * dr
        cc = c + 2 * dc
        target = None
        while 0 <= rr < h and 0 <= cc < w:
            other = cell_to_id.get((rr, cc))
            if other is not None and other != spec["id"]:
                target = other
                break
            rr += dr
            cc += dc
        targets.append(target)
    if targets != list(range(1, len(specs))) + [None]:
        return None

    expected = _render_expected((h, w), bg, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    count = rng.randint(ARROW_COUNT_RANGE[0], ARROW_COUNT_RANGE[1])
    legend_corner = rng.choice(["tl", "tr", "bl", "br"])
    bg = rng.choice(NON_INSTR_COLORS)
    arrow_colors = [color for color in NON_INSTR_COLORS if color != bg]
    rng.shuffle(arrow_colors)
    arrow_colors = arrow_colors[:count]

    _, forbidden = _placement_forbidden(legend_corner, h, w)
    path = _build_centers(count, h, w, forbidden, rng)
    if path is None:
        return None
    centers, new_dirs = path
    ops = _choose_ops(count, rng)

    specs = []
    for idx in range(count):
        new_dir = new_dirs[idx]
        op = ops[idx]
        specs.append(
            {
                "id": idx,
                "color": arrow_colors[idx],
                "center": centers[idx],
                "new_dir": new_dir,
                "op": op,
                "orig_dir": _invert_op(new_dir, op),
            }
        )

    return _render_input(h, w, bg, legend_corner, specs)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
