from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4 = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ROTATE_DIR = {
    "up": "left",
    "left": "down",
    "down": "right",
    "right": "up",
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[dict] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox_from_cells(cells)})
    return comps


def _central_block_candidates(length: int, gap_len: int) -> list[list[int]]:
    slack = length - gap_len
    starts = {slack // 2, (slack + 1) // 2}
    return [list(range(start, start + gap_len)) for start in sorted(starts)]


def _central_block(length: int, gap_len: int) -> list[int]:
    return _central_block_candidates(length, gap_len)[0]


def _valid_gap(indices: list[int], edge_len: int) -> bool:
    if len(indices) not in (1, 2):
        return False
    if indices != list(range(indices[0], indices[0] + len(indices))):
        return False
    if indices[0] == 0 or indices[-1] == edge_len - 1:
        return False
    return indices in _central_block_candidates(edge_len, len(indices))


def _parse_cup(grid: np.ndarray, comp: dict) -> dict:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 3, "cup bbox must be at least 3x3"

    top_missing = [c for c in range(w) if int(patch[0, c]) != color]
    bottom_missing = [c for c in range(w) if int(patch[h - 1, c]) != color]
    left_missing = [r for r in range(h) if int(patch[r, 0]) != color]
    right_missing = [r for r in range(h) if int(patch[r, w - 1]) != color]

    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    if _valid_gap(top_missing, w):
        candidates.append(("up", [(0, c) for c in top_missing]))
    if _valid_gap(bottom_missing, w):
        candidates.append(("down", [(h - 1, c) for c in bottom_missing]))
    if _valid_gap(left_missing, h):
        candidates.append(("left", [(r, 0) for r in left_missing]))
    if _valid_gap(right_missing, h):
        candidates.append(("right", [(r, w - 1) for r in right_missing]))

    assert len(candidates) == 1, "each component must have exactly one central opening"
    opening, rel_gap = candidates[0]
    gap_rel = set(rel_gap)

    for rr in range(h):
        for cc in range(w):
            on_boundary = rr in (0, h - 1) or cc in (0, w - 1)
            val = int(patch[rr, cc])
            if on_boundary:
                if (rr, cc) in gap_rel:
                    assert val == 0, "opening cells must be black in the input"
                else:
                    assert val == color, "all non-opening boundary cells must match cup color"
            else:
                assert val == 0, "cup interior must be black in the input"

    body = {(r, c) for r, c in comp["cells"]}
    gap = {(r0 + rr, c0 + cc) for rr, cc in gap_rel}
    fill = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    assert fill, "cup must enclose at least one interior cell"
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "gap_cells": tuple(sorted(gap)),
        "body_cells": body,
        "fill_cells": fill,
    }


def _solve_from_cups(shape: tuple[int, int], cups: list[dict]) -> tuple[np.ndarray, dict]:
    out = np.zeros(shape, dtype=int)
    solid_cells: set[tuple[int, int]] = set()
    for cup in cups:
        color = int(cup["color"])
        for cell in cup["body_cells"] | cup["fill_cells"]:
            out[cell] = color
            solid_cells.add(cell)

    candidates: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}
    for idx, cup in enumerate(cups):
        dr, dc = DIRS4[cup["opening"]]
        r0, c0, _r1, _c1 = cup["bbox"]
        for gr, gc in cup["gap_cells"]:
            rr, cc = gr, gc
            dist = 0
            while _in_bounds(shape, rr, cc):
                candidates.setdefault((rr, cc), []).append(
                    (dist, r0, c0, idx, int(cup["color"]))
                )
                rr += dr
                cc += dc
                dist += 1

    winner_map: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    collision_cells: list[tuple[int, int]] = []
    tie_cells: list[tuple[int, int]] = []
    strict_cells: list[tuple[int, int]] = []
    for cell, items in candidates.items():
        uniq = {}
        for item in items:
            uniq[item[3]] = min(item, uniq.get(item[3], item))
        values = list(uniq.values())
        if len(values) > 1:
            collision_cells.append(cell)
        winner = min(values)
        winner_map[cell] = winner
        if len(values) > 1:
            best_dist = winner[0]
            if sum(1 for value in values if value[0] == best_dist) > 1:
                tie_cells.append(cell)
            else:
                strict_cells.append(cell)

    ray_cells: set[tuple[int, int]] = set()
    for cell, winner in winner_map.items():
        out[cell] = winner[4]
        ray_cells.add(cell)

    pre_clean = out.copy()
    cleaned: list[tuple[int, int]] = []
    for r, c in sorted(ray_cells):
        color = int(pre_clean[r, c])
        same_neighbors = 0
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc) and int(pre_clean[nr, nc]) == color:
                same_neighbors += 1
        if same_neighbors < 1:
            cleaned.append((r, c))

    for r, c in cleaned:
        out[r, c] = 0

    return out, {
        "winner_map": winner_map,
        "collision_cells": collision_cells,
        "tie_cells": tie_cells,
        "strict_cells": strict_cells,
        "ray_cells": ray_cells,
        "cleaned_cells": cleaned,
        "solid_cells": solid_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each open cup, extend rays from every opening cell, resolve multi-source collisions by source distance, then erase unsupported ray singletons.

    Input structure:
    - The grid has black background and 2 or 3 monochromatic cup outlines of different colors.
    - Each cup is a thin rectangular frame with exactly one central gap of length 1 or 2 on one side.
    - The cup interiors are black, and at least one horizontal-opening ray and one vertical-opening ray can intersect.

    Rule:
    - Detect each cup's opening side and fill all black interior cells inside its bounding box with the cup color.
    - From every gap cell, extend a straight same-color ray outward in the opening direction up to the grid edge.
    - If different cups cover the same ray cell, keep the cup whose opening cell is closer in Manhattan distance; if tied, keep the cup whose bbox top-left is lexicographically smaller.
    - After all rays are resolved, erase any ray cell whose final same-color 4-neighbor count is zero, without iterating.

    Key insight:
    - The output depends on three linked stages: cup interiors identify the sources, source-to-cell distance resolves crossed rays, and the collision-truncated ray leftovers are then pruned by one-pass adjacency support.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    comps = _components_by_color(g)
    assert 2 <= len(comps) <= 3, "expected 2 or 3 cup components"
    colors = [int(comp["color"]) for comp in comps]
    assert len(colors) == len(set(colors)), "cups must use distinct colors"
    cups = [_parse_cup(g, comp) for comp in comps]
    out, _stats = _solve_from_cups(g.shape, cups)
    return out


def _make_cup_meta(
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    opening: str,
    gap_len: int,
    gap_variant: int = 0,
) -> dict:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if opening in ("up", "down"):
        gap_cols = _central_block_candidates(width, gap_len)[gap_variant]
        row = 0 if opening == "up" else height - 1
        for col in gap_cols:
            patch[row, col] = 0
    else:
        gap_rows = _central_block_candidates(height, gap_len)[gap_variant]
        col = 0 if opening == "left" else width - 1
        for row in gap_rows:
            patch[row, col] = 0

    body = set()
    gap = set()
    for rr in range(height):
        for cc in range(width):
            cell = (top + rr, left + cc)
            if int(patch[rr, cc]) == color:
                body.add(cell)
            elif rr in (0, height - 1) or cc in (0, width - 1):
                gap.add(cell)

    fill = {(r, c) for r in range(top + 1, top + height - 1) for c in range(left + 1, left + width - 1)}
    return {
        "color": color,
        "bbox": (top, left, top + height - 1, left + width - 1),
        "opening": opening,
        "gap_cells": tuple(sorted(gap)),
        "body_cells": body,
        "fill_cells": fill,
    }


def _draw_input(shape: tuple[int, int], cups: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for cup in cups:
        for r, c in cup["body_cells"]:
            grid[r, c] = int(cup["color"])
    return grid


def _rotate_coord(coord: tuple[int, int], shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return w - 1 - c, r
    if k == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _rotate_dir(name: str, k: int) -> str:
    out = name
    for _ in range(k % 4):
        out = ROTATE_DIR[out]
    return out


def _rotate_cups(cups: list[dict], shape: tuple[int, int], k: int) -> tuple[list[dict], tuple[int, int]]:
    new_shape = np.rot90(np.zeros(shape, dtype=int), k).shape
    rotated = []
    for cup in cups:
        body = {_rotate_coord(cell, shape, k) for cell in cup["body_cells"]}
        fill = {_rotate_coord(cell, shape, k) for cell in cup["fill_cells"]}
        gap = {_rotate_coord(cell, shape, k) for cell in cup["gap_cells"]}
        rotated.append(
            {
                "color": int(cup["color"]),
                "bbox": _bbox_from_cells(body),
                "opening": _rotate_dir(cup["opening"], k),
                "gap_cells": tuple(sorted(gap)),
                "body_cells": body,
                "fill_cells": fill,
            }
        )
    return rotated, new_shape


def _far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _add_optional_cup(shape: tuple[int, int], cups: list[dict], color: int) -> None:
    h = random.choice([3, 4, 5])
    w = random.choice([4, 5])
    gap_len = 2 if (h >= 4 and random.random() < 0.75) else 1
    opening = "up"
    for top in range(1, 4):
        for left in range(1, 5):
            cup = _make_cup_meta(top, left, h, w, color, opening, gap_len)
            if all(_far_enough(cup["bbox"], other["bbox"], pad=1) for other in cups):
                cups.append(cup)
                return


def _build_blueprint(forced_mode: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(100):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        shape = (h, w)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.choice([2, 3]))
        mode = forced_mode or random.choices(
            ["vertical_near", "horizontal_near", "tie"],
            weights=[4, 4, 3],
            k=1,
        )[0]

        a_h = 3 if mode == "horizontal_near" else random.choice([3, 5])
        a_w = random.choice([3, 4, 5])
        a_gap = 1
        b_h = random.choice([3, 4, 5])
        b_gap = 2 if random.random() < 0.45 else 1
        b_w_choices = [4, 5] if b_gap == 2 else [3, 4, 5]
        b_w = random.choice(b_w_choices)

        if mode in ("vertical_near", "tie"):
            i_r = random.randint(7, h - 6)
            i_c = w - 2
            max_d_a = min(8, i_c - a_w)
            max_d_b = i_r - b_h
            if max_d_a < 4 or max_d_b < 1:
                continue
            d_a = random.randint(4, max_d_a)
            if mode == "tie":
                if d_a > max_d_b:
                    continue
                d_b = d_a
            else:
                d_b_high = min(d_a - 1, max_d_b)
                if d_b_high < 1:
                    continue
                d_b = random.randint(1, d_b_high)
        else:
            i_r = h - 2
            i_c = random.randint(7, w - 6)
            max_d_a = min(5, i_c - a_w)
            max_d_b = min(8, i_r - b_h)
            if max_d_a < 2 or max_d_b < 3:
                continue
            d_a = random.randint(2, max_d_a)
            if max_d_b < d_a + 1:
                continue
            d_b = random.randint(d_a + 1, max_d_b)

        a_top = i_r - (a_h // 2)
        a_left = i_c - d_a - a_w + 1
        b_gap_variant = len(_central_block_candidates(b_w, b_gap)) - 1
        gap_right = _central_block_candidates(b_w, b_gap)[b_gap_variant][-1]
        b_left = i_c - gap_right
        b_top = i_r - d_b - b_h + 1

        if a_top < 1 or a_left < 1:
            continue
        if b_top < 1 or b_left < 1 or b_left + b_w > w:
            continue

        cup_a = _make_cup_meta(a_top, a_left, a_h, a_w, colors[0], "right", a_gap)
        cup_b = _make_cup_meta(
            b_top,
            b_left,
            b_h,
            b_w,
            colors[1],
            "down",
            b_gap,
            gap_variant=b_gap_variant,
        )
        if not _far_enough(cup_a["bbox"], cup_b["bbox"], pad=1):
            continue

        cups = [cup_a, cup_b]
        if len(colors) == 3:
            _add_optional_cup(shape, cups, colors[2])

        base_inp = _draw_input(shape, cups)
        rotate_k = random.randint(0, 3)
        rot_inp = np.rot90(base_inp, rotate_k).copy()
        rot_cups, rot_shape = _rotate_cups(cups, shape, rotate_k)
        expected, stats = _solve_from_cups(rot_shape, rot_cups)
        if not stats["collision_cells"] or not stats["cleaned_cells"]:
            continue
        if mode == "tie" and not stats["tie_cells"]:
            continue
        if mode != "tie" and not stats["strict_cells"]:
            continue
        meta = {"mode": mode, "cups": rot_cups, "stats": stats}
        return rot_inp, expected, meta
    raise RuntimeError("failed to build blueprint")


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    stats = meta["stats"]
    if not stats["collision_cells"]:
        return False
    if meta["mode"] == "tie":
        if not stats["tie_cells"]:
            return False
    else:
        if not stats["strict_cells"]:
            return False
    if not stats["cleaned_cells"]:
        return False

    parsed = [_parse_cup(inp, comp) for comp in _components_by_color(inp)]
    if len(parsed) != len(meta["cups"]):
        return False
    parsed_out, parsed_stats = _solve_from_cups(inp.shape, parsed)
    if not np.array_equal(parsed_out, expected):
        return False
    if set(parsed_stats["collision_cells"]) != set(stats["collision_cells"]):
        return False
    if set(parsed_stats["cleaned_cells"]) != set(stats["cleaned_cells"]):
        return False

    pre_clean = expected.copy()
    for r, c in stats["cleaned_cells"]:
        pre_clean[r, c] = next(
            winner[4] for cell, winner in stats["winner_map"].items() if cell == (r, c)
        )
    if np.array_equal(pre_clean, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        mode = random.choices(
            ["vertical_near", "horizontal_near", "tie"],
            weights=[4, 4, 3],
            k=1,
        )[0]
        inp, expected, meta = _build_blueprint(mode)
        if _validate_semantics(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid ray-collision cup puzzle")
