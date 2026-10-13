from __future__ import annotations

from itertools import combinations
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r0 for r0, _ in cells]
            cols = [c0 for _, c0 in cells]
            comps.append(
                {
                    "color": color,
                    "cells": frozenset(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(cells),
                }
            )
    return comps


def _filled_rect_bbox(cells: frozenset[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _shift_cells(cells: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(r + dr), int(c + dc)) for r, c in cells)


def _horizontal_candidate(comp_a: dict, idx_a: int, comp_b: dict, idx_b: int, width: int) -> dict | None:
    if int(comp_a["color"]) == int(comp_b["color"]):
        return None

    ar0, ac0, ar1, ac1 = map(int, comp_a["bbox"])
    br0, bc0, br1, bc1 = map(int, comp_b["bbox"])
    if not _intervals_overlap(ar0, ar1, br0, br1):
        return None

    if ac1 < bc0:
        left, left_idx = comp_a, int(idx_a)
        right, right_idx = comp_b, int(idx_b)
    elif bc1 < ac0:
        left, left_idx = comp_b, int(idx_b)
        right, right_idx = comp_a, int(idx_a)
    else:
        return None

    max_d = min(int(width - 1 - left["bbox"][3]), int(right["bbox"][1]))
    for d in range(1, max_d + 1):
        left_cells = _shift_cells(left["cells"], 0, d)
        right_cells = _shift_cells(right["cells"], 0, -d)
        if left_cells & right_cells:
            continue
        union = left_cells | right_cells
        rect = _filled_rect_bbox(union)
        if rect is None:
            continue
        _r0, c0, _r1, c1 = rect
        if not any(c == c0 for _r, c in left_cells):
            continue
        if not any(c == c1 for _r, c in right_cells):
            continue
        return {
            "pair": (min(left_idx, right_idx), max(left_idx, right_idx)),
            "placements": {left_idx: left_cells, right_idx: right_cells},
            "final_cells": union,
        }
    return None


def _vertical_candidate(comp_a: dict, idx_a: int, comp_b: dict, idx_b: int, height: int) -> dict | None:
    if int(comp_a["color"]) == int(comp_b["color"]):
        return None

    ar0, ac0, ar1, ac1 = map(int, comp_a["bbox"])
    br0, bc0, br1, bc1 = map(int, comp_b["bbox"])
    if not _intervals_overlap(ac0, ac1, bc0, bc1):
        return None

    if ar1 < br0:
        top, top_idx = comp_a, int(idx_a)
        bottom, bottom_idx = comp_b, int(idx_b)
    elif br1 < ar0:
        top, top_idx = comp_b, int(idx_b)
        bottom, bottom_idx = comp_a, int(idx_a)
    else:
        return None

    max_d = min(int(height - 1 - top["bbox"][2]), int(bottom["bbox"][0]))
    for d in range(1, max_d + 1):
        top_cells = _shift_cells(top["cells"], d, 0)
        bottom_cells = _shift_cells(bottom["cells"], -d, 0)
        if top_cells & bottom_cells:
            continue
        union = top_cells | bottom_cells
        rect = _filled_rect_bbox(union)
        if rect is None:
            continue
        r0, _c0, r1, _c1 = rect
        if not any(r == r0 for r, _c in top_cells):
            continue
        if not any(r == r1 for r, _c in bottom_cells):
            continue
        return {
            "pair": (min(top_idx, bottom_idx), max(top_idx, bottom_idx)),
            "placements": {top_idx: top_cells, bottom_idx: bottom_cells},
            "final_cells": union,
        }
    return None


def _render_solution(grid: np.ndarray, bg: int, comps: list[dict], candidates: list[dict], chosen: list[int]) -> np.ndarray:
    out = np.full_like(grid, int(bg))
    for cand_idx in chosen:
        cand = candidates[int(cand_idx)]
        for comp_idx, cells in cand["placements"].items():
            color = int(comps[int(comp_idx)]["color"])
            for r, c in cells:
                out[int(r), int(c)] = color
    return out


def _solve_matchings(grid: np.ndarray, bg: int, comps: list[dict], candidates: list[dict]) -> list[np.ndarray]:
    n = len(comps)
    by_comp: list[list[int]] = [[] for _ in range(n)]
    for idx, cand in enumerate(candidates):
        i, j = cand["pair"]
        by_comp[int(i)].append(int(idx))
        by_comp[int(j)].append(int(idx))

    assert all(by_comp[i] for i in range(n)), "every fragment must have at least one compatible partner"

    solutions: list[list[int]] = []

    def dfs(unmatched: tuple[int, ...], occupied: frozenset[tuple[int, int]], chosen: list[int]) -> None:
        if not unmatched:
            solutions.append(chosen.copy())
            return
        i = int(unmatched[0])
        for cand_idx in by_comp[i]:
            cand = candidates[int(cand_idx)]
            a, b = cand["pair"]
            j = int(b if a == i else a)
            if j not in unmatched:
                continue
            if cand["final_cells"] & occupied:
                continue
            next_unmatched = tuple(x for x in unmatched if x not in (i, j))
            chosen.append(int(cand_idx))
            dfs(next_unmatched, occupied | cand["final_cells"], chosen)
            chosen.pop()

    dfs(tuple(range(n)), frozenset(), [])
    assert solutions, "failed to pair all fragments"

    rendered: list[np.ndarray] = []
    for chosen in solutions:
        out = _render_solution(grid, bg, comps, candidates, chosen)
        if not any(np.array_equal(out, prev) for prev in rendered):
            rendered.append(out)
    assert len(rendered) == 1, "pairing is ambiguous"
    return rendered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, slide each pair toward its midpoint, and merge into a full rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    comps = _find_components(g, bg)
    assert comps and len(comps) % 2 == 0, "expected an even number of foreground fragments"

    h, w = g.shape
    candidates: list[dict] = []
    for i, j in combinations(range(len(comps)), 2):
        horiz = _horizontal_candidate(comps[i], i, comps[j], j, w)
        if horiz is not None:
            candidates.append(horiz)
        vert = _vertical_candidate(comps[i], i, comps[j], j, h)
        if vert is not None:
            candidates.append(vert)

    outputs = _solve_matchings(g, bg, comps, candidates)
    return outputs[0]


# --------------- generator ---------------

BG = 5
FG_COLORS = tuple(c for c in range(10) if c != BG)
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
PAIR_COUNT_CHOICES = (1, 2, 3)
PAIR_COUNT_WEIGHTS = (3, 5, 2)
H_RECT_H_RANGE = (2, 5)
H_RECT_W_RANGE = (4, 8)
V_RECT_H_RANGE = (4, 8)
V_RECT_W_RANGE = (2, 5)
SHIFT_RANGE = (2, 4)
PAIR_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400


def _profile(length: int, span: int) -> list[int]:
    assert length >= 1 and span >= 2
    vals = [random.randint(1, span - 1)]
    for _ in range(length - 1):
        step = random.choice((-2, -1, 0, 1, 2))
        nxt = min(span - 1, max(1, vals[-1] + step))
        vals.append(int(nxt))
    if length > 1 and span > 2 and len(set(vals)) == 1:
        idx = random.randrange(length)
        vals[idx] = vals[idx] + 1 if vals[idx] < span - 1 else vals[idx] - 1
    return vals


def _pair_masks(axis: str, rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray]:
    if axis == "h":
        left = np.zeros((rect_h, rect_w), dtype=bool)
        right = np.zeros((rect_h, rect_w), dtype=bool)
        for r, cut in enumerate(_profile(rect_h, rect_w)):
            left[int(r), : int(cut)] = True
            right[int(r), int(cut) :] = True
        return left, right

    top = np.zeros((rect_h, rect_w), dtype=bool)
    bottom = np.zeros((rect_h, rect_w), dtype=bool)
    for c, cut in enumerate(_profile(rect_w, rect_h)):
        top[: int(cut), int(c)] = True
        bottom[int(cut) :, int(c)] = True
    return top, bottom


def _sample_pair_spec(color_a: int, color_b: int) -> dict:
    axis = random.choice(("h", "v"))
    if axis == "h":
        rect_h = random.randint(*H_RECT_H_RANGE)
        rect_w = random.randint(*H_RECT_W_RANGE)
    else:
        rect_h = random.randint(*V_RECT_H_RANGE)
        rect_w = random.randint(*V_RECT_W_RANGE)
    shift = random.randint(*SHIFT_RANGE)
    mask_a, mask_b = _pair_masks(axis, rect_h, rect_w)
    return {
        "axis": axis,
        "rect_h": rect_h,
        "rect_w": rect_w,
        "shift": shift,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "colors": (int(color_a), int(color_b)),
    }


def _envelope_bbox(spec: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    rect_h = int(spec["rect_h"])
    rect_w = int(spec["rect_w"])
    shift = int(spec["shift"])
    if spec["axis"] == "h":
        return int(r0), int(c0 - shift), int(r0 + rect_h - 1), int(c0 + rect_w - 1 + shift)
    return int(r0 - shift), int(c0), int(r0 + rect_h - 1 + shift), int(c0 + rect_w - 1)


def _expand_bbox(box: tuple[int, int, int, int], pad: int, height: int, width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, int(r0 - pad)),
        max(0, int(c0 - pad)),
        min(int(height - 1), int(r1 + pad)),
        min(int(width - 1), int(c1 + pad)),
    )


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (int(a[2]) < int(b[0]) or int(b[2]) < int(a[0]) or int(a[3]) < int(b[1]) or int(b[3]) < int(a[1]))


def _placement_options(
    spec: dict,
    height: int,
    width: int,
    blocked: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    opts: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for r0 in range(height - int(spec["rect_h"]) + 1):
        for c0 in range(width - int(spec["rect_w"]) + 1):
            env = _envelope_bbox(spec, r0, c0)
            if env[0] < 0 or env[1] < 0 or env[2] >= height or env[3] >= width:
                continue
            padded = _expand_bbox(env, PAIR_PADDING, height, width)
            if any(_boxes_intersect(padded, other) for other in blocked):
                continue
            opts.append((int(r0), int(c0), padded))
    random.shuffle(opts)
    return opts


def _cells_from_mask(mask: np.ndarray, r0: int, c0: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask))


def _place_pair(spec: dict, r0: int, c0: int) -> dict:
    out_a = _cells_from_mask(spec["mask_a"], r0, c0)
    out_b = _cells_from_mask(spec["mask_b"], r0, c0)
    shift = int(spec["shift"])
    if spec["axis"] == "h":
        in_a = _shift_cells(out_a, 0, -shift)
        in_b = _shift_cells(out_b, 0, shift)
    else:
        in_a = _shift_cells(out_a, -shift, 0)
        in_b = _shift_cells(out_b, shift, 0)
    return {
        "colors": spec["colors"],
        "input_a": in_a,
        "input_b": in_b,
        "output_a": out_a,
        "output_b": out_b,
    }


def _strict_bg_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(v): int(c) for v, c in zip(vals, counts)}
    bg_count = count_by_color.get(int(bg), 0)
    return all(int(c) < bg_count for color, c in count_by_color.items() if int(color) != int(bg))


def _paint_cells(grid: np.ndarray, cells: frozenset[tuple[int, int]], color: int) -> bool:
    for r, c in cells:
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        grid[r, c] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    colors = random.sample(FG_COLORS, 2 * pair_count)

    specs = [
        _sample_pair_spec(colors[2 * i], colors[2 * i + 1])
        for i in range(pair_count)
    ]
    specs.sort(
        key=lambda spec: (
            (int(spec["rect_h"]) + (2 * int(spec["shift"]) if spec["axis"] == "v" else 0))
            * (int(spec["rect_w"]) + (2 * int(spec["shift"]) if spec["axis"] == "h" else 0))
        ),
        reverse=True,
    )

    blocked: list[tuple[int, int, int, int]] = []
    placements: list[dict] = []
    for spec in specs:
        options = _placement_options(spec, height, width, blocked)
        if not options:
            return None
        r0, c0, padded = random.choice(options)
        blocked.append(padded)
        placements.append(_place_pair(spec, r0, c0))

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for placed in placements:
        color_a, color_b = placed["colors"]
        if not _paint_cells(inp, placed["input_a"], color_a):
            return None
        if not _paint_cells(inp, placed["input_b"], color_b):
            return None
        if not _paint_cells(out, placed["output_a"], color_a):
            return None
        if not _paint_cells(out, placed["output_b"], color_b):
            return None

    if not _strict_bg_dominant(inp, BG):
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
