from __future__ import annotations

from collections import deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            out.append(
                {
                    "color": color,
                    "cells": cells,
                    "cell_set": set(cells),
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return out


def _is_marker(comp: dict[str, object]) -> bool:
    if int(comp["area"]) != 2:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    return (r1 - r0 + 1, c1 - c0 + 1) in {(1, 2), (2, 1)}


def _marker_orientation(comp: dict[str, object]) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    return "h" if (r1 - r0 + 1, c1 - c0 + 1) == (1, 2) else "v"


def _touches(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> bool:
    for r, c in cells_a:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in cells_b:
                return True
    return False


def _pair_info(
    grid: np.ndarray,
    comp_a: dict[str, object],
    comp_b: dict[str, object],
    orient: str,
    bg: int,
) -> dict[str, object] | None:
    if int(comp_a["color"]) != int(comp_b["color"]):
        return None
    if orient == "v":
        left, right = sorted((comp_a, comp_b), key=lambda comp: comp["bbox"][1])
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        if (lr0, lr1) != (rr0, rr1):
            return None
        crack_col = lc1 + 1
        if rc0 != crack_col + 1:
            return None
        rows_left = {r for r, _ in left["cells"]}
        rows_right = {r for r, _ in right["cells"]}
        rows = []
        for r in range(lr0, lr1 + 1):
            if int(grid[r, crack_col]) != bg:
                return None
            if r not in rows_left or r not in rows_right:
                return None
            rows.append(r)
        return {
            "orient": "v",
            "first": left,
            "second": right,
            "crack_col": crack_col,
            "span": rows,
            "bbox": (lr0, lc0, rr1, rc1),
        }

    top, bottom = sorted((comp_a, comp_b), key=lambda comp: comp["bbox"][0])
    tr0, tc0, tr1, tc1 = top["bbox"]
    br0, bc0, br1, bc1 = bottom["bbox"]
    if (tc0, tc1) != (bc0, bc1):
        return None
    crack_row = tr1 + 1
    if br0 != crack_row + 1:
        return None
    cols_top = {c for _, c in top["cells"]}
    cols_bottom = {c for _, c in bottom["cells"]}
    cols = []
    for c in range(tc0, tc1 + 1):
        if int(grid[crack_row, c]) != bg:
            return None
        if c not in cols_top or c not in cols_bottom:
            return None
        cols.append(c)
    return {
        "orient": "h",
        "first": top,
        "second": bottom,
        "crack_row": crack_row,
        "span": cols,
        "bbox": (tr0, tc0, br1, bc1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each cracked solid shape and widen the crack into a colored gap.

    Input structure:
    - bg is the most common color.
    - Each object is a large monochrome shape visually split into two same-colored halves by a 1-cell straight bg crack.
    - A separate 2-cell marker sits just outside the shape; the marker is perpendicular to the crack and gives the gap-fill color.

    Rule:
    - Detect every pair of same-colored large components whose shared bbox contains a single straight bg crack, either vertical or horizontal.
    - Use the nearby 2-cell marker to choose which pair belongs together and to read the color that will fill the widened crack.
    - Translate the two halves one cell away from the crack along the crack normal, turning the original 1-cell crack into a 3-cell gap.
    - Remove the marker and fill the 3-cell gap with the marker color.

    Key insight:
    - The real object is not either half alone; it is the pair of same-colored halves plus the internal straight bg slit, while the tiny external domino supplies both the crack orientation cue and the gap color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    comps = _components(g, bg)
    markers = [comp for comp in comps if _is_marker(comp)]
    large = [comp for comp in comps if not _is_marker(comp)]

    ops: list[dict[str, object]] = []
    used_large_ids: set[int] = set()

    for marker in markers:
        expected_orient = "v" if _marker_orientation(marker) == "h" else "h"
        marker_set = marker["cell_set"]
        candidates: list[tuple[int, int, dict[str, object]]] = []
        for i in range(len(large)):
            if i in used_large_ids:
                continue
            for j in range(i + 1, len(large)):
                if j in used_large_ids:
                    continue
                info = _pair_info(g, large[i], large[j], expected_orient, bg)
                if info is None:
                    continue
                union_cells = set(info["first"]["cells"]) | set(info["second"]["cells"])
                if not _touches(marker_set, union_cells):
                    continue
                r0, c0, r1, c1 = info["bbox"]
                score = (r1 - r0 + 1) * (c1 - c0 + 1)
                candidates.append((score, i, j, info))
        if not candidates:
            continue
        _score, i, j, info = max(candidates, key=lambda item: item[0])
        used_large_ids.add(i)
        used_large_ids.add(j)
        ops.append(
            {
                "marker_cells": marker["cells"],
                "fill_color": int(marker["color"]),
                "info": info,
            }
        )

    out = g.copy()
    for op in ops:
        info = op["info"]
        for r, c in op["marker_cells"]:
            out[r, c] = bg
        for comp in (info["first"], info["second"]):
            for r, c in comp["cells"]:
                out[r, c] = bg

    for op in ops:
        info = op["info"]
        fill_color = int(op["fill_color"])
        if info["orient"] == "v":
            for r, c in info["first"]["cells"]:
                nr, nc = r, c - 1
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(info["first"]["color"])
            for r, c in info["second"]["cells"]:
                nr, nc = r, c + 1
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(info["second"]["color"])
            crack_col = int(info["crack_col"])
            for r in info["span"]:
                for c in (crack_col - 1, crack_col, crack_col + 1):
                    if 0 <= r < h and 0 <= c < w:
                        out[r, c] = fill_color
        else:
            for r, c in info["first"]["cells"]:
                nr, nc = r - 1, c
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(info["first"]["color"])
            for r, c in info["second"]["cells"]:
                nr, nc = r + 1, c
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(info["second"]["color"])
            crack_row = int(info["crack_row"])
            for c in info["span"]:
                for r in (crack_row - 1, crack_row, crack_row + 1):
                    if 0 <= r < h and 0 <= c < w:
                        out[r, c] = fill_color

    return out


def _random_profile(length: int, max_depth: int, rng: random.Random) -> list[int]:
    vals = [rng.randint(2, max_depth) for _ in range(length)]
    vals[rng.randrange(length)] = max_depth
    if max_depth > 2 and rng.random() < 0.7:
        vals[rng.randrange(length)] = 2
    return vals


def _marker_candidates(
    shape_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    orient: str,
    patch_shape: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    h, w = patch_shape
    r0, c0, r1, c1 = bbox
    candidates: list[list[tuple[int, int]]] = []

    def add(cells: list[tuple[int, int]]) -> None:
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            return
        if any((r, c) in shape_cells for r, c in cells):
            return
        if _touches(set(cells), shape_cells):
            candidates.append(cells)

    if orient == "h":
        for c in range(c0 - 1, c1 + 1):
            add([(r0 - 1, c), (r0 - 1, c + 1)])
            add([(r1 + 1, c), (r1 + 1, c + 1)])
        for r in range(r0, r1 + 1):
            add([(r, c0 - 2), (r, c0 - 1)])
            add([(r, c1 + 1), (r, c1 + 2)])
    else:
        for r in range(r0 - 1, r1 + 1):
            add([(r, c0 - 1), (r + 1, c0 - 1)])
            add([(r, c1 + 1), (r + 1, c1 + 1)])
        for c in range(c0, c1 + 1):
            add([(r0 - 2, c), (r0 - 1, c)])
            add([(r1 + 1, c), (r1 + 2, c)])

    uniq: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for cells in candidates:
        key = tuple(sorted(cells))
        if key not in seen:
            seen.add(key)
            uniq.append(cells)
    return uniq


def _build_vertical_object(shape_color: int, fill_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    height = rng.randint(6, 10)
    left_max = rng.randint(2, 4)
    right_max = rng.randint(2, 4)
    bbox_w = left_max + 1 + right_max
    patch_h, patch_w = height + 4, bbox_w + 4
    bbox_r0, bbox_c0 = 2, 2
    crack_col = bbox_c0 + left_max
    left_widths = _random_profile(height, left_max, rng)
    right_widths = _random_profile(height, right_max, rng)

    left_cells: set[tuple[int, int]] = set()
    right_cells: set[tuple[int, int]] = set()
    for rel_r in range(height):
        r = bbox_r0 + rel_r
        for c in range(crack_col - left_widths[rel_r], crack_col):
            left_cells.add((r, c))
        for c in range(crack_col + 1, crack_col + 1 + right_widths[rel_r]):
            right_cells.add((r, c))

    shape_cells = left_cells | right_cells
    bbox = (bbox_r0, bbox_c0, bbox_r0 + height - 1, bbox_c0 + bbox_w - 1)
    marker = rng.choice(_marker_candidates(shape_cells, bbox, "h", (patch_h, patch_w)))

    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)
    for r, c in left_cells | right_cells:
        inp[r, c] = shape_color
    for r, c in marker:
        inp[r, c] = fill_color
    for r, c in left_cells:
        out[r, c - 1] = shape_color
    for r, c in right_cells:
        out[r, c + 1] = shape_color
    for r in range(bbox_r0, bbox_r0 + height):
        out[r, crack_col - 1 : crack_col + 2] = fill_color
    return inp, out


def _build_horizontal_object(shape_color: int, fill_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    width = rng.randint(6, 10)
    top_max = rng.randint(2, 4)
    bottom_max = rng.randint(2, 4)
    bbox_h = top_max + 1 + bottom_max
    patch_h, patch_w = bbox_h + 4, width + 4
    bbox_r0, bbox_c0 = 2, 2
    crack_row = bbox_r0 + top_max
    top_heights = _random_profile(width, top_max, rng)
    bottom_heights = _random_profile(width, bottom_max, rng)

    top_cells: set[tuple[int, int]] = set()
    bottom_cells: set[tuple[int, int]] = set()
    for rel_c in range(width):
        c = bbox_c0 + rel_c
        for r in range(crack_row - top_heights[rel_c], crack_row):
            top_cells.add((r, c))
        for r in range(crack_row + 1, crack_row + 1 + bottom_heights[rel_c]):
            bottom_cells.add((r, c))

    shape_cells = top_cells | bottom_cells
    bbox = (bbox_r0, bbox_c0, bbox_r0 + bbox_h - 1, bbox_c0 + width - 1)
    marker = rng.choice(_marker_candidates(shape_cells, bbox, "v", (patch_h, patch_w)))

    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)
    for r, c in top_cells | bottom_cells:
        inp[r, c] = shape_color
    for r, c in marker:
        inp[r, c] = fill_color
    for r, c in top_cells:
        out[r - 1, c] = shape_color
    for r, c in bottom_cells:
        out[r + 1, c] = shape_color
    out[crack_row - 1 : crack_row + 2, bbox_c0 : bbox_c0 + width] = fill_color
    return inp, out


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    mask = src != 0
    dst[top : top + src.shape[0], left : left + src.shape[1]][mask] = src[mask]


def _place_rectangles(
    grids: list[tuple[np.ndarray, np.ndarray]],
    canvas_shape: tuple[int, int],
    bg: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = canvas_shape
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    for src_in, src_out in sorted(grids, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        ph, pw = src_in.shape
        choices: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                rr0 = max(0, r - 1)
                rr1 = min(h, r + ph + 1)
                cc0 = max(0, c - 1)
                cc1 = min(w, c + pw + 1)
                if not occ[rr0:rr1, cc0:cc1].any():
                    choices.append((r, c))
        if not choices:
            return None
        top, left = rng.choice(choices)
        _paste_nonzero(inp, src_in, top, left)
        _paste_nonzero(out, src_out, top, left)
        occ[top : top + ph, left : left + pw] = True
    return inp, out


def _sample_object_count(rng: random.Random) -> int:
    return rng.choice([1, 2, 3])


def _sample_canvas(num_objects: int, rng: random.Random) -> tuple[int, int]:
    if num_objects == 1:
        return rng.randint(14, 22), rng.randint(14, 22)
    if num_objects == 2:
        return rng.randint(18, 26), rng.randint(18, 28)
    return rng.randint(22, 30), rng.randint(22, 30)


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = list(range(1, 10))
    bg = 0
    for _ in range(300):
        num_objects = _sample_object_count(rng)
        canvas = _sample_canvas(num_objects, rng)
        if max(canvas) > 30:
            continue
        rng.shuffle(colors)
        if 2 * num_objects > len(colors):
            continue

        objects: list[tuple[np.ndarray, np.ndarray]] = []
        for idx in range(num_objects):
            shape_color = colors[2 * idx]
            fill_color = colors[2 * idx + 1]
            if rng.random() < 0.5:
                objects.append(_build_vertical_object(shape_color, fill_color, rng))
            else:
                objects.append(_build_horizontal_object(shape_color, fill_color, rng))

        placed = _place_rectangles(objects, canvas, bg, rng)
        if placed is None:
            continue
        inp, out = placed
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid cracked-shape puzzle")
