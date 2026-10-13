import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}


GEN_BG = 0
GEN_ARROW = 5
GEN_BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_H_RANGE = (18, 29)
GEN_W_RANGE = (18, 29)
GEN_RECT_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
GEN_RECT_H_RANGE = (4, 8)
GEN_RECT_W_RANGE = (4, 8)
GEN_RECT_GAP = 3
GEN_RECT_PLACE_TRIES = 120
GEN_ARROW_DEPTH_MAX = 5
GEN_ARROW_ATTEMPTS = 6
GEN_SAMPLE_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _perps(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (-dc, dr), (dc, -dr)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(coords),
                }
            )
    return comps


def _split_rectangles_and_arrows(comps: list[dict]) -> tuple[list[dict], list[dict]]:
    rectangles = []
    arrows = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if comp["area"] == h * w and h >= 3 and w >= 3:
            rectangles.append(comp)
        else:
            arrows.append(comp)
    return rectangles, arrows


def _rect_id_grid(rectangles: list[dict], shape: tuple[int, int]) -> np.ndarray:
    rect_ids = np.full(shape, -1, dtype=int)
    for idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        rect_ids[r0 : r1 + 1, c0 : c1 + 1] = idx
    return rect_ids


def _in_rect(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _parse_arrow(comp: dict, rect_ids: np.ndarray) -> dict:
    h, w = rect_ids.shape
    hits = set()
    for r, c in comp["coords"]:
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(rect_ids[nr, nc]) != -1:
                hits.add((int(r), int(c), int(dr), int(dc), int(rect_ids[nr, nc])))
    assert len(hits) == 1, "each arrow must point to exactly one rectangle"

    tip_r, tip_c, dr, dc, rect_id = next(iter(hits))
    coord_set = comp["coord_set"]
    depth = 0
    shaft = set()
    while (tip_r - dr * (depth + 1), tip_c - dc * (depth + 1)) in coord_set:
        depth += 1
        shaft.add((tip_r - dr * depth, tip_c - dc * depth))
    assert 1 <= depth <= GEN_ARROW_DEPTH_MAX

    back_r = int(tip_r - dr)
    back_c = int(tip_c - dc)
    perp1, perp2 = _perps(dr, dc)
    wing1 = (int(back_r + perp1[0]), int(back_c + perp1[1]))
    wing2 = (int(back_r + perp2[0]), int(back_c + perp2[1]))
    expected = {(int(tip_r), int(tip_c)), wing1, wing2} | shaft
    assert coord_set == expected, "arrow geometry does not match the expected shaft+V pattern"

    return {
        "rect_id": int(rect_id),
        "tip": (int(tip_r), int(tip_c)),
        "dir": (int(dr), int(dc)),
        "depth": int(depth),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat grey arrows as stamping instructions: erase the arrows and carve 3-wide rectangular holes
    into adjacent solid rectangles, with hole depth equal to the arrow shaft length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    comps = _components(g, bg)
    rectangles, arrows = _split_rectangles_and_arrows(comps)
    assert rectangles and arrows

    rect_ids = _rect_id_grid(rectangles, g.shape)
    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = bg

    for arrow in arrows:
        info = _parse_arrow(arrow, rect_ids)
        rect = rectangles[int(info["rect_id"])]
        rect_color = int(rect["color"])
        rect_bbox = rect["bbox"]
        tip_r, tip_c = info["tip"]
        dr, dc = info["dir"]
        depth = int(info["depth"])
        perp1, perp2 = _perps(dr, dc)

        for step in range(1, depth + 1):
            center_r = int(tip_r + dr * step)
            center_c = int(tip_c + dc * step)
            for off_r, off_c in ((0, 0), perp1, perp2):
                rr = int(center_r + off_r)
                cc = int(center_c + off_c)
                if _in_rect(rr, cc, rect_bbox) and int(out[rr, cc]) == rect_color:
                    out[rr, cc] = bg

    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _arrow_shape(
    bbox: tuple[int, int, int, int], side: str, center: int, depth: int
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    dr, dc = SIDE_TO_DIR[side]
    if side == "left":
        tip = (int(center), int(c0 - 1))
    elif side == "right":
        tip = (int(center), int(c1 + 1))
    elif side == "top":
        tip = (int(r0 - 1), int(center))
    else:
        tip = (int(r1 + 1), int(center))

    back_r = int(tip[0] - dr)
    back_c = int(tip[1] - dc)
    perp1, perp2 = _perps(dr, dc)
    cells = {tip}
    for step in range(1, depth + 1):
        cells.add((int(tip[0] - dr * step), int(tip[1] - dc * step)))
    cells.add((int(back_r + perp1[0]), int(back_c + perp1[1])))
    cells.add((int(back_r + perp2[0]), int(back_c + perp2[1])))
    return tip, (int(dr), int(dc)), cells


def _add_chebyshev_zone(zone: set[tuple[int, int]], cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                zone.add((int(r + dr), int(c + dc)))


def _valid_arrow_candidate(
    grid: np.ndarray,
    rect_ids: np.ndarray,
    rect_idx: int,
    bbox: tuple[int, int, int, int],
    side: str,
    center: int,
    depth: int,
    blocked: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    tip, direction, cells = _arrow_shape(bbox, side, center, depth)
    dr, dc = direction

    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if int(grid[r, c]) != GEN_BG or (r, c) in blocked:
            return None

    for nr, nc in DIR4:
        rr = int(tip[0] + nr)
        cc = int(tip[1] + nc)
        if 0 <= rr < h and 0 <= cc < w and int(rect_ids[rr, cc]) != -1:
            if (nr, nc) != (dr, dc) or int(rect_ids[rr, cc]) != rect_idx:
                return None
    rr = int(tip[0] + dr)
    cc = int(tip[1] + dc)
    if not (0 <= rr < h and 0 <= cc < w and int(rect_ids[rr, cc]) == rect_idx):
        return None

    for r, c in cells:
        if (r, c) == tip:
            continue
        for nr, nc in DIR4:
            rr = int(r + nr)
            cc = int(c + nc)
            if 0 <= rr < h and 0 <= cc < w and int(rect_ids[rr, cc]) != -1:
                return None

    return {
        "side": side,
        "center": int(center),
        "depth": int(depth),
        "tip": tip,
        "dir": direction,
        "cells": cells,
    }


def _arrow_candidates_for_rect(
    grid: np.ndarray,
    rect_ids: np.ndarray,
    rect_idx: int,
    bbox: tuple[int, int, int, int],
    blocked: set[tuple[int, int]],
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    rect_h = int(r1 - r0 + 1)
    rect_w = int(c1 - c0 + 1)
    candidates = []

    for side in ("left", "right"):
        max_depth = min(GEN_ARROW_DEPTH_MAX, rect_w)
        for center in range(r0 + 1, r1):
            for depth in range(1, max_depth + 1):
                cand = _valid_arrow_candidate(grid, rect_ids, rect_idx, bbox, side, center, depth, blocked)
                if cand is not None:
                    candidates.append(cand)

    for side in ("top", "bottom"):
        max_depth = min(GEN_ARROW_DEPTH_MAX, rect_h)
        for center in range(c0 + 1, c1):
            for depth in range(1, max_depth + 1):
                cand = _valid_arrow_candidate(grid, rect_ids, rect_idx, bbox, side, center, depth, blocked)
                if cand is not None:
                    candidates.append(cand)

    return candidates


def _place_rectangles() -> tuple[int, int, list[dict]] | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    count = random.choice(GEN_RECT_COUNT_CHOICES)
    colors = random.sample(GEN_BLOCK_COLORS, count)
    rectangles = []

    for color in colors:
        placed = False
        for _ in range(GEN_RECT_PLACE_TRIES):
            rect_h = random.randint(*GEN_RECT_H_RANGE)
            rect_w = random.randint(*GEN_RECT_W_RANGE)
            if rect_h >= h - 2 or rect_w >= w - 2:
                continue
            r0 = random.randint(1, h - rect_h - 1)
            c0 = random.randint(1, w - rect_w - 1)
            bbox = (int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1))
            if all(_boxes_separated(bbox, other["bbox"], GEN_RECT_GAP) for other in rectangles):
                rectangles.append({"color": int(color), "bbox": bbox})
                placed = True
                break
        if not placed:
            return None

    return h, w, rectangles


def _background_is_unique_majority(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
    bg_count = count_map.get(GEN_BG, 0)
    return all(bg_count > count for color, count in count_map.items() if color != GEN_BG)


def _attempt_sample() -> np.ndarray | None:
    placed = _place_rectangles()
    if placed is None:
        return None
    full_h, full_w, rectangles = placed

    grid = np.full((full_h, full_w), GEN_BG, dtype=int)
    rect_ids = np.full((full_h, full_w), -1, dtype=int)
    for idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
        rect_ids[r0 : r1 + 1, c0 : c1 + 1] = idx

    blocked: set[tuple[int, int]] = set()
    total_arrows = 0
    order = list(range(len(rectangles)))
    random.shuffle(order)

    for rect_idx in order:
        bbox = rectangles[rect_idx]["bbox"]
        placed_here = 0
        goal = random.choice([1, 1, 2, 2, 3])

        for _ in range(GEN_ARROW_ATTEMPTS):
            candidates = _arrow_candidates_for_rect(grid, rect_ids, rect_idx, bbox, blocked)
            if not candidates:
                break
            by_side: dict[str, list[dict]] = defaultdict(list)
            for cand in candidates:
                by_side[cand["side"]].append(cand)
            side = random.choice(list(by_side))
            cand = random.choice(by_side[side])
            for r, c in cand["cells"]:
                grid[r, c] = GEN_ARROW
            _add_chebyshev_zone(blocked, cand["cells"])
            placed_here += 1
            total_arrows += 1
            if placed_here >= goal:
                break

        if placed_here == 0:
            return None

    if total_arrows == 0 or not _background_is_unique_majority(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_SAMPLE_ATTEMPTS):
        grid = _attempt_sample()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-stamping sample")
