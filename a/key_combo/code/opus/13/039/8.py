from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PAIR_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 600


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
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _pair_boxes_for_color(g: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    comps = _components(g == color)
    assert len(comps) == 2, "each pair color must form exactly two components"
    boxes = []
    for pts in comps:
        box = _bbox(pts)
        r0, c0, r1, c1 = box
        assert r1 - r0 == 1 and c1 - c0 == 1, "each component must be a 2x2 rectangle"
        assert len(pts) == 4, "rectangle area must be 4"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be solid"
        boxes.append(box)
    boxes.sort(key=lambda box: (box[0], box[1]))
    return boxes


def _pair_info(g: np.ndarray) -> list[dict]:
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    assert 3 <= len(colors) <= 4, "expected 3 or 4 pair colors"

    pairs: list[dict] = []
    for color in colors:
        upper, lower = _pair_boxes_for_color(g, color)
        ur0, uc0, ur1, uc1 = upper
        lr0, lc0, _lr1, lc1 = lower
        assert ur1 < lr0, "paired rectangles must be vertically separated"

        if uc1 < lc0:
            dr, dc = 1, 1
            start_corner = (ur1, uc1)
            end_corner = (lr0, lc0)
        else:
            assert lc1 < uc0, "paired rectangles must be diagonally separated"
            dr, dc = 1, -1
            start_corner = (ur1, uc0)
            end_corner = (lr0, lc1)

        row_delta = end_corner[0] - start_corner[0]
        col_delta = end_corner[1] - start_corner[1]
        assert row_delta > 0 and abs(row_delta) == abs(col_delta), "facing corners must be on one 45-degree diagonal"

        path: list[tuple[int, int]] = []
        r, c = start_corner[0] + dr, start_corner[1] + dc
        while (r, c) != end_corner:
            path.append((r, c))
            r += dr
            c += dc
        assert path, "there must be at least one diagonal gap cell"

        pairs.append(
            {
                "color": int(color),
                "upper": upper,
                "lower": lower,
                "order_key": (upper[1], upper[0]),
                "path": path,
            }
        )

    pairs.sort(key=lambda pair: pair["order_key"])
    for order, pair in enumerate(pairs):
        pair["order"] = order
    return pairs


def _draw_lines(g: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for pair in pairs:
        color = int(pair["color"])
        for r, c in pair["path"]:
            if out[r, c] == BG:
                out[r, c] = color
    return out


def _enclosed_regions(g: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                regions.append(pts)
    return regions


def _boundary_counts(
    g: np.ndarray,
    region: list[tuple[int, int]],
) -> tuple[Counter[int], set[tuple[int, int]]]:
    h, w = g.shape
    boundary: set[tuple[int, int]] = set()
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != BG:
                boundary.add((int(nr), int(nc)))
    counts = Counter(int(g[r, c]) for r, c in boundary)
    return counts, boundary


def _region_signature(info: dict) -> tuple:
    return (
        int(info["size"]),
        tuple(sorted((int(color), int(count)) for color, count in info["counts"].items())),
        int(info["winner"]),
    )


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    pairs = _pair_info(g)
    order_map = {int(pair["color"]): int(pair["order"]) for pair in pairs}
    drawn = _draw_lines(g, pairs)
    filled = drawn.copy()

    regions_info = []
    for region in _enclosed_regions(drawn):
        counts, boundary = _boundary_counts(drawn, region)
        assert counts, "enclosed regions must have a non-background boundary"
        best = min(counts, key=lambda color: (-counts[color], order_map[color]))
        for r, c in region:
            filled[r, c] = best
        regions_info.append(
            {
                "cells": region,
                "size": len(region),
                "boundary": boundary,
                "counts": counts,
                "winner": int(best),
                "top_count": max(counts.values()),
                "second_count": sorted(counts.values(), reverse=True)[1] if len(counts) >= 2 else 0,
            }
        )

    path_sets = {int(pair["color"]): set(pair["path"]) for pair in pairs}
    intersections = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if path_sets[pairs[i]["color"]] & path_sets[pairs[j]["color"]]:
                intersections += 1

    return {
        "pairs": pairs,
        "drawn": drawn,
        "filled": filled,
        "regions": regions_info,
        "intersections": intersections,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal pair connectors, detect the enclosed background regions they create, and fill each region with the boundary-majority line color.

    Input structure:
    - The background is black.
    - There are 3 or 4 distinct foreground colors, and each color appears as exactly two separate solid 2x2 rectangles.
    - Each same-color pair is diagonally offset so the upper rectangle and lower rectangle have facing corners on one 45-degree diagonal.

    Rule:
    - Sort the pairs by the upper rectangle's top-left column, breaking ties by row.
    - For each pair, draw the open diagonal segment between the two facing corners using that pair's color.
    - If a diagonal step lands on an already occupied cell, keep the earlier color and continue the line.
    - After all lines are drawn, find every 4-connected background component that does not touch the grid edge.
    - Count the non-background boundary pixels around each enclosed region by color, and fill the region with the color having the largest boundary count.
    - If the largest count ties, choose the color whose line was drawn earlier.

    Key insight:
    - No single diagonal pair closes a region by itself; the enclosed cavities emerge only from multiple diagonal connectors crossing, and the winning fill color is determined by visible boundary contribution rather than by object identity alone.
    """
    return _analyze(np.asarray(grid, dtype=int))["filled"]


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _segment_boxes(seg: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    p0 = seg["p0"]
    p1 = seg["p1"]
    if seg["type"] == "desc":
        upper = (p0[0] - 2, p0[1] - 2, p0[0] - 1, p0[1] - 1)
        lower = (p1[0] + 1, p1[1] + 1, p1[0] + 2, p1[1] + 2)
    else:
        upper = (p0[0] - 2, p0[1] + 1, p0[0] - 1, p0[1] + 2)
        lower = (p1[0] + 1, p1[1] - 2, p1[0] + 2, p1[1] - 1)
    return upper, lower


def _segment_cells(seg: dict) -> list[tuple[int, int]]:
    dr, dc = (1, 1) if seg["type"] == "desc" else (1, -1)
    r, c = seg["p0"]
    pts: list[tuple[int, int]] = []
    while True:
        pts.append((int(r), int(c)))
        if (r, c) == seg["p1"]:
            break
        r += dr
        c += dc
    return pts


def _shift_segment(seg: dict, dr: int, dc: int) -> dict:
    return {
        "id": int(seg["id"]),
        "type": str(seg["type"]),
        "p0": (int(seg["p0"][0] + dr), int(seg["p0"][1] + dc)),
        "p1": (int(seg["p1"][0] + dr), int(seg["p1"][1] + dc)),
    }


def _mirror_segments(segs: list[dict], rows: int, cols: int, flip_h: bool, flip_v: bool) -> list[dict]:
    out = []
    for seg in segs:
        pts = _segment_cells(seg)
        transformed = [
            ((rows - 1 - r) if flip_v else r, (cols - 1 - c) if flip_h else c)
            for r, c in pts
        ]
        transformed.sort(key=lambda rc: (rc[0], rc[1]))
        p0 = transformed[0]
        p1 = transformed[-1]
        out.append(
            {
                "id": int(seg["id"]),
                "type": "desc" if p1[1] > p0[1] else "asc",
                "p0": p0,
                "p1": p1,
            }
        )
    return out


def _build_segments(mode: str, n_pairs: int) -> tuple[int, int, list[dict]] | None:
    gap_pos = random.choice([4, 6])
    gap_neg = random.choice([2, 4, 6])
    k1 = random.choice([0, 2, 4, 6])
    k2 = k1 + gap_pos
    s1 = random.choice(list(range(k2 + 8, k2 + 17, 2)))
    s2 = s1 + gap_neg

    if mode == "tie":
        tails = [(random.randint(0, 1), random.randint(0, 1)) for _ in range(4)]
    else:
        tails = [(random.randint(0, 2), random.randint(0, 2)) for _ in range(4)]

    segs: list[dict] = []
    for idx, k in enumerate([k1, k2]):
        before, after = tails[idx]
        r0 = (s1 - k) // 2 - before
        r1 = (s2 - k) // 2 + after
        segs.append({"id": idx, "type": "desc", "p0": (r0, r0 + k), "p1": (r1, r1 + k)})
    for idx, s in enumerate([s1, s2], start=2):
        before, after = tails[idx]
        r0 = (s - k2) // 2 - before
        r1 = (s - k1) // 2 + after
        segs.append({"id": idx, "type": "asc", "p0": (r0, s - r0), "p1": (r1, s - r1)})

    if n_pairs == 3:
        omit = random.randrange(4)
        segs = [seg for seg in segs if seg["id"] != omit]

    all_boxes = [box for seg in segs for box in _segment_boxes(seg)]
    if not all_boxes:
        return None
    min_r = min(box[0] for box in all_boxes)
    min_c = min(box[1] for box in all_boxes)
    segs = [_shift_segment(seg, 2 - min_r, 2 - min_c) for seg in segs]

    all_boxes = [box for seg in segs for box in _segment_boxes(seg)]
    max_r = max(box[2] for box in all_boxes)
    max_c = max(box[3] for box in all_boxes)
    rows = max_r + 1 + random.randint(2, 5)
    cols = max_c + 1 + random.randint(2, 5)
    if max(rows, cols) > 30:
        return None

    shift_r = random.randint(0, rows - max_r - 1)
    shift_c = random.randint(0, cols - max_c - 1)
    segs = [_shift_segment(seg, shift_r, shift_c) for seg in segs]
    segs = _mirror_segments(
        segs,
        rows,
        cols,
        flip_h=bool(random.getrandbits(1)),
        flip_v=bool(random.getrandbits(1)),
    )

    for seg, color in zip(segs, random.sample(PAIR_COLORS, len(segs))):
        seg["color"] = int(color)

    rects: list[tuple[int, int, int, int]] = []
    occupied_rects: set[tuple[int, int]] = set()
    for seg in segs:
        for box in _segment_boxes(seg):
            r0, c0, r1, c1 = box
            if min(r0, c0) < 0 or r1 >= rows or c1 >= cols:
                return None
            for old in rects:
                if _rects_overlap(box, old):
                    return None
            rects.append(box)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    occupied_rects.add((r, c))

    for seg in segs:
        own_rects = set()
        for box in _segment_boxes(seg):
            r0, c0, r1, c1 = box
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    own_rects.add((r, c))
        for cell in _segment_cells(seg):
            if cell in occupied_rects and cell not in own_rects:
                return None

    return rows, cols, segs


def _render_input(rows: int, cols: int, segs: list[dict]) -> np.ndarray:
    g = np.full((rows, cols), BG, dtype=int)
    for seg in segs:
        for box in _segment_boxes(seg):
            r0, c0, r1, c1 = box
            g[r0 : r1 + 1, c0 : c1 + 1] = int(seg["color"])
    return g


def _validate_candidate(inp: np.ndarray, mode: str) -> bool:
    stats = _analyze(inp)
    regions = stats["regions"]
    if stats["intersections"] < 2 or not regions:
        return False
    if max(int(region["size"]) for region in regions) < 5:
        return False
    if not all(len(region["counts"]) >= 2 for region in regions):
        return False

    if mode == "tie" and not any(region["top_count"] == region["second_count"] for region in regions):
        return False
    if mode == "mixed" and not any(region["top_count"] > region["second_count"] for region in regions):
        return False

    full_sig = sorted(_region_signature(region) for region in regions)
    colors = [pair["color"] for pair in stats["pairs"]]
    pair_changed = False
    for removed_color in colors:
        reduced = inp.copy()
        reduced[reduced == removed_color] = BG
        try:
            reduced_stats = _analyze(reduced)
        except Exception:
            pair_changed = True
            break
        reduced_sig = sorted(_region_signature(region) for region in reduced_stats["regions"])
        if reduced_sig != full_sig:
            pair_changed = True
            break
    if not pair_changed:
        return False

    return np.array_equal(main(inp), stats["filled"])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        mode = random.choice(["tie", "mixed"])
        n_pairs = random.choices([3, 4], weights=[3, 5], k=1)[0]
        built = _build_segments(mode, n_pairs)
        if built is None:
            continue
        rows, cols, segs = built
        inp = _render_input(rows, cols, segs)
        try:
            if _validate_candidate(inp, mode):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid enclosed-diagonal ARC input")
