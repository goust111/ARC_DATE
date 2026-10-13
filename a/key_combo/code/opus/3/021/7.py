from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 4
CENTER_COLOR = 5
GUIDE_COLORS = (1, 2, 3, 6, 7, 8, 9)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(int(r) - 1, int(c)), (int(r) + 1, int(c)), (int(r), int(c) - 1), (int(r), int(c) + 1)]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: Iterable[int] = ()) -> list[int]:
    blocked = {int(v) for v in exclude}
    return sorted(int(v) for v in np.unique(grid) if int(v) not in blocked)


def bbox_coords(coords: Iterable[Coord]) -> BBox:
    pts = list(coords)
    rows = [int(r) for r, _c in pts]
    cols = [int(c) for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def shift_bbox(box: BBox, dr: int, dc: int) -> BBox:
    return (
        int(box[0] + dr),
        int(box[1] + dc),
        int(box[2] + dr),
        int(box[3] + dc),
    )


def bbox_intersection(a: BBox, b: BBox) -> BBox | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def bbox_contains(box: BBox, rc: Coord) -> bool:
    r, c = map(int, rc)
    return int(box[0]) <= r <= int(box[2]) and int(box[1]) <= c <= int(box[3])


def normalize_coords(coords: Iterable[Coord]) -> list[Coord]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0, c0, _r1, _c1 = bbox_coords(pts)
    return sorted((int(r - r0), int(c - c0)) for r, c in pts)


def shift_coords(coords: Iterable[Coord], dr: int, dc: int) -> list[Coord]:
    return [(int(r) + int(dr), int(c) + int(dc)) for r, c in coords]


def components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(g == int(color)):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((int(cr), int(cc)))
            for nr, nc in neighbors4(cr, cc):
                if not in_bounds(g.shape, nr, nc):
                    continue
                if seen[nr, nc] or int(g[nr, nc]) != int(color):
                    continue
                seen[nr, nc] = True
                stack.append((int(nr), int(nc)))
        comps.append(sorted(comp))
    return comps


def is_connected(coords: Iterable[Coord]) -> bool:
    pts = {tuple(map(int, rc)) for rc in coords}
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def paint_cells(grid: np.ndarray, coords: Iterable[Coord], color: int) -> None:
    for r, c in coords:
        if in_bounds(grid.shape, r, c):
            grid[int(r), int(c)] = int(color)


def frame_border_coords(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = map(int, box)
    out: list[Coord] = []
    for c in range(c0, c1 + 1):
        out.append((r0, c))
        out.append((r1, c))
    for r in range(r0 + 1, r1):
        out.append((r, c0))
        out.append((r, c1))
    return out


def strict_interior_coords(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = map(int, box)
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def find_unique_translation(center_coords: Iterable[Coord], partial_coords: Iterable[Coord]) -> Coord:
    center = {tuple(map(int, rc)) for rc in center_coords}
    part = [tuple(map(int, rc)) for rc in partial_coords]
    assert center and part
    pr0, pc0 = part[0]
    candidates: set[Coord] = set()
    for sr, sc in center:
        dr = int(pr0 - sr)
        dc = int(pc0 - sc)
        if all((int(r - dr), int(c - dc)) in center for r, c in part):
            candidates.add((int(dr), int(dc)))
    assert len(candidates) == 1, "partial copy must determine a unique displacement vector"
    return next(iter(candidates))


def enumerate_tiled_copies(
    center_coords: list[Coord],
    center_bbox: BBox,
    vec: Coord,
    shape: tuple[int, int],
) -> list[dict]:
    dr, dc = map(int, vec)
    copies: list[dict] = []
    k = 1
    while True:
        shifted = shift_coords(center_coords, k * dr, k * dc)
        visible = [(r, c) for r, c in shifted if in_bounds(shape, r, c)]
        if not visible:
            break
        box = shift_bbox(center_bbox, k * dr, k * dc)
        copies.append(
            {
                "k": int(k),
                "coords": shifted,
                "visible_coords": visible,
                "bbox": box,
                "fully_in_bounds": len(visible) == len(center_coords),
            }
        )
        k += 1
        if k > max(shape) * 2:
            break
    return copies


def detect_frame(grid: np.ndarray, bg: int, center_color: int) -> dict:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict] = []
    for color in colors_present(g, exclude={bg, center_color}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        if not coords:
            continue
        box = bbox_coords(coords)
        if bbox_h(box) < 3 or bbox_w(box) < 3:
            continue
        border = frame_border_coords(box)
        if any(int(g[r, c]) != int(color) for r, c in border):
            continue
        pattern = [
            (int(r), int(c))
            for r, c in strict_interior_coords(box)
            if int(g[r, c]) == int(color)
        ]
        if not pattern:
            continue
        candidates.append(
            {
                "color": int(color),
                "bbox": box,
                "border": border,
                "pattern": sorted(pattern),
            }
        )
    assert len(candidates) == 1, "expected one patterned hollow frame"
    return candidates[0]


def _solve(grid: np.ndarray, *, overlay_mode: str = "bbox", max_k: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    center_components = components_of_color(g, CENTER_COLOR)
    assert center_components, "expected a central gray sprite"
    center = max(center_components, key=len)
    center_box = bbox_coords(center)

    frame = detect_frame(g, bg, CENTER_COLOR)
    frame_color = int(frame["color"])

    guides: list[dict] = []
    for color in colors_present(g, exclude={bg, CENTER_COLOR, frame_color}):
        for coords in components_of_color(g, color):
            vec = find_unique_translation(center, coords)
            copies = enumerate_tiled_copies(center, center_box, vec, g.shape)
            guides.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "vector": vec,
                    "copies": copies,
                }
            )

    assert 1 <= len(guides) <= 2, "expected one or two partial guide copies"
    guides.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))

    out = g.copy()
    frame_box = frame["bbox"]
    pattern = [tuple(map(int, rc)) for rc in frame["pattern"]]

    for guide in guides:
        color = int(guide["color"])
        for copy in guide["copies"]:
            if max_k is not None and int(copy["k"]) > int(max_k):
                continue
            paint_cells(out, copy["visible_coords"], color)
            if bbox_intersection(copy["bbox"], frame_box) is None:
                continue
            if overlay_mode == "bbox":
                overlay_cells = [rc for rc in pattern if bbox_contains(copy["bbox"], rc)]
            elif overlay_mode == "shape":
                copy_set = {tuple(map(int, rc)) for rc in copy["coords"]}
                overlay_cells = [rc for rc in pattern if tuple(map(int, rc)) in copy_set]
            else:
                raise ValueError(f"unknown overlay_mode: {overlay_mode}")
            paint_cells(out, overlay_cells, color)

    paint_cells(out, center, CENTER_COLOR)
    paint_cells(out, frame["border"], frame_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the central gray sprite along guide displacement vectors and stamp frame-pattern overlaps in guide colors.

    Input structure:
    - Black background with one largest gray 4-connected sprite near the center.
    - One or two smaller colored guide components, each a translated connected subset of the gray sprite.
    - One hollow rectangular frame whose border and sparse interior pattern share one color.

    Rule:
    - Take the largest gray component as the source sprite.
    - For each colored guide, find the unique translation vector whose shifted gray sprite covers that guide subset.
    - Repeat the full gray sprite at 1x, 2x, 3x, ... of that vector, recoloring every repeated copy with the guide color until the copy leaves the grid.
    - Whenever a repeated copy's bbox intersects the frame bbox, recolor the frame's interior pattern cells that lie inside that bbox with the guide color.
    - Keep the original gray sprite unchanged and keep the frame border in its original frame color.

    Key insight:
    - The guide subset does not mark just one destination: its displacement vector becomes both the tiling direction and the tiling period, while frame interaction depends on bbox overlap rather than exact sprite-pixel overlap.
    """
    return _solve(np.asarray(grid, dtype=int), overlay_mode="bbox")


def random_connected_shape() -> list[Coord]:
    for _ in range(300):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        target = random.randint(5, min(8, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in list(cells):
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        cells = set(normalize_coords(cells))
        box = bbox_coords(cells)
        if bbox_h(box) < 2 or bbox_w(box) < 2:
            continue
        if len(cells) == bbox_h(box) * bbox_w(box):
            continue
        return sorted(cells)
    raise RuntimeError("failed to sample an irregular connected sprite")


def subset_occurrence_count(shape_cells: list[Coord], subset_cells: list[Coord]) -> int:
    shape = {tuple(map(int, rc)) for rc in shape_cells}
    subset = [tuple(map(int, rc)) for rc in subset_cells]
    seed_r, seed_c = subset[0]
    seen: set[Coord] = set()
    for tr, tc in shape:
        dr = int(tr - seed_r)
        dc = int(tc - seed_c)
        shifted = {(int(r + dr), int(c + dc)) for r, c in subset}
        if shifted <= shape:
            seen.add((int(dr), int(dc)))
    return len(seen)


def subset_candidates(shape_cells: list[Coord]) -> list[dict]:
    ordered = list(shape_cells)
    n = len(ordered)
    full_box = bbox_coords(shape_cells)
    out: list[dict] = []
    for mask in range(1, 1 << n):
        size = int(mask.bit_count())
        if size < 2 or size > n - 2:
            continue
        subset = [ordered[i] for i in range(n) if (mask >> i) & 1]
        if not is_connected(subset):
            continue
        if subset_occurrence_count(shape_cells, subset) != 1:
            continue
        sub_box = bbox_coords(subset)
        tl_vec = (int(sub_box[0] - full_box[0]), int(sub_box[1] - full_box[1]))
        br_vec = (int(sub_box[2] - full_box[2]), int(sub_box[3] - full_box[3]))
        score = 0
        score += int(sub_box[0] > full_box[0])
        score += int(sub_box[1] > full_box[1])
        score += int(sub_box[2] < full_box[2])
        score += int(sub_box[3] < full_box[3])
        score += int(bbox_h(sub_box) * bbox_w(sub_box) > len(subset))
        out.append(
            {
                "cells": sorted(subset),
                "bbox": sub_box,
                "size": size,
                "score": score,
                "tl_offset": tl_vec,
                "br_offset": br_vec,
            }
        )
    random.shuffle(out)
    out.sort(key=lambda item: (-int(item["score"]), -int(item["size"])))
    return out


def direction_key(vec: Coord) -> Coord:
    dr, dc = map(int, vec)
    g = math.gcd(abs(dr), abs(dc))
    return int(dr // g), int(dc // g)


def vector_pool(box: BBox) -> list[Coord]:
    h = bbox_h(box)
    w = bbox_w(box)
    pool: list[Coord] = []
    for dr in range(-6, 7):
        for dc in range(-6, 7):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) < 2 or max(abs(dr), abs(dc)) > 6:
                continue
            if abs(dr) + abs(dc) < 3:
                continue
            if abs(dr) < h and abs(dc) < w:
                continue
            pool.append((int(dr), int(dc)))
    random.shuffle(pool)
    return pool


def copies_overlap(base_occ: set[Coord], guides: list[dict]) -> bool:
    occupied = set(base_occ)
    for guide in guides:
        for copy in guide["copies"]:
            for rc in copy["visible_coords"]:
                cell = tuple(map(int, rc))
                if cell in occupied:
                    return True
                occupied.add(cell)
    return False


def select_subsets(candidates: list[dict], n_guides: int, center_box: BBox) -> list[dict] | None:
    filtered = [
        cand
        for cand in candidates
        if tuple(cand["tl_offset"]) != (0, 0) and tuple(cand["br_offset"]) != (0, 0)
    ]
    source = filtered if len(filtered) >= n_guides else candidates
    if len(source) < n_guides:
        return None
    chosen: list[dict] = []
    used: set[tuple[Coord, ...]] = set()
    for cand in source:
        key = tuple(cand["cells"])
        if key in used:
            continue
        chosen.append(cand)
        used.add(key)
        if len(chosen) == n_guides:
            break
    if len(chosen) != n_guides:
        return None
    return chosen


def build_input_grid(case: dict) -> np.ndarray:
    H, W = map(int, case["shape"])
    grid = np.full((H, W), BG, dtype=int)
    paint_cells(grid, case["center_abs"], CENTER_COLOR)
    for guide in case["guides"]:
        paint_cells(grid, guide["partial_abs"], guide["color"])
    paint_cells(grid, case["frame"]["border"], FRAME_COLOR)
    paint_cells(grid, case["frame"]["pattern"], FRAME_COLOR)
    return grid


def simulate_case_output(case: dict, *, overlay_mode: str = "bbox", max_k: int | None = None) -> np.ndarray:
    out = build_input_grid(case)
    frame_box = case["frame"]["bbox"]
    pattern = [tuple(map(int, rc)) for rc in case["frame"]["pattern"]]
    for guide in case["guides"]:
        color = int(guide["color"])
        for copy in guide["copies"]:
            if max_k is not None and int(copy["k"]) > int(max_k):
                continue
            paint_cells(out, copy["visible_coords"], color)
            if bbox_intersection(copy["bbox"], frame_box) is None:
                continue
            if overlay_mode == "bbox":
                overlay_cells = [rc for rc in pattern if bbox_contains(copy["bbox"], rc)]
            elif overlay_mode == "shape":
                copy_set = {tuple(map(int, rc)) for rc in copy["coords"]}
                overlay_cells = [rc for rc in pattern if tuple(map(int, rc)) in copy_set]
            else:
                raise ValueError(f"bad overlay_mode: {overlay_mode}")
            paint_cells(out, overlay_cells, color)
    paint_cells(out, case["center_abs"], CENTER_COLOR)
    paint_cells(out, case["frame"]["border"], FRAME_COLOR)
    return out


def choose_frame(case: dict) -> dict | None:
    H, W = map(int, case["shape"])
    center_box = case["center_bbox_abs"]
    forbidden_boxes = [center_box] + [guide["copies"][0]["bbox"] for guide in case["guides"]]
    input_occ = set(case["center_abs"])
    for guide in case["guides"]:
        input_occ |= set(guide["partial_abs"])

    targets = []
    for gi, guide in enumerate(case["guides"]):
        full = [copy for copy in guide["copies"] if copy["fully_in_bounds"] and int(copy["k"]) >= 2]
        for copy in full:
            targets.append((gi, copy))
    random.shuffle(targets)

    for gi, target in targets:
        target_box = target["bbox"]
        target_set = {tuple(map(int, rc)) for rc in target["coords"]}
        tr0, tc0, tr1, tc1 = map(int, target_box)
        for _ in range(200):
            fh = random.randint(max(5, bbox_h(case["center_bbox_rel"]) + 1), min(9, H - 1))
            fw = random.randint(max(5, bbox_w(case["center_bbox_rel"]) + 1), min(9, W - 1))
            r_lo = max(0, tr0 - fh + 2)
            r_hi = min(H - fh, tr1 - 1)
            c_lo = max(0, tc0 - fw + 2)
            c_hi = min(W - fw, tc1 - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            fr0 = random.randint(r_lo, r_hi)
            fc0 = random.randint(c_lo, c_hi)
            box = (int(fr0), int(fc0), int(fr0 + fh - 1), int(fc0 + fw - 1))
            interior = strict_interior_coords(box)
            overlap_box = bbox_intersection((fr0 + 1, fc0 + 1, fr0 + fh - 2, fc0 + fw - 2), target_box)
            if overlap_box is None:
                continue
            if any(bbox_intersection(box, bad) is not None for bad in forbidden_boxes):
                continue
            other_hit = False
            for gj, other in enumerate(case["guides"]):
                for copy in other["copies"]:
                    if gj == gi and int(copy["k"]) == int(target["k"]):
                        continue
                    if bbox_intersection(box, copy["bbox"]) is not None:
                        other_hit = True
                        break
                if other_hit:
                    break
            if other_hit:
                continue

            overlap_cells = [rc for rc in interior if bbox_contains(target_box, rc)]
            if len(overlap_cells) < 2 or len(overlap_cells) == len(interior):
                continue
            filled_overlap = [rc for rc in overlap_cells if tuple(map(int, rc)) in target_set]
            blank_overlap = [rc for rc in overlap_cells if tuple(map(int, rc)) not in target_set]
            outside_overlap = [rc for rc in interior if rc not in overlap_cells]
            if not filled_overlap or not blank_overlap or not outside_overlap:
                continue

            chosen_filled = random.choice(filled_overlap)
            chosen_blank = random.choice(blank_overlap)
            pattern = {
                chosen_filled,
                chosen_blank,
                random.choice(outside_overlap),
            }
            remaining = [rc for rc in interior if rc not in pattern]
            random.shuffle(remaining)
            extra_count = random.randint(0, min(3, len(remaining)))
            pattern.update(remaining[:extra_count])

            border = frame_border_coords(box)
            frame_cells = set(border) | set(pattern)
            if frame_cells & input_occ:
                continue
            return {
                "bbox": box,
                "border": sorted(border),
                "pattern": sorted(pattern),
                "target_guide": int(gi),
                "target_k": int(target["k"]),
                "blank_overlap_cell": tuple(map(int, chosen_blank)),
                "filled_overlap_cell": tuple(map(int, chosen_filled)),
            }
    return None


def build_case() -> dict | None:
    H = random.randint(18, 28)
    W = random.randint(18, 28)
    center_rel = random_connected_shape()
    center_bbox_rel = bbox_coords(center_rel)
    sub_cands = subset_candidates(center_rel)
    n_guides = 2 if random.random() < 0.55 else 1
    subsets = select_subsets(sub_cands, n_guides, center_bbox_rel)
    if subsets is None:
        return None

    for _ in range(120):
        r0 = random.randint(2, H - bbox_h(center_bbox_rel) - 3)
        c0 = random.randint(2, W - bbox_w(center_bbox_rel) - 3)
        center_abs = shift_coords(center_rel, r0, c0)
        center_bbox_abs = shift_bbox(center_bbox_rel, r0, c0)

        feasible: list[dict] = []
        for vec in vector_pool(center_bbox_rel):
            copies = enumerate_tiled_copies(center_abs, center_bbox_abs, vec, (H, W))
            if len(copies) < 2:
                continue
            if not copies[0]["fully_in_bounds"] or not copies[1]["fully_in_bounds"]:
                continue
            if {tuple(map(int, rc)) for rc in copies[0]["coords"]} & {tuple(map(int, rc)) for rc in center_abs}:
                continue
            feasible.append({"vector": vec, "copies": copies, "dir": direction_key(vec)})
        random.shuffle(feasible)

        if n_guides == 1:
            combos = [[item] for item in feasible]
        else:
            combos = []
            for i in range(len(feasible)):
                for j in range(i + 1, len(feasible)):
                    if feasible[i]["dir"] == feasible[j]["dir"]:
                        continue
                    combos.append([feasible[i], feasible[j]])
            random.shuffle(combos)

        for combo in combos[:120]:
            guide_colors = random.sample(GUIDE_COLORS, len(combo))
            guides: list[dict] = []
            for idx, info in enumerate(combo):
                subset = subsets[idx]
                partial_abs = shift_coords(subset["cells"], info["vector"][0] + r0, info["vector"][1] + c0)
                guides.append(
                    {
                        "color": int(guide_colors[idx]),
                        "vector": tuple(map(int, info["vector"])),
                        "partial_abs": sorted(partial_abs),
                        "partial_bbox": bbox_coords(partial_abs),
                        "partial_rel": subset["cells"],
                        "copies": info["copies"],
                    }
                )
            if copies_overlap(set(center_abs), guides):
                continue

            case = {
                "shape": (H, W),
                "center_rel": center_rel,
                "center_abs": sorted(center_abs),
                "center_bbox_rel": center_bbox_rel,
                "center_bbox_abs": center_bbox_abs,
                "guides": guides,
            }
            frame = choose_frame(case)
            if frame is None:
                continue
            case["frame"] = frame
            return case
    return None


def validate_case(case: dict) -> bool:
    inp = build_input_grid(case)
    expected = simulate_case_output(case, overlay_mode="bbox")
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    if not np.array_equal(_solve(inp, overlay_mode="bbox"), expected):
        return False

    center_components = components_of_color(inp, CENTER_COLOR)
    if len(center_components) != 1:
        return False

    frame_hits = []
    for gi, guide in enumerate(case["guides"]):
        vec = find_unique_translation(case["center_abs"], guide["partial_abs"])
        if tuple(vec) != tuple(guide["vector"]):
            return False
        tl_guess = (
            int(guide["partial_bbox"][0] - case["center_bbox_abs"][0]),
            int(guide["partial_bbox"][1] - case["center_bbox_abs"][1]),
        )
        if tl_guess == tuple(guide["vector"]):
            return False
        if len([copy for copy in guide["copies"] if copy["fully_in_bounds"]]) < 2:
            return False
        added = [
            (r, c)
            for r, c in np.argwhere(expected == int(guide["color"]))
            if int(inp[r, c]) != int(guide["color"])
        ]
        if not added:
            return False
        for copy in guide["copies"]:
            if bbox_intersection(copy["bbox"], case["frame"]["bbox"]) is not None:
                frame_hits.append((int(gi), int(copy["k"])))

    if frame_hits != [(int(case["frame"]["target_guide"]), int(case["frame"]["target_k"]))]:
        return False

    overlayless = simulate_case_output(case, overlay_mode="bbox", max_k=1)
    if np.array_equal(expected, overlayless):
        return False

    no_pattern = build_input_grid(case)
    for guide in case["guides"]:
        for copy in guide["copies"]:
            paint_cells(no_pattern, copy["visible_coords"], guide["color"])
    paint_cells(no_pattern, case["center_abs"], CENTER_COLOR)
    paint_cells(no_pattern, case["frame"]["border"], FRAME_COLOR)
    if np.array_equal(expected, no_pattern):
        return False

    shape_overlay = simulate_case_output(case, overlay_mode="shape")
    if np.array_equal(expected, shape_overlay):
        return False

    blank_cell = case["frame"]["blank_overlap_cell"]
    filled_cell = case["frame"]["filled_overlap_cell"]
    target_color = int(case["guides"][case["frame"]["target_guide"]]["color"])
    if int(expected[blank_cell]) != target_color or int(expected[filled_cell]) != target_color:
        return False
    if int(shape_overlay[blank_cell]) == target_color:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        case = build_case()
        if case is None:
            continue
        if validate_case(case):
            return build_input_grid(case)
    raise RuntimeError("failed to generate a valid displacement-tiling frame-overlay puzzle")
