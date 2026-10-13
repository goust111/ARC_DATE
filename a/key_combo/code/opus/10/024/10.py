from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Coord = Tuple[int, int]
BBox = Tuple[int, int, int, int]

BLUE = 1
PALETTE = (1, 2, 3, 4)
NON_BLUE = (2, 3, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

HALF_REGIONS = ("lt", "lb", "c", "rt", "rb")
HALF_EDGES = ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4))


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _perpendiculars(direction: Coord) -> Tuple[Coord, Coord]:
    dr, dc = direction
    if dr == 0:
        return (-1, 0), (1, 0)
    return (0, -1), (0, 1)


def _marker_cells(center: Coord, direction: Coord) -> Tuple[Coord, Coord, Coord]:
    r, c = center
    (p1r, p1c), (p2r, p2c) = _perpendiculars(direction)
    dr, dc = direction
    stem = (r + dr, c + dc)
    diag1 = (r - dr + p1r, c - dc + p1c)
    diag2 = (r - dr + p2r, c - dc + p2c)
    return stem, diag1, diag2


def _label_mask(mask: np.ndarray) -> Tuple[np.ndarray, List[List[Coord]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: List[List[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            cid = len(components)
            q = deque([(r, c)])
            labels[r, c] = cid
            cells: List[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and labels[nr, nc] < 0
                    ):
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            components.append(cells)
    return labels, components


def _bbox(cells: Sequence[Coord]) -> BBox:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(cells: Sequence[Coord]) -> int:
    r0, c0, r1, c1 = _bbox(cells)
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _build_component_map(grid: np.ndarray, blocked: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    h, w = grid.shape
    comp_id = np.full((h, w), -1, dtype=int)
    comp_color: List[int] = []
    for color in PALETTE:
        labels, comps = _label_mask((grid == color) & (~blocked))
        for local_id, cells in enumerate(comps):
            cid = len(comp_color)
            comp_color.append(color)
            for r, c in cells:
                comp_id[r, c] = cid
    return comp_id, comp_color


def _is_singleton_blue(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLUE:
        return False
    return all(
        not _in_bounds(grid.shape, r + dr, c + dc) or int(grid[r + dr, c + dc]) != BLUE
        for dr, dc in DIRS4
    )


def _resolve_marker_cover(candidates: List[dict], singletons: set[Coord]) -> List[dict]:
    by_cell: Dict[Coord, List[int]] = {cell: [] for cell in singletons}
    for idx, candidate in enumerate(candidates):
        for cell in candidate["cells"]:
            if cell in by_cell:
                by_cell[cell].append(idx)

    best: List[dict] | None = None

    def dfs(remaining: set[Coord], chosen: List[int]) -> None:
        nonlocal best
        if not remaining:
            if best is None or len(chosen) < len(best):
                best = [candidates[i] for i in chosen]
            return
        if best is not None and len(chosen) >= len(best):
            return
        cell = min(remaining, key=lambda xy: len(by_cell.get(xy, [])))
        for idx in by_cell.get(cell, []):
            marker_cells = candidates[idx]["cells"]
            if not marker_cells <= remaining:
                continue
            dfs(remaining - marker_cells, chosen + [idx])

    dfs(set(singletons), [])
    return [] if best is None else best


def _detect_markers(grid: np.ndarray) -> List[dict]:
    singleton_blue = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if _is_singleton_blue(grid, r, c)
    }
    if not singleton_blue:
        return []

    blocked = np.zeros_like(grid, dtype=bool)
    for r, c in singleton_blue:
        blocked[r, c] = True
    comp_id, comp_color = _build_component_map(grid, blocked)

    candidates: List[dict] = []
    seen: set[Tuple[frozenset[Coord], Coord]] = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            for direction in DIRS4:
                stem, diag1, diag2 = _marker_cells((r, c), direction)
                cells = {stem, diag1, diag2}
                if not all(_in_bounds(grid.shape, rr, cc) for rr, cc in cells):
                    continue
                if not cells <= singleton_blue:
                    continue
                (p1r, p1c), (p2r, p2c) = _perpendiculars(direction)
                dr, dc = direction
                source_probe_1 = (r - 2 * dr + p1r, c - 2 * dc + p1c)
                source_probe_2 = (r - 2 * dr + p2r, c - 2 * dc + p2c)
                target_probe = (r + 2 * dr, c + 2 * dc)
                probes = (source_probe_1, source_probe_2, target_probe)
                if not all(_in_bounds(grid.shape, rr, cc) for rr, cc in probes):
                    continue
                src1 = int(comp_id[source_probe_1])
                src2 = int(comp_id[source_probe_2])
                dst = int(comp_id[target_probe])
                if src1 < 0 or src2 < 0 or dst < 0 or src1 != src2 or src1 == dst:
                    continue
                key = (frozenset(cells), direction)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "center": (r, c),
                        "direction": direction,
                        "cells": frozenset(cells),
                        "source_comp": src1,
                        "target_comp": dst,
                    }
                )

    markers = _resolve_marker_cover(candidates, singleton_blue)
    if markers and set().union(*(m["cells"] for m in markers)) == singleton_blue:
        return markers
    return []


def _final_component_colors(comp_color: List[int], markers: List[dict]) -> List[int]:
    redirect: Dict[int, int] = {}
    for marker in markers:
        target = int(marker["target_comp"])
        source = int(marker["source_comp"])
        if target in redirect and redirect[target] != source:
            raise ValueError("ambiguous target recolor")
        redirect[target] = source

    memo: Dict[int, int] = {}
    visiting: set[int] = set()

    def resolve(cid: int) -> int:
        if cid in memo:
            return memo[cid]
        if cid in visiting:
            raise ValueError("cyclic control dependency")
        visiting.add(cid)
        nxt = redirect.get(cid)
        color = int(comp_color[cid]) if nxt is None else resolve(nxt)
        visiting.remove(cid)
        memo[cid] = color
        return color

    return [resolve(cid) for cid in range(len(comp_color))]


def _apply_controls(grid: np.ndarray, *, erase_markers: bool) -> np.ndarray:
    markers = _detect_markers(grid)
    if not markers:
        return grid.copy()

    control = np.zeros_like(grid, dtype=bool)
    for marker in markers:
        for r, c in marker["cells"]:
            control[r, c] = True
    comp_id, comp_color = _build_component_map(grid, control)

    # Rebind source/target ids using the final control mask to avoid accidental reliance
    # on non-marker blue singletons.
    rebound: List[dict] = []
    for marker in markers:
        r, c = marker["center"]
        dr, dc = marker["direction"]
        (p1r, p1c), (p2r, p2c) = _perpendiculars(marker["direction"])
        src1 = int(comp_id[r - 2 * dr + p1r, c - 2 * dc + p1c])
        src2 = int(comp_id[r - 2 * dr + p2r, c - 2 * dc + p2c])
        dst = int(comp_id[r + 2 * dr, c + 2 * dc])
        if src1 < 0 or src2 < 0 or dst < 0 or src1 != src2 or src1 == dst:
            raise ValueError("invalid control marker geometry")
        rebound.append(
            {
                "cells": marker["cells"],
                "source_comp": src1,
                "target_comp": dst,
            }
        )

    final_color = _final_component_colors(comp_color, rebound)
    out = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = final_color[cid]
    if erase_markers:
        for marker in rebound:
            color = final_color[int(marker["source_comp"])]
            for r, c in marker["cells"]:
                out[r, c] = color
    return out


def _dominant_detail(half_grid: np.ndarray) -> dict:
    colors = [int(v) for v in np.unique(half_grid) if int(v) != 0]
    counts = {color: int(np.sum(half_grid == color)) for color in colors}
    best_count = max(counts.values())
    tied = [color for color, count in counts.items() if count == best_count]
    bbox_scores: Dict[int, int] = {}
    for color in tied:
        labels, comps = _label_mask(half_grid == color)
        bbox_scores[color] = max(_bbox_area(comp) for comp in comps)
    best_bbox = max(bbox_scores.values())
    winners = [color for color in tied if bbox_scores[color] == best_bbox]
    return {
        "color": min(winners),
        "counts": counts,
        "count_tie": len(tied) > 1,
        "bbox_scores": bbox_scores,
        "bbox_tie": len(winners) > 1,
    }


def _draw_split_border(grid: np.ndarray, top_color: int, bottom_color: int) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    mid = h // 2
    out[0, :] = top_color
    out[:mid, 0] = top_color
    out[:mid, w - 1] = top_color
    out[h - 1, :] = bottom_color
    out[mid:, 0] = bottom_color
    out[mid:, w - 1] = bottom_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor T-marked target regions, erase the controls, then draw a split-color outer border from the post-recolor halfwise dominant colors.

    Input structure:
    - The grid is fully tiled by connected monochrome regions using colors 1-4, with no black background.
    - One or two blue 3-pixel control markers sit on region boundaries; each marker has one stem pixel and two diagonal source pixels.
    - The grid height is even, so the canvas has a top half and a bottom half.

    Rule:
    - For each blue control marker, use the stem direction to find the adjacent target region and use the two source-side diagonals to identify the source region.
    - Recolor every target region to the source region's color, then replace each marker pixel with that source region's final color.
    - On the cleaned post-recolor grid, find the dominant non-black color in the top half and in the bottom half; count ties are broken by the largest connected region bbox area within that half.
    - Draw a 1-pixel outer border whose top-half edge pixels use the top dominant color and whose bottom-half edge pixels use the bottom dominant color.

    Key insight:
    - The border colors are not read directly from the original mosaic: the control markers first alter the region-color totals, and those updated halfwise totals determine the final split border.
    """
    grid = np.asarray(grid, dtype=int)
    recolored = _apply_controls(grid, erase_markers=True)
    mid = recolored.shape[0] // 2
    top_color = _dominant_detail(recolored[:mid, :])["color"]
    bottom_color = _dominant_detail(recolored[mid:, :])["color"]
    return _draw_split_border(recolored, top_color, bottom_color)


def _all_assignments() -> List[Dict[str, int]]:
    assignments: List[Dict[str, int]] = []
    for lt in PALETTE:
        for lb in PALETTE:
            for center in PALETTE:
                for rt in PALETTE:
                    for rb in PALETTE:
                        values = (lt, lb, center, rt, rb)
                        if any(values[a] == values[b] for a, b in HALF_EDGES):
                            continue
                        assignments.append(dict(zip(HALF_REGIONS, values)))
    return assignments


ASSIGNMENTS = _all_assignments()


def _random_partition(total: int, pieces: int, minimum: int) -> List[int]:
    extras = total - pieces * minimum
    cuts = sorted(random.sample(range(extras + pieces - 1), pieces - 1))
    out: List[int] = []
    prev = -1
    for cut in cuts + [extras + pieces - 1]:
        out.append(cut - prev)
        prev = cut
    return [part + minimum - 1 for part in out]


def _half_boxes(row0: int, half_h: int, widths: Sequence[int], split_h: int) -> Dict[str, BBox]:
    c1, c2, c3 = map(int, widths)
    top_h = int(split_h)
    bot_h = half_h - top_h
    c12 = c1 + c2
    return {
        "lt": (row0, 0, row0 + top_h - 1, c1 - 1),
        "lb": (row0 + top_h, 0, row0 + half_h - 1, c1 - 1),
        "c": (row0, c1, row0 + half_h - 1, c12 - 1),
        "rt": (row0, c12, row0 + top_h - 1, c12 + c3 - 1),
        "rb": (row0 + top_h, c12, row0 + half_h - 1, c12 + c3 - 1),
    }


def _paint_regions(shape: Tuple[int, int], boxes: Dict[str, BBox], assignment: Dict[str, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for name, (r0, c0, r1, c1) in boxes.items():
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(assignment[name])
    return grid


def _half_placements(boxes: Dict[str, BBox]) -> List[dict]:
    placements: List[dict] = []

    def add(source: str, target: str, direction: Coord) -> None:
        sr0, sc0, sr1, sc1 = boxes[source]
        tr0, tc0, tr1, tc1 = boxes[target]
        if direction == (0, 1):
            if sc1 + 1 != tc0 or sc1 - sc0 + 1 < 3 or tc1 - tc0 + 1 < 2:
                return
            lo, hi = max(sr0, tr0) + 1, min(sr1, tr1) - 1
            if lo > hi:
                return
        elif direction == (0, -1):
            if tc1 + 1 != sc0 or sc1 - sc0 + 1 < 3 or tc1 - tc0 + 1 < 2:
                return
            lo, hi = max(sr0, tr0) + 1, min(sr1, tr1) - 1
            if lo > hi:
                return
        elif direction == (1, 0):
            if sr1 + 1 != tr0 or sr1 - sr0 + 1 < 3 or tr1 - tr0 + 1 < 2:
                return
            lo, hi = max(sc0, tc0) + 1, min(sc1, tc1) - 1
            if lo > hi:
                return
        else:
            if tr1 + 1 != sr0 or sr1 - sr0 + 1 < 3 or tr1 - tr0 + 1 < 2:
                return
            lo, hi = max(sc0, tc0) + 1, min(sc1, tc1) - 1
            if lo > hi:
                return
        placements.append({"source": source, "target": target, "direction": direction})

    add("lt", "c", (0, 1))
    add("c", "lt", (0, -1))
    add("lb", "c", (0, 1))
    add("c", "lb", (0, -1))
    add("c", "rt", (0, 1))
    add("rt", "c", (0, -1))
    add("c", "rb", (0, 1))
    add("rb", "c", (0, -1))
    add("lt", "lb", (1, 0))
    add("lb", "lt", (-1, 0))
    add("rt", "rb", (1, 0))
    add("rb", "rt", (-1, 0))
    return placements


def _placement_center(boxes: Dict[str, BBox], placement: dict) -> Coord:
    source = placement["source"]
    target = placement["target"]
    direction = placement["direction"]
    sr0, sc0, sr1, sc1 = boxes[source]
    tr0, tc0, tr1, tc1 = boxes[target]
    if direction == (0, 1):
        r = random.randint(max(sr0, tr0) + 1, min(sr1, tr1) - 1)
        return r, sc1
    if direction == (0, -1):
        r = random.randint(max(sr0, tr0) + 1, min(sr1, tr1) - 1)
        return r, sc0
    if direction == (1, 0):
        c = random.randint(max(sc0, tc0) + 1, min(sc1, tc1) - 1)
        return sr1, c
    c = random.randint(max(sc0, tc0) + 1, min(sc1, tc1) - 1)
    return sr0, c


def _half_profile(
    half_shape: Tuple[int, int],
    local_boxes: Dict[str, BBox],
    assignment: Dict[str, int],
    placement: dict | None,
) -> dict:
    grid = _paint_regions(half_shape, local_boxes, assignment)
    before = _dominant_detail(grid)
    after = grid.copy()
    if placement is not None:
        src = placement["source"]
        dst = placement["target"]
        dst_box = local_boxes[dst]
        after[dst_box[0] : dst_box[2] + 1, dst_box[1] : dst_box[3] + 1] = int(assignment[src])
    detail = _dominant_detail(after)
    return {"before": before, "after": detail, "grid": grid, "after_grid": after}


def _boxes_key(local_boxes: Dict[str, BBox]) -> Tuple[BBox, ...]:
    return tuple(local_boxes[name] for name in HALF_REGIONS)


@lru_cache(maxsize=None)
def _find_half_options_cached(
    half_shape: Tuple[int, int],
    boxes_key: Tuple[BBox, ...],
    require_marker: bool,
    mode: str,
) -> Tuple[dict, ...]:
    local_boxes = dict(zip(HALF_REGIONS, boxes_key))
    placements = _half_placements(local_boxes)
    options: List[dict] = []
    candidates = placements if require_marker else [None] + placements
    for placement in candidates:
        for assignment in ASSIGNMENTS:
            if BLUE not in assignment.values():
                continue
            if placement is not None:
                if assignment[placement["source"]] not in NON_BLUE:
                    continue
                if assignment[placement["target"]] not in NON_BLUE:
                    continue
            profile = _half_profile(half_shape, local_boxes, assignment, placement)
            before = profile["before"]
            after = profile["after"]
            if placement is None:
                if require_marker:
                    continue
            else:
                if before["color"] == after["color"]:
                    continue
            if mode == "tie":
                if not after["count_tie"] or after["bbox_tie"]:
                    continue
            options.append(
                {
                    "assignment": assignment,
                    "placement": placement,
                    "before": before,
                    "after": after,
                }
            )
    return tuple(options)


def _find_half_options(
    half_shape: Tuple[int, int],
    local_boxes: Dict[str, BBox],
    require_marker: bool,
    mode: str,
) -> List[dict]:
    return list(_find_half_options_cached(half_shape, _boxes_key(local_boxes), require_marker, mode))


def _overlay_marker(grid: np.ndarray, boxes: Dict[str, BBox], placement: dict) -> dict:
    center = _placement_center(boxes, placement)
    stem, diag1, diag2 = _marker_cells(center, placement["direction"])
    for r, c in (stem, diag1, diag2):
        grid[r, c] = BLUE
    return {"center": center, "direction": placement["direction"], "cells": {stem, diag1, diag2}}


def _assemble_case() -> Tuple[np.ndarray, dict]:
    productive_layouts = (
        (12, 12, (3, 3, 6), 3, 3),
        (12, 13, (3, 4, 6), 3, 3),
        (14, 14, (4, 4, 6), 3, 4),
        (16, 15, (4, 5, 6), 4, 4),
    )
    for _ in range(120):
        height, width, widths, top_split, bot_split = random.choice(productive_layouts)
        half_h = height // 2

        top_local = _half_boxes(0, half_h, widths, top_split)
        bot_local = _half_boxes(0, half_h, widths, bot_split)
        top_global = _half_boxes(0, half_h, widths, top_split)
        bot_global = _half_boxes(half_h, half_h, widths, bot_split)

        top_mode = random.choice(["flip", "flip", "tie"])
        top_options = _find_half_options((half_h, width), top_local, True, top_mode)
        if not top_options:
            continue
        top_choice = random.choice(top_options)

        want_second = random.random() < 0.55
        bottom_mode = random.choice(["flip", "tie"]) if want_second else "flip"
        bottom_options = _find_half_options((half_h, width), bot_local, want_second, bottom_mode)
        bottom_options = [
            option
            for option in bottom_options
            if option["after"]["color"] != top_choice["after"]["color"]
            and top_choice["assignment"]["lb"] != option["assignment"]["lt"]
            and top_choice["assignment"]["c"] != option["assignment"]["c"]
            and top_choice["assignment"]["rb"] != option["assignment"]["rt"]
        ]
        if not bottom_options:
            continue
        bottom_choice = random.choice(bottom_options)

        base = np.zeros((height, width), dtype=int)
        top_half = _paint_regions((half_h, width), top_local, top_choice["assignment"])
        bot_half = _paint_regions((half_h, width), bot_local, bottom_choice["assignment"])
        base[:half_h, :] = top_half
        base[half_h:, :] = bot_half
        if set(np.unique(base)) != set(PALETTE):
            continue

        inp = base.copy()
        markers = []
        markers.append(_overlay_marker(inp, top_global, top_choice["placement"]))
        if bottom_choice["placement"] is not None:
            markers.append(_overlay_marker(inp, bot_global, bottom_choice["placement"]))

        after_clean = _apply_controls(inp, erase_markers=True)
        after_dirty = _apply_controls(inp, erase_markers=False)
        if np.array_equal(after_clean, inp):
            continue

        top_before = _dominant_detail(base[:half_h, :])
        bot_before = _dominant_detail(base[half_h:, :])
        top_after = _dominant_detail(after_clean[:half_h, :])
        bot_after = _dominant_detail(after_clean[half_h:, :])
        if top_after["color"] == bot_after["color"]:
            continue
        if top_before["color"] == top_after["color"] and bot_before["color"] == bot_after["color"]:
            continue
        if top_mode == "tie" and (not top_after["count_tie"] or top_after["bbox_tie"]):
            continue
        if want_second and bottom_choice["placement"] is not None and bottom_mode == "tie":
            if not bot_after["count_tie"] or bot_after["bbox_tie"]:
                continue

        erase_sensitive = (
            _dominant_detail(after_dirty[:half_h, :])["color"] != top_after["color"]
            or _dominant_detail(after_dirty[half_h:, :])["color"] != bot_after["color"]
        )
        return inp, {
            "marker_count": len(markers),
            "top_mode": top_mode,
            "bottom_mode": bottom_mode if bottom_choice["placement"] is not None else "none",
            "erase_sensitive": erase_sensitive,
        }
    raise RuntimeError("failed to generate a valid sample")


def generate_input() -> np.ndarray:
    for _ in range(80):
        grid, _meta = _assemble_case()
        if _detect_markers(grid):
            return grid
    raise RuntimeError("generator exhausted retries")
