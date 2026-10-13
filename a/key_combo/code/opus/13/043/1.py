from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]

ANCHOR_PALETTE = (1, 3, 4, 8)
MARKER_PALETTE = (2, 6, 7, 9)
SIDES = ("left", "right", "up", "down")
HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("up", "down")
GEN_RETRY_LIMIT = 400
PAIR_RETRY_LIMIT = 300
GAP_RANGE = (1, 6)
GRID_RANGE = (14, 19)


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[Coord]:
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "coord_set": set(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    components.sort(key=lambda comp: comp["bbox"])
    return components


def _classify_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    components = _find_components(g, bg)
    anchors: list[dict] = []
    markers: list[dict] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
            if comp["coord_set"] == {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}:
                anchors.append(comp)
                continue
        if comp["area"] == 1:
            markers.append(comp)
            continue
        raise AssertionError("expected only solid 2x2 anchors and isolated marker pixels")

    assert 2 <= len(anchors) <= 4
    assert len(anchors) == len(markers)
    assert len({int(comp["color"]) for comp in anchors}) == len(anchors)
    assert len({int(comp["color"]) for comp in markers}) == len(markers)
    markers.sort(key=lambda comp: comp["coords"][0])
    anchors.sort(key=lambda comp: comp["bbox"][:2])
    return bg, anchors, markers


def _candidate_for_marker_anchor(grid: np.ndarray, bg: int, marker: Coord, anchor: dict) -> dict | None:
    mr, mc = marker
    r0, c0, _r1, _c1 = anchor["bbox"]
    rows = (r0, r0 + 1)
    cols = (c0, c0 + 1)

    if mr in rows:
        if mc < c0:
            gap_cells = tuple((mr, c) for c in range(mc + 1, c0))
            if all(int(grid[r, c]) == bg for r, c in gap_cells):
                return {"side": "left", "gap": len(gap_cells)}
        if mc > c0 + 1:
            gap_cells = tuple((mr, c) for c in range(c0 + 2, mc))
            if all(int(grid[r, c]) == bg for r, c in gap_cells):
                return {"side": "right", "gap": len(gap_cells)}

    if mc in cols:
        if mr < r0:
            gap_cells = tuple((r, mc) for r in range(mr + 1, r0))
            if all(int(grid[r, c]) == bg for r, c in gap_cells):
                return {"side": "up", "gap": len(gap_cells)}
        if mr > r0 + 1:
            gap_cells = tuple((r, mc) for r in range(r0 + 2, mr))
            if all(int(grid[r, c]) == bg for r, c in gap_cells):
                return {"side": "down", "gap": len(gap_cells)}

    return None


def _solve_matching(grid: np.ndarray, bg: int, anchors: list[dict], markers: list[dict]) -> dict:
    candidate_maps: list[list[dict]] = []
    for marker_index, marker in enumerate(markers):
        marker_coord = marker["coords"][0]
        candidates: list[dict] = []
        for anchor_index, anchor in enumerate(anchors):
            candidate = _candidate_for_marker_anchor(grid, bg, marker_coord, anchor)
            if candidate is not None:
                candidate = dict(candidate)
                candidate["anchor_index"] = anchor_index
                candidate["marker_index"] = marker_index
                candidates.append(candidate)
        assert candidates, "each marker must see at least one anchor"
        candidate_maps.append(candidates)

    solutions: list[tuple[dict, ...]] = []

    def backtrack(marker_index: int, used_anchors: set[int], current: list[dict]) -> None:
        if marker_index == len(candidate_maps):
            solutions.append(tuple(current))
            return
        for candidate in candidate_maps[marker_index]:
            anchor_index = int(candidate["anchor_index"])
            if anchor_index in used_anchors:
                continue
            used_anchors.add(anchor_index)
            current.append(candidate)
            backtrack(marker_index + 1, used_anchors, current)
            current.pop()
            used_anchors.remove(anchor_index)
            if len(solutions) > 1:
                return

    backtrack(0, set(), [])
    assert len(solutions) == 1, "marker-anchor assignment must be unique"
    marker_to_anchor = tuple(solutions[0])
    anchor_to_marker = {
        int(candidate["anchor_index"]): candidate for candidate in marker_to_anchor
    }
    assert len(anchor_to_marker) == len(anchors)
    return {
        "marker_candidates": tuple(tuple(cands) for cands in candidate_maps),
        "anchor_to_marker": anchor_to_marker,
    }


def _ray_segments(shape: tuple[int, int], anchor: dict, side: str) -> list[tuple[Coord, Coord]]:
    h, w = shape
    r0, c0, _r1, _c1 = anchor["bbox"]
    if side == "left":
        return [((r0, c), (r0 + 1, c)) for c in range(c0 + 2, w)]
    if side == "right":
        return [((r0, c), (r0 + 1, c)) for c in range(c0 - 1, -1, -1)]
    if side == "up":
        return [((r, c0), (r, c0 + 1)) for r in range(r0 + 2, h)]
    if side == "down":
        return [((r, c0), (r, c0 + 1)) for r in range(r0 - 1, -1, -1)]
    raise ValueError(f"unknown side: {side}")


def _render_from_analysis(grid: np.ndarray, anchors: list[dict], markers: list[dict], assignment: dict, bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for marker in markers:
        mr, mc = marker["coords"][0]
        out[mr, mc] = bg

    for anchor_index, anchor in enumerate(anchors):
        candidate = assignment["anchor_to_marker"][anchor_index]
        marker = markers[int(candidate["marker_index"])]
        ray = _ray_segments(out.shape, anchor, str(candidate["side"]))
        gap = int(candidate["gap"])
        cutoff = max(0, len(ray) - gap)
        body_color = int(marker["color"])
        tail_color = int(anchor["color"])
        for idx, segment in enumerate(ray):
            color = body_color if idx < cutoff else tail_color
            for r, c in segment:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each marker to the unique 2x2 anchor it sees, then emit an opposite-facing ray whose tail length encodes the marker-anchor gap.

    Input structure
    - The background is black.
    - Foreground objects are 2-4 solid monochrome 2x2 anchor squares and the same number of isolated marker pixels.
    - Every marker has a clear horizontal or vertical background sightline to exactly one anchor side, and every anchor is matched once.
    - No anchor's outgoing ray corridor contains another input object.

    Rule
    - Pair each marker with the unique anchor that it sees along a row or column, measuring the background gap G between the marker and the nearest anchor pixel.
    - For each anchor, shoot a 2-cell-wide ray from the side opposite the matched marker, starting just outside the anchor and extending to the grid boundary.
    - Paint the ray body with the marker color, then repaint the last G ray positions from the far end with the anchor color; if the ray is shorter than G, repaint the whole ray.
    - Remove the marker pixels and keep the anchors unchanged.
    - If rays overlap, later anchors in top-to-bottom, then left-to-right anchor order overwrite earlier rays.

    Key insight
    - The sightline fixes both the pairing and the ray direction, while the geometric gap independently determines how much of the far ray tail changes from marker color to anchor color.
    """
    g = np.asarray(grid, dtype=int)
    bg, anchors, markers = _classify_objects(g)
    assignment = _solve_matching(g, bg, anchors, markers)
    return _render_from_analysis(g, anchors, markers, assignment, bg)


def _pair_geometry(
    h: int,
    w: int,
    anchor_top_left: Coord,
    side: str,
    align: int,
    gap: int,
    anchor_color: int,
    marker_color: int,
) -> dict | None:
    r0, c0 = anchor_top_left
    if not (0 <= r0 < h - 1 and 0 <= c0 < w - 1):
        return None

    anchor_cells = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
    if side == "left":
        marker = (r0 + align, c0 - gap - 1)
        gap_cells = {(r0 + align, c) for c in range(c0 - gap, c0)}
    elif side == "right":
        marker = (r0 + align, c0 + gap + 2)
        gap_cells = {(r0 + align, c) for c in range(c0 + 2, c0 + gap + 2)}
    elif side == "up":
        marker = (r0 - gap - 1, c0 + align)
        gap_cells = {(r, c0 + align) for r in range(r0 - gap, r0)}
    elif side == "down":
        marker = (r0 + gap + 2, c0 + align)
        gap_cells = {(r, c0 + align) for r in range(r0 + 2, r0 + gap + 2)}
    else:
        raise ValueError(f"unknown side: {side}")

    object_cells = set(anchor_cells)
    object_cells.add(marker)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in object_cells | gap_cells):
        return None

    anchor = {"bbox": (r0, c0, r0 + 1, c0 + 1), "color": anchor_color}
    ray = _ray_segments((h, w), anchor, side)
    ray_cells = {cell for segment in ray for cell in segment}
    if object_cells & gap_cells:
        return None
    if object_cells & ray_cells:
        return None
    return {
        "anchor_top_left": (r0, c0),
        "anchor_cells": frozenset(anchor_cells),
        "marker": marker,
        "marker_color": marker_color,
        "anchor_color": anchor_color,
        "side": side,
        "align": align,
        "gap": gap,
        "orientation": "horizontal" if side in HORIZONTAL_SIDES else "vertical",
        "gap_cells": frozenset(gap_cells),
        "ray_cells": frozenset(ray_cells),
        "ray_len": len(ray),
    }


def _proposal_pool(
    h: int,
    w: int,
    side: str,
    anchor_color: int,
    marker_color: int,
    short_ray: bool,
    prefer_body: bool,
) -> list[dict]:
    proposals: list[dict] = []
    for gap in range(GAP_RANGE[0], GAP_RANGE[1] + 1):
        for align in (0, 1):
            for r0 in range(h - 1):
                for c0 in range(w - 1):
                    proposal = _pair_geometry(h, w, (r0, c0), side, align, gap, anchor_color, marker_color)
                    if proposal is None or proposal["ray_len"] <= 0:
                        continue
                    ray_len = int(proposal["ray_len"])
                    if short_ray:
                        if not (ray_len < gap):
                            continue
                    elif prefer_body:
                        if not (ray_len > gap):
                            continue
                    else:
                        if not (ray_len >= gap):
                            continue
                    proposals.append(proposal)
    random.shuffle(proposals)
    return proposals


def _objects_ok_against_existing(proposal: dict, placed: list[dict]) -> bool:
    existing_objects = set()
    reserved_bg = set()
    for pair in placed:
        existing_objects |= set(pair["anchor_cells"])
        existing_objects.add(pair["marker"])
        reserved_bg |= set(pair["gap_cells"])
        reserved_bg |= set(pair["ray_cells"])

    for cell in proposal["anchor_cells"] | {proposal["marker"]}:
        if cell in existing_objects or cell in reserved_bg:
            return False
        if any(neighbor in existing_objects for neighbor in _neighbors8(*cell)):
            return False

    clear_cells = set(proposal["gap_cells"]) | set(proposal["ray_cells"])
    if clear_cells & existing_objects:
        return False
    return True


def _place_pair(
    h: int,
    w: int,
    side: str,
    anchor_color: int,
    marker_color: int,
    placed: list[dict],
    short_ray: bool,
    prefer_body: bool,
    force_overlap: bool,
) -> dict | None:
    proposals = _proposal_pool(h, w, side, anchor_color, marker_color, short_ray, prefer_body)
    for proposal in proposals[:PAIR_RETRY_LIMIT * 10]:
        if not _objects_ok_against_existing(proposal, placed):
            continue
        if force_overlap:
            has_overlap = any(
                proposal["orientation"] != pair["orientation"] and set(proposal["ray_cells"]) & set(pair["ray_cells"])
                for pair in placed
            )
            if not has_overlap:
                continue
        return proposal
    return None


def _build_input(h: int, w: int, pairs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for pair in pairs:
        for r, c in pair["anchor_cells"]:
            grid[r, c] = int(pair["anchor_color"])
        mr, mc = pair["marker"]
        grid[mr, mc] = int(pair["marker_color"])
    return grid


def _meaningful_overlap_exists(anchors: list[dict], markers: list[dict], assignment: dict, shape: tuple[int, int]) -> bool:
    painted: list[dict[Coord, int]] = []
    for anchor_index, anchor in enumerate(anchors):
        candidate = assignment["anchor_to_marker"][anchor_index]
        marker = markers[int(candidate["marker_index"])]
        ray = _ray_segments(shape, anchor, str(candidate["side"]))
        gap = int(candidate["gap"])
        cutoff = max(0, len(ray) - gap)
        cell_colors: dict[Coord, int] = {}
        for idx, segment in enumerate(ray):
            color = int(marker["color"]) if idx < cutoff else int(anchor["color"])
            for cell in segment:
                cell_colors[cell] = color
        painted.append(cell_colors)

    for later in range(len(painted)):
        for earlier in range(later):
            overlap = set(painted[later]) & set(painted[earlier])
            if any(painted[later][cell] != painted[earlier][cell] for cell in overlap):
                return True
    return False


def _validate_instance(grid: np.ndarray, expect_overlap: bool, expect_trunc: bool) -> bool:
    try:
        bg, anchors, markers = _classify_objects(grid)
        assignment = _solve_matching(grid, bg, anchors, markers)
        out = _render_from_analysis(grid, anchors, markers, assignment, bg)
        if not np.array_equal(out, main(grid)):
            return False
    except AssertionError:
        return False

    gaps: list[int] = []
    orientations: set[str] = set()
    has_visible_body = False
    has_truncation = False
    for marker_candidates in assignment["marker_candidates"]:
        if len(marker_candidates) != 1:
            return False
    for anchor_index, anchor in enumerate(anchors):
        candidate = assignment["anchor_to_marker"][anchor_index]
        gap = int(candidate["gap"])
        ray_len = len(_ray_segments(grid.shape, anchor, str(candidate["side"])))
        gaps.append(gap)
        orientations.add("horizontal" if str(candidate["side"]) in HORIZONTAL_SIDES else "vertical")
        has_visible_body = has_visible_body or ray_len > gap
        has_truncation = has_truncation or ray_len < gap

    if orientations != {"horizontal", "vertical"}:
        return False
    if len(set(gaps)) < 2:
        return False
    if max(gaps) < 2:
        return False
    if not has_visible_body:
        return False
    if expect_trunc and not has_truncation:
        return False
    if expect_overlap and not _meaningful_overlap_exists(anchors, markers, assignment, grid.shape):
        return False

    if np.array_equal(out, grid):
        return False
    if len({int(v) for v in np.unique(out)} - {bg}) < len(anchors):
        return False
    return True


def _side_sequence(pair_count: int, expect_overlap: bool) -> list[str]:
    sides = [random.choice(HORIZONTAL_SIDES), random.choice(VERTICAL_SIDES)]
    for _ in range(pair_count - 2):
        sides.append(random.choice(SIDES))
    if expect_overlap:
        return sides
    random.shuffle(sides)
    return sides


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    pair_count = random.choices([2, 3, 4], weights=[2, 4, 4], k=1)[0]
    anchor_colors = random.sample(ANCHOR_PALETTE, pair_count)
    marker_colors = random.sample(MARKER_PALETTE, pair_count)

    expect_overlap = pair_count >= 2 and random.random() < 0.65
    expect_trunc = random.random() < 0.65
    trunc_index = random.randrange(pair_count) if expect_trunc else -1
    sides = _side_sequence(pair_count, expect_overlap)

    placed: list[dict] = []
    for index in range(pair_count):
        force_overlap = expect_overlap and index == 1
        short_ray = index == trunc_index
        prefer_body = not short_ray or random.random() < 0.85
        proposal = _place_pair(
            h,
            w,
            sides[index],
            anchor_colors[index],
            marker_colors[index],
            placed,
            short_ray=short_ray,
            prefer_body=prefer_body,
            force_overlap=force_overlap,
        )
        if proposal is None:
            return None
        placed.append(proposal)

    grid = _build_input(h, w, placed)
    if not _validate_instance(grid, expect_overlap, expect_trunc):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sightline-ray ARC instance")
