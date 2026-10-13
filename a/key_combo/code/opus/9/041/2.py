from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3

INPUT_MIN_SIDE = 6
INPUT_MAX_SIDE = 9
MIN_INPUT_POINTS = 4
MAX_INPUT_POINTS = 8
GEN_RETRIES = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_cells(out_shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = out_shape
    cells: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += 1
        cc += 1
    return cells


def _diagonal_offset_copy(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((2 * h, 2 * w), BLACK, dtype=int)
    for r, c in np.argwhere(g == GREEN):
        for rr, cc in _trace_cells(out.shape, int(r), int(c)):
            out[rr, cc] = GREEN
    return out


def _classify_components(raw: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[tuple[int, int]]]:
    comps = _components(raw == GREEN)
    classified = np.full_like(raw, BLACK)
    anchors: list[list[tuple[int, int]]] = []
    singletons: list[tuple[int, int]] = []
    for comp in comps:
        if len(comp) >= 2:
            anchors.append(comp)
            for r, c in comp:
                classified[r, c] = BLUE
        else:
            r, c = comp[0]
            singletons.append((r, c))
            classified[r, c] = GREEN
    return classified, anchors, singletons


def _marker_target(
    marker: tuple[int, int], anchors: list[list[tuple[int, int]]]
) -> dict | None:
    r, c = marker
    per_anchor: list[dict] = []
    for idx, comp in enumerate(anchors):
        best_for_anchor: dict | None = None
        for ar, ac in comp:
            if ar == r:
                dist = abs(ac - c)
                if dist > 0:
                    step = 1 if ac > c else -1
                    cand = {
                        "anchor_idx": idx,
                        "dist": dist,
                        "end": (r, ac - step),
                        "orientation": "horizontal",
                    }
                    if best_for_anchor is None or cand["dist"] < best_for_anchor["dist"]:
                        best_for_anchor = cand
            if ac == c:
                dist = abs(ar - r)
                if dist > 0:
                    step = 1 if ar > r else -1
                    cand = {
                        "anchor_idx": idx,
                        "dist": dist,
                        "end": (ar - step, c),
                        "orientation": "vertical",
                    }
                    if best_for_anchor is None or cand["dist"] < best_for_anchor["dist"]:
                        best_for_anchor = cand
        if best_for_anchor is not None:
            per_anchor.append(best_for_anchor)

    if not per_anchor:
        return None

    best_dist = min(item["dist"] for item in per_anchor)
    tied = [item for item in per_anchor if item["dist"] == best_dist]
    assert len(tied) == 1, "nearest aligned anchor must be unique"
    chosen = tied[0].copy()
    chosen["candidate_count"] = len(per_anchor)
    return chosen


def _paint_ray(out: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    end_r, end_c = end
    cells: list[tuple[int, int]] = []
    if r == end_r:
        step = 1 if end_c > c else -1
        for cc in range(c + step, end_c + step, step):
            out[r, cc] = GREEN
            cells.append((r, cc))
    elif c == end_c:
        step = 1 if end_r > r else -1
        for rr in range(r + step, end_r + step, step):
            out[rr, c] = GREEN
            cells.append((rr, c))
    else:
        raise AssertionError("ray must be axis-aligned")
    return cells


def _analyze(grid: np.ndarray) -> dict:
    raw = _diagonal_offset_copy(grid)
    classified, anchors, singletons = _classify_components(raw)
    out = classified.copy()
    marker_infos: list[dict] = []
    for marker in sorted(singletons):
        target = _marker_target(marker, anchors)
        info = {"coord": marker, "target": target, "ray_cells": []}
        if target is not None:
            info["ray_cells"] = _paint_ray(out, marker, target["end"])
        marker_infos.append(info)
    return {
        "raw": raw,
        "classified": classified,
        "output": out,
        "anchors": anchors,
        "singletons": singletons,
        "marker_infos": marker_infos,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diagonally tile the sparse green motif, recolor emergent multi-pixel components as anchors, and extend aligned singleton rays toward the nearest anchor.

    Input structure:
    - The input is a small black canvas containing only isolated green pixels; no two green pixels are 4-neighbor adjacent.
    - Green pixels do not carry fixed roles in the input: every pixel starts as the same kind of sparse seed.
    - The only geometry cue is the set of seed positions under the standard `(row, col)` coordinate system.

    Rule:
    - Double the canvas height and width, fill it with black, and copy each green seed along the down-right diagonal at offsets `(r+k, c+k)` while in bounds.
    - On that doubled canvas, compute 4-connected green components; recolor every component of area at least 2 to blue, while keeping every area-1 component green.
    - For each remaining green singleton, if some blue anchor component has a pixel in the same row or the same column, connect the singleton to the nearest such anchor by a straight green ray that stops in the cell just before the anchor.
    - If a singleton is aligned with multiple anchors, use the nearest anchor component; if it is aligned with none, leave it unchanged.

    Key insight:
    - Anchor versus marker status is not visible in the input itself: it emerges only after diagonal offset copying creates new 4-connected overlaps, and the later ray step must use that emergent classification rather than the original seed positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 15, "input must stay small enough that doubling remains ARC-sized"
    assert set(map(int, np.unique(g))) <= {BLACK, GREEN}, "expected only black and green in the input"

    fg = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert MIN_INPUT_POINTS <= len(fg) <= MAX_INPUT_POINTS, "unexpected number of foreground pixels"
    diags = {r - c for r, c in fg}
    assert len(diags) == len(fg), "each input pixel must define a distinct diagonal trace"
    for r, c in fg:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                assert not (g[nr, nc] == GREEN), "input green pixels must be isolated under 4-connectivity"

    meta = _analyze(g)
    assert len(meta["anchors"]) >= 1, "tiling must create at least one multi-pixel anchor component"
    assert len(meta["singletons"]) >= 2, "tiling must leave at least two singleton marker pixels"
    return meta["output"]


def _can_add_point(points: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    diag = r - c
    occupied = set(points)
    if candidate in occupied:
        return False
    if diag in {pr - pc for pr, pc in points}:
        return False
    for pr, pc in points:
        if abs(pr - r) + abs(pc - c) == 1:
            return False
    return True


def _sample_anchor_pair(
    h: int, w: int, points: list[tuple[int, int]], upper_bias: bool
) -> list[tuple[int, int]] | None:
    row_limit = max(1, h - 3)
    col_limit = max(1, w - 3)
    for _ in range(120):
        if upper_bias:
            r = int(np.random.randint(0, row_limit))
            c = int(np.random.randint(0, col_limit))
        else:
            r = int(np.random.randint(1, h - 2))
            c = int(np.random.randint(1, w - 2))
        gap = int(np.random.randint(2, min(5, h + 1)))
        variant = int(np.random.randint(2))
        if variant == 0:
            p1 = (r, c)
            p2 = (r + gap, c + gap - 1)
        else:
            p1 = (r, c)
            p2 = (r + gap - 1, c + gap)
        if not (0 <= p1[0] < h and 0 <= p1[1] < w and 0 <= p2[0] < h and 0 <= p2[1] < w):
            continue
        if _can_add_point(points, p1) and _can_add_point(points + [p1], p2):
            return [p1, p2]
    return None


def _sample_extra_point(
    h: int, w: int, points: list[tuple[int, int]], prefer_late: bool
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not _can_add_point(points, (r, c)):
                continue
            if prefer_late and r + c < h + w - 4:
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _all_pixels_matter(grid: np.ndarray, target: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    for r, c in coords:
        mutated = grid.copy()
        mutated[r, c] = BLACK
        try:
            alt = main(mutated)
        except Exception:
            continue
        if np.array_equal(alt, target):
            return False
    return True


def _valid_generated_example(
    grid: np.ndarray,
    *,
    want_two_anchors: bool,
    want_unaligned: bool,
    want_multi_choice: bool,
    want_orientation: str,
) -> bool:
    meta = _analyze(grid)
    raw = meta["raw"]
    classified = meta["classified"]
    out = meta["output"]
    anchors = meta["anchors"]
    infos = meta["marker_infos"]

    if len(anchors) < 1 or len(meta["singletons"]) < 2:
        return False
    if want_two_anchors and len(anchors) < 2:
        return False
    if np.array_equal(raw, classified):
        return False
    if np.array_equal(classified, out):
        return False

    nontrivial = [info for info in infos if info["ray_cells"]]
    if not nontrivial:
        return False
    if want_orientation not in {"horizontal", "vertical"}:
        return False
    if not any(info["target"] and info["target"]["orientation"] == want_orientation and info["ray_cells"] for info in infos):
        return False

    if want_unaligned and not any(info["target"] is None for info in infos):
        return False
    if want_multi_choice and not any(info["target"] and info["target"]["candidate_count"] >= 2 for info in infos):
        return False

    if np.count_nonzero(out == BLUE) < 2:
        return False
    if np.count_nonzero((out == GREEN) & (classified == BLACK)) < 1:
        return False
    if np.count_nonzero(out != BLACK) > out.size // 2:
        return False

    return _all_pixels_matter(grid, out)


def _attempt_generate() -> np.ndarray | None:
    h = int(np.random.randint(INPUT_MIN_SIDE, INPUT_MAX_SIDE + 1))
    w = int(np.random.randint(INPUT_MIN_SIDE, INPUT_MAX_SIDE + 1))
    want_two_anchors = bool(np.random.randint(2))
    want_unaligned = bool(np.random.randint(2))
    want_multi_choice = bool(want_two_anchors and np.random.randint(2))
    want_orientation = "horizontal" if bool(np.random.randint(2)) else "vertical"

    points: list[tuple[int, int]] = []
    pair = _sample_anchor_pair(h, w, points, upper_bias=True)
    if pair is None:
        return None
    points.extend(pair)

    if want_two_anchors:
        pair2 = _sample_anchor_pair(h, w, points, upper_bias=False)
        if pair2 is None:
            return None
        points.extend(pair2)

    target_count = int(np.random.randint(max(4, len(points)), MAX_INPUT_POINTS + 1))
    while len(points) < target_count:
        prefer_late = want_unaligned and len(points) == target_count - 1
        extra = _sample_extra_point(h, w, points, prefer_late=prefer_late)
        if extra is None:
            return None
        points.append(extra)

    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in points:
        grid[r, c] = GREEN

    try:
        result = main(grid)
    except Exception:
        return None
    if result.shape != (2 * h, 2 * w):
        return None

    if not _valid_generated_example(
        grid,
        want_two_anchors=want_two_anchors,
        want_unaligned=want_unaligned,
        want_multi_choice=want_multi_choice,
        want_orientation=want_orientation,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-anchor-ray ARC input")
