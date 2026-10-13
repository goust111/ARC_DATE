from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 0
TARGET_COLOR = 5
LINE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRS = ORTH_DIRS + DIAG_DIRS

GEN_RETRIES = 400
PLACEMENT_TRIES = 250


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochromatic_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append({"color": color, "coords": coords})
    return comps


def _is_target_component(g: np.ndarray, comp: dict) -> bool:
    coords = comp["coords"]
    color = int(comp["color"])
    if len(coords) != 9:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    return bool((patch == color).all())


def _segment_info(comp: dict) -> dict | None:
    color = int(comp["color"])
    coords = sorted((int(r), int(c)) for r, c in comp["coords"])
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    deltas = {r - c for r, c in coords}
    sigmas = {r + c for r, c in coords}

    if len(rows) == 1:
        row = next(iter(rows))
        ordered = sorted(coords, key=lambda p: p[1])
        c0 = ordered[0][1]
        if [c for _r, c in ordered] != list(range(c0, c0 + len(ordered))):
            return None
        return {
            "color": color,
            "length": len(ordered),
            "kind": "orth",
            "axis": "h",
            "row": row,
            "c0": ordered[0][1],
            "c1": ordered[-1][1],
        }

    if len(cols) == 1:
        col = next(iter(cols))
        ordered = sorted(coords, key=lambda p: p[0])
        r0 = ordered[0][0]
        if [r for r, _c in ordered] != list(range(r0, r0 + len(ordered))):
            return None
        return {
            "color": color,
            "length": len(ordered),
            "kind": "orth",
            "axis": "v",
            "col": col,
            "r0": ordered[0][0],
            "r1": ordered[-1][0],
        }

    if len(deltas) == 1:
        delta = next(iter(deltas))
        ordered = sorted(coords, key=lambda p: p[0])
        for (r0, c0), (r1, c1) in zip(ordered, ordered[1:], strict=False):
            if (r1 - r0, c1 - c0) != (1, 1):
                return None
        return {
            "color": color,
            "length": len(ordered),
            "kind": "diag",
            "axis": "d1",
            "delta": delta,
            "r0": ordered[0][0],
            "r1": ordered[-1][0],
        }

    if len(sigmas) == 1:
        sigma = next(iter(sigmas))
        ordered = sorted(coords, key=lambda p: p[0])
        for (r0, c0), (r1, c1) in zip(ordered, ordered[1:], strict=False):
            if (r1 - r0, c1 - c0) != (1, -1):
                return None
        return {
            "color": color,
            "length": len(ordered),
            "kind": "diag",
            "axis": "d2",
            "sigma": sigma,
            "r0": ordered[0][0],
            "r1": ordered[-1][0],
        }

    return None


def _distance_to_target(seg: dict, center: tuple[int, int]) -> int | None:
    r, c = map(int, center)
    axis = seg["axis"]
    if axis == "h":
        if r != int(seg["row"]):
            return None
        if c < int(seg["c0"]):
            return int(seg["c0"]) - c
        if c > int(seg["c1"]):
            return c - int(seg["c1"])
        return None

    if axis == "v":
        if c != int(seg["col"]):
            return None
        if r < int(seg["r0"]):
            return int(seg["r0"]) - r
        if r > int(seg["r1"]):
            return r - int(seg["r1"])
        return None

    if axis == "d1":
        if r - c != int(seg["delta"]):
            return None
        if r < int(seg["r0"]):
            return int(seg["r0"]) - r
        if r > int(seg["r1"]):
            return r - int(seg["r1"])
        return None

    if r + c != int(seg["sigma"]):
        return None
    if r < int(seg["r0"]):
        return int(seg["r0"]) - r
    if r > int(seg["r1"]):
        return r - int(seg["r1"])
    return None


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[int, list[dict]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(int(v) for v in g.ravel()).most_common(1)[0][0])
    components = _monochromatic_components(g, bg)

    targets: list[dict] = []
    segments: list[dict] = []
    for comp in components:
        if _is_target_component(g, comp):
            r0, c0, r1, c1 = _bbox(comp["coords"])
            targets.append(
                {
                    "color": int(comp["color"]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "center": (int(r0 + 1), int(c0 + 1)),
                }
            )
        else:
            info = _segment_info(comp)
            assert info is not None, "non-target foreground objects must be straight segments"
            segments.append(info)

    assert targets, "expected at least one 3x3 target block"
    centers = [tuple(map(int, target["center"])) for target in targets]
    by_target: dict[int, list[dict]] = {idx: [] for idx in range(len(targets))}

    for seg in segments:
        candidates: list[tuple[int, int]] = []
        for idx, center in enumerate(centers):
            dist = _distance_to_target(seg, center)
            if dist is not None:
                candidates.append((int(dist), int(idx)))
        assert candidates, "each segment must point to a target"
        best_dist = min(dist for dist, _idx in candidates)
        best = [idx for dist, idx in candidates if dist == best_dist]
        assert len(best) == 1, "segment-to-target match must be unambiguous"
        by_target[best[0]].append(seg)

    for idx in range(len(targets)):
        assert len(by_target[idx]) == 2, "each target must be pointed to by exactly two segments"

    return g, targets, by_target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray 3x3 target block using the colors of the two line segments that point to it.

    Input structure:
    - The background is black.
    - Several solid gray 3x3 squares act as target regions and do not touch each other.
    - Other foreground objects are monochrome straight line segments: horizontal, vertical,
      or 45-degree diagonal, and each segment's supporting line passes through one target center.
    - Every target is associated with exactly two differently colored segments.

    Rule:
    - Keep every line segment unchanged.
    - For each target, identify the two segments whose extension reaches its center and count
      the number of pixels in each segment.
    - If one segment is longer, use its color on the target's outer ring and use the shorter
      segment's color on the center pixel.
    - If the lengths tie, the horizontal/vertical segment colors the outer ring and the
      diagonal segment colors the center pixel.

    Key insight:
    The scattered colored pixels are not decorative fragments: they are directional arrows
    aimed at gray targets, and the target fill depends on comparing arrow lengths rather than
    on any fixed color identity.
    """
    g, targets, by_target = _analyze_grid(grid)
    out = g.copy()

    for idx, target in enumerate(targets):
        seg_a, seg_b = by_target[idx]
        if int(seg_a["length"]) > int(seg_b["length"]):
            outer, inner = int(seg_a["color"]), int(seg_b["color"])
        elif int(seg_b["length"]) > int(seg_a["length"]):
            outer, inner = int(seg_b["color"]), int(seg_a["color"])
        else:
            if seg_a["kind"] == "orth" and seg_b["kind"] == "diag":
                outer, inner = int(seg_a["color"]), int(seg_b["color"])
            elif seg_b["kind"] == "orth" and seg_a["kind"] == "diag":
                outer, inner = int(seg_b["color"]), int(seg_a["color"])
            else:
                raise AssertionError("tie case requires one orthogonal and one diagonal segment")

        r0, c0, r1, c1 = map(int, target["bbox"])
        cr, cc = map(int, target["center"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(outer)
        out[cr, cc] = int(inner)

    return out


def _segment_fill(colors: tuple[int, int], kinds: tuple[str, str], lengths: tuple[int, int]) -> tuple[int, int]:
    if lengths[0] > lengths[1]:
        return int(colors[0]), int(colors[1])
    if lengths[1] > lengths[0]:
        return int(colors[1]), int(colors[0])
    if kinds[0] == "orth" and kinds[1] == "diag":
        return int(colors[0]), int(colors[1])
    if kinds[1] == "orth" and kinds[0] == "diag":
        return int(colors[1]), int(colors[0])
    raise AssertionError("tie case must mix orthogonal and diagonal segments")


def _build_cluster(colors: tuple[int, int], size_bias: int) -> dict:
    tie_case = bool(random.getrandbits(1))
    if tie_case:
        dirs = [random.choice(ORTH_DIRS), random.choice(DIAG_DIRS)]
    else:
        dirs = random.sample(ALL_DIRS, 2)

    max_len = 4 if size_bias >= 3 else 5
    max_gap = 2 if size_bias >= 3 else 3
    min_len = 2
    min_gap = 1

    if tie_case:
        length = random.randint(min_len, max_len)
        lengths = [length, length]
    else:
        lengths = [random.randint(min_len, max_len), random.randint(min_len, max_len)]
        while lengths[0] == lengths[1]:
            lengths[1] = random.randint(min_len, max_len)

    gaps = [random.randint(min_gap, max_gap), random.randint(min_gap, max_gap)]

    cells: dict[tuple[int, int], int] = {}
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            cells[(dr, dc)] = TARGET_COLOR

    kinds: list[str] = []
    segments: list[dict] = []
    for color, direction, length, gap in zip(colors, dirs, lengths, gaps, strict=False):
        kind = "orth" if direction in ORTH_DIRS else "diag"
        kinds.append(kind)
        start = 2 + gap
        for step in range(start, start + length):
            cell = (int(direction[0] * step), int(direction[1] * step))
            assert cell not in cells
            cells[cell] = int(color)
        segments.append(
            {
                "color": int(color),
                "dir": tuple(map(int, direction)),
                "kind": kind,
                "length": int(length),
            }
        )

    rr = [r for r, _c in cells]
    cc = [c for _r, c in cells]
    outer_color, inner_color = _segment_fill(
        (int(colors[0]), int(colors[1])),
        (kinds[0], kinds[1]),
        (int(lengths[0]), int(lengths[1])),
    )
    return {
        "cells": cells,
        "bbox": (min(rr), min(cc), max(rr), max(cc)),
        "segments": segments,
        "fill": (int(outer_color), int(inner_color)),
    }


def _place_clusters(h: int, w: int, clusters: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []

    order = sorted(range(len(clusters)), key=lambda idx: len(clusters[idx]["cells"]), reverse=True)
    for idx in order:
        cluster = clusters[idx]
        min_r, min_c, max_r, max_c = map(int, cluster["bbox"])
        lo_r = -min_r
        hi_r = h - 1 - max_r
        lo_c = -min_c
        hi_c = w - 1 - max_c
        if lo_r > hi_r or lo_c > hi_c:
            return None

        placed = False
        for _attempt in range(PLACEMENT_TRIES):
            cr = random.randint(lo_r, hi_r)
            cc = random.randint(lo_c, hi_c)
            abs_cells = [(int(cr + r), int(cc + c), int(color)) for (r, c), color in cluster["cells"].items()]
            if any(blocked[r, c] for r, c, _color in abs_cells):
                continue

            for r, c, color in abs_cells:
                grid[r, c] = int(color)
            for r, c, _color in abs_cells:
                for rr in range(max(0, r - 1), min(h, r + 2)):
                    for cc2 in range(max(0, c - 1), min(w, c + 2)):
                        blocked[rr, cc2] = True

            placements.append(
                {
                    "center": (int(cr), int(cc)),
                    "fill": tuple(map(int, cluster["fill"])),
                }
            )
            placed = True
            break

        if not placed:
            return None

    return grid, placements


def _expected_output(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for item in placements:
        cr, cc = map(int, item["center"])
        outer, inner = map(int, item["fill"])
        out[cr - 1 : cr + 2, cc - 1 : cc + 2] = int(outer)
        out[cr, cc] = int(inner)
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_RETRIES):
        target_count = random.randint(1, 3)
        colors = random.sample(LINE_COLORS, 2 * target_count)
        clusters = [
            _build_cluster((int(colors[2 * i]), int(colors[2 * i + 1])), target_count)
            for i in range(target_count)
        ]

        max_h = max(int(cluster["bbox"][2] - cluster["bbox"][0] + 1) for cluster in clusters)
        max_w = max(int(cluster["bbox"][3] - cluster["bbox"][1] + 1) for cluster in clusters)
        low_h = {1: 12, 2: 17, 3: 22}[target_count]
        low_w = {1: 12, 2: 17, 3: 22}[target_count]
        h_min = max(low_h, max_h + 2)
        w_min = max(low_w, max_w + 2)
        if h_min > 30 or w_min > 30:
            continue

        h = random.randint(h_min, 30)
        w = random.randint(w_min, 30)
        placed = _place_clusters(h, w, clusters)
        if placed is None:
            continue

        grid, placements = placed
        expected = _expected_output(grid, placements)

        try:
            analyzed_grid, targets, by_target = _analyze_grid(grid)
        except AssertionError:
            continue
        if len(targets) != target_count:
            continue
        if sum(len(v) for v in by_target.values()) != 2 * target_count:
            continue
        if not np.array_equal(analyzed_grid, grid):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise AssertionError("failed to generate a valid puzzle instance")
