from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import random

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
ARROW_LENGTHS = (3, 4, 5)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
ARROW_COUNT_CHOICES = (2, 3, 4, 5)
WITH_CROSS_PROB = 0.5
TRUNCATE_PROB = 0.4
CLUE_MARGIN_MIN = 2

GEN_ATTEMPTS = 500
LOCAL_ATTEMPTS = 250
SUBSET_LIMIT = 80


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).ravel())
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if _in_bounds(nr, nc, h, w) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _capacity_from(end: tuple[int, int], step: tuple[int, int], h: int, w: int) -> int:
    r, c = end
    dr, dc = step
    count = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(r, c, h, w):
            return count
        count += 1


def _ray_cells(end: tuple[int, int], step: tuple[int, int], length: int, h: int, w: int) -> list[tuple[int, int]]:
    r, c = end
    dr, dc = step
    ray: list[tuple[int, int]] = []
    for _ in range(length):
        r += dr
        c += dc
        if not _in_bounds(r, c, h, w):
            break
        ray.append((int(r), int(c)))
    return ray


def _open_run(grid: np.ndarray, bg: int, end: tuple[int, int], step: tuple[int, int]) -> int:
    h, w = grid.shape
    r, c = end
    dr, dc = step
    count = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(r, c, h, w) or int(grid[r, c]) != int(bg):
            return count
        count += 1


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(rr, cc, h, w):
                    out.add((int(rr), int(cc)))
    return out


def _build_arrow_spec(
    h: int,
    w: int,
    direction: str,
    length: int,
    tip: tuple[int, int],
) -> dict | None:
    tr, tc = tip
    if direction == "right":
        cells = [(tr, tc - length + 1 + i) for i in range(length)]
        tail = (tr, tc - length + 1)
    elif direction == "left":
        cells = [(tr, tc + i) for i in range(length)]
        tail = (tr, tc + length - 1)
    elif direction == "down":
        cells = [(tr - length + 1 + i, tc) for i in range(length)]
        tail = (tr - length + 1, tc)
    else:
        assert direction == "up"
        cells = [(tr + i, tc) for i in range(length)]
        tail = (tr + length - 1, tc)

    if not all(_in_bounds(r, c, h, w) for r, c in cells):
        return None

    step = DIRS[direction]
    alt_step = DIRS[OPPOSITE[direction]]
    ray = _ray_cells(tip, step, length, h, w)
    cells_set = set(cells)
    return {
        "direction": direction,
        "length": int(length),
        "tip": (int(tr), int(tc)),
        "tail": (int(tail[0]), int(tail[1])),
        "cells": cells,
        "cells_set": cells_set,
        "buffer": _expand_cells(cells_set, h, w),
        "ray": ray,
        "ray_set": set(ray),
        "capacity": _capacity_from(tip, step, h, w),
        "alt_capacity": _capacity_from(tail, alt_step, h, w),
        "truncated": len(ray) < int(length),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, int, set[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    h, w = g.shape
    bg = _bg_color(g)

    non_bg = g != bg
    marker_candidates: list[list[tuple[int, int]]] = []
    for comp in _components(non_bg):
        colors = {int(g[r, c]) for r, c in comp}
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) == 2 and len(colors) == 2 and r0 == r1 and c1 == c0 + 1:
            marker_candidates.append(comp)

    assert len(marker_candidates) == 1, "expected a unique two-color horizontal marker pair"
    marker_cells = set(marker_candidates[0])
    mr = min(r for r, _c in marker_cells)
    right_c = max(c for _r, c in marker_cells)
    cross_color = int(g[mr, right_c])

    arrow_mask = non_bg.copy()
    for r, c in marker_cells:
        arrow_mask[r, c] = False

    arrows: list[dict] = []
    for comp in _components(arrow_mask):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each arrow must be monochrome"
        color = int(next(iter(colors)))
        area = len(comp)
        assert 3 <= area <= 5, "each arrow must have 3-5 pixels"

        r0, c0, r1, c1 = _bbox(comp)
        if r0 == r1:
            cols = sorted(c for _r, c in comp)
            assert cols == list(range(c0, c1 + 1)), "horizontal arrows must be contiguous"
            ends = [((r0, c0), DIRS["left"]), ((r0, c1), DIRS["right"])]
        else:
            assert c0 == c1, "arrows must be straight horizontal or vertical segments"
            rows = sorted(r for r, _c in comp)
            assert rows == list(range(r0, r1 + 1)), "vertical arrows must be contiguous"
            ends = [((r0, c0), DIRS["up"]), ((r1, c0), DIRS["down"])]

        cands = []
        for end, step in ends:
            cands.append(
                {
                    "tip": (int(end[0]), int(end[1])),
                    "step": (int(step[0]), int(step[1])),
                    "capacity": _capacity_from(end, step, h, w),
                    "open_run": _open_run(g, bg, end, step),
                    "ray": _ray_cells(end, step, area, h, w),
                }
            )

        arrows.append(
            {
                "color": color,
                "length": int(area),
                "cells_set": set(comp),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cands": cands,
            }
        )

    arrows.sort(key=lambda obj: (obj["bbox"], obj["color"]))
    assert arrows, "expected at least one arrow"
    return int(bg), int(cross_color), marker_cells, arrows


def _assignment_score(grid: np.ndarray, bg: int, arrows: list[dict], bits: tuple[int, ...]) -> tuple[int, ...]:
    coverage: dict[tuple[int, int], set[int]] = defaultdict(set)
    pos_capacity = 0
    capacity_margin = 0
    pos_open = 0
    open_margin = 0

    for i, bit in enumerate(bits):
        cand = arrows[i]["cands"][bit]
        other = arrows[i]["cands"][1 - bit]
        if cand["capacity"] > other["capacity"]:
            pos_capacity += 1
        if cand["open_run"] > other["open_run"]:
            pos_open += 1
        capacity_margin += int(cand["capacity"]) - int(other["capacity"])
        open_margin += int(cand["open_run"]) - int(other["open_run"])
        for cell in cand["ray"]:
            coverage[cell].add(i)

    visible_overlap = 0
    visible_single = 0
    hidden_hits = 0
    for cell, owners in coverage.items():
        if int(grid[cell]) != int(bg):
            hidden_hits += 1
        elif len(owners) >= 2:
            visible_overlap += 1
        else:
            visible_single += 1

    return (
        int(visible_overlap),
        int(pos_capacity),
        int(capacity_margin),
        int(pos_open),
        int(open_margin),
        int(visible_single),
        -int(hidden_hits),
    )


def _choose_assignment(grid: np.ndarray, bg: int, arrows: list[dict]) -> tuple[int, ...]:
    best_bits: tuple[int, ...] | None = None
    best_score: tuple[int, ...] | None = None
    for bits in itertools.product((0, 1), repeat=len(arrows)):
        bits = tuple(int(v) for v in bits)
        score = _assignment_score(grid, bg, arrows, bits)
        if best_score is None or score > best_score or (score == best_score and bits < best_bits):
            best_score = score
            best_bits = bits
    assert best_bits is not None
    return best_bits


def _render_assignment(
    grid: np.ndarray,
    bg: int,
    cross_color: int,
    arrows: list[dict],
    bits: tuple[int, ...],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    coverage: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, bit in enumerate(bits):
        for cell in arrows[i]["cands"][bit]["ray"]:
            coverage[cell].append(i)

    for cell, owners in coverage.items():
        if int(grid[cell]) != int(bg):
            continue
        out[cell] = int(cross_color if len(set(owners)) >= 2 else arrows[owners[0]]["color"])
    return out


def _infer_tips_by_color(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    bg, _cross_color, _marker_cells, arrows = _extract_scene(grid)
    bits = _choose_assignment(np.asarray(grid, dtype=int), bg, arrows)
    return {int(arrows[i]["color"]): tuple(arrows[i]["cands"][bits[i]]["tip"]) for i in range(len(arrows))}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast finite rays from short colored arrows and recolor visible ray crossings by the marker color.

    Input structure:
    - The background is the dominant color.
    - Several non-background monochrome arrows appear as short 1-cell-thick horizontal or vertical line sprites; each arrow has 3-5 pixels total, and the surrounding layout disambiguates which end is the tip.
    - A separate two-cell horizontal marker with two different colors is the unique crossing-color indicator, and the right marker cell gives the color used at ray intersections.

    Rule:
    - For each arrow, count its pixels to obtain ray length N.
    - Starting one cell beyond the arrow tip, draw a straight ray of N cells in the arrow's pointing direction, truncating at the grid boundary.
    - Paint background cells covered by exactly one ray with that arrow's color.
    - When rays from different arrows cover the same background cell, recolor that cell to the marker's right-hand color; keep all original arrow cells and the marker pair unchanged.

    Key insight:
    - The arrow size encodes the ray range, while the separate two-cell marker independently supplies the crossing color, so the solver must decode both the length rule and the legend-like recolor rule.
    """
    g = np.asarray(grid, dtype=int)
    bg, cross_color, _marker_cells, arrows = _extract_scene(g)
    bits = _choose_assignment(g, bg, arrows)
    return _render_assignment(g, bg, cross_color, arrows, bits)


def _spec_cache(h: int, w: int) -> dict[tuple[str, int], list[dict]]:
    cache: dict[tuple[str, int], list[dict]] = {}
    for direction in DIRS:
        for length in ARROW_LENGTHS:
            specs: list[dict] = []
            for r in range(h):
                for c in range(w):
                    spec = _build_arrow_spec(h, w, direction, length, (r, c))
                    if spec is None:
                        continue
                    if spec["capacity"] <= 0:
                        continue
                    if int(spec["capacity"]) - int(spec["alt_capacity"]) < CLUE_MARGIN_MIN:
                        continue
                    specs.append(spec)
            cache[(direction, length)] = specs
    return cache


def _subset(items: list[dict], limit: int = SUBSET_LIMIT) -> list[dict]:
    if len(items) <= limit:
        out = list(items)
        random.shuffle(out)
        return out
    return random.sample(items, limit)


def _spec_conflicts(
    spec: dict,
    object_buffer: set[tuple[int, int]],
    object_cells: set[tuple[int, int]],
    used_rays: set[tuple[int, int]],
    allowed_overlaps: set[tuple[int, int]] | None = None,
) -> bool:
    allowed = allowed_overlaps or set()
    if spec["cells_set"] & object_buffer:
        return True
    if spec["cells_set"] & used_rays:
        return True
    if spec["ray_set"] & object_cells:
        return True
    if (spec["ray_set"] & used_rays) - allowed:
        return True
    return False


def _pick_cross_pair(h: int, w: int, cache: dict[tuple[str, int], list[dict]]) -> tuple[dict, dict] | None:
    for _ in range(LOCAL_ATTEMPTS):
        cross = (random.randint(2, h - 3), random.randint(2, w - 3))
        h_dir = random.choice(("left", "right"))
        v_dir = random.choice(("up", "down"))
        h_len = random.choice(ARROW_LENGTHS)
        v_len = random.choice(ARROW_LENGTHS)

        hs = [s for s in cache[(h_dir, h_len)] if cross in s["ray"]]
        vs = [s for s in cache[(v_dir, v_len)] if cross in s["ray"]]
        if random.random() < TRUNCATE_PROB:
            filtered = [s for s in hs if s["truncated"]]
            hs = filtered or hs
        if random.random() < TRUNCATE_PROB:
            filtered = [s for s in vs if s["truncated"]]
            vs = filtered or vs
        if not hs or not vs:
            continue

        for h_spec in _subset(hs):
            for v_spec in _subset(vs):
                if h_spec["cells_set"] & v_spec["buffer"]:
                    continue
                if v_spec["cells_set"] & h_spec["buffer"]:
                    continue
                if h_spec["ray_set"] & v_spec["cells_set"]:
                    continue
                if v_spec["ray_set"] & h_spec["cells_set"]:
                    continue
                if (h_spec["ray_set"] & v_spec["ray_set"]) != {cross}:
                    continue
                return h_spec, v_spec
    return None


def _pick_free_arrow(
    cache: dict[tuple[str, int], list[dict]],
    object_buffer: set[tuple[int, int]],
    object_cells: set[tuple[int, int]],
    used_rays: set[tuple[int, int]],
) -> dict | None:
    for _ in range(LOCAL_ATTEMPTS):
        direction = random.choice(tuple(DIRS))
        length = random.choice(ARROW_LENGTHS)
        specs = cache[(direction, length)]
        if random.random() < TRUNCATE_PROB:
            filtered = [s for s in specs if s["truncated"]]
            specs = filtered or specs
        for spec in _subset(specs):
            if not _spec_conflicts(spec, object_buffer, object_cells, used_rays):
                return spec
    return None


def _pick_marker(
    h: int,
    w: int,
    object_buffer: set[tuple[int, int]],
    used_rays: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rows = list(range(h))
    random.shuffle(rows)
    edge_rows = [0, h - 1] if h > 1 else [0]
    rows = edge_rows + [r for r in rows if r not in edge_rows]
    for r in rows:
        cols = list(range(w - 1))
        random.shuffle(cols)
        for c in cols:
            left = (int(r), int(c))
            right = (int(r), int(c + 1))
            if {left, right} & object_buffer:
                continue
            if left in used_rays or right in used_rays:
                continue
            return left, right
    return None


def _direct_output(grid: np.ndarray, bg: int, cross_color: int, arrows: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    coverage: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, arrow in enumerate(arrows):
        for cell in arrow["spec"]["ray"]:
            coverage[cell].append(idx)
    for cell, owners in coverage.items():
        if int(grid[cell]) != int(bg):
            continue
        out[cell] = int(cross_color if len(set(owners)) >= 2 else arrows[owners[0]]["color"])
    return out


def _make_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    arrow_count = random.choice(ARROW_COUNT_CHOICES)
    with_cross = random.random() < WITH_CROSS_PROB

    if h < 6 or w < 6:
        return None

    colors = list(range(10))
    random.shuffle(colors)
    bg = int(colors[0])
    left_marker_color = int(colors[1])
    cross_color = int(colors[2])
    arrow_colors = [int(c) for c in colors[3 : 3 + arrow_count]]
    if len(arrow_colors) != arrow_count:
        return None

    cache = _spec_cache(h, w)
    object_cells: set[tuple[int, int]] = set()
    object_buffer: set[tuple[int, int]] = set()
    used_rays: set[tuple[int, int]] = set()
    arrows: list[dict] = []

    if with_cross:
        pair = _pick_cross_pair(h, w, cache)
        if pair is None:
            return None
        for color, spec in zip(arrow_colors[:2], pair):
            arrows.append({"color": int(color), "spec": spec})
            object_cells |= spec["cells_set"]
            object_buffer |= spec["buffer"]
            used_rays |= spec["ray_set"]
        color_start = 2
    else:
        color_start = 0

    for color in arrow_colors[color_start:]:
        spec = _pick_free_arrow(cache, object_buffer, object_cells, used_rays)
        if spec is None:
            return None
        arrows.append({"color": int(color), "spec": spec})
        object_cells |= spec["cells_set"]
        object_buffer |= spec["buffer"]
        used_rays |= spec["ray_set"]

    marker = _pick_marker(h, w, object_buffer, used_rays)
    if marker is None:
        return None
    left_cell, right_cell = marker

    grid = np.full((h, w), int(bg), dtype=int)
    for arrow in arrows:
        for r, c in arrow["spec"]["cells"]:
            grid[r, c] = int(arrow["color"])
    grid[left_cell] = int(left_marker_color)
    grid[right_cell] = int(cross_color)

    expected = _direct_output(grid, bg, cross_color, arrows)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    inferred = _infer_tips_by_color(grid)
    intended = {int(arrow["color"]): tuple(arrow["spec"]["tip"]) for arrow in arrows}
    if inferred != intended:
        return None

    visible_crosses = int(np.count_nonzero((grid == bg) & (expected == cross_color)))
    if with_cross and visible_crosses == 0:
        return None
    if not with_cross and visible_crosses != 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _make_attempt()
            if grid is not None:
                return grid
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
