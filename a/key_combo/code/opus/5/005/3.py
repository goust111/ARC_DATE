from __future__ import annotations

import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


BG = 0
GREEN = 3
GREY = 5
MARKER_COLORS = (1, 2, 4, 6)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = _neighbors4 if connectivity == 4 else _neighbors8
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in steps(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _segment_records(grid: np.ndarray) -> list[dict]:
    segments = []
    for comp in _components(grid == GREEN, connectivity=8):
        sums = {r + c for r, c in comp}
        assert len(sums) == 1, "green components must be anti-diagonal segments"
        dvals = sorted(r - c for r, c in comp)
        for a, b in zip(dvals, dvals[1:], strict=False):
            assert b - a == 2, "segment must be contiguous on its anti-diagonal"
        segments.append(
            {
                "coords": sorted(comp),
                "s": int(next(iter(sums))),
                "dvals": tuple(dvals),
                "dset": set(dvals),
                "dmin": int(dvals[0]),
                "dmax": int(dvals[-1]),
                "length": len(comp),
            }
        )
    return segments


def _pair_segments(segments: list[dict]) -> list[tuple[dict, dict, tuple[int, ...]]]:
    edges = []
    for i, a in enumerate(segments):
        for j in range(i + 1, len(segments)):
            b = segments[j]
            common = tuple(sorted(a["dset"] & b["dset"]))
            if common:
                edges.append((len(common), abs(a["s"] - b["s"]), i, j, common))
    edges.sort(reverse=True)

    matched: set[int] = set()
    pairs = []
    for overlap, _dist, i, j, common in edges:
        if overlap <= 0 or i in matched or j in matched:
            continue
        matched.add(i)
        matched.add(j)
        pairs.append((segments[i], segments[j], common))

    assert len(matched) == len(segments), "each green segment must have exactly one partner"
    return pairs


def _domain_mask(shape: tuple[int, int], seg_a: dict, seg_b: dict, common_dvals: tuple[int, ...]) -> np.ndarray:
    s_lo, s_hi = sorted((seg_a["s"], seg_b["s"]))
    d_lo, d_hi = min(common_dvals), max(common_dvals)
    rr, cc = np.indices(shape)
    return (rr + cc > s_lo) & (rr + cc < s_hi) & (rr - cc > d_lo - 1) & (rr - cc < d_hi + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-split diagonal parallelogram subregion with its singleton marker color.

    Input structure:
    - Black background surrounds 2-3 pairs of green anti-diagonal segments.
    - Each pair of green segments lies on two parallel anti-diagonals and delimits one diagonal fill domain.
    - Inside each domain, grey horizontal/vertical wall lines split the interior into 2-3 4-connected subregions, and each subregion contains exactly one colored singleton marker.

    Rule:
    - Pair green segments by their shared main-diagonal span, then recover the discrete parallelogram interior between each pair.
    - Within each interior, treat grey pixels as blocking walls and find the 4-connected non-wall subregions.
    - Recolor every pixel of a subregion to the color of its unique internal singleton marker.
    - Keep the green boundary segments, grey walls, and all pixels outside the paired diagonal domains unchanged.

    Key insight:
    - The colored singletons do not act globally; they label wall-defined topological subregions inside a geometry-first diagonal fill domain determined by the matched green segment pair.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    segments = _segment_records(g)
    pairs = _pair_segments(segments)

    for seg_a, seg_b, common_dvals in pairs:
        domain = _domain_mask(g.shape, seg_a, seg_b, common_dvals)
        regions = _components(domain & (g != GREY), connectivity=4)
        for region in regions:
            markers = [(r, c) for r, c in region if int(g[r, c]) not in {BG, GREEN, GREY}]
            assert len(markers) == 1, "each wall-defined subregion must contain exactly one marker"
            marker_color = int(g[markers[0]])
            assert marker_color in MARKER_COLORS, "marker colors must come from the visible palette"
            for r, c in region:
                out[r, c] = marker_color

    return out


def _make_segment(r0: int, c0: int, length: int) -> list[tuple[int, int]]:
    return [(r0 + i, c0 + length - 1 - i) for i in range(length)]


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded_bbox(b: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = b
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _domain_spans(domain: np.ndarray, orientation: str) -> list[dict]:
    spans = []
    h, w = domain.shape
    if orientation == "H":
        for r in range(h):
            cols = np.flatnonzero(domain[r])
            if cols.size == 0:
                continue
            spans.append({"orientation": "H", "axis": int(r), "lo": int(cols[0]), "hi": int(cols[-1]), "inner": int(cols[-1] - cols[0] + 1)})
    else:
        for c in range(w):
            rows = np.flatnonzero(domain[:, c])
            if rows.size == 0:
                continue
            spans.append({"orientation": "V", "axis": int(c), "lo": int(rows[0]), "hi": int(rows[-1]), "inner": int(rows[-1] - rows[0] + 1)})
    return spans


def _wall_candidates(domain: np.ndarray, grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    candidates = []
    for orientation in ("H", "V"):
        for span in _domain_spans(domain, orientation):
            inner = span["inner"]
            if inner < 2:
                continue
            max_total = min(10, w if orientation == "H" else h)
            min_total = max(5, inner)
            for total in range(min_total, max_total + 1):
                extra = total - inner
                offsets = list(range(extra + 1))
                random.shuffle(offsets)
                for left_extra in offsets:
                    start = span["lo"] - left_extra
                    end = span["hi"] + (extra - left_extra)
                    if start < 0 or end >= (w if orientation == "H" else h):
                        continue
                    if orientation == "H":
                        cells = [(span["axis"], c) for c in range(start, end + 1)]
                    else:
                        cells = [(r, span["axis"]) for r in range(start, end + 1)]
                    if any(int(grid[r, c]) != BG for r, c in cells):
                        continue
                    candidates.append(
                        {
                            "orientation": orientation,
                            "axis": span["axis"],
                            "cells": cells,
                        }
                    )
                    break
    random.shuffle(candidates)
    return candidates


def _region_bool(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _marker_candidates(region: list[tuple[int, int]], barriers: np.ndarray) -> list[tuple[int, int]]:
    h, w = barriers.shape
    candidates = []
    for r, c in region:
        if all(not (0 <= nr < h and 0 <= nc < w and barriers[nr, nc]) for nr, nc in _neighbors8(r, c)):
            candidates.append((r, c))
    return candidates


def _evaluate_wall_set(domain: np.ndarray, grid: np.ndarray, walls: list[dict]) -> dict | None:
    wall_mask = np.zeros_like(domain, dtype=bool)
    for wall in walls:
        for r, c in wall["cells"]:
            if wall_mask[r, c] or int(grid[r, c]) != BG:
                return None
            wall_mask[r, c] = True

    regions = _components(domain & ~wall_mask, connectivity=4)
    if len(regions) != len(walls) + 1:
        return None

    barriers = wall_mask | (grid == GREEN)
    marker_positions = []
    for region in regions:
        if len(region) - 1 < 6:
            return None
        options = _marker_candidates(region, barriers)
        if not options:
            return None
        marker_positions.append(random.choice(options))

    return {"walls": walls, "regions": regions, "marker_positions": marker_positions}


def _choose_walls(domain: np.ndarray, grid: np.ndarray, wall_count: int) -> dict | None:
    candidates = _wall_candidates(domain, grid)
    if wall_count == 1:
        for cand in candidates:
            solution = _evaluate_wall_set(domain, grid, [cand])
            if solution is not None:
                return solution
        return None

    for orientation in ("H", "V"):
        same = [cand for cand in candidates if cand["orientation"] == orientation]
        random.shuffle(same)
        for wall_a, wall_b in combinations(same, 2):
            if abs(int(wall_a["axis"]) - int(wall_b["axis"])) < 2:
                continue
            solution = _evaluate_wall_set(domain, grid, [wall_a, wall_b])
            if solution is not None:
                return solution
    return None


def _global_grey_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _components(grid == GREY, connectivity=4)


def _validate_scene(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if get_bg(grid) != BG:
        return False

    try:
        segments = _segment_records(grid)
        if len(segments) not in {4, 6}:
            return False
        pairs = _pair_segments(segments)
    except AssertionError:
        return False

    grey_components = _global_grey_components(grid)
    used_domain = np.zeros_like(grid, dtype=bool)
    marker_pixels_total = 0

    for seg_a, seg_b, common_dvals in pairs:
        domain = _domain_mask(grid.shape, seg_a, seg_b, common_dvals)
        if not domain.any() or np.any(used_domain & domain):
            return False
        used_domain |= domain

        touching_walls = []
        for comp in grey_components:
            if any(domain[r, c] for r, c in comp):
                touching_walls.append(comp)
        if len(touching_walls) not in {1, 2}:
            return False
        for comp in touching_walls:
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if len(rows) != 1 and len(cols) != 1:
                return False

        regions = _components(domain & (grid != GREY), connectivity=4)
        if len(regions) not in {2, 3}:
            return False
        seen_colors = set()
        for region in regions:
            markers = [(r, c) for r, c in region if int(grid[r, c]) in MARKER_COLORS]
            if len(markers) != 1:
                return False
            marker_pixels_total += 1
            mr, mc = markers[0]
            color = int(grid[mr, mc])
            if color in seen_colors or len(region) - 1 < 6:
                return False
            seen_colors.add(color)
            for nr, nc in _neighbors8(mr, mc):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) in {GREEN, GREY}:
                    return False

    colored = np.argwhere(np.isin(grid, MARKER_COLORS))
    if int(colored.shape[0]) != marker_pixels_total:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out[grid == GREEN] != GREEN) or np.any(out[grid == GREY] != GREY):
        return False
    return True


def _try_place_pair(grid: np.ndarray, reserved: np.ndarray, used_dsets: set[tuple[int, ...]]) -> bool:
    h, w = grid.shape
    length = 5
    delta = 4
    side = length + delta
    pad = 2

    for _ in range(200):
        r0 = random.randint(pad, h - side - pad)
        c0 = random.randint(pad, w - side - pad)

        top = _make_segment(r0, c0, length)
        bottom = [(r + delta, c + delta) for r, c in top]
        dvals = tuple(sorted(r - c for r, c in top))
        if dvals in used_dsets:
            continue

        bbox = _bbox(top + bottom)
        ex_r0, ex_c0, ex_r1, ex_c1 = _expanded_bbox(bbox, pad, grid.shape)
        if reserved[ex_r0 : ex_r1 + 1, ex_c0 : ex_c1 + 1].any():
            continue

        temp = grid.copy()
        _paint(temp, top, GREEN)
        _paint(temp, bottom, GREEN)
        seg_a = {"s": top[0][0] + top[0][1], "dset": set(dvals), "dvals": dvals}
        seg_b = {"s": bottom[0][0] + bottom[0][1], "dset": set(dvals), "dvals": dvals}
        domain = _domain_mask(temp.shape, seg_a, seg_b, dvals)
        orientation = random.choice(("H", "V"))
        spans = _domain_spans(domain, orientation)
        center = spans[len(spans) // 2]
        if orientation == "H":
            wall_cells = [(center["axis"], c) for c in range(center["lo"], center["hi"] + 1)]
        else:
            wall_cells = [(r, center["axis"]) for r in range(center["lo"], center["hi"] + 1)]
        _paint(temp, wall_cells, GREY)

        regions = _components(domain & (temp != GREY), connectivity=4)
        if len(regions) != 2 or any(len(region) - 1 < 6 for region in regions):
            continue
        barriers = (temp == GREY) | (temp == GREEN)
        marker_positions = []
        for region in regions:
            options = _marker_candidates(region, barriers)
            if not options:
                break
            marker_positions.append(random.choice(options))
        if len(marker_positions) != 2:
            continue

        colors = random.sample(MARKER_COLORS, len(regions))
        for color, (r, c) in zip(colors, marker_positions, strict=False):
            temp[r, c] = int(color)

        grid[:, :] = temp
        reserved[ex_r0 : ex_r1 + 1, ex_c0 : ex_c1 + 1] = True
        used_dsets.add(dvals)
        return True

    return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(26, 30)
        w = random.randint(26, 30)
        pair_count = random.choice((2, 3))
        grid = np.full((h, w), BG, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        used_dsets: set[tuple[int, ...]] = set()

        ok = True
        for _pair_idx in range(pair_count):
            if not _try_place_pair(grid, reserved, used_dsets):
                ok = False
                break

        if ok and _validate_scene(grid):
            return grid

    raise RuntimeError("failed to generate a valid diagonal wall-partition scene")
