import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _round_half_up(x: float) -> int:
    if x >= 0:
        return int(np.floor(x + 0.5))
    return int(np.ceil(x - 0.5))


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _path_and_anchors(grid: np.ndarray) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BLACK)
    path_comp: list[tuple[int, int]] | None = None
    anchors: list[tuple[int, int]] = []
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        if len(comp) == 1 and colors == {GREY}:
            anchors.append(comp[0])
            continue
        if path_comp is None or len(comp) > len(path_comp):
            path_comp = comp
    assert path_comp is not None, "expected one non-black path component"
    path = set(path_comp)
    clean_anchors = []
    for r, c in anchors:
        if all((nr, nc) not in path for nr, nc in _neighbors4(r, c)):
            clean_anchors.append((r, c))
    return path, sorted(clean_anchors)


def _ordered_path(path: set[tuple[int, int]]) -> list[tuple[int, int]]:
    degrees = {
        pt: sum((nbr in path) for nbr in _neighbors4(pt[0], pt[1]))
        for pt in path
    }
    assert all(d in (1, 2) for d in degrees.values()), "path must be unbranched"
    ends = sorted(pt for pt, d in degrees.items() if d == 1)
    assert len(ends) == 2, "path must have exactly two endpoints"

    order = [ends[0]]
    prev = None
    cur = ends[0]
    while True:
        nxts = [nbr for nbr in _neighbors4(cur[0], cur[1]) if nbr in path and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must be deterministic"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(path), "path traversal must visit every cell exactly once"
    return order


def _segments_from_grid(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    path, anchors = _path_and_anchors(g)
    order = _ordered_path(path)

    segments: list[dict] = []
    current_color = int(g[order[0]])
    current_cells = [order[0]]
    for cell in order[1:]:
        color = int(g[cell])
        if color == current_color:
            current_cells.append(cell)
            continue
        segments.append(_make_segment(current_color, current_cells))
        current_color = color
        current_cells = [cell]
    segments.append(_make_segment(current_color, current_cells))

    colors = [seg["color"] for seg in segments]
    assert 2 <= len(segments) <= 4, "path must split into 2..4 monochrome segments"
    assert len(set(colors)) == len(colors), "each path color must appear in one contiguous segment"
    assert GREY not in colors, "grey is reserved for anchors"
    return segments, anchors


def _make_segment(color: int, cells: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    centroid = (
        _round_half_up(sum(rows) / len(rows)),
        _round_half_up(sum(cols) / len(cols)),
    )
    return {
        "color": int(color),
        "cells": list(cells),
        "count": len(cells),
        "centroid": centroid,
    }


def _distance2(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = a[0] - b[0]
    dc = a[1] - b[1]
    return dr * dr + dc * dc


def _assign_targets(
    segments: list[dict], anchors: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], bool]]:
    remaining = list(anchors)
    targets: list[tuple[tuple[int, int], bool]] = []
    for seg in segments:
        if remaining:
            centroid = seg["centroid"]
            idx, anchor = min(
                enumerate(remaining),
                key=lambda item: (_distance2(item[1], centroid), item[1][0], item[1][1]),
            )
            targets.append((anchor, True))
            remaining.pop(idx)
        else:
            targets.append((seg["centroid"], False))
    return targets


def _compact_dims(count: int, reserve_center: bool) -> tuple[int, int]:
    need = count + (1 if reserve_center else 0)
    best_key = None
    best_dims = None
    limit = count + 2
    for h in range(1, limit + 1):
        for w in range(1, limit + 1):
            area = h * w
            if area < need:
                continue
            key = (area, abs(h - w), max(h, w), h)
            if best_key is None or key < best_key:
                best_key = key
                best_dims = (h, w)
    assert best_dims is not None
    return best_dims


def _placement_cells(
    center: tuple[int, int], count: int, reserve_center: bool
) -> list[tuple[int, int]]:
    h, w = _compact_dims(count, reserve_center)
    r0 = _round_half_up(center[0] - (h - 1) / 2.0)
    c0 = _round_half_up(center[1] - (w - 1) / 2.0)
    cells: list[tuple[int, int]] = []
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if reserve_center and (r, c) == center:
                continue
            cells.append((r, c))
    return cells[:count]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    segments, anchors = _segments_from_grid(g)
    targets = _assign_targets(segments, anchors)

    out = np.full((h, w), BLACK, dtype=int)
    anchor_set = set(anchors)
    for seg, (target, anchored) in zip(segments, targets):
        for r, c in _placement_cells(target, seg["count"], reserve_center=anchored):
            if 0 <= r < h and 0 <= c < w and (r, c) not in anchor_set:
                out[r, c] = seg["color"]
    for r, c in anchors:
        out[r, c] = GREY
    return out


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        d0 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d1 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d0 != d1:
            turns += 1
    return turns


def _valid_next_steps(
    pos: tuple[int, int],
    visited: set[tuple[int, int]],
    h: int,
    w: int,
    margin: int,
) -> list[tuple[int, int]]:
    out = []
    for nr, nc in _neighbors4(pos[0], pos[1]):
        if not (margin <= nr < h - margin and margin <= nc < w - margin):
            continue
        if (nr, nc) in visited:
            continue
        seen_neighbors = [nbr for nbr in _neighbors4(nr, nc) if nbr in visited]
        if seen_neighbors != [pos] and set(seen_neighbors) != {pos}:
            continue
        if len(seen_neighbors) != 1:
            continue
        out.append((nr, nc))
    return out


def _random_path(
    rng: random.Random, h: int, w: int, length: int, margin: int = 2
) -> list[tuple[int, int]] | None:
    def onward_degree(pos: tuple[int, int], visited: set[tuple[int, int]]) -> int:
        return len(_valid_next_steps(pos, visited, h, w, margin))

    for _ in range(120):
        start = (rng.randrange(margin, h - margin), rng.randrange(margin, w - margin))
        path = [start]
        visited = {start}

        def dfs() -> bool:
            if len(path) == length:
                return True
            cur = path[-1]
            candidates = _valid_next_steps(cur, visited, h, w, margin)
            rng.shuffle(candidates)
            candidates.sort(key=lambda pt: (-onward_degree(pt, visited | {pt}), rng.random()))
            for nxt in candidates:
                path.append(nxt)
                visited.add(nxt)
                if dfs():
                    return True
                visited.remove(nxt)
                path.pop()
            return False

        if dfs() and _path_turns(path) >= 1:
            return path
    return None


def _split_total(
    rng: random.Random, total: int, parts: int, lo: int = 2, hi: int = 8
) -> list[int] | None:
    for _ in range(100):
        vals = []
        remaining = total
        ok = True
        for i in range(parts):
            slots = parts - i - 1
            low = max(lo, remaining - hi * slots)
            high = min(hi, remaining - lo * slots)
            if low > high:
                ok = False
                break
            val = remaining if i == parts - 1 else rng.randint(low, high)
            vals.append(val)
            remaining -= val
        if ok and remaining == 0:
            return vals
    return None


def _fits_inside(
    cells: list[tuple[int, int]], h: int, w: int
) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _non_adjacent_to_path(cell: tuple[int, int], path: set[tuple[int, int]]) -> bool:
    if cell in path:
        return False
    return all((nbr not in path) for nbr in _neighbors4(cell[0], cell[1]))


def _choose_anchor_for_segment(
    rng: random.Random,
    centroid: tuple[int, int],
    count: int,
    path: set[tuple[int, int]],
    h: int,
    w: int,
    existing_anchors: list[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if not _non_adjacent_to_path(cell, path):
                continue
            if cell in existing_anchors:
                continue
            placement = _placement_cells(cell, count, reserve_center=True)
            if not _fits_inside(placement, h, w):
                continue
            if any(pt in existing_anchors for pt in placement):
                continue
            if any(pt in occupied_cells for pt in placement):
                continue
            d = _distance2(cell, centroid)
            if any(_distance2(anchor, centroid) <= d for anchor in existing_anchors):
                continue
            candidates.append((d, abs(r - centroid[0]) + abs(c - centroid[1]), rng.random(), cell))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def _choose_extra_anchor(
    rng: random.Random,
    path: set[tuple[int, int]],
    h: int,
    w: int,
    anchors: list[tuple[int, int]],
    segment_targets: list[tuple[tuple[int, int], bool]],
    segments: list[dict],
    occupied_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    required = []
    for seg, (target, anchored) in zip(segments, segment_targets):
        if anchored:
            required.append((seg["centroid"], target))

    candidates = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if not _non_adjacent_to_path(cell, path):
                continue
            if cell in anchors or cell in occupied_cells:
                continue
            if any(_distance2(cell, centroid) <= _distance2(anchor, centroid) for centroid, anchor in required):
                continue
            score = min(
                [_distance2(cell, centroid) for centroid, _ in required] + [_distance2(cell, pt) for pt in path]
            )
            candidates.append((-score, rng.random(), cell))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _final_cells_and_targets(
    segments: list[dict], anchors: list[tuple[int, int]]
) -> tuple[list[tuple[tuple[int, int], bool]], list[list[tuple[int, int]]]]:
    targets = _assign_targets(segments, anchors)
    cells = [
        _placement_cells(target, seg["count"], reserve_center=anchored)
        for seg, (target, anchored) in zip(segments, targets)
    ]
    return targets, cells


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 23)
    w = rng.randint(15, 23)
    segment_count = rng.randint(2, 4)
    total_len = rng.randint(segment_count * 3, min(segment_count * 7, 24))
    lengths = _split_total(rng, total_len, segment_count, lo=2, hi=8)
    if lengths is None:
        return None

    path_list = _random_path(rng, h, w, total_len)
    if path_list is None:
        return None
    path = set(path_list)

    grid = np.full((h, w), BLACK, dtype=int)
    colors = rng.sample(PALETTE, segment_count)
    idx = 0
    for color, length in zip(colors, lengths):
        for cell in path_list[idx : idx + length]:
            grid[cell] = color
        idx += length

    segments, _ = _segments_from_grid(grid)
    if [seg["count"] for seg in segments] != lengths:
        return None

    enough_anchors = rng.choice([False, True])
    if enough_anchors:
        anchor_count = rng.randint(segment_count, segment_count + 2)
    else:
        anchor_count = rng.randint(0, segment_count - 1)

    anchors: list[tuple[int, int]] = []
    occupied_final: set[tuple[int, int]] = set()
    anchored_segments = min(anchor_count, segment_count)
    for seg in segments[:anchored_segments]:
        anchor = _choose_anchor_for_segment(
            rng,
            seg["centroid"],
            seg["count"],
            path,
            h,
            w,
            anchors,
            occupied_final,
        )
        if anchor is None:
            return None
        anchors.append(anchor)
        occupied_final.update(_placement_cells(anchor, seg["count"], reserve_center=True))

    targets, planned_cells = _final_cells_and_targets(segments, anchors)
    anchor_set = set(anchors)
    for cells in planned_cells:
        if not _fits_inside(cells, h, w):
            return None
        if any(pt in anchor_set for pt in cells):
            return None

    all_cells: set[tuple[int, int]] = set()
    for cells in planned_cells:
        if any(pt in all_cells for pt in cells):
            return None
        all_cells.update(cells)

    while len(anchors) < anchor_count:
        extra = _choose_extra_anchor(
            rng,
            path,
            h,
            w,
            anchors,
            targets,
            segments,
            all_cells,
        )
        if extra is None:
            return None
        anchors.append(extra)

    for r, c in anchors:
        grid[r, c] = GREY

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len(set(colors)) != len(colors):
        return None
    if _path_turns(path_list) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
