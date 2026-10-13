from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _find_rectangles(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    owner = np.full((h, w), -1, dtype=int)
    rects: list[dict] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components(g == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            hh = int(r1 - r0 + 1)
            ww = int(c1 - c0 + 1)
            area = int(coords.shape[0])
            if hh < 2 or ww < 2 or area != hh * ww:
                continue
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            rid = len(rects)
            owner[coords[:, 0], coords[:, 1]] = rid
            rects.append(
                {
                    "id": rid,
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": area,
                    "coords": coords,
                }
            )
    return rects, owner


def _intervals_1d(ids: np.ndarray) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < len(ids):
        rid = int(ids[i])
        if rid < 0:
            i += 1
            continue
        j = i
        while j + 1 < len(ids) and int(ids[j + 1]) == rid:
            j += 1
        out.append((i, j, rid))
        i = j + 1
    return out


def _longest_run(values: np.ndarray, color: int) -> int:
    best = 0
    cur = 0
    for v in map(int, values):
        if v == color:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _edges_for_channel(
    grid: np.ndarray,
    owner: np.ndarray,
    bg: int,
    aux_colors: list[int],
    channel_color: int,
) -> tuple[set[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    allowed = np.asarray([bg] + aux_colors, dtype=int)
    edges: set[tuple[int, int]] = set()
    hits = 0
    support = 0

    for r in range(h):
        intervals = _intervals_1d(owner[r])
        for (_s1, e1, a), (s2, _e2, b) in zip(intervals, intervals[1:]):
            corridor = g[r, e1 + 1 : s2]
            if corridor.size == 0:
                continue
            if not np.all(np.isin(corridor, allowed)):
                continue
            run = _longest_run(corridor, channel_color)
            if run < 2:
                continue
            edges.add((min(a, b), max(a, b)))
            hits += 1
            support += run

    for c in range(w):
        intervals = _intervals_1d(owner[:, c])
        for (_s1, e1, a), (s2, _e2, b) in zip(intervals, intervals[1:]):
            corridor = g[e1 + 1 : s2, c]
            if corridor.size == 0:
                continue
            if not np.all(np.isin(corridor, allowed)):
                continue
            run = _longest_run(corridor, channel_color)
            if run < 2:
                continue
            edges.add((min(a, b), max(a, b)))
            hits += 1
            support += run

    return edges, hits, support


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair noisy straight connectors between solid rectangles, then recolor each
    connected rectangle group to the color of its unique largest rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    rects, owner = _find_rectangles(g, bg)
    if not rects:
        return np.full_like(g, bg)

    rect_mask = owner >= 0
    aux_colors = sorted({int(v) for v in g[~rect_mask] if int(v) != bg})

    edges: set[tuple[int, int]] = set()
    if aux_colors:
        best_score = None
        for color in aux_colors:
            cand_edges, hits, support = _edges_for_channel(g, owner, bg, aux_colors, color)
            score = (len(cand_edges), hits, support)
            if best_score is None or score > best_score:
                best_score = score
                edges = cand_edges

    dsu = _DSU(len(rects))
    for a, b in edges:
        dsu.union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for rid in range(len(rects)):
        groups[dsu.find(rid)].append(rid)

    out = np.full_like(g, bg)
    for ids in groups.values():
        winner = max(ids, key=lambda rid: rects[rid]["area"])
        winner_color = int(rects[winner]["color"])
        for rid in ids:
            coords = rects[rid]["coords"]
            out[coords[:, 0], coords[:, 1]] = winner_color
    return out


# --------------- generator ---------------

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
RECT_COUNT_RANGE = (3, 6)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 5)
GROUP_COUNT_MAX = 3
CHANNEL_GAP_RANGE = (3, 7)
BREAK_CHANNEL_PROB = 0.65
BACKGROUND_NOISE_RANGE = (4, 14)
FREE_PLACE_ATTEMPTS = 80
ATTACH_PLACE_ATTEMPTS = 140
GEN_ATTEMPTS = 500


def _in_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_free(
    owner: np.ndarray,
    reserved: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: int = 1,
) -> bool:
    h, w = owner.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return not np.any(owner[rr0 : rr1 + 1, cc0 : cc1 + 1] >= 0) and not np.any(
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]
    )


def _corridor_clear(
    owner: np.ndarray,
    reserved: np.ndarray,
    cells: list[tuple[int, int]],
    parent_bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = owner.shape
    for i, (r, c) in enumerate(cells):
        if owner[r, c] >= 0 or reserved[r, c]:
            return False
        touches_parent = False
        touches_child = False
        for nr, nc in _neighbors4(r, c, h, w):
            if reserved[nr, nc]:
                return False
            if _in_bbox(nr, nc, parent_bbox):
                if i != 0:
                    return False
                touches_parent = True
                continue
            if _in_bbox(nr, nc, child_bbox):
                if i != len(cells) - 1:
                    return False
                touches_child = True
                continue
            if owner[nr, nc] >= 0:
                return False
        if i == 0 and not touches_parent:
            return False
        if i == len(cells) - 1 and not touches_child:
            return False
    return True


def _random_group_sizes(total_rects: int) -> list[int]:
    max_groups = min(GROUP_COUNT_MAX, total_rects - 1)
    group_count = random.randint(1, max_groups)
    sizes = [1] * group_count
    for _ in range(total_rects - group_count):
        sizes[random.randrange(group_count)] += 1
    random.shuffle(sizes)
    return sizes


def _place_free_rect(
    owner: np.ndarray,
    reserved: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = owner.shape
    for _ in range(FREE_PLACE_ATTEMPTS):
        rh = random.randint(*RECT_H_RANGE)
        rw = random.randint(*RECT_W_RANGE)
        if h - rh - 1 <= 1 or w - rw - 1 <= 1:
            continue
        r0 = random.randint(1, h - rh - 1)
        c0 = random.randint(1, w - rw - 1)
        bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if _bbox_free(owner, reserved, bbox):
            return bbox
    return None


def _attach_rect(
    rects: list[dict],
    group_ids: list[int],
    owner: np.ndarray,
    reserved: np.ndarray,
) -> tuple[int, tuple[int, int, int, int], list[tuple[int, int]]] | None:
    h, w = owner.shape
    for _ in range(ATTACH_PLACE_ATTEMPTS):
        parent_id = random.choice(group_ids)
        parent_bbox = rects[parent_id]["bbox"]
        pr0, pc0, pr1, pc1 = parent_bbox
        rh = random.randint(*RECT_H_RANGE)
        rw = random.randint(*RECT_W_RANGE)
        gap = random.randint(*CHANNEL_GAP_RANGE)
        orient = random.choice(("horizontal", "vertical"))
        side = random.choice((-1, 1))

        if orient == "horizontal":
            line_r = random.randint(pr0, pr1)
            child_r0_lo = max(1, line_r - rh + 1)
            child_r0_hi = min(h - rh - 1, line_r)
            if child_r0_lo > child_r0_hi:
                continue
            cr0 = random.randint(child_r0_lo, child_r0_hi)
            cr1 = cr0 + rh - 1
            if side < 0:
                cc1 = pc0 - gap - 1
                cc0 = cc1 - rw + 1
                cells = [(line_r, c) for c in range(cc1 + 1, pc0)]
            else:
                cc0 = pc1 + gap + 1
                cc1 = cc0 + rw - 1
                cells = [(line_r, c) for c in range(pc1 + 1, cc0)]
        else:
            line_c = random.randint(pc0, pc1)
            child_c0_lo = max(1, line_c - rw + 1)
            child_c0_hi = min(w - rw - 1, line_c)
            if child_c0_lo > child_c0_hi:
                continue
            cc0 = random.randint(child_c0_lo, child_c0_hi)
            cc1 = cc0 + rw - 1
            if side < 0:
                cr1 = pr0 - gap - 1
                cr0 = cr1 - rh + 1
                cells = [(r, line_c) for r in range(cr1 + 1, pr0)]
            else:
                cr0 = pr1 + gap + 1
                cr1 = cr0 + rh - 1
                cells = [(r, line_c) for r in range(pr1 + 1, cr0)]

        child_bbox = (cr0, cc0, cr1, cc1)
        if cr0 < 1 or cc0 < 1 or cr1 >= h - 1 or cc1 >= w - 1:
            continue
        if not cells:
            continue
        if not _bbox_free(owner, reserved, child_bbox):
            continue
        if not _corridor_clear(owner, reserved, cells, parent_bbox, child_bbox):
            continue
        return parent_id, child_bbox, cells
    return None


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _place_background_noise(grid: np.ndarray, bg: int, noise: int, count: int) -> int:
    h, w = grid.shape
    placed = 0
    for _ in range(count * 50):
        r = random.randrange(h)
        c = random.randrange(w)
        if int(grid[r, c]) != bg:
            continue
        ok = True
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if int(grid[rr, cc]) == noise:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        grid[r, c] = noise
        placed += 1
        if placed >= count:
            break
    return placed


def _expected_output(shape: tuple[int, int], bg: int, rects: list[dict], groups: list[list[int]]) -> np.ndarray | None:
    out = np.full(shape, bg, dtype=int)
    for ids in groups:
        areas = [int(rects[rid]["area"]) for rid in ids]
        max_area = max(areas)
        if areas.count(max_area) != 1:
            return None
        winner = ids[areas.index(max_area)]
        color = int(rects[winner]["color"])
        for rid in ids:
            _paint_rect(out, rects[rid]["bbox"], color)
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    total_rects = random.randint(*RECT_COUNT_RANGE)
    groups = _random_group_sizes(total_rects)
    if total_rects + 3 > 10:
        return None

    palette = random.sample(list(range(10)), total_rects + 3)
    bg, channel, noise = map(int, palette[:3])
    rect_colors = list(map(int, palette[3:]))

    owner = np.full((h, w), -1, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    rects: list[dict] = []
    group_rect_ids: list[list[int]] = []
    connections: list[dict] = []

    color_i = 0
    for gid, group_size in enumerate(groups):
        bbox = _place_free_rect(owner, reserved)
        if bbox is None:
            return None
        rid = len(rects)
        _paint_rect(owner, bbox, rid)
        rects.append(
            {
                "id": rid,
                "group": gid,
                "color": rect_colors[color_i],
                "bbox": bbox,
                "area": (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1),
            }
        )
        color_i += 1
        ids = [rid]

        for _ in range(group_size - 1):
            attached = _attach_rect(rects, ids, owner, reserved)
            if attached is None:
                return None
            parent_id, child_bbox, cells = attached
            rid = len(rects)
            _paint_rect(owner, child_bbox, rid)
            rects.append(
                {
                    "id": rid,
                    "group": gid,
                    "color": rect_colors[color_i],
                    "bbox": child_bbox,
                    "area": (child_bbox[2] - child_bbox[0] + 1) * (child_bbox[3] - child_bbox[1] + 1),
                }
            )
            color_i += 1
            ids.append(rid)
            for r, c in cells:
                reserved[r, c] = True
            connections.append({"a": parent_id, "b": rid, "cells": cells})

        group_rect_ids.append(ids)

    expected = _expected_output((h, w), bg, rects, group_rect_ids)
    if expected is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for rect in rects:
        _paint_rect(inp, rect["bbox"], rect["color"])
    for conn in connections:
        for r, c in conn["cells"]:
            inp[r, c] = channel

    for conn in connections:
        cells = conn["cells"]
        n = len(cells)
        if n < 4 or random.random() >= BREAK_CHANNEL_PROB:
            continue
        options = [i for i in range(1, n - 1) if max(i, n - i - 1) >= 2]
        if not options:
            continue
        idx = random.choice(options)
        r, c = cells[idx]
        inp[r, c] = noise

    bg_noise_target = random.randint(*BACKGROUND_NOISE_RANGE)
    if _place_background_noise(inp, bg, noise, bg_noise_target) < 2:
        return None

    if _bg_color(inp) != bg:
        return None
    if channel not in inp or noise not in inp:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid instance")
