from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
LEGEND_ENTRY_RANGE = (2, 4)
CHAIN_COUNT_RANGE = (1, 4)
COMPONENT_COUNT_CHOICES = [1, 2, 3, 4]
COMPONENT_COUNT_WEIGHTS = [1, 4, 3, 2]
FRAGMENT_COUNT_CHOICES = [1, 2, 3]
FRAGMENT_COUNT_WEIGHTS = [3, 4, 1]
FRAGMENT_LEN_RANGE = (2, 6)
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            out.append(np.array(pts, dtype=int))
    return out


def _extract_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    for r, c in np.argwhere(grid == GREY):
        r = int(r)
        c = int(c)
        horiz = (
            c - 1 >= 0
            and c + 1 < w
            and grid[r, c - 1] not in (BG, GREY)
            and grid[r, c + 1] not in (BG, GREY)
            and (r == 0 or grid[r - 1, c] == BG)
            and (r == h - 1 or grid[r + 1, c] == BG)
            and (c - 2 < 0 or grid[r, c - 2] == BG)
            and (c + 2 >= w or grid[r, c + 2] == BG)
        )
        vert = (
            r - 1 >= 0
            and r + 1 < h
            and grid[r - 1, c] not in (BG, GREY)
            and grid[r + 1, c] not in (BG, GREY)
            and (c == 0 or grid[r, c - 1] == BG)
            and (c == w - 1 or grid[r, c + 1] == BG)
            and (r - 2 < 0 or grid[r - 2, c] == BG)
            and (r + 2 >= h or grid[r + 2, c] == BG)
        )
        assert horiz != vert, "every grey pixel must belong to exactly one legend bar"

        if horiz:
            src = int(grid[r, c - 1])
            dst = int(grid[r, c + 1])
            legend_mask[r, c - 1 : c + 2] = True
        else:
            src = int(grid[r - 1, c])
            dst = int(grid[r + 1, c])
            legend_mask[r - 1 : r + 2, c] = True

        assert src not in mapping or mapping[src] == dst, "legend source must map consistently"
        mapping[src] = dst

    assert mapping, "legend must be present"
    return mapping, legend_mask


def _fill_same_color_gaps(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    changed = True
    while changed:
        changed = False
        nxt = out.copy()
        for r in range(h):
            for c in range(w):
                if int(out[r, c]) != BG:
                    continue
                candidates: set[int] = set()
                if 0 < c < w - 1 and int(out[r, c - 1]) == int(out[r, c + 1]) and int(out[r, c - 1]) != BG:
                    candidates.add(int(out[r, c - 1]))
                if 0 < r < h - 1 and int(out[r - 1, c]) == int(out[r + 1, c]) and int(out[r - 1, c]) != BG:
                    candidates.add(int(out[r - 1, c]))
                if len(candidates) == 1:
                    nxt[r, c] = next(iter(candidates))
                    changed = True
                else:
                    assert len(candidates) <= 1, "gap filling must be unambiguous"
        out = nxt
    return out


def _line_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    idx = 0
    for pts in _components(grid != BG):
        colors = {int(grid[r, c]) for r, c in pts}
        assert len(colors) == 1, "each component must be monochrome"
        color = next(iter(colors))
        rows = pts[:, 0]
        cols = pts[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert r0 == r1 or c0 == c1, "each path segment must be straight"
        orientation = "h" if r0 == r1 else "v"
        if orientation == "h":
            assert int(len(pts)) == c1 - c0 + 1, "horizontal segment must be solid after stitching"
            line = r0
            start = c0
            end = c1
        else:
            assert int(len(pts)) == r1 - r0 + 1, "vertical segment must be solid after stitching"
            line = c0
            start = r0
            end = r1
        comps.append(
            {
                "id": idx,
                "color": int(color),
                "cells": [(int(r), int(c)) for r, c in pts],
                "orientation": orientation,
                "line": int(line),
                "start": int(start),
                "end": int(end),
            }
        )
        idx += 1
    return comps


def _build_chains(components: list[dict]) -> list[dict]:
    chains: list[dict] = []
    by_row: dict[int, list[dict]] = defaultdict(list)
    by_col: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        if comp["orientation"] == "h":
            by_row[int(comp["line"])].append(comp)
        else:
            by_col[int(comp["line"])].append(comp)

    for row, comps in by_row.items():
        comps = sorted(comps, key=lambda comp: int(comp["start"]))
        current = [comps[0]]
        gaps: list[tuple[int, int]] = []
        for prev, nxt in zip(comps, comps[1:]):
            if int(nxt["start"]) == int(prev["end"]) + 2:
                current.append(nxt)
                gaps.append((int(row), int(prev["end"]) + 1))
            else:
                chains.append({"components": current, "gaps": gaps})
                current = [nxt]
                gaps = []
        chains.append({"components": current, "gaps": gaps})

    for col, comps in by_col.items():
        comps = sorted(comps, key=lambda comp: int(comp["start"]))
        current = [comps[0]]
        gaps: list[tuple[int, int]] = []
        for prev, nxt in zip(comps, comps[1:]):
            if int(nxt["start"]) == int(prev["end"]) + 2:
                current.append(nxt)
                gaps.append((int(prev["end"]) + 1, int(col)))
            else:
                chains.append({"components": current, "gaps": gaps})
                current = [nxt]
                gaps = []
        chains.append({"components": current, "gaps": gaps})

    return chains


def _chain_plan(colors: list[int], mapping: dict[int, int]) -> list[int] | None:
    current = int(colors[0])
    plan: list[int] = []
    for _ in range(1, len(colors)):
        if current not in mapping:
            return None
        current = int(mapping[current])
        plan.append(int(current))
    return plan


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stitch same-color path fragments, then recolor downstream segments at
    cross-color junctions according to grey-centered legend bars.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _extract_legend(g)

    work = g.copy()
    work[legend_mask] = BG
    work = _fill_same_color_gaps(work)
    out = work.copy()

    chains = _build_chains(_line_components(work))
    for chain in chains:
        comps = chain["components"]
        if len(comps) < 2:
            continue

        colors = [int(comp["color"]) for comp in comps]
        forward = _chain_plan(colors, mapping)
        backward = _chain_plan(colors[::-1], mapping)
        valid = [(1, forward), (-1, backward)]
        valid = [(direction, plan) for direction, plan in valid if plan is not None]
        assert len(valid) == 1, "junction direction must be unique"

        direction, plan = valid[0]
        if direction == 1:
            ordered_comps = comps
            ordered_gaps = chain["gaps"]
        else:
            ordered_comps = list(reversed(comps))
            ordered_gaps = list(reversed(chain["gaps"]))

        for i, target in enumerate(plan, start=1):
            gr, gc = ordered_gaps[i - 1]
            out[int(gr), int(gc)] = int(target)
            for r, c in ordered_comps[i]["cells"]:
                out[int(r), int(c)] = int(target)

    return out


def _legend_layout(
    h: int,
    w: int,
    entries: list[tuple[int, int]],
    bar_orientation: str,
    corner: str,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    count = len(entries)
    if bar_orientation == "h":
        box_h = 2 * count - 1
        box_w = 3
    else:
        box_h = 3
        box_w = 2 * count - 1

    if corner.startswith("top"):
        top = 1
    else:
        top = h - box_h - 2
    if corner.endswith("left"):
        left = 1
    else:
        left = w - box_w - 2

    bars: list[tuple[int, int, int, int]] = []
    for i, _entry in enumerate(entries):
        if bar_orientation == "h":
            r = top + 2 * i
            c = left
            bars.append((r, c, r, c + 2))
        else:
            r = top
            c = left + 2 * i
            bars.append((r, c, r + 2, c))
    return bars, (top, left, top + box_h - 1, left + box_w - 1)


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(h - 1, int(r1) + 1)
    cc1 = min(w - 1, int(c1) + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_fragments(force_split: bool) -> list[int]:
    frag_count = random.choices(FRAGMENT_COUNT_CHOICES, weights=FRAGMENT_COUNT_WEIGHTS, k=1)[0]
    if force_split:
        frag_count = max(2, frag_count)
    return [random.randint(*FRAGMENT_LEN_RANGE) for _ in range(frag_count)]


def _build_component_spec(
    force_split: bool,
) -> tuple[list[int], int]:
    frags = _sample_fragments(force_split)
    total_len = sum(frags) + len(frags) - 1
    return frags, total_len


def _build_chain_pattern(
    colors: list[int],
    force_first_split: bool,
) -> tuple[np.ndarray, list[tuple[int, int]], bool]:
    cells: list[int] = []
    spans: list[tuple[int, int]] = []
    had_split = False
    pos = 0
    for i, color in enumerate(colors):
        frags, _total_len = _build_component_spec(force_split=force_first_split and i == 0)
        if len(frags) > 1:
            had_split = True
        start = pos
        for j, frag_len in enumerate(frags):
            cells.extend([int(color)] * int(frag_len))
            pos += int(frag_len)
            if j + 1 < len(frags):
                cells.append(BG)
                pos += 1
        end = pos - 1
        spans.append((start, end))
        if i + 1 < len(colors):
            cells.append(BG)
            pos += 1
    return np.array(cells, dtype=int), spans, had_split


def _place_chain(
    grid: np.ndarray,
    blocked: np.ndarray,
    pattern: np.ndarray,
    orientation: str,
) -> bool:
    h, w = grid.shape
    length = int(pattern.shape[0])

    for _ in range(PLACEMENT_ATTEMPTS):
        if orientation == "h":
            if length > w - 2:
                return False
            row = random.randint(1, h - 2)
            col0 = random.randint(1, w - length - 1)
            if blocked[row - 1 : row + 2, col0 - 1 : col0 + length + 1].any():
                continue
            for offset, color in enumerate(pattern):
                if int(color) != BG:
                    grid[row, col0 + offset] = int(color)
            _mark_blocked(blocked, row, col0, row, col0 + length - 1)
            return True

        if length > h - 2:
            return False
        col = random.randint(1, w - 2)
        row0 = random.randint(1, h - length - 1)
        if blocked[row0 - 1 : row0 + length + 1, col - 1 : col + 2].any():
            continue
        for offset, color in enumerate(pattern):
            if int(color) != BG:
                grid[row0 + offset, col] = int(color)
        _mark_blocked(blocked, row0, col, row0 + length - 1, col)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend_count = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_count + 1 > len(PATH_COLORS):
        return None

    states = random.sample(PATH_COLORS, legend_count + 1)
    mapping = {int(states[i]): int(states[i + 1]) for i in range(legend_count)}
    entries = [(int(src), int(dst)) for src, dst in mapping.items()]
    bar_orientation = random.choice(["h", "v"])
    corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    bars, legend_box = _legend_layout(h, w, entries, bar_orientation, corner)
    if legend_box[0] < 0 or legend_box[1] < 0 or legend_box[2] >= h or legend_box[3] >= w:
        return None

    for (src, dst), (r0, c0, r1, c1) in zip(entries, bars):
        if bar_orientation == "h":
            grid[r0, c0] = int(src)
            grid[r0, c0 + 1] = GREY
            grid[r0, c0 + 2] = int(dst)
        else:
            grid[r0, c0] = int(src)
            grid[r0 + 1, c0] = GREY
            grid[r0 + 2, c0] = int(dst)
    _mark_blocked(blocked, *legend_box)

    filler_pool = [color for color in PATH_COLORS if color not in states[:-1]]
    if not filler_pool:
        return None

    chain_count = random.randint(*CHAIN_COUNT_RANGE)
    transition_chain_seen = False
    split_seen = False
    used_specs = 0

    for chain_index in range(chain_count):
        if chain_index == 0:
            comp_count = random.randint(2, min(4, legend_count + 1))
        else:
            comp_count = random.choices(COMPONENT_COUNT_CHOICES, weights=COMPONENT_COUNT_WEIGHTS, k=1)[0]
            if comp_count > legend_count + 1:
                comp_count = legend_count + 1

        if comp_count >= 2:
            max_start = legend_count - (comp_count - 1)
            if max_start < 0:
                return None
            start_idx = random.randint(0, max_start)
            colors = [int(states[start_idx])]
            current = int(states[start_idx])
            for step in range(comp_count - 1):
                target = int(mapping[current])
                choices = [color for color in filler_pool if int(color) != int(target)]
                if not choices:
                    choices = filler_pool
                colors.append(int(random.choice(choices)))
                current = target
            transition_chain_seen = True
        else:
            colors = [int(random.choice(PATH_COLORS))]

        pattern, _spans, had_split = _build_chain_pattern(colors, force_first_split=(chain_index == 0))
        split_seen = split_seen or had_split
        orientation = random.choice(["h", "v"])
        if random.choice([True, False]):
            pattern = pattern[::-1].copy()

        if not _place_chain(grid, blocked, pattern, orientation):
            return None
        used_specs += 1

    if not transition_chain_seen or not split_seen or used_specs == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
