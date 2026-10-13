from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
MAGENTA = 6
TEAL = 8
PAIR_COLORS = (BLUE, RED, YELLOW, MAGENTA)
GEN_ATTEMPTS = 400


Rect = tuple[int, int, int, int]
Frame = dict[str, int | Rect]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if lo > hi:
        return None
    return lo, hi


def _find_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert _get_bg(g) == BLACK, "expected black background"
    assert TEAL not in set(map(int, np.unique(g))), "teal must be absent from input"

    frames: list[Frame] = []
    used_colors = set(map(int, np.unique(g))) - {BLACK}
    assert used_colors and used_colors.issubset(set(PAIR_COLORS)), "unexpected foreground colors"

    for color in sorted(used_colors):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h >= 4 and w >= 4, "frames must have non-empty interiors"
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            assert np.array_equal(sub == color, border), "component must be a 1-pixel hollow rectangle"
            assert np.all(sub[1:-1, 1:-1] == BLACK), "frame interior must be black"
            frames.append(
                {
                    "color": color,
                    "outer": (r0, c0, r1, c1),
                    "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                }
            )

    counts = Counter(int(frame["color"]) for frame in frames)
    assert len(counts) in (2, 3), "expected 2 or 3 color pairs"
    assert all(v == 2 for v in counts.values()), "each color must appear exactly twice"
    assert len(frames) in (4, 6), "expected 4 or 6 frames"
    return frames


def _fill_stage(shape: tuple[int, int], frames: list[Frame]) -> tuple[np.ndarray, dict[int, list[Rect]]]:
    stage = np.zeros(shape, dtype=int)
    solids: dict[int, list[Rect]] = {}
    for frame in frames:
        color = int(frame["color"])
        rect = frame["inner"]
        solids.setdefault(color, []).append(rect)
        r0, c0, r1, c1 = rect
        stage[r0 : r1 + 1, c0 : c1 + 1] = color
    return stage, solids


def _bridge_candidate(stage: np.ndarray, rect_a: Rect, rect_b: Rect) -> dict[str, object] | None:
    ra0, ca0, ra1, ca1 = rect_a
    rb0, cb0, rb1, cb1 = rect_b
    candidates: list[dict[str, object]] = []

    row_overlap = _interval_overlap(ra0, ra1, rb0, rb1)
    if row_overlap is not None:
        lo, hi = row_overlap
        if hi - lo + 1 >= 3:
            if ca1 < cb0:
                left, right = rect_a, rect_b
            elif cb1 < ca0:
                left, right = rect_b, rect_a
            else:
                left = right = None
            if left is not None:
                lr0, lc0, lr1, lc1 = left
                rr0, rc0, rr1, rc1 = right
                gap = rc0 - lc1 - 1
                if gap >= 1 and np.all(stage[lo : hi + 1, lc1 + 1 : rc0] == BLACK):
                    rows = range(lo + 1, hi)
                    cols = range(lc1 + 1, rc0)
                    cells = [(r, c) for r in rows for c in cols]
                    candidates.append(
                        {
                            "orientation": "H",
                            "cells": cells,
                            "size": len(cells),
                            "shrunk_span": (lo + 1, hi - 1, lc1 + 1, rc0 - 1),
                        }
                    )

    col_overlap = _interval_overlap(ca0, ca1, cb0, cb1)
    if col_overlap is not None:
        lo, hi = col_overlap
        if hi - lo + 1 >= 3:
            if ra1 < rb0:
                top, bottom = rect_a, rect_b
            elif rb1 < ra0:
                top, bottom = rect_b, rect_a
            else:
                top = bottom = None
            if top is not None:
                tr0, tc0, tr1, tc1 = top
                br0, bc0, br1, bc1 = bottom
                gap = br0 - tr1 - 1
                if gap >= 1 and np.all(stage[tr1 + 1 : br0, lo : hi + 1] == BLACK):
                    rows = range(tr1 + 1, br0)
                    cols = range(lo + 1, hi)
                    cells = [(r, c) for r in rows for c in cols]
                    candidates.append(
                        {
                            "orientation": "V",
                            "cells": cells,
                            "size": len(cells),
                            "shrunk_span": (tr1 + 1, br0 - 1, lo + 1, hi - 1),
                        }
                    )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (int(item["size"]), str(item["orientation"])))
    return candidates[0]


def _pair_bridge_cells(stage: np.ndarray, rect_a: Rect, rect_b: Rect) -> list[tuple[int, int]]:
    candidate = _bridge_candidate(stage, rect_a, rect_b)
    assert candidate is not None, "same-color pair must have a valid bridge"
    return list(candidate["cells"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame's interior with its own color, erase the frame border, then bridge each same-color pair with teal.

    Input structure:
    - The grid has a black background and 4 or 6 one-pixel-thick hollow rectangular frames.
    - Frame colors are chosen from blue, red, yellow, and magenta, with each used color appearing exactly twice.
    - Every frame encloses a black interior, and each same-color pair faces across a clear horizontal or vertical background gap.

    Rule:
    - Replace every frame by a solid rectangle occupying its strict interior and colored with the frame's border color.
    - Erase all original border pixels to black.
    - For each same-color solid-rectangle pair, find the valid horizontal or vertical facing gap whose shared span is at least 3 cells.
    - Shrink the shared span by 1 cell at both ends, then fill every background cell in the remaining gap with teal.
    - If both orientations are valid, use the bridge with fewer painted gap cells.

    Key insight:
    - The bridge targets are not the visible hollow frames but the solid rectangles created after first filling the interiors and deleting the borders.
    """
    frames = _find_frames(np.asarray(grid, dtype=int))
    out, solids = _fill_stage(np.asarray(grid, dtype=int).shape, frames)

    for color, rects in sorted(solids.items()):
        assert len(rects) == 2, f"color {color} must have exactly two filled rectangles"
        for r, c in _pair_bridge_cells(out, rects[0], rects[1]):
            assert out[r, c] == BLACK
            out[r, c] = TEAL

    return out


def _wrong_outer_bridge_cells(shape: tuple[int, int], outer_a: Rect, outer_b: Rect) -> set[tuple[int, int]]:
    blank = np.zeros(shape, dtype=int)
    candidate = _bridge_candidate(blank, outer_a, outer_b)
    if candidate is None:
        return set()
    return set(candidate["cells"])


def _draw_frame(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_horizontal_pair(region_h: int, region_w: int, rng: random.Random) -> tuple[Rect, Rect] | None:
    for _ in range(120):
        max_h = min(6, region_h - 2)
        max_w = min(6, region_w - 5)
        if max_h < 3 or max_w < 3:
            return None
        h1 = rng.randint(3, max_h)
        h2 = rng.randint(3, max_h)
        w1 = rng.randint(3, max_w)
        w2 = rng.randint(3, max_w)
        max_gap = min(7, region_w - w1 - w2 - 2)
        if max_gap < 3:
            continue
        gap = rng.randint(3, max_gap)
        c1_max = region_w - (w1 + gap + w2) - 1
        if c1_max < 1:
            continue
        c1 = rng.randint(1, c1_max)
        c2 = c1 + w1 + gap
        overlap = rng.randint(3, min(h1, h2))
        modes = []
        delta_down = h1 - overlap
        r1_lo = max(1, 1 - delta_down)
        r1_hi = min(region_h - h1 - 1, region_h - h2 - 1 - delta_down)
        if r1_lo <= r1_hi:
            modes.append(("down", r1_lo, r1_hi))
        delta_up = h2 - overlap
        r1_lo = max(1 + delta_up, 1)
        r1_hi = min(region_h - h1 - 1, region_h - h2 - 1 + delta_up)
        if r1_lo <= r1_hi:
            modes.append(("up", r1_lo, r1_hi))
        if not modes:
            continue
        mode, lo, hi = rng.choice(modes)
        r1 = rng.randint(lo, hi)
        r2 = r1 + (h1 - overlap) if mode == "down" else r1 - (h2 - overlap)
        rect1 = (r1, c1, r1 + h1 - 1, c1 + w1 - 1)
        rect2 = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)
        return rect1, rect2
    return None


def _sample_vertical_pair(region_h: int, region_w: int, rng: random.Random) -> tuple[Rect, Rect] | None:
    for _ in range(120):
        max_h = min(6, region_h - 5)
        max_w = min(6, region_w - 2)
        if max_h < 3 or max_w < 3:
            return None
        h1 = rng.randint(3, max_h)
        h2 = rng.randint(3, max_h)
        w1 = rng.randint(3, max_w)
        w2 = rng.randint(3, max_w)
        max_gap = min(7, region_h - h1 - h2 - 2)
        if max_gap < 3:
            continue
        gap = rng.randint(3, max_gap)
        r1_max = region_h - (h1 + gap + h2) - 1
        if r1_max < 1:
            continue
        r1 = rng.randint(1, r1_max)
        r2 = r1 + h1 + gap
        overlap = rng.randint(3, min(w1, w2))
        modes = []
        delta_right = w1 - overlap
        c1_lo = max(1, 1 - delta_right)
        c1_hi = min(region_w - w1 - 1, region_w - w2 - 1 - delta_right)
        if c1_lo <= c1_hi:
            modes.append(("right", c1_lo, c1_hi))
        delta_left = w2 - overlap
        c1_lo = max(1 + delta_left, 1)
        c1_hi = min(region_w - w1 - 1, region_w - w2 - 1 + delta_left)
        if c1_lo <= c1_hi:
            modes.append(("left", c1_lo, c1_hi))
        if not modes:
            continue
        mode, lo, hi = rng.choice(modes)
        c1 = rng.randint(lo, hi)
        c2 = c1 + (w1 - overlap) if mode == "right" else c1 - (w2 - overlap)
        rect1 = (r1, c1, r1 + h1 - 1, c1 + w1 - 1)
        rect2 = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)
        return rect1, rect2
    return None


def _region_dims(orientation: str) -> tuple[int, int]:
    if orientation == "H":
        return 6, 11
    return 11, 6


def _partition_regions(orientations: list[str], rng: random.Random) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    count = len(orientations)
    mins = [_region_dims(o) for o in orientations]

    if count == 2:
        for split in rng.sample(["vertical", "horizontal"], 2):
            if split == "vertical":
                need_h = max(h for h, _ in mins)
                need_w = mins[0][1] + mins[1][1] + 1
                if need_h > 20 or need_w > 20:
                    continue
                h = rng.randint(max(15, need_h), 20)
                w = rng.randint(max(15, need_w), 20)
                left_min = mins[0][1]
                right_min = mins[1][1]
                left_w = rng.randint(left_min, w - 1 - right_min)
                regs = [(0, 0, h, left_w), (0, left_w + 1, h, w - left_w - 1)]
                if all(regs[i][2] >= mins[i][0] and regs[i][3] >= mins[i][1] for i in range(2)):
                    return (h, w), regs
            else:
                need_h = mins[0][0] + mins[1][0] + 1
                need_w = max(w0 for _, w0 in mins)
                if need_h > 20 or need_w > 20:
                    continue
                h = rng.randint(max(15, need_h), 20)
                w = rng.randint(max(15, need_w), 20)
                top_min = mins[0][0]
                bot_min = mins[1][0]
                top_h = rng.randint(top_min, h - 1 - bot_min)
                regs = [(0, 0, top_h, w), (top_h + 1, 0, h - top_h - 1, w)]
                if all(regs[i][2] >= mins[i][0] and regs[i][3] >= mins[i][1] for i in range(2)):
                    return (h, w), regs
        return None

    indices = list(range(3))
    patterns = []
    for top_idx in indices:
        rem = [i for i in indices if i != top_idx]
        patterns.append(("top", top_idx, rem[0], rem[1]))
    for left_idx in indices:
        rem = [i for i in indices if i != left_idx]
        patterns.append(("left", left_idx, rem[0], rem[1]))
    rng.shuffle(patterns)

    for kind, pivot, a, b in patterns:
        hp, wp = mins[pivot]
        ha, wa = mins[a]
        hb, wb = mins[b]
        if kind == "top":
            need_h = hp + 1 + max(ha, hb)
            need_w = max(wp, wa + 1 + wb)
            if need_h > 20 or need_w > 20:
                continue
            h = rng.randint(max(16, need_h), 20)
            w = rng.randint(max(16, need_w), 20)
            top_h = rng.randint(hp, h - 1 - max(ha, hb))
            left_w = rng.randint(wa, w - 1 - wb)
            regs = [None, None, None]
            regs[pivot] = (0, 0, top_h, w)
            regs[a] = (top_h + 1, 0, h - top_h - 1, left_w)
            regs[b] = (top_h + 1, left_w + 1, h - top_h - 1, w - left_w - 1)
        else:
            need_h = max(hp, ha + 1 + hb)
            need_w = wp + 1 + max(wa, wb)
            if need_h > 20 or need_w > 20:
                continue
            h = rng.randint(max(16, need_h), 20)
            w = rng.randint(max(16, need_w), 20)
            left_w = rng.randint(wp, w - 1 - max(wa, wb))
            top_h = rng.randint(ha, h - 1 - hb)
            regs = [None, None, None]
            regs[pivot] = (0, 0, h, left_w)
            regs[a] = (0, left_w + 1, top_h, w - left_w - 1)
            regs[b] = (top_h + 1, left_w + 1, h - top_h - 1, w - left_w - 1)
        if all(regs[i][2] >= mins[i][0] and regs[i][3] >= mins[i][1] for i in indices):
            return (h, w), regs  # type: ignore[arg-type]
    return None


def _validate_semantics(grid: np.ndarray) -> None:
    frames = _find_frames(grid)
    stage, solids = _fill_stage(grid.shape, frames)
    frame_by_color: dict[int, list[Frame]] = {}
    for frame in frames:
        frame_by_color.setdefault(int(frame["color"]), []).append(frame)

    output = main(grid)
    assert np.array_equal(output, output.astype(int))
    assert not np.array_equal(grid, output), "transform must change the grid"

    bridge_pixels = set(map(tuple, np.argwhere(output == TEAL)))
    assert bridge_pixels, "output must contain teal bridges"

    shortcut_counterexample = False
    orientations = set()
    for color in sorted(solids):
        rects = solids[color]
        cells = set(_pair_bridge_cells(stage, rects[0], rects[1]))
        assert cells, "bridge must paint at least one pixel"
        assert cells.issubset(bridge_pixels), "expected bridge pixels missing from output"
        cand = _bridge_candidate(stage, rects[0], rects[1])
        assert cand is not None
        orientations.add(str(cand["orientation"]))
        wrong = _wrong_outer_bridge_cells(grid.shape, frame_by_color[color][0]["outer"], frame_by_color[color][1]["outer"])
        if wrong and wrong != cells:
            shortcut_counterexample = True

    assert orientations.issubset({"H", "V"})
    assert shortcut_counterexample, "at least one pair should punish direct border bridging"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([2, 3])
    colors = rng.sample(list(PAIR_COLORS), pair_count)

    if pair_count == 2:
        orientation_options = [["H", "V"], ["V", "H"], ["H", "H"], ["V", "V"]]
        weights = [0.45, 0.25, 0.15, 0.15]
        orientations = rng.choices(orientation_options, weights=weights, k=1)[0][:]
    else:
        orientations = ["H", "V", rng.choice(["H", "V"])]
        rng.shuffle(orientations)

    partition = _partition_regions(orientations, rng)
    if partition is None:
        return None
    (h, w), regions = partition
    grid = np.zeros((h, w), dtype=int)

    for color, orientation, (r0, c0, rh, rw) in zip(colors, orientations, regions):
        if orientation == "H":
            pair = _sample_horizontal_pair(rh, rw, rng)
        else:
            pair = _sample_vertical_pair(rh, rw, rng)
        if pair is None:
            return None
        for inner in pair:
            ir0, ic0, ir1, ic1 = inner
            outer = (r0 + ir0 - 1, c0 + ic0 - 1, r0 + ir1 + 1, c0 + ic1 + 1)
            _draw_frame(grid, outer, color)

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
