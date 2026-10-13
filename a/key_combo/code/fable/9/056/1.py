from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
VISIBLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _parse_grid(grid: np.ndarray) -> tuple[list[int], int, np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 4 <= h <= 30 and 4 <= w <= 30

    separator = int(g[1, 0])
    assert np.all(g[1, :] == separator), "second row must be a full separator line"

    legend: list[int] = []
    c = 0
    while c < w and int(g[0, c]) != BLACK:
        legend.append(int(g[0, c]))
        c += 1
    assert 2 <= len(legend) < w, "top row must start with a non-empty legend strip"
    assert np.all(g[0, c:] == BLACK), "cells after the legend strip must stay black"

    main = g[2:, :]
    sources = np.argwhere((main != BLACK) & (main != separator))
    assert sources.shape == (1, 2), "main region must contain exactly one colored source pixel"
    src = tuple(map(int, sources[0]))
    return legend, separator, main, src, int(main[src])


def _distance_map(main: np.ndarray, src: tuple[int, int], wall_color: int) -> np.ndarray:
    h, w = main.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque([src])
    dist[src] = 0
    while q:
        r, c = q.popleft()
        step = dist[r, c] + 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if dist[nr, nc] != -1 or int(main[nr, nc]) == wall_color:
                continue
            dist[nr, nc] = step
            q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """Color geodesic wavefront layers from the source using the legend strip while walls block expansion.

    Input structure:
    - The top row begins with a contiguous colored legend strip and is black elsewhere.
    - The second row is a full grey separator line.
    - Below the separator, the main region is mostly black and contains grey wall cells plus exactly one non-grey colored source pixel.

    Rule:
    - Work only inside the main region.
    - Starting from the source, compute 4-neighbor shortest-path distances through non-wall cells.
    - Recolor each reachable black cell at distance d with the legend color indexed by `(d - 1) mod k`, where `k` is the legend length.
    - Keep the source, the grey walls, unreachable black pockets, the legend strip, and the separator line unchanged.

    Key insight:
    - The colored bands are not plain Manhattan diamonds: walls bend the wavefront, so equal-color bands follow obstacle-avoiding geodesic distance from the source.
    """
    legend, separator, main_region, src, _source_color = _parse_grid(grid)
    dist = _distance_map(main_region, src, separator)

    out = np.asarray(grid, dtype=int).copy()
    k = len(legend)
    for r, c in np.argwhere((main_region == BLACK) & (dist > 0)):
        out[r + 2, c] = legend[(int(dist[r, c]) - 1) % k]
    return out


def _manhattan_shortcut_output(grid: np.ndarray) -> np.ndarray:
    legend, _separator, main_region, src, _source_color = _parse_grid(grid)
    rows, cols = np.indices(main_region.shape)
    manhattan = np.abs(rows - src[0]) + np.abs(cols - src[1])
    out = np.asarray(grid, dtype=int).copy()
    k = len(legend)
    for r, c in np.argwhere(main_region == BLACK):
        d = int(manhattan[r, c])
        if d > 0:
            out[r + 2, c] = legend[(d - 1) % k]
    return out


def _region_clear(main_region: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 0) -> bool:
    h, w = main_region.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return bool(np.all(main_region[r0:r1, c0:c1] == BLACK))


def _place_ring(main_region: np.ndarray) -> bool:
    h, w = main_region.shape
    if h < 7 or w < 8:
        return False
    for _ in range(50):
        ring_h = random.randint(3, min(6, h - 2))
        ring_w = random.randint(3, min(8, w - 2))
        top = random.randint(1, h - ring_h - 1)
        left = random.randint(1, w - ring_w - 1)
        if not _region_clear(main_region, top, left, ring_h, ring_w, pad=1):
            continue
        main_region[top, left : left + ring_w] = GREY
        main_region[top + ring_h - 1, left : left + ring_w] = GREY
        main_region[top : top + ring_h, left] = GREY
        main_region[top : top + ring_h, left + ring_w - 1] = GREY
        return True
    return False


def _place_stub(main_region: np.ndarray, info: dict[str, int | str]) -> None:
    h, w = main_region.shape
    if info["orientation"] == "v":
        c = int(info["fixed"])
        start = int(info["start"])
        end = int(info["end"])
        if end - start < 2:
            return
        choices = []
        if c >= 3:
            choices.append(-1)
        if c <= w - 4:
            choices.append(1)
        if not choices:
            return
        direction = random.choice(choices)
        length = random.randint(2, min(4, c - 1 if direction < 0 else w - c - 2))
        row = random.randint(start + 1, end - 1)
        for offset in range(1, length + 1):
            main_region[row, c + direction * offset] = GREY
        return

    r = int(info["fixed"])
    start = int(info["start"])
    end = int(info["end"])
    if end - start < 2:
        return
    choices = []
    if r >= 3:
        choices.append(-1)
    if r <= h - 4:
        choices.append(1)
    if not choices:
        return
    direction = random.choice(choices)
    length = random.randint(2, min(4, r - 1 if direction < 0 else h - r - 2))
    col = random.randint(start + 1, end - 1)
    for offset in range(1, length + 1):
        main_region[r + direction * offset, col] = GREY


def _place_barrier(main_region: np.ndarray) -> dict[str, int | str] | None:
    h, w = main_region.shape
    if h < 7 or w < 7:
        return None

    if random.random() < 0.5:
        min_len = max(4, h // 2)
        max_len = h - 2
        if min_len > max_len or w < 7:
            return None
        length = random.randint(min_len, max_len)
        top = random.randint(1, h - length - 1)
        bottom = top + length - 1
        col = random.randint(3, w - 4)
        main_region[top : bottom + 1, col] = GREY
        info: dict[str, int | str] = {"orientation": "v", "fixed": col, "start": top, "end": bottom}
    else:
        min_len = max(4, w // 2)
        max_len = w - 2
        if min_len > max_len or h < 7:
            return None
        length = random.randint(min_len, max_len)
        left = random.randint(1, w - length - 1)
        right = left + length - 1
        row = random.randint(3, h - 4)
        main_region[row, left : right + 1] = GREY
        info = {"orientation": "h", "fixed": row, "start": left, "end": right}

    if random.random() < 0.55:
        _place_stub(main_region, info)
    return info


def _source_candidates(main_region: np.ndarray, info: dict[str, int | str] | None) -> list[tuple[int, int]]:
    h, w = main_region.shape
    if info is None:
        return [(r, c) for r in range(h) for c in range(w) if int(main_region[r, c]) == BLACK]

    candidates: list[tuple[int, int]] = []
    if info["orientation"] == "v":
        c_wall = int(info["fixed"])
        start = int(info["start"])
        end = int(info["end"])
        sides = []
        if c_wall >= 1:
            sides.append(range(0, c_wall))
        if c_wall + 1 < w:
            sides.append(range(c_wall + 1, w))
        random.shuffle(sides)
        for cols in sides:
            local = [(r, c) for r in range(start, end + 1) for c in cols if int(main_region[r, c]) == BLACK]
            if local:
                candidates.extend(local)
                break
    else:
        r_wall = int(info["fixed"])
        start = int(info["start"])
        end = int(info["end"])
        sides = []
        if r_wall >= 1:
            sides.append(range(0, r_wall))
        if r_wall + 1 < h:
            sides.append(range(r_wall + 1, h))
        random.shuffle(sides)
        for rows in sides:
            local = [(r, c) for r in rows for c in range(start, end + 1) if int(main_region[r, c]) == BLACK]
            if local:
                candidates.extend(local)
                break
    return candidates


def _analyze_case(grid: np.ndarray) -> dict[str, int | bool]:
    legend, separator, main_region, src, source_color = _parse_grid(grid)
    dist = _distance_map(main_region, src, separator)
    rows, cols = np.indices(main_region.shape)
    manhattan = np.abs(rows - src[0]) + np.abs(cols - src[1])
    reachable_black = (main_region == BLACK) & (dist > 0)
    unreachable_black = (main_region == BLACK) & (dist < 0)
    out = main(grid)

    used_legend_colors = {
        int(out[r + 2, c])
        for r, c in np.argwhere(reachable_black)
        if int(out[r + 2, c]) in legend
    }
    return {
        "legend_len": len(legend),
        "source_color": source_color,
        "wall_count": int(np.sum(main_region == separator)),
        "reachable_black_count": int(np.sum(reachable_black)),
        "unreachable_black_count": int(np.sum(unreachable_black)),
        "max_dist": int(dist.max()),
        "detour": bool(np.any(reachable_black & (dist > manhattan))),
        "used_legend_color_count": len(used_legend_colors),
        "new_color_count": int(np.sum(out[2:, :] != grid[2:, :])),
    }


def _semantic_checks(grid: np.ndarray, kind: str, legend: list[int], source_color: int) -> bool:
    out = main(grid)
    summary = _analyze_case(grid)

    if np.array_equal(out, grid):
        return False
    if source_color in legend:
        return False
    if summary["source_color"] != source_color:
        return False
    if summary["reachable_black_count"] < 8:
        return False
    if summary["max_dist"] < len(legend) + 2:
        return False
    if summary["used_legend_color_count"] != len(legend):
        return False
    if summary["new_color_count"] != summary["reachable_black_count"]:
        return False
    if int(np.sum(out == source_color)) != 1:
        return False

    if kind == "open":
        return summary["wall_count"] == 0 and summary["unreachable_black_count"] == 0 and not summary["detour"]

    if summary["wall_count"] == 0 or not summary["detour"]:
        return False
    if np.array_equal(out, _manhattan_shortcut_output(grid)):
        return False
    if kind == "pocket" and summary["unreachable_black_count"] == 0:
        return False
    return True


def _build_candidate(kind: str) -> np.ndarray | None:
    if kind == "open":
        rows = random.randint(10, 18)
        cols = random.randint(12, 22)
    elif kind == "barrier":
        rows = random.randint(11, 20)
        cols = random.randint(13, 24)
    else:
        rows = random.randint(12, 20)
        cols = random.randint(14, 24)

    grid = np.zeros((rows, cols), dtype=int)
    grid[1, :] = GREY

    legend_len = random.randint(3, 6)
    legend = random.sample(VISIBLE_COLORS, legend_len)
    source_color = random.choice([color for color in VISIBLE_COLORS if color not in legend])
    grid[0, :legend_len] = np.array(legend, dtype=int)

    main_region = grid[2:, :]
    barrier = None
    if kind in {"barrier", "pocket"}:
        barrier = _place_barrier(main_region)
        if barrier is None:
            return None

    candidates = _source_candidates(main_region, barrier)
    if not candidates:
        return None
    src = random.choice(candidates)
    main_region[src] = source_color

    if kind == "pocket" and not _place_ring(main_region):
        return None

    if kind == "barrier" and random.random() < 0.35:
        _place_ring(main_region)

    if _semantic_checks(grid, kind, legend, source_color):
        return grid
    return None


def generate_input() -> np.ndarray:
    kinds = ["open", "barrier", "pocket"]
    weights = [0.3, 0.35, 0.35]
    for _ in range(GEN_RETRIES):
        kind = random.choices(kinds, weights=weights, k=1)[0]
        candidate = _build_candidate(kind)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid obstacle wavefront input")
