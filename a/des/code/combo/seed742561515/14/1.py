from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    candidates: list[tuple[int, int, int]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) != 4 or r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                candidates.append((int(c0), int(r0), int(color)))

    assert candidates, "expected at least one 2x2 legend block"

    by_col: dict[int, list[tuple[int, int]]] = {}
    for c0, r0, color in candidates:
        by_col.setdefault(int(c0), []).append((int(r0), int(color)))

    best_size = max(len(entries) for entries in by_col.values())
    best_col = min(c0 for c0, entries in by_col.items() if len(entries) == best_size)
    entries = sorted(by_col[best_col], key=lambda item: item[0])
    return [color for _row, color in entries]


def _is_rectangular_border(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(sub == color, border)


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components(grid == color):
            bbox = _bbox(comp)
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if len(comp) != 2 * h + 2 * w - 4:
                continue
            if _is_rectangular_border(grid, color, bbox):
                frames.append((int(r0), int(c0), int(r1), int(c1), int(color)))
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each frame's interior markers using the legend color indexed by marker count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend_colors = _find_legend_colors(g, bg)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one framed region"

    out = g.copy()
    for r0, c0, r1, c1, frame_color in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (interior != bg) & (interior != frame_color)
        count = int(np.count_nonzero(marker_mask))
        assert 1 <= count <= len(legend_colors), "marker count must index a legend color"
        out[r0 + 1 : r1, c0 + 1 : c1][marker_mask] = legend_colors[count - 1]

    return out


GEN_BG = 0
GEN_FRAME = 5
GEN_ATTEMPTS = 300


def _sample_inner_size(rng: np.random.Generator, count: int) -> tuple[int, int] | None:
    for _ in range(80):
        inner_h = int(rng.integers(2, 8))
        inner_w = int(rng.integers(2, 10))
        area = inner_h * inner_w
        if (area + 1) // 2 < count:
            continue
        if area < count + max(2, count // 2):
            continue
        return inner_h, inner_w
    return None


def _sample_marker_cells(
    rng: np.random.Generator, inner_h: int, inner_w: int, count: int
) -> list[tuple[int, int]] | None:
    parities = [0, 1]
    rng.shuffle(parities)
    for parity in parities:
        cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == parity]
        if len(cells) < count:
            continue
        choice = rng.choice(len(cells), size=count, replace=False)
        return [cells[int(idx)] for idx in choice]
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    legend_count = int(rng.integers(3, 7))
    frame_count_max = min(4, legend_count)
    if rng.random() < 0.25:
        frame_count = 1
    else:
        frame_count = int(rng.integers(2, frame_count_max + 1))

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(palette)
    legend_colors = palette[:legend_count]
    marker_palette = palette[legend_count:]
    if not marker_palette:
        return None

    counts = list(rng.integers(1, legend_count + 1, size=frame_count))
    if frame_count > 1 and len(set(counts)) == 1:
        counts[-1] = 1 + (counts[-1] % legend_count)

    frame_specs: list[dict[str, object]] = []
    for count0 in counts:
        count = int(count0)
        inner_size = _sample_inner_size(rng, count)
        if inner_size is None:
            return None
        inner_h, inner_w = inner_size
        cells = _sample_marker_cells(rng, inner_h, inner_w, count)
        if cells is None:
            return None
        frame_specs.append(
            {
                "count": count,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "marker_color": int(marker_palette[int(rng.integers(0, len(marker_palette)))]),
                "cells": cells,
            }
        )

    legend_h = 3 * legend_count - 1
    max_frame_h = max(int(spec["outer_h"]) for spec in frame_specs)
    max_frame_w = max(int(spec["outer_w"]) for spec in frame_specs)

    min_h = max(legend_h + 2, max_frame_h + 2, 18)
    min_w = max(13 + max_frame_w, 22)
    if min_h > 30 or min_w > 30:
        return None

    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    legend_left = int(rng.integers(1, 3))
    legend_top = int(rng.integers(1, h - legend_h))
    frame_left_min = legend_left + 6
    if frame_left_min + max_frame_w + 1 >= w:
        return None

    grid = np.full((h, w), GEN_BG, dtype=int)

    for idx, color in enumerate(legend_colors):
        r0 = legend_top + 3 * idx
        c0 = legend_left
        grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)

    occupied = np.zeros((h, w), dtype=bool)
    _mark_occupied(occupied, legend_top, legend_left, legend_h, 2)

    placed_specs: list[dict[str, object]] = []
    order = list(range(len(frame_specs)))
    order.sort(key=lambda idx: int(frame_specs[idx]["outer_h"]) * int(frame_specs[idx]["outer_w"]), reverse=True)

    for idx in order:
        spec = frame_specs[idx]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(1, h - outer_h))
            c0 = int(rng.integers(frame_left_min, w - outer_w))
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue
            _mark_occupied(occupied, r0, c0, outer_h, outer_w)
            spec["r0"] = r0
            spec["c0"] = c0
            placed_specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for spec in placed_specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        marker_color = int(spec["marker_color"])
        count = int(spec["count"])

        grid[r0, c0 : c0 + outer_w] = GEN_FRAME
        grid[r0 + outer_h - 1, c0 : c0 + outer_w] = GEN_FRAME
        grid[r0 : r0 + outer_h, c0] = GEN_FRAME
        grid[r0 : r0 + outer_h, c0 + outer_w - 1] = GEN_FRAME

        expected[r0, c0 : c0 + outer_w] = GEN_FRAME
        expected[r0 + outer_h - 1, c0 : c0 + outer_w] = GEN_FRAME
        expected[r0 : r0 + outer_h, c0] = GEN_FRAME
        expected[r0 : r0 + outer_h, c0 + outer_w - 1] = GEN_FRAME

        target_color = int(legend_colors[count - 1])
        for dr, dc in spec["cells"]:
            rr = r0 + 1 + int(dr)
            cc = c0 + 1 + int(dc)
            grid[rr, cc] = marker_color
            expected[rr, cc] = target_color

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
