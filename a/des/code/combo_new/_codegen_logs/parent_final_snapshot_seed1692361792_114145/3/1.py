from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for rr, cc in _neighbors4(r, c, h, w):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            parts.append(coords)
    return parts


def _segment_legend(row: np.ndarray, bg: int) -> tuple[int, int, list[int]] | None:
    cols = np.flatnonzero(row != int(bg))
    if cols.size == 0:
        return None
    start = int(cols[0])
    end = int(cols[-1]) + 1
    if end - start != int(cols.size):
        return None
    colors = [int(x) for x in row[start:end].tolist()]
    if len(colors) != len(set(colors)):
        return None
    return start, end, colors


def _legend_info(grid: np.ndarray, bg: int) -> tuple[str, int, int, list[int]]:
    candidates: list[tuple[str, int, int, list[int]]] = []
    top = _segment_legend(grid[0], bg)
    if top is not None:
        candidates.append(("top", top[0], top[1], top[2]))
    bottom = _segment_legend(grid[-1], bg)
    if bottom is not None:
        candidates.append(("bottom", bottom[0], bottom[1], bottom[2]))
    assert len(candidates) == 1, "expected exactly one legend row on the top or bottom edge"
    return candidates[0]


def _prefix(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix_sum: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(
        prefix_sum[r1 + 1, c1 + 1]
        - prefix_sum[r0, c1 + 1]
        - prefix_sum[r1 + 1, c0]
        + prefix_sum[r0, c0]
    )


def _frame_from_bbox(
    body: np.ndarray,
    fence_color: int,
    bbox: tuple[int, int, int, int],
    max_holes: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    patch = body[r0 : r1 + 1, c0 : c1 + 1]
    if (
        int(patch[0, 0]) != int(fence_color)
        or int(patch[0, -1]) != int(fence_color)
        or int(patch[-1, 0]) != int(fence_color)
        or int(patch[-1, -1]) != int(fence_color)
    ):
        return None

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    color_mask = patch == int(fence_color)
    if np.any(color_mask & ~border_mask):
        return None

    if h > 2 and w > 2 and np.any(patch[1:-1, 1:-1] != 0):
        return None

    holes = int(border_mask.sum() - color_mask.sum())
    if holes < 1 or holes > int(max_holes):
        return None

    cells = frozenset(
        (int(r0 + rr), int(c0 + cc))
        for rr, cc in np.argwhere(color_mask)
    )
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "cells": cells,
        "holes": holes,
        "area": len(cells),
    }


def _frames_from_components(body: np.ndarray, fence_color: int, max_holes: int) -> list[dict[str, object]] | None:
    comps = _components(body == int(fence_color))
    if not comps:
        return []

    frames: list[dict[str, object]] = []
    covered: set[tuple[int, int]] = set()
    for coords in comps:
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        frame = _frame_from_bbox(
            body,
            fence_color,
            (min(rows), min(cols), max(rows), max(cols)),
            max_holes,
        )
        if frame is None or set(coords) != set(frame["cells"]):
            return None
        frames.append(frame)
        covered.update(coords)

    all_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(body == int(fence_color))
    }
    return frames if covered == all_cells else None


def _scan_frame_candidates(body: np.ndarray, fence_color: int, max_holes: int) -> list[dict[str, object]]:
    h, w = body.shape
    fence_mask = body == int(fence_color)
    prefix_sum = _prefix(fence_mask)
    candidates: dict[tuple[int, int, int, int], dict[str, object]] = {}

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not fence_mask[r0, c0]:
                continue
            for r1 in range(r0 + 2, h):
                if not fence_mask[r1, c0]:
                    continue
                for c1 in range(c0 + 2, w):
                    if not fence_mask[r0, c1] or not fence_mask[r1, c1]:
                        continue
                    border_count = _rect_sum(prefix_sum, r0, c0, r1, c1) - _rect_sum(
                        prefix_sum, r0 + 1, c0 + 1, r1 - 1, c1 - 1
                    )
                    perimeter = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    holes = int(perimeter - border_count)
                    if holes < 1 or holes > int(max_holes):
                        continue
                    frame = _frame_from_bbox(body, fence_color, (r0, c0, r1, c1), max_holes)
                    if frame is not None:
                        candidates[(r0, c0, r1, c1)] = frame
    return list(candidates.values())


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _solve_cover(
    candidates: list[dict[str, object]],
    all_cells: set[tuple[int, int]],
) -> list[dict[str, object]] | None:
    by_cell: dict[tuple[int, int], list[int]] = {cell: [] for cell in all_cells}
    for idx, cand in enumerate(candidates):
        for cell in cand["cells"]:
            if cell in by_cell:
                by_cell[cell].append(idx)

    ordered = sorted(
        range(len(candidates)),
        key=lambda i: (
            int(candidates[i]["bbox"][0]),
            int(candidates[i]["bbox"][1]),
            int(candidates[i]["bbox"][2]),
            int(candidates[i]["bbox"][3]),
        ),
    )

    def search(
        remaining: set[tuple[int, int]],
        chosen_ids: list[int],
    ) -> list[int] | None:
        if not remaining:
            return chosen_ids

        best_cell: tuple[int, int] | None = None
        best_options: list[int] | None = None
        for cell in remaining:
            options: list[int] = []
            for idx in by_cell[cell]:
                cand = candidates[idx]
                if not cand["cells"] <= remaining:
                    continue
                if any(_boxes_overlap(cand["bbox"], candidates[j]["bbox"]) for j in chosen_ids):
                    continue
                options.append(idx)
            if not options:
                return None
            if best_options is None or len(options) < len(best_options):
                best_cell = cell
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_cell is not None and best_options is not None
        for idx in sorted(best_options, key=lambda i: ordered.index(i)):
            result = search(remaining - set(candidates[idx]["cells"]), chosen_ids + [idx])
            if result is not None:
                return result
        return None

    solution = search(set(all_cells), [])
    if solution is None:
        return None
    return [candidates[idx] for idx in solution]


def _find_frames(body: np.ndarray, fence_color: int, max_holes: int) -> list[dict[str, object]]:
    direct = _frames_from_components(body, fence_color, max_holes)
    if direct is not None:
        return sorted(direct, key=lambda frame: frame["bbox"])

    all_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(body == int(fence_color))
    }
    candidates = _scan_frame_candidates(body, fence_color, max_holes)
    solved = _solve_cover(candidates, all_cells)
    assert solved is not None, "failed to reconstruct fence rectangles"
    covered = set().union(*(set(frame["cells"]) for frame in solved)) if solved else set()
    assert covered == all_cells, "reconstructed rectangles must cover all fence pixels"
    return sorted(solved, key=lambda frame: frame["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow fence using the legend color indexed by its number of border gaps.

    Input structure:
    - The background is black.
    - Several hollow rectangular fences are drawn in one shared non-background color, with some single-pixel border gaps left as background.
    - A horizontal legend strip sits on the top or bottom edge; its distinct colors read left-to-right as counts 1, 2, 3, ...

    Rule:
    - Find each fence and count how many missing border pixels it has.
    - Use that count as a 1-based index into the legend strip.
    - Recolor all present pixels of that fence border with the selected legend color.
    - Keep the missing pixels as background and leave the legend unchanged.

    Key insight:
    - The fence geometry supplies a hole count, and the edge legend converts that count into the repaint color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    side, _start, _end, legend_colors = _legend_info(g, bg)

    if side == "top":
        body = g[1:, :]
        row_offset = 1
    else:
        body = g[:-1, :]
        row_offset = 0

    non_bg_colors = {int(x) for x in np.unique(body) if int(x) != int(bg)}
    assert len(non_bg_colors) == 1, "expected exactly one fence color outside the legend row"
    fence_color = next(iter(non_bg_colors))

    frames = _find_frames(body, fence_color, len(legend_colors))
    out = g.copy()
    for frame in frames:
        color = int(legend_colors[int(frame["holes"]) - 1])
        for r, c in frame["cells"]:
            out[int(r) + row_offset, int(c)] = color
    return out


# --------------- generator ---------------

BG = 0
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 26)
LEGEND_LEN_RANGE = (3, 5)
FRAME_COUNT_RANGE = (2, 4)
FRAME_H_RANGE = (4, 8)
FRAME_W_RANGE = (4, 9)
PLACEMENT_PADDING = 1
PLACEMENT_RETRIES = 200
FRAME_RETRIES = 200
GENERATION_RETRIES = 400


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _border_hole_positions(height: int, width: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for c in range(1, width - 1):
        positions.append((0, c))
        positions.append((height - 1, c))
    for r in range(1, height - 1):
        positions.append((r, 0))
        positions.append((r, width - 1))
    return positions


def _make_frame_patch(color: int, height: int, width: int, holes: int) -> np.ndarray | None:
    choices = _border_hole_positions(height, width)
    if holes > len(choices):
        return None
    for _ in range(FRAME_RETRIES):
        patch = np.zeros((height, width), dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        for r, c in random.sample(choices, holes):
            patch[r, c] = BG
        if _is_connected(patch == int(color)):
            return patch
    return None


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_hole_counts(frame_count: int, legend_len: int) -> list[int]:
    for _ in range(FRAME_RETRIES):
        counts = [random.randint(1, legend_len) for _ in range(frame_count)]
        if frame_count == 1 or len(set(counts)) >= min(2, legend_len):
            return counts
    return [1 + (i % legend_len) for i in range(frame_count)]


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    occupied: list[tuple[int, int, int, int]],
    legend_side: str,
) -> tuple[int, int] | None:
    h, w = patch.shape
    grid_h, grid_w = grid.shape
    if legend_side == "top":
        r_min = 2
        r_max = grid_h - h
    else:
        r_min = 0
        r_max = grid_h - h - 2
    c_min = 0
    c_max = grid_w - w
    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_boxes_too_close(bbox, other, PLACEMENT_PADDING) for other in occupied):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]
        occupied.append(bbox)
        return r0, c0
    return None


def _draw_legend(grid: np.ndarray, colors: list[int], side: str) -> None:
    start = random.randint(0, grid.shape[1] - len(colors))
    row = 0 if side == "top" else grid.shape[0] - 1
    grid[row, start : start + len(colors)] = np.array(colors, dtype=int)


def _sample_frame_shape(hole_count: int) -> tuple[int, int, np.ndarray] | None:
    dims = [
        (height, width)
        for height in range(FRAME_H_RANGE[0], FRAME_H_RANGE[1] + 1)
        for width in range(FRAME_W_RANGE[0], FRAME_W_RANGE[1] + 1)
        if len(_border_hole_positions(height, width)) >= hole_count
    ]
    random.shuffle(dims)
    for height, width in dims:
        patch = _make_frame_patch(1, height, width, hole_count)
        if patch is not None:
            return height, width, patch
    return None


def _build_sample() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    legend_side = random.choice(["top", "bottom"])
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)

    colors = list(range(1, 10))
    random.shuffle(colors)
    legend_colors = colors[:legend_len]
    fence_color = int(colors[legend_len])

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _draw_legend(grid, legend_colors, legend_side)

    occupied: list[tuple[int, int, int, int]] = []
    frames: list[tuple[np.ndarray, int, int, int]] = []
    for hole_count in _sample_hole_counts(frame_count, legend_len):
        shape = _sample_frame_shape(hole_count)
        if shape is None:
            return None
        _height, _width, raw_patch = shape
        patch = np.where(raw_patch == 1, fence_color, BG)
        placed = _place_patch(grid, patch, occupied, legend_side)
        if placed is None:
            return None
        frames.append((patch, placed[0], placed[1], hole_count))

    out = grid.copy()
    for patch, r0, c0, hole_count in frames:
        color = int(legend_colors[hole_count - 1])
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch == fence_color] = color

    if int(np.count_nonzero(grid == BG)) <= max(int(np.count_nonzero(grid == fence_color)), legend_len):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sample = _build_sample()
        if sample is None:
            continue
        inp, expected = sample
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid fence-legend sample")
