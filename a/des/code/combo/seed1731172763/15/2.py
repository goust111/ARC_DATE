from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
EDGE_ORDER = ("top", "bottom", "left", "right")
MAX_GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _parse_strip_runs(values: np.ndarray, bg: int) -> list[tuple[int, int]] | None:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(values.shape[0])
    while i < n:
        if int(values[i]) == bg:
            i += 1
            continue
        j = i
        while j < n and int(values[j]) != bg:
            j += 1
        if j - i != 2:
            return None
        runs.append((int(i), int(j)))
        i = j
    return runs or None


def _detect_strip(
    grid: np.ndarray, bg: int
) -> tuple[str, dict[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, dict[int, int], list[tuple[int, int]]]] = []

    top_runs = _parse_strip_runs(g[0, :], bg)
    if top_runs is not None:
        mapping = {int(g[0, c0]): int(g[0, c0 + 1]) for c0, _ in top_runs}
        cells = [(0, c) for c0, c1 in top_runs for c in range(c0, c1)]
        candidates.append(("top", mapping, cells))

    bottom_runs = _parse_strip_runs(g[h - 1, :], bg)
    if bottom_runs is not None:
        mapping = {int(g[h - 1, c0]): int(g[h - 1, c0 + 1]) for c0, _ in bottom_runs}
        cells = [(h - 1, c) for c0, c1 in bottom_runs for c in range(c0, c1)]
        candidates.append(("bottom", mapping, cells))

    left_runs = _parse_strip_runs(g[:, 0], bg)
    if left_runs is not None:
        mapping = {int(g[r0, 0]): int(g[r0 + 1, 0]) for r0, _ in left_runs}
        cells = [(r, 0) for r0, r1 in left_runs for r in range(r0, r1)]
        candidates.append(("left", mapping, cells))

    right_runs = _parse_strip_runs(g[:, w - 1], bg)
    if right_runs is not None:
        mapping = {int(g[r0, w - 1]): int(g[r0 + 1, w - 1]) for r0, _ in right_runs}
        cells = [(r, w - 1) for r0, r1 in right_runs for r in range(r0, r1)]
        candidates.append(("right", mapping, cells))

    if not candidates:
        raise ValueError("no edge strip found")

    candidates.sort(key=lambda item: (-len(item[2]), EDGE_ORDER.index(item[0])))
    return candidates[0]


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            r0 = int(coords_arr[:, 0].min())
            c0 = int(coords_arr[:, 1].min())
            r1 = int(coords_arr[:, 0].max())
            c1 = int(coords_arr[:, 1].max())
            rel = coords_arr - np.array([r0, c0], dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (r0, c0, r1, c1),
                    "rel": rel,
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                }
            )
    return objects


def _sort_key(obj: dict[str, object], edge: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = obj["bbox"]  # type: ignore[misc]
    color = int(obj["color"])
    if edge == "top":
        return (int(r0), int(c0), int(r1), color)
    if edge == "bottom":
        return (int(h - 1 - r1), int(c0), int(r0), color)
    if edge == "left":
        return (int(c0), int(r0), int(c1), color)
    return (int(w - 1 - c1), int(r0), int(c0), color)


def _pack_objects(
    shape: tuple[int, int], objects: list[dict[str, object]], edge: str, bg: int
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    ordered = sorted(objects, key=lambda obj: _sort_key(obj, edge, shape))

    for obj in ordered:
        color = int(obj["paint_color"])
        if color == bg:
            continue

        rel = np.asarray(obj["rel"], dtype=int)
        r0, c0, _r1, _c1 = obj["bbox"]  # type: ignore[misc]
        height = int(obj["height"])
        width = int(obj["width"])

        if edge == "top":
            anchors = range(0, int(r0) + 1)
            candidates = ((cand_r0, int(c0)) for cand_r0 in anchors)
        elif edge == "bottom":
            anchors = range(h - height, int(r0) - 1, -1)
            candidates = ((cand_r0, int(c0)) for cand_r0 in anchors)
        elif edge == "left":
            anchors = range(0, int(c0) + 1)
            candidates = ((int(r0), cand_c0) for cand_c0 in anchors)
        else:
            anchors = range(w - width, int(c0) - 1, -1)
            candidates = ((int(r0), cand_c0) for cand_c0 in anchors)

        placed = False
        for cand_r0, cand_c0 in candidates:
            rr = rel[:, 0] + int(cand_r0)
            cc = rel[:, 1] + int(cand_c0)
            if np.any(occupied[rr, cc]):
                continue
            occupied[rr, cc] = True
            out[rr, cc] = color
            placed = True
            break

        if not placed:
            raise ValueError("unable to pack object without overlap")

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a color-translation strip on one edge, recolor the interior objects,
    then pack them toward that same edge while preserving their orthogonal
    positions and avoiding overlap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    edge, mapping, strip_cells = _detect_strip(g, bg)

    work = g.copy()
    for r, c in strip_cells:
        work[r, c] = bg

    objects = _extract_objects(work, bg)
    for obj in objects:
        obj["paint_color"] = int(mapping.get(int(obj["color"]), int(obj["color"])))

    return _pack_objects(g.shape, objects, edge, bg)


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    return np.array(norm, dtype=int)


def _random_polyomino() -> np.ndarray:
    for _ in range(80):
        raw_h = random.randint(2, 4)
        raw_w = random.randint(2, 5)
        target_area = random.randint(3, min(7, raw_h * raw_w))
        cells = {(random.randrange(raw_h), random.randrange(raw_w))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < raw_h and 0 <= cc < raw_w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue
        norm = _normalize_cells(cells)
        h = int(norm[:, 0].max()) + 1
        w = int(norm[:, 1].max()) + 1
        if h <= 4 and w <= 5:
            return norm
    raise RuntimeError("failed to sample polyomino")


def _random_shape() -> np.ndarray:
    if random.random() < 0.4:
        h = random.randint(1, 3)
        w = random.randint(2, 5)
        if random.random() < 0.5:
            h, w = w, h
        cells = {(r, c) for r in range(h) for c in range(w)}
        return _normalize_cells(cells)
    return _random_polyomino()


def _paint_cells(grid: np.ndarray, cells: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr = cells[:, 0] + int(r0)
    cc = cells[:, 1] + int(c0)
    grid[rr, cc] = int(color)


def _can_place(occupied: np.ndarray, cells: np.ndarray, r0: int, c0: int) -> bool:
    rr = cells[:, 0] + int(r0)
    cc = cells[:, 1] + int(c0)
    return not bool(np.any(occupied[rr, cc]))


def _sample_output_layout(
    h: int, w: int, edge: str, shapes: list[np.ndarray]
) -> list[dict[str, object]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[dict[str, object]] = []

    for cells in shapes:
        height = int(cells[:, 0].max()) + 1
        width = int(cells[:, 1].max()) + 1

        placed = False
        for _ in range(80):
            if edge in ("top", "bottom"):
                if w - width - 2 < 0:
                    return None
                c0 = random.randint(1, w - width - 1)
                row_iter = range(0, h - height + 1) if edge == "top" else range(h - height, -1, -1)
                for r0 in row_iter:
                    if _can_place(occupied, cells, r0, c0):
                        rr = cells[:, 0] + int(r0)
                        cc = cells[:, 1] + int(c0)
                        occupied[rr, cc] = True
                        placements.append(
                            {
                                "cells": cells,
                                "out_r0": int(r0),
                                "out_c0": int(c0),
                                "height": height,
                                "width": width,
                            }
                        )
                        placed = True
                        break
            else:
                if h - height - 2 < 0:
                    return None
                r0 = random.randint(1, h - height - 1)
                col_iter = range(0, w - width + 1) if edge == "left" else range(w - width, -1, -1)
                for c0 in col_iter:
                    if _can_place(occupied, cells, r0, c0):
                        rr = cells[:, 0] + int(r0)
                        cc = cells[:, 1] + int(c0)
                        occupied[rr, cc] = True
                        placements.append(
                            {
                                "cells": cells,
                                "out_r0": int(r0),
                                "out_c0": int(c0),
                                "height": height,
                                "width": width,
                            }
                        )
                        placed = True
                        break
            if placed:
                break
        if not placed:
            return None

    return placements


def _sample_input_colors(output_colors: list[int], mapped_indices: list[int]) -> list[int] | None:
    mapped_set = set(mapped_indices)
    input_colors: list[int | None] = [None] * len(output_colors)
    used_input = set()

    for i, out_color in enumerate(output_colors):
        if i not in mapped_set:
            input_colors[i] = int(out_color)
            used_input.add(int(out_color))

    remaining = [color for color in PALETTE if color not in used_input]
    mapped_indices = list(mapped_indices)
    random.shuffle(mapped_indices)

    def backtrack(pos: int, pool: list[int]) -> bool:
        if pos == len(mapped_indices):
            return True
        idx = mapped_indices[pos]
        options = [color for color in pool if color != output_colors[idx]]
        random.shuffle(options)
        for color in options:
            input_colors[idx] = int(color)
            next_pool = [value for value in pool if value != color]
            if backtrack(pos + 1, next_pool):
                return True
        input_colors[idx] = None
        return False

    if not backtrack(0, remaining):
        return None
    return [int(color) for color in input_colors]


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _place_strip(
    grid: np.ndarray, edge: str, pairs: list[tuple[int, int]]
) -> None:
    h, w = grid.shape
    span = w if edge in ("top", "bottom") else h
    used = 2 * len(pairs) + max(0, len(pairs) - 1)
    extra = span - used
    if extra < 0:
        raise ValueError("strip does not fit")
    gaps = _random_composition(extra, len(pairs) + 1)

    pos = int(gaps[0])
    if edge == "top":
        row = 0
        for i, (src, dst) in enumerate(pairs):
            grid[row, pos] = int(src)
            grid[row, pos + 1] = int(dst)
            pos += 2
            if i < len(pairs) - 1:
                pos += 1 + int(gaps[i + 1])
    elif edge == "bottom":
        row = h - 1
        for i, (src, dst) in enumerate(pairs):
            grid[row, pos] = int(src)
            grid[row, pos + 1] = int(dst)
            pos += 2
            if i < len(pairs) - 1:
                pos += 1 + int(gaps[i + 1])
    elif edge == "left":
        col = 0
        for i, (src, dst) in enumerate(pairs):
            grid[pos, col] = int(src)
            grid[pos + 1, col] = int(dst)
            pos += 2
            if i < len(pairs) - 1:
                pos += 1 + int(gaps[i + 1])
    else:
        col = w - 1
        for i, (src, dst) in enumerate(pairs):
            grid[pos, col] = int(src)
            grid[pos + 1, col] = int(dst)
            pos += 2
            if i < len(pairs) - 1:
                pos += 1 + int(gaps[i + 1])


def _edge_shift_limit(placement: dict[str, object], edge: str, shape: tuple[int, int]) -> int:
    h, w = shape
    out_r0 = int(placement["out_r0"])
    out_c0 = int(placement["out_c0"])
    height = int(placement["height"])
    width = int(placement["width"])
    if edge == "top":
        return int(h - (out_r0 + height))
    if edge == "bottom":
        return int(out_r0)
    if edge == "left":
        return int(w - (out_c0 + width))
    return int(out_c0)


def _shift_input_position(
    placement: dict[str, object], edge: str, shift: int
) -> tuple[int, int]:
    out_r0 = int(placement["out_r0"])
    out_c0 = int(placement["out_c0"])
    if edge == "top":
        return int(out_r0 + shift), int(out_c0)
    if edge == "bottom":
        return int(out_r0 - shift), int(out_c0)
    if edge == "left":
        return int(out_r0), int(out_c0 + shift)
    return int(out_r0), int(out_c0 - shift)


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(EDGE_ORDER)
    h = random.randint(12, 18)
    w = random.randint(12, 18)

    max_pairs = (w + 1) // 3 if edge in ("top", "bottom") else (h + 1) // 3
    if max_pairs < 1:
        return None

    object_count = random.randint(2, 5)
    shapes = [_random_shape() for _ in range(object_count)]
    placements = _sample_output_layout(h, w, edge, shapes)
    if placements is None:
        return None

    shift_cap = min(_edge_shift_limit(placement, edge, (h, w)) for placement in placements)
    if shift_cap < 2:
        return None
    shift = random.randint(2, shift_cap)

    max_mapped = min(object_count, max_pairs)
    mapped_count = random.randint(1, max_mapped)
    mapped_indices = sorted(random.sample(range(object_count), mapped_count))

    output_colors = random.sample(PALETTE, object_count)
    input_colors = _sample_input_colors(output_colors, mapped_indices)
    if input_colors is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    mapping_pairs: list[tuple[int, int]] = []
    for idx, placement in enumerate(placements):
        cells = np.asarray(placement["cells"], dtype=int)
        in_r0, in_c0 = _shift_input_position(placement, edge, shift)
        out_r0 = int(placement["out_r0"])
        out_c0 = int(placement["out_c0"])
        _paint_cells(inp, cells, in_r0, in_c0, input_colors[idx])
        _paint_cells(expected, cells, out_r0, out_c0, output_colors[idx])
        if idx in mapped_indices:
            mapping_pairs.append((int(input_colors[idx]), int(output_colors[idx])))

    _place_strip(inp, edge, mapping_pairs)

    if Counter(inp.ravel()).most_common(1)[0][0] != BG:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
