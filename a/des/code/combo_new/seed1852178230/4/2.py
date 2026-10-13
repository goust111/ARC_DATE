from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components_same_color(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == BG or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            top_r = min(r for r, _ in cells)
            top_c = min(c for r, c in cells if r == top_r)
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "size": len(cells),
                    "top_r": top_r,
                    "top_c": top_c,
                }
            )

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor scattered monochrome clusters by size order using a bottom-edge legend.

    Input structure:
    - Background is black.
    - Above the last row, several non-black monochrome clusters appear as 4-connected components.
    - The last row is a legend: distinct single-pixel colors appear left-to-right, with black gaps between them.

    Rule:
    - Ignore the last row and measure the area of every 4-connected same-color cluster.
    - Sort clusters by increasing area; break ties by the row of the topmost pixel, then by its column.
    - Repaint the sorted clusters with the legend colors in left-to-right order.
    - Return a grid of the same size with the recolored clusters and a cleared black last row.

    Key insight: The legend colors are assigned only after globally ordering the interior clusters, so the mapping depends on size and tie-break position rather than spatial proximity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 1

    legend_colors = [int(v) for v in g[-1, :] if int(v) != BG]
    interior = g[:-1, :]
    comps = _components_same_color(interior)
    comps.sort(key=lambda comp: (int(comp["size"]), int(comp["top_r"]), int(comp["top_c"])))
    assert len(legend_colors) >= len(comps), "legend must provide at least one color per cluster"

    out = np.full((h, w), BG, dtype=int)
    for comp, new_color in zip(comps, legend_colors, strict=False):
        for r, c in comp["cells"]:
            out[int(r), int(c)] = int(new_color)
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in cells) + 1
    return _normalize_cells({(c, h - 1 - r) for r, c in cells})


def _flip_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    w = max(c for _, c in cells) + 1
    return _normalize_cells({(r, w - 1 - c) for r, c in cells})


def _random_shape(size: int) -> tuple[tuple[int, int], ...]:
    if size == 1:
        return ((0, 0),)

    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != size:
            continue

        norm = _normalize_cells(cells)
        for _ in range(random.randint(0, 3)):
            norm = _rotate_cells(norm)
        if random.random() < 0.5:
            norm = _flip_cells(norm)

        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h <= 4 and w <= 4:
            return norm

    raise RuntimeError("failed to sample a compact connected shape")


def _shape_spec(size: int, color: int) -> dict[str, object]:
    cells = _random_shape(size)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    top_col_offset = min(c for r, c in cells if r == 0)
    return {
        "size": size,
        "color": color,
        "cells": cells,
        "h": h,
        "w": w,
        "top_col_offset": top_col_offset,
    }


def _candidate_positions(
    spec: dict[str, object],
    body_h: int,
    width: int,
    blocked: np.ndarray,
    fixed_row: int | None = None,
) -> list[tuple[int, int]]:
    shape_h = int(spec["h"])
    shape_w = int(spec["w"])

    row_start = 1
    row_stop = body_h - shape_h - 1
    col_start = 1
    col_stop = width - shape_w - 1
    if row_stop < row_start or col_stop < col_start:
        return []

    if fixed_row is None:
        rows = list(range(row_start, row_stop + 1))
        random.shuffle(rows)
    else:
        if not (row_start <= fixed_row <= row_stop):
            return []
        rows = [fixed_row]

    cols = list(range(col_start, col_stop + 1))
    random.shuffle(cols)
    positions: list[tuple[int, int]] = []
    for r0 in rows:
        for c0 in cols:
            if all(not blocked[r0 + r, c0 + c] for r, c in spec["cells"]):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _mark_blocked(blocked: np.ndarray, spec: dict[str, object], r0: int, c0: int) -> None:
    h, w = map(int, blocked.shape)
    for dr, dc in spec["cells"]:
        rr = r0 + dr
        cc = c0 + dc
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_spec(spec: dict[str, object], r0: int, c0: int) -> None:
    spec["r0"] = r0
    spec["c0"] = c0


def _top_anchor(spec: dict[str, object]) -> tuple[int, int]:
    return int(spec["r0"]), int(spec["c0"]) + int(spec["top_col_offset"])


def _pair_mode_ok(mode: str, specs: list[dict[str, object]]) -> bool:
    keys = [(int(spec["size"]),) + _top_anchor(spec) for spec in specs]
    if mode == "unique":
        return len({size for size, _, _ in keys}) == len(keys)
    if mode == "row_tie":
        return any(
            i < j and keys[i][0] == keys[j][0] and keys[i][1] != keys[j][1]
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
        )
    if mode == "col_tie":
        return any(
            i < j and keys[i][0] == keys[j][0] and keys[i][1] == keys[j][1] and keys[i][2] != keys[j][2]
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
        )
    return False


def _build_expected(shape: tuple[int, int], specs: list[dict[str, object]], legend_colors: list[int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    ordered = sorted(
        specs,
        key=lambda spec: (
            int(spec["size"]),
            int(spec["r0"]),
            int(spec["c0"]) + int(spec["top_col_offset"]),
        ),
    )
    for spec, new_color in zip(ordered, legend_colors, strict=False):
        for dr, dc in spec["cells"]:
            out[int(spec["r0"]) + dr, int(spec["c0"]) + dc] = int(new_color)
    return out


def _legend_row(width: int, legend_colors: list[int]) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    n = len(legend_colors)
    min_len = 2 * n - 1
    slack = width - min_len
    extras = [0] * (n + 1)
    for _ in range(slack):
        extras[random.randrange(n + 1)] += 1

    c = extras[0]
    for i, color in enumerate(legend_colors):
        row[c] = int(color)
        if i < n - 1:
            c += 2 + extras[i + 1]
    return row


def _sample_legend_colors(cluster_colors: list[int]) -> list[int]:
    if random.random() < 0.5:
        legend = cluster_colors[:]
        random.shuffle(legend)
        return legend
    return random.sample(COLORS, len(cluster_colors))


def _sample_sizes(mode: str, count: int) -> list[int]:
    if mode == "unique":
        return random.sample(list(range(1, 9)), count)

    dup_size = random.randint(1, 8)
    others = random.sample([size for size in range(1, 9) if size != dup_size], count - 2)
    sizes = [dup_size, dup_size] + others
    random.shuffle(sizes)
    return sizes


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(("unique", "row_tie", "col_tie"))
    count = random.randint(3, 7) if mode == "unique" else random.randint(4, 7)
    sizes = _sample_sizes(mode, count)
    cluster_colors = random.sample(COLORS, count)
    legend_colors = _sample_legend_colors(cluster_colors)

    specs = [_shape_spec(size, color) for size, color in zip(sizes, cluster_colors, strict=False)]
    pair_indices = []
    if mode != "unique":
        dup_size = next(size for size in sizes if sizes.count(size) == 2)
        pair_indices = [i for i, size in enumerate(sizes) if size == dup_size]

    max_h = max(int(spec["h"]) for spec in specs)
    legend_min_w = 2 * count - 1
    pair_need_w = 0
    if mode == "col_tie":
        pair_need_w = sum(int(specs[i]["w"]) for i in pair_indices) + 4

    min_h = max(10, max_h + 5, count + 6)
    max_h_grid = min(24, min_h + 6)
    min_w = max(12, legend_min_w + 2, pair_need_w + 2)
    max_w_grid = min(28, min_w + 10)
    if min_h > max_h_grid or min_w > max_w_grid:
        return None

    height = random.randint(min_h, max_h_grid)
    width = random.randint(min_w, max_w_grid)
    body_h = height - 2
    blocked = np.zeros((body_h, width), dtype=bool)

    if mode == "col_tie":
        a = specs[pair_indices[0]]
        b = specs[pair_indices[1]]
        candidate_rows = list(range(1, body_h - max(int(a["h"]), int(b["h"])) - 1 + 1))
        random.shuffle(candidate_rows)
        placed = False
        for row in candidate_rows:
            for a_pos in _candidate_positions(a, body_h, width, blocked, fixed_row=row):
                temp = blocked.copy()
                _mark_blocked(temp, a, a_pos[0], a_pos[1])
                b_positions = [
                    pos
                    for pos in _candidate_positions(b, body_h, width, temp, fixed_row=row)
                    if pos[1] + int(b["top_col_offset"]) != a_pos[1] + int(a["top_col_offset"])
                ]
                if not b_positions:
                    continue
                b_pos = random.choice(b_positions)
                _place_spec(a, *a_pos)
                _mark_blocked(blocked, a, a_pos[0], a_pos[1])
                _place_spec(b, *b_pos)
                _mark_blocked(blocked, b, b_pos[0], b_pos[1])
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    elif mode == "row_tie":
        a = specs[pair_indices[0]]
        b = specs[pair_indices[1]]
        placed = False
        for a_pos in _candidate_positions(a, body_h, width, blocked):
            temp = blocked.copy()
            _mark_blocked(temp, a, a_pos[0], a_pos[1])
            b_positions = [pos for pos in _candidate_positions(b, body_h, width, temp) if pos[0] != a_pos[0]]
            if not b_positions:
                continue
            b_pos = random.choice(b_positions)
            _place_spec(a, *a_pos)
            _mark_blocked(blocked, a, a_pos[0], a_pos[1])
            _place_spec(b, *b_pos)
            _mark_blocked(blocked, b, b_pos[0], b_pos[1])
            placed = True
            break
        if not placed:
            return None

    remaining = [spec for spec in specs if "r0" not in spec]
    remaining.sort(key=lambda spec: (int(spec["h"]) * int(spec["w"]), int(spec["size"])), reverse=True)
    for spec in remaining:
        positions = _candidate_positions(spec, body_h, width, blocked)
        if not positions:
            return None
        pos = random.choice(positions)
        _place_spec(spec, *pos)
        _mark_blocked(blocked, spec, pos[0], pos[1])

    if not _pair_mode_ok(mode, specs):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        for dr, dc in spec["cells"]:
            grid[int(spec["r0"]) + dr, int(spec["c0"]) + dc] = int(spec["color"])
    grid[-1, :] = _legend_row(width, legend_colors)

    expected = _build_expected((height, width), specs, legend_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
