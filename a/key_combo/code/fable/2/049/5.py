from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
WINDOW_GRID = tuple((r, c) for r in range(3) for c in range(3))
NEIGHBORS8 = {
    node: tuple(
        (node[0] + dr, node[1] + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0) and 0 <= node[0] + dr < 3 and 0 <= node[1] + dc < 3
    )
    for node in WINDOW_GRID
}
CHAIN_LENGTH_PATTERNS = (
    (3, 2),
    (4, 2),
    (5,),
    (3, 3),
    (4, 3),
    (3, 2, 2),
    (5, 2),
    (4, 2, 2),
    (6,),
    (5, 3),
    (4, 4),
)
WINDOW_SIZES = (3, 5, 7, 9)
MAX_GENERATION_ATTEMPTS = 400
MAX_PATH_ATTEMPTS = 200


def _get_bg(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    return int(Counter(map(int, flat)).most_common(1)[0][0])


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _spans_between_bands(total: int, bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(bands) == 2
    spans = [
        (0, bands[0][0] - 1),
        (bands[0][1] + 1, bands[1][0] - 1),
        (bands[1][1] + 1, total - 1),
    ]
    assert all(start <= end for start, end in spans)
    return spans


def _find_separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []

    for color in map(int, np.unique(g)):
        row_ids = [r for r in range(h) if bool(np.all(g[r, :] == color))]
        col_ids = [c for c in range(w) if bool(np.all(g[:, c] == color))]
        row_bands = _group_runs(row_ids)
        col_bands = _group_runs(col_ids)
        if len(row_bands) == 2 and len(col_bands) == 2:
            candidates.append((color, row_bands, col_bands))

    assert len(candidates) == 1, "expected exactly one separator color with two row and two col bands"
    return candidates[0]


def _window_top_left(cell_h: int, cell_w: int, wr: int, wc: int) -> tuple[int, int]:
    return wr * (cell_h + 1), wc * (cell_w + 1)


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    separator_color, row_bands, col_bands = _find_separator_structure(g)
    row_spans = _spans_between_bands(h, row_bands)
    col_spans = _spans_between_bands(w, col_bands)

    heights = [r1 - r0 + 1 for r0, r1 in row_spans]
    widths = [c1 - c0 + 1 for c0, c1 in col_spans]
    assert len(set(heights)) == 1 and len(set(widths)) == 1, "windows must have uniform size"
    cell_h = int(heights[0])
    cell_w = int(widths[0])
    assert cell_h >= 3 and cell_w >= 3 and cell_h % 2 == 1 and cell_w % 2 == 1

    non_separator = np.ones_like(g, dtype=bool)
    for r0, r1 in row_bands:
        non_separator[r0 : r1 + 1, :] = False
    for c0, c1 in col_bands:
        non_separator[:, c0 : c1 + 1] = False
    assert bool(non_separator.any())
    bg = _get_bg(g[non_separator])
    assert bg == BLACK, "window background must be black"

    center_r = cell_h // 2
    center_c = cell_w // 2

    windows: dict[tuple[int, int], dict[str, object]] = {}
    indegree = {node: 0 for node in WINDOW_GRID}

    for wr, (r0, r1) in enumerate(row_spans):
        for wc, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            marker_coords = np.argwhere(patch != bg)
            assert marker_coords.shape == (1, 2), "each window must contain exactly one colored marker"

            mr, mc = map(int, marker_coords[0])
            dr = int(mr - center_r)
            dc = int(mc - center_c)
            assert dr in (-1, 0, 1) and dc in (-1, 0, 1), "marker must be center or one of eight offsets"

            color = int(patch[mr, mc])
            assert color not in (bg, separator_color)

            target = None
            if (dr, dc) != (0, 0):
                target = (wr + dr, wc + dc)
                assert 0 <= target[0] < 3 and 0 <= target[1] < 3, "pointer must target an existing neighbor"
                indegree[target] += 1

            windows[(wr, wc)] = {
                "row_span": (int(r0), int(r1)),
                "col_span": (int(c0), int(c1)),
                "marker": (int(mr), int(mc)),
                "offset": (int(dr), int(dc)),
                "color": color,
                "target": target,
            }

    assert max(indegree.values()) <= 1, "each window may be targeted by at most one pointer"

    heads = [node for node in WINDOW_GRID if windows[node]["target"] is not None and indegree[node] == 0]
    offcenter = {node for node in WINDOW_GRID if windows[node]["target"] is not None}
    chains: list[list[tuple[int, int]]] = []
    chain_members: set[tuple[int, int]] = set()

    for head in heads:
        chain: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        cursor = head
        head_color = int(windows[head]["color"])

        while True:
            assert cursor not in seen, "pointers must not form a cycle"
            seen.add(cursor)
            chain.append(cursor)
            chain_members.add(cursor)

            target = windows[cursor]["target"]
            if target is None:
                break
            cursor = target

        assert len(chain) >= 2, "each chain must end at a distinct centered tail"
        assert windows[chain[-1]]["offset"] == (0, 0), "chain tails must use centered markers"
        for middle in chain[1:-1]:
            assert int(windows[middle]["color"]) != head_color
        chains.append(chain)

    pointed = {windows[node]["target"] for node in offcenter}
    assert offcenter.issubset(chain_members), "every off-center marker must belong to a head-started chain"
    assert pointed.issubset(chain_members), "every pointed window must belong to some chain"

    isolates = [
        node
        for node in WINDOW_GRID
        if windows[node]["target"] is None and indegree[node] == 0
    ]

    return {
        "grid": g,
        "bg": bg,
        "separator_color": int(separator_color),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "windows": windows,
        "indegree": indegree,
        "heads": heads,
        "chains": chains,
        "isolates": isolates,
    }


def _paint_output(analysis: dict[str, object], color_by_window: dict[tuple[int, int], int]) -> np.ndarray:
    g = np.asarray(analysis["grid"], dtype=int)
    out = np.full_like(g, int(analysis["bg"]))
    sep = int(analysis["separator_color"])

    for r0, r1 in analysis["row_bands"]:
        out[int(r0) : int(r1) + 1, :] = sep
    for c0, c1 in analysis["col_bands"]:
        out[:, int(c0) : int(c1) + 1] = sep

    for node, color in color_by_window.items():
        r0, r1 = analysis["windows"][node]["row_span"]
        c0, c1 = analysis["windows"][node]["col_span"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return out


def _expected_output(analysis: dict[str, object]) -> np.ndarray:
    color_by_window: dict[tuple[int, int], int] = {}
    windows = analysis["windows"]
    for chain in analysis["chains"]:
        head_color = int(windows[chain[0]]["color"])
        for node in chain:
            color_by_window[node] = head_color
    return _paint_output(analysis, color_by_window)


def _naive_own_color_output(analysis: dict[str, object]) -> np.ndarray:
    color_by_window: dict[tuple[int, int], int] = {}
    windows = analysis["windows"]
    for chain in analysis["chains"]:
        for node in chain:
            color_by_window[node] = int(windows[node]["color"])
    return _paint_output(analysis, color_by_window)


def _naive_parent_color_output(analysis: dict[str, object]) -> np.ndarray:
    color_by_window: dict[tuple[int, int], int] = {}
    windows = analysis["windows"]
    for chain in analysis["chains"]:
        color_by_window[chain[0]] = int(windows[chain[0]]["color"])
        for prev, node in zip(chain, chain[1:]):
            color_by_window[node] = int(windows[prev]["color"])
    return _paint_output(analysis, color_by_window)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace every off-center window marker backward to its unpointed chain head, then flood the whole chain with that head color.

    Input structure:
    - Black background with grey full-row and full-column separators that split the grid into a 3x3 array of equal windows.
    - Each window contains exactly one non-black marker pixel, either at the window center or at one of the eight neighboring offset positions.
    - An off-center marker's offset direction points to an existing adjacent window; centered markers do not point onward.

    Rule:
    - Link windows into pointer chains: a window points to the neighbor indicated by its local offset, and an off-center window with no incoming pointer is a chain head.
    - For each chain, follow pointers until the centered tail and recolor every window on that chain with the chain head's marker color.
    - Windows that belong to no chain become entirely black, the original marker pixels disappear under the fill, and the grey separators stay unchanged.

    Key insight:
    - Intermediate marker colors are only forwarding addresses; the visible fill color is determined by tracing all the way back to the unique unpointed source window rather than by the current marker or the immediate predecessor.
    """
    analysis = _analyze_input(grid)
    return _expected_output(analysis)


def _extend_path(
    path: list[tuple[int, int]],
    available: set[tuple[int, int]],
    length: int,
) -> bool:
    if len(path) == length:
        return True

    neighbors = [node for node in NEIGHBORS8[path[-1]] if node in available and node not in path]
    random.shuffle(neighbors)
    for node in neighbors:
        path.append(node)
        if _extend_path(path, available, length):
            return True
        path.pop()
    return False


def _sample_simple_path(available: set[tuple[int, int]], length: int) -> list[tuple[int, int]] | None:
    starts = list(available)
    random.shuffle(starts)
    for start in starts:
        path = [start]
        if _extend_path(path, available, length):
            if random.random() < 0.5:
                path.reverse()
            return path
    return None


def _path_edge_kinds(paths: list[list[tuple[int, int]]]) -> tuple[bool, bool]:
    has_diagonal = False
    has_cardinal = False
    for path in paths:
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            dr = int(r1 - r0)
            dc = int(c1 - c0)
            if abs(dr) == 1 and abs(dc) == 1:
                has_diagonal = True
            else:
                has_cardinal = True
    return has_diagonal, has_cardinal


def _sample_paths(lengths: tuple[int, ...]) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    target_nodes = sum(lengths)
    assert target_nodes < len(WINDOW_GRID)

    for _ in range(MAX_PATH_ATTEMPTS):
        available = set(WINDOW_GRID)
        paths: list[list[tuple[int, int]]] = []
        ok = True
        for length in sorted(lengths, reverse=True):
            path = _sample_simple_path(available, length)
            if path is None:
                ok = False
                break
            paths.append(path)
            available.difference_update(path)
        if not ok:
            continue

        has_diagonal, has_cardinal = _path_edge_kinds(paths)
        if not has_diagonal or not has_cardinal:
            continue

        random.shuffle(paths)
        isolates = list(available)
        random.shuffle(isolates)
        return paths, isolates

    return None


def _assign_marker_colors(
    paths: list[list[tuple[int, int]]],
    isolates: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    head_colors = random.sample(PALETTE, len(paths))

    for head_color, path in zip(head_colors, paths):
        colors[path[0]] = int(head_color)
        non_head_choices = [color for color in PALETTE if color != head_color]
        for node in path[1:]:
            colors[node] = int(random.choice(non_head_choices))

    for node in isolates:
        colors[node] = int(random.choice(PALETTE))

    assert len(colors) == 9
    return colors


def _render_input(
    cell_h: int,
    cell_w: int,
    paths: list[list[tuple[int, int]]],
    isolates: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
) -> np.ndarray:
    height = 3 * cell_h + 2
    width = 3 * cell_w + 2
    grid = np.full((height, width), BLACK, dtype=int)

    for sep_r in (cell_h, 2 * cell_h + 1):
        grid[sep_r, :] = GREY
    for sep_c in (cell_w, 2 * cell_w + 1):
        grid[:, sep_c] = GREY

    center_r = cell_h // 2
    center_c = cell_w // 2
    offsets = {node: (0, 0) for node in WINDOW_GRID}

    for path in paths:
        for current, nxt in zip(path, path[1:]):
            offsets[current] = (int(nxt[0] - current[0]), int(nxt[1] - current[1]))
        offsets[path[-1]] = (0, 0)

    for node in isolates:
        offsets[node] = (0, 0)

    for wr, wc in WINDOW_GRID:
        r0, c0 = _window_top_left(cell_h, cell_w, wr, wc)
        dr, dc = offsets[(wr, wc)]
        marker_r = r0 + center_r + int(dr)
        marker_c = c0 + center_c + int(dc)
        grid[marker_r, marker_c] = int(colors[(wr, wc)])

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_input(grid)
    except AssertionError:
        return False

    if not analysis["chains"]:
        return False
    if not any(len(chain) >= 3 for chain in analysis["chains"]):
        return False
    if not analysis["isolates"]:
        return False

    has_diagonal, has_cardinal = _path_edge_kinds(analysis["chains"])
    if not has_diagonal or not has_cardinal:
        return False

    expected = _expected_output(analysis)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, _naive_own_color_output(analysis)):
        return False
    if np.array_equal(expected, _naive_parent_color_output(analysis)):
        return False

    head_colors = {int(analysis["windows"][chain[0]]["color"]) for chain in analysis["chains"]}
    if len(head_colors) != len(analysis["chains"]):
        return False

    for chain in analysis["chains"]:
        head_color = int(analysis["windows"][chain[0]]["color"])
        if not any(int(analysis["windows"][node]["color"]) != head_color for node in chain[1:]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        cell_h = random.choice(WINDOW_SIZES)
        cell_w = random.choice(WINDOW_SIZES)
        pattern = random.choice(CHAIN_LENGTH_PATTERNS)
        sampled = _sample_paths(pattern)
        if sampled is None:
            continue

        paths, isolates = sampled
        colors = _assign_marker_colors(paths, isolates)
        grid = _render_input(cell_h, cell_w, paths, isolates, colors)
        if _semantic_checks(grid):
            return grid

    raise RuntimeError("failed to generate a valid pointer-chain recoloring puzzle")
