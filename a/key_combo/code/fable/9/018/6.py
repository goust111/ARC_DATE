from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BG = 0
MIN_RECT_SPAN = 4
MAX_RECT_SPAN = 6
CHAIN_SPAN = 6
CHAIN_OVERLAP = 3
MAX_GENERATION_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _visible_positions(arr: np.ndarray, bg: int) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = defaultdict(list)
    for idx, value in enumerate(arr.tolist()):
        value = int(value)
        if value != bg:
            positions[value].append(idx)
    return dict(positions)


def _intervals_from_band(arr: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    intervals: dict[int, tuple[int, int]] = {}
    for color, positions in _visible_positions(arr, bg).items():
        assert len(positions) >= 2, "each shadow must expose both endpoints"
        intervals[color] = (int(positions[0]), int(positions[-1]))
    return intervals


def _band_relations(
    arr: np.ndarray, intervals: dict[int, tuple[int, int]]
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], list[int]]]:
    colors = sorted(intervals)
    relations: set[tuple[int, int]] = set()
    evidence: dict[tuple[int, int], list[int]] = {}
    for i, a in enumerate(colors):
        a0, a1 = intervals[a]
        for b in colors[i + 1 :]:
            b0, b1 = intervals[b]
            lo = max(a0, b0)
            hi = min(a1, b1)
            if hi <= lo:
                continue
            forbidden = {a0, a1, b0, b1}
            visible = []
            for pos in range(lo, hi + 1):
                if pos in forbidden:
                    continue
                visible.append((pos, int(arr[pos])))
            if not visible:
                continue
            seen_a = [pos for pos, color in visible if color == a]
            seen_b = [pos for pos, color in visible if color == b]
            assert not (seen_a and seen_b), "pairwise visibility is inconsistent"
            if seen_a:
                relations.add((b, a))
                evidence[(b, a)] = seen_a
            elif seen_b:
                relations.add((a, b))
                evidence[(a, b)] = seen_b
    return relations, evidence


def _topological_total_order(colors: list[int], relations: set[tuple[int, int]]) -> list[int]:
    if len(colors) <= 1:
        return list(colors)

    followers: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower, upper in relations:
        if upper not in followers[lower]:
            followers[lower].add(upper)
            indegree[upper] += 1

    order: list[int] = []
    frontier = sorted(color for color in colors if indegree[color] == 0)
    while frontier:
        assert len(frontier) == 1, "shadow order is not uniquely determined"
        current = frontier.pop()
        order.append(current)
        for nxt in sorted(followers[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)
                frontier.sort()

    assert len(order) == len(colors), "shadow order must be acyclic"
    rank = {color: idx for idx, color in enumerate(order)}
    for a in colors:
        for b in colors:
            if a == b:
                continue
            assert rank[a] != rank[b]
    for lower, upper in relations:
        assert rank[lower] < rank[upper]
    return order


def _analyze_shadows(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H >= 6 and W >= 6
    bg = _get_bg(g)
    assert np.all(g[:-1, :-1] == bg), "main area must be blank in the input"

    bottom = g[-1, :-1]
    right = g[:-1, -1]
    bottom_intervals = _intervals_from_band(bottom, bg)
    right_intervals = _intervals_from_band(right, bg)
    colors = sorted(bottom_intervals)
    assert colors == sorted(right_intervals), "every rectangle color must appear on both bands"

    bottom_relations, bottom_evidence = _band_relations(bottom, bottom_intervals)
    right_relations, right_evidence = _band_relations(right, right_intervals)
    relations = bottom_relations | right_relations
    order = _topological_total_order(colors, relations)

    return {
        "bg": bg,
        "bottom": bottom_intervals,
        "right": right_intervals,
        "order": order,
        "bottom_relations": bottom_relations,
        "right_relations": right_relations,
        "bottom_evidence": bottom_evidence,
        "right_evidence": right_evidence,
    }


def _draw_hollow_rect(grid: np.ndarray, rows: tuple[int, int], cols: tuple[int, int], color: int) -> None:
    r0, r1 = rows
    c0, c1 = cols
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hollow rectangles from the bottom and right shadow bands, then redraw them in layer order.

    Input structure:
    - The main area is blank; only the bottom row and right column contain colored shadow fragments.
    - Each foreground color belongs to one hidden axis-aligned hollow rectangle.
    - A color's bottom-row shadow gives its column span, and its right-column shadow gives its row span.

    Rule:
    - Recover each shadow interval by taking the full span between that color's two outermost visible pixels on the corresponding band.
    - Read pairwise occlusion from overlapping interval interiors: on a shared non-endpoint band cell, the visible color is the upper one.
    - Merge both bands' occlusion relations into the unique total layer order, then draw every hollow rectangle from bottom layer to top layer.
    - Clear both shadow bands back to background in the output.

    Key insight:
    - The shadow bands jointly encode both rectangle geometry and stacking order: endpoints restore broken intervals, while non-endpoint overlaps reveal who covers whom.
    """
    info = _analyze_shadows(grid)
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, int(info["bg"]))

    bottom_intervals: dict[int, tuple[int, int]] = info["bottom"]  # type: ignore[assignment]
    right_intervals: dict[int, tuple[int, int]] = info["right"]  # type: ignore[assignment]
    order: list[int] = info["order"]  # type: ignore[assignment]

    for color in order:
        _draw_hollow_rect(out, right_intervals[color], bottom_intervals[color], color)
    out[-1, :] = int(info["bg"])
    out[:, -1] = int(info["bg"])
    return out


def _find_runs(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    runs = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev))
        start = prev = pos
    runs.append((start, prev))
    return runs


def _split_adjacent_edges(n: int, rng: np.random.Generator) -> set[int]:
    while True:
        mask = rng.integers(0, 2, size=n - 1)
        chosen = {idx for idx, bit in enumerate(mask.tolist()) if bit}
        if 0 < len(chosen) < n - 1:
            return chosen


def _chain_block(size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    lengths = [CHAIN_SPAN for _ in range(size)]
    intervals = [(0, lengths[0] - 1)]
    for idx in range(1, size):
        prev_end = intervals[-1][1]
        start = prev_end - (CHAIN_OVERLAP - 1)
        end = start + lengths[idx] - 1
        intervals.append((start, end))
    return intervals


def _axis_intervals(n: int, overlap_edges: set[int], rng: np.random.Generator) -> list[tuple[int, int]]:
    blocks: list[list[int]] = []
    start = 0
    for edge in range(n - 1):
        if edge not in overlap_edges:
            blocks.append(list(range(start, edge + 1)))
            start = edge + 1
    blocks.append(list(range(start, n)))
    rng.shuffle(blocks)

    intervals: list[tuple[int, int] | None] = [None] * n
    cursor = int(rng.integers(0, 2))
    for block_idx, block in enumerate(blocks):
        if block_idx > 0:
            cursor += int(rng.integers(1, 4))
        local = _chain_block(len(block), rng)
        for local_idx, layer_idx in enumerate(block):
            lo, hi = local[local_idx]
            intervals[layer_idx] = (cursor + lo, cursor + hi)
        cursor += local[-1][1] + 1

    result = [interval for interval in intervals if interval is not None]
    starts = [lo for lo, _ in result]
    ends = [hi for _, hi in result]
    endpoints = starts + ends
    assert len(set(starts)) == n and len(set(ends)) == n
    assert len(set(endpoints)) == len(endpoints)
    return result  # type: ignore[return-value]


def _render_band(length: int, intervals: dict[int, tuple[int, int]], order: list[int], bg: int) -> np.ndarray:
    arr = np.full(length, bg, dtype=int)
    for color in order:
        lo, hi = intervals[color]
        arr[lo : hi + 1] = color
    for color, (lo, hi) in intervals.items():
        arr[lo] = color
        arr[hi] = color
    return arr


def _build_output(
    main_h: int,
    main_w: int,
    row_intervals: dict[int, tuple[int, int]],
    col_intervals: dict[int, tuple[int, int]],
    order: list[int],
    bg: int,
) -> np.ndarray:
    out = np.full((main_h + 1, main_w + 1), bg, dtype=int)
    for color in order:
        _draw_hollow_rect(out, row_intervals[color], col_intervals[color], color)
    out[-1, :] = bg
    out[:, -1] = bg
    return out


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    order: list[int],
    row_intervals: dict[int, tuple[int, int]],
    col_intervals: dict[int, tuple[int, int]],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    info = _analyze_shadows(inp)
    if info["order"] != order:
        return False
    if info["right"] != row_intervals:
        return False
    if info["bottom"] != col_intervals:
        return False
    if not info["bottom_relations"] or not info["right_relations"]:
        return False

    bg = int(info["bg"])
    bottom_positions = _visible_positions(inp[-1, :-1], bg)
    right_positions = _visible_positions(inp[:-1, -1], bg)
    if not any(len(_find_runs(pos)) >= 2 for pos in bottom_positions.values()):
        return False
    if not any(len(_find_runs(pos)) >= 2 for pos in right_positions.values()):
        return False

    main_area = expected[:-1, :-1]
    for color in order:
        if not np.any(main_area == color):
            return False
    if np.count_nonzero(main_area != bg) < 20:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(3, 5))
    bottom_edges = _split_adjacent_edges(n, rng)
    right_edges = set(range(n - 1)) - bottom_edges

    layer_colors = rng.choice(np.arange(1, 10), size=n, replace=False).tolist()
    order = [int(color) for color in layer_colors]

    col_list = _axis_intervals(n, bottom_edges, rng)
    row_list = _axis_intervals(n, right_edges, rng)

    col_intervals = {color: col_list[idx] for idx, color in enumerate(order)}
    row_intervals = {color: row_list[idx] for idx, color in enumerate(order)}

    main_w = max(hi for _, hi in col_list) + 1 + int(rng.integers(0, 3))
    main_h = max(hi for _, hi in row_list) + 1 + int(rng.integers(0, 3))
    if max(main_h + 1, main_w + 1) > 30:
        return None

    inp = np.full((main_h + 1, main_w + 1), BG, dtype=int)
    inp[-1, :-1] = _render_band(main_w, col_intervals, order, BG)
    inp[:-1, -1] = _render_band(main_h, row_intervals, order, BG)
    inp[-1, -1] = BG

    expected = _build_output(main_h, main_w, row_intervals, col_intervals, order, BG)
    if not _validate_generated_case(inp, expected, order, row_intervals, col_intervals):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-rectangle input")
