import random
from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
LEGEND_AREA_WIDTH = 3
MAIN_START_COL = 5
GEN_RETRIES = 1200


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + dr * k, c0 + dc * k) for k in range(steps + 1)]


def _mono_components(grid: np.ndarray, *, exclude: Iterable[int]) -> list[dict]:
    h, w = grid.shape
    banned = set(map(int, exclude))
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in banned or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "r0": min(rows),
                    "c0": min(cols),
                    "r1": max(rows),
                    "c1": max(cols),
                }
            )
    return comps


def _legend_entries(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _mono_components(grid, exclude={bg, BLUE})
    entries = []
    for comp in comps:
        cols = {c for _, c in comp["cells"]}
        if comp["c1"] >= LEGEND_AREA_WIDTH:
            continue
        if len(cols) != 1:
            continue
        height = comp["r1"] - comp["r0"] + 1
        if len(comp["cells"]) != height:
            continue
        entries.append(
            {
                "color": int(comp["color"]),
                "height": int(height),
                "cells": list(comp["cells"]),
                "r0": int(comp["r0"]),
                "c0": int(comp["c0"]),
            }
        )
    entries.sort(key=lambda e: (e["r0"], e["c0"]))
    return entries


def _segments_from_blue_endpoints(grid: np.ndarray) -> list[dict]:
    points = [tuple(map(int, pt)) for pt in np.argwhere(grid == BLUE)]
    assert len(points) >= 2 and len(points) % 2 == 0, "blue endpoints must come in pairs"

    adjacency: dict[int, list[int]] = {i: [] for i in range(len(points))}
    for i in range(len(points)):
        r0, c0 = points[i]
        for j in range(i + 1, len(points)):
            r1, c1 = points[j]
            if r0 != r1 and abs(r1 - r0) == abs(c1 - c0):
                adjacency[i].append(j)
                adjacency[j].append(i)

    assert all(len(v) == 1 for v in adjacency.values()), "each blue endpoint must have exactly one diagonal partner"

    segments = []
    seen_edges = set()
    for i, js in adjacency.items():
        j = js[0]
        edge = tuple(sorted((i, j)))
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        p = points[edge[0]]
        q = points[edge[1]]
        path = diag_path(p, q)
        segments.append(
            {
                "endpoints": (p, q),
                "path": path,
                "length": len(path),
                "center": ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0),
            }
        )

    path_cells = [cell for seg in segments for cell in seg["path"]]
    assert len(set(path_cells)) == len(path_cells), "diagonal segments must not share pixels"
    return segments


def _render_output(shape: tuple[int, int], bg: int, legend: list[dict], ranked_segments: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for seg in ranked_segments:
        for r, c in seg["path"]:
            out[r, c] = BLUE
    for entry in legend:
        idx = int(entry["height"]) - 1
        assert 0 <= idx < len(ranked_segments)
        for r, c in ranked_segments[idx]["path"]:
            out[r, c] = int(entry["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect diagonal blue endpoint pairs, rank the resulting segments by length, and recolor them from a left-side legend.

    Input structure:
    - The background is black, and the leftmost few columns contain several 1-cell-wide monochrome legend blocks with distinct colors and heights.
    - The main area contains several blue endpoint pairs; each pair lies on a shared 45-degree diagonal, and different pairs do not create intersecting completed segments.

    Rule:
    - Identify every blue endpoint pair and draw the full blue diagonal segment between its endpoints, inclusive.
    - Rank those completed diagonal segments by pixel length from longest to shortest.
    - For each legend block of color K and height k, recolor the rank-k segment to color K.
    - Remove the legend by turning its cells back to background, while leaving any unrecolored segments blue.

    Key insight:
    - The legend height is not a target segment length; it is a rank index that becomes meaningful only after the diagonal segments are first completed and measured.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _legend_entries(g, bg)
    assert legend, "legend must be present"
    segments = _segments_from_blue_endpoints(g)
    ranked = sorted(
        segments,
        key=lambda seg: (-int(seg["length"]), seg["center"][0], seg["center"][1], seg["endpoints"]),
    )

    out = g.copy()
    for seg in segments:
        for r, c in seg["path"]:
            out[r, c] = BLUE
    for entry in legend:
        idx = int(entry["height"]) - 1
        assert 0 <= idx < len(ranked), "legend rank exceeds segment count"
        for r, c in ranked[idx]["path"]:
            out[r, c] = int(entry["color"])
    for entry in legend:
        for r, c in entry["cells"]:
            out[r, c] = bg
    return out


def _choose_heights(n: int) -> list[int]:
    while True:
        heights = list(range(1, n + 1))
        random.shuffle(heights)
        if heights != list(range(1, n + 1)) and heights != list(range(n, 0, -1)):
            return heights


def _choose_slopes(n: int) -> list[int]:
    if random.random() < 0.72:
        slopes = [1, -1]
        while len(slopes) < n:
            slopes.append(random.choice([1, -1]))
        random.shuffle(slopes)
        return slopes
    slope = random.choice([1, -1])
    return [slope] * n


def _segment_path(length: int, slope: int, height: int, width: int) -> list[tuple[int, int]] | None:
    delta = length - 1
    if slope == 1:
        if height - length < 0 or width - MAIN_START_COL - length < 0:
            return None
        r0 = random.randint(0, height - length)
        c0 = random.randint(MAIN_START_COL, width - length)
        return [(r0 + k, c0 + k) for k in range(length)]
    if height - length < 0 or width - 1 - (MAIN_START_COL + delta) < 0:
        return None
    r0 = random.randint(0, height - length)
    c0 = random.randint(MAIN_START_COL + delta, width - 1)
    return [(r0 + k, c0 - k) for k in range(length)]


def _path_clear(path: list[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    return all(cell not in blocked for cell in path)


def _add_blocked(path: list[tuple[int, int]], blocked: set[tuple[int, int]], height: int, width: int) -> None:
    for r, c in path:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((nr, nc))


def _legend_layout(height: int, n: int, legend_heights: list[int], legend_colors: list[int]) -> list[dict] | None:
    gaps = [random.randint(1, 2) for _ in range(n - 1)]
    needed = sum(legend_heights) + sum(gaps)
    if needed > height:
        return None
    slack = height - needed
    top_margin = random.randint(0, min(2, slack))
    row = top_margin

    legend_cols = [random.randint(0, LEGEND_AREA_WIDTH - 1) for _ in range(n)]
    if len(set(legend_cols)) == 1:
        legend_cols[random.randrange(n)] = (legend_cols[0] + 1) % LEGEND_AREA_WIDTH

    entries = []
    for i, (seg_height, color) in enumerate(zip(legend_heights, legend_colors)):
        cells = [(row + d, legend_cols[i]) for d in range(seg_height)]
        entries.append(
            {
                "color": int(color),
                "height": int(seg_height),
                "cells": cells,
                "r0": row,
                "c0": legend_cols[i],
            }
        )
        row += seg_height
        if i < n - 1:
            row += gaps[i]
    return entries


def _orders_too_simple(segments: list[dict]) -> bool:
    rank_order = [idx for idx, _ in sorted(enumerate(segments), key=lambda x: -x[1]["length"])]
    row_order = [idx for idx, _ in sorted(enumerate(segments), key=lambda x: x[1]["center"][0])]
    col_order = [idx for idx, _ in sorted(enumerate(segments), key=lambda x: x[1]["center"][1])]
    reversed_rank = list(reversed(rank_order))
    return row_order in (rank_order, reversed_rank) or col_order in (rank_order, reversed_rank)


def _alt_ranked(segments: list[dict], mode: str) -> list[dict]:
    if mode == "desc":
        return sorted(segments, key=lambda seg: (-seg["length"], seg["center"][0], seg["center"][1]))
    if mode == "asc":
        return sorted(segments, key=lambda seg: (seg["length"], seg["center"][0], seg["center"][1]))
    if mode == "row":
        return sorted(segments, key=lambda seg: (seg["center"][0], seg["center"][1]))
    if mode == "col":
        return sorted(segments, key=lambda seg: (seg["center"][1], seg["center"][0]))
    raise ValueError(mode)


def _semantic_checks(inp: np.ndarray, legend: list[dict], segments: list[dict]) -> bool:
    if len(legend) != len(segments):
        return False
    n = len(segments)
    if sorted(entry["height"] for entry in legend) != list(range(1, n + 1)):
        return False
    if len({entry["color"] for entry in legend}) != n:
        return False
    if any(len(seg["path"]) != seg["length"] for seg in segments):
        return False
    if len({seg["length"] for seg in segments}) != n:
        return False
    if _orders_too_simple(segments):
        return False

    try:
        parsed_legend = _legend_entries(inp, BG)
        parsed_segments = _segments_from_blue_endpoints(inp)
    except AssertionError:
        return False
    if len(parsed_legend) != n or len(parsed_segments) != n:
        return False

    expected = _render_output(inp.shape, BG, legend, _alt_ranked(segments, "desc"))
    if not np.array_equal(main(inp), expected):
        return False
    if BLUE in set(map(int, expected.flatten())):
        return False

    if np.array_equal(expected, _render_output(inp.shape, BG, legend, _alt_ranked(segments, "asc"))):
        return False
    if np.array_equal(expected, _render_output(inp.shape, BG, legend, _alt_ranked(segments, "row"))):
        return False
    if np.array_equal(expected, _render_output(inp.shape, BG, legend, _alt_ranked(segments, "col"))):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    height = random.randint(14, 18)
    width = random.randint(17, 23)

    legend_colors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], n)
    legend_heights = _choose_heights(n)
    legend = _legend_layout(height, n, legend_heights, legend_colors)
    if legend is None:
        return None

    max_length = min(height, width - MAIN_START_COL)
    if max_length < n + 3:
        return None
    lengths = random.sample(list(range(4, max_length + 1)), n)
    slopes = _choose_slopes(n)

    blocked: set[tuple[int, int]] = set()
    segments = []
    for length, slope in sorted(zip(lengths, slopes), reverse=True):
        placed = None
        for _ in range(300):
            path = _segment_path(length, slope, height, width)
            if path is None or not _path_clear(path, blocked):
                continue
            placed = path
            break
        if placed is None:
            return None
        p = placed[0]
        q = placed[-1]
        segments.append(
            {
                "endpoints": (p, q),
                "path": placed,
                "length": length,
                "center": ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0),
            }
        )
        _add_blocked(placed, blocked, height, width)

    inp = np.full((height, width), BG, dtype=int)
    for entry in legend:
        for r, c in entry["cells"]:
            inp[r, c] = int(entry["color"])
    for seg in segments:
        p, q = seg["endpoints"]
        inp[p] = BLUE
        inp[q] = BLUE

    try:
        if not _semantic_checks(inp, legend, segments):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
