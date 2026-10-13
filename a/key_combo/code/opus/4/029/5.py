import itertools

import numpy as np


PALETTE = np.array([1, 2, 3, 4, 7], dtype=int)
GEN_RETRIES = 400


def _row_objects(grid: np.ndarray) -> list[dict]:
    rows = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] != 0)
        if cols.size == 0:
            continue
        c0 = int(cols[0])
        c1 = int(cols[-1])
        assert np.all(grid[r, c0 : c1 + 1] != 0), "each visible line row must stay contiguous"
        rows.append(
            {
                "row": int(r),
                "c0": c0,
                "c1": c1,
                "length": int(c1 - c0 + 1),
                "segment": grid[r, c0 : c1 + 1].copy(),
                "candidates": tuple(dict.fromkeys([int(grid[r, c0]), int(grid[r, c1])])),
                "colors": {int(v) for v in grid[r, c0 : c1 + 1] if int(v) != 0},
            }
        )
    assert len(rows) % 2 == 1, "the number of hidden segments must be odd"
    assert 3 <= len(rows) <= 5, "expected 3-5 visible segment rows"
    return rows


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int] | None:
    indeg = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indeg[dst] += 1

    remaining = set(colors)
    order = []
    while remaining:
        frontier = sorted(color for color in remaining if indeg[color] == 0)
        if len(frontier) != 1:
            return None
        color = frontier[0]
        order.append(color)
        remaining.remove(color)
        for dst in edges[color]:
            indeg[dst] -= 1
    return order


def _infer_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, int], list[int]]:
    rows = _row_objects(grid)
    all_colors = sorted({int(v) for v in np.unique(grid) if int(v) != 0})
    assert len(all_colors) == len(rows), "each hidden line must have a unique color"

    valid = []
    for choices in itertools.product(*(row["candidates"] for row in rows)):
        if len(set(choices)) != len(rows):
            continue
        if set(choices) != set(all_colors):
            continue

        own_by_row = {idx: int(color) for idx, color in enumerate(choices)}
        color_to_row = {int(color): idx for idx, color in enumerate(choices)}
        edges = {int(color): set() for color in all_colors}
        ok = True

        for idx, row in enumerate(rows):
            own = own_by_row[idx]
            if int(np.count_nonzero(row["segment"] == own)) < 2:
                ok = False
                break
            for visible in row["colors"]:
                if visible == own:
                    continue
                edges[visible].add(own)

        if not ok:
            continue

        topo = _unique_topological_order(all_colors, edges)
        if topo is not None:
            valid.append((own_by_row, topo))

    assert len(valid) == 1, "row-to-color assignment must be uniquely recoverable"
    own_by_row, topo = valid[0]
    return rows, own_by_row, topo


def _expected_output(rows: list[dict], topo: list[int], own_by_row: dict[int, int]) -> np.ndarray:
    median_row = rows[len(rows) // 2]
    width = median_row["length"]
    out = np.zeros((len(rows), width), dtype=int)
    for r, color in enumerate(topo):
        out[r, :] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the occluding line segments by depth and redraw them to the median row's width.

    Input structure:
    - The background is black, and 5 one-pixel-high horizontal segment rows appear at distinct y positions.
    - Each row shows one hidden colored segment, but interior cells may be recolored by higher-depth segments whose column ranges overlap it.
    - Each hidden segment has a unique non-black color and keeps at least one visible endpoint plus at least two visible self-colored cells.

    Rule:
    - For each non-black row, take its contiguous non-black span as that segment's full horizontal range.
    - Infer the row's original color from the uniquely consistent endpoint-color assignment, then treat every other visible color on that row as an occluder above it.
    - Topologically sort those occlusion relations from topmost to bottommost.
    - In the original row order, pick the median segment row and use its span length as the output width.
    - Output one solid row per segment, ordered by occlusion depth, filling every row completely with that segment's original color.

    Key insight:
    - The same overlapped scene encodes two independent signals: row spans reveal the median alignment width, while foreign colors embedded inside a row reveal which other segments sit above it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rows, own_by_row, topo = _infer_scene(g)
    return _expected_output(rows, topo, own_by_row)


def _render_scene(segments: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    ordered = sorted(segments, key=lambda seg: seg["depth"])
    for seg in segments:
        r = int(seg["row"])
        c0 = int(seg["c0"])
        c1 = int(seg["c1"])
        for c in range(c0, c1 + 1):
            visible = int(seg["color"])
            for upper in ordered:
                if upper["depth"] > seg["depth"]:
                    break
                if int(upper["c0"]) <= c <= int(upper["c1"]):
                    visible = int(upper["color"])
                    break
            grid[r, c] = visible
    return grid


def _scene_output(segments: list[dict], rows_sorted: list[int]) -> np.ndarray:
    by_row = sorted(segments, key=lambda seg: seg["row"])
    width = int(by_row[len(by_row) // 2]["c1"] - by_row[len(by_row) // 2]["c0"] + 1)
    out = np.zeros((len(segments), width), dtype=int)
    for r, seg in enumerate(sorted(segments, key=lambda seg: seg["depth"])):
        out[r, :] = int(seg["color"])
    return out


def _build_intervals(rng: np.random.Generator, n: int) -> list[tuple[int, int]]:
    starts = [2 * (n - 1 - idx) for idx in range(n)]
    for _ in range(100):
        lengths = [int(rng.integers(4, 11)) for _ in range(n)]
        if len(set(lengths)) < 2:
            continue
        intervals = [(start, start + length - 1) for start, length in zip(starts, lengths)]
        if max(end for _, end in intervals) <= 17:
            return intervals
    raise AssertionError("failed to construct interval chain")


def _generate_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n = 5
    colors = rng.permutation(PALETTE).tolist()
    intervals = _build_intervals(rng, n)

    row_perm = rng.permutation(n).tolist()
    if row_perm == list(range(n)):
        return None

    top_gap = int(rng.integers(1, 3))
    gaps = [int(rng.integers(2, 4)) for _ in range(n - 1)]
    row_positions = [top_gap]
    for gap in gaps:
        row_positions.append(row_positions[-1] + gap)

    depth_segments = []
    for depth, color in enumerate(colors):
        c0, c1 = intervals[depth]
        depth_segments.append(
            {
                "depth": depth,
                "color": int(color),
                "row": int(row_positions[row_perm[depth]]),
                "c0": int(c0),
                "c1": int(c1),
            }
        )

    max_end = max(seg["c1"] for seg in depth_segments)
    h = row_positions[-1] + int(rng.integers(2, 4))
    w = max_end + int(rng.integers(3, 7))
    shift = int(rng.integers(1, max(2, w - max_end - 1)))
    for seg in depth_segments:
        seg["c0"] += shift
        seg["c1"] += shift
    w = max(seg["c1"] for seg in depth_segments) + int(rng.integers(2, 5))

    grid = _render_scene(depth_segments, (h, w))
    expected = _scene_output(depth_segments, row_positions)

    physical_order = [seg["color"] for seg in sorted(depth_segments, key=lambda seg: seg["row"])]
    depth_order = [seg["color"] for seg in sorted(depth_segments, key=lambda seg: seg["depth"])]
    lengths = [int(seg["c1"] - seg["c0"] + 1) for seg in sorted(depth_segments, key=lambda seg: seg["row"])]
    median_width = lengths[len(lengths) // 2]

    if physical_order == depth_order:
        return None
    if all(length == median_width for length in lengths):
        return None
    if median_width == int(sorted(depth_segments, key=lambda seg: seg["depth"])[0]["c1"] - sorted(depth_segments, key=lambda seg: seg["depth"])[0]["c0"] + 1):
        return None

    rows, own_by_row, topo = _infer_scene(grid)
    recovered = _expected_output(rows, topo, own_by_row)
    if not np.array_equal(recovered, expected):
        return None

    naive_row = np.zeros_like(expected)
    for r, color in enumerate(physical_order):
        naive_row[r, :] = int(color)
    if np.array_equal(naive_row, expected):
        return None

    topo_width_shortcut = np.zeros((n, lengths[0]), dtype=int)
    for r, color in enumerate(depth_order):
        topo_width_shortcut[r, :] = int(color)
    if np.array_equal(topo_width_shortcut, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        result = _generate_attempt(rng)
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid occlusion-order input")
