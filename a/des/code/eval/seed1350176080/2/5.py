from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be unique"
    return int(ranked[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _pipe_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != int(bg))
    assert counts, "expected non-background structure"

    scored: list[tuple[int, int, int]] = []
    for color, total in counts.items():
        comps = _components(grid == int(color))
        largest = max(len(comp) for comp in comps)
        scored.append((largest, int(total), int(color)))

    scored.sort(reverse=True)
    assert len(scored) == 1 or scored[0][:2] > scored[1][:2], "pipe color must dominate other foreground colors"
    return int(scored[0][2])


def _build_pipe_graph(mask: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "expected pipe pixels"
    nodes = set(coords)
    graph = {node: set() for node in nodes}
    for r, c in nodes:
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr in nodes:
                graph[(r, c)].add(nbr)

    seen = set()
    stack = [coords[0]]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph[node] - seen)
    assert len(seen) == len(nodes), "pipe network must be connected"

    edge_count = sum(len(v) for v in graph.values()) // 2
    assert edge_count == len(nodes) - 1, "pipe network must be a tree"
    assert all(1 <= len(v) <= 4 for v in graph.values()), "pipe degrees must stay within orthogonal limits"
    return graph


def _parse_legend_and_sources(
    grid: np.ndarray,
    bg: int,
    pipe: int,
) -> tuple[dict[frozenset[int], int], list[tuple[int, int, int]]]:
    mask = (grid != int(bg)) & (grid != int(pipe))
    comps = _components(mask)
    mix_map: dict[frozenset[int], int] = {}
    sources: list[tuple[int, int, int]] = []

    for comp in comps:
        if len(comp) == 1:
            r, c = comp[0]
            sources.append((int(r), int(c), int(grid[r, c])))
            continue

        if len(comp) == 3:
            rows = {r for r, _c in comp}
            cols = sorted(c for _r, c in comp)
            if len(rows) == 1 and cols == list(range(cols[0], cols[0] + 3)):
                row = next(iter(rows))
                left, mid, right = cols
                c1 = int(grid[row, left])
                out = int(grid[row, mid])
                c2 = int(grid[row, right])
                key = frozenset((c1, c2))
                assert len(key) == 2, "legend inputs must use two distinct colors"
                if key in mix_map:
                    assert mix_map[key] == out, "legend must be consistent"
                mix_map[key] = out
                continue

        raise AssertionError("expected only isolated sources and 1x3 legend rows outside the pipe")

    assert mix_map, "expected at least one legend row"
    assert sources, "expected at least one source"
    source_colors = [color for _r, _c, color in sources]
    assert len(set(source_colors)) == len(source_colors), "source colors must be unique"
    return mix_map, sources


def _merge_colors(colors: list[int], mix_map: dict[frozenset[int], int]) -> int | None:
    uniq: list[int] = []
    for color in colors:
        if color not in uniq:
            uniq.append(int(color))

    if not uniq:
        return None
    if len(uniq) == 1:
        return uniq[0]
    assert len(uniq) == 2, "generator guarantees binary mixes at each reduction step"
    key = frozenset(uniq)
    assert key in mix_map, "missing legend entry for encountered color pair"
    return int(mix_map[key])


def _final_signal_color(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    leaf_sources: dict[tuple[int, int], int],
    mix_map: dict[frozenset[int], int],
) -> int:
    remain = {node: set(neis) for node, neis in graph.items()}
    pending = {node: [] for node in graph}
    removed: set[tuple[int, int]] = set()
    q = deque(node for node, neis in remain.items() if len(neis) <= 1)
    last_signal: int | None = None

    while q:
        node = q.popleft()
        if node in removed or len(remain[node]) > 1:
            continue

        colors = list(pending[node])
        intrinsic = leaf_sources.get(node)
        if intrinsic is not None:
            colors.append(int(intrinsic))
        signal = _merge_colors(colors, mix_map)
        if signal is not None:
            last_signal = signal

        nbrs = list(remain[node])
        removed.add(node)
        if nbrs:
            nbr = nbrs[0]
            remain[nbr].remove(node)
            if signal is not None:
                pending[nbr].append(signal)
            if len(remain[nbr]) <= 1:
                q.append(nbr)

    assert len(removed) == len(graph), "reduction should consume the whole tree"
    assert last_signal is not None, "expected at least one propagated signal"
    return int(last_signal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colored signals through a tree-shaped pipe network, mix colors at
    junctions according to the legend, and paint each port leaf with the final
    arriving color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    pipe = _pipe_color(g, bg)
    mix_map, sources = _parse_legend_and_sources(g, bg, pipe)
    graph = _build_pipe_graph(g == int(pipe))

    leaves = [node for node, neis in graph.items() if len(neis) == 1]
    assert leaves, "expected pipe leaves"

    leaf_sources: dict[tuple[int, int], int] = {}
    for sr, sc, color in sources:
        touching = []
        for dr, dc in DIR4:
            nbr = (sr + dr, sc + dc)
            if nbr in graph:
                touching.append(nbr)
        assert len(touching) == 1, "each source must touch exactly one pipe pixel"
        leaf = touching[0]
        assert len(graph[leaf]) == 1, "sources must attach to pipe leaves"
        assert leaf not in leaf_sources, "each leaf may carry at most one source"
        leaf_sources[leaf] = int(color)

    ports = [leaf for leaf in leaves if leaf not in leaf_sources]
    assert leaf_sources, "expected at least one source leaf"
    assert ports, "expected at least one port leaf"

    final_color = _final_signal_color(graph, leaf_sources, mix_map)
    out = g.copy()
    for r, c in ports:
        out[r, c] = final_color
    return out


def _draw_segment(cells: set[tuple[int, int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            cells.add((int(r0), int(c)))
        return
    assert c0 == c1, "segments must be orthogonal"
    for r in range(min(r0, r1), max(r0, r1) + 1):
        cells.add((int(r), int(c0)))


def _add_port_fan(
    pipe_cells: set[tuple[int, int]],
    split_node: tuple[int, int],
    port_count: int,
) -> list[tuple[int, int]]:
    assert 1 <= port_count <= 3
    r, c = split_node
    ports: list[tuple[int, int]] = []

    right_len = random.randint(2, 4)
    right_leaf = (r, c + right_len)
    _draw_segment(pipe_cells, split_node, right_leaf)
    ports.append(right_leaf)

    if port_count == 1:
        return ports

    if port_count == 2:
        side = random.choice(["up", "down"])
        delta = -random.randint(2, 4) if side == "up" else random.randint(2, 4)
        leaf = (r + delta, c)
        _draw_segment(pipe_cells, split_node, leaf)
        ports.append(leaf)
        return ports

    up_leaf = (r - random.randint(2, 4), c)
    down_leaf = (r + random.randint(2, 4), c)
    _draw_segment(pipe_cells, split_node, up_leaf)
    _draw_segment(pipe_cells, split_node, down_leaf)
    ports.extend([up_leaf, down_leaf])
    return ports


def _add_source(
    source_cells: dict[tuple[int, int], int],
    leaf: tuple[int, int],
    outward: tuple[int, int],
    color: int,
) -> None:
    sr = leaf[0] + outward[0]
    sc = leaf[1] + outward[1]
    source_cells[(sr, sc)] = int(color)


def _two_t_layout(
    source_colors: list[int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    pipe_cells: set[tuple[int, int]] = set()
    source_cells: dict[tuple[int, int], int] = {}

    final_mix = (0, 0)
    left_leaf = (0, -random.randint(2, 5))
    side = random.choice(["up", "down"])
    vertical_delta = -random.randint(2, 4) if side == "up" else random.randint(2, 4)
    vertical_leaf = (vertical_delta, 0)
    split = (0, random.randint(2, 4))
    port_count = random.randint(1, 3)

    _draw_segment(pipe_cells, left_leaf, final_mix)
    _draw_segment(pipe_cells, vertical_leaf, final_mix)
    _draw_segment(pipe_cells, final_mix, split)
    ports = _add_port_fan(pipe_cells, split, port_count)

    _add_source(source_cells, left_leaf, (0, -1), source_colors[0])
    _add_source(source_cells, vertical_leaf, (-1, 0) if side == "up" else (1, 0), source_colors[1])

    return pipe_cells, source_cells, ports, [(source_colors[0], source_colors[1], -1)]


def _three_chain_t_layout(
    source_colors: list[int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    pipe_cells: set[tuple[int, int]] = set()
    source_cells: dict[tuple[int, int], int] = {}

    first_mix = (0, 0)
    final_mix = (0, random.randint(3, 5))
    split = (0, final_mix[1] + random.randint(2, 4))
    port_count = random.randint(1, 3)

    left_leaf = (0, -random.randint(2, 5))
    side_a = random.choice(["up", "down"])
    delta_a = -random.randint(2, 4) if side_a == "up" else random.randint(2, 4)
    leaf_a = (delta_a, 0)

    side_f = random.choice(["up", "down"])
    delta_f = -random.randint(2, 4) if side_f == "up" else random.randint(2, 4)
    leaf_f = (delta_f, final_mix[1])

    _draw_segment(pipe_cells, left_leaf, first_mix)
    _draw_segment(pipe_cells, leaf_a, first_mix)
    _draw_segment(pipe_cells, first_mix, final_mix)
    _draw_segment(pipe_cells, leaf_f, final_mix)
    _draw_segment(pipe_cells, final_mix, split)
    ports = _add_port_fan(pipe_cells, split, port_count)

    _add_source(source_cells, left_leaf, (0, -1), source_colors[0])
    _add_source(source_cells, leaf_a, (-1, 0) if side_a == "up" else (1, 0), source_colors[1])
    _add_source(source_cells, leaf_f, (-1, 0) if side_f == "up" else (1, 0), source_colors[2])

    return pipe_cells, source_cells, ports, [(source_colors[0], source_colors[1], -1), (-1, source_colors[2], -1)]


def _three_cross_layout(
    source_colors: list[int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    pipe_cells: set[tuple[int, int]] = set()
    source_cells: dict[tuple[int, int], int] = {}

    first_mix = (0, 0)
    final_mix = (0, random.randint(3, 5))
    split = (0, final_mix[1] + random.randint(2, 4))
    extra_ports = random.randint(1, 2)

    left_leaf = (0, -random.randint(2, 5))
    side_a = random.choice(["up", "down"])
    delta_a = -random.randint(2, 4) if side_a == "up" else random.randint(2, 4)
    leaf_a = (delta_a, 0)

    source_side = random.choice(["up", "down"])
    source_delta = -random.randint(2, 4) if source_side == "up" else random.randint(2, 4)
    source_leaf = (source_delta, final_mix[1])
    port_delta = random.randint(2, 4) if source_side == "up" else -random.randint(2, 4)
    direct_port = (port_delta, final_mix[1])

    _draw_segment(pipe_cells, left_leaf, first_mix)
    _draw_segment(pipe_cells, leaf_a, first_mix)
    _draw_segment(pipe_cells, first_mix, final_mix)
    _draw_segment(pipe_cells, source_leaf, final_mix)
    _draw_segment(pipe_cells, direct_port, final_mix)
    _draw_segment(pipe_cells, final_mix, split)
    ports = [direct_port]
    ports.extend(_add_port_fan(pipe_cells, split, extra_ports))

    _add_source(source_cells, left_leaf, (0, -1), source_colors[0])
    _add_source(source_cells, leaf_a, (-1, 0) if side_a == "up" else (1, 0), source_colors[1])
    _add_source(source_cells, source_leaf, (-1, 0) if source_side == "up" else (1, 0), source_colors[2])

    return pipe_cells, source_cells, ports, [(source_colors[0], source_colors[1], -1), (-1, source_colors[2], -1)]


def _four_cross_layout(
    source_colors: list[int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    pipe_cells: set[tuple[int, int]] = set()
    source_cells: dict[tuple[int, int], int] = {}

    final_mix = (0, 0)
    top_mix = (-random.randint(3, 5), 0)
    left_mix = (0, -random.randint(3, 5))
    split = (0, random.randint(2, 4))
    extra_ports = random.randint(1, 2)

    source1_leaf = (top_mix[0], top_mix[1] - random.randint(2, 4))
    source2_leaf = (top_mix[0] - random.randint(2, 4), top_mix[1])
    source3_leaf = (left_mix[0], left_mix[1] - random.randint(2, 4))
    source4_leaf = (left_mix[0] + random.randint(2, 4), left_mix[1])
    direct_port = (random.randint(2, 4), 0)

    _draw_segment(pipe_cells, source1_leaf, top_mix)
    _draw_segment(pipe_cells, source2_leaf, top_mix)
    _draw_segment(pipe_cells, top_mix, final_mix)
    _draw_segment(pipe_cells, source3_leaf, left_mix)
    _draw_segment(pipe_cells, source4_leaf, left_mix)
    _draw_segment(pipe_cells, left_mix, final_mix)
    _draw_segment(pipe_cells, final_mix, direct_port)
    _draw_segment(pipe_cells, final_mix, split)
    ports = [direct_port]
    ports.extend(_add_port_fan(pipe_cells, split, extra_ports))

    _add_source(source_cells, source1_leaf, (0, -1), source_colors[0])
    _add_source(source_cells, source2_leaf, (-1, 0), source_colors[1])
    _add_source(source_cells, source3_leaf, (0, -1), source_colors[2])
    _add_source(source_cells, source4_leaf, (1, 0), source_colors[3])

    return pipe_cells, source_cells, ports, [
        (source_colors[0], source_colors[1], -1),
        (source_colors[2], source_colors[3], -1),
        (-1, -1, -1),
    ]


def _legend_rows_for_topology(
    topology: str,
    source_colors: list[int],
    mix_colors: list[int],
) -> tuple[list[tuple[int, int, int]], int]:
    if topology == "two_t":
        rows = [(source_colors[0], mix_colors[0], source_colors[1])]
        return rows, mix_colors[0]
    if topology in {"three_chain_t", "three_cross"}:
        rows = [
            (source_colors[0], mix_colors[0], source_colors[1]),
            (mix_colors[0], mix_colors[1], source_colors[2]),
        ]
        return rows, mix_colors[1]

    assert topology == "four_cross"
    rows = [
        (source_colors[0], mix_colors[0], source_colors[1]),
        (source_colors[2], mix_colors[1], source_colors[3]),
        (mix_colors[0], mix_colors[2], mix_colors[1]),
    ]
    return rows, mix_colors[2]


def _build_topology(
    topology: str,
    source_colors: list[int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], list[tuple[int, int]]]:
    if topology == "two_t":
        pipe_cells, source_cells, ports, _ = _two_t_layout(source_colors)
        return pipe_cells, source_cells, ports
    if topology == "three_chain_t":
        pipe_cells, source_cells, ports, _ = _three_chain_t_layout(source_colors)
        return pipe_cells, source_cells, ports
    if topology == "three_cross":
        pipe_cells, source_cells, ports, _ = _three_cross_layout(source_colors)
        return pipe_cells, source_cells, ports
    assert topology == "four_cross"
    pipe_cells, source_cells, ports, _ = _four_cross_layout(source_colors)
    return pipe_cells, source_cells, ports


def _flip_cells(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    flip_ud: bool,
    flip_lr: bool,
) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        rr = h - 1 - r if flip_ud else r
        cc = w - 1 - c if flip_lr else c
        out.add((int(rr), int(cc)))
    return out


def _flip_dict(
    cells: dict[tuple[int, int], int],
    h: int,
    w: int,
    flip_ud: bool,
    flip_lr: bool,
) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for (r, c), color in cells.items():
        rr = h - 1 - r if flip_ud else r
        cc = w - 1 - c if flip_lr else c
        out[(int(rr), int(cc))] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    topology = random.choice(["two_t", "three_chain_t", "three_cross", "four_cross"])
    source_count = {"two_t": 2, "three_chain_t": 3, "three_cross": 3, "four_cross": 4}[topology]
    mix_count = source_count - 1

    colors = list(range(10))
    random.shuffle(colors)
    bg = colors.pop()
    pipe = colors.pop()
    source_colors = [colors.pop() for _ in range(source_count)]
    mix_colors = [colors.pop() for _ in range(mix_count)]

    legend_rows, final_color = _legend_rows_for_topology(topology, source_colors, mix_colors)
    random.shuffle(legend_rows)
    legend_h = 2 * len(legend_rows) + 1
    legend_w = 5
    pipe_cells, source_cells, ports = _build_topology(topology, source_colors)

    local_rows = [r for r, _c in pipe_cells] + [r for r, _c in source_cells] + [r for r, _c in ports]
    local_cols = [c for _r, c in pipe_cells] + [c for _r, c in source_cells] + [c for _r, c in ports]
    x0 = legend_w + random.randint(2, 4) - min(local_cols)
    y0 = legend_h + random.randint(2, 5) - min(local_rows)

    pipe_cells = {(r + y0, c + x0) for r, c in pipe_cells}
    source_cells = {(r + y0, c + x0): color for (r, c), color in source_cells.items()}
    ports = [(r + y0, c + x0) for r, c in ports]

    all_rows = [r for r, _c in pipe_cells] + [r for r, _c in source_cells] + [r for r, _c in ports]
    all_cols = [c for _r, c in pipe_cells] + [c for _r, c in source_cells] + [c for _r, c in ports]

    max_row = max(max(all_rows), legend_h - 1)
    max_col = max(max(all_cols), legend_w - 1)
    h = max_row + random.randint(3, 6) + 1
    w = max_col + random.randint(3, 7) + 1
    grid = np.full((h, w), int(bg), dtype=int)

    for i, (c1, out, c2) in enumerate(legend_rows):
        r = 1 + 2 * i
        grid[r, 1] = int(c1)
        grid[r, 2] = int(out)
        grid[r, 3] = int(c2)

    for r, c in pipe_cells:
        if not (0 <= r < h and 0 <= c < w):
            return None
        grid[r, c] = int(pipe)
    for (r, c), color in source_cells.items():
        if not (0 <= r < h and 0 <= c < w):
            return None
        if grid[r, c] != int(bg):
            return None
        grid[r, c] = int(color)

    expected = grid.copy()
    for r, c in ports:
        expected[r, c] = int(final_color)

    flip_ud = random.random() < 0.5
    flip_lr = random.random() < 0.5
    if flip_ud:
        grid = np.flipud(grid)
        expected = np.flipud(expected)
    if flip_lr:
        grid = np.fliplr(grid)
        expected = np.fliplr(expected)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-mixing task")
