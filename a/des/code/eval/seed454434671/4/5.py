from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BG = 0
PIPE = 5

BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

N = (-1, 0)
S = (1, 0)
W = (0, -1)
E = (0, 1)
DIRS4 = (N, S, W, E)
LEFT_TURN = {N: W, W: S, S: E, E: N}
RIGHT_TURN = {N: E, E: S, S: W, W: N}

PRIMARY_COLORS = (RED, BLUE, YELLOW)
PAIR_RESULTS = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _straight_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("straight path requires shared row or column")


def _rotate_cell(cell: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = cell
    for _ in range(turns % 4):
        r, c = -c, r
    return r, c


def _reflect_cell(cell: tuple[int, int]) -> tuple[int, int]:
    return cell[0], -cell[1]


def _apply_transform(
    cells: list[tuple[int, int]],
    turns: int,
    reflect: bool,
) -> list[tuple[int, int]]:
    out = [_rotate_cell(cell, turns) for cell in cells]
    if reflect:
        out = [_reflect_cell(cell) for cell in out]
    return out


def _legend_spec(block: np.ndarray) -> tuple[dict[frozenset[int], int], set[int]] | None:
    if block.shape != (3, 3):
        return None
    if not np.all((block != BG) & (block != PIPE)):
        return None

    pair_rows = [frozenset((int(block[r, 0]), int(block[r, 1]))) for r in range(3)]
    if any(len(pair) != 2 for pair in pair_rows):
        return None

    sources = set()
    for pair in pair_rows:
        sources |= set(pair)
    if len(sources) != 3:
        return None

    want_pairs = {frozenset((a, b)) for i, a in enumerate(sorted(sources)) for b in sorted(sources)[i + 1 :]}
    if set(pair_rows) != want_pairs:
        return None

    mix_map = {pair_rows[r]: int(block[r, 2]) for r in range(3)}
    if len(set(mix_map.values())) != 3:
        return None
    return mix_map, sources


def _parse_legend(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], dict[frozenset[int], int], dict[int, frozenset[int]], dict[frozenset[int], int], set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for r in range(h - 2):
        for c in range(w - 2):
            parsed = _legend_spec(g[r : r + 3, c : c + 3])
            if parsed is None:
                continue
            mix_map, sources = parsed
            candidates.append((r, c, mix_map, sources))

    assert len(candidates) == 1, f"expected exactly one legend, found {len(candidates)}"
    r0, c0, mix_map, source_colors = candidates[0]

    color_parts: dict[int, frozenset[int]] = {}
    parts_to_color: dict[frozenset[int], int] = {}
    for idx, color in enumerate(sorted(source_colors)):
        part = frozenset((idx,))
        color_parts[int(color)] = part
        parts_to_color[part] = int(color)
    for pair, result in mix_map.items():
        merged = frozenset().union(*(color_parts[color] for color in pair))
        color_parts[int(result)] = merged
        parts_to_color[merged] = int(result)

    return (r0, c0, r0 + 2, c0 + 2), mix_map, color_parts, parts_to_color, source_colors


def _merge_colors(
    colors: list[int],
    color_parts: dict[int, frozenset[int]],
    parts_to_color: dict[frozenset[int], int],
) -> int | None:
    distinct = {int(color) for color in colors if color is not None}
    if not distinct:
        return None
    parts = frozenset().union(*(color_parts[color] for color in distinct))
    out = parts_to_color.get(parts)
    assert out is not None, f"unsupported color union {sorted(distinct)}"
    return out


def _pipe_neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((nr, nc))
    return out


def _build_graph(
    pipe_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], dict[tuple[int, int], int], list[dict[str, object]], list[list[tuple[int, int]]]]:
    pipe_cells = [tuple(map(int, cell)) for cell in np.argwhere(pipe_mask)]
    assert pipe_cells, "expected a pipe network"

    degrees = {cell: len(_pipe_neighbors(pipe_mask, cell)) for cell in pipe_cells}
    node_cells = sorted(cell for cell, deg in degrees.items() if deg != 2)
    assert node_cells, "pipe graph must contain at least one endpoint or junction"
    node_index = {cell: idx for idx, cell in enumerate(node_cells)}

    segments: list[dict[str, object]] = []
    adjacency: list[list[tuple[int, int]]] = [[] for _ in node_cells]
    seen_edges: set[frozenset[tuple[int, int]]] = set()

    for start in node_cells:
        start_id = node_index[start]
        for nxt in _pipe_neighbors(pipe_mask, start):
            edge = frozenset((start, nxt))
            if edge in seen_edges:
                continue
            path = [start]
            prev = start
            cur = nxt
            seen_edges.add(edge)
            while cur not in node_index:
                path.append(cur)
                next_cells = [cell for cell in _pipe_neighbors(pipe_mask, cur) if cell != prev]
                assert len(next_cells) == 1, "intermediate pipe cells must continue uniquely"
                nxt2 = next_cells[0]
                seen_edges.add(frozenset((cur, nxt2)))
                prev, cur = cur, nxt2
            path.append(cur)
            end_id = node_index[cur]
            seg_id = len(segments)
            segments.append({"u": start_id, "v": end_id, "path": path})
            adjacency[start_id].append((end_id, seg_id))
            adjacency[end_id].append((start_id, seg_id))

    seen_nodes: set[int] = set()
    component_count = 0
    for node_id in range(len(node_cells)):
        if node_id in seen_nodes:
            continue
        component_count += 1
        stack = [node_id]
        seen_nodes.add(node_id)
        while stack:
            cur = stack.pop()
            for nxt, _seg_id in adjacency[cur]:
                if nxt not in seen_nodes:
                    seen_nodes.add(nxt)
                    stack.append(nxt)
    assert len(segments) == len(node_cells) - component_count, "generator expects a forest-shaped pipe graph"

    return node_cells, node_index, degrees, segments, adjacency


def _find_sources(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    source_colors: set[int],
    node_index: dict[tuple[int, int], int],
    degrees: dict[tuple[int, int], int],
) -> dict[int, int]:
    r0, c0, r1, c1 = legend_bbox
    base_colors: dict[int, int] = {}
    for r, c in np.argwhere((grid != BG) & (grid != PIPE)):
        rr, cc = int(r), int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        color = int(grid[rr, cc])
        assert color in source_colors, "non-legend colored pixels must be signal sources"
        touching = []
        for dr, dc in DIRS4:
            pipe_cell = (rr + dr, cc + dc)
            if pipe_cell in node_index and degrees[pipe_cell] <= 1:
                touching.append(pipe_cell)
        assert len(touching) == 1, "each source must touch exactly one pipe dead-end"
        node_id = node_index[touching[0]]
        assert node_id not in base_colors, "each pipe dead-end may have at most one source"
        base_colors[node_id] = color
    assert base_colors, "expected at least one signal source"
    return base_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colors along grey pipes, mixing them at junctions according to the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend_bbox, _mix_map, color_parts, parts_to_color, source_colors = _parse_legend(g)
    pipe_mask = g == PIPE
    node_cells, node_index, degrees, segments, adjacency = _build_graph(pipe_mask)
    base_colors = _find_sources(g, legend_bbox, source_colors, node_index, degrees)

    @lru_cache(maxsize=None)
    def message(src: int, dst: int) -> int | None:
        inputs: list[int] = []
        if src in base_colors:
            inputs.append(base_colors[src])
        for nxt, _seg_id in adjacency[src]:
            if nxt == dst:
                continue
            incoming = message(nxt, src)
            if incoming is not None:
                inputs.append(incoming)
        return _merge_colors(inputs, color_parts, parts_to_color)

    def node_color(node_id: int) -> int | None:
        inputs: list[int] = []
        if node_id in base_colors:
            inputs.append(base_colors[node_id])
        for nxt, _seg_id in adjacency[node_id]:
            incoming = message(nxt, node_id)
            if incoming is not None:
                inputs.append(incoming)
        return _merge_colors(inputs, color_parts, parts_to_color)

    out = g.copy()
    for node_id, cell in enumerate(node_cells):
        color = node_color(node_id)
        if color is not None:
            out[cell] = color

    for segment in segments:
        u = int(segment["u"])
        v = int(segment["v"])
        path = list(segment["path"])
        forward = message(u, v)
        backward = message(v, u)
        if forward is None and backward is None:
            continue
        if forward is not None and backward is not None:
            assert forward == backward, "conflicting signals cannot meet on a non-junction segment"
            color = forward
        else:
            color = forward if forward is not None else backward
        for cell in path[1:-1]:
            out[cell] = int(color)

    return out


def _leaf_branch(
    start: tuple[int, int],
    direction: tuple[int, int],
    rng: random.Random,
    allow_bend: bool = True,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    pos = start
    path = [start]
    leg1 = rng.randint(2, 4)
    for _ in range(leg1):
        pos = _add(pos, direction)
        path.append(pos)
    final_dir = direction
    if allow_bend and rng.random() < 0.45:
        final_dir = rng.choice((LEFT_TURN[direction], RIGHT_TURN[direction]))
        leg2 = rng.randint(1, 2)
        for _ in range(leg2):
            pos = _add(pos, final_dir)
            path.append(pos)
    return path, _add(pos, final_dir)


def _pick_mix_pair(rng: random.Random) -> tuple[tuple[int, int], int]:
    pair = rng.choice(list(PAIR_RESULTS.keys()))
    ordered = tuple(sorted(pair))
    return ordered, PAIR_RESULTS[pair]


def _build_single_junction(rng: random.Random, same_color: bool) -> tuple[list[list[tuple[int, int]]], list[tuple[tuple[int, int], int]]]:
    directions = rng.sample(list(DIRS4), rng.choice((3, 4)))
    max_sources = len(directions) - 1
    source_count = rng.randint(2, max_sources)
    source_dirs = directions[:source_count]
    passive_dirs = directions[source_count:]

    if same_color:
        colors = [rng.choice(PRIMARY_COLORS)] * source_count
    else:
        pair, _result = _pick_mix_pair(rng)
        colors = [pair[0], pair[1]]
        while len(colors) < source_count:
            colors.append(rng.choice(pair))
        rng.shuffle(colors)

    center = (0, 0)
    paths: list[list[tuple[int, int]]] = []
    sources: list[tuple[tuple[int, int], int]] = []
    for direction, color in zip(source_dirs, colors, strict=False):
        path, source = _leaf_branch(center, direction, rng)
        paths.append(path)
        sources.append((source, color))
    for direction in passive_dirs:
        path, _source = _leaf_branch(center, direction, rng)
        paths.append(path)
    return paths, sources


def _build_chain(rng: random.Random) -> tuple[list[list[tuple[int, int]]], list[tuple[tuple[int, int], int]]]:
    gap = rng.randint(4, 6)
    j1 = (0, 0)
    j2 = (0, gap)

    pair, _result = _pick_mix_pair(rng)
    source_count = rng.choice((2, 3))
    colors = [pair[0], pair[1]]
    while len(colors) < source_count:
        colors.append(rng.choice(pair))
    rng.shuffle(colors)

    left_dirs = rng.sample([N, S, W], source_count)
    right_dirs = rng.sample([N, S, E], rng.choice((2, 3)))

    paths = [_straight_path(j1, j2)]
    sources: list[tuple[tuple[int, int], int]] = []
    for direction, color in zip(left_dirs, colors, strict=False):
        path, source = _leaf_branch(j1, direction, rng)
        paths.append(path)
        sources.append((source, color))
    for direction in right_dirs:
        path, _source = _leaf_branch(j2, direction, rng)
        paths.append(path)
    return paths, sources


def _build_double_mix(rng: random.Random) -> tuple[list[list[tuple[int, int]]], list[tuple[tuple[int, int], int]]]:
    left_gap = rng.randint(4, 6)
    right_gap = rng.randint(4, 6)
    jc = (0, 0)
    jl = (0, -left_gap)
    jr = (0, right_gap)

    pair, _result = _pick_mix_pair(rng)
    left_count = rng.choice((2, 3))
    right_count = rng.choice((2, 3))
    left_colors = [pair[0], pair[1]]
    right_colors = [pair[0], pair[1]]
    while len(left_colors) < left_count:
        left_colors.append(rng.choice(pair))
    while len(right_colors) < right_count:
        right_colors.append(rng.choice(pair))
    rng.shuffle(left_colors)
    rng.shuffle(right_colors)

    paths = [_straight_path(jl, jc), _straight_path(jc, jr)]
    sources: list[tuple[tuple[int, int], int]] = []
    for direction, color in zip(rng.sample([N, S, W], left_count), left_colors, strict=False):
        path, source = _leaf_branch(jl, direction, rng)
        paths.append(path)
        sources.append((source, color))
    for direction, color in zip(rng.sample([N, S, E], right_count), right_colors, strict=False):
        path, source = _leaf_branch(jr, direction, rng)
        paths.append(path)
        sources.append((source, color))
    for direction in rng.sample([N, S], rng.choice((1, 2))):
        path, _source = _leaf_branch(jc, direction, rng)
        paths.append(path)
    return paths, sources


def _build_family(family: int, rng: random.Random) -> tuple[list[list[tuple[int, int]]], list[tuple[tuple[int, int], int]]]:
    if family == 0:
        return _build_single_junction(rng, same_color=False)
    if family == 1:
        return _build_single_junction(rng, same_color=True)
    if family == 2:
        return _build_chain(rng)
    if family == 3:
        return _build_double_mix(rng)
    raise AssertionError("unknown family")


def _build_idle_component(
    existing_paths: list[list[tuple[int, int]]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    used = [cell for path in existing_paths for cell in path]
    min_c = min(cell[1] for cell in used)
    max_c = max(cell[1] for cell in used)
    max_r = max(cell[0] for cell in used)
    center = (max_r + 7, (min_c + max_c) // 2)

    paths: list[list[tuple[int, int]]] = []
    for direction in rng.sample(list(DIRS4), rng.choice((3, 4))):
        path, _source = _leaf_branch(center, direction, rng)
        paths.append(path)
    return paths


def _normalize_paths(
    paths: list[list[tuple[int, int]]],
    sources: list[tuple[tuple[int, int], int]],
    turns: int,
    reflect: bool,
) -> tuple[list[list[tuple[int, int]]], list[tuple[tuple[int, int], int]]]:
    transformed_paths = [_apply_transform(path, turns, reflect) for path in paths]
    transformed_sources = [
        (_apply_transform([cell], turns, reflect)[0], color) for cell, color in sources
    ]
    all_cells = [cell for path in transformed_paths for cell in path] + [cell for cell, _ in transformed_sources]
    min_r = min(cell[0] for cell in all_cells)
    min_c = min(cell[1] for cell in all_cells)
    dr, dc = -min_r, -min_c
    shifted_paths = [[(r + dr, c + dc) for r, c in path] for path in transformed_paths]
    shifted_sources = [((r + dr, c + dc), color) for (r, c), color in transformed_sources]
    return shifted_paths, shifted_sources


def _boxes_far(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_candidate(
    paths: list[list[tuple[int, int]]],
    sources: list[tuple[tuple[int, int], int]],
    rng: random.Random,
) -> np.ndarray | None:
    pipe_cells = {cell for path in paths for cell in path}
    all_cells = list(pipe_cells) + [cell for cell, _ in sources]

    max_r = max(cell[0] for cell in all_cells)
    max_c = max(cell[1] for cell in all_cells)
    net_h = max_r + 1
    net_w = max_c + 1

    for _ in range(PLACEMENT_ATTEMPTS):
        min_h = max(12, net_h + 3)
        min_w = max(12, net_w + 3)
        if min_h > 30 or min_w > 30:
            return None
        height = rng.randint(min_h, max(18, min(min_h + 8, 30)))
        width = rng.randint(min_w, max(18, min(min_w + 8, 30)))

        corner = rng.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            legend_r, legend_c = 0, 0
        elif corner == "tr":
            legend_r, legend_c = 0, width - 3
        elif corner == "bl":
            legend_r, legend_c = height - 3, 0
        else:
            legend_r, legend_c = height - 3, width - 3
        legend_box = (legend_r, legend_c, legend_r + 2, legend_c + 2)

        if net_h > height or net_w > width:
            continue
        off_r = rng.randint(0, height - net_h)
        off_c = rng.randint(0, width - net_w)
        network_box = (off_r, off_c, off_r + net_h - 1, off_c + net_w - 1)
        if not _boxes_far(network_box, legend_box, gap=2):
            continue

        grid = np.full((height, width), BG, dtype=int)
        for r, c in pipe_cells:
            grid[off_r + r, off_c + c] = PIPE
        for (r, c), color in sources:
            rr, cc = off_r + r, off_c + c
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            if grid[rr, cc] != BG:
                return None
            grid[rr, cc] = color

        rows = list(PAIR_RESULTS.items())
        rng.shuffle(rows)
        for rr, (pair, result) in enumerate(rows):
            a, b = tuple(pair)
            if rng.random() < 0.5:
                a, b = b, a
            grid[legend_r + rr, legend_c + 0] = a
            grid[legend_r + rr, legend_c + 1] = b
            grid[legend_r + rr, legend_c + 2] = result
        return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    family = rng.randrange(4)
    paths, sources = _build_family(family, rng)
    if rng.random() < 0.45:
        paths.extend(_build_idle_component(paths, rng))
    paths, sources = _normalize_paths(paths, sources, turns=rng.randrange(4), reflect=(rng.random() < 0.5))
    grid = _place_candidate(paths, sources, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any((grid == PIPE) & (out != grid)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
