from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BASE_GRAPHS = [
    {
        (1, 1),
        (1, 0),
        (1, 2),
        (0, 1),
        (2, 1),
        (1, 3),
        (0, 2),
        (3, 1),
    },
    {
        (1, 1),
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 2),
    },
    {
        (1, 1),
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
        (0, 2),
        (0, 3),
        (2, 2),
    },
]

NODE_SIZE = 5
INNER_SIZE = 3
DICT_PATTERNS = 4
DICT_GAP = 1
MAX_ATTEMPTS = 300
MAX_SLOT_SPAN = 5


def _most_common_color(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _line_color(grid: np.ndarray, bg: int) -> int:
    fg = grid[grid != bg]
    assert fg.size > 0, "expected foreground pixels"
    vals, cnts = np.unique(fg, return_counts=True)
    return int(vals[cnts.argmax()])


def _extract_patterns(grid: np.ndarray, bg: int, line: int) -> tuple[list[np.ndarray], int]:
    mask = (grid != bg) & (grid != line)
    comps = _components(mask)
    assert len(comps) == DICT_PATTERNS, "expected four dictionary patterns"

    items: list[tuple[int, int, np.ndarray]] = []
    inner = None
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert patch.shape[0] == patch.shape[1], "patterns must be square"
        if inner is None:
            inner = int(patch.shape[0])
        assert patch.shape == (inner, inner), "all patterns must have the same size"
        items.append((c0, r0, patch))

    assert inner is not None and inner >= 1
    items.sort(key=lambda item: (item[0], item[1]))
    return [patch for _, _, patch in items], inner


def _find_nodes(grid: np.ndarray, bg: int, line: int, node_size: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    nodes: list[tuple[int, int]] = []
    for r0 in range(h - node_size + 1):
        for c0 in range(w - node_size + 1):
            sub = grid[r0 : r0 + node_size, c0 : c0 + node_size]
            if not np.all(sub[1:-1, 1:-1] == bg):
                continue
            if (
                np.all(sub[0, :] == line)
                and np.all(sub[-1, :] == line)
                and np.all(sub[:, 0] == line)
                and np.all(sub[:, -1] == line)
            ):
                nodes.append((r0, c0))
    assert nodes, "expected at least one hollow node"
    return nodes


def _compute_degrees(grid: np.ndarray, line: int, node_size: int, nodes: list[tuple[int, int]]) -> list[int]:
    mid = node_size // 2
    degrees = [0] * len(nodes)

    by_row: dict[int, list[tuple[int, int, int]]] = {}
    by_col: dict[int, list[tuple[int, int, int]]] = {}
    for idx, (r0, c0) in enumerate(nodes):
        by_row.setdefault(r0 + mid, []).append((c0 + mid, idx, c0))
        by_col.setdefault(c0 + mid, []).append((r0 + mid, idx, r0))

    for row, items in by_row.items():
        items.sort()
        for (_, idx_a, c0_a), (_, idx_b, c0_b) in zip(items, items[1:], strict=False):
            start = c0_a + node_size - 1
            end = c0_b
            if np.all(grid[row, start : end + 1] == line):
                degrees[idx_a] += 1
                degrees[idx_b] += 1

    for col, items in by_col.items():
        items.sort()
        for (_, idx_a, r0_a), (_, idx_b, r0_b) in zip(items, items[1:], strict=False):
            start = r0_a + node_size - 1
            end = r0_b
            if np.all(grid[start : end + 1, col] == line):
                degrees[idx_a] += 1
                degrees[idx_b] += 1

    return degrees


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow square node with the dictionary pattern indexed by its path degree.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    line = _line_color(g, bg)
    patterns, inner = _extract_patterns(g, bg, line)
    node_size = inner + 2
    nodes = _find_nodes(g, bg, line, node_size)
    degrees = _compute_degrees(g, line, node_size, nodes)

    out = g.copy()
    for (r0, c0), degree in zip(nodes, degrees, strict=False):
        assert 1 <= degree <= len(patterns), "node degree must map to a dictionary pattern"
        out[r0 + 1 : r0 + 1 + inner, c0 + 1 : c0 + 1 + inner] = patterns[degree - 1]
    return out


def _normalize_slots(nodes: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in nodes)
    min_c = min(c for _, c in nodes)
    return {(r - min_r, c - min_c) for r, c in nodes}


def _rotate_slots(nodes: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(-c, r) for r, c in nodes}


def _mirror_slots(nodes: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, -c) for r, c in nodes}


def _slot_degrees(nodes: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {
        node: sum((node[0] + dr, node[1] + dc) in nodes for dr, dc in DIRS4)
        for node in nodes
    }


def _slot_span(nodes: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in nodes]
    cols = [c for _, c in nodes]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _leaf_candidates(nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cands = set()
    for r, c in nodes:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in nodes:
                continue
            adj = sum((nr + rr, nc + cc) in nodes for rr, cc in DIRS4)
            if adj == 1:
                cands.add((nr, nc))
    return list(cands)


def _build_slot_graph(rng: random.Random) -> set[tuple[int, int]]:
    nodes = set(rng.choice(BASE_GRAPHS))
    extra_steps = rng.randint(0, 3)
    for _ in range(extra_steps):
        cands = _leaf_candidates(nodes)
        rng.shuffle(cands)
        added = False
        for cand in cands:
            trial = set(nodes)
            trial.add(cand)
            norm = _normalize_slots(trial)
            span_r, span_c = _slot_span(norm)
            deg_values = set(_slot_degrees(norm).values())
            if span_r > MAX_SLOT_SPAN or span_c > MAX_SLOT_SPAN:
                continue
            if max(deg_values) > 4:
                continue
            if {1, 2, 3, 4}.issubset(deg_values):
                nodes = norm
                added = True
                break
        if not added:
            break

    nodes = _normalize_slots(nodes)
    for _ in range(rng.randrange(4)):
        nodes = _rotate_slots(nodes)
    if rng.choice((False, True)):
        nodes = _mirror_slots(nodes)
    return _normalize_slots(nodes)


def _random_pattern_mask(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        target = rng.randint(4, 8)
        cells = {(rng.randrange(INNER_SIZE), rng.randrange(INNER_SIZE))}
        while len(cells) < target:
            r, c = rng.choice(tuple(cells))
            dr, dc = rng.choice(DIRS4)
            nr, nc = r + dr, c + dc
            if 0 <= nr < INNER_SIZE and 0 <= nc < INNER_SIZE:
                cells.add((nr, nc))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows != set(range(INNER_SIZE)) or cols != set(range(INNER_SIZE)):
            continue

        mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to create a full-span pattern")


def _make_patterns(bg: int, colors: list[int], rng: random.Random) -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    seen = set()
    while len(patterns) < DICT_PATTERNS:
        mask = _random_pattern_mask(rng)
        key = tuple(int(x) for x in mask.ravel())
        if key in seen:
            continue
        seen.add(key)
        patch = np.full((INNER_SIZE, INNER_SIZE), bg, dtype=int)
        patch[mask] = colors[len(patterns)]
        patterns.append(patch)
    return patterns


def _draw_node(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0, c0 : c0 + NODE_SIZE] = color
    grid[r0 + NODE_SIZE - 1, c0 : c0 + NODE_SIZE] = color
    grid[r0 : r0 + NODE_SIZE, c0] = color
    grid[r0 : r0 + NODE_SIZE, c0 + NODE_SIZE - 1] = color


def _render_network(
    grid: np.ndarray,
    nodes: set[tuple[int, int]],
    top: int,
    left: int,
    pitch_r: int,
    pitch_c: int,
    line: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    pixel_nodes: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in nodes:
        r0 = top + sr * pitch_r
        c0 = left + sc * pitch_c
        pixel_nodes[(sr, sc)] = (r0, c0)
        _draw_node(grid, r0, c0, line)

    for sr, sc in nodes:
        r0, c0 = pixel_nodes[(sr, sc)]
        if (sr, sc + 1) in nodes:
            right_c0 = pixel_nodes[(sr, sc + 1)][1]
            row = r0 + NODE_SIZE // 2
            grid[row, c0 + NODE_SIZE : right_c0] = line
        if (sr + 1, sc) in nodes:
            down_r0 = pixel_nodes[(sr + 1, sc)][0]
            col = c0 + NODE_SIZE // 2
            grid[r0 + NODE_SIZE : down_r0, col] = line
    return pixel_nodes


def _expected_output(
    grid: np.ndarray,
    pixel_nodes: dict[tuple[int, int], tuple[int, int]],
    slot_nodes: set[tuple[int, int]],
    patterns: list[np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    degrees = _slot_degrees(slot_nodes)
    for node, degree in degrees.items():
        r0, c0 = pixel_nodes[node]
        out[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE] = patterns[degree - 1]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg, line, *pattern_colors = rng.sample(range(10), 2 + DICT_PATTERNS)
    patterns = _make_patterns(bg, pattern_colors, rng)
    slot_nodes = _build_slot_graph(rng)
    slot_h, slot_w = _slot_span(slot_nodes)

    pitch_r = rng.randint(6, 7)
    pitch_c = rng.randint(6, 7)
    net_h = NODE_SIZE + (slot_h - 1) * pitch_r
    net_w = NODE_SIZE + (slot_w - 1) * pitch_c

    dict_h = INNER_SIZE
    dict_w = DICT_PATTERNS * INNER_SIZE + (DICT_PATTERNS - 1) * DICT_GAP
    sep_v = rng.randint(2, 4)
    margin_top = rng.randint(1, 3)
    margin_bottom = rng.randint(1, 3)
    margin_left = rng.randint(1, 3)
    margin_right = rng.randint(1, 3)
    extra_h = rng.randint(0, 4)
    extra_w = rng.randint(0, 6)

    dict_on_top = rng.choice((False, True))
    dict_on_left = rng.choice((False, True))

    if dict_on_top:
        height = margin_top + dict_h + sep_v + net_h + margin_bottom + extra_h
        dict_r = margin_top
        net_r_min = margin_top + dict_h + sep_v
        net_r_max = height - margin_bottom - net_h
    else:
        height = margin_top + net_h + sep_v + dict_h + margin_bottom + extra_h
        dict_r = height - margin_bottom - dict_h
        net_r_min = margin_top
        net_r_max = dict_r - sep_v - net_h

    width = max(margin_left + dict_w + margin_right, margin_left + net_w + margin_right) + extra_w
    dict_c = margin_left if dict_on_left else width - margin_right - dict_w
    net_c_min = margin_left
    net_c_max = width - margin_right - net_w

    if net_r_min > net_r_max or net_c_min > net_c_max:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for idx, patch in enumerate(patterns):
        c0 = dict_c + idx * (INNER_SIZE + DICT_GAP)
        grid[dict_r : dict_r + INNER_SIZE, c0 : c0 + INNER_SIZE] = patch

    net_r = rng.randint(net_r_min, net_r_max)
    net_c = rng.randint(net_c_min, net_c_max)
    pixel_nodes = _render_network(grid, slot_nodes, net_r, net_c, pitch_r, pitch_c, line)
    expected = _expected_output(grid, pixel_nodes, slot_nodes, patterns)

    if _most_common_color(grid) != bg:
        return None
    if _line_color(grid, bg) != line:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
