from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
PERP: dict[str, tuple[str, str]] = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}

BG_COLOR = 0
ARROW_COLOR = 7
BLOCK_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
BLOCK_SIZE = 2

GEN_RETRIES = 500
CHAIN_RETRIES = 250
NODE_RETRIES = 160


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_square_component(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != 4:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    return (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _is_arrow_component(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != 3:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    return (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _block_cells(pos: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = pos
    return {
        (r, c),
        (r, c + 1),
        (r + 1, c),
        (r + 1, c + 1),
    }


def _shift(pos: tuple[int, int], direction: str, steps: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return pos[0] + dr * steps, pos[1] + dc * steps


def _steps_to_edge(pos: tuple[int, int], direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    if direction == "U":
        return r
    if direction == "D":
        return h - BLOCK_SIZE - r
    if direction == "L":
        return c
    return w - BLOCK_SIZE - c


def _collect_scene(
    grid: np.ndarray,
) -> tuple[int, int, dict[int, set[tuple[int, int]]], list[tuple[int, tuple[int, int]]]]:
    bg = _background_color(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]

    arrow_candidates: list[tuple[int, list[list[tuple[int, int]]]]] = []
    for color in colors:
        comps = _components(grid == color)
        if comps and all(_is_arrow_component(comp) for comp in comps):
            arrow_candidates.append((color, comps))
    assert len(arrow_candidates) == 1, "expected one arrow color"

    arrow_color, arrow_comps = arrow_candidates[0]
    arrows = {idx: set(comp) for idx, comp in enumerate(arrow_comps)}

    blocks: list[tuple[int, tuple[int, int]]] = []
    for color in colors:
        if color == arrow_color:
            continue
        comps = _components(grid == color)
        assert len(comps) == 1, "each block color must form one component"
        comp = comps[0]
        assert _is_square_component(comp), "non-arrow components must be 2x2 squares"
        r0, c0, _r1, _c1 = _bbox(comp)
        blocks.append((color, (r0, c0)))

    blocks.sort(key=lambda item: item[0])
    return bg, int(arrow_color), arrows, blocks


def _arrow_side_for_block(pos: tuple[int, int], arrow_cells: set[tuple[int, int]]) -> str | None:
    r, c = pos
    hits: list[str] = []
    if any(rr == r - 1 and c <= cc <= c + 1 for rr, cc in arrow_cells):
        hits.append("U")
    if any(rr == r + 2 and c <= cc <= c + 1 for rr, cc in arrow_cells):
        hits.append("D")
    if any(cc == c - 1 and r <= rr <= r + 1 for rr, cc in arrow_cells):
        hits.append("L")
    if any(cc == c + 2 and r <= rr <= r + 1 for rr, cc in arrow_cells):
        hits.append("R")
    if len(hits) != 1:
        return None
    return hits[0]


def _direction_from_arrow(pos: tuple[int, int], arrow_cells: set[tuple[int, int]]) -> str:
    r, c = pos
    side = _arrow_side_for_block(pos, arrow_cells)
    assert side is not None, "arrow must be adjacent on exactly one side"

    if side == "U":
        near = [(rr, cc) for rr, cc in arrow_cells if rr == r - 1 and c <= cc <= c + 1]
        assert len(near) == 1
        near_r, near_c = near[0]
        far = (near_r - 1, near_c)
        assert far in arrow_cells
        tip = next(iter(arrow_cells - {near[0], far}))
        assert tip[0] == far[0] and abs(tip[1] - far[1]) == 1
        return "L" if tip[1] < far[1] else "R"

    if side == "D":
        near = [(rr, cc) for rr, cc in arrow_cells if rr == r + 2 and c <= cc <= c + 1]
        assert len(near) == 1
        near_r, near_c = near[0]
        far = (near_r + 1, near_c)
        assert far in arrow_cells
        tip = next(iter(arrow_cells - {near[0], far}))
        assert tip[0] == far[0] and abs(tip[1] - far[1]) == 1
        return "L" if tip[1] < far[1] else "R"

    if side == "L":
        near = [(rr, cc) for rr, cc in arrow_cells if cc == c - 1 and r <= rr <= r + 1]
        assert len(near) == 1
        near_r, near_c = near[0]
        far = (near_r, near_c - 1)
        assert far in arrow_cells
        tip = next(iter(arrow_cells - {near[0], far}))
        assert tip[1] == far[1] and abs(tip[0] - far[0]) == 1
        return "U" if tip[0] < far[0] else "D"

    near = [(rr, cc) for rr, cc in arrow_cells if cc == c + 2 and r <= rr <= r + 1]
    assert len(near) == 1
    near_r, near_c = near[0]
    far = (near_r, near_c + 1)
    assert far in arrow_cells
    tip = next(iter(arrow_cells - {near[0], far}))
    assert tip[1] == far[1] and abs(tip[0] - far[0]) == 1
    return "U" if tip[0] < far[0] else "D"


def _adjacent_arrow_ids(
    pos: tuple[int, int],
    arrows: dict[int, set[tuple[int, int]]],
) -> list[int]:
    out = []
    for arrow_id, cells in arrows.items():
        if _arrow_side_for_block(pos, cells) is not None:
            out.append(arrow_id)
    return out


def _follow_chain(
    start_pos: tuple[int, int],
    start_arrow_id: int,
    arrows: dict[int, set[tuple[int, int]]],
    shape: tuple[int, int],
) -> tuple[int, int]:
    pos = start_pos
    arrow_id = start_arrow_id
    seen: set[tuple[tuple[int, int], int]] = set()

    while True:
        state = (pos, arrow_id)
        assert state not in seen, "arrow chain must not cycle"
        seen.add(state)

        direction = _direction_from_arrow(pos, arrows[arrow_id])
        max_steps = _steps_to_edge(pos, direction, shape)

        next_pos: tuple[int, int] | None = None
        next_arrow_id: int | None = None
        for step in range(1, max_steps + 1):
            candidate = _shift(pos, direction, step)
            hits = [aid for aid in _adjacent_arrow_ids(candidate, arrows) if aid != arrow_id]
            if hits:
                assert len(hits) == 1, "movement must meet one next arrow at a time"
                next_pos = candidate
                next_arrow_id = hits[0]
                break

        if next_pos is None:
            return _shift(pos, direction, max_steps)

        pos = next_pos
        arrow_id = int(next_arrow_id)


def _solve_scene(
    grid: np.ndarray,
) -> tuple[int, int, dict[int, set[tuple[int, int]]], list[tuple[int, tuple[int, int]]], dict[int, tuple[int, int]]]:
    bg, arrow_color, arrows, blocks = _collect_scene(grid)
    finals: dict[int, tuple[int, int]] = {}
    for color, pos in blocks:
        hits = _adjacent_arrow_ids(pos, arrows)
        assert len(hits) == 1, "each block needs one adjacent starting arrow"
        finals[color] = _follow_chain(pos, hits[0], arrows, grid.shape)
    return bg, arrow_color, arrows, blocks, finals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 2x2 colored block along the chain of adjacent 3-cell L arrows until
    it reaches the final edge position.
    """
    g = np.asarray(grid, dtype=int)
    bg, arrow_color, _arrows, blocks, finals = _solve_scene(g)

    out = g.copy()
    for _color, pos in blocks:
        for r, c in _block_cells(pos):
            out[r, c] = bg

    occupied: set[tuple[int, int]] = set()
    for color, final_pos in sorted(finals.items()):
        cells = _block_cells(final_pos)
        for r, c in cells:
            assert out[r, c] != arrow_color, "moved block must not cover an arrow"
            assert (r, c) not in occupied, "final block positions must stay disjoint"
        occupied |= cells
        for r, c in cells:
            out[r, c] = int(color)

    return out


def _arrow_cells(
    pos: tuple[int, int],
    direction: str,
    side: str,
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    h, w = shape
    r, c = pos

    if direction == "R":
        near_c = c
        if side == "U":
            cells = {(r - 1, near_c), (r - 2, near_c), (r - 2, near_c + 1)}
        else:
            cells = {(r + 2, near_c), (r + 3, near_c), (r + 3, near_c + 1)}
    elif direction == "L":
        near_c = c + 1
        if side == "U":
            cells = {(r - 1, near_c), (r - 2, near_c), (r - 2, near_c - 1)}
        else:
            cells = {(r + 2, near_c), (r + 3, near_c), (r + 3, near_c - 1)}
    elif direction == "D":
        near_r = r
        if side == "L":
            cells = {(near_r, c - 1), (near_r, c - 2), (near_r + 1, c - 2)}
        else:
            cells = {(near_r, c + 2), (near_r, c + 3), (near_r + 1, c + 3)}
    else:
        near_r = r + 1
        if side == "L":
            cells = {(near_r, c - 1), (near_r, c - 2), (near_r - 1, c - 2)}
        else:
            cells = {(near_r, c + 2), (near_r, c + 3), (near_r - 1, c + 3)}

    if all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
        return cells
    return None


def _side_options(pos: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[str]:
    out: list[str] = []
    for side in PERP[direction]:
        cells = _arrow_cells(pos, direction, side, shape)
        if cells is not None:
            out.append(side)
    return out


def _trace_positions(
    nodes: list[tuple[int, int]],
    directions: list[str],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    positions = [nodes[0]]
    for idx in range(len(nodes) - 1):
        cur = nodes[idx]
        nxt = nodes[idx + 1]
        steps = abs(nxt[0] - cur[0]) + abs(nxt[1] - cur[1])
        positions.extend(_shift(cur, directions[idx], step) for step in range(1, steps + 1))

    final_steps = _steps_to_edge(nodes[-1], directions[-1], shape)
    positions.extend(_shift(nodes[-1], directions[-1], step) for step in range(1, final_steps + 1))
    return positions, positions[-1]


def _path_cells(positions: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for pos in positions:
        out |= _block_cells(pos)
    return out


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _touches_4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _choose_directions(n_arrows: int) -> list[str]:
    directions = [random.choice(tuple(DIRS))]
    for _ in range(1, n_arrows):
        directions.append(random.choice(PERP[directions[-1]]))
    return directions


def _build_nodes(
    shape: tuple[int, int],
    directions: list[str],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]] | None:
    h, w = shape
    top_left_h = h - BLOCK_SIZE + 1
    top_left_w = w - BLOCK_SIZE + 1

    for _ in range(NODE_RETRIES):
        start = (random.randrange(top_left_h), random.randrange(top_left_w))
        if not _side_options(start, directions[0], shape):
            continue

        nodes = [start]
        used_positions = {start}
        ok = True
        for idx, direction in enumerate(directions[:-1]):
            cur = nodes[-1]
            max_len = min(_steps_to_edge(cur, direction, shape), 8)
            candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
            for length in range(3, max_len + 1):
                nxt = _shift(cur, direction, length)
                traversed = {_shift(cur, direction, step) for step in range(1, length + 1)}
                if traversed & used_positions:
                    continue
                if not _side_options(nxt, directions[idx + 1], shape):
                    continue
                candidates.append((nxt, traversed))
            if not candidates:
                ok = False
                break
            nxt, traversed = random.choice(candidates)
            nodes.append(nxt)
            used_positions |= traversed

        if not ok:
            continue

        final_steps = _steps_to_edge(nodes[-1], directions[-1], shape)
        if final_steps < 2:
            continue
        tail = {_shift(nodes[-1], directions[-1], step) for step in range(1, final_steps + 1)}
        if tail & used_positions:
            continue

        positions, final_pos = _trace_positions(nodes, directions, shape)
        if len(set(positions)) != len(positions):
            continue
        return nodes, positions, final_pos
    return None


def _choose_arrows(
    nodes: list[tuple[int, int]],
    directions: list[str],
    positions: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[set[tuple[int, int]]] | None:
    options: list[list[set[tuple[int, int]]]] = []
    for pos, direction in zip(nodes, directions):
        node_options: list[set[tuple[int, int]]] = []
        for side in _side_options(pos, direction, shape):
            cells = _arrow_cells(pos, direction, side, shape)
            assert cells is not None
            if not cells.isdisjoint(path_cells):
                continue
            adjacent_positions = [p for p in positions if _arrow_side_for_block(p, cells) is not None]
            if set(adjacent_positions) != {pos}:
                continue
            node_options.append(cells)
        if not node_options:
            return None
        random.shuffle(node_options)
        options.append(node_options)

    order = sorted(range(len(nodes)), key=lambda idx: len(options[idx]))
    chosen: list[set[tuple[int, int]] | None] = [None] * len(nodes)
    used_cells: set[tuple[int, int]] = set()

    def rec(depth: int) -> bool:
        nonlocal used_cells
        if depth == len(order):
            return True
        idx = order[depth]
        for cells in options[idx]:
            if cells & used_cells:
                continue
            if any(prev is not None and _touches_4(cells, prev) for prev in chosen):
                continue
            chosen[idx] = cells
            old_used = used_cells
            used_cells = used_cells | cells
            if rec(depth + 1):
                return True
            used_cells = old_used
            chosen[idx] = None
        return False

    if not rec(0):
        return None
    return [cells for cells in chosen if cells is not None]


def _render_input(shape: tuple[int, int], chains: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for chain in chains:
        for arrow in chain["arrows"]:  # type: ignore[index]
            for r, c in arrow:
                grid[r, c] = ARROW_COLOR
    for chain in chains:
        color = int(chain["color"])
        start = chain["nodes"][0]  # type: ignore[index]
        for r, c in _block_cells(start):
            grid[r, c] = color
    return grid


def _render_output(shape: tuple[int, int], chains: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for chain in chains:
        for arrow in chain["arrows"]:  # type: ignore[index]
            for r, c in arrow:
                grid[r, c] = ARROW_COLOR
    for chain in chains:
        color = int(chain["color"])
        final_pos = chain["final"]  # type: ignore[index]
        for r, c in _block_cells(final_pos):
            grid[r, c] = color
    return grid


def _build_chain(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    color: int,
    require_multi: bool,
) -> dict[str, object] | None:
    for _ in range(CHAIN_RETRIES):
        if require_multi:
            n_arrows = random.choice([2, 2, 3, 3, 4])
        else:
            n_arrows = random.choice([1, 1, 2, 2, 3, 4])
        directions = _choose_directions(n_arrows)
        built = _build_nodes(shape, directions)
        if built is None:
            continue
        nodes, positions, final_pos = built
        path_cells = _path_cells(positions)
        if path_cells & forbidden:
            continue

        arrows = _choose_arrows(nodes, directions, positions, path_cells, shape)
        if arrows is None:
            continue
        arrow_cells = set().union(*arrows)
        if arrow_cells & forbidden:
            continue

        reserve = _dilate(path_cells, shape, 2) | _dilate(arrow_cells, shape, 1)
        chain: dict[str, object] = {
            "color": color,
            "nodes": nodes,
            "directions": directions,
            "positions": positions,
            "final": final_pos,
            "arrows": arrows,
            "reserve": reserve,
        }

        local_inp = _render_input(shape, [chain])
        local_out = _render_output(shape, [chain])
        try:
            if not np.array_equal(main(local_inp), local_out):
                continue
        except AssertionError:
            continue
        return chain
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(17, 24)
        w = random.randint(17, 24)
        shape = (h, w)

        n_chains = random.choice([1, 2, 2, 3])
        colors = random.sample(BLOCK_COLORS, n_chains)
        require_multi = random.random() < 0.7

        chains: list[dict[str, object]] = []
        forbidden: set[tuple[int, int]] = set()
        ok = True
        for idx, color in enumerate(colors):
            chain = _build_chain(shape, forbidden, color, require_multi and idx == 0)
            if chain is None:
                ok = False
                break
            chains.append(chain)
            forbidden |= chain["reserve"]  # type: ignore[operator]
        if not ok:
            continue

        inp = _render_input(shape, chains)
        expected = _render_output(shape, chains)

        try:
            _bg, _arrow_color, _arrows, blocks, finals = _solve_scene(inp)
        except AssertionError:
            continue
        if len(blocks) != n_chains:
            continue

        planned = {int(chain["color"]): chain["final"] for chain in chains}
        if finals != planned:
            continue

        try:
            out = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(out, expected):
            continue

        if require_multi and not any(len(chain["nodes"]) >= 2 for chain in chains):
            continue
        return inp

    raise RuntimeError("failed to generate a valid scene")
