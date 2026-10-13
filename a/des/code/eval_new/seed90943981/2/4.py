from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GENERATION_RETRIES = 500
COARSE_HEIGHT_RANGE = (4, 11)
COARSE_WIDTH_RANGE = (5, 13)
CHAIN_COUNT_RANGE = (1, 3)
CHAIN_LENGTH_RANGE = (2, 6)
PALETTE_GAP_RANGE = (0, 2)
SIDES = ("top", "left", "right")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_VALUES = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(value) for value in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(bool(v) for v in mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask) - 1)))
    return runs


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_grid_boundary(top_left: tuple[int, int], height: int, width: int) -> bool:
    r0, c0 = map(int, top_left)
    return bool(r0 == 0 or c0 == 0 or r0 + 1 == height - 1 or c0 + 1 == width - 1)


def _extract_palette(grid: np.ndarray, bg: int) -> list[int]:
    runs = _runs(grid[-1] != bg)
    assert runs, "expected a palette strip on the bottom row"
    start, end = max(runs, key=lambda run: int(run[1] - run[0]))
    palette = [int(value) for value in grid[-1, start : end + 1]]
    assert len(palette) >= 2, "palette must encode at least two chain positions"
    assert all(int(color) != int(bg) for color in palette)
    return palette


def _find_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    work = np.asarray(grid[:-1], dtype=int)
    blocks: list[dict] = []
    for color in sorted(int(value) for value in np.unique(work) if int(value) != int(bg)):
        for comp in _components(work == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            assert len(comp) == 4, "each chain element must be a 2x2 block"
            assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "foreground objects must be solid 2x2 blocks"
            blocks.append(
                {
                    "color": int(color),
                    "top_left": (int(r0), int(c0)),
                }
            )
    assert blocks, "expected at least one chain block"
    return blocks


def _build_graph(blocks: list[dict]) -> list[list[int]]:
    pos_to_idx = {tuple(block["top_left"]): idx for idx, block in enumerate(blocks)}
    graph = [set() for _ in blocks]
    for idx, block in enumerate(blocks):
        r0, c0 = map(int, block["top_left"])
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            neighbor = pos_to_idx.get((int(r0 + dr), int(c0 + dc)))
            if neighbor is not None:
                graph[idx].add(int(neighbor))
    return [sorted(neighbors) for neighbors in graph]


def _ordered_chains(blocks: list[dict], graph: list[list[int]], height: int, width: int) -> list[list[int]]:
    seen: set[int] = set()
    chains: list[list[int]] = []
    for start in range(len(blocks)):
        if start in seen:
            continue
        stack = [int(start)]
        component: list[int] = []
        seen.add(int(start))
        while stack:
            idx = stack.pop()
            component.append(int(idx))
            for neighbor in graph[idx]:
                if neighbor not in seen:
                    seen.add(int(neighbor))
                    stack.append(int(neighbor))

        degrees = {idx: len(graph[idx]) for idx in component}
        assert all(int(deg) <= 2 for deg in degrees.values()), "each component must be a simple chain"

        if len(component) == 1:
            order = component
        else:
            endpoints = [idx for idx in component if int(degrees[idx]) == 1]
            assert len(endpoints) == 2, "a chain must have exactly two endpoints"
            head_candidates = [
                idx
                for idx in endpoints
                if _touches_grid_boundary(tuple(blocks[idx]["top_left"]), height, width)
            ]
            assert len(head_candidates) == 1, "expected a unique boundary-adjacent chain head"
            head = int(head_candidates[0])
            order = [int(head)]
            prev = -1
            cur = int(head)
            while True:
                nxt = [neighbor for neighbor in graph[cur] if int(neighbor) != int(prev)]
                if not nxt:
                    break
                assert len(nxt) == 1, "chain traversal must be unambiguous"
                prev, cur = int(cur), int(nxt[0])
                order.append(int(cur))
            assert len(order) == len(component), "chain traversal must visit every block exactly once"

        chains.append(order)
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 2x2 chain by its topological position using the bottom palette strip.

    Input structure:
    - The dominant color is the background.
    - Above the last row, every foreground object is a solid 2x2 block.
    - Blocks touch edge-to-edge to form one or more simple chains, and one endpoint of each chain touches the grid boundary.
    - The bottom row contains one contiguous palette strip of single pixels whose left-to-right order defines replacement colors.

    Rule:
    - Read the palette strip from left to right, then erase the bottom row back to the background.
    - Group the 2x2 blocks into chains by shared-side adjacency.
    - For each chain, start from the unique endpoint block that touches the grid boundary and follow the chain order to the other endpoint.
    - Replace the 1st block with palette color 1, the 2nd block with palette color 2, and so on independently for every chain.

    Key insight:
    - Original block colors are irrelevant; the solution depends on recovering the chain topology, identifying the boundary-touching head, and using the bottom strip as a position-indexed palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    palette = _extract_palette(g, bg)
    blocks = _find_blocks(g, bg)
    graph = _build_graph(blocks)
    chains = _ordered_chains(blocks, graph, *g.shape)

    out = g.copy()
    out[-1, :] = int(bg)
    for order in chains:
        assert len(order) <= len(palette), "palette must cover the longest chain"
        for idx, block_idx in enumerate(order):
            r0, c0 = map(int, blocks[block_idx]["top_left"])
            out[r0 : r0 + 2, c0 : c0 + 2] = int(palette[idx])
    return out


def _touches_side(cell: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = map(int, cell)
    return bool(r == 0 or c == 0 or c == cols - 1)


def _is_interior(cell: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = map(int, cell)
    return bool(r > 0 and 0 < c < cols - 1)


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((int(nr), int(nc)))
    return out


def _expanded_forbidden(cells: set[tuple[int, int]], rows: int, cols: int) -> set[tuple[int, int]]:
    out = set(cells)
    for cell in list(cells):
        out.update(_neighbors(cell, rows, cols))
    return out


def _start_pairs(rows: int, cols: int, side: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if side == "top":
        if rows < 2 or cols < 3:
            return pairs
        for c in range(1, cols - 1):
            pairs.append(((0, int(c)), (1, int(c))))
    elif side == "left":
        if cols < 2:
            return pairs
        for r in range(1, rows):
            pairs.append(((int(r), 0), (int(r), 1)))
    else:
        assert side == "right"
        if cols < 2:
            return pairs
        for r in range(1, rows):
            pairs.append(((int(r), cols - 1), (int(r), cols - 2)))
    return pairs


def _valid_extension(
    candidate: tuple[int, int],
    path: list[tuple[int, int]],
    current: tuple[int, int],
    used: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> bool:
    if candidate in used or candidate in forbidden:
        return False
    if not _is_interior(candidate, rows, cols):
        return False
    for cell in path:
        if cell == current:
            continue
        if abs(int(cell[0]) - int(candidate[0])) + abs(int(cell[1]) - int(candidate[1])) == 1:
            return False
    return True


def _find_path(
    rows: int,
    cols: int,
    length: int,
    side: str,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    pairs = _start_pairs(rows, cols, side)
    random.shuffle(pairs)

    def dfs(
        path: list[tuple[int, int]],
        used: set[tuple[int, int]],
        current: tuple[int, int],
        previous: tuple[int, int],
    ) -> bool:
        if len(path) == length:
            return True
        options = [cell for cell in _neighbors(current, rows, cols) if cell != previous]
        random.shuffle(options)
        options.sort(
            key=lambda cell: sum(
                1
                for nxt in _neighbors(cell, rows, cols)
                if nxt != current and _is_interior(nxt, rows, cols) and nxt not in used and nxt not in forbidden
            )
        )
        for cell in options:
            if not _valid_extension(cell, path, current, used, forbidden, rows, cols):
                continue
            path.append(cell)
            used.add(cell)
            if dfs(path, used, cell, current):
                return True
            used.remove(cell)
            path.pop()
        return False

    for head, second in pairs:
        if head in forbidden or second in forbidden:
            continue
        path = [head, second]
        used = {head, second}
        if length == 2 or dfs(path, used, second, head):
            return path
    return None


def _validate_paths(paths: list[list[tuple[int, int]]], rows: int, cols: int) -> None:
    all_cells: list[tuple[int, int]] = []
    for path in paths:
        assert len(path) >= 2
        assert _touches_side(path[0], rows, cols)
        assert all(not _touches_side(cell, rows, cols) for cell in path[1:])
        for idx, cell in enumerate(path):
            for jdx, other in enumerate(path):
                if idx == jdx:
                    continue
                dist = abs(int(cell[0]) - int(other[0])) + abs(int(cell[1]) - int(other[1]))
                if dist == 1:
                    assert abs(idx - jdx) == 1
        all_cells.extend(path)
    assert len(all_cells) == len(set(all_cells))
    for idx, path_a in enumerate(paths):
        for path_b in paths[idx + 1 :]:
            for cell_a in path_a:
                for cell_b in path_b:
                    dist = abs(int(cell_a[0]) - int(cell_b[0])) + abs(int(cell_a[1]) - int(cell_b[1]))
                    assert dist != 1


def _render_input(
    rows: int,
    cols: int,
    gap_rows: int,
    bg: int,
    paths: list[list[tuple[int, int]]],
    chain_colors: list[list[int]],
    palette: list[int],
    palette_start: int,
) -> np.ndarray:
    height = 2 * rows + gap_rows + 1
    width = 2 * cols
    grid = np.full((height, width), bg, dtype=int)
    for path, colors in zip(paths, chain_colors):
        for cell, color in zip(path, colors):
            r0 = 2 * int(cell[0])
            c0 = 2 * int(cell[1])
            grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)
    grid[-1, palette_start : palette_start + len(palette)] = np.asarray(palette, dtype=int)
    return grid


def _render_output(
    rows: int,
    cols: int,
    gap_rows: int,
    bg: int,
    paths: list[list[tuple[int, int]]],
    palette: list[int],
    palette_start: int,
) -> np.ndarray:
    out = _render_input(rows, cols, gap_rows, bg, paths, [[bg] * len(path) for path in paths], palette, palette_start)
    out[-1, :] = int(bg)
    for path in paths:
        for idx, cell in enumerate(path):
            r0 = 2 * int(cell[0])
            c0 = 2 * int(cell[1])
            out[r0 : r0 + 2, c0 : c0 + 2] = int(palette[idx])
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*COARSE_HEIGHT_RANGE)
    cols = random.randint(*COARSE_WIDTH_RANGE)
    gap_rows = random.randint(*PALETTE_GAP_RANGE)
    chain_count = random.randint(*CHAIN_COUNT_RANGE)
    max_chain_len = random.randint(CHAIN_LENGTH_RANGE[0], min(CHAIN_LENGTH_RANGE[1], rows + cols - 2))
    chain_lengths = [int(max_chain_len)] + [random.randint(CHAIN_LENGTH_RANGE[0], max_chain_len) for _ in range(chain_count - 1)]
    random.shuffle(chain_lengths)
    sides = random.sample(SIDES, k=chain_count)

    paths: list[list[tuple[int, int]]] = []
    forbidden: set[tuple[int, int]] = set()
    for side, length in zip(sides, chain_lengths):
        path = _find_path(rows, cols, int(length), side, forbidden)
        if path is None:
            return None
        paths.append(path)
        forbidden.update(_expanded_forbidden(set(path), rows, cols))

    _validate_paths(paths, rows, cols)

    bg = random.choice(COLOR_VALUES)
    usable_colors = [color for color in COLOR_VALUES if int(color) != int(bg)]
    palette = random.sample(usable_colors, max_chain_len)
    chain_colors = [random.sample(usable_colors, len(path)) for path in paths]

    width = 2 * cols
    palette_start = random.randint(0, width - len(palette))
    inp = _render_input(rows, cols, gap_rows, bg, paths, chain_colors, palette, palette_start)
    out = _render_output(rows, cols, gap_rows, bg, paths, palette, palette_start)

    if inp.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if int((inp == bg).sum()) <= int(inp.size // 2):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
