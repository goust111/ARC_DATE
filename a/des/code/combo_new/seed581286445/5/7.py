from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

ARROW_PATTERNS = {
    "right": frozenset({(0, 0), (1, 1), (2, 0)}),
    "left": frozenset({(0, 1), (1, 0), (2, 1)}),
    "down": frozenset({(0, 0), (0, 2), (1, 1)}),
    "up": frozenset({(0, 1), (1, 0), (1, 2)}),
}

GRID_LIMIT = 30
LATTICE_SIZE = 5
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    deltas = DIR4 if connectivity == 4 else DIR8
    out: list[tuple[int, int]] = []
    for dr, dc in deltas:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            out.append((rr, cc))
    return out


def _components_of_color(grid: np.ndarray, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors(cr, cc, grid.shape, connectivity):
                if not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _decode_arrow(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) != 3:
        return None
    norm = _normalize(coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return direction
    return None


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _infer_arrow_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, np.unique(grid))) - {bg})
    candidates: list[int] = []
    for color in colors:
        comps = _components_of_color(grid, color, connectivity=8)
        if comps and all(_decode_arrow(comp) is not None for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one arrow color"
    return candidates[0]


def _extract_squares(grid: np.ndarray, bg: int, arrow_color: int) -> list[dict[str, object]]:
    squares: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, arrow_color}):
        for comp in _components_of_color(grid, color, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            side_r = r1 - r0 + 1
            side_c = c1 - c0 + 1
            assert side_r == side_c, "every block must be square"
            assert len(comp) == side_r * side_c, "every block must be solid"
            squares.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "side": int(side_r),
                }
            )
    assert squares, "expected at least one colored block"
    side = int(squares[0]["side"])
    assert side >= 3, "square size must support the 3-pixel arrow"
    assert all(int(square["side"]) == side for square in squares), "all blocks must have the same size"
    return squares


def _arrow_endpoints(
    squares: list[dict[str, object]],
    direction: str,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    above: list[tuple[int, int]] = []
    below: list[tuple[int, int]] = []

    for idx, square in enumerate(squares):
        sr0, sc0, sr1, sc1 = square["bbox"]  # type: ignore[assignment]
        sr0 = int(sr0)
        sc0 = int(sc0)
        sr1 = int(sr1)
        sc1 = int(sc1)
        if _intervals_overlap(r0, r1, sr0, sr1):
            if sc1 < c0:
                left.append((c0 - sc1, idx))
            if sc0 > c1:
                right.append((sc0 - c1, idx))
        if _intervals_overlap(c0, c1, sc0, sc1):
            if sr1 < r0:
                above.append((r0 - sr1, idx))
            if sr0 > r1:
                below.append((sr0 - r1, idx))

    if direction == "right":
        assert left and right
        return min(left)[1], min(right)[1]
    if direction == "left":
        assert left and right
        return min(right)[1], min(left)[1]
    if direction == "down":
        assert above and below
        return min(above)[1], min(below)[1]
    assert direction == "up"
    assert above and below
    return min(below)[1], min(above)[1]


def _extract_graph(
    grid: np.ndarray,
    squares: list[dict[str, object]],
    arrow_color: int,
) -> tuple[dict[int, int], dict[int, int]]:
    pred: dict[int, int] = {}
    succ: dict[int, int] = {}
    for comp in _components_of_color(grid, arrow_color, connectivity=8):
        direction = _decode_arrow(comp)
        assert direction is not None, "invalid arrow component"
        src, dst = _arrow_endpoints(squares, direction, _bbox(comp))
        assert src != dst
        assert src not in succ, "chains may not branch outward"
        assert dst not in pred, "chains may not branch inward"
        succ[src] = dst
        pred[dst] = src

    heads = [idx for idx in range(len(squares)) if idx not in pred]
    assert heads, "each chain needs a head"
    visited: set[int] = set()
    for head in heads:
        cur = head
        while True:
            assert cur not in visited, "cycles are not allowed"
            visited.add(cur)
            if cur not in succ:
                break
            cur = succ[cur]
    assert len(visited) == len(squares), "every block must belong to a chain"
    return pred, succ


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each square block's color one step downstream along the grey arrow chains.

    Input structure:
    - Background is black.
    - Several separated solid square blocks of one uniform non-background color
      appear in the grid, and every block has the same size.
    - Grey 3-pixel arrowheads sit between horizontally or vertically aligned
      nearest block pairs; each arrowhead points toward one of the two blocks.
    - The blocks and arrows form one or more non-branching directed chains.

    Rule:
    - Decode every grey arrowhead to recover the color-transfer direction
      between its two neighboring square blocks.
    - For each chain, replace every block's color with the original color of
      its unique upstream predecessor.
    - Recolor each chain head, which has no upstream predecessor, to the black
      background instead.
    - Clear every grey arrow pixel to background; all transfers happen
      simultaneously from the original input colors.

    Key insight:
    - The arrows define a directed path graph over equal-sized square blocks,
      so the output is a synchronous one-step color shift along each chain
      rather than a sequential overwrite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    arrow_color = _infer_arrow_color(g, bg)
    squares = _extract_squares(g, bg, arrow_color)
    pred, _succ = _extract_graph(g, squares, arrow_color)

    out = np.full_like(g, bg)
    for idx, square in enumerate(squares):
        color = bg if idx not in pred else int(squares[pred[idx]]["color"])
        for r, c in square["coords"]:  # type: ignore[index]
            out[int(r), int(c)] = color
    return out


def _orth_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < LATTICE_SIZE and 0 <= cc < LATTICE_SIZE:
            out.append((rr, cc))
    return out


def _first_visible(
    cell: tuple[int, int],
    dr: int,
    dc: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    rr = cell[0] + dr
    cc = cell[1] + dc
    while 0 <= rr < LATTICE_SIZE and 0 <= cc < LATTICE_SIZE:
        if (rr, cc) in occupied:
            return (rr, cc)
        rr += dr
        cc += dc
    return None


def _can_start(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if cell in occupied:
        return False
    if any(nei in occupied for nei in _orth_neighbors(cell)):
        return False
    return all(_first_visible(cell, dr, dc, occupied) is None for dr, dc in DIR4)


def _can_extend(
    cell: tuple[int, int],
    prev: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    if cell in occupied:
        return False
    if any(nei in occupied and nei != prev for nei in _orth_neighbors(cell)):
        return False
    visible = {
        hit
        for dr, dc in DIR4
        if (hit := _first_visible(cell, dr, dc, occupied)) is not None
    }
    return visible <= {prev}


def _grow_from(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target_len: int,
) -> bool:
    if len(path) == target_len:
        return True
    current = path[-1]
    candidates = _orth_neighbors(current)
    random.shuffle(candidates)
    for nxt in candidates:
        if not _can_extend(nxt, current, occupied):
            continue
        path.append(nxt)
        occupied.add(nxt)
        if _grow_from(path, occupied, target_len):
            return True
        occupied.remove(nxt)
        path.pop()
    return False


def _build_chain(length: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(LATTICE_SIZE) for c in range(LATTICE_SIZE)]
    random.shuffle(starts)
    for start in starts:
        if not _can_start(start, occupied):
            continue
        trial = [start]
        local = set(occupied)
        local.add(start)
        if _grow_from(trial, local, length):
            return trial
    return None


def _edge_set(chains: list[list[tuple[int, int]]]) -> set[frozenset[tuple[int, int]]]:
    edges: set[frozenset[tuple[int, int]]] = set()
    for chain in chains:
        for i in range(len(chain) - 1):
            edges.add(frozenset((chain[i], chain[i + 1])))
    return edges


def _layout_is_valid(chains: list[list[tuple[int, int]]]) -> bool:
    occupied = {cell for chain in chains for cell in chain}
    edges = _edge_set(chains)
    visible: set[frozenset[tuple[int, int]]] = set()
    for cell in occupied:
        deg = 0
        for nei in _orth_neighbors(cell):
            if frozenset((cell, nei)) in edges:
                deg += 1
        if deg > 2:
            return False
        for dr, dc in DIR4:
            hit = _first_visible(cell, dr, dc, occupied)
            if hit is not None:
                visible.add(frozenset((cell, hit)))
    return visible == edges


def _sample_chain_lengths(chain_count: int) -> list[int]:
    if chain_count == 1:
        return [random.randint(2, 5)]
    if chain_count == 2:
        options = [
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3],
            [2, 4],
            [4, 2],
            [3, 4],
            [4, 3],
        ]
        return random.choice(options)
    return random.choice([[2, 2, 2], [2, 2, 3], [2, 3, 2], [3, 2, 2]])


def _generate_chain_layout(chain_count: int) -> list[list[tuple[int, int]]] | None:
    lengths = _sample_chain_lengths(chain_count)
    chains: list[list[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    for length in lengths:
        chain = _build_chain(length, occupied)
        if chain is None:
            return None
        chains.append(chain)
        occupied.update(chain)
    if not _layout_is_valid(chains):
        return None
    for chain in chains:
        if random.random() < 0.5:
            chain.reverse()
    return chains


def _sample_block_palette(count: int) -> list[int]:
    if random.random() < 0.25 and count >= 3:
        while True:
            colors = [random.choice(BLOCK_COLORS) for _ in range(count)]
            if len(set(colors)) >= 2:
                return colors
    return random.sample(BLOCK_COLORS, count)


def _arrow_coords(
    src: tuple[int, int],
    dst: tuple[int, int],
    side: int,
    src_offsets: dict[tuple[tuple[int, int], tuple[int, int]], int],
) -> list[tuple[int, int]]:
    sr, sc = src
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    offset = src_offsets[(src, dst)]
    if dr == 0 and dc > 0:
        col = sc + side
        row = sr + offset
        return [(row, col), (row + 1, col + 1), (row + 2, col)]
    if dr == 0 and dc < 0:
        col = sc - 2
        row = sr + offset
        return [(row, col + 1), (row + 1, col), (row + 2, col + 1)]
    if dr > 0 and dc == 0:
        row = sr + side
        col = sc + offset
        return [(row, col), (row, col + 2), (row + 1, col + 1)]
    row = sr - 2
    col = sc + offset
    return [(row, col + 1), (row + 1, col), (row + 1, col + 2)]


def _render_layout(chains: list[list[tuple[int, int]]], side: int) -> np.ndarray | None:
    step = side + 3
    all_cells = [cell for chain in chains for cell in chain]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    norm = [[(r - min_r, c - min_c) for r, c in chain] for chain in chains]

    max_r = max(r for chain in norm for r, _ in chain)
    max_c = max(c for chain in norm for _, c in chain)
    needed_h = max_r * step + side
    needed_w = max_c * step + side
    if needed_h > GRID_LIMIT - 2 or needed_w > GRID_LIMIT - 2:
        return None

    extra_h = random.randint(2, min(8, GRID_LIMIT - needed_h))
    extra_w = random.randint(2, min(8, GRID_LIMIT - needed_w))
    top_pad = random.randint(1, extra_h - 1)
    left_pad = random.randint(1, extra_w - 1)
    bottom_pad = extra_h - top_pad
    right_pad = extra_w - left_pad

    grid = np.full((needed_h + top_pad + bottom_pad, needed_w + left_pad + right_pad), BLACK, dtype=int)
    phys_chains: list[list[tuple[int, int]]] = []
    for chain in norm:
        phys_chain = []
        for r, c in chain:
            phys_chain.append((top_pad + r * step, left_pad + c * step))
        phys_chains.append(phys_chain)

    colors = _sample_block_palette(sum(len(chain) for chain in phys_chains))
    color_idx = 0
    for chain in phys_chains:
        for r, c in chain:
            grid[r : r + side, c : c + side] = colors[color_idx]
            color_idx += 1

    src_offsets: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for chain in phys_chains:
        for i in range(len(chain) - 1):
            src = chain[i]
            dst = chain[i + 1]
            src_offsets[(src, dst)] = random.randint(0, side - 3)

    for chain in phys_chains:
        for i in range(len(chain) - 1):
            for r, c in _arrow_coords(chain[i], chain[i + 1], side, src_offsets):
                grid[r, c] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    chain_count = random.choice((1, 2, 3))
    side = 3 if chain_count == 3 else random.choice((3, 4))
    chains = _generate_chain_layout(chain_count)
    if chains is None:
        return None
    grid = _render_layout(chains, side)
    if grid is None:
        return None
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and max(grid.shape) <= GRID_LIMIT):
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.any(out == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-transfer puzzle")
