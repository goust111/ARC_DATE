from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PALETTE = (3, 4, 5, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _grid_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == BLUE)]
    cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == BLUE)]
    assert len(rows) >= 3 and len(cols) >= 3
    row_steps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    col_steps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1
    assert row_steps[0] == col_steps[0]
    return rows, cols


def _extract_cells(g: np.ndarray) -> tuple[list[int], list[int], int, dict[tuple[int, int], np.ndarray]]:
    rows, cols = _grid_lines(g)
    cell_size = rows[1] - rows[0] - 1
    assert cell_size >= 3 and cell_size % 2 == 1

    patches: dict[tuple[int, int], np.ndarray] = {}
    for rr in range(len(rows) - 1):
        for cc in range(len(cols) - 1):
            r0, r1 = rows[rr] + 1, rows[rr + 1]
            c0, c1 = cols[cc] + 1, cols[cc + 1]
            patch = g[r0:r1, c0:c1].copy()
            patch[(patch == BLUE) | (patch == RED)] = BLACK
            patches[(rr, cc)] = patch
    return rows, cols, cell_size, patches


def _arrow_target(
    g: np.ndarray,
    rows: list[int],
    cols: list[int],
    cell_size: int,
    cell: tuple[int, int],
) -> tuple[int, int] | None:
    rr, cc = cell
    mid = cell_size // 2
    center_r = rows[rr] + 1 + mid
    center_c = cols[cc] + 1 + mid
    candidates: list[tuple[int, int]] = []

    if rr > 0 and g[rows[rr] - 1, center_c] == RED:
        candidates.append((rr - 1, cc))
    if rr + 1 < len(rows) - 1 and g[rows[rr + 1] + 1, center_c] == RED:
        candidates.append((rr + 1, cc))
    if cc > 0 and g[center_r, cols[cc] - 1] == RED:
        candidates.append((rr, cc - 1))
    if cc + 1 < len(cols) - 1 and g[center_r, cols[cc + 1] + 1] == RED:
        candidates.append((rr, cc + 1))

    assert len(candidates) <= 1
    return candidates[0] if candidates else None


def _overlay_earlier(base: np.ndarray, add: np.ndarray) -> np.ndarray:
    out = base.copy()
    mask = (out == BLACK) & (add != BLACK)
    out[mask] = add[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rows, cols, cell_size, patches = _extract_cells(g)
    occupied = {
        cell
        for cell, patch in patches.items()
        if np.any(patch != BLACK)
    }
    assert occupied

    outgoing: dict[tuple[int, int], tuple[int, int]] = {}
    indeg = {cell: 0 for cell in occupied}
    for cell in occupied:
        dst = _arrow_target(g, rows, cols, cell_size, cell)
        if dst is None:
            continue
        assert dst in occupied
        outgoing[cell] = dst
        indeg[dst] += 1

    assert all(v <= 1 for v in indeg.values())

    out = g.copy()
    seen: set[tuple[int, int]] = set()
    starts = [cell for cell in occupied if indeg[cell] == 0]
    assert starts

    for start in starts:
        if start in seen:
            continue
        chain: list[tuple[int, int]] = []
        cur = start
        local_seen: set[tuple[int, int]] = set()
        while True:
            assert cur not in local_seen
            local_seen.add(cur)
            seen.add(cur)
            chain.append(cur)
            if cur not in outgoing:
                break
            cur = outgoing[cur]

        cumulative = np.full((cell_size, cell_size), BLACK, dtype=int)
        for idx, cell in enumerate(chain):
            patch = patches[cell]
            if idx == 0:
                cumulative = patch.copy()
            else:
                cumulative = _overlay_earlier(cumulative, patch)

            rr, cc = cell
            r0, r1 = rows[rr] + 1, rows[rr + 1]
            c0, c1 = cols[cc] + 1, cols[cc + 1]
            region = out[r0:r1, c0:c1]
            keep_red = region == RED
            region[~keep_red] = BLACK
            region[cumulative != BLACK] = cumulative[cumulative != BLACK]

    assert seen == occupied
    return out


def _shape_templates(side: int) -> list[set[tuple[int, int]]]:
    m = side // 2
    return [
        {(m, m)},
        {(m - 1, m), (m, m), (m + 1, m)},
        {(m, m - 1), (m, m), (m, m + 1)},
        {(m - 1, m), (m, m - 1), (m, m), (m, m + 1), (m + 1, m)},
        {(m - 1, m - 1), (m - 1, m), (m, m - 1), (m, m)},
        {(m - 1, m - 1), (m, m), (m + 1, m + 1)},
        {(m - 1, m + 1), (m, m), (m + 1, m - 1)},
        {(m - 1, m - 1), (m, m - 1), (m + 1, m - 1), (m + 1, m), (m + 1, m + 1)},
    ]


def _random_patch(cell_size: int, color: int) -> np.ndarray:
    patch = np.full((cell_size, cell_size), BLACK, dtype=int)
    inner = cell_size - 2
    template = random.choice(_shape_templates(inner))
    for r, c in template:
        patch[r + 1, c + 1] = color
    if inner >= 5 and random.random() < 0.35:
        extra = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        rr = cell_size // 2 + extra[0]
        cc = cell_size // 2 + extra[1]
        patch[rr, cc] = color
    return patch


def _draw_grid(board_h: int, board_w: int, cell_size: int) -> np.ndarray:
    h = board_h * cell_size + board_h + 1
    w = board_w * cell_size + board_w + 1
    g = np.full((h, w), BLACK, dtype=int)
    for r in range(0, h, cell_size + 1):
        g[r, :] = BLUE
    for c in range(0, w, cell_size + 1):
        g[:, c] = BLUE
    return g


def _cell_origin(cell: tuple[int, int], cell_size: int) -> tuple[int, int]:
    rr, cc = cell
    return rr * (cell_size + 1) + 1, cc * (cell_size + 1) + 1


def _place_patch(g: np.ndarray, cell: tuple[int, int], patch: np.ndarray, cell_size: int) -> None:
    r0, c0 = _cell_origin(cell, cell_size)
    region = g[r0 : r0 + cell_size, c0 : c0 + cell_size]
    region[patch != BLACK] = patch[patch != BLACK]


def _place_arrow(g: np.ndarray, src: tuple[int, int], dst: tuple[int, int], cell_size: int) -> None:
    sr, sc = src
    dr, dc = dst
    r0, c0 = _cell_origin(src, cell_size)
    mid = cell_size // 2
    if dr == sr - 1:
        g[r0 - 2, c0 + mid] = RED
    elif dr == sr + 1:
        g[r0 + cell_size + 1, c0 + mid] = RED
    elif dc == sc - 1:
        g[r0 + mid, c0 - 2] = RED
    elif dc == sc + 1:
        g[r0 + mid, c0 + cell_size + 1] = RED
    else:
        raise AssertionError("arrow endpoints must be adjacent")


def _neighbors(cell: tuple[int, int], board_h: int, board_w: int) -> list[tuple[int, int]]:
    rr, cc = cell
    out = []
    for dr, dc in DIRS.values():
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < board_h and 0 <= nc < board_w:
            out.append((nr, nc))
    random.shuffle(out)
    return out


def _random_path(
    free_cells: set[tuple[int, int]],
    board_h: int,
    board_w: int,
    length: int,
) -> list[tuple[int, int]] | None:
    starts = list(free_cells)
    random.shuffle(starts)
    for start in starts:
        path = [start]
        used = {start}
        while len(path) < length:
            choices = [
                nb
                for nb in _neighbors(path[-1], board_h, board_w)
                if nb in free_cells and nb not in used
            ]
            if not choices:
                break
            nxt = random.choice(choices)
            path.append(nxt)
            used.add(nxt)
        if len(path) == length:
            return path
    return None


def _split_lengths(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1)) if parts > 1 else []
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    random.shuffle(vals)
    return vals


def _gen_attempt() -> np.ndarray | None:
    board_h = random.randint(2, 4)
    board_w = random.randint(2, 4)
    cell_size = random.choice([5, 7])
    total_cells = board_h * board_w
    active_count = random.randint(max(2, total_cells // 2), total_cells)

    max_chains = min(active_count, 4)
    chain_count = random.randint(1, max_chains)
    lengths = _split_lengths(active_count, chain_count)
    if all(length == 1 for length in lengths) and active_count >= 2:
        lengths[0] += 1
        lengths[-1] -= 1
        if lengths[-1] == 0:
            return None

    free_cells = {(r, c) for r in range(board_h) for c in range(board_w)}
    chains: list[list[tuple[int, int]]] = []
    for length in lengths:
        path = _random_path(free_cells, board_h, board_w, length)
        if path is None:
            return None
        chains.append(path)
        for cell in path:
            free_cells.remove(cell)

    g = _draw_grid(board_h, board_w, cell_size)
    all_cells = {cell for chain in chains for cell in chain}

    for chain in chains:
        colors = random.sample(PALETTE, k=min(len(PALETTE), len(chain)))
        if len(colors) < len(chain):
            colors = [random.choice(PALETTE) for _ in chain]
        shared_base = random.random() < 0.7
        base_patch = None
        if shared_base:
            base_patch = _random_patch(cell_size, colors[0])

        for idx, cell in enumerate(chain):
            color = colors[idx] if idx < len(colors) else random.choice(PALETTE)
            patch = _random_patch(cell_size, color)
            if shared_base and base_patch is not None and idx > 0:
                mask = base_patch != BLACK
                patch[mask] = color
            _place_patch(g, cell, patch, cell_size)

        for src, dst in zip(chain, chain[1:]):
            _place_arrow(g, src, dst, cell_size)

    if len(all_cells) != active_count:
        return None

    try:
        out = main(g)
    except Exception:
        return None

    if out.shape != g.shape:
        return None
    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
