from __future__ import annotations

import random

import numpy as np


_BG = 0
_ARROW = 5
_BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MAX_GENERATION_ATTEMPTS = 400
_CHAIN_COUNT_RANGE = (1, 3)
_CHAIN_LENGTH_RANGE = (1, 4)
_GRID_SIDE_RANGE = (18, 26)
_BLOCK_HEIGHT_RANGE = (2, 4)
_BLOCK_WIDTH_RANGE = (2, 4)
_GAP_RANGE = (3, 6)
_EXTEND_RETRIES = 120
_INITIAL_BLOCK_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or bool(seen[r0, c0]):
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in _DIRS:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1 = min(rows)
            r2 = max(rows)
            c1 = min(cols)
            c2 = max(cols)
            components.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r1), int(c1), int(r2), int(c2)),
                    "area": int(len(coords)),
                }
            )

    return components


def _is_solid_rect(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _point_in_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(r0 <= r <= r1 and c0 <= c <= c1)


def _parse_arrow(component: dict) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in component["coords"]}
    candidates: list[tuple[int, int, int, int]] = []

    for r, c in pts:
        for dr, dc in _DIRS:
            br = int(r - dr)
            bc = int(c - dc)
            pr1 = int(-dc)
            pc1 = int(dr)
            pr2 = int(dc)
            pc2 = int(-dr)
            required = {
                (br, bc),
                (br + pr1, bc + pc1),
                (br + pr2, bc + pc2),
            }
            if not required <= pts:
                continue
            if (r + dr, c + dc) in pts:
                continue
            candidates.append((int(r), int(c), int(dr), int(dc)))

    assert len(candidates) == 1, f"expected one arrow tip, got {candidates}"
    tip_r, tip_c, dr, dc = candidates[0]

    tail_r = int(tip_r - dr)
    tail_c = int(tip_c - dc)
    while (tail_r - dr, tail_c - dc) in pts:
        tail_r -= dr
        tail_c -= dc

    base_r = int(tip_r - dr)
    base_c = int(tip_c - dc)
    allowed = {(int(tip_r), int(tip_c))}
    cur_r, cur_c = int(tail_r), int(tail_c)
    while True:
        allowed.add((int(cur_r), int(cur_c)))
        if (cur_r, cur_c) == (base_r, base_c):
            break
        cur_r += dr
        cur_c += dc

    allowed.add((int(base_r - dc), int(base_c + dr)))
    allowed.add((int(base_r + dc), int(base_c - dr)))
    assert allowed == pts, f"arrow has unexpected pixels: {sorted(pts - allowed)}"

    return (int(tail_r), int(tail_c)), (int(tip_r), int(tip_c)), (int(dr), int(dc))


def _trace_block(
    grid: np.ndarray,
    block_at: np.ndarray,
    bg: int,
    arrow_color: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
) -> int:
    h, w = grid.shape
    r = int(start_r)
    c = int(start_c)
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value != bg and value != arrow_color:
            block_idx = int(block_at[r, c])
            assert block_idx >= 0
            return block_idx
        r += dr
        c += dc
    raise AssertionError("arrow does not terminate at a block")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the grey arrow chains between solid colored rectangles.

    Each arrow defines a directed edge from one rectangle to another. Along every
    chain, the head rectangle's color propagates through all downstream rectangles.
    All grey arrow pixels are then removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    components = _find_components(g, bg)
    if not components:
        return g.copy()

    blocks = [comp for comp in components if _is_solid_rect(comp)]
    arrow_components = [comp for comp in components if not _is_solid_rect(comp)]

    if not arrow_components:
        return g.copy()

    arrow_colors = {int(comp["color"]) for comp in arrow_components}
    assert len(arrow_colors) == 1, f"expected one arrow color, got {arrow_colors}"
    arrow_color = int(next(iter(arrow_colors)))
    blocks = [comp for comp in blocks if int(comp["color"]) != arrow_color]
    assert blocks, "expected at least one colored block"

    block_at = np.full(g.shape, -1, dtype=int)
    for idx, block in enumerate(blocks):
        r0, c0, r1, c1 = block["bbox"]
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(block["color"]))
        assert np.all(block_at[r0 : r1 + 1, c0 : c1 + 1] == -1)
        block_at[r0 : r1 + 1, c0 : c1 + 1] = int(idx)

    incoming = [-1] * len(blocks)
    outgoing = [-1] * len(blocks)
    arrow_mask = np.zeros_like(g, dtype=bool)

    for comp in arrow_components:
        assert int(comp["color"]) == arrow_color
        (tail_r, tail_c), (tip_r, tip_c), (dr, dc) = _parse_arrow(comp)
        src = _trace_block(g, block_at, bg, arrow_color, tail_r - dr, tail_c - dc, -dr, -dc)
        dst = _trace_block(g, block_at, bg, arrow_color, tip_r + dr, tip_c + dc, dr, dc)
        assert src != dst
        assert outgoing[src] == -1, "a block has multiple outgoing arrows"
        assert incoming[dst] == -1, "a block has multiple incoming arrows"
        outgoing[src] = int(dst)
        incoming[dst] = int(src)
        coords = comp["coords"]
        arrow_mask[coords[:, 0], coords[:, 1]] = True

    out = g.copy()
    out[arrow_mask] = bg

    for idx, block in enumerate(blocks):
        head = int(idx)
        seen: set[int] = set()
        while incoming[head] != -1:
            assert head not in seen, "arrow graph must be acyclic"
            seen.add(head)
            head = int(incoming[head])
        color = int(blocks[head]["color"])
        coords = block["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _random_block_size() -> tuple[int, int]:
    return (
        random.randint(*_BLOCK_HEIGHT_RANGE),
        random.randint(*_BLOCK_WIDTH_RANGE),
    )


def _box_clear(grid: np.ndarray, box: tuple[int, int, int, int], pad: int) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(h - 1, int(r1 + pad))
    cc1 = min(w - 1, int(c1 + pad))
    return bool(np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == _BG))


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _arrow_cells(
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    dr: int,
    dc: int,
    line: int,
) -> list[tuple[int, int]]:
    sr0, sc0, sr1, sc1 = source_box
    tr0, tc0, tr1, tc1 = target_box

    if (dr, dc) == (0, 1):
        assert sr0 <= line <= sr1 and tr0 <= line <= tr1
        base_c = int(tc0 - 2)
        cells = [(int(line), int(c)) for c in range(int(sc1 + 1), int(tc0 - 1))]
        cells.append((int(line), int(tc0 - 1)))
        cells.append((int(line - 1), int(base_c)))
        cells.append((int(line + 1), int(base_c)))
        return cells

    if (dr, dc) == (0, -1):
        assert sr0 <= line <= sr1 and tr0 <= line <= tr1
        base_c = int(tc1 + 2)
        cells = [(int(line), int(c)) for c in range(int(tc1 + 2), int(sc0))]
        cells.append((int(line), int(tc1 + 1)))
        cells.append((int(line - 1), int(base_c)))
        cells.append((int(line + 1), int(base_c)))
        return cells

    if (dr, dc) == (1, 0):
        assert sc0 <= line <= sc1 and tc0 <= line <= tc1
        base_r = int(tr0 - 2)
        cells = [(int(r), int(line)) for r in range(int(sr1 + 1), int(tr0 - 1))]
        cells.append((int(tr0 - 1), int(line)))
        cells.append((int(base_r), int(line - 1)))
        cells.append((int(base_r), int(line + 1)))
        return cells

    assert (dr, dc) == (-1, 0)
    assert sc0 <= line <= sc1 and tc0 <= line <= tc1
    base_r = int(tr1 + 2)
    cells = [(int(r), int(line)) for r in range(int(tr1 + 2), int(sr0))]
    cells.append((int(tr1 + 1), int(line)))
    cells.append((int(base_r), int(line - 1)))
    cells.append((int(base_r), int(line + 1)))
    return cells


def _cells_in_bounds(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _arrow_has_clean_neighbors(
    grid: np.ndarray,
    source_box: tuple[int, int, int, int],
    cells: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if int(grid[r, c]) != _BG:
            return False
        for dr, dc in _DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(grid[nr, nc]) == _BG:
                continue
            if not _point_in_box(nr, nc, source_box):
                return False
    return True


def _place_initial_block(grid: np.ndarray, color: int) -> dict | None:
    h, w = grid.shape
    for _ in range(_INITIAL_BLOCK_RETRIES):
        bh, bw = _random_block_size()
        if h - bh - 1 < 1 or w - bw - 1 < 1:
            return None
        r0 = random.randint(1, h - bh - 1)
        c0 = random.randint(1, w - bw - 1)
        box = (int(r0), int(c0), int(r0 + bh - 1), int(c0 + bw - 1))
        if not _box_clear(grid, box, pad=1):
            continue
        _paint_box(grid, box, color)
        return {"bbox": box, "color": int(color)}
    return None


def _extend_chain(grid: np.ndarray, source: dict, color: int) -> dict | None:
    source_box = source["bbox"]

    for _ in range(_EXTEND_RETRIES):
        dr, dc = random.choice(_DIRS)
        bh, bw = _random_block_size()
        gap = random.randint(*_GAP_RANGE)

        if dr == 0:
            line_r = random.randint(source_box[0], source_box[2])
            target_row_offset = random.randint(0, bh - 1)
            tr0 = int(line_r - target_row_offset)
            if dc == 1:
                tc0 = int(source_box[3] + gap)
            else:
                tc0 = int(source_box[1] - gap - bw + 1)
            target_box = (int(tr0), int(tc0), int(tr0 + bh - 1), int(tc0 + bw - 1))
            line = int(line_r)
        else:
            line_c = random.randint(source_box[1], source_box[3])
            target_col_offset = random.randint(0, bw - 1)
            tc0 = int(line_c - target_col_offset)
            if dr == 1:
                tr0 = int(source_box[2] + gap)
            else:
                tr0 = int(source_box[0] - gap - bh + 1)
            target_box = (int(tr0), int(tc0), int(tr0 + bh - 1), int(tc0 + bw - 1))
            line = int(line_c)

        if not _box_clear(grid, target_box, pad=1):
            continue

        arrow_cells = _arrow_cells(source_box, target_box, dr, dc, line)
        if len(set(arrow_cells)) != len(arrow_cells):
            continue
        if not _cells_in_bounds(arrow_cells, grid.shape):
            continue
        if not _arrow_has_clean_neighbors(grid, source_box, arrow_cells):
            continue

        _paint_box(grid, target_box, color)
        for r, c in arrow_cells:
            grid[r, c] = _ARROW
        return {"bbox": target_box, "color": int(color)}

    return None


def _choose_chain_lengths() -> list[int]:
    while True:
        chain_count = random.randint(*_CHAIN_COUNT_RANGE)
        lengths = [random.randint(*_CHAIN_LENGTH_RANGE) for _ in range(chain_count)]
        if sum(lengths) > len(_BLOCK_COLORS):
            continue
        if max(lengths) <= 1:
            continue
        return lengths


def _expected_output(grid: np.ndarray, chains: list[list[dict]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[out == _ARROW] = _BG
    for chain in chains:
        head_color = int(chain[0]["color"])
        for block in chain:
            _paint_box(out, block["bbox"], head_color)
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_SIDE_RANGE)
    w = random.randint(*_GRID_SIDE_RANGE)
    grid = np.full((h, w), _BG, dtype=int)

    lengths = _choose_chain_lengths()
    colors = list(_BLOCK_COLORS)
    random.shuffle(colors)
    color_iter = iter(colors)

    chains: list[list[dict]] = []
    for length in lengths:
        head = _place_initial_block(grid, next(color_iter))
        if head is None:
            return None

        chain = [head]
        current = head
        for _ in range(length - 1):
            nxt = _extend_chain(grid, current, next(color_iter))
            if nxt is None:
                return None
            chain.append(nxt)
            current = nxt
        chains.append(chain)

    expected = _expected_output(grid, chains)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-chain puzzle")
