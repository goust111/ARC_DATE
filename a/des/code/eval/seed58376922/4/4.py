import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
PIPE_COLORS = {GREY, YELLOW, ORANGE}
INPUT_COLORS = (1, 2, 3, 6, 8, 9)

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _neighbors(r: int, c: int, h: int, w: int, conn8: bool = False):
    dirs = DIR8 if conn8 else DIR4
    for dr, dc in dirs:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _mask_components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, h, w, conn8=conn8):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _same_color_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            col = int(g[r, c])
            if col == BLACK or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, h, w, conn8=False):
                    if not bool(seen[nr, nc]) and int(g[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": col, "cells": cells})
    return comps


def _find_square_blocks(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    blocks: list[dict] = []
    cell_to_block: dict[tuple[int, int], int] = {}
    for comp in _same_color_components(grid):
        cells = comp["cells"]
        if len(cells) != 4:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if max(rs) - min(rs) != 1 or max(cs) - min(cs) != 1:
            continue
        cell_set = set(cells)
        ok = True
        for rr in range(min(rs), max(rs) + 1):
            for cc in range(min(cs), max(cs) + 1):
                if (rr, cc) not in cell_set:
                    ok = False
        if not ok:
            continue
        idx = len(blocks)
        block = {"color": int(comp["color"]), "cells": cell_set}
        blocks.append(block)
        for cell in cell_set:
            cell_to_block[cell] = idx
    return blocks, cell_to_block


def _diagonal_touch(endpoint: tuple[int, int], block_cells: set[tuple[int, int]]) -> bool:
    r, c = endpoint
    return any(abs(br - r) == 1 and abs(bc - c) == 1 for br, bc in block_cells)


def _path_between(
    cells: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if goal not in prev:
        return None
    path: list[tuple[int, int]] = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each circuit's 2x2 output block based on whether its pipe path is broken.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    blocks, cell_to_block = _find_square_blocks(g)
    comps = _mask_components(g != BLACK, conn8=True)

    for comp in comps:
        comp_set = set(comp)
        comp_block_ids = {cell_to_block[cell] for cell in comp if cell in cell_to_block}
        if not comp_block_ids:
            continue

        input_blocks = [blocks[idx] for idx in comp_block_ids if int(blocks[idx]["color"]) not in PIPE_COLORS]
        output_blocks = [blocks[idx] for idx in comp_block_ids if int(blocks[idx]["color"]) == GREY]
        if len(input_blocks) != 1 or len(output_blocks) != 1:
            continue

        input_block = input_blocks[0]
        output_block = output_blocks[0]
        pipe_cells = {
            cell
            for cell in comp
            if int(g[cell]) in PIPE_COLORS and cell not in output_block["cells"]
        }
        if not pipe_cells:
            continue

        endpoints = []
        for cell in pipe_cells:
            r, c = cell
            deg = sum((r + dr, c + dc) in pipe_cells for dr, dc in DIR4)
            if deg <= 1:
                endpoints.append(cell)

        input_endpoints = [cell for cell in endpoints if _diagonal_touch(cell, input_block["cells"])]
        output_endpoints = [cell for cell in endpoints if _diagonal_touch(cell, output_block["cells"])]
        if len(input_endpoints) != 1 or len(output_endpoints) != 1:
            continue

        path = _path_between(pipe_cells, input_endpoints[0], output_endpoints[0])
        if path is None:
            continue

        broken = any(int(g[r, c]) == ORANGE for r, c in path)
        fill = BLACK if broken else int(input_block["color"])
        for r, c in output_block["cells"]:
            out[r, c] = fill

    return out


def _make_block(endpoint: tuple[int, int], corner_dr: int, corner_dc: int) -> set[tuple[int, int]]:
    r, c = endpoint
    r0 = r - 2 if corner_dr == -1 else r + 1
    c0 = c - 2 if corner_dc == -1 else c + 1
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _normalize_cells(cells: dict[tuple[int, int], int]):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c): col for (r, c), col in cells.items()}
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    return shifted, h, w


def _render_cells(cells: dict[tuple[int, int], int], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for (r, c), col in cells.items():
        grid[r, c] = int(col)
    return grid


def _shape_l(rng: random.Random):
    hdir = rng.choice(((0, 1), (0, -1)))
    vdir = rng.choice(((1, 0), (-1, 0)))
    hlen = rng.randint(3, 7)
    vlen = rng.randint(3, 7)
    j = (0, 0)
    pipe = {j}
    for k in range(1, hlen + 1):
        pipe.add((j[0] + hdir[0] * k, j[1] + hdir[1] * k))
    for k in range(1, vlen + 1):
        pipe.add((j[0] + vdir[0] * k, j[1] + vdir[1] * k))
    endpoints = [
        {"cell": (j[0] + hdir[0] * hlen, j[1] + hdir[1] * hlen), "dir": hdir},
        {"cell": (j[0] + vdir[0] * vlen, j[1] + vdir[1] * vlen), "dir": vdir},
    ]
    return pipe, endpoints, (0, 1), "L"


def _shape_t(rng: random.Random):
    if rng.random() < 0.5:
        stem_dir = rng.choice(((1, 0), (-1, 0)))
        cross_dir = rng.choice(((0, 1), (0, -1)))
    else:
        stem_dir = rng.choice(((0, 1), (0, -1)))
        cross_dir = rng.choice(((1, 0), (-1, 0)))
    stem_len = rng.randint(3, 7)
    arm_a = rng.randint(2, 6)
    arm_b = rng.randint(2, 6)
    j = (0, 0)
    pipe = {j}
    for k in range(1, stem_len + 1):
        pipe.add((j[0] + stem_dir[0] * k, j[1] + stem_dir[1] * k))
    for k in range(1, arm_a + 1):
        pipe.add((j[0] + cross_dir[0] * k, j[1] + cross_dir[1] * k))
    for k in range(1, arm_b + 1):
        pipe.add((j[0] - cross_dir[0] * k, j[1] - cross_dir[1] * k))
    endpoints = [
        {"cell": (j[0] + stem_dir[0] * stem_len, j[1] + stem_dir[1] * stem_len), "dir": stem_dir},
        {"cell": (j[0] + cross_dir[0] * arm_a, j[1] + cross_dir[1] * arm_a), "dir": cross_dir},
        {"cell": (j[0] - cross_dir[0] * arm_b, j[1] - cross_dir[1] * arm_b), "dir": (-cross_dir[0], -cross_dir[1])},
    ]
    used = rng.choice(((0, 1), (0, 2), (1, 2)))
    return pipe, endpoints, used, "T"


def _build_circuit(rng: random.Random, input_color: int, conductive: bool) -> dict | None:
    for _ in range(80):
        if rng.random() < 0.5:
            pipe_cells, endpoints, used_pair, kind = _shape_l(rng)
        else:
            pipe_cells, endpoints, used_pair, kind = _shape_t(rng)

        chosen = [endpoints[used_pair[0]], endpoints[used_pair[1]]]
        if rng.random() < 0.5:
            input_ep, output_ep = chosen
        else:
            output_ep, input_ep = chosen

        occupied = set(pipe_cells)
        placements = {}
        ok = True
        for name, endpoint in (("input", input_ep), ("output", output_ep)):
            signs = [-1, 1]
            rng.shuffle(signs)
            placed = None
            for sign in signs:
                dr, dc = endpoint["dir"]
                if dr == 0:
                    corner = (sign, dc)
                else:
                    corner = (dr, sign)
                block = _make_block(endpoint["cell"], corner[0], corner[1])
                if block & occupied:
                    continue
                placed = block
                break
            if placed is None:
                ok = False
                break
            placements[name] = placed
            occupied |= placed
        if not ok:
            continue

        route = _path_between(pipe_cells, input_ep["cell"], output_ep["cell"])
        if route is None:
            continue
        endpoints_set = {input_ep["cell"], output_ep["cell"]}
        internal = [cell for cell in route if cell not in endpoints_set]
        max_switches = min(2, len(internal))
        if conductive:
            switch_count = rng.randint(0, max_switches)
        else:
            if max_switches == 0:
                continue
            switch_count = rng.randint(1, max_switches)

        switch_cells = rng.sample(internal, switch_count) if switch_count else []
        switch_map: dict[tuple[int, int], int] = {}
        if conductive:
            for cell in switch_cells:
                switch_map[cell] = YELLOW
        else:
            orange_count = 1 if switch_count == 1 else rng.randint(1, switch_count)
            orange_cells = set(rng.sample(switch_cells, orange_count))
            for cell in switch_cells:
                switch_map[cell] = ORANGE if cell in orange_cells else YELLOW

        cells = {cell: GREY for cell in pipe_cells}
        cells.update(switch_map)
        for cell in placements["input"]:
            cells[cell] = input_color
        for cell in placements["output"]:
            cells[cell] = GREY

        shifted, h, w = _normalize_cells(cells)
        local_grid = _render_cells(shifted, h, w)
        if len(_mask_components(local_grid != BLACK, conn8=True)) != 1:
            continue

        output_cells = {
            (r - min(r0 for r0, _ in cells), c - min(c0 for _, c0 in cells))
            for r, c in placements["output"]
        }
        return {
            "grid": local_grid,
            "output_cells": output_cells,
            "fill_color": input_color if conductive else BLACK,
            "kind": kind,
        }
    return None


def _row_layout(rng: random.Random, circuits: list[dict]) -> tuple[np.ndarray, list[dict]]:
    n = len(circuits)
    order = list(range(n))
    rng.shuffle(order)

    if n == 2:
        row_count = rng.choice((1, 2))
    elif n == 3:
        row_count = rng.choice((1, 2))
    else:
        row_count = 2

    if row_count == 1:
        rows = [order]
    else:
        split = 1 if n == 2 else (rng.choice((1, 2)) if n == 3 else 2)
        rows = [order[:split], order[split:]]
        if rng.random() < 0.5:
            rows.reverse()

    row_heights = []
    row_widths = []
    row_gaps = []
    for row in rows:
        heights = [int(circuits[idx]["grid"].shape[0]) for idx in row]
        widths = [int(circuits[idx]["grid"].shape[1]) for idx in row]
        gaps = [rng.randint(3, 5) for _ in range(max(0, len(row) - 1))]
        row_heights.append(max(heights))
        row_widths.append(sum(widths) + sum(gaps))
        row_gaps.append(gaps)

    top = rng.randint(1, 4)
    bottom = rng.randint(1, 4)
    left = rng.randint(1, 4)
    right = rng.randint(1, 4)
    between_rows = [rng.randint(3, 5) for _ in range(max(0, len(rows) - 1))]
    h = top + bottom + sum(row_heights) + sum(between_rows)
    w = left + right + max(row_widths)

    grid = np.full((h, w), BLACK, dtype=int)
    metas: list[dict] = []

    cur_r = top
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        row_start_c = left + rng.randint(0, max(0, max(row_widths) - row_w))
        cur_c = row_start_c
        for pos, circuit_idx in enumerate(row):
            circuit = circuits[circuit_idx]
            patch = circuit["grid"]
            ph, pw = patch.shape
            off_r = cur_r + rng.randint(0, row_h - ph)
            nz = np.argwhere(patch != BLACK)
            for rr, cc in nz:
                grid[off_r + int(rr), cur_c + int(cc)] = int(patch[int(rr), int(cc)])
            metas.append(
                {
                    "output_cells": {(off_r + r, cur_c + c) for r, c in circuit["output_cells"]},
                    "fill_color": int(circuit["fill_color"]),
                }
            )
            cur_c += pw
            if pos < len(row_gaps[row_idx]):
                cur_c += row_gaps[row_idx][pos]
        cur_r += row_h
        if row_idx < len(between_rows):
            cur_r += between_rows[row_idx]
    return grid, metas


def _expected_from_meta(grid: np.ndarray, metas: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for meta in metas:
        for r, c in meta["output_cells"]:
            out[r, c] = int(meta["fill_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_circuits = rng.randint(2, 4)
    colors = rng.sample(INPUT_COLORS, num_circuits)

    conductive_count = num_circuits // 2
    states = [True] * conductive_count + [False] * conductive_count
    if len(states) < num_circuits:
        states.append(rng.choice((True, False)))
    rng.shuffle(states)

    circuits = []
    for idx in range(num_circuits):
        circuit = _build_circuit(rng, colors[idx], states[idx])
        if circuit is None:
            return None
        circuits.append(circuit)

    grid, metas = _row_layout(rng, circuits)
    expected = _expected_from_meta(grid, metas)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate a valid circuit puzzle")
