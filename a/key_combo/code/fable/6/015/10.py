from __future__ import annotations

import random
from functools import reduce
from math import gcd
from typing import Iterable

import numpy as np

from arc_utils import find_components, get_bg


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return out


def _solid_square_component(grid: np.ndarray, comp: dict, bg: int) -> tuple[bool, int, int] | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 2 or int(comp["area"]) != h * w:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(sub[0, 0])
    if color == int(bg) or not np.all(sub == color):
        return None
    return True, h, color


def _infer_blocks_and_pieces(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], list[np.ndarray], int, int]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False)

    block_candidates: list[tuple[int, dict, int, int]] = []
    for idx, comp in enumerate(comps):
        info = _solid_square_component(grid, comp, bg)
        if info is None:
            continue
        _, size, color = info
        block_candidates.append((idx, comp, size, color))
    assert block_candidates, "expected square main blocks"

    sizes = [size for _, _, size, _ in block_candidates]
    vals, cnts = np.unique(np.asarray(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])

    block_indices = {idx for idx, _, size, _ in block_candidates if int(size) == int(block_size)}
    blocks: list[tuple[int, int, int]] = []
    pieces: list[np.ndarray] = []

    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
        if idx in block_indices:
            blocks.append((int(r0), int(c0), int(sub[0, 0])))
        else:
            pieces.append(sub)

    assert pieces, "expected scattered fragments"
    assert len(blocks) >= 3, "existing blocks must disambiguate the minimap"

    rows = sorted({r for r, _, _ in blocks})
    cols = sorted({c for _, c, _ in blocks})
    diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    diffs += [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    diffs = [d for d in diffs if d > 0]
    assert diffs, "block lattice must expose a positive step"
    step = reduce(gcd, diffs)
    assert step > block_size, "block lattice step must exceed block size"
    return blocks, pieces, block_size, int(step)


def _piece_orientations(piece: np.ndarray, bg: int) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for flip in (False, True):
        base = np.fliplr(piece) if flip else piece
        for k in range(4):
            arr = np.rot90(base, k)
            key = (arr.shape, arr.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(np.asarray(arr, dtype=int).copy())
    return out


def _enumerate_assemblies(piece_arrays: list[np.ndarray], bg: int, limit: int = 256) -> list[np.ndarray]:
    total_area = sum(int(np.count_nonzero(piece != bg)) for piece in piece_arrays)
    results: list[np.ndarray] = []
    seen_results: set[tuple[tuple[int, int], bytes]] = set()

    for H, W in _factor_pairs(total_area):
        max_h = max(max(ori.shape[0] for ori in _piece_orientations(piece, bg)) for piece in piece_arrays)
        max_w = max(max(ori.shape[1] for ori in _piece_orientations(piece, bg)) for piece in piece_arrays)
        if H < max_h or W < max_w:
            continue

        placements: list[list[tuple[int, list[tuple[int, int, int]], int]]] = []
        for piece in piece_arrays:
            opts: list[tuple[int, list[tuple[int, int, int]], int]] = []
            seen_place: set[tuple[int, tuple[tuple[int, int, int], ...]]] = set()
            for ori in _piece_orientations(piece, bg):
                hh, ww = ori.shape
                occ = np.argwhere(ori != bg)
                cells_local = [(int(r), int(c), int(ori[r, c])) for r, c in occ]
                for r0 in range(H - hh + 1):
                    for c0 in range(W - ww + 1):
                        mask = 0
                        cells_abs: list[tuple[int, int, int]] = []
                        for r, c, color in cells_local:
                            rr = int(r0 + r)
                            cc = int(c0 + c)
                            mask |= 1 << int(rr * W + cc)
                            cells_abs.append((rr, cc, color))
                        key = (mask, tuple(cells_abs))
                        if key in seen_place:
                            continue
                        seen_place.add(key)
                        opts.append((mask, cells_abs, int(len(cells_abs))))
            placements.append(opts)

        full_mask = (1 << int(H * W)) - 1
        remaining = set(range(len(piece_arrays)))
        canvas = np.full((H, W), int(bg), dtype=int)

        def search(used_mask: int) -> None:
            if len(results) >= limit:
                return
            if not remaining:
                if used_mask == full_mask:
                    key = (canvas.shape, canvas.tobytes())
                    if key not in seen_results:
                        seen_results.add(key)
                        results.append(canvas.copy())
                return

            empty_mask = (~used_mask) & full_mask
            if empty_mask == 0:
                return
            cell = (empty_mask & -empty_mask).bit_length() - 1

            candidate_groups: list[tuple[int, int, list[tuple[int, list[tuple[int, int, int]], int]]]] = []
            for pi in sorted(remaining):
                usable = [pl for pl in placements[pi] if ((pl[0] >> cell) & 1) and (pl[0] & used_mask) == 0]
                if usable:
                    candidate_groups.append((len(usable), pi, usable))
            if not candidate_groups:
                return

            candidate_groups.sort(key=lambda item: (item[0], item[1]))
            for _, pi, usable in candidate_groups:
                remaining.remove(pi)
                for mask, cells_abs, _ in usable:
                    for r, c, color in cells_abs:
                        canvas[r, c] = color
                    search(used_mask | mask)
                    for r, c, _ in cells_abs:
                        canvas[r, c] = int(bg)
                    if len(results) >= limit:
                        remaining.add(pi)
                        return
                remaining.add(pi)

        search(0)
    assert results, "failed to assemble fragment rectangle"
    return results


def _valid_base(
    minimap: np.ndarray,
    blocks: Iterable[tuple[int, int, int]],
    step: int,
    block_size: int,
    shape: tuple[int, int],
    base: tuple[int, int],
) -> bool:
    H, W = shape
    rows, cols = minimap.shape
    base_r, base_c = base
    for rr in range(rows):
        for cc in range(cols):
            r0 = int(base_r + rr * step)
            c0 = int(base_c + cc * step)
            if not (0 <= r0 <= H - block_size and 0 <= c0 <= W - block_size):
                return False
    for r0, c0, color in blocks:
        dr = int(r0 - base_r)
        dc = int(c0 - base_c)
        if dr % step != 0 or dc % step != 0:
            return False
        rr = dr // step
        cc = dc // step
        if not (0 <= rr < rows and 0 <= cc < cols):
            return False
        if int(minimap[rr, cc]) != int(color):
            return False
    return True


def _choose_minimap_and_base(
    candidates: list[np.ndarray],
    blocks: list[tuple[int, int, int]],
    step: int,
    block_size: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    fits: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    for idx, cand in enumerate(candidates):
        bases: dict[tuple[int, int], int] = {}
        rows, cols = cand.shape
        for rr in range(rows):
            for cc in range(cols):
                color = int(cand[rr, cc])
                for r0, c0, bcol in blocks:
                    if color != int(bcol):
                        continue
                    base = (int(r0 - rr * step), int(c0 - cc * step))
                    bases[base] = bases.get(base, 0) + 1
        for base, votes in bases.items():
            if _valid_base(cand, blocks, step, block_size, shape, base):
                fits.append((int(votes), idx, cand, base))

    assert fits, "no assembled minimap matches the existing blocks"
    fits.sort(key=lambda item: (-item[0], item[1], item[3][0], item[3][1]))
    votes, _, cand, base = fits[0]
    top_count = sum(1 for item in fits if item[0] == votes)
    assert top_count == 1, "existing blocks must uniquely resolve the minimap orientation"
    return cand, base


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered fragments into a rectangular minimap, then use that
    minimap to fill the missing large color blocks in the main lattice.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = get_bg(grid)
    blocks, pieces, block_size, step = _infer_blocks_and_pieces(grid, bg)
    candidates = _enumerate_assemblies(pieces, bg)
    minimap, base = _choose_minimap_and_base(candidates, blocks, step, block_size, grid.shape)

    out = grid.copy()
    base_r, base_c = base
    for rr in range(minimap.shape[0]):
        for cc in range(minimap.shape[1]):
            color = int(minimap[rr, cc])
            r0 = int(base_r + rr * step)
            c0 = int(base_c + cc * step)
            out[r0 : r0 + block_size, c0 : c0 + block_size] = color
    return out


# --------------- generator (o5x) ---------------

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_TRIES = 500


def _random_minimap() -> np.ndarray:
    rows = random.randint(2, 4)
    cols = random.randint(3, 4)
    palette = random.sample(PALETTE, random.randint(3, min(5, rows * cols)))
    arr = np.array([[random.choice(palette) for _ in range(cols)] for _ in range(rows)], dtype=int)
    if len(set(map(int, arr.ravel()))) < 3:
        return _random_minimap()
    return arr


def _random_partition(rows: int, cols: int, pieces_n: int) -> list[list[tuple[int, int]]]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = random.sample(cells, pieces_n)
    owner = {seed: idx for idx, seed in enumerate(seeds)}
    parts = [[seed] for seed in seeds]
    remaining = set(cells) - set(seeds)

    while remaining:
        frontier: list[tuple[tuple[int, int], int]] = []
        for r, c in sorted(remaining):
            adj = set()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in owner:
                    adj.add(owner[nb])
            if adj:
                frontier.append(((r, c), random.choice(sorted(adj))))
        if not frontier:
            raise RuntimeError("partition frontier vanished")
        cell, idx = random.choice(frontier)
        owner[cell] = idx
        parts[idx].append(cell)
        remaining.remove(cell)
    return [sorted(part) for part in parts]


def _piece_from_cells(minimap: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in cells:
        out[r - r0, c - c0] = int(minimap[r, c])
    return out


def _random_piece_set(minimap: np.ndarray) -> list[np.ndarray]:
    rows, cols = minimap.shape
    total = rows * cols
    for _ in range(200):
        pieces_n = random.randint(2, min(4, total // 2))
        cells_parts = _random_partition(rows, cols, pieces_n)
        if min(len(part) for part in cells_parts) < 2:
            continue
        pieces = [_piece_from_cells(minimap, part) for part in cells_parts]
        if any(len(set(map(int, piece[piece != BG].ravel()))) < 2 for piece in pieces):
            continue
        return pieces
    raise RuntimeError("failed to partition minimap into multicolor fragments")


def _random_transform(piece: np.ndarray) -> np.ndarray:
    arr = np.asarray(piece, dtype=int)
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    k = random.randint(0, 3)
    return np.rot90(arr, k).copy()


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_TRIES):
        minimap = _random_minimap()
        rows, cols = minimap.shape
        pieces = _random_piece_set(minimap)
        pieces = [_random_transform(piece) for piece in pieces]

        block_size = random.randint(2, 4)
        gap = random.randint(1, 2)
        step = block_size + gap
        main_h = (rows - 1) * step + block_size
        main_w = (cols - 1) * step + block_size

        visible: list[tuple[int, int]] = []
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(cells)
        keep_n = random.randint(max(3, (rows * cols) // 2), rows * cols - 1)
        visible = cells[:keep_n]
        if len({r for r, _ in visible}) < 2 or len({c for _, c in visible}) < 2:
            continue

        stack_h = sum(piece.shape[0] for piece in pieces) + max(0, len(pieces) - 1) * 2
        frag_w = max(piece.shape[1] for piece in pieces)

        H = max(main_h + 2, stack_h + 2)
        W = main_w + 4 + frag_w + 1
        if H > 30 or W > 30:
            continue

        grid = np.full((H, W), BG, dtype=int)
        expected = np.full((H, W), BG, dtype=int)

        base_r = 1
        base_c = 1
        for rr in range(rows):
            for cc in range(cols):
                color = int(minimap[rr, cc])
                r0 = int(base_r + rr * step)
                c0 = int(base_c + cc * step)
                expected[r0 : r0 + block_size, c0 : c0 + block_size] = color
                if (rr, cc) in visible:
                    grid[r0 : r0 + block_size, c0 : c0 + block_size] = color

        frag_c0 = main_w + 4
        cur_r = 1
        for piece in pieces:
            occ = np.argwhere(piece != BG)
            for r, c in occ:
                grid[cur_r + r, frag_c0 + c] = int(piece[r, c])
                expected[cur_r + r, frag_c0 + c] = int(piece[r, c])
            cur_r += int(piece.shape[0]) + 2

        try:
            got = main(grid)
        except AssertionError:
            continue
        if np.array_equal(got, expected):
            return grid, expected
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _build_instance()[0]
