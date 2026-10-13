from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, find_components, get_bg
except ModuleNotFoundError:
    utils_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(utils_dir) not in sys.path:
        sys.path.append(str(utils_dir))
    from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
GREY = Color.GREY
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
MAX_GEN_ATTEMPTS = 600
PATH_SEARCH_BUDGET = 60000


def _neighbors4(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    h, w = shape
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _true_run_lengths(line: np.ndarray) -> list[int]:
    runs: list[int] = []
    start = None
    for idx, value in enumerate(line.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append(idx - start)
            start = None
    if start is not None:
        runs.append(len(line) - start)
    return runs


def _infer_block_size(mask: np.ndarray) -> int:
    lengths: list[int] = []
    for row in mask:
        lengths.extend(_true_run_lengths(row))
    for col in mask.T:
        lengths.extend(_true_run_lengths(col))
    assert lengths, "expected occupied chain pixels"
    block = lengths[0]
    for length in lengths[1:]:
        block = math.gcd(block, length)
    assert block >= 1, "invalid block size"
    return int(block)


def _coarse_chain(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert len(components) == 1, "expected a single multicolor chain component"
    comp = components[0]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub != bg

    block = _infer_block_size(mask)
    h, w = sub.shape
    assert h % block == 0 and w % block == 0, "chain bbox must align to square block size"

    coarse_h = h // block
    coarse_w = w // block
    occ = np.zeros((coarse_h, coarse_w), dtype=bool)
    colors = np.full((coarse_h, coarse_w), bg, dtype=int)

    for rr in range(coarse_h):
        for cc in range(coarse_w):
            patch = sub[rr * block : (rr + 1) * block, cc * block : (cc + 1) * block]
            vals = set(map(int, patch.flatten()))
            if vals == {bg}:
                continue
            assert len(vals) == 1, "each coarse cell must be a solid-color square"
            color = next(iter(vals))
            assert color != bg
            occ[rr, cc] = True
            colors[rr, cc] = color

    order = _order_path(occ)
    return {
        "bg": bg,
        "bbox": (r0, c0, r1, c1),
        "block": block,
        "occ": occ,
        "colors": colors,
        "order": order,
    }


def _order_path(occ: np.ndarray) -> list[tuple[int, int]]:
    occupied = {(int(r), int(c)) for r, c in np.argwhere(occ)}
    assert occupied, "empty coarse chain"

    degrees = {cell: sum((nbr in occupied) for nbr in _neighbors4(cell, occ.shape)) for cell in occupied}
    assert all(deg in (1, 2) for deg in degrees.values()), "chain must be unbranched"

    ends = sorted([cell for cell, deg in degrees.items() if deg == 1])
    assert len(ends) == 2, "chain must have exactly two endpoints"

    path = [ends[0]]
    prev = None
    cur = ends[0]
    while True:
        nxts = [nbr for nbr in _neighbors4(cur, occ.shape) if nbr in occupied and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path walk must stay unique"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(occupied), "path walk must cover all blocks"
    return path


def _visible_segment(block_colors: list[int]) -> tuple[int, int, list[int]]:
    visible = [idx for idx, color in enumerate(block_colors) if int(color) != int(GREY)]
    assert visible, "expected a colored middle segment"
    start = visible[0]
    end = visible[-1]
    assert start > 0 and end + 1 < len(block_colors), "visible segment must sit strictly inside the chain"
    assert visible == list(range(start, end + 1)), "visible blocks must form one continuous segment in chain order"
    return start, end, block_colors[start : end + 1]


def _majority_period(seq: list[int]) -> tuple[int, list[int]]:
    n = len(seq)
    candidates: list[tuple[int, list[int]]] = []
    for period in range(1, n + 1):
        if n % period != 0 or n // period < 3:
            continue
        motif: list[int] = []
        valid = True
        for residue in range(period):
            bucket = seq[residue::period]
            counts = Counter(map(int, bucket))
            best_color, best_count = counts.most_common(1)[0]
            second = counts.most_common(2)[1][1] if len(counts) > 1 else 0
            if best_count <= second:
                valid = False
                break
            motif.append(int(best_color))
        if valid:
            candidates.append((period, motif))
    assert candidates, "failed to infer a strict-majority period"
    return candidates[0]


def _paint_block(out: np.ndarray, bbox: tuple[int, int, int, int], block: int, cell: tuple[int, int], color: int) -> None:
    r0, c0, _r1, _c1 = bbox
    rr, cc = cell
    sr = r0 + rr * block
    sc = c0 + cc * block
    out[sr : sr + block, sc : sc + block] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the true cyclic colors on a bent square-block chain, fix the corrupted blocks, and extend the cycle across all grey blocks.

    Input structure:
    - The background contains one 4-connected unbranched chain of equal solid square blocks that touch edge-to-edge and may bend arbitrarily.
    - A continuous middle segment of the chain is colored, while the blocks on both ends are grey.
    - The visible colored segment follows a repeated color cycle with a few corrupted blocks; the correct color is the strict majority at every cycle position.

    Rule:
    - Linearize the square blocks from one chain endpoint to the other.
    - Read the colors of the visible middle segment in chain order and infer the shortest cycle whose residue classes have unique strict-majority colors.
    - Recolor the visible corrupted blocks to those majority colors and extend the same cycle from that segment toward both endpoints.
    - Paint every grey block as a whole square with its inferred cycle color.

    Key insight:
    - The periodic signal lives on the snake-like block order, not on rows or columns, so the chain must be straightened conceptually before majority-based period recovery and bidirectional extrapolation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    chain = _coarse_chain(g)
    order = chain["order"]
    block_colors = [int(chain["colors"][r, c]) for r, c in order]
    start, _end, visible_seq = _visible_segment(block_colors)
    period, motif = _majority_period(visible_seq)

    out = g.copy()
    for idx, cell in enumerate(order):
        expected = motif[(idx - start) % period]
        _paint_block(out, chain["bbox"], chain["block"], cell, expected)
    return out


def _can_add(used: set[tuple[int, int]], tail: tuple[int, int], nxt: tuple[int, int], shape: tuple[int, int]) -> bool:
    if nxt in used:
        return False
    touching = [nbr for nbr in _neighbors4(nxt, shape) if nbr in used]
    return touching == [tail]


def _onward_count(used: set[tuple[int, int]], tail: tuple[int, int], shape: tuple[int, int]) -> int:
    return sum(_can_add(used, tail, nbr, shape) for nbr in _neighbors4(tail, shape))


def _search_induced_path(shape: tuple[int, int], length: int) -> list[tuple[int, int]] | None:
    h, w = shape
    starts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(starts)

    for start in starts:
        budget = [0]
        path = [start]
        used = {start}

        def dfs() -> bool:
            budget[0] += 1
            if budget[0] > PATH_SEARCH_BUDGET:
                return False
            if len(path) == length:
                return True

            tail = path[-1]
            candidates = []
            for nbr in _neighbors4(tail, shape):
                if not _can_add(used, tail, nbr, shape):
                    continue
                used.add(nbr)
                onward = _onward_count(used, nbr, shape)
                used.remove(nbr)
                candidates.append((onward, random.random(), nbr))
            candidates.sort(key=lambda item: (item[0], item[1]))

            for _onward, _rnd, nbr in candidates:
                path.append(nbr)
                used.add(nbr)
                if dfs():
                    return True
                used.remove(nbr)
                path.pop()
            return False

        if dfs():
            return path.copy()
    return None


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for idx in range(1, len(path) - 1):
        dr1 = path[idx][0] - path[idx - 1][0]
        dc1 = path[idx][1] - path[idx - 1][1]
        dr2 = path[idx + 1][0] - path[idx][0]
        dc2 = path[idx + 1][1] - path[idx][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _sample_path(length: int, block: int) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    max_coarse = (30 - 6) // block
    dims = [(h, w) for h in range(4, max_coarse + 1) for w in range(4, max_coarse + 1) if h * w >= length + 4]
    random.shuffle(dims)

    for shape in dims:
        path = _search_induced_path(shape, length)
        if path is None:
            continue
        if _turn_count(path) < 2:
            continue
        return shape, path
    return None


def _render_input(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    block: int,
    left_grey: int,
    motif: list[int],
    repeats: int,
    errors: dict[int, int],
) -> tuple[np.ndarray, list[int]]:
    coarse_h, coarse_w = shape
    margin_top = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    extra_bottom = random.randint(1, 3)
    extra_right = random.randint(1, 3)
    height = coarse_h * block + margin_top + extra_bottom
    width = coarse_w * block + margin_left + extra_right
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BG, dtype=int)
    visible_len = len(motif) * repeats
    total = len(path)
    seq = [GREY] * total
    for idx in range(visible_len):
        chain_idx = left_grey + idx
        seq[chain_idx] = int(motif[idx % len(motif)])
    for idx, wrong_color in errors.items():
        seq[idx] = int(wrong_color)

    for idx, (rr, cc) in enumerate(path):
        sr = margin_top + rr * block
        sc = margin_left + cc * block
        grid[sr : sr + block, sc : sc + block] = int(seq[idx])
    return grid, seq


def _expected_output(shape: tuple[int, int], path: list[tuple[int, int]], block: int, left_grey: int, motif: list[int], canvas_shape: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    coarse_h, coarse_w = shape
    _ = coarse_h, coarse_w
    out = np.full(canvas_shape, BG, dtype=int)
    top, left = offset
    for idx, (rr, cc) in enumerate(path):
        color = int(motif[(idx - left_grey) % len(motif)])
        sr = top + rr * block
        sc = left + cc * block
        out[sr : sr + block, sc : sc + block] = color
    return out


def _order_based_sequence(order: list[tuple[int, int]], color_map: dict[tuple[int, int], int]) -> list[int]:
    return [int(color_map[cell]) for cell in order]


def _shortcut_output(
    order: list[tuple[int, int]],
    color_map: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int] | None:
    seq = _order_based_sequence(order, color_map)
    visible = [idx for idx, color in enumerate(seq) if color != int(GREY)]
    if not visible:
        return None
    start = visible[0]
    end = visible[-1]
    if start == 0 or end + 1 == len(seq):
        return None
    if visible != list(range(start, end + 1)):
        return None
    try:
        period, motif = _majority_period(seq[start : end + 1])
    except AssertionError:
        return None
    return {cell: int(motif[(idx - start) % period]) for idx, cell in enumerate(order)}


def _validate_case(
    inp: np.ndarray,
    path: list[tuple[int, int]],
    left_grey: int,
    right_grey: int,
    motif: list[int],
    repeats: int,
    errors: dict[int, int],
) -> bool:
    if np.count_nonzero(inp == BG) <= inp.size // 2:
        return False

    total = len(path)
    visible_len = len(motif) * repeats
    if not (left_grey >= 1 and right_grey >= 1 and left_grey + visible_len + right_grey == total):
        return False
    if len(errors) < 1:
        return False

    try:
        produced = main(inp)
        chain = _coarse_chain(inp)
    except AssertionError:
        return False

    order = chain["order"]
    if len(order) != total:
        return False
    if _turn_count(order) < 2:
        return False

    block_colors = [int(chain["colors"][r, c]) for r, c in order]
    start, end, visible_seq = _visible_segment(block_colors)
    if start != left_grey or end != total - right_grey - 1:
        return False

    try:
        period, recovered = _majority_period(visible_seq)
    except AssertionError:
        return False
    if period != len(motif) or recovered != motif:
        return False

    if np.array_equal(produced, inp):
        return False
    if any(color == GREY for color in produced.flatten()):
        return False

    coarse_colors_out = {}
    for idx, cell in enumerate(order):
        coarse_colors_out[cell] = int(motif[(idx - left_grey) % len(motif)])

    color_map_in = {cell: int(block_colors[idx]) for idx, cell in enumerate(order)}
    row_order = sorted(order)
    col_order = sorted(order, key=lambda cell: (cell[1], cell[0]))
    for alt_order in (row_order, list(reversed(row_order)), col_order, list(reversed(col_order))):
        if alt_order == order or alt_order == list(reversed(order)):
            continue
        shortcut = _shortcut_output(alt_order, color_map_in)
        if shortcut == coarse_colors_out:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    period = random.choice((2, 3, 4))
    repeats = random.choice((3, 4)) if period < 4 else 3
    left_grey = random.randint(1, 3)
    right_grey = random.randint(1, 3)
    total = left_grey + period * repeats + right_grey
    block = random.choice((2, 3))

    sampled = _sample_path(total, block)
    if sampled is None:
        return None
    shape, path = sampled

    motif = random.sample(PALETTE, period)
    visible_indices = list(range(left_grey, left_grey + period * repeats))

    available_errors = visible_indices[:]
    random.shuffle(available_errors)
    error_count = random.randint(1, min(period, 3))
    errors: dict[int, int] = {}
    used_residues: set[int] = set()
    for idx in available_errors:
        residue = (idx - left_grey) % period
        if residue in used_residues:
            continue
        expected = motif[residue]
        neighbors = set()
        if idx > 0:
            neighbors.add(motif[(idx - 1 - left_grey) % period])
        if idx + 1 < total and left_grey <= idx + 1 < left_grey + period * repeats:
            neighbors.add(motif[(idx + 1 - left_grey) % period])
        pool = [color for color in PALETTE if color != expected and color not in neighbors and color not in motif]
        if not pool:
            pool = [color for color in PALETTE if color != expected and color not in neighbors]
        errors[idx] = random.choice(pool)
        used_residues.add(residue)
        if len(errors) == error_count:
            break
    if len(errors) != error_count:
        return None

    grid, _seq = _render_input(shape, path, block, left_grey, motif, repeats, errors)
    if not _validate_case(grid, path, left_grey, right_grey, motif, repeats, errors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-chain periodic-repair input")
