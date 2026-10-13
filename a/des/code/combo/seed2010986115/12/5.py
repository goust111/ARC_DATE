from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


BG = 8
GRADIENT = (1, 2, 3, 4, 6)  # blue, red, green, yellow, magenta
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

# Concrete arrow realization used by both solver and generator:
# each arrow is one 2x2 L-triomino rotation with a unique tail cell.
REL_CELLS = {
    "U": ((0, 0), (-1, 0), (-1, 1)),
    "R": ((0, 0), (0, 1), (1, 1)),
    "D": ((0, 0), (1, -1), (1, 0)),
    "L": ((0, 0), (0, -1), (-1, -1)),
}
NORM_SHAPES = {
    frozenset({(0, 0), (0, 1), (1, 0)}): ("U", (1, 0)),
    frozenset({(0, 0), (0, 1), (1, 1)}): ("R", (0, 0)),
    frozenset({(0, 1), (1, 0), (1, 1)}): ("D", (0, 1)),
    frozenset({(0, 0), (1, 0), (1, 1)}): ("L", (1, 1)),
}

# --------------- solver helpers ---------------


@dataclass(frozen=True)
class Arrow:
    color: int
    tail: tuple[int, int]
    direction: str
    cells: tuple[tuple[int, int], ...]


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(pts)
    return comps


def _arrow_from_tail(tail: tuple[int, int], direction: str, color: int) -> Arrow:
    tr, tc = map(int, tail)
    cells = tuple(sorted((tr + dr, tc + dc) for dr, dc in REL_CELLS[direction]))
    return Arrow(color=int(color), tail=(tr, tc), direction=direction, cells=cells)


def _detect_arrows(grid: np.ndarray) -> tuple[list[Arrow], int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    arrows: list[Arrow] = []
    for pts in _components8(g != bg):
        assert len(pts) == 3, "each foreground component must contain exactly three pixels"
        colors = {int(g[r, c]) for r, c in pts}
        assert len(colors) == 1, "each arrow must be monochrome"
        r0 = min(r for r, _ in pts)
        c0 = min(c for _, c in pts)
        norm = frozenset((r - r0, c - c0) for r, c in pts)
        assert norm in NORM_SHAPES, "unexpected arrow shape"
        direction, tail_rel = NORM_SHAPES[norm]
        tail = (int(r0 + tail_rel[0]), int(c0 + tail_rel[1]))
        arrows.append(_arrow_from_tail(tail, direction, int(next(iter(colors)))))
    arrows.sort(key=lambda arrow: (arrow.tail[0], arrow.tail[1], arrow.direction))
    return arrows, bg


def _candidate_key(src: Arrow, dst: Arrow) -> tuple[int, int, int, int, int] | None:
    if src is dst:
        return None

    dr, dc = DIRS[src.direction]
    sr, sc = src.tail
    tr, tc = dst.tail
    if dr == 0:
        primary = (tc - sc) * dc
        orth = tr - sr
    else:
        primary = (tr - sr) * dr
        orth = tc - sc
    if primary <= 0 or abs(int(orth)) > 1:
        return None
    return (int(primary), abs(int(orth)), int(orth), int(tr), int(tc))


def _build_successors(arrows: list[Arrow]) -> list[int | None]:
    succ: list[int | None] = [None] * len(arrows)
    for i, src in enumerate(arrows):
        best_idx = None
        best_key = None
        for j, dst in enumerate(arrows):
            key = _candidate_key(src, dst)
            if key is None:
                continue
            if best_key is None or key < best_key:
                best_idx = j
                best_key = key
        succ[i] = best_idx
    return succ


def _extract_chains(arrows: list[Arrow], succ: list[int | None]) -> list[list[int]]:
    indeg = [0] * len(arrows)
    for dst in succ:
        if dst is not None:
            indeg[int(dst)] += 1

    order = sorted(range(len(arrows)), key=lambda i: (arrows[i].tail[0], arrows[i].tail[1], arrows[i].direction))
    starts = [i for i in order if indeg[i] == 0]

    chains: list[list[int]] = []
    seen: set[int] = set()
    for start in starts + order:
        if start in seen:
            continue
        cur = start
        local: set[int] = set()
        chain: list[int] = []
        while cur is not None and cur not in seen and cur not in local:
            chain.append(int(cur))
            seen.add(int(cur))
            local.add(int(cur))
            cur = succ[int(cur)]
        if chain:
            chains.append(chain)
    return chains


def _render_transformed(shape: tuple[int, int], chains: list[list[Arrow]], bg: int) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    for chain in chains:
        assert len(chain) <= len(GRADIENT), "chain length exceeds gradient palette"
        dr, dc = DIRS[chain[-1].direction]
        shift_r = 2 * dr
        shift_c = 2 * dc
        for i, arrow in enumerate(chain):
            color = int(GRADIENT[i])
            for r, c in arrow.cells:
                rr = int(r + shift_r)
                cc = int(c + shift_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain the scattered arrows by pointer direction, recolor each chain by position,
    then translate the whole chain by the terminal arrow's direction.

    Input structure
    - Background is azure.
    - Each arrow is a monochrome 3-pixel L-triomino whose rotation determines its direction.
    - Arrows do not touch each other.

    Rule
    - For each arrow A, the nearest tail pixel ahead of A in A's direction, with orthogonal
      offset at most 1, is the next arrow in the chain.
    - Recolor every chain from start to end using blue, red, green, yellow, magenta.
    - Translate the entire chain by 2 cells in the final arrow's pointing direction.
      Cells shifted outside the grid are truncated.
    """
    g = np.asarray(grid, dtype=int)
    arrows, bg = _detect_arrows(g)
    if not arrows:
        return np.full_like(g, int(bg))

    succ = _build_successors(arrows)
    chain_indices = _extract_chains(arrows, succ)
    chains = [[arrows[i] for i in chain] for chain in chain_indices]
    return _render_transformed(g.shape, chains, bg)


# --------------- generator ---------------

# Grid sizing and retry budgets.
GEN_H_RANGE = (16, 24)
GEN_W_RANGE = (16, 24)
GEN_MAX_ATTEMPTS = 400
GEN_CHAIN_SHAPE_ATTEMPTS = 150
GEN_PLAN_ATTEMPTS = 120
GEN_PLACE_ATTEMPTS = 400

# Chain complexity.
GEN_CHAIN_COUNT_CHOICES = (1, 2, 3)
GEN_CHAIN_LEN_RANGE = (1, 5)
GEN_TOTAL_ARROWS_RANGE = (3, 8)
GEN_GAP_RANGE = (4, 6)
GEN_CLIP_PROB = 0.35


def _step_tail(tail: tuple[int, int], direction: str, gap: int, orth: int) -> tuple[int, int]:
    r, c = map(int, tail)
    if direction == "R":
        return (int(r + orth), int(c + gap))
    if direction == "L":
        return (int(r + orth), int(c - gap))
    if direction == "D":
        return (int(r + gap), int(c + orth))
    return (int(r - gap), int(c + orth))


def _cells_touch_halo(cells: tuple[tuple[int, int], ...], halo: set[tuple[int, int]]) -> bool:
    return any((int(r), int(c)) in halo for r, c in cells)


def _add_to_halo(cells: tuple[tuple[int, int], ...], halo: set[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((int(r + dr), int(c + dc)))


def _build_relative_chain(length: int) -> list[tuple[tuple[int, int], str]] | None:
    for _ in range(GEN_CHAIN_SHAPE_ATTEMPTS):
        plan: list[tuple[tuple[int, int], str]] = []
        tails = [(0, 0)]
        directions: list[str] = []
        for _step in range(max(0, length - 1)):
            direction = random.choice(tuple(DIRS))
            gap = random.randint(*GEN_GAP_RANGE)
            orth = random.choice((-1, 0, 1))
            directions.append(direction)
            tails.append(_step_tail(tails[-1], direction, gap, orth))
        directions.append(random.choice(tuple(DIRS)))
        plan = list(zip(tails, directions))

        halo: set[tuple[int, int]] = set()
        arrows = []
        ok = True
        for idx, (tail, direction) in enumerate(plan):
            arrow = _arrow_from_tail(tail, direction, idx + 1)
            if _cells_touch_halo(arrow.cells, halo):
                ok = False
                break
            _add_to_halo(arrow.cells, halo)
            arrows.append(arrow)
        if not ok:
            continue

        expected = [i + 1 for i in range(len(arrows) - 1)] + [None]
        if _build_successors(arrows) != expected:
            continue
        return plan
    return None


def _sample_chain_lengths() -> list[int]:
    for _ in range(200):
        chain_count = random.choice(GEN_CHAIN_COUNT_CHOICES)
        lengths = [random.randint(*GEN_CHAIN_LEN_RANGE) for _ in range(chain_count)]
        total = sum(lengths)
        if not (GEN_TOTAL_ARROWS_RANGE[0] <= total <= GEN_TOTAL_ARROWS_RANGE[1]):
            continue
        if max(lengths) == 1:
            continue
        return lengths
    raise RuntimeError("failed to sample chain lengths")


def _offset_ranges(cells: list[tuple[int, int]], h: int, w: int) -> tuple[range, range] | None:
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    row_lo = -min_r
    row_hi = h - 1 - max_r
    col_lo = -min_c
    col_hi = w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return range(int(row_lo), int(row_hi) + 1), range(int(col_lo), int(col_hi) + 1)


def _visible_after_shift(
    cells: tuple[tuple[int, int], ...], shift: tuple[int, int], h: int, w: int
) -> list[tuple[int, int]]:
    vis: list[tuple[int, int]] = []
    sr, sc = shift
    for r, c in cells:
        rr = int(r + sr)
        cc = int(c + sc)
        if 0 <= rr < h and 0 <= cc < w:
            vis.append((rr, cc))
    return vis


def _place_chain(
    rel_plan: list[tuple[tuple[int, int], str]],
    h: int,
    w: int,
    input_halo: set[tuple[int, int]],
    output_halo: set[tuple[int, int]],
    require_clip: bool,
) -> list[Arrow] | None:
    rel_arrows = [_arrow_from_tail(tail, direction, 1) for tail, direction in rel_plan]
    all_input_cells = [cell for arrow in rel_arrows for cell in arrow.cells]
    ranges = _offset_ranges(all_input_cells, h, w)
    if ranges is None:
        return None
    row_range, col_range = ranges

    offsets = [(r, c) for r in row_range for c in col_range]
    random.shuffle(offsets)
    shift_dir = DIRS[rel_arrows[-1].direction]
    shift = (2 * shift_dir[0], 2 * shift_dir[1])

    for off_r, off_c in offsets[:GEN_PLACE_ATTEMPTS]:
        abs_arrows = [_arrow_from_tail((tail[0] + off_r, tail[1] + off_c), direction, 1) for tail, direction in rel_plan]
        if any(_cells_touch_halo(arrow.cells, input_halo) for arrow in abs_arrows):
            continue

        vis_by_arrow = [_visible_after_shift(arrow.cells, shift, h, w) for arrow in abs_arrows]
        visible_sizes = [len(vis) for vis in vis_by_arrow]
        if min(visible_sizes) < 2:
            continue
        is_clipped = any(size < 3 for size in visible_sizes)
        if require_clip != is_clipped:
            continue

        all_vis = [cell for vis in vis_by_arrow for cell in vis]
        if any(cell in output_halo for cell in all_vis):
            continue
        return abs_arrows
    return None


def _intended_successors(chains: list[list[Arrow]]) -> dict[tuple[int, int], tuple[tuple[int, int] | None, str]]:
    intended: dict[tuple[int, int], tuple[tuple[int, int] | None, str]] = {}
    for chain in chains:
        for i, arrow in enumerate(chain):
            next_tail = chain[i + 1].tail if i + 1 < len(chain) else None
            intended[arrow.tail] = (next_tail, arrow.direction)
    return intended


def _plan_is_valid(grid: np.ndarray, chains: list[list[Arrow]]) -> bool:
    try:
        detected, _ = _detect_arrows(grid)
    except Exception:
        return False

    intended = _intended_successors(chains)
    if len(detected) != len(intended):
        return False

    tail_to_idx = {arrow.tail: i for i, arrow in enumerate(detected)}
    if set(tail_to_idx) != set(intended):
        return False
    for tail, (_next_tail, direction) in intended.items():
        if detected[tail_to_idx[tail]].direction != direction:
            return False

    succ = _build_successors(detected)
    for tail, (next_tail, _direction) in intended.items():
        actual_idx = succ[tail_to_idx[tail]]
        actual_next = None if actual_idx is None else detected[actual_idx].tail
        if actual_next != next_tail:
            return False

    expected = _render_transformed(grid.shape, chains, BG)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    lengths = _sample_chain_lengths()

    rel_chains: list[list[tuple[tuple[int, int], str]]] = []
    for length in lengths:
        rel_chain = _build_relative_chain(length)
        if rel_chain is None:
            return None
        rel_chains.append(rel_chain)

    clip_idx = None
    if random.random() < GEN_CLIP_PROB:
        clip_idx = random.randrange(len(rel_chains))

    order = list(range(len(rel_chains)))
    order.sort(key=lambda i: (0 if i == clip_idx else 1, -len(rel_chains[i])))

    input_halo: set[tuple[int, int]] = set()
    output_halo: set[tuple[int, int]] = set()
    placed: list[list[Arrow] | None] = [None] * len(rel_chains)

    for idx in order:
        abs_chain = _place_chain(
            rel_chains[idx],
            h,
            w,
            input_halo,
            output_halo,
            require_clip=(idx == clip_idx),
        )
        if abs_chain is None:
            return None
        placed[idx] = abs_chain
        for arrow in abs_chain:
            _add_to_halo(arrow.cells, input_halo)
        shift_dir = DIRS[abs_chain[-1].direction]
        shift = (2 * shift_dir[0], 2 * shift_dir[1])
        for arrow in abs_chain:
            vis = tuple(_visible_after_shift(arrow.cells, shift, h, w))
            _add_to_halo(vis, output_halo)

    chains = [chain for chain in placed if chain is not None]
    colors = [c for c in range(10) if c != BG]
    grid = np.full((h, w), BG, dtype=int)
    for chain in chains:
        chain_palette = random.sample(colors, random.randint(1, min(3, len(colors))))
        for arrow in chain:
            color = random.choice(chain_palette)
            for r, c in arrow.cells:
                grid[r, c] = color

    if not _plan_is_valid(grid, chains):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate input after {GEN_MAX_ATTEMPTS} attempts")
