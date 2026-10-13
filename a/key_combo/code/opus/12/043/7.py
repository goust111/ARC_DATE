from __future__ import annotations

from collections import deque
import random

import numpy as np


INPUT_HEIGHT = 4
VALID_WIDTHS = (6, 8, 10)
SIDE_SAMPLING = (6, 8, 10, 10, 10)
NONZERO_PALETTE = (5, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def _read_blocks_reverse_row_major(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    blocks: list[np.ndarray] = []
    for br in range(h // 2 - 1, -1, -1):
        for bc in range(w // 2 - 1, -1, -1):
            blocks.append(grid[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2].copy())
    return blocks


def _layer_sequences_from_blocks(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layers = np.asarray([block.reshape(4) for block in blocks], dtype=int)
    return layers[:, 0], layers[:, 1], layers[:, 2], layers[:, 3]


def _render_quadrant(layer_seq: np.ndarray, first_corner: str) -> np.ndarray:
    side = int(len(layer_seq))
    out = np.empty((side, side), dtype=int)
    r0 = c0 = 0
    r1 = c1 = side - 1
    corner = first_corner
    opposite = {"ul": "lr", "ur": "ll", "ll": "ur", "lr": "ul"}

    for color in layer_seq:
        if corner == "ul":
            out[r0, c0 : c1 + 1] = int(color)
            if r0 + 1 <= r1:
                out[r0 + 1 : r1 + 1, c0] = int(color)
            r0 += 1
            c0 += 1
        elif corner == "ur":
            out[r0, c0 : c1 + 1] = int(color)
            if r0 + 1 <= r1:
                out[r0 + 1 : r1 + 1, c1] = int(color)
            r0 += 1
            c1 -= 1
        elif corner == "ll":
            out[r1, c0 : c1 + 1] = int(color)
            if r0 <= r1 - 1:
                out[r0 : r1, c0] = int(color)
            r1 -= 1
            c0 += 1
        else:
            out[r1, c0 : c1 + 1] = int(color)
            if r0 <= r1 - 1:
                out[r0 : r1, c1] = int(color)
            r1 -= 1
            c1 -= 1
        corner = opposite[corner]

    return out


def _assemble_raw_output(
    ul_seq: np.ndarray,
    ur_seq: np.ndarray,
    ll_seq: np.ndarray,
    lr_seq: np.ndarray,
) -> np.ndarray:
    side = int(len(ul_seq))
    out = np.empty((2 * side, 2 * side), dtype=int)
    out[0:side, 0:side] = _render_quadrant(ul_seq, "ul")
    out[0:side, side : 2 * side] = _render_quadrant(ur_seq, "ur")
    out[side : 2 * side, 0:side] = _render_quadrant(ll_seq, "ll")
    out[side : 2 * side, side : 2 * side] = _render_quadrant(lr_seq, "lr")
    return out


def _count_enclosed_black_regions(quadrant: np.ndarray) -> int:
    h, w = quadrant.shape
    seen = np.zeros((h, w), dtype=bool)
    enclosed = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(quadrant[r, c]) != 0:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while queue:
                cr, cc = queue.popleft()
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(quadrant[nr, nc]) == 0:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not touches_border:
                enclosed += 1
    return enclosed


def _recolor_for_count(count: int) -> int:
    if count == 0:
        return 1
    if count == 1:
        return 2
    if count == 2:
        return 3
    return 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 2x2-block strip into nested quadrants, count enclosed black regions, and recolor each quadrant by that count.

    Input structure:
    - The input is a 4xN strip with even width N in 6..10, partitionable into 2x2 blocks.
    - Reverse row-major block order determines the concentric layer order.
    - Each block contributes one color per quadrant layer: top-left, top-right, bottom-left, bottom-right.

    Rule:
    - Read blocks from bottom-right to top-left in reverse row-major order.
    - Expand those block pixels into four SxS nested layer patterns, where each quadrant is peeled one layer at a time by alternating between opposite corners and S is the number of 2x2 blocks.
    - In each quadrant, count 4-connected black regions that do not touch the quadrant border.
    - Keep black cells black and recolor every non-black cell by the enclosed-region count: 0->1, 1->2, 2->3, >=3->4.

    Key insight:
    - The strip only encodes concentric layers; the decisive signal is the topology of black gaps after expansion, not the raw block colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == INPUT_HEIGHT, "input must have exactly four rows"
    assert w in VALID_WIDTHS and w % 2 == 0, "input width must be one of 6, 8, 10"

    blocks = _read_blocks_reverse_row_major(g)
    side = len(blocks)
    assert side == w, "the 4xN strip must contain exactly N 2x2 blocks"

    raw = _assemble_raw_output(*_layer_sequences_from_blocks(blocks))
    out = raw.copy()
    quadrants = [
        (slice(0, side), slice(0, side)),
        (slice(0, side), slice(side, 2 * side)),
        (slice(side, 2 * side), slice(0, side)),
        (slice(side, 2 * side), slice(side, 2 * side)),
    ]
    for rs, cs in quadrants:
        quadrant = raw[rs, cs]
        fill_color = _recolor_for_count(_count_enclosed_black_regions(quadrant))
        recolored = quadrant.copy()
        recolored[recolored != 0] = fill_color
        out[rs, cs] = recolored
    return out


_PATTERN_CACHE: dict[int, dict[int, dict[bool, list[tuple[int, ...]]]]] = {}


def _binary_patterns_by_count(side: int) -> dict[int, dict[bool, list[tuple[int, ...]]]]:
    cached = _PATTERN_CACHE.get(side)
    if cached is not None:
        return cached

    by_count: dict[int, dict[bool, list[tuple[int, ...]]]] = {}
    for mask in range(1, 1 << side):
        bits = tuple(1 if (mask >> idx) & 1 else 0 for idx in range(side))
        seq = np.array([9 if bit else 0 for bit in bits], dtype=int)
        enclosed = _count_enclosed_black_regions(_render_quadrant(seq, "ul"))
        by_count.setdefault(enclosed, {False: [], True: []})[bits[0] == 0].append(bits)

    _PATTERN_CACHE[side] = by_count
    return by_count


def _choose_binary_pattern(side: int, enclosed_count: int, prefer_prefix_zero: bool | None) -> list[int]:
    by_count = _binary_patterns_by_count(side)
    if enclosed_count not in by_count:
        raise ValueError("no feasible topology count")

    groups = by_count[enclosed_count]
    pool: list[tuple[int, ...]] = []
    if prefer_prefix_zero is not None and groups[prefer_prefix_zero]:
        pool.extend(groups[prefer_prefix_zero])
    if groups[False]:
        pool.extend(groups[False])
    if groups[True]:
        pool.extend(groups[True])
    if not pool:
        raise ValueError("empty candidate pool")
    return list(random.choice(pool))


def _colorize_layers(bits: list[int]) -> np.ndarray:
    seq: list[int] = []
    last_run_color: int | None = None
    prev_bit: int | None = None
    for bit in bits:
        if bit == 0:
            seq.append(0)
            prev_bit = 0
            continue
        if prev_bit != 1 or last_run_color is None or random.random() < 0.4:
            choices = [color for color in NONZERO_PALETTE if color != last_run_color]
            last_run_color = random.choice(choices or list(NONZERO_PALETTE))
        seq.append(last_run_color)
        prev_bit = 1
    return np.array(seq, dtype=int)


def _build_input_from_sequences(
    ul_seq: np.ndarray,
    ur_seq: np.ndarray,
    ll_seq: np.ndarray,
    lr_seq: np.ndarray,
) -> np.ndarray:
    side = int(len(ul_seq))
    grid = np.zeros((INPUT_HEIGHT, side), dtype=int)
    layer_idx = 0
    for br in range(INPUT_HEIGHT // 2 - 1, -1, -1):
        for bc in range(side // 2 - 1, -1, -1):
            block = np.array(
                [
                    [int(ul_seq[layer_idx]), int(ur_seq[layer_idx])],
                    [int(ll_seq[layer_idx]), int(lr_seq[layer_idx])],
                ],
                dtype=int,
            )
            grid[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2] = block
            layer_idx += 1
    return grid


def _quadrant_slices(side: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, side), slice(0, side)),
        (slice(0, side), slice(side, 2 * side)),
        (slice(side, 2 * side), slice(0, side)),
        (slice(side, 2 * side), slice(side, 2 * side)),
    ]


def _validate_generated(grid: np.ndarray, target_counts: list[int]) -> bool:
    if grid.shape[0] != INPUT_HEIGHT or grid.shape[1] not in VALID_WIDTHS:
        return False

    blocks = _read_blocks_reverse_row_major(grid)
    sequences = _layer_sequences_from_blocks(blocks)
    if any(int(np.count_nonzero(seq)) == 0 for seq in sequences):
        return False

    raw = _assemble_raw_output(*sequences)
    actual_counts: list[int] = []
    for rs, cs in _quadrant_slices(len(blocks)):
        actual_counts.append(_count_enclosed_black_regions(raw[rs, cs]))
    if actual_counts != target_counts:
        return False
    if len(set(actual_counts)) < 2 or max(actual_counts) < 1:
        return False

    out = main(grid)
    if out.shape != (2 * len(blocks), 2 * len(blocks)):
        return False

    for (rs, cs), enclosed in zip(_quadrant_slices(len(blocks)), actual_counts):
        raw_quadrant = raw[rs, cs]
        out_quadrant = out[rs, cs]
        if not np.array_equal(out_quadrant == 0, raw_quadrant == 0):
            return False
        nonzero_values = set(map(int, out_quadrant[out_quadrant != 0].tolist()))
        if len(nonzero_values) != 1:
            return False
        if next(iter(nonzero_values)) != _recolor_for_count(enclosed):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        side = random.choice(SIDE_SAMPLING)
        available_counts = sorted(_binary_patterns_by_count(side))
        lower_counts = [count for count in available_counts if count in (0, 1, 2)]
        high_counts = [count for count in available_counts if count >= 3]
        if len(lower_counts) < 2:
            continue

        if high_counts and set((0, 1, 2)).issubset(available_counts):
            target_counts = [0, 1, 2, random.choice(high_counts)]
        else:
            base = sorted(lower_counts)[:]
            while len(base) < 4:
                base.append(random.choice(lower_counts))
            target_counts = random.sample(base, 4)
        random.shuffle(target_counts)

        prefix_zero_candidates = [
            idx for idx, enclosed in enumerate(target_counts) if _binary_patterns_by_count(side).get(enclosed, {}).get(True)
        ]
        if not prefix_zero_candidates:
            continue
        prefix_zero_index = random.choice(prefix_zero_candidates)
        sequences: list[np.ndarray] = []
        for idx, enclosed in enumerate(target_counts):
            prefer_prefix_zero = True if idx == prefix_zero_index else None
            bits = _choose_binary_pattern(side, enclosed, prefer_prefix_zero)
            sequences.append(_colorize_layers(bits))

        grid = _build_input_from_sequences(*sequences)
        if _validate_generated(grid, target_counts):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
