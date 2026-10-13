from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _outer_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Return (axis, index, color) for the sole full non-background line."""
    candidates: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != bg:
            candidates.append((0, r, int(grid[r, 0])))
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg:
            candidates.append((1, c, int(grid[0, c])))
    assert len(candidates) == 1, "expected one full non-background divider line"
    return candidates[0]


def _split_legend(legend: np.ndarray, axis: int, sep: int) -> list[np.ndarray]:
    """Split a one-dimensional legend strip at its full separator lines."""
    split_axis = 1 if axis == 0 else 0
    if split_axis == 1:
        cuts = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == sep)]
        limit = legend.shape[1]
    else:
        cuts = [r for r in range(legend.shape[0]) if np.all(legend[r, :] == sep)]
        limit = legend.shape[0]
    assert cuts, "legend blocks must be separated"

    blocks: list[np.ndarray] = []
    start = 0
    for cut in cuts + [limit]:
        assert cut > start, "empty or adjacent legend slots"
        block = legend[:, start:cut] if split_axis == 1 else legend[start:cut, :]
        blocks.append(block.copy())
        start = cut + 1
    return blocks


def _edge_signature(block: np.ndarray, bg: int, side: str) -> frozenset[int]:
    mask = block != bg
    if side == "left":
        return frozenset(map(int, np.where(mask[:, 0])[0]))
    if side == "right":
        return frozenset(map(int, np.where(mask[:, -1])[0]))
    if side == "top":
        return frozenset(map(int, np.where(mask[0, :])[0]))
    if side == "bottom":
        return frozenset(map(int, np.where(mask[-1, :])[0]))
    raise ValueError(side)


def _chain_orders(blocks: list[np.ndarray], bg: int, axis: int) -> list[tuple[int, ...]]:
    if axis == 0:
        outgoing, incoming = "right", "left"
    else:
        outgoing, incoming = "bottom", "top"
    out_edges = [_edge_signature(block, bg, outgoing) for block in blocks]
    in_edges = [_edge_signature(block, bg, incoming) for block in blocks]

    def connects(a: int, b: int) -> bool:
        # A match means that the complete visible foreground positions coincide.
        return bool(out_edges[a]) and out_edges[a] == in_edges[b]

    return [
        perm
        for perm in itertools.permutations(range(len(blocks)))
        if all(connects(perm[i], perm[i + 1]) for i in range(len(perm) - 1))
    ]


def _decode(grid: np.ndarray) -> tuple[np.ndarray, int, list[int], dict[int, tuple[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background(g)
    divider_axis, divider_index, sep = _outer_divider(g, bg)

    if divider_axis == 0:
        regions = [g[:divider_index, :], g[divider_index + 1 :, :]]
    else:
        regions = [g[:, :divider_index], g[:, divider_index + 1 :]]
    assert all(region.size for region in regions), "divider must have a region on both sides"
    legend_flags = [bool(np.any(region == sep)) for region in regions]
    assert sum(legend_flags) == 1, "only the legend region contains internal separators"
    legend_i = legend_flags.index(True)
    legend, canvas = regions[legend_i], regions[1 - legend_i]

    blocks = _split_legend(legend, divider_axis, sep)
    assert 3 <= len(blocks) <= 6
    shapes = {block.shape for block in blocks}
    assert len(shapes) == 1, "legend blocks must be equal-sized"
    bh, bw = blocks[0].shape
    assert bh == bw and bh >= 3, "legend blocks must be equal-sized squares"

    block_colors: list[int] = []
    for block in blocks:
        colors = set(map(int, block.ravel())) - {bg}
        assert len(colors) == 1 and sep not in colors, "each legend block has one foreground color"
        block_colors.append(next(iter(colors)))
    assert len(set(block_colors)) == len(block_colors), "legend colors must be distinct"

    orders = _chain_orders(blocks, bg, divider_axis)
    assert len(orders) == 1, "legend must encode exactly one complete chain"
    chain = [block_colors[i] for i in orders[0]]

    canvas_colors = set(map(int, canvas.ravel()))
    assert canvas_colors <= {bg, *chain}, "canvas may contain only background and legend colors"
    markers: dict[int, tuple[str, int]] = {}
    for color in chain:
        coords = np.argwhere(canvas == color)
        assert coords.shape == (2, 2), "each legend color must have one two-pixel marker"
        rows = sorted(set(map(int, coords[:, 0])))
        cols = sorted(set(map(int, coords[:, 1])))
        if len(rows) == 1 and cols[1] == cols[0] + 1:
            markers[color] = ("h", rows[0])
        elif len(cols) == 1 and rows[1] == rows[0] + 1:
            markers[color] = ("v", cols[0])
        else:
            raise AssertionError("marker must be a horizontal or vertical domino")
    return canvas.copy(), bg, chain, markers


def main(grid: np.ndarray) -> np.ndarray:
    """Order legend colors by matching block edges, then layer the corresponding full canvas bars.

    Input structure:
    - One full separator line divides a legend strip from a canvas; the legend contains
      equal square, single-color pattern blocks separated by shorter lines of that color.
    - Facing foreground positions on the blocks encode one unique chain.
    - Every legend color occurs on the canvas as exactly one horizontal or vertical domino.

    Rule:
    - Reconstruct the unique chain by matching a block's trailing edge to the next
      block's leading edge. Extend horizontal dominoes across their complete rows and
      vertical dominoes across their complete columns, painting in chain order.
      Later (chain-tail) colors overwrite earlier colors at crossings.

    Key insight:
    - Domino orientation selects row versus column, but crossing priority comes only
      from the independently scrambled legend's edge-matched chain.
    """
    canvas, bg, chain, markers = _decode(np.asarray(grid, dtype=int))
    out = np.full_like(canvas, bg)
    for color in chain:
        orient, index = markers[color]
        if orient == "h":
            out[index, :] = color
        else:
            out[:, index] = color
    return out


# ------------------------------ generator ---------------------------------

_PALETTE = tuple(range(10))
_ATTEMPTS = 300


def _grow_pattern(mask: np.ndarray, additions: int) -> None:
    """Add connected interior cells without changing any edge signature."""
    side = mask.shape[0]
    for _ in range(additions):
        choices: list[tuple[int, int]] = []
        for r in range(1, side - 1):
            for c in range(1, side - 1):
                if mask[r, c]:
                    continue
                if mask[r - 1, c] or mask[r + 1, c] or mask[r, c - 1] or mask[r, c + 1]:
                    choices.append((r, c))
        if not choices:
            return
        mask[random.choice(choices)] = True


def _horizontal_pattern(side: int, color: int, left: int | None, right: int | None, bg: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    bend = random.randint(1, side - 2)
    if left is not None and right is not None:
        mask[left, : bend + 1] = True
        mask[min(left, right) : max(left, right) + 1, bend] = True
        mask[right, bend:] = True
    elif left is not None:
        mask[left, : bend + 1] = True
    elif right is not None:
        mask[right, bend:] = True
    else:
        raise AssertionError("a chain block must expose at least one joining edge")
    _grow_pattern(mask, random.randint(1, side))
    block = np.full((side, side), bg, dtype=int)
    block[mask] = color
    return block


def _make_chain_blocks(n: int, side: int, colors: list[int], axis: int, bg: int) -> list[np.ndarray]:
    links = random.sample(range(1, side - 1), n - 1)
    blocks = []
    for i, color in enumerate(colors):
        left = links[i - 1] if i else None
        right = links[i] if i < n - 1 else None
        block = _horizontal_pattern(side, color, left, right, bg)
        blocks.append(block if axis == 0 else block.T.copy())
    return blocks


def _scramble(blocks: list[np.ndarray]) -> list[np.ndarray]:
    order = list(range(len(blocks)))
    while order == list(range(len(blocks))):
        random.shuffle(order)
    return [blocks[i] for i in order]


def _render_legend(blocks: list[np.ndarray], axis: int, bg: int, sep: int) -> np.ndarray:
    side = blocks[0].shape[0]
    length = len(blocks) * side + len(blocks) - 1
    shape = (side, length) if axis == 0 else (length, side)
    legend = np.full(shape, bg, dtype=int)
    cursor = 0
    for i, block in enumerate(blocks):
        if axis == 0:
            legend[:, cursor : cursor + side] = block
        else:
            legend[cursor : cursor + side, :] = block
        cursor += side
        if i < len(blocks) - 1:
            if axis == 0:
                legend[:, cursor] = sep
            else:
                legend[cursor, :] = sep
            cursor += 1
    return legend


def _orientation_sequences(n: int) -> list[tuple[str, ...]]:
    """Sequences that visibly disprove both fixed row-first and fixed column-first rules."""
    valid = []
    for seq in itertools.product(("h", "v"), repeat=n):
        has_h_before_v = any(seq[i] == "h" and seq[j] == "v" for i in range(n) for j in range(i + 1, n))
        has_v_before_h = any(seq[i] == "v" and seq[j] == "h" for i in range(n) for j in range(i + 1, n))
        if has_h_before_v and has_v_before_h:
            valid.append(seq)
    return valid


def _place_markers(shape: tuple[int, int], colors: list[int], orientations: tuple[str, ...], bg: int) -> tuple[np.ndarray, dict[int, tuple[str, int]]] | None:
    height, width = shape
    canvas = np.full(shape, bg, dtype=int)
    occupied: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    markers: dict[int, tuple[str, int]] = {}

    def options(orient: str) -> list[tuple[int, list[tuple[int, int]]]]:
        result: list[tuple[int, list[tuple[int, int]]]] = []
        if orient == "h":
            for r in range(1, height - 1):
                if r in used_rows:
                    continue
                for c in range(1, width - 2):
                    result.append((r, [(r, c), (r, c + 1)]))
        else:
            for c in range(1, width - 1):
                if c in used_cols:
                    continue
                for r in range(1, height - 2):
                    result.append((c, [(r, c), (r + 1, c)]))
        random.shuffle(result)
        return result

    def separated(cells: list[tuple[int, int]]) -> bool:
        return all(
            (r + dr, c + dc) not in occupied
            for r, c in cells
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        )

    def rec(i: int) -> bool:
        if i == len(colors):
            return True
        color, orient = colors[i], orientations[i]
        for index, cells in options(orient):
            if not separated(cells):
                continue
            for r, c in cells:
                canvas[r, c] = color
                occupied.add((r, c))
            (used_rows if orient == "h" else used_cols).add(index)
            markers[color] = (orient, index)
            if rec(i + 1):
                return True
            del markers[color]
            (used_rows if orient == "h" else used_cols).remove(index)
            for r, c in cells:
                canvas[r, c] = bg
                occupied.remove((r, c))
        return False

    return (canvas, markers) if rec(0) else None


def _render_expected(shape: tuple[int, int], bg: int, colors: list[int], markers: dict[int, tuple[str, int]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in colors:
        orient, index = markers[color]
        if orient == "h":
            out[index, :] = color
        else:
            out[:, index] = color
    return out


def _combine(legend: np.ndarray, canvas: np.ndarray, axis: int, sep: int, legend_first: bool) -> np.ndarray:
    if axis == 0:
        line = np.full((1, legend.shape[1]), sep, dtype=int)
        parts = (legend, line, canvas) if legend_first else (canvas, line, legend)
        return np.vstack(parts)
    line = np.full((legend.shape[0], 1), sep, dtype=int)
    parts = (legend, line, canvas) if legend_first else (canvas, line, legend)
    return np.hstack(parts)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice((3, 4))
    side = random.randint(n + 1, 6)
    axis = random.choice((0, 1))
    bg, sep, *colors = random.sample(_PALETTE, n + 2)

    ordered_blocks = _make_chain_blocks(n, side, colors, axis, bg)
    blocks = _scramble(ordered_blocks)
    if len(_chain_orders(blocks, bg, axis)) != 1:
        return None
    legend = _render_legend(blocks, axis, bg, sep)

    length = legend.shape[1] if axis == 0 else legend.shape[0]
    canvas_shape = (random.randint(8, 13), length) if axis == 0 else (length, random.randint(8, 13))
    orientations = random.choice(_orientation_sequences(n))
    placed = _place_markers(canvas_shape, colors, orientations, bg)
    if placed is None:
        return None
    canvas, markers = placed
    expected = _render_expected(canvas_shape, bg, colors, markers)

    # Every key dependency must be observable: reversing priority changes crossings,
    # and examples contain both a later horizontal and a later vertical winner.
    reversed_expected = _render_expected(canvas_shape, bg, list(reversed(colors)), markers)
    if np.array_equal(expected, reversed_expected):
        return None
    winners = set()
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            oa, _ = markers[color_a]
            ob, _ = markers[color_b]
            if oa != ob:
                winners.add(ob)
    if winners != {"h", "v"}:
        return None

    inp = _combine(legend, canvas, axis, sep, random.choice((True, False)))
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-layering input")
