"""ARC task: relocate one-hole blocks to the cells named by their holes."""

from __future__ import annotations

import numpy as np


GREEN = 3
MAX_GENERATION_ATTEMPTS = 300

_rng = np.random.default_rng()
_generation_count = 0
_seen_outputs: set[tuple[tuple[int, int], bytes]] = set()


def _grid_order(shape: tuple[int, int]) -> int:
    """Recover n from a square side of n*n+n-1, for the stated n in {3, 4}."""
    height, width = map(int, shape)
    assert height == width, "expected a square separated grid"
    candidates = [n for n in (3, 4) if n * n + n - 1 == height]
    assert len(candidates) == 1, "grid side must encode n=3 or n=4"
    return candidates[0]


def _cell_slice(n: int, cell: tuple[int, int]) -> tuple[slice, slice]:
    tile_r, tile_c = cell
    step = n + 1
    r0, c0 = tile_r * step, tile_c * step
    return slice(r0, r0 + n), slice(c0, c0 + n)


def _separator_mask(side: int, n: int) -> np.ndarray:
    separator_axis = np.arange(side) % (n + 1) == n
    return separator_axis[:, None] | separator_axis[None, :]


def _parse_for_background(
    grid: np.ndarray, n: int, background: int
) -> dict[tuple[int, int], tuple[int, tuple[int, int], np.ndarray]] | None:
    """Parse cells under one background hypothesis, or reject the hypothesis."""
    blocks: dict[tuple[int, int], tuple[int, tuple[int, int], np.ndarray]] = {}
    for tile_r in range(n):
        for tile_c in range(n):
            cell = (tile_r, tile_c)
            rs, cs = _cell_slice(n, cell)
            tile = grid[rs, cs]

            if np.all(tile == background):
                continue
            if int(np.count_nonzero(tile == background)) != 1:
                return None

            solid_values = np.unique(tile[tile != background])
            if len(solid_values) != 1:
                return None
            color = int(solid_values[0])
            if color == GREEN:
                return None

            raw_hole = np.argwhere(tile == background)[0]
            hole = (int(raw_hole[0]), int(raw_hole[1]))
            blocks[cell] = (color, hole, tile.copy())

    if not blocks:
        return None
    occupied = set(blocks)
    addresses = [record[1] for record in blocks.values()]
    if len(addresses) != len(set(addresses)) or set(addresses) != occupied:
        return None
    return blocks


def _analyze(
    grid: np.ndarray,
) -> tuple[int, int, dict[tuple[int, int], tuple[int, tuple[int, int], np.ndarray]]]:
    """Identify geometry, background, blocks, and their local-coordinate addresses."""
    n = _grid_order(grid.shape)
    side = int(grid.shape[0])
    separators = _separator_mask(side, n)
    assert np.all(grid[separators] == GREEN), "separator rows and columns must be green"
    assert not np.any(grid[~separators] == GREEN), "green is reserved for separators"

    interior = grid[~separators]
    hypotheses = []
    for raw_color in np.unique(interior):
        background = int(raw_color)
        if background == GREEN:
            continue
        blocks = _parse_for_background(grid, n, background)
        if blocks is not None:
            frequency = int(np.count_nonzero(interior == background))
            hypotheses.append((frequency, background, blocks))

    assert hypotheses, "no background interpretation satisfies the one-hole address rule"
    # Empty cells normally make the background the most frequent interior color.  If
    # every cell is occupied by one same-colored block, the true background instead
    # has the smallest frequency, so semantic parsing above is primary.  Multiple
    # valid hypotheses can only arise in that latter two-color case.
    if len(hypotheses) > 1:
        minimum = min(item[0] for item in hypotheses)
        hypotheses = [item for item in hypotheses if item[0] == minimum]
    assert len(hypotheses) == 1, "background interpretation must be unique"
    _, background, blocks = hypotheses[0]
    return n, background, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Relocate every one-hole block to the array cell named by its hole.

    Input structure:
    - Green one-cell separator lines divide a square canvas into an n by n array
      of n by n cells, where n is 3 or 4.
    - Each occupied cell contains a solid single-colored n by n block with exactly
      one background hole; empty cells are entirely background.
    - A hole's local (row, column) coordinate is an address, and all addresses are
      distinct and form a permutation of the occupied cell coordinates.

    Rule:
    - Move each complete block, including its hole, into the cell whose array
      coordinate equals that hole's local coordinate.  Perform all moves
      simultaneously, keep fixed-point blocks in place, clear vacated cells to
      background, and leave every green separator unchanged.

    Key insight:
    - The hole is an address label rather than damage: after the unique
      permutation, every block's local hole coordinate truthfully names the cell
      containing that block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, background, blocks = _analyze(g)

    out = g.copy()
    for tile_r in range(n):
        for tile_c in range(n):
            out[_cell_slice(n, (tile_r, tile_c))] = background

    # Read every source from the unchanged input and write it to its address, so
    # cycles are simultaneous rather than depending on iteration order.
    for _source, (_color, address, tile) in blocks.items():
        out[_cell_slice(n, address)] = tile
    return out


def _permutation_with_fixed_points(
    positions: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[tuple[tuple[int, int], ...]]]:
    """Make fixed points plus one or two nontrivial cycles, one of length >= 3."""
    shuffled = positions.copy()
    _rng.shuffle(shuffled)
    max_fixed = len(shuffled) - 3
    if max_fixed == 1 or _rng.random() < 0.45:
        fixed_count = 1
    else:
        fixed_count = int(_rng.integers(2, max_fixed + 1))

    fixed = shuffled[:fixed_count]
    moving = shuffled[fixed_count:]
    cycles: list[tuple[tuple[int, int], ...]] = [(cell,) for cell in fixed]

    if len(moving) >= 5 and _rng.random() < 0.4:
        split = int(_rng.integers(2, len(moving) - 2))
        moving_cycles = [moving[:split], moving[split:]]
    else:
        moving_cycles = [moving]
    cycles.extend(tuple(cycle) for cycle in moving_cycles)

    address_of: dict[tuple[int, int], tuple[int, int]] = {}
    for cycle in cycles:
        if len(cycle) == 1:
            address_of[cycle[0]] = cycle[0]
        else:
            for index, source in enumerate(cycle):
                address_of[source] = cycle[(index + 1) % len(cycle)]
    return address_of, cycles


def _render_pair(
    n: int,
    background: int,
    colors: dict[tuple[int, int], int],
    address_of: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    side = n * n + n - 1
    inp = np.full((side, side), background, dtype=int)
    out = np.full((side, side), background, dtype=int)
    separators = _separator_mask(side, n)
    inp[separators] = GREEN
    out[separators] = GREEN

    for source, address in address_of.items():
        tile = np.full((n, n), colors[source], dtype=int)
        tile[address] = background
        inp[_cell_slice(n, source)] = tile
        out[_cell_slice(n, address)] = tile
    return inp, out


def _semantically_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    n: int,
    background: int,
    colors: dict[tuple[int, int], int],
    address_of: dict[tuple[int, int], tuple[int, int]],
    cycles: list[tuple[tuple[int, int], ...]],
) -> bool:
    """Check the rule, branch coverage, uniqueness, and a counterfactual label change."""
    try:
        actual = main(inp)
        if not np.array_equal(actual, expected) or np.array_equal(inp, expected):
            return False

        parsed_n, parsed_bg, parsed = _analyze(inp)
        if parsed_n != n or parsed_bg != background:
            return False
        if set(parsed) != set(address_of):
            return False
        if {cell for cell, dst in address_of.items() if cell == dst} == set():
            return False
        if not any(len(cycle) >= 3 for cycle in cycles):
            return False

        # Output self-consistency: every surviving block's hole names its own cell.
        _, output_bg, output_blocks = _analyze(expected)
        if output_bg != background:
            return False
        if any(cell != record[1] for cell, record in output_blocks.items()):
            return False

        # Counterexample to shortcuts that ignore the hole labels: exchange two
        # differently colored moving labels.  The occupied cells stay identical,
        # but the correct output must change.
        moving = [cell for cell, dst in address_of.items() if cell != dst]
        pair = next(
            (
                (a, b)
                for i, a in enumerate(moving)
                for b in moving[i + 1 :]
                if colors[a] != colors[b]
            ),
            None,
        )
        if pair is None:
            return False
        a, b = pair
        alternate = inp.copy()
        for source, new_address in ((a, address_of[b]), (b, address_of[a])):
            tile = np.full((n, n), colors[source], dtype=int)
            tile[new_address] = background
            alternate[_cell_slice(n, source)] = tile
        alternate_out = main(alternate)
        if np.array_equal(alternate_out, expected):
            return False
    except (AssertionError, ValueError, StopIteration):
        return False
    return True


def _gen_attempt(n: int) -> tuple[np.ndarray, np.ndarray] | None:
    all_positions = [(r, c) for r in range(n) for c in range(n)]
    maximum = n * n - 2  # Keep visible empty cells to disambiguate background.
    minimum = 4 if n == 3 else 5
    count = int(_rng.integers(minimum, maximum + 1))
    chosen_indices = _rng.choice(n * n, size=count, replace=False)
    positions = [all_positions[int(index)] for index in chosen_indices]

    address_of, cycles = _permutation_with_fixed_points(positions)
    moving = [cell for cell in positions if address_of[cell] != cell]

    palette = [color for color in range(10) if color != GREEN]
    background = int(_rng.choice(palette))
    block_choices = [color for color in palette if color != background]
    palette_size = int(_rng.integers(2, min(4, count) + 1))
    block_palette = [int(color) for color in _rng.choice(block_choices, size=palette_size, replace=False)]

    color_values = [block_palette[index % palette_size] for index in range(count)]
    _rng.shuffle(color_values)
    colors = {cell: int(color_values[index]) for index, cell in enumerate(positions)}
    # Guarantee that the counterfactual label swap is visibly distinguishable.
    colors[moving[0]] = block_palette[0]
    colors[moving[1]] = block_palette[1]

    inp, expected = _render_pair(n, background, colors, address_of)
    if not _semantically_valid(inp, expected, n, background, colors, address_of, cycles):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with both fixed and cyclic moves."""
    global _generation_count
    # Alternation guarantees that even a three-example sample covers n=3 and n=4;
    # all remaining structure, colors, positions, counts, and cycles stay random.
    n = 3 + (_generation_count % 2)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(n)
        if result is None:
            continue
        inp, expected = result
        output_key = (tuple(map(int, expected.shape)), expected.tobytes())
        if output_key in _seen_outputs:
            continue
        _seen_outputs.add(output_key)
        _generation_count += 1
        return inp
    raise RuntimeError("failed to generate a valid one-hole relocation input")
