from __future__ import annotations

import random
from collections import deque

import numpy as np


PALETTE = list(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (row, column) coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(np.asarray(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _transform(mask: np.ndarray, index: int) -> np.ndarray:
    base = np.fliplr(mask) if index >= 4 else mask
    return np.rot90(base, index % 4).copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Canonical key under all rotations and reflections."""
    keys = []
    for index in range(8):
        changed = _transform(mask.astype(bool), index)
        keys.append((changed.shape, changed.tobytes()))
    return min(keys)


def _parse_legend(top: np.ndarray, bg: int, size: int) -> dict[tuple[tuple[int, int], bytes], int]:
    top_colors = set(map(int, np.unique(top))) - {bg}
    swatches: list[tuple[int, int]] = []

    for color in top_colors:
        for coords in _components(top == color):
            r0, c0, r1, c1 = _bbox(coords)
            if len(coords) == 4 and (r0, r1, c1 - c0 + 1) == (0, 1, 2):
                swatches.append((c0, color))

    swatches.sort()
    assert len(swatches) >= 2, "legend must contain at least two 2x2 color swatches"
    swatch_colors = [color for _c, color in swatches]
    assert len(swatch_colors) == len(set(swatch_colors)), "swatch colors must be distinct"

    neutral_candidates = top_colors - set(swatch_colors)
    assert len(neutral_candidates) == 1, "stencils must share one neutral drawing color"
    neutral = next(iter(neutral_candidates))

    stencils: list[tuple[int, int, np.ndarray]] = []
    for coords in _components(top == neutral):
        r0, c0, r1, c1 = _bbox(coords)
        mask = _mask_from_coords(coords)
        assert (r0, r1) == (0, size - 1), "each stencil must span the legend height"
        assert mask.shape == (size, size), "each stencil must have a square full-size bbox"
        stencils.append((c0, c1, mask))
    stencils.sort(key=lambda item: item[0])
    assert len(stencils) == len(swatches), "swatches and stencils must pair one-to-one"

    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    for i, ((swatch_c0, color), (stencil_c0, stencil_c1, mask)) in enumerate(zip(swatches, stencils)):
        assert swatch_c0 < stencil_c0, "each swatch must precede its paired stencil"
        if i + 1 < len(swatches):
            assert stencil_c1 < swatches[i + 1][0], "legend pairs must not interleave"
        key = _shape_key(mask)
        assert key not in mapping, "stencil shapes must be unique up to rotation/reflection"
        mapping[key] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Reverse-decode a block picture by translating invariant shapes through paired color swatches.

    Input structure:
    - A top legend contains left-to-right pairs of a 2x2 color swatch followed by a square
      neutral-color stencil; all stencils have one size and are inequivalent under rotation/reflection.
    - One or more blank rows separate the legend from a lower rectangle partitioned into equal
      square blocks. A block is empty or contains a monochrome transformed copy of one stencil.

    Rule:
    - Pair swatches and stencils by their left-to-right pair order. Split the lower rectangle using
      the stencil size, ignore every occupied block's drawing color, and match its binary shape
      to a stencil under all rotations and reflections. Emit the paired swatch color per block;
      emit the background color for an empty block.

    Key insight:
    - Shape, not the deliberately unrelated block color, is the index that reverse-compresses each
      large block to one output pixel; orientation is irrelevant but the visible legend pairing is not.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background(g)

    occupied_rows = np.any(g != bg, axis=1)
    row_runs = _runs(occupied_rows)
    assert len(row_runs) >= 2 and row_runs[0][0] == 0, "legend and body must be blank-row separated"
    legend_r1 = row_runs[0][1]
    size = legend_r1 + 1
    assert size >= 3
    top = g[:size, :]
    mapping = _parse_legend(top, bg, size)

    body_rows = np.flatnonzero(occupied_rows & (np.arange(g.shape[0]) > legend_r1))
    assert body_rows.size > 0
    body_r0, body_r1 = int(body_rows.min()), int(body_rows.max())
    assert body_r0 > legend_r1 + 1, "at least one blank separator row is required"

    lower = g[body_r0 : body_r1 + 1, :]
    body_coords = np.argwhere(lower != bg)
    assert body_coords.size > 0
    body_c0 = int(body_coords[:, 1].min())
    body_c1 = int(body_coords[:, 1].max())
    body = lower[:, body_c0 : body_c1 + 1]
    assert body.shape[0] % size == 0 and body.shape[1] % size == 0

    block_rows = body.shape[0] // size
    block_cols = body.shape[1] // size
    out = np.full((block_rows, block_cols), bg, dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            block = body[br * size : (br + 1) * size, bc * size : (bc + 1) * size]
            mask = block != bg
            if not np.any(mask):
                continue
            drawing_colors = set(map(int, np.unique(block[mask])))
            assert len(drawing_colors) == 1, "each occupied block must be monochrome"
            key = _shape_key(mask)
            assert key in mapping, "every occupied block shape must match exactly one stencil"
            out[br, bc] = mapping[key]
    return out


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = a
    cells = {(r, c)}
    while (r, c) != b:
        steps: list[tuple[int, int]] = []
        if r < b[0]:
            steps.append((1, 0))
        elif r > b[0]:
            steps.append((-1, 0))
        if c < b[1]:
            steps.append((0, 1))
        elif c > b[1]:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r, c = r + dr, c + dc
        cells.add((r, c))
    return cells


def _make_shape(size: int, area: int) -> np.ndarray | None:
    """Make a connected, full-bbox, deliberately asymmetric shape of exact area."""
    for _ in range(250):
        top = (0, random.randrange(size))
        bottom = (size - 1, random.randrange(size))
        cells = _random_path(top, bottom)
        for anchor in ((random.randrange(size), 0), (random.randrange(size), size - 1)):
            target = min(cells, key=lambda p: abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]))
            cells |= _random_path(anchor, target)
        if len(cells) > area:
            continue

        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        mask = np.zeros((size, size), dtype=bool)
        rr, cc = zip(*cells)
        mask[np.asarray(rr), np.asarray(cc)] = True
        if len({_transform(mask, i).tobytes() for i in range(8)}) == 8:
            return mask
    return None


def _make_shapes(count: int, size: int) -> list[np.ndarray] | None:
    for _ in range(40):
        area = random.randint(2 * size + 2, size * size - 3)
        shapes: list[np.ndarray] = []
        keys: set[tuple[tuple[int, int], bytes]] = set()
        for _index in range(count):
            shape = _make_shape(size, area)
            if shape is None:
                break
            key = _shape_key(shape)
            if key in keys:
                break
            keys.add(key)
            shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _legend_width(count: int, size: int) -> int:
    return count * (size + 5) - 2


VALID_SPECS = [
    (count, size, rows, cols)
    for count in (2, 3)
    for size in (4, 5, 6)
    for rows in range(2, 5)
    for cols in range(2, 6)
    if _legend_width(count, size) <= 30
    and size + 2 + rows * size <= 30
    and cols * size <= 30
    and rows * cols >= 2 * count + 1
]


def _sample_active(rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    total = rows * cols
    low = max(2 * count, int(np.ceil(0.55 * total)))
    if low > total - 1:
        return None
    active_count = random.randint(low, total - 1)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(80):
        active = random.sample(all_cells, active_count)
        aset = set(active)
        if (
            any(r == 0 for r, _c in aset)
            and any(r == rows - 1 for r, _c in aset)
            and any(c == 0 for _r, c in aset)
            and any(c == cols - 1 for _r, c in aset)
        ):
            random.shuffle(active)
            return active
    return None


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    swatches: list[int],
    swatch_cols: list[int],
    shapes: list[np.ndarray],
    active: list[tuple[int, int]],
    labels: dict[tuple[int, int], int],
    orientations: dict[tuple[int, int], int],
    block_colors: dict[tuple[int, int], int],
    decoys: list[int],
    body_r0: int,
    body_c0: int,
    size: int,
) -> bool:
    """Check rule semantics and anti-shortcut properties, not just generator closure."""
    try:
        if _background(grid) != bg or not np.array_equal(main(grid), expected):
            return False
        if set(map(int, np.unique(expected))) != {bg, *swatches}:
            return False
        if len({int(shape.sum()) for shape in shapes}) != 1:
            return False
        if len({_shape_key(shape) for shape in shapes}) != len(shapes):
            return False
        if any(len({_transform(shape, i).tobytes() for i in range(8)}) != 8 for shape in shapes):
            return False

        common = block_colors[active[0]]
        for label in range(len(shapes)):
            cells = [cell for cell in active if labels[cell] == label]
            if len({block_colors[cell] for cell in cells}) < 2:
                return False
            if not any(block_colors[cell] == common for cell in cells):
                return False
            if not any(1 <= orientations[cell] <= 3 for cell in cells):
                return False
            if not any(4 <= orientations[cell] <= 7 for cell in cells):
                return False
        if set(block_colors.values()) & set(swatches):
            return False
        if len(set(block_colors.values())) < 3:
            return False

        # Repainting every block changes visible input colors but must not alter output.
        recolored = grid.copy()
        for cell in active:
            r, c = cell
            block = recolored[
                body_r0 + r * size : body_r0 + (r + 1) * size,
                body_c0 + c * size : body_c0 + (c + 1) * size,
            ]
            old = block_colors[cell]
            new = next(color for color in decoys if color != old)
            block[block == old] = new
        if not np.array_equal(main(recolored), expected):
            return False

        # A further quarter-turn of one full-bbox block must preserve its decoded pixel.
        rotated = grid.copy()
        r, c = active[0]
        block = rotated[
            body_r0 + r * size : body_r0 + (r + 1) * size,
            body_c0 + c * size : body_c0 + (c + 1) * size,
        ]
        color = block_colors[(r, c)]
        turned = np.rot90(block != bg)
        block[:, :] = bg
        block[turned] = color
        if not np.array_equal(main(rotated), expected):
            return False

        # Cyclically changing visible swatches must cyclically change decoded colors.
        changed = grid.copy()
        color_map: dict[int, int] = {}
        for i, col in enumerate(swatch_cols):
            new_color = swatches[(i + 1) % len(swatches)]
            changed[0:2, col : col + 2] = new_color
            color_map[swatches[i]] = new_color
        changed_expected = np.full_like(expected, bg)
        for old, new in color_map.items():
            changed_expected[expected == old] = new
        if np.array_equal(changed_expected, expected):
            return False
        if not np.array_equal(main(changed), changed_expected):
            return False
    except (AssertionError, ValueError, StopIteration):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count, size, rows, cols = random.choice(VALID_SPECS)
    shapes = _make_shapes(count, size)
    if shapes is None:
        return None

    bg = random.choice(PALETTE)
    visible = random.sample([color for color in PALETTE if color != bg], count + 1)
    swatches = visible[:count]
    neutral = visible[-1]
    decoys = [color for color in PALETTE if color not in {bg, neutral, *swatches}]
    random.shuffle(decoys)
    if len(decoys) < 3:
        return None

    gap = random.randint(1, 2)
    legend_width = _legend_width(count, size)
    body_width = cols * size
    base_width = max(legend_width, body_width)
    width = base_width + random.randint(0, min(3, 30 - base_width))
    height = size + gap + rows * size
    grid = np.full((height, width), bg, dtype=int)

    swatch_cols: list[int] = []
    for i, (color, shape) in enumerate(zip(swatches, shapes)):
        pair_c0 = i * (size + 5)
        swatch_cols.append(pair_c0)
        grid[0:2, pair_c0 : pair_c0 + 2] = color
        stencil_c0 = pair_c0 + 3
        patch = grid[0:size, stencil_c0 : stencil_c0 + size]
        patch[shape] = neutral

    active = _sample_active(rows, cols, count)
    if active is None:
        return None
    labels: dict[tuple[int, int], int] = {}
    orientations: dict[tuple[int, int], int] = {}
    block_colors: dict[tuple[int, int], int] = {}
    for label in range(count):
        first, second = active[2 * label], active[2 * label + 1]
        labels[first] = labels[second] = label
        orientations[first] = random.randint(1, 3)
        orientations[second] = random.randint(4, 7)
        block_colors[first] = decoys[0]
        block_colors[second] = decoys[1 + label % (len(decoys) - 1)]
    for cell in active[2 * count :]:
        labels[cell] = random.randrange(count)
        orientations[cell] = random.randrange(8)
        block_colors[cell] = random.choice(decoys)

    # Ensure at least three visibly different decoy colors occur.
    for index, cell in enumerate(active[: min(3, len(active))]):
        if index == 0:
            block_colors[cell] = decoys[0]
        elif index == 1 and labels[cell] == labels[active[0]]:
            block_colors[cell] = decoys[1]
        elif index == 2:
            block_colors[cell] = decoys[2]

    # Reassert the two-color-per-shape and shared-color construction after the color coverage tweak.
    for label in range(count):
        pair = (active[2 * label], active[2 * label + 1])
        block_colors[pair[0]] = decoys[0]
        block_colors[pair[1]] = decoys[1 + label % (len(decoys) - 1)]
    if len(set(block_colors.values())) < 3 and len(active) > 2 * count:
        block_colors[active[-1]] = decoys[2]
    if len(set(block_colors.values())) < 3:
        # With count >= 2, the second mandatory colors can always supply a third color.
        block_colors[active[3]] = decoys[2]

    body_r0 = size + gap
    body_c0 = random.randint(0, width - body_width)
    expected = np.full((rows, cols), bg, dtype=int)
    for r, c in active:
        label = labels[(r, c)]
        mask = _transform(shapes[label], orientations[(r, c)])
        color = block_colors[(r, c)]
        block = grid[
            body_r0 + r * size : body_r0 + (r + 1) * size,
            body_c0 + c * size : body_c0 + (c + 1) * size,
        ]
        block[mask] = color
        expected[r, c] = swatches[label]

    if not _validate_semantics(
        grid,
        expected,
        bg,
        swatches,
        swatch_cols,
        shapes,
        active,
        labels,
        orientations,
        block_colors,
        decoys,
        body_r0,
        body_c0,
        size,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-to-color reverse-decoding puzzle")
