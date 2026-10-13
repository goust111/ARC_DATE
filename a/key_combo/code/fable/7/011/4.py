"""ARC task: turn a color-pointer chain of hollow frames into nested rings."""

from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 200


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components using the shared ARC (row, column) convention."""
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(cells)
    return components


def _frame_from_component(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None
    perimeter = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return (r0, c0, r1, c1) if set(cells) == perimeter else None


def _read_frames(grid: np.ndarray) -> dict[int, dict[str, object]]:
    """Find complete monochrome rectangular perimeters, ignoring isolated pointers."""
    frames: dict[int, dict[str, object]] = {}
    for color in (int(x) for x in np.unique(grid) if int(x) != BACKGROUND):
        candidates = []
        for component in _components4(grid == color):
            box = _frame_from_component(component)
            if box is not None:
                candidates.append(box)
        assert len(candidates) == 1, f"color {color} must form exactly one hollow frame"
        frames[color] = {"box": candidates[0]}

    assert len(frames) >= 2, "expected a chain of at least two frames"
    frame_colors = set(frames)
    for color, frame in frames.items():
        r0, c0, r1, c1 = frame["box"]  # type: ignore[misc]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        values = [int(x) for x in interior.ravel() if int(x) != BACKGROUND]
        assert color not in values, "a hollow frame may not contain its own color"
        pointer_colors = set(values)
        assert len(pointer_colors) <= 1, "a frame interior has at most one pointer color"
        if pointer_colors:
            target = pointer_colors.pop()
            assert target in frame_colors and target != color, "pointer must name another frame"
            frame["target"] = target
            frame["count"] = len(values)
        else:
            frame["target"] = None
            frame["count"] = 0
    return frames


def _chain(frames: dict[int, dict[str, object]]) -> tuple[list[int], list[int]]:
    indegree = {color: 0 for color in frames}
    for frame in frames.values():
        target = frame["target"]
        if target is not None:
            indegree[int(target)] += 1
    heads = [color for color, degree in indegree.items() if degree == 0]
    assert len(heads) == 1, "the color references must have one unique head"
    assert all(degree == 1 for color, degree in indegree.items() if color != heads[0])

    colors: list[int] = []
    counts: list[int] = []
    current: int | None = heads[0]
    seen: set[int] = set()
    while current is not None:
        assert current not in seen, "the references may not contain a cycle"
        seen.add(current)
        colors.append(current)
        frame = frames[current]
        target = frame["target"]
        if target is not None:
            count = int(frame["count"])
            assert count > 0
            counts.append(count)
            current = int(target)
        else:
            current = None
    assert len(seen) == len(frames), "every frame must belong to the same chain"
    assert len(counts) == len(colors) - 1
    return colors, counts


def _nested_output(colors: list[int], counts: list[int]) -> np.ndarray:
    side = 2 * (1 + sum(counts[:-1])) + counts[-1]
    assert side <= 30, "nested result exceeds the ARC size limit"
    out = np.zeros((side, side), dtype=int)
    offset = 0
    thicknesses = [1, *counts[:-1], counts[-1]]
    for index, color in enumerate(colors):
        out[offset : side - offset, offset : side - offset] = color
        if index < len(colors) - 1:
            offset += thicknesses[index]
    assert side - 2 * offset == counts[-1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress a color-reference chain of hollow frames into concentric square rings.

    Input structure:
    - A black grid contains separated, differently colored hollow rectangular frames.
    - A non-tail frame contains one or more pixels of one other frame color strictly
      inside it; the tail frame has an empty interior.

    Rule:
    - Interpret the interior color of frame A as a pointer A→B and follow the unique
      head-to-tail chain. Draw the head as a one-cell outer ring. Each later non-tail
      color forms the next ring, whose thickness is its pixel count in its predecessor.
      Draw the tail as the central solid square with side equal to its incoming count.

    Key insight:
    - Pointer color determines nesting order and pointer count determines thickness;
      the frames' positions and rectangular sizes are decoys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors, counts = _chain(_read_frames(g))
    return _nested_output(colors, counts)


def _pointer_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    candidates = list(itertools.product(range(1, height - 1), range(1, width - 1)))
    valid = [
        combo
        for combo in itertools.combinations(candidates, count)
        if all(abs(r1 - r2) + abs(c1 - c2) > 1 for (r1, c1), (r2, c2) in itertools.combinations(combo, 2))
    ]
    assert valid
    return list(random.choice(valid))


def _oracle(colors: list[int], counts: list[int]) -> np.ndarray:
    """Independent ring renderer used by the generator's semantic checks."""
    side = 2 + 2 * sum(counts[:-1]) + counts[-1]
    expected = np.empty((side, side), dtype=int)
    start, end = 0, side
    for i, color in enumerate(colors[:-1]):
        thickness = 1 if i == 0 else counts[i - 1]
        expected[start : start + thickness, start:end] = color
        expected[end - thickness : end, start:end] = color
        expected[start:end, start : start + thickness] = color
        expected[start:end, end - thickness : end] = color
        start += thickness
        end -= thickness
    assert end - start == counts[-1]
    expected[start:end, start:end] = colors[-1]
    return expected


def _render_input(
    colors: list[int], counts: list[int], sizes: list[tuple[int, int]], spatial_order: list[int]
) -> np.ndarray:
    n = len(colors)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    height = rows * 9 + random.randint(1, 3)
    width = cols * 9 + random.randint(1, 3)
    base_r = random.randint(0, height - rows * 9)
    base_c = random.randint(0, width - cols * 9)
    grid = np.zeros((height, width), dtype=int)

    for slot, chain_index in enumerate(spatial_order):
        h, w = sizes[chain_index]
        slot_r, slot_c = divmod(slot, cols)
        r0 = base_r + slot_r * 9 + random.randint(1, 8 - h)
        c0 = base_c + slot_c * 9 + random.randint(1, 8 - w)
        color = colors[chain_index]
        grid[r0, c0 : c0 + w] = color
        grid[r0 + h - 1, c0 : c0 + w] = color
        grid[r0 : r0 + h, c0] = color
        grid[r0 : r0 + h, c0 + w - 1] = color
        if chain_index < n - 1:
            for dr, dc in _pointer_cells(h, w, counts[chain_index]):
                grid[r0 + dr, c0 + dc] = colors[chain_index + 1]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 6)
    colors = random.sample(PALETTE, n)

    counts = [random.randint(1, 3) for _ in range(n - 1)]
    counts[random.randrange(n - 1)] = 1
    counts[random.randrange(n - 1)] = random.choice((2, 3))
    if 2 * (1 + sum(counts[:-1])) + counts[-1] > 30:
        return None

    size_pool = [
        (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (5, 7),
        (6, 4), (6, 5), (6, 6), (6, 7), (7, 4), (7, 5), (7, 6), (7, 7),
    ]
    sizes = random.sample(size_pool, n)
    areas = [h * w for h, w in sizes]
    if areas == sorted(areas) or areas == sorted(areas, reverse=True):
        return None

    spatial_order = list(range(n))
    random.shuffle(spatial_order)
    if spatial_order in (list(range(n)), list(reversed(range(n)))):
        return None

    inp = _render_input(colors, counts, sizes, spatial_order)
    expected = _oracle(colors, counts)
    if not np.array_equal(main(inp), expected):
        return None

    # Semantic, rather than merely round-trip, checks: every encoded count matters,
    # and permuting all geometric size decoys leaves the independently derived answer.
    for i in range(len(counts)):
        changed = counts.copy()
        changed[i] = 2 if counts[i] == 1 else counts[i] - 1
        if _oracle(colors, changed).shape == expected.shape and np.array_equal(_oracle(colors, changed), expected):
            return None
    decoy = _render_input(colors, counts, sizes[1:] + sizes[:1], spatial_order[1:] + spatial_order[:1])
    if not np.array_equal(main(decoy), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate pointer chains with independent size decoys."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid pointer-chain instance")
