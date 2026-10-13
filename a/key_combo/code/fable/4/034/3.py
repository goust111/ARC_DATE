from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]


def _background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    largest = max(counts.values())
    winners = [color for color, count in counts.items() if count == largest]
    assert len(winners) == 1, "expected a unique most-common background color"
    return winners[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as arrays of (row, column) coords."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        todo = deque([(sr, sc)])
        cells: list[Coord] = []
        while todo:
            r, c = todo.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _normalize(coords: list[Coord] | tuple[Coord, ...] | np.ndarray) -> Shape:
    pts = [(int(r), int(c)) for r, c in coords]
    assert pts, "a sprite must contain at least one cell"
    min_r = min(r for r, _c in pts)
    min_c = min(c for _r, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _rotations(coords: list[Coord] | tuple[Coord, ...] | np.ndarray) -> tuple[Shape, ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    variants: set[Shape] = set()
    current = pts
    for _ in range(4):
        variants.add(_normalize(current))
        current = [(c, -r) for r, c in current]
    return tuple(sorted(variants))


def _variants(coords: list[Coord] | tuple[Coord, ...] | np.ndarray) -> tuple[Shape, ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    variants = set(_rotations(pts))
    variants.update(_rotations([(r, -c) for r, c in pts]))
    return tuple(sorted(variants))


def _canonical(coords: list[Coord] | tuple[Coord, ...] | np.ndarray) -> Shape:
    return min(_variants(coords))


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Find the color and bend (row, col) of the top/left-anchored L divider."""
    height, width = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        mask = grid == color
        for r in range(2, height):
            c = -1
            while c + 1 < width and mask[r, c + 1]:
                c += 1
            if c >= 3 and bool(mask[: r + 1, c].all()):
                candidates.append((color, r, c))
    assert len(candidates) == 1, "expected one L-shaped divider from the top edge to the left edge"
    return candidates[0]


def _legend_entries(grid: np.ndarray, bg: int, divider_r: int, divider_c: int) -> list[dict]:
    legend = grid[:divider_r, :divider_c]
    entries: list[dict] = []
    for color in map(int, np.unique(legend)):
        if color == bg:
            continue
        comps = _components(legend == color)
        assert len(comps) == 1, "each legend color must form exactly one sprite"
        coords = comps[0]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        entries.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "key": _canonical(coords),
            }
        )
    assert entries, "expected at least one legend sprite"
    entries.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    tops = [item["bbox"][0] for item in entries]
    assert len(tops) == len(set(tops)), "legend sprites must have an unambiguous top-to-bottom order"
    for upper, lower in zip(entries, entries[1:]):
        assert upper["bbox"][2] < lower["bbox"][0], "legend sprite bounding boxes must not overlap vertically"
    keys = [item["key"] for item in entries]
    assert len(keys) == len(set(keys)), "legend shapes must be unique under rotation and reflection"
    colors = [item["color"] for item in entries]
    assert len(colors) == len(set(colors)), "legend sprites must use distinct colors"
    return entries


def _query_entries(
    grid: np.ndarray,
    bg: int,
    divider_color: int,
    legend_entries: list[dict],
) -> list[dict]:
    template_colors = {int(item["color"]) for item in legend_entries}
    role_colors = set(map(int, np.unique(grid))) - {bg, divider_color} - template_colors
    assert len(role_colors) == 1, "expected exactly one placeholder query color"
    query_color = next(iter(role_colors))
    entries: list[dict] = []
    for coords in _components(grid == query_color):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        local = _normalize(coords)
        entries.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": local,
                "key": _canonical(local),
            }
        )
    assert len(entries) == len(legend_entries), "legend and query sprites must correspond one-to-one"
    keys = [item["key"] for item in entries]
    assert len(keys) == len(set(keys)), "each legend shape must have exactly one query sprite"
    return entries


def _stack(shapes: list[Shape], colors: list[int], bg: int) -> np.ndarray:
    assert shapes and len(shapes) == len(colors)
    sizes = [_shape_size(shape) for shape in shapes]
    out = np.full((sum(h for h, _w in sizes), max(w for _h, w in sizes)), bg, dtype=int)
    row = 0
    for shape, color, (height, _width) in zip(shapes, colors, sizes):
        rr, cc = np.asarray(shape, dtype=int).T
        out[row + rr, cc] = int(color)
        row += height
    assert max(out.shape) <= 30, "stacked output exceeds the ARC 30x30 limit"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Match placeholder sprites to a vertical legend, recolor them, and stack them in legend order.

    Input structure:
    - A dominant background contains an L-shaped divider anchored to the top and left edges.
    - Distinctly colored, vertically ordered template sprites lie inside the upper-left legend region.
    - Outside the divider is one placeholder-colored query sprite per template, separated from the others.

    Rule:
    - Match every query to its unique legend template under rotation and reflection equivalence.
    - Following the legend's top-to-bottom order, crop each matched query in its original orientation,
      recolor it with the template color, and stack the crops vertically with touching bounding boxes and left edges.
    - Fill unused cells with the input background; the output width is the widest query bounding box.

    Key insight:
    - Shape equivalence selects the dictionary entry, but the legend supplies color and output order while each
      scattered query supplies the orientation that must be preserved in the compact stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2-D grid"
    assert max(g.shape) <= 30, "input exceeds the ARC 30x30 limit"

    bg = _background(g)
    divider_color, divider_r, divider_c = _find_divider(g, bg)
    legend = _legend_entries(g, bg, divider_r, divider_c)
    queries = _query_entries(g, bg, divider_color, legend)
    query_by_key = {item["key"]: item for item in queries}

    ordered_shapes: list[Shape] = []
    ordered_colors: list[int] = []
    for item in legend:
        key = item["key"]
        assert key in query_by_key, "every legend shape must have one matching query"
        ordered_shapes.append(query_by_key[key]["shape"])
        ordered_colors.append(int(item["color"]))
    return _stack(ordered_shapes, ordered_colors, bg)


# A small catalog of connected free polyominoes.  Canonicalization below removes
# accidental duplicates, while transformations still randomize both legend and query orientations.
_BASE_SHAPES: tuple[Shape, ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)),
    ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)),
)


def _catalog() -> list[Shape]:
    by_key: dict[Shape, Shape] = {}
    for shape in _BASE_SHAPES:
        by_key.setdefault(_canonical(shape), _normalize(shape))
    return list(by_key.values())


_SHAPE_CATALOG = _catalog()


def _is_chiral(shape: Shape) -> bool:
    return bool(set(_variants(shape)) - set(_rotations(shape)))


def _patch(shape: Shape, color: int, bg: int) -> np.ndarray:
    height, width = _shape_size(shape)
    patch = np.full((height, width), bg, dtype=int)
    rr, cc = np.asarray(shape, dtype=int).T
    patch[rr, cc] = int(color)
    return patch


def _bbox_gap_ok(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for er0, ec0, er1, ec1 in boxes:
        dr = er0 - r1 - 1 if r1 < er0 else r0 - er1 - 1 if er1 < r0 else 0
        dc = ec0 - c1 - 1 if c1 < ec0 else c0 - ec1 - 1 if ec1 < c0 else 0
        if max(dr, dc) < 1:
            return False
    return True


def _query_positions(
    shape: Shape,
    grid_shape: tuple[int, int],
    divider_r: int,
    divider_c: int,
    existing: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    height, width = _shape_size(shape)
    grid_h, grid_w = grid_shape
    regions = (
        (0, divider_r - 1, divider_c + 2, grid_w - 1),
        (divider_r + 2, grid_h - 1, 0, grid_w - 1),
    )
    choices: list[tuple[int, int, int, int]] = []
    for min_r, max_r, min_c, max_c in regions:
        for r0 in range(min_r, max_r - height + 2):
            for c0 in range(min_c, max_c - width + 2):
                box = (r0, c0, r0 + height - 1, c0 + width - 1)
                if _bbox_gap_ok(box, existing):
                    choices.append(box)
    return choices


def _render_stack_by_indices(
    query_shapes: list[Shape], colors: list[int], order: list[int], bg: int
) -> np.ndarray:
    return _stack([query_shapes[i] for i in order], [colors[i] for i in order], bg)


def _gen_attempt() -> np.ndarray | None:
    num_sprites = random.randint(3, 5)
    chiral_shapes = [shape for shape in _SHAPE_CATALOG if _is_chiral(shape)]
    if not chiral_shapes:
        raise RuntimeError("generator catalog needs a chiral shape")

    chosen = [random.choice(chiral_shapes)]
    remaining = [shape for shape in _SHAPE_CATALOG if _canonical(shape) != _canonical(chosen[0])]
    chosen.extend(random.sample(remaining, num_sprites - 1))
    random.shuffle(chosen)
    chiral_index = next(i for i, shape in enumerate(chosen) if _is_chiral(shape))

    role_colors = random.sample(range(10), num_sprites + 3)
    bg, divider_color, query_color = role_colors[:3]
    template_colors = role_colors[3:]

    divider_c = random.randint(7, 10)
    legend_shapes: list[Shape] = []
    for shape in chosen:
        choices = [v for v in _variants(shape) if _shape_size(v)[0] <= 4 and _shape_size(v)[1] <= divider_c - 2]
        if not choices:
            return None
        legend_shapes.append(random.choice(choices))

    divider_r = 2 + sum(_shape_size(shape)[0] for shape in legend_shapes) + num_sprites - 1
    if divider_r > 26:
        return None
    min_height = max(22, divider_r + 3)
    if min_height > 30:
        return None
    grid_h = random.randint(min_height, 30)
    grid_w = random.randint(max(22, divider_c + 8), 30)

    query_shapes: list[Shape] = []
    for i, shape in enumerate(legend_shapes):
        choices = list(_variants(shape))
        if i == chiral_index:
            reflection_only = list(set(choices) - set(_rotations(shape)))
            if not reflection_only:
                return None
            query_shapes.append(random.choice(reflection_only))
        else:
            query_shapes.append(random.choice(choices))

    # Make rotation visibly necessary even when the forced reflected sprite has a square bbox.
    if all(query_shapes[i] == legend_shapes[i] for i in range(num_sprites)):
        for i, shape in enumerate(legend_shapes):
            alternatives = [v for v in _variants(shape) if v != shape]
            if alternatives:
                query_shapes[i] = random.choice(alternatives)
                break
    if all(query_shapes[i] == legend_shapes[i] for i in range(num_sprites)):
        return None
    if sum(_shape_size(shape)[0] for shape in query_shapes) > 30:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    row = 1
    for shape, color in zip(legend_shapes, template_colors):
        height, width = _shape_size(shape)
        col = random.randint(1, divider_c - width - 1)
        patch = _patch(shape, color, bg)
        grid[row : row + height, col : col + width][patch != bg] = color
        row += height + 1
    assert row == divider_r
    grid[divider_r, : divider_c + 1] = divider_color
    grid[: divider_r + 1, divider_c] = divider_color

    query_boxes: list[tuple[int, int, int, int]] = []
    for shape in query_shapes:
        choices = _query_positions(shape, grid.shape, divider_r, divider_c, query_boxes)
        if not choices:
            return None
        box = random.choice(choices)
        r0, c0, r1, c1 = box
        patch = _patch(shape, query_color, bg)
        mask = patch != bg
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1][mask] != bg):
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1][mask] = query_color
        query_boxes.append(box)

    # Spatial reading order must contradict the legend order, making the legend's sequence indispensable.
    spatial_order = sorted(range(num_sprites), key=lambda i: (query_boxes[i][0], query_boxes[i][1]))
    if spatial_order == list(range(num_sprites)):
        return None

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = _render_stack_by_indices(query_shapes, template_colors, list(range(num_sprites)), bg)
    wrong_spatial = _render_stack_by_indices(query_shapes, template_colors, spatial_order, bg)
    wrong_legend_orientation = _stack(legend_shapes, template_colors, bg)
    wrong_no_recolor = _stack(query_shapes, [query_color] * num_sprites, bg)
    if np.array_equal(expected, wrong_spatial):
        return None
    if np.array_equal(expected, wrong_legend_orientation):
        return None
    if np.array_equal(expected, wrong_no_recolor):
        return None

    # Independent semantic checks: unique keys, one reflected-only match, and a hand-built oracle.
    if len({_canonical(shape) for shape in legend_shapes}) != num_sprites:
        return None
    if _canonical(query_shapes[chiral_index]) != _canonical(legend_shapes[chiral_index]):
        return None
    if query_shapes[chiral_index] in set(_rotations(legend_shapes[chiral_index])):
        return None
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate instance satisfying all solver assumptions."""
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-dictionary stack puzzle")
