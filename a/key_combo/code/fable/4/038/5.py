from __future__ import annotations

from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
_RNG = np.random.default_rng()
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MAX_GENERATION_ATTEMPTS = 300


def _background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _foreground_components(grid: np.ndarray, background: int) -> list[set[Coord]]:
    """Find 4-connected multicolor foreground components in (row, column) coordinates."""
    foreground = grid != int(background)
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[set[Coord]] = []
    height, width = grid.shape

    for start_r, start_c in np.argwhere(foreground):
        start = (int(start_r), int(start_c))
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        component: set[Coord] = set()
        while queue:
            r, c = queue.pop()
            component.add((r, c))
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and foreground[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(component)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Align all partial copies by their common anchor pixels, union their shapes, and complete every copy in its own color.

    Input structure:
    - A uniquely dominant background contains at least three separated, 4-connected partial copies.
    - Every copy uses one private body color and contains exactly one pixel of a common anchor color.
    - The anchor occupies the same template coordinate in every copy; the aligned partial masks are distinct and complementary.

    Rule:
    - Translate each copy's occupied coordinates so that its anchor is at relative coordinate (0, 0), and take their union as the hidden template.
    - At every original anchor position, paint all missing template coordinates with that copy's private color.
    - Preserve every anchor in the common anchor color and leave all unrelated background unchanged.

    Key insight:
    - No complete template is visible: its coordinates are recoverable only by anchor-aligned union, and every partial copy contributes coordinates absent from all the others.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    background = _background(g)
    components = _foreground_components(g, background)
    assert len(components) >= 3, "expected at least three partial copies"

    color_sets = [{int(g[r, c]) for r, c in component} for component in components]
    common_colors = set.intersection(*color_sets)
    assert len(common_colors) == 1, "copies must share exactly one anchor color"
    anchor_color = next(iter(common_colors))

    copies = []
    private_colors: list[int] = []
    for component, colors in zip(components, color_sets):
        assert len(colors) == 2 and anchor_color in colors, "each copy needs one anchor and one body color"
        anchors = [(r, c) for r, c in component if int(g[r, c]) == anchor_color]
        assert len(anchors) == 1, "each copy must contain exactly one anchor pixel"
        private_color = next(iter(colors - {anchor_color}))
        anchor_r, anchor_c = anchors[0]
        offsets = {(r - anchor_r, c - anchor_c) for r, c in component}
        assert (0, 0) in offsets and len(offsets) >= 3, "partial copy is too small"
        copies.append((component, (anchor_r, anchor_c), private_color, offsets))
        private_colors.append(private_color)

    assert len(set(private_colors)) == len(private_colors), "body colors must be different"
    template = set().union(*(copy[3] for copy in copies))
    assert len(template) > max(len(copy[3]) for copy in copies), "no complete template may be visible"

    # Semantic promise of the puzzle: removing any copy loses template information.
    for index, copy in enumerate(copies):
        union_without = set().union(*(other[3] for j, other in enumerate(copies) if j != index))
        assert copy[3] - union_without, "every copy must make an indispensable contribution"

    out = g.copy()
    height, width = g.shape
    for component, (anchor_r, anchor_c), private_color, _offsets in copies:
        for dr, dc in template:
            r, c = anchor_r + dr, anchor_c + dc
            assert 0 <= r < height and 0 <= c < width, "completed copy would leave the grid"
            if int(g[r, c]) != background:
                assert (r, c) in component, "completed copies must not collide"
            out[r, c] = anchor_color if (dr, dc) == (0, 0) else private_color
    return out


def _connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    reached = {next(iter(cells))}
    stack = list(reached)
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS:
            neighbor = (r + dr, c + dc)
            if neighbor in cells and neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    return reached == cells


def _shortest_path(allowed: set[Coord], starts: set[Coord], goal: Coord) -> set[Coord]:
    """Return one shortest path from any start to goal, including both ends."""
    assert starts and starts <= allowed and goal in allowed
    queue = deque(starts)
    parent: dict[Coord, Coord | None] = {cell: None for cell in starts}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = {cell}
            while parent[cell] is not None:
                cell = parent[cell]  # type: ignore[assignment]
                path.add(cell)
            return path
        r, c = cell
        for dr, dc in _NEIGHBORS:
            neighbor = (r + dr, c + dc)
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    raise AssertionError("allowed template region must connect starts to goal")


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _make_template(rng: np.random.Generator, copy_count: int):
    """Create a repeated, mirror-symmetric comb whose off-axis anchor breaks the symmetry."""
    height = int(rng.integers(5, 8))
    width = int(rng.choice((5, 7)))
    center = width // 2
    parity_options = [parity for parity in (0, 1) if len(range(parity, height, 2)) >= 3]
    parity = int(rng.choice(parity_options))
    bar_rows = list(range(parity, height, 2))

    template: set[Coord] = set()
    for r in range(height):
        radius = center if r in bar_rows else int(rng.integers(0, 2))
        for c in range(center - radius, center + radius + 1):
            template.add((r, c))

    anchor_row = int(rng.choice(bar_rows))
    anchor_col = center + int(rng.choice((-1, 1)))
    anchor = (anchor_row, anchor_col)
    leaves = {(r, 0) for r in bar_rows} | {(r, width - 1) for r in bar_rows}
    assert len(leaves) >= copy_count and anchor not in leaves and _connected(template)

    transposed = bool(rng.integers(0, 2))
    if transposed:
        template = {(c, r) for r, c in template}
        anchor = (anchor[1], anchor[0])
        leaves = {(c, r) for r, c in leaves}
        height, width = width, height

    # Express all template coordinates relative to the anchor, matching main's reference frame.
    ar, ac = anchor
    relative = {(r - ar, c - ac) for r, c in template}
    relative_leaves = {(r - ar, c - ac) for r, c in leaves}
    return relative, (height, width), relative_leaves, transposed


def _assign_exclusive_leaves(
    rng: np.random.Generator, leaves: set[Coord], copy_count: int, transposed: bool
) -> list[set[Coord]]:
    """Give each copy a private boundary witness and force two unequal fragment bboxes."""
    groups = [set() for _ in range(copy_count)]
    if transposed:
        low = [cell for cell in leaves if cell[0] == min(r for r, _ in leaves)]
        high = [cell for cell in leaves if cell[0] == max(r for r, _ in leaves)]
    else:
        low = [cell for cell in leaves if cell[1] == min(c for _, c in leaves)]
        high = [cell for cell in leaves if cell[1] == max(c for _, c in leaves)]

    first = low[int(rng.integers(len(low)))]
    second = high[int(rng.integers(len(high)))]
    groups[0].add(first)
    groups[1].add(second)
    remaining = list(leaves - {first, second})
    rng.shuffle(remaining)
    for index in range(2, copy_count):
        groups[index].add(remaining.pop())

    for leaf in remaining:
        # Copy 0 never reaches the high extreme; copy 1 never reaches the low extreme.
        if leaf in high:
            choices = list(range(1, copy_count))
        elif leaf in low:
            choices = [0] + list(range(2, copy_count))
        else:
            choices = list(range(copy_count))
        groups[int(rng.choice(choices))].add(leaf)
    return groups


def _grow_masks(
    rng: np.random.Generator,
    template: set[Coord],
    exclusive_groups: list[set[Coord]],
) -> list[set[Coord]]:
    all_exclusive = set().union(*exclusive_groups)
    masks: list[set[Coord]] = []
    allowed_regions: list[set[Coord]] = []

    for owned in exclusive_groups:
        allowed = template - (all_exclusive - owned)
        mask = {(0, 0)}
        for witness in owned:
            mask |= _shortest_path(allowed, mask, witness)

        low_target = max(len(mask), int(np.ceil(0.42 * len(template))))
        high_target = max(low_target, min(len(allowed), int(np.ceil(0.63 * len(template)))))
        target = int(rng.integers(low_target, high_target + 1))
        while len(mask) < target:
            frontier = {
                (r + dr, c + dc)
                for r, c in mask
                for dr, dc in _NEIGHBORS
                if (r + dr, c + dc) in allowed and (r + dr, c + dc) not in mask
            }
            assert frontier
            mask.add(list(frontier)[int(rng.integers(len(frontier)))])
        masks.append(mask)
        allowed_regions.append(allowed)

    # Assign every still-unseen nonexclusive template cell to one eligible copy, adding a path if needed.
    missing = list(template - set().union(*masks))
    rng.shuffle(missing)
    for cell in missing:
        eligible = [i for i, allowed in enumerate(allowed_regions) if cell in allowed]
        index = int(rng.choice(eligible))
        masks[index] |= _shortest_path(allowed_regions[index], masks[index], cell)
    return masks


def _layout_origins(
    rng: np.random.Generator, count: int, sprite_h: int, sprite_w: int
) -> tuple[tuple[int, int], list[Coord]]:
    layouts = []
    for columns in range(2, min(3, count) + 1):
        rows = (count + columns - 1) // columns
        for row_gap in (2, 3):
            for col_gap in (2, 3):
                content_h = rows * sprite_h + (rows - 1) * row_gap
                content_w = columns * sprite_w + (columns - 1) * col_gap
                if content_h + 2 <= 30 and content_w + 2 <= 30:
                    layouts.append((rows, columns, row_gap, col_gap, content_h, content_w))
    assert layouts
    rows, columns, row_gap, col_gap, content_h, content_w = layouts[int(rng.integers(len(layouts)))]
    grid_h = int(rng.integers(content_h + 2, min(30, content_h + 6) + 1))
    grid_w = int(rng.integers(content_w + 2, min(30, content_w + 6) + 1))
    top = int(rng.integers(1, grid_h - content_h))
    left = int(rng.integers(1, grid_w - content_w))
    slots = [
        (top + r * (sprite_h + row_gap), left + c * (sprite_w + col_gap))
        for r in range(rows)
        for c in range(columns)
    ]
    chosen = rng.choice(len(slots), size=count, replace=False)
    origins = [slots[int(index)] for index in chosen]
    rng.shuffle(origins)
    return (grid_h, grid_w), origins


def _render_expected(
    shape: tuple[int, int],
    background: int,
    template: set[Coord],
    masks: list[set[Coord]],
    anchors: list[Coord],
    body_colors: list[int],
    anchor_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, background, dtype=int)
    expected = inp.copy()
    for mask, (anchor_r, anchor_c), body_color in zip(masks, anchors, body_colors):
        for dr, dc in template:
            value = anchor_color if (dr, dc) == (0, 0) else body_color
            expected[anchor_r + dr, anchor_c + dc] = value
        for dr, dc in mask:
            value = anchor_color if (dr, dc) == (0, 0) else body_color
            inp[anchor_r + dr, anchor_c + dc] = value
    return inp, expected


def _semantics_hold(
    inp: np.ndarray,
    expected: np.ndarray,
    template: set[Coord],
    masks: list[set[Coord]],
    exclusive_groups: list[set[Coord]],
    anchor_color: int,
    transposed: bool,
) -> bool:
    """Validate the intended rule independently of a mere generator/solver round trip."""
    if set().union(*masks) != template or any((0, 0) not in mask for mask in masks):
        return False
    if any(mask == template or not _connected(mask) for mask in masks):
        return False
    if len({_bbox(mask) for mask in masks}) < 2:
        return False
    if len({frozenset(template - mask) for mask in masks}) != len(masks):
        return False
    for i, (mask, owned) in enumerate(zip(masks, exclusive_groups)):
        others = set().union(*(other for j, other in enumerate(masks) if j != i))
        if not owned or not owned <= (mask - others):
            return False

    # The template has a visible mirror repetition, but its anchor is deliberately off that mirror axis.
    r0, c0, r1, c1 = _bbox(template)
    if transposed:
        mirrored = {(r0 + r1 - r, c) for r, c in template}
        anchor_off_axis = 2 * 0 != r0 + r1
    else:
        mirrored = {(r, c0 + c1 - c) for r, c in template}
        anchor_off_axis = 2 * 0 != c0 + c1
    if mirrored != template or not anchor_off_axis:
        return False
    if np.array_equal(inp, expected) or not np.array_equal(expected[inp != _background(inp)], inp[inp != _background(inp)]):
        return False
    if int(np.count_nonzero(inp == anchor_color)) != len(masks):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    copy_count = int(rng.integers(3, 6))
    palette = list(map(int, rng.choice(10, size=copy_count + 2, replace=False)))
    background, anchor_color = palette[:2]
    body_colors = palette[2:]

    template, (sprite_h, sprite_w), leaves, transposed = _make_template(rng, copy_count)
    exclusive_groups = _assign_exclusive_leaves(rng, leaves, copy_count, transposed)
    masks = _grow_masks(rng, template, exclusive_groups)
    shape, origins = _layout_origins(rng, copy_count, sprite_h, sprite_w)
    min_r, min_c, _max_r, _max_c = _bbox(template)
    anchors = [(top - min_r, left - min_c) for top, left in origins]
    order = list(map(int, rng.permutation(copy_count)))
    masks = [masks[i] for i in order]
    exclusive_groups = [exclusive_groups[i] for i in order]

    inp, expected = _render_expected(
        shape, background, template, masks, anchors, body_colors, anchor_color
    )
    try:
        if _background(inp) != background:
            return None
        if not _semantics_hold(
            inp, expected, template, masks, exclusive_groups, anchor_color, transposed
        ):
            return None
        if not np.array_equal(main(inp), expected):
            return None
    except (AssertionError, ValueError):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate anchor-aligned reconstruction instance."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-completion input")
