import random
from collections import Counter, deque

import numpy as np


GREY = 5
BLACK = 0
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

TEMPLATE_RETRIES = 300
FRAGMENT_RETRIES = 300
PLACEMENT_RETRIES = 400
GENERATION_RETRIES = 300
WORK_SHAPE = (30, 30)


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[Coord] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _offset_graph_neighbors(cell: Coord, shape_cells: frozenset[Coord]) -> list[Coord]:
    r, c = cell
    return [(nr, nc) for nr, nc in _neighbors8(r, c) if (nr, nc) in shape_cells]


def _parse_fragment(grid: np.ndarray, comp: list[Coord], bg: int) -> tuple[Coord, int, frozenset[Coord]]:
    grey_cells = [(r, c) for r, c in comp if int(grid[r, c]) == GREY]
    assert len(grey_cells) == 1, "each fragment must contain exactly one grey anchor"
    anchor = grey_cells[0]

    colors = {int(grid[r, c]) for r, c in comp if int(grid[r, c]) not in {bg, GREY}}
    assert len(colors) == 1, "each fragment must have exactly one visible color"
    color = colors.pop()

    ar, ac = anchor
    offsets = frozenset((r - ar, c - ac) for r, c in comp)
    assert (0, 0) in offsets
    return anchor, color, offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden template by grey-anchor alignment, then complete every fragment into that template.

    Input structure:
    - A black background contains several separated monochrome fragments, each with exactly one grey anchor pixel.
    - Every fragment is a translated incomplete copy of one shared template, with all grey anchors occupying the same template cell.
    - No input fragment is complete, but the union of all anchor-aligned fragments covers the full template.

    Rule:
    - Split the non-background pixels into connected fragments.
    - For each fragment, treat its grey cell as the origin and record all fragment cell offsets relative to that anchor.
    - Take the union of those offsets to recover the full template, then stamp that full offset set back at every anchor using the fragment's own color, including the anchor cell.

    Key insight:
    - The largest visible fragment is still incomplete; only the union of all fragments after anchor registration reveals the template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _components(g != bg)
    assert comps, "expected at least one fragment"

    fragments = [_parse_fragment(g, comp, bg) for comp in comps]
    template = frozenset().union(*(offsets for _anchor, _color, offsets in fragments))
    assert template, "template must be nonempty"

    out = np.full_like(g, bg)
    painted: set[Coord] = set()
    for anchor, color, _offsets in fragments:
        ar, ac = anchor
        cells = {(ar + dr, ac + dc) for dr, dc in template}
        for r, c in cells:
            assert _in_bounds(g.shape, r, c), "completed template must stay in bounds"
        assert not (cells & painted), "completed fragments must not overlap"
        for r, c in cells:
            out[r, c] = color
        painted |= cells

    return out


def _choose_anchor(cells: set[Coord]) -> Coord:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    interior = [cell for cell in cells if r0 < cell[0] < r1 and c0 < cell[1] < c1]
    if interior:
        return random.choice(interior)

    semi_interior = [cell for cell in cells if r0 < cell[0] < r1 or c0 < cell[1] < c1]
    if semi_interior:
        return random.choice(semi_interior)

    degree_two = [cell for cell in cells if len(_offset_graph_neighbors(cell, frozenset(cells))) >= 2]
    if degree_two:
        return random.choice(degree_two)

    return random.choice(sorted(cells))


def _generate_template() -> frozenset[Coord] | None:
    for _ in range(TEMPLATE_RETRIES):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = random.randint(6, min(h * w - 1, 10))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}

        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue
        if len({r for r, _c in cells}) < 2 or len({c for _r, c in cells}) < 2:
            continue
        if len(cells) == h * w:
            continue

        anchor = _choose_anchor(cells)
        ar, ac = anchor
        offsets = frozenset((r - ar, c - ac) for r, c in cells)
        drs = [dr for dr, _dc in offsets]
        dcs = [dc for _dr, dc in offsets]
        if not ((min(drs) < 0 < max(drs)) or (min(dcs) < 0 < max(dcs))):
            continue
        return offsets
    return None


def _path_in_shape(shape_cells: frozenset[Coord], start: Coord, goal: Coord) -> list[Coord]:
    q = deque([start])
    parent = {start: None}
    while q:
        cell = q.popleft()
        if cell == goal:
            break
        for nxt in _offset_graph_neighbors(cell, shape_cells):
            if nxt not in parent:
                parent[nxt] = cell
                q.append(nxt)
    assert goal in parent, "shape must be connected"

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _grow_subset(shape_cells: frozenset[Coord], initial: set[Coord], target_size: int) -> frozenset[Coord]:
    subset = set(initial)
    frontier = {
        nxt
        for cell in subset
        for nxt in _offset_graph_neighbors(cell, shape_cells)
        if nxt not in subset
    }
    while len(subset) < target_size:
        assert frontier, "connected growth frontier unexpectedly empty"
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        subset.add(cell)
        for nxt in _offset_graph_neighbors(cell, shape_cells):
            if nxt not in subset:
                frontier.add(nxt)
    return frozenset(subset)


def _partition_nonempty(items: list[Coord], k: int) -> list[list[Coord]]:
    random.shuffle(items)
    groups = [[] for _ in range(k)]
    for idx, cell in enumerate(items):
        groups[idx % k].append(cell)
    return groups


def _generate_fragments(shape_cells: frozenset[Coord]) -> list[frozenset[Coord]] | None:
    non_anchor = [cell for cell in shape_cells if cell != (0, 0)]
    area = len(shape_cells)
    min_size = 4 if area >= 7 else 3

    for _ in range(FRAGMENT_RETRIES):
        k = random.randint(3, min(5, len(non_anchor)))
        groups = _partition_nonempty(non_anchor[:], k)
        fragments: list[frozenset[Coord]] = []
        ok = True

        for group in groups:
            initial = {(0, 0)}
            for cell in group:
                initial.update(_path_in_shape(shape_cells, (0, 0), cell))

            if len(initial) >= area:
                ok = False
                break

            lo = max(min_size, len(initial))
            hi = area - 1
            if lo > hi:
                ok = False
                break

            target = random.randint(lo, hi)
            fragments.append(_grow_subset(shape_cells, initial, target))

        if not ok:
            continue

        if frozenset().union(*fragments) != shape_cells:
            continue
        if any(len(fragment) == area for fragment in fragments):
            continue
        if len({tuple(sorted(fragment)) for fragment in fragments}) == 1:
            continue
        return fragments

    return None


def _stamp_full(anchor: Coord, offsets: frozenset[Coord]) -> set[Coord]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in offsets}


def _dilate_cells(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
        out.add((r, c))
    return out


def _place_anchors(offsets: frozenset[Coord], count: int) -> list[Coord] | None:
    h, w = WORK_SHAPE
    min_dr = min(dr for dr, _dc in offsets)
    max_dr = max(dr for dr, _dc in offsets)
    min_dc = min(dc for _dr, dc in offsets)
    max_dc = max(dc for _dr, dc in offsets)

    row_lo = -min_dr
    row_hi = h - 1 - max_dr
    col_lo = -min_dc
    col_hi = w - 1 - max_dc

    anchors: list[Coord] = []
    forbidden: set[Coord] = set()
    for _ in range(PLACEMENT_RETRIES):
        if len(anchors) == count:
            return anchors
        anchor = (
            random.randint(row_lo, row_hi),
            random.randint(col_lo, col_hi),
        )
        full_cells = _stamp_full(anchor, offsets)
        if any(cell in forbidden for cell in full_cells):
            continue
        anchors.append(anchor)
        forbidden |= _dilate_cells(full_cells, WORK_SHAPE)

    return None


def _crop_box(full_cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in full_cells]
    cols = [c for _r, c in full_cells]
    top = min(rows)
    bottom = max(rows)
    left = min(cols)
    right = max(cols)

    margin_top = random.randint(0, min(3, top))
    margin_left = random.randint(0, min(3, left))
    margin_bottom = random.randint(0, min(3, WORK_SHAPE[0] - 1 - bottom))
    margin_right = random.randint(0, min(3, WORK_SHAPE[1] - 1 - right))
    return (
        top - margin_top,
        bottom + margin_bottom,
        left - margin_left,
        right + margin_right,
    )


def _generate_one() -> np.ndarray | None:
    template = _generate_template()
    if template is None:
        return None

    fragments = _generate_fragments(template)
    if fragments is None:
        return None

    anchors = _place_anchors(template, len(fragments))
    if anchors is None:
        return None

    colors = random.sample(COLOR_POOL, len(fragments))
    canvas = np.full(WORK_SHAPE, BLACK, dtype=int)
    all_full_cells: set[Coord] = set()

    for anchor, fragment, color in zip(anchors, fragments, colors):
        full_cells = _stamp_full(anchor, template)
        all_full_cells |= full_cells
        ar, ac = anchor
        canvas[ar, ac] = GREY
        for dr, dc in fragment:
            if (dr, dc) == (0, 0):
                continue
            canvas[ar + dr, ac + dc] = color

    top, bottom, left, right = _crop_box(all_full_cells)
    grid = canvas[top : bottom + 1, left : right + 1].copy()

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ghost-template input")
