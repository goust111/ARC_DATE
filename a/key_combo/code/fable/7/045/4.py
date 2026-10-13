import random
from collections import Counter, defaultdict

import numpy as np


Coord = tuple[int, int]
Offset = tuple[int, int]

BG = 0
COLOR_POOL = list(range(1, 10))

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)

MASK_RADIUS = 3
MASK_AREA_RANGE = (10, 15)
MASK_RETRIES = 200

SOURCE_COUNT_CHOICES = (3, 4)
CLUSTER_RADIUS = 2
LAYOUT_RETRIES = 200
OWNERSHIP_RETRIES = 200
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(cells: set[Coord]) -> list[set[Coord]]:
    remaining = set(cells)
    out: list[set[Coord]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = {start}
        remaining.remove(start)
        while stack:
            r, c = stack.pop()
            for nbr in _neighbors4(r, c):
                if nbr in remaining:
                    remaining.remove(nbr)
                    comp.add(nbr)
                    stack.append(nbr)
        out.append(comp)
    return out


def _color_components(grid: np.ndarray, color: int) -> list[set[Coord]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))}
    return _components(coords)


def _transform_offset(offset: Offset, kind: int) -> Offset:
    dr, dc = offset
    if kind == 0:
        return dr, dc
    if kind == 1:
        return dr, -dc
    if kind == 2:
        return -dr, dc
    if kind == 3:
        return -dr, -dc
    if kind == 4:
        return dc, dr
    if kind == 5:
        return dc, -dr
    if kind == 6:
        return -dc, dr
    return -dc, -dr


def _is_asymmetric(offsets: set[Offset]) -> bool:
    return sum({ _transform_offset(offset, kind) for offset in offsets } == offsets for kind in range(8)) == 1


def _offset_bbox(offsets: set[Offset]) -> tuple[int, int, int, int]:
    rows = [dr for dr, _dc in offsets]
    cols = [dc for _dr, dc in offsets]
    return min(rows), min(cols), max(rows), max(cols)


def _translate_offsets(origin: Coord, offsets: set[Offset]) -> set[Coord]:
    r0, c0 = origin
    return {(r0 + dr, c0 + dc) for dr, dc in offsets}


def _mask_visible_enough(visible_cells: set[Coord]) -> bool:
    if len(visible_cells) < 2:
        return False
    return all(len(comp) >= 2 for comp in _components(visible_cells))


def _gen_mask_offsets() -> set[Offset]:
    allowed = {
        (dr, dc)
        for dr in range(-MASK_RADIUS, MASK_RADIUS + 1)
        for dc in range(-MASK_RADIUS, MASK_RADIUS + 1)
        if max(abs(dr), abs(dc)) >= 2
    }
    for _ in range(MASK_RETRIES):
        target_area = random.randint(*MASK_AREA_RANGE)
        start = random.choice(tuple(allowed))
        mask = {start}
        while len(mask) < target_area:
            frontier = []
            for dr, dc in mask:
                for nbr in _neighbors4(dr, dc):
                    if nbr in allowed and nbr not in mask:
                        frontier.append(nbr)
            if not frontier:
                break
            mask.add(random.choice(frontier))
        if len(mask) != target_area:
            continue
        r0, c0, r1, c1 = _offset_bbox(mask)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric sprite mask")


def _control_ranges(mask: set[Offset], h: int, w: int) -> tuple[range, range]:
    min_r, min_c, max_r, max_c = _offset_bbox(mask)
    row_lo = max(0, -min_r)
    row_hi = min(h - 1, h - 1 - max_r)
    col_lo = max(0, -min_c)
    col_hi = min(w - 1, w - 1 - max_c)
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _sample_source_layout(mask: set[Offset], h: int, w: int, n_sources: int) -> tuple[list[Coord], list[set[Coord]]] | None:
    row_range, col_range = _control_ranges(mask, h, w)
    cluster = [(dr, dc) for dr in range(-CLUSTER_RADIUS, CLUSTER_RADIUS + 1) for dc in range(-CLUSTER_RADIUS, CLUSTER_RADIUS + 1)]
    for _ in range(LAYOUT_RETRIES):
        base = (random.choice(tuple(row_range)), random.choice(tuple(col_range)))
        deltas = random.sample(cluster, n_sources)
        positions = [(base[0] + dr, base[1] + dc) for dr, dc in deltas]
        if len(set(positions)) != n_sources:
            continue
        if any(pos[0] not in row_range or pos[1] not in col_range for pos in positions):
            continue
        copies = [_translate_offsets(pos, mask) for pos in positions]
        union_cells = set().union(*copies)
        if any(pos in union_cells for pos in positions):
            continue
        ok = True
        for i, cells in enumerate(copies):
            others = set().union(*(copies[j] for j in range(n_sources) if j != i))
            overlap = cells & others
            unique = cells - others
            if len(overlap) < 3 or len(unique) < 2:
                ok = False
                break
        if ok:
            return positions, copies
    return None


def _expanded(cells: set[Coord]) -> set[Coord]:
    out = set(cells)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _choose_target_position(mask: set[Offset], h: int, w: int, source_positions: list[Coord], source_copies: list[set[Coord]]) -> Coord | None:
    row_range, col_range = _control_ranges(mask, h, w)
    source_cells = set().union(*source_copies)
    source_controls = set(source_positions)
    forbidden = _expanded(source_cells | source_controls)
    candidates: list[Coord] = []
    for r in row_range:
        for c in col_range:
            pos = (r, c)
            cells = _translate_offsets(pos, mask)
            if pos in forbidden:
                continue
            if cells & forbidden:
                continue
            candidates.append(pos)
    if not candidates:
        return None
    return random.choice(candidates)


def _assign_visibility(mask: set[Offset], source_positions: list[Coord], source_copies: list[set[Coord]]) -> tuple[list[set[Coord]], list[set[Offset]]] | None:
    cover_map: dict[Coord, list[int]] = defaultdict(list)
    for idx, cells in enumerate(source_copies):
        for cell in cells:
            cover_map[cell].append(idx)

    full_mask = set(mask)
    n_sources = len(source_positions)
    for _ in range(OWNERSHIP_RETRIES):
        params = [
            (
                random.choice([-2, -1, 1, 2]),
                random.choice([-2, -1, 1, 2]),
                random.randint(-6, 6),
                random.random(),
            )
            for _ in range(n_sources)
        ]
        visible_cells = [set() for _ in range(n_sources)]
        visible_offsets = [set() for _ in range(n_sources)]
        for (r, c), owners in cover_map.items():
            if len(owners) == 1:
                owner = owners[0]
            else:
                owner = max(
                    owners,
                    key=lambda idx: (
                        params[idx][0] * r + params[idx][1] * c + params[idx][2],
                        params[idx][3],
                    ),
                )
            visible_cells[owner].add((r, c))
            anchor_r, anchor_c = source_positions[owner]
            visible_offsets[owner].add((r - anchor_r, c - anchor_c))

        if set().union(*visible_offsets) != full_mask:
            continue
        if any(len(offsets) >= len(full_mask) for offsets in visible_offsets):
            continue
        if any(not _mask_visible_enough(cells) for cells in visible_cells):
            continue
        return visible_cells, visible_offsets
    return None


def _render_input(
    h: int,
    w: int,
    source_positions: list[Coord],
    source_colors: list[int],
    target_position: Coord,
    target_color: int,
    visible_cells: list[set[Coord]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for color, cells in zip(source_colors, visible_cells):
        for r, c in cells:
            grid[r, c] = int(color)
    for color, pos in zip(source_colors, source_positions):
        grid[pos] = int(color)
    grid[target_position] = int(target_color)
    return grid


def _render_output(grid: np.ndarray, target_position: Coord, target_color: int, mask: set[Offset]) -> np.ndarray:
    out = np.array(grid, copy=True)
    tr, tc = target_position
    for dr, dc in mask:
        rr = tr + dr
        cc = tc + dc
        if out[rr, cc] == BG:
            out[rr, cc] = int(target_color)
    return out


def _has_unique_singleton_controls(grid: np.ndarray, colors: list[int]) -> bool:
    for color in colors:
        comps = _color_components(grid, color)
        if sum(len(comp) == 1 for comp in comps) != 1:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the missing stamp by unioning all visible same-shape fragments in control-relative coordinates.

    Input structure:
    - A black background contains several uniquely colored singleton control pixels.
    - For every control color except one, additional pixels of that same color form a visible fragment of one shared sprite mask.
    - The fragments come from copies of the same mask placed at identical relative offsets from their own control pixels, but different overlaps hide different parts.

    Rule:
    - For each color, use its singleton pixel as the control origin and convert all other visible same-color pixels into relative offsets.
    - Union the relative offsets from every color that has a fragment to recover the full sprite mask.
    - Find the unique control color with no fragment, and stamp the recovered mask at that control position using that color, only onto black cells.

    Key insight:
    - No single copy is complete; the full mask exists only as the cross-copy union of visible offsets once every fragment is expressed in the coordinate system of its control pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)

    union_offsets: set[Offset] = set()
    target_color: int | None = None
    target_control: Coord | None = None

    for color in colors:
        comps = _color_components(g, color)
        singletons = [next(iter(comp)) for comp in comps if len(comp) == 1]
        assert len(singletons) == 1, "expected exactly one singleton control for each color"
        control = singletons[0]

        offsets = {
            (int(r - control[0]), int(c - control[1]))
            for r, c in np.argwhere(g == int(color))
            if (int(r), int(c)) != control
        }
        if offsets:
            union_offsets |= offsets
        else:
            assert target_color is None, "expected a unique missing-stamp control"
            target_color = int(color)
            target_control = control

    assert union_offsets, "expected at least one visible fragment"
    assert target_color is not None and target_control is not None, "expected exactly one target control"

    out = g.copy()
    tr, tc = target_control
    for dr, dc in union_offsets:
        rr = tr + dr
        cc = tc + dc
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "recovered stamp must stay in bounds"
        if out[rr, cc] == bg:
            out[rr, cc] = target_color
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    mask = _gen_mask_offsets()
    n_sources = random.choice(SOURCE_COUNT_CHOICES)

    layout = _sample_source_layout(mask, h, w, n_sources)
    if layout is None:
        raise RuntimeError("failed to place overlapping source copies")
    source_positions, source_copies = layout

    target_position = _choose_target_position(mask, h, w, source_positions, source_copies)
    if target_position is None:
        raise RuntimeError("failed to place target control")

    ownership = _assign_visibility(mask, source_positions, source_copies)
    if ownership is None:
        raise RuntimeError("failed to assign coherent visible fragments")
    visible_cells, _visible_offsets = ownership

    colors = random.sample(COLOR_POOL, n_sources + 1)
    source_colors = colors[:-1]
    target_color = colors[-1]

    inp = _render_input(
        h,
        w,
        source_positions,
        source_colors,
        target_position,
        target_color,
        visible_cells,
    )
    if not _has_unique_singleton_controls(inp, colors):
        raise RuntimeError("invalid singleton-control structure")

    out = _render_output(inp, target_position, target_color, mask)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, out = _generate_case()
        except RuntimeError:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate valid input for puzzle 4")
