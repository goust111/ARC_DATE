import random
from collections import deque

import numpy as np


BG = 0
RED = 2
SEP = 5
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    assert axis in (0, 1)
    if axis == 0:
        counts = np.sum(grid == SEP, axis=1)
        other = grid.shape[1]
    else:
        counts = np.sum(grid == SEP, axis=0)
        other = grid.shape[0]
    threshold = max(2, other // 3 + 1)
    best = None
    best_score = None
    for step in range(2, len(counts)):
        if (len(counts) - 1) % step != 0:
            continue
        indices = list(range(0, len(counts), step))
        if len(indices) < 4:
            continue
        if min(int(counts[i]) for i in indices) < threshold:
            continue
        off_score = sum(int(counts[i]) for i in range(len(counts)) if i not in set(indices))
        on_score = sum(int(counts[i]) for i in indices)
        score = on_score - off_score
        if best is None or score > best_score:
            best = indices
            best_score = score
    assert best is not None, "expected a regular separator lattice"
    return best


def _tile_spans(separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(separators, separators[1:]):
        assert b - a > 1, "separator rows/cols must leave non-empty tile interiors"
        spans.append((a + 1, b - 1))
    return spans


def _label_black_regions(grid: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    touches_edge: list[bool] = []
    component_id = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = component_id
            hit_edge = r == 0 or r == h - 1 or c == 0 or c == w - 1
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if grid[nr, nc] != BG or labels[nr, nc] != -1:
                        continue
                    labels[nr, nc] = component_id
                    hit_edge |= nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1
                    stack.append((nr, nc))
            touches_edge.append(hit_edge)
            component_id += 1
    return labels, touches_edge


def _tile_analysis(grid: np.ndarray):
    row_seps = _separator_indices(grid, axis=0)
    col_seps = _separator_indices(grid, axis=1)
    row_spans = _tile_spans(row_seps)
    col_spans = _tile_spans(col_seps)
    tile_hs = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_ws = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_hs) == 1 and len(tile_ws) == 1, "tile sizes must be uniform"

    labels, touches_edge = _label_black_regions(grid)
    template_tile = None
    template_patch = None
    marked_tiles: dict[tuple[int, int], tuple[int, int]] = {}
    tile_status: dict[tuple[int, int], bool] = {}

    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(tile)))
            template_colors = colors - {BG, SEP}
            marker_coords = np.argwhere(tile == SEP)
            key = (tr, tc)
            if template_colors:
                assert template_tile is None, "expected exactly one template tile"
                template_tile = key
                template_patch = tile.copy()
                assert not len(marker_coords), "template tile must not contain a grey marker"
            elif len(marker_coords):
                assert len(marker_coords) == 1, "each marked tile must contain exactly one internal grey pixel"
                assert colors <= {BG, SEP}, "marked tiles must otherwise contain only black background"
                marker = marker_coords[0]
                marked_tiles[key] = (r0 + int(marker[0]), c0 + int(marker[1]))
            else:
                assert colors == {BG}, "unmarked non-template tiles must be plain background"

            black_ids = {
                int(labels[rr, cc])
                for rr in range(r0, r1 + 1)
                for cc in range(c0, c1 + 1)
                if labels[rr, cc] >= 0
            }
            assert black_ids, "each tile must retain some background pixels"
            statuses = {touches_edge[idx] for idx in black_ids}
            assert len(statuses) == 1, "a tile's background must belong to one topological class"
            tile_status[key] = bool(next(iter(statuses)))

    assert template_tile is not None and template_patch is not None, "template tile missing"
    assert marked_tiles, "expected at least one marked tile"
    return row_spans, col_spans, template_tile, template_patch, marked_tiles, tile_status


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use separator-gap topology to choose whether each grey-marked tile gets the template stamp or only a red recolor.

    Input structure:
    - Grey separator rows and columns, including an outer frame, partition the canvas into equal rectangular tiles.
    - Exactly one tile contains a small non-black, non-grey colored template pattern; other non-template tiles are black inside except for optional single grey marker pixels.
    - Some separator segments are erased to black, creating 4-connected background passages between tile interiors and sometimes to the grid edge.

    Rule:
    - Split the black pixels into 4-connected regions and classify each tile by whether its interior black background belongs to an edge-touching region (open) or a fully enclosed region (closed).
    - For every closed marked tile, replace the entire tile interior with an exact copy of the template tile interior.
    - For every open marked tile, change only its internal grey marker pixel to red while leaving the rest of that tile unchanged.
    - Keep the template tile, all unmarked tiles, and all remaining separator pixels unchanged.

    Key insight:
    - The grey marker only signals candidate target tiles; the decisive cue is the hidden black-background connectivity induced by separator gaps, which gates stamp-vs-recolor behavior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_spans, col_spans, _template_tile, template_patch, marked_tiles, tile_status = _tile_analysis(g)
    out = g.copy()
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            key = (tr, tc)
            if key not in marked_tiles:
                continue
            if tile_status[key]:
                rr, cc = marked_tiles[key]
                out[rr, cc] = RED
            else:
                out[r0 : r1 + 1, c0 : c1 + 1] = template_patch
    return out


def _tile_bounds(tile: tuple[int, int], tile_h: int, tile_w: int) -> tuple[int, int, int, int]:
    tr, tc = tile
    r0 = 1 + tr * (tile_h + 1)
    c0 = 1 + tc * (tile_w + 1)
    return r0, c0, r0 + tile_h - 1, c0 + tile_w - 1


def _tile_neighbors(tile: tuple[int, int], rows: int, cols: int):
    tr, tc = tile
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = tr + dr, tc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _boundary_tiles(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(rows) for c in range(cols) if r in (0, rows - 1) or c in (0, cols - 1)]


def _interior_tiles(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]


def _shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    prev = {start: None}
    while queue:
        tile = queue.popleft()
        if tile == goal:
            path = []
            cur = tile
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return list(reversed(path))
        for nxt in _tile_neighbors(tile, rows, cols):
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = tile
            queue.append(nxt)
    return None


def _random_template(tile_h: int, tile_w: int, rng: random.Random) -> np.ndarray:
    patch = np.zeros((tile_h, tile_w), dtype=int)
    area = tile_h * tile_w
    count = rng.randint(2, min(5, area - 1))
    coords = rng.sample([(r, c) for r in range(tile_h) for c in range(tile_w)], count)
    palette = rng.sample(TEMPLATE_COLORS, rng.randint(1, min(2, len(TEMPLATE_COLORS))))
    for r, c in coords:
        patch[r, c] = rng.choice(palette)
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1 and len(cols) == 1:
        rr, cc = coords[0]
        nr = min(tile_h - 1, rr + 1)
        patch[nr, cc] = rng.choice(palette)
    return patch


def _marker_position(tile_h: int, tile_w: int, rng: random.Random) -> tuple[int, int]:
    row_choices = list(range(1, tile_h - 1)) or list(range(tile_h))
    col_choices = list(range(1, tile_w - 1)) or list(range(tile_w))
    return rng.choice(row_choices), rng.choice(col_choices)


def _erase_vertical_gap(grid: np.ndarray, tile: tuple[int, int], sep_col: int, tile_h: int, rng: random.Random) -> None:
    r0, _, r1, _ = _tile_bounds(tile, tile_h, 1)
    length = rng.randint(1, min(2, tile_h))
    start = rng.randint(r0, r1 - length + 1)
    grid[start : start + length, sep_col] = BG


def _erase_horizontal_gap(grid: np.ndarray, tile: tuple[int, int], sep_row: int, tile_w: int, rng: random.Random) -> None:
    _, c0, _, c1 = _tile_bounds(tile, 1, tile_w)
    length = rng.randint(1, min(2, tile_w))
    start = rng.randint(c0, c1 - length + 1)
    grid[sep_row, start : start + length] = BG


def _erase_tile_edge_gap(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    tile_h: int,
    tile_w: int,
    rng: random.Random,
) -> None:
    ar, ac = a
    br, bc = b
    assert abs(ar - br) + abs(ac - bc) == 1
    if ar == br:
        row_tile = a
        sep_col = (min(ac, bc) + 1) * (tile_w + 1)
        _erase_vertical_gap(grid, row_tile, sep_col, tile_h, rng)
    else:
        col_tile = a
        sep_row = (min(ar, br) + 1) * (tile_h + 1)
        _erase_horizontal_gap(grid, col_tile, sep_row, tile_w, rng)


def _erase_boundary_gap(
    grid: np.ndarray,
    tile: tuple[int, int],
    side: str,
    tile_h: int,
    tile_w: int,
    rng: random.Random,
) -> None:
    tr, tc = tile
    if side == "top":
        _erase_horizontal_gap(grid, (tr, tc), 0, tile_w, rng)
    elif side == "bottom":
        _erase_horizontal_gap(grid, (tr, tc), grid.shape[0] - 1, tile_w, rng)
    elif side == "left":
        _erase_vertical_gap(grid, (tr, tc), 0, tile_h, rng)
    else:
        assert side == "right"
        _erase_vertical_gap(grid, (tr, tc), grid.shape[1] - 1, tile_h, rng)


def _open_region_plan(rows: int, cols: int, template_tile: tuple[int, int], rng: random.Random):
    total = rows * cols
    boundary = [tile for tile in _boundary_tiles(rows, cols) if tile != template_tile]
    interiors = [tile for tile in _interior_tiles(rows, cols) if tile != template_tile]
    seed = rng.choice(boundary)
    target_interior = rng.choice(interiors)
    path = _shortest_path(seed, target_interior, rows, cols, {template_tile})
    if path is None:
        return None

    open_tiles = set(path)
    edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
    max_open = min(total - 3, max(len(open_tiles), total // 2))
    target_size = rng.randint(len(open_tiles), max_open)

    while len(open_tiles) < target_size:
        frontier = []
        for tile in sorted(open_tiles):
            for nxt in _tile_neighbors(tile, rows, cols):
                if nxt in open_tiles or nxt == template_tile:
                    continue
                frontier.append((tile, nxt))
        if not frontier:
            break
        edge = rng.choice(frontier)
        edges.add(edge)
        open_tiles.add(edge[1])

    seed_sides = []
    sr, sc = seed
    if sr == 0:
        seed_sides.append("top")
    if sr == rows - 1:
        seed_sides.append("bottom")
    if sc == 0:
        seed_sides.append("left")
    if sc == cols - 1:
        seed_sides.append("right")
    return seed, rng.choice(seed_sides), open_tiles, edges, target_interior


def _choose_closed_pair(
    rows: int,
    cols: int,
    template_tile: tuple[int, int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates = []
    for r in range(rows):
        for c in range(cols):
            tile = (r, c)
            if tile == template_tile or tile in forbidden:
                continue
            for nxt in _tile_neighbors(tile, rows, cols):
                if nxt <= tile:
                    continue
                if nxt == template_tile or nxt in forbidden:
                    continue
                candidates.append((tile, nxt))
    if not candidates:
        return None
    preferred = [pair for pair in candidates if pair[0] in _interior_tiles(rows, cols) or pair[1] in _interior_tiles(rows, cols)]
    return rng.choice(preferred or candidates)


def _render_base_grid(rows: int, cols: int, tile_h: int, tile_w: int) -> np.ndarray:
    h = rows * tile_h + rows + 1
    w = cols * tile_w + cols + 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(0, h, tile_h + 1):
        grid[r, :] = SEP
    for c in range(0, w, tile_w + 1):
        grid[:, c] = SEP
    return grid


def _place_template(grid: np.ndarray, tile: tuple[int, int], patch: np.ndarray, tile_h: int, tile_w: int) -> None:
    r0, c0, r1, c1 = _tile_bounds(tile, tile_h, tile_w)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _place_marker(grid: np.ndarray, tile: tuple[int, int], tile_h: int, tile_w: int, rng: random.Random) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _tile_bounds(tile, tile_h, tile_w)
    dr, dc = _marker_position(tile_h, tile_w, rng)
    rr, cc = r0 + dr, c0 + dc
    grid[rr, cc] = SEP
    return rr, cc


def _expected_from_meta(
    inp: np.ndarray,
    template_tile: tuple[int, int],
    template_patch: np.ndarray,
    tile_h: int,
    tile_w: int,
    open_marked: set[tuple[int, int]],
    closed_marked: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for tile in open_marked:
        r0, c0, r1, c1 = _tile_bounds(tile, tile_h, tile_w)
        marker = np.argwhere(out[r0 : r1 + 1, c0 : c1 + 1] == SEP)
        assert len(marker) == 1
        rr, cc = marker[0]
        out[r0 + int(rr), c0 + int(cc)] = RED
    for tile in closed_marked:
        r0, c0, r1, c1 = _tile_bounds(tile, tile_h, tile_w)
        out[r0 : r1 + 1, c0 : c1 + 1] = template_patch
    r0, c0, r1, c1 = _tile_bounds(template_tile, tile_h, tile_w)
    assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], template_patch)
    return out


def _semantic_validate(
    inp: np.ndarray,
    template_tile: tuple[int, int],
    template_patch: np.ndarray,
    open_tiles: set[tuple[int, int]],
    seed_tile: tuple[int, int],
    interior_open_tile: tuple[int, int],
    closed_pair: tuple[tuple[int, int], tuple[int, int]],
    marked_open: set[tuple[int, int]],
    marked_closed: set[tuple[int, int]],
) -> bool:
    try:
        row_spans, col_spans, seen_template, seen_patch, marked_tiles, tile_status = _tile_analysis(inp)
        if seen_template != template_tile or not np.array_equal(seen_patch, template_patch):
            return False

        intended_open = {
            (tr, tc): ((tr, tc) in open_tiles)
            for tr in range(len(row_spans))
            for tc in range(len(col_spans))
            if (tr, tc) != template_tile
        }
        for tile, is_open in intended_open.items():
            if tile_status[tile] != is_open:
                return False

        if not marked_open or not marked_closed:
            return False
        if not marked_open <= set(marked_tiles) or not marked_closed <= set(marked_tiles):
            return False
        if interior_open_tile not in marked_open or interior_open_tile == seed_tile:
            return False
        if closed_pair[0] not in marked_closed and closed_pair[1] not in marked_closed:
            return False

        r0, c0, r1, c1 = _tile_bounds(template_tile, seen_patch.shape[0], seen_patch.shape[1])
        if not (
            np.all(inp[r0 - 1, c0 : c1 + 1] == SEP)
            and np.all(inp[r1 + 1, c0 : c1 + 1] == SEP)
            and np.all(inp[r0 : r1 + 1, c0 - 1] == SEP)
            and np.all(inp[r0 : r1 + 1, c1 + 1] == SEP)
        ):
            return False

        expected = _expected_from_meta(
            inp,
            template_tile,
            template_patch,
            seen_patch.shape[0],
            seen_patch.shape[1],
            marked_open,
            marked_closed,
        )
        return np.array_equal(main(inp), expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    rows = rng.randint(3, 4)
    cols = rng.randint(3, 4)
    tile_h = rng.randint(3, 5)
    tile_w = rng.randint(3, 5)

    all_tiles = [(r, c) for r in range(rows) for c in range(cols)]
    interior_tiles = _interior_tiles(rows, cols)
    if len(interior_tiles) == 1:
        template_tile = rng.choice([tile for tile in all_tiles if tile != interior_tiles[0]])
    else:
        template_tile = rng.choice(all_tiles)
    open_plan = _open_region_plan(rows, cols, template_tile, rng)
    if open_plan is None:
        return None
    seed_tile, seed_side, open_tiles, open_edges, interior_open_tile = open_plan

    closed_pair = _choose_closed_pair(rows, cols, template_tile, open_tiles, rng)
    if closed_pair is None:
        return None

    available_closed = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in open_tiles and (r, c) != template_tile
    }
    if len(available_closed) < 2:
        return None

    grid = _render_base_grid(rows, cols, tile_h, tile_w)
    template_patch = _random_template(tile_h, tile_w, rng)
    _place_template(grid, template_tile, template_patch, tile_h, tile_w)

    _erase_boundary_gap(grid, seed_tile, seed_side, tile_h, tile_w, rng)
    for a, b in open_edges:
        _erase_tile_edge_gap(grid, a, b, tile_h, tile_w, rng)
    _erase_tile_edge_gap(grid, closed_pair[0], closed_pair[1], tile_h, tile_w, rng)

    marked_open = {interior_open_tile}
    other_open = list(open_tiles - marked_open)
    rng.shuffle(other_open)
    for tile in other_open[: rng.randint(0, min(2, len(other_open)))]:
        marked_open.add(tile)

    marked_closed = {rng.choice(closed_pair)}
    other_closed = list(available_closed - marked_closed)
    rng.shuffle(other_closed)
    extra_closed = rng.randint(0, min(2, len(other_closed)))
    for tile in other_closed[:extra_closed]:
        marked_closed.add(tile)

    for tile in sorted(marked_open | marked_closed):
        _place_marker(grid, tile, tile_h, tile_w, rng)

    if not _semantic_validate(
        grid,
        template_tile,
        template_patch,
        open_tiles,
        seed_tile,
        interior_open_tile,
        closed_pair,
        marked_open,
        marked_closed,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topological stamp input")
