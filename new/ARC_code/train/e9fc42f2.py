from collections import defaultdict, deque

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location
import math
import random


def _connector_dir(sprite: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = sprite.shape
    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append((-1, 0))
    if r == h - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == w - 1:
        dirs.append((0, 1))
    assert len(dirs) == 1
    return dirs[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble the foreground objects by docking equal-colored boundary markers edge-to-edge.

    Input structure
    The grid has a dominant background color and several 4-connected multicolor foreground
    objects.  One foreground color is the body color; the less frequent foreground colors are
    boundary connector markers, and each marker color appears once as a free end or twice as a
    matched docking pair.

    Rule
    Crop each foreground object to its bounding box and translate the objects so that the two
    occurrences of every paired connector color touch on their exposed boundary sides.  Preserve
    each object's orientation and colors, then return the tight crop around the assembled
    foreground on the original background color.

    Key insight
    The colored marker pixels are ports: matching colors specify which objects must touch, and
    the marker's side of its object determines the direction of the translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    fg = g[g != bg]
    assert fg.size > 0
    values, counts = np.unique(fg, return_counts=True)
    max_count = int(counts.max())
    body_candidates = [int(v) for v, n in zip(values, counts) if int(n) == max_count]
    assert len(body_candidates) == 1
    body = body_candidates[0]
    marker_colors = {int(v) for v in values if int(v) != body}
    assert marker_colors

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) >= 2

    sprites: list[np.ndarray] = []
    marker_occurrences: dict[int, list[tuple[int, tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    for idx, comp in enumerate(components):
        sprite = crop_bbox(g, comp["bbox"])
        sprites.append(sprite)
        marker_locs = np.argwhere(np.isin(sprite, list(marker_colors)))
        assert marker_locs.size > 0
        for rr, cc in marker_locs:
            r, c = int(rr), int(cc)
            direction = _connector_dir(sprite, r, c)
            marker_occurrences[int(sprite[r, c])].append((idx, (r, c), direction))

    adjacency: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in sprites]
    paired_markers = 0
    for occurrences in marker_occurrences.values():
        assert len(occurrences) in (1, 2)
        if len(occurrences) == 1:
            continue
        (a_idx, a_pos, a_dir), (b_idx, b_pos, b_dir) = occurrences
        assert a_idx != b_idx
        assert b_dir == (-a_dir[0], -a_dir[1])
        delta = (a_pos[0] + a_dir[0] - b_pos[0], a_pos[1] + a_dir[1] - b_pos[1])
        adjacency[a_idx].append((b_idx, delta))
        adjacency[b_idx].append((a_idx, (-delta[0], -delta[1])))
        paired_markers += 1
    assert paired_markers == len(sprites) - 1

    placements: dict[int, tuple[int, int]] = {0: (0, 0)}
    queue = deque([0])
    while queue:
        idx = queue.popleft()
        base_r, base_c = placements[idx]
        for nbr, (dr, dc) in adjacency[idx]:
            pos = (base_r + dr, base_c + dc)
            if nbr in placements:
                assert placements[nbr] == pos
            else:
                placements[nbr] = pos
                queue.append(nbr)
    assert len(placements) == len(sprites)

    placed_fg: list[tuple[int, int]] = []
    for idx, sprite in enumerate(sprites):
        base_r, base_c = placements[idx]
        for rr, cc in np.argwhere(sprite != bg):
            placed_fg.append((base_r + int(rr), base_c + int(cc)))

    r0 = min(r for r, _ in placed_fg)
    c0 = min(c for _, c in placed_fg)
    r1 = max(r for r, _ in placed_fg)
    c1 = max(c for _, c in placed_fg)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)

    for idx, sprite in enumerate(sprites):
        base_r, base_c = placements[idx]
        rr, cc = base_r - r0, base_c - c0
        mask = sprite != bg
        target = out[rr : rr + sprite.shape[0], cc : cc + sprite.shape[1]]
        assert target.shape == sprite.shape
        assert not np.any((target != bg) & mask)
        paste(out, sprite, rr, cc, transparent=bg)

    return out

# --------------- generator (o5x) ---------------

# Palette and marker vocabulary.
_GEN_BG, _GEN_BODY = 2, 1
_GEN_MARKERS = [3, 4, 8]

# Port sides and their assembly directions.
_GEN_SIDES = ("T", "B", "L", "R")
_GEN_DIRS = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
_GEN_OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}

# Canvas and scattering constraints.
_GEN_CANVAS_SHAPE = (13, 13)
_GEN_MAX_OUTPUT_SPAN = 13
_GEN_SCATTER_PADDING, _GEN_PADDING_CONNECTIVITY = 2, 8
_GEN_SHADOW_BG = 0

# Sprite count, size, and special-case height controls.
_GEN_SPRITE_COUNT_CHOICES = [2, 3, 4]
_GEN_MIN_SPRITES, _GEN_MID_SPRITES, _GEN_MAX_SPRITES = 2, 3, 4
_GEN_HEIGHT_RANGE, _GEN_COMPACT_HEIGHT_RANGE, _GEN_WIDTH_RANGE = (3, 8), (3, 6), (3, 6)
_GEN_FOUR_SPRITE_HEIGHT_CHOICES = [3, 4, 5, 5, 6]
_GEN_THREE_SPRITE_COMPACT_PROB = 0.7

# Sprite fill and port-count constraints.
_GEN_INTERIOR_MARGIN = 1
_GEN_MIN_DENSITY, _GEN_MAX_DENSITY = 0.36, 0.89
_GEN_MAX_PORTS_PER_SPRITE = 3

# Generation retry budget.
_GEN_MAX_ATTEMPTS = 300


def _gen_neighbors4(r, c, h, w):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _gen_boundary_coord(side, h, w):
    low = _GEN_INTERIOR_MARGIN
    if side == "T":
        return 0, random.randint(low, w - 1 - low)
    if side == "B":
        return h - 1, random.randint(low, w - 1 - low)
    if side == "L":
        return random.randint(low, h - 1 - low), 0
    return random.randint(low, h - 1 - low), w - 1


def _gen_inner_neighbor(side, r, c):
    dr, dc = _GEN_DIRS[side]
    return r - dr, c - dc


def _gen_add_path(cells, start, end):
    r, c = start
    er, ec = end
    cells.add((r, c))
    axes = [0, 1]
    random.shuffle(axes)
    for axis in axes:
        while (axis == 0 and r != er) or (axis == 1 and c != ec):
            if axis == 0:
                r += 1 if er > r else -1
            else:
                c += 1 if ec > c else -1
            cells.add((r, c))


def _gen_seed_sprite_cells(h, w, ports):
    port_locs = {}
    cells = set()
    port_sides = {side for _, side in ports}

    for color, side in ports:
        r, c = _gen_boundary_coord(side, h, w)
        port_locs[color] = (r, c)
        cells.add((r, c))
        cells.add(_gen_inner_neighbor(side, r, c))

    for side in _GEN_SIDES:
        if side in port_sides:
            continue
        r, c = _gen_boundary_coord(side, h, w)
        cells.add((r, c))

    return cells, port_locs


def _gen_connect_sprite_cells(cells, h, w):
    low = _GEN_INTERIOR_MARGIN
    center = (random.randint(low, h - 1 - low), random.randint(low, w - 1 - low))
    cells.add(center)
    for cell in list(cells):
        _gen_add_path(cells, cell, center)


def _gen_frontier(cells, h, w):
    return {
        nbr
        for r, c in cells
        for nbr in _gen_neighbors4(r, c, h, w)
        if nbr not in cells
    }


def _gen_grow_sprite_cells(cells, h, w):
    area = h * w
    min_fg = max(len(cells), math.ceil(_GEN_MIN_DENSITY * area))
    max_fg = math.floor(_GEN_MAX_DENSITY * area)
    if min_fg > max_fg:
        return False

    target = random.randint(min_fg, max_fg)
    while len(cells) < target:
        frontier = _gen_frontier(cells, h, w)
        if not frontier:
            return False
        cells.add(random.choice(tuple(frontier)))
    return True


def _gen_paint_sprite(h, w, cells, ports, port_locs):
    sprite = np.full((h, w), _GEN_BG, dtype=int)
    for r, c in cells:
        sprite[r, c] = _GEN_BODY
    for color, _side in ports:
        r, c = port_locs[color]
        sprite[r, c] = color

    for color, side in ports:
        r, c = port_locs[color]
        ir, ic = _gen_inner_neighbor(side, r, c)
        if sprite[ir, ic] != _GEN_BODY:
            return None

    return sprite


def _gen_build_sprite(h, w, ports):
    cells, port_locs = _gen_seed_sprite_cells(h, w, ports)
    _gen_connect_sprite_cells(cells, h, w)
    if not _gen_grow_sprite_cells(cells, h, w):
        return None, None

    sprite = _gen_paint_sprite(h, w, cells, ports, port_locs)
    if sprite is None:
        return None, None
    return sprite, port_locs


def _gen_tree(n):
    edges = []
    for child in range(1, n):
        parent = random.randrange(child)
        edges.append((parent, child))
    return edges


def _gen_unpaired_marker_count(n):
    if n == _GEN_MIN_SPRITES:
        return 1
    if n == _GEN_MID_SPRITES:
        return random.randint(0, 1)
    return 0


def _gen_port_plan(n):
    edges = _gen_tree(n)
    unpaired = _gen_unpaired_marker_count(n)
    colors = random.sample(_GEN_MARKERS, len(edges) + unpaired)
    paired_colors = colors[: len(edges)]
    unpaired_color = colors[-1] if unpaired else None

    used_sides = [set() for _ in range(n)]
    ports = [[] for _ in range(n)]
    edge_records = []

    for (parent, child), color in zip(edges, paired_colors):
        choices = [
            side
            for side in _GEN_SIDES
            if side not in used_sides[parent]
            and _GEN_OPPOSITE[side] not in used_sides[child]
        ]
        if not choices:
            return None
        side = random.choice(choices)
        child_side = _GEN_OPPOSITE[side]
        used_sides[parent].add(side)
        used_sides[child].add(child_side)
        ports[parent].append((color, side))
        ports[child].append((color, child_side))
        edge_records.append((parent, child, color, side))

    if unpaired_color is not None:
        choices = [
            idx
            for idx in range(n)
            if len(ports[idx]) < _GEN_MAX_PORTS_PER_SPRITE
            and any(side not in used_sides[idx] for side in _GEN_SIDES)
        ]
        if not choices:
            return None
        idx = random.choice(choices)
        side = random.choice([side for side in _GEN_SIDES if side not in used_sides[idx]])
        ports[idx].append((unpaired_color, side))
        used_sides[idx].add(side)

    if any(len(p) == 0 or len(p) > _GEN_MAX_PORTS_PER_SPRITE for p in ports):
        return None
    return ports, edge_records


def _gen_dims(n):
    dims = []
    for _ in range(n):
        if n == _GEN_MAX_SPRITES:
            h = random.choice(_GEN_FOUR_SPRITE_HEIGHT_CHOICES)
        elif n == _GEN_MID_SPRITES and random.random() < _GEN_THREE_SPRITE_COMPACT_PROB:
            h = random.randint(*_GEN_COMPACT_HEIGHT_RANGE)
        else:
            h = random.randint(*_GEN_HEIGHT_RANGE)
        w = random.randint(*_GEN_WIDTH_RANGE)
        dims.append((h, w))
    return dims


def _gen_placements(edge_records, port_locs):
    placements = {0: (0, 0)}
    pending = list(edge_records)
    while pending:
        progressed = False
        rest = []
        for parent, child, color, side in pending:
            if parent not in placements:
                rest.append((parent, child, color, side))
                continue
            pr, pc = placements[parent]
            ar, ac = port_locs[parent][color]
            br, bc = port_locs[child][color]
            dr, dc = _GEN_DIRS[side]
            placements[child] = (pr + ar + dr - br, pc + ac + dc - bc)
            progressed = True
        if not progressed:
            return None
        pending = rest
    return placements


def _gen_render_output(sprites, placements):
    coords = []
    seen = set()
    for idx, sprite in enumerate(sprites):
        br, bc = placements[idx]
        for rr, cc in np.argwhere(sprite != _GEN_BG):
            pos = (br + int(rr), bc + int(cc))
            if pos in seen:
                return None
            seen.add(pos)
            coords.append(pos)
    if not coords:
        return None

    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    r1 = max(r for r, _ in coords)
    c1 = max(c for _, c in coords)
    if r1 - r0 + 1 > _GEN_MAX_OUTPUT_SPAN or c1 - c0 + 1 > _GEN_MAX_OUTPUT_SPAN:
        return None

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), _GEN_BG, dtype=int)
    for idx, sprite in enumerate(sprites):
        br, bc = placements[idx]
        rr, cc = br - r0, bc - c0
        target = out[rr : rr + sprite.shape[0], cc : cc + sprite.shape[1]]
        mask = sprite != _GEN_BG
        if target.shape != sprite.shape or np.any((target != _GEN_BG) & mask):
            return None
        paste(out, sprite, rr, cc, transparent=_GEN_BG)
    return out


def _gen_scatter_input(sprites):
    canvas = np.full(_GEN_CANVAS_SHAPE, _GEN_BG, dtype=int)
    shadow = np.full(_GEN_CANVAS_SHAPE, _GEN_SHADOW_BG, dtype=int)
    order = sorted(range(len(sprites)), key=lambda i: sprites[i].size, reverse=True)
    for idx in order:
        sprite = sprites[idx]
        rect = np.ones(sprite.shape, dtype=int)
        try:
            r, c = random_free_location(
                shadow,
                rect,
                transparent=_GEN_SHADOW_BG,
                padding=_GEN_SCATTER_PADDING,
                padding_connectivity=_GEN_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(shadow, rect, r, c)
        paste(canvas, sprite, r, c, transparent=_GEN_BG)
    return canvas


def _gen_attempt():
    n = random.choice(_GEN_SPRITE_COUNT_CHOICES)
    plan = _gen_port_plan(n)
    if plan is None:
        return None
    ports, edge_records = plan

    sprites = []
    port_locs = []
    for (h, w), sprite_ports in zip(_gen_dims(n), ports):
        sprite, locs = _gen_build_sprite(h, w, sprite_ports)
        if sprite is None:
            return None
        sprites.append(sprite)
        port_locs.append(locs)

    placements = _gen_placements(edge_records, port_locs)
    if placements is None:
        return None
    out = _gen_render_output(sprites, placements)
    if out is None:
        return None

    inp = _gen_scatter_input(sprites)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e9fc42f2 input")
