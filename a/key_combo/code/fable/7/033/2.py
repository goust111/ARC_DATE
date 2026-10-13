"""ARC task: assemble connector-marked pieces, weld, and fill new holes."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
YELLOW = 4
_GEN_CALL_COUNT = 0
_GEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected coordinate arrays, using the common (row, col) convention."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        coords = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for rr, cc in _neighbors(r, c):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _hole_mask(occupied: np.ndarray) -> np.ndarray:
    """Background cells not 4-connected to the boundary of this local canvas."""
    h, w = occupied.shape
    exterior = np.zeros_like(occupied, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not occupied[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not occupied[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors(r, c):
            if (0 <= rr < h and 0 <= cc < w and not occupied[rr, cc]
                    and not exterior[rr, cc]):
                exterior[rr, cc] = True
                queue.append((rr, cc))
    return (~occupied) & (~exterior)


def _normalize_render(
    pieces: list[dict], placements: dict[int, tuple[int, int]], bg: int,
    body: int, marker_colors: set[int],
) -> np.ndarray | None:
    """Validate one complete placement and render its welded, hole-filled crop."""
    cells: dict[tuple[int, int], tuple[int, int]] = {}
    for idx, piece in enumerate(pieces):
        br, bc = placements[idx]
        for (r, c), color in piece["cells"].items():
            pos = (br + r, bc + c)
            if pos in cells:
                return None
            cells[pos] = (idx, color)

    # Every cross-piece contact must be exactly a matching colored dock.
    contacts: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for pos, (idx, color) in cells.items():
        for nbr in _neighbors(*pos):
            if nbr not in cells or cells[nbr][0] == idx:
                continue
            nidx, ncolor = cells[nbr]
            if color != ncolor or color not in marker_colors:
                return None
            contacts.add(tuple(sorted((pos, nbr))))

    for color in marker_colors:
        pair = [pos for pos, (_idx, value) in cells.items() if value == color]
        if len(pair) != 2 or abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]) != 1:
            return None
        contacts.discard(tuple(sorted(pair)))
    if contacts:
        return None

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), (_idx, color) in cells.items():
        out[r - r0, c - c0] = body if color in marker_colors else color
    holes = _hole_mask(out != bg)
    if not np.any(holes):
        return None
    out[holes] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble matching connector-marked pieces, weld their seams, fill emerged holes yellow, and crop tightly.

    Input structure:
    - A uniquely dominant background contains several separated 4-connected pieces sharing
      one body color. Every rarer connector color occurs exactly twice on two different pieces,
      and no individual piece contains an enclosed background hole.

    Rule:
    - Translate pieces without rotation or reflection so each equal-colored connector pair
      touches edge-to-edge. Reject overlaps and any cross-piece contact not made by a matching
      connector; the constraints determine one normalized assembly. Recolor all connectors to
      the body color, fill every enclosed background region yellow, and return the tight crop.

    Key insight:
    - The colored boundary pixels are both docking instructions and seam cells: only after the
      uniquely constrained assembly is welded do the yellow enclosed holes emerge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    values, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    bg_candidates = [int(v) for v, n in zip(values, counts) if int(n) == max_count]
    assert len(bg_candidates) == 1, "background must be uniquely most common"
    bg = bg_candidates[0]

    fg_values = [int(v) for v in values if int(v) != bg]
    assert len(fg_values) >= 3 and YELLOW not in fg_values, "expected body and connector colors"
    fg_counts = {v: int(np.count_nonzero(g == v)) for v in fg_values}
    body_count = max(fg_counts.values())
    body_candidates = [v for v, n in fg_counts.items() if n == body_count]
    assert len(body_candidates) == 1, "body color must be uniquely most frequent foreground"
    body = body_candidates[0]
    marker_colors = set(fg_values) - {body}
    assert marker_colors and all(fg_counts[c] == 2 for c in marker_colors)

    comps = _components(g != bg)
    assert 2 <= len(comps) <= 6, "expected several separated pieces"
    pieces: list[dict] = []
    occurrences: dict[int, list[tuple[int, tuple[int, int]]]] = {c: [] for c in marker_colors}
    for idx, coords in enumerate(comps):
        r0, c0 = coords.min(axis=0)
        local: dict[tuple[int, int], int] = {}
        for r, c in coords:
            pos = (int(r - r0), int(c - c0))
            color = int(g[int(r), int(c)])
            local[pos] = color
            if color in marker_colors:
                occurrences[color].append((idx, pos))
        assert body in local.values(), "each piece must contain body cells"
        h = max(r for r, _ in local) + 1
        w = max(c for _, c in local) + 1
        local_mask = np.zeros((h, w), dtype=bool)
        for pos in local:
            local_mask[pos] = True
        assert not np.any(_hole_mask(local_mask)), "an input piece may not contain a hole"
        pieces.append({"cells": local})

    edges = []
    for color, pair in occurrences.items():
        assert len(pair) == 2 and pair[0][0] != pair[1][0]
        (a, pa), (b, pb) = pair
        edges.append((a, b, pa, pb, color))

    graph = [set() for _ in pieces]
    for a, b, _pa, _pb, _color in edges:
        graph[a].add(b)
        graph[b].add(a)
    reached, queue = {0}, deque([0])
    while queue:
        i = queue.popleft()
        for j in graph[i] - reached:
            reached.add(j)
            queue.append(j)
    assert len(reached) == len(pieces), "connector graph must be connected"

    rendered: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    placements: dict[int, tuple[int, int]] = {0: (0, 0)}

    def overlaps(idx: int, pos: tuple[int, int]) -> bool:
        occupied = set()
        for j, (jr, jc) in placements.items():
            occupied.update((jr + r, jc + c) for r, c in pieces[j]["cells"])
        return any((pos[0] + r, pos[1] + c) in occupied for r, c in pieces[idx]["cells"])

    def search() -> None:
        if len(placements) == len(pieces):
            out = _normalize_render(pieces, placements, bg, body, marker_colors)
            if out is not None:
                rendered[(out.shape, out.tobytes())] = out
            return

        candidates = [i for i in range(len(pieces)) if i not in placements and graph[i] & placements.keys()]
        if not candidates:
            return
        idx = min(candidates, key=lambda i: -len(graph[i] & placements.keys()))
        constraints = []
        for a, b, pa, pb, _color in edges:
            if a == idx and b in placements:
                constraints.append((pa, pb, placements[b]))
            elif b == idx and a in placements:
                constraints.append((pb, pa, placements[a]))
        own, other, opos = constraints[0]
        options = set()
        for dr, dc in DIR4:
            options.add((opos[0] + other[0] + dr - own[0], opos[1] + other[1] + dc - own[1]))
        for pos in options:
            ok = True
            for p_own, p_other, base in constraints:
                distance = abs((pos[0] + p_own[0]) - (base[0] + p_other[0])) + abs((pos[1] + p_own[1]) - (base[1] + p_other[1]))
                if distance != 1:
                    ok = False
                    break
            if ok and not overlaps(idx, pos):
                placements[idx] = pos
                search()
                del placements[idx]

    search()
    assert len(rendered) == 1, f"expected one uniquely determined assembly, got {len(rendered)}"
    return next(iter(rendered.values())).copy()


def _rectangle_cycle(h: int, w: int, r0: int = 0, c0: int = 0) -> list[tuple[int, int]]:
    cycle = [(r0, c0 + c) for c in range(w)]
    cycle += [(r0 + r, c0 + w - 1) for r in range(1, h)]
    cycle += [(r0 + h - 1, c0 + c) for c in range(w - 2, -1, -1)]
    cycle += [(r0 + r, c0) for r in range(h - 2, 0, -1)]
    return cycle


def _partition_cycle(cycle: list[tuple[int, int]], count: int, forbidden: set[tuple[int, int]]):
    """Split a cycle into path pieces; cut endpoints become the paired docks."""
    n = len(cycle)
    for _ in range(300):
        extras = n - 4 * count
        if extras < 0:
            return None
        lengths = [4] * count
        for _j in range(extras):
            lengths[random.randrange(count)] += 1
        random.shuffle(lengths)
        start = random.randrange(n)
        ordered = cycle[start:] + cycle[:start]
        pieces, endpoints, at = [], [], 0
        for length in lengths:
            pieces.append(set(ordered[at:at + length]))
            at += length
        for i in range(count):
            endpoints.append((ordered[sum(lengths[:i + 1]) - 1], ordered[sum(lengths[:i + 1]) % n]))
        if all(a not in forbidden and b not in forbidden for a, b in endpoints):
            return pieces, endpoints
    return None


def _expected_from_final(cells: dict[tuple[int, int], int], bg: int, body: int, markers: set[int]) -> np.ndarray:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    r1 = max(r for r, _ in cells)
    c1 = max(c for _, c in cells)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), color in cells.items():
        out[r - r0, c - c0] = body if color in markers else color
    out[_hole_mask(out != bg)] = YELLOW
    return out


def _scatter(piece_maps: list[dict[tuple[int, int], int]], bg: int) -> np.ndarray | None:
    h, w = random.randint(22, 30), random.randint(22, 30)
    canvas = np.full((h, w), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()
    order = list(range(len(piece_maps)))
    random.shuffle(order)
    for idx in order:
        cells = piece_maps[idx]
        ph = max(r for r, _ in cells) + 1
        pw = max(c for _, c in cells) + 1
        choices = [(r0, c0) for r0 in range(1, h - ph) for c0 in range(1, w - pw)]
        random.shuffle(choices)
        placed = None
        for r0, c0 in choices[:800]:
            coords = {(r0 + r, c0 + c) for r, c in cells}
            if any((rr, cc) in occupied for r, c in coords for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)):
                continue
            placed = coords
            for (r, c), color in cells.items():
                canvas[r0 + r, c0 + c] = color
            occupied.update(coords)
            break
        if placed is None:
            return None
    return canvas


def _gen_attempt(two_holes: bool, one_ring_pieces: int) -> np.ndarray | None:
    palette = list(range(10))
    bg = random.choice([0, 1, 2, 3, 5, 6, 7, 8, 9])
    body = random.choice([c for c in palette if c not in {bg, YELLOW}])

    if not two_holes:
        h, w = random.randint(6, 10), random.randint(6, 10)
        partition = _partition_cycle(_rectangle_cycle(h, w), one_ring_pieces, set())
        if partition is None:
            return None
        piece_sets, endpoint_pairs = partition
    else:
        h1, w1 = random.randint(6, 8), random.randint(6, 8)
        h2, w2 = random.randint(6, 8), random.randint(6, 8)
        bridge_len = random.randint(2, 4)
        top2 = random.randint(0, max(0, h1 - h2)) if h1 >= h2 else 0
        left_cycle = _rectangle_cycle(h1, w1)
        right_c0 = w1 + bridge_len
        right_cycle = _rectangle_cycle(h2, w2, top2, right_c0)
        attach_r = min(h1 - 2, max(1, top2 + h2 // 2))
        left_attach = (attach_r, w1 - 1)
        right_attach = (attach_r, right_c0)
        forbidden = {left_attach, right_attach}
        left_part = _partition_cycle(left_cycle, 2, forbidden)
        right_part = _partition_cycle(right_cycle, 2, forbidden)
        if left_part is None or right_part is None:
            return None
        piece_sets = left_part[0] + right_part[0]
        endpoint_pairs = left_part[1] + right_part[1]
        owner = next(i for i, cells in enumerate(piece_sets) if left_attach in cells)
        bridge = {(attach_r, c) for c in range(w1, right_c0)}
        piece_sets[owner].update(bridge)
        endpoint_pairs.append(((attach_r, right_c0 - 1), right_attach))

    marker_count = len(endpoint_pairs)
    available = [c for c in palette if c not in {bg, body, YELLOW}]
    if marker_count > len(available):
        return None
    marker_palette = random.sample(available, marker_count)
    coord_piece = {coord: i for i, cells in enumerate(piece_sets) for coord in cells}
    piece_maps = [{coord: body for coord in cells} for cells in piece_sets]
    for color, (a, b) in zip(marker_palette, endpoint_pairs):
        if a not in coord_piece or b not in coord_piece or coord_piece[a] == coord_piece[b]:
            return None
        piece_maps[coord_piece[a]][a] = color
        piece_maps[coord_piece[b]][b] = color

    # Normalize each sprite to its own bbox before scattering.
    local_maps = []
    for cells in piece_maps:
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        local = {(r - r0, c - c0): color for (r, c), color in cells.items()}
        mask = np.zeros((max(r for r, _ in local) + 1, max(c for _, c in local) + 1), dtype=bool)
        for pos in local:
            mask[pos] = True
        if np.any(_hole_mask(mask)):
            return None
        local_maps.append(local)

    final_cells = {coord: color for cells in piece_maps for coord, color in cells.items()}
    expected = _expected_from_final(final_cells, bg, body, set(marker_palette))
    expected_holes = len(_components(expected == YELLOW))
    if expected_holes != (2 if two_holes else 1):
        return None
    inp = _scatter(local_maps, bg)
    if inp is None or np.count_nonzero(inp == bg) <= np.count_nonzero(inp != bg):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    # Semantic anti-shortcut checks: assembly changes geometry and welding removes all ports.
    raw_crop_coords = np.argwhere(inp != bg)
    rr0, cc0 = raw_crop_coords.min(axis=0)
    rr1, cc1 = raw_crop_coords.max(axis=0)
    raw_crop = inp[rr0:rr1 + 1, cc0:cc1 + 1]
    if raw_crop.shape == actual.shape and np.array_equal(raw_crop, actual):
        return None
    if any(np.any(actual == color) for color in marker_palette) or not np.any(actual == YELLOW):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate varied, uniquely solvable one- and two-hole assembly instances."""
    global _GEN_CALL_COUNT
    call = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    # Every consecutive four calls cover 2, 3, and 4 pieces; every first three
    # calls include the less common two-hole branch.
    phase = call % 4
    two_holes = phase == 1
    piece_count = (2, 4, 3, 4)[phase]
    for _ in range(500):
        grid = _gen_attempt(two_holes, piece_count)
        if grid is not None and max(grid.shape) <= 30:
            out = main(grid)
            signature = (out.shape, out.tobytes())
            if signature in _GEN_OUTPUT_SIGS:
                continue
            _GEN_OUTPUT_SIGS.add(signature)
            return grid
    raise RuntimeError("failed to generate a valid connector-hole assembly")
