from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# Minimal arc_utils-style helpers kept local so SourceFileLoader can import this
# file without requiring sys.path edits.
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE_SIDE = {"T": "B", "B": "T", "L": "R", "R": "L"}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    dst[r0 : r0 + h, c0 : c0 + w] = view
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = DIR4 if int(connectivity) == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g) if bg is None else int(bg)
    out: list[dict] = []
    if monochromatic:
        for color in sorted(set(map(int, g.ravel())) - {bg}):
            for coords in components_mask(g == color, connectivity=connectivity):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out
    for coords in components_mask(g != bg, connectivity=connectivity):
        out.append({"coords": coords, "bbox": bbox_coords(coords), "area": int(coords.shape[0])})
    return out


def _connector_dir(sprite: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = sprite.shape
    dirs = []
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


def _parse_legend(legend: np.ndarray, bg: int) -> tuple[dict[int, int], list[tuple[int, int]]]:
    comps = find_components(legend, bg=bg, connectivity=4, monochromatic=False)
    assert comps
    partner: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for comp in comps:
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        assert len(coords) == 2
        (r1, c1), (r2, c2) = sorted(coords)
        assert r1 == r2 and c2 == c1 + 1
        a = int(legend[r1, c1])
        b = int(legend[r2, c2])
        assert a != bg and b != bg and a != b
        assert a not in partner and b not in partner
        partner[a] = b
        partner[b] = a
        pairs.append((a, b))
    return partner, pairs


def _extract_objects(region: np.ndarray, bg: int) -> list[dict]:
    components = find_components(region, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) >= 2
    objects = []
    seen_ports: set[int] = set()
    for comp in components:
        sprite = crop_bbox(region, comp["bbox"])
        fg = sprite[sprite != bg]
        assert fg.size > 0
        colors, counts = np.unique(fg, return_counts=True)
        max_count = int(counts.max())
        body_candidates = [int(color) for color, count in zip(colors, counts, strict=False) if int(count) == max_count]
        assert len(body_candidates) == 1
        body = body_candidates[0]
        port_mask = (sprite != bg) & (sprite != body)
        port_coords = np.argwhere(port_mask)
        assert port_coords.size > 0
        port_colors = [int(sprite[r, c]) for r, c in port_coords]
        assert len(port_colors) == len(set(port_colors))
        ports = {}
        for rr, cc in port_coords:
            r, c = int(rr), int(cc)
            color = int(sprite[r, c])
            assert color not in seen_ports
            seen_ports.add(color)
            ports[color] = {"pos": (r, c), "dir": _connector_dir(sprite, r, c)}
        objects.append({"sprite": sprite, "body": body, "ports": ports})
    return objects


def _try_split(legend: np.ndarray, main: np.ndarray, bg: int) -> dict | None:
    if legend.size == 0 or main.size == 0:
        return None
    try:
        partner, pairs = _parse_legend(legend, bg)
        objects = _extract_objects(main, bg)
        port_colors = {color for obj in objects for color in obj["ports"]}
        assert port_colors == set(partner)
        return {"partner": partner, "pairs": pairs, "objects": objects, "bg": bg}
    except AssertionError:
        return None


def _split_input(grid: np.ndarray, bg: int) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []

    for r in range(h):
        if int(g[r, 0]) == bg or not bool(np.all(g[r] == g[r, 0])):
            continue
        if r > 0:
            info = _try_split(g[:r, :], g[r + 1 :, :], bg)
            if info is not None:
                candidates.append(info)
        if r + 1 < h:
            info = _try_split(g[r + 1 :, :], g[:r, :], bg)
            if info is not None:
                candidates.append(info)

    for c in range(w):
        if int(g[0, c]) == bg or not bool(np.all(g[:, c] == g[0, c])):
            continue
        if c > 0:
            info = _try_split(g[:, :c], g[:, c + 1 :], bg)
            if info is not None:
                candidates.append(info)
        if c + 1 < w:
            info = _try_split(g[:, c + 1 :], g[:, :c], bg)
            if info is not None:
                candidates.append(info)

    assert len(candidates) == 1
    return candidates[0]


def _assemble(objects: list[dict], pairs: list[tuple[int, int]], bg: int) -> np.ndarray:
    port_occurrences = {}
    for idx, obj in enumerate(objects):
        for color, info in obj["ports"].items():
            port_occurrences[color] = (idx, info["pos"], info["dir"])

    adjacency: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in objects]
    for a, b in pairs:
        a_idx, a_pos, a_dir = port_occurrences[a]
        b_idx, b_pos, b_dir = port_occurrences[b]
        assert a_idx != b_idx
        assert b_dir == (-a_dir[0], -a_dir[1])
        dr = a_pos[0] + a_dir[0] - b_pos[0]
        dc = a_pos[1] + a_dir[1] - b_pos[1]
        adjacency[a_idx].append((b_idx, (dr, dc)))
        adjacency[b_idx].append((a_idx, (-dr, -dc)))

    placements: dict[int, tuple[int, int]] = {0: (0, 0)}
    q = deque([0])
    while q:
        idx = q.popleft()
        br, bc = placements[idx]
        for nbr, delta in adjacency[idx]:
            pos = (br + delta[0], bc + delta[1])
            if nbr in placements:
                assert placements[nbr] == pos
            else:
                placements[nbr] = pos
                q.append(nbr)
    assert len(placements) == len(objects)

    owner: dict[tuple[int, int], int] = {}
    pixel: dict[tuple[int, int], int] = {}
    all_coords = []
    for idx, obj in enumerate(objects):
        br, bc = placements[idx]
        for rr, cc in np.argwhere(obj["sprite"] != bg):
            pos = (br + int(rr), bc + int(cc))
            assert pos not in owner
            owner[pos] = idx
            pixel[pos] = int(obj["sprite"][rr, cc])
            all_coords.append(pos)

    expected_contacts = set()
    for a, b in pairs:
        a_idx, a_pos, a_dir = port_occurrences[a]
        b_idx, b_pos, _ = port_occurrences[b]
        pa = (placements[a_idx][0] + a_pos[0], placements[a_idx][1] + a_pos[1])
        pb = (placements[b_idx][0] + b_pos[0], placements[b_idx][1] + b_pos[1])
        assert pb == (pa[0] + a_dir[0], pa[1] + a_dir[1])
        expected_contacts.add(tuple(sorted((pa, pb))))

    actual_contacts = set()
    for r, c in owner:
        for dr, dc in ((1, 0), (0, 1)):
            nbr = (r + dr, c + dc)
            if nbr in owner and owner[nbr] != owner[(r, c)]:
                actual_contacts.add(tuple(sorted(((r, c), nbr))))
    assert actual_contacts == expected_contacts

    coords = np.array(all_coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), color in pixel.items():
        out[r - r0, c - c0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the legend's unequal color pairs as a port dictionary, then dock the scattered objects into one tight assembly.

    Input structure:
    - A full-row or full-column divider isolates a small legend region from the main region.
    - The legend contains several disconnected horizontal 2-pixel color pairs; each pair states that the two colors are partners.
    - The main region contains several 4-connected multicolor objects with one dominant body color and boundary port pixels of other colors.

    Rule:
    - Read every horizontal legend pair as a bidirectional partner mapping between two different port colors.
    - Crop each main-region object to its bbox and keep its orientation fixed.
    - For each legend-defined partner pair, translate the two objects so the two port pixels touch edge-to-edge in the outward direction implied by their boundary sides.
    - Discard the legend and divider, then return the tight crop of the assembled foreground with all original colors preserved.

    Key insight:
    - Equal colors never identify a match here; only the corner legend tells which unequal port colors belong together, while the port side determines the docking direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    info = _split_input(g, bg)
    return _assemble(info["objects"], info["pairs"], bg)


GEN_ATTEMPTS = 500
GEN_OUTPUT_MAX_SPAN = 14
GEN_SCATTER_PADDING = 2
GEN_CANVAS_MIN = 20
GEN_CANVAS_MAX = 26
GEN_MIN_DENSITY = 0.45
GEN_MAX_DENSITY = 0.8


def _add_path(cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
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


def _boundary_coord(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "T":
        return 0, random.randint(1, w - 2)
    if side == "B":
        return h - 1, random.randint(1, w - 2)
    if side == "L":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _inner_neighbor(side: str, r: int, c: int) -> tuple[int, int]:
    dr, dc = SIDE_TO_DIR[side]
    return r - dr, c - dc


def _seed_sprite_cells(h: int, w: int, ports: list[tuple[int, str]]) -> tuple[set[tuple[int, int]], dict[int, tuple[int, int]]]:
    cells: set[tuple[int, int]] = set()
    used: set[tuple[int, int]] = set()
    port_locs: dict[int, tuple[int, int]] = {}
    used_sides = {side for _, side in ports}
    for color, side in ports:
        coord = _boundary_coord(side, h, w)
        while coord in used:
            coord = _boundary_coord(side, h, w)
        used.add(coord)
        port_locs[color] = coord
        cells.add(coord)
        cells.add(_inner_neighbor(side, *coord))
    for side in SIDE_TO_DIR:
        if side in used_sides:
            continue
        coord = _boundary_coord(side, h, w)
        while coord in used:
            coord = _boundary_coord(side, h, w)
        used.add(coord)
        cells.add(coord)
    return cells, port_locs


def _grow_sprite_cells(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    target_min = max(len(cells), int(np.ceil(GEN_MIN_DENSITY * h * w)))
    target_max = max(target_min, int(np.floor(GEN_MAX_DENSITY * h * w)))
    target = random.randint(target_min, target_max)
    while len(cells) < target:
        frontier = {
            (nr, nc)
            for r, c in list(cells)
            for dr, dc in DIR4
            for nr, nc in [(r + dr, c + dc)]
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
        }
        if not frontier:
            return False
        cells.add(random.choice(tuple(frontier)))
    return True


def _build_sprite(h: int, w: int, ports: list[tuple[int, str]], bg: int, body: int) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | tuple[None, None]:
    for _ in range(50):
        cells, port_locs = _seed_sprite_cells(h, w, ports)
        center = (random.randint(1, h - 2), random.randint(1, w - 2))
        cells.add(center)
        for cell in list(cells):
            _add_path(cells, cell, center)
        if not _grow_sprite_cells(cells, h, w):
            continue
        sprite = np.full((h, w), bg, dtype=int)
        for r, c in cells:
            sprite[r, c] = body
        for color, side in ports:
            pr, pc = port_locs[color]
            ir, ic = _inner_neighbor(side, pr, pc)
            if sprite[ir, ic] != body:
                break
            sprite[pr, pc] = color
        else:
            return sprite, port_locs
    return None, None


def _gen_tree(n: int) -> list[tuple[int, int]]:
    return [(random.randrange(child), child) for child in range(1, n)]


def _port_plan(n: int, colors: list[int]) -> tuple[list[list[tuple[int, str]]], list[tuple[int, int, int, int, str]], list[tuple[int, int]], dict[int, int]] | None:
    edges = _gen_tree(n)
    ports = [[] for _ in range(n)]
    used_sides = [set() for _ in range(n)]
    pair_records = []
    pairs = []
    partner: dict[int, int] = {}

    for edge_idx, (parent, child) in enumerate(edges):
        choices = [
            side
            for side in SIDE_TO_DIR
            if side not in used_sides[parent] and OPPOSITE_SIDE[side] not in used_sides[child]
        ]
        if not choices:
            return None
        side = random.choice(choices)
        child_side = OPPOSITE_SIDE[side]
        a = colors[2 * edge_idx]
        b = colors[2 * edge_idx + 1]
        used_sides[parent].add(side)
        used_sides[child].add(child_side)
        ports[parent].append((a, side))
        ports[child].append((b, child_side))
        pair_records.append((parent, child, a, b, side))
        pairs.append((a, b))
        partner[a] = b
        partner[b] = a

    if any(len(port_list) == 0 for port_list in ports):
        return None
    return ports, pair_records, pairs, partner


def _output_placements(pair_records: list[tuple[int, int, int, int, str]], port_locs: list[dict[int, tuple[int, int]]]) -> dict[int, tuple[int, int]] | None:
    placements = {0: (0, 0)}
    pending = list(pair_records)
    while pending:
        progressed = False
        rest = []
        for parent, child, a, b, side in pending:
            if parent not in placements:
                rest.append((parent, child, a, b, side))
                continue
            pr, pc = placements[parent]
            pa = port_locs[parent][a]
            pb = port_locs[child][b]
            dr, dc = SIDE_TO_DIR[side]
            placements[child] = (pr + pa[0] + dr - pb[0], pc + pa[1] + dc - pb[1])
            progressed = True
        if not progressed:
            return None
        pending = rest
    return placements


def _render_output(
    sprites: list[np.ndarray],
    placements: dict[int, tuple[int, int]],
    pair_records: list[tuple[int, int, int, int, str]],
    port_locs: list[dict[int, tuple[int, int]]],
    bg: int,
) -> np.ndarray | None:
    owner: dict[tuple[int, int], int] = {}
    colors: dict[tuple[int, int], int] = {}
    all_coords = []
    for idx, sprite in enumerate(sprites):
        br, bc = placements[idx]
        for rr, cc in np.argwhere(sprite != bg):
            pos = (br + int(rr), bc + int(cc))
            if pos in owner:
                return None
            owner[pos] = idx
            colors[pos] = int(sprite[rr, cc])
            all_coords.append(pos)

    if not all_coords:
        return None
    arr = np.array(all_coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    if r1 - r0 + 1 > GEN_OUTPUT_MAX_SPAN or c1 - c0 + 1 > GEN_OUTPUT_MAX_SPAN:
        return None

    expected = set()
    for parent, child, a, b, side in pair_records:
        pa = (placements[parent][0] + port_locs[parent][a][0], placements[parent][1] + port_locs[parent][a][1])
        pb = (placements[child][0] + port_locs[child][b][0], placements[child][1] + port_locs[child][b][1])
        dr, dc = SIDE_TO_DIR[side]
        if pb != (pa[0] + dr, pa[1] + dc):
            return None
        expected.add(tuple(sorted((pa, pb))))

    actual = set()
    for r, c in owner:
        for dr, dc in ((1, 0), (0, 1)):
            nbr = (r + dr, c + dc)
            if nbr in owner and owner[nbr] != owner[(r, c)]:
                actual.add(tuple(sorted(((r, c), nbr))))
    if actual != expected:
        return None

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), color in colors.items():
        out[r - r0, c - c0] = color
    return out


def _rects_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah + pad <= br or br + bh + pad <= ar or ac + aw + pad <= bc or bc + bw + pad <= ac


def _scatter_positions(shape: tuple[int, int], sprites: list[np.ndarray]) -> dict[int, tuple[int, int]] | None:
    h, w = shape
    order = sorted(range(len(sprites)), key=lambda idx: sprites[idx].shape[0] * sprites[idx].shape[1], reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    positions: dict[int, tuple[int, int]] = {}
    for idx in order:
        sh, sw = sprites[idx].shape
        if sh > h or sw > w:
            return None
        placed = False
        for _ in range(250):
            r = random.randint(0, h - sh)
            c = random.randint(0, w - sw)
            cand = (r, c, sh, sw)
            if all(_rects_separate(cand, other, GEN_SCATTER_PADDING) for other in rects):
                rects.append(cand)
                positions[idx] = (r, c)
                placed = True
                break
        if not placed:
            return None
    return positions


def _render_region(shape: tuple[int, int], sprites: list[np.ndarray], positions: dict[int, tuple[int, int]], bg: int) -> np.ndarray:
    region = np.full(shape, bg, dtype=int)
    for idx, sprite in enumerate(sprites):
        r, c = positions[idx]
        paste(region, sprite, r, c, transparent=bg)
    return region


def _legend_corner(side: str) -> str:
    choices = {
        "top": ["tl", "tr"],
        "bottom": ["bl", "br"],
        "left": ["tl", "bl"],
        "right": ["tr", "br"],
    }[side]
    return random.choice(choices)


def _render_legend(shape: tuple[int, int], pairs: list[tuple[int, int]], bg: int, corner: str) -> np.ndarray:
    legend = np.full(shape, bg, dtype=int)
    block_h = 2 * len(pairs) - 1
    if corner in {"tl", "tr"}:
        r0 = 1
    else:
        r0 = shape[0] - block_h - 1
    if corner in {"tl", "bl"}:
        c0 = 1
    else:
        c0 = shape[1] - 3
    for i, (a, b) in enumerate(pairs):
        rr = r0 + 2 * i
        left, right = (a, b) if random.random() < 0.5 else (b, a)
        legend[rr, c0] = left
        legend[rr, c0 + 1] = right
    return legend


def _compose_input(
    side: str,
    main_region: np.ndarray,
    legend: np.ndarray,
    divider: int,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    mh, mw = main_region.shape
    lh, lw = legend.shape
    if side == "top":
        grid = np.full((lh + 1 + mh, mw), bg, dtype=int)
        grid[:lh, :] = legend
        grid[lh, :] = divider
        grid[lh + 1 :, :] = main_region
        return grid, (lh + 1, 0)
    if side == "bottom":
        grid = np.full((mh + 1 + lh, mw), bg, dtype=int)
        grid[:mh, :] = main_region
        grid[mh, :] = divider
        grid[mh + 1 :, :] = legend
        return grid, (0, 0)
    if side == "left":
        grid = np.full((mh, lw + 1 + mw), bg, dtype=int)
        grid[:, :lw] = legend
        grid[:, lw] = divider
        grid[:, lw + 1 :] = main_region
        return grid, (0, lw + 1)
    grid = np.full((mh, mw + 1 + lw), bg, dtype=int)
    grid[:, :mw] = main_region
    grid[:, mw] = divider
    grid[:, mw + 1 :] = legend
    return grid, (0, 0)


def _port_catalog(
    input_origin: tuple[int, int],
    positions: dict[int, tuple[int, int]],
    ports_by_obj: list[list[tuple[int, str]]],
    port_locs: list[dict[int, tuple[int, int]]],
) -> dict[int, dict]:
    base_r, base_c = input_origin
    catalog = {}
    for idx, sprite_ports in enumerate(ports_by_obj):
        obj_r, obj_c = positions[idx]
        for color, side in sprite_ports:
            pr, pc = port_locs[idx][color]
            catalog[color] = {
                "obj": idx,
                "pos": (base_r + obj_r + pr, base_c + obj_c + pc),
                "dir": SIDE_TO_DIR[side],
            }
    return catalog


def _has_direction_ambiguity(catalog: dict[int, dict], partner: dict[int, int]) -> bool:
    for color, info in catalog.items():
        target_dir = (-info["dir"][0], -info["dir"][1])
        candidates = [
            other_color
            for other_color, other in catalog.items()
            if other["obj"] != info["obj"] and other["dir"] == target_dir
        ]
        if len(candidates) >= 2 and partner[color] in candidates:
            return True
    return False


def _nearest_wrong_opposite(catalog: dict[int, dict], partner: dict[int, int]) -> bool:
    for color, info in catalog.items():
        target_dir = (-info["dir"][0], -info["dir"][1])
        candidates = [
            (other_color, other)
            for other_color, other in catalog.items()
            if other["obj"] != info["obj"] and other["dir"] == target_dir
        ]
        if len(candidates) < 2:
            continue
        correct = next((other for other in candidates if other[0] == partner[color]), None)
        if correct is None:
            continue
        cr, cc = correct[1]["pos"]
        d_correct = abs(info["pos"][0] - cr) + abs(info["pos"][1] - cc)
        for other_color, other in candidates:
            if other_color == partner[color]:
                continue
            dist = abs(info["pos"][0] - other["pos"][0]) + abs(info["pos"][1] - other["pos"][1])
            if dist < d_correct:
                return True
    return False


def _canvas_and_regions(pair_count: int, side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    h = random.randint(GEN_CANVAS_MIN, GEN_CANVAS_MAX)
    w = random.randint(GEN_CANVAS_MIN, GEN_CANVAS_MAX)
    if side in {"top", "bottom"}:
        legend_h = 2 * pair_count + random.randint(2, 3)
        return (h - legend_h - 1, w), (legend_h, w)
    legend_w = random.randint(4, 6)
    return (h, w - legend_w - 1), (h, legend_w)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice([3, 4, 4])
    palette = list(range(10))
    bg = random.choice(palette)
    body = random.choice([c for c in palette if c != bg])
    divider = random.choice([c for c in palette if c not in {bg, body}])
    port_pool = [c for c in palette if c not in {bg, body, divider}]
    port_colors = random.sample(port_pool, 2 * (n_objects - 1))

    plan = _port_plan(n_objects, port_colors)
    if plan is None:
        return None
    ports_by_obj, pair_records, pairs, partner = plan

    sprites = []
    port_locs = []
    for idx in range(n_objects):
        if n_objects == 4:
            h = random.randint(3, 5)
            w = random.randint(3, 5)
        else:
            h = random.randint(3, 6)
            w = random.randint(3, 6)
        sprite, locs = _build_sprite(h, w, ports_by_obj[idx], bg, body)
        if sprite is None:
            return None
        sprites.append(sprite)
        port_locs.append(locs)

    placements = _output_placements(pair_records, port_locs)
    if placements is None:
        return None
    out = _render_output(sprites, placements, pair_records, port_locs, bg)
    if out is None:
        return None

    for _ in range(80):
        side = random.choice(["top", "bottom", "left", "right"])
        main_shape, legend_shape = _canvas_and_regions(len(pairs), side)
        if min(main_shape) < 8 or min(legend_shape) < 4:
            continue
        positions = _scatter_positions(main_shape, sprites)
        if positions is None:
            continue
        main_region = _render_region(main_shape, sprites, positions, bg)
        legend = _render_legend(legend_shape, pairs, bg, _legend_corner(side))
        inp, main_origin = _compose_input(side, main_region, legend, divider, bg)
        if inp.ndim != 2 or max(inp.shape) > 30:
            continue
        catalog = _port_catalog(main_origin, positions, ports_by_obj, port_locs)
        if not _has_direction_ambiguity(catalog, partner):
            continue
        if not _nearest_wrong_opposite(catalog, partner):
            continue
        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-mediated docking input")
