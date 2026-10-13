from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr, sc in np.argwhere(grid == color):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if _in_bounds((h, w), rr, cc) and not seen[rr, cc] and int(grid[rr, cc]) == color:
                    seen[rr, cc] = True
                    q.append((rr, cc))

        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        comps.append(
            {
                "coords": [(int(r), int(c)) for r, c in coords],
                "coord_set": {(int(r), int(c)) for r, c in coords},
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": len(coords),
                "markers": [],
            }
        )
    return comps


def _piece_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    candidates = [(count, -color, color) for color, count in counts.items() if color != int(bg)]
    assert candidates, "expected non-background puzzle pieces and markers"
    return int(max(candidates)[2])


def _marker_pairs(grid: np.ndarray, bg: int, piece_color: int, pieces: list[dict]) -> dict[int, tuple[dict, dict]]:
    owner: dict[tuple[int, int], int] = {}
    for i, piece in enumerate(pieces):
        for p in piece["coords"]:
            owner[p] = i

    pairs: dict[int, tuple[dict, dict]] = {}
    for color in sorted(set(map(int, grid.ravel())) - {int(bg), int(piece_color)}):
        coords = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
        if len(coords) != 2:
            continue

        infos: list[dict] = []
        for r, c in coords:
            adjacent: list[tuple[int, tuple[int, int]]] = []
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                pid = owner.get((rr, cc))
                if pid is not None:
                    adjacent.append((pid, (dr, dc)))

            piece_ids = {pid for pid, _ in adjacent}
            assert len(piece_ids) == 1, "each marker pixel must touch exactly one piece"
            pid = next(iter(piece_ids))
            inward = []
            for d in DIR4:
                if any(pid2 == pid and dd == d for pid2, dd in adjacent):
                    inward.append(d)
            assert inward, "marker must have an inward direction"
            info = {"piece": pid, "pos": (r, c), "inward": inward, "color": int(color)}
            pieces[pid]["markers"].append(info)
            infos.append(info)

        assert infos[0]["piece"] != infos[1]["piece"], "paired markers must be on different pieces"
        pairs[int(color)] = (infos[0], infos[1])

    assert pairs, "expected at least one paired interface marker"
    return pairs


def _outward_match(placed_marker: dict, other_marker: dict) -> tuple[int, int]:
    placed_out = [(-dr, -dc) for dr, dc in placed_marker["inward"]]
    other_in = list(other_marker["inward"])
    for d in DIR4:
        if d in placed_out and d in other_in:
            return d
    return placed_out[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble same-colored puzzle fragments by matching paired colored edge markers.

    Input structure:
    - bg = most common color.
    - Several 4-connected regions of one shared piece color form separated fragments.
    - Small colored marker pixels sit on fragment edges; each marker color appears exactly twice, on two different fragments.
    - Marker colors are distinct from both the background and the shared piece color.

    Rule:
    - Choose the largest piece-color fragment as the anchor and keep it at its input position.
    - For each paired marker color, when one marked fragment is already placed and the other is not, translate the other fragment so its same-color marker is 4-adjacent just outside the placed marker.
    - Only translations are allowed; rotations and reflections are not used. Newly placed fragments expose their other markers and continue the assembly.
    - After all fragments are assembled, replace every marker pixel with the shared piece color.

    Key insight: Matching marker colors are docking instructions; the marker's side on its fragment gives the direction of the translation, and the markers disappear into the completed same-colored shape.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background(g)
    piece_color = _piece_color(g, bg)
    pieces = _components_of_color(g, piece_color)
    assert pieces, "expected at least one piece"
    pairs = _marker_pairs(g, bg, piece_color, pieces)

    anchor = max(
        range(len(pieces)),
        key=lambda i: (int(pieces[i]["area"]), -int(pieces[i]["bbox"][0]), -int(pieces[i]["bbox"][1])),
    )
    offsets: dict[int, tuple[int, int]] = {anchor: (0, 0)}

    changed = True
    while changed:
        changed = False
        for a, b in pairs.values():
            pa, pb = int(a["piece"]), int(b["piece"])
            if pa in offsets and pb not in offsets:
                dr, dc = _outward_match(a, b)
                ar, ac = a["pos"]
                br, bc = b["pos"]
                off_a = offsets[pa]
                target = (ar + off_a[0] + dr, ac + off_a[1] + dc)
                offsets[pb] = (target[0] - br, target[1] - bc)
                changed = True
            elif pb in offsets and pa not in offsets:
                dr, dc = _outward_match(b, a)
                br, bc = b["pos"]
                ar, ac = a["pos"]
                off_b = offsets[pb]
                target = (br + off_b[0] + dr, bc + off_b[1] + dc)
                offsets[pa] = (target[0] - ar, target[1] - ac)
                changed = True

    assert len(offsets) == len(pieces), "all pieces must be connected by marker pairs"

    out = np.full_like(g, int(bg))
    painted: set[tuple[int, int]] = set()
    for pid, piece in enumerate(pieces):
        off_r, off_c = offsets[pid]
        cells = list(piece["coords"]) + [m["pos"] for m in piece["markers"]]
        for r, c in cells:
            rr, cc = int(r) + int(off_r), int(c) + int(off_c)
            assert _in_bounds((h, w), rr, cc), "assembled piece must remain inside the grid"
            if (rr, cc) in painted:
                assert int(out[rr, cc]) == int(piece_color), "assembled pieces must not conflict"
            out[rr, cc] = int(piece_color)
            painted.add((rr, cc))

    return out


# --------------- generator ---------------

GEN_RETRIES = 300
PLACE_RETRIES = 300
MIN_GRID = 18
MAX_GRID = 30
MIN_PIECES = 3
MAX_PIECES = 5


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _random_polyomino(area: int, rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    p = (r + dr, c + dc)
                    if p not in cells:
                        frontier.append(p)
            cells.add(rng.choice(frontier))
            cells = _normalize(cells)
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            if max(rs) - min(rs) + 1 > 6 or max(cs) - min(cs) + 1 > 6:
                break
        if len(cells) == area:
            return _normalize(cells)
    raise RuntimeError("failed to create a random piece")


def _single_touch_port(cells: set[tuple[int, int]], direction: tuple[int, int], used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    ports = []
    dr, dc = direction
    for r, c in sorted(cells):
        marker = (r + dr, c + dc)
        if marker in cells or marker in used:
            continue
        touches = 0
        for nr, nc in ((marker[0] + rr, marker[1] + cc) for rr, cc in DIR4):
            if (nr, nc) in cells:
                touches += 1
        if touches == 1:
            ports.append(marker)
    return ports


def _piece_cells(piece: dict, origin: tuple[int, int], include_markers: bool = True) -> list[tuple[int, int]]:
    out = [(origin[0] + r, origin[1] + c) for r, c in piece["cells"]]
    if include_markers:
        out.extend((origin[0] + r, origin[1] + c) for r, c in piece["markers"])
    return out


def _bbox_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _dilate_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _build_specs(rng: random.Random) -> tuple[list[dict], list[dict]] | None:
    n = rng.randint(MIN_PIECES, MAX_PIECES)
    anchor_area = rng.randint(10, 14)
    pieces = [{"cells": _random_polyomino(anchor_area, rng), "markers": []}]
    for _ in range(1, n):
        pieces.append({"cells": _random_polyomino(rng.randint(4, 8), rng), "markers": []})

    parents = [0]
    edges = []
    used_markers = [set() for _ in range(n)]
    for child in range(1, n):
        parent = rng.randrange(child)
        direction = rng.choice(DIR4)
        parent_ports = _single_touch_port(pieces[parent]["cells"], direction, used_markers[parent])
        child_ports = _single_touch_port(pieces[child]["cells"], (-direction[0], -direction[1]), used_markers[child])
        if not parent_ports or not child_ports:
            return None
        pm = rng.choice(parent_ports)
        cm = rng.choice(child_ports)
        used_markers[parent].add(pm)
        used_markers[child].add(cm)
        pieces[parent]["markers"].append(pm)
        pieces[child]["markers"].append(cm)
        edges.append({"parent": parent, "child": child, "dir": direction, "pm": pm, "cm": cm})
        parents.append(parent)

    origins = {0: (0, 0)}
    for edge in edges:
        parent = edge["parent"]
        child = edge["child"]
        pr, pc = origins[parent]
        dr, dc = edge["dir"]
        pmr, pmc = edge["pm"]
        cmr, cmc = edge["cm"]
        target = (pr + pmr + dr, pc + pmc + dc)
        origins[child] = (target[0] - cmr, target[1] - cmc)

    occupied: set[tuple[int, int]] = set()
    for i, piece in enumerate(pieces):
        for cell in _piece_cells(piece, origins[i], include_markers=True):
            if cell in occupied:
                return None
            occupied.add(cell)

    rel_box = _bbox_cells(list(occupied))
    if rel_box[2] - rel_box[0] + 1 > 18 or rel_box[3] - rel_box[1] + 1 > 18:
        return None

    for piece in pieces:
        piece["markers"] = []
    for edge in edges:
        pieces[edge["parent"]]["markers"].append(edge["pm"])
        pieces[edge["child"]]["markers"].append(edge["cm"])
    return pieces, edges


def _paint_piece(
    grid: np.ndarray,
    piece: dict,
    origin: tuple[int, int],
    piece_color: int,
    marker_colors: dict[tuple[int, tuple[int, int]], int],
    pid: int,
) -> None:
    for r, c in piece["cells"]:
        grid[origin[0] + r, origin[1] + c] = int(piece_color)
    for marker in piece["markers"]:
        grid[origin[0] + marker[0], origin[1] + marker[1]] = int(marker_colors[(pid, marker)])


def _place_scattered(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    piece: dict,
    pid: int,
    final_origin: tuple[int, int],
    piece_color: int,
    marker_colors: dict[tuple[int, tuple[int, int]], int],
    rng: random.Random,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    h, w = grid.shape
    local = [(r, c) for r, c in piece["cells"]] + list(piece["markers"])
    r0, c0, r1, c1 = _bbox_cells(local)
    rows = list(range(-r0, h - r1))
    cols = list(range(-c0, w - c1))
    rng.shuffle(rows)
    rng.shuffle(cols)
    candidates = [(r, c) for r in rows for c in cols]
    rng.shuffle(candidates)

    for origin in candidates[:PLACE_RETRIES]:
        if origin == final_origin:
            continue
        cells = set(_piece_cells(piece, origin, include_markers=True))
        if cells & reserved:
            continue
        _paint_piece(grid, piece, origin, piece_color, marker_colors, pid)
        return origin, cells
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _build_specs(rng)
    if specs is None:
        return None
    pieces, edges = specs

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors.pop()
    piece_color = colors.pop()
    if len(colors) < len(edges):
        return None
    edge_colors = colors[: len(edges)]

    marker_colors: dict[tuple[int, tuple[int, int]], int] = {}
    origins_rel = {0: (0, 0)}
    for edge, color in zip(edges, edge_colors):
        marker_colors[(edge["parent"], edge["pm"])] = int(color)
        marker_colors[(edge["child"], edge["cm"])] = int(color)
        parent = edge["parent"]
        child = edge["child"]
        pr, pc = origins_rel[parent]
        dr, dc = edge["dir"]
        target = (pr + edge["pm"][0] + dr, pc + edge["pm"][1] + dc)
        origins_rel[child] = (target[0] - edge["cm"][0], target[1] - edge["cm"][1])

    rel_cells = []
    for pid, piece in enumerate(pieces):
        rel_cells.extend(_piece_cells(piece, origins_rel[pid], include_markers=True))
    rr0, cc0, rr1, cc1 = _bbox_cells(rel_cells)
    asm_h = rr1 - rr0 + 1
    asm_w = cc1 - cc0 + 1
    if asm_h > 18 or asm_w > 18:
        return None

    min_h = max(MIN_GRID, asm_h + 8)
    min_w = max(MIN_GRID, asm_w + 8)
    if min_h > MAX_GRID or min_w > MAX_GRID:
        return None
    h = rng.randint(min_h, MAX_GRID)
    w = rng.randint(min_w, MAX_GRID)

    shift_r = rng.randint(1 - rr0, h - rr1 - 2)
    shift_c = rng.randint(1 - cc0, w - cc1 - 2)
    final_origins = {pid: (shift_r + o[0], shift_c + o[1]) for pid, o in origins_rel.items()}

    expected = np.full((h, w), int(bg), dtype=int)
    for pid, piece in enumerate(pieces):
        for r, c in _piece_cells(piece, final_origins[pid], include_markers=True):
            expected[r, c] = int(piece_color)

    inp = np.full((h, w), int(bg), dtype=int)
    _paint_piece(inp, pieces[0], final_origins[0], piece_color, marker_colors, 0)
    occupied = set(_piece_cells(pieces[0], final_origins[0], include_markers=True))
    reserved = _dilate_cells(occupied, h, w)

    for pid in range(1, len(pieces)):
        placed = _place_scattered(
            inp,
            reserved,
            pieces[pid],
            pid,
            final_origins[pid],
            piece_color,
            marker_colors,
            rng,
        )
        if placed is None:
            return None
        _origin, cells = placed
        reserved |= _dilate_cells(cells, h, w)

    counts = Counter(map(int, inp.ravel()))
    if counts[int(bg)] != max(counts.values()):
        return None
    if any(counts[int(c)] != 2 for c in edge_colors):
        return None
    piece_areas = [len(p["cells"]) for p in pieces]
    if piece_areas[0] <= max(piece_areas[1:]):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle-piece assembly input")
