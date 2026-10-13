from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

ARC_COLORS = tuple(range(10))
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8: tuple[Coord, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
DIAG_DIRS: tuple[Coord, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
AXIS_DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_EMIT_DIRS: tuple[Coord, ...] = DIAG_DIRS + AXIS_DIRS


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sign(v: int) -> int:
    return 0 if v == 0 else (1 if v > 0 else -1)


def _components8(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[Coord] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _edge_marker(grid: np.ndarray, bg: int) -> tuple[Coord, int, Coord]:
    h, w = grid.shape
    edge_cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == bg:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                edge_cells.append((r, c))

    assert len(edge_cells) == 1, "expected exactly one non-background edge marker"
    r, c = edge_cells[0]
    assert (r in (0, h - 1)) ^ (c in (0, w - 1)), "edge marker must not be a corner"
    if r == 0:
        direction = (1, 0)
    elif r == h - 1:
        direction = (-1, 0)
    elif c == 0:
        direction = (0, 1)
    else:
        direction = (0, -1)
    return (r, c), int(grid[r, c]), direction


def _triangle_cells(front: Coord, direction: Coord) -> tuple[Coord, Coord, Coord]:
    """Cells for the sprite whose foremost pixel is front and whose ray goes direction."""
    r, c = front
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return (front, (r - dr, c), (r, c - dc))
    if dr:
        return (front, (r - dr, c - 1), (r - dr, c + 1))
    return (front, (r - 1, c - dc), (r + 1, c - dc))


def _triangle_info(comp: list[Coord]) -> dict:
    assert len(comp) == 3, "every foreground object away from the edge marker must be a 3-cell triangle"
    cells = set(comp)

    deg4 = {
        p: sum((p[0] + dr, p[1] + dc) in cells for dr, dc in DIRS4)
        for p in comp
    }
    elbows = [p for p, degree in deg4.items() if degree == 2]
    if len(elbows) == 1:
        front = elbows[0]
        fr, fc = front
        arms = []
        for r, c in comp:
            if (r, c) == front:
                continue
            dr, dc = r - fr, c - fc
            assert abs(dr) + abs(dc) == 1, "L triangle arms must be 4-neighbors of the elbow"
            arms.append((dr, dc))
        direction = (-_sign(arms[0][0] + arms[1][0]), -_sign(arms[0][1] + arms[1][1]))
        assert abs(direction[0]) + abs(direction[1]) == 2, "L triangle must encode a diagonal direction"
        return {"cells": tuple(sorted(cells)), "front": front, "direction": direction, "kind": "L"}

    deg8 = {
        p: sum((p[0] + dr, p[1] + dc) in cells for dr, dc in DIRS8)
        for p in comp
    }
    apexes = [p for p, degree in deg8.items() if degree == 2]
    assert len(apexes) == 1, "V triangle must have one apex touching both diagonal legs"
    front = apexes[0]
    fr, fc = front
    offsets = [(r - fr, c - fc) for r, c in comp if (r, c) != front]
    assert all(abs(dr) == 1 and abs(dc) == 1 for dr, dc in offsets), "V legs must be diagonal from the apex"
    direction = (-(offsets[0][0] + offsets[1][0]) // 2, -(offsets[0][1] + offsets[1][1]) // 2)
    assert abs(direction[0]) + abs(direction[1]) == 1, "V triangle must encode a cardinal direction"
    return {"cells": tuple(sorted(cells)), "front": front, "direction": direction, "kind": "V"}


def _find_triangles(grid: np.ndarray, bg: int, marker: Coord) -> list[dict]:
    fg = (grid != bg).copy()
    fg[marker] = False
    comps = _components8(fg)
    assert comps, "expected at least one triangle"
    triangles = []
    for comp in comps:
        info = _triangle_info(comp)
        colors = {int(grid[r, c]) for r, c in info["cells"]}
        assert len(colors) == 1, "each triangle must be monochrome"
        info["color"] = colors.pop()
        triangles.append(info)
    return triangles


def _trace(
    grid: np.ndarray,
    out: np.ndarray,
    start: Coord,
    direction: Coord,
    fuse_color: int,
    bg: int,
    cell_to_triangle: dict[Coord, int],
) -> tuple[int | None, list[Coord]]:
    h, w = grid.shape
    r, c = start
    painted: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        pos = (r, c)
        tri_id = cell_to_triangle.get(pos)
        if tri_id is not None:
            return tri_id, painted
        if int(grid[r, c]) != bg:
            return None, painted
        out[r, c] = fuse_color
        painted.append(pos)
        r += direction[0]
        c += direction[1]
    return None, painted


def _solve(grid: np.ndarray, *, collect_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    marker, fuse_color, fuse_dir = _edge_marker(g, bg)
    triangles = _find_triangles(g, bg, marker)

    cell_to_triangle: dict[Coord, int] = {}
    for idx, tri in enumerate(triangles):
        for cell in tri["cells"]:
            assert cell not in cell_to_triangle, "triangles must not overlap"
            cell_to_triangle[cell] = idx

    out = g.copy()
    lit: set[int] = set()
    q: deque[tuple[Coord, Coord, int | None]] = deque(
        [((marker[0] + fuse_dir[0], marker[1] + fuse_dir[1]), fuse_dir, None)]
    )
    ray_painted: list[list[Coord]] = []
    hit_events: list[tuple[int | None, int | None]] = []
    repeated_hits = 0

    while q:
        start, direction, parent = q.popleft()
        hit, painted = _trace(g, out, start, direction, fuse_color, bg, cell_to_triangle)
        ray_painted.append(painted)
        hit_events.append((parent, hit))
        if hit is None:
            continue
        if hit in lit:
            repeated_hits += 1
            continue
        lit.add(hit)
        tri = triangles[hit]
        for r, c in tri["cells"]:
            out[r, c] = fuse_color
        fr, fc = tri["front"]
        dr, dc = tri["direction"]
        q.append(((fr + dr, fc + dc), (dr, dc), hit))

    if collect_meta:
        return out, {
            "bg": bg,
            "marker": marker,
            "fuse_color": fuse_color,
            "triangles": triangles,
            "lit": lit,
            "ray_painted": ray_painted,
            "hit_events": hit_events,
            "repeated_hits": repeated_hits,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light a chain of triangle sprites from a single edge fuse ray, recoloring only reached sprites.

    Input structure:
    - The dominant color is the background. Exactly one non-background single cell sits on a non-corner edge and is the fuse color.
    - All other foreground objects are separated monochrome 3-cell triangle sprites: 2x2 L-shapes missing one corner or diagonal-legged V-shapes.
    - Sprite colors are visible object labels only; their geometry, not their color id, determines the outgoing direction.

    Rule:
    - The edge fuse fires straight inward; every original background cell on the ray becomes the fuse color until the ray hits a triangle or leaves the grid.
    - A newly hit triangle is recolored entirely to the fuse color, then emits one new ray from its foremost pixel outward: L-shapes fire diagonally away from the missing 2x2 corner, and V-shapes fire opposite their open side.
    - Rays stop at the first triangle they meet. A triangle already lit by an earlier ray stops the later ray but does not emit again; never-hit triangles keep their original color and position.

    Key insight:
    - Recoloring is governed by reachability in the geometry-encoded ray chain, so the solution must trace propagation step by step instead of recoloring every triangle.
    """
    return _solve(grid)


# ----------------------------- generator -----------------------------

GEN_RETRIES = 500
GEN_GRID_MIN = 15
GEN_GRID_MAX = 25
GEN_CHAIN_RANGE = (3, 5)
GEN_DISTRACTOR_RANGE = (2, 4)
GEN_MIN_RAY_GAP = 2


def _ray_cells(shape: tuple[int, int], start: Coord, direction: Coord) -> list[Coord]:
    h, w = shape
    r, c = start
    cells: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += direction[0]
        c += direction[1]
    return cells


def _candidate_placements(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            for direction in ALL_EMIT_DIRS:
                cells = _triangle_cells((r, c), direction)
                if not all(1 <= rr < h - 1 and 1 <= cc < w - 1 for rr, cc in cells):
                    continue
                kind = "L" if abs(direction[0]) + abs(direction[1]) == 2 else "V"
                out.append(
                    {
                        "front": (r, c),
                        "direction": direction,
                        "cells": tuple(sorted(cells)),
                        "cell_set": frozenset(cells),
                        "kind": kind,
                    }
                )
    return out


def _halo(cells: tuple[Coord, ...], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _first_intersection_distance(ray: list[Coord], cells: set[Coord]) -> int | None:
    for dist, cell in enumerate(ray):
        if cell in cells:
            return dist
    return None


def _fits_placement(placement: dict, occupied_halo: set[Coord], blocked: set[Coord]) -> bool:
    return not (placement["cell_set"] & occupied_halo) and not (placement["cell_set"] & blocked)


def _choose_chain_sprite(
    rng: random.Random,
    shape: tuple[int, int],
    placements: list[dict],
    occupied: set[Coord],
    occupied_halo: set[Coord],
    source: Coord,
    ray_dir: Coord,
    emit_dir: Coord,
) -> dict | None:
    ray = _ray_cells(shape, source, ray_dir)
    candidates = []
    for placement in placements:
        if placement["direction"] != emit_dir:
            continue
        if not _fits_placement(placement, occupied_halo, set()):
            continue
        dist = _first_intersection_distance(ray, set(placement["cells"]))
        if dist is None or dist < GEN_MIN_RAY_GAP:
            continue
        blocked_before = False
        for earlier in ray[:dist]:
            if earlier in occupied:
                blocked_before = True
                break
        if not blocked_before:
            candidates.append((dist, placement))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    short_pool = candidates[: min(len(candidates), 20)]
    return rng.choice(short_pool)[1]


def _add_placement(
    placement: dict,
    shape: tuple[int, int],
    placed: list[dict],
    occupied: set[Coord],
    occupied_halo: set[Coord],
) -> None:
    placed.append(dict(placement))
    occupied.update(placement["cells"])
    occupied_halo.update(_halo(placement["cells"], shape))


def _make_marker(rng: random.Random, shape: tuple[int, int]) -> tuple[Coord, Coord]:
    h, w = shape
    side = rng.choice(("top", "bottom", "left", "right"))
    if side == "top":
        c = rng.randrange(2, w - 2)
        return (0, c), (1, 0)
    if side == "bottom":
        c = rng.randrange(2, w - 2)
        return (h - 1, c), (-1, 0)
    if side == "left":
        r = rng.randrange(2, h - 2)
        return (r, 0), (0, 1)
    r = rng.randrange(2, h - 2)
    return (r, w - 1), (0, -1)


def _initial_emit_dirs(rng: random.Random, chain_len: int) -> list[Coord]:
    dirs = [rng.choice(ALL_EMIT_DIRS) for _ in range(chain_len)]
    if not any(abs(dr) + abs(dc) == 2 for dr, dc in dirs):
        dirs[rng.randrange(chain_len)] = rng.choice(DIAG_DIRS)
    if not any(abs(dr) + abs(dc) == 1 for dr, dc in dirs):
        dirs[rng.randrange(chain_len)] = rng.choice(AXIS_DIRS)
    return dirs


def _place_chain(
    rng: random.Random,
    shape: tuple[int, int],
    placements: list[dict],
    marker: Coord,
    fuse_dir: Coord,
    chain_len: int,
) -> tuple[list[dict], set[Coord], set[Coord]] | None:
    placed: list[dict] = []
    occupied = {marker}
    occupied_halo = _halo((marker,), shape)
    source = (marker[0] + fuse_dir[0], marker[1] + fuse_dir[1])
    ray_dir = fuse_dir
    emit_dirs = _initial_emit_dirs(rng, chain_len)

    for emit_dir in emit_dirs:
        choice = _choose_chain_sprite(
            rng, shape, placements, occupied, occupied_halo, source, ray_dir, emit_dir
        )
        if choice is None:
            return None
        _add_placement(choice, shape, placed, occupied, occupied_halo)
        fr, fc = choice["front"]
        dr, dc = choice["direction"]
        source = (fr + dr, fc + dc)
        ray_dir = (dr, dc)

    return placed, occupied, occupied_halo


def _choose_distractor(
    rng: random.Random,
    placements: list[dict],
    shape: tuple[int, int],
    occupied_halo: set[Coord],
    required_kind: str | None = None,
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if required_kind is not None and placement["kind"] != required_kind:
            continue
        if _fits_placement(placement, occupied_halo, set()):
            return placement
    return None


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    fuse: int,
    marker: Coord,
    placements: list[dict],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    grid[marker] = fuse
    colors = [c for c in ARC_COLORS if c not in (bg, fuse)]
    rng.shuffle(colors)
    assert len(colors) >= len(placements)
    for placement, color in zip(placements, colors):
        for r, c in placement["cells"]:
            grid[r, c] = int(color)
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    triangles = meta["triangles"]
    lit = set(meta["lit"])
    total = len(triangles)
    if not (4 <= total <= 8 and 2 <= len(lit) < total):
        return False
    kinds = {tri["kind"] for tri in triangles}
    lit_kinds = {triangles[i]["kind"] for i in lit}
    if kinds != {"L", "V"} or lit_kinds != {"L", "V"}:
        return False
    if sum(len(cells) for cells in meta["ray_painted"]) < len(lit) + 2:
        return False
    if not any(parent is not None and hit is not None for parent, hit in meta["hit_events"]):
        return False
    if meta["repeated_hits"] <= 0:
        return False
    if not any(i not in lit for i in range(total)):
        return False
    if np.array_equal(inp, out):
        return False

    fuse = int(meta["fuse_color"])
    for i in lit:
        if not all(int(out[r, c]) == fuse for r, c in triangles[i]["cells"]):
            return False
    for i, tri in enumerate(triangles):
        if i in lit:
            continue
        if not all(int(out[r, c]) == int(inp[r, c]) for r, c in tri["cells"]):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = rng.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    shape = (h, w)
    placements = _candidate_placements(shape)
    marker, fuse_dir = _make_marker(rng, shape)
    chain_len = rng.randint(*GEN_CHAIN_RANGE)

    chain = _place_chain(rng, shape, placements, marker, fuse_dir, chain_len)
    if chain is None:
        return None
    placed, occupied, occupied_halo = chain

    kinds_present = {p["kind"] for p in placed}
    distractor_count = rng.randint(*GEN_DISTRACTOR_RANGE)
    for i in range(distractor_count):
        need_kind = None
        if i == 0 and "L" not in kinds_present:
            need_kind = "L"
        elif i == 0 and "V" not in kinds_present:
            need_kind = "V"
        choice = _choose_distractor(rng, placements, shape, occupied_halo, need_kind)
        if choice is None:
            return None
        _add_placement(choice, shape, placed, occupied, occupied_halo)
        kinds_present.add(choice["kind"])

    if len(placed) > 8:
        return None

    colors = list(ARC_COLORS)
    bg = rng.choice(colors)
    fuse = rng.choice([c for c in colors if c != bg])
    inp = _render_grid(shape, bg, fuse, marker, placed, rng)

    try:
        out, meta = _solve(inp, collect_meta=True)
    except Exception:
        return None
    if not _semantic_checks(inp, out, meta):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid chain-lighting input after {GEN_RETRIES} attempts")
