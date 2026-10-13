from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
MAROON = 9
PORTAL_COLORS = [1, 2, 3, 6, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


Coord = tuple[int, int]


@dataclass(frozen=True)
class Portal:
    color: int
    center: Coord
    marker_dir: Coord

    @property
    def marker(self) -> Coord:
        return (self.center[0] + self.marker_dir[0], self.center[1] + self.marker_dir[1])


def _add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1])


def _scale(d: Coord, k: int) -> Coord:
    return (d[0] * k, d[1] * k)


def _neg(d: Coord) -> Coord:
    return (-d[0], -d[1])


def _in_bounds(h: int, w: int, cell: Coord) -> bool:
    return 0 <= cell[0] < h and 0 <= cell[1] < w


def _unit(delta_r: int, delta_c: int) -> Coord | None:
    if delta_r == 0 and delta_c == 0:
        return None
    if delta_r == 0:
        return (0, 1 if delta_c > 0 else -1)
    if delta_c == 0:
        return (1 if delta_r > 0 else -1, 0)
    return None


def _segment_cells(a: Coord, b: Coord) -> tuple[Coord, ...]:
    step = _unit(b[0] - a[0], b[1] - a[1])
    assert step is not None, "segment endpoints must share a row or column"
    dist = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return tuple((a[0] + step[0] * k, a[1] + step[1] * k) for k in range(dist + 1))


def _portal_cells(center: Coord, marker_dir: Coord, h: int, w: int) -> set[Coord] | None:
    r, c = center
    if marker_dir[0] != 0:
        body = {(r, c - 1), (r, c), (r, c + 1)}
    else:
        body = {(r - 1, c), (r, c), (r + 1, c)}
    marker = (r + marker_dir[0], c + marker_dir[1])
    cells = set(body)
    cells.add(marker)
    if all(_in_bounds(h, w, cell) for cell in cells):
        return cells
    return None


def _body_cells(center: Coord, marker_dir: Coord) -> set[Coord]:
    r, c = center
    if marker_dir[0] != 0:
        return {(r, c - 1), (r, c), (r, c + 1)}
    return {(r - 1, c), (r, c), (r + 1, c)}


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _portal_from_component(grid: np.ndarray, color: int, coords: Iterable[Coord]) -> Portal:
    pts = sorted((int(r), int(c)) for r, c in coords)
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) == 1 and len(cols) == 3 and cols[2] - cols[0] == 2:
        center = (rows[0], cols[1])
        side_dirs = [(-1, 0), (1, 0)]
    elif len(cols) == 1 and len(rows) == 3 and rows[2] - rows[0] == 2:
        center = (rows[1], cols[0])
        side_dirs = [(0, -1), (0, 1)]
    else:
        raise AssertionError("portal body must be a straight 3-pixel segment")

    h, w = grid.shape
    marker_dirs = [d for d in side_dirs if _in_bounds(h, w, _add(center, d)) and int(grid[_add(center, d)]) == MAROON]
    assert len(marker_dirs) == 1, "each portal must have exactly one entrance marker"
    return Portal(color=int(color), center=center, marker_dir=marker_dirs[0])


def _parse_portals(grid: np.ndarray) -> dict[int, tuple[Portal, Portal]]:
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in {BG, YELLOW, MAROON})
    portals: dict[int, tuple[Portal, Portal]] = {}
    for color in colors:
        comps = _components(grid == color)
        assert len(comps) == 2, "each portal color must appear as exactly two separate portals"
        pair = tuple(_portal_from_component(grid, color, comp) for comp in comps)
        portals[color] = pair  # type: ignore[assignment]
    assert portals, "expected at least one portal pair"
    return portals


def _source_coord(grid: np.ndarray) -> Coord:
    ys = np.argwhere(grid == YELLOW)
    assert ys.shape == (1, 2), "expected a unique yellow source"
    return (int(ys[0, 0]), int(ys[0, 1]))


def _initial_candidates(grid: np.ndarray, source: Coord, portals: dict[int, tuple[Portal, Portal]]) -> list[tuple[int, Portal, Coord]]:
    candidates: list[tuple[int, Portal, Coord]] = []
    for pair in portals.values():
        for portal in pair:
            dr = portal.center[0] - source[0]
            dc = portal.center[1] - source[1]
            step = _unit(dr, dc)
            if step is None or portal.marker_dir != _neg(step):
                continue
            cells = _segment_cells(source, portal.center)[1:-1]
            if portal.marker not in cells:
                continue
            if all((cell == portal.marker) or int(grid[cell]) == BG for cell in cells):
                candidates.append((abs(dr) + abs(dc), portal, step))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Trace a yellow light ray through paired portals, letting each paired entrance marker determine the exit direction.

    Input structure:
    - The background is Black.
    - Each portal color forms exactly two separate 3-pixel straight segments;
      every segment is horizontal or vertical and has one Maroon marker adjacent
      to its middle pixel on a perpendicular side.
    - A single Yellow source pixel has a clear row-or-column sightline toward
      the marker side of the nearest portal on its beam path.

    Rule:
    - Start a ray at the Yellow source and move in the straight direction of the
      nearest visible portal; every traversed Black cell becomes Yellow.
    - When the ray reaches a portal from its marker side, teleport to the other
      portal of the same color.
    - The teleported ray leaves that paired portal in the direction from the
      portal center toward its own Maroon marker, then continues straight.
    - Repeat portal transfers until the ray leaves the grid; portal pixels,
      Maroon markers, and the original Yellow source stay unchanged.

    Key insight:
    - A portal pair is matched by color, but the outgoing direction is not the
      incoming direction; it is redefined by the paired portal's marker side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    portals = _parse_portals(g)
    center_to_portal: dict[Coord, Portal] = {}
    pair_lookup: dict[Coord, Portal] = {}
    for pair in portals.values():
        a, b = pair
        center_to_portal[a.center] = a
        center_to_portal[b.center] = b
        pair_lookup[a.center] = b
        pair_lookup[b.center] = a

    source = _source_coord(g)
    candidates = _initial_candidates(g, source, portals)
    assert candidates, "source must see at least one portal"
    best_dist = min(dist for dist, _portal, _step in candidates)
    best = [(portal, step) for dist, portal, step in candidates if dist == best_dist]
    assert len(best) == 1, "source must have a unique nearest visible portal"

    pos = source
    direction = best[0][1]
    out = g.copy()
    seen_states: set[tuple[Coord, Coord]] = set()
    max_steps = h * w * 12

    for _ in range(max_steps):
        state = (pos, direction)
        assert state not in seen_states, "beam entered a portal loop"
        seen_states.add(state)

        next_pos = _add(pos, direction)
        if not _in_bounds(h, w, next_pos):
            break

        next2 = _add(next_pos, direction)
        portal = center_to_portal.get(next2)
        if (
            portal is not None
            and portal.marker == next_pos
            and portal.marker_dir == _neg(direction)
        ):
            paired = pair_lookup[next2]
            pos = paired.center
            direction = paired.marker_dir
            continue

        if int(g[next_pos]) == BG:
            out[next_pos] = YELLOW
        pos = next_pos
    else:
        raise AssertionError("beam simulation did not terminate")

    return out


@dataclass(frozen=True)
class GenPortal:
    center: Coord
    marker_dir: Coord


@dataclass(frozen=True)
class EntryCandidate:
    portal: GenPortal
    portal_cells: frozenset[Coord]
    segment: tuple[Coord, ...]


def _distance_to_edge(origin: Coord, direction: Coord, h: int, w: int) -> int:
    if direction == (-1, 0):
        return origin[0]
    if direction == (1, 0):
        return h - 1 - origin[0]
    if direction == (0, -1):
        return origin[1]
    return w - 1 - origin[1]


def _path_to_edge(
    origin: Coord,
    direction: Coord,
    allow_first_marker: bool,
    h: int,
    w: int,
    blocked: set[Coord],
) -> tuple[Coord, ...] | None:
    edge_dist = _distance_to_edge(origin, direction, h, w)
    min_dist = 2 if allow_first_marker else 1
    if edge_dist < min_dist:
        return None
    segment = tuple(_add(origin, _scale(direction, k)) for k in range(1, edge_dist + 1))
    check_cells = segment[1:] if allow_first_marker else segment
    if any(cell in blocked for cell in check_cells):
        return None
    return segment


def _entry_candidates(
    origin: Coord,
    direction: Coord,
    allow_first_marker: bool,
    h: int,
    w: int,
    occupied: set[Coord],
    reserved: set[Coord],
) -> list[EntryCandidate]:
    blocked = occupied | reserved
    min_dist = 4 if allow_first_marker else 3
    max_dist = _distance_to_edge(origin, direction, h, w)
    candidates: list[EntryCandidate] = []
    for dist in range(min_dist, max_dist + 1):
        center = _add(origin, _scale(direction, dist))
        portal = GenPortal(center=center, marker_dir=_neg(direction))
        portal_cells = _portal_cells(center, portal.marker_dir, h, w)
        if portal_cells is None:
            continue
        segment = tuple(_add(origin, _scale(direction, k)) for k in range(1, dist + 1))
        check_cells = segment[1:] if allow_first_marker else segment
        fresh = set(check_cells) | portal_cells
        if fresh.isdisjoint(blocked):
            candidates.append(
                EntryCandidate(
                    portal=portal,
                    portal_cells=frozenset(portal_cells),
                    segment=segment,
                )
            )
    random.shuffle(candidates)
    return candidates


def _all_exit_candidates(
    direction: Coord,
    remaining: int,
    h: int,
    w: int,
    occupied: set[Coord],
    reserved: set[Coord],
) -> list[GenPortal]:
    blocked = occupied | reserved
    out: list[GenPortal] = []
    for r in range(h):
        for c in range(w):
            portal = GenPortal(center=(r, c), marker_dir=direction)
            portal_cells = _portal_cells(portal.center, portal.marker_dir, h, w)
            if portal_cells is None or not portal_cells.isdisjoint(blocked):
                continue
            occ2 = occupied | portal_cells
            res2 = reserved | portal_cells
            if remaining == 0:
                final_path = _path_to_edge(portal.center, direction, True, h, w, occ2 | res2)
                if final_path is None:
                    continue
            else:
                if not _entry_candidates(portal.center, direction, True, h, w, occ2, res2):
                    continue
            out.append(portal)
    random.shuffle(out)
    return out


def _build_used_chain(
    origin: Coord,
    direction: Coord,
    allow_first_marker: bool,
    remaining: int,
    h: int,
    w: int,
    occupied: set[Coord],
    reserved: set[Coord],
) -> tuple[list[tuple[GenPortal, GenPortal]], set[Coord], set[Coord]] | None:
    if remaining == 0:
        final_path = _path_to_edge(origin, direction, allow_first_marker, h, w, occupied | reserved)
        if final_path is None:
            return None
        return ([], occupied, reserved | set(final_path))

    for entry in _entry_candidates(origin, direction, allow_first_marker, h, w, occupied, reserved):
        occupied1 = occupied | set(entry.portal_cells)
        reserved1 = reserved | set(entry.segment) | set(entry.portal_cells)

        dir_options = DIRS[:]
        random.shuffle(dir_options)
        for out_dir in dir_options:
            exits = _all_exit_candidates(out_dir, remaining - 1, h, w, occupied1, reserved1)
            for exit in exits[:80]:
                exit_cells = _portal_cells(exit.center, exit.marker_dir, h, w)
                assert exit_cells is not None
                occupied2 = occupied1 | exit_cells
                reserved2 = reserved1 | exit_cells
                suffix = _build_used_chain(
                    exit.center,
                    out_dir,
                    True,
                    remaining - 1,
                    h,
                    w,
                    occupied2,
                    reserved2,
                )
                if suffix is None:
                    continue
                pairs, occ_final, res_final = suffix
                return ([(entry.portal, exit)] + pairs, occ_final, res_final)
    return None


def _draw_portal(grid: np.ndarray, portal: GenPortal, color: int) -> None:
    for r, c in _body_cells(portal.center, portal.marker_dir):
        grid[r, c] = color
    mr, mc = portal.center[0] + portal.marker_dir[0], portal.center[1] + portal.marker_dir[1]
    grid[mr, mc] = MAROON


def _place_distractor_pair(
    h: int,
    w: int,
    occupied: set[Coord],
    reserved: set[Coord],
) -> tuple[tuple[GenPortal, GenPortal], set[Coord]] | None:
    occ = set(occupied)
    pair: list[GenPortal] = []
    for _ in range(2):
        candidates: list[GenPortal] = []
        blocked = occ | reserved
        for r in range(h):
            for c in range(w):
                for direction in DIRS:
                    portal = GenPortal(center=(r, c), marker_dir=direction)
                    cells = _portal_cells(portal.center, portal.marker_dir, h, w)
                    if cells is not None and cells.isdisjoint(blocked):
                        candidates.append(portal)
        if not candidates:
            return None
        portal = random.choice(candidates)
        cells = _portal_cells(portal.center, portal.marker_dir, h, w)
        assert cells is not None
        occ |= cells
        pair.append(portal)
    return ((pair[0], pair[1]), occ)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)

    used_count = random.choice([1, 2, 3])
    max_distractors = max(0, len(PORTAL_COLORS) - used_count)
    distractor_count = random.randint(0, min(2, max_distractors))

    source_positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(source_positions)

    chain_result = None
    for source in source_positions[:80]:
        dir_options = DIRS[:]
        random.shuffle(dir_options)
        for direction in dir_options:
            occupied = {source}
            reserved = {source}
            chain_result = _build_used_chain(source, direction, False, used_count, h, w, occupied, reserved)
            if chain_result is not None:
                break
        if chain_result is not None:
            break
    if chain_result is None:
        return None

    used_pairs, occupied, reserved = chain_result
    colors = PORTAL_COLORS[:]
    random.shuffle(colors)
    used_colors = colors[:used_count]
    extra_colors = colors[used_count : used_count + distractor_count]

    distractor_pairs: list[tuple[GenPortal, GenPortal]] = []
    for _ in extra_colors:
        placed = _place_distractor_pair(h, w, occupied, reserved)
        if placed is None:
            return None
        pair, occupied = placed
        distractor_pairs.append(pair)

    grid = np.full((h, w), BG, dtype=int)
    portal_cells_used: set[Coord] = set()
    all_pairs = used_pairs + distractor_pairs
    for pair in all_pairs:
        for portal in pair:
            cells = _portal_cells(portal.center, portal.marker_dir, h, w)
            assert cells is not None
            portal_cells_used |= cells
    source_cells = occupied - portal_cells_used
    assert len(source_cells) == 1
    source = next(iter(source_cells))
    grid[source] = YELLOW

    for color, pair in zip(used_colors, used_pairs):
        for portal in pair:
            _draw_portal(grid, portal, color)
    for color, pair in zip(extra_colors, distractor_pairs):
        for portal in pair:
            _draw_portal(grid, portal, color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == YELLOW)) <= 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-ray ARC input")
