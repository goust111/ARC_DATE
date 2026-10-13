from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np


BG = 8
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

LATTICE_MARGIN = 8
LATTICE_SPACING = 6
CONNECTOR_LEN = 2
GEN_RETRIES = 400

_RNG = random.Random()


@dataclass(frozen=True)
class Node:
    r0: int
    c0: int
    r1: int
    c1: int
    seed_color: int | None
    seed_count: int

    @property
    def outer_h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def outer_w(self) -> int:
        return self.c1 - self.c0 + 1


@dataclass(frozen=True)
class Slot:
    vr: int
    vc: int
    side: str


@dataclass(frozen=True)
class Placement:
    node: Node
    connector: tuple[tuple[int, int], ...]
    slot: Slot


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _border_cells(node: Node):
    for c in range(node.c0, node.c1 + 1):
        yield node.r0, c
        yield node.r1, c
    for r in range(node.r0 + 1, node.r1):
        yield r, node.c0
        yield r, node.c1


def _find_nodes(grid: np.ndarray) -> list[Node]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[Node] = []
    for outer_h in range(4, 7):
        for outer_w in range(4, 7):
            inner_h = outer_h - 2
            inner_w = outer_w - 2
            if not (2 <= inner_h <= 4 and 2 <= inner_w <= 4):
                continue
            for r0 in range(h - outer_h + 1):
                r1 = r0 + outer_h - 1
                for c0 in range(w - outer_w + 1):
                    c1 = c0 + outer_w - 1
                    if not np.all(g[r0, c0 : c1 + 1] == GREY):
                        continue
                    if not np.all(g[r1, c0 : c1 + 1] == GREY):
                        continue
                    if not np.all(g[r0 : r1 + 1, c0] == GREY):
                        continue
                    if not np.all(g[r0 : r1 + 1, c1] == GREY):
                        continue

                    interior = g[r0 + 1 : r1, c0 + 1 : c1]
                    if interior.size == 0 or np.any(interior == GREY):
                        continue

                    values = set(map(int, interior.flat))
                    non_bg = {v for v in values if v != BG}
                    if len(non_bg) > 1:
                        continue

                    seed_count = int(np.count_nonzero(interior != BG))
                    if seed_count > 3:
                        continue

                    seed_color: int | None = None
                    if seed_count:
                        seed_color = int(next(iter(non_bg)))
                        if seed_color not in SEED_COLORS:
                            continue

                    attached = False
                    for br, bc in _border_cells(Node(r0, c0, r1, c1, seed_color, seed_count)):
                        for nr, nc in _neighbors4(br, bc, h, w):
                            if r0 <= nr <= r1 and c0 <= nc <= c1:
                                continue
                            if int(g[nr, nc]) == GREY:
                                attached = True
                                break
                        if attached:
                            break
                    if not attached:
                        continue
                    candidates.append(Node(r0, c0, r1, c1, seed_color, seed_count))

    selected: list[Node] = []
    used = np.zeros_like(g, dtype=bool)
    for node in sorted(candidates, key=lambda n: (-(n.outer_h * n.outer_w), n.r0, n.c0)):
        if np.any(used[node.r0 : node.r1 + 1, node.c0 : node.c1 + 1]):
            continue
        used[node.r0 : node.r1 + 1, node.c0 : node.c1 + 1] = True
        selected.append(node)
    return selected


def _border_mask(shape: tuple[int, int], nodes: list[Node]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for node in nodes:
        for r, c in _border_cells(node):
            mask[r, c] = True
    return mask


def _source_starts(
    grid: np.ndarray, nodes: list[Node], border_mask: np.ndarray
) -> tuple[list[Node], list[set[tuple[int, int]]], list[dict[tuple[int, int], set[str]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pipe_mask = (g == GREY) & (~border_mask)

    seeded = [node for node in nodes if node.seed_count > 0 and node.seed_color is not None]
    starts: list[set[tuple[int, int]]] = []
    axes_by_source: list[dict[tuple[int, int], set[str]]] = []

    for node in seeded:
        entry_cells: set[tuple[int, int]] = set()
        entry_axes: dict[tuple[int, int], set[str]] = defaultdict(set)
        for br, bc in _border_cells(node):
            for nr, nc in _neighbors4(br, bc, h, w):
                if node.r0 <= nr <= node.r1 and node.c0 <= nc <= node.c1:
                    continue
                if not pipe_mask[nr, nc]:
                    continue
                entry_cells.add((nr, nc))
                if nr == br:
                    entry_axes[(nr, nc)].add("h")
                if nc == bc:
                    entry_axes[(nr, nc)].add("v")
        starts.append(entry_cells)
        axes_by_source.append(entry_axes)
    return seeded, starts, axes_by_source


def _propagate(
    pipe_mask: np.ndarray, starts: list[set[tuple[int, int]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = pipe_mask.shape
    owner = -np.ones((h, w), dtype=int)
    dist = -np.ones((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    initial: dict[tuple[int, int], set[int]] = defaultdict(set)
    for sid, cells in enumerate(starts):
        for cell in cells:
            if pipe_mask[cell]:
                initial[cell].add(sid)

    frontier: list[tuple[int, int, int]] = []
    for (r, c), sids in initial.items():
        if len(sids) == 1:
            sid = next(iter(sids))
            owner[r, c] = sid
            dist[r, c] = 0
            frontier.append((r, c, sid))
        else:
            blocked[r, c] = True

    time = 0
    while frontier:
        proposals: dict[tuple[int, int], set[int]] = defaultdict(set)
        for r, c, sid in frontier:
            for nr, nc in _neighbors4(r, c, h, w):
                if not pipe_mask[nr, nc]:
                    continue
                if owner[nr, nc] != -1 or blocked[nr, nc]:
                    continue
                proposals[(nr, nc)].add(sid)

        next_frontier: list[tuple[int, int, int]] = []
        for (r, c), sids in proposals.items():
            if owner[r, c] != -1 or blocked[r, c]:
                continue
            if len(sids) == 1:
                sid = next(iter(sids))
                owner[r, c] = sid
                dist[r, c] = time + 1
                next_frontier.append((r, c, sid))
            else:
                blocked[r, c] = True
        frontier = next_frontier
        time += 1

    return owner, dist, blocked


def _axes_for_cell(
    r: int,
    c: int,
    sid: int,
    pipe_mask: np.ndarray,
    owner: np.ndarray,
    start_axes: dict[tuple[int, int], set[str]],
) -> set[str]:
    h, w = pipe_mask.shape
    axes = set(start_axes.get((r, c), set()))
    for nr, nc in _neighbors4(r, c, h, w):
        if not pipe_mask[nr, nc]:
            continue
        if owner[nr, nc] == sid:
            axes.add("h" if nr == r else "v")
    if axes:
        return axes
    for nr, nc in _neighbors4(r, c, h, w):
        if pipe_mask[nr, nc]:
            axes.add("h" if nr == r else "v")
    return axes


def _paint_widened(
    grid: np.ndarray,
    nodes: list[Node],
    seeded: list[Node],
    owner: np.ndarray,
    dist: np.ndarray,
    blocked: np.ndarray,
    start_axes: list[dict[tuple[int, int], set[str]]],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    border = _border_mask(g.shape, nodes)
    pipe_mask = (g == GREY) & (~border)

    best_time = np.full((h, w), 10**9, dtype=int)
    best_color = -np.ones((h, w), dtype=int)
    tied = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            sid = int(owner[r, c])
            if sid < 0:
                continue
            color = int(seeded[sid].seed_color)
            radius = int(seeded[sid].seed_count) - 1
            axes = _axes_for_cell(r, c, sid, pipe_mask, owner, start_axes[sid])
            targets = {(r, c)}
            for axis in axes:
                if axis == "h":
                    for dr in range(-radius, radius + 1):
                        targets.add((r + dr, c))
                else:
                    for dc in range(-radius, radius + 1):
                        targets.add((r, c + dc))

            for tr, tc in targets:
                if not (0 <= tr < h and 0 <= tc < w):
                    continue
                if border[tr, tc]:
                    continue
                if int(g[tr, tc]) != BG and int(owner[tr, tc]) != sid:
                    continue

                arrival = int(dist[r, c])
                if arrival < best_time[tr, tc]:
                    best_time[tr, tc] = arrival
                    best_color[tr, tc] = color
                    tied[tr, tc] = False
                elif arrival == best_time[tr, tc] and best_color[tr, tc] != color:
                    tied[tr, tc] = True

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                out[r, c] = GREY
                continue
            if best_color[r, c] != -1 and not tied[r, c]:
                out[r, c] = best_color[r, c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread each node's colored seeds through the grey pipe network.

    - Grey framed rectangles are nodes; the count of same-color seed pixels
      inside each node determines the spread width.
    - Diffusion runs simultaneously on the 1-pixel-wide pipe centerline.
      First arrival claims a pipe cell; same-time collisions leave that pipe
      cell grey as a divider.
    - After the centerline competition is resolved, each claimed pipe cell is
      widened orthogonally by radius 0/1/2 for seed counts 1/2/3.
    - Node borders remain grey.
    """
    g = np.asarray(grid, dtype=int)
    nodes = _find_nodes(g)
    border = _border_mask(g.shape, nodes)
    pipe_mask = (g == GREY) & (~border)

    seeded, starts, start_axes = _source_starts(g, nodes, border)
    if not seeded:
        return g.copy()

    owner, dist, blocked = _propagate(pipe_mask, starts)
    return _paint_widened(g, nodes, seeded, owner, dist, blocked, start_axes)


def _vertex_coords(rows: int, cols: int) -> dict[tuple[int, int], tuple[int, int]]:
    coords: dict[tuple[int, int], tuple[int, int]] = {}
    for vr in range(rows):
        for vc in range(cols):
            coords[(vr, vc)] = (
                LATTICE_MARGIN + vr * LATTICE_SPACING,
                LATTICE_MARGIN + vc * LATTICE_SPACING,
            )
    return coords


def _abstract_neighbors(v: tuple[int, int], rows: int, cols: int):
    vr, vc = v
    for dr, dc in DIR4:
        nr, nc = vr + dr, vc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _all_abstract_edges(rows: int, cols: int) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges = set()
    for vr in range(rows):
        for vc in range(cols):
            if vr + 1 < rows:
                edges.add(tuple(sorted(((vr, vc), (vr + 1, vc)))))
            if vc + 1 < cols:
                edges.add(tuple(sorted(((vr, vc), (vr, vc + 1)))))
    return edges


def _random_graph(rows: int, cols: int, rng: random.Random):
    vertices = [(vr, vc) for vr in range(rows) for vc in range(cols)]
    start = rng.choice(vertices)
    visited = {start}
    stack = [start]
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while stack:
        cur = stack[-1]
        options = [nxt for nxt in _abstract_neighbors(cur, rows, cols) if nxt not in visited]
        if not options:
            stack.pop()
            continue
        nxt = rng.choice(options)
        visited.add(nxt)
        stack.append(nxt)
        edges.add(tuple(sorted((cur, nxt))))

    extras = list(_all_abstract_edges(rows, cols) - edges)
    rng.shuffle(extras)
    extra_count = rng.randint(1, min(4, len(extras)))
    for edge in extras[:extra_count]:
        edges.add(edge)

    degree = defaultdict(int)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    if max(degree.values(), default=0) < 3:
        interior = [(vr, vc) for vr in range(1, rows - 1) for vc in range(1, cols - 1)]
        rng.shuffle(interior)
        for hub in interior:
            missing = [
                tuple(sorted((hub, nbr)))
                for nbr in _abstract_neighbors(hub, rows, cols)
                if tuple(sorted((hub, nbr))) not in edges
            ]
            rng.shuffle(missing)
            for edge in missing:
                edges.add(edge)
                a, b = edge
                degree[a] += 1
                degree[b] += 1
                if degree[hub] >= 3:
                    break
            if degree[hub] >= 3:
                break
    return edges


def _render_graph(grid: np.ndarray, edges, coords: dict[tuple[int, int], tuple[int, int]]) -> None:
    for a, b in edges:
        r1, c1 = coords[a]
        r2, c2 = coords[b]
        if r1 == r2:
            c0, c1b = sorted((c1, c2))
            grid[r1, c0 : c1b + 1] = GREY
        else:
            r0, r1b = sorted((r1, r2))
            grid[r0 : r1b + 1, c1] = GREY


def _boundary_slots(rows: int, cols: int) -> list[Slot]:
    slots: list[Slot] = []
    for vc in range(cols):
        slots.append(Slot(0, vc, "up"))
        slots.append(Slot(rows - 1, vc, "down"))
    for vr in range(rows):
        slots.append(Slot(vr, 0, "left"))
        slots.append(Slot(vr, cols - 1, "right"))
    dedup = {}
    for slot in slots:
        dedup[(slot.vr, slot.vc, slot.side)] = slot
    return list(dedup.values())


def _layout_from_slot(slot: Slot, pr: int, pc: int, outer_h: int, outer_w: int) -> tuple[Node, tuple[tuple[int, int], ...]]:
    if slot.side == "up":
        r0 = pr - CONNECTOR_LEN - outer_h + 1
        c0 = pc - outer_w // 2
        connector = tuple((r, pc) for r in range(r0 + outer_h, pr + 1))
    elif slot.side == "down":
        r0 = pr + CONNECTOR_LEN
        c0 = pc - outer_w // 2
        connector = tuple((r, pc) for r in range(pr, r0))
    elif slot.side == "left":
        r0 = pr - outer_h // 2
        c0 = pc - CONNECTOR_LEN - outer_w + 1
        connector = tuple((pr, c) for c in range(c0 + outer_w, pc + 1))
    else:
        r0 = pr - outer_h // 2
        c0 = pc + CONNECTOR_LEN
        connector = tuple((pr, c) for c in range(pc, c0))

    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    return Node(r0, c0, r1, c1, None, 0), connector


def _reserve_ok(mask: np.ndarray, node: Node, connector: tuple[tuple[int, int], ...]) -> bool:
    h, w = mask.shape
    if not (0 <= node.r0 < h and 0 <= node.c0 < w and 0 <= node.r1 < h and 0 <= node.c1 < w):
        return False
    pad_r0 = max(0, node.r0 - 1)
    pad_c0 = max(0, node.c0 - 1)
    pad_r1 = min(h - 1, node.r1 + 1)
    pad_c1 = min(w - 1, node.c1 + 1)
    if np.any(mask[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1]):
        return False
    for r, c in connector:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if mask[r, c]:
            return False
    return True


def _commit_node(
    grid: np.ndarray,
    reserved: np.ndarray,
    node: Node,
    connector: tuple[tuple[int, int], ...],
    color: int | None,
    count: int,
    rng: random.Random,
) -> Placement:
    actual = Node(node.r0, node.c0, node.r1, node.c1, color, count)
    grid[node.r0 : node.r1 + 1, node.c0 : node.c1 + 1] = BG
    grid[node.r0, node.c0 : node.c1 + 1] = GREY
    grid[node.r1, node.c0 : node.c1 + 1] = GREY
    grid[node.r0 : node.r1 + 1, node.c0] = GREY
    grid[node.r0 : node.r1 + 1, node.c1] = GREY
    for r, c in connector:
        grid[r, c] = GREY

    if color is not None and count > 0:
        inner = [(r, c) for r in range(node.r0 + 1, node.r1) for c in range(node.c0 + 1, node.c1)]
        rng.shuffle(inner)
        for r, c in inner[:count]:
            grid[r, c] = color

    pad_r0 = max(0, node.r0 - 1)
    pad_c0 = max(0, node.c0 - 1)
    pad_r1 = min(grid.shape[0] - 1, node.r1 + 1)
    pad_c1 = min(grid.shape[1] - 1, node.c1 + 1)
    reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
    for r, c in connector:
        reserved[r, c] = True
    return Placement(actual, connector, Slot(-1, -1, ""))


def _place_nodes(grid: np.ndarray, rows: int, cols: int, coords, rng: random.Random) -> list[Placement] | None:
    reserved = np.zeros_like(grid, dtype=bool)
    slots = _boundary_slots(rows, cols)
    rng.shuffle(slots)

    seeded_count = rng.randint(2, min(5, len(SEED_COLORS)))
    colors = rng.sample(list(SEED_COLORS), seeded_count)
    seed_counts = [rng.randint(1, 3) for _ in range(seeded_count)]
    if seeded_count >= 2 and len(set(seed_counts)) == 1:
        idx = rng.randrange(seeded_count)
        seed_counts[idx] = 1 + (seed_counts[idx] % 3)

    empty_count = 0
    if len(slots) - seeded_count >= 1 and rng.random() < 0.45:
        empty_count = rng.randint(1, min(2, len(slots) - seeded_count))

    placements: list[Placement] = []
    requested = [(colors[i], seed_counts[i]) for i in range(seeded_count)] + [(None, 0)] * empty_count
    for color, count in requested:
        placed = False
        rng.shuffle(slots)
        for slot in slots:
            pr, pc = coords[(slot.vr, slot.vc)]
            for _ in range(20):
                inner_h = rng.randint(2, 4)
                inner_w = rng.randint(2, 4)
                outer_h = inner_h + 2
                outer_w = inner_w + 2
                node, connector = _layout_from_slot(slot, pr, pc, outer_h, outer_w)
                if not _reserve_ok(reserved, node, connector):
                    continue
                placement = _commit_node(grid, reserved, node, connector, color, count, rng)
                placements.append(Placement(placement.node, connector, slot))
                slots.remove(slot)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return placements


def _validate_generated(grid: np.ndarray, placements: list[Placement]) -> bool:
    parsed = _find_nodes(grid)
    expected = sorted((p.node.r0, p.node.c0, p.node.r1, p.node.c1, p.node.seed_color, p.node.seed_count) for p in placements)
    got = sorted((n.r0, n.c0, n.r1, n.c1, n.seed_color, n.seed_count) for n in parsed)
    if expected != got:
        return False

    border = _border_mask(grid.shape, parsed)
    pipe_mask = (grid == GREY) & (~border)
    seeded, starts, _ = _source_starts(grid, parsed, border)
    if len(seeded) < 2:
        return False
    if len({node.seed_count for node in seeded}) < min(2, len(seeded)):
        return False
    owner, _dist, blocked = _propagate(pipe_mask, starts)
    if not np.any(owner >= 0):
        return False
    if not np.any(blocked):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    for node in parsed:
        if not np.all(out[node.r0, node.c0 : node.c1 + 1] == GREY):
            return False
        if not np.all(out[node.r1, node.c0 : node.c1 + 1] == GREY):
            return False
        if not np.all(out[node.r0 : node.r1 + 1, node.c0] == GREY):
            return False
        if not np.all(out[node.r0 : node.r1 + 1, node.c1] == GREY):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(3, 4)
    cols = rng.randint(3, 4)
    h = 2 * LATTICE_MARGIN + LATTICE_SPACING * (rows - 1) + 1
    w = 2 * LATTICE_MARGIN + LATTICE_SPACING * (cols - 1) + 1

    grid = np.full((h, w), BG, dtype=int)
    coords = _vertex_coords(rows, cols)
    edges = _random_graph(rows, cols, rng)
    _render_graph(grid, edges, coords)

    placements = _place_nodes(grid, rows, cols, coords, rng)
    if placements is None:
        return None
    if not _validate_generated(grid, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
