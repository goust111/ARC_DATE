import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
MAROON = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)
GENERATE_ATTEMPTS = 400
PLACE_ATTEMPTS = 200


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _gray_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GRAY:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != GRAY:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append(comp)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored source through its adjacent gray wire network and recolor only
    the gray degree-1 endpoints. Conflicting source colors turn the endpoint maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _gray_components(g):
        comp_set = set(comp)
        source_colors: set[int] = set()
        endpoints: list[tuple[int, int]] = []
        for r, c in comp:
            gray_degree = 0
            for nr, nc in neighbors4(r, c):
                if not in_bounds(g.shape, nr, nc):
                    continue
                value = int(g[nr, nc])
                if value == GRAY:
                    gray_degree += 1
                elif value != BLACK:
                    source_colors.add(value)
            if gray_degree == 1:
                endpoints.append((r, c))

        assert source_colors, "every wire network must touch at least one signal source"
        assert endpoints, "every wire network must have at least one receiver endpoint"

        endpoint_color = next(iter(source_colors)) if len(source_colors) == 1 else MAROON
        for r, c in endpoints:
            out[r, c] = endpoint_color

    return out


def _expand_tree(
    nodes: set[tuple[int, int]], edges: set[frozenset[tuple[int, int]]]
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    max_r = max(r for r, _ in nodes)
    max_c = max(c for _, c in nodes)
    patch = np.full((max_r * 2 + 1, max_c * 2 + 1), BLACK, dtype=int)
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in nodes:
        rr, cc = 2 * r, 2 * c
        mapping[(r, c)] = (rr, cc)
        patch[rr, cc] = GRAY
    for edge in edges:
        (r1, c1), (r2, c2) = tuple(edge)
        rr1, cc1 = mapping[(r1, c1)]
        rr2, cc2 = mapping[(r2, c2)]
        patch[(rr1 + rr2) // 2, (cc1 + cc2) // 2] = GRAY
    return patch, mapping


def _wire_degrees(patch: np.ndarray) -> dict[tuple[int, int], int]:
    coords = np.argwhere(patch == GRAY)
    degrees: dict[tuple[int, int], int] = {}
    for r0, c0 in coords:
        r = int(r0)
        c = int(c0)
        degrees[(r, c)] = sum(
            1
            for nr, nc in neighbors4(r, c)
            if in_bounds(patch.shape, nr, nc) and int(patch[nr, nc]) == GRAY
        )
    return degrees


def _attach_sources(
    rng: random.Random, patch: np.ndarray, n_sources: int
) -> np.ndarray | None:
    result = patch.copy()
    degrees = _wire_degrees(result)
    slots: list[tuple[int, int, int, int]] = []
    for (r, c), degree in degrees.items():
        if degree < 2:
            continue
        for nr, nc in neighbors4(r, c):
            if in_bounds(result.shape, nr, nc) and int(result[nr, nc]) == BLACK:
                slots.append((r, c, nr, nc))
    if len(slots) < n_sources:
        return None

    rng.shuffle(slots)
    chosen: list[tuple[int, int]] = []
    used_targets: set[tuple[int, int]] = set()
    used_hosts: set[tuple[int, int]] = set()
    for hr, hc, sr, sc in slots:
        if (sr, sc) in used_targets or (hr, hc) in used_hosts:
            continue
        chosen.append((sr, sc))
        used_targets.add((sr, sc))
        used_hosts.add((hr, hc))
        if len(chosen) == n_sources:
            break
    if len(chosen) < n_sources:
        return None

    colors = list(SOURCE_COLORS)
    rng.shuffle(colors)
    for idx, (r, c) in enumerate(chosen):
        result[r, c] = colors[idx]
    return result


def _random_path(
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[frozenset[tuple[int, int]]]]:
    h = rng.randint(4, 7)
    w = rng.randint(4, 7)
    length = rng.randint(4, 7)
    for _ in range(200):
        cell = (rng.randrange(h), rng.randrange(w))
        path = [cell]
        seen = {cell}
        while len(path) < length:
            r, c = path[-1]
            choices = [
                (nr, nc)
                for nr, nc in neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen
            ]
            if not choices:
                break
            nxt = rng.choice(choices)
            path.append(nxt)
            seen.add(nxt)
        if len(path) == length:
            edges = {
                frozenset((path[i], path[i + 1]))
                for i in range(len(path) - 1)
            }
            return set(path), edges
    raise RuntimeError("failed to build path")

def _random_branch_tree(
    rng: random.Random, min_leaves: int
) -> tuple[set[tuple[int, int]], set[frozenset[tuple[int, int]]]]:
    for _ in range(300):
        h = rng.randint(5, 8)
        w = rng.randint(5, 8)
        target = rng.randint(6, 11)
        start = (rng.randrange(h), rng.randrange(w))
        nodes = {start}
        edges: set[frozenset[tuple[int, int]]] = set()
        for _ in range(target - 1):
            frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for r, c in nodes:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in nodes:
                        frontier.append(((r, c), (nr, nc)))
            if not frontier:
                break
            parent, child = rng.choice(frontier)
            nodes.add(child)
            edges.add(frozenset((parent, child)))

        degrees = {node: 0 for node in nodes}
        for edge in edges:
            a, b = tuple(edge)
            degrees[a] += 1
            degrees[b] += 1
        leaf_count = sum(1 for degree in degrees.values() if degree == 1)
        if any(degree >= 3 for degree in degrees.values()) and leaf_count >= min_leaves:
            return nodes, edges
    raise RuntimeError("failed to build branch tree")


def _build_component(rng: random.Random, kind: str) -> np.ndarray:
    for _ in range(200):
        if kind == "linear":
            nodes, edges = _random_path(rng)
            patch, _ = _expand_tree(nodes, edges)
            patch = _attach_sources(rng, patch, 1)
        elif kind == "branch":
            nodes, edges = _random_branch_tree(rng, min_leaves=3)
            patch, _ = _expand_tree(nodes, edges)
            patch = _attach_sources(rng, patch, 1)
        else:
            nodes, edges = _random_branch_tree(rng, min_leaves=3)
            patch, _ = _expand_tree(nodes, edges)
            patch = _attach_sources(rng, patch, 2)
        if patch is not None:
            return patch
    raise RuntimeError(f"failed to build component: {kind}")


def _can_place(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    cells = np.argwhere(patch != BLACK)
    for r0, c0 in cells:
        r = top + int(r0)
        c = left + int(c0)
        if canvas[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)].any():
            return False
    return True


def _stamp(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    cells = np.argwhere(patch != BLACK)
    for r0, c0 in cells:
        canvas[top + int(r0), left + int(c0)] = int(patch[int(r0), int(c0)])


def _validate_input(grid: np.ndarray) -> None:
    gray_components = _gray_components(grid)
    assert gray_components
    for comp in gray_components:
        comp_set = set(comp)
        endpoints = 0
        source_cells: set[tuple[int, int]] = set()
        for r, c in comp:
            gray_degree = 0
            for nr, nc in neighbors4(r, c):
                if not in_bounds(grid.shape, nr, nc):
                    continue
                value = int(grid[nr, nc])
                if value == GRAY:
                    gray_degree += 1
                elif value != BLACK:
                    source_cells.add((nr, nc))
            if gray_degree == 1:
                endpoints += 1
        assert endpoints >= 1
        assert source_cells

    for r0, c0 in np.argwhere((grid != BLACK) & (grid != GRAY)):
        r = int(r0)
        c = int(c0)
        touched = 0
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GRAY:
                touched += 1
        assert touched >= 1


def generate_input() -> np.ndarray:
    rng = random.Random()
    kinds = ("linear", "branch", "conflict")
    for _ in range(GENERATE_ATTEMPTS):
        h = rng.randint(16, 28)
        w = rng.randint(16, 28)
        grid = np.full((h, w), BLACK, dtype=int)
        n_components = rng.randint(1, 3)

        component_kinds = [rng.choice(kinds) for _ in range(n_components)]
        if n_components >= 2 and all(kind == "linear" for kind in component_kinds):
            component_kinds[rng.randrange(n_components)] = rng.choice(("branch", "conflict"))

        patches = [_build_component(rng, kind) for kind in component_kinds]
        order = list(range(len(patches)))
        rng.shuffle(order)

        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            placed = False
            for _ in range(PLACE_ATTEMPTS):
                top = rng.randint(0, h - ph)
                left = rng.randint(0, w - pw)
                if _can_place(grid, patch, top, left):
                    _stamp(grid, patch, top, left)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        _validate_input(grid)
        return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_ATTEMPTS} attempts")
