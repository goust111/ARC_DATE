from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5

DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
AROUND8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_source_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    del shape
    return r == 0 or c == 0


def _is_exit_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == h - 1 or c == w - 1


def _green_components(g: np.ndarray) -> list[set[tuple[int, int]]]:
    green = {tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)}
    seen: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []

    for start in sorted(green):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: set[tuple[int, int]] = set()
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for dr, dc in DIAGONALS:
                nb = (r + dr, c + dc)
                if nb in green and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(comp)
    return components


def _ordered_path(comp: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    assert len(comp) >= 2, "each ray trail must contain at least two green cells"

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in comp:
        nbrs = []
        for dr, dc in DIAGONALS:
            nb = (r + dr, c + dc)
            if nb in comp:
                nbrs.append(nb)
        adj[(r, c)] = sorted(nbrs)

    endpoints = [p for p, nbrs in adj.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, "each ray trail must be a simple open path"
    assert all(1 <= len(nbrs) <= 2 for nbrs in adj.values()), "ray trails cannot branch or self-touch diagonally"

    sources = [p for p in endpoints if _is_source_edge(shape, *p)]
    exits = [p for p in endpoints if _is_exit_edge(shape, *p) and not _is_source_edge(shape, *p)]
    assert len(sources) == 1, "each trail needs one visible source endpoint on the top or left edge"
    assert len(exits) == 1, "each trail needs one visible exit endpoint on the bottom or right edge"

    start = sources[0]
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nexts = [p for p in adj[cur] if p != prev]
        if not nexts:
            break
        assert len(nexts) == 1, "ray trail ordering must be unique"
        nxt = nexts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(comp), "ray trail must not contain disconnected pieces"
    assert path[-1] == exits[0], "ordered trail must run from source edge to exit edge"
    return path


def _trace_paths(g: np.ndarray) -> list[list[tuple[int, int]]]:
    paths = [_ordered_path(comp, g.shape) for comp in _green_components(g)]
    assert paths, "expected at least one green ray trail"
    return paths


def _contacts_from_paths(
    paths: list[list[tuple[int, int]]], shape: tuple[int, int], g: np.ndarray
) -> list[tuple[str, tuple[int, int]]]:
    contacts: list[tuple[str, tuple[int, int]]] = []
    for path in paths:
        for i in range(1, len(path) - 1):
            pr, pc = path[i - 1]
            r, c = path[i]
            nr, nc = path[i + 1]
            dr1, dc1 = r - pr, c - pc
            dr2, dc2 = nr - r, nc - c
            assert (dr1, dc1) in DIAGONALS and (dr2, dc2) in DIAGONALS

            if (dr1, dc1) == (dr2, dc2):
                continue
            if dr2 == -dr1 and dc2 == dc1:
                orient = "h"
            elif dr2 == dr1 and dc2 == -dc1:
                orient = "v"
            else:
                raise AssertionError("a bend must flip exactly one diagonal direction component")

            contact = (r + dr1, c + dc1)
            assert _in_bounds(shape, *contact), "mirror contact cell must stay inside the grid"
            assert int(g[contact]) == BLACK, "mirror contact cell must be erased, not part of a trail"
            contacts.append((orient, contact))
    return contacts


def _contact_groups(
    contacts: list[tuple[str, tuple[int, int]]]
) -> dict[tuple[str, int], set[int]]:
    groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    for orient, (r, c) in contacts:
        if orient == "h":
            groups[(orient, r)].add(c)
        else:
            groups[(orient, c)].add(r)
    return groups


def _draw_reconstructed_mirrors(g: np.ndarray, contacts: list[tuple[str, tuple[int, int]]]) -> np.ndarray:
    out = g.copy()
    groups = _contact_groups(contacts)
    for (orient, fixed), values in groups.items():
        assert len(values) >= 2, "each reconstructed mirror line needs at least two contact cells"
        lo, hi = min(values), max(values)
        for v in range(lo, hi + 1):
            r, c = (fixed, v) if orient == "h" else (v, fixed)
            assert int(out[r, c]) in (BLACK, GREY), "reconstructed mirror segment cannot overwrite a ray trail"
            out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct erased gray mirror segments from bends in green diagonal ray trails.

    Input structure:
    - A black grid contains green one-cell-wide diagonal ray trails and no visible mirror pixels.
    - Each trail is a simple diagonal path with one endpoint on the top or left source edge and the other on the bottom or right exit edge.
    - Some trails may pass straight through without bending; bent trails flip one diagonal direction component at each turn.

    Rule:
    - Preserve every green trail pixel.
    - For every bend, use the source-to-exit trail order to step one cell farther in the incoming direction; that erased cell is the mirror contact.
    - A row-direction flip reconstructs a horizontal mirror line, and a column-direction flip reconstructs a vertical mirror line.
    - Merge all contacts on the same reconstructed row or column by drawing the full gray segment between their extremes.

    Key insight:
    - The mirror is invisible; the bend gives its orientation, while multiple collinear contact cells determine the segment span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"
    assert set(map(int, np.unique(g))).issubset({BLACK, GREEN}), "input should contain only black background and green trails"

    paths = _trace_paths(g)
    contacts = _contacts_from_paths(paths, g.shape, g)
    if not contacts:
        return g.copy()
    return _draw_reconstructed_mirrors(g, contacts)


# --------------- generator ---------------

GEN_ATTEMPTS = 800
SUBSET_ATTEMPTS = 260
GRID_MIN = 14
GRID_MAX = 26
MAX_RAYS = 7


def _gen_place_mirrors(h: int, w: int) -> dict[tuple[int, int], str] | None:
    count = random.randint(2, 4)
    orientations = ["h", "v"]
    orientations.extend(random.choice(["h", "v"]) for _ in range(count - 2))
    random.shuffle(orientations)

    mirrors: dict[tuple[int, int], str] = {}
    used_h_rows: set[int] = set()
    used_v_cols: set[int] = set()

    for orient in orientations:
        placed = False
        for _ in range(100):
            if orient == "h":
                length = random.randint(4, min(10, w - 5))
                r = random.randint(3, h - 4)
                c0 = random.randint(2, w - length - 2)
                cells = [(r, c) for c in range(c0, c0 + length)]
                if r in used_h_rows:
                    continue
            else:
                length = random.randint(4, min(10, h - 5))
                c = random.randint(3, w - 4)
                r0 = random.randint(2, h - length - 2)
                cells = [(r, c) for r in range(r0, r0 + length)]
                if c in used_v_cols:
                    continue

            if any(cell in mirrors for cell in cells):
                continue
            if any((rr + dr, cc + dc) in mirrors for rr, cc in cells for dr, dc in AROUND8):
                continue

            for cell in cells:
                mirrors[cell] = orient
            if orient == "h":
                used_h_rows.add(cells[0][0])
            else:
                used_v_cols.add(cells[0][1])
            placed = True
            break
        if not placed:
            return None
    return mirrors


def _gen_simulate_ray(
    shape: tuple[int, int],
    mirrors: dict[tuple[int, int], str],
    start: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[str, tuple[int, int]]]] | None:
    h, w = shape
    r, c = start
    dr, dc = 1, 1
    path = [(r, c)]
    contacts: list[tuple[str, tuple[int, int]]] = []
    seen_states = {(r, c, dr, dc)}
    path_cells = {(r, c)}

    for _ in range(h * w * 4):
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            break

        mirror_orient = mirrors.get((nr, nc))
        if mirror_orient is not None:
            contacts.append((mirror_orient, (nr, nc)))
            if mirror_orient == "h":
                dr = -dr
            else:
                dc = -dc
            state = (r, c, dr, dc)
            if state in seen_states:
                return None
            seen_states.add(state)
            continue

        r, c = nr, nc
        if (r, c) in path_cells:
            return None
        path.append((r, c))
        path_cells.add((r, c))
        state = (r, c, dr, dc)
        if state in seen_states:
            return None
        seen_states.add(state)
    else:
        return None

    end = path[-1]
    if not _is_exit_edge(shape, *end) or _is_source_edge(shape, *end):
        return None
    if len(path) < 5:
        return None
    return path, contacts


def _gen_ray_candidates(
    shape: tuple[int, int], mirrors: dict[tuple[int, int], str]
) -> list[tuple[list[tuple[int, int]], list[tuple[str, tuple[int, int]]]]]:
    h, w = shape
    starts = [(0, c) for c in range(1, w - 1)]
    starts.extend((r, 0) for r in range(1, h - 1))
    random.shuffle(starts)

    candidates = []
    for start in starts:
        ray = _gen_simulate_ray(shape, mirrors, start)
        if ray is not None:
            candidates.append(ray)
    return candidates


def _gen_paths_do_not_touch(paths: list[list[tuple[int, int]]]) -> bool:
    occupied: set[tuple[int, int]] = set()
    for path in paths:
        cells = set(path)
        if len(cells) != len(path):
            return False
        for r, c in cells:
            if (r, c) in occupied:
                return False
            for dr, dc in DIAGONALS:
                if (r + dr, c + dc) in occupied:
                    return False
        occupied.update(cells)
    return True


def _gen_draw_paths(shape: tuple[int, int], paths: list[list[tuple[int, int]]]) -> np.ndarray:
    g = np.full(shape, BLACK, dtype=int)
    for path in paths:
        for r, c in path:
            g[r, c] = GREEN
    return g


def _gen_semantic_ok(g: np.ndarray) -> bool:
    try:
        paths = _trace_paths(g)
        contacts = _contacts_from_paths(paths, g.shape, g)
        groups = _contact_groups(contacts)
        solved = _draw_reconstructed_mirrors(g, contacts)
    except AssertionError:
        return False

    if not contacts or not groups:
        return False
    if {orient for orient, _fixed in groups} != {"h", "v"}:
        return False
    if any(len(values) < 2 for values in groups.values()):
        return False
    if not any(len(_contacts_from_paths([path], g.shape, g)) >= 2 for path in paths):
        return False
    if int(np.count_nonzero(solved == GREY)) < 5:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    mirrors = _gen_place_mirrors(h, w)
    if mirrors is None:
        return None

    candidates = _gen_ray_candidates((h, w), mirrors)
    bent = [ray for ray in candidates if ray[1]]
    straight = [ray for ray in candidates if not ray[1]]
    if len(bent) < 3:
        return None

    for _ in range(SUBSET_ATTEMPTS):
        random.shuffle(bent)
        chosen = bent[: random.randint(3, min(MAX_RAYS, len(bent)))]
        if straight and random.random() < 0.45:
            chosen = chosen + [random.choice(straight)]
        random.shuffle(chosen)

        paths = [path for path, _contacts in chosen]
        if not _gen_paths_do_not_touch(paths):
            continue
        g = _gen_draw_paths((h, w), paths)
        if not _gen_semantic_ok(g):
            continue

        try:
            out = main(g)
        except AssertionError:
            continue
        if out.shape == g.shape and max(out.shape) <= 30:
            return g
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid hidden-mirror ray puzzle input")
