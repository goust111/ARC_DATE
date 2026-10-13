from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import product

import numpy as np


BG = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 600


Coord = tuple[int, int]


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _line_cells(a: Coord, b: Coord) -> list[Coord]:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        step = _sign(c2 - c1)
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = _sign(r2 - r1)
        return [(r, c1) for r in range(r1, r2 + step, step)]
    raise ValueError("line must be axis-aligned")


def _build_route(p: Coord, q: Coord, order: str) -> dict:
    if p[0] == q[0] or p[1] == q[1]:
        raise ValueError("route endpoints must differ in both row and column")
    if order == "HV":
        bend = (p[0], q[1])
    elif order == "VH":
        bend = (q[0], p[1])
    else:
        raise ValueError("order must be HV or VH")

    first = _line_cells(p, bend)
    second = _line_cells(bend, q)
    dirs: dict[Coord, set[str]] = defaultdict(set)
    for cell in first:
        dirs[cell].add("H" if p[0] == bend[0] else "V")
    for cell in second:
        dirs[cell].add("H" if bend[0] == q[0] else "V")

    cells = set(first)
    cells.update(second)
    h_cells = {cell for cell, ds in dirs.items() if "H" in ds}
    v_cells = {cell for cell, ds in dirs.items() if "V" in ds}

    if order == "HV":
        p_into = (0, _sign(q[1] - p[1]))
        q_into = (-_sign(q[0] - p[0]), 0)
    else:
        p_into = (_sign(q[0] - p[0]), 0)
        q_into = (0, -_sign(q[1] - p[1]))

    return {
        "p": p,
        "q": q,
        "order": order,
        "bend": bend,
        "cells": cells,
        "dirs": {cell: frozenset(values) for cell, values in dirs.items()},
        "h_cells": h_cells,
        "v_cells": v_cells,
        "p_into": p_into,
        "q_into": q_into,
    }


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(grid == color):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[Coord] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _segment_info(comp: list[Coord]) -> dict:
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        endpoints = [(row, cols[0]), (row, cols[-1])]
        orient = "H"
    elif len(cols) == 1:
        col = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1))
        endpoints = [(rows[0], col), (rows[-1], col)]
        orient = "V"
    else:
        raise AssertionError("component is not a straight segment")
    assert 2 <= len(comp) <= 4
    return {
        "cells": set(comp),
        "endpoints": endpoints,
        "orient": orient,
        "length": len(comp),
    }


def _orthogonal_cross(dirs_a: frozenset[str], dirs_b: frozenset[str]) -> bool:
    return ("H" in dirs_a and "V" in dirs_b) or ("V" in dirs_a and "H" in dirs_b)


def _enumerate_candidates(grid: np.ndarray, color: int, yellow_cells: set[Coord]) -> list[dict]:
    comps = _components_of_color(grid, color)
    assert len(comps) == 2, "each foreground color must form exactly two segments"
    segs = [_segment_info(comp) for comp in comps]
    occupied_non_path = {
        (int(r), int(c))
        for r, c in np.argwhere((grid != BG) & (grid != YELLOW) & (grid != color))
    }

    candidates: list[dict] = []
    seen_keys: set[tuple[tuple[Coord, ...], tuple[Coord, ...]]] = set()
    for a, b in product(segs[0]["endpoints"], segs[1]["endpoints"]):
        if a[0] == b[0] or a[1] == b[1]:
            continue
        for order in ("HV", "VH"):
            route = _build_route(a, b, order)
            route_only = route["cells"] - {a, b}
            if route_only & segs[0]["cells"]:
                continue
            if route_only & segs[1]["cells"]:
                continue
            if route_only & occupied_non_path:
                continue
            if any(int(grid[r, c]) not in (BG, YELLOW) for r, c in route_only):
                continue
            yellow_on_route = route["cells"] & yellow_cells
            cells = set(route["cells"])
            cells.update(segs[0]["cells"])
            cells.update(segs[1]["cells"])
            key = (tuple(sorted(route["cells"])), tuple(sorted(cells)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "color": color,
                    "segments": segs,
                    "endpoint_pair": (a, b),
                    "order": order,
                    "route_cells": set(route["cells"]),
                    "route_dirs": route["dirs"],
                    "cells": cells,
                    "yellow_count": len(yellow_on_route),
                    "endpoint_dist": abs(a[0] - b[0]) + abs(a[1] - b[1]),
                    "route_len": len(route["cells"]),
                }
            )
    assert candidates, f"no route candidates for color {color}"
    return candidates


def _solve_paths(grid: np.ndarray, require_unique: bool = False) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(x) for x in np.unique(g) if int(x) not in (BG, YELLOW))
    yellow_cells = {(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)}
    assert colors, "expected at least one non-background color"
    assert yellow_cells, "expected at least one yellow crossing marker"

    candidate_lists = [_enumerate_candidates(g, color, yellow_cells) for color in colors]
    valid: list[tuple[tuple[int, int, int, int], tuple[dict, ...]]] = []

    for combo in product(*candidate_lists):
        if any(cand["yellow_count"] == 0 for cand in combo):
            continue
        overlap_counts: Counter[Coord] = Counter()
        ok = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                shared = combo[i]["cells"] & combo[j]["cells"]
                for cell in shared:
                    if cell not in yellow_cells:
                        ok = False
                        break
                    dirs_i = combo[i]["route_dirs"].get(cell, frozenset())
                    dirs_j = combo[j]["route_dirs"].get(cell, frozenset())
                    if not _orthogonal_cross(dirs_i, dirs_j):
                        ok = False
                        break
                    overlap_counts[cell] += 1
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        cover_counts: Counter[Coord] = Counter()
        for cand in combo:
            for cell in cand["route_cells"] & yellow_cells:
                cover_counts[cell] += 1
        if set(cover_counts) != yellow_cells:
            continue
        if any(cover_counts[cell] != 2 for cell in yellow_cells):
            continue

        score = (
            sum(cand["yellow_count"] for cand in combo),
            -sum(cand["endpoint_dist"] for cand in combo),
            -sum(cand["route_len"] for cand in combo),
            -sum(cand["color"] for cand in combo),
        )
        valid.append((score, combo))

    assert valid, "no valid path interpretation"
    valid.sort(key=lambda item: item[0], reverse=True)
    best_score = valid[0][0]
    best_combos = [combo for score, combo in valid if score == best_score]
    assert best_combos, "internal error"
    if require_unique:
        assert len(best_combos) == 1, "path interpretation should be unique"
    chosen = best_combos[0]
    return {int(color): cand for color, cand in zip(colors, chosen)}


def _render_output(shape: tuple[int, int], paths: dict[int, dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    order = sorted(
        ((color, len(info["cells"])) for color, info in paths.items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    for color, _length in order:
        for r, c in paths[color]["cells"]:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of same-colored fragments with the yellow-guided L path, then resolve crossings by path length.

    Input structure:
    - The background is black.
    - Each non-yellow foreground color appears as exactly two short straight segments,
      each 2-4 pixels long and either horizontal or vertical.
    - Yellow singleton pixels mark every crossing point where two completed paths
      would overlap.
    - For each color, one endpoint from each segment can be connected by a one-bend
      Manhattan path; the correct bend choice is the one consistent with the yellow markers.

    Rule:
    - For every color, identify the two visible segments and enumerate endpoint pairs
      together with both possible L-shaped connectors.
    - Keep the global interpretation in which every yellow pixel is explained as a
      crossing of exactly two differently colored paths and no unmarked overlaps occur.
    - Fill all cells of every reconstructed path, including the original segments and
      the missing connector.
    - At crossings, the shorter full path is drawn on top of the longer one, so yellow
      disappears in the output.

    Key insight:
    The yellow markers do not belong to any one color; they jointly disambiguate which
    endpoints and which bend belong to each same-colored pair, and only after the full
    paths are recovered can the crossing-layer rule be applied by comparing path sizes.
    """
    g = np.asarray(grid, dtype=int)
    paths = _solve_paths(g)
    return _render_output(g.shape, paths)


def _random_route(rng: random.Random, h: int, w: int) -> dict | None:
    rows = list(range(3, h - 3))
    cols = list(range(3, w - 3))
    if len(rows) < 6 or len(cols) < 6:
        return None
    for _ in range(80):
        pr = rng.choice(rows)
        pc = rng.choice(cols)
        qr = rng.choice(rows)
        qc = rng.choice(cols)
        if pr == qr or pc == qc:
            continue
        if abs(pr - qr) < 3 or abs(pc - qc) < 3:
            continue
        order = rng.choice(["HV", "VH"])
        return _build_route((pr, pc), (qr, qc), order)
    return None


def _route_overlap_check(route: dict, existing: list[dict], used_yellows: set[Coord]) -> list[Coord] | None:
    hits: list[Coord] = []
    for old in existing:
        overlap = route["cells"] & old["cells"]
        if not overlap:
            continue
        if len(overlap) != 1:
            return None
        cell = next(iter(overlap))
        if cell in used_yellows:
            return None
        if cell in {route["p"], route["q"], route["bend"], old["p"], old["q"], old["bend"]}:
            return None
        if not _orthogonal_cross(route["dirs"].get(cell, frozenset()), old["dirs"].get(cell, frozenset())):
            return None
        hits.append(cell)
    return hits


def _cells_by_orientation(route: dict, orient: str) -> list[Coord]:
    if orient == "H":
        base = route["h_cells"]
    else:
        base = route["v_cells"]
    blocked = {route["p"], route["q"], route["bend"]}
    return [cell for cell in base if cell not in blocked]


def _route_through_cell(rng: random.Random, h: int, w: int, cell: Coord, orient: str) -> dict | None:
    rows = list(range(3, h - 3))
    cols = list(range(3, w - 3))
    r, c = cell

    def choose_sides(values: list[int], pivot: int) -> tuple[int, int] | None:
        below = [v for v in values if v <= pivot - 2]
        above = [v for v in values if v >= pivot + 2]
        if not below or not above:
            return None
        a = rng.choice(below)
        b = rng.choice(above)
        if rng.random() < 0.5:
            a, b = b, a
        return a, b

    for _ in range(120):
        order = rng.choice(["HV", "VH"])
        if orient == "V":
            side_rows = choose_sides(rows, r)
            if side_rows is None:
                continue
            ra, rb = side_rows
            if order == "HV":
                p_col_choices = [x for x in cols if abs(x - c) >= 3]
                if not p_col_choices:
                    continue
                p = (ra, rng.choice(p_col_choices))
                q = (rb, c)
            else:
                q_col_choices = [x for x in cols if abs(x - c) >= 3]
                if not q_col_choices:
                    continue
                p = (ra, c)
                q = (rb, rng.choice(q_col_choices))
        else:
            side_cols = choose_sides(cols, c)
            if side_cols is None:
                continue
            ca, cb = side_cols
            if order == "HV":
                q_row_choices = [x for x in rows if abs(x - r) >= 3]
                if not q_row_choices:
                    continue
                p = (r, ca)
                q = (rng.choice(q_row_choices), cb)
            else:
                p_row_choices = [x for x in rows if abs(x - r) >= 3]
                if not p_row_choices:
                    continue
                p = (rng.choice(p_row_choices), ca)
                q = (r, cb)
        if p[0] == q[0] or p[1] == q[1]:
            continue
        route = _build_route(p, q, order)
        if cell in {route["p"], route["q"], route["bend"]}:
            continue
        if orient not in route["dirs"].get(cell, frozenset()):
            continue
        return route
    return None


def _tail_options(
    endpoint: Coord,
    forbidden_dir: tuple[int, int],
    blocked: set[Coord],
    h: int,
    w: int,
) -> list[list[Coord]]:
    options: list[list[Coord]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if (dr, dc) == forbidden_dir:
            continue
        for length in [2, 3, 4]:
            cells = [endpoint]
            ok = True
            for step in range(1, length):
                rr = endpoint[0] + dr * step
                cc = endpoint[1] + dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if (rr, cc) in blocked:
                    ok = False
                    break
                cells.append((rr, cc))
            if ok:
                options.append(cells)
    return options


def _attach_tails(rng: random.Random, routes: list[dict]) -> list[dict] | None:
    h = max(max(r for r, _ in route["cells"]) for route in routes) + 1
    w = max(max(c for _, c in route["cells"]) for route in routes) + 1
    blocked = set().union(*(route["cells"] for route in routes))
    attached: list[dict] = []
    tail_cells_used: set[Coord] = set()
    for route in routes:
        p_opts = _tail_options(route["p"], route["p_into"], blocked | tail_cells_used, h, w)
        if not p_opts:
            return None
        rng.shuffle(p_opts)
        chosen_p = None
        chosen_q = None
        for p_tail in p_opts:
            local_blocked = blocked | tail_cells_used | set(p_tail[1:])
            q_opts = _tail_options(route["q"], route["q_into"], local_blocked, h, w)
            if not q_opts:
                continue
            rng.shuffle(q_opts)
            chosen_p = p_tail
            chosen_q = q_opts[0]
            break
        if chosen_p is None or chosen_q is None:
            return None
        tail_cells_used.update(chosen_p[1:])
        tail_cells_used.update(chosen_q[1:])
        data = dict(route)
        data["tail_a"] = set(chosen_p)
        data["tail_b"] = set(chosen_q)
        data["cells_full"] = set(route["cells"]) | set(chosen_p) | set(chosen_q)
        attached.append(data)
    return attached


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 22)
    w = rng.randint(14, 22)
    n_paths = rng.choice([2, 2, 3])
    colors = rng.sample(PALETTE, n_paths)

    first = _random_route(rng, h, w)
    if first is None:
        return None
    routes = [first]
    yellow_cells: set[Coord] = set()
    crossings = [0]

    for _idx in range(1, n_paths):
        target = rng.randrange(len(routes))
        target_route = routes[target]
        if rng.random() < 0.5:
            base_cells = _cells_by_orientation(target_route, "H")
            need_orient = "V"
        else:
            base_cells = _cells_by_orientation(target_route, "V")
            need_orient = "H"
        if not base_cells:
            return None

        placed = None
        for _ in range(180):
            anchor = rng.choice(base_cells)
            cand = _route_through_cell(rng, h, w, anchor, need_orient)
            if cand is None:
                continue
            hits = _route_overlap_check(cand, routes, yellow_cells)
            if hits is None or not hits:
                continue
            placed = cand
            for cell in hits:
                yellow_cells.add(cell)
            break
        if placed is None:
            return None
        routes.append(placed)

    if len(yellow_cells) == 0:
        return None
    for route in routes:
        if not (route["cells"] & yellow_cells):
            return None

    attached = _attach_tails(rng, routes)
    if attached is None:
        return None

    expected_paths: dict[int, dict] = {}
    for color, route in zip(colors, attached):
        expected_paths[int(color)] = {"cells": set(route["cells_full"])}

    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            shared = attached[i]["cells_full"] & attached[j]["cells_full"]
            if any(cell not in yellow_cells for cell in shared):
                return None
            if shared and len(expected_paths[colors[i]]["cells"]) == len(expected_paths[colors[j]]["cells"]):
                return None

    inp = np.full((h, w), BG, dtype=int)
    for color, route in zip(colors, attached):
        for r, c in route["tail_a"] | route["tail_b"]:
            inp[r, c] = int(color)
    for r, c in yellow_cells:
        inp[r, c] = YELLOW

    try:
        solved_paths = _solve_paths(inp, require_unique=True)
    except AssertionError:
        return None
    expected = _render_output(inp.shape, expected_paths)
    actual = _render_output(inp.shape, solved_paths)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
