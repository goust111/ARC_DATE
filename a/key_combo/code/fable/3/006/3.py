from __future__ import annotations

from collections import deque
from math import gcd
import random

import numpy as np


Coord = tuple[int, int]
BG = 0
GREY = 5
DIR4: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _color_components(g: np.ndarray, cells: set[Coord]) -> list[dict[str, object]]:
    remaining = set(cells)
    comps: list[dict[str, object]] = []
    while remaining:
        start = next(iter(remaining))
        color = int(g[start])
        q = deque([start])
        remaining.remove(start)
        pts: set[Coord] = {start}
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                p = (nr, nc)
                if p in remaining and int(g[p]) == color:
                    remaining.remove(p)
                    pts.add(p)
                    q.append(p)
        comps.append({"color": color, "cells": pts})
    return comps


def _path_endpoints(cells: set[Coord]) -> list[Coord] | None:
    if len(cells) < 2:
        return None
    degrees: dict[Coord, int] = {}
    for r, c in cells:
        deg = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        if deg not in (1, 2):
            return None
        degrees[(r, c)] = deg
    ends = [p for p, deg in degrees.items() if deg == 1]
    if len(ends) != 2:
        return None
    return ends


def _ordered_path(cells: set[Coord], start: Coord, end: Coord) -> list[Coord]:
    ordered = [start]
    prev: Coord | None = None
    cur = start
    while cur != end:
        nxts = [p for p in _neighbors4(*cur) if p in cells and p != prev]
        assert len(nxts) == 1, "path must be a single unbranched chain"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
        assert len(ordered) <= len(cells), "path must not loop"
    assert len(ordered) == len(cells), "ordered path must consume every path cell"
    return ordered


def _path_contacts(marker_cells: set[Coord], path_cells: set[Coord]) -> set[Coord]:
    contacts: set[Coord] = set()
    for r, c in marker_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in path_cells:
                contacts.add((nr, nc))
    return contacts


def _component_record(g: np.ndarray, comp: list[Coord]) -> dict[str, object]:
    comp_set = set(comp)
    pieces = _color_components(g, comp_set)
    candidates: list[dict[str, object]] = []

    for path_idx, piece in enumerate(pieces):
        path_cells = set(piece["cells"])  # type: ignore[arg-type]
        endpoints = _path_endpoints(path_cells)
        if endpoints is None:
            continue

        marker_by_end: dict[Coord, dict[str, object]] = {}
        ok = True
        for idx, marker in enumerate(pieces):
            if idx == path_idx:
                continue
            marker_cells = set(marker["cells"])  # type: ignore[arg-type]
            marker_color = int(marker["color"])
            if marker_color == int(piece["color"]) or not (1 <= len(marker_cells) <= 3):
                ok = False
                break
            contacts = _path_contacts(marker_cells, path_cells)
            if len(contacts) != 1:
                ok = False
                break
            endpoint = next(iter(contacts))
            if endpoint not in endpoints or endpoint in marker_by_end:
                ok = False
                break
            marker_by_end[endpoint] = marker

        if not ok or set(marker_by_end) != set(endpoints):
            continue

        a, b = endpoints
        ma = marker_by_end[a]
        mb = marker_by_end[b]
        if int(ma["color"]) == int(mb["color"]):
            continue

        ordered = _ordered_path(path_cells, a, b)
        candidates.append(
            {
                "path": ordered,
                "start_marker": ma,
                "end_marker": mb,
            }
        )

    assert len(candidates) == 1, "each foreground component must contain exactly one marked path"
    return candidates[0]


def _paint_path(out: np.ndarray, record: dict[str, object]) -> None:
    path: list[Coord] = record["path"]  # type: ignore[assignment]
    start_marker: dict[str, object] = record["start_marker"]  # type: ignore[assignment]
    end_marker: dict[str, object] = record["end_marker"]  # type: ignore[assignment]

    start_cells: set[Coord] = set(start_marker["cells"])  # type: ignore[arg-type]
    end_cells: set[Coord] = set(end_marker["cells"])  # type: ignore[arg-type]
    for r, c in start_cells | end_cells:
        out[r, c] = BG

    start_color = int(start_marker["color"])
    end_color = int(end_marker["color"])
    start_speed = len(start_cells)
    end_speed = len(end_cells)
    last = len(path) - 1

    for i, (r, c) in enumerate(path):
        left = i * end_speed
        right = (last - i) * start_speed
        if left < right:
            out[r, c] = start_color
        elif left > right:
            out[r, c] = end_color
        else:
            out[r, c] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marked path by the two endpoint markers racing toward one another.

    Input structure:
    - A black background contains one or more separated 4-connected, unbranched, monochrome paths.
    - Each path endpoint touches exactly one same-color marker component of size 1 to 3 pixels.
    - The two endpoint markers on a path have different colors and are the only non-path pixels in that component.

    Rule:
    - For every path cell, measure its step distance along the path to each endpoint.
    - Divide each distance by that endpoint marker's pixel count; the smaller quotient wins and recolors the path cell.
    - If the two quotients are equal, recolor that path cell grey; erase both endpoint markers to black.

    Key insight:
    - The marker size is the runner speed, so the color boundary is a weighted split of the path rather than the geometric midpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert BG in set(map(int, np.unique(g))), "black background must be present"

    out = g.copy()
    comps = _components(g != BG)
    assert comps, "expected at least one marked path"
    for comp in comps:
        record = _component_record(g, comp)
        _paint_path(out, record)
    return out


# --------------- generator ---------------

PALETTE = tuple(c for c in range(1, 10) if c != GREY)
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
PATH_LEN_RANGE = (5, 13)
MAX_ATTEMPTS = 600


def _add(a: Coord, b: Coord, k: int = 1) -> Coord:
    return a[0] + b[0] * k, a[1] + b[1] * k


def _side_dirs(d: Coord) -> tuple[Coord, Coord]:
    dr, dc = d
    return (-dc, dr), (dc, -dr)


def _transform(p: Coord, kind: int) -> Coord:
    r, c = p
    transforms = (
        (r, c),
        (c, -r),
        (-r, -c),
        (-c, r),
        (r, -c),
        (-r, c),
        (c, r),
        (-c, -r),
    )
    return transforms[kind]


def _base_path(n: int, rng: random.Random) -> list[Coord] | None:
    kinds = ["straight", "elbow"]
    if n >= 6:
        kinds.extend(["zigzag", "zigzag"])
    rng.shuffle(kinds)

    for kind in kinds:
        if kind == "straight":
            pts = [(0, c) for c in range(n)]
        elif kind == "elbow":
            a = rng.randint(2, n - 1)
            b = n - a + 1
            pts = [(0, c) for c in range(a)]
            pts.extend((r, a - 1) for r in range(1, b))
        else:
            triples = []
            for a in range(2, n - 2):
                for b in range(2, n - a + 1):
                    c = n - a - b + 2
                    if c >= 2:
                        triples.append((a, b, c))
            rng.shuffle(triples)
            pts = []
            for a, b, c in triples:
                sign = rng.choice((1, -1))
                if sign < 0 and b < 3:
                    continue
                candidate = [(0, x) for x in range(a)]
                candidate.extend((r, a - 1) for r in range(1, b))
                candidate.extend((b - 1, a - 1 + sign * x) for x in range(1, c))
                if len(set(candidate)) == n and _path_endpoints(set(candidate)) is not None:
                    pts = candidate
                    break
            if not pts:
                continue

        if len(set(pts)) == n and _path_endpoints(set(pts)) is not None:
            return pts
    return None


def _marker_options(endpoint: Coord, outward: Coord, size: int) -> list[set[Coord]]:
    base = _add(endpoint, outward)
    side_a, side_b = _side_dirs(outward)
    if size == 1:
        return [{base}]
    if size == 2:
        return [
            {base, _add(base, outward)},
            {base, _add(base, side_a)},
            {base, _add(base, side_b)},
        ]
    return [
        {base, _add(base, outward), _add(base, outward, 2)},
        {base, _add(base, outward), _add(_add(base, outward), side_a)},
        {base, _add(base, outward), _add(_add(base, outward), side_b)},
        {base, _add(base, side_a), _add(base, side_b)},
    ]


def _choose_marker(endpoint: Coord, outward: Coord, size: int, path_cells: set[Coord], rng: random.Random) -> set[Coord] | None:
    options = _marker_options(endpoint, outward, size)
    rng.shuffle(options)
    for cells in options:
        if cells & path_cells:
            continue
        if _path_contacts(cells, path_cells) == {endpoint}:
            return cells
    return None


def _choose_profile(profile: str, rng: random.Random) -> tuple[int, int, int]:
    if profile == "equal":
        speed = rng.randint(1, 3)
        n = rng.choice([5, 7, 9, 11, 13])
        return speed, speed, n

    pairs = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
    s0, s1 = rng.choice(pairs)

    if profile == "unequal_tie":
        period = (s0 + s1) // gcd(s0, s1)
        choices = [period * k + 1 for k in range(1, 6) if PATH_LEN_RANGE[0] <= period * k + 1 <= PATH_LEN_RANGE[1]]
        n = rng.choice(choices)
        return s0, s1, n

    choices = [
        n
        for n in range(PATH_LEN_RANGE[0], PATH_LEN_RANGE[1] + 1)
        if (s0 * (n - 1)) % (s0 + s1) != 0
    ]
    return s0, s1, rng.choice(choices)


def _make_local_record(profile: str, rng: random.Random) -> dict[str, object] | None:
    s0, s1, n = _choose_profile(profile, rng)
    for _ in range(80):
        base = _base_path(n, rng)
        if base is None:
            continue
        kind = rng.randrange(8)
        path = [_transform(p, kind) for p in base]
        path_cells = set(path)
        if len(path_cells) != n or _path_endpoints(path_cells) is None:
            continue

        start, end = path[0], path[-1]
        start_out = (start[0] - path[1][0], start[1] - path[1][1])
        end_out = (end[0] - path[-2][0], end[1] - path[-2][1])
        m0 = _choose_marker(start, start_out, s0, path_cells, rng)
        if m0 is None:
            continue
        m1 = _choose_marker(end, end_out, s1, path_cells, rng)
        if m1 is None or m0 & m1:
            continue

        all_cells = path_cells | m0 | m1
        min_r = min(r for r, _ in all_cells)
        min_c = min(c for _, c in all_cells)

        def norm(p: Coord) -> Coord:
            return p[0] - min_r, p[1] - min_c

        norm_path = [norm(p) for p in path]
        norm_m0 = {norm(p) for p in m0}
        norm_m1 = {norm(p) for p in m1}
        all_norm = set(norm_path) | norm_m0 | norm_m1
        height = max(r for r, _ in all_norm) + 1
        width = max(c for _, c in all_norm) + 1

        path_color = rng.choice(PALETTE)
        marker_pool = [c for c in PALETTE if c != path_color]
        color0, color1 = rng.sample(marker_pool, 2)

        return {
            "profile": profile,
            "path": norm_path,
            "start_marker": {"cells": norm_m0, "color": color0},
            "end_marker": {"cells": norm_m1, "color": color1},
            "path_color": path_color,
            "height": height,
            "width": width,
        }
    return None


def _can_place(cells: set[Coord], occupied: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _shift_cells(cells, dr: int, dc: int):
    if isinstance(cells, list):
        return [(r + dr, c + dc) for r, c in cells]
    return {(r + dr, c + dc) for r, c in cells}


def _place_record(grid: np.ndarray, local: dict[str, object], occupied: set[Coord], rng: random.Random) -> dict[str, object] | None:
    h, w = grid.shape
    oh = int(local["height"])
    ow = int(local["width"])
    margin = 1
    positions = [
        (r, c)
        for r in range(margin, h - oh - margin + 1)
        for c in range(margin, w - ow - margin + 1)
    ]
    rng.shuffle(positions)
    local_path: list[Coord] = local["path"]  # type: ignore[assignment]
    sm: dict[str, object] = local["start_marker"]  # type: ignore[assignment]
    em: dict[str, object] = local["end_marker"]  # type: ignore[assignment]
    local_cells = set(local_path) | set(sm["cells"]) | set(em["cells"])  # type: ignore[arg-type]

    for r0, c0 in positions:
        shifted_cells = _shift_cells(local_cells, r0, c0)
        if not _can_place(shifted_cells, occupied, grid.shape):
            continue

        path = _shift_cells(local_path, r0, c0)
        m0 = _shift_cells(set(sm["cells"]), r0, c0)  # type: ignore[arg-type]
        m1 = _shift_cells(set(em["cells"]), r0, c0)  # type: ignore[arg-type]
        path_color = int(local["path_color"])
        color0 = int(sm["color"])
        color1 = int(em["color"])

        for rr, cc in path:
            grid[rr, cc] = path_color
        for rr, cc in m0:
            grid[rr, cc] = color0
        for rr, cc in m1:
            grid[rr, cc] = color1

        occupied.update(shifted_cells)
        return {
            "profile": local["profile"],
            "path": path,
            "start_marker": {"cells": m0, "color": color0},
            "end_marker": {"cells": m1, "color": color1},
        }
    return None


def _expected_from_records(shape: tuple[int, int], records: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for record in records:
        _paint_path(out, record)
    return out


def _has_unequal_nonmid_tie(record: dict[str, object]) -> bool:
    path: list[Coord] = record["path"]  # type: ignore[assignment]
    sm: dict[str, object] = record["start_marker"]  # type: ignore[assignment]
    em: dict[str, object] = record["end_marker"]  # type: ignore[assignment]
    s0 = len(set(sm["cells"]))  # type: ignore[arg-type]
    s1 = len(set(em["cells"]))  # type: ignore[arg-type]
    if s0 == s1:
        return False
    last = len(path) - 1
    ties = [i for i in range(len(path)) if i * s1 == (last - i) * s0]
    return bool(ties) and ties[0] * 2 != last


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, records: list[dict[str, object]]) -> bool:
    profiles = {str(rec["profile"]) for rec in records}
    if not {"equal", "unequal_tie", "unequal_no_tie"} <= profiles:
        return False
    if not any(_has_unequal_nonmid_tie(rec) for rec in records):
        return False
    if np.array_equal(grid, expected):
        return False

    for rec in records:
        path: list[Coord] = rec["path"]  # type: ignore[assignment]
        sm: dict[str, object] = rec["start_marker"]  # type: ignore[assignment]
        em: dict[str, object] = rec["end_marker"]  # type: ignore[assignment]
        m0 = set(sm["cells"])  # type: ignore[arg-type]
        m1 = set(em["cells"])  # type: ignore[arg-type]
        if not (1 <= len(m0) <= 3 and 1 <= len(m1) <= 3 and int(sm["color"]) != int(em["color"])):
            return False
        if any(expected[p] != BG for p in m0 | m1):
            return False
        if rec["profile"] == "unequal_no_tie" and any(expected[p] == GREY for p in path):
            return False
        if rec["profile"] in {"equal", "unequal_tie"} and not any(expected[p] == GREY for p in path):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    profiles = ["equal", "unequal_tie", "unequal_no_tie"]
    if rng.random() < 0.45:
        profiles.append(rng.choice(profiles))
    rng.shuffle(profiles)

    locals_: list[dict[str, object]] = []
    for profile in profiles:
        rec = _make_local_record(profile, rng)
        if rec is None:
            return None
        locals_.append(rec)

    grid = np.full((h, w), BG, dtype=int)
    occupied: set[Coord] = set()
    placed: list[dict[str, object]] = []
    for local in sorted(locals_, key=lambda x: int(x["height"]) * int(x["width"]), reverse=True):
        rec = _place_record(grid, local, occupied, rng)
        if rec is None:
            return None
        placed.append(rec)

    expected = _expected_from_records(grid.shape, placed)
    if not _semantic_ok(grid, expected, placed):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-ended race input")
