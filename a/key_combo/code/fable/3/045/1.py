from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SHAPES: dict[str, tuple[Coord, ...]] = {
    "solid": (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ),
    "x": ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)),
    "diamond": ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
}
_CYCLE = ("solid", "x", "diamond")
_FULL_BOX = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _get_bg(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in _N4:
        yield r + dr, c + dc


def _paint_token(g: np.ndarray, center: Coord, color: int, shape: str) -> None:
    cr, cc = center
    for dr, dc in _SHAPES[shape]:
        g[cr + dr, cc + dc] = int(color)


def _token_cells(center: Coord, shape: str) -> set[Coord]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in _SHAPES[shape]}


def _box_cells(center: Coord) -> set[Coord]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in _FULL_BOX}


def _find_tokens(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    tokens: list[dict] = []
    used: set[Coord] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in used or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            for name, offsets in _SHAPES.items():
                on = {(r + dr, c + dc) for dr, dc in offsets}
                off = _box_cells((r, c)) - on
                if any(int(g[rr, cc]) != color for rr, cc in on):
                    continue
                if any(int(g[rr, cc]) != bg for rr, cc in off):
                    continue
                # A token is a clean 3x3 symbol, not part of a larger same-color object.
                if any(
                    _in_bounds(g.shape, nr, nc)
                    and (nr, nc) not in on
                    and int(g[nr, nc]) == color
                    for rr, cc in on
                    for nr, nc in _neighbors4(rr, cc)
                ):
                    continue
                tokens.append({"center": (r, c), "color": color, "shape": name, "cells": on})
                used.update(on)
                break
    return tokens


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[Coord]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[Coord] = deque([(r, c)])
        comp: list[Coord] = []
        while q:
            pr, pc = q.popleft()
            comp.append((pr, pc))
            for nr, nc in _neighbors4(pr, pc):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _endpoint_dirs(comp: list[Coord]) -> dict[Coord, Coord]:
    cells = set(comp)
    dirs: dict[Coord, Coord] = {}
    for r, c in comp:
        ns = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in cells]
        assert len(ns) <= 2, "wire paths must not branch"
        if len(ns) == 1:
            nr, nc = ns[0]
            dirs[(r, c)] = (r - nr, c - nc)
    assert len(dirs) == 2, "each wire must have exactly two endpoints"
    return dirs


def _dominant_wire_color(g: np.ndarray, comp: list[Coord]) -> int:
    counts = Counter(int(g[r, c]) for r, c in comp)
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "wire color must be unique by area inside each path"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit each endpoint token to the opposite end of its wire, advancing its shape by the bead count.

    Input structure:
    - The background is the dominant color.
    - Several disjoint one-cell-wide wire paths are drawn on the background; each path has a clean 3x3 token slot at exactly one end and an empty matching slot at the other end.
    - Each source token is one of three 3x3 shapes: solid block, X, or diamond, and bead pixels embedded in the same wire path use colors different from the wire color.
    - The bead counts on the different wires are distinct.

    Rule:
    - Keep the source tokens, wires, and bead pixels unchanged.
    - For each wire, count its bead pixels, then advance the source token shape through the fixed cycle solid -> X -> diamond -> solid by that many steps.
    - Draw the advanced shape in the empty slot at the opposite endpoint, using the source token color.

    Key insight:
    - The beads are counters, not obstacles; wires with bead counts congruent to 0 mod 3 copy the shape, while the other wires expose the cyclic shape advance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must stay within 30x30"

    bg = _get_bg(g)
    tokens = _find_tokens(g, bg)
    assert tokens, "expected at least one source token"

    token_cells: set[Coord] = set()
    token_by_center: dict[Coord, dict] = {}
    for token in tokens:
        center = token["center"]
        assert center not in token_by_center, "token centers must be unique"
        token_by_center[center] = token
        token_cells.update(token["cells"])

    path_mask = (g != bg)
    for r, c in token_cells:
        path_mask[r, c] = False

    comps = _components(path_mask)
    assert comps, "expected wire paths"

    out = g.copy()
    used_tokens: set[Coord] = set()
    bead_counts: list[int] = []

    for comp in comps:
        assert len(comp) >= 3, "wire path is too short"
        endpoint_dirs = _endpoint_dirs(comp)
        wire_color = _dominant_wire_color(g, comp)
        bead_count = sum(1 for r, c in comp if int(g[r, c]) != wire_color)
        bead_counts.append(bead_count)

        attached: list[tuple[Coord, Coord, dict]] = []
        for endpoint, outward in endpoint_dirs.items():
            er, ec = endpoint
            dr, dc = outward
            center = (er + 2 * dr, ec + 2 * dc)
            if center in token_by_center:
                attached.append((endpoint, outward, token_by_center[center]))

        assert len(attached) == 1, "each wire must have exactly one source token"
        source_endpoint, _source_outward, token = attached[0]
        used_tokens.add(token["center"])

        dest_candidates = [ep for ep in endpoint_dirs if ep != source_endpoint]
        assert len(dest_candidates) == 1
        dest_endpoint = dest_candidates[0]
        dest_outward = endpoint_dirs[dest_endpoint]
        tr = dest_endpoint[0] + 2 * dest_outward[0]
        tc = dest_endpoint[1] + 2 * dest_outward[1]
        assert 1 <= tr < h - 1 and 1 <= tc < w - 1, "destination token slot must fit"
        assert (tr, tc) not in token_by_center, "opposite endpoint must be empty"

        shape_index = _CYCLE.index(token["shape"])
        result_shape = _CYCLE[(shape_index + bead_count) % len(_CYCLE)]
        _paint_token(out, (tr, tc), token["color"], result_shape)

    assert used_tokens == set(token_by_center), "every token must be attached to one wire"
    assert len(bead_counts) == len(set(bead_counts)), "wire bead counts must be distinct"
    return out


# --------------- generator ---------------

_BG = 0
_WIRE_COLOR = 8
_BEAD_COLORS = (5,)
_TOKEN_COLORS = (1, 2, 3, 4, 6, 7, 9)
_GEN_RETRIES = 500
_PLACE_RETRIES = 300


def _expanded(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _path_from_steps(start: Coord, dirs_and_lengths: list[tuple[Coord, int]]) -> list[Coord] | None:
    path = [start]
    r, c = start
    for (dr, dc), length in dirs_and_lengths:
        for _ in range(length):
            r += dr
            c += dc
            if (r, c) in path:
                return None
            path.append((r, c))
    return path


def _sample_bead_indices(path_len: int, bead_count: int) -> list[int] | None:
    candidates = list(range(1, path_len - 1))
    for _ in range(80):
        random.shuffle(candidates)
        chosen: list[int] = []
        for idx in candidates:
            if all(abs(idx - other) > 1 for other in chosen):
                chosen.append(idx)
                if len(chosen) == bead_count:
                    return sorted(chosen)
    return None


def _make_counts(n_wires: int) -> list[int]:
    counts = [
        random.choice([0, 3]),
        random.choice([1, 4]),
        random.choice([2, 5]),
    ]
    if n_wires == 4:
        remaining = [k for k in range(6) if k not in counts]
        counts.append(random.choice(remaining))
    random.shuffle(counts)
    return counts


def _make_shapes(n_wires: int) -> list[str]:
    shapes = list(_CYCLE)
    random.shuffle(shapes)
    while len(shapes) < n_wires:
        shapes.append(random.choice(_CYCLE))
    random.shuffle(shapes)
    return shapes


def _random_candidate_path(shape: tuple[int, int], min_cells: int) -> list[Coord] | None:
    h, w = shape
    dirs = list(_N4)
    random.shuffle(dirs)
    kind = "bend" if random.random() < 0.45 else "straight"

    if kind == "straight":
        d = random.choice(dirs)
        length = random.randint(max(1, min_cells - 1), min(16, max(1, min(h, w) - 5)))
        start = (random.randint(3, h - 4), random.randint(3, w - 4))
        return _path_from_steps(start, [(d, length)])

    d1 = random.choice(dirs)
    perp = [(d1[1], d1[0]), (-d1[1], -d1[0])]
    d2 = random.choice(perp)
    total_steps = random.randint(max(2, min_cells - 1), 18)
    len1 = random.randint(2, total_steps - 1)
    len2 = total_steps - len1
    start = (random.randint(3, h - 4), random.randint(3, w - 4))
    return _path_from_steps(start, [(d1, len1), (d2, len2)])


def _candidate_fits(
    path: list[Coord],
    shape: tuple[int, int],
    reserved: set[Coord],
) -> tuple[Coord, Coord, set[Coord]] | None:
    if path is None or len(path) < 3:
        return None
    h, w = shape
    if any(not _in_bounds(shape, r, c) for r, c in path):
        return None

    start = path[0]
    second = path[1]
    end = path[-1]
    before_end = path[-2]
    source_out = (start[0] - second[0], start[1] - second[1])
    dest_out = (end[0] - before_end[0], end[1] - before_end[1])
    source_center = (start[0] + 2 * source_out[0], start[1] + 2 * source_out[1])
    dest_center = (end[0] + 2 * dest_out[0], end[1] + 2 * dest_out[1])

    for center in (source_center, dest_center):
        r, c = center
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return None

    path_set = set(path)
    if len(path_set) != len(path):
        return None
    if any(cell in reserved for cell in path_set):
        return None

    source_box = _box_cells(source_center)
    dest_box = _box_cells(dest_center)
    if path_set & (source_box | dest_box):
        return None

    all_cells = path_set | source_box | dest_box
    if any(cell in reserved for cell in all_cells):
        return None
    return source_center, dest_center, all_cells


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        source_index = _CYCLE.index(spec["shape"])
        result_shape = _CYCLE[(source_index + spec["bead_count"]) % len(_CYCLE)]
        _paint_token(out, spec["dest_center"], spec["token_color"], result_shape)
    return out


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    counts = [int(spec["bead_count"]) for spec in specs]
    if len(counts) != len(set(counts)):
        return False
    if {c % 3 for c in counts} != {0, 1, 2}:
        return False
    if not any(c >= 3 and c % 3 == 0 for c in counts):
        return False

    changed = 0
    unchanged = 0
    for spec in specs:
        result_shape = _CYCLE[(_CYCLE.index(spec["shape"]) + spec["bead_count"]) % 3]
        if result_shape == spec["shape"]:
            unchanged += 1
        else:
            changed += 1
        for cell in spec["path"]:
            if int(inp[cell]) != int(expected[cell]):
                return False
        for cell in _token_cells(spec["source_center"], spec["shape"]):
            if int(inp[cell]) != int(expected[cell]):
                return False
    return changed > 0 and unchanged > 0


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    n_wires = random.choice([3, 4])
    counts = _make_counts(n_wires)
    shapes = _make_shapes(n_wires)
    token_colors = random.sample(_TOKEN_COLORS, n_wires)

    grid = np.full((h, w), _BG, dtype=int)
    reserved: set[Coord] = set()
    specs: list[dict] = []

    for i in range(n_wires):
        bead_count = counts[i]
        min_cells = max(8, bead_count * 2 + 3)
        placed = False
        for _ in range(_PLACE_RETRIES):
            path = _random_candidate_path((h, w), min_cells)
            fit = _candidate_fits(path, (h, w), reserved)
            if fit is None:
                continue
            bead_indices = _sample_bead_indices(len(path), bead_count)
            if bead_indices is None:
                continue

            source_center, dest_center, all_cells = fit
            for r, c in path:
                grid[r, c] = _WIRE_COLOR
            for idx in bead_indices:
                r, c = path[idx]
                grid[r, c] = random.choice(_BEAD_COLORS)
            _paint_token(grid, source_center, token_colors[i], shapes[i])

            reserved |= _expanded(all_cells, (h, w))
            specs.append(
                {
                    "path": tuple(path),
                    "source_center": source_center,
                    "dest_center": dest_center,
                    "shape": shapes[i],
                    "token_color": token_colors[i],
                    "bead_count": bead_count,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, specs)
    if not _semantic_ok(grid, expected, specs):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bead-count wire transmission input")
