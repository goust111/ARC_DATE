from __future__ import annotations

import random
from typing import Callable

import numpy as np


BLACK = 0
GREY = 5
TOKEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SHAPE_SOLID = 0
SHAPE_X = 1
SHAPE_DIAMOND = 2

SHAPE_MASKS = {
    SHAPE_SOLID: np.ones((3, 3), dtype=bool),
    SHAPE_X: np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    SHAPE_DIAMOND: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
}

ZONE = 13
TEMPLATE_EVEN = {
    0: [(6, 3), (6, 9)],
    2: [(6, 3), (6, 7), (4, 7), (4, 9)],
    4: [(6, 3), (6, 5), (4, 5), (4, 8), (8, 8), (8, 9)],
}
TEMPLATE_ODD = {
    1: [(6, 3), (6, 9), (3, 9)],
    3: [(6, 3), (6, 7), (5, 7), (5, 9), (3, 9)],
    5: [(6, 3), (6, 4), (8, 4), (8, 8), (5, 8), (5, 9), (3, 9)],
}
TRANSFORMS: tuple[Callable[[tuple[int, int]], tuple[int, int]], ...] = (
    lambda p: (p[0], p[1]),
    lambda p: (p[1], ZONE - 1 - p[0]),
    lambda p: (ZONE - 1 - p[0], ZONE - 1 - p[1]),
    lambda p: (ZONE - 1 - p[1], p[0]),
    lambda p: (ZONE - 1 - p[0], p[1]),
    lambda p: (p[1], p[0]),
    lambda p: (p[0], ZONE - 1 - p[1]),
    lambda p: (ZONE - 1 - p[1], ZONE - 1 - p[0]),
)

GEN_RETRIES = 400
_GEN_RNG = random.Random()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _draw_shape(grid: np.ndarray, center: tuple[int, int], color: int, shape_idx: int) -> None:
    r0 = center[0] - 1
    c0 = center[1] - 1
    mask = SHAPE_MASKS[shape_idx]
    for dr in range(3):
        for dc in range(3):
            rr = r0 + dr
            cc = c0 + dc
            if mask[dr, dc]:
                grid[rr, cc] = color
            else:
                grid[rr, cc] = BLACK


def _detect_tokens(grid: np.ndarray) -> dict[tuple[int, int], dict[str, int]]:
    h, w = grid.shape
    tokens: dict[tuple[int, int], dict[str, int]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            sub = grid[r - 1 : r + 2, c - 1 : c + 2]
            values = sorted({int(v) for v in sub.flatten()} - {BLACK})
            if len(values) != 1:
                continue
            color = values[0]
            if color == GREY:
                continue
            for shape_idx, mask in SHAPE_MASKS.items():
                if np.all(sub[mask] == color) and np.all(sub[~mask] == BLACK):
                    tokens[(r, c)] = {"shape": shape_idx, "color": color}
                    break
    return tokens


def _wire_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_wire_path(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(comp)
    degree = {}
    for p in comp:
        degree[p] = sum((nr, nc) in pts for nr, nc in _neighbors4(*p))
    endpoints = [p for p, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "wire must be a simple path with two endpoints"
    start = min(endpoints)
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(*cur) if p in pts and p != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "wire may not branch"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    assert len(path) == len(comp), "wire path must visit every wire cell exactly once"
    return path


def _bend_count(path: list[tuple[int, int]]) -> int:
    bends = 0
    prev_dir = None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        step = (r1 - r0, c1 - c0)
        if prev_dir is not None and step != prev_dir:
            bends += 1
        prev_dir = step
    return bends


def _parse_scene(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    tokens = _detect_tokens(g)
    wires: list[dict[str, object]] = []
    for comp in _wire_components(g):
        path = _trace_wire_path(comp)
        bends = _bend_count(path)
        endpoints: list[dict[str, object]] = []
        for endpoint, neighbor in ((path[0], path[1]), (path[-1], path[-2])):
            outward = (endpoint[0] - neighbor[0], endpoint[1] - neighbor[1])
            center = (endpoint[0] + 2 * outward[0], endpoint[1] + 2 * outward[1])
            assert 1 <= center[0] < h - 1 and 1 <= center[1] < w - 1, "token center out of bounds"
            token = tokens.get(center)
            endpoints.append(
                {
                    "endpoint": endpoint,
                    "center": center,
                    "outward": outward,
                    "token": token,
                }
            )
        wires.append({"path": path, "bends": bends, "ends": endpoints})
    return wires


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each token across its grey wire and advance its shape by the wire's bend count.

    Input structure:
    - Black background contains 2-4 independent grey 4-connected wires, each a single-pixel path with two endpoints.
    - Some wires have one colored 3x3 token placed just outside one endpoint; tokens are solid blocks, X patterns, or diamonds.
    - The opposite endpoint of an active wire has an empty 3x3 black target area; some wires may have no token on either end.

    Rule:
    - For every wire, identify its two endpoints and trace the unique path between them.
    - Count how many times the path changes between horizontal and vertical motion.
    - If exactly one endpoint has a token of shape S, draw at the opposite endpoint the same color with shape (S + bends) mod 3 under the cycle solid -> X -> diamond -> solid.
    - Keep the source token, wires, and empty-only wires unchanged.

    Key insight:
    - The transmission target depends on both endpoint pairing and the wire geometry: the token follows its wire, but the number of right-angle bends determines how far the shape advances around the three-shape cycle.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for wire in _parse_scene(g):
        ends = wire["ends"]
        src_candidates = [end for end in ends if end["token"] is not None]
        if not src_candidates:
            continue
        assert len(src_candidates) == 1, "at most one token may appear on a wire"
        src = src_candidates[0]
        dst = ends[1] if ends[0] is src else ends[0]
        token = src["token"]
        assert token is not None
        dst_shape = (int(token["shape"]) + int(wire["bends"])) % 3
        _draw_shape(out, dst["center"], int(token["color"]), dst_shape)
    return out


def _expand_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [vertices[0]]
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        assert dr == 0 or dc == 0
        cr, cc = r0, c0
        while (cr, cc) != (r1, c1):
            cr += dr
            cc += dc
            cells.append((cr, cc))
    return cells


def _make_local_path(bends: int, rng: random.Random) -> list[tuple[int, int]]:
    vertices = TEMPLATE_EVEN[bends] if bends % 2 == 0 else TEMPLATE_ODD[bends]
    while True:
        transform = TRANSFORMS[rng.randrange(len(TRANSFORMS))]
        path = _expand_vertices([transform(p) for p in vertices])
        if len(set(path)) == len(path):
            return path


def _random_bend(rng: random.Random, *, prefer_wrap: bool) -> int:
    choices = [0, 1, 2, 3, 4, 5]
    if prefer_wrap:
        weights = [1, 1, 1, 2, 2, 2]
        return rng.choices(choices, weights=weights, k=1)[0]
    return rng.choice(choices)


def _sample_active_specs(active_count: int, rng: random.Random) -> list[dict[str, int]]:
    specs: list[dict[str, int]]
    while True:
        specs = []
        if active_count >= 3:
            base_shape = rng.randrange(3)
            alt_shape = rng.choice([s for s in range(3) if s != base_shape])
            bend0 = _random_bend(rng, prefer_wrap=True)
            bend1_options = [b for b in range(6) if b % 3 != bend0 % 3]
            bend1 = rng.choice(bend1_options)
            specs.extend(
                [
                    {"shape": base_shape, "bends": bend0},
                    {"shape": base_shape, "bends": bend1},
                    {"shape": alt_shape, "bends": bend0},
                ]
            )
            while len(specs) < active_count:
                specs.append({"shape": rng.randrange(3), "bends": _random_bend(rng, prefer_wrap=True)})
        else:
            if rng.random() < 0.5:
                shape = rng.randrange(3)
                bend0 = _random_bend(rng, prefer_wrap=True)
                bend1 = rng.choice([b for b in range(6) if b % 3 != bend0 % 3])
                specs = [{"shape": shape, "bends": bend0}, {"shape": shape, "bends": bend1}]
            else:
                bend = _random_bend(rng, prefer_wrap=True)
                shape0 = rng.randrange(3)
                shape1 = rng.choice([s for s in range(3) if s != shape0])
                specs = [{"shape": shape0, "bends": bend}, {"shape": shape1, "bends": bend}]
        targets = {(spec["shape"] + spec["bends"]) % 3 for spec in specs}
        if len(targets) >= 2:
            return specs


def _build_layout(n_wires: int, rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    gap = rng.randint(1, 3)
    if n_wires <= 2 and rng.random() < 0.5:
        rows, cols = n_wires, 1
    elif n_wires == 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2
    height = rows * ZONE + (rows - 1) * gap
    width = cols * ZONE + (cols - 1) * gap
    origins: list[tuple[int, int]] = []
    for rr in range(rows):
        for cc in range(cols):
            origins.append((rr * (ZONE + gap), cc * (ZONE + gap)))
    rng.shuffle(origins)
    return origins[:n_wires], (height, width)


def _place_wire(
    grid: np.ndarray,
    origin: tuple[int, int],
    bends: int,
    token_shape: int | None,
    token_color: int | None,
    rng: random.Random,
) -> dict[str, object]:
    local_path = _make_local_path(bends, rng)
    global_path = [(origin[0] + r, origin[1] + c) for r, c in local_path]
    for r, c in global_path:
        grid[r, c] = GREY

    meta: dict[str, object] = {
        "path": global_path,
        "bends": bends,
        "source_shape": token_shape,
        "token_color": token_color,
        "source_center": None,
        "target_center": None,
    }
    if token_shape is None or token_color is None:
        return meta

    start = global_path[0]
    second = global_path[1]
    end = global_path[-1]
    prev = global_path[-2]
    src_outward = (start[0] - second[0], start[1] - second[1])
    dst_outward = (end[0] - prev[0], end[1] - prev[1])
    src_center = (start[0] + 2 * src_outward[0], start[1] + 2 * src_outward[1])
    dst_center = (end[0] + 2 * dst_outward[0], end[1] + 2 * dst_outward[1])
    _draw_shape(grid, src_center, token_color, token_shape)
    meta["source_center"] = src_center
    meta["target_center"] = dst_center
    return meta


def _semantic_checks(grid: np.ndarray, metas: list[dict[str, object]], active_count: int) -> bool:
    parsed = _parse_scene(grid)
    if len(parsed) != len(metas):
        return False
    parsed_bends = sorted(int(wire["bends"]) for wire in parsed)
    meta_bends = sorted(int(meta["bends"]) for meta in metas)
    if parsed_bends != meta_bends:
        return False
    token_lines = []
    empty_lines = 0
    for wire in parsed:
        present = [end for end in wire["ends"] if end["token"] is not None]
        if len(present) == 1:
            token_lines.append(
                {
                    "shape": int(present[0]["token"]["shape"]),
                    "bends": int(wire["bends"]),
                    "target": (int(present[0]["token"]["shape"]) + int(wire["bends"])) % 3,
                }
            )
        elif len(present) == 0:
            empty_lines += 1
        else:
            return False
    if len(token_lines) != active_count:
        return False
    if active_count >= 2:
        same_shape_diff_bends = any(
            a["shape"] == b["shape"] and a["bends"] != b["bends"] and a["target"] != b["target"]
            for i, a in enumerate(token_lines)
            for b in token_lines[i + 1 :]
        )
        same_bend_diff_shape = any(
            a["bends"] == b["bends"] and a["shape"] != b["shape"] and a["target"] != b["target"]
            for i, a in enumerate(token_lines)
            for b in token_lines[i + 1 :]
        )
        if active_count >= 3 and not (same_shape_diff_bends and same_bend_diff_shape):
            return False
        if active_count == 2 and not (same_shape_diff_bends or same_bend_diff_shape):
            return False
    if empty_lines and active_count == len(parsed):
        return False
    return True


def _expected_output(inp: np.ndarray, metas: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for meta in metas:
        if meta["source_shape"] is None:
            continue
        target_shape = (int(meta["source_shape"]) + int(meta["bends"])) % 3
        _draw_shape(out, meta["target_center"], int(meta["token_color"]), target_shape)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_wires = rng.choices([2, 3, 4], weights=[1, 3, 2], k=1)[0]
    empty_count = 0 if n_wires == 2 else rng.choice([0, 1])
    active_count = n_wires - empty_count
    specs = _sample_active_specs(active_count, rng)
    rng.shuffle(specs)
    colors = rng.sample(TOKEN_COLORS, active_count)
    origins, shape = _build_layout(n_wires, rng)
    grid = np.full(shape, BLACK, dtype=int)

    metas: list[dict[str, object]] = []
    for idx, origin in enumerate(origins):
        if idx < active_count:
            spec = specs[idx]
            meta = _place_wire(
                grid,
                origin,
                int(spec["bends"]),
                int(spec["shape"]),
                colors[idx],
                rng,
            )
        else:
            meta = _place_wire(grid, origin, _random_bend(rng, prefer_wrap=True), None, None, rng)
        metas.append(meta)

    if not _semantic_checks(grid, metas, active_count):
        return None

    expected = _expected_output(grid, metas)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bend-coded token transmission input")
