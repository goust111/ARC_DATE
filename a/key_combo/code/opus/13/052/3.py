from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
ROLE_PATH = 1
ROLE_STAMP = 2

GRID_H_RANGE = (17, 20)
GRID_W_RANGE = (17, 20)
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _range_inclusive(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _normalize_pair(color: int, coords: np.ndarray) -> dict[str, object]:
    pts = sorted((int(r), int(c)) for r, c in coords)
    assert len(pts) == 2
    a, b = pts
    assert a[0] != b[0] and a[1] != b[1]
    assert a[0] < b[0]
    return {"color": int(color), "a": a, "b": b}


def _marker_cells(specs: list[dict[str, object]]) -> set[tuple[int, int]]:
    return {tuple(spec["a"]) for spec in specs} | {tuple(spec["b"]) for spec in specs}


def _elbow(spec: dict[str, object], route: str = "horizontal") -> tuple[int, int]:
    ar, ac = spec["a"]
    br, bc = spec["b"]
    if route == "horizontal":
        return int(ar), int(bc)
    return int(br), int(ac)


def _path_geom(spec: dict[str, object], route: str = "horizontal") -> set[tuple[int, int]]:
    ar, ac = spec["a"]
    br, bc = spec["b"]
    cells: set[tuple[int, int]] = set()
    if route == "horizontal":
        for c in _range_inclusive(int(ac), int(bc)):
            cells.add((int(ar), int(c)))
        for r in _range_inclusive(int(ar), int(br)):
            cells.add((int(r), int(bc)))
        return cells
    for r in _range_inclusive(int(ar), int(br)):
        cells.add((int(r), int(ac)))
    for c in _range_inclusive(int(ac), int(bc)):
        cells.add((int(br), int(c)))
    return cells


def _stamp_geom(
    spec: dict[str, object],
    shape: tuple[int, int],
    route: str = "horizontal",
) -> set[tuple[int, int]]:
    er, ec = _elbow(spec, route)
    cells = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = er + dr, ec + dc
        if _in_bounds(shape, rr, cc):
            cells.add((rr, cc))
    return cells


def _collect_claims(
    specs: list[dict[str, object]],
    shape: tuple[int, int],
    *,
    route: str = "horizontal",
    include_stamps: bool = True,
) -> dict[tuple[int, int], dict[int, int]]:
    markers = _marker_cells(specs)
    claims: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)

    for spec in specs:
        color = int(spec["color"])
        for cell in _path_geom(spec, route):
            if cell in markers:
                continue
            claims[cell][color] = claims[cell].get(color, 0) | ROLE_PATH
        if not include_stamps:
            continue
        for cell in _stamp_geom(spec, shape, route):
            if cell in markers:
                continue
            claims[cell][color] = claims[cell].get(color, 0) | ROLE_STAMP

    return claims


def _render_on_grid(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    *,
    route: str = "horizontal",
    include_stamps: bool = True,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    claims = _collect_claims(specs, out.shape, route=route, include_stamps=include_stamps)

    for (r, c), by_color in claims.items():
        if len(by_color) >= 2:
            out[r, c] = ORANGE
            continue
        color, role = next(iter(by_color.items()))
        out[r, c] = int(color) if role & ROLE_STAMP else GREY

    return out


def _extract_specs(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g) if int(v) != BLACK]
    assert 3 <= len(colors) <= 4
    assert set(colors).issubset(set(MARKER_COLORS))
    assert int(np.count_nonzero(g)) == 2 * len(colors)

    specs = []
    for color in sorted(colors):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2)
        specs.append(_normalize_pair(color, coords))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with a horizontal-first grey bend, stamp the elbow neighbors, and turn multi-pair overlaps orange.

    Input structure:
    - The grid has black background and 3-4 non-black marker colors chosen from blue, red, green, and yellow.
    - Each marker color appears exactly twice, and each same-color pair lies on different rows and different columns.
    - The markers are isolated points; no pre-drawn paths, stamps, grey pixels, or orange collision pixels appear in the input.

    Rule:
    - Pair the two markers of each color; for every pair, let A be the marker with the smaller row and B the one with the larger row.
    - Draw a horizontal-first Manhattan polyline from A to B: move along A's row until B's column, then move downward along that column to B.
    - Path cells paint grey only on non-marker cells; the bend cell is the elbow.
    - Stamp the elbow's four orthogonal neighbors with that pair's marker color, again skipping original marker cells.
    - After all pairs are drawn, any non-marker cell claimed by two or more different pairs, whether by a grey path or by a colored stamp, becomes orange.

    Key insight:
    - The elbow is not arbitrary: the smaller-row marker fixes a unique horizontal-first bend, and that derived bend both positions the colored stamp and determines where overlap collisions must turn orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    specs = _extract_specs(g)
    return _render_on_grid(g, specs, route="horizontal", include_stamps=True)


def _make_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        ar, ac = spec["a"]
        br, bc = spec["b"]
        grid[int(ar), int(ac)] = color
        grid[int(br), int(bc)] = color
    return grid


def _sample_increasing_with_gap(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    compressed_hi = int(hi) - (count - 1) * (gap - 1)
    if compressed_hi - int(lo) + 1 < count:
        return None
    picks = sorted(random.sample(range(int(lo), compressed_hi + 1), count))
    return [int(v + i * (gap - 1)) for i, v in enumerate(picks)]


def _structure_ok(shape: tuple[int, int], specs: list[dict[str, object]]) -> bool:
    h, w = shape
    markers = list(_marker_cells(specs))
    if len(markers) != 2 * len(specs):
        return False

    for spec in specs:
        ar, ac = spec["a"]
        br, bc = spec["b"]
        if not (_in_bounds(shape, ar, ac) and _in_bounds(shape, br, bc)):
            return False
        if not (ar < br and ac != bc):
            return False
        er, ec = _elbow(spec, "horizontal")
        if not (1 <= er < h - 1 and 1 <= ec < w - 1):
            return False
        if abs(ac - bc) + abs(br - ar) < 6:
            return False
        if len(_stamp_geom(spec, shape, "horizontal")) != 4:
            return False

    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) < 4:
                return False

    for i, spec in enumerate(specs):
        path = _path_geom(spec, "horizontal")
        stamp = _stamp_geom(spec, shape, "horizontal")
        for j, other in enumerate(specs):
            if i == j:
                continue
            for marker in (tuple(other["a"]), tuple(other["b"])):
                if marker in path or marker in stamp:
                    return False

    return True


def _metrics(shape: tuple[int, int], specs: list[dict[str, object]]) -> dict[str, object]:
    inp = _make_input(shape, specs)
    output = _render_on_grid(inp, specs, route="horizontal", include_stamps=True)
    vertical_output = _render_on_grid(inp, specs, route="vertical", include_stamps=True)
    path_only_output = _render_on_grid(inp, specs, route="horizontal", include_stamps=False)
    claims = _collect_claims(specs, shape, route="horizontal", include_stamps=True)

    colors = [int(spec["color"]) for spec in specs]
    single_path_only = {color: set() for color in colors}
    single_stamp_visible = {color: set() for color in colors}
    collision_cells: list[tuple[int, int]] = []
    path_collision_cells: list[tuple[int, int]] = []
    stamp_collision_cells: list[tuple[int, int]] = []

    for cell, by_color in claims.items():
        if len(by_color) >= 2:
            collision_cells.append(cell)
            if sum(bool(role & ROLE_PATH) for role in by_color.values()) >= 2:
                path_collision_cells.append(cell)
            if any(role & ROLE_STAMP for role in by_color.values()):
                stamp_collision_cells.append(cell)
            continue
        color, role = next(iter(by_color.items()))
        if role == ROLE_PATH:
            single_path_only[int(color)].add(cell)
        if role & ROLE_STAMP:
            single_stamp_visible[int(color)].add(cell)

    return {
        "input": inp,
        "output": output,
        "vertical_output": vertical_output,
        "path_only_output": path_only_output,
        "claims": claims,
        "single_path_only": single_path_only,
        "single_stamp_visible": single_stamp_visible,
        "collision_cells": collision_cells,
        "path_collision_cells": path_collision_cells,
        "stamp_collision_cells": stamp_collision_cells,
    }


def _semantic_ok(
    shape: tuple[int, int],
    specs: list[dict[str, object]],
    *,
    require_stamp_collision: bool,
) -> bool:
    if not _structure_ok(shape, specs):
        return False

    info = _metrics(shape, specs)
    if len(info["path_collision_cells"]) < 2:
        return False
    if not (2 <= len(info["collision_cells"]) <= 10):
        return False
    if np.array_equal(info["output"], info["vertical_output"]):
        return False
    if np.array_equal(info["output"], info["path_only_output"]):
        return False
    if not np.any(info["output"] == ORANGE):
        return False

    for spec in specs:
        color = int(spec["color"])
        if len(info["single_stamp_visible"][color]) < 2:
            return False
        if len(info["single_path_only"][color]) < 2:
            return False

    if not np.array_equal(main(info["input"]), info["output"]):
        return False

    if require_stamp_collision:
        stamp_cells = set(info["stamp_collision_cells"])
        if not stamp_cells:
            return False
        changed = False
        for r, c in stamp_cells:
            if int(info["output"][r, c]) != int(info["path_only_output"][r, c]):
                changed = True
                break
        if not changed:
            return False

    return True


def _build_base_specs(shape: tuple[int, int], colors: list[int]) -> list[dict[str, object]] | None:
    h, w = shape
    for _ in range(PAIR_ATTEMPTS):
        rows = _sample_increasing_with_gap(1, h - 2, 6, 2)
        if rows is None:
            return None
        r1, r2, b2, r3, b3, b1 = rows
        if b1 - r1 < 8:
            continue

        c2v = random.randint(2, w - 8)
        c1v = random.randint(c2v + 3, w - 5)
        c3v = random.randint(c1v + 3, w - 2)
        c2a = random.randint(c1v + 2, w - 2)
        c3a = random.randint(1, c1v - 2)
        if random.random() < 0.5:
            c1a = random.randint(1, c1v - 3)
        else:
            c1a = random.randint(c1v + 3, w - 2)

        specs = [
            {"color": int(colors[0]), "a": (r1, c1a), "b": (b1, c1v)},
            {"color": int(colors[1]), "a": (r2, c2a), "b": (b2, c2v)},
            {"color": int(colors[2]), "a": (r3, c3a), "b": (b3, c3v)},
        ]
        if _structure_ok(shape, specs) and len(_metrics(shape, specs)["path_collision_cells"]) >= 2:
            return specs
    return None


def _build_isolated_extra_pair(
    shape: tuple[int, int],
    color: int,
    specs: list[dict[str, object]],
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PAIR_ATTEMPTS):
        ar = random.randint(1, h - 4)
        br = random.randint(ar + 2, h - 2)
        if random.random() < 0.5:
            ac = random.randint(1, w - 4)
            bc = random.randint(ac + 2, w - 2)
        else:
            ac = random.randint(3, w - 2)
            bc = random.randint(1, ac - 2)

        spec = {"color": int(color), "a": (ar, ac), "b": (br, bc)}
        combined = specs + [spec]
        if not _structure_ok(shape, combined):
            continue

        info = _metrics(shape, combined)
        if len(info["collision_cells"]) > 8:
            continue
        if len(info["single_stamp_visible"][int(color)]) < 2:
            continue
        if len(info["single_path_only"][int(color)]) < 2:
            continue
        if sum(1 for by_color in info["claims"].values() if int(color) in by_color and len(by_color) >= 2) > 1:
            continue
        return spec
    return None


def _build_stamp_collision_pair(
    shape: tuple[int, int],
    color: int,
    specs: list[dict[str, object]],
) -> dict[str, object] | None:
    h, w = shape
    base_info = _metrics(shape, specs)
    trunk_color = int(specs[0]["color"])
    targets = list(base_info["single_path_only"][trunk_color])
    random.shuffle(targets)

    marker_cells = _marker_cells(specs)
    for tr, tc in targets:
        if not (2 <= tc < w - 2):
            continue
        for dc in random.sample((-1, 1), 2):
            elbow_c = tc + dc
            if (tr, elbow_c) in marker_cells or not (1 <= elbow_c < w - 1):
                continue

            if dc > 0:
                starts = list(range(elbow_c + 2, w - 1))
            else:
                starts = list(range(1, elbow_c - 1))
            bottoms = list(range(tr + 2, h - 1))
            random.shuffle(starts)
            random.shuffle(bottoms)

            for ac in starts[:10]:
                for br in bottoms[:10]:
                    spec = {"color": int(color), "a": (tr, ac), "b": (br, elbow_c)}
                    combined = specs + [spec]
                    if not _structure_ok(shape, combined):
                        continue

                    info = _metrics(shape, combined)
                    claim = info["claims"].get((tr, tc), {})
                    if len(info["collision_cells"]) > 10:
                        continue
                    if int(color) not in claim or len(claim) < 2:
                        continue
                    if not (claim[int(color)] & ROLE_STAMP):
                        continue
                    if int(info["output"][tr, tc]) == int(info["path_only_output"][tr, tc]):
                        continue
                    if len(info["single_stamp_visible"][int(color)]) < 2:
                        continue
                    if len(info["single_path_only"][int(color)]) < 2:
                        continue
                    return spec
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    include_stamp_collision = random.random() < 0.65
    n_pairs = 4 if include_stamp_collision else random.choice((3, 4))
    colors = random.sample(MARKER_COLORS, n_pairs)

    specs = _build_base_specs(shape, colors[:3])
    if specs is None:
        return None

    if n_pairs == 4:
        if include_stamp_collision:
            extra = _build_stamp_collision_pair(shape, colors[3], specs)
        else:
            extra = _build_isolated_extra_pair(shape, colors[3], specs)
        if extra is None:
            return None
        specs = specs + [extra]

    if not _semantic_ok(shape, specs, require_stamp_collision=include_stamp_collision):
        return None

    return _make_input(shape, specs)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-3 input")
