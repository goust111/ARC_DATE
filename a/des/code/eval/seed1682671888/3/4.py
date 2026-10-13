import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
GREY = 5
YELLOW = 4
SEED_COLORS = [1, 2, 3, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
GEN_RETRIES = 400
SCENARIOS = ("no_cross", "unequal", "equal_tiebreak", "mixed")


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _grey_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if bool(mask[rr, cc])]


def _trace_path(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    prev = None
    cur = start

    while True:
        neigh = _grey_neighbors(mask, cur[0], cur[1])
        deg = len(neigh)
        if prev is None:
            assert deg <= 1, "seeded endpoint must be a grey endpoint"
            if deg == 0:
                break
            nxt = neigh[0]
        else:
            if deg == 1:
                assert neigh[0] == prev, "path endpoint must point back to previous cell"
                break
            if deg == 2:
                assert prev in neigh, "non-crossing path step must include the previous cell"
                nxt = neigh[0] if neigh[1] == prev else neigh[1]
            elif deg == 4:
                dr = cur[0] - prev[0]
                dc = cur[1] - prev[1]
                nxt = (cur[0] + dr, cur[1] + dc)
                assert nxt in neigh, "crossing traversal must continue straight"
            else:
                raise AssertionError("grey degrees must be 0, 1, 2, or 4")
        path.append(nxt)
        prev, cur = cur, nxt

    return path


def _extract_seeded_paths(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    grey_mask = g == GREY
    grey_cells = [tuple(x) for x in np.argwhere(grey_mask)]
    assert grey_cells, "grid must contain grey path pixels"

    degrees = {cell: len(_grey_neighbors(grey_mask, cell[0], cell[1])) for cell in grey_cells}
    assert all(d in (0, 1, 2, 4) for d in degrees.values()), "invalid grey skeleton degree"
    endpoints = {cell for cell, d in degrees.items() if d <= 1}

    seeds = []
    seen_seeded_endpoints = set()
    for r, c in np.argwhere((g != BG) & (g != GREY)):
        r = int(r)
        c = int(c)
        adj_endpoints = [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if (rr, cc) in endpoints]
        assert len(adj_endpoints) == 1, "each seed must touch exactly one grey endpoint"
        endpoint = adj_endpoints[0]
        assert endpoint not in seen_seeded_endpoints, "each seeded endpoint must be unique"
        seen_seeded_endpoints.add(endpoint)
        seeds.append({"seed": (r, c), "color": int(g[r, c]), "endpoint": endpoint})

    assert seeds, "expected at least one seed"

    paths = []
    owners: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, info in enumerate(sorted(seeds, key=lambda d: (d["seed"][0], d["seed"][1]))):
        cells = _trace_path(grey_mask, info["endpoint"])
        paths.append(
            {
                "index": idx,
                "seed": info["seed"],
                "color": info["color"],
                "endpoint": info["endpoint"],
                "cells": cells,
                "total_length": len(cells),
                "distance": {cell: dist for dist, cell in enumerate(cells, start=1)},
            }
        )
        for cell in cells:
            owners[cell].append((idx, len(paths[-1]["cells"])))

    assert set(owners) == set(grey_cells), "all grey pixels must belong to seeded paths"
    for cell, traced_by in owners.items():
        assert len(traced_by) <= 2, "a crossing pixel can belong to at most two paths"
        if len(traced_by) == 2:
            assert degrees[cell] == 4, "shared pixels must be true crossings"

    return paths, owners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each seeded grey path and mark true crossings in yellow.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    paths, owners = _extract_seeded_paths(g)
    path_by_index = {path["index"]: path for path in paths}

    for cell, traced_by in owners.items():
        best = min(
            (
                path_by_index[path_idx]["distance"][cell],
                path_by_index[path_idx]["total_length"],
                path_idx,
            )
            for path_idx, _ in traced_by
        )
        winner = path_by_index[best[2]]
        out[cell] = winner["color"]

    for cell, traced_by in owners.items():
        if len(traced_by) >= 2:
            out[cell] = YELLOW

    return out


def _hline(row: int, c0: int, c1: int, seed_color: int) -> dict:
    return {"kind": "h", "row": row, "c0": c0, "c1": c1, "seed_side": "left", "color": seed_color}


def _vline(col: int, r0: int, r1: int, seed_color: int) -> dict:
    return {"kind": "v", "col": col, "r0": r0, "r1": r1, "seed_side": "top", "color": seed_color}


def _line_cells(line: dict) -> list[tuple[int, int]]:
    if line["kind"] == "h":
        return [(line["row"], c) for c in range(line["c0"], line["c1"] + 1)]
    return [(r, line["col"]) for r in range(line["r0"], line["r1"] + 1)]


def _line_seed(line: dict) -> tuple[int, int]:
    if line["kind"] == "h":
        return (line["row"], line["c0"] - 1)
    return (line["r0"] - 1, line["col"])


def _owners_from_lines(lines: list[dict]) -> dict[tuple[int, int], list[int]]:
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, line in enumerate(lines):
        for cell in _line_cells(line):
            owners[cell].append(idx)
    return owners


def _render_lines(h: int, w: int, lines: list[dict]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    owners = _owners_from_lines(lines)

    seed_cells = set()
    for line in lines:
        seed = _line_seed(line)
        if not (0 <= seed[0] < h and 0 <= seed[1] < w):
            return None
        if seed in owners or seed in seed_cells:
            return None
        seed_cells.add(seed)

    grey_mask = np.zeros((h, w), dtype=bool)
    for cell, traced_by in owners.items():
        if len(traced_by) > 2:
            return None
        if len(traced_by) == 2:
            if lines[traced_by[0]]["kind"] == lines[traced_by[1]]["kind"]:
                return None
        grey_mask[cell] = True

    for r, c in owners:
        for rr, cc in _neighbors4(r, c, h, w):
            if not grey_mask[rr, cc]:
                continue
            common = set(owners[(r, c)]) & set(owners[(rr, cc)])
            if not common:
                return None

    for cell in owners:
        deg = len(_grey_neighbors(grey_mask, cell[0], cell[1]))
        if deg not in (1, 2, 4):
            return None

    for cell in owners:
        grid[cell] = GREY
    for line in lines:
        seed = _line_seed(line)
        if sum(1 for rr, cc in _neighbors4(seed[0], seed[1], h, w) if grey_mask[rr, cc]) != 1:
            return None
        grid[seed] = int(line["color"])

    return grid


def _sample_rows_or_cols(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    random.shuffle(vals)
    chosen = []
    for v in vals:
        if all(abs(v - u) >= gap for u in chosen):
            chosen.append(v)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _extra_parallel_line(h: int, w: int, existing: list[dict], orient: str) -> dict | None:
    for _ in range(80):
        color = random.choice(SEED_COLORS)
        if orient == "h":
            row = random.randint(1, h - 2)
            c0 = random.randint(1, w - 5)
            c1 = random.randint(c0 + 3, w - 2)
            line = _hline(row, c0, c1, color)
        else:
            col = random.randint(1, w - 2)
            r0 = random.randint(1, h - 5)
            r1 = random.randint(r0 + 3, h - 2)
            line = _vline(col, r0, r1, color)
        if _render_lines(h, w, existing + [line]) is not None:
            return line
    return None


def _build_no_cross(h: int, w: int) -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    count = random.randint(2, 4)
    lines = []
    if orient == "h":
        rows = _sample_rows_or_cols(1, h - 2, count, 3)
        if rows is None:
            return None
        for row in rows:
            c0 = random.randint(1, w - 5)
            c1 = random.randint(c0 + 3, w - 2)
            lines.append(_hline(row, c0, c1, random.choice(SEED_COLORS)))
    else:
        cols = _sample_rows_or_cols(1, w - 2, count, 3)
        if cols is None:
            return None
        for col in cols:
            r0 = random.randint(1, h - 5)
            r1 = random.randint(r0 + 3, h - 2)
            lines.append(_vline(col, r0, r1, random.choice(SEED_COLORS)))
    return _render_lines(h, w, lines)


def _build_unequal(h: int, w: int) -> np.ndarray | None:
    for _ in range(120):
        row = random.randint(2, h - 3)
        c0 = random.randint(1, w - 7)
        c1 = random.randint(c0 + 4, w - 2)
        col = random.randint(c0 + 1, c1 - 1)
        r0 = random.randint(1, row - 1)
        r1 = random.randint(row + 1, h - 2)
        dh = col - c0 + 1
        dv = row - r0 + 1
        if dh == dv:
            continue
        lines = [
            _hline(row, c0, c1, random.choice(SEED_COLORS)),
            _vline(col, r0, r1, random.choice(SEED_COLORS)),
        ]
        if random.random() < 0.6:
            extra = _extra_parallel_line(h, w, lines, random.choice(["h", "v"]))
            if extra is not None:
                lines.append(extra)
        grid = _render_lines(h, w, lines)
        if grid is not None:
            return grid
    return None


def _build_equal_tiebreak(h: int, w: int) -> np.ndarray | None:
    for _ in range(120):
        d = random.randint(2, min(h, w) // 3 + 1)
        c0 = random.randint(1, w - d - 4)
        r0 = random.randint(1, h - d - 4)
        col = c0 + d - 1
        row = r0 + d - 1
        c1 = random.randint(col + 1, w - 2)
        r1 = random.randint(row + 1, h - 2)
        if (c1 - c0) == (r1 - r0):
            continue
        lines = [
            _hline(row, c0, c1, random.choice(SEED_COLORS)),
            _vline(col, r0, r1, random.choice(SEED_COLORS)),
        ]
        if random.random() < 0.5:
            extra = _extra_parallel_line(h, w, lines, random.choice(["h", "v"]))
            if extra is not None:
                lines.append(extra)
        grid = _render_lines(h, w, lines)
        if grid is not None:
            return grid
    return None


def _build_mixed(h: int, w: int) -> np.ndarray | None:
    for _ in range(160):
        row = random.randint(4, h - 3)
        c0 = random.randint(1, w - 9)
        c1 = random.randint(c0 + 6, w - 2)

        if c1 - c0 < 4:
            continue
        col_a_hi = min(c1 - 3, c0 + max(2, (c1 - c0) // 2))
        if col_a_hi < c0 + 1:
            continue
        col_a = random.randint(c0 + 1, col_a_hi)
        d_a = col_a - c0 + 1
        r0_a = row - d_a + 1
        if r0_a < 1:
            continue
        r1_a = random.randint(row + 1, h - 2)
        if (c1 - c0) == (r1_a - r0_a):
            continue

        if col_a + 3 > c1 - 1:
            continue
        col_b = random.randint(col_a + 3, c1 - 1)
        r0_b = random.randint(1, row - 1)
        r1_b = random.randint(row + 1, h - 2)
        d_b_h = col_b - c0 + 1
        d_b_v = row - r0_b + 1
        if d_b_h == d_b_v:
            continue

        lines = [
            _hline(row, c0, c1, random.choice(SEED_COLORS)),
            _vline(col_a, r0_a, r1_a, random.choice(SEED_COLORS)),
            _vline(col_b, r0_b, r1_b, random.choice(SEED_COLORS)),
        ]
        if random.random() < 0.5:
            extra = _extra_parallel_line(h, w, lines, "h")
            if extra is not None:
                lines.append(extra)
        grid = _render_lines(h, w, lines)
        if grid is not None:
            return grid
    return None


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=int, copy=True)


def _crossing_kinds(grid: np.ndarray) -> Counter:
    paths, owners = _extract_seeded_paths(grid)
    path_by_index = {path["index"]: path for path in paths}
    kinds = Counter()
    for cell, traced_by in owners.items():
        if len(traced_by) != 2:
            continue
        a_idx, _ = traced_by[0]
        b_idx, _ = traced_by[1]
        pa = path_by_index[a_idx]
        pb = path_by_index[b_idx]
        da = pa["distance"][cell]
        db = pb["distance"][cell]
        if da != db:
            kinds["unequal"] += 1
        elif pa["total_length"] != pb["total_length"]:
            kinds["equal_tiebreak"] += 1
        else:
            kinds["perfect_tie"] += 1
    return kinds


def _valid_generated_grid(grid: np.ndarray, scenario: str) -> bool:
    try:
        out = main(grid)
        assert out.shape == grid.shape
        kinds = _crossing_kinds(grid)
        if scenario == "no_cross":
            assert sum(kinds.values()) == 0
        elif scenario == "unequal":
            assert kinds["unequal"] >= 1
        elif scenario == "equal_tiebreak":
            assert kinds["equal_tiebreak"] >= 1
        elif scenario == "mixed":
            assert kinds["unequal"] >= 1 and kinds["equal_tiebreak"] >= 1
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(SCENARIOS)
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    builder = {
        "no_cross": _build_no_cross,
        "unequal": _build_unequal,
        "equal_tiebreak": _build_equal_tiebreak,
        "mixed": _build_mixed,
    }[scenario]

    grid = builder(h, w)
    if grid is None:
        return None

    grid = _random_transform(grid)
    if not _valid_generated_grid(grid, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
