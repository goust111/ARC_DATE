from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
PALETTE = (1, 3, 4, 7)
GEN_ATTEMPTS = 400


def _chebyshev_adjacent(p: tuple[int, int], q: tuple[int, int]) -> bool:
    return max(abs(int(p[0]) - int(q[0])), abs(int(p[1]) - int(q[1]))) <= 1


def _normalize_pair(color: int, p0: tuple[int, int], p1: tuple[int, int]) -> dict:
    a, b = sorted((tuple(map(int, p0)), tuple(map(int, p1))), key=lambda p: (p[0], p[1]))
    return {"color": int(color), "a": a, "b": b}


def _path_cells(pair: dict, *, corner_mode: str = "target_col") -> list[tuple[int, int]]:
    ar, ac = pair["a"]
    br, bc = pair["b"]
    if ar == br or ac == bc:
        raise ValueError("each endpoint pair must differ in both row and column")

    cells: list[tuple[int, int]] = []
    if corner_mode == "target_col":
        step_c = 1 if bc > ac else -1
        for c in range(ac, bc + step_c, step_c):
            cells.append((ar, c))
        for r in range(ar + 1, br + 1):
            cells.append((r, bc))
        return cells

    if corner_mode == "target_row":
        for r in range(ar, br + 1):
            cells.append((r, ac))
        step_c = 1 if bc > ac else -1
        for c in range(ac + step_c, bc + step_c, step_c):
            cells.append((br, c))
        return cells

    raise ValueError(f"unknown corner mode: {corner_mode}")


def _path_info(pair: dict, *, corner_mode: str = "target_col") -> dict:
    cells = _path_cells(pair, corner_mode=corner_mode)
    ar, ac = pair["a"]
    br, bc = pair["b"]
    return {
        "color": int(pair["color"]),
        "a": tuple(map(int, pair["a"])),
        "b": tuple(map(int, pair["b"])),
        "length": int(abs(br - ar) + abs(bc - ac)),
        "cells": cells,
        "cellset": set(cells),
    }


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    pairs: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape == (2, 2), "each non-black color must appear exactly twice"
        p0 = tuple(map(int, coords[0]))
        p1 = tuple(map(int, coords[1]))
        assert p0[0] != p1[0] and p0[1] != p1[1], "same-color endpoints cannot share row/col"
        pairs.append(_normalize_pair(color, p0, p1))
    assert 3 <= len(pairs) <= 4, "expected 3 or 4 endpoint pairs"
    return pairs


def _sorted_paths(pairs: list[dict], *, corner_mode: str = "target_col", order: str = "spec") -> list[dict]:
    paths = [_path_info(pair, corner_mode=corner_mode) for pair in pairs]

    if order == "spec":
        keys = [(path["length"], path["a"][0]) for path in paths]
        assert len(keys) == len(set(keys)), "generator must avoid unresolved sort ties"
        return sorted(paths, key=lambda path: (path["length"], path["a"][0]), reverse=True)
    if order == "length_asc":
        return sorted(paths, key=lambda path: (path["length"], path["a"][0]))
    if order == "a_row_desc":
        return sorted(paths, key=lambda path: (path["a"][0], path["length"]), reverse=True)
    if order == "a_row_asc":
        return sorted(paths, key=lambda path: (path["a"][0], path["length"]))
    if order == "tie_wrong":
        return sorted(paths, key=lambda path: (-path["length"], path["a"][0]))
    raise ValueError(f"unknown order: {order}")


def _render(shape: tuple[int, int], pairs: list[dict], *, corner_mode: str = "target_col", order: str = "spec", preserve_endpoints: bool = True) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    paths = _sorted_paths(pairs, corner_mode=corner_mode, order=order)
    for path in paths:
        for r, c in path["cells"]:
            out[int(r), int(c)] = int(path["color"])
    if preserve_endpoints:
        for pair in pairs:
            out[pair["a"]] = int(pair["color"])
            out[pair["b"]] = int(pair["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each color-matched endpoint pair, then overlay shorter L-paths on top of longer ones.

    Input structure:
    - Black (0) background with 3 or 4 endpoint pairs.
    - Each pair is two isolated pixels of the same color, with colors drawn from blue, green, yellow, and orange.
    - The two endpoints of any pair are in different rows and different columns.

    Rule:
    - For each same-color pair, choose the upper pixel as start A and the lower pixel as end B.
    - Draw an L-shaped path of that color from A across A's row to B's column, then down that column to B, including both endpoints.
    - Compute each path's Manhattan length |A.row-B.row| + |A.col-B.col|.
    - Draw the paths from larger length to smaller length; if two lengths tie, draw the one with larger A.row first.
    - After all path drawing, restore every endpoint pixel to its original color so endpoints are never overwritten.

    Key insight:
    - At intersections the visible color is decided by computed path length, not by spatial left-right order or by color identity: shorter paths sit on top of longer ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    pairs = _extract_pairs(g)
    return _render(g.shape, pairs, corner_mode="target_col", order="spec", preserve_endpoints=True)


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    vals = list(range(int(lo), int(hi) + 1))
    if len(vals) < count:
        return None
    for _ in range(300):
        sample = sorted(random.sample(vals, count))
        if all(sample[i + 1] - sample[i] >= min_gap for i in range(count - 1)):
            return sample
    return None


def _grid_from_pairs(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for pair in pairs:
        grid[pair["a"]] = int(pair["color"])
        grid[pair["b"]] = int(pair["color"])
    return grid


def _pairwise_endpoint_checks(pairs: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    endpoints: list[tuple[int, int]] = []
    for pair in pairs:
        for p in (pair["a"], pair["b"]):
            r, c = p
            if not (1 <= r < h - 1 and 1 <= c < w - 1):
                return False
            endpoints.append(p)
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if _chebyshev_adjacent(endpoints[i], endpoints[j]):
                return False
    return True


def _crossing_map(paths: list[dict]) -> dict[tuple[int, int], list[int]]:
    cell_to_paths: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, path in enumerate(paths):
        for cell in path["cellset"]:
            cell_to_paths[cell].append(idx)
    return {cell: ids for cell, ids in cell_to_paths.items() if len(ids) >= 2}


def _semantic_checks(pairs: list[dict], shape: tuple[int, int], *, want_tie: bool, want_endpoint_overlap: bool) -> bool:
    if not _pairwise_endpoint_checks(pairs, shape):
        return False
    if len({pair["color"] for pair in pairs}) != len(pairs):
        return False

    true_paths = _sorted_paths(pairs, corner_mode="target_col", order="spec")
    keys = [(path["length"], path["a"][0]) for path in true_paths]
    if len(keys) != len(set(keys)):
        return False

    horizontal_dirs = {int(np.sign(path["b"][1] - path["a"][1])) for path in true_paths}
    if horizontal_dirs != {-1, 1}:
        return False

    cell_to_paths = _crossing_map(true_paths)
    endpoint_set = {pair["a"] for pair in pairs} | {pair["b"] for pair in pairs}
    non_endpoint_crossings = [cell for cell in cell_to_paths if cell not in endpoint_set]
    if len(non_endpoint_crossings) < 2:
        return False
    if len(non_endpoint_crossings) > 8:
        return False

    true_out = _render(shape, pairs, corner_mode="target_col", order="spec", preserve_endpoints=True)
    input_grid = _grid_from_pairs(shape, pairs)
    if np.array_equal(true_out, input_grid):
        return False

    shortest_visible = False
    for cell, ids in cell_to_paths.items():
        if cell in endpoint_set:
            continue
        lengths = [true_paths[i]["length"] for i in ids]
        best_len = min(lengths)
        if lengths.count(best_len) == 1:
            top_idx = ids[lengths.index(best_len)]
            if int(true_out[cell]) == int(true_paths[top_idx]["color"]):
                shortest_visible = True
                break
    if not shortest_visible:
        return False

    wrong_length = _render(shape, pairs, corner_mode="target_col", order="length_asc", preserve_endpoints=True)
    wrong_row_desc = _render(shape, pairs, corner_mode="target_col", order="a_row_desc", preserve_endpoints=True)
    wrong_row_asc = _render(shape, pairs, corner_mode="target_col", order="a_row_asc", preserve_endpoints=True)
    wrong_corner = _render(shape, pairs, corner_mode="target_row", order="spec", preserve_endpoints=True)

    diff_count = sum(
        not np.array_equal(true_out, candidate)
        for candidate in (wrong_length, wrong_row_desc, wrong_row_asc, wrong_corner)
    )
    if diff_count < 2:
        return False
    if np.array_equal(true_out, wrong_corner):
        return False

    endpoint_hits = 0
    for idx, path in enumerate(true_paths):
        own = {path["a"], path["b"]}
        for endpoint in endpoint_set - own:
            if endpoint in path["cellset"]:
                endpoint_hits += 1
    if want_endpoint_overlap:
        if endpoint_hits == 0:
            return False
        no_restore = _render(shape, pairs, corner_mode="target_col", order="spec", preserve_endpoints=False)
        if np.array_equal(true_out, no_restore):
            return False
    else:
        if endpoint_hits != 0:
            return False

    if want_tie:
        tie_lengths = Counter(path["length"] for path in true_paths)
        if max(tie_lengths.values()) < 2:
            return False
        wrong_tie = _render(shape, pairs, corner_mode="target_col", order="tie_wrong", preserve_endpoints=True)
        if np.array_equal(true_out, wrong_tie):
            return False
    else:
        if len({path["length"] for path in true_paths}) != len(true_paths):
            return False

    return np.array_equal(main(input_grid), true_out)


def _base_pairs(shape: tuple[int, int], colors: list[int]) -> list[dict] | None:
    h, w = shape
    rows = _sample_spaced_values(1, h - 2, 6, 2)
    cols = _sample_spaced_values(1, w - 2, 5, 2)
    if rows is None or cols is None:
        return None

    y1, y2, y3, y4, y5, y6 = rows
    x1, x2, x3, x4, x5 = cols
    pairs = [
        _normalize_pair(colors[0], (y3, x1), (y6, x4)),
        _normalize_pair(colors[1], (y1, x2), (y4, x3)),
        _normalize_pair(colors[2], (y2, x5), (y5, x2)),
    ]

    if random.random() < 0.5:
        mirrored = []
        for pair in pairs:
            a = (pair["a"][0], w - 1 - pair["a"][1])
            b = (pair["b"][0], w - 1 - pair["b"][1])
            mirrored.append(_normalize_pair(pair["color"], a, b))
        pairs = mirrored
    return pairs


def _find_fourth_pair(shape: tuple[int, int], pairs: list[dict], color: int, *, want_tie: bool, want_endpoint_overlap: bool) -> dict | None:
    h, w = shape
    base_paths = [_path_info(pair, corner_mode="target_col") for pair in pairs]
    endpoint_set = {pair["a"] for pair in pairs} | {pair["b"] for pair in pairs}
    length_targets = [path["length"] for path in base_paths]
    target_cols = [pair["a"][1] for pair in pairs] + [pair["b"][1] for pair in pairs]

    for _ in range(2500):
        r0 = random.randint(1, h - 3)
        c0 = random.randint(1, w - 2)
        if want_endpoint_overlap and random.random() < 0.6:
            endpoint = random.choice([pair["b"] for pair in pairs])
            r0 = endpoint[0]
            c0 = random.choice([c for c in range(1, w - 1) if c != endpoint[1]])
        if r0 >= h - 2:
            continue
        r1 = random.randint(r0 + 1, h - 2)
        if want_tie and random.random() < 0.7:
            target_len = random.choice(length_targets)
            vertical = r1 - r0
            horizontal = target_len - vertical
            if horizontal <= 0:
                continue
            if random.random() < 0.5:
                c1 = c0 + horizontal
            else:
                c1 = c0 - horizontal
        else:
            c1 = random.randint(1, w - 2)
        if c1 == c0 or not (1 <= c1 < w - 1):
            continue

        pair = _normalize_pair(color, (r0, c0), (r1, c1))
        endpoints = (pair["a"], pair["b"])
        if any(_chebyshev_adjacent(p, q) for p in endpoints for q in endpoint_set):
            continue
        if _chebyshev_adjacent(pair["a"], pair["b"]):
            continue

        path = _path_info(pair, corner_mode="target_col")
        if want_tie:
            same_len = [base for base in base_paths if base["length"] == path["length"] and base["a"][0] != path["a"][0]]
            if not same_len:
                continue
            if not any(len((path["cellset"] & base["cellset"]) - endpoint_set) >= 1 for base in same_len):
                continue
        else:
            if path["length"] in length_targets:
                continue

        intersections = [len(path["cellset"] & base["cellset"]) for base in base_paths]
        if max(intersections, default=0) > 2:
            continue
        if sum(count > 0 for count in intersections) == 0:
            continue

        endpoint_hits = [endpoint for endpoint in endpoint_set if endpoint in path["cellset"]]
        if want_endpoint_overlap:
            if len(endpoint_hits) == 0:
                continue
        else:
            if endpoint_hits:
                continue

        if want_endpoint_overlap and pair["a"][1] in target_cols and pair["b"][1] in target_cols and random.random() < 0.5:
            continue
        return pair
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(13, 18)
        w = random.randint(13, 18)
        shape = (h, w)

        branch = random.choices(
            ["plain3", "plain4", "tie4", "endpoint4", "tie_endpoint4"],
            weights=[3, 2, 3, 2, 3],
            k=1,
        )[0]
        want_fourth = branch != "plain3"
        want_tie = branch in {"tie4", "tie_endpoint4"}
        want_endpoint_overlap = branch in {"endpoint4", "tie_endpoint4"}

        colors = random.sample(list(PALETTE), 4 if want_fourth else 3)
        pairs = _base_pairs(shape, colors)
        if pairs is None:
            continue

        if want_fourth:
            fourth = _find_fourth_pair(
                shape,
                pairs,
                colors[3],
                want_tie=want_tie,
                want_endpoint_overlap=want_endpoint_overlap,
            )
            if fourth is None:
                continue
            pairs = pairs + [fourth]

        if not _semantic_checks(pairs, shape, want_tie=want_tie, want_endpoint_overlap=want_endpoint_overlap):
            continue

        return _grid_from_pairs(shape, pairs)

    raise RuntimeError("failed to generate a valid ARC sample")
