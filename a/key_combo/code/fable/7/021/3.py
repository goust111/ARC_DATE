from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("arc_utils_fable_7_021_3", _ARC_UTILS_PATH)
assert _ARC_SPEC is not None and _ARC_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
find_components = _ARC_UTILS.find_components
neighbors4 = _ARC_UTILS.neighbors4

BG = Color.BLACK
PATH_COLOR = Color.GREY
SEPARATOR_COLOR = Color.MAROON
STRIPE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL]
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _on_global_boundary(r: int, c: int, shape: tuple[int, int]) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _stripe_sequence(region: np.ndarray) -> dict | None:
    if region.size == 0:
        return None
    h, w = region.shape
    rows_constant = all(np.all(region[r, :] == region[r, 0]) for r in range(h))
    cols_constant = all(np.all(region[:, c] == region[0, c]) for c in range(w))
    if rows_constant == cols_constant:
        return None

    if rows_constant:
        raw = [int(region[r, 0]) for r in range(h)]
        orientation = "horizontal"
    else:
        raw = [int(region[0, c]) for c in range(w)]
        orientation = "vertical"

    compressed = [raw[0]]
    for color in raw[1:]:
        if color != compressed[-1]:
            compressed.append(color)

    if len(compressed) < 2:
        return None
    if len(set(compressed)) != len(compressed):
        return None
    return {"orientation": orientation, "raw": raw, "compressed": compressed}


def _ordered_path(region: np.ndarray, row_offset: int, full_shape: tuple[int, int]) -> dict | None:
    if region.size == 0:
        return None
    bg = _mode_color(region)
    if bg == PATH_COLOR:
        return None
    if np.any((region != bg) & (region != PATH_COLOR)):
        return None

    comps = find_components(region, bg=bg, colors=[PATH_COLOR], connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != 1:
        return None

    mask = region == PATH_COLOR
    if bbox(mask) is None:
        return None

    degrees: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(mask):
        deg = 0
        for nr, nc in neighbors4(int(r), int(c)):
            if _in_bounds(region.shape, nr, nc) and mask[nr, nc]:
                deg += 1
        degrees[(int(r), int(c))] = deg

    if any(deg > 2 for deg in degrees.values()):
        return None
    endpoints = [coord for coord, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return None

    boundary_cells = []
    for r, c in degrees:
        gr = row_offset + r
        if _on_global_boundary(gr, c, full_shape):
            boundary_cells.append((r, c))
    if len(boundary_cells) != 1 or boundary_cells[0] not in endpoints:
        return None

    start = boundary_cells[0]
    ordered_local = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    while len(ordered_local) < int(mask.sum()):
        cur = ordered_local[-1]
        next_steps = []
        for nr, nc in neighbors4(*cur):
            nxt = (int(nr), int(nc))
            if _in_bounds(region.shape, *nxt) and mask[nxt] and nxt != prev:
                next_steps.append(nxt)
        if len(next_steps) != 1:
            return None
        nxt = next_steps[0]
        if nxt in seen:
            return None
        seen.add(nxt)
        ordered_local.append(nxt)
        prev = cur

    ordered_global = [(row_offset + r, c) for r, c in ordered_local]
    if _on_global_boundary(*ordered_global[-1], full_shape):
        return None
    return {"background": bg, "ordered_global": ordered_global}


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, _w = g.shape
    candidates = []
    for sep_row in range(1, h - 1):
        if not np.all(g[sep_row, :] == g[sep_row, 0]):
            continue
        top = _stripe_sequence(g[:sep_row, :])
        bottom = _stripe_sequence(g[sep_row + 1 :, :])
        top_path = _ordered_path(g[:sep_row, :], 0, g.shape)
        bottom_path = _ordered_path(g[sep_row + 1 :, :], sep_row + 1, g.shape)

        if top is not None and bottom_path is not None and len(top["raw"]) == len(bottom_path["ordered_global"]):
            candidates.append(
                {
                    "separator_row": sep_row,
                    "stripe_side": "top",
                    "stripe_orientation": top["orientation"],
                    "sequence": top["raw"],
                    "compressed": top["compressed"],
                    "background": bottom_path["background"],
                    "path": bottom_path["ordered_global"],
                }
            )
        if bottom is not None and top_path is not None and len(bottom["raw"]) == len(top_path["ordered_global"]):
            candidates.append(
                {
                    "separator_row": sep_row,
                    "stripe_side": "bottom",
                    "stripe_orientation": bottom["orientation"],
                    "sequence": bottom["raw"],
                    "compressed": bottom["compressed"],
                    "background": top_path["background"],
                    "path": top_path["ordered_global"],
                }
            )

    assert len(candidates) == 1, "expected a unique stripe/path split"
    return candidates[0]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Transcribe the full stripe-by-thickness color sequence onto the grey path starting from its border endpoint.

    Input structure:
    - A single full-width separator row splits the grid into a stripe side and a path side.
    - The stripe side is completely filled by either horizontal or vertical uniform color bands with distinct stripe colors and non-uniform thicknesses.
    - The other side is background plus one grey 4-connected one-pixel-wide path with exactly two endpoints, and exactly one endpoint lies on the outer grid border.

    Rule:
    - Detect which side is the stripe block and read it along its stripe axis, preserving repeated colors caused by stripe thickness.
    - Walk the grey path from the endpoint on the grid border and recolor each successive path cell with the next item of that full sequence.
    - Return a same-size grid where every non-path cell, including the stripe block and separator row, is reset to background.

    Key insight:
    - The stripe block is only a one-dimensional sequence with run lengths, and the grey path is the new one-dimensional carrier that must receive that exact uncompressed order.
    """
    info = _analyze_grid(np.asarray(input_grid, dtype=int))
    out = np.full(np.asarray(input_grid).shape, int(info["background"]), dtype=int)
    for (r, c), color in zip(info["path"], info["sequence"]):
        out[r, c] = int(color)
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    for _ in range(200):
        cuts = sorted(random.sample(range(1, total), parts - 1))
        values = [cuts[0], *[b - a for a, b in zip(cuts, cuts[1:])], total - cuts[-1]] if cuts else [total]
        if max(values) > 1:
            return values
    raise RuntimeError("failed to sample a nontrivial partition")


def _stripe_region(height: int, width: int, orientation: str) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    seq_len = width if orientation == "vertical" else height
    max_parts = min(6, seq_len - 1, len(STRIPE_COLORS))
    min_parts = min(3, max_parts)
    n_stripes = random.randint(min_parts, max_parts)
    thicknesses = _random_partition(seq_len, n_stripes)
    colors = random.sample(STRIPE_COLORS, n_stripes)
    region = np.full((height, width), BG, dtype=int)
    raw = []
    cursor = 0
    for thickness, color in zip(thicknesses, colors):
        raw.extend([int(color)] * int(thickness))
        if orientation == "vertical":
            region[:, cursor : cursor + thickness] = int(color)
        else:
            region[cursor : cursor + thickness, :] = int(color)
        cursor += thickness
    return region, thicknesses, colors, raw


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _allowed_start_positions(shape: tuple[int, int], stripe_side: str) -> list[tuple[int, int]]:
    h, w = shape
    starts = []
    if stripe_side == "top":
        starts.extend((h - 1, c) for c in range(1, w - 1))
    else:
        starts.extend((0, c) for c in range(1, w - 1))
    starts.extend((r, 0) for r in range(1, h - 1))
    starts.extend((r, w - 1) for r in range(1, h - 1))
    random.shuffle(starts)
    return starts


def _reachable_free_count(shape: tuple[int, int], start: tuple[int, int], visited: set[tuple[int, int]]) -> int:
    stack = [start]
    seen = {start}
    total = 0
    while stack:
        r, c = stack.pop()
        total += 1
        for nr, nc in neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if not _in_bounds(shape, *nxt) or nxt in seen or nxt in visited:
                continue
            if nxt[0] in (0, shape[0] - 1) or nxt[1] in (0, shape[1] - 1):
                continue
            seen.add(nxt)
            stack.append(nxt)
    return total


def _path_candidates(shape: tuple[int, int], current: tuple[int, int], visited: set[tuple[int, int]], remaining: int) -> list[tuple[int, int]]:
    options = []
    for nr, nc in neighbors4(*current):
        nxt = (int(nr), int(nc))
        if not _in_bounds(shape, *nxt) or nxt in visited:
            continue
        if nxt[0] in (0, shape[0] - 1) or nxt[1] in (0, shape[1] - 1):
            continue

        occupied_neighbors = 0
        for ar, ac in neighbors4(*nxt):
            if _in_bounds(shape, ar, ac) and (int(ar), int(ac)) in visited:
                occupied_neighbors += 1
        if occupied_neighbors != 1:
            continue

        trial_visited = set(visited)
        trial_visited.add(nxt)
        if remaining - 1 > _reachable_free_count(shape, nxt, trial_visited):
            continue
        future_degree = 0
        for ar, ac in neighbors4(*nxt):
            probe = (int(ar), int(ac))
            if not _in_bounds(shape, *probe) or probe in trial_visited:
                continue
            if probe[0] in (0, shape[0] - 1) or probe[1] in (0, shape[1] - 1):
                continue
            future_degree += 1
        options.append((future_degree, random.random(), nxt))
    options.sort(reverse=True)
    return [item[2] for item in options]


def _dfs_path(shape: tuple[int, int], target_len: int, path: list[tuple[int, int]], visited: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return path[:] if path[-1][0] not in (0, shape[0] - 1) and path[-1][1] not in (0, shape[1] - 1) else None

    remaining = target_len - len(path)
    for nxt in _path_candidates(shape, path[-1], visited, remaining):
        path.append(nxt)
        visited.add(nxt)
        solved = _dfs_path(shape, target_len, path, visited)
        if solved is not None:
            return solved
        visited.remove(nxt)
        path.pop()
    return None


def _generate_path(shape: tuple[int, int], length: int, stripe_side: str, require_turn: bool) -> list[tuple[int, int]] | None:
    h, w = shape
    if h < 5 or w < 5 or length < 2:
        return None
    if 1 + (h - 2) * (w - 2) < length:
        return None

    for start in _allowed_start_positions(shape, stripe_side):
        inside_neighbors = []
        for nr, nc in neighbors4(*start):
            nxt = (int(nr), int(nc))
            if _in_bounds(shape, *nxt) and nxt[0] not in (0, h - 1) and nxt[1] not in (0, w - 1):
                inside_neighbors.append(nxt)
        random.shuffle(inside_neighbors)
        for first in inside_neighbors:
            path = [start, first]
            visited = {start, first}
            solved = _dfs_path(shape, length, path, visited)
            if solved is None:
                continue
            if require_turn and _turn_count(solved) == 0:
                continue
            return solved
    return None


def _render_case(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    stripe_region = meta["stripe_region"]
    path_region = np.full(meta["path_shape"], BG, dtype=int)
    for r, c in meta["path_local"]:
        path_region[r, c] = PATH_COLOR

    separator = np.full((1, meta["width"]), SEPARATOR_COLOR, dtype=int)
    if meta["stripe_side"] == "top":
        grid = np.vstack([stripe_region, separator, path_region])
        path_row_offset = stripe_region.shape[0] + 1
    else:
        grid = np.vstack([path_region, separator, stripe_region])
        path_row_offset = 0

    expected = np.full_like(grid, BG)
    global_path = [(path_row_offset + r, c) for r, c in meta["path_local"]]
    for (r, c), color in zip(global_path, meta["raw_sequence"]):
        expected[r, c] = int(color)
    meta["path_global"] = global_path
    return grid, expected


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    orientation = random.choice(["vertical", "horizontal"])
    stripe_side = random.choice(["top", "bottom"])

    if orientation == "vertical":
        width = random.randint(8, 18)
        stripe_height = random.randint(4, 10)
        path_height = random.randint(5, min(12, 29 - stripe_height))
    else:
        stripe_height = random.randint(6, 14)
        width = random.randint(8, 18)
        path_height = random.randint(5, min(12, 29 - stripe_height))

    stripe_region, thicknesses, stripe_colors, raw = _stripe_region(stripe_height, width, orientation)
    require_turn = random.random() < 0.8
    path_local = _generate_path((path_height, width), len(raw), stripe_side, require_turn=require_turn)
    if path_local is None:
        return None

    meta = {
        "orientation": orientation,
        "stripe_side": stripe_side,
        "width": width,
        "stripe_height": stripe_height,
        "path_shape": (path_height, width),
        "stripe_region": stripe_region,
        "thicknesses": thicknesses,
        "stripe_colors": stripe_colors,
        "raw_sequence": raw,
        "path_local": path_local,
    }
    grid, expected = _render_case(meta)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if len(raw) == len(meta["thicknesses"]):
        return None
    if raw == raw[::-1]:
        return None

    try:
        parsed = _analyze_grid(grid)
    except AssertionError:
        return None
    if parsed["stripe_orientation"] != orientation or parsed["stripe_side"] != stripe_side:
        return None
    if parsed["sequence"] != raw or parsed["path"] != meta["path_global"]:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for fable/7/021/3")


def _validate_generated_case(grid: np.ndarray, out: np.ndarray) -> dict:
    parsed = _analyze_grid(grid)
    assert isinstance(grid, np.ndarray) and isinstance(out, np.ndarray)
    assert grid.ndim == 2 and out.ndim == 2
    assert grid.shape == out.shape
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    assert len(parsed["sequence"]) == len(parsed["path"])
    assert len(parsed["compressed"]) < len(parsed["sequence"]), "generator must keep thickness meaningful"
    assert parsed["sequence"] != parsed["sequence"][::-1], "border endpoint must matter"
    for (r, c), color in zip(parsed["path"], parsed["sequence"]):
        assert out[r, c] == color
    bg = parsed["background"]
    painted = np.zeros_like(out, dtype=bool)
    for r, c in parsed["path"]:
        painted[r, c] = True
    assert np.all(out[~painted] == bg)
    return parsed
