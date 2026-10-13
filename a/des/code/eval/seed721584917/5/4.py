from __future__ import annotations

import random
from collections import Counter
from itertools import permutations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARM_DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
CORNERS = ("tl", "tr", "bl", "br")


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _edge_run(values: np.ndarray, color: int, reverse: bool = False) -> int:
    seq = values[::-1] if reverse else values
    count = 0
    for value in seq:
        if int(value) != int(color):
            break
        count += 1
    return int(count)


def _legend_candidate(grid: np.ndarray, bg: int, corner: str):
    h, w = grid.shape
    if corner == "tl":
        border = int(grid[0, 0])
        width = _edge_run(grid[0, :], border)
        height = _edge_run(grid[:, 0], border)
        r0, c0 = 0, 0
    elif corner == "tr":
        border = int(grid[0, w - 1])
        width = _edge_run(grid[0, :], border, reverse=True)
        height = _edge_run(grid[:, w - 1], border)
        r0, c0 = 0, w - width
    elif corner == "bl":
        border = int(grid[h - 1, 0])
        width = _edge_run(grid[h - 1, :], border)
        height = _edge_run(grid[:, 0], border, reverse=True)
        r0, c0 = h - height, 0
    else:
        border = int(grid[h - 1, w - 1])
        width = _edge_run(grid[h - 1, :], border, reverse=True)
        height = _edge_run(grid[:, w - 1], border, reverse=True)
        r0, c0 = h - height, w - width

    if border == int(bg) or width < 4 or height < 4:
        return None
    crop = np.asarray(grid[r0 : r0 + height, c0 : c0 + width], dtype=int)
    groups = _parse_legend_crop(crop, bg, border)
    if groups is None:
        return None
    return {
        "border": int(border),
        "crop": crop,
        "groups": groups,
        "rect": (int(r0), int(c0), int(height), int(width)),
    }


def _parse_legend_crop(crop: np.ndarray, bg: int, border: int) -> list[dict[int, int]] | None:
    h, w = crop.shape
    if w != 4 or h < 4:
        return None
    if not np.all(crop[0, :] == border) or not np.all(crop[-1, :] == border):
        return None
    if not np.all(crop[:, 0] == border) or not np.all(crop[:, -1] == border):
        return None

    groups: list[dict[int, int]] = []
    current: dict[int, int] = {}
    for r in range(1, h - 1):
        row = crop[r, :]
        if np.all(row == border):
            if not current:
                return None
            groups.append(current)
            current = {}
            continue
        if int(row[0]) != int(border) or int(row[-1]) != int(border):
            return None
        src = int(row[1])
        dst = int(row[2])
        if src in {int(bg), int(border)} or dst in {int(bg), int(border)}:
            return None
        if src in current:
            return None
        current[src] = dst

    if not current:
        return None
    groups.append(current)
    return groups


def _find_legend(grid: np.ndarray, bg: int):
    candidates = []
    for corner in CORNERS:
        cand = _legend_candidate(grid, bg, corner)
        if cand is not None:
            candidates.append(cand)
    assert len(candidates) == 1, "expected exactly one corner legend"

    chosen = candidates[0]
    r0, c0, height, width = chosen["rect"]
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r0 + height, c0 : c0 + width] = True
    mappings = {idx + 1: group for idx, group in enumerate(chosen["groups"])}
    return legend_mask, mappings


def _color_components(grid: np.ndarray, bg: int, forbidden: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if visited[r0, c0] or forbidden[r0, c0] or color == int(bg):
                continue
            stack = [(int(r0), int(c0))]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if visited[rr, cc] or forbidden[rr, cc]:
                        continue
                    if int(grid[rr, cc]) != int(color):
                        continue
                    visited[rr, cc] = True
                    stack.append((int(rr), int(cc)))
            comps.append({"color": int(color), "coords": np.array(coords, dtype=int)})
    return comps


def _tower_from_component(coords: np.ndarray):
    if int(coords.shape[0]) < 5:
        return None

    rows, row_counts = np.unique(coords[:, 0], return_counts=True)
    cols, col_counts = np.unique(coords[:, 1], return_counts=True)
    if int(row_counts.max()) < 3 or int(col_counts.max()) < 3:
        return None
    if int(np.count_nonzero(row_counts == row_counts.max())) != 1:
        return None
    if int(np.count_nonzero(col_counts == col_counts.max())) != 1:
        return None

    center_r = int(rows[int(np.argmax(row_counts))])
    center_c = int(cols[int(np.argmax(col_counts))])
    rmin = int(coords[:, 0].min())
    rmax = int(coords[:, 0].max())
    cmin = int(coords[:, 1].min())
    cmax = int(coords[:, 1].max())

    if not (rmin < center_r < rmax and cmin < center_c < cmax):
        return None

    expected = {
        (int(r), int(center_c)) for r in range(rmin, rmax + 1)
    } | {
        (int(center_r), int(c)) for c in range(cmin, cmax + 1)
    }
    if expected != set(map(tuple, coords.tolist())):
        return None

    lengths = {
        "N": int(center_r - rmin),
        "S": int(rmax - center_r),
        "W": int(center_c - cmin),
        "E": int(cmax - center_c),
    }
    return {
        "center": (int(center_r), int(center_c)),
        "bbox": (int(rmin), int(rmax), int(cmin), int(cmax)),
        "lengths": lengths,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each tower's satellites using the legend group selected by arm-length spread.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    legend_mask, mappings = _find_legend(g, bg)
    components = _color_components(g, bg, legend_mask)

    towers = []
    for comp in components:
        tower = _tower_from_component(comp["coords"])
        if tower is None:
            assert int(comp["coords"].shape[0]) == 1, "non-legend foreground must be towers or single satellites"
            continue
        tower["color"] = int(comp["color"])
        towers.append(tower)
    assert towers, "expected at least one signal tower"

    h, w = g.shape
    out = g.copy()
    for tower in towers:
        lengths = tower["lengths"]
        diff = max(lengths.values()) - min(lengths.values())
        assert diff in mappings, "missing legend mapping group"
        mapping = mappings[diff]

        center_r, center_c = tower["center"]
        rmin, rmax, cmin, cmax = tower["bbox"]
        satellites = {
            "N": (rmin - 1, center_c),
            "S": (rmax + 1, center_c),
            "W": (center_r, cmin - 1),
            "E": (center_r, cmax + 1),
        }
        for rr, cc in satellites.values():
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if legend_mask[rr, cc]:
                continue
            color = int(g[rr, cc])
            if color in {int(bg), int(tower["color"])}:
                continue
            assert color in mapping, "satellite color missing from selected mapping table"
            out[rr, cc] = int(mapping[color])

    return out


# --------------- generator ---------------

BG = 0
GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
MAX_GROUP_CHOICES = (2, 3)
TOWER_COUNT_MAX = 4
SOURCE_COLOR_COUNT = 4
MIN_ARM = 1
MAX_MIN_ARM = 3
PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _derangements(colors: list[int]) -> list[tuple[int, ...]]:
    out = []
    for perm in permutations(colors):
        if all(int(a) != int(b) for a, b in zip(colors, perm)):
            out.append(tuple(int(x) for x in perm))
    return out


def _sample_lengths(diff: int) -> dict[str, int]:
    lo = random.randint(MIN_ARM, MAX_MIN_ARM)
    hi = lo + int(diff)
    arms = [random.randint(lo, hi) for _ in range(4)]
    lo_idx, hi_idx = random.sample(range(4), 2)
    arms[lo_idx] = lo
    arms[hi_idx] = hi
    random.shuffle(arms)
    return {"N": int(arms[0]), "S": int(arms[1]), "W": int(arms[2]), "E": int(arms[3])}


def _tower_cells(center: tuple[int, int], lengths: dict[str, int]) -> list[tuple[int, int]]:
    r0, c0 = center
    cells = {(int(r0), int(c0))}
    for name, (dr, dc) in ARM_DIRS.items():
        for step in range(1, int(lengths[name]) + 1):
            cells.add((int(r0 + dr * step), int(c0 + dc * step)))
    return sorted(cells)


def _satellite_slots(center: tuple[int, int], lengths: dict[str, int]) -> dict[str, tuple[int, int]]:
    r0, c0 = center
    slots = {}
    for name, (dr, dc) in ARM_DIRS.items():
        step = int(lengths[name]) + 1
        slots[name] = (int(r0 + dr * step), int(c0 + dc * step))
    return slots


def _mark_with_margin(mask: np.ndarray, points: list[tuple[int, int]] | tuple[tuple[int, int], ...], radius: int = 1) -> None:
    h, w = mask.shape
    for r, c in points:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True


def _paste_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> tuple[int, int]:
    h, w = grid.shape
    lh, lw = legend.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - lw
    elif corner == "bl":
        r0, c0 = h - lh, 0
    else:
        r0, c0 = h - lh, w - lw
    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    return int(r0), int(c0)


def _build_legend(border: int, source_colors: list[int], group_maps: dict[int, dict[int, int]]) -> np.ndarray:
    rows: list[list[int]] = []
    for group_idx in range(1, len(group_maps) + 1):
        order = source_colors[:]
        random.shuffle(order)
        mapping = group_maps[group_idx]
        for src in order:
            rows.append([int(border), int(src), int(mapping[src]), int(border)])
        if group_idx < len(group_maps):
            rows.append([int(border)] * 4)

    legend = np.full((len(rows) + 2, 4), int(border), dtype=int)
    for idx, row in enumerate(rows, start=1):
        legend[idx, :] = np.array(row, dtype=int)
    return legend


def _try_place_tower(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    tower_color: int,
    source_colors: list[int],
    mapping: dict[int, int],
    diff: int,
) -> bool:
    h, w = inp.shape
    lengths = _sample_lengths(diff)

    min_r = int(lengths["N"]) + 2
    max_r = h - int(lengths["S"]) - 3
    min_c = int(lengths["W"]) + 2
    max_c = w - int(lengths["E"]) - 3
    if min_r > max_r or min_c > max_c:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        center = (random.randint(min_r, max_r), random.randint(min_c, max_c))
        tower = _tower_cells(center, lengths)
        slots = _satellite_slots(center, lengths)
        reserve = tower + list(slots.values())
        if any(occupied[r, c] for r, c in reserve):
            continue

        for r, c in tower:
            inp[r, c] = int(tower_color)
            out[r, c] = int(tower_color)

        sat_count = random.randint(1, 4)
        chosen_dirs = random.sample(list(slots), sat_count)
        for name in chosen_dirs:
            rr, cc = slots[name]
            src = int(random.choice(source_colors))
            inp[rr, cc] = src
            out[rr, cc] = int(mapping[src])

        _mark_with_margin(occupied, reserve, radius=1)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    border = random.randint(1, 9)
    palette = [color for color in range(1, 10) if color != border]
    source_colors = random.sample(palette, SOURCE_COLOR_COUNT)
    tower_palette = [color for color in palette if color not in source_colors]
    if len(tower_palette) < 2:
        return None

    max_group = random.choice(MAX_GROUP_CHOICES)
    all_derangements = _derangements(source_colors)
    if len(all_derangements) < max_group:
        return None
    perms = random.sample(all_derangements, max_group)
    group_maps = {
        idx + 1: {int(src): int(dst) for src, dst in zip(source_colors, perm)}
        for idx, perm in enumerate(perms)
    }

    legend = _build_legend(border, source_colors, group_maps)
    if legend.shape[0] >= h or legend.shape[1] >= w:
        return None

    corner = random.choice(CORNERS)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    _paste_legend(inp, legend, corner)
    _paste_legend(out, legend, corner)

    legend_mask = inp != BG
    legend_only = legend_mask & (out == inp)
    occupied = np.zeros((h, w), dtype=bool)
    _mark_with_margin(occupied, list(map(tuple, np.argwhere(legend_only))), radius=1)

    tower_count = random.randint(max_group, TOWER_COUNT_MAX)
    if tower_count > len(tower_palette):
        return None
    tower_colors = random.sample(tower_palette, tower_count)

    diffs = list(range(1, max_group + 1))
    while len(diffs) < tower_count:
        diffs.append(random.randint(1, max_group))
    random.shuffle(diffs)

    for tower_color, diff in zip(tower_colors, diffs):
        if not _try_place_tower(inp, out, occupied, tower_color, source_colors, group_maps[diff], diff):
            return None

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
