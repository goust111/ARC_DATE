from __future__ import annotations

import random
from itertools import combinations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _is_exact_tower(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = grid.shape
    if r + 1 >= h or c + 1 >= w:
        return False
    color = int(grid[r, c])
    if color == bg:
        return False
    block = grid[r : r + 2, c : c + 2]
    if not np.all(block == color):
        return False

    rr0, rr1 = max(0, r - 1), min(h, r + 3)
    cc0, cc1 = max(0, c - 1), min(w, c + 3)
    hood = grid[rr0:rr1, cc0:cc1] == color
    hood = hood.copy()
    hood[r - rr0 : r - rr0 + 2, c - cc0 : c - cc0 + 2] = False
    return not bool(hood.any())


def _find_towers(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    towers: list[dict[str, int]] = []
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            if _is_exact_tower(grid, r, c, bg):
                towers.append({"r": int(r), "c": int(c), "color": int(grid[r, c])})

    assert towers, "expected at least one 2x2 signal tower"
    colors = [tower["color"] for tower in towers]
    assert len(colors) == len(set(colors)), "signal tower colors must be distinct"
    return towers


def _parse_left_palette(grid: np.ndarray, bg: int, tower_mask: np.ndarray) -> dict[tuple[int, int], int]:
    palette_mask = (grid != bg) & ~tower_mask
    entries: list[tuple[int, int, int]] = []
    used = np.zeros_like(palette_mask, dtype=bool)

    for r in range(grid.shape[0]):
        cols = np.flatnonzero(palette_mask[r])
        if cols.size == 0:
            continue
        if not np.array_equal(cols, np.array([0, 1, 2])):
            raise AssertionError("invalid left-edge palette row")
        a, out, b = map(int, grid[r, 0:3])
        if bg in (a, out, b):
            raise AssertionError("palette entries must be non-background")
        entries.append((a, out, b))
        used[r, 0:3] = True

    if not entries or not np.array_equal(used, palette_mask):
        raise AssertionError("left-edge palette parse failed")
    return _entries_to_mix_map(entries)


def _parse_top_palette(grid: np.ndarray, bg: int, tower_mask: np.ndarray) -> dict[tuple[int, int], int]:
    palette_mask = (grid != bg) & ~tower_mask
    entries: list[tuple[int, int, int]] = []
    used = np.zeros_like(palette_mask, dtype=bool)

    for c in range(grid.shape[1]):
        rows = np.flatnonzero(palette_mask[:, c])
        if rows.size == 0:
            continue
        if not np.array_equal(rows, np.array([0, 1, 2])):
            raise AssertionError("invalid top-edge palette column")
        a, out, b = map(int, grid[0:3, c])
        if bg in (a, out, b):
            raise AssertionError("palette entries must be non-background")
        entries.append((a, out, b))
        used[0:3, c] = True

    if not entries or not np.array_equal(used, palette_mask):
        raise AssertionError("top-edge palette parse failed")
    return _entries_to_mix_map(entries)


def _entries_to_mix_map(entries: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    mix_map: dict[tuple[int, int], int] = {}
    for a, out, b in entries:
        if a == b:
            raise AssertionError("palette endpoints must differ")
        key = tuple(sorted((int(a), int(b))))
        if key in mix_map and mix_map[key] != int(out):
            raise AssertionError("palette pair must map to a single result color")
        mix_map[key] = int(out)
    return mix_map


def _find_mix_map(grid: np.ndarray, bg: int, towers: list[dict[str, int]]) -> dict[tuple[int, int], int]:
    tower_mask = np.zeros_like(grid, dtype=bool)
    for tower in towers:
        tower_mask[tower["r"] : tower["r"] + 2, tower["c"] : tower["c"] + 2] = True

    candidates: list[dict[tuple[int, int], int]] = []
    for parser in (_parse_left_palette, _parse_top_palette):
        try:
            candidates.append(parser(grid, bg, tower_mask))
        except AssertionError:
            continue

    assert len(candidates) == 1, "expected exactly one valid edge palette"
    return candidates[0]


def _ordered_pairs(towers: list[dict[str, int]]) -> list[tuple[dict[str, int], dict[str, int]]]:
    ordered = sorted(towers, key=lambda tower: (tower["c"], tower["r"]))
    pairs: list[tuple[dict[str, int], dict[str, int]]] = []
    for i, first in enumerate(ordered):
        for second in ordered[i + 1 :]:
            if first["r"] == second["r"] or first["c"] == second["c"]:
                pairs.append((first, second))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mix aligned 2x2 signal towers using an edge palette and stamp the result at each pair midpoint.

    Input structure:
    - The dominant color is background.
    - Each signal tower is a monochrome 2x2 block with a unique color.
    - The palette lives on the left edge as 1x3 rows or on the top edge as 3x1 columns.

    Rule:
    - For every tower pair sharing a row or a column, look up the mixed color from the palette.
    - The mixed 2x2 block is drawn at the floored midpoint between the two tower centers.
    - Pairs are processed left-to-right, then top-to-bottom by tower anchors, so later stamps overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    towers = _find_towers(g, bg)
    mix_map = _find_mix_map(g, bg, towers)

    out = g.copy()
    for first, second in _ordered_pairs(towers):
        key = tuple(sorted((first["color"], second["color"])))
        assert key in mix_map, "missing palette entry for aligned tower pair"
        mid_r = (first["r"] + second["r"] + 1) // 2
        mid_c = (first["c"] + second["c"] + 1) // 2
        assert mid_r + 1 < g.shape[0] and mid_c + 1 < g.shape[1], "midpoint block must fit"
        out[mid_r : mid_r + 2, mid_c : mid_c + 2] = mix_map[key]
    return out


ALL_COLORS = tuple(range(10))
SIZE_MIN = 14
SIZE_MAX = 26
TOWER_GAP = 3
TABLE_GAP = 2
SCENE_MARGIN = 1
RESERVED_BAND = 4
MAX_ATTEMPTS = 400


def _sample_spaced_values(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picked = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _scene_bounds(shape: tuple[int, int], orientation: str) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = RESERVED_BAND + 1 if orientation == "top" else SCENE_MARGIN
    c0 = RESERVED_BAND + 1 if orientation == "left" else SCENE_MARGIN
    r1 = h - 2 - SCENE_MARGIN
    c1 = w - 2 - SCENE_MARGIN
    return r0, r1, c0, c1


def _sample_cross_positions(shape: tuple[int, int], orientation: str) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = _scene_bounds(shape, orientation)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    for _ in range(200):
        mid_r = random.randint(r0 + 2, r1 - 2)
        mid_c = random.randint(c0 + 2, c1 - 2)
        max_dr = min(mid_r - r0, r1 - mid_r)
        max_dc = min(mid_c - c0, c1 - mid_c)
        if max_dr < 2 or max_dc < 2:
            continue
        dr = random.randint(2, max_dr)
        dc = random.randint(2, max_dc)
        return [
            (mid_r, mid_c - dc),
            (mid_r, mid_c + dc),
            (mid_r - dr, mid_c),
            (mid_r + dr, mid_c),
        ]
    return None


def _sample_grid_positions(shape: tuple[int, int], orientation: str, tower_count: int) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = _scene_bounds(shape, orientation)
    rows = _sample_spaced_values(r0, r1, 2, TOWER_GAP)
    cols = _sample_spaced_values(c0, c1, 2, TOWER_GAP)
    if rows is None or cols is None:
        return None

    positions = [(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])]
    if tower_count == 3:
        missing = random.randrange(4)
        positions = [pos for i, pos in enumerate(positions) if i != missing]
    random.shuffle(positions)
    return positions


def _pair_midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return ((a[0] + b[0] + 1) // 2, (a[1] + b[1] + 1) // 2)


def _aligned_pairs(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    ordered = sorted(range(len(positions)), key=lambda i: (positions[i][1], positions[i][0]))
    for i, first_idx in enumerate(ordered):
        for second_idx in ordered[i + 1 :]:
            first = positions[first_idx]
            second = positions[second_idx]
            if first[0] == second[0] or first[1] == second[1]:
                pairs.append((first_idx, second_idx))
    return pairs


def _has_overlap_branch(positions: list[tuple[int, int]]) -> bool:
    mids = [_pair_midpoint(positions[i], positions[j]) for i, j in _aligned_pairs(positions)]
    return len(mids) != len(set(mids))


def _choose_mix_map(
    tower_colors: list[int],
    aligned_color_pairs: list[tuple[int, int]],
    force_distinct_overlap: bool,
    bg: int,
) -> dict[tuple[int, int], int]:
    available = [color for color in ALL_COLORS if color != bg]
    mix_map: dict[tuple[int, int], int] = {}
    overlap_results: list[int] = []

    for pair in combinations(tower_colors, 2):
        a, b = sorted(map(int, pair))
        options = [color for color in available if color not in (a, b)]
        mix_map[(a, b)] = random.choice(options)

    if force_distinct_overlap and len(aligned_color_pairs) >= 2:
        special = aligned_color_pairs[:2]
        used: list[int] = []
        for key in special:
            a, b = key
            options = [color for color in available if color not in (a, b) and color not in used]
            if not options:
                options = [color for color in available if color not in (a, b)]
            chosen = random.choice(options)
            mix_map[key] = chosen
            used.append(chosen)
            overlap_results.append(chosen)
        if len(overlap_results) == 2 and overlap_results[0] == overlap_results[1]:
            raise AssertionError("overlap results must be visually distinct")

    return mix_map


def _sample_palette_slots(length: int, count: int) -> list[int] | None:
    return _sample_spaced_values(0, length - 1, count, TABLE_GAP)


def _place_palette(
    grid: np.ndarray,
    orientation: str,
    tower_colors: list[int],
    mix_map: dict[tuple[int, int], int],
) -> None:
    entries = []
    for a, b in combinations(tower_colors, 2):
        key = tuple(sorted((int(a), int(b))))
        if random.random() < 0.5:
            entries.append((int(a), mix_map[key], int(b)))
        else:
            entries.append((int(b), mix_map[key], int(a)))
    random.shuffle(entries)

    if orientation == "left":
        rows = _sample_palette_slots(grid.shape[0], len(entries))
        assert rows is not None, "palette rows must fit"
        for row, (a, out, b) in zip(rows, entries):
            grid[row, 0:3] = np.array([a, out, b], dtype=int)
    else:
        cols = _sample_palette_slots(grid.shape[1], len(entries))
        assert cols is not None, "palette columns must fit"
        for col, (a, out, b) in zip(cols, entries):
            grid[0:3, col] = np.array([a, out, b], dtype=int)


def _render_input(
    shape: tuple[int, int],
    orientation: str,
    positions: list[tuple[int, int]],
    tower_colors: list[int],
    mix_map: dict[tuple[int, int], int],
    bg: int,
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    _place_palette(grid, orientation, tower_colors, mix_map)
    for (r, c), color in zip(positions, tower_colors):
        grid[r : r + 2, c : c + 2] = int(color)
    return grid


def _valid_generation(grid: np.ndarray, expect_overlap: bool) -> bool:
    try:
        bg = _most_common_color(grid)
        towers = _find_towers(grid, bg)
        positions = [(tower["r"], tower["c"]) for tower in towers]
        overlap_present = _has_overlap_branch(positions)
        if overlap_present != expect_overlap:
            return False
        mix_map = _find_mix_map(grid, bg, towers)
        midpoint_colors: dict[tuple[int, int], list[int]] = {}
        for first, second in _ordered_pairs(towers):
            key = tuple(sorted((first["color"], second["color"])))
            midpoint_colors.setdefault(
                _pair_midpoint((first["r"], first["c"]), (second["r"], second["c"])),
                [],
            ).append(mix_map[key])
        if expect_overlap:
            if not any(len(colors) > 1 and len(set(colors)) > 1 for colors in midpoint_colors.values()):
                return False
        _ = main(grid)
    except Exception:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice(ALL_COLORS)
    orientation = random.choice(("left", "top"))
    layout = random.choice(("cross", "grid"))
    tower_count = 4 if layout == "cross" else random.choice((3, 4))
    tower_colors = random.sample([color for color in ALL_COLORS if color != bg], tower_count)
    shape = (random.randint(SIZE_MIN, SIZE_MAX), random.randint(SIZE_MIN, SIZE_MAX))

    if layout == "cross":
        positions = _sample_cross_positions(shape, orientation)
        expect_overlap = True
    else:
        positions = _sample_grid_positions(shape, orientation, tower_count)
        expect_overlap = False
    if positions is None:
        return None

    aligned_pairs = _aligned_pairs(positions)
    if not aligned_pairs:
        return None

    aligned_color_pairs = [
        tuple(sorted((tower_colors[i], tower_colors[j])))
        for i, j in aligned_pairs
    ]
    mix_map = _choose_mix_map(tower_colors, aligned_color_pairs, expect_overlap, bg)
    grid = _render_input(shape, orientation, positions, tower_colors, mix_map, bg)
    if not _valid_generation(grid, expect_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
