from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(10))
BLACK = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
]

GEN_RETRIES = 500


def _most_common_value(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts
    value, _count = counts.most_common(1)[0]
    return int(value)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _panel_start(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid != BLACK, axis=1))
    assert rows.size > 0, "expected a non-black scene panel below the strip"
    start = int(rows[0])
    assert 0 < start < grid.shape[0], "expected both instruction strip and scene panel"
    assert np.all(grid[start:] != BLACK), "scene panel must contain no black cells"
    return start


def _parse_markers(top: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(top != BLACK)
    assert coords.size > 0, "expected marker pixels in the top strip"
    assert np.all(coords[:, 1] == 0), "marker pixels must sit in column 0"
    rows = [int(r) for r in coords[:, 0]]
    assert len(rows) == len(set(rows)), "expected at most one marker per row"
    markers = [(int(r), int(top[r, 0])) for r in sorted(rows)]
    colors = [color for _row, color in markers]
    assert 1 <= len(markers) <= 3
    assert len(colors) == len(set(colors)), "marker colors must be distinct"
    return markers


def _count_map(scene: np.ndarray, colors: list[int]) -> dict[int, int]:
    return {int(color): len(_components_of_color(scene, int(color))) for color in colors}


def _filtered_scene(scene: np.ndarray, panel_bg: int, keep_colors: set[int]) -> np.ndarray:
    out = scene.copy()
    erase_mask = (scene != int(panel_bg)) & ~np.isin(scene, list(keep_colors))
    out[erase_mask] = int(panel_bg)
    return out


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    panel_start = _panel_start(g)
    top = g[:panel_start]
    scene = g[panel_start:]

    markers = _parse_markers(top)
    marker_colors = [color for _row, color in markers]
    panel_bg = _most_common_value(scene)
    assert panel_bg != BLACK, "scene background must be non-black"
    assert panel_bg not in marker_colors, "marker colors must differ from panel background"

    counts = _count_map(scene, marker_colors)
    filtered = _filtered_scene(scene, panel_bg, set(marker_colors))
    shift = sum(counts.values())
    assert 0 < shift < scene.shape[0], "surviving object count must define a non-trivial row roll"

    info = {
        "panel_start": panel_start,
        "scene": scene,
        "panel_bg": panel_bg,
        "markers": markers,
        "counts": counts,
        "filtered": filtered,
        "shift": shift,
    }
    return np.roll(filtered, shift=shift, axis=0), info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the scene by top-strip marker colors, then roll the surviving scene downward by the surviving-object count.

    Input structure:
    - The grid is split into a black instruction strip on top and a non-black scene panel below.
    - The instruction strip contains 1-3 distinct colored marker pixels, each in column 0 and at most one per row.
    - The scene panel has one uniform non-black background color plus several separated 4-connected foreground objects in multiple colors.

    Rule:
    - For each marker color, count the number of 4-connected objects of that color in the scene panel.
    - Treat those counts as the lengths of same-color bars that would replace the top markers; markers with count 0 disappear.
    - Erase every scene object whose color is not in the marker-color set by repainting it to the panel background.
    - Sum all marker-color object counts to obtain S, then cyclically roll the filtered scene panel downward by S rows.
    - Remove the instruction strip and output only the rolled scene panel.

    Key insight:
    - The top strip is both a color filter and a count encoder: the same selected object counts determine which scene objects survive and how far the surviving scene is cyclically shifted.
    """
    out, _info = _solve(grid)
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _can_place(panel: np.ndarray, panel_bg: int, shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> bool:
    h, w = panel.shape
    cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if any(panel[r, c] != int(panel_bg) for r, c in cells):
        return False
    for r, c in cells:
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            return False
        for dr, dc in DIRS8:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and panel[nr, nc] != int(panel_bg):
                return False
    return True


def _place_object(panel: np.ndarray, panel_bg: int, color: int, shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = panel.shape
    sh, sw = _shape_dims(shape)
    candidates = [
        (r0, c0)
        for r0 in range(1, h - sh)
        for c0 in range(1, w - sw)
        if _can_place(panel, panel_bg, shape, r0, c0)
    ]
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    for dr, dc in shape:
        panel[r0 + dr, c0 + dc] = int(color)
    return True


def _split_positive_sum(total: int, parts: int, cap: int) -> list[int] | None:
    if not (parts <= total <= parts * cap):
        return None
    values = [1] * parts
    remaining = total - parts
    while remaining > 0:
        indices = [i for i, value in enumerate(values) if value < cap]
        if not indices:
            return None
        idx = random.choice(indices)
        values[idx] += 1
        remaining -= 1
    random.shuffle(values)
    return values


def _choose_shift(panel_h: int, present_markers: int) -> int:
    cap = 4 * present_markers
    valid = [s for s in range(present_markers, min(panel_h - 1, cap) + 1)]
    assert valid
    branch = random.choices(
        ["low", "middle", "high"],
        weights=[2, 3, 2],
        k=1,
    )[0]
    if branch == "low":
        pool = valid[: max(1, min(2, len(valid)))]
    elif branch == "high":
        pool = valid[-max(1, min(2, len(valid))):]
    else:
        lo = len(valid) // 3
        hi = max(lo + 1, len(valid) - len(valid) // 3)
        pool = valid[lo:hi] or valid
    return int(random.choice(pool))


def _nontrivial_output_checks(scene: np.ndarray, filtered: np.ndarray, out: np.ndarray, shift: int, marker_count: int) -> bool:
    panel_h = scene.shape[0]
    if np.array_equal(filtered, out):
        return False
    for other in range(panel_h):
        if other != shift and np.array_equal(np.roll(filtered, shift=other, axis=0), out):
            return False
    total_objects = sum(
        len(_components_of_color(scene, color))
        for color in map(int, np.unique(scene))
        if color != _most_common_value(scene)
    )
    if total_objects != shift and np.array_equal(np.roll(filtered, shift=total_objects % panel_h, axis=0), out):
        return False
    kept_pixels = int(np.count_nonzero(filtered != _most_common_value(scene)))
    if kept_pixels % panel_h != shift and np.array_equal(np.roll(filtered, shift=kept_pixels % panel_h, axis=0), out):
        return False
    if marker_count != shift and np.array_equal(np.roll(filtered, shift=marker_count % panel_h, axis=0), out):
        return False
    if np.array_equal(np.roll(scene, shift=shift, axis=0), out):
        return False
    return True


def _make_attempt() -> np.ndarray | None:
    top_h = random.choice([3, 4])
    panel_h = random.randint(8, 12)
    width = random.randint(12, 18)

    marker_total = random.choice([1, 2, 3])
    zero_marker = bool(marker_total >= 2 and random.random() < 0.35)
    present_markers = marker_total - 1 if zero_marker else marker_total
    if present_markers <= 0:
        return None

    actual_color_count = random.randint(max(2, present_markers + 1), 4)
    panel_bg = random.choice(PALETTE[1:])
    available = [c for c in PALETTE[1:] if c != panel_bg]
    actual_colors = random.sample(available, actual_color_count)
    marked_present_colors = random.sample(actual_colors, present_markers)
    unmarked_colors = [color for color in actual_colors if color not in marked_present_colors]
    if not unmarked_colors:
        return None

    marker_colors = marked_present_colors[:]
    if zero_marker:
        absent_choices = [c for c in available if c not in actual_colors]
        if not absent_choices:
            return None
        marker_colors.append(random.choice(absent_choices))
    random.shuffle(marker_colors)

    shift = _choose_shift(panel_h, present_markers)
    selected_counts = _split_positive_sum(shift, present_markers, cap=4)
    if selected_counts is None:
        return None

    unmarked_total = random.randint(1, min(5, panel_h - 1))
    unmarked_parts = _split_positive_sum(unmarked_total, len(unmarked_colors), cap=4)
    if unmarked_parts is None:
        return None

    color_to_count = {color: 0 for color in marker_colors}
    for color, count in zip(marked_present_colors, selected_counts):
        color_to_count[color] = int(count)
    for color, count in zip(unmarked_colors, unmarked_parts):
        color_to_count[color] = int(count)

    panel = np.full((panel_h, width), panel_bg, dtype=int)
    objects: list[int] = []
    for color in actual_colors:
        objects.extend([int(color)] * int(color_to_count[color]))
    random.shuffle(objects)

    for color in objects:
        shape = random.choice(SHAPES)
        if not _place_object(panel, panel_bg, color, shape):
            return None

    grid = np.zeros((top_h + panel_h, width), dtype=int)
    grid[top_h:] = panel
    marker_rows = sorted(random.sample(range(top_h), marker_total))
    for row, color in zip(marker_rows, marker_colors):
        grid[row, 0] = int(color)

    try:
        out, info = _solve(grid)
    except AssertionError:
        return None

    counts = info["counts"]
    if sum(counts.values()) != shift:
        return None
    if zero_marker and 0 not in counts.values():
        return None
    if any(counts[color] == 0 for color in marked_present_colors):
        return None
    if any(len(_components_of_color(info["scene"], color)) == 0 for color in unmarked_colors):
        return None

    filtered = info["filtered"]
    panel_bg = int(info["panel_bg"])
    if not np.any((info["scene"] != panel_bg) & ~np.isin(info["scene"], marker_colors)):
        return None
    if np.any((filtered != panel_bg) & ~np.isin(filtered, marker_colors)):
        return None
    if len({int(c) for c in np.unique(filtered) if c != panel_bg}) != present_markers:
        return None
    if not _nontrivial_output_checks(info["scene"], filtered, out, shift, marker_total):
        return None

    bars = {color: counts[color] for color in marker_colors}
    if len(set(bars.values())) == 1 and marker_total > 1 and shift == marker_total * next(iter(bars.values())):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
