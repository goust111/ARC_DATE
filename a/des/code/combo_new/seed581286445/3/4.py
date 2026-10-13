from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
NONSPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8)
PANEL_KEYS = ((0, 0), (0, 1), (1, 0), (1, 1))
DIRECTION_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_SHAPES = {
    "right": ((0, 3), (1, 0), (1, 1), (1, 2), (2, 3)),
    "left": ((0, 0), (1, 1), (1, 2), (1, 3), (2, 0)),
    "up": ((0, 0), (0, 2), (1, 1), (2, 1), (3, 1)),
    "down": ((0, 1), (1, 1), (2, 1), (3, 0), (3, 2)),
}
SHAPE_LIBRARY = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
)
TEMPLATES = (
    {(0, 0): "right", (0, 1): "left", (1, 0): "right", (1, 1): "left"},
    {(0, 0): "down", (0, 1): "down", (1, 0): "up", (1, 1): "up"},
    {(0, 0): "right", (0, 1): "down", (1, 1): "left", (1, 0): "up"},
    {(0, 0): "down", (1, 0): "right", (1, 1): "up", (0, 1): "left"},
)
PANEL_MIN = 7
PANEL_MAX = 14
MAX_GENERATION_ATTEMPTS = 400


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    min_r = min(rows)
    min_c = min(cols)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


NORMALIZED_ARROWS = {
    direction: _normalize(list(coords)) for direction, coords in ARROW_SHAPES.items()
}


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return max(rows) + 1, max(cols) + 1


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected nonempty values"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nbr = (r + dr, c + dc)
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(coords)


def _find_separator_axes(g: np.ndarray) -> tuple[int, int, int]:
    height, width = g.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(g):
        value = int(color)
        rows = [r for r in range(height) if np.all(g[r, :] == value)]
        cols = [c for c in range(width) if np.all(g[:, c] == value)]
        if rows and cols:
            candidates.append((value, rows, cols))
    assert len(candidates) == 1, "expected exactly one separator color"
    color, rows, cols = candidates[0]
    assert len(rows) == 1 and len(cols) == 1, "expected a single separator row and column"
    return color, rows[0], cols[0]


def _target_panel(panel_key: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIRECTION_STEPS[direction]
    return panel_key[0] + dr, panel_key[1] + dc


def _arrow_direction(coords: list[tuple[int, int]]) -> str:
    normalized = _normalize(coords)
    matches = [direction for direction, shape in NORMALIZED_ARROWS.items() if shape == normalized]
    assert len(matches) == 1, "expected one recognizable arrow"
    return matches[0]


def _parse_panel(patch: np.ndarray) -> dict[str, object]:
    assert patch.ndim == 2 and patch.size > 0
    assert GREY not in set(map(int, np.unique(patch))), "separator color should not appear inside a panel"

    base = _mode_color(patch)
    assert base not in {GREY, MAROON}, "base color must differ from separator and arrow"

    arrow_cells = [tuple(map(int, rc)) for rc in np.argwhere(patch == MAROON)]
    assert len(arrow_cells) == 5, "expected a five-pixel arrow"
    direction = _arrow_direction(arrow_cells)

    pattern_mask = (patch != base) & (patch != MAROON)
    pattern_cells = [tuple(map(int, rc)) for rc in np.argwhere(pattern_mask)]
    assert 3 <= len(pattern_cells) <= 6, "expected a small 3-6 pixel pattern"
    pattern_colors = set(map(int, patch[pattern_mask].reshape(-1)))
    assert len(pattern_colors) == 1, "pattern pixels must share one color"

    pattern_set = set(pattern_cells)
    assert _is_connected(pattern_set), "pattern must be 4-connected"
    assert pattern_set.isdisjoint(set(arrow_cells)), "pattern and arrow must be separate"

    return {
        "base": int(base),
        "direction": direction,
        "pattern_rel": tuple(sorted(pattern_cells)),
        "pattern_set": frozenset(pattern_set),
        "shape": patch.shape,
        "arrow_set": frozenset(arrow_cells),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each panel's pattern into its arrow-indicated neighbor using the source panel's base color.

    Input structure:
    - The grid is split into a 2x2 set of rectangular panels by one full grey row and one full grey column.
    - Each panel is filled by its own dominant base color and also contains one small 4-connected pattern in another color.
    - Each panel also contains one maroon arrow made from a 3-pixel straight segment plus two diagonal head pixels.

    Rule:
    - Read each arrow direction to choose the orthogonally adjacent target panel.
    - Copy the source panel's pattern mask to the same relative coordinates inside that target panel.
    - Paint copied cells with the source panel's base color, but do not overwrite the target panel's original pattern cells.
    - Erase every maroon arrow back to its panel base color while keeping all original patterns unchanged.

    Key insight:
    - The arrow is a transfer instruction rather than a moving object: the pattern shape stays in place, while its silhouette is stamped into the neighboring panel using the source background color instead of the pattern color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator_color, sep_r, sep_c = _find_separator_axes(g)
    assert separator_color == GREY, "expected grey separator lines"

    row_bands = ((0, sep_r), (sep_r + 1, g.shape[0]))
    col_bands = ((0, sep_c), (sep_c + 1, g.shape[1]))
    assert row_bands[0][0] < row_bands[0][1] < row_bands[1][0] < row_bands[1][1]
    assert col_bands[0][0] < col_bands[0][1] < col_bands[1][0] < col_bands[1][1]

    panels: dict[tuple[int, int], dict[str, object]] = {}
    out = g.copy()
    for panel_key in PANEL_KEYS:
        pr, pc = panel_key
        r0, r1 = row_bands[pr]
        c0, c1 = col_bands[pc]
        patch = g[r0:r1, c0:c1]
        info = _parse_panel(patch)
        info.update({"r0": r0, "c0": c0, "h": patch.shape[0], "w": patch.shape[1]})
        panels[panel_key] = info

        panel_out = out[r0:r1, c0:c1]
        panel_out[patch == MAROON] = int(info["base"])

    paints: dict[tuple[int, int], int] = {}
    for panel_key, info in panels.items():
        target_key = _target_panel(panel_key, str(info["direction"]))
        assert target_key in panels, "arrow must point to an adjacent panel"
        target = panels[target_key]

        for rel_r, rel_c in info["pattern_rel"]:
            assert rel_r < int(target["h"]) and rel_c < int(target["w"]), "projected pattern must stay in bounds"
            if (rel_r, rel_c) in target["pattern_set"]:
                continue
            rr = int(target["r0"]) + rel_r
            cc = int(target["c0"]) + rel_c
            previous = paints.get((rr, cc))
            if previous is not None:
                assert previous == int(info["base"]), "projection colors must not conflict"
            else:
                paints[(rr, cc)] = int(info["base"])

    for (r, c), color in paints.items():
        out[r, c] = color
    return out


def _transform_shape(coords: tuple[tuple[int, int], ...], rotation: int, mirror: bool) -> tuple[tuple[int, int], ...]:
    points = list(coords)
    if mirror:
        points = [(r, -c) for r, c in points]
    for _ in range(rotation):
        points = [(c, -r) for r, c in points]
    return _normalize(points)


def _all_shape_variants() -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for shape in SHAPE_LIBRARY:
        for rotation in range(4):
            for mirror in (False, True):
                variants.add(_transform_shape(shape, rotation, mirror))
    return sorted(variants)


SHAPE_VARIANTS = _all_shape_variants()


def _sample_extent_pair() -> tuple[int, int] | None:
    first = random.randint(PANEL_MIN, PANEL_MAX)
    second_max = min(PANEL_MAX, 29 - first)
    if second_max < PANEL_MIN:
        return None
    second = random.randint(PANEL_MIN, second_max)
    return first, second


def _placement_axis(limit: int, size: int, prefer_margin: bool) -> range:
    if prefer_margin and limit >= size + 2:
        return range(1, limit - size)
    return range(0, limit - size + 1)


def _sample_pattern(limit_h: int, limit_w: int) -> frozenset[tuple[int, int]] | None:
    roomy = [
        shape
        for shape in SHAPE_VARIANTS
        if _shape_dims(shape)[0] <= limit_h - 2 and _shape_dims(shape)[1] <= limit_w - 2
    ]
    eligible = roomy
    prefer_margin = True
    if not eligible:
        eligible = [shape for shape in SHAPE_VARIANTS if _shape_dims(shape)[0] <= limit_h and _shape_dims(shape)[1] <= limit_w]
        prefer_margin = False
    if not eligible:
        return None

    shape = random.choice(eligible)
    shape_h, shape_w = _shape_dims(shape)
    top = random.choice(list(_placement_axis(limit_h, shape_h, prefer_margin)))
    left = random.choice(list(_placement_axis(limit_w, shape_w, prefer_margin)))
    return frozenset((r + top, c + left) for r, c in shape)


def _too_close(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]], pad: int) -> bool:
    for ar, ac in a:
        for br, bc in b:
            if abs(ar - br) <= pad and abs(ac - bc) <= pad:
                return True
    return False


def _sample_arrow(panel_h: int, panel_w: int, direction: str, pattern: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]] | None:
    shape = NORMALIZED_ARROWS[direction]
    shape_h, shape_w = _shape_dims(shape)
    for pad in (1, 0):
        candidates = []
        for top in _placement_axis(panel_h, shape_h, prefer_margin=True):
            for left in _placement_axis(panel_w, shape_w, prefer_margin=True):
                coords = frozenset((r + top, c + left) for r, c in shape)
                if _too_close(coords, pattern, pad):
                    continue
                candidates.append(coords)
        if candidates:
            return random.choice(candidates)
    return None


def _panel_specs(top_h: int, bottom_h: int, left_w: int, right_w: int) -> dict[tuple[int, int], dict[str, int]]:
    return {
        (0, 0): {"r0": 0, "c0": 0, "h": top_h, "w": left_w},
        (0, 1): {"r0": 0, "c0": left_w + 1, "h": top_h, "w": right_w},
        (1, 0): {"r0": top_h + 1, "c0": 0, "h": bottom_h, "w": left_w},
        (1, 1): {"r0": top_h + 1, "c0": left_w + 1, "h": bottom_h, "w": right_w},
    }


def _render_input(
    top_h: int,
    bottom_h: int,
    left_w: int,
    right_w: int,
    base_colors: dict[tuple[int, int], int],
    pattern_colors: dict[tuple[int, int], int],
    patterns: dict[tuple[int, int], frozenset[tuple[int, int]]],
    arrows: dict[tuple[int, int], frozenset[tuple[int, int]]],
) -> np.ndarray:
    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w
    grid = np.full((height, width), BLACK, dtype=int)
    grid[top_h, :] = GREY
    grid[:, left_w] = GREY

    specs = _panel_specs(top_h, bottom_h, left_w, right_w)
    for panel_key, spec in specs.items():
        r0 = spec["r0"]
        c0 = spec["c0"]
        h = spec["h"]
        w = spec["w"]
        grid[r0 : r0 + h, c0 : c0 + w] = int(base_colors[panel_key])
        for rel_r, rel_c in patterns[panel_key]:
            grid[r0 + rel_r, c0 + rel_c] = int(pattern_colors[panel_key])
        for rel_r, rel_c in arrows[panel_key]:
            grid[r0 + rel_r, c0 + rel_c] = MAROON
    return grid


def _gen_attempt() -> np.ndarray | None:
    row_sizes = _sample_extent_pair()
    col_sizes = _sample_extent_pair()
    if row_sizes is None or col_sizes is None:
        return None
    top_h, bottom_h = row_sizes
    left_w, right_w = col_sizes
    specs = _panel_specs(top_h, bottom_h, left_w, right_w)

    directions = dict(random.choice(TEMPLATES))
    targets = {panel_key: _target_panel(panel_key, direction) for panel_key, direction in directions.items()}

    base_values = random.sample(NONSPECIAL_COLORS, 4)
    base_colors = {panel_key: base_values[i] for i, panel_key in enumerate(PANEL_KEYS)}
    pattern_colors = {
        panel_key: random.choice([color for color in NONSPECIAL_COLORS if color != base_colors[panel_key]])
        for panel_key in PANEL_KEYS
    }

    patterns: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    overlap_source = random.choice(PANEL_KEYS) if random.random() < 0.5 else None
    if overlap_source is not None:
        overlap_target = targets[overlap_source]
        next_target = targets[overlap_target]
        shared_h = min(specs[overlap_source]["h"], specs[overlap_target]["h"], specs[next_target]["h"])
        shared_w = min(specs[overlap_source]["w"], specs[overlap_target]["w"], specs[next_target]["w"])
        pattern = _sample_pattern(shared_h, shared_w)
        if pattern is None:
            return None
        patterns[overlap_source] = pattern
        patterns[overlap_target] = pattern

    for panel_key in PANEL_KEYS:
        if panel_key in patterns:
            continue
        target = targets[panel_key]
        limit_h = min(specs[panel_key]["h"], specs[target]["h"])
        limit_w = min(specs[panel_key]["w"], specs[target]["w"])
        pattern = _sample_pattern(limit_h, limit_w)
        if pattern is None:
            return None
        patterns[panel_key] = pattern

    arrows: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for panel_key in PANEL_KEYS:
        arrow = _sample_arrow(specs[panel_key]["h"], specs[panel_key]["w"], directions[panel_key], patterns[panel_key])
        if arrow is None:
            return None
        arrows[panel_key] = arrow

    inp = _render_input(top_h, bottom_h, left_w, right_w, base_colors, pattern_colors, patterns, arrows)
    try:
        out = main(inp)
    except AssertionError:
        return None

    # Require at least one visible projection besides arrow erasure.
    if not np.any((out != inp) & (inp != MAROON)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-projection input")
