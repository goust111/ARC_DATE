from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
GREY = 5
TOWER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _mode(values) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _parse_scene(grid: np.ndarray) -> tuple[int, int, np.ndarray, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    terrain = _mode(g[-1, :].tolist())
    border = np.concatenate([g[0, :], g[:-1, 0], g[:-1, -1]])
    bg = _mode(border.tolist())

    surface = np.zeros(w, dtype=int)
    towers: list[dict[str, int]] = []
    seen_colors: set[int] = set()

    for c in range(w):
        terrain_rows = np.flatnonzero(g[:, c] == terrain)
        assert terrain_rows.size > 0, "every column must contain terrain"
        s = int(terrain_rows[0])
        surface[c] = s
        assert bool(np.all(g[s:, c] == terrain)), "terrain must fill to the bottom"

        if s == 0:
            continue

        above = g[:s, c]
        non_bg_rows = [r for r, v in enumerate(above.tolist()) if int(v) not in (bg, terrain)]
        if not non_bg_rows:
            assert bool(np.all(above == bg)), "empty sky columns must stay background"
            continue

        top = int(non_bg_rows[0])
        bottom = int(non_bg_rows[-1])
        color = int(g[top, c])
        assert bottom == s - 1, "tower must stand directly on the terrain"
        assert bool(np.all(g[top:s, c] == color)), "tower must be a solid vertical segment"
        assert color not in seen_colors, "tower colors must be unique"
        if top > 0:
            assert bool(np.all(g[:top, c] == bg)), "space above a tower must be background"

        towers.append(
            {
                "col": int(c),
                "surface": int(s),
                "top": int(top),
                "height": int(s - top),
                "color": int(color),
            }
        )
        seen_colors.add(color)

    towers.sort(key=lambda tower: tower["col"])
    assert len(towers) >= 2, "expected multiple towers"
    return bg, terrain, surface, towers


def _claim_better(a: tuple[int, int, int, int, int], b: tuple[int, int, int, int, int]) -> bool:
    if a[0] != b[0]:
        return a[0] > b[0]
    if a[1] != b[1]:
        return a[1] < b[1]
    return a[2] < b[2]


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, _terrain, surface, towers = _parse_scene(g)
    tower_by_col = {tower["col"]: tower for tower in towers}

    claims: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = defaultdict(list)
    terrain_stops = 0
    tower_stops = 0

    for tower_index, tower in enumerate(towers):
        row = int(tower["top"])
        col = int(tower["col"])
        height = int(tower["height"])
        color = int(tower["color"])

        for step in (-1, 1):
            cc = col + step
            while 0 <= cc < w:
                if int(surface[cc]) <= row:
                    terrain_stops += 1
                    break

                blocker = tower_by_col.get(int(cc))
                if blocker is not None and int(blocker["height"]) >= height:
                    tower_stops += 1
                    break

                occupied_by_tower = blocker is not None and int(blocker["top"]) <= row < int(blocker["surface"])
                if not occupied_by_tower:
                    claims[(int(row), int(cc))].append((height, abs(int(cc - col)), col, color, tower_index))
                cc += step

    out = g.copy()
    overlap_higher = 0
    overlap_equal = 0
    ambiguous_ties = 0

    for (r, c), options in claims.items():
        best = options[0]
        for option in options[1:]:
            if _claim_better(option, best):
                best = option

        if len(options) > 1:
            if any(option[0] < best[0] for option in options):
                overlap_higher += 1
            if any(option[0] == best[0] and option[1] > best[1] for option in options):
                overlap_equal += 1
            if any(
                option[0] == best[0] and option[1] == best[1] and option[2] != best[2]
                for option in options
            ):
                ambiguous_ties += 1

        out[int(r), int(c)] = int(best[3])

    meta = {
        "tower_count": len(towers),
        "terrain_stops": int(terrain_stops),
        "tower_stops": int(tower_stops),
        "overlap_higher": int(overlap_higher),
        "overlap_equal": int(overlap_equal),
        "ambiguous_ties": int(ambiguous_ties),
        "changed_pixels": int(np.count_nonzero(out != g)),
    }
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each tower's horizontal signal cover over a terrain profile, with taller towers dominating overlaps.

    Input structure:
    - The bottom of the grid is a solid grey terrain mass whose top contour varies by column.
    - Colored one-cell-wide vertical towers stand directly on that contour; the rest of the sky is black.
    - Each tower has a unique non-grey, non-black color and a visible height encoded by its vertical length.

    Rule:
    - Keep the terrain and all tower pixels unchanged.
    - For every tower, start from its top row and extend a horizontal signal left and right across the sky.
    - Stop a signal when it reaches the grid edge, a column where the terrain rises to the signal row or above, or another tower whose height is at least the current tower's height.
    - If multiple signals want the same sky pixel, the taller tower wins; if the heights match, keep the color of the nearer tower.

    Key insight:
    - The output is a line-of-sight propagation problem on a single horizontal row per tower, with terrain blocking by absolute height and signal priority determined by tower height.
    """
    return _solve(np.asarray(grid, dtype=int), return_meta=False)


def _random_surface(h: int, w: int) -> np.ndarray:
    low = max(4, h // 3)
    high = h - 2
    cur = random.randint(max(low, high - 5), high - 1)
    surface = np.zeros(w, dtype=int)
    for c in range(w):
        surface[c] = int(cur)
        cur = max(low, min(high, cur + random.choice([-1, 0, 0, 1])))

    for _ in range(random.randint(1, 3)):
        center = random.randint(1, w - 2)
        width = random.randint(1, 3)
        delta = random.choice([-2, -1, 1])
        for c in range(max(0, center - width), min(w, center + width + 1)):
            weight = width - abs(c - center) + 1
            surface[c] = int(np.clip(surface[c] + delta * weight, low, high))

    return surface


def _clear_interval(surface: np.ndarray, start: int, end: int, top_row: int, h: int) -> None:
    lo = min(start, end)
    hi = max(start, end)
    for c in range(lo, hi + 1):
        min_surface = min(h - 2, top_row + 1 + random.randint(0, 2))
        surface[c] = max(int(surface[c]), int(min_surface))


def _near_ridge_cols(w: int, anchor: int, direction: int) -> list[int] | None:
    distance = random.randint(2, 4)
    center = anchor + direction * distance
    if not (1 <= center <= w - 2):
        return None
    width = random.randint(1, 2)
    if direction < 0:
        start = max(1, center - width + 1)
    else:
        start = center
    if start + width - 1 > w - 2:
        start = w - 1 - width
    cols = list(range(start, start + width))
    if any(not (1 <= c <= w - 2) for c in cols):
        return None
    return cols


def _set_ridge(surface: np.ndarray, cols: list[int], top_row: int) -> None:
    ridge_top = max(1, top_row - 1)
    ridge_bottom = top_row
    for c in cols:
        surface[c] = min(int(surface[c]), random.randint(ridge_top, ridge_bottom))


def _sample_pair_columns(w: int, *, odd_gap: bool) -> tuple[int, int] | None:
    gaps = []
    for gap in range(5, min(12, w - 3)):
        if odd_gap and gap % 2 == 0:
            continue
        gaps.append(gap)
    if not gaps:
        return None
    gap = random.choice(gaps)
    start_hi = w - gap - 3
    if start_hi < 2:
        return None
    c1 = random.randint(2, start_hi)
    return int(c1), int(c1 + gap)


def _sample_extra_columns(w: int, count: int, taken: list[int]) -> list[int] | None:
    picked: list[int] = []
    for _ in range(300):
        if len(picked) == count:
            return sorted(picked)
        c = random.randint(1, w - 2)
        if any(abs(c - other) < 3 for other in taken + picked):
            continue
        picked.append(int(c))
    return None


def _render_grid(h: int, w: int, surface: np.ndarray, towers: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for c in range(w):
        grid[int(surface[c]) :, c] = GREY
    for tower in towers:
        col = int(tower["col"])
        top = int(tower["top"])
        bottom = int(tower["surface"])
        grid[top:bottom, col] = int(tower["color"])
    return grid


def _add_extra_towers(
    h: int,
    surface: np.ndarray,
    colors: list[int],
    cols: list[int],
    used_top_rows: set[int],
    towers: list[dict[str, int]],
) -> None:
    for color, col in zip(colors, cols, strict=False):
        top_choices = [row for row in range(1, h - 3) if row not in used_top_rows]
        assert top_choices, "need a free top row for an extra tower"
        top = int(random.choice(top_choices))
        max_height = min(6, h - 2 - top)
        assert max_height >= 1
        height = int(random.randint(1, max_height))
        surface[col] = int(top + height)
        towers.append(
            {
                "col": int(col),
                "top": int(top),
                "height": int(height),
                "surface": int(top + height),
                "color": int(color),
            }
        )
        used_top_rows.add(top)


def _build_equal_theme() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(17, 28)
    num_towers = random.choice([2, 3, 4])
    pair = _sample_pair_columns(w, odd_gap=True)
    if pair is None:
        return None
    c1, c2 = pair

    top_row = random.randint(2, h - 8)
    max_height = min(5, h - 2 - top_row)
    if max_height < 2:
        return None
    pair_height = random.randint(2, max_height)

    surface = _random_surface(h, w)
    _clear_interval(surface, c1, c2, top_row, h)
    surface[c1] = int(top_row + pair_height)
    surface[c2] = int(top_row + pair_height)

    direction = random.choice([-1, 1])
    ridge_cols = _near_ridge_cols(w, c1 if direction < 0 else c2, direction)
    if ridge_cols is None:
        return None
    if direction < 0:
        _clear_interval(surface, ridge_cols[-1] + 1, c1 - 1, top_row, h)
    else:
        _clear_interval(surface, c2 + 1, ridge_cols[0] - 1, top_row, h)
    _set_ridge(surface, ridge_cols, top_row)

    colors = random.sample(TOWER_COLORS, num_towers)
    towers = [
        {"col": int(c1), "top": int(top_row), "height": int(pair_height), "surface": int(top_row + pair_height), "color": int(colors[0])},
        {"col": int(c2), "top": int(top_row), "height": int(pair_height), "surface": int(top_row + pair_height), "color": int(colors[1])},
    ]

    extra_cols = _sample_extra_columns(w, num_towers - 2, [c1, c2] + ridge_cols)
    if extra_cols is None:
        return None
    _add_extra_towers(h, surface, colors[2:], extra_cols, {int(top_row)}, towers)
    return _render_grid(h, w, surface, towers)


def _build_override_theme() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(17, 28)
    num_towers = random.choice([2, 3, 4])
    pair = _sample_pair_columns(w, odd_gap=False)
    if pair is None:
        return None
    c1, c2 = pair

    top_row = random.randint(2, h - 8)
    max_tall = min(6, h - 2 - top_row)
    if max_tall < 3:
        return None
    tall_height = random.randint(3, max_tall)
    short_height = random.randint(1, tall_height - 1)
    if random.random() < 0.5:
        h1, h2 = tall_height, short_height
    else:
        h1, h2 = short_height, tall_height

    surface = _random_surface(h, w)
    _clear_interval(surface, c1, c2, top_row, h)
    surface[c1] = int(top_row + h1)
    surface[c2] = int(top_row + h2)

    direction = random.choice([-1, 1])
    ridge_cols = _near_ridge_cols(w, c1 if direction < 0 else c2, direction)
    if ridge_cols is None:
        return None
    if direction < 0:
        _clear_interval(surface, ridge_cols[-1] + 1, c1 - 1, top_row, h)
    else:
        _clear_interval(surface, c2 + 1, ridge_cols[0] - 1, top_row, h)
    _set_ridge(surface, ridge_cols, top_row)

    colors = random.sample(TOWER_COLORS, num_towers)
    towers = [
        {"col": int(c1), "top": int(top_row), "height": int(h1), "surface": int(top_row + h1), "color": int(colors[0])},
        {"col": int(c2), "top": int(top_row), "height": int(h2), "surface": int(top_row + h2), "color": int(colors[1])},
    ]

    extra_cols = _sample_extra_columns(w, num_towers - 2, [c1, c2] + ridge_cols)
    if extra_cols is None:
        return None
    _add_extra_towers(h, surface, colors[2:], extra_cols, {int(top_row)}, towers)
    return _render_grid(h, w, surface, towers)


def _build_terrain_theme() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 26)
    num_towers = random.choice([2, 3, 4])
    cols = _sample_extra_columns(w, num_towers, [])
    if cols is None:
        return None

    surface = _random_surface(h, w)
    top_row = random.randint(2, h - 8)
    max_height = min(6, h - 2 - top_row)
    if max_height < 2:
        return None
    focus_height = random.randint(2, max_height)
    focus_index = random.randrange(num_towers)
    focus_col = int(cols[focus_index])
    surface[focus_col] = int(top_row + focus_height)

    ridge_options = [
        (direction, cols)
        for direction in (-1, 1)
        for cols in [_near_ridge_cols(w, focus_col, direction)]
        if cols is not None
    ]
    if not ridge_options:
        return None
    ridge_dir, ridge_cols = random.choice(ridge_options)
    if ridge_dir < 0:
        _clear_interval(surface, ridge_cols[-1] + 1, focus_col - 1, top_row, h)
    else:
        _clear_interval(surface, focus_col + 1, ridge_cols[0] - 1, top_row, h)
    _set_ridge(surface, ridge_cols, top_row)

    colors = random.sample(TOWER_COLORS, num_towers)
    towers: list[dict[str, int]] = []
    used_top_rows = {int(top_row)}
    for i, (color, col) in enumerate(zip(colors, cols, strict=False)):
        if i == focus_index:
            towers.append(
                {
                    "col": int(col),
                    "top": int(top_row),
                    "height": int(focus_height),
                    "surface": int(top_row + focus_height),
                    "color": int(color),
                }
            )
            continue

        top_choices = [row for row in range(1, h - 3) if row not in used_top_rows]
        if not top_choices:
            return None
        top = int(random.choice(top_choices))
        max_other = min(6, h - 2 - top)
        if max_other < 1:
            return None
        height = int(random.randint(1, max_other))
        surface[col] = int(top + height)
        towers.append(
            {
                "col": int(col),
                "top": int(top),
                "height": int(height),
                "surface": int(top + height),
                "color": int(color),
            }
        )
        used_top_rows.add(top)

    return _render_grid(h, w, surface, towers)


def generate_input() -> np.ndarray:
    builders = {
        "equal": _build_equal_theme,
        "override": _build_override_theme,
        "terrain": _build_terrain_theme,
    }

    for _ in range(GEN_RETRIES):
        theme = random.choice(list(builders))
        try:
            grid = builders[theme]()
            if grid is None:
                continue

            out, meta = _solve(grid, return_meta=True)
        except AssertionError:
            continue
        if meta["ambiguous_ties"] != 0:
            continue
        if meta["changed_pixels"] < 4:
            continue
        if theme == "equal" and not (meta["overlap_equal"] > 0 and meta["terrain_stops"] > 0):
            continue
        if theme == "override" and not (meta["overlap_higher"] > 0 and meta["terrain_stops"] > 0):
            continue
        if theme == "terrain" and not (meta["terrain_stops"] > 0 and meta["tower_count"] >= 2):
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid tower-signal puzzle input")
