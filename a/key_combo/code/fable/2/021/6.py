from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.flatten())))
    if exclude is None:
        return colors
    return [c for c in colors if c not in exclude]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _foreground_components(grid: np.ndarray, bg: int = BLACK) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit inside 30x30"
    assert int(get_bg(g)) == BLACK, "background must be black"

    comps = _foreground_components(g, bg=BLACK)
    assert comps, "input must contain foreground objects"
    pattern_comps = [coords for coords in comps if len(coords) > 1]
    singletons = [coords[0] for coords in comps if len(coords) == 1]

    assert len(pattern_comps) == 1, "expected exactly one non-singleton connected pattern"
    assert len(singletons) >= 4 and len(singletons) % 2 == 0, "expected several singleton sample pixels"

    pattern = pattern_comps[0]
    pattern_bbox = bbox_coords(pattern)
    pattern_colors = {int(g[r, c]) for r, c in pattern}
    assert len(pattern_colors) >= 2, "pattern must be multi-color"

    row_pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in singletons:
        row_pairs[int(r)].append((int(c), int(g[r, c])))

    assert all(len(items) == 2 for items in row_pairs.values()), "each sample row must contain exactly one pixel pair"

    axis2_values = {items[0][0] + items[1][0] for items in row_pairs.values()}
    assert len(axis2_values) == 1, "all sample-pair midpoints must lie on one vertical axis"
    axis2 = axis2_values.pop()

    _r0, c0, _r1, c1 = pattern_bbox
    if 2 * c1 <= axis2 - 2:
        pattern_side = "left"
    elif 2 * c0 >= axis2 + 2:
        pattern_side = "right"
    else:
        raise AssertionError("pattern must lie strictly on one side of the inferred axis")

    mapping: dict[int, int] = {}
    target_colors: set[int] = set()
    normalized_pairs: list[dict[str, int]] = []
    for row, items in sorted(row_pairs.items()):
        left, right = sorted(items, key=lambda item: item[0])
        lc, lcol = left
        rc, rcol = right
        assert lc + rc == axis2, "pair endpoints must be mirror positions on their row"
        if pattern_side == "left":
            assert 2 * lc <= axis2 - 2, "sample pixels must remain isolated across the axis gap"
            src_color, dst_color = lcol, rcol
        else:
            assert 2 * rc >= axis2 + 2, "sample pixels must remain isolated across the axis gap"
            src_color, dst_color = rcol, lcol

        assert src_color in pattern_colors, "pattern-side sample color must come from the pattern palette"
        assert dst_color not in pattern_colors, "far-side sample color must be outside the pattern palette"
        if src_color in mapping:
            assert mapping[src_color] == dst_color, "duplicate sample colors must imply the same recolor rule"
        mapping[src_color] = dst_color
        target_colors.add(dst_color)
        normalized_pairs.append(
            {
                "row": int(row),
                "left_col": int(lc),
                "right_col": int(rc),
                "src_color": int(src_color),
                "dst_color": int(dst_color),
            }
        )

    assert set(mapping) == pattern_colors, "sample pairs must define a recolor target for every pattern color"
    assert len(target_colors) == len(mapping), "recolor targets must be unique"

    return {
        "grid": g,
        "pattern": pattern,
        "pattern_bbox": pattern_bbox,
        "pattern_colors": pattern_colors,
        "axis2": int(axis2),
        "pattern_side": pattern_side,
        "mapping": mapping,
        "pairs": normalized_pairs,
    }


def _complete_with_axis(parsed: dict, axis2: int, recolor: bool) -> np.ndarray | None:
    g = parsed["grid"]
    out = g.copy()
    for r, c in parsed["pattern"]:
        mc = int(axis2 - int(c))
        if not (0 <= mc < g.shape[1]):
            return None
        if int(g[r, mc]) != BLACK:
            return None
        out[r, mc] = int(parsed["mapping"][int(g[r, c])]) if recolor else int(g[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the lone connected pattern across a hidden vertical axis and recolor the mirrored copy from sample pairs.

    Input structure:
    - Black background with one connected multi-color pattern entirely on one side of an unseen vertical axis.
    - Several isolated single-pixel pairs; each pair lies on one row, and all pair midpoints share the same vertical axis.
    - In every pair, the endpoint on the pattern side uses a pattern color, while the far endpoint uses a distinct color absent from the pattern.

    Rule:
    - Detect the unique non-singleton foreground component as the source pattern and read the hidden axis from the common midpoint of all singleton pairs.
    - Use each pair as a sample of the composite transformation: pattern-side color maps to far-side color, and the two endpoints are mirror positions across that axis.
    - Keep the input unchanged, then place a mirrored copy of the source pattern on the opposite side and recolor every mirrored pixel with the inferred color dictionary.

    Key insight:
    - Each singleton pair simultaneously encodes geometry and color: its midpoint reveals the unseen mirror axis, and its two colors reveal how a mirrored source pixel must be recolored.
    """
    parsed = _parse_grid(grid)
    out = _complete_with_axis(parsed, parsed["axis2"], recolor=True)
    assert out is not None, "mirrored copy must fit cleanly into the background"
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_mask(height: int, width: int, min_cells: int, max_cells: int) -> np.ndarray:
    target = random.randint(min_cells, max_cells)
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while mask.sum() < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    return _crop_mask(mask)


def _colorize_mask(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = np.argwhere(mask)
    seed_coords = coords[random.sample(range(len(coords)), len(colors))]
    random.shuffle(colors)
    out = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        distances = [abs(int(r) - int(sr)) + abs(int(c) - int(sc)) for sr, sc in seed_coords]
        nearest = min(range(len(colors)), key=lambda idx: (distances[idx], idx))
        out[r, c] = int(colors[nearest])
    return out


def _make_pattern(src_colors: list[int]) -> np.ndarray | None:
    for _ in range(80):
        height = random.randint(4, 8)
        width = random.randint(4, 7)
        area_low = max(len(src_colors) + 4, (height * width) // 3)
        area_high = min(height * width - 1, (2 * height * width) // 3 + 2)
        if area_low > area_high:
            continue
        mask = _random_connected_mask(height, width, area_low, area_high)
        h, w = mask.shape
        if h < 3 or w < 3 or mask.sum() < len(src_colors) + 4:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        pattern = _colorize_mask(mask, src_colors.copy())
        counts = Counter(map(int, pattern[pattern != BLACK].flatten()))
        if set(counts) != set(src_colors):
            continue
        if min(counts.values()) < 2:
            continue
        return pattern
    return None


def _valid_c0_range(width: int, pattern_width: int, axis2: int, side: str) -> tuple[int, int] | None:
    if side == "left":
        low = max(1, axis2 - (width - 2))
        high = min((axis2 - 2) // 2 - (pattern_width - 1), width - pattern_width - 1)
    else:
        low = max((axis2 + 3) // 2, 1)
        high = min(width - pattern_width - 1, axis2 - pattern_width)
    if low > high:
        return None
    return low, high


def _choose_sample_rows(height: int, pattern_r0: int, pattern_height: int, count: int) -> list[int] | None:
    blocked = set(range(pattern_r0 - 1, pattern_r0 + pattern_height + 1))
    available = [r for r in range(1, height - 1) if r not in blocked]
    for _ in range(80):
        rows = sorted(random.sample(available, count))
        if all(rows[idx + 1] - rows[idx] >= 2 for idx in range(len(rows) - 1)):
            return rows
    return None


def _sample_side_columns(width: int, axis2: int, side: str) -> list[int]:
    if side == "left":
        return [c for c in range(1, width - 1) if 2 * c <= axis2 - 2]
    return [c for c in range(1, width - 1) if 2 * c >= axis2 + 2]


def _pick_sample_columns(candidates: list[int], count: int) -> list[int] | None:
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return [candidates[0]] * count
    for _ in range(80):
        cols = [random.choice(candidates) for _ in range(count)]
        if count == 1 or len(set(cols)) >= 2:
            return cols
    return None


def _build_candidate() -> np.ndarray | None:
    n_colors = random.randint(2, 4)
    src_colors = random.sample(PALETTE, n_colors)
    dst_colors = random.sample([c for c in PALETTE if c not in src_colors], n_colors)
    pattern = _make_pattern(src_colors)
    if pattern is None:
        return None

    ph, pw = pattern.shape
    min_height = ph + 2 * n_colors + 4
    max_height = min(30, min_height + 6)
    min_width = max(14, 2 * pw + 6)
    if min_height > 30 or min_width > 30:
        return None

    height = random.randint(min_height, max_height)
    width = random.randint(min_width, 30)
    side = random.choice(("left", "right"))
    parity = random.choice((0, 1))

    axis_candidates = [axis2 for axis2 in range(4, 2 * width - 4) if axis2 % 2 == parity and axis2 != width - 1]
    random.shuffle(axis_candidates)
    for axis2 in axis_candidates:
        c0_range = _valid_c0_range(width, pw, axis2, side)
        if c0_range is None:
            continue
        low, high = c0_range
        c0 = random.randint(low, high)

        r0_candidates = list(range(2, height - ph - 1))
        random.shuffle(r0_candidates)
        for r0 in r0_candidates:
            sample_rows = _choose_sample_rows(height, r0, ph, n_colors)
            if sample_rows is None:
                continue
            side_cols = _sample_side_columns(width, axis2, side)
            chosen_side_cols = _pick_sample_columns(side_cols, n_colors)
            if chosen_side_cols is None:
                continue

            grid = np.zeros((height, width), dtype=int)
            grid[r0 : r0 + ph, c0 : c0 + pw] = pattern

            pair_specs = list(zip(src_colors, dst_colors, sample_rows, chosen_side_cols))
            random.shuffle(pair_specs)
            for src_color, dst_color, row, src_col in pair_specs:
                dst_col = axis2 - src_col
                if not (0 <= dst_col < width):
                    return None
                if side == "left":
                    left_col, right_col = src_col, dst_col
                    left_color, right_color = src_color, dst_color
                else:
                    left_col, right_col = dst_col, src_col
                    left_color, right_color = dst_color, src_color
                if grid[row, left_col] != BLACK or grid[row, right_col] != BLACK:
                    break
                grid[row, left_col] = int(left_color)
                grid[row, right_col] = int(right_color)
            else:
                return grid
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    parsed = _parse_grid(grid)
    out = main(grid)
    assert not np.array_equal(out, grid), "output must add a mirrored recolored copy"

    mirrored_same_color = _complete_with_axis(parsed, parsed["axis2"], recolor=False)
    assert mirrored_same_color is not None
    assert not np.array_equal(out, mirrored_same_color), "recolor step must be necessary"

    center_axis2 = grid.shape[1] - 1
    if center_axis2 != parsed["axis2"]:
        center_completion = _complete_with_axis(parsed, center_axis2, recolor=True)
        if center_completion is not None:
            assert not np.array_equal(out, center_completion), "grid-center axis must not accidentally solve the task"

    output_comps = _foreground_components(out, bg=BLACK)
    non_singletons = [coords for coords in output_comps if len(coords) > 1]
    assert len(non_singletons) >= 2, "output should contain both source and mirrored patterns"
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            if _validate_generated(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid paired-pixel mirror input")
