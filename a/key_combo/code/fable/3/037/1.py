import random

import numpy as np


BG = 0
BOX = 5
SIDE_TO_POS = {0: (0, 1), 1: (1, 2), 2: (2, 1), 3: (1, 0)}
POS_TO_SIDE = {pos: side for side, pos in SIDE_TO_POS.items()}
GEN_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _singleton_side(sprite: np.ndarray) -> int:
    values, counts = np.unique(sprite, return_counts=True)
    assert len(values) == 2 and sorted(counts.tolist()) == [1, 8], "legend sprite must be 8+1 two-color"
    singleton_color = int(values[np.argmin(counts)])
    locs = np.argwhere(sprite == singleton_color)
    assert locs.shape == (1, 2), "legend sprite must have one singleton cell"
    pos = tuple(map(int, locs[0]))
    assert pos in POS_TO_SIDE, "singleton must sit on an edge midpoint"
    return POS_TO_SIDE[pos]


def _rotate_to_side(sprite: np.ndarray, target_side: int) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(sprite, k)
        if _singleton_side(rotated) == target_side:
            return rotated
    raise AssertionError("legend sprite should support all four quarter-turn orientations")


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(grid == BOX):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue

        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(local, border):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        non_bg = np.argwhere(inner != bg)
        if non_bg.shape[0] != 9:
            continue
        sr0, sc0, sr1, sc1 = _bbox(non_bg)
        if (sr1 - sr0 + 1, sc1 - sc0 + 1) != (3, 3):
            continue

        sprite = inner[sr0 : sr1 + 1, sc0 : sc1 + 1]
        if np.any(sprite == bg):
            continue
        try:
            _singleton_side(sprite)
        except AssertionError:
            continue
        candidates.append(((r0, c0, r1, c1), sprite.copy()))

    assert len(candidates) == 1, "expected one hollow grey legend box with one 3x3 legend sprite"
    return candidates[0]


def _find_bars(grid: np.ndarray, bg: int) -> tuple[str, list[int], int]:
    h, w = grid.shape
    full_rows = [r for r in range(h) if grid[r, 0] != bg and np.all(grid[r, :] == grid[r, 0])]
    full_cols = [c for c in range(w) if grid[0, c] != bg and np.all(grid[:, c] == grid[0, c])]

    if full_rows and not full_cols:
        colors = {int(grid[r, 0]) for r in full_rows}
        assert len(colors) == 1, "all spanning rows must share one color"
        return "rows", sorted(full_rows), colors.pop()
    if full_cols and not full_rows:
        colors = {int(grid[0, c]) for c in full_cols}
        assert len(colors) == 1, "all spanning columns must share one color"
        return "cols", sorted(full_cols), colors.pop()
    raise AssertionError("expected either spanning rows or spanning columns, but not both")


def _make_sprite(major: int, minor: int, side: int) -> np.ndarray:
    sprite = np.full((3, 3), int(major), dtype=int)
    sprite[SIDE_TO_POS[int(side)]] = int(minor)
    return sprite


def _sample_spaced(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _sample_spaced_values(values: list[int], count: int, min_gap: int) -> list[int] | None:
    if len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _max_spaced_count(values: list[int], min_gap: int) -> int:
    count = 0
    last = -10**9
    for value in sorted(values):
        if value - last >= min_gap:
            count += 1
            last = value
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each control pixel toward the first spanning bar in each perpendicular direction, then stamp the rotated legend sprite at every hit.

    Input structure:
    - The background is black.
    - One non-background color forms several full parallel spanning rows or full parallel spanning columns.
    - A grey hollow rectangle contains one filled 3x3 two-color legend sprite whose unique singleton cell sits on an edge midpoint.
    - All other foreground cells are same-color singleton control pixels outside the bars and outside the legend box.

    Rule:
    - For each control pixel, cast rays in the two directions perpendicular to the bars and stop each ray at the first bar cell it meets.
    - Recolor only the empty cells strictly between the control pixel and each hit cell with the control color; directions with no bar produce nothing.
    - At every hit cell, center a 3x3 copy of the legend sprite rotated so its singleton edge cell faces back toward the control pixel that generated that hit.
    - The legend box stays unchanged; later stamps overwrite bar cells and any ray pixels underneath them.

    Key insight:
    - The hit marker is not fixed: the input legend sprite acts as a reusable orientation key, and each ray chooses a different quarter-turn based on its own incoming direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    legend_bbox, sprite = _find_legend(g, bg)
    box_r0, box_c0, box_r1, box_c1 = legend_bbox
    orientation, bars, _bar_color = _find_bars(g, bg)

    control_positions: list[tuple[int, int]] = []
    for r, c in zip(*np.where(g != bg)):
        rr = int(r)
        cc = int(c)
        if box_r0 <= rr <= box_r1 and box_c0 <= cc <= box_c1:
            continue
        if orientation == "rows" and rr in bars:
            continue
        if orientation == "cols" and cc in bars:
            continue
        control_positions.append((rr, cc))

    assert control_positions, "expected at least one control pixel"
    control_colors = {int(g[r, c]) for r, c in control_positions}
    assert len(control_colors) == 1, "all control pixels must share one color"
    control_color = control_colors.pop()

    out = g.copy()
    hits: list[tuple[int, int, int]] = []

    if orientation == "rows":
        for r, c in sorted(control_positions):
            above = [bar_r for bar_r in bars if bar_r < r]
            below = [bar_r for bar_r in bars if bar_r > r]
            if above:
                hit_r = above[-1]
                assert r - hit_r >= 2, "stamp would overlap its control pixel"
                assert np.all(g[hit_r + 1 : r, c] == bg), "vertical ray path must be empty"
                out[hit_r + 1 : r, c] = control_color
                hits.append((hit_r, c, 2))
            if below:
                hit_r = below[0]
                assert hit_r - r >= 2, "stamp would overlap its control pixel"
                assert np.all(g[r + 1 : hit_r, c] == bg), "vertical ray path must be empty"
                out[r + 1 : hit_r, c] = control_color
                hits.append((hit_r, c, 0))
    else:
        for r, c in sorted(control_positions):
            left = [bar_c for bar_c in bars if bar_c < c]
            right = [bar_c for bar_c in bars if bar_c > c]
            if left:
                hit_c = left[-1]
                assert c - hit_c >= 2, "stamp would overlap its control pixel"
                assert np.all(g[r, hit_c + 1 : c] == bg), "horizontal ray path must be empty"
                out[r, hit_c + 1 : c] = control_color
                hits.append((r, hit_c, 1))
            if right:
                hit_c = right[0]
                assert hit_c - c >= 2, "stamp would overlap its control pixel"
                assert np.all(g[r, c + 1 : hit_c] == bg), "horizontal ray path must be empty"
                out[r, c + 1 : hit_c] = control_color
                hits.append((r, hit_c, 3))

    centers = [(r, c) for r, c, _side in hits]
    assert len(set(centers)) == len(centers), "each hit cell should receive exactly one oriented stamp"

    for hit_r, hit_c, side in hits:
        assert 1 <= hit_r < g.shape[0] - 1 and 1 <= hit_c < g.shape[1] - 1, "stamps must stay in bounds"
        out[hit_r - 1 : hit_r + 2, hit_c - 1 : hit_c + 2] = _rotate_to_side(sprite, side)

    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    max_bars = min(4, (h - 5) // 4)
    if max_bars < 2:
        return None

    bar_count = random.randint(2, max_bars)
    bar_rows = _sample_spaced(2, h - 3, bar_count, 4)
    if bar_rows is None:
        return None

    box_row_starts = [r0 for r0 in range(h - 4) if all(bar_r not in range(r0, r0 + 5) for bar_r in bar_rows)]
    box_col_starts = list(range(2, w - 5))
    if not box_row_starts or not box_col_starts:
        return None

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    bar_color, control_color, major, minor = random.sample(palette, 4)
    sprite = _make_sprite(major, minor, random.randrange(4))

    for _ in range(80):
        box_r0 = random.choice(box_row_starts)
        box_c0 = random.choice(box_col_starts)

        allowed_cols = [c for c in range(1, w - 1) if c <= box_c0 - 2 or c >= box_c0 + 6]
        max_controls = min(4, _max_spaced_count(allowed_cols, 4))
        if max_controls < 2:
            continue

        control_count = random.randint(2, max_controls)
        control_cols = _sample_spaced_values(allowed_cols, control_count, 4)
        if control_cols is None:
            continue

        categories: list[tuple[object, list[int]]] = []
        top_rows = list(range(0, bar_rows[0] - 1))
        bottom_rows = list(range(bar_rows[-1] + 2, h))
        if top_rows:
            categories.append(("top", top_rows))
        for upper, lower in zip(bar_rows, bar_rows[1:]):
            mid_rows = list(range(upper + 2, lower - 1))
            if mid_rows:
                categories.append((("between", upper, lower), mid_rows))
        if bottom_rows:
            categories.append(("bottom", bottom_rows))

        between_categories = [item for item in categories if isinstance(item[0], tuple)]
        edge_categories = [item for item in categories if item[0] in {"top", "bottom"}]
        if not between_categories or not edge_categories:
            continue

        picked_categories = [random.choice(between_categories), random.choice(edge_categories)]
        while len(picked_categories) < control_count:
            picked_categories.append(random.choice(categories))
        random.shuffle(picked_categories)

        control_points = [(random.choice(rows), col) for col, (_kind, rows) in zip(control_cols, picked_categories)]

        grid = np.zeros((h, w), dtype=int)
        grid[bar_rows, :] = bar_color
        grid[box_r0, box_c0 : box_c0 + 5] = BOX
        grid[box_r0 + 4, box_c0 : box_c0 + 5] = BOX
        grid[box_r0 : box_r0 + 5, box_c0] = BOX
        grid[box_r0 : box_r0 + 5, box_c0 + 4] = BOX
        grid[box_r0 + 1 : box_r0 + 4, box_c0 + 1 : box_c0 + 4] = sprite
        for r, c in control_points:
            grid[r, c] = control_color

        if random.random() < 0.5:
            grid = grid.T.copy()

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, out):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 037/1")
