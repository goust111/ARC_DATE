from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
CENTER = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

BAR_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

WIDTH_CHOICES = (15, 17, 19)
BAR_COUNT_RANGE = (2, 4)
MAIN_HEIGHT_CHOICES = (11, 13, 15)
OBJECT_SIZE_CHOICES = (2, 3, 3, 4, 4, 5)
TOTAL_OBJECTS_RANGE = (4, 8)
MAX_GEN_ATTEMPTS = 400


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _grow_polyomino(size: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                frontier.add(nxt)

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return np.array(norm, dtype=int)


def _choose_bar_colors(count: int) -> list[int]:
    colors = list(BAR_COLORS)
    if count == 4:
        chosen = colors
    elif random.random() < 0.7:
        chosen = [YELLOW] + random.sample([c for c in colors if c != YELLOW], count - 1)
    else:
        chosen = random.sample(colors, count)
    random.shuffle(chosen)
    return chosen


def _count_partitions(total: int, parts: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, acc: list[int]) -> None:
        if slots == 0:
            if remaining == 0:
                out.append(tuple(acc))
            return
        lo = max(1, remaining - 4 * (slots - 1))
        hi = min(4, remaining - (slots - 1))
        for value in range(lo, hi + 1):
            acc.append(value)
            rec(remaining - value, slots - 1, acc)
            acc.pop()

    rec(total, parts, [])
    return out


def _choose_counts_and_lengths(color_count: int) -> tuple[list[int], list[int], set[int]]:
    while True:
        total_objects = random.randint(*TOTAL_OBJECTS_RANGE)
        partitions = [p for p in _count_partitions(total_objects, color_count) if max(p) >= 2]
        if not partitions:
            continue
        counts = list(random.choice(partitions))
        order = list(range(color_count))
        random.shuffle(order)
        counts = [counts[i] for i in order]

        candidates = [
            matched
            for mask in range(1, (1 << color_count) - 1)
            if any(counts[i] >= 2 for i in range(color_count) if mask & (1 << i))
            for matched in [{i for i in range(color_count) if mask & (1 << i)}]
        ]
        if not candidates:
            continue
        matched = random.choice(candidates)
        lengths: list[int] = []
        close_mismatch_exists = False
        for i, count in enumerate(counts):
            if i in matched:
                lengths.append(count)
                continue
            choices = [v for v in range(1, 5) if v != count]
            near = [v for v in choices if abs(v - count) == 1]
            if near and random.random() < 0.75:
                value = random.choice(near)
                close_mismatch_exists = True
            else:
                value = random.choice(choices)
                close_mismatch_exists = close_mismatch_exists or abs(value - count) == 1
            lengths.append(value)
        if close_mismatch_exists:
            return counts, lengths, matched


def _place_objects(
    main_h: int,
    left_w: int,
    color_counts: dict[int, int],
) -> dict[int, list[np.ndarray]] | None:
    blocked = np.zeros((main_h, left_w), dtype=bool)
    placed: dict[int, list[np.ndarray]] = {color: [] for color in color_counts}

    objects = [color for color, count in color_counts.items() for _ in range(count)]
    random.shuffle(objects)

    for color in objects:
        success = False
        for _ in range(240):
            shape = _grow_polyomino(random.choice(OBJECT_SIZE_CHOICES))
            h = int(shape[:, 0].max()) + 1
            w = int(shape[:, 1].max()) + 1
            if h > main_h or w > left_w:
                continue
            r0 = random.randint(0, main_h - h)
            c0 = random.randint(0, left_w - w)
            coords = shape + np.array([r0, c0], dtype=int)
            if any(blocked[r, c] for r, c in coords):
                continue

            coords_set = {tuple(map(int, rc)) for rc in coords}
            comp = np.array(sorted(coords_set), dtype=int)
            placed[color].append(comp)
            for r, c in comp:
                blocked[r, c] = True
                for dr, dc in DIR8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < main_h and 0 <= nc < left_w:
                        blocked[nr, nc] = True
            success = True
            break
        if not success:
            return None

    return placed


def _render_input(
    bar_colors: list[int],
    lengths: list[int],
    placements: dict[int, list[np.ndarray]],
    bar_count: int,
    main_h: int,
    width: int,
) -> np.ndarray:
    height = bar_count + main_h
    cc = width // 2
    cr = bar_count + main_h // 2
    grid = np.zeros((height, width), dtype=int)

    for r, (color, length) in enumerate(zip(bar_colors, lengths)):
        grid[r, :length] = int(color)

    grid[bar_count:, cc] = YELLOW
    grid[cr, cc] = CENTER

    for color, objects in placements.items():
        for comp in objects:
            for rr, cc_local in comp:
                grid[bar_count + int(rr), int(cc_local)] = int(color)

    return grid


def _parse_structure(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w % 2 == 1

    centers = np.argwhere(g == CENTER)
    assert centers.shape == (1, 2)
    cr, cc = map(int, centers[0])

    divider_rows = np.flatnonzero(g[:, cc] != BG)
    assert divider_rows.size > 0
    main_start = int(divider_rows[0])
    assert 2 <= main_start <= 4
    assert cr >= main_start
    assert cr - main_start == (h - 1) - cr
    assert np.all(g[:main_start, cc] == BG)
    assert np.all(np.isin(g[main_start:, cc], [CENTER, YELLOW]))
    assert g[cr, cc] == CENTER
    assert np.count_nonzero(g[main_start:, cc] == CENTER) == 1
    divider_only = np.delete(g[main_start:, cc], cr - main_start)
    assert np.all(divider_only == YELLOW)

    bars: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r in range(main_start):
        nz = np.flatnonzero(g[r] != BG)
        assert nz.size > 0 and int(nz[0]) == 0
        length = int(nz[-1]) + 1
        color = int(g[r, 0])
        assert color in BAR_COLORS
        assert 1 <= length <= 4
        assert np.all(g[r, :length] == color)
        assert np.all(g[r, length:] == BG)
        assert color not in seen_colors
        seen_colors.add(color)
        bars.append((r, color, length))

    assert np.all(g[main_start:, cc + 1 :] == BG)
    left_colors = {int(v) for v in np.unique(g[main_start:, :cc]) if int(v) != BG}
    assert left_colors.issubset(seen_colors)
    return cr, cc, main_start, bars


def _left_components_by_color(grid: np.ndarray, main_start: int, cc: int, colors: list[int]) -> dict[int, list[np.ndarray]]:
    left = grid[main_start:, :cc]
    out: dict[int, list[np.ndarray]] = {}
    for color in colors:
        out[color] = _components_mask(left == color, connectivity=4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter colors whose left-side object count matches their top bar length, then point-reflect only those objects across the center marker.

    Input structure:
    - A black background contains 2-4 colored instruction bars in the top rows, one bar per row starting at column 0.
    - Below the bars, a yellow vertical divider fills the center column, except for one red center marker on that divider.
    - The left half of the main area contains several separated 4-connected objects whose colors come from the bar colors; the right half is empty.

    Rule:
    - For each bar color, count the number of 4-connected objects of that color strictly on the left side of the divider.
    - A color passes iff that object count equals the length of its bar.
    - Erase every left-side object and top bar whose color does not pass.
    - Keep every passing left-side object and paint its 180-degree point reflection through the red center marker onto the right side.
    - Keep the divider and center marker unchanged.

    Key insight:
    - The top bars are expected counts rather than summaries: only colors whose actual object counts match their bar lengths survive and get reflected.
    """
    g = np.asarray(grid, dtype=int)
    cr, cc, main_start, bars = _parse_structure(g)
    colors = [color for _row, color, _length in bars]
    counts = {color: len(comps) for color, comps in _left_components_by_color(g, main_start, cc, colors).items()}
    passed = {color for _row, color, length in bars if counts[color] == length}

    out = g.copy()
    for _row, color, length in bars:
        if color not in passed:
            out[_row, :length] = BG

    left_mask = np.zeros(g.shape, dtype=bool)
    left_mask[main_start:, :cc] = True
    for color in colors:
        color_mask = left_mask & (g == color)
        if color not in passed:
            out[color_mask] = BG
            continue
        for r, c in np.argwhere(color_mask):
            rr = 2 * cr - int(r)
            rc = 2 * cc - int(c)
            assert main_start <= rr < g.shape[0]
            assert cc < rc < g.shape[1]
            assert out[rr, rc] == BG
            out[rr, rc] = color

    out[main_start:, cc] = YELLOW
    out[cr, cc] = CENTER
    return out


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        cr, cc, main_start, bars = _parse_structure(inp)
        colors = [color for _row, color, _length in bars]
        comps_by_color = _left_components_by_color(inp, main_start, cc, colors)
        counts = {color: len(comps) for color, comps in comps_by_color.items()}
        assert 4 <= sum(counts.values()) <= 8
        assert all(1 <= counts[color] <= 4 for color in colors)
        for color in colors:
            for comp in comps_by_color[color]:
                assert 2 <= len(comp) <= 5
        passed = {color for _row, color, length in bars if counts[color] == length}
        failed = set(colors) - passed
        assert passed and failed
        assert any(counts[color] >= 2 for color in passed)
        out = main(inp)

        added = np.argwhere((inp == BG) & (out != BG))
        removed = np.argwhere((inp != BG) & (out == BG))
        assert len(added) > 0
        assert len(removed) > 0
        assert np.all(inp[main_start:, cc + 1 :] == BG)

        for row, color, length in bars:
            if color in passed:
                assert np.all(out[row, :length] == color)
            else:
                assert np.all(out[row, :length] == BG)
            assert np.all(out[row, length:] == BG)

        for color in colors:
            comps = comps_by_color[color]
            for comp in comps:
                for lr, lc in comp:
                    gr = main_start + int(lr)
                    gc = int(lc)
                    rr = 2 * cr - gr
                    rc = 2 * cc - gc
                    if color in passed:
                        assert out[gr, gc] == color
                        assert out[rr, rc] == color
                    else:
                        assert out[gr, gc] == BG
        assert np.all(out[main_start:, cc] == np.where(np.arange(inp.shape[0] - main_start) == cr - main_start, CENTER, YELLOW))

        if YELLOW in colors:
            full_yellow = _components_mask(inp[main_start:, :] == YELLOW, connectivity=4)
            assert len(full_yellow) == counts[YELLOW] + 1
        return True
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    bar_count = random.randint(*BAR_COUNT_RANGE)
    main_h = random.choice(MAIN_HEIGHT_CHOICES)
    width = random.choice(WIDTH_CHOICES)
    color_count = bar_count
    bar_colors = _choose_bar_colors(color_count)
    counts, lengths, _matched = _choose_counts_and_lengths(color_count)
    color_counts = dict(zip(bar_colors, counts))

    left_w = width // 2
    usable_left = left_w - 1
    if usable_left < 4:
        return None
    placements = _place_objects(main_h, usable_left, color_counts)
    if placements is None:
        return None

    inp = _render_input(bar_colors, lengths, placements, bar_count, main_h, width)
    if not _semantic_check(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance for puzzle 3")
