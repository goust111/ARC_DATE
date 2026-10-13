import random
from collections import Counter

import numpy as np


BG = 0
OBJECT_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    assert len(non_bg) == 3

    seeds = []
    for color in non_bg:
        coords = np.argwhere(g == color)
        if len(coords) == 1:
            r, c = map(int, coords[0])
            if c in (0, w - 1):
                seeds.append((color, r, c))
    assert len(seeds) == 2

    side_cols = {c for _color, _r, c in seeds}
    assert len(side_cols) == 1
    side_col = next(iter(side_cols))
    assert side_col in (0, w - 1)

    seed_rows = {r for _color, r, _c in seeds}
    assert len(seed_rows) == 2

    seed_colors = {color for color, _r, _c in seeds}
    object_colors = [color for color in non_bg if color not in seed_colors]
    assert len(object_colors) == 1
    object_color = object_colors[0]

    return g, bg, object_color, seeds


def _stripe_row_colors(height: int, seeds) -> tuple[list[int | None], int]:
    (color_a, row_a, _), (color_b, row_b, _) = seeds
    gap = abs(row_a - row_b)
    assert gap > 0
    period = 2 * gap
    row_colors: list[int | None] = [None] * height
    for r in range(height):
        if (r - row_a) % period == 0:
            row_colors[r] = int(color_a)
        elif (r - row_b) % period == 0:
            row_colors[r] = int(color_b)
    return row_colors, gap


def _vote_counts(coords: np.ndarray, row_colors: list[int | None], color_a: int, color_b: int) -> tuple[int, int]:
    count_a = 0
    count_b = 0
    for r, _c in coords:
        row_color = row_colors[int(r)]
        if row_color == color_a:
            count_a += 1
        elif row_color == color_b:
            count_b += 1
    return count_a, count_b


def main(grid: np.ndarray) -> np.ndarray:
    """Use two border seeds to define invisible periodic stripes, then recolor or erase each gray object by stripe-majority overlap.

    Input structure:
    - The grid has one dominant background color, exactly two different-color seed pixels on the same left or right edge, and several gray 4-connected objects.
    - The seed pixels are singletons; every other foreground object is monochrome gray and spatially separated.

    Rule:
    - The two seed rows define invisible horizontal stripes: each seed row is a stripe of that seed's color, and the two stripe colors repeat upward and downward with period twice the row gap.
    - For each gray 4-connected object, count how many of its pixels lie on rows covered by each seed color's invisible stripes.
    - Recolor the whole object to the color with the larger count; if the counts are equal, erase the whole object to the background.
    - Keep the seed pixels unchanged, and never draw the stripes themselves.

    Key insight:
    - The visible objects are judged by overlap with an unpainted periodic row schedule controlled only by the two edge seeds, so the decisive information is hidden stripe coverage count, not object shape alone.
    """
    g, bg, object_color, seeds = _parse_input(grid)
    h, _w = g.shape

    row_colors, _gap = _stripe_row_colors(h, seeds)
    color_a, color_b = int(seeds[0][0]), int(seeds[1][0])

    out = g.copy()
    for coords in _components_of_color(g, object_color):
        count_a, count_b = _vote_counts(coords, row_colors, color_a, color_b)
        if count_a > count_b:
            target = color_a
        elif count_b > count_a:
            target = color_b
        else:
            target = bg
        out[coords[:, 0], coords[:, 1]] = target
    return out


def _mark_blocked(blocked: np.ndarray, coords: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _can_place(coords: set[tuple[int, int]], blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r in (0, h - 1) or c in (0, w - 1):
            return False
        if blocked[r, c]:
            return False
    return True


def _make_two_stripe_object(
    h: int,
    w: int,
    top_row: int,
    bottom_row: int,
    top_color: int,
    bottom_color: int,
    winner,
) -> set[tuple[int, int]] | None:
    assert top_row < bottom_row
    for _ in range(50):
        if winner == "tie":
            seg_top = seg_bottom = random.randint(2, 5)
        elif winner == top_color:
            seg_top = random.randint(3, 5)
            seg_bottom = random.randint(2, seg_top - 1)
        elif winner == bottom_color:
            seg_bottom = random.randint(3, 5)
            seg_top = random.randint(2, seg_bottom - 1)
        else:
            return None

        anchor = random.randint(2, w - 3)
        left_top = random.randint(0, seg_top - 1)
        left_bottom = random.randint(0, seg_bottom - 1)
        start_top = anchor - left_top
        start_bottom = anchor - left_bottom
        end_top = start_top + seg_top - 1
        end_bottom = start_bottom + seg_bottom - 1
        if start_top < 1 or start_bottom < 1 or end_top > w - 2 or end_bottom > w - 2:
            continue

        coords = {(top_row, c) for c in range(start_top, end_top + 1)}
        coords |= {(bottom_row, c) for c in range(start_bottom, end_bottom + 1)}
        coords |= {(r, anchor) for r in range(top_row, bottom_row + 1)}

        gap_rows = [r for r in range(top_row + 1, bottom_row) if r not in (top_row, bottom_row)]
        if gap_rows and random.random() < 0.7:
            rr = random.choice(gap_rows)
            side = random.choice([-1, 1])
            cc = anchor + side
            if 1 <= cc <= w - 2:
                coords.add((rr, cc))
        if gap_rows and random.random() < 0.35:
            rr = random.choice(gap_rows)
            side = random.choice([-1, 1])
            cc = anchor + side
            if 1 <= cc <= w - 2:
                coords.add((rr, cc))
        return coords
    return None


def _make_gap_object(w: int, rows: list[int]) -> set[tuple[int, int]] | None:
    if not rows:
        return None
    for _ in range(50):
        band_h = random.randint(1, min(3, len(rows)))
        start_idx = random.randint(0, len(rows) - band_h)
        picked_rows = rows[start_idx : start_idx + band_h]

        first_len = random.randint(2, 5)
        first_start = random.randint(1, w - first_len - 1)
        segments = [(first_start, first_len)]
        prev_start, prev_len = first_start, first_len

        ok = True
        for _ in picked_rows[1:]:
            cur_len = random.randint(1, 5)
            overlap_left = max(1, prev_start - cur_len + 1)
            overlap_right = min(w - cur_len - 1, prev_start + prev_len - 1)
            if overlap_left > overlap_right:
                ok = False
                break
            cur_start = random.randint(overlap_left, overlap_right)
            segments.append((cur_start, cur_len))
            prev_start, prev_len = cur_start, cur_len
        if not ok:
            continue

        coords = set()
        for r, (start, length) in zip(picked_rows, segments):
            for c in range(start, start + length):
                coords.add((r, c))
        if len(coords) >= 3:
            return coords
    return None


def _place_object(grid: np.ndarray, blocked: np.ndarray, coords: set[tuple[int, int]]) -> None:
    for r, c in coords:
        grid[r, c] = OBJECT_COLOR
    _mark_blocked(blocked, coords)


def _classify_component(coords: np.ndarray, row_colors: list[int | None], color_a: int, color_b: int):
    count_a, count_b = _vote_counts(coords, row_colors, color_a, color_b)
    if count_a == count_b == 0:
        return "zero", count_a, count_b
    if count_a == count_b:
        return "tie", count_a, count_b
    if count_a > count_b:
        return color_a, count_a, count_b
    return color_b, count_a, count_b


def _validate_example(grid: np.ndarray) -> bool:
    g, bg, object_color, seeds = _parse_input(grid)
    h, _w = g.shape
    row_colors, gap = _stripe_row_colors(h, seeds)
    assert gap >= 2

    color_a, color_b = int(seeds[0][0]), int(seeds[1][0])
    seed_rows = {int(seeds[0][1]), int(seeds[1][1])}
    comps = _components_of_color(g, object_color)
    assert 5 <= len(comps) <= 8

    counts = Counter()
    uses_repeated_stripe = False
    for coords in comps:
        label, _ca, _cb = _classify_component(coords, row_colors, color_a, color_b)
        counts[label] += 1
        covered_rows = {int(r) for r, _c in coords if row_colors[int(r)] is not None}
        if any(r not in seed_rows for r in covered_rows):
            uses_repeated_stripe = True

    assert counts[color_a] >= 1
    assert counts[color_b] >= 1
    assert counts["tie"] >= 1
    assert counts["zero"] >= 1
    assert uses_repeated_stripe

    out = main(g)
    assert out.shape == g.shape
    assert not np.array_equal(out, g)
    assert np.all((out != bg) <= (g != bg))

    for color, r, c in seeds:
        assert int(out[r, c]) == int(color)

    saw_recolored_a = False
    saw_recolored_b = False
    saw_erased = False
    for coords in comps:
        label, _ca, _cb = _classify_component(coords, row_colors, color_a, color_b)
        vals = {int(out[r, c]) for r, c in coords}
        assert len(vals) == 1
        target = next(iter(vals))
        if label == "zero" or label == "tie":
            assert target == bg
            saw_erased = True
        elif label == color_a:
            assert target == color_a
            saw_recolored_a = True
        else:
            assert target == color_b
            saw_recolored_b = True

    assert saw_recolored_a and saw_recolored_b and saw_erased
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)

    max_gap = min(5, (h - 2) // 4)
    if max_gap < 2:
        return None
    gap = random.randint(2, max_gap)
    top_seed_row = random.randint(1, h - 3 * gap - 1)
    bottom_seed_row = top_seed_row + gap
    side_col = random.choice([0, w - 1])
    seed_color_a, seed_color_b = random.sample(SEED_COLORS, 2)

    grid = np.full((h, w), BG, dtype=int)
    grid[top_seed_row, side_col] = seed_color_a
    grid[bottom_seed_row, side_col] = seed_color_b

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, {(top_seed_row, side_col), (bottom_seed_row, side_col)})

    seeds = [(seed_color_a, top_seed_row, side_col), (seed_color_b, bottom_seed_row, side_col)]
    row_colors, _ = _stripe_row_colors(h, seeds)

    base_pair = (top_seed_row, bottom_seed_row)
    middle_pair = (bottom_seed_row, top_seed_row + 2 * gap)
    repeat_pair = (top_seed_row + 2 * gap, bottom_seed_row + 2 * gap)
    gap_bands = [
        list(range(top_seed_row + 1, bottom_seed_row)),
        list(range(bottom_seed_row + 1, top_seed_row + 2 * gap)),
        list(range(top_seed_row + 2 * gap + 1, bottom_seed_row + 2 * gap)),
    ]
    gap_bands = [band for band in gap_bands if band]

    specs = [
        ("pair", base_pair, random.choice([seed_color_a, seed_color_b])),
        ("pair", repeat_pair, seed_color_b if random.random() < 0.5 else seed_color_a),
        ("pair", middle_pair, "tie"),
        ("gap", random.choice(gap_bands), "zero"),
    ]
    mandatory_winners = {spec[2] for spec in specs if spec[0] == "pair"}
    if seed_color_a not in mandatory_winners:
        specs.append(("pair", random.choice([base_pair, repeat_pair]), seed_color_a))
    if seed_color_b not in mandatory_winners:
        specs.append(("pair", random.choice([base_pair, repeat_pair]), seed_color_b))

    extra_count = random.randint(1, 2)
    pair_choices = [base_pair, middle_pair, repeat_pair]
    for _ in range(extra_count):
        kind = random.choice(["pair", "pair", "gap"])
        if kind == "pair":
            specs.append(("pair", random.choice(pair_choices), random.choice([seed_color_a, seed_color_b, "tie"])))
        else:
            specs.append(("gap", random.choice(gap_bands), "zero"))

    if not (5 <= len(specs) <= 8):
        return None

    for kind, payload, winner in specs:
        placed = False
        for _ in range(80):
            if kind == "pair":
                row0, row1 = payload
                top_row, bottom_row = sorted((row0, row1))
                top_color = row_colors[top_row]
                bottom_color = row_colors[bottom_row]
                if top_color is None or bottom_color is None:
                    continue
                coords = _make_two_stripe_object(h, w, top_row, bottom_row, top_color, bottom_color, winner)
            else:
                coords = _make_gap_object(w, payload)
            if coords is None or not _can_place(coords, blocked):
                continue
            _place_object(grid, blocked, coords)
            placed = True
            break
        if not placed:
            return None

    try:
        _validate_example(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-stripe input")
