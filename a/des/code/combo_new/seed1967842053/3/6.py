import random

import numpy as np


BLACK = 0


def _same_color_run_end(row, start):
    color = row[start]
    end = start
    while end + 1 < row.size and row[end + 1] == color:
        end += 1
    return end


def _color_runs(values):
    runs = []
    if values.size == 0:
        return runs
    start = 0
    color = values[0]
    for idx in range(1, values.size):
        if values[idx] != color:
            runs.append((start, idx - 1, int(color)))
            start = idx
            color = values[idx]
    runs.append((start, values.size - 1, int(color)))
    return runs


def _first_nonzero_interval(row, min_col):
    cols = np.flatnonzero(row[min_col:] != BLACK)
    if cols.size == 0:
        return None
    start = int(cols[0] + min_col)
    end = int(cols[-1] + min_col)
    return start, end


def main(grid: np.ndarray) -> np.ndarray:
    """Sort right-side multicolor stripe bands by the left priority tower and flip bands whose key segment has even length.

    Input structure:
    The background is black.  A vertical priority tower sits on the left as one-row blocks of distinct colors, read from top to bottom.  On the right are one-row horizontal multicolor stripe bands, separated by black rows; each band has exactly one contiguous segment whose color appears in the tower.

    Rule:
    Detect the tower order, find each stripe band's tower-colored key segment, and sort all bands by that key color's tower rank.  While moving a band into sorted order, keep its left-right order if the key segment length is odd, and mirror the whole band horizontally if that length is even.  The output keeps the same canvas, tower, stripe columns, and black separator rows.

    Key insight:
    The tower supplies only the inter-row priority; the hidden independent condition is that the parity of the key segment length controls whether the row itself is mirrored.
    """
    arr = np.asarray(grid)
    out = np.zeros_like(arr)
    nz = np.argwhere(arr != BLACK)
    if nz.size == 0:
        return out

    min_col = int(nz[:, 1].min())
    tower_rows = [r for r in range(arr.shape[0]) if arr[r, min_col] != BLACK]
    if not tower_rows:
        return arr.copy()

    priority = []
    seen = set()
    for r in tower_rows:
        color = int(arr[r, min_col])
        if color not in seen:
            priority.append(color)
            seen.add(color)
        end = _same_color_run_end(arr[r], min_col)
        out[r, min_col : end + 1] = arr[r, min_col : end + 1]

    rank = {color: idx for idx, color in enumerate(priority)}
    tower_end = max(_same_color_run_end(arr[r], min_col) for r in tower_rows)

    stripes = []
    for r in range(arr.shape[0]):
        interval = _first_nonzero_interval(arr[r], tower_end + 1)
        if interval is None:
            continue
        start, end = interval
        values = arr[r, start : end + 1].copy()
        if np.any(values == BLACK):
            used = np.flatnonzero(values != BLACK)
            start += int(used[0])
            end = start + int(used[-1] - used[0])
            values = arr[r, start : end + 1].copy()

        key = None
        for a, b, color in _color_runs(values):
            if color in rank:
                key = (rank[color], b - a + 1)
                break
        if key is None:
            continue
        ordered = values if key[1] % 2 == 1 else values[::-1]
        stripes.append(
            {
                "row": r,
                "start": start,
                "values": ordered.copy(),
                "rank": key[0],
            }
        )

    target_rows = sorted(s["row"] for s in stripes)
    stripes.sort(key=lambda item: (item["rank"], item["row"]))
    for target_row, stripe in zip(target_rows, stripes):
        start = stripe["start"]
        values = stripe["values"]
        out[target_row, start : start + values.size] = values

    return out


def _choose_segment_colors(segment_count, key_pos, key_color, filler_colors):
    colors = []
    for idx in range(segment_count):
        if idx == key_pos:
            colors.append(key_color)
            continue
        choices = [c for c in filler_colors if not colors or c != colors[-1]]
        if idx + 1 == key_pos and key_color in choices and len(choices) > 1:
            choices = [c for c in choices if c != key_color]
        colors.append(random.choice(choices))
    return colors


def _make_stripe(key_color, filler_colors, max_width, odd_key):
    while True:
        segment_count = random.randint(3, 5)
        key_pos = random.randrange(segment_count)
        key_len = random.choice([1, 3, 5] if odd_key else [2, 4])
        lengths = [random.randint(1, 4) for _ in range(segment_count)]
        lengths[key_pos] = key_len
        if sum(lengths) > max_width:
            continue
        colors = _choose_segment_colors(segment_count, key_pos, key_color, filler_colors)
        stripe = []
        for color, length in zip(colors, lengths):
            stripe.extend([color] * length)
        return np.array(stripe, dtype=int)


def generate_input() -> np.ndarray:
    n = random.randint(3, 6)
    palette = list(range(1, 10))
    tower_colors = random.sample(palette, n)
    filler_colors = [c for c in palette if c not in tower_colors]

    height_needed = max(n, 2 * n - 1)
    height = random.randint(height_needed, 30)
    stripe_top = random.randint(0, height - (2 * n - 1))
    tower_top = random.randint(0, height - n)

    left_margin = random.randint(0, 2)
    tower_width = random.randint(1, 3)
    gap = random.randint(1, 3)
    stripe_start = left_margin + tower_width + gap
    right_margin = random.randint(0, 2)
    max_width = 30 - stripe_start - right_margin
    max_width = max(8, max_width)

    parities = [False, True] + [bool(random.getrandbits(1)) for _ in range(n - 2)]
    random.shuffle(parities)

    stripe_order = tower_colors[:]
    random.shuffle(stripe_order)
    if stripe_order == tower_colors and n > 1:
        stripe_order[0], stripe_order[1] = stripe_order[1], stripe_order[0]

    stripes = [
        _make_stripe(color, filler_colors, max_width, odd_key=parity)
        for color, parity in zip(stripe_order, parities)
    ]
    width = random.randint(stripe_start + max(len(s) for s in stripes) + right_margin, 30)

    grid = np.zeros((height, width), dtype=int)
    for idx, color in enumerate(tower_colors):
        row = tower_top + idx
        grid[row, left_margin : left_margin + tower_width] = color

    for idx, stripe in enumerate(stripes):
        row = stripe_top + 2 * idx
        grid[row, stripe_start : stripe_start + stripe.size] = stripe

    return grid
