import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GENERATION_RETRIES = 500


def _parse_legend(row: np.ndarray) -> dict[tuple[int, int], int]:
    values = [int(x) for x in row if int(x) != BG]
    assert values and len(values) % 3 == 0, "legend row must contain one or more [A, C, B] triplets"

    mixes: dict[tuple[int, int], int] = {}
    for i in range(0, len(values), 3):
        a, mixed, b = values[i : i + 3]
        assert a != b, "legend endpoints must describe two different line colors"
        key = tuple(sorted((a, b)))
        if key in mixes:
            assert mixes[key] == mixed, "duplicate legend entries must agree"
        mixes[key] = mixed
    return mixes


def _endpoint_pairs(body: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    pairs: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(body != BG):
        color = int(body[r, c])
        pairs.setdefault(color, []).append((int(r), int(c)))

    assert pairs, "expected at least one colored marker pair"
    for color, coords in pairs.items():
        assert len(coords) == 2, f"color {color} must appear exactly twice"
    return pairs


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise AssertionError("paired markers must align horizontally or vertically")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair and recolor line crossings via the bottom mixing legend.

    Input structure:
    - The background is black.
    - Above the bottom row, every non-black color appears exactly twice as single-pixel markers.
    - Each same-color marker pair lies on one shared row or one shared column.
    - The bottom row is a legend made of one or more [A, C, B] color triplets.

    Rule:
    - Treat the last row as a preserved legend and work on all rows above it.
    - For each color pair, draw the full horizontal or vertical segment between its two markers, including endpoints.
    - Whenever two different segments share one pixel, replace that pixel by the legend's mixing color for the unordered pair of segment colors.
    - Leave the legend row unchanged.

    Key insight:
    - The visible dots are endpoints of hidden straight connectors, and the bottom-row triplets are a lookup table that determines the crossing color instead of simple overpainting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    body = g[:-1]
    legend = g[-1]
    mixes = _parse_legend(legend)
    pairs = _endpoint_pairs(body)

    claims: dict[tuple[int, int], list[int]] = {}
    for color, coords in pairs.items():
        a, b = coords
        for cell in _segment_cells(a, b):
            claims.setdefault(cell, []).append(int(color))

    out = np.full_like(body, BG)
    for (r, c), colors in claims.items():
        unique = sorted(set(colors))
        assert 1 <= len(unique) <= 2, "puzzle structure only supports pairwise crossings"
        if len(unique) == 1:
            out[r, c] = unique[0]
        else:
            key = tuple(unique)
            assert key in mixes, f"missing legend entry for crossing colors {key}"
            out[r, c] = mixes[key]

    return np.vstack([out, legend[None, :]])


def _choose_count(rng: np.random.Generator) -> tuple[int, int]:
    layouts = ((1, 1), (1, 2), (2, 1), (2, 2))
    return layouts[int(rng.integers(0, len(layouts)))]


def _sample_positions(
    rng: np.random.Generator,
    low: int,
    high: int,
    count: int,
    min_gap: int = 2,
) -> list[int] | None:
    candidates = np.arange(low, high + 1)
    if len(candidates) < count:
        return None
    for _ in range(100):
        picks = sorted(int(x) for x in rng.choice(candidates, size=count, replace=False))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _legend_row(
    rng: np.random.Generator,
    width: int,
    entries: list[tuple[int, int, int]],
) -> np.ndarray | None:
    values: list[int] = []
    for i, entry in enumerate(entries):
        values.extend(entry)
        if i != len(entries) - 1:
            values.append(BG)

    if len(values) > width:
        return None

    row = np.zeros(width, dtype=int)
    start = int(rng.integers(0, width - len(values) + 1))
    row[start : start + len(values)] = values
    return row


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    horiz_count, vert_count = _choose_count(rng)
    cross_count = horiz_count * vert_count
    legend_min_width = 4 * cross_count - 1

    body_h = int(rng.integers(7, 18))
    min_body_w = max(7, legend_min_width)
    body_w = int(rng.integers(min_body_w, 29))

    rows = _sample_positions(rng, 1, body_h - 2, horiz_count)
    cols = _sample_positions(rng, 1, body_w - 2, vert_count)
    if rows is None or cols is None:
        return None

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    if min_row < 1 or max_row > body_h - 2 or min_col < 1 or max_col > body_w - 2:
        return None

    needed_colors = horiz_count + vert_count + cross_count
    if needed_colors > len(COLOR_POOL):
        return None
    colors = list(int(x) for x in rng.choice(np.array(COLOR_POOL), size=needed_colors, replace=False))
    line_colors = colors[: horiz_count + vert_count]
    mix_colors = colors[horiz_count + vert_count :]

    body = np.zeros((body_h, body_w), dtype=int)
    horizontal_specs: list[dict[str, int]] = []
    vertical_specs: list[dict[str, int]] = []

    for i, row in enumerate(rows):
        left = int(rng.integers(0, min_col))
        right = int(rng.integers(max_col + 1, body_w))
        color = line_colors[i]
        body[row, left] = color
        body[row, right] = color
        horizontal_specs.append({"row": row, "left": left, "right": right, "color": color})

    for j, col in enumerate(cols):
        top = int(rng.integers(0, min_row))
        bottom = int(rng.integers(max_row + 1, body_h))
        color = line_colors[horiz_count + j]
        body[top, col] = color
        body[bottom, col] = color
        vertical_specs.append({"top": top, "bottom": bottom, "col": col, "color": color})

    legend_entries: list[tuple[int, int, int]] = []
    mix_index = 0
    for h in horizontal_specs:
        for v in vertical_specs:
            a = int(h["color"])
            b = int(v["color"])
            mixed = mix_colors[mix_index]
            mix_index += 1
            if rng.random() < 0.5:
                legend_entries.append((a, mixed, b))
            else:
                legend_entries.append((b, mixed, a))
    rng.shuffle(legend_entries)

    legend = _legend_row(rng, body_w, legend_entries)
    if legend is None:
        return None

    grid = np.vstack([body, legend[None, :]])
    out = main(grid)

    body_out = out[:-1]
    if np.array_equal(body, body_out):
        return None

    mixed_pixels = {
        int(body_out[h["row"], v["col"]])
        for h in horizontal_specs
        for v in vertical_specs
    }
    if len(mixed_pixels) != cross_count:
        return None
    if any(pixel in {spec["color"] for spec in horizontal_specs + vertical_specs} for pixel in mixed_pixels):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
