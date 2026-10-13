import random

import numpy as np


PALETTE = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ordered_nonzero_colors(values) -> list[int]:
    seen = set()
    ordered = []
    for value in values:
        color = int(value)
        if color == 0 or color in seen:
            continue
        seen.add(color)
        ordered.append(color)
    return ordered


def _ordered_band_colors(band: np.ndarray) -> list[int]:
    return _ordered_nonzero_colors(band.flatten())


def _ordered_marker_colors(marker_row: np.ndarray) -> list[int]:
    return _ordered_nonzero_colors(marker_row)


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    mask_a = patch == int(a)
    mask_b = patch == int(b)
    out[mask_a] = int(b)
    out[mask_b] = int(a)
    return out


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    assert bg == 0, "expected black background"

    nonempty_rows = np.any(g != bg, axis=1)
    assert bool(nonempty_rows[0]), "band must start at the top row"

    band_h = 0
    while band_h < g.shape[0] and bool(nonempty_rows[band_h]):
        band_h += 1
    assert band_h >= 1, "expected a non-empty top band"

    band = g[:band_h]
    assert np.all(band != bg), "band rows must be fully colored"
    band_colors = _ordered_band_colors(band)
    assert len(band_colors) >= 2, "band must contain at least two colors"

    marker_row = None
    for r in range(band_h, g.shape[0]):
        if bool(nonempty_rows[r]):
            marker_row = r
            break
    assert marker_row is not None, "expected a marker row below the band"
    assert marker_row > band_h, "expected at least one blank row between band and marker"
    assert not np.any(nonempty_rows[band_h:marker_row]), "gap rows must be empty"
    assert not np.any(nonempty_rows[marker_row + 1 :]), "rows below the marker must be empty"

    marker = g[marker_row]
    marker_positions = np.flatnonzero(marker != bg)
    assert marker_positions.size >= 2, "marker row must contain sparse colored pixels"
    assert np.all(np.diff(marker_positions) >= 2), "marker pixels must be isolated"

    marker_colors = _ordered_marker_colors(marker)
    assert len(marker_colors) == 2, "marker row must contain exactly two colors"
    assert not (set(marker_colors) & set(band_colors)), "marker colors must not appear in the band"

    reflect_start = marker_row + 1
    reflect_end = reflect_start + band_h
    assert reflect_end <= g.shape[0], "reflected band must fit below the marker"

    return g, marker_row, band, band_colors, marker_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the top color band across the marker row and swap the marker-coded band colors inside the reflected copy.

    Input structure:
    - The grid has a black background, a full-width colored band occupying the top consecutive non-black rows, and one sparse marker row below it.
    - The marker row is the first non-black row below the band, with isolated pixels in exactly two colors that do not appear in the band.
    - All rows between the band and the marker are black, and all rows below the marker are black in the input.

    Rule:
    - Read the two marker colors from left to right as M1 then M2.
    - Read the distinct band colors from top-left to bottom-right as C1, C2, ...
    - Reflect the whole band vertically across the marker row so the last band row lands immediately below the marker.
    - In that reflected copy only, swap every C1 pixel with C2 and every C2 pixel with C1; leave any other band colors unchanged.
    - Keep the original band and the marker row unchanged.

    Key insight:
    - The same sparse marker row plays two roles at once: its position is the reflection axis, while its two-color composition signals that the reflected copy must exchange the first two band colors in scan order.
    """
    g, marker_row, band, band_colors, marker_colors = _parse_grid(grid)
    assert len(marker_colors) == 2
    c1, c2 = band_colors[:2]

    reflected = band[::-1].copy()
    swapped = _swap_colors(reflected, c1, c2)

    out = g.copy()
    out[marker_row + 1 : marker_row + 1 + band.shape[0], :] = swapped
    return out


def _generate_band(height: int, width: int, ordered_colors: list[int]) -> np.ndarray:
    n_colors = len(ordered_colors)
    band = np.zeros((height, width), dtype=int)
    mode = random.choice(["checker", "shifted", "segments", "motif"])

    if mode == "checker":
        row_step = random.randint(1, n_colors)
        for r in range(height):
            phase = random.randint(0, n_colors - 1)
            for c in range(width):
                idx = (phase + c + r * row_step) % n_colors
                band[r, c] = ordered_colors[idx]
    elif mode == "shifted":
        base = [ordered_colors[(c // random.randint(1, 3)) % n_colors] for c in range(width)]
        for r in range(height):
            shift = random.randint(0, width - 1)
            band[r] = np.roll(base, shift + r * random.randint(1, 3))
    elif mode == "segments":
        for r in range(height):
            cuts = sorted(random.sample(range(1, width), n_colors - 1))
            spans = [0] + cuts + [width]
            row_order = ordered_colors[:]
            random.shuffle(row_order)
            for idx in range(n_colors):
                band[r, spans[idx] : spans[idx + 1]] = row_order[idx]
    else:
        motif_len = random.randint(n_colors + 1, min(width, n_colors + 5))
        motif = [random.choice(ordered_colors) for _ in range(motif_len)]
        for idx, color in enumerate(ordered_colors):
            motif[idx] = color
        motif = np.array(motif, dtype=int)
        for r in range(height):
            shift = random.randint(0, motif_len - 1)
            row = np.roll(motif, shift + r * random.randint(1, 2))
            band[r] = np.resize(row, width)

    for idx, color in enumerate(ordered_colors):
        band[0, idx] = color

    for color in ordered_colors:
        if int(np.sum(band == color)) < 3:
            cols = random.sample(range(width), k=min(width, 3))
            rows = random.sample(range(height), k=min(height, 2))
            for r in rows:
                for c in cols:
                    band[r, c] = color

    return band


def _marker_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(width))
    random.shuffle(cols)
    chosen = []
    for col in cols:
        if all(abs(col - prev) >= 2 for prev in chosen):
            chosen.append(col)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _generate_marker_row(width: int, ordered_colors: list[int]) -> np.ndarray | None:
    total = random.randint(4, min(8, (width + 1) // 2))
    cols = _marker_columns(width, total)
    if cols is None:
        return None

    row = np.zeros(width, dtype=int)
    sequence = [ordered_colors[0], ordered_colors[1]]
    for _ in range(total - 2):
        sequence.append(random.choice(ordered_colors))
    tail = sequence[1:]
    random.shuffle(tail)
    sequence = [ordered_colors[0], *tail]
    if ordered_colors[1] not in sequence[1:]:
        sequence[-1] = ordered_colors[1]

    for col, color in zip(cols, sequence):
        row[col] = color
    return row


def _plain_reflection_output(grid: np.ndarray) -> np.ndarray:
    g, marker_row, band, band_colors, _marker_colors = _parse_grid(grid)
    out = g.copy()
    out[marker_row + 1 : marker_row + 1 + band.shape[0], :] = band[::-1]
    return out


def _nonreflected_swap_output(grid: np.ndarray) -> np.ndarray:
    g, marker_row, band, band_colors, _marker_colors = _parse_grid(grid)
    c1, c2 = band_colors[:2]
    out = g.copy()
    out[marker_row + 1 : marker_row + 1 + band.shape[0], :] = _swap_colors(band, c1, c2)
    return out


def _validate_example(grid: np.ndarray) -> bool:
    try:
        g, marker_row, band, band_colors, marker_colors = _parse_grid(grid)
        c1, c2 = band_colors[:2]
        reflected = band[::-1]
        swapped = _swap_colors(reflected, c1, c2)
        if np.array_equal(swapped, reflected):
            return False
        if np.array_equal(band, reflected):
            return False
        if np.sum(band == c1) < 3 or np.sum(band == c2) < 3:
            return False

        out = main(g)
        if np.array_equal(out, _plain_reflection_output(g)):
            return False
        if np.array_equal(out, _nonreflected_swap_output(g)):
            return False
        expected_patch = swapped
        actual_patch = out[marker_row + 1 : marker_row + 1 + band.shape[0], :]
        if not np.array_equal(actual_patch, expected_patch):
            return False
        return True
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    band_h = random.randint(3, 5)
    gap = random.randint(1, 3)
    min_height = band_h + gap + 1 + band_h
    if min_height > 30:
        return None

    height = random.randint(min_height, min(30, min_height + 6))
    width = random.randint(11, 22)

    band_color_count = 3 if random.random() < 0.65 else 2
    band_colors = random.sample(PALETTE, band_color_count)
    random.shuffle(band_colors)
    marker_colors = random.sample([c for c in PALETTE if c not in band_colors], 2)
    if random.random() < 0.5:
        marker_colors.reverse()

    band = _generate_band(band_h, width, band_colors)
    if np.any(band == 0):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:band_h] = band

    marker_row_idx = band_h + gap
    marker_row = _generate_marker_row(width, marker_colors)
    if marker_row is None:
        return None
    grid[marker_row_idx] = marker_row

    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
