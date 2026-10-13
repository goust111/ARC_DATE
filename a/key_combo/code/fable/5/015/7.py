from __future__ import annotations

import random

import numpy as np


def _consecutive_groups(rows: list[int]) -> list[tuple[int, int]]:
    """Turn sorted row indices into inclusive consecutive intervals."""
    groups: list[tuple[int, int]] = []
    for row in rows:
        if not groups or row != groups[-1][1] + 1:
            groups.append((row, row))
        else:
            groups[-1] = (groups[-1][0], row)
    return groups


def _band_slices(grid: np.ndarray) -> tuple[int, list[slice]]:
    """Find the unique full-row separator color and the equal-height bands."""
    height, _width = grid.shape
    full_rows: dict[int, list[int]] = {}
    for r in range(height):
        color = int(grid[r, 0])
        if bool(np.all(grid[r, :] == color)):
            full_rows.setdefault(color, []).append(r)

    candidates: list[tuple[int, list[slice]]] = []
    for color, rows in full_rows.items():
        groups = _consecutive_groups(rows)
        if not groups:
            continue

        slices: list[slice] = []
        start = 0
        valid = True
        for r0, r1 in groups:
            if r0 <= start:
                valid = False
                break
            slices.append(slice(start, r0))
            start = r1 + 1
        if not valid or start >= height:
            continue
        slices.append(slice(start, height))

        band_heights = [int(s.stop - s.start) for s in slices]
        separator_heights = [r1 - r0 + 1 for r0, r1 in groups]
        if (
            len(slices) >= 2
            and len(set(band_heights)) == 1
            and len(set(separator_heights)) == 1
        ):
            candidates.append((color, slices))

    assert len(candidates) == 1, "separator rows must uniquely split equal-height horizontal bands"
    return candidates[0]


def _singleton_pixel(strip: np.ndarray) -> tuple[int, int, int]:
    """Return (color, row, col) for the strip's unique frequency-one pixel."""
    colors, counts = np.unique(strip, return_counts=True)
    singleton_colors = colors[counts == 1]
    assert len(singleton_colors) == 1, "each band must contain exactly one singleton color"

    color = int(singleton_colors[0])
    coords = np.argwhere(strip == color)
    assert coords.shape == (1, 2)
    return color, int(coords[0, 0]), int(coords[0, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore independently scrambled horizontal bands by vertically aligning their hidden singleton-color anchors.

    Input structure:
    - Uniform horizontal separator lines divide the grid into equal-height bands.
    - Within every band, exactly one color occurs once; that pixel is the hidden anchor.
    - The lower bands are independently cyclically shifted horizontally, while the top band is unchanged.

    Rule:
    - Count colors separately over every complete band, excluding only the separator lines.
    - Keep the top band fixed and cyclically shift each lower whole band until its singleton pixel shares the top anchor's column.
    - Preserve separator lines and every pixel color.

    Key insight:
    - The anchor is defined by within-band frequency rather than a fixed color, so anchor alignment uniquely reveals each inverse cyclic shift.
    """
    g = np.asarray(grid)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0

    _separator_color, bands = _band_slices(g)
    anchors = [_singleton_pixel(g[band, :]) for band in bands]
    target_col = anchors[0][2]

    out = g.copy()
    for band, (_color, _row, anchor_col) in zip(bands[1:], anchors[1:]):
        shift = (target_col - anchor_col) % g.shape[1]
        out[band, :] = np.roll(g[band, :], shift=shift, axis=1)
    return out


# ------------------------------- generator ---------------------------------

_PALETTE = tuple(range(10))
_GENERATION_ATTEMPTS = 300


def _random_position(available: set[tuple[int, int]], rng: random.Random) -> tuple[int, int]:
    pos = rng.choice(tuple(available))
    available.remove(pos)
    return pos


def _place_random(
    strip: np.ndarray,
    available: set[tuple[int, int]],
    color: int,
    count: int,
    rng: random.Random,
) -> None:
    for _ in range(count):
        r, c = _random_position(available, rng)
        strip[r, c] = color


def _place_separated_pair(
    strip: np.ndarray,
    available: set[tuple[int, int]],
    color: int,
    rng: random.Random,
) -> bool:
    """Place two same-colored pixels that are not even diagonally adjacent."""
    first_choices = list(available)
    rng.shuffle(first_choices)
    for first in first_choices:
        seconds = [
            pos
            for pos in available
            if pos != first and max(abs(pos[0] - first[0]), abs(pos[1] - first[1])) > 1
        ]
        if seconds:
            second = rng.choice(seconds)
            available.remove(first)
            available.remove(second)
            strip[first] = color
            strip[second] = color
            return True
    return False


def _make_aligned_strip(
    band_height: int,
    width: int,
    background: int,
    anchor_colors: list[int],
    own_index: int,
    motif_colors: tuple[int, int],
    anchor_col: int,
    rng: random.Random,
) -> tuple[np.ndarray, int]:
    """Build one sparse band whose only frequency-one color is its own anchor."""
    strip = np.full((band_height, width), background, dtype=int)
    anchor_row = rng.randrange(band_height)
    own_color = anchor_colors[own_index]
    strip[anchor_row, anchor_col] = own_color

    available = {
        (r, c)
        for r in range(band_height)
        for c in range(width)
        if (r, c) != (anchor_row, anchor_col)
    }

    # Make the hidden anchor locally non-isolated: uniqueness is statistical,
    # not a connected-component or neighborhood clue.
    neighbor_col = anchor_col + 1 if anchor_col + 1 < width else anchor_col - 1
    available.remove((anchor_row, neighbor_col))
    strip[anchor_row, neighbor_col] = motif_colors[0]

    # Every other band's anchor color occurs twice here.  Thus no anchor color
    # is a global singleton, and each decoy pair also defeats local-isolation shortcuts.
    for i, color in enumerate(anchor_colors):
        if i != own_index and not _place_separated_pair(strip, available, color, rng):
            raise ValueError("could not place separated anchor-color decoys")

    # Both motif colors have repeated occurrences in every band.
    _place_random(strip, available, motif_colors[0], rng.randint(2, 4), rng)
    _place_random(strip, available, motif_colors[1], rng.randint(2, 5), rng)

    # Avoid uniform background rows, which could masquerade as separators.
    for r in range(band_height):
        if bool(np.all(strip[r, :] == background)):
            choices = [pos for pos in available if pos[0] == r]
            if not choices:
                raise ValueError("could not break a uniform band row")
            pos = rng.choice(choices)
            available.remove(pos)
            strip[pos] = motif_colors[1]

    colors, counts = np.unique(strip, return_counts=True)
    singletons = colors[counts == 1]
    if singletons.tolist() != [own_color]:
        raise ValueError("constructed band does not have the intended singleton")
    if int(np.count_nonzero(strip == background)) < 2:
        raise ValueError("background became a singleton")
    if any(bool(np.all(strip[r, :] == strip[r, 0])) for r in range(band_height)):
        raise ValueError("constructed band contains a uniform row")
    return strip, anchor_row


def _compose(
    strips: list[np.ndarray], separator_color: int, separator_height: int = 1
) -> np.ndarray:
    width = strips[0].shape[1]
    separator = np.full((separator_height, width), separator_color, dtype=int)
    pieces: list[np.ndarray] = [strips[0]]
    for strip in strips[1:]:
        pieces.extend((separator.copy(), strip))
    return np.vstack(pieces)


def _sample_distinct_shifts(count: int, width: int, rng: random.Random) -> list[int]:
    """Choose distinct nonzero shifts, including short and long cyclic moves."""
    assert count >= 2
    low = list(range(1, width // 2 + 1))
    high = list(range(width // 2 + 1, width))
    shifts = [rng.choice(low), rng.choice(high)]
    remaining = [s for s in range(1, width) if s not in shifts]
    shifts.extend(rng.sample(remaining, count - 2))
    rng.shuffle(shifts)
    return shifts


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    aligned: list[np.ndarray],
    scrambled: list[np.ndarray],
    anchor_colors: list[int],
    anchor_rows: list[int],
    anchor_col: int,
    shifts: list[int],
    separator_color: int,
) -> bool:
    """Check rule-level invariants independently of the solver implementation."""
    band_count = len(aligned)
    band_height, width = aligned[0].shape

    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if np.array_equal(inp, expected) or not np.array_equal(scrambled[0], aligned[0]):
        return False
    if len(set(shifts)) != len(shifts) or any(shift == 0 for shift in shifts):
        return False

    # The only full uniform rows are the unchanged separators.
    separator_rows = [i * (band_height + 1) - 1 for i in range(1, band_count)]
    uniform_rows = [r for r in range(inp.shape[0]) if bool(np.all(inp[r, :] == inp[r, 0]))]
    if uniform_rows != separator_rows:
        return False
    if any(not bool(np.all(inp[r, :] == separator_color)) for r in separator_rows):
        return False

    for i, strip in enumerate(scrambled):
        colors, counts = np.unique(strip, return_counts=True)
        singleton_colors = colors[counts == 1]
        if singleton_colors.tolist() != [anchor_colors[i]]:
            return False
        coords = np.argwhere(strip == anchor_colors[i])
        observed = (int(coords[0, 0]), int(coords[0, 1]))
        applied_shift = 0 if i == 0 else shifts[i - 1]
        if observed != (anchor_rows[i], (anchor_col + applied_shift) % width):
            return False
        if not np.array_equal(np.roll(strip, -applied_shift, axis=1), aligned[i]):
            return False

        # Whole-band movement must affect non-anchor content as well.
        if i > 0:
            diff = strip != aligned[i]
            non_anchor_diff = diff & (strip != anchor_colors[i]) & (aligned[i] != anchor_colors[i])
            if int(non_anchor_diff.sum()) < 2:
                return False

    # Anchor colors are deliberately useless under global-frequency reasoning.
    global_counts = {color: int(np.count_nonzero(inp == color)) for color in anchor_colors}
    if len(set(global_counts.values())) != 1 or any(count <= 1 for count in global_counts.values()):
        return False

    # No single common correction can restore every lower band.
    if any(
        all(np.array_equal(np.roll(scrambled[i], correction, axis=1), aligned[i])
            for i in range(1, band_count))
        for correction in range(width)
    ):
        return False

    # The top anchor touches another color, excluding an "isolated pixel" reading.
    ar = anchor_rows[0]
    adjacent_cols = [c for c in (anchor_col - 1, anchor_col + 1) if 0 <= c < width]
    if not any(int(aligned[0][ar, c]) != int(aligned[0][ar, anchor_col]) for c in adjacent_cols):
        return False
    return np.array_equal(expected, _compose(aligned, separator_color))


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    layouts = [
        (band_count, band_height)
        for band_count in range(3, 6)
        for band_height in range(3, 6)
        if band_count * band_height + band_count - 1 <= 30
    ]
    band_count, band_height = rng.choice(layouts)
    width = rng.randint(10, 24)

    separator_color, background = rng.sample(_PALETTE, 2)
    remaining_colors = [c for c in _PALETTE if c not in (separator_color, background)]
    chosen = rng.sample(remaining_colors, band_count + 2)
    anchor_colors = chosen[:band_count]
    motif_colors = (chosen[-2], chosen[-1])

    if rng.random() < 0.4:
        anchor_col = rng.choice([0, 1, width - 2, width - 1])
    else:
        anchor_col = rng.randint(2, width - 3)

    aligned: list[np.ndarray] = []
    anchor_rows: list[int] = []
    try:
        for i in range(band_count):
            strip, anchor_row = _make_aligned_strip(
                band_height,
                width,
                background,
                anchor_colors,
                i,
                motif_colors,
                anchor_col,
                rng,
            )
            aligned.append(strip)
            anchor_rows.append(anchor_row)
    except ValueError:
        return None

    shifts = _sample_distinct_shifts(band_count - 1, width, rng)
    scrambled = [aligned[0].copy()]
    scrambled.extend(
        np.roll(strip, shift=shift, axis=1)
        for strip, shift in zip(aligned[1:], shifts)
    )

    inp = _compose(scrambled, separator_color)
    expected = _compose(aligned, separator_color)
    if not _validate_semantics(
        inp,
        expected,
        aligned,
        scrambled,
        anchor_colors,
        anchor_rows,
        anchor_col,
        shifts,
        separator_color,
    ):
        return None

    try:
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return None
    return inp if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate hidden-anchor alignment input."""
    rng = random.Random()
    for _ in range(_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GENERATION_ATTEMPTS} attempts")
