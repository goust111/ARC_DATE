from __future__ import annotations

import numpy as np


BLUE = 1
TEAL = 8
NON_CARRIER_COLORS = (2, 3, 4, 5, 6, 7, 9)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _row_carriers(grid: np.ndarray) -> list[int]:
    carriers: list[int] = []
    for row in grid:
        blue_count = int(np.count_nonzero(row == BLUE))
        teal_count = int(np.count_nonzero(row == TEAL))
        if blue_count == teal_count:
            raise AssertionError("each row must have a unique carrier-majority color")
        carriers.append(BLUE if blue_count > teal_count else TEAL)
    return carriers


def _bands_from_rows(row_carriers: list[int]) -> list[tuple[int, int, int]]:
    bands: list[tuple[int, int, int]] = []
    start = 0
    for r in range(1, len(row_carriers) + 1):
        if r == len(row_carriers) or row_carriers[r] != row_carriers[start]:
            bands.append((start, r - 1, row_carriers[start]))
            start = r
    return bands


def _segment_slope(coords: np.ndarray) -> str:
    diffs = {int(c) - int(r) for r, c in coords}
    sums = {int(c) + int(r) for r, c in coords}
    if len(diffs) == 1:
        return "down_right"
    if len(sums) == 1:
        return "down_left"
    raise AssertionError("segment must be a single diagonal")


def _band_fill_mask(
    r0: int,
    r1: int,
    width: int,
    slope: str,
    phase: int,
) -> np.ndarray:
    mask = np.zeros((r1 - r0 + 1, width), dtype=bool)
    for rr, r in enumerate(range(r0, r1 + 1)):
        cols = np.arange(width)
        if slope == "down_right":
            mask[rr, :] = ((cols - r) % 3) == phase
        else:
            mask[rr, :] = ((cols + r) % 3) == phase
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each diagonal segment to the carrier edge, then fill its modulo-3 diagonal congruence class inside that carrier band.

    Input structure:
    - The whole grid is partitioned into full-width horizontal carrier bands colored blue (1) and teal (8), alternating by band.
    - Each band contains exactly one non-carrier diagonal segment of a unique color; all segments share the same slope direction.
    - Every segment stays entirely inside its own carrier band and no two bands' segments touch each other.

    Rule:
    - Infer each row's carrier color by majority and group consecutive equal rows into carrier bands.
    - In every band, locate the unique diagonal segment and clear its original cells back to the carrier color.
    - If the carrier is blue, slide the whole segment horizontally right until its rightmost cell reaches the grid's right edge; if teal, slide it left until its leftmost cell reaches column 0.
    - Using the slid segment's diagonal phase as the anchor, color every cell in the same modulo-3 diagonal congruence class within that band's row range, clipping to the band and leaving any already non-carrier cell untouched; bands are processed top-down so upper bands would win any boundary conflict.

    Key insight:
    - The carrier color does not merely choose a direction of motion: sliding first changes the diagonal's modulo-3 phase, and that new phase is what determines the final interlaced fill pattern.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("input must be a 2D grid")

    h, w = g.shape
    row_carriers = _row_carriers(g)
    bands = _bands_from_rows(row_carriers)
    out = g.copy()

    global_slope: str | None = None
    band_payloads: list[tuple[int, int, int, int, np.ndarray, str]] = []

    for r0, r1, carrier in bands:
        band = g[r0 : r1 + 1, :]
        non_carrier_mask = (band != BLUE) & (band != TEAL)
        comps = _components8(non_carrier_mask)
        if len(comps) != 1:
            raise AssertionError("each carrier band must contain exactly one diagonal segment")
        local_coords = comps[0]
        coords = local_coords.copy()
        coords[:, 0] += r0
        segment_color = int(g[coords[0, 0], coords[0, 1]])
        if not np.all(g[coords[:, 0], coords[:, 1]] == segment_color):
            raise AssertionError("each segment must be monochromatic")

        slope = _segment_slope(coords)
        if global_slope is None:
            global_slope = slope
        elif slope != global_slope:
            raise AssertionError("all bands must share one slope direction")

        out[coords[:, 0], coords[:, 1]] = carrier
        band_payloads.append((r0, r1, carrier, segment_color, coords, slope))

    for r0, r1, carrier, segment_color, coords, slope in band_payloads:
        if carrier == BLUE:
            delta = (w - 1) - int(coords[:, 1].max())
        elif carrier == TEAL:
            delta = -int(coords[:, 1].min())
        else:
            raise AssertionError("carrier color must be blue or teal")

        moved = coords.copy()
        moved[:, 1] += delta
        if np.any(moved[:, 1] < 0) or np.any(moved[:, 1] >= w):
            raise AssertionError("slid segment left the grid")

        if slope == "down_right":
            phase = int((moved[0, 1] - moved[0, 0]) % 3)
        else:
            phase = int((moved[0, 1] + moved[0, 0]) % 3)

        fill_mask = _band_fill_mask(r0, r1, w, slope, phase)
        band_view = out[r0 : r1 + 1, :]
        paintable = fill_mask & (band_view == carrier)
        band_view[paintable] = segment_color

    return out


def _segment_coords(r_start: int, length: int, carrier: int, width: int, slope: str) -> np.ndarray:
    rows = np.arange(r_start, r_start + length, dtype=int)
    if slope == "down_right":
        if carrier == BLUE:
            cols = np.arange(width - length, width, dtype=int)
        else:
            cols = np.arange(0, length, dtype=int)
    else:
        if carrier == BLUE:
            cols = np.arange(width - 1, width - length - 1, -1, dtype=int)
        else:
            cols = np.arange(length - 1, -1, -1, dtype=int)
    return np.column_stack([rows, cols])


def _shift_input_coords(moved: np.ndarray, carrier: int, gap: int) -> np.ndarray:
    coords = moved.copy()
    if carrier == BLUE:
        coords[:, 1] -= int(gap)
    else:
        coords[:, 1] += int(gap)
    return coords


def _touching(a: np.ndarray, b: np.ndarray) -> bool:
    for ra, ca in a:
        for rb, cb in b:
            if max(abs(int(ra) - int(rb)), abs(int(ca) - int(cb))) <= 1:
                return True
    return False


def _choose_gap(max_gap: int, residue: int, rng: np.random.Generator) -> int | None:
    choices = [gap for gap in range(1, max_gap + 1) if gap % 3 == residue]
    if not choices:
        choices = list(range(1, max_gap + 1))
    if not choices:
        return None
    return int(choices[int(rng.integers(0, len(choices)))])


def _validate_generated_case(
    grid: np.ndarray,
    bands: list[tuple[int, int, int]],
    segments: list[dict[str, object]],
) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape or not np.any(out != grid):
        return False

    saw_phase_sensitive = False
    saw_short_segment_expansion = False
    width = grid.shape[1]
    for (r0, r1, _carrier), info in zip(bands, segments):
        slope = str(info["slope"])
        moved = np.asarray(info["moved"], dtype=int)
        inp = np.asarray(info["input"], dtype=int)
        length = int(info["length"])
        band_height = int(r1 - r0 + 1)
        if slope == "down_right":
            old_phase = int((inp[0, 1] - inp[0, 0]) % 3)
            new_phase = int((moved[0, 1] - moved[0, 0]) % 3)
        else:
            old_phase = int((inp[0, 1] + inp[0, 0]) % 3)
            new_phase = int((moved[0, 1] + moved[0, 0]) % 3)

        wrong = _band_fill_mask(r0, r1, width, slope, old_phase)
        correct = _band_fill_mask(r0, r1, width, slope, new_phase)
        if not np.array_equal(wrong, correct):
            saw_phase_sensitive = True

        if length < band_height:
            seed_rows = set(int(r) for r in moved[:, 0])
            colored_rows = {
                r
                for r in range(r0, r1 + 1)
                if np.any(out[r, :] == int(info["color"]))
            }
            if any(r not in seed_rows for r in colored_rows):
                saw_short_segment_expansion = True

    if not saw_phase_sensitive:
        return False
    if not saw_short_segment_expansion:
        return False

    color_count = len({int(v) for v in np.unique(grid)} - {BLUE, TEAL})
    return color_count == len(segments)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(400):
        band_count = int(rng.integers(3, 5))
        heights = rng.choice([3, 4], size=band_count, replace=True)
        if 4 not in heights:
            heights[int(rng.integers(0, band_count))] = 4
        height = int(np.sum(heights))
        width = int(rng.integers(10, 16))

        start_carrier = BLUE if int(rng.integers(0, 2)) == 0 else TEAL
        carriers = [start_carrier if i % 2 == 0 else (TEAL if start_carrier == BLUE else BLUE) for i in range(band_count)]
        slope = "down_right" if int(rng.integers(0, 2)) == 0 else "down_left"

        segment_colors = rng.choice(np.array(NON_CARRIER_COLORS, dtype=int), size=band_count, replace=False)
        designated_short_band = int(rng.choice(np.flatnonzero(heights == 4)))

        residues = [1, 2]
        while len(residues) < band_count:
            residues.append(int(rng.integers(0, 3)))
        rng.shuffle(residues)

        grid = np.zeros((height, width), dtype=int)
        bands: list[tuple[int, int, int]] = []
        segments: list[dict[str, object]] = []

        row = 0
        valid = True
        for idx, (band_h, carrier, color, residue) in enumerate(zip(heights, carriers, segment_colors, residues)):
            band_h = int(band_h)
            r0, r1 = row, row + band_h - 1
            row = r1 + 1
            bands.append((r0, r1, int(carrier)))
            grid[r0 : r1 + 1, :] = int(carrier)

            if idx == designated_short_band:
                length = 3
            else:
                length = 3 if band_h == 3 else int(rng.choice([3, 4]))
            if length > band_h:
                valid = False
                break

            moved_options: list[tuple[np.ndarray, np.ndarray, int, int]] = []
            max_gap = width - length
            gap = _choose_gap(max_gap, int(residue), rng)
            if gap is None:
                valid = False
                break

            for r_start in range(r0, r1 - length + 2):
                moved = _segment_coords(r_start, length, int(carrier), width, slope)
                inp = _shift_input_coords(moved, int(carrier), gap)
                if np.any(inp[:, 1] < 0) or np.any(inp[:, 1] >= width):
                    continue
                moved_options.append((inp, moved, r_start, gap))

            rng.shuffle(moved_options)

            placed = False
            for inp, moved, r_start, gap in moved_options:
                if any(_touching(inp, np.asarray(prev["input"], dtype=int)) for prev in segments):
                    continue
                grid[inp[:, 0], inp[:, 1]] = int(color)
                segments.append(
                    {
                        "color": int(color),
                        "carrier": int(carrier),
                        "slope": slope,
                        "length": int(length),
                        "input": inp.copy(),
                        "moved": moved.copy(),
                        "r_start": int(r_start),
                        "gap": int(gap),
                    }
                )
                placed = True
                break

            if not placed:
                valid = False
                break

        if not valid:
            continue

        if _validate_generated_case(grid, bands, segments):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
