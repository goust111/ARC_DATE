from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
HEIGHT_RANGE = (3, 5)
WIDTH_RANGE = (12, 18)
SLOT_COUNT_RANGE = (5, 7)
SHIFT_CHOICES = (-3, -2, -1, 1, 2, 3)
SUBSET_SIZES = (4, 5, 6)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _other_color(color: int, colors: tuple[int, int]) -> int:
    return colors[1] if color == colors[0] else colors[0]


def _binary_min_period(bits: np.ndarray) -> int:
    n = int(bits.size)
    for p in range(1, n):
        if n % p != 0:
            continue
        if all(int(bits[i]) == int(bits[i % p]) for i in range(n)):
            return p
    return n


def _make_band(height: int, width: int, colors: tuple[int, int]) -> np.ndarray | None:
    for _ in range(300):
        period = random.randint(4, min(7, width - 1))
        motif = np.random.randint(0, 2, size=period, dtype=int)
        if motif.min() == motif.max() or _binary_min_period(motif) != period:
            continue

        phases = [random.randrange(period) for _ in range(height)]
        inversions = [random.randrange(2) for _ in range(height)]
        if len(set(phases)) == 1 and len(set(inversions)) == 1:
            continue

        rows = []
        for phase, inv in zip(phases, inversions):
            row = np.resize(np.roll(motif, phase), width)
            if inv:
                row = 1 - row
            rows.append(row)
        band01 = np.asarray(rows, dtype=int)

        if np.any(band01.sum(axis=1) == 0) or np.any(band01.sum(axis=1) == width):
            continue
        if np.any(band01.sum(axis=0) == 0) or np.any(band01.sum(axis=0) == height):
            continue
        if np.array_equal(band01, np.flipud(band01)):
            continue
        if np.array_equal(band01, 1 - np.flipud(band01)):
            continue
        if len({tuple(row.tolist()) for row in band01}) < 2:
            continue

        return np.where(band01 == 0, colors[0], colors[1]).astype(int)
    return None


def _render_output(
    band: np.ndarray,
    band_top: int,
    shift: int,
    out_shape: tuple[int, int],
    colors: tuple[int, int],
) -> np.ndarray:
    height, width = band.shape
    out_h, out_w = out_shape
    out = np.full((out_h, out_w), BG, dtype=int)

    for r in range(out_h):
        k = (r - band_top) // height
        local_r = r - (band_top + k * height)
        src_r = height - 1 - local_r if k % 2 else local_r
        delta = k * shift
        for c in range(out_w):
            base_c = c - delta
            if 0 <= base_c < width:
                color = int(band[src_r, base_c])
                if k % 2:
                    color = _other_color(color, colors)
                out[r, c] = color

    return out


def _matching_shifts(
    band: np.ndarray,
    residues: list[tuple[int, int]],
    residue_values: list[int],
    band_top: int,
    k: int,
    colors: tuple[int, int],
    *,
    use_flip: bool,
    use_swap: bool,
) -> list[int]:
    height, width = band.shape
    slot_top = band_top + k * height
    matches: list[int] = []

    for shift in range(-(width - 1), width):
        delta = k * shift
        ok = True
        for (r, c), value in zip(residues, residue_values):
            local_r = r - slot_top
            if not (0 <= local_r < height):
                ok = False
                break
            src_r = height - 1 - local_r if (k % 2 and use_flip) else local_r
            base_c = c - delta
            if not (0 <= base_c < width):
                ok = False
                break
            expected = int(band[src_r, base_c])
            if k % 2 and use_swap:
                expected = _other_color(expected, colors)
            if expected != int(value):
                ok = False
                break
        if ok:
            matches.append(int(shift))

    return matches


def _find_reference_band(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray, tuple[int, int]]:
    full_rows = np.flatnonzero(np.all(grid != bg, axis=1))
    assert full_rows.size > 0, "expected one complete row band"
    band_top = int(full_rows[0])
    band_bottom = int(full_rows[-1])
    assert np.array_equal(full_rows, np.arange(band_top, band_bottom + 1)), "complete band must be contiguous"

    band = grid[band_top : band_bottom + 1, :].copy()
    colors = tuple(sorted(int(x) for x in np.unique(band)))
    assert bg not in colors and len(colors) == 2, "band must use exactly two foreground colors"
    return band_top, band, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the glide-inverted ribbon from one full row band and a sparse remote residue.

    Input structure:
    - The canvas has background cells, one complete full-width horizontal band, and a few foreground residue pixels elsewhere.
    - The complete band uses exactly two foreground colors.
    - All residue pixels come from one other vertical copy of the same band.

    Rule:
    - Treat the full band as copy 0.
    - Each adjacent copy is made from the previous one by vertical reflection, swapping the two foreground colors, and shifting horizontally by a fixed offset `s`.
    - Use the remote residue to determine the unique offset `s`, then extend the copies upward and downward across the whole canvas, cropping cells that shift out of bounds.

    Key insight:
    - The sparse residue is evidence from one odd-parity copy, so the shift can only be recovered by matching reflection, color swap, and horizontal glide together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    band_top, band, colors = _find_reference_band(g, bg)
    band_height, width = band.shape

    residues = [
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if not (band_top <= int(r) < band_top + band_height)
    ]
    assert residues, "expected a remote residue copy"

    ks = {(r - band_top) // band_height for r, _ in residues}
    assert len(ks) == 1 and 0 not in ks, "all residue pixels must belong to one non-zero copy"
    k = int(next(iter(ks)))
    residue_values = [int(g[r, c]) for r, c in residues]

    shifts = _matching_shifts(
        band,
        residues,
        residue_values,
        band_top,
        k,
        colors,
        use_flip=True,
        use_swap=True,
    )
    assert len(shifts) == 1, "residue must determine a unique glide offset"

    return _render_output(band, band_top, shifts[0], g.shape, colors)


def _candidate_ks(slot_count: int, base_slot: int, shift: int) -> list[int]:
    ks = [slot - base_slot for slot in range(slot_count) if slot != base_slot and (slot - base_slot) % 2]
    if shift > 0:
        ks.sort(key=lambda k: (-abs(k), k))
    else:
        ks.sort(key=lambda k: (-abs(k), -k))
    return ks


def _pick_residue_subset(
    full: np.ndarray,
    band: np.ndarray,
    band_top: int,
    base_slot: int,
    shift: int,
    colors: tuple[int, int],
) -> tuple[int, list[tuple[int, int]]] | None:
    band_height, _width = band.shape
    slot_count = full.shape[0] // band_height

    for k in _candidate_ks(slot_count, base_slot, shift):
        slot_top = band_top + k * band_height
        patch = full[slot_top : slot_top + band_height, :]
        visible = np.argwhere(patch != BG)
        if visible.shape[0] < SUBSET_SIZES[0]:
            continue
        if len({int(patch[r, c]) for r, c in visible}) < 2:
            continue

        pool: list[tuple[int, int]] = []
        for local_r in range(band_height):
            cols = np.flatnonzero(patch[local_r] != BG)
            if cols.size == 0:
                continue
            picks = [int(cols[0]), int(cols[len(cols) // 2]), int(cols[-1])]
            for c in picks:
                cell = (slot_top + local_r, c)
                if cell not in pool:
                    pool.append(cell)
        pool.sort()
        if len(pool) < SUBSET_SIZES[0]:
            continue

        pool_values = [int(full[r, c]) for r, c in pool]
        for size in SUBSET_SIZES:
            if len(pool) < size:
                continue
            for choice in itertools.combinations(range(len(pool)), size):
                residues = [pool[i] for i in choice]
                values = [pool_values[i] for i in choice]
                if len(set(values)) < 2:
                    continue
                if len({r for r, _ in residues}) < 2:
                    continue
                if len({c for _, c in residues}) < 2:
                    continue

                full_matches = _matching_shifts(
                    band,
                    residues,
                    values,
                    band_top,
                    k,
                    colors,
                    use_flip=True,
                    use_swap=True,
                )
                if full_matches != [shift]:
                    continue

                if shift in _matching_shifts(
                    band,
                    residues,
                    values,
                    band_top,
                    k,
                    colors,
                    use_flip=False,
                    use_swap=True,
                ):
                    continue
                if shift in _matching_shifts(
                    band,
                    residues,
                    values,
                    band_top,
                    k,
                    colors,
                    use_flip=True,
                    use_swap=False,
                ):
                    continue

                return k, residues

    return None


def _build_case() -> np.ndarray | None:
    for _ in range(MAX_ATTEMPTS):
        band_height = random.randint(*HEIGHT_RANGE)
        max_slots = min(SLOT_COUNT_RANGE[1], 30 // band_height)
        if max_slots < SLOT_COUNT_RANGE[0]:
            continue
        slot_count = random.randint(SLOT_COUNT_RANGE[0], max_slots)
        out_h = band_height * slot_count
        out_w = random.randint(*WIDTH_RANGE)
        colors = tuple(random.sample(range(1, 10), 2))

        band = _make_band(band_height, out_w, colors)
        if band is None:
            continue

        base_slot = random.randint(1, slot_count - 2)
        band_top = base_slot * band_height
        shift = random.choice(SHIFT_CHOICES)
        full = _render_output(band, band_top, shift, (out_h, out_w), colors)

        chosen = _pick_residue_subset(full, band, band_top, base_slot, shift, colors)
        if chosen is None:
            continue

        _k, residues = chosen
        inp = np.full((out_h, out_w), BG, dtype=int)
        inp[band_top : band_top + band_height, :] = band
        for r, c in residues:
            inp[r, c] = full[r, c]

        solved = main(inp)
        if not np.array_equal(solved, full):
            continue

        bg = _get_bg(inp)
        ref_top, ref_band, ref_colors = _find_reference_band(inp, bg)
        outside = [
            (int(r), int(c))
            for r, c in np.argwhere(inp != bg)
            if not (ref_top <= int(r) < ref_top + ref_band.shape[0])
        ]
        outside_values = [int(inp[r, c]) for r, c in outside]
        kset = {(r - ref_top) // ref_band.shape[0] for r, _ in outside}
        if len(kset) != 1:
            continue
        residue_k = int(next(iter(kset)))
        shifts = _matching_shifts(
            ref_band,
            outside,
            outside_values,
            ref_top,
            residue_k,
            ref_colors,
            use_flip=True,
            use_swap=True,
        )
        if shifts != [shift]:
            continue
        if shift in _matching_shifts(
            ref_band,
            outside,
            outside_values,
            ref_top,
            residue_k,
            ref_colors,
            use_flip=False,
            use_swap=True,
        ):
            continue
        if shift in _matching_shifts(
            ref_band,
            outside,
            outside_values,
            ref_top,
            residue_k,
            ref_colors,
            use_flip=True,
            use_swap=False,
        ):
            continue

        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residue-glide ribbon input")
