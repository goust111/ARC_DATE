from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
PERIODS = (3, 4)
PALETTE = tuple(color for color in range(10) if color not in (BLACK, RED))
_RNG = np.random.default_rng()


def _find_red_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    red = np.argwhere(grid == RED)
    assert red.size, "expected a red frame"
    r0, c0 = red.min(axis=0)
    r1, c1 = red.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

    mask = np.zeros_like(grid, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    assert np.array_equal(grid == RED, mask), "red pixels must form a single rectangular border"
    return r0, c0, r1, c1


def _outside_box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = map(int, shape)
    r0, c0, r1, c1 = box
    mask = np.ones((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _line_densities(occupied: np.ndarray, valid_mask: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    counts = occupied.sum(axis=axis)
    totals = valid_mask.sum(axis=axis)
    densities = np.zeros_like(counts, dtype=float)
    nz = totals > 0
    densities[nz] = counts[nz] / totals[nz]
    return densities, totals


def _best_pattern(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, int, int]:
    h, w = grid.shape
    valid_mask = _outside_box_mask((h, w), box)
    occupied = (grid != BLACK) & (grid != RED) & valid_mask

    row_density, row_total = _line_densities(occupied, valid_mask, axis=1)
    col_density, col_total = _line_densities(occupied, valid_mask, axis=0)

    best = None
    for orientation, densities, totals in (
        ("horizontal", row_density, row_total),
        ("vertical", col_density, col_total),
    ):
        n = int(densities.shape[0])
        idx = np.arange(n)
        valid_lines = totals > 0
        for period in PERIODS:
            for phase in range(period):
                stripe_lines = valid_lines & ((idx - phase) % period == 0)
                gap_lines = valid_lines & ~((idx - phase) % period == 0)
                if not stripe_lines.any() or not gap_lines.any():
                    continue

                stripe = densities[stripe_lines]
                gap = densities[gap_lines]
                separation = float(stripe.mean() - gap.mean())
                score = separation + 0.4 * float(np.median(stripe) - np.median(gap))
                score += 0.2 * float(stripe.min() - gap.max())
                cand = (score, separation, orientation, int(period), int(phase))
                if best is None or cand > best:
                    best = cand

    assert best is not None, "failed to infer stripe pattern"
    _score, _sep, orientation, period, phase = best
    return orientation, period, phase


def _stripe_cell_mask(shape: tuple[int, int], orientation: str, period: int, phase: int) -> np.ndarray:
    h, w = map(int, shape)
    if orientation == "horizontal":
        base = (np.arange(h)[:, None] - int(phase)) % int(period) == 0
        return np.broadcast_to(base, (h, w))
    assert orientation == "vertical"
    base = (np.arange(w)[None, :] - int(phase)) % int(period) == 0
    return np.broadcast_to(base, (h, w))


def _noise_color(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    orientation: str,
    period: int,
    phase: int,
) -> int:
    valid_mask = _outside_box_mask(grid.shape, box)
    stripe_mask = _stripe_cell_mask(grid.shape, orientation, period, phase)
    gap_mask = valid_mask & ~stripe_mask

    colors = [int(color) for color in np.unique(grid) if int(color) not in (BLACK, RED)]
    assert len(colors) >= 2, "expected both stripe and noise colors"

    gap_counts = {color: int(np.count_nonzero((grid == color) & gap_mask)) for color in colors}
    noise = max(colors, key=lambda color: (gap_counts[color], int(np.count_nonzero(grid == color))))
    assert gap_counts[noise] > 0, "noise color must appear between stripes"
    return int(noise)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the red frame and redraw the stripe pattern's negative inside it using the noise color.

    Input structure:
    - The background is black, plus one red 1-pixel-thick rectangular frame whose interior is black.
    - Outside that frame, the grid carries a full horizontal or vertical stripe pattern with width 1 and period 3 or 4.
    - A second non-red color appears as sparse noise, replacing some stripe pixels and also landing in stripe gaps.

    Rule:
    - Locate the red frame and ignore its covered area while inferring the stripe direction, period, and phase from the repeated line densities.
    - Identify the noise color as the non-red color that appears on non-stripe rows or columns.
    - Return only the red frame's bounding box, preserving the frame and filling its interior with the stripe pattern's negative: stripe lines stay black, while the in-between lines become the noise color.

    Key insight:
    - The same sparse color plays two roles: it is an outlier when recovering the dominant stripe lattice, but it becomes the paint for the lattice's complement inside the framed crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    box = _find_red_frame(g)
    orientation, period, phase = _best_pattern(g, box)
    noise = _noise_color(g, box, orientation, period, phase)

    r0, c0, r1, c1 = box
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    out[0, :] = RED
    out[-1, :] = RED
    out[:, 0] = RED
    out[:, -1] = RED

    if orientation == "horizontal":
        for global_r in range(r0 + 1, r1):
            if (global_r - phase) % period != 0:
                out[global_r - r0, 1:-1] = noise
    else:
        for global_c in range(c0 + 1, c1):
            if (global_c - phase) % period != 0:
                out[1:-1, global_c - c0] = noise

    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _sample_distinct_colors() -> tuple[int, int]:
    colors = _RNG.choice(np.array(PALETTE, dtype=int), size=2, replace=False)
    return int(colors[0]), int(colors[1])


def _sample_geometry(orientation: str, period: int) -> tuple[int, int, int, int, int, int]:
    h = _randint(14, 26)
    w = _randint(14, 26)

    if orientation == "horizontal":
        frame_h = _randint(max(6, period + 3), min(12, h - 3))
        frame_w = _randint(5, min(12, w - 3))
    else:
        frame_h = _randint(5, min(12, h - 3))
        frame_w = _randint(max(6, period + 3), min(12, w - 3))

    r0 = _randint(1, h - frame_h - 1)
    c0 = _randint(1, w - frame_w - 1)
    return h, w, r0, c0, frame_h, frame_w


def _sample_noise_points(
    box_mask: np.ndarray,
    stripe_mask: np.ndarray,
    orientation: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = box_mask.shape
    stripe_points: list[tuple[int, int]] = []
    gap_points: list[tuple[int, int]] = []

    if orientation == "horizontal":
        stripe_lines = [r for r in range(h) if bool(stripe_mask[r, 0])]
        gap_lines = [r for r in range(h) if not bool(stripe_mask[r, 0])]
        stripe_take = _RNG.choice(
            stripe_lines,
            size=max(2, min(len(stripe_lines), _randint(max(2, len(stripe_lines) // 3), max(2, (2 * len(stripe_lines)) // 3)))),
            replace=False,
        )
        gap_take = _RNG.choice(
            gap_lines,
            size=max(2, min(len(gap_lines), _randint(max(2, len(gap_lines) // 3), max(2, (2 * len(gap_lines)) // 3)))),
            replace=False,
        )

        for r in map(int, stripe_take):
            cols = np.flatnonzero(~box_mask[r])
            if cols.size == 0:
                continue
            k = min(int(cols.size), _randint(1, 2 if cols.size >= 8 else 1))
            picked = _RNG.choice(cols, size=k, replace=False)
            stripe_points.extend((int(r), int(c)) for c in picked.tolist())

        for r in map(int, gap_take):
            cols = np.flatnonzero(~box_mask[r])
            if cols.size == 0:
                continue
            k = min(int(cols.size), _randint(1, 2 if cols.size >= 12 and _choice((False, True)) else 1))
            picked = _RNG.choice(cols, size=k, replace=False)
            gap_points.extend((int(r), int(c)) for c in picked.tolist())
    else:
        stripe_lines = [c for c in range(w) if bool(stripe_mask[0, c])]
        gap_lines = [c for c in range(w) if not bool(stripe_mask[0, c])]
        stripe_take = _RNG.choice(
            stripe_lines,
            size=max(2, min(len(stripe_lines), _randint(max(2, len(stripe_lines) // 3), max(2, (2 * len(stripe_lines)) // 3)))),
            replace=False,
        )
        gap_take = _RNG.choice(
            gap_lines,
            size=max(2, min(len(gap_lines), _randint(max(2, len(gap_lines) // 3), max(2, (2 * len(gap_lines)) // 3)))),
            replace=False,
        )

        for c in map(int, stripe_take):
            rows = np.flatnonzero(~box_mask[:, c])
            if rows.size == 0:
                continue
            k = min(int(rows.size), _randint(1, 2 if rows.size >= 8 else 1))
            picked = _RNG.choice(rows, size=k, replace=False)
            stripe_points.extend((int(r), int(c)) for r in picked.tolist())

        for c in map(int, gap_take):
            rows = np.flatnonzero(~box_mask[:, c])
            if rows.size == 0:
                continue
            k = min(int(rows.size), _randint(1, 2 if rows.size >= 12 and _choice((False, True)) else 1))
            picked = _RNG.choice(rows, size=k, replace=False)
            gap_points.extend((int(r), int(c)) for r in picked.tolist())

    return stripe_points, gap_points


def _expected_output(box: tuple[int, int, int, int], orientation: str, period: int, phase: int, noise: int) -> np.ndarray:
    r0, c0, r1, c1 = box
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    out[0, :] = RED
    out[-1, :] = RED
    out[:, 0] = RED
    out[:, -1] = RED

    if orientation == "horizontal":
        for global_r in range(r0 + 1, r1):
            if (global_r - phase) % period != 0:
                out[global_r - r0, 1:-1] = int(noise)
    else:
        for global_c in range(c0 + 1, c1):
            if (global_c - phase) % period != 0:
                out[1:-1, global_c - c0] = int(noise)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray, int, int]:
    orientation = _choice(("horizontal", "vertical"))
    period = _choice(PERIODS)
    phase = _randint(0, period - 1)
    stripe_color, noise_color = _sample_distinct_colors()
    h, w, r0, c0, frame_h, frame_w = _sample_geometry(orientation, period)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    box = (r0, c0, r1, c1)

    grid = np.full((h, w), BLACK, dtype=int)
    stripe_mask = _stripe_cell_mask((h, w), orientation, period, phase)
    grid[stripe_mask] = int(stripe_color)

    box_mask = np.zeros((h, w), dtype=bool)
    box_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    stripe_points, gap_points = _sample_noise_points(box_mask, stripe_mask, orientation)
    for r, c in stripe_points + gap_points:
        grid[r, c] = int(noise_color)

    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED

    expected = _expected_output(box, orientation, period, phase, noise_color)
    return grid, expected, len(stripe_points), len(gap_points)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid, expected, stripe_noise, gap_noise = _generate_case()
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue

        if gap_noise < 2 or stripe_noise < 2:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("generate_input failed to create a valid sample")
