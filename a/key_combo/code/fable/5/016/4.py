from __future__ import annotations

from collections import Counter

import numpy as np


def _fold_at_period(
    grid: np.ndarray, period: int
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Recover modal residue classes and return their mismatching cells."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    p = int(period)
    if p < 1 or h % p or w % p:
        return None

    template = np.empty((p, p), dtype=int)
    mismatches: list[tuple[int, int]] = []
    for lr in range(p):
        for lc in range(p):
            values = g[lr::p, lc::p].ravel()
            colors, counts = np.unique(values, return_counts=True)
            top = int(counts.max())
            if int(np.count_nonzero(counts == top)) != 1:
                return None
            modal = int(colors[int(np.argmax(counts))])
            template[lr, lc] = modal
            for rr, cc in np.argwhere(g[lr::p, lc::p] != modal):
                mismatches.append((lr + int(rr) * p, lc + int(cc) * p))
    return template, mismatches


def _infer_period(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, list[tuple[int, int]]]:
    """Choose the smallest square period attaining the least sparse residue error."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or min(g.shape) < 2:
        raise ValueError("expected a two-dimensional tiled grid")
    h, w = map(int, g.shape)

    candidates: list[tuple[int, int, np.ndarray, list[tuple[int, int]]]] = []
    for p in range(2, min(h, w) + 1):
        if h % p or w % p:
            continue
        copies = (h // p) * (w // p)
        if copies < 4:
            continue
        folded = _fold_at_period(g, p)
        if folded is None:
            continue
        template, errors = folded
        if not (1 <= len(errors) < p * p):
            continue

        # Each hidden-pixel coordinate is represented by one carrier tile.
        residues = [(r % p, c % p) for r, c in errors]
        if len(set(residues)) != len(residues):
            continue
        candidates.append((len(errors), p, template, errors))

    if not candidates:
        raise ValueError("no uniquely recoverable noisy square period")
    candidates.sort(key=lambda item: (item[0], item[1]))
    error_count, p, template, errors = candidates[0]
    best_periods = [item[1] for item in candidates if item[0] == error_count]
    assert p == min(best_periods)
    return p, template, errors


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).ravel()))
    largest = max(counts.values())
    winners = [color for color, count in counts.items() if count == largest]
    if len(winners) != 1:
        raise ValueError("expected a unique modal background color")
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Fold scattered deviations from a periodic tiling into their hidden basic-block pattern.

    Input structure:
    - The grid is a multi-color square basic pattern tiled periodically in both
      directions, with a few cells changed from their residue-class template colors.
    - Each changed cell occupies a different coordinate modulo the basic period.

    Rule:
    - Infer the smallest supported square period and take the unique modal color
      in every modulo-coordinate class to recover its template.
    - Compare the input with that template, then place every deviating cell's color
      at its `(row mod p, column mod p)` coordinate in a new `p x p` grid whose
      base color is the unique modal color of the recovered template.

    Key insight:
    - The large tiling is only a coordinate carrier: anomalies stored in different
      copies combine under coordinate-modulo folding into a pattern never shown whole.
    """
    g = np.asarray(grid, dtype=int)
    p, template, errors = _infer_period(g)
    background = _unique_mode(template)

    out = np.full((p, p), background, dtype=int)
    written: set[tuple[int, int]] = set()
    for r, c in errors:
        local = (int(r % p), int(c % p))
        if local in written:
            raise ValueError("two anomalies compete for one folded coordinate")
        written.add(local)
        out[local] = int(g[r, c])
    return out


_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_MAX_GENERATION_ATTEMPTS = 500


def _exact_clean_period(grid: np.ndarray) -> int | None:
    """Independent exact-period oracle used only by the generator."""
    h, w = map(int, grid.shape)
    for p in range(1, min(h, w) + 1):
        if h % p or w % p:
            continue
        tile = grid[:p, :p]
        if np.array_equal(np.tile(tile, (h // p, w // p)), grid):
            return p
    return None


def _grow_connected_coords(
    p: int, size: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Grow a 4-connected local-coordinate motif without wrapping at edges."""
    start = (int(rng.integers(p)), int(rng.integers(p)))
    cells = {start}
    while len(cells) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < p and 0 <= cc < p and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            raise RuntimeError("connected motif growth unexpectedly stalled")
        ordered = sorted(frontier)
        cells.add(ordered[int(rng.integers(len(ordered)))])
    return sorted(cells)


def _make_template(
    p: int, bg: int, motif_colors: list[int], rng: np.random.Generator
) -> np.ndarray:
    """Create a primitive, background-dominant, genuinely multi-color tile."""
    template = np.full((p, p), bg, dtype=int)
    foreground_count = max(len(motif_colors), int(round(0.36 * p * p)))
    foreground_count = min(foreground_count, (p * p - 1) // 2)
    positions = rng.choice(p * p, size=foreground_count, replace=False)
    assigned = list(motif_colors)
    assigned.extend(
        int(rng.choice(motif_colors))
        for _ in range(foreground_count - len(motif_colors))
    )
    rng.shuffle(assigned)
    for flat, color in zip(positions, assigned):
        template.flat[int(flat)] = int(color)
    return template


def _semantic_checks(
    inp: np.ndarray,
    clean: np.ndarray,
    template: np.ndarray,
    expected: np.ndarray,
    p: int,
    records: list[tuple[int, int, int, int]],
) -> bool:
    """Check the stated semantics, not merely generator/solver round-tripping."""
    if max(inp.shape) > 30 or expected.shape != (p, p):
        return False
    if _exact_clean_period(clean) != p:
        return False

    try:
        inferred_p, inferred_template, errors = _infer_period(inp)
        actual = main(inp)
    except (AssertionError, ValueError):
        return False
    if inferred_p != p or not np.array_equal(inferred_template, template):
        return False
    if not np.array_equal(actual, expected):
        return False

    record_cells = {(r, c) for r, c, _lr, _lc in records}
    if set(errors) != record_cells:
        return False
    local_coords = {(lr, lc) for _r, _c, lr, lc in records}
    if len(local_coords) != len(records):
        return False

    # The carriers must be distributed: no visible tile contains the answer.
    carrier_tiles = {(r // p, c // p) for r, c, _lr, _lc in records}
    if len(carrier_tiles) < 3:
        return False
    per_tile = Counter((r // p, c // p) for r, c, _lr, _lc in records)
    if max(per_tile.values()) != 1:
        return False
    h, w = inp.shape
    for r0 in range(0, h, p):
        for c0 in range(0, w, p):
            if np.array_equal(inp[r0 : r0 + p, c0 : c0 + p], expected):
                return False

    bg = _unique_mode(template)
    template_colors = set(map(int, template.ravel()))
    anomaly_colors = {int(inp[r, c]) for r, c in record_cells}
    if not anomaly_colors.issubset(template_colors) or bg in anomaly_colors:
        return False
    if len(anomaly_colors) < 2 or len(np.unique(expected)) < 3:
        return False

    underlying = {int(template[lr, lc]) for lr, lc in local_coords}
    if bg not in underlying or len(underlying - {bg}) == 0:
        return False
    if np.array_equal(expected, template):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    p = int(rng.integers(3, 7))
    max_tiles = 30 // p
    tile_rows = int(rng.integers(3, max_tiles + 1))
    tile_cols = int(rng.integers(3, max_tiles + 1))

    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    motif_colors = palette[1:4]
    template = _make_template(p, bg, motif_colors, rng)
    clean = np.tile(template, (tile_rows, tile_cols))
    if _exact_clean_period(clean) != p:
        return None

    anomaly_count = int(rng.integers(5, min(8, p * p - 1) + 1))
    local_coords = _grow_connected_coords(p, anomaly_count, rng)
    underlying = [int(template[lr, lc]) for lr, lc in local_coords]
    if underlying.count(bg) < 2 or sum(color != bg for color in underlying) < 2:
        return None

    tile_ids = rng.choice(tile_rows * tile_cols, size=anomaly_count, replace=False)
    inp = clean.copy()
    expected = np.full((p, p), bg, dtype=int)
    records: list[tuple[int, int, int, int]] = []
    chosen_colors: list[int] = []
    for (lr, lc), tile_id in zip(local_coords, tile_ids):
        expected_color = int(template[lr, lc])
        choices = [color for color in motif_colors if color != expected_color]
        anomaly_color = int(rng.choice(choices))
        tr, tc = divmod(int(tile_id), tile_cols)
        r, c = tr * p + lr, tc * p + lc
        inp[r, c] = anomaly_color
        expected[lr, lc] = anomaly_color
        records.append((r, c, lr, lc))
        chosen_colors.append(anomaly_color)
    if len(set(chosen_colors)) < 2:
        return None

    if not _semantic_checks(inp, clean, template, expected, p, records):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse, semantically validated periodic-residue puzzle input."""
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        result = _generate_attempt(rng)
        if result is None:
            continue
        inp, expected = result
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return inp
    raise RuntimeError("failed to generate a non-degenerate periodic residue puzzle")
