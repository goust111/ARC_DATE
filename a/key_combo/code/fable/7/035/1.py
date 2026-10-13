"""ARC task: repair a periodic fence, then drop the marker pixels."""

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _fence_description(grid: np.ndarray, bg: int):
    """Return (color, top, bottom, period, repaired_band)."""
    counts = Counter(map(int, grid.ravel()))
    colors = [color for color in counts if color != bg]
    assert len(colors) >= 2, "expected a fence and marker pixels"
    fence = max(colors, key=lambda color: counts[color])

    coords = np.argwhere(grid == fence)
    assert len(coords), "expected visible fence cells"
    top, bottom = int(coords[:, 0].min()), int(coords[:, 0].max())
    assert 0 < top <= bottom < grid.shape[0] - 1
    band = grid[top : bottom + 1]
    height, width = band.shape

    candidates: list[tuple[int, np.ndarray]] = []
    for period in range(2, width // 2 + 1):
        if width % period:
            continue
        tile = np.zeros((height, period), dtype=bool)
        for residue in range(period):
            tile[:, residue] = np.any(
                band[:, residue::period] == fence, axis=1
            )

        # The stated motif consists of solid fence columns and whole-column gates.
        solid = np.all(tile, axis=0)
        empty = ~np.any(tile, axis=0)
        if np.all(solid | empty) and np.any(solid) and np.any(empty):
            candidates.append((period, tile))

    assert candidates, "could not infer a periodic fence with a true gate"
    period, tile = min(candidates, key=lambda item: item[0])
    repaired = np.full((height, width), bg, dtype=int)
    for c in range(width):
        repaired[tile[:, c % period], c] = fence
    return fence, top, bottom, period, repaired


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a horizontal periodic fence and drop every marker through only its true gates.

    Input structure:
    - The dominant background surrounds a full-width, several-row-high fence band.
    - One frequent foreground color forms a horizontally repeated fence motif whose
      whole-background motif columns are gates; some fence cells are erased damage.
    - All other foreground cells are isolated colored markers above the band.

    Rule:
    - Infer the smallest supported horizontal period containing both solid fence
      columns and whole-column gates, and restore every erased fence cell.
    - Clear the markers and drop each vertically. A marker over a restored solid
      column stops immediately above the band; one over a true gate reaches the
      bottom row. Multiple markers in a column stack upward in their original order.

    Key insight:
    - A visible opening is passable only when its residue is open in every period
      instance: collision is tested against the reconstructed motif, not the damage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    bg = _background(g)
    fence, top, bottom, _period, repaired = _fence_description(g, bg)

    marker_mask = (g != bg) & (g != fence)
    marker_coords = np.argwhere(marker_mask)
    assert len(marker_coords), "expected marker pixels"
    assert np.all(marker_coords[:, 0] < top), "all markers must start above the fence"

    out = g.copy()
    out[marker_mask] = bg
    out[top : bottom + 1] = repaired

    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in marker_coords:
        by_col[int(c)].append((int(r), int(g[int(r), int(c)])))

    height = g.shape[0]
    for c, markers in by_col.items():
        # Lower input markers remain below upper input markers after simultaneous fall.
        markers.sort(reverse=True)
        blocked = bool(np.any(repaired[:, c] != bg))
        base = top - 1 if blocked else height - 1
        assert base - len(markers) + 1 >= 0, "marker stack does not fit"
        for offset, (_old_r, color) in enumerate(markers):
            out[base - offset, c] = color
    return out


def _candidate_periods(damaged_band: np.ndarray, fence: int) -> list[int]:
    h, w = damaged_band.shape
    result = []
    for period in range(2, w // 2 + 1):
        if w % period:
            continue
        tile = np.zeros((h, period), dtype=bool)
        for residue in range(period):
            tile[:, residue] = np.any(
                damaged_band[:, residue::period] == fence, axis=1
            )
        solid = np.all(tile, axis=0)
        empty = ~np.any(tile, axis=0)
        if np.all(solid | empty) and np.any(solid) and np.any(empty):
            result.append(period)
    return result


def _expected_output(
    inp: np.ndarray,
    bg: int,
    fence: int,
    top: int,
    bottom: int,
    gate_residues: set[int],
    period: int,
) -> np.ndarray:
    """Independent construction oracle using the sampled latent motif."""
    marker_mask = (inp != bg) & (inp != fence)
    out = inp.copy()
    out[marker_mask] = bg
    for c in range(inp.shape[1]):
        out[top : bottom + 1, c] = (
            bg if c % period in gate_residues else fence
        )

    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(marker_mask):
        by_col[int(c)].append((int(r), int(inp[int(r), int(c)])))
    for c, markers in by_col.items():
        markers.sort(reverse=True)
        base = inp.shape[0] - 1 if c % period in gate_residues else top - 1
        for offset, (_r, color) in enumerate(markers):
            out[base - offset, c] = color
    return out


def _try_generate() -> np.ndarray | None:
    palette = list(range(10))
    bg, fence = random.sample(palette, 2)
    marker_palette = random.sample([x for x in palette if x not in (bg, fence)], random.randint(2, 3))

    period = random.randint(4, 8)
    repeats = random.randint(3, min(5, 30 // period))
    width = period * repeats
    band_h = random.randint(2, 4)
    top = random.randint(7, 11)
    below = random.randint(4, 8)
    height = top + band_h + below
    if height > 30:
        return None
    bottom = top + band_h - 1

    gate_count = random.randint(1, min(2, period - 2))
    gate_residues = set(random.sample(range(period), gate_count))
    wall_residues = [r for r in range(period) if r not in gate_residues]

    inp = np.full((height, width), bg, dtype=int)
    for c in range(width):
        if c % period not in gate_residues:
            inp[top : bottom + 1, c] = fence

    # Erase a complete wall column in one instance: the required false-door foil.
    fake_residue = random.choice(wall_residues)
    fake_instance = random.randrange(repeats)
    fake_col = fake_instance * period + fake_residue
    inp[top : bottom + 1, fake_col] = bg

    # Add scattered cell damage, while retaining support in other instances.
    wall_cells = [
        (r, c)
        for r in range(top, bottom + 1)
        for c in range(width)
        if c % period in wall_residues and c != fake_col
    ]
    random.shuffle(wall_cells)
    for r, c in wall_cells[: random.randint(band_h, min(len(wall_cells), band_h * 3))]:
        inp[r, c] = bg

    # Reject damage that hides any required motif cell or creates a shorter explanation.
    band = inp[top : bottom + 1]
    candidates = _candidate_periods(band, fence)
    if not candidates or min(candidates) != period:
        return None

    true_col = random.randrange(repeats) * period + random.choice(tuple(gate_residues))
    columns = [fake_col, true_col]
    # Force stacking, then add both passable and blocked random columns.
    stack_col = random.choice(columns)
    columns.append(stack_col)
    for _ in range(random.randint(2, 5)):
        if random.random() < 0.5:
            residue = random.choice(tuple(gate_residues))
        else:
            residue = random.choice(wall_residues)
        columns.append(random.randrange(repeats) * period + residue)

    used_rows: dict[int, set[int]] = defaultdict(set)
    for c in columns:
        choices = [r for r in range(0, top - 1) if r not in used_rows[c]]
        random.shuffle(choices)
        chosen = next((r for r in choices if all(abs(r - old) > 1 for old in used_rows[c])), None)
        if chosen is None:
            return None
        used_rows[c].add(chosen)
        inp[chosen, c] = random.choice(marker_palette)

    # Semantic checks: both outcomes, stacking, real damage, and the false door matter.
    if not (fake_col in used_rows and true_col in used_rows and any(len(v) >= 2 for v in used_rows.values())):
        return None
    expected = _expected_output(inp, bg, fence, top, bottom, gate_residues, period)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected) or np.array_equal(inp, expected):
        return None
    if not np.all(expected[top : bottom + 1, fake_col] == fence):
        return None
    # A solver colliding against the damaged input would send this marker downward.
    if not np.all(inp[top : bottom + 1, fake_col] == bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate examples with explicit shortcut foils."""
    for _ in range(500):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a uniquely decodable periodic-fence task")
