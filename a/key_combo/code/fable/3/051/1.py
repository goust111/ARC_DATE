from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
ALL_NON_BG_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
MIN_INTERIOR_SIZE = 6
MAX_INTERIOR_SIZE = 12


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _wall_mask(row_labels: np.ndarray, col_labels: np.ndarray) -> np.ndarray:
    return row_labels[:, None] == col_labels[None, :]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 2 and w >= 2, "grid must contain an interior plus label row/column"

    bg = int(g[h - 1, w - 1])
    inner = g[: h - 1, : w - 1]
    row_labels = g[: h - 1, w - 1]
    col_labels = g[h - 1, : w - 1]
    assert np.all(row_labels != bg), "row labels must be non-background"
    assert np.all(col_labels != bg), "column labels must be non-background"

    label_colors = set(map(int, row_labels)) | set(map(int, col_labels))
    seeds = np.argwhere(inner != bg)
    for r, c in seeds:
        assert int(inner[r, c]) not in label_colors, "seed colors must not appear in the labels"

    walls = _wall_mask(row_labels, col_labels)
    open_mask = ~walls
    components = _components_from_mask(open_mask)
    assert components, "expected at least one non-wall compartment"

    out = g.copy()
    for comp in components:
        seed_colors = {
            int(inner[r, c])
            for r, c in comp
            if int(inner[r, c]) != bg
        }
        assert len(seed_colors) <= 1, "each compartment may contain at most one seed color"
        if not seed_colors:
            continue
        fill_color = next(iter(seed_colors))
        out[comp[:, 0], comp[:, 1]] = fill_color

    out[: h - 1, : w - 1][walls] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer invisible walls from matching row/column labels and flood each seeded compartment.

    Input structure:
    - The bottom row is a non-background column-label strip and the rightmost column is a non-background row-label strip; the bottom-right corner is background.
    - The interior region uses a black background with a few colored seed pixels, and seed colors never appear in the label strips.
    - An interior cell is an invisible wall exactly when its row label matches its column label.

    Rule:
    - Project the row/column label equalities into the interior to recover the wall cells.
    - Split the non-wall interior into 4-connected compartments.
    - If a compartment contains one seed, flood the whole compartment with that seed color; empty compartments and wall cells stay black, and the label strips remain unchanged.

    Key insight:
    - The barriers are never drawn directly; they are the shadow of label-color equality between the right edge and the bottom edge.
    """
    return _solve(grid)


def _partition_positive(total: int, parts: int) -> list[int]:
    lengths = [1] * parts
    for _ in range(total - parts):
        lengths[random.randrange(parts)] += 1
    random.shuffle(lengths)
    return lengths


def _make_band_colors(palette: tuple[int, ...], n_bands: int) -> list[int]:
    seq = random.sample(list(palette), len(palette))
    while len(seq) < n_bands:
        positions = list(range(len(seq) + 1))
        random.shuffle(positions)
        inserted = False
        for pos in positions:
            left = seq[pos - 1] if pos > 0 else None
            right = seq[pos] if pos < len(seq) else None
            choices = [color for color in palette if color != left and color != right]
            if choices:
                seq.insert(pos, random.choice(choices))
                inserted = True
                break
        if not inserted:
            raise AssertionError("failed to insert another band color")
    return seq


def _build_label_sequence(length: int, palette: tuple[int, ...]) -> np.ndarray:
    max_bands = min(length, len(palette) + 2)
    n_bands = random.randint(len(palette), max_bands)
    band_colors = _make_band_colors(palette, n_bands)
    band_lengths = _partition_positive(length, n_bands)
    seq = np.empty(length, dtype=int)
    start = 0
    for color, span in zip(band_colors, band_lengths):
        seq[start : start + span] = color
        start += span
    return seq


def _select_palette_size(h: int, w: int) -> int:
    if min(h, w) >= 7 and random.random() < 0.45:
        return 3
    return 2


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(MIN_INTERIOR_SIZE, MAX_INTERIOR_SIZE)
    w = random.randint(MIN_INTERIOR_SIZE, MAX_INTERIOR_SIZE)
    palette_size = _select_palette_size(h, w)
    label_palette = tuple(random.sample(ALL_NON_BG_COLORS, palette_size))

    row_labels = _build_label_sequence(h, label_palette)
    col_labels = _build_label_sequence(w, label_palette)
    walls = _wall_mask(row_labels, col_labels)
    wall_ratio = float(np.mean(walls))
    if wall_ratio < 0.18 or wall_ratio > 0.58:
        return None

    components = _components_from_mask(~walls)
    if not (2 <= len(components) <= 8):
        return None

    component_areas = [int(comp.shape[0]) for comp in components]
    if sum(area >= 4 for area in component_areas) < 2:
        return None

    seedable = [comp for comp in components if int(comp.shape[0]) >= 3]
    if len(seedable) < 2:
        return None

    max_seeds = min(4, len(seedable))
    if max_seeds < 2:
        return None
    if len(seedable) >= 3 and random.random() < 0.4:
        n_seeds = random.randint(2, max(2, max_seeds - 1))
    else:
        n_seeds = random.randint(2, max_seeds)

    chosen_components = random.sample(seedable, n_seeds)
    seed_palette = [color for color in ALL_NON_BG_COLORS if color not in label_palette]
    seed_colors = random.sample(seed_palette, n_seeds)

    grid = np.full((h + 1, w + 1), BACKGROUND, dtype=int)
    grid[:h, w] = row_labels
    grid[h, :w] = col_labels

    total_growth = 0
    for comp, color in zip(chosen_components, seed_colors):
        coord = tuple(map(int, comp[random.randrange(len(comp))]))
        grid[coord] = color
        total_growth += int(comp.shape[0]) - 1

    if total_growth < 4:
        return None

    out = _solve(grid)
    interior_delta = int(np.count_nonzero(out[:h, :w] != grid[:h, :w]))
    if interior_delta < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-wall flood-fill input")
