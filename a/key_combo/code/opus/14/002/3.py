from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
TOP_COLORS = (1, 2, 3, 4, 5, 6)
MARKER_COLORS = (7, 8, 9)
DIAG_DIRS = ((1, -1), (1, 1))
ORTH_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
EMITTER_COUNT_RANGE = (4, 6)
MARKER_COUNT_OPTIONS = (2, 3)
MIN_REGION_SIZE = 6
MAX_GENERATION_RETRIES = 600


def _cast_rays(
    shape: tuple[int, int],
    emitters: list[tuple[int, int]],
    *,
    reverse: bool = False,
) -> tuple[np.ndarray, int]:
    """Return the ray canvas and the number of ray-occlusion stops."""
    h, w = shape
    ray = np.zeros((h, w), dtype=int)
    ordered = list(reversed(emitters)) if reverse else emitters
    occlusions = 0

    for col, color in ordered:
        ray[0, col] = color
        for dr, dc in DIAG_DIRS:
            r, c = dr, col + dc
            while 0 <= r < h and 0 <= c < w:
                if ray[r, c] != BG:
                    occlusions += 1
                    break
                ray[r, c] = color
                r += dr
                c += dc
    return ray, occlusions


def _zero_regions(grid: np.ndarray) -> tuple[np.ndarray, list[frozenset[tuple[int, int]]]]:
    """Label 4-connected zero regions and return (labels, region-cell-sets)."""
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[frozenset[tuple[int, int]]] = []

    next_label = 0
    for sr in range(h):
        for sc in range(w):
            if grid[sr, sc] != BG or labels[sr, sc] != -1:
                continue
            q = deque([(sr, sc)])
            labels[sr, sc] = next_label
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ORTH_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and labels[nr, nc] == -1:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            regions.append(frozenset(cells))
            next_label += 1
    return labels, regions


def _fill_from_seed(out: np.ndarray, seed: tuple[int, int], color: int) -> None:
    """4-fill all background cells reachable from *seed* with *color*."""
    r0, c0 = seed
    assert out[r0, c0] == BG, "marker must start in a background cell after ray casting"

    q = deque([(r0, c0)])
    out[r0, c0] = color
    while q:
        r, c = q.popleft()
        for dr, dc in ORTH_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and out[nr, nc] == BG:
                out[nr, nc] = color
                q.append((nr, nc))


def _solve_with_order(grid: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3

    top_cols = [int(c) for c in np.flatnonzero(g[0] != BG)]
    assert top_cols, "top row must contain at least one emitter"
    emitters = [(c, int(g[0, c])) for c in top_cols]

    ray, _ = _cast_rays((h, w), emitters, reverse=reverse)
    out = ray.copy()

    marker_cells = np.argwhere(g[1:] != BG)
    markers = sorted((int(r + 1), int(c), int(g[r + 1, c])) for r, c in marker_cells)
    for r, c, color in markers:
        assert out[r, c] == BG, "markers must lie inside ray-bounded background regions"
        _fill_from_seed(out, (r, c), color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast occluding diagonal rays from the top row, then flood each marked partition with the marker color.

    Input structure:
    - A black grid has 4-6 distinct non-black pixels on the top row and a few scattered single-pixel markers lower down.
    - Top-row pixels are the ray emitters; lower markers each sit inside a different 4-connected background region.
    - No other foreground structure is needed: the rays themselves will create the partitions.

    Rule:
    - Process the non-black top-row pixels from left to right.
    - Each emitter sends one down-left and one down-right 45-degree ray, coloring every visited cell until the next step would hit an already-colored cell or leave the grid.
    - After all rays are drawn, 4-fill from every lower marker using its own color across background cells only; rays, top-row emitters, earlier fills, and the grid boundary stop the fill.

    Key insight:
    - Earlier rays block later ones, so the left-to-right processing order determines the partition shapes that the scattered marker pixels end up flooding.
    """
    return _solve_with_order(np.asarray(grid, dtype=int), reverse=False)


def _gap_has_even_pair(cols: list[int]) -> bool:
    return any((cols[i + 1] - cols[i]) % 2 == 0 for i in range(len(cols) - 1))


def _choose_top_columns(width: int, count: int) -> list[int] | None:
    """Sample non-contiguous-ish top-row columns with at least one even gap pair."""
    if width - 2 < count:
        return None

    for _ in range(80):
        cols = sorted(random.sample(range(1, width - 1), count))
        if not _gap_has_even_pair(cols):
            continue
        if all(cols[i + 1] == cols[i] + 1 for i in range(len(cols) - 1)):
            continue
        return cols
    return None


def _cell_score(cell: tuple[int, int], region: frozenset[tuple[int, int]], shape: tuple[int, int]) -> int:
    r, c = cell
    h, w = shape
    same_neighbors = sum((r + dr, c + dc) in region for dr, dc in ORTH_DIRS)
    edge_margin = min(r - 2, h - 1 - r, c, w - 1 - c)
    return 10 * same_neighbors + edge_margin


def _region_records(
    normal_ray: np.ndarray,
    reverse_ray: np.ndarray,
) -> list[dict[str, object]]:
    """Describe usable normal-order regions and how they behave under reverse order."""
    normal_labels, normal_regions = _zero_regions(normal_ray)
    reverse_labels, reverse_regions = _zero_regions(reverse_ray)
    records: list[dict[str, object]] = []

    for idx, region in enumerate(normal_regions):
        usable_cells = [cell for cell in region if cell[0] >= 2]
        if len(usable_cells) < MIN_REGION_SIZE or len(region) < MIN_REGION_SIZE:
            continue

        scored = sorted(
            ((cell, _cell_score(cell, region, normal_ray.shape)) for cell in usable_cells),
            key=lambda item: (item[1], item[0][0], item[0][1]),
            reverse=True,
        )

        best_any = scored[0]
        best_sensitive: tuple[tuple[int, int], int] | None = None
        region_changed = False
        for cell, score in scored:
            r, c = cell
            if reverse_ray[r, c] != BG:
                best_sensitive = (cell, score)
                region_changed = True
                break
            rev_label = reverse_labels[r, c]
            if rev_label == -1:
                best_sensitive = (cell, score)
                region_changed = True
                break
            if reverse_regions[rev_label] != region:
                best_sensitive = (cell, score)
                region_changed = True
                break

        chosen_cell, chosen_score = best_sensitive if best_sensitive is not None else best_any
        records.append(
            {
                "label": idx,
                "region": region,
                "size": len(region),
                "cell": chosen_cell,
                "score": chosen_score,
                "sensitive": region_changed,
            }
        )

    records.sort(key=lambda rec: (int(rec["sensitive"]), int(rec["size"]), int(rec["score"])), reverse=True)
    return records


def _choose_marked_regions(records: list[dict[str, object]]) -> list[dict[str, object]] | None:
    if len(records) < 3:
        return None

    count_options = [count for count in MARKER_COUNT_OPTIONS if len(records) >= count + 1]
    if not count_options:
        return None

    marker_count = random.choice(count_options)
    sensitive = [rec for rec in records if bool(rec["sensitive"])]
    if not sensitive:
        return None

    chosen = [random.choice(sensitive)]
    remaining = [rec for rec in records if rec is not chosen[0]]
    if len(remaining) < marker_count - 1:
        return None

    chosen.extend(random.sample(remaining, marker_count - 1))
    return chosen


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    emitter_count = random.randint(*EMITTER_COUNT_RANGE)
    cols = _choose_top_columns(w, emitter_count)
    if cols is None:
        return None

    top_colors = random.sample(TOP_COLORS, emitter_count)
    emitters = list(zip(cols, top_colors))
    normal_ray, occlusions = _cast_rays((h, w), emitters, reverse=False)
    reverse_ray, _ = _cast_rays((h, w), emitters, reverse=True)
    if occlusions == 0:
        return None

    records = _region_records(normal_ray, reverse_ray)
    chosen_regions = _choose_marked_regions(records)
    if chosen_regions is None:
        return None

    marker_colors = random.sample(MARKER_COLORS, len(chosen_regions))

    inp = np.zeros((h, w), dtype=int)
    for col, color in emitters:
        inp[0, col] = color
    for rec, color in zip(chosen_regions, marker_colors):
        r, c = rec["cell"]  # type: ignore[index]
        inp[r, c] = color

    out = main(inp)
    try:
        alt = _solve_with_order(inp, reverse=True)
    except AssertionError:
        alt = None

    # Keep only examples where order visibly changes the marked outcome.
    if alt is not None:
        out_fill = np.isin(out, MARKER_COLORS)
        alt_fill = np.isin(alt, MARKER_COLORS)
        if np.array_equal(out_fill, alt_fill):
            return None

    # Each chosen marker should expand beyond a singleton for visual clarity.
    for color in marker_colors:
        if int(np.count_nonzero(out == color)) < 4:
            return None

    # At least one sizable region must stay black after all fills.
    out_labels, out_regions = _zero_regions(out)
    del out_labels
    if not out_regions or max(len(region) for region in out_regions) < MIN_REGION_SIZE:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition fill input")
