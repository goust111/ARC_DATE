from __future__ import annotations

from collections import Counter

import numpy as np


def _contiguous_bands(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    bands: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        bands.append((start, prev + 1))
        start = prev = idx
    bands.append((start, prev + 1))
    return bands


def _find_panels(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, list[np.ndarray]]] = []

    for color in np.unique(grid):
        full_cols = [c for c in range(w) if bool(np.all(grid[:, c] == int(color)))]
        bands = _contiguous_bands(full_cols)
        if len(bands) < 2:
            continue

        sep_widths = {end - start for start, end in bands}
        if len(sep_widths) != 1:
            continue

        spans: list[tuple[int, int]] = []
        cursor = 0
        ok = True
        for start, end in bands:
            if cursor >= start:
                ok = False
                break
            spans.append((cursor, start))
            cursor = end
        if cursor >= w:
            ok = False
        else:
            spans.append((cursor, w))
        if not ok or len(spans) < 3:
            continue

        widths = [end - start for start, end in spans]
        if len(set(widths)) != 1 or widths[0] != h:
            continue

        panels = [grid[:, start:end] for start, end in spans]
        candidates.append((int(color), panels))

    assert len(candidates) == 1, "expected one separator-based panel layout"
    return candidates[0]


def _dominant_background(panels: list[np.ndarray]) -> int:
    values = np.concatenate([panel.reshape(-1) for panel in panels])
    counts = Counter(map(int, values))
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _majority_color(values: np.ndarray, global_counts: Counter[int]) -> int:
    counts: Counter[int] = Counter()
    order: list[int] = []
    for value in values:
        color = int(value)
        if color not in counts:
            order.append(color)
        counts[color] += 1

    best = order[0]
    best_key = (counts[best], global_counts[best])
    for color in order[1:]:
        key = (counts[color], global_counts[color])
        if key > best_key:
            best = color
            best_key = key
    return int(best)


def _consensus_pattern(panels: list[np.ndarray]) -> tuple[np.ndarray, int]:
    stack = np.stack(panels, axis=0)
    n_panels, h, w = map(int, stack.shape)
    assert h == w and n_panels >= 3

    global_counts = Counter(map(int, stack.reshape(-1)))
    bg = _dominant_background(panels)
    consensus = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            consensus[r, c] = _majority_color(stack[:, r, c], global_counts)
    return consensus, bg


def _tiled_square(pattern: np.ndarray, size: int) -> np.ndarray:
    tile_h, tile_w = map(int, pattern.shape)
    reps_r = (int(size) + tile_h - 1) // tile_h
    reps_c = (int(size) + tile_w - 1) // tile_w
    tiled = np.tile(pattern, (reps_r, reps_c))
    return tiled[:size, :size]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a consensus pattern from noisy panels, then size the output by its non-background count.

    Input structure:
    - Several equal square panels are arranged in one row.
    - Full-height separator columns of one color divide the panels.
    - Most panels contain the same small pattern; a minority have a few pixel edits.

    Rule:
    - For each local panel coordinate, take the most frequent color across panels.
    - This majority vote reconstructs the consensus pattern.
    - Let K be the number of non-background pixels in that consensus pattern.
    - Output a K x K grid by tiling the consensus pattern from the upper-left and cropping.

    Key insight:
    - The task has two stages: denoise by cross-panel voting, then convert a pixel count into output size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep_color, panels = _find_panels(g)
    consensus, bg = _consensus_pattern(panels)
    k = int(np.count_nonzero(consensus != int(bg)))
    assert k > 0, "consensus pattern must contain foreground"
    return _tiled_square(consensus, k)


# --------------- generator ---------------

PANEL_SIDE_RANGE = (4, 6)
PANEL_COUNT_OPTIONS = (3, 5, 7)
COLOR_POOL = tuple(range(10))
MAX_FILL_RATIO = 0.65
MAX_GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 80
PANEL_CORRUPT_ATTEMPTS = 120


def _non_monochrome_columns(panel: np.ndarray) -> bool:
    return all(len(set(map(int, panel[:, c]))) >= 2 for c in range(panel.shape[1]))


def _sample_target_foreground(n: int, rng: np.random.Generator) -> int:
    max_k = min(n * (n - 1), int(np.floor(MAX_FILL_RATIO * n * n)))
    assert max_k >= n
    mode = int(rng.integers(3))
    if mode == 0:
        return n
    if mode == 1:
        lo = n + 1
        hi = max(lo, (n + max_k) // 2)
        return int(rng.integers(lo, hi + 1))
    lo = max(n + 1, (n + max_k) // 2 + 1)
    hi = max_k
    return int(rng.integers(lo, hi + 1))


def _support_mask(n: int, target_k: int, rng: np.random.Generator) -> np.ndarray | None:
    assert n <= target_k <= n * (n - 1)

    for _ in range(MASK_ATTEMPTS):
        mask = np.zeros((n, n), dtype=bool)
        perm = rng.permutation(n)
        mask[np.arange(n), perm] = True
        row_counts = np.ones(n, dtype=int)
        col_counts = np.ones(n, dtype=int)
        filled = n

        while filled < target_k:
            candidates = [
                (r, c)
                for r in range(n)
                for c in range(n)
                if (not mask[r, c]) and row_counts[r] < n - 1 and col_counts[c] < n - 1
            ]
            if not candidates:
                break

            slack = np.array(
                [(n - 1 - row_counts[r]) + (n - 1 - col_counts[c]) for r, c in candidates],
                dtype=float,
            )
            probs = slack / slack.sum()
            r, c = candidates[int(rng.choice(len(candidates), p=probs))]
            mask[r, c] = True
            row_counts[r] += 1
            col_counts[c] += 1
            filled += 1

        if filled == target_k and _non_monochrome_columns(np.where(mask, 1, 0)):
            return mask
    return None


def _paint_consensus(mask: np.ndarray, bg: int, sep: int, rng: np.random.Generator) -> np.ndarray | None:
    n = int(mask.shape[0])
    target_k = int(mask.sum())
    bg_count = n * n - target_k
    if bg_count <= 1:
        return None

    min_fg_colors = max(1, (target_k + (bg_count - 2)) // (bg_count - 1))
    available = [color for color in COLOR_POOL if color not in {bg, sep}]
    if min_fg_colors > len(available):
        return None

    upper = min(len(available), target_k, max(min_fg_colors, 4))
    n_fg = int(rng.integers(min_fg_colors, upper + 1))
    palette = [int(color) for color in rng.choice(available, size=n_fg, replace=False)]
    rng.shuffle(palette)

    panel = np.full((n, n), bg, dtype=int)
    positions = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    order_mode = int(rng.integers(3))
    if order_mode == 0:
        positions.sort(key=lambda rc: (rc[0], rc[1]))
    elif order_mode == 1:
        positions.sort(key=lambda rc: (rc[1], rc[0]))
    else:
        positions.sort(key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))

    offset = int(rng.integers(n_fg))
    for idx, (r, c) in enumerate(positions):
        panel[r, c] = palette[(idx + offset) % n_fg]

    if not _non_monochrome_columns(panel):
        return None

    counts = Counter(map(int, panel.reshape(-1)))
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    if winners != [bg]:
        return None
    return panel


def _sample_consensus_panel(bg: int, sep: int, rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(PANEL_SIDE_RANGE[0], PANEL_SIDE_RANGE[1] + 1))
    target_k = _sample_target_foreground(n, rng)
    mask = _support_mask(n, target_k, rng)
    if mask is None:
        return None
    return _paint_consensus(mask, bg, sep, rng)


def _corrupt_panel(
    panel: np.ndarray,
    bg: int,
    sep: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    out = panel.copy()
    n = int(panel.shape[0])
    available = [color for color in COLOR_POOL if color != sep]
    desired_changes = int(rng.integers(1, min(n + 1, n * n // 2 + 1)))
    changed = 0

    for flat in rng.permutation(n * n):
        if changed >= desired_changes:
            break
        r, c = divmod(int(flat), n)
        current = int(out[r, c])
        choices = [color for color in available if color != current]
        rng.shuffle(choices)
        for color in choices:
            candidate = out.copy()
            candidate[r, c] = int(color)
            if not _non_monochrome_columns(candidate):
                continue
            out = candidate
            changed += 1
            break

    if changed == 0:
        return None
    if np.array_equal(out, panel):
        return None
    return out


def _assemble_input(panels: list[np.ndarray], sep: int) -> np.ndarray:
    n = int(panels[0].shape[0])
    width = len(panels) * n + (len(panels) - 1)
    grid = np.full((n, width), sep, dtype=int)
    cursor = 0
    for idx, panel in enumerate(panels):
        grid[:, cursor : cursor + n] = panel
        cursor += n
        if idx != len(panels) - 1:
            cursor += 1
    return grid


def _expected_output(consensus: np.ndarray, bg: int) -> np.ndarray:
    k = int(np.count_nonzero(consensus != bg))
    return _tiled_square(consensus, k)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg, sep = [int(color) for color in rng.choice(COLOR_POOL, size=2, replace=False)]
    consensus = _sample_consensus_panel(bg, sep, rng)
    if consensus is None:
        return None

    n_panels = int(rng.choice(PANEL_COUNT_OPTIONS))
    max_noisy = (n_panels - 1) // 2
    n_noisy = int(rng.integers(1, max_noisy + 1))
    noisy_indices = set(int(idx) for idx in rng.choice(n_panels, size=n_noisy, replace=False))

    panels: list[np.ndarray] = []
    for idx in range(n_panels):
        if idx not in noisy_indices:
            panels.append(consensus.copy())
            continue
        corrupted = _corrupt_panel(consensus, bg, sep, rng)
        if corrupted is None:
            return None
        panels.append(corrupted)

    grid = _assemble_input(panels, sep)
    expected = _expected_output(consensus, bg)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
