from __future__ import annotations

import random

import numpy as np


def _invert_panel(panel: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    return np.where(panel == a, b, a).astype(int)


def _aligned_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected a square 2x2 panel canvas with one divider row/column"

    split = h // 2
    line_color = int(g[split, 0])
    assert np.all(g[split, :] == line_color), "expected a full horizontal divider"
    assert np.all(g[:, split] == line_color), "expected a full vertical divider"

    tl = g[:split, :split]
    tr = g[:split, split + 1 :]
    bl = g[split + 1 :, :split]
    br = g[split + 1 :, split + 1 :]
    assert tl.shape == tr.shape == bl.shape == br.shape
    assert tl.shape[0] == tl.shape[1], "each panel must be square"

    return [
        tl.copy(),
        np.rot90(tr, 3),
        np.rot90(br, 2),
        np.rot90(bl, 1),
    ]


def _best_reconstruction(aligned: list[np.ndarray]) -> np.ndarray:
    colors_found = sorted({int(v) for panel in aligned for v in np.unique(panel)})
    assert len(colors_found) == 2, "expected exactly two panel colors"
    colors = (colors_found[0], colors_found[1])

    scored: list[tuple[tuple[int, int], int, list[np.ndarray]]] = []
    for inverted_idx in range(4):
        restored = [panel.copy() for panel in aligned]
        restored[inverted_idx] = _invert_panel(restored[inverted_idx], colors)

        count_a = sum(panel == colors[0] for panel in restored)
        count_b = 4 - count_a
        margin = np.abs(count_a - count_b)
        ties = int(np.count_nonzero(margin == 0))
        score = (-ties, int(margin.sum()))
        scored.append((score, inverted_idx, restored))

    best_score = max(score for score, _idx, _restored in scored)
    winners = [(idx, restored) for score, idx, restored in scored if score == best_score]
    assert len(winners) == 1, "expected a unique inverted panel"

    _best_idx, restored = winners[0]
    count_a = sum(panel == colors[0] for panel in restored)
    count_b = 4 - count_a
    assert np.all(count_a != count_b), "expected a strict per-cell majority after restoration"
    return np.where(count_a > count_b, colors[0], colors[1]).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the original binary panel by undoing quadrant rotations, fixing one inverted panel, and taking a majority vote.

    Input structure:
    - The canvas is split by one full row and one full column into four equal square panels.
    - The four panels are rotated copies of the same two-color pattern in the 0, 90, 180, and 270 degree views.
    - Exactly one panel is globally color-inverted, and each panel also contains a few single-cell binary corruptions.

    Rule:
    - Rotate the top-right, bottom-right, and bottom-left panels back into the top-left panel's orientation.
    - Identify the unique panel whose colors were globally swapped, and invert it back.
    - After alignment and restoration, take the strict cellwise majority color across the four copies.

    Key insight:
    - The four panels only become redundant after moving them into one shared coordinate system; then the wrong global inversion is the only choice that restores a strong 3-to-1 or 4-to-0 consensus everywhere.
    """
    return _best_reconstruction(_aligned_panels(np.asarray(grid, dtype=int)))


PALETTE = tuple(range(10))
PANEL_SIZE_RANGE = (5, 12)
FILL_RATIO_RANGE = (0.28, 0.62)
GROW_RETRIES = 80
GENERATION_RETRIES = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < h and 0 <= c + dc < w]


def _random_connected_mask(size: int) -> np.ndarray | None:
    target = random.randint(
        max(3, int(np.ceil(size * size * FILL_RATIO_RANGE[0]))),
        min(size * size - 3, int(np.floor(size * size * FILL_RATIO_RANGE[1]))),
    )
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = {start}

    while int(mask.sum()) < target and frontier:
        base_r, base_c = random.choice(tuple(frontier))
        candidates = [(r, c) for r, c in _neighbors(mask.shape, base_r, base_c) if not mask[r, c]]
        if not candidates:
            frontier.remove((base_r, base_c))
            continue
        rr, cc = random.choice(candidates)
        mask[rr, cc] = True
        frontier.add((rr, cc))

    if int(mask.sum()) != target:
        return None
    if not np.any(~mask):
        return None
    if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
        return None
    return mask


def _random_base_pattern(size: int, colors: tuple[int, int]) -> np.ndarray | None:
    for _ in range(GROW_RETRIES):
        mask = _random_connected_mask(size)
        if mask is None:
            continue
        a, b = colors
        panel = np.full((size, size), a, dtype=int)
        panel[mask] = b
        return panel
    return None


def _corrupt_panel_cell(panel: np.ndarray, pos: tuple[int, int], colors: tuple[int, int]) -> None:
    r, c = pos
    a, b = colors
    panel[r, c] = b if int(panel[r, c]) == a else a


def _aligned_to_local(panel_idx: int, pos: tuple[int, int], size: int) -> tuple[int, int]:
    r, c = pos
    if panel_idx == 0:
        return r, c
    if panel_idx == 1:
        return size - 1 - c, r
    if panel_idx == 2:
        return c, size - 1 - r
    if panel_idx == 3:
        return size - 1 - r, size - 1 - c
    raise AssertionError("invalid panel index")


def _build_input() -> np.ndarray | None:
    size = random.randint(*PANEL_SIZE_RANGE)
    panel_colors = tuple(random.sample(PALETTE, 2))
    divider_color = random.choice([c for c in PALETTE if c not in panel_colors])

    base = _random_base_pattern(size, panel_colors)
    if base is None:
        return None

    panels = [
        base.copy(),
        np.rot90(base, 1).copy(),
        np.rot90(base, 3).copy(),
        np.rot90(base, 2).copy(),
    ]
    inverted_idx = random.randrange(4)
    panels[inverted_idx] = _invert_panel(panels[inverted_idx], panel_colors)

    aligned_cells = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(aligned_cells)
    if len(aligned_cells) < 4:
        return None

    # Every panel gets at least one corruption, and each aligned coordinate is corrupted in at most one panel.
    for panel_idx in range(4):
        local = _aligned_to_local(panel_idx, aligned_cells[panel_idx], size)
        _corrupt_panel_cell(panels[panel_idx], local, panel_colors)

    extra_noise_max = max(0, min(size * size - 4, size * size // 6))
    extra_noise = random.randint(0, extra_noise_max)
    for pos in aligned_cells[4 : 4 + extra_noise]:
        panel_idx = random.randrange(4)
        local = _aligned_to_local(panel_idx, pos, size)
        _corrupt_panel_cell(panels[panel_idx], local, panel_colors)

    canvas = np.full((2 * size + 1, 2 * size + 1), divider_color, dtype=int)
    canvas[:size, :size] = panels[0]
    canvas[:size, size + 1 :] = panels[1]
    canvas[size + 1 :, :size] = panels[2]
    canvas[size + 1 :, size + 1 :] = panels[3]

    if not np.array_equal(main(canvas), base):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-vote input")
