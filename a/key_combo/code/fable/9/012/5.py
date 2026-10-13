from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
TRANSFORMS = ("mirror_lr", "mirror_ud", "rotate_cw", "rotate_180")
PANEL_COUNT = 7
PANEL_SIZE_CHOICES = (3,)
COLOR_POOL = tuple(range(1, 10))
BASE_ATTEMPTS = 400
CHAIN_ATTEMPTS = 400
GENERATION_ATTEMPTS = 400


def _apply_transform(panel: np.ndarray, transform: str) -> np.ndarray:
    if transform == "mirror_lr":
        return np.fliplr(panel)
    if transform == "mirror_ud":
        return np.flipud(panel)
    if transform == "rotate_cw":
        return np.rot90(panel, -1)
    if transform == "rotate_180":
        return np.rot90(panel, 2)
    raise ValueError(f"unknown transform: {transform}")


def _apply_inverse_transform(panel: np.ndarray, transform: str) -> np.ndarray:
    if transform == "rotate_cw":
        return np.rot90(panel, 1)
    return _apply_transform(panel, transform)


def _all_dihedral_variants(panel: np.ndarray) -> list[np.ndarray]:
    return [
        panel,
        np.rot90(panel, -1),
        np.rot90(panel, 2),
        np.rot90(panel, 1),
        np.fliplr(panel),
        np.flipud(panel),
        panel.T,
        np.rot90(panel, 2).T,
    ]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _matching_transforms(prev_panel: np.ndarray, next_panel: np.ndarray) -> list[str]:
    return [
        transform
        for transform in TRANSFORMS
        if np.array_equal(_apply_transform(prev_panel, transform), next_panel)
    ]


def _parse_grid(grid: np.ndarray) -> tuple[list[np.ndarray], list[int | None], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    separator = int(g[-1, 0])
    assert np.all(g[-1] == separator), "bottom separator band must be uniform"

    panel_size = g.shape[0] - 2
    assert panel_size >= 3, "panels must have positive interior"

    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == separator)]
    panel_runs: list[tuple[int, int]] = []
    start = None
    for c in range(g.shape[1]):
        if c in separator_cols:
            if start is not None:
                panel_runs.append((start, c - 1))
                start = None
        elif start is None:
            start = c
    if start is not None:
        panel_runs.append((start, g.shape[1] - 1))

    assert panel_runs, "expected at least one panel run"
    assert all(c1 - c0 + 1 == panel_size for c0, c1 in panel_runs), "all panels must be square"
    panel_count = len(panel_runs)
    assert panel_count >= 4, "need several panels to decode the chain"

    panels: list[np.ndarray] = []
    symbols: list[int | None] = [None]
    for idx, (c0, c1) in enumerate(panel_runs):
        panels.append(g[1 : 1 + panel_size, c0 : c1 + 1].copy())
        if idx > 0:
            symbol_col = c0 + panel_size // 2
            symbols.append(int(g[0, symbol_col]))

    flat_panels = [idx for idx, panel in enumerate(panels) if np.all(panel == int(panel[0, 0]))]
    assert len(flat_panels) == 1, "expected exactly one fully blank panel"
    missing_idx = flat_panels[0]
    assert 0 < missing_idx < panel_count - 1, "missing panel must be non-terminal"
    return panels, symbols, missing_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the per-symbol rigid-motion chain and reconstruct the missing square panel.

    Input structure:
    - A separator color divides the grid into one horizontal row of equal-size square panels.
    - The leftmost panel is the seed; for every later panel, the top separator band contains one
      symbol pixel whose color encodes the rigid transform from the previous panel.
    - Exactly one non-endpoint panel is blanked out to pure background, while its symbol pixel remains visible.

    Rule:
    - For every adjacent pair of visible panels, compare the later panel against the earlier one and
      infer which rigid transform its symbol color stands for.
    - Use the decoded symbol above the missing panel to transform the left neighbor forward by one step.
    - Cross-check with the right neighbor by applying the inverse of the next symbol's transform; both
      directions must agree, and the agreed square is the output.

    Key insight:
    - The symbol-color codebook is not global and must be solved inside each input from the other
      visible panel transitions, so the missing panel can only be recovered after decoding the chain.
    """
    panels, symbols, missing_idx = _parse_grid(grid)

    symbol_to_transform: dict[int, str] = {}
    for idx in range(1, len(panels)):
        if idx == missing_idx or idx - 1 == missing_idx:
            continue
        symbol = symbols[idx]
        assert symbol is not None
        matches = _matching_transforms(panels[idx - 1], panels[idx])
        assert len(matches) == 1, "visible panel pair must imply exactly one transform"
        if symbol in symbol_to_transform:
            assert symbol_to_transform[symbol] == matches[0], "symbol mapping must stay consistent"
        else:
            symbol_to_transform[symbol] = matches[0]

    left_symbol = symbols[missing_idx]
    right_symbol = symbols[missing_idx + 1]
    assert left_symbol in symbol_to_transform, "missing symbol must be decodable from other pairs"
    assert right_symbol in symbol_to_transform, "right-adjacent symbol must be decodable from other pairs"

    from_left = _apply_transform(panels[missing_idx - 1], symbol_to_transform[left_symbol])
    from_right = _apply_inverse_transform(panels[missing_idx + 1], symbol_to_transform[right_symbol])
    assert np.array_equal(from_left, from_right), "both directions must reconstruct the same panel"
    return from_left


def _random_base_panel(size: int, fg_colors: list[int]) -> np.ndarray:
    palette = [BACKGROUND, *fg_colors]
    for _ in range(BASE_ATTEMPTS):
        probs = np.random.dirichlet(np.ones(len(palette)))
        probs[0] = min(max(probs[0], 0.20), 0.45)
        probs[1:] = probs[1:] / probs[1:].sum() * (1.0 - probs[0])
        panel = np.random.choice(palette, size=(size, size), p=probs).astype(int)

        for color in fg_colors:
            if not np.any(panel == color):
                r = random.randrange(size)
                c = random.randrange(size)
                panel[r, c] = color

        if not np.any(panel == BACKGROUND):
            continue
        if any(int(np.sum(panel == color)) < 2 for color in fg_colors):
            continue
        mask = panel != BACKGROUND
        if _bbox(mask) != (0, 0, size - 1, size - 1):
            continue
        filled = int(mask.sum())
        if not (max(size + 1, 2 * len(fg_colors)) <= filled <= size * size - 2):
            continue
        if len({variant.tobytes() for variant in _all_dihedral_variants(panel)}) != 8:
            continue
        return panel
    raise RuntimeError("failed to build a non-degenerate base panel")


def _sequence_states(sequence: list[str], size: int) -> list[np.ndarray] | None:
    marker = np.arange(size * size, dtype=int).reshape(size, size)
    states = [marker]
    seen = {marker.tobytes()}
    for transform in sequence:
        nxt = _apply_transform(states[-1], transform)
        key = nxt.tobytes()
        if key in seen:
            return None
        seen.add(key)
        states.append(nxt)
    return states


def _build_transform_sequence() -> tuple[list[str], int]:
    for _ in range(CHAIN_ATTEMPTS):
        missing_idx = random.randint(1, PANEL_COUNT - 2)
        visible_order = random.sample(list(TRANSFORMS), k=4)
        if random.random() < 0.35:
            hidden_left = hidden_right = random.choice(visible_order)
        else:
            hidden_left, hidden_right = random.sample(visible_order, k=2)

        visible_iter = iter(visible_order)
        sequence: list[str] = []
        for pos in range(PANEL_COUNT - 1):
            if pos == missing_idx - 1:
                sequence.append(hidden_left)
            elif pos == missing_idx:
                sequence.append(hidden_right)
            else:
                sequence.append(next(visible_iter))

        if _sequence_states(sequence, 4) is not None:
            return sequence, missing_idx
    raise RuntimeError("failed to build a valid transform chain")


def _render_grid(
    panels: list[np.ndarray],
    sequence: list[str],
    separator: int,
    transform_to_symbol: dict[str, int],
    left_border: bool,
    right_border: bool,
) -> np.ndarray:
    panel_size = panels[0].shape[0]
    width = PANEL_COUNT * panel_size + (PANEL_COUNT - 1) + int(left_border) + int(right_border)
    grid = np.full((panel_size + 2, width), separator, dtype=int)
    start_col = int(left_border)
    for idx, panel in enumerate(panels):
        c0 = start_col + idx * (panel_size + 1)
        grid[1 : 1 + panel_size, c0 : c0 + panel_size] = panel
        if idx > 0:
            symbol_col = c0 + panel_size // 2
            grid[0, symbol_col] = transform_to_symbol[sequence[idx - 1]]
    return grid


def _all_consistent_mappings(
    panels: list[np.ndarray], symbols: list[int | None], missing_idx: int
) -> list[dict[int, str]]:
    symbol_colors = sorted({int(symbol) for symbol in symbols[1:] if symbol is not None})
    if len(symbol_colors) != 4:
        return []

    valid: list[dict[int, str]] = []
    for perm in itertools.permutations(TRANSFORMS):
        mapping = dict(zip(symbol_colors, perm))
        ok = True
        for idx in range(1, len(panels)):
            if idx == missing_idx or idx - 1 == missing_idx:
                continue
            symbol = symbols[idx]
            assert symbol is not None
            if not np.array_equal(_apply_transform(panels[idx - 1], mapping[symbol]), panels[idx]):
                ok = False
                break
        if not ok:
            continue
        left = _apply_transform(panels[missing_idx - 1], mapping[symbols[missing_idx]])
        right = _apply_inverse_transform(panels[missing_idx + 1], mapping[symbols[missing_idx + 1]])
        if np.array_equal(left, right):
            valid.append(mapping)
    return valid


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    panel_size = random.choice(PANEL_SIZE_CHOICES)
    fg_count = random.choice((2, 3))
    chosen = random.sample(COLOR_POOL, k=1 + 4 + fg_count)
    separator = chosen[0]
    symbol_colors = chosen[1:5]
    fg_colors = chosen[5:]

    base_panel = _random_base_panel(panel_size, fg_colors)
    sequence, missing_idx = _build_transform_sequence()
    transform_to_symbol = dict(zip(random.sample(list(TRANSFORMS), k=4), symbol_colors))
    left_border = bool(random.getrandbits(1))
    right_border = bool(random.getrandbits(1))

    full_panels = [base_panel]
    for transform in sequence:
        full_panels.append(_apply_transform(full_panels[-1], transform))

    missing_panel = full_panels[missing_idx].copy()
    assert not any(
        np.array_equal(missing_panel, panel)
        for idx, panel in enumerate(full_panels)
        if idx != missing_idx
    )

    visible_panels = [panel.copy() for panel in full_panels]
    visible_panels[missing_idx][:] = BACKGROUND
    grid = _render_grid(
        visible_panels,
        sequence,
        separator,
        transform_to_symbol,
        left_border,
        right_border,
    )

    symbols: list[int | None] = [None]
    for transform in sequence:
        symbols.append(transform_to_symbol[transform])

    for idx in range(1, len(full_panels)):
        if idx == missing_idx or idx - 1 == missing_idx:
            continue
        assert len(_matching_transforms(full_panels[idx - 1], full_panels[idx])) == 1

    mappings = _all_consistent_mappings(visible_panels, symbols, missing_idx)
    assert len(mappings) == 1, "visible chain must imply a unique symbol codebook"
    assert np.array_equal(main(grid), missing_panel)
    return grid, missing_panel


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid, expected = _build_case()
        out = main(grid)
        if out.shape == expected.shape and np.array_equal(out, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
