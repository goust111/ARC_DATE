"""ARC task: extrapolate one generation of an adjacent-gap growth process."""

from __future__ import annotations

import itertools
import random

import numpy as np


ORANGE = 7
PINK = 6


def _offspring(active: np.ndarray) -> np.ndarray:
    """Return midpoints of every visible active-background-active triple."""
    active = np.asarray(active, dtype=bool)
    children = np.zeros_like(active)
    if active.size >= 3:
        children[1:-1] = active[:-2] & ~active[1:-1] & active[2:]
    return children


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    """Parse equal-width panels using their visible shared vertical borders."""
    h, w = grid.shape
    assert h >= 5 and w >= 5, "expected a framed horizontal sequence"
    frame = int(grid[0, 0])
    assert np.all(grid[0] == frame) and np.all(grid[-1] == frame)

    separators = [c for c in range(w) if np.all(grid[:, c] == frame)]
    assert len(separators) >= 3, "expected at least two consecutive panels"
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = np.diff(separators)
    assert np.all(strides == strides[0]) and int(strides[0]) >= 3

    panel_w = int(strides[0]) + 1
    panels = [grid[:, a : b + 1] for a, b in zip(separators, separators[1:])]
    for panel in panels:
        assert panel.shape == (h, panel_w)
        assert np.all(panel[0] == frame) and np.all(panel[-1] == frame)
        assert np.all(panel[:, 0] == frame) and np.all(panel[:, -1] == frame)
    return panels, frame, panel_w


def main(grid: np.ndarray) -> np.ndarray:
    """Advance the framed adjacent-gap growth sequence by one generation.

    Input structure:
    - Equal-width time panels are arranged horizontally and share vertical border
      columns; every panel has the same border color and interior background.
    - The first panel contains one horizontal orange seed generation.  Each later
      panel preserves all earlier generations and adds the next one two rows lower.

    Rule:
    - In each generation, an interior position is born exactly when its left and
      right neighbors are active and its own position is background: active-gap-active.
    - Generation colors alternate orange and pink.  Copy the last panel into a
      standalone complete frame and add one more generation two rows below; if
      there are no qualifying gaps, preserve the last panel without new pixels.

    Key insight:
    - The panels are successive states of a cumulative growth process, so the
      answer advances the local reproduction rule rather than translating a row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    panels, frame, panel_w = _split_panels(g)

    interior_values = np.concatenate([p[1:-1, 1:-1].ravel() for p in panels])
    values, counts = np.unique(interior_values, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    assert background not in (frame, ORANGE, PINK)
    assert set(map(int, values)) <= {background, ORANGE, PINK}

    first = panels[0]
    seed_coords = np.argwhere(first[1:-1, 1:-1] != background)
    assert seed_coords.size and len(set(map(int, seed_coords[:, 0]))) == 1
    seed_row = int(seed_coords[0, 0]) + 1
    assert np.all(first[seed_row, 1:-1][first[seed_row, 1:-1] != background] == ORANGE)
    expected = np.full_like(first, background)
    expected[0, :] = expected[-1, :] = frame
    expected[:, 0] = expected[:, -1] = frame
    active = first[seed_row, 1:-1] == ORANGE
    expected[seed_row, 1:-1][active] = ORANGE
    assert np.array_equal(first, expected), "first panel must contain only the seed row"

    current_row = seed_row
    current_color = ORANGE
    for generation, panel in enumerate(panels[1:], start=1):
        next_row = current_row + 2
        assert next_row < g.shape[0] - 1, "generation must remain inside the frame"
        children = _offspring(active)
        assert np.any(children), "observed adjacent frames must add one nonempty generation"
        next_color = PINK if current_color == ORANGE else ORANGE
        expected = expected.copy()
        expected[next_row, 1:-1][children] = next_color
        assert np.array_equal(panel, expected), f"panel {generation + 1} is not the next growth state"
        active, current_row, current_color = children, next_row, next_color

    output_row = current_row + 2
    assert output_row < g.shape[0] - 1, "next generation must fit inside the output frame"
    children = _offspring(active)
    out = panels[-1].copy()
    if np.any(children):
        next_color = PINK if current_color == ORANGE else ORANGE
        out[output_row, 1:-1][children] = next_color
    return out


_GEN_CALL_INDEX = 0
_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _candidate_seeds(width: int, require_next: bool) -> list[np.ndarray]:
    """Enumerate seeds with two visible growth steps and an anti-copy distractor."""
    candidates: list[np.ndarray] = []
    for bits in itertools.product((False, True), repeat=width):
        seed = np.asarray(bits, dtype=bool)
        gen1 = _offspring(seed)
        gen2 = _offspring(gen1)
        gen3 = _offspring(gen2)
        has_adjacent_pair = bool(np.any(seed[:-1] & seed[1:]))
        if (
            np.any(gen1)
            and np.any(gen2)
            and bool(np.any(gen3)) == require_next
            and has_adjacent_pair
            and not np.array_equal(seed, gen1)
        ):
            candidates.append(seed)
    return candidates


def _make_panel(
    height: int,
    panel_w: int,
    frame: int,
    background: int,
    generations: list[np.ndarray],
    seed_row: int,
) -> np.ndarray:
    panel = np.full((height, panel_w), background, dtype=int)
    panel[0, :] = panel[-1, :] = frame
    panel[:, 0] = panel[:, -1] = frame
    for index, active in enumerate(generations):
        color = ORANGE if index % 2 == 0 else PINK
        panel[seed_row + 2 * index, 1:-1][active] = color
    return panel


def _join_shared(panels: list[np.ndarray]) -> np.ndarray:
    stride = panels[0].shape[1] - 1
    joined = np.empty((panels[0].shape[0], len(panels) * stride + 1), dtype=int)
    for index, panel in enumerate(panels):
        joined[:, index * stride : index * stride + panel.shape[1]] = panel
    return joined


def generate_input() -> np.ndarray:
    """Generate varied, uniquely parsed examples covering both prediction branches."""
    global _GEN_CALL_INDEX
    require_next = bool(_GEN_CALL_INDEX % 2)
    _GEN_CALL_INDEX += 1

    palette = [color for color in range(10) if color not in (ORANGE, PINK)]
    for _ in range(500):
        interior_w = 8 if require_next else random.choice((7, 8))
        panel_w = interior_w + 2
        height = random.randint(9, 16)
        seed_row = random.randint(1, height - 8)
        frame, background = random.sample(palette, 2)
        seed = random.choice(_candidate_seeds(interior_w, require_next)).copy()

        generations = [seed]
        generations.append(_offspring(generations[-1]))
        generations.append(_offspring(generations[-1]))
        panels = [
            _make_panel(height, panel_w, frame, background, generations[: count + 1], seed_row)
            for count in range(3)
        ]
        inp = _join_shared(panels)
        expected = _make_panel(
            height,
            panel_w,
            frame,
            background,
            generations + [_offspring(generations[-1])],
            seed_row,
        )
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        if max(inp.shape) > 30 or max(actual.shape) > 30:
            continue
        input_key = (tuple(map(int, inp.shape)), inp.tobytes())
        output_key = (tuple(map(int, actual.shape)), actual.tobytes())
        if input_key in _SEEN_INPUTS or output_key in _SEEN_OUTPUTS:
            continue
        _SEEN_INPUTS.add(input_key)
        _SEEN_OUTPUTS.add(output_key)
        return inp
    raise RuntimeError("failed to generate a fresh valid growth sequence")
