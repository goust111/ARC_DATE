from __future__ import annotations

import random

import numpy as np


RED = 2
ARC_COLORS = tuple(range(10))
DIVIDER_COLORS = tuple(color for color in ARC_COLORS if color not in {0, RED})

# Calls rotate through these cases so any first three generated examples include
# a genuinely two-axis roll and both kinds of zero-coordinate counterexample.
_SHIFT_CASES = ("both", "zero_row", "zero_col", "edge")
_generation_count = 0
_seen_output_keys: set[tuple[int, int, int, int, int]] = set()


def _matching_rolls(left: np.ndarray, right: np.ndarray) -> list[tuple[int, int]]:
    """Return all panel-local (dr, dc) rolls that map left exactly to right."""
    h, w = left.shape
    return [
        (dr, dc)
        for dr in range(h)
        for dc in range(w)
        if np.array_equal(np.roll(left, shift=(dr, dc), axis=(0, 1)), right)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the unique cyclic offset between two panels and encode it as one red cell.

    Input structure:
    - One solid vertical divider separates two equally sized multicolor panels.
    - The right panel is a cyclic row-and-column roll of the left panel, and the
      panel pattern has no nonzero cyclic self-alignment.

    Rule:
    - In panel-local coordinates, find the unique pair (dr, dc) for which rolling
      the left panel down dr rows and right dc columns produces the right panel.
    - Return a panel-sized grid in the divider color, with only cell (dr, dc) red.

    Key insight:
    - The output records the transformation parameter itself; its red cell's row
      and column are the two cyclic offsets, rather than a transformed panel image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, total_w = map(int, g.shape)
    assert h >= 2 and total_w >= 3 and total_w % 2 == 1, (
        "a one-column divider must separate equal-width panels"
    )

    divider_col = total_w // 2
    divider_color = int(g[0, divider_col])
    assert divider_color != RED, "the divider must contrast with the red output marker"
    assert np.all(g[:, divider_col] == divider_color), "center column must be a solid divider"

    solid_columns = [
        c for c in range(total_w) if np.all(g[:, c] == int(g[0, c]))
    ]
    assert solid_columns == [divider_col], "expected exactly one solid divider column"

    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    assert left.shape == right.shape and left.shape[1] > 0

    matches = _matching_rolls(left, right)
    assert len(matches) == 1, "panel pattern must determine one unique cyclic offset"
    dr, dc = matches[0]

    out = np.full(left.shape, divider_color, dtype=int)
    out[dr, dc] = RED
    return out


def _manual_roll(panel: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Independent coordinate-level oracle for a down/right cyclic roll."""
    h, w = panel.shape
    out = np.empty_like(panel)
    for r in range(h):
        for c in range(w):
            out[(r + dr) % h, (c + dc) % w] = panel[r, c]
    return out


def _sample_shift(h: int, w: int, case: str) -> tuple[int, int]:
    row_values = [v for v in range(1, h) if 2 * v != h] or list(range(1, h))
    col_values = [v for v in range(1, w) if 2 * v != w] or list(range(1, w))

    if case == "zero_row":
        return 0, random.choice(col_values)
    if case == "zero_col":
        return random.choice(row_values), 0
    if case == "edge":
        choices = [
            (h - 1, random.choice(col_values)),
            (random.choice(row_values), w - 1),
            (h - 1, w - 1),
        ]
        return random.choice(choices)

    pairs = [(dr, dc) for dr in row_values for dc in col_values if dr != dc]
    if not pairs:
        pairs = [(dr, dc) for dr in row_values for dc in col_values]
    return random.choice(pairs)


def _random_patchwork(h: int, w: int, divider_color: int) -> np.ndarray:
    """Make a dense, blob-like multicolor panel whose roll is easy to compare visually."""
    available = [color for color in ARC_COLORS if color != divider_color]
    color_count = random.randint(3, min(5, len(available)))
    if 0 in available and random.random() < 0.75:
        palette = [0] + random.sample([c for c in available if c != 0], color_count - 1)
    else:
        palette = random.sample(available, color_count)

    seed_cells = random.sample([(r, c) for r in range(h) for c in range(w)], color_count)
    rr, cc = np.indices((h, w))
    row_scale = random.uniform(0.75, 1.45)
    col_scale = random.uniform(0.75, 1.45)
    scores = []
    for sr, sc in seed_cells:
        distance = row_scale * np.abs(rr - sr) + col_scale * np.abs(cc - sc)
        noise = np.random.random((h, w)) * random.uniform(0.05, 0.35)
        scores.append(distance + noise + random.uniform(-0.2, 0.2))

    owners = np.argmin(np.stack(scores, axis=0), axis=0)
    palette_array = np.asarray(palette, dtype=int)
    return palette_array[owners]


def _panel_is_nondegenerate(panel: np.ndarray, shifted: np.ndarray) -> bool:
    h, w = panel.shape
    counts = [int(np.sum(panel == color)) for color in np.unique(panel)]
    minimum_count = max(3, (h * w) // 24)
    if len(counts) < 3 or min(counts) < minimum_count or max(counts) > 0.68 * h * w:
        return False

    # Avoid stripes that could be mistaken for another full divider.
    if any(np.all(panel[:, c] == panel[0, c]) for c in range(w)):
        return False
    if any(np.all(panel[r, :] == panel[r, 0]) for r in range(h)):
        return False

    if _matching_rolls(panel, panel) != [(0, 0)]:
        return False
    difference_ratio = float(np.mean(panel != shifted))
    return difference_ratio >= 0.30


def _semantic_candidate_is_valid(
    inp: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    divider_color: int,
    dr: int,
    dc: int,
) -> bool:
    """Check the stated rule independently, including uniqueness and non-degeneracy."""
    h, w = left.shape
    if inp.shape != (h, 2 * w + 1) or max(inp.shape) > 30:
        return False
    if not np.array_equal(right, _manual_roll(left, dr, dc)):
        return False
    if (dr, dc) == (0, 0) or _matching_rolls(left, right) != [(dr, dc)]:
        return False

    center = w
    solid_columns = [
        c for c in range(inp.shape[1]) if np.all(inp[:, c] == int(inp[0, c]))
    ]
    if solid_columns != [center] or not np.all(inp[:, center] == divider_color):
        return False
    if divider_color in set(map(int, np.unique(left))):
        return False
    if not _panel_is_nondegenerate(left, right):
        return False

    # Wrong one-axis shortcuts must fail whenever the corresponding true offset is nonzero.
    if dc != 0 and np.array_equal(_manual_roll(left, dr, 0), right):
        return False
    if dr != 0 and np.array_equal(_manual_roll(left, 0, dc), right):
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    expected = np.full((h, w), divider_color, dtype=int)
    expected[dr, dc] = RED
    if not np.array_equal(solved, expected):
        return False
    return int(np.sum(solved == RED)) == 1 and np.all(solved[solved != RED] == divider_color)


def generate_input() -> np.ndarray:
    """Generate a varied valid pair of panels with scheduled shift counterexamples."""
    global _generation_count

    case = _SHIFT_CASES[_generation_count % len(_SHIFT_CASES)]
    for _ in range(500):
        h = random.randint(5, 12)
        w = random.randint(5, 12)
        divider_color = random.choice(DIVIDER_COLORS)
        dr, dc = _sample_shift(h, w, case)
        output_key = (h, w, divider_color, dr, dc)
        if output_key in _seen_output_keys:
            continue

        left = _random_patchwork(h, w, divider_color)
        right = _manual_roll(left, dr, dc)
        inp = np.concatenate(
            [left, np.full((h, 1), divider_color, dtype=int), right], axis=1
        )
        if not _semantic_candidate_is_valid(inp, left, right, divider_color, dr, dc):
            continue

        _seen_output_keys.add(output_key)
        _generation_count += 1
        return inp

    raise RuntimeError("failed to generate a valid unique cyclic-offset input")
