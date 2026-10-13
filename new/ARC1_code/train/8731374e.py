from __future__ import annotations

import itertools

import numpy as np

from arc_utils import colors_present, paste


def _find_marked_rectangle(g: np.ndarray) -> tuple[int, int, int, int, int, int]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, int, int, int, int, int]] = []

    for a, b in itertools.combinations(colors_present(g), 2):
        allowed = (g == a) | (g == b)
        run_heights = np.zeros(width, dtype=int)

        for r1 in range(height):
            run_heights = np.where(allowed[r1], run_heights + 1, 0)
            for c0 in range(width):
                rect_h = height + 1
                for c1 in range(c0, width):
                    if run_heights[c1] == 0:
                        break
                    rect_h = min(rect_h, int(run_heights[c1]))
                    rect_w = c1 - c0 + 1
                    if rect_h < 3 or rect_w < 3:
                        continue

                    r0 = r1 - rect_h + 1
                    crop = g[r0 : r1 + 1, c0 : c1 + 1]
                    count_a = int(np.count_nonzero(crop == a))
                    count_b = int(np.count_nonzero(crop == b))
                    if count_a == 0 or count_b == 0 or count_a == count_b:
                        continue

                    base, marker = (a, b) if count_a > count_b else (b, a)
                    marker_coords = np.argwhere(crop == marker)
                    interior = (
                        (marker_coords[:, 0] > 0)
                        & (marker_coords[:, 0] < rect_h - 1)
                        & (marker_coords[:, 1] > 0)
                        & (marker_coords[:, 1] < rect_w - 1)
                    )
                    if not bool(np.all(interior)):
                        continue

                    area = rect_h * rect_w
                    candidates.append((area, r0, c0, r1, c1, int(base), int(marker)))

    assert candidates, "expected a marked two-color rectangle"
    max_area = max(area for area, *_rest in candidates)
    winners = [candidate for candidate in candidates if candidate[0] == max_area]
    assert len(winners) == 1, "expected a unique largest marked rectangle"
    _area, r0, c0, r1, c1, base, marker = winners[0]
    return r0, c0, r1, c1, base, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each sparse marker inside the dense rectangle into a full marker-color row and column.

    Input structure
    - A noisy grid contains one largest rectangle using exactly two colors.
    - Within that rectangle, one color is the dominant base and the other appears only as sparse interior marker pixels.

    Rule
    - Crop to the marked rectangle's shape.
    - Start with the rectangle's base color everywhere, then for every marker pixel paint its whole row and whole column with the marker color.

    Key insight
    - The small marker pixels are row/column controls; the surrounding noisy cells are distractors and the output keeps only the expanded marked rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1, base, marker = _find_marked_rectangle(g)
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    marker_coords = np.argwhere(crop == marker)
    assert marker_coords.size > 0, "expected at least one marker pixel"

    out = np.full(crop.shape, base, dtype=int)
    for r, c in marker_coords:
        out[int(r), :] = marker
        out[:, int(c)] = marker

    return out

# --------------- generator (o5x) ---------------
# Rectangle, marker, and palette sampling.
RECT_H_RANGE, RECT_W_RANGE = (6, 11), (6, 10)
MARKER_COUNT_RANGE, INTERIOR_GAP = (1, 4), 1
NON_BLACK_COLORS = np.arange(1, 10)
# Canvas placement, noise, and retry policy.
GRID_H_RANGE, GRID_W_RANGE = (16, 28), (17, 24)
PLACEMENT_GAP, NOISE_COLOR_RANGE = 2, (0, 10)
MAX_ATTEMPTS = 300

def _render_rectangles(shape, base, marker, rows, cols):
    """Return the sparse input rectangle and cross-expanded oracle output."""
    rect = np.full(shape, base, dtype=int)
    expected = np.full(shape, base, dtype=int)
    rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    expected[rows, :] = marker
    expected[:, cols] = marker
    rect[rows, cols] = marker
    return rect, expected

def _sample_canvas(rng: np.random.Generator, rect: np.ndarray) -> np.ndarray:
    """Fill a noisy canvas and paste the rectangle with the required edge gap."""
    rect_h, rect_w = rect.shape
    grid_h = int(rng.integers(max(GRID_H_RANGE[0], rect_h + 2 * PLACEMENT_GAP), GRID_H_RANGE[1]))
    grid_w = int(rng.integers(max(GRID_W_RANGE[0], rect_w + 2 * PLACEMENT_GAP), GRID_W_RANGE[1]))
    grid = rng.integers(*NOISE_COLOR_RANGE, size=(grid_h, grid_w), dtype=int)
    r0 = int(rng.integers(PLACEMENT_GAP, grid_h - rect_h - PLACEMENT_GAP + 1))
    c0 = int(rng.integers(PLACEMENT_GAP, grid_w - rect_w - PLACEMENT_GAP + 1))
    paste(grid, rect, r0, c0, transparent=None)
    return grid

def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rect_h = int(rng.integers(*RECT_H_RANGE))
    rect_w = int(rng.integers(*RECT_W_RANGE))
    base, marker = map(int, rng.choice(NON_BLACK_COLORS, size=2, replace=False))
    n_markers = int(rng.integers(*MARKER_COUNT_RANGE))
    marker_rows = rng.choice(np.arange(INTERIOR_GAP, rect_h - INTERIOR_GAP), size=n_markers, replace=False)
    marker_cols = rng.choice(np.arange(INTERIOR_GAP, rect_w - INTERIOR_GAP), size=n_markers, replace=False)
    rect, expected = _render_rectangles((rect_h, rect_w), base, marker, marker_rows, marker_cols)
    grid = _sample_canvas(rng, rect)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None

def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8731374e input")
