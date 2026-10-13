from __future__ import annotations

import numpy as np


BACKGROUND = 0
NEGATIVE_BG = 8
INPUT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

MIN_RECT_SIZE = 3
MAX_RECT_SIZE = 7
MIN_OBJECTS = 2
MAX_OBJECTS = 3
MIN_HEIGHT = 8
MAX_HEIGHT = 14
MIN_WIDTH = 8
MAX_WIDTH = 15
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

    return components


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    components = _find_components(grid)
    assert components, "expected at least one foreground rectangle"

    rectangles: list[dict[str, int]] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= MIN_RECT_SIZE and width >= MIN_RECT_SIZE, "rectangles must have nonempty interiors"
        assert int(comp["area"]) == height * width, "each component must be a filled rectangle"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "rectangle bbox must be monochromatic and solid"
        rectangles.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})

    return rectangles


def _render_output(shape: tuple[int, int], rectangles: list[dict[str, int]], touches_left: bool) -> np.ndarray:
    h, w = map(int, shape)
    mirrored_half = np.full((h, w), NEGATIVE_BG, dtype=int)

    for rect in rectangles:
        r0 = int(rect["r0"])
        r1 = int(rect["r1"])
        c0 = int(rect["c0"])
        c1 = int(rect["c1"])
        color = int(rect["color"])

        mc0 = w - 1 - c1
        mc1 = w - 1 - c0
        mirrored_half[r0 : r1 + 1, mc0 : mc1 + 1] = color
        mirrored_half[r0 + 1 : r1, mc0 + 1 : mc1] = BACKGROUND

    original = np.zeros((h, w), dtype=int)
    for rect in rectangles:
        original[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = int(rect["color"])

    if touches_left:
        return np.concatenate((mirrored_half, original), axis=1)
    return np.concatenate((original, mirrored_half), axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-concatenate the border-touching rectangles, but invert only the interior of each mirrored rectangle.

    Input structure:
    - The grid is a black background with several filled axis-aligned rectangles in non-black colors.
    - Every rectangle has height and width at least 3, so each one contains a one-cell border and a nonempty interior.
    - All rectangles touch the same vertical border of the grid, either the left edge or the right edge.

    Rule:
    - Detect the shared touched vertical border and horizontally mirror the whole grid across that edge.
    - Concatenate the mirrored copy on the touched side: mirrored-left + original if the rectangles touch the left border, otherwise original + mirrored-right.
    - Keep the original half unchanged.
    - In the mirrored half, recolor background cells to 8, keep every mirrored rectangle border in its original color, and recolor every mirrored rectangle interior cell to black.

    Key insight:
    - The mirrored side is not a global negative; only rectangle interiors undergo the foreground-to-background half of the negative mapping, while the rectangle borders stay colored and the external background becomes cyan.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h > 0 and w > 0, "expected a nonempty grid"

    rectangles = _extract_rectangles(g)
    foreground = g != BACKGROUND
    touches_left = bool(np.any(foreground[:, 0]))
    touches_right = bool(np.any(foreground[:, -1]))
    assert touches_left ^ touches_right, "foreground must touch exactly one vertical border"

    for rect in rectangles:
        if touches_left:
            assert int(rect["c0"]) == 0, "every rectangle must touch the left border"
        else:
            assert int(rect["c1"]) == w - 1, "every rectangle must touch the right border"

    return _render_output(g.shape, rectangles, touches_left)


def _split_extra(total: int, slots: int, rng: np.random.Generator) -> np.ndarray:
    if total == 0:
        return np.zeros(slots, dtype=int)
    probs = np.full(slots, 1.0 / slots, dtype=float)
    return rng.multinomial(total, probs)


def _sample_rows(count: int, height: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    min_needed = count * MIN_RECT_SIZE + (count - 1)
    if min_needed > height:
        raise RuntimeError("height is too small for the requested rectangle count")

    rect_heights = np.full(count, MIN_RECT_SIZE, dtype=int)
    slack = height - min_needed
    extra_height_budget = int(rng.integers(0, min(slack, count * (MAX_RECT_SIZE - MIN_RECT_SIZE)) + 1))
    for _ in range(extra_height_budget):
        eligible = np.flatnonzero(rect_heights < MAX_RECT_SIZE)
        if eligible.size == 0:
            break
        idx = int(rng.choice(eligible))
        rect_heights[idx] += 1

    extras = _split_extra(slack - extra_height_budget, count + 1, rng)
    rows: list[tuple[int, int]] = []
    r = int(extras[0])
    for idx, rect_h in enumerate(rect_heights.tolist()):
        rows.append((r, int(rect_h)))
        r += int(rect_h)
        if idx < count - 1:
            r += 1 + int(extras[idx + 1])
    return rows


def _expected_from_rectangles(grid: np.ndarray, rectangles: list[dict[str, int]], touches_left: bool) -> np.ndarray:
    expected = _render_output(grid.shape, rectangles, touches_left)
    assert np.array_equal(expected[:, : grid.shape[1]], grid) or np.array_equal(expected[:, grid.shape[1] :], grid)
    return expected


def _make_rectangles(height: int, width: int, count: int, touches_left: bool, rng: np.random.Generator) -> list[dict[str, int]] | None:
    rows = _sample_rows(count, height, rng)
    max_rect_width = min(MAX_RECT_SIZE, width - 1)
    if max_rect_width < MIN_RECT_SIZE:
        return None

    rect_widths = rng.integers(MIN_RECT_SIZE, max_rect_width + 1, size=count)
    row_heights = np.array([rh for _, rh in rows], dtype=int)
    eligible = np.flatnonzero(row_heights >= 4)
    if eligible.size and max_rect_width >= 4 and not np.any((rect_widths >= 4) & (row_heights >= 4)):
        idx = int(rng.choice(eligible))
        rect_widths[idx] = max(int(rect_widths[idx]), 4)

    colors = rng.choice(INPUT_COLORS, size=count, replace=False)
    rectangles: list[dict[str, int]] = []
    for (r0, rect_h), rect_w, color in zip(rows, rect_widths.tolist(), colors.tolist()):
        rect_w = int(rect_w)
        c0 = 0 if touches_left else width - rect_w
        rectangles.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + rect_h - 1),
                "c1": int(c0 + rect_w - 1),
            }
        )
    return rectangles


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    h, w = map(int, inp.shape)
    assert out.shape == (h, 2 * w)
    rectangles = _extract_rectangles(inp)
    fg = inp != BACKGROUND
    touches_left = bool(np.any(fg[:, 0]))
    touches_right = bool(np.any(fg[:, -1]))
    assert touches_left ^ touches_right

    original_half = out[:, w:] if touches_left else out[:, :w]
    mirror_half = out[:, :w] if touches_left else out[:, w:]
    assert np.array_equal(original_half, inp)

    reflected = np.fliplr(inp)
    bg_mask = reflected == BACKGROUND
    assert np.all(mirror_half[bg_mask] == NEGATIVE_BG)

    for rect in rectangles:
        r0 = int(rect["r0"])
        r1 = int(rect["r1"])
        c0 = int(rect["c0"])
        c1 = int(rect["c1"])
        color = int(rect["color"])
        mc0 = w - 1 - c1
        mc1 = w - 1 - c0

        assert np.all(mirror_half[r0, mc0 : mc1 + 1] == color)
        assert np.all(mirror_half[r1, mc0 : mc1 + 1] == color)
        assert np.all(mirror_half[r0 : r1 + 1, mc0] == color)
        assert np.all(mirror_half[r0 : r1 + 1, mc1] == color)
        assert np.all(mirror_half[r0 + 1 : r1, mc0 + 1 : mc1] == BACKGROUND)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    min_height = max(MIN_HEIGHT, count * MIN_RECT_SIZE + (count - 1))
    height = int(rng.integers(min_height, MAX_HEIGHT + 1))
    width = int(rng.integers(MIN_WIDTH, MAX_WIDTH + 1))
    touches_left = bool(rng.integers(0, 2))

    rectangles = _make_rectangles(height, width, count, touches_left, rng)
    if rectangles is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for rect in rectangles:
        grid[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = int(rect["color"])

    expected = _expected_from_rectangles(grid, rectangles, touches_left)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        _semantic_checks(grid, actual)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
