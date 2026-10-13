from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEMPLATE_COLORS = (3, 4, 6, 7, 8, 9)
OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Use only signatures whose colored cells span the full 3x3 bbox, so the library
# encodes a unique center even though the center pixel itself is blue.
SIGNATURES: tuple[frozenset[tuple[int, int]], ...] = tuple(
    frozenset(OFFSETS[i] for i in range(8) if (mask >> i) & 1)
    for mask in range(1, 1 << 8)
    if 3 <= mask.bit_count() <= 6
    and any(OFFSETS[i][0] == -1 for i in range(8) if (mask >> i) & 1)
    and any(OFFSETS[i][0] == 1 for i in range(8) if (mask >> i) & 1)
    and any(OFFSETS[i][1] == -1 for i in range(8) if (mask >> i) & 1)
    and any(OFFSETS[i][1] == 1 for i in range(8) if (mask >> i) & 1)
)


def _checker_color(r: int, c: int) -> int:
    return BLACK if (r + c) % 2 == 0 else GREY


def _paint_checkerboard(grid: np.ndarray, row_start: int) -> None:
    h, w = grid.shape
    for r in range(row_start, h):
        for c in range(w):
            grid[r, c] = _checker_color(r, c)


def _parse_library(grid: np.ndarray) -> dict[frozenset[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 7
    lib = g[:3]
    colors = sorted(int(v) for v in np.unique(lib) if int(v) not in {BLUE})
    assert colors, "expected at least one stamp template in the blue library band"

    signatures: dict[frozenset[tuple[int, int]], int] = {}
    for color in colors:
        pts = np.argwhere(lib == color)
        assert pts.size > 0
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        assert int(r0) == 0 and int(r1) == 2, "template color must span all three library rows"
        assert int(c1) - int(c0) == 2, "template color must span a full 3-column stamp bbox"

        center_c = int(c0) + 1
        sig = frozenset((int(r) - 1, int(c) - center_c) for r, c in pts)
        assert sig and (0, 0) not in sig
        assert sig not in signatures, "stamp signatures must be unique"
        signatures[sig] = color

    return signatures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read 3x3 stamp signatures from the blue library band, match each marker's
    surrounding red pattern to a signature, and repaint that signature in the
    marker's own color on the checkerboard.
    """
    g = np.asarray(grid, dtype=int)
    signatures = _parse_library(g)
    valid_marker_colors = set(signatures.values())
    h, w = g.shape

    out = g.copy()
    for r in range(3, h):
        for c in range(w):
            color = int(g[r, c])
            if color not in valid_marker_colors:
                continue

            sig = frozenset(
                (dr, dc)
                for dr, dc in OFFSETS
                if 0 <= r + dr < h and 0 <= c + dc < w and int(g[r + dr, c + dc]) == RED
            )
            assert sig in signatures, "every marker's red neighborhood must match a library signature"

            for dr, dc in sig:
                out[r + dr, c + dc] = color

    for r, c in np.argwhere(out == RED):
        out[r, c] = _checker_color(int(r), int(c))

    return out


def _render_library(width: int, templates: list[tuple[int, frozenset[tuple[int, int]]]]) -> np.ndarray:
    lib = np.full((3, width), BLUE, dtype=int)
    col = 0
    for i, (color, sig) in enumerate(templates):
        for dr, dc in sig:
            lib[1 + dr, col + 1 + dc] = color
        col += 3
        if i + 1 < len(templates):
            col += random.randint(1, 3)
    return lib


def _sample_template_layout(template_count: int) -> tuple[int, list[int]]:
    gaps = [random.randint(1, 3) for _ in range(template_count - 1)]
    width = template_count * 3 + sum(gaps) + random.randint(0, 4)
    return width, gaps


def _centers_available(height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(4, height - 1) for c in range(1, width - 1)]


def _choose_marker_centers(height: int, width: int, marker_count: int) -> list[tuple[int, int]] | None:
    candidates = _centers_available(height, width)
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) > 2 or abs(c - cc) > 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == marker_count:
                return chosen
    return None


def _make_input() -> np.ndarray | None:
    template_count = random.randint(3, 5)
    library_colors = random.sample(TEMPLATE_COLORS, template_count)
    library_signatures = random.sample(list(SIGNATURES), template_count)
    templates = list(zip(library_colors, library_signatures))

    width, gaps = _sample_template_layout(template_count)
    min_height = 10
    max_height = 18
    height = random.randint(min_height, max_height)
    marker_count = random.randint(2, min(6, (height - 5) * max(1, width - 2) // 10))
    centers = _choose_marker_centers(height, width, marker_count)
    if centers is None:
        return None

    grid = np.empty((height, width), dtype=int)
    grid[:3, :] = BLUE
    _paint_checkerboard(grid, 3)

    col = 0
    for i, (color, sig) in enumerate(templates):
        for dr, dc in sig:
            grid[1 + dr, col + 1 + dc] = color
        col += 3
        if i < len(gaps):
            col += gaps[i]

    marker_colors = [random.choice(library_colors) for _ in range(marker_count)]
    marker_signatures = [random.choice(library_signatures) for _ in range(marker_count)]

    if template_count > 1 and all(
        marker_signatures[i] == dict(templates)[marker_colors[i]] for i in range(marker_count)
    ):
        idx = random.randrange(marker_count)
        alternatives = [sig for sig in library_signatures if sig != dict(templates)[marker_colors[idx]]]
        marker_signatures[idx] = random.choice(alternatives)

    for (r, c), color, sig in zip(centers, marker_colors, marker_signatures):
        grid[r, c] = color
        for dr, dc in sig:
            rr, cc = r + dr, c + dc
            grid[rr, cc] = RED

    try:
        out = main(grid)
    except Exception:
        return None

    for (r, c), color, sig in zip(centers, marker_colors, marker_signatures):
        assert int(out[r, c]) == color
        for dr, dc in sig:
            assert int(out[r + dr, c + dc]) == color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _make_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-library puzzle")
