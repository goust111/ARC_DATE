"""ARC task: repair a mirror half encoded by a color-bijection codebook."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color; generated grids make this role unique."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], "background must be unique"
    return int(ordered[0][0])


def _split_halves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Crop the equal halves around the specified central separator column."""
    h, full_w = grid.shape
    assert full_w >= 5 and full_w % 2 == 1, "a central column must split equal halves"
    mid = full_w // 2
    separator = grid[:, mid]
    assert np.all(separator == separator[0]), "central separator must be monochrome"
    sep_color = int(separator[0])
    assert not np.any(grid[:, :mid] == sep_color)
    assert not np.any(grid[:, mid + 1 :] == sep_color)
    return grid[:, :mid].copy(), grid[:, mid + 1 :].copy(), sep_color


def _marker_and_side(
    grid: np.ndarray, half_w: int, excluded: set[int]
) -> tuple[int, str]:
    """Find the unique twice-occurring aligned color confined to one half."""
    candidates: list[tuple[int, str]] = []
    for color in map(int, np.unique(grid)):
        if color in excluded:
            continue
        pts = np.argwhere(grid == color)
        if len(pts) != 2:
            continue
        if not (pts[0, 0] == pts[1, 0] or pts[0, 1] == pts[1, 1]):
            continue
        cols = pts[:, 1]
        if bool(np.all(cols < half_w)):
            candidates.append((color, "left"))
        elif bool(np.all(cols > half_w)):
            candidates.append((color, "right"))
    assert len(candidates) == 1, "exactly one aligned marker pair must select a half"
    return candidates[0]


def _decode_map(
    polluted: np.ndarray, healthy_mirror: np.ndarray, bg: int, marker: int
) -> dict[int, int]:
    """Infer polluted-color -> true-color from all ordinary non-hole pairs."""
    pairs: dict[int, int] = {}
    for coded, true in zip(polluted.ravel(), healthy_mirror.ravel()):
        coded, true = int(coded), int(true)
        if coded in (bg, marker) or true == bg:
            continue
        if coded in pairs:
            assert pairs[coded] == true, "the color codebook must be consistent"
        else:
            pairs[coded] = true
    assert pairs, "the color codebook must have visible examples"
    assert len(set(pairs.values())) == len(pairs), "the codebook must be bijective"

    # Every non-marker coded color that ever needs inverse translation must be known.
    needed = set(map(int, np.unique(polluted))) - {bg, marker}
    assert needed == set(pairs), "every polluted foreground color must be decodable"
    return pairs


def _solve_parts(grid: np.ndarray):
    """Infer roles and return aligned polluted/healthy halves plus the codebook."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and g.shape[1] <= 30
    left, right, separator = _split_halves(g)
    bg = _background(g)
    assert bg != separator
    marker, side = _marker_and_side(g, left.shape[1], {bg, separator})

    if side == "left":
        polluted = left
        healthy_mirror = right[:, ::-1]
    else:
        polluted = right
        healthy_mirror = left[:, ::-1]

    decode = _decode_map(polluted, healthy_mirror, bg, marker)
    return polluted, healthy_mirror, bg, marker, side, decode


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair and crop the marker-selected mirror half by decoding its uniform color substitution.

    Input structure:
    - A monochrome central separator column divides the grid into equal-width mirror halves on a uniquely most-common background.
    - One half is true-colored and the other has every foreground color changed by one bijection; each half also has background-colored holes, with no paired holes.
    - Exactly two pixels of a unique marker color are aligned in one row or column and lie in the encoded half.

    Rule:
    - The marker pair selects the encoded half, which is returned at its original half-grid size and orientation.
    - Align the healthy half by reflecting it across the separator and infer encoded-color to true-color pairs wherever both aligned cells are non-background and non-marker.
    - Copy a non-hole healthy cell directly; where the healthy cell is a hole, inverse-translate the encoded cell through the inferred bijection. Replace marker pixels by the same rule.

    Key insight: The apparently coherent damaged half is a color codebook, and holes in the healthy mirror force use of its inverse rather than simple mirror copying.
    """
    polluted, healthy, bg, marker, _side, decode = _solve_parts(grid)
    out = np.full(polluted.shape, bg, dtype=int)

    for r in range(polluted.shape[0]):
        for c in range(polluted.shape[1]):
            true = int(healthy[r, c])
            coded = int(polluted[r, c])
            if true != bg:
                out[r, c] = true
            elif coded == bg:
                out[r, c] = bg
            else:
                assert coded != marker, "a marker over a healthy-side hole is not decodable"
                assert coded in decode
                out[r, c] = decode[coded]
    return out


def _grow_true_image(h: int, w: int, bg: int, colors: list[int]) -> np.ndarray:
    """Grow three balanced colored regions into a sparse, readable true image."""
    target = random.randint(max(18, int(0.38 * h * w)), int(0.52 * h * w))
    grid = np.full((h, w), bg, dtype=int)
    seeds = random.sample([(r, c) for r in range(h) for c in range(w)], len(colors))
    regions: list[set[tuple[int, int]]] = []
    for color, point in zip(colors, seeds):
        grid[point] = color
        regions.append({point})

    turns = 0
    while int(np.count_nonzero(grid != bg)) < target and turns < h * w * 20:
        turns += 1
        idx = turns % len(colors)
        frontier: set[tuple[int, int]] = set()
        for r, c in regions[idx]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                    frontier.add((nr, nc))
        if not frontier:
            continue
        point = random.choice(tuple(frontier))
        grid[point] = colors[idx]
        regions[idx].add(point)
    return grid


def _aligned_marker_cells(true: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Choose a horizontal or vertical aligned foreground pair with visible spacing."""
    h, w = true.shape
    orientation = random.choice(("row", "column"))
    choices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if orientation == "row":
        for r in range(h):
            cols = [c for c in range(w) if true[r, c] != bg]
            choices.extend(((r, a), (r, b)) for a in cols for b in cols if b >= a + 2)
    else:
        for c in range(w):
            rows = [r for r in range(h) if true[r, c] != bg]
            choices.extend(((a, c), (b, c)) for a in rows for b in rows if b >= a + 2)
    return random.choice(choices) if choices else None


def _derangement(colors: list[int]) -> dict[int, int]:
    """Return a random visible bijection in which every foreground color changes."""
    shifted = colors[1:] + colors[:1]
    if random.random() < 0.5:
        shifted = colors[-1:] + colors[:-1]
    return dict(zip(colors, shifted))


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    polluted_holes: set[tuple[int, int]],
    healthy_holes: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
    expected_side: str,
    foreground: set[int],
) -> bool:
    """Check the described semantics and reject shortcut-friendly samples."""
    try:
        polluted, healthy, bg, marker, side, decode = _solve_parts(grid)
        solved = main(grid)
    except (AssertionError, ValueError):
        return False
    if side != expected_side or not np.array_equal(solved, expected):
        return False
    if not polluted_holes or not healthy_holes or not marker_cells:
        return False
    if set(decode.values()) != foreground or len(decode) != len(foreground):
        return False

    # A one-way mirror copy fails at every healthy-side hole.
    mirror_only = healthy.copy()
    if np.array_equal(mirror_only, expected):
        return False
    if not all(healthy[p] == bg and expected[p] != bg for p in healthy_holes):
        return False

    # Inverse-translating only the damaged half fails at its holes and markers.
    damaged_only = np.full(expected.shape, bg, dtype=int)
    for p in np.ndindex(expected.shape):
        value = int(polluted[p])
        if value in decode:
            damaged_only[p] = decode[value]
    if np.array_equal(damaged_only, expected):
        return False
    if not all(polluted[p] == bg and expected[p] != bg for p in polluted_holes):
        return False
    if not all(polluted[p] == marker and healthy[p] == expected[p] != bg for p in marker_cells):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(7, 13)
    w = random.randint(5, 10)
    bg, separator, marker, *foreground = random.sample(range(10), 6)
    true = _grow_true_image(h, w, bg, foreground)
    counts = [int(np.count_nonzero(true == color)) for color in foreground]
    if min(counts) < 5 or len(set(map(int, np.unique(true)))) != 4:
        return None

    marker_pair = _aligned_marker_cells(true, bg)
    if marker_pair is None:
        return None
    marker_cells = set(marker_pair)

    available = [p for p in np.ndindex(true.shape) if true[p] != bg and p not in marker_cells]
    random.shuffle(available)
    n_polluted = random.randint(1, min(3, len(available) // 4))
    polluted_holes = set(available[:n_polluted])
    remaining = [p for p in available[n_polluted:] if p not in polluted_holes]
    n_healthy = random.randint(1, min(3, len(remaining) // 4))
    healthy_holes = set(remaining[:n_healthy])
    if polluted_holes & healthy_holes:
        return None

    encode = _derangement(foreground)
    polluted = np.full_like(true, bg)
    for p in np.ndindex(true.shape):
        value = int(true[p])
        polluted[p] = bg if value == bg else encode[value]
    for p in polluted_holes:
        polluted[p] = bg
    for p in marker_cells:
        polluted[p] = marker

    healthy_aligned = true.copy()
    for p in healthy_holes:
        healthy_aligned[p] = bg

    side = random.choice(("left", "right"))
    sep_col = np.full((h, 1), separator, dtype=int)
    if side == "left":
        grid = np.concatenate((polluted, sep_col, healthy_aligned[:, ::-1]), axis=1)
    else:
        grid = np.concatenate((healthy_aligned[:, ::-1], sep_col, polluted), axis=1)

    if _background(grid) != bg:
        return None
    if not _semantic_check(
        grid,
        true,
        polluted_holes,
        healthy_holes,
        marker_cells,
        side,
        set(foreground),
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance with both indispensable hole branches."""
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            assert max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid color-codebook repair puzzle")
