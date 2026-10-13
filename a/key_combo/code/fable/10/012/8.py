from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PERIMETER_ORDER = (
    (0, 1),  # top
    (0, 2),  # top-right
    (1, 2),  # right
    (2, 2),  # bottom-right
    (2, 1),  # bottom
    (2, 0),  # bottom-left
    (1, 0),  # left
    (0, 0),  # top-left
)
CORNER_MATCH_INDICES = {1, 3, 5, 7}
EDGE_MATCH_INDICES = {0, 2, 4, 6}
ALL_COLORS = tuple(range(10))
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _non_bg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    mask = grid != int(bg)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _ring_match_index(patch: np.ndarray, bg: int) -> int:
    assert patch.shape == (3, 3), "ring patch must be 3x3"
    center = int(patch[1, 1])
    assert center != bg, "ring center must be a foreground key"

    perimeter = [int(patch[r, c]) for r, c in PERIMETER_ORDER]
    assert all(color != bg for color in perimeter), "ring perimeter must be foreground"
    matches = [idx for idx, color in enumerate(perimeter) if color == center]
    assert len(matches) == 1, "exactly one perimeter pixel must match the key color"
    return matches[0]


def _rotate_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    match_idx = _ring_match_index(patch, bg)
    perimeter = [int(patch[r, c]) for r, c in PERIMETER_ORDER]
    rotated = perimeter[match_idx:] + perimeter[:match_idx]

    out = patch.copy()
    for (r, c), color in zip(PERIMETER_ORDER, rotated):
        out[r, c] = int(color)
    out[1, 1] = int(bg)
    return out


def _parse_rings(grid: np.ndarray, bg: int) -> list[dict]:
    rings: list[dict] = []
    for comp in _non_bg_components(grid, bg):
        assert len(comp) == 9, "each object must be a 3x3 filled block"
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each object must have a 3x3 bbox"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "rings must be solid 3x3 foreground blocks"
        rings.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "patch": patch.copy(),
                "match_idx": _ring_match_index(patch, bg),
            }
        )
    assert rings, "expected at least one compass ring"
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each 3x3 compass ring along its perimeter until the key-matching pixel reaches the top, then extinguish the key.

    Input structure:
    - A dominant background contains one or more separated 3x3 foreground blocks.
    - Every such 3x3 block is a compass ring: all eight perimeter cells are foreground, and the center cell is a non-background key.
    - In each block, exactly one perimeter cell has the same color as the center key.

    Rule:
    - For each 3x3 block, read the eight perimeter cells in cyclic perimeter order.
    - Find the unique perimeter cell whose color matches the center key color.
    - Cyclically rotate the entire eight-cell perimeter sequence until that matching cell lands at the top-middle position.
    - Replace the center key with the background color.

    Key insight:
    - The center color does not stay in place; it specifies how far the perimeter cycle must move, and the motion is along the eight-step ring order rather than a 90-degree matrix rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = get_bg(g)
    out = g.copy()
    for ring in _parse_rings(g, bg):
        r0 = int(ring["r0"])
        c0 = int(ring["c0"])
        out[r0 : r0 + 3, c0 : c0 + 3] = _rotate_patch(ring["patch"], bg)
    return out


def _rectangles_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _sample_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(height - 2) for c in range(width - 2)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for r0, c0 in candidates:
        box = (r0, c0, r0 + 2, c0 + 2)
        if all(_rectangles_separated(box, old_box, padding=1) for old_box in boxes):
            chosen.append((r0, c0))
            boxes.append(box)
            if len(chosen) == count:
                return chosen
    return None


def _sample_grid_shape(count: int) -> tuple[int, int] | None:
    viable = []
    for height in range(8, 19):
        for width in range(8, 19):
            if height * width <= 18 * count:
                continue
            if _sample_positions(height, width, count) is not None:
                viable.append((height, width))
    return random.choice(viable) if viable else None


def _non_periodic(values: list[int]) -> bool:
    return all(values != values[k:] + values[:k] for k in range(1, len(values)))


def _ring_from_match_index(bg: int, match_idx: int) -> np.ndarray:
    palette = [color for color in ALL_COLORS if color != bg]
    perimeter = random.sample(palette, 8)
    key_color = perimeter[0]
    perimeter[0], perimeter[match_idx] = perimeter[match_idx], perimeter[0]
    assert len(set(perimeter)) == 8
    assert _non_periodic(perimeter)

    patch = np.full((3, 3), bg, dtype=int)
    for (r, c), color in zip(PERIMETER_ORDER, perimeter):
        patch[r, c] = int(color)
    patch[1, 1] = int(perimeter[match_idx])
    return patch


def _wrong_patch_variants(patch: np.ndarray, bg: int) -> list[np.ndarray]:
    wrong = []
    cleared = patch.copy()
    cleared[1, 1] = int(bg)
    wrong.append(cleared)
    for k in range(1, 4):
        rotated = np.rot90(patch, k=k).copy()
        rotated[1, 1] = int(bg)
        wrong.append(rotated)
    return wrong


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    for ring in _parse_rings(g, bg):
        r0 = int(ring["r0"])
        c0 = int(ring["c0"])
        out[r0 : r0 + 3, c0 : c0 + 3] = _rotate_patch(ring["patch"], bg)
    return out


def _sample_match_indices(count: int) -> list[int]:
    if count == 1:
        return [random.choice(tuple(CORNER_MATCH_INDICES | EDGE_MATCH_INDICES))]

    indices = [random.choice(tuple(CORNER_MATCH_INDICES)), random.choice(tuple(EDGE_MATCH_INDICES))]
    pool = list(range(8))
    while len(indices) < count:
        if len(indices) == 2 and count >= 3 and random.random() < 0.45:
            indices.append(0)
        else:
            indices.append(random.choice(pool))
    random.shuffle(indices)
    return indices


def _build_input() -> np.ndarray | None:
    bg = random.choice(ALL_COLORS)
    count = random.choice((1, 2, 2, 3, 3, 4))
    shape = _sample_grid_shape(count)
    if shape is None:
        return None
    height, width = shape
    positions = _sample_positions(height, width, count)
    if positions is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for (r0, c0), match_idx in zip(positions, _sample_match_indices(count)):
        grid[r0 : r0 + 3, c0 : c0 + 3] = _ring_from_match_index(bg, match_idx)
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    bg = get_bg(inp)
    rings = _parse_rings(inp, bg)
    if len(rings) < 1:
        return False

    match_indices = [int(ring["match_idx"]) for ring in rings]
    if len(rings) >= 2:
        if not any(idx in CORNER_MATCH_INDICES for idx in match_indices):
            return False
        if not any(idx in EDGE_MATCH_INDICES for idx in match_indices):
            return False

    if all(idx == 0 for idx in match_indices):
        return False

    disproves_shortcuts = False
    cleared_only = inp.copy()
    for ring in rings:
        r0 = int(ring["r0"])
        c0 = int(ring["c0"])
        cleared_only[r0 + 1, c0 + 1] = int(bg)

        true_patch = out[r0 : r0 + 3, c0 : c0 + 3]
        if all(not np.array_equal(true_patch, wrong) for wrong in _wrong_patch_variants(ring["patch"], bg)):
            disproves_shortcuts = True

        if int(out[r0 + 1, c0 + 1]) != int(bg):
            return False
        if int(true_patch[0, 1]) != int(ring["patch"][1, 1]):
            return False

    if np.array_equal(out, cleared_only):
        return False
    if not disproves_shortcuts:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _build_input()
        if inp is None:
            continue
        expected = _expected_output(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected) and _semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid compass-ring input")
