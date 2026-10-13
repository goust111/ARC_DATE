from __future__ import annotations

import random

import numpy as np


BG = 0
NON_BG_COLORS = list(range(1, 10))
SIZE_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _template_rotations(template: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(template, k).copy() for k in range(4)]


def _unique_matching_rotation(subgrid: np.ndarray, rotations: list[np.ndarray], bg: int) -> int | None:
    visible = subgrid != bg
    if not np.any(visible):
        return None
    matches = [k for k, rot in enumerate(rotations) if np.array_equal(subgrid[visible], rot[visible])]
    return matches[0] if len(matches) == 1 else None


def _extract_template(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    pts = np.argwhere(grid != bg)
    assert pts.size, "expected a non-empty template"
    top_left = min((int(r), int(c)) for r, c in pts)
    r0, c0 = top_left

    size_r = 0
    while r0 + size_r < grid.shape[0] and int(grid[r0 + size_r, c0]) != bg:
        size_r += 1
    size_c = 0
    while c0 + size_c < grid.shape[1] and int(grid[r0, c0 + size_c]) != bg:
        size_c += 1

    assert size_r == size_c and size_r >= 2, "template must be a filled square"
    size = size_r
    template = np.asarray(grid[r0 : r0 + size, c0 : c0 + size], dtype=int)
    assert template.shape == (size, size)
    assert np.all(template != bg), "template cannot contain background pixels"
    return template, (r0, c0)


def _has_bg_frame(grid: np.ndarray, r: int, c: int, size: int, bg: int) -> bool:
    h, w = grid.shape
    if r <= 0 or c <= 0 or r + size >= h or c + size >= w:
        return False
    patch = grid[r - 1 : r + size + 1, c - 1 : c + size + 1].copy()
    patch[1:-1, 1:-1] = bg
    return bool(np.all(patch == bg))


def _boundary_visible(subgrid: np.ndarray, bg: int) -> bool:
    return (
        bool(np.any(subgrid[0, :] != bg))
        and bool(np.any(subgrid[-1, :] != bg))
        and bool(np.any(subgrid[:, 0] != bg))
        and bool(np.any(subgrid[:, -1] != bg))
    )


def _find_pieces(
    grid: np.ndarray,
    template_pos: tuple[int, int],
    size: int,
    rotations: list[np.ndarray],
    bg: int,
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    found: list[tuple[int, int, int]] = []
    used = np.zeros((h, w), dtype=bool)

    for r in range(1, h - size):
        for c in range(1, w - size):
            if (r, c) == template_pos:
                continue
            if used[r : r + size, c : c + size].any():
                continue
            sub = grid[r : r + size, c : c + size]
            if not np.any(sub != bg):
                continue
            if not _has_bg_frame(grid, r, c, size, bg):
                continue
            if not _boundary_visible(sub, bg):
                continue
            rot_k = _unique_matching_rotation(sub, rotations, bg)
            if rot_k is None:
                continue
            found.append((r, c, rot_k))
            used[r : r + size, c : c + size] = True

    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each damaged rotated copy of the template by inferring its orientation and filling its gaps.

    Input structure:
    - The background is black.
    - Near the upper-left sits one complete n×n square template made of non-background colors only.
    - Elsewhere are several isolated n×n squares that are rotated copies of the template with some cells erased to background.

    Rule:
    - Read the complete upper-left template and enumerate its four 90-degree rotations.
    - For each other n×n square, find the unique rotation whose colors agree with every non-background cell that is still visible.
    - Replace the background gaps inside that square with the colors from the matched rotated template, while leaving all existing visible cells and the original template unchanged.

    Key insight:
    - The missing cells do not destroy the orientation cue: the surviving colored cells still identify exactly one rotation of the full template, and that orientation tells you how to restore the erased cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    template, template_pos = _extract_template(g, bg)
    size = int(template.shape[0])
    rotations = _template_rotations(template)

    out = g.copy()
    for r, c, rot_k in _find_pieces(g, template_pos, size, rotations, bg):
        sub = out[r : r + size, c : c + size]
        fill = sub == bg
        sub[fill] = rotations[rot_k][fill]
    return out


def _mark_forbidden(mask: np.ndarray, r: int, c: int, size: int) -> None:
    h, w = mask.shape
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(h, r + size + 1)
    c1 = min(w, c + size + 1)
    mask[r0:r1, c0:c1] = True


def _random_template(size: int) -> np.ndarray:
    colors = random.sample(NON_BG_COLORS, size * size)
    random.shuffle(colors)
    return np.array(colors, dtype=int).reshape(size, size)


def _random_visible_mask(size: int) -> np.ndarray | None:
    all_cells = [(r, c) for r in range(size) for c in range(size)]
    max_missing = size * size - 1
    for _ in range(MASK_ATTEMPTS):
        missing_count = random.randint(1, max_missing)
        missing = set(random.sample(all_cells, missing_count))
        visible = np.ones((size, size), dtype=bool)
        for r, c in missing:
            visible[r, c] = False

        if not _boundary_visible(visible.astype(int), 0):
            continue
        if size % 2 == 1 and visible.sum() == 1 and visible[size // 2, size // 2]:
            continue
        return visible
    return None


def _place_square(
    canvas: np.ndarray,
    forbidden: np.ndarray,
    square: np.ndarray,
    allow_edge: bool,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    size = square.shape[0]
    row_lo = 0 if allow_edge else 1
    col_lo = 0 if allow_edge else 1
    row_hi = h - size if allow_edge else h - size - 1
    col_hi = w - size if allow_edge else w - size - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if forbidden[r : r + size, c : c + size].any():
            continue
        canvas[r : r + size, c : c + size] = square
        _mark_forbidden(forbidden, r, c, size)
        return r, c
    return None


def _place_template(
    canvas: np.ndarray,
    expected: np.ndarray,
    forbidden: np.ndarray,
    template: np.ndarray,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    size = template.shape[0]
    row_cap = min(1, h - size)
    col_cap = min(1, w - size)
    candidates = [(r, c) for r in range(row_cap + 1) for c in range(col_cap + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if forbidden[r : r + size, c : c + size].any():
            continue
        canvas[r : r + size, c : c + size] = template
        expected[r : r + size, c : c + size] = template
        _mark_forbidden(forbidden, r, c, size)
        return r, c
    return None


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    size = random.choice(SIZE_CHOICES)
    piece_lo, piece_hi = (3, 7) if size == 2 else (2, 5)
    piece_count = random.randint(piece_lo, piece_hi)

    min_side = max(10, (size + 2) * 3)
    max_side = 30
    grid_h = random.randint(min_side, max_side)
    grid_w = random.randint(min_side, max_side)
    canvas = np.full((grid_h, grid_w), BG, dtype=int)
    expected = canvas.copy()
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)

    template = _random_template(size)
    template_pos = _place_template(canvas, expected, forbidden, template)
    assert template_pos is not None
    tr, tc = template_pos

    rotations = _template_rotations(template)
    placements: list[tuple[int, int, int]] = []
    for _ in range(piece_count):
        visible = _random_visible_mask(size)
        if visible is None:
            raise AssertionError("failed to build a visible mask")
        rot_k = random.randrange(4)
        full_piece = rotations[rot_k]
        damaged = np.where(visible, full_piece, BG).astype(int)
        pos = _place_square(canvas, forbidden, damaged, allow_edge=False)
        if pos is None:
            raise AssertionError("failed to place a damaged piece")
        r, c = pos
        expected[r : r + size, c : c + size] = full_piece
        placements.append((r, c, rot_k))

    assert np.array_equal(expected[tr : tr + size, tc : tc + size], template)

    found = _find_pieces(canvas, (tr, tc), size, rotations, BG)
    if len(found) != piece_count:
        raise AssertionError("piece detection count mismatch")
    found_map = {(r, c): k for r, c, k in found}
    for r, c, rot_k in placements:
        if found_map.get((r, c)) != rot_k:
            raise AssertionError("piece orientation mismatch")
        sub = canvas[r : r + size, c : c + size]
        if _unique_matching_rotation(sub, rotations, BG) != rot_k:
            raise AssertionError("piece must have a unique orientation")

    produced = main(canvas)
    if not np.array_equal(produced, expected):
        raise AssertionError("generator produced an unsolved example")
    return canvas, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, _ = _build_example()
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
