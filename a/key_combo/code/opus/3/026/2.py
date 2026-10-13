from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MATCHED_COLOR = 1
UNMATCHED_COLOR = 2
SYMMETRIC_COLOR = 3
INPUT_FOREGROUND = 7
LATTICE_COLOR = 8

CELL_COUNT_OPTIONS = (3, 4)
CELL_SIZE_OPTIONS = (3, 5)
MAX_FOREGROUND_PIXELS = 6
MAX_ROLE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400

ROLE_EMPTY = "empty"
ROLE_SYMMETRIC = "symmetric"
ROLE_MIXED_ASYMMETRIC = "mixed_asymmetric"
ROLE_BBOX_TRICK = "bbox_trick"
ROLE_ASYMMETRIC = "asymmetric"


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bbox in arc_utils style: (r0, c0, r1, c1)."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _cell_ranges_from_walls(length: int, walls: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for left, right in zip(walls, walls[1:]):
        if right - left > 1:
            ranges.append((left + 1, right - 1))
    assert ranges, f"expected at least one cell band in length {length}"
    return ranges


def _parse_lattice(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    H, W = g.shape

    row_walls = [r for r in range(H) if np.all(g[r, :] == LATTICE_COLOR)]
    col_walls = [c for c in range(W) if np.all(g[:, c] == LATTICE_COLOR)]
    assert len(row_walls) >= 2 and len(col_walls) >= 2, "expected full teal lattice rows and columns"

    row_ranges = _cell_ranges_from_walls(H, row_walls)
    col_ranges = _cell_ranges_from_walls(W, col_walls)
    row_heights = {r1 - r0 + 1 for r0, r1 in row_ranges}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_ranges}
    assert len(row_heights) == 1 and len(col_widths) == 1, "expected equal-sized lattice cells"

    cell_h = row_heights.pop()
    cell_w = col_widths.pop()
    assert cell_w % 2 == 1, "cell width must be odd for a unique vertical symmetry axis"
    return row_ranges, col_ranges, int(cell_h), int(cell_w)


def _foreground_mask(cell: np.ndarray) -> np.ndarray:
    return (cell != BACKGROUND) & (cell != LATTICE_COLOR)


def _mirror_mask(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask)


def _is_cell_symmetric(mask: np.ndarray) -> bool:
    return bool(mask.any()) and np.array_equal(mask, _mirror_mask(mask))


def _is_bbox_symmetric(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    if box is None:
        return False
    cropped = _crop_bbox(mask, box)
    return np.array_equal(cropped, np.fliplr(cropped))


def _classify_mask(mask: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    mirrored = _mirror_mask(mask)
    matched = mask & mirrored
    unmatched = mask & ~mirrored
    if not mask.any():
        return ROLE_EMPTY, matched, unmatched
    if np.array_equal(mask, mirrored):
        return ROLE_SYMMETRIC, matched, unmatched
    return ROLE_ASYMMETRIC, matched, unmatched


def _transform_cell(cell: np.ndarray) -> np.ndarray:
    out = np.full(cell.shape, BACKGROUND, dtype=int)
    mask = _foreground_mask(cell)
    role, matched, unmatched = _classify_mask(mask)
    if role == ROLE_SYMMETRIC:
        out[mask] = SYMMETRIC_COLOR
    elif role == ROLE_ASYMMETRIC:
        out[matched] = MATCHED_COLOR
        out[unmatched] = UNMATCHED_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each teal lattice cell by testing its foreground against that cell's own vertical mirror axis.

    Input structure
    - Black background with one-pixel teal lattice lines, including the outer border, splitting the canvas into equal cells.
    - Each cell interior may contain orange foreground pixels or may be empty.
    - Every cell interior has odd width, so each cell has a single vertical mirror axis.

    Rule
    - Preserve the teal lattice and the black background.
    - For each cell, compare every foreground pixel with the pixel at its left-right reflected position inside that same cell.
    - If the full foreground set is mirror-symmetric, recolor all foreground pixels in that cell green.
    - Otherwise recolor foreground pixels whose mirrors also exist blue, and recolor unmatched foreground pixels red.

    Key insight
    The mirror test is local to each lattice cell rather than to the foreground bbox or the whole canvas, and that local symmetry classification selects between all-green recoloring and blue/red per-pixel recoloring.
    """
    g = np.asarray(grid, dtype=int)
    row_ranges, col_ranges, _cell_h, _cell_w = _parse_lattice(g)

    out = g.copy()
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_cell(g[r0 : r1 + 1, c0 : c1 + 1])
    return out


def _blank_lattice(n_cells: int, cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    size = n_cells * cell_size + (n_cells + 1)
    grid = np.full((size, size), BACKGROUND, dtype=int)
    walls = [k * (cell_size + 1) for k in range(n_cells + 1)]
    grid[walls, :] = LATTICE_COLOR
    grid[:, walls] = LATTICE_COLOR
    ranges = [(walls[i] + 1, walls[i + 1] - 1) for i in range(n_cells)]
    return grid, ranges, ranges


def _symmetric_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    axis = w // 2
    pair_positions = [(r, c) for r in range(h) for c in range(axis)]
    center_positions = [(r, axis) for r in range(h)]

    for _ in range(MAX_ROLE_ATTEMPTS):
        mask = np.zeros((h, w), dtype=bool)
        max_pairs = min(len(pair_positions), MAX_FOREGROUND_PIXELS // 2)
        n_pairs = rng.randrange(max_pairs + 1)
        remaining = MAX_FOREGROUND_PIXELS - 2 * n_pairs
        n_centers = rng.randrange(min(len(center_positions), remaining) + 1)
        total = 2 * n_pairs + n_centers
        if total == 0:
            continue

        for r, c in rng.sample(pair_positions, n_pairs):
            mask[r, c] = True
            mask[r, w - 1 - c] = True
        for r, c in rng.sample(center_positions, n_centers):
            mask[r, c] = True
        if _is_cell_symmetric(mask):
            return mask
    raise RuntimeError("failed to generate symmetric cell mask")


def _mixed_asymmetric_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    for _ in range(MAX_ROLE_ATTEMPTS):
        base = _symmetric_mask(h, w, rng)
        if not (1 <= int(base.sum()) <= MAX_FOREGROUND_PIXELS - 1):
            continue

        candidates = []
        for r in range(h):
            for c in range(w):
                mirror_c = w - 1 - c
                if c == mirror_c:
                    continue
                if base[r, c] or base[r, mirror_c]:
                    continue
                candidates.append((r, c))

        rng.shuffle(candidates)
        legal_extras: list[tuple[int, int]] = []
        seen_mirror_pairs: set[tuple[int, int]] = set()
        for r, c in candidates:
            mirror_coord = (r, w - 1 - c)
            if mirror_coord in seen_mirror_pairs:
                continue
            legal_extras.append((r, c))
            seen_mirror_pairs.add((r, c))

        max_extra = min(2, len(legal_extras), MAX_FOREGROUND_PIXELS - int(base.sum()))
        if max_extra < 1:
            continue

        mask = base.copy()
        for r, c in rng.sample(legal_extras, rng.randint(1, max_extra)):
            mask[r, c] = True

        role, matched, unmatched = _classify_mask(mask)
        if role == ROLE_ASYMMETRIC and matched.any() and unmatched.any():
            return mask
    raise RuntimeError("failed to generate mixed asymmetric cell mask")


def _bbox_trick_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    shapes = [
        (bh, bw)
        for bh in range(1, h + 1)
        for bw in range(1, w)
        if bh * bw <= MAX_FOREGROUND_PIXELS
    ]
    rng.shuffle(shapes)
    for _ in range(MAX_ROLE_ATTEMPTS):
        for bh, bw in shapes:
            col_starts = [c0 for c0 in range(w - bw + 1) if 2 * c0 + bw - 1 != w - 1]
            if not col_starts:
                continue
            r0 = rng.randrange(h - bh + 1)
            c0 = rng.choice(col_starts)
            mask = np.zeros((h, w), dtype=bool)
            mask[r0 : r0 + bh, c0 : c0 + bw] = True
            if _is_bbox_symmetric(mask) and not _is_cell_symmetric(mask):
                return mask
    raise RuntimeError("failed to generate bbox-trick cell mask")


def _random_asymmetric_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    coords = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(MAX_ROLE_ATTEMPTS):
        count = rng.randint(1, min(MAX_FOREGROUND_PIXELS, len(coords)))
        chosen = rng.sample(coords, count)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if not _is_cell_symmetric(mask):
            return mask
    raise RuntimeError("failed to generate asymmetric cell mask")


def _mask_for_role(role: str, h: int, w: int, rng: random.Random) -> np.ndarray:
    if role == ROLE_EMPTY:
        return np.zeros((h, w), dtype=bool)
    if role == ROLE_SYMMETRIC:
        return _symmetric_mask(h, w, rng)
    if role == ROLE_MIXED_ASYMMETRIC:
        return _mixed_asymmetric_mask(h, w, rng)
    if role == ROLE_BBOX_TRICK:
        return _bbox_trick_mask(h, w, rng)
    if role == ROLE_ASYMMETRIC:
        return _random_asymmetric_mask(h, w, rng)
    raise ValueError(f"unknown role: {role}")


def _sample_roles(total_cells: int, rng: random.Random) -> list[str]:
    roles = [
        ROLE_EMPTY,
        ROLE_SYMMETRIC,
        ROLE_SYMMETRIC,
        ROLE_MIXED_ASYMMETRIC,
        ROLE_BBOX_TRICK,
    ]
    extras = [
        ROLE_SYMMETRIC,
        ROLE_MIXED_ASYMMETRIC,
        ROLE_BBOX_TRICK,
        ROLE_ASYMMETRIC,
        ROLE_EMPTY,
    ]
    while len(roles) < total_cells:
        roles.append(rng.choice(extras))
    rng.shuffle(roles)
    return roles


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.argwhere(mask).T if mask.any() else (np.array([], dtype=int), np.array([], dtype=int))
    if rows.size:
        grid[top + rows, left + cols] = color


def _render_candidate(rng: random.Random) -> np.ndarray:
    n_cells = rng.choice(CELL_COUNT_OPTIONS)
    cell_size = rng.choice(CELL_SIZE_OPTIONS)
    grid, row_ranges, col_ranges = _blank_lattice(n_cells, cell_size)
    roles = _sample_roles(n_cells * n_cells, rng)

    used_nonempty_masks: set[bytes] = set()
    for index, role in enumerate(roles):
        r_band = index // n_cells
        c_band = index % n_cells
        r0, r1 = row_ranges[r_band]
        c0, c1 = col_ranges[c_band]

        mask = None
        for _ in range(MAX_ROLE_ATTEMPTS):
            candidate = _mask_for_role(role, r1 - r0 + 1, c1 - c0 + 1, rng)
            key = candidate.tobytes()
            if role == ROLE_EMPTY or key not in used_nonempty_masks or rng.random() < 0.25:
                mask = candidate
                break
        assert mask is not None

        if role != ROLE_EMPTY:
            used_nonempty_masks.add(mask.tobytes())
        _paint_mask(grid, r0, c0, mask, INPUT_FOREGROUND)
    return grid


def _semantic_expectation(grid: np.ndarray) -> np.ndarray:
    row_ranges, col_ranges, _cell_h, _cell_w = _parse_lattice(grid)
    out = np.full(grid.shape, BACKGROUND, dtype=int)
    out[np.asarray(grid) == LATTICE_COLOR] = LATTICE_COLOR
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = _foreground_mask(cell)
            role, matched, unmatched = _classify_mask(mask)
            if role == ROLE_SYMMETRIC:
                out[r0 : r1 + 1, c0 : c1 + 1][mask] = SYMMETRIC_COLOR
            elif role == ROLE_ASYMMETRIC:
                out[r0 : r1 + 1, c0 : c1 + 1][matched] = MATCHED_COLOR
                out[r0 : r1 + 1, c0 : c1 + 1][unmatched] = UNMATCHED_COLOR
    return out


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) - {BACKGROUND, INPUT_FOREGROUND, LATTICE_COLOR}:
        return False

    try:
        row_ranges, col_ranges, cell_h, cell_w = _parse_lattice(grid)
    except AssertionError:
        return False

    if len(row_ranges) not in CELL_COUNT_OPTIONS or len(col_ranges) not in CELL_COUNT_OPTIONS:
        return False
    if len(row_ranges) != len(col_ranges):
        return False
    if cell_h not in CELL_SIZE_OPTIONS or cell_w != cell_h:
        return False

    cell_masks: list[np.ndarray] = []
    has_empty = False
    has_symmetric = False
    has_mixed_asymmetric = False
    has_bbox_counterexample = False

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            mask = _foreground_mask(grid[r0 : r1 + 1, c0 : c1 + 1])
            cell_masks.append(mask)
            role, matched, unmatched = _classify_mask(mask)
            if role == ROLE_EMPTY:
                has_empty = True
            elif role == ROLE_SYMMETRIC:
                has_symmetric = True
            else:
                if matched.any() and unmatched.any():
                    has_mixed_asymmetric = True
                if _is_bbox_symmetric(mask):
                    has_bbox_counterexample = True

    unique_nonempty = {mask.tobytes() for mask in cell_masks if mask.any()}
    if len(unique_nonempty) < 3:
        return False
    if not (has_empty and has_symmetric and has_mixed_asymmetric and has_bbox_counterexample):
        return False

    expected = _semantic_expectation(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if INPUT_FOREGROUND in set(np.unique(actual)):
        return False
    if not {MATCHED_COLOR, UNMATCHED_COLOR, SYMMETRIC_COLOR}.issubset(set(np.unique(actual))):
        return False
    if not np.array_equal(actual == LATTICE_COLOR, grid == LATTICE_COLOR):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _render_candidate(rng)
        if _validate_generated_input(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid lattice symmetry example")
