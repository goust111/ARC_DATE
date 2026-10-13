from __future__ import annotations

import numpy as np


BLACK = 0
FG_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _normalize_cells(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.asarray(cells, dtype=int)
    coords = coords.copy()
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[order]
    return tuple((int(r), int(c)) for r, c in coords)


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    base = np.asarray(cells, dtype=int)
    for flip in (False, True):
        coords = base.copy()
        if flip:
            coords[:, 1] *= -1
        for _ in range(4):
            key = _normalize_cells(coords)
            if key not in seen:
                seen.add(key)
                out.append(key)
            coords = np.stack([coords[:, 1], -coords[:, 0]], axis=1)
    return out


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _local_neighborhood(mask: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((3, 3), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[dr + 1, dc + 1] = bool(mask[rr, cc])
    return out


def _non_black_color(grid: np.ndarray) -> int | None:
    colors = sorted(set(map(int, np.unique(grid))) - {BLACK})
    if not colors:
        return None
    assert len(colors) == 1, "expected exactly one foreground color"
    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each foreground pixel into a 3x3 block stamped by the complement of its own local neighborhood.

    Input structure:
    - The input is a black background with pixels of exactly one foreground color.
    - Foreground pixels may be isolated or form small local clusters.
    - A pixel's relevant context is its centered 3x3 neighborhood, with out-of-bounds treated as black.

    Rule:
    - The output canvas is exactly 3 times the input height and width.
    - Every black input cell maps to an all-black 3x3 output block.
    - For every foreground input cell, inspect its 3x3 neighborhood:
      neighborhood positions that are black become foreground in the block, while
      neighborhood positions that are foreground, including the center itself, stay black.

    Key insight:
    - This is not one global stamp reused everywhere: each foreground cell prints a different
      3x3 complement mask derived from its own local surroundings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = map(int, g.shape)
    fg = _non_black_color(g)
    out = np.zeros((h * 3, w * 3), dtype=int)
    if fg is None:
        return out

    mask = g == fg
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    for r, c in np.argwhere(mask):
        block = np.where(padded[r : r + 3, c : c + 3], BLACK, fg)
        out[3 * r : 3 * r + 3, 3 * c : 3 * c + 3] = block
    return out


BASE_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "single": ((0, 0),),
    "domino": ((0, 0), (0, 1)),
    "diag_pair": ((0, 0), (1, 1)),
    "line3": ((0, 0), (0, 1), (0, 2)),
    "l3": ((0, 0), (1, 0), (1, 1)),
    "square2": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "t4": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "plus5": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "ring8": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
}
SHAPE_VARIANTS = {name: _shape_variants(cells) for name, cells in BASE_SHAPES.items()}
CLUSTER_NAMES = ("domino", "diag_pair", "line3", "l3", "square2", "t4", "plus5", "ring8")
DIAGONAL_CLUSTER_NAMES = ("diag_pair", "square2", "ring8")
ORTHOGONAL_CLUSTER_NAMES = ("domino", "line3", "l3", "t4", "plus5")
BORDER_SHAPE_NAMES = ("single", "domino", "diag_pair", "line3", "l3")


def _translated_cells(cells: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + r, c0 + c) for r, c in cells}


def _can_place(mask: np.ndarray, cells: tuple[tuple[int, int], ...], r0: int, c0: int) -> bool:
    h, w = mask.shape
    placed = _translated_cells(cells, r0, c0)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in placed):
        return False
    if any(mask[r, c] for r, c in placed):
        return False

    rows = [r for r, _ in placed]
    cols = [c for _, c in placed]
    for rr in range(max(0, min(rows) - 1), min(h - 1, max(rows) + 1) + 1):
        for cc in range(max(0, min(cols) - 1), min(w - 1, max(cols) + 1) + 1):
            if mask[rr, cc] and (rr, cc) not in placed:
                return False
    return True


def _place_shape(
    mask: np.ndarray,
    rng: np.random.Generator,
    shape_name: str,
    *,
    touch_border: bool,
) -> bool:
    variants = SHAPE_VARIANTS[shape_name][:]
    rng.shuffle(variants)
    h, w = mask.shape
    for cells in variants:
        sh, sw = _shape_dims(cells)
        if sh > h or sw > w:
            continue
        origins = [
            (r0, c0)
            for r0 in range(h - sh + 1)
            for c0 in range(w - sw + 1)
            if not touch_border or r0 == 0 or c0 == 0 or r0 + sh == h or c0 + sw == w
        ]
        if not origins:
            continue
        origins = list(origins)
        rng.shuffle(origins)
        for r0, c0 in origins:
            if _can_place(mask, cells, r0, c0):
                for r, c in cells:
                    mask[r0 + r, c0 + c] = True
                return True
    return False


def _has_border_touch(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _has_isolated(mask: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        neighborhood = _local_neighborhood(mask, int(r), int(c))
        if int(neighborhood.sum()) == 1:
            return True
    return False


def _has_nonisolated(mask: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        if int(_local_neighborhood(mask, int(r), int(c)).sum()) >= 2:
            return True
    return False


def _has_diagonal_contact(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if mask[r, c] and mask[r + 1, c + 1]:
                return True
            if mask[r + 1, c] and mask[r, c + 1]:
                return True
    return False


def _has_full_neighborhood(mask: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        if _local_neighborhood(mask, int(r), int(c)).all():
            return True
    return False


def _pattern_count(mask: np.ndarray) -> int:
    patterns = {
        tuple(_local_neighborhood(mask, int(r), int(c)).astype(int).ravel().tolist())
        for r, c in np.argwhere(mask)
    }
    return len(patterns)


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(4, 11))
    w = int(rng.integers(4, 11))
    fg = int(rng.choice(FG_COLORS))
    mask = np.zeros((h, w), dtype=bool)

    wants_diagonal = bool(rng.integers(0, 2))
    border_shape = str(rng.choice(BORDER_SHAPE_NAMES))
    primary_cluster_pool = DIAGONAL_CLUSTER_NAMES if wants_diagonal else ORTHOGONAL_CLUSTER_NAMES
    primary_cluster = str(rng.choice(primary_cluster_pool))
    secondary_cluster = str(rng.choice(CLUSTER_NAMES))

    plan: list[tuple[str, bool]] = [(border_shape, True)]
    if border_shape == "single":
        plan.append((primary_cluster, False))
    else:
        plan.append(("single", False))
        plan.append((primary_cluster, False))
    plan.append((secondary_cluster, False))

    extra_count = int(rng.integers(0, 4))
    for _ in range(extra_count):
        name = str(rng.choice(("single",) + CLUSTER_NAMES))
        touch_border = bool(rng.random() < 0.25)
        plan.append((name, touch_border))

    for shape_name, touch_border in plan:
        if not _place_shape(mask, rng, shape_name, touch_border=touch_border):
            return None

    fg_count = int(mask.sum())
    if fg_count < 4 or fg_count > max(12, (h * w) // 2):
        return None
    if not _has_border_touch(mask):
        return None
    if not _has_isolated(mask) or not _has_nonisolated(mask):
        return None
    if wants_diagonal and not _has_diagonal_contact(mask):
        return None
    if _has_full_neighborhood(mask):
        return None
    if _pattern_count(mask) < 3:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[mask] = fg
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid local-complement stamp input")
