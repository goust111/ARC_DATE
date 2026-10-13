from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_RETRIES = 400
_GEN_CALL_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors8(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def _mask_connected_8(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c, mask.shape):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _dihedral_variants(grid: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(grid, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _is_asymmetric(grid: np.ndarray) -> bool:
    variants = _dihedral_variants(grid)
    return not any(np.array_equal(grid, variant) for variant in variants[1:])


def _scale_patch(grid: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def _extract_clean_patch(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    fg_mask = (g != BLACK) & (g != GREEN)
    box = _bbox(fg_mask)
    assert box is not None, "expected at least one non-black non-green foreground pixel"
    patch = _crop(g, box)
    factor = int(np.count_nonzero(patch == GREEN))
    assert 2 <= factor <= 4, "expected 2-4 green marker pixels inside the pattern bbox"
    clean = patch.copy()
    clean[clean == GREEN] = BLACK
    return clean, factor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale the local foreground pattern by a factor encoded by the number of green marker pixels in its bbox.

    Input structure:
    - The grid is mostly black background with a single small central pattern.
    - Non-black non-green pixels form a 3x3 or 4x4 red/blue foreground pattern whose bbox may also contain black holes.
    - Green marker pixels appear only on black cells inside that foreground bbox.

    Rule:
    - Find the minimal bounding box of all non-black non-green pixels.
    - Count the green marker pixels inside that bbox; this count N is the upscale factor.
    - Treat the green cells as black, then expand every cell of the bbox, including black background cells, into an N x N solid block.

    Key insight:
    - The marker positions do not matter; their count alone encodes the zoom factor that must be applied to the local pixel pattern.
    """
    clean, factor = _extract_clean_patch(grid)
    return _scale_patch(clean, factor)


def _pattern_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_mask(mask: np.ndarray, factor: int) -> bool:
    size = int(mask.shape[0])
    if mask.shape != (size, size):
        return False
    if mask.sum() < 4:
        return False
    if (size * size - int(mask.sum())) < factor:
        return False
    if not _pattern_has_full_bbox(mask):
        return False
    if not _mask_connected_8(mask):
        return False
    return True


def _random_colored_patch(size: int, factor: int) -> np.ndarray | None:
    max_fg = size * size - factor
    if max_fg < 4:
        return None
    for _ in range(300):
        fg_count = int(np.random.randint(4, max_fg + 1))
        flat = np.zeros(size * size, dtype=int)
        flat[np.random.choice(size * size, size=fg_count, replace=False)] = 1
        mask = flat.reshape(size, size).astype(bool)
        if not _valid_mask(mask, factor):
            continue

        patch = np.zeros((size, size), dtype=int)
        fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        color_bits = np.random.randint(0, 2, size=len(fg_coords))
        if color_bits.sum() == 0 or color_bits.sum() == len(fg_coords):
            continue
        for (r, c), bit in zip(fg_coords, color_bits):
            patch[r, c] = RED if int(bit) else BLUE
        if not _is_asymmetric(patch):
            continue
        return patch
    return None


def _marker_candidates(patch: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    zero_coords = [tuple(map(int, rc)) for rc in np.argwhere(patch == BLACK)]
    boundary = []
    interior = []
    last = patch.shape[0] - 1
    for r, c in zero_coords:
        if r in (0, last) or c in (0, last):
            boundary.append((r, c))
        else:
            interior.append((r, c))
    return boundary, interior


def _choose_markers(patch: np.ndarray, factor: int, mode: int) -> list[tuple[int, int]] | None:
    boundary, interior = _marker_candidates(patch)
    zero_coords = boundary + interior
    if len(zero_coords) < factor:
        return None

    if mode == 0 and boundary and interior:
        forced = [boundary[int(np.random.randint(len(boundary)))], interior[int(np.random.randint(len(interior)))]]
        pool = [cell for cell in zero_coords if cell not in forced]
        extra = list(np.random.choice(len(pool), size=factor - len(forced), replace=False)) if factor > 2 else []
        return forced + [pool[i] for i in extra]

    if mode == 1 and len(interior) >= factor:
        picks = np.random.choice(len(interior), size=factor, replace=False)
        return [interior[int(i)] for i in picks]

    picks = np.random.choice(len(zero_coords), size=factor, replace=False)
    return [zero_coords[int(i)] for i in picks]


def _render_input(patch: np.ndarray, markers: list[tuple[int, int]], top: int, left: int, shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch
    for r, c in markers:
        grid[top + r, left + c] = GREEN
    return grid


def _pick_canvas(size: int) -> tuple[int, int, int, int]:
    rows = int(np.random.randint(11, 16))
    cols = int(np.random.randint(11, 16))
    base_r = (rows - size) // 2
    base_c = (cols - size) // 2
    r_choices = [r for r in range(max(0, base_r - 1), min(rows - size, base_r + 1) + 1)]
    c_choices = [c for c in range(max(0, base_c - 1), min(cols - size, base_c + 1) + 1)]
    top = int(r_choices[int(np.random.randint(len(r_choices)))])
    left = int(c_choices[int(np.random.randint(len(c_choices)))])
    return rows, cols, top, left


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED, GREEN}):
        return False

    fg_mask = (grid != BLACK) & (grid != GREEN)
    box = _bbox(fg_mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    patch = _crop(grid, box)
    size = int(patch.shape[0])
    if patch.shape[0] != patch.shape[1] or size not in (3, 4):
        return False
    if not _pattern_has_full_bbox((patch != BLACK) & (patch != GREEN)):
        return False
    if np.count_nonzero(grid != BLACK) != np.count_nonzero(patch != BLACK):
        return False

    marker_count = int(np.count_nonzero(patch == GREEN))
    if not 2 <= marker_count <= 4:
        return False
    clean = patch.copy()
    clean[clean == GREEN] = BLACK
    if not _mask_connected_8(clean != BLACK):
        return False
    if not _is_asymmetric(clean):
        return False
    if len(np.unique(clean[clean != BLACK])) != 2:
        return False

    out = main(grid)
    if out.shape != (size * marker_count, size * marker_count):
        return False

    alt_counts = []
    for alt in (2, 3, 4):
        if alt == marker_count:
            continue
        alt_out = _scale_patch(clean, alt)
        alt_counts.append(not np.array_equal(out, alt_out))
    return all(alt_counts)


def _gen_attempt() -> np.ndarray | None:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    factor = [2, 3, 4][(_GEN_CALL_COUNT - 1) % 3]
    size = 3 if _GEN_CALL_COUNT % 2 else 4
    marker_mode = (_GEN_CALL_COUNT - 1) % 3

    patch = _random_colored_patch(size, factor)
    if patch is None:
        return None
    markers = _choose_markers(patch, factor, marker_mode)
    if markers is None:
        return None

    rows, cols, top, left = _pick_canvas(size)
    grid = _render_input(patch, markers, top, left, (rows, cols))
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
