from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
MASK_COLOR = 3
TEMPLATE_COLOR = 5
PATTERN_SIZE = 7
MASK_SIZE = 3
CELL_SIZE = 3
CELL_GAP = 1
LATTICE_SIZE = 3
CELL_STRIDE = CELL_SIZE + CELL_GAP
SAMPLE_IDXS = (0, 2, 4)
PATTERN_PALETTE = np.array([1, 2, 4, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 500

# All masks are 4-connected, use exactly five pixels, and occupy a 3x3 bbox.
TEMPLATE_MASKS = (
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    m = n - 1
    coords = {
        (r, c),
        (c, m - r),
        (m - r, m - c),
        (m - c, r),
        (r, m - c),
        (m - r, c),
        (c, r),
        (m - c, m - r),
    }
    return tuple(sorted(coords))


def _all_d4_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = _d4_orbit(r, c, n)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


PATTERN_ORBITS = _all_d4_orbits(PATTERN_SIZE)


def _recover_pattern(pattern: np.ndarray) -> np.ndarray:
    restored = pattern.copy()
    n = pattern.shape[0]
    for r in range(n):
        for c in range(n):
            if int(pattern[r, c]) != MASK_COLOR:
                continue
            visible_colors = {
                int(pattern[rr, cc])
                for rr, cc in _d4_orbit(r, c, n)
                if int(pattern[rr, cc]) != MASK_COLOR
            }
            assert len(visible_colors) == 1
            restored[r, c] = visible_colors.pop()
    return restored


def _find_pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid != BACKGROUND
    candidates = []
    for coords in _components_from_mask(mask):
        values = {int(grid[r, c]) for r, c in coords}
        if MASK_COLOR not in values:
            continue
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 == PATTERN_SIZE and c1 - c0 + 1 == PATTERN_SIZE:
            candidates.append(bbox)
    assert len(candidates) == 1
    return candidates[0]


def _find_template_slots(grid: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    components = _components_from_mask(grid == TEMPLATE_COLOR)
    assert len(components) == LATTICE_SIZE * LATTICE_SIZE

    items: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in components:
        bbox = _bbox_from_coords(coords)
        items.append((bbox, coords))

    row_positions = sorted({bbox[0] for bbox, _coords in items})
    col_positions = sorted({bbox[1] for bbox, _coords in items})
    assert len(row_positions) == LATTICE_SIZE
    assert len(col_positions) == LATTICE_SIZE
    row_to_i = {row: i for i, row in enumerate(row_positions)}
    col_to_j = {col: j for j, col in enumerate(col_positions)}

    slots: dict[tuple[int, int], np.ndarray] = {}
    for bbox, coords in items:
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 == CELL_SIZE
        assert c1 - c0 + 1 == CELL_SIZE
        key = (row_to_i[r0], col_to_j[c0])
        slots[key] = coords

    assert len(slots) == LATTICE_SIZE * LATTICE_SIZE
    return slots


def _sample_control_matrix(pattern: np.ndarray) -> np.ndarray:
    return pattern[np.ix_(SAMPLE_IDXS, SAMPLE_IDXS)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the masked symmetric lookup pattern and use its sampled colors to repaint the template lattice.

    Input structure:
    - The background is black, with a left-side 3x3 lattice of identical gray template objects in 3x3 boxes separated by one-cell gaps.
    - Each template object is a small 4-connected shape with black holes inside its 3x3 bounding box.
    - Elsewhere there is one 7x7 multicolor square pattern that is symmetric under all rotations and reflections of the square, except for one solid 3x3 green mask.

    Rule:
    - Restore every green cell in the 7x7 pattern by finding a non-masked cell in the same D4 symmetry orbit; all visible orbit mates share one color.
    - Read a 3x3 control matrix from the restored pattern at rows and columns 0, 2, and 4 relative to the pattern's top-left corner.
    - Recolor each gray template object at lattice position (i, j) with the control color at (i, j), changing only the gray foreground pixels and leaving black holes unchanged.
    - Replace the green mask with the recovered colors and leave all other pixels unchanged.

    Key insight:
    - The symmetric square is not decorative; after repairing its masked cells, it becomes a positional color legend that drives the recoloring of the left template lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()

    pr0, pc0, pr1, pc1 = _find_pattern_bbox(g)
    pattern = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    restored = _recover_pattern(pattern)
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = restored

    control = _sample_control_matrix(restored)
    slots = _find_template_slots(g)
    for (i, j), coords in slots.items():
        out[coords[:, 0], coords[:, 1]] = int(control[i, j])

    return out


def _stamp_lattice(
    grid: np.ndarray,
    top: int,
    left: int,
    template_mask: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    for i in range(LATTICE_SIZE):
        for j in range(LATTICE_SIZE):
            color = TEMPLATE_COLOR if colors is None else int(colors[i, j])
            r0 = top + i * CELL_STRIDE
            c0 = left + j * CELL_STRIDE
            block = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            block[template_mask] = color


def _make_full_pattern() -> np.ndarray:
    palette_size = int(np.random.randint(3, 6))
    palette = np.random.choice(PATTERN_PALETTE, size=palette_size, replace=False)
    pattern = np.zeros((PATTERN_SIZE, PATTERN_SIZE), dtype=int)
    for orbit in PATTERN_ORBITS:
        color = int(np.random.choice(palette))
        for r, c in orbit:
            pattern[r, c] = color
    return pattern


def _valid_mask_position(pattern: np.ndarray, top: int, left: int) -> bool:
    sampled_colors = []
    for r in SAMPLE_IDXS:
        for c in SAMPLE_IDXS:
            if top <= r < top + MASK_SIZE and left <= c < left + MASK_SIZE:
                sampled_colors.append(int(pattern[r, c]))
    if len(sampled_colors) < 2 or len(set(sampled_colors)) < 2:
        return False

    for r in range(top, top + MASK_SIZE):
        for c in range(left, left + MASK_SIZE):
            visible = [
                (rr, cc)
                for rr, cc in _d4_orbit(r, c, PATTERN_SIZE)
                if not (top <= rr < top + MASK_SIZE and left <= cc < left + MASK_SIZE)
            ]
            if not visible:
                return False
            if len({int(pattern[rr, cc]) for rr, cc in visible}) != 1:
                return False

    return True


def _choose_mask_position(pattern: np.ndarray) -> tuple[int, int] | None:
    candidates = [
        (r0, c0)
        for r0 in range(PATTERN_SIZE - MASK_SIZE + 1)
        for c0 in range(PATTERN_SIZE - MASK_SIZE + 1)
        if _valid_mask_position(pattern, r0, c0)
    ]
    if not candidates:
        return None
    idx = int(np.random.randint(len(candidates)))
    return candidates[idx]


def _layout_dimensions(
    left_top: tuple[int, int],
    right_top: tuple[int, int],
    gap_right: int,
    gap_bottom: int,
) -> tuple[int, int]:
    left_r, left_c = left_top
    right_r, right_c = right_top
    height = max(left_r + CELL_SIZE + (LATTICE_SIZE - 1) * CELL_STRIDE, right_r + PATTERN_SIZE) + gap_bottom
    width = right_c + PATTERN_SIZE + gap_right
    return height, width


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    template_mask = TEMPLATE_MASKS[int(np.random.randint(len(TEMPLATE_MASKS)))]
    full_pattern = _make_full_pattern()
    control = _sample_control_matrix(full_pattern)
    if len(set(map(int, control.flatten()))) < 3:
        return None

    mask_pos = _choose_mask_position(full_pattern)
    if mask_pos is None:
        return None
    mask_r, mask_c = mask_pos

    left_top = (int(np.random.randint(1, 4)), int(np.random.randint(1, 4)))
    gap = int(np.random.randint(2, 6))
    right_top = (
        int(np.random.randint(1, 6)),
        left_top[1] + CELL_SIZE + (LATTICE_SIZE - 1) * CELL_STRIDE + gap,
    )
    bottom_margin = int(np.random.randint(1, 4))
    right_margin = int(np.random.randint(1, 4))
    height, width = _layout_dimensions(left_top, right_top, right_margin, bottom_margin)
    if height > 30 or width > 30:
        return None

    output_grid = np.zeros((height, width), dtype=int)
    _stamp_lattice(output_grid, left_top[0], left_top[1], template_mask, colors=control)
    rr, rc = right_top
    output_grid[rr : rr + PATTERN_SIZE, rc : rc + PATTERN_SIZE] = full_pattern

    input_grid = output_grid.copy()
    _stamp_lattice(input_grid, left_top[0], left_top[1], template_mask, colors=None)
    input_grid[rr + mask_r : rr + mask_r + MASK_SIZE, rc + mask_c : rc + mask_c + MASK_SIZE] = MASK_COLOR

    if not _validate_instance(input_grid, output_grid, full_pattern, mask_pos):
        return None

    return input_grid, output_grid


def _validate_instance(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    full_pattern: np.ndarray,
    mask_pos: tuple[int, int],
) -> bool:
    if input_grid.ndim != 2 or output_grid.ndim != 2:
        return False
    if max(input_grid.shape) > 30 or max(output_grid.shape) > 30:
        return False
    if np.array_equal(input_grid, output_grid):
        return False

    restored = main(input_grid)
    if not np.array_equal(restored, output_grid):
        return False

    if len(_components_from_mask(input_grid == TEMPLATE_COLOR)) != 9:
        return False

    mr, mc = mask_pos
    sampled_masked = sum(
        mr <= r < mr + MASK_SIZE and mc <= c < mc + MASK_SIZE
        for r in SAMPLE_IDXS
        for c in SAMPLE_IDXS
    )
    if sampled_masked < 2:
        return False

    control = _sample_control_matrix(full_pattern)
    if len(set(map(int, control.flatten()))) < 3:
        return False

    for r in range(mr, mr + MASK_SIZE):
        for c in range(mc, mc + MASK_SIZE):
            visible = [
                (rr, cc)
                for rr, cc in _d4_orbit(r, c, PATTERN_SIZE)
                if not (mr <= rr < mr + MASK_SIZE and mc <= cc < mc + MASK_SIZE)
            ]
            if not visible:
                return False
            if len({int(full_pattern[rr, cc]) for rr, cc in visible}) != 1:
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
