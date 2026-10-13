import itertools
from collections import deque

import numpy as np


GRAY = 5
BLUE = 1
RED = 2
MAX_GENERATION_ATTEMPTS = 600


def _run_start_indices(matrix: np.ndarray) -> np.ndarray:
    starts = [0]
    for idx in range(1, matrix.shape[0]):
        if not np.array_equal(matrix[idx], matrix[idx - 1]):
            starts.append(idx)
    return np.array(starts, dtype=int)


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    rows = _run_start_indices(grid)
    cols = _run_start_indices(grid.T)
    return grid[np.ix_(rows, cols)].copy()


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_sprite(base: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    comps = _components4(base == BLUE)
    assert len(comps) == 1
    sprite_coords = comps[0]
    assert 3 <= len(sprite_coords) <= 6
    r0, c0, r1, c1 = _bbox(sprite_coords)
    mask = base[r0 : r1 + 1, c0 : c1 + 1] == BLUE
    return mask, sprite_coords


def _extract_blue_mask_any(grid: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == BLUE)]
    assert coords
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE


def _stamp(base: np.ndarray, sprite_mask: np.ndarray, controls: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    h, w = base.shape
    sh, sw = sprite_mask.shape
    for r, c in controls:
        top = r - sh // 2
        left = c - sw // 2
        for dr, dc in np.argwhere(sprite_mask):
            rr = top + int(dr)
            cc = left + int(dc)
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == GRAY:
                out[rr, cc] = BLUE
    return out


def _stamp_from_grid(base: np.ndarray, *, solid_bbox: bool = False) -> np.ndarray:
    sprite_mask, _ = _extract_sprite(base)
    if solid_bbox:
        sprite_mask = np.ones_like(sprite_mask, dtype=bool)
    controls = [tuple(map(int, coord)) for coord in np.argwhere(base == RED)]
    assert 2 <= len(controls) <= 4
    return _stamp(base, sprite_mask, controls)


def _raw_direct_stamp(grid: np.ndarray) -> np.ndarray:
    sprite_mask = _extract_blue_mask_any(grid)
    controls = [tuple(map(int, coord)) for coord in np.argwhere(grid == RED)]
    assert controls
    return _stamp(grid, sprite_mask, controls)


def main(grid: np.ndarray) -> np.ndarray:
    """Compress repeated rows and columns, then stamp the recovered blue sprite at each recovered red control.

    Input structure:
    - The grid has no black background; it is a smaller gray-dominant master grid stretched by repeating each row and each column 1-3 times in non-uniform adjacent runs.
    - In the recovered master, gray fills most cells, blue forms exactly one 4-connected sprite, and red forms 2-4 isolated control pixels.

    Rule:
    - Collapse every maximal run of identical adjacent rows and every maximal run of identical adjacent columns by keeping only the first row or column from each run.
    - In the compressed master, take the blue component's bounding box mask as the sprite stamp.
    - For each red control pixel in row-major order, align the center of that bounding box to the control pixel using floor half-sizes, and paint the sprite mask in blue.
    - The stamp is clipped to the grid bounds; only gray cells are recolored, while existing blue and red cells stay unchanged.

    Key insight:
    - The blue sprite's true shape and the red controls' true coordinates are both hidden by unequal row and column repetition, so the stamp can only be recovered after compression.
    """
    g = np.asarray(grid, dtype=int)
    base = _compress_grid(g)
    return _stamp_from_grid(base)


def _shape_is_connected(mask: np.ndarray) -> bool:
    return len(_components4(mask)) == 1


def _enumerate_sprite_masks() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[bytes] = set()
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3)):
        cells = [(r, c) for r in range(h) for c in range(w)]
        for size in range(3, min(6, h * w) + 1):
            for combo in itertools.combinations(cells, size):
                mask = np.zeros((h, w), dtype=bool)
                for r, c in combo:
                    mask[r, c] = True
                if not _shape_is_connected(mask):
                    continue
                if mask.all():
                    continue
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
                    continue
                key = mask.tobytes()
                if key not in seen:
                    seen.add(key)
                    shapes.append(mask)
    assert shapes
    return shapes


SPRITE_MASKS = _enumerate_sprite_masks()
SPRITE_MASKS_WITH_CENTER = [m for m in SPRITE_MASKS if m[m.shape[0] // 2, m.shape[1] // 2]]
SPRITE_MASKS_WITHOUT_CENTER = [m for m in SPRITE_MASKS if not m[m.shape[0] // 2, m.shape[1] // 2]]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _stamp_box(control: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = control
    sh, sw = shape
    top = r - sh // 2
    left = c - sw // 2
    return top, left, top + sh - 1, left + sw - 1


def _is_clipped(control: tuple[int, int], shape: tuple[int, int], grid_shape: tuple[int, int]) -> bool:
    top, left, bottom, right = _stamp_box(control, shape)
    h, w = grid_shape
    return top < 0 or left < 0 or bottom >= h or right >= w


def _expand_grid(base: np.ndarray, row_repeats: np.ndarray, col_repeats: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(base, row_repeats, axis=0), col_repeats, axis=1)


def _sample_repeats(
    rng: np.random.Generator,
    length: int,
    total_min: int,
    total_max: int,
) -> np.ndarray | None:
    for _ in range(80):
        reps = rng.integers(1, 4, size=length, dtype=int)
        if len(set(map(int, reps))) == 1:
            continue
        total = int(reps.sum())
        if total_min <= total <= total_max:
            return reps
    return None


def _valid_control_candidates(base: np.ndarray) -> list[tuple[int, int]]:
    sprite_mask, sprite_coords = _extract_sprite(base)
    h, w = base.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if base[r, c] != GRAY:
                continue
            if min(_manhattan((r, c), xy) for xy in sprite_coords) < 2:
                continue
            out.append((r, c))
    assert sprite_mask.size > 0
    return out


def _select_controls(
    rng: np.random.Generator,
    base: np.ndarray,
    sprite_mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    candidates = _valid_control_candidates(base)
    rng.shuffle(candidates)
    count = int(rng.integers(2, 5))
    want_clip = bool(rng.integers(0, 2))
    clipped = [p for p in candidates if _is_clipped(p, sprite_mask.shape, base.shape)]
    unclipped = [p for p in candidates if not _is_clipped(p, sprite_mask.shape, base.shape)]
    if want_clip and not clipped:
        return None
    selected: list[tuple[int, int]] = []
    pool = candidates[:]
    if want_clip:
        first = clipped[int(rng.integers(len(clipped)))]
        selected.append(first)
        pool = [p for p in pool if p != first]
    rng.shuffle(pool)
    for p in pool:
        if all(_manhattan(p, q) >= 2 for q in selected):
            selected.append(p)
            if len(selected) == count:
                break
    if len(selected) != count:
        return None
    if not want_clip and len(unclipped) < count:
        return None
    return sorted(selected)


def _paint_base(
    grid_shape: tuple[int, int],
    sprite_mask: np.ndarray,
    sprite_top_left: tuple[int, int],
    controls: list[tuple[int, int]],
) -> np.ndarray:
    base = np.full(grid_shape, GRAY, dtype=int)
    r0, c0 = sprite_top_left
    sh, sw = sprite_mask.shape
    base[r0 : r0 + sh, c0 : c0 + sw][sprite_mask] = BLUE
    for r, c in controls:
        base[r, c] = RED
    return base


def _relevant_repeats_ok(base: np.ndarray, row_repeats: np.ndarray, col_repeats: np.ndarray) -> bool:
    important = np.argwhere((base == BLUE) | (base == RED))
    row_hit = any(int(row_repeats[int(r)]) > 1 for r, _ in important)
    col_hit = any(int(col_repeats[int(c)]) > 1 for _, c in important)
    return row_hit and col_hit


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    base_h = int(rng.integers(7, 10))
    base_w = int(rng.integers(7, 10))

    center_branch = bool(rng.integers(0, 2))
    shape_bank = SPRITE_MASKS_WITH_CENTER if center_branch else SPRITE_MASKS_WITHOUT_CENTER
    sprite_mask = shape_bank[int(rng.integers(len(shape_bank)))]
    sh, sw = sprite_mask.shape

    sprite_r = int(rng.integers(0, base_h - sh + 1))
    sprite_c = int(rng.integers(0, base_w - sw + 1))

    temp = _paint_base((base_h, base_w), sprite_mask, (sprite_r, sprite_c), [])
    controls = _select_controls(rng, temp, sprite_mask)
    if controls is None:
        return None

    base = _paint_base((base_h, base_w), sprite_mask, (sprite_r, sprite_c), controls)
    compressed_out = _stamp_from_grid(base)
    if np.array_equal(base, compressed_out):
        return None
    if np.array_equal(_stamp_from_grid(base, solid_bbox=True), compressed_out):
        return None

    row_repeats = _sample_repeats(rng, base_h, 16, 25)
    col_repeats = _sample_repeats(rng, base_w, 16, 25)
    if row_repeats is None or col_repeats is None:
        return None
    if not _relevant_repeats_ok(base, row_repeats, col_repeats):
        return None

    expanded = _expand_grid(base, row_repeats, col_repeats)
    if max(expanded.shape) > 25:
        return None
    if not np.array_equal(_compress_grid(expanded), base):
        return None

    actual = main(expanded)
    if not np.array_equal(actual, compressed_out):
        return None

    direct_shortcut = _compress_grid(_raw_direct_stamp(expanded))
    if np.array_equal(direct_shortcut, compressed_out):
        return None

    if np.array_equal(_compress_grid(expanded), compressed_out):
        return None

    base_blue = _components4(base == BLUE)
    if len(base_blue) != 1:
        return None
    if any(_manhattan(a, b) < 2 for a, b in itertools.combinations(controls, 2)):
        return None
    for ctrl in controls:
        if min(_manhattan(ctrl, cell) for cell in base_blue[0]) < 2:
            return None

    return expanded


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
