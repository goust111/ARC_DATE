from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
MARKER = 6
SQUARE_COLORS = (1, 2, 3, 4, 7, 8)
SQUARE_SIZES = (4, 5)
SCALES = (2, 3)
GEN_ATTEMPTS = 400

_CORNER_ROLES = {
    ((0, 0), (0, 1), (1, 0)): "TL",
    ((0, 0), (0, 1), (1, 1)): "TR",
    ((0, 0), (1, 0), (1, 1)): "BL",
    ((0, 1), (1, 0), (1, 1)): "BR",
}


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _corner_role(coords: np.ndarray) -> tuple[str, tuple[int, int]]:
    assert coords.shape[0] == 3, "each marker corner must contain three cells"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    assert (r1 - r0, c1 - c0) == (1, 1), "each marker corner must fit in a 2x2 box"
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    assert rel in _CORNER_ROLES, "marker component must be an L triomino"
    role = _CORNER_ROLES[rel]
    if role == "TL":
        key = (r0, c0)
    elif role == "TR":
        key = (r0, c1)
    elif role == "BL":
        key = (r1, c0)
    else:
        key = (r1, c1)
    return role, key


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _mask_components(grid == MARKER, connectivity=4)
    assert len(comps) == 4, "expected exactly four corner markers"

    corners: dict[str, tuple[int, int]] = {}
    for comp in comps:
        role, key = _corner_role(comp)
        assert role not in corners, "expected one corner of each role"
        corners[role] = key

    assert set(corners) == {"TL", "TR", "BL", "BR"}
    top, left = corners["TL"]
    top2, right = corners["TR"]
    bottom, left2 = corners["BL"]
    bottom2, right2 = corners["BR"]
    assert top == top2 and left == left2 and bottom == bottom2 and right == right2
    assert bottom > top + 1 and right > left + 1
    return top, left, bottom, right


def _extract_multicolor_square(grid: np.ndarray) -> np.ndarray:
    mask = ~np.isin(grid, (BG, GRAY, MARKER))
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one multicolor square"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h == w, "multicolor object must be square"
    assert np.all(~np.isin(patch, (BG, GRAY, MARKER))), "multicolor square must be dense and isolated"
    assert h in SQUARE_SIZES, "square size must match the puzzle specification"
    return patch


def _extract_pattern(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid == GRAY)
    assert coords.size > 0, "expected one gray scaled pattern"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(np.isin(patch, (BG, GRAY))), "gray pattern bbox may contain only gray or black"
    assert patch.shape[0] == patch.shape[1], "scaled pattern bbox must be square"
    return patch


def _downsample_mask(pattern: np.ndarray, target_size: int) -> np.ndarray:
    ph, pw = pattern.shape
    assert ph == pw and ph % target_size == 0 and pw % target_size == 0
    scale = ph // target_size
    assert scale == pw // target_size and scale >= 2

    mask = np.zeros((target_size, target_size), dtype=bool)
    for r in range(target_size):
        for c in range(target_size):
            block = pattern[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            if np.all(block == GRAY):
                mask[r, c] = True
            else:
                assert np.all(block == BG), "scaled pattern must be constant on each scale block"
    return mask


def _render_from_seed(shape: tuple[int, int], frame: tuple[int, int, int, int], seed: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    top, left, _bottom, _right = frame
    h, w = seed.shape
    ir0 = top + 1
    ic0 = left + 1
    out[ir0 : ir0 + h, ic0 : ic0 + w] = seed
    out[ir0 : ir0 + h, ic0 + w : ic0 + 2 * w] = np.fliplr(seed)
    out[ir0 + h : ir0 + 2 * h, ic0 : ic0 + w] = np.flipud(seed)
    out[ir0 + h : ir0 + 2 * h, ic0 + w : ic0 + 2 * w] = np.flipud(np.fliplr(seed))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the large gray stencil, mask the multicolor square, and expand the seed by fourfold symmetry inside the marked frame.

    Input structure:
    - The background is black.
    - One large gray object is a square integer scale-up of a smaller binary mask,
      with gray cells on black inside its own bbox.
    - One dense multicolor square sits elsewhere, and four magenta 2x2 L-corners
      mark the corners of a larger symmetry frame.

    Rule:
    - Crop the gray pattern, the multicolor square, and the marker-defined frame.
    - Downsample the gray pattern to the square size and use the result as a binary
      mask on the multicolor square, producing a top-left seed pattern.
    - Place that seed in the top-left quadrant of the frame interior, then fill the
      other three quadrants with its horizontal, vertical, and double reflections.
    - Keep the magenta corner markers fixed and clear every other input object.

    Key insight:
    - The gray object encodes which square cells survive, while the corner markers
      independently define the exact 2x-by-2x symmetry domain for expanding that seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    square = _extract_multicolor_square(g)
    n = int(square.shape[0])
    pattern = _extract_pattern(g)
    frame = _find_frame(g)
    top, left, bottom, right = frame
    assert bottom - top - 1 == 2 * n and right - left - 1 == 2 * n

    mask = _downsample_mask(pattern, n)
    seed = np.where(mask, square, BG)

    out = _render_from_seed(g.shape, frame, seed)
    out[g == MARKER] = MARKER
    return out


def _bool_component_count(mask: np.ndarray) -> int:
    return len(_mask_components(mask.astype(bool), connectivity=4))


def _all_rows_and_cols_mixed(mask: np.ndarray) -> bool:
    return all(0 < int(row.sum()) < row.size for row in mask) and all(
        0 < int(col.sum()) < col.size for col in mask.T
    )


def _make_mask(size: int, require_disconnected: bool) -> np.ndarray | None:
    min_on = (size * size * 2 + 4) // 5
    boundary_coords = (
        [(0, c) for c in range(size)]
        + [(size - 1, c) for c in range(size)]
        + [(r, 0) for r in range(1, size - 1)]
        + [(r, size - 1) for r in range(1, size - 1)]
    )

    for _ in range(300):
        count = random.randint(min_on, size * size - 1)
        mask = np.zeros((size, size), dtype=bool)
        for idx in random.sample(range(size * size), count):
            mask.flat[idx] = True
        if not _all_rows_and_cols_mixed(mask):
            continue
        boundary_vals = [bool(mask[r, c]) for r, c in boundary_coords]
        if not any(boundary_vals) or all(boundary_vals):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        comp_count = _bool_component_count(mask)
        if require_disconnected and comp_count <= 1:
            continue
        if not require_disconnected and random.random() < 0.5 and comp_count > 1:
            continue
        return mask
    return None


def _make_square(size: int) -> np.ndarray | None:
    for _ in range(300):
        square = np.array(
            [random.choice(SQUARE_COLORS) for _ in range(size * size)],
            dtype=int,
        ).reshape(size, size)
        if len(set(int(v) for v in square.ravel())) < 4:
            continue
        if any(np.array_equal(square, variant) for variant in (np.fliplr(square), np.flipud(square), np.rot90(square, 2))):
            continue
        if not all(len(set(int(v) for v in row)) >= 2 for row in square):
            continue
        if not all(len(set(int(v) for v in col)) >= 2 for col in square.T):
            continue
        return square
    return None


def _valid_seed(seed: np.ndarray) -> bool:
    kept = seed[seed != BG]
    if kept.size == 0 or kept.size == seed.size:
        return False
    if len(set(int(v) for v in kept)) < 3:
        return False
    if np.array_equal(seed, np.fliplr(seed)) or np.array_equal(seed, np.flipud(seed)):
        return False
    return not np.array_equal(seed, np.rot90(seed, 2))


def _scaled_pattern(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.where(np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)), GRAY, BG)


def _corner_patch(role: str) -> np.ndarray:
    patch = np.zeros((2, 2), dtype=int)
    if role == "TL":
        patch[0, 0] = patch[0, 1] = patch[1, 0] = MARKER
    elif role == "TR":
        patch[0, 0] = patch[0, 1] = patch[1, 1] = MARKER
    elif role == "BL":
        patch[0, 0] = patch[1, 0] = patch[1, 1] = MARKER
    else:
        patch[0, 1] = patch[1, 0] = patch[1, 1] = MARKER
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == patch.shape
    non_bg = patch != BG
    assert not np.any(target[non_bg] != BG)
    target[non_bg] = patch[non_bg]


def _place_corners(grid: np.ndarray, frame_top: int, frame_left: int, size: int) -> None:
    frame_bottom = frame_top + 2 * size + 1
    frame_right = frame_left + 2 * size + 1
    _paste(grid, _corner_patch("TL"), frame_top, frame_left)
    _paste(grid, _corner_patch("TR"), frame_top, frame_right - 1)
    _paste(grid, _corner_patch("BL"), frame_bottom - 1, frame_left)
    _paste(grid, _corner_patch("BR"), frame_bottom - 1, frame_right - 1)


def _can_place_box(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = used.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _mark_box(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _sample_layout(height: int, width: int, size: int, scale: int) -> dict[str, tuple[int, int]] | None:
    boxes = {
        "pattern": (size * scale, size * scale),
        "frame": (2 * size + 2, 2 * size + 2),
        "square": (size, size),
    }
    order = sorted(boxes, key=lambda name: boxes[name][0] * boxes[name][1], reverse=True)
    used = np.zeros((height, width), dtype=bool)
    placements: dict[str, tuple[int, int]] = {}

    for name in order:
        h, w = boxes[name]
        candidates = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            if _can_place_box(used, r0, c0, h, w, pad=1):
                placements[name] = (r0, c0)
                _mark_box(used, r0, c0, h, w, pad=1)
                break
        else:
            return None
    return placements


def _latent_expected(
    shape: tuple[int, int],
    frame_top: int,
    frame_left: int,
    seed: np.ndarray,
) -> np.ndarray:
    size = int(seed.shape[0])
    frame = (frame_top, frame_left, frame_top + 2 * size + 1, frame_left + 2 * size + 1)
    out = _render_from_seed(shape, frame, seed)
    _place_corners(out, frame_top, frame_left, size)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    size = random.choice(SQUARE_SIZES)
    scale = random.choice(SCALES)
    require_disconnected = random.random() < 0.5

    mask = _make_mask(size, require_disconnected=require_disconnected)
    if mask is None:
        return None
    square = _make_square(size)
    if square is None:
        return None

    seed = np.where(mask, square, BG)
    if not _valid_seed(seed):
        return None

    min_side = max(size * scale + 4, 2 * size + 7, 20)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    layout = _sample_layout(height, width, size, scale)
    if layout is None:
        return None

    grid = np.zeros((height, width), dtype=int)

    pattern = _scaled_pattern(mask, scale)
    pr, pc = layout["pattern"]
    _paste(grid, pattern, pr, pc)

    sr, sc = layout["square"]
    _paste(grid, square, sr, sc)

    fr, fc = layout["frame"]
    _place_corners(grid, fr, fc, size)

    expected = _latent_expected(grid.shape, fr, fc, seed)

    gray_coords = np.argwhere(grid == GRAY)
    color_coords = np.argwhere(~np.isin(grid, (BG, GRAY, MARKER)))
    assert gray_coords.size > 0 and color_coords.size > 0
    assert _find_frame(grid) == (fr, fc, fr + 2 * size + 1, fc + 2 * size + 1)
    assert _extract_multicolor_square(grid).shape == (size, size)
    assert _extract_pattern(grid).shape == (size * scale, size * scale)
    ir0 = fr + 1
    ic0 = fc + 1
    assert np.array_equal(expected[ir0 : ir0 + size, ic0 : ic0 + size], seed)
    assert np.array_equal(expected[ir0 : ir0 + size, ic0 + size : ic0 + 2 * size], np.fliplr(seed))
    assert np.array_equal(expected[ir0 + size : ir0 + 2 * size, ic0 : ic0 + size], np.flipud(seed))
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            result = _generate_case()
        except AssertionError:
            continue
        if result is None:
            continue
        grid, expected = result
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
