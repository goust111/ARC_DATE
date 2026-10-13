import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

MISSING_TO_COLOR = {
    (0, 0): GREEN,
    (0, 1): TEAL,
    (1, 0): YELLOW,
    (1, 1): PINK,
}

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS_4X4 = [(0, 0), (0, 3), (3, 0), (3, 3)]
PROTRUSION_CANDIDATES = [
    (-1, 1),
    (-1, 2),
    (4, 1),
    (4, 2),
    (1, -1),
    (2, -1),
    (1, 4),
    (2, 4),
]
MAX_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the border-touching red shape and fill the mirrored background with a color decoded from its embedded orange L marker.

    Input structure:
    - The grid has black background and one 4-connected foreground shape made of red cells.
    - Exactly three orange cells replace three corners of one internal 2x2 foreground block, forming an L-triomino marker.
    - The foreground touches exactly one vertical border, touches neither horizontal border, and does not touch the opposite vertical border.

    Rule:
    - Before reflecting, find the orange marker's 2x2 bounding box and read which corner is not orange.
    - Decode the missing corner as a fill color: top-left -> green, top-right -> teal, bottom-left -> yellow, bottom-right -> pink.
    - Horizontally reflect the whole input to make the mirrored half; in that half, black becomes the decoded color and every foreground cell becomes black.
    - Restore the original half by erasing orange marker cells back to red, then concatenate original and mirrored halves with the reflection seam on the border the foreground touches.

    Key insight:
    - The touched vertical border determines which side receives the mirrored negative half, while the embedded L marker must be decoded before reflection because mirroring flips its orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == ORANGE), "expected an orange marker"

    marker_color = _decode_marker_color(g)
    foreground = g != BLACK
    touches_left = bool(np.any(foreground[:, 0]))
    touches_right = bool(np.any(foreground[:, -1]))
    assert touches_left != touches_right, "foreground must touch exactly one vertical border"

    mirrored = np.fliplr(g)
    mirrored_half = np.full_like(mirrored, marker_color)
    mirrored_half[mirrored != BLACK] = BLACK

    original_half = g.copy()
    original_half[original_half == ORANGE] = RED

    if touches_left:
        return np.concatenate((mirrored_half, original_half), axis=1)
    return np.concatenate((original_half, mirrored_half), axis=1)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")


def _decode_marker_color(grid: np.ndarray) -> int:
    orange = np.argwhere(grid == ORANGE)
    assert orange.shape[0] == 3, "expected exactly three orange cells"

    r0, c0 = orange.min(axis=0)
    r1, c1 = orange.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "marker must fit in one 2x2 box"

    missing = None
    for dr in range(2):
        for dc in range(2):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            if grid[rr, cc] != ORANGE:
                assert grid[rr, cc] == RED, "missing corner must remain foreground red"
                missing = (dr, dc)
    assert missing in MISSING_TO_COLOR, "marker orientation must map to a color"
    return MISSING_TO_COLOR[missing]


def _gen_attempt() -> np.ndarray | None:
    H, W = _pick_dimensions()
    touched_left = bool(random.getrandbits(1))
    missing_corner = random.choice(list(MISSING_TO_COLOR))

    local_shape = _build_local_shape()
    if local_shape is None:
        return None

    shape_coords = np.argwhere(local_shape)
    r0, c0 = shape_coords.min(axis=0)
    r1, c1 = shape_coords.max(axis=0)
    patch = local_shape[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    ph, pw = patch.shape

    if pw >= W or ph >= H:
        return None
    if H - ph < 2:
        return None

    top = random.randint(1, H - ph - 1)
    left = 0 if touched_left else W - pw

    grid = np.zeros((H, W), dtype=int)
    for pr, pc in np.argwhere(patch):
        grid[top + int(pr), left + int(pc)] = RED

    marker_origin = _choose_marker_origin(patch)
    if marker_origin is None:
        return None
    mr, mc = marker_origin
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing_corner:
                grid[top + mr + dr, left + mc + dc] = ORANGE

    if not _valid_input(grid):
        return None
    return grid


def _pick_dimensions() -> tuple[int, int]:
    H = random.randint(6, 10)
    valid_widths = [w for w in range(max(8, H + 1), 15) if w % 2 == 0]
    W = random.choice(valid_widths)
    return H, W


def _build_local_shape() -> np.ndarray | None:
    shape = np.ones((4, 4), dtype=bool)

    removable = CORNERS_4X4[:]
    random.shuffle(removable)
    remove_count = random.randint(1, 4)
    for rr, cc in removable[:remove_count]:
        shape[rr, cc] = False

    candidates = PROTRUSION_CANDIDATES[:]
    random.shuffle(candidates)
    protrusion_count = random.randint(0, 2)
    cells = {(int(r), int(c)) for r, c in np.argwhere(shape)}
    for rr, cc in candidates[:protrusion_count]:
        cells.add((rr, cc))

    norm_cells = _normalize_cells(cells)
    if len(norm_cells) < 11 or len(norm_cells) > 18:
        return None

    patch = _cells_to_mask(norm_cells)
    if not _is_connected(patch):
        return None
    if _choose_marker_origin(patch) is None:
        return None
    if len(_marker_origins(patch)) != 1:
        return None

    return patch


def _choose_marker_origin(patch: np.ndarray) -> tuple[int, int] | None:
    origins = _marker_origins(patch)
    if len(origins) != 1:
        return None
    return origins[0]


def _marker_origins(patch: np.ndarray) -> list[tuple[int, int]]:
    H, W = patch.shape
    out = []
    for r in range(H - 1):
        for c in range(W - 1):
            block = patch[r : r + 2, c : c + 2]
            if not np.all(block):
                continue
            ok = True
            for br in range(2):
                for bc in range(2):
                    rr = r + br
                    cc = c + bc
                    for nr, nc in DIR4:
                        ar = rr + nr
                        ac = cc + nc
                        if not (0 <= ar < H and 0 <= ac < W and patch[ar, ac]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                out.append((r, c))
    return out


def _valid_input(grid: np.ndarray) -> bool:
    H, W = grid.shape
    if W % 2 != 0 or 2 * W > 30 or H > 30:
        return False

    present = set(map(int, np.unique(grid)))
    if not present <= {BLACK, RED, ORANGE}:
        return False
    if np.count_nonzero(grid == ORANGE) != 3:
        return False

    foreground = grid != BLACK
    if not np.any(foreground):
        return False
    if np.any(foreground[0]) or np.any(foreground[-1]):
        return False
    touches_left = bool(np.any(foreground[:, 0]))
    touches_right = bool(np.any(foreground[:, -1]))
    if touches_left == touches_right:
        return False
    if not _is_connected(foreground):
        return False

    orange = np.argwhere(grid == ORANGE)
    r0, c0 = orange.min(axis=0)
    r1, c1 = orange.max(axis=0)
    if int(r1 - r0) != 1 or int(c1 - c0) != 1:
        return False
    red_in_box = 0
    for dr in range(2):
        for dc in range(2):
            val = int(grid[int(r0 + dr), int(c0 + dc)])
            if val == RED:
                red_in_box += 1
            elif val != ORANGE:
                return False
    if red_in_box != 1:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == (H, 2 * W)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    patch = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    return patch


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    H, W = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            nxt = (rr, cc)
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == int(mask.sum())
