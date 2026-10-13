import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GEN_RETRIES = 400
H_MIN, H_MAX = 8, 18
HALF_W_MIN, HALF_W_MAX = 7, 12


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(sorted(pts), dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sort_key(
    coords: np.ndarray,
    width: int,
    direction: str,
    *,
    tie_break: str = "top",
    reverse_distance: bool = False,
) -> tuple[int, int, int]:
    r0, c0, _r1, c1 = _bbox(coords)
    if direction == "right":
        distance = width - 1 - c1
    else:
        distance = c0
    primary = -distance if reverse_distance else distance
    if tie_break == "top":
        secondary = r0
    elif tie_break == "bottom":
        secondary = -r0
    else:
        raise ValueError(f"unknown tie_break: {tie_break}")
    return int(primary), int(secondary), int(c0)


def _slide_component(
    coords: np.ndarray, occupied: np.ndarray, width: int, direction: str
) -> tuple[np.ndarray, int]:
    step = 1 if direction == "right" else -1
    delta = 0
    while True:
        trial = coords.copy()
        trial[:, 1] += step * (delta + 1)
        if int(trial[:, 1].min()) < 0 or int(trial[:, 1].max()) >= width:
            break
        if np.any(occupied[trial[:, 0], trial[:, 1]]):
            break
        delta += 1
    moved = coords.copy()
    moved[:, 1] += step * delta
    occupied[moved[:, 0], moved[:, 1]] = True
    return moved, int(delta)


def _pack_side(
    mask: np.ndarray,
    direction: str,
    *,
    tie_break: str = "top",
    reverse_distance: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    h, w = mask.shape
    occupied = np.zeros((h, w), dtype=bool)
    meta: list[dict] = []
    comps = sorted(
        _components(mask),
        key=lambda pts: _sort_key(
            pts, w, direction, tie_break=tie_break, reverse_distance=reverse_distance
        ),
    )
    for coords in comps:
        moved, shift = _slide_component(coords, occupied, w, direction)
        key = _sort_key(coords, w, direction, tie_break="top", reverse_distance=False)
        meta.append(
            {
                "orig": coords,
                "moved": moved,
                "shift": shift,
                "distance": key[0],
                "top": int(coords[:, 0].min()),
            }
        )
    return occupied, meta


def _flush_independent(mask: np.ndarray, direction: str) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for coords in _components(mask):
        moved = coords.copy()
        if direction == "right":
            moved[:, 1] += w - 1 - int(coords[:, 1].max())
        else:
            moved[:, 1] -= int(coords[:, 1].min())
        out[moved[:, 0], moved[:, 1]] = True
    return out


def _aligned_xor(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    return np.fliplr(left_mask) ^ right_mask


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, f"expected one full grey separator column, got {cols}"
    sep = int(cols[0])
    assert sep * 2 + 1 == grid.shape[1], "separator must split the grid into equal halves"
    return sep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress blue and red fragments toward a central grey column, then output their mirror-aligned symmetric difference.

    Input structure:
    - A single full grey column sits exactly in the middle and splits the grid into equal left and right panes.
    - The left pane contains one or more blue 4-connected components on a black background.
    - The right pane contains one or more red 4-connected components on the same background.

    Rule:
    - Pack the left blue components rightward and the right red components leftward.
    - On each side, process components from nearest to the separator to farthest; if two components start at the same separator distance, slide the higher one first.
    - A component keeps moving until one more step would hit the separator or a component that has already stopped on that side.
    - After both panes are packed, compare mirrored separator-distance columns: the cell adjacent to the separator is output column 0, the next pair is output column 1, and so on.
    - Paint output cells green where exactly one packed side has foreground and black otherwise.

    Key insight:
    - The two colors are not comparable in their raw positions; only the inward queueing creates a common separator-centered coordinate frame in which the XOR has meaning.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _separator_col(g)

    left = g[:, :sep] == BLUE
    right = g[:, sep + 1 :] == RED

    packed_left, _ = _pack_side(left, "right")
    packed_right, _ = _pack_side(right, "left")
    xor_mask = _aligned_xor(packed_left, packed_right)
    return np.where(xor_mask, GREEN, BG).astype(int)


def _normalize_shape(shape: np.ndarray) -> np.ndarray:
    coords = np.argwhere(shape)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return shape[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_variants(shape: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for k in range(4):
        rotated = np.rot90(shape, k)
        for variant in (rotated, np.fliplr(rotated)):
            norm = _normalize_shape(variant.astype(bool))
            key = (tuple(map(int, norm.shape)), tuple(map(int, norm.ravel())))
            if key not in seen:
                seen.add(key)
                out.append(norm)
    return out


def _build_shape_bank() -> list[np.ndarray]:
    raws = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1], [1], [1]],
        [[1, 1], [1, 0]],
        [[1, 1], [0, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[0, 0, 1], [1, 1, 1]],
        [[1, 0], [1, 1], [1, 0]],
    ]
    bank: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for raw in raws:
        for variant in _shape_variants(np.array(raw, dtype=bool)):
            key = (tuple(map(int, variant.shape)), tuple(map(int, variant.ravel())))
            if key not in seen:
                seen.add(key)
                bank.append(variant)
    return bank


_SHAPE_BANK = _build_shape_bank()
_TIE_UPPER = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
_TIE_LOWER = np.array([[0, 1], [0, 1], [1, 1]], dtype=bool)
_TIE_BLOCKER = np.array([[1, 0], [1, 1]], dtype=bool)


def _offset_coords(shape: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(shape).astype(int)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _can_place(mask: np.ndarray, coords: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if mask[rr, cc]:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return False
    return True


def _stamp(mask: np.ndarray, coords: np.ndarray) -> None:
    mask[coords[:, 0], coords[:, 1]] = True


def _meaningful_tie(mask: np.ndarray, direction: str) -> bool:
    comps = _components(mask)
    if len(comps) < 3:
        return False
    w = mask.shape[1]
    distances = [_sort_key(comp, w, direction, tie_break="top")[0] for comp in comps]
    if len(set(distances)) == len(distances):
        return False
    top_version, _ = _pack_side(mask, direction, tie_break="top")
    bottom_version, _ = _pack_side(mask, direction, tie_break="bottom")
    return not np.array_equal(top_version, bottom_version)


def _unique_distance_top(mask: np.ndarray, direction: str) -> bool:
    w = mask.shape[1]
    keys = [(_sort_key(comp, w, direction)[0], int(comp[:, 0].min())) for comp in _components(mask)]
    return len(keys) == len(set(keys))


def _build_tie_side(h: int, half_w: int) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((h, half_w), dtype=bool)
        tie_right = random.randint(3, half_w - 3)
        blocker_right = random.randint(tie_right + 2, half_w - 1)
        upper_top = random.randint(0, h - 5)
        lower_top = upper_top + 2
        blocker_top = min(h - 2, upper_top + random.choice([0, 1, 2]))

        placements = [
            _offset_coords(_TIE_UPPER, upper_top, tie_right - 2),
            _offset_coords(_TIE_LOWER, lower_top, tie_right - 1),
            _offset_coords(_TIE_BLOCKER, blocker_top, blocker_right - 1),
        ]
        ok = True
        for coords in placements:
            if not _can_place(mask, coords):
                ok = False
                break
            _stamp(mask, coords)
        if not ok:
            continue
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if _meaningful_tie(mask, "right") and _unique_distance_top(mask, "right"):
            return mask
    return None


def _add_random_components(
    mask: np.ndarray, min_total: int, max_total: int, max_tries: int = 200
) -> np.ndarray | None:
    target = random.randint(min_total, max_total)
    out = mask.copy()
    tries = 0
    while len(_components(out)) < target and tries < max_tries:
        tries += 1
        shape = random.choice(_SHAPE_BANK)
        sh, sw = map(int, shape.shape)
        if sh > out.shape[0] or sw > out.shape[1]:
            continue
        candidates = [
            (r, c)
            for r in range(out.shape[0] - sh + 1)
            for c in range(out.shape[1] - sw + 1)
        ]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            coords = _offset_coords(shape, r0, c0)
            if _can_place(out, coords):
                _stamp(out, coords)
                break
    if len(_components(out)) < min_total:
        return None
    return out


def _make_grid(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    h, half_w = left_mask.shape
    grid = np.full((h, 2 * half_w + 1), BG, dtype=int)
    grid[:, half_w] = GREY
    grid[:, :half_w][left_mask] = BLUE
    grid[:, half_w + 1 :][right_mask] = RED
    return grid


def _informative_instance(grid: np.ndarray, tied_side: str) -> bool:
    sep = _separator_col(grid)
    left = grid[:, :sep] == BLUE
    right = grid[:, sep + 1 :] == RED
    if len(_components(left)) < 2 or len(_components(right)) < 2:
        return False
    if not _unique_distance_top(left, "right") or not _unique_distance_top(right, "left"):
        return False

    packed_left, left_meta = _pack_side(left, "right")
    packed_right, right_meta = _pack_side(right, "left")
    standard = _aligned_xor(packed_left, packed_right)
    if sum(item["shift"] for item in left_meta) == 0 or sum(item["shift"] for item in right_meta) == 0:
        return False

    green = int(standard.sum())
    if green < 4 or green > standard.size - 4:
        return False

    raw = _aligned_xor(left, right)
    if np.array_equal(standard, raw):
        return False

    naive = _aligned_xor(_flush_independent(left, "right"), _flush_independent(right, "left"))
    if np.array_equal(standard, naive):
        return False

    far = _aligned_xor(
        _pack_side(left, "right", reverse_distance=True)[0],
        _pack_side(right, "left", reverse_distance=True)[0],
    )
    if np.array_equal(standard, far):
        return False

    if tied_side == "left":
        if not _meaningful_tie(left, "right"):
            return False
        tie_out = _aligned_xor(_pack_side(left, "right", tie_break="bottom")[0], packed_right)
        if np.array_equal(standard, tie_out):
            return False
    else:
        if not _meaningful_tie(right, "left"):
            return False
        tie_out = _aligned_xor(packed_left, _pack_side(right, "left", tie_break="bottom")[0])
        if np.array_equal(standard, tie_out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(H_MIN, H_MAX)
    half_w = random.randint(HALF_W_MIN, HALF_W_MAX)

    tie_side = random.choice(["left", "right"])
    tie_local = _build_tie_side(h, half_w)
    if tie_local is None:
        return None

    if tie_side == "left":
        left_seed = tie_local
        right_seed = np.zeros((h, half_w), dtype=bool)
    else:
        left_seed = np.zeros((h, half_w), dtype=bool)
        right_seed = np.fliplr(tie_local)

    left = _add_random_components(left_seed, 3 if tie_side == "left" else 2, 5 if tie_side == "left" else 4)
    right = _add_random_components(right_seed, 3 if tie_side == "right" else 2, 5 if tie_side == "right" else 4)
    if left is None or right is None:
        return None

    grid = _make_grid(left, right)
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != (h, half_w):
        return None
    if not _informative_instance(grid, tie_side):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid inward-pack mirror-xor input")
