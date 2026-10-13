import random
from typing import Dict, List, Optional, Tuple

import numpy as np


BG = 0
SHELF_COLOR = 5
GEN_ATTEMPTS = 400
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

SPRITE_LIBRARY: Tuple[np.ndarray, ...] = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
)


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _rotation_variants(mask: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (rot.shape, rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot)
    return out


def _find_shelf(grid: np.ndarray) -> Tuple[int, int]:
    h, _w = grid.shape
    for r in range(h - 1, -1, -1):
        row = grid[r]
        color = int(row[0])
        if color == BG or not np.all(row == color):
            continue
        if r > 0 and np.any((grid[r - 1] != BG) & (grid[r - 1] != color)):
            return r, color
    raise AssertionError("could not find shelf row")


def _find_markers(grid: np.ndarray, shelf_row: int, shelf_color: int) -> Dict[int, int]:
    assert shelf_row > 0
    markers: Dict[int, int] = {}
    for c, value in enumerate(grid[shelf_row - 1]):
        color = int(value)
        if color in (BG, shelf_color):
            continue
        assert color not in markers, "marker colors must be unique"
        markers[color] = int(c)
    assert markers, "expected at least one color marker"
    return markers


def _match_key_to_notch(key_mask: np.ndarray, notch_mask: np.ndarray) -> bool:
    for rot in _rotation_variants(key_mask):
        if rot.shape == notch_mask.shape and np.array_equal(rot, notch_mask):
            return True
    return False


def _frame_info(grid: np.ndarray, color: int, coords: np.ndarray) -> Optional[dict]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    comp_mask = np.zeros((h, w), dtype=bool)
    comp_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(comp_mask & ~border):
        return None
    if not (comp_mask[0, 0] and comp_mask[0, -1] and comp_mask[-1, 0] and comp_mask[-1, -1]):
        return None

    missing = border & ~comp_mask
    if not np.any(missing):
        return None
    if missing[0, 0] or missing[0, -1] or missing[-1, 0] or missing[-1, -1]:
        return None

    sides = []
    if np.any(missing[0, :]):
        sides.append("top")
    if np.any(missing[-1, :]):
        sides.append("bottom")
    if np.any(missing[:, 0]):
        sides.append("left")
    if np.any(missing[:, -1]):
        sides.append("right")
    if len(sides) != 1:
        return None

    side = sides[0]
    if side in ("top", "bottom"):
        idx = np.where(missing[0 if side == "top" else -1, :])[0]
    else:
        idx = np.where(missing[:, 0 if side == "left" else -1])[0]
    if len(idx) == 0 or int(idx[-1] - idx[0] + 1) != int(len(idx)):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    sprite_mask = (interior != BG) & (interior != color)
    if not np.any(sprite_mask):
        return None
    sprite_coords = np.argwhere(sprite_mask)
    sr0, sc0 = sprite_coords.min(axis=0)
    sr1, sc1 = sprite_coords.max(axis=0)
    sprite_patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

    notch_coords = np.argwhere(missing)
    notch_mask = _tight_mask(notch_coords)
    return {
        "bbox": (r0, c0, r1, c1),
        "notch_mask": notch_mask,
        "sprite_patch": sprite_patch,
    }


def _parse_lock(grid: np.ndarray, color: int, cutoff_row: int) -> dict:
    color_mask = grid == color
    color_mask[cutoff_row:, :] = False
    comps = _components(color_mask)
    assert comps, "expected frame and key components for every marker color"

    frame = None
    key_masks: List[np.ndarray] = []
    for comp in comps:
        info = _frame_info(grid, color, comp)
        if info is not None:
            assert frame is None, "each color should have exactly one lock frame"
            frame = info
        else:
            key_masks.append(_tight_mask(comp))
    assert frame is not None, "missing lock frame"
    assert key_masks, "missing key component"
    assert any(_match_key_to_notch(km, frame["notch_mask"]) for km in key_masks), "key must match notch up to rotation"
    return frame


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= grid.shape[0] and left + w <= grid.shape[1]
    mask = patch != BG
    region = grid[top : top + h, left : left + w]
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    shelf_row, shelf_color = _find_shelf(g)
    markers = _find_markers(g, shelf_row, shelf_color)

    out = np.full_like(g, BG)
    out[shelf_row, :] = shelf_color

    cutoff_row = shelf_row - 1
    for color, target_col in markers.items():
        frame = _parse_lock(g, color, cutoff_row)
        sprite_patch = frame["sprite_patch"]
        h, w = sprite_patch.shape
        top = shelf_row - h
        left = target_col - w // 2
        assert top >= 0 and left >= 0 and left + w <= g.shape[1]
        _place_patch(out, sprite_patch, int(top), int(left))

    return out


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _make_lock_and_key(
    frame_color: int,
    sprite_color: int,
    sprite_mask: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sprite_patch = _colorize(sprite_mask, sprite_color)
    sh, sw = sprite_mask.shape

    inner_h = sh + rng.randint(2, 4)
    inner_w = sw + rng.randint(2, 4)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    side = rng.choice(("top", "bottom", "left", "right"))
    span = frame_w if side in ("top", "bottom") else frame_h
    notch_len = rng.randint(2, min(4, span - 2))

    lock = np.zeros((frame_h, frame_w), dtype=int)
    lock[0, :] = frame_color
    lock[-1, :] = frame_color
    lock[:, 0] = frame_color
    lock[:, -1] = frame_color

    if side in ("top", "bottom"):
        start = rng.randint(1, frame_w - notch_len - 1)
        row = 0 if side == "top" else frame_h - 1
        lock[row, start : start + notch_len] = BG
        notch_is_horizontal = True
    else:
        start = rng.randint(1, frame_h - notch_len - 1)
        col = 0 if side == "left" else frame_w - 1
        lock[start : start + notch_len, col] = BG
        notch_is_horizontal = False

    sprite_r = 1 + rng.randint(0, inner_h - sh - 2)
    sprite_c = 1 + rng.randint(0, inner_w - sw - 2)
    region = lock[sprite_r : sprite_r + sh, sprite_c : sprite_c + sw]
    region[sprite_mask] = sprite_color

    same_orientation = rng.random() < 0.5
    if notch_is_horizontal == same_orientation:
        key_mask = np.ones((1, notch_len), dtype=bool)
    else:
        key_mask = np.ones((notch_len, 1), dtype=bool)
    key_patch = _colorize(key_mask, frame_color)
    return lock, key_patch, sprite_patch


def _find_free_spot(
    occupied: np.ndarray,
    patch_shape: Tuple[int, int],
    rng: random.Random,
    padding: int = 1,
    tries: int = 200,
) -> Optional[Tuple[int, int]]:
    ph, pw = patch_shape
    h, w = occupied.shape
    if ph > h or pw > w:
        return None
    for _ in range(tries):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        r0 = max(0, top - padding)
        r1 = min(h, top + ph + padding)
        c0 = max(0, left - padding)
        c1 = min(w, left + pw + padding)
        if not occupied[r0:r1, c0:c1].any():
            return top, left
    return None


def _build_expected_output(
    shape: Tuple[int, int],
    shelf_row: int,
    marker_map: Dict[int, int],
    sprite_by_color: Dict[int, np.ndarray],
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    out[shelf_row, :] = SHELF_COLOR
    for color, target_col in marker_map.items():
        sprite_patch = sprite_by_color[color]
        sh, sw = sprite_patch.shape
        top = shelf_row - sh
        left = target_col - sw // 2
        assert top >= 0 and left >= 0 and left + sw <= w
        _place_patch(out, sprite_patch, top, left)
    return out


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    count = rng.randint(2, 4)
    colors = list(rng.sample(FRAME_COLORS, 2 * count))
    frame_colors = colors[:count]
    sprite_colors = colors[count:]

    sprite_masks = [rng.choice(SPRITE_LIBRARY) for _ in range(count)]
    sprite_patches = [_colorize(mask, color) for mask, color in zip(sprite_masks, sprite_colors)]
    max_sprite_w = max(int(patch.shape[1]) for patch in sprite_patches)

    step = max_sprite_w + rng.randint(2, 4)
    start = rng.randint(max_sprite_w // 2 + 1, max_sprite_w // 2 + 3)
    marker_order = list(range(count))
    rng.shuffle(marker_order)
    marker_cols = [start + step * i for i in range(count)]
    width_min = marker_cols[-1] + max_sprite_w // 2 + 2

    locks: List[np.ndarray] = []
    keys: List[np.ndarray] = []
    sprite_by_frame: Dict[int, np.ndarray] = {}
    for idx in range(count):
        lock, key, sprite_patch = _make_lock_and_key(frame_colors[idx], sprite_colors[idx], sprite_masks[idx], rng)
        locks.append(lock)
        keys.append(key)
        sprite_by_frame[frame_colors[idx]] = sprite_patch

    max_obj_w = max(max(int(lock.shape[1]) for lock in locks), max(int(key.shape[1]) for key in keys))
    width = max(width_min, max_obj_w + 4, rng.randint(width_min, width_min + 10))
    height = rng.randint(18, 24)
    shelf_row = height - 1
    top_area_h = shelf_row - 1
    if top_area_h < max(int(lock.shape[0]) for lock in locks) + 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[shelf_row, :] = SHELF_COLOR

    marker_map: Dict[int, int] = {}
    for order_idx, marker_col in zip(marker_order, marker_cols):
        if marker_col >= width:
            return None
        frame_color = frame_colors[order_idx]
        grid[shelf_row - 1, marker_col] = frame_color
        marker_map[frame_color] = marker_col

    occupied = np.zeros((top_area_h, width), dtype=bool)
    objects: List[Tuple[np.ndarray, Tuple[int, int]]] = []
    object_patches = locks + keys
    object_patches.sort(key=lambda patch: int(patch.shape[0] * patch.shape[1]), reverse=True)
    for patch in object_patches:
        pos = _find_free_spot(occupied, patch.shape, rng)
        if pos is None:
            return None
        top, left = pos
        occupied[top : top + patch.shape[0], left : left + patch.shape[1]] = True
        objects.append((patch, pos))

    for patch, (top, left) in objects:
        _place_patch(grid, patch, top, left)

    expected = _build_expected_output(grid.shape, shelf_row, marker_map, sprite_by_frame)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
