from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_SIDES = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120
GRID_MIN = 12
GRID_MAX = 30
PAIR_COUNT_OPTIONS = (1, 2, 3, 4)
FRAME_H_RANGE = (5, 9)
FRAME_W_RANGE = (5, 9)
GAP_RANGE = (1, 3)
COLOR_POOL = tuple(range(1, 10))


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    return np.array_equal(_coords_to_mask(coords), _frame_mask((h, w)))


def _find_pair_parts(g: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    comps = _components(g == int(color))
    assert len(comps) == 2, "each color must form exactly one frame and one key"
    frame_comps = [comp for comp in comps if _is_rect_frame(comp)]
    assert len(frame_comps) == 1, "each color must have exactly one rectangular frame"
    frame = frame_comps[0]
    key = comps[0] if frame is not comps[0] else comps[1]
    return frame, key


def _side_from_positions(frame: np.ndarray, key: np.ndarray) -> str:
    r0, c0, r1, c1 = _bbox(frame)
    frame_center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0], dtype=float)
    key_center = key.mean(axis=0)
    dr, dc = key_center - frame_center
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _aligned_origin(
    frame_bbox: tuple[int, int, int, int], key_shape: tuple[int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    fh = r1 - r0 + 1
    fw = c1 - c0 + 1
    kh, kw = key_shape
    if side == "up":
        return r0 - kh + 1, c0 + (fw - kw) // 2
    if side == "down":
        return r1, c0 + (fw - kw) // 2
    if side == "left":
        return r0 + (fh - kh) // 2, c0 - kw + 1
    if side == "right":
        return r0 + (fh - kh) // 2, c1
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut same-colored door frames using nearby key shapes, then remove the keys.

    Input structure:
    - The background is black.
    - Each foreground color appears in exactly two 4-connected objects: a 1-pixel-thick rectangular frame with a black interior, and a nearby solid key shape.
    - Different colors encode different frame/key pairs, and the key is separated from its frame.

    Rule:
    - Pair each key with the frame of the same color.
    - Determine whether the key lies above, below, left, or right of the frame from the key centroid's main direction relative to the frame center.
    - Align the cropped key shape to the corresponding frame side with edge contact and centered placement along that side, then erase every frame pixel covered by that aligned key template.
    - Erase the original key object from the grid.

    Key insight:
    The key is not moved into the frame interior; its shape is a cutting stencil whose side is chosen by the key's relative position around the matching frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        frame, key = _find_pair_parts(g, color)
        frame_bbox = _bbox(frame)
        key_mask = _coords_to_mask(key)
        side = _side_from_positions(frame, key)

        out[key[:, 0], key[:, 1]] = BG

        top, left = _aligned_origin(frame_bbox, key_mask.shape, side)
        for kr, kc in np.argwhere(key_mask):
            rr = top + int(kr)
            cc = left + int(kc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == color:
                out[rr, cc] = BG

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    norm = np.array([(r - r0, c - c0) for r, c in cells], dtype=int)
    order = np.lexsort((norm[:, 1], norm[:, 0]))
    return norm[order]


def _transform_coords(coords: np.ndarray, op: int) -> np.ndarray:
    arr = coords.copy()
    if op & 1:
        arr[:, 1] *= -1
    if op & 2:
        arr[:, 0] *= -1
    if op & 4:
        arr = arr[:, [1, 0]]
    return _normalize_cells(set(map(tuple, arr.tolist())))


def _random_key_mask(max_h: int, max_w: int) -> np.ndarray:
    max_h = max(1, min(int(max_h), 4))
    max_w = max(1, min(int(max_w), 4))
    max_area = min(6, max_h * max_w)

    for _ in range(120):
        size = random.randint(1, max_area)
        cells = {(0, 0)}
        while len(cells) < size:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    frontier.add((r + dr, c + dc))
            frontier.difference_update(cells)
            cells.add(random.choice(tuple(frontier)))
        coords = _normalize_cells(cells)
        coords = _transform_coords(coords, random.randint(0, 7))
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        if r1 + 1 <= max_h and c1 + 1 <= max_w:
            mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            return mask

    fallback = np.ones((1, min(max_w, 2)), dtype=bool)
    return fallback[: max_h, : max_w]


def _pair_patch(
    frame_h: int, frame_w: int, key_mask: np.ndarray, side: str, gap: int, color: int
) -> np.ndarray:
    kh, kw = key_mask.shape
    if side in {"up", "down"}:
        patch_h = frame_h + gap + kh
        patch_w = max(frame_w, kw)
        frame_r = kh + gap if side == "up" else 0
        frame_c = (patch_w - frame_w) // 2
        key_r = 0 if side == "up" else frame_h + gap
        key_c = (patch_w - kw) // 2
    else:
        patch_h = max(frame_h, kh)
        patch_w = frame_w + gap + kw
        frame_r = (patch_h - frame_h) // 2
        frame_c = kw + gap if side == "left" else 0
        key_r = (patch_h - kh) // 2
        key_c = 0 if side == "left" else frame_w + gap

    patch = np.zeros((patch_h, patch_w), dtype=int)
    frame = _frame_mask((frame_h, frame_w))
    patch[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w][frame] = int(color)
    patch[key_r : key_r + kh, key_c : key_c + kw][key_mask] = int(color)
    return patch


def _make_pair_spec(color: int, side: str) -> np.ndarray:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    if side in {"up", "down"}:
        key_mask = _random_key_mask(max_h=min(4, frame_h), max_w=frame_w)
    else:
        key_mask = _random_key_mask(max_h=frame_h, max_w=min(4, frame_w))
    gap = random.randint(*GAP_RANGE)
    return _pair_patch(frame_h, frame_w, key_mask, side, gap, color)


def _reserve_ok(reserved: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    h, w = reserved.shape
    ph, pw = patch.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + ph + 1)
    c1 = min(w, left + pw + 1)
    return not reserved[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    if ph > gh or pw > gw:
        return False

    candidates = [
        (r, c)
        for r in range(gh - ph + 1)
        for c in range(gw - pw + 1)
        if _reserve_ok(reserved, r, c, patch)
    ]
    if not candidates:
        return False

    top, left = random.choice(candidates)
    mask = patch != BG
    grid[top : top + ph, left : left + pw][mask] = patch[mask]

    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(gh, top + ph + 1)
    c1 = min(gw, left + pw + 1)
    reserved[r0:r1, c0:c1] = True
    return True


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    if not colors:
        return False

    for color in colors:
        try:
            frame, key = _find_pair_parts(grid, color)
        except AssertionError:
            return False
        if key.shape[0] > 6:
            return False
        if _side_from_positions(frame, key) not in {"up", "down", "left", "right"}:
            return False

    out = main(grid)
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_OPTIONS)
    colors = random.sample(COLOR_POOL, n_pairs)
    sides = random.sample(ALL_SIDES, k=n_pairs)
    patches = [_make_pair_spec(color, side) for color, side in zip(colors, sides)]
    patches.sort(key=lambda patch: int(patch.shape[0] * patch.shape[1]), reverse=True)

    max_ph = max(int(p.shape[0]) for p in patches)
    max_pw = max(int(p.shape[1]) for p in patches)
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    min_side = max(
        GRID_MIN,
        max(max_ph, max_pw),
        int(np.ceil(np.sqrt(total_area * 1.8))),
    )
    min_side = min(min_side, GRID_MAX)

    for _ in range(PLACE_ATTEMPTS):
        gh = random.randint(min_side, GRID_MAX)
        gw = random.randint(min_side, GRID_MAX)
        grid = np.zeros((gh, gw), dtype=int)
        reserved = np.zeros((gh, gw), dtype=bool)
        ok = True
        for patch in patches:
            if not _place_patch(grid, reserved, patch):
                ok = False
                break
        if ok and _validate_input(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
