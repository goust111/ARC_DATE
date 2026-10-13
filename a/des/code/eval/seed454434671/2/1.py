from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
WINDOW_COLOR = 8
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNER_BY_SIDE = {
    "up": "top_left",
    "right": "top_right",
    "left": "bottom_left",
    "down": "bottom_right",
}
STAMP_COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _stamp_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        base = np.fliplr(mask) if flipped else mask
        for turns in range(4):
            variant = np.rot90(base, turns)
            sig = _mask_signature(variant)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(variant.copy())
    return out


_BASE_STAMP_MASKS = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]], dtype=bool),
]
STAMP_MASK_CATALOG: list[np.ndarray] = []
_stamp_seen: set[tuple[tuple[int, int], bytes]] = set()
for _base_mask in _BASE_STAMP_MASKS:
    for _variant in _stamp_variants(_base_mask):
        _sig = _mask_signature(_variant)
        if _sig not in _stamp_seen:
            _stamp_seen.add(_sig)
            STAMP_MASK_CATALOG.append(_variant)


def _frame_border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _corner_origin(corner: str, grid_shape: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    ph, pw = patch_shape
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, w - pw
    if corner == "bottom_left":
        return h - ph, 0
    if corner == "bottom_right":
        return h - ph, w - pw
    raise ValueError(corner)


def _find_corner_stamps(grid: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    h, w = grid.shape
    mask = (grid != bg) & (grid != FRAME_COLOR) & (grid != WINDOW_COLOR)
    stamps: dict[str, np.ndarray] = {}
    for cells in _components(mask):
        r0, c0, r1, c1 = _bbox(cells)
        if r0 == 0 and c0 == 0:
            corner = "top_left"
        elif r0 == 0 and c1 == w - 1:
            corner = "top_right"
        elif r1 == h - 1 and c0 == 0:
            corner = "bottom_left"
        elif r1 == h - 1 and c1 == w - 1:
            corner = "bottom_right"
        else:
            continue
        stamps[corner] = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert set(stamps) == set(CORNER_BY_SIDE.values())
    return stamps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for cells in _components(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_border_mask(sub.shape)
        if np.all(sub[border] == FRAME_COLOR) and not np.any(sub[1:-1, 1:-1] == FRAME_COLOR):
            frames.append((r0, c0, r1, c1))
    assert frames
    return sorted(frames)


def _window_side(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = frame_bbox
    inner = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == WINDOW_COLOR)
    assert inner.shape == (2, 2)
    inner = inner + np.array([r0 + 1, c0 + 1])
    (ra, ca), (rb, cb) = inner.tolist()
    if ra == rb and abs(ca - cb) == 1:
        if ra == r0 + 1:
            return "up"
        if ra == r1 - 1:
            return "down"
    if ca == cb and abs(ra - rb) == 1:
        if ca == c0 + 1:
            return "left"
        if ca == c1 - 1:
            return "right"
    raise AssertionError("window marker is not aligned with a frame side")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = int(np.bincount(g.ravel()).argmax())
    stamps = _find_corner_stamps(g, bg)
    frames = _find_frames(g)

    out = g.copy()
    for frame_bbox in frames:
        side = _window_side(g, frame_bbox)
        stamp = stamps[CORNER_BY_SIDE[side]]
        r0, c0, r1, c1 = frame_bbox
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        stamp_h, stamp_w = stamp.shape
        assert stamp_h <= inner_h and stamp_w <= inner_w
        top = r0 + 1 + (inner_h - stamp_h) // 2
        left = c0 + 1 + (inner_w - stamp_w) // 2
        target = out[top : top + stamp_h, left : left + stamp_w]
        stamp_mask = stamp != bg
        draw_mask = stamp_mask & (target == bg)
        target[draw_mask] = stamp[draw_mask]
    return out


def _sample_stamps() -> dict[str, np.ndarray]:
    colors = random.sample(STAMP_COLOR_POOL, 4)
    masks = [mask.copy() for mask in random.sample(STAMP_MASK_CATALOG, 4)]
    corner_order = ["top_left", "top_right", "bottom_left", "bottom_right"]
    stamps: dict[str, np.ndarray] = {}
    for corner, color, mask in zip(corner_order, colors, masks):
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = color
        stamps[corner] = patch
    return stamps


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    target = grid[top : top + ph, left : left + pw]
    patch_mask = patch != BG
    assert np.all(target[patch_mask] == BG)
    target[patch_mask] = patch[patch_mask]


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR


def _draw_window(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], side: str) -> None:
    r0, c0, r1, c1 = frame_bbox
    if side == "up":
        col = random.randint(c0 + 1, c1 - 2)
        grid[r0 + 1, col : col + 2] = WINDOW_COLOR
        return
    if side == "down":
        col = random.randint(c0 + 1, c1 - 2)
        grid[r1 - 1, col : col + 2] = WINDOW_COLOR
        return
    if side == "left":
        row = random.randint(r0 + 1, r1 - 2)
        grid[row : row + 2, c0 + 1] = WINDOW_COLOR
        return
    if side == "right":
        row = random.randint(r0 + 1, r1 - 2)
        grid[row : row + 2, c1 - 1] = WINDOW_COLOR
        return
    raise ValueError(side)


def _reserve(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    occupied[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _place_frames(
    occupied: np.ndarray, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    h, w = occupied.shape
    boxes: list[tuple[int, int, int, int]] = []
    for outer_h, outer_w in sorted(sizes, key=lambda dims: dims[0] * dims[1], reverse=True):
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(2, h - outer_h):
            r1 = r0 + outer_h - 1
            for c0 in range(2, w - outer_w):
                c1 = c0 + outer_w - 1
                if not occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                    candidates.append((r0, c0, r1, c1))
        if not candidates:
            return None
        bbox = random.choice(candidates)
        boxes.append(bbox)
        _reserve(occupied, bbox, pad=1)
    random.shuffle(boxes)
    return boxes


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(30, 38)
    width = random.randint(30, 38)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    stamps = _sample_stamps()
    for corner, patch in stamps.items():
        top, left = _corner_origin(corner, grid.shape, patch.shape)
        _paint_patch(grid, patch, top, left)
        ph, pw = patch.shape
        _reserve(occupied, (top, left, top + ph - 1, left + pw - 1), pad=1)

    frame_count = random.randint(2, 4)
    frame_sizes = [(random.randint(9, 12), random.randint(9, 12)) for _ in range(frame_count)]
    frame_boxes = _place_frames(occupied, frame_sizes)
    if frame_boxes is None:
        return None

    sides = random.sample(list(CORNER_BY_SIDE), frame_count)
    for frame_bbox, side in zip(frame_boxes, sides):
        _draw_frame(grid, frame_bbox)
        _draw_window(grid, frame_bbox, side)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
