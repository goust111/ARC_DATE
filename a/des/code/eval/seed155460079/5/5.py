from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

NOTCH_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
]

GRID_RANGE = (18, 30)
RECT_SIDE_RANGE = (5, 8)
OBJECT_COUNT_CHOICES = [2, 3, 4]
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_trays(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    trays: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub == GRAY, border):
            trays.append((r0, c0, r1, c1))
    return trays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each same-colored edge notch with the matching rotated fragment, then clear the trays.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    tray_mask = np.zeros_like(g, dtype=bool)
    fragments: dict[int, np.ndarray] = {}
    for r0, c0, r1, c1 in _find_trays(g):
        tray_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(v) for v in np.unique(inner) if int(v) not in (BG, GRAY)]
        if len(colors) != 1:
            continue
        color = int(colors[0])
        fragments[color] = _tight_crop(inner == color)
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    clean = g.copy()
    clean[tray_mask] = BG

    for color in [int(v) for v in np.unique(clean) if int(v) not in (BG, GRAY)]:
        comps = _components(clean == color)
        if not comps:
            continue
        major = max(comps, key=len)
        r0, c0, r1, c1 = _bbox(major)
        sub = clean[r0 : r1 + 1, c0 : c1 + 1]
        gap = sub == BG
        if not gap.any() or color not in fragments:
            continue
        gap_shape = _tight_crop(gap)
        match_count = 0
        for k in range(4):
            rotated = np.rot90(fragments[color], k)
            if rotated.shape == gap_shape.shape and np.array_equal(rotated, gap_shape):
                match_count += 1
        if match_count == 1:
            target = out[r0 : r1 + 1, c0 : c1 + 1]
            target[gap] = color

    return out


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _make_object(color: int) -> dict[str, np.ndarray]:
    for _ in range(100):
        notch = np.rot90(random.choice(NOTCH_LIBRARY), random.randrange(4))
        side = random.choice(["top", "bottom", "left", "right"])
        rect_h = random.randint(*RECT_SIDE_RANGE)
        rect_w = random.randint(*RECT_SIDE_RANGE)
        notch_h, notch_w = notch.shape

        if side in ("top", "bottom"):
            if rect_w < notch_w + 2 or rect_h < notch_h + 1:
                continue
            row = 0 if side == "top" else rect_h - notch_h
            col = random.randint(1, rect_w - notch_w - 1)
        else:
            if rect_h < notch_h + 2 or rect_w < notch_w + 1:
                continue
            row = random.randint(1, rect_h - notch_h - 1)
            col = 0 if side == "left" else rect_w - notch_w

        full_shape = np.full((rect_h, rect_w), color, dtype=int)
        input_shape = full_shape.copy()
        window = input_shape[row : row + notch_h, col : col + notch_w]
        window[notch] = BG

        rotated_keys = {_rotation_key(np.rot90(notch, k)) for k in range(4)}
        if len(rotated_keys) != 4:
            continue

        tray_frag = np.rot90(notch, random.randrange(4))
        tray_side = max(tray_frag.shape) + 2
        tray = np.full((tray_side, tray_side), BG, dtype=int)
        tray[0, :] = GRAY
        tray[-1, :] = GRAY
        tray[:, 0] = GRAY
        tray[:, -1] = GRAY

        inner_h = tray_side - 2
        inner_w = tray_side - 2
        off_r = random.randint(0, inner_h - tray_frag.shape[0])
        off_c = random.randint(0, inner_w - tray_frag.shape[1])
        tray[1 + off_r : 1 + off_r + tray_frag.shape[0], 1 + off_c : 1 + off_c + tray_frag.shape[1]][tray_frag] = color

        return {"input_shape": input_shape, "output_shape": full_shape, "tray": tray}
    raise RuntimeError("failed to build a valid object")


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(h, top + height + PLACEMENT_PADDING)
    c1 = min(w, left + width + PLACEMENT_PADDING)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(canvas: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    starts = [(r, c) for r in range(canvas.shape[0] - h + 1) for c in range(canvas.shape[1] - w + 1)]
    random.shuffle(starts)
    for top, left in starts:
        if _can_place(occupied, top, left, h, w):
            canvas[top : top + h, left : left + w] = patch
            occupied[top : top + h, left : left + w] = True
            return top, left
    return None


def _gen_attempt() -> np.ndarray | None:
    obj_count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, obj_count)
    objects = [_make_object(color) for color in colors]

    min_side = max(GRID_RANGE[0], 6 * obj_count + random.randint(0, 2))
    max_side = GRID_RANGE[1]
    if min_side > max_side:
        return None
    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    items: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for obj in objects:
        items.append(("shape", obj["input_shape"], obj["output_shape"]))
        items.append(("tray", obj["tray"], None))
    random.shuffle(items)

    for kind, patch, solved in items:
        pos = _place_patch(inp, occupied, patch)
        if pos is None:
            return None
        top, left = pos
        if kind == "shape":
            assert solved is not None
            h, w = solved.shape
            out[top : top + h, left : left + w] = solved

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
