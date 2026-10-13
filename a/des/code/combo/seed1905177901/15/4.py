from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    """Return 4-connected monochrome foreground components in row-major order."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), bg, dtype=int)
            patch[arr[:, 0] - int(r0), arr[:, 1] - int(c0)] = color
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                }
            )

    return components


def _edge_touch_count(coords: np.ndarray, shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    touched = 0
    if np.any(coords[:, 0] == 0):
        touched += 1
    if np.any(coords[:, 0] == h - 1):
        touched += 1
    if np.any(coords[:, 1] == 0):
        touched += 1
    if np.any(coords[:, 1] == w - 1):
        touched += 1
    return touched


def _transform_patch(patch: np.ndarray, edge_count: int) -> np.ndarray | None:
    if edge_count == 0:
        return None
    if edge_count == 1:
        return np.fliplr(patch)
    if edge_count == 2:
        return np.rot90(patch, -1)
    return patch.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each foreground object according to how many grid edges it touches,
    then sort the surviving objects by that count and lay them out left-to-right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_components(g, bg=BG)
    kept: list[tuple[int, int, np.ndarray]] = []
    for idx, obj in enumerate(objects):
        edge_count = _edge_touch_count(obj["coords"], g.shape)
        transformed = _transform_patch(obj["patch"], edge_count)
        if transformed is not None:
            kept.append((edge_count, idx, transformed))

    if not kept:
        return np.full((1, 1), BG, dtype=int)

    kept.sort(key=lambda item: (int(item[0]), int(item[1])))
    out_h = max(int(patch.shape[0]) for _, _, patch in kept)
    out_w = sum(int(patch.shape[1]) for _, _, patch in kept) + len(kept) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c0 = 0
    for _edge_count, _idx, patch in kept:
        h, w = map(int, patch.shape)
        out[:h, c0 : c0 + w] = patch
        c0 += w + 1

    return out


def _path(a: tuple[int, int], b: tuple[int, int], horizontal_first: bool) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]

    def step_cols(cur_r: int, cur_c: int, dst_c: int) -> tuple[int, int]:
        if cur_c == dst_c:
            return cur_r, cur_c
        dc = 1 if dst_c > cur_c else -1
        while cur_c != dst_c:
            cur_c += dc
            cells.append((cur_r, cur_c))
        return cur_r, cur_c

    def step_rows(cur_r: int, cur_c: int, dst_r: int) -> tuple[int, int]:
        if cur_r == dst_r:
            return cur_r, cur_c
        dr = 1 if dst_r > cur_r else -1
        while cur_r != dst_r:
            cur_r += dr
            cells.append((cur_r, cur_c))
        return cur_r, cur_c

    if horizontal_first:
        r, c = step_cols(r, c, tc)
        r, c = step_rows(r, c, tr)
    else:
        r, c = step_rows(r, c, tr)
        r, c = step_cols(r, c, tc)
    return cells


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                pts.add((nr, nc))
    return list(pts)


def _random_connected_mask(
    h: int,
    w: int,
    *,
    min_fill: float = 0.35,
    max_fill: float = 0.75,
    require_hflip_change: bool = False,
) -> np.ndarray:
    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        pivot = (random.randrange(h), random.randrange(w))

        for anchor in anchors:
            for cell in _path(anchor, pivot, horizontal_first=bool(random.randrange(2))):
                mask[cell] = True

        area = int(mask.sum())
        lo = max(area, int(np.ceil(min_fill * h * w)))
        hi = max(lo, int(np.floor(max_fill * h * w)))
        target = random.randint(lo, hi)

        while int(mask.sum()) < target:
            choices = _frontier(mask)
            if not choices:
                break
            mask[random.choice(choices)] = True

        if require_hflip_change and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    raise RuntimeError("failed to generate a non-degenerate connected mask")


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _can_place(
    bbox: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
    *,
    pad: int = 1,
) -> bool:
    return all(not _boxes_conflict(bbox, other, pad=pad) for other in existing)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    assert not np.any((view != BG) & mask)
    view[mask] = int(color)


def _build_canonical_layout() -> np.ndarray | None:
    h3 = random.randint(2, 4)
    h1 = random.randint(3, 5)
    w1 = random.randint(3, 6)
    h2 = random.randint(3, 5)
    w2 = random.randint(3, 5)
    if h2 == w2:
        if w2 < 5:
            w2 += 1
        else:
            h2 -= 1
    h0 = random.randint(3, 5)
    w0 = random.randint(3, 5)

    min_h = h3 + max(h1, h0) + h2 + 5
    min_w = w1 + w0 + w2 + 6
    if min_h > 22 or min_w > 26:
        return None
    H = random.randint(min_h, min(22, min_h + 4))
    W = random.randint(min_w, min(26, min_w + 5))

    r1_lo = h3 + 1
    r1_hi = H - h2 - h1 - 2
    if r1_lo > r1_hi:
        return None
    r1 = random.randint(r1_lo, r1_hi)

    top_bbox = (0, 0, h3 - 1, W - 1)
    e1_bbox = (r1, 0, r1 + h1 - 1, w1 - 1)
    e2_bbox = (H - h2, W - w2, H - 1, W - 1)
    boxes = [top_bbox, e1_bbox, e2_bbox]

    for _ in range(80):
        r0 = random.randint(h3 + 1, H - h0 - 2)
        c0 = random.randint(1, W - w0 - 2)
        e0_bbox = (r0, c0, r0 + h0 - 1, c0 + w0 - 1)
        if _can_place(e0_bbox, boxes, pad=1):
            break
    else:
        return None
    boxes.append(e0_bbox)

    extra_count = random.randint(0, 2)
    extras: list[tuple[int, int, int, int]] = []
    for _ in range(extra_count):
        placed = False
        for _ in range(60):
            eh = random.randint(2, 4)
            ew = random.randint(2, 4)
            rr = random.randint(1, H - eh - 2)
            cc = random.randint(1, W - ew - 2)
            bbox = (rr, cc, rr + eh - 1, cc + ew - 1)
            if _can_place(bbox, boxes + extras, pad=1):
                extras.append(bbox)
                placed = True
                break
        if not placed:
            break

    colors = random.sample(PALETTE, 4 + len(extras))
    grid = np.full((H, W), BG, dtype=int)

    top_mask = _random_connected_mask(h3, W, min_fill=0.30, max_fill=0.65)
    e1_mask = _random_connected_mask(h1, w1, min_fill=0.35, max_fill=0.75, require_hflip_change=True)
    e2_mask = _random_connected_mask(h2, w2, min_fill=0.35, max_fill=0.80)
    e0_mask = _random_connected_mask(h0, w0, min_fill=0.35, max_fill=0.75)

    _paint_mask(grid, top_mask, top_bbox[0], top_bbox[1], colors[0])
    _paint_mask(grid, e1_mask, e1_bbox[0], e1_bbox[1], colors[1])
    _paint_mask(grid, e2_mask, e2_bbox[0], e2_bbox[1], colors[2])
    _paint_mask(grid, e0_mask, e0_bbox[0], e0_bbox[1], colors[3])

    for i, bbox in enumerate(extras, start=4):
        r0, c0, r1, c1 = bbox
        mask = _random_connected_mask(r1 - r0 + 1, c1 - c0 + 1, min_fill=0.40, max_fill=0.85)
        _paint_mask(grid, mask, r0, c0, colors[i])

    return grid


def _apply_global_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _valid_generated_grid(grid: np.ndarray) -> bool:
    objects = _find_components(grid, bg=BG)
    if len(objects) < 4:
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False

    counts = [_edge_touch_count(obj["coords"], grid.shape) for obj in objects]
    if 0 not in counts or 1 not in counts or 2 not in counts or not any(count >= 3 for count in counts):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if out.ndim != 2 or out.size == 0:
        return False
    if int(np.sum(out != BG)) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        base = _build_canonical_layout()
        if base is None:
            continue
        grid = _apply_global_transform(base)
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate valid input")
