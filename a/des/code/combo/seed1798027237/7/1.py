from __future__ import annotations

from collections import deque
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _dominant_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _nearest_side(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    side = _nearest_side_or_none(bbox, shape)
    assert side is not None, "rectangle must be uniquely closest to one boundary"
    return side


def _nearest_side_or_none(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    h, w = shape
    dists = {
        "top": int(r0),
        "bottom": int(h - 1 - r1),
        "left": int(c0),
        "right": int(w - 1 - c1),
    }
    best = min(dists.values())
    sides = [side for side, dist in dists.items() if dist == best]
    return sides[0] if len(sides) == 1 else None


def _hole_shape(region: np.ndarray, bg: int) -> np.ndarray:
    bg_mask = region == bg
    holes = []
    h, w = region.shape
    for comp in _components(bg_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        hole = np.zeros_like(bg_mask, dtype=bool)
        for r, c in comp:
            hole[r, c] = True
        holes.append(hole)
    assert len(holes) == 1, "expected exactly one interior hole per rectangle"
    return _crop_mask(holes[0])


def _align_start(center: float, size: int) -> int:
    return int(np.floor(center - (size - 1) / 2 + 0.5))


def _stamp_origin(
    side: str,
    rect_bbox: tuple[int, int, int, int],
    stamp_shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect_bbox
    h, w = grid_shape
    sh, sw = stamp_shape
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    if side == "top":
        return 0, _align_start(center_c, sw)
    if side == "bottom":
        return h - sh, _align_start(center_c, sw)
    if side == "left":
        return _align_start(center_r, sh), 0
    return _align_start(center_r, sh), w - sw


def _mirrored(mask: np.ndarray, side: str) -> np.ndarray:
    return np.flipud(mask) if side in ("top", "bottom") else np.fliplr(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each rectangle's interior-hole silhouette onto its nearest grid edge.

    Input structure
    - A black square background contains several monochrome rectangles.
    - Each rectangle has one connected black hole strictly inside it.
    - Every rectangle is uniquely closest to one grid boundary.

    Rule
    - Keep all rectangles and holes unchanged.
    - Extract each hole shape relative to its own bounding box.
    - Mirror the hole toward the nearest boundary and stamp it there using the
      rectangle's color, centered against the rectangle along the orthogonal axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1]

    bg = _dominant_color(g)
    out = g.copy()

    for comp in _components(g != bg):
        r0, c0, r1, c1 = _bbox(comp)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, region[region != bg].tolist()))
        assert len(colors) == 1, "each component must be monochrome"
        color = int(next(iter(colors)))

        hole = _hole_shape(region, bg)
        side = _nearest_side((r0, c0, r1, c1), g.shape)
        stamp = _mirrored(hole, side)
        sr, sc = _stamp_origin(side, (r0, c0, r1, c1), stamp.shape, g.shape)
        sh, sw = stamp.shape
        assert 0 <= sr and 0 <= sc and sr + sh <= g.shape[0] and sc + sw <= g.shape[1]
        out[sr : sr + sh, sc : sc + sw][stamp] = color

    return out


# --------------- generator ---------------

GRID_SIZE_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (1, 4)
RAW_HOLE_SIDE_RANGE = (1, 5)
RECT_MARGIN_RANGE = (1, 3)
HOLE_GAP_EXTRA_RANGE = (0, 2)
PLACEMENT_ATTEMPTS = 240
GENERATION_ATTEMPTS = 400


def _random_connected_shape() -> np.ndarray:
    for _ in range(200):
        raw_h = random.randint(*RAW_HOLE_SIDE_RANGE)
        raw_w = random.randint(*RAW_HOLE_SIDE_RANGE)
        target = random.randint(1, raw_h * raw_w)
        seed = (random.randrange(raw_h), random.randrange(raw_w))
        cells = {seed}
        frontier = {pos for pos in _neighbors4(*seed) if 0 <= pos[0] < raw_h and 0 <= pos[1] < raw_w}
        while len(cells) < target:
            pos = random.choice(tuple(frontier))
            frontier.remove(pos)
            cells.add(pos)
            for nxt in _neighbors4(*pos):
                if 0 <= nxt[0] < raw_h and 0 <= nxt[1] < raw_w and nxt not in cells:
                    frontier.add(nxt)
        mask = np.zeros((raw_h, raw_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if _hole_shape_is_valid(mask):
            for _ in range(random.randrange(4)):
                mask = np.rot90(mask)
            if random.random() < 0.5:
                mask = np.flipud(mask)
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            return mask
    raise RuntimeError("failed to build a valid hole shape")


def _hole_shape_is_valid(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    if len(_components(mask)) != 1:
        return False
    comp_false = _components(~mask)
    h, w = mask.shape
    for comp in comp_false:
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            return False
    return True


def _inflate_bbox_mask(n: int, bbox: tuple[int, int, int, int], pad: int = 1) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((n, n), dtype=bool)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(n - 1, r1 + pad)
    cc1 = min(n - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _placeable(
    blocked: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    stamp_bbox: tuple[int, int, int, int],
) -> bool:
    n = blocked.shape[0]
    rect_mask = _inflate_bbox_mask(n, rect_bbox, pad=1)
    stamp_mask = _inflate_bbox_mask(n, stamp_bbox, pad=1)
    return not bool(np.any(blocked & (rect_mask | stamp_mask)))


def _stamp_bbox(side: str, rect_bbox: tuple[int, int, int, int], hole: np.ndarray, n: int) -> tuple[int, int, int, int]:
    stamp = _mirrored(hole, side)
    sr, sc = _stamp_origin(side, rect_bbox, stamp.shape, (n, n))
    sh, sw = stamp.shape
    return sr, sc, sr + sh - 1, sc + sw - 1


def _candidate_rect(side: str, rect_h: int, rect_w: int, hole: np.ndarray, n: int) -> tuple[int, int, int, int] | None:
    hh, hw = hole.shape
    if side in ("top", "bottom"):
        gap = random.randint(hh + HOLE_GAP_EXTRA_RANGE[0], hh + HOLE_GAP_EXTRA_RANGE[1])
        if rect_w > n or gap + rect_h > n:
            return None
        r0 = gap if side == "top" else n - gap - rect_h
        c0 = random.randint(0, n - rect_w)
    else:
        gap = random.randint(hw + HOLE_GAP_EXTRA_RANGE[0], hw + HOLE_GAP_EXTRA_RANGE[1])
        if rect_h > n or gap + rect_w > n:
            return None
        r0 = random.randint(0, n - rect_h)
        c0 = gap if side == "left" else n - gap - rect_w

    bbox = (int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1))
    if _nearest_side_or_none(bbox, (n, n)) != side:
        return None
    sb = _stamp_bbox(side, bbox, hole, n)
    if not (0 <= sb[0] <= sb[2] < n and 0 <= sb[1] <= sb[3] < n):
        return None
    if not (sb[2] < bbox[0] or sb[0] > bbox[2] or sb[3] < bbox[1] or sb[1] > bbox[3]):
        return None
    return bbox


def _apply_object(
    grid: np.ndarray,
    color: int,
    rect_bbox: tuple[int, int, int, int],
    hole: np.ndarray,
    margins: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = rect_bbox
    top_margin, left_margin = margins
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    hh, hw = hole.shape
    hr0 = r0 + top_margin
    hc0 = c0 + left_margin
    region = grid[hr0 : hr0 + hh, hc0 : hc0 + hw]
    region[hole] = 0


def _expected_output(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    n = inp.shape[0]
    for obj in objects:
        stamp = _mirrored(obj["hole"], obj["side"])
        sr, sc = _stamp_origin(obj["side"], obj["bbox"], stamp.shape, (n, n))
        out[sr : sr + stamp.shape[0], sc : sc + stamp.shape[1]][stamp] = int(obj["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GRID_SIZE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    sides = random.sample(SIDES, count)
    colors = random.sample(range(1, 10), count)

    grid = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=bool)
    objects: list[dict] = []

    for side, color in zip(sides, colors, strict=False):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            hole = _random_connected_shape()
            hh, hw = hole.shape
            top_margin = random.randint(*RECT_MARGIN_RANGE)
            bottom_margin = random.randint(*RECT_MARGIN_RANGE)
            left_margin = random.randint(*RECT_MARGIN_RANGE)
            right_margin = random.randint(*RECT_MARGIN_RANGE)
            rect_h = hh + top_margin + bottom_margin
            rect_w = hw + left_margin + right_margin
            bbox = _candidate_rect(side, rect_h, rect_w, hole, n)
            if bbox is None:
                continue
            stamp_bbox = _stamp_bbox(side, bbox, hole, n)
            if not _placeable(blocked, bbox, stamp_bbox):
                continue

            _apply_object(grid, color, bbox, hole, (top_margin, left_margin))
            blocked |= _inflate_bbox_mask(n, bbox, pad=1)
            blocked |= _inflate_bbox_mask(n, stamp_bbox, pad=1)
            objects.append(
                {
                    "bbox": bbox,
                    "color": int(color),
                    "hole": hole.copy(),
                    "side": side,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, objects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
