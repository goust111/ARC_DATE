import random
from typing import Iterable

import numpy as np


BG = 8
BLACK = 0
GREY = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_component(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _point_to_rect_distance(center: tuple[float, float], bbox: tuple[int, int, int, int]) -> float:
    r, c = center
    r0, c0, r1, c1 = bbox
    if r < r0:
        dr = r0 - r
    elif r > r1:
        dr = r - r1
    else:
        dr = 0.0
    if c < c0:
        dc = c0 - c
    elif c > c1:
        dc = c - c1
    else:
        dc = 0.0
    return float(dr * dr + dc * dc)


def _combine_centered(masks: Iterable[np.ndarray]) -> np.ndarray | None:
    masks = list(masks)
    if not masks:
        return None
    h = max(int(mask.shape[0]) for mask in masks)
    w = max(int(mask.shape[1]) for mask in masks)
    out = np.zeros((h, w), dtype=bool)
    for mask in masks:
        mh, mw = map(int, mask.shape)
        roff = (h - mh) // 2
        coff = (w - mw) // 2
        out[roff : roff + mh, coff : coff + mw] |= mask
    return out


def _paste_centered(rect_bbox: tuple[int, int, int, int], mask: np.ndarray) -> tuple[slice, slice]:
    r0, c0, r1, c1 = rect_bbox
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1
    mh, mw = map(int, mask.shape)
    assert mh <= rh and mw <= rw
    rr = r0 + (rh - mh) // 2
    cc = c0 + (rw - mw) // 2
    return slice(rr, rr + mh), slice(cc, cc + mw)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group nearby black/grey markers by the nearest large solid rectangle, merge same-color
    marker shapes into centered templates, carve the grey template from the rectangle, and
    print the black template on top with carve taking priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    objects = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = int(coords.shape[0])
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": area,
                    "solid": area == h * w,
                }
            )

    rectangles = [obj for obj in objects if obj["solid"] and obj["area"] >= 12]
    markers = [obj for obj in objects if obj not in rectangles]
    assert rectangles, "expected at least one large solid rectangle"
    assert markers, "expected marker components"

    out = np.full_like(g, bg)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])

    groups = {idx: {GREY: [], BLACK: []} for idx in range(len(rectangles))}
    for marker in markers:
        center = tuple(np.mean(marker["coords"], axis=0))
        dists = [_point_to_rect_distance(center, rect["bbox"]) for rect in rectangles]
        owner = int(np.argmin(dists))
        groups[owner].setdefault(int(marker["color"]), []).append(_mask_from_component(marker["coords"]))

    for idx, rect in enumerate(rectangles):
        carve_mask = _combine_centered(groups[idx].get(GREY, []))
        print_mask = _combine_centered(groups[idx].get(BLACK, []))

        if carve_mask is not None:
            rs, cs = _paste_centered(rect["bbox"], carve_mask)
            out[rs, cs][carve_mask] = bg

        if print_mask is not None:
            rs, cs = _paste_centered(rect["bbox"], print_mask)
            paintable = print_mask & (out[rs, cs] != bg)
            out[rs, cs][paintable] = BLACK

    return out


SHAPES = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
]

GRID_MIN = 26
GRID_MAX = 34
RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (7, 12)
RECT_GAP = 6
MARKER_GAP = 1
GEN_RETRIES = 400


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _sample_rectangles(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        for _ in range(200):
            rh = random.randint(*RECT_SIDE_RANGE)
            rw = random.randint(*RECT_SIDE_RANGE)
            r0 = random.randint(1, h - rh - 2)
            c0 = random.randint(1, w - rw - 2)
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_expanded_overlap(bbox, other, RECT_GAP) for other in rects):
                continue
            rects.append(bbox)
            break
        else:
            return None
    rects.sort(key=lambda b: (b[0], b[1]))
    return rects


def _marker_modes() -> tuple[int, int]:
    mode = random.choice(["grey_only", "black_only", "mixed"])
    if mode == "grey_only":
        return random.randint(2, 4), 0
    if mode == "black_only":
        return 0, random.randint(2, 4)
    total = random.randint(2, 4)
    grey_count = random.randint(1, total - 1)
    return grey_count, total - grey_count


def _shape_mask() -> np.ndarray:
    return random.choice(SHAPES).astype(bool)


def _territory_positions(rect: tuple[int, int, int, int], mask: np.ndarray, grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = rect
    mh, mw = map(int, mask.shape)
    positions = []
    for side in random.sample(["top", "bottom", "left", "right"], 4):
        if side == "top":
            rr = range(max(0, r0 - 5 - mh), r0 - 1 - mh + 1)
            cc = range(max(0, c0 - 2), min(w - mw, c1 + 2 - mw + 1) + 1)
        elif side == "bottom":
            rr = range(r1 + 2, min(h - mh, r1 + 5) + 1)
            cc = range(max(0, c0 - 2), min(w - mw, c1 + 2 - mw + 1) + 1)
        elif side == "left":
            rr = range(max(0, r0 - 2), min(h - mh, r1 + 2 - mh + 1) + 1)
            cc = range(max(0, c0 - 5 - mw), c0 - 1 - mw + 1)
        else:
            rr = range(max(0, r0 - 2), min(h - mh, r1 + 2 - mh + 1) + 1)
            cc = range(c1 + 2, min(w - mw, c1 + 5) + 1)
        for tr in rr:
            for tc in cc:
                positions.append((tr, tc))
    random.shuffle(positions)
    return positions


def _center_of_mask(top: int, left: int, mask: np.ndarray) -> tuple[float, float]:
    pts = np.argwhere(mask)
    center = pts.mean(axis=0)
    return float(top + center[0]), float(left + center[1])


def _place_markers(
    rects: list[tuple[int, int, int, int]], grid_shape: tuple[int, int]
) -> list[tuple[int, int, np.ndarray, int]] | None:
    placed: list[tuple[int, int, np.ndarray, int]] = []
    occupied_boxes: list[tuple[int, int, int, int]] = []

    for rect in rects:
        grey_count, black_count = _marker_modes()
        colors = [GREY] * grey_count + [BLACK] * black_count
        random.shuffle(colors)
        for color in colors:
            success = False
            for _ in range(200):
                mask = _shape_mask()
                mh, mw = map(int, mask.shape)
                for tr, tc in _territory_positions(rect, mask, grid_shape):
                    bbox = (tr, tc, tr + mh - 1, tc + mw - 1)
                    if any(_expanded_overlap(bbox, other, MARKER_GAP) for other in occupied_boxes):
                        continue
                    center = _center_of_mask(tr, tc, mask)
                    dists = [_point_to_rect_distance(center, other) for other in rects]
                    best = min(range(len(rects)), key=lambda i: dists[i])
                    second = min(d for i, d in enumerate(dists) if i != best)
                    if rects[best] != rect or second - dists[best] < 4.0:
                        continue
                    occupied_boxes.append(bbox)
                    placed.append((tr, tc, mask, color))
                    success = True
                    break
                if success:
                    break
            if not success:
                return None
    return placed


def _render(rects: list[tuple[int, int, int, int]], markers: list[tuple[int, int, np.ndarray, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    rect_colors = random.sample(list(RECT_COLORS), len(rects))
    for bbox, color in zip(rects, rect_colors):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    for tr, tc, mask, color in markers:
        mh, mw = map(int, mask.shape)
        patch = grid[tr : tr + mh, tc : tc + mw]
        patch[mask] = color
    return grid


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        rect_count = random.randint(*RECT_COUNT_RANGE)
        rects = _sample_rectangles(h, w, rect_count)
        if rects is None:
            continue
        markers = _place_markers(rects, (h, w))
        if markers is None:
            continue
        grid = _render(rects, markers, (h, w))
        if _valid_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid sample")
