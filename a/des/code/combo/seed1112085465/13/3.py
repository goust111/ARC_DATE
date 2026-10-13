import random
from collections import deque

import numpy as np


MAROON = 9
COLOR_POOL = tuple(range(9))
GRID_H_RANGE = (8, 18)
GRID_W_RANGE = (8, 18)
RECT_COUNT_RANGE = (1, 4)
MAX_RECT_SIDE = 6
MAX_AREA_RATIO = 0.38
GEN_ATTEMPTS = 400

Rect = tuple[int, int, int, int]


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    parity = ((rows + cols) & 1) == 0
    return np.where(parity, even_color, odd_color).astype(int)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)

    return comps


def _bbox(cells: list[tuple[int, int]]) -> Rect:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_mask(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _render_expected(clean: np.ndarray, rects: list[Rect]) -> np.ndarray:
    out = clean.copy()
    anomaly_mask = _rect_mask(clean.shape, rects)
    h, w = clean.shape

    for r0, c0, r1, c1 in rects:
        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(h - 1, r1 + 1)
        ec1 = min(w - 1, c1 + 1)
        for r in range(er0, er1 + 1):
            for c in range(ec0, ec1 + 1):
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue
                if anomaly_mask[r, c]:
                    continue
                out[r, c] = MAROON

    return out


def _infer_checkerboard(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g)]
    assert len(colors) >= 3, "expected two checker colors and one anomaly color"

    rows, cols = np.indices(g.shape)
    even_mask = ((rows + cols) & 1) == 0
    odd_mask = ~even_mask

    scored: list[tuple[int, int, int]] = []
    for even_color in colors:
        even_hits = int(np.count_nonzero(g[even_mask] == even_color))
        for odd_color in colors:
            if odd_color == even_color:
                continue
            odd_hits = int(np.count_nonzero(g[odd_mask] == odd_color))
            scored.append((even_hits + odd_hits, even_color, odd_color))

    scored.sort(reverse=True)
    best_score, even_color, odd_color = scored[0]
    if len(scored) > 1:
        assert best_score > scored[1][0], "checkerboard phase is ambiguous"

    expected = _checkerboard(g.shape, even_color, odd_color)
    anomaly_mask = g != expected
    assert np.any(anomaly_mask), "expected at least one anomaly block"

    anomaly_colors = {int(v) for v in g[anomaly_mask]}
    assert len(anomaly_colors) == 1, "anomaly cells must share one color"
    assert next(iter(anomaly_colors)) not in (even_color, odd_color)

    return expected, anomaly_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair rectangular anomaly blocks in a two-color checkerboard and draw a
    one-cell maroon halo around each block's bounding box.
    """
    g = np.asarray(grid, dtype=int)
    expected, anomaly_mask = _infer_checkerboard(g)
    out = expected.copy()

    for comp in _components(anomaly_mask):
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "anomaly components must be solid rectangles"

        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(g.shape[0] - 1, r1 + 1)
        ec1 = min(g.shape[1] - 1, c1 + 1)

        for r in range(er0, er1 + 1):
            for c in range(ec0, ec1 + 1):
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue
                if anomaly_mask[r, c]:
                    continue
                out[r, c] = MAROON

    return out


def _sample_rect_size(height: int, width: int) -> tuple[int, int]:
    max_h = min(MAX_RECT_SIDE, height - 1)
    max_w = min(MAX_RECT_SIDE, width - 1)
    assert max_h >= 1 and max_w >= 1

    if random.random() < 0.35:
        if random.random() < 0.5:
            return 1, random.randint(1, max_w)
        return random.randint(1, max_h), 1
    return random.randint(1, max_h), random.randint(1, max_w)


def _can_place_rect(occupied: np.ndarray, rect: Rect) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if occupied[r, c]:
                return False
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _stamp_rect(occupied: np.ndarray, rect: Rect) -> None:
    r0, c0, r1, c1 = rect
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_rect(height: int, width: int, occupied: np.ndarray, max_area: int, touch_edge: bool = False) -> Rect | None:
    for _ in range(120):
        rh, rw = _sample_rect_size(height, width)
        rect_area = rh * rw
        if rect_area > max_area:
            continue

        if touch_edge:
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                r0 = 0
                c0 = random.randint(0, width - rw)
            elif side == "bottom":
                r0 = height - rh
                c0 = random.randint(0, width - rw)
            elif side == "left":
                r0 = random.randint(0, height - rh)
                c0 = 0
            else:
                r0 = random.randint(0, height - rh)
                c0 = width - rw
        else:
            r0 = random.randint(0, height - rh)
            c0 = random.randint(0, width - rw)

        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if _can_place_rect(occupied, rect):
            return rect
    return None


def _expanded_boxes_overlap(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ar0, ac0, ar1, ac1 = ar0 - 1, ac0 - 1, ar1 + 1, ac1 + 1
    br0, bc0, br1, bc1 = br0 - 1, bc0 - 1, br1 + 1, bc1 + 1
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _overlap_pair(height: int, width: int, max_area: int) -> list[Rect] | None:
    for _ in range(180):
        h1, w1 = _sample_rect_size(height, width)
        h2, w2 = _sample_rect_size(height, width)
        if h1 * w1 + h2 * w2 > max_area:
            continue

        r1 = random.randint(0, height - h1)
        c1 = random.randint(0, width - w1)
        horizontal = random.random() < 0.5

        if horizontal:
            side = random.choice((-1, 1))
            c2 = c1 + w1 + 2 if side == 1 else c1 - w2 - 2
            if not (0 <= c2 <= width - w2):
                continue
            low = max(0, r1 - h2 + 1)
            high = min(height - h2, r1 + h1 - 1)
            if low > high:
                continue
            r2 = random.randint(low, high)
        else:
            side = random.choice((-1, 1))
            r2 = r1 + h1 + 2 if side == 1 else r1 - h2 - 2
            if not (0 <= r2 <= height - h2):
                continue
            low = max(0, c1 - w2 + 1)
            high = min(width - w2, c1 + w1 - 1)
            if low > high:
                continue
            c2 = random.randint(low, high)

        rects = [
            (r1, c1, r1 + h1 - 1, c1 + w1 - 1),
            (r2, c2, r2 + h2 - 1, c2 + w2 - 1),
        ]
        occupied = np.zeros((height, width), dtype=bool)
        if not _can_place_rect(occupied, rects[0]):
            continue
        _stamp_rect(occupied, rects[0])
        if not _can_place_rect(occupied, rects[1]):
            continue
        if not _expanded_boxes_overlap(rects[0], rects[1]):
            continue
        return rects
    return None


def _build_rectangles(height: int, width: int) -> list[Rect] | None:
    mode = random.choice(("free", "edge", "overlap"))
    target_count = random.randint(*RECT_COUNT_RANGE)
    if mode == "overlap":
        target_count = max(2, target_count)

    max_area = max(6, int(height * width * MAX_AREA_RATIO))
    used_area = 0
    rects: list[Rect] = []
    occupied = np.zeros((height, width), dtype=bool)

    if mode == "edge":
        rect = _random_rect(height, width, occupied, max_area - used_area, touch_edge=True)
        if rect is None:
            return None
        rects.append(rect)
        _stamp_rect(occupied, rect)
        used_area += _rect_area(rect)
    elif mode == "overlap":
        pair = _overlap_pair(height, width, max_area - used_area)
        if pair is None:
            return None
        for rect in pair:
            rects.append(rect)
            _stamp_rect(occupied, rect)
            used_area += _rect_area(rect)

    while len(rects) < target_count:
        rect = _random_rect(height, width, occupied, max_area - used_area, touch_edge=False)
        if rect is None:
            break
        rects.append(rect)
        _stamp_rect(occupied, rect)
        used_area += _rect_area(rect)

    min_count = 2 if mode == "overlap" else 1
    return rects if len(rects) >= min_count else None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    colors = list(COLOR_POOL)
    random.shuffle(colors)
    even_color, odd_color, anomaly_color = colors[:3]
    clean = _checkerboard((height, width), even_color, odd_color)

    rects = _build_rectangles(height, width)
    if rects is None:
        return None

    anomaly_mask = _rect_mask((height, width), rects)
    if {int(v) for v in clean[~anomaly_mask]} != {even_color, odd_color}:
        return None

    inp = clean.copy()
    inp[anomaly_mask] = anomaly_color
    expected = _render_expected(clean, rects)

    try:
        solved = main(inp)
    except Exception:
        return None

    if np.array_equal(solved, expected) and not np.array_equal(inp, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard anomaly task")
