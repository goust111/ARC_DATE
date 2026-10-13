from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
OBJECT_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

GRID_MIN = 16
GRID_MAX = 30
PAIR_COUNT_OPTIONS = [2, 2, 3, 3, 4]
SINGLE_COUNT_MAX = 2
RECT_SHORT_MIN = 1
RECT_SHORT_MAX = 5
RECT_LONG_MAX = 6
PADDING = 1

MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 250
SINGLE_PLACEMENT_ATTEMPTS = 200

Rect = tuple[int, int, int, int]


def _find_components(g: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = g.shape
    mask = g == int(color)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_shape(box: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _short_side(box: Rect) -> int:
    h, w = _rect_shape(box)
    return min(h, w)


def _assert_solid_rectangle(g: np.ndarray, color: int, coords: np.ndarray) -> Rect:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    h, w = _rect_shape(box)
    assert coords.shape[0] == h * w, "component must fill its bounding box"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)), "rectangle bbox must be solid"
    return box


def _gap_info(a: Rect, b: Rect) -> tuple[int, Rect]:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    if ar0 == br0 and ar1 == br1:
        if ac1 < bc0:
            gap = int(bc0 - ac1 - 1)
            gap_box = (ar0, ac1 + 1, ar1, bc0 - 1)
        elif bc1 < ac0:
            gap = int(ac0 - bc1 - 1)
            gap_box = (ar0, bc1 + 1, ar1, ac0 - 1)
        else:
            raise AssertionError("aligned horizontal rectangles must be separated")
        assert gap > 0, "paired rectangles must have a background gap"
        return gap, gap_box

    if ac0 == bc0 and ac1 == bc1:
        if ar1 < br0:
            gap = int(br0 - ar1 - 1)
            gap_box = (ar1 + 1, ac0, br0 - 1, ac1)
        elif br1 < ar0:
            gap = int(ar0 - br1 - 1)
            gap_box = (br1 + 1, ac0, ar0 - 1, ac1)
        else:
            raise AssertionError("aligned vertical rectangles must be separated")
        assert gap > 0, "paired rectangles must have a background gap"
        return gap, gap_box

    raise AssertionError("same-color pair must share an exact row span or column span")


def _paint_rect(g: np.ndarray, box: Rect, color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the gap between close same-color rectangles with red.

    Input structure:
    - Background is black.
    - Each foreground object is a solid monochrome rectangle.
    - Some colors appear once; some appear exactly twice as an aligned pair.

    Rule:
    - For each same-color pair, compute the background gap between the two rectangles.
    - Let the threshold be the smaller short-side length of the two rectangles.
    - If the gap is at most that threshold, fill the whole gap rectangle with red.
    - Otherwise leave the pair unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        components = _find_components(g, color)
        assert len(components) in (1, 2), "each foreground color must appear once or twice"
        if len(components) == 1:
            continue

        boxes = [_assert_solid_rectangle(g, color, comp) for comp in components]
        gap, gap_box = _gap_info(boxes[0], boxes[1])
        threshold = min(_short_side(boxes[0]), _short_side(boxes[1]))
        r0, c0, r1, c1 = gap_box
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BG), "gap region must be background"
        if gap <= threshold:
            out[r0 : r1 + 1, c0 : c1 + 1] = RED

    return out


def _expanded(box: Rect, h: int, w: int, pad: int = PADDING) -> Rect:
    r0, c0, r1, c1 = box
    return (
        max(0, int(r0 - pad)),
        max(0, int(c0 - pad)),
        min(h - 1, int(r1 + pad)),
        min(w - 1, int(c1 + pad)),
    )


def _reserve_ok(reserved: np.ndarray, box: Rect) -> bool:
    r0, c0, r1, c1 = _expanded(box, reserved.shape[0], reserved.shape[1])
    return not bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_reserved(reserved: np.ndarray, box: Rect) -> None:
    r0, c0, r1, c1 = _expanded(box, reserved.shape[0], reserved.shape[1])
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _try_place_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    orientation: str,
    branch: str,
) -> dict[str, object] | None:
    h, w = grid.shape

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        if orientation == "h":
            rect_h = random.randint(RECT_SHORT_MIN, min(RECT_SHORT_MAX, h - 2))
            w1 = random.randint(RECT_SHORT_MIN, min(RECT_LONG_MAX, w - 2))
            w2 = random.randint(RECT_SHORT_MIN, min(RECT_LONG_MAX, w - 2))
            threshold = min(min(rect_h, w1), min(rect_h, w2))
            max_gap = w - 2 - w1 - w2
            if branch == "fill":
                gap_lo, gap_hi = 1, min(threshold, max_gap)
            else:
                gap_lo, gap_hi = threshold + 1, min(max_gap, threshold + 5)
            if gap_lo > gap_hi:
                continue
            gap = random.randint(gap_lo, gap_hi)
            total_w = w1 + gap + w2
            if total_w > w - 2:
                continue
            r0 = random.randint(1, h - rect_h - 1)
            c0 = random.randint(1, w - total_w - 1)
            box1 = (r0, c0, r0 + rect_h - 1, c0 + w1 - 1)
            box2 = (r0, c0 + w1 + gap, r0 + rect_h - 1, c0 + total_w - 1)
            hull = (r0, c0, r0 + rect_h - 1, c0 + total_w - 1)
            fill_box = (r0, c0 + w1, r0 + rect_h - 1, c0 + w1 + gap - 1)
        else:
            rect_w = random.randint(RECT_SHORT_MIN, min(RECT_SHORT_MAX, w - 2))
            h1 = random.randint(RECT_SHORT_MIN, min(RECT_LONG_MAX, h - 2))
            h2 = random.randint(RECT_SHORT_MIN, min(RECT_LONG_MAX, h - 2))
            threshold = min(min(h1, rect_w), min(h2, rect_w))
            max_gap = h - 2 - h1 - h2
            if branch == "fill":
                gap_lo, gap_hi = 1, min(threshold, max_gap)
            else:
                gap_lo, gap_hi = threshold + 1, min(max_gap, threshold + 5)
            if gap_lo > gap_hi:
                continue
            gap = random.randint(gap_lo, gap_hi)
            total_h = h1 + gap + h2
            if total_h > h - 2:
                continue
            r0 = random.randint(1, h - total_h - 1)
            c0 = random.randint(1, w - rect_w - 1)
            box1 = (r0, c0, r0 + h1 - 1, c0 + rect_w - 1)
            box2 = (r0 + h1 + gap, c0, r0 + total_h - 1, c0 + rect_w - 1)
            hull = (r0, c0, r0 + total_h - 1, c0 + rect_w - 1)
            fill_box = (r0 + h1, c0, r0 + h1 + gap - 1, c0 + rect_w - 1)

        if not _reserve_ok(reserved, hull):
            continue

        _paint_rect(grid, box1, color)
        _paint_rect(grid, box2, color)
        _mark_reserved(reserved, hull)
        return {
            "color": int(color),
            "boxes": (box1, box2),
            "gap": int(gap),
            "threshold": int(threshold),
            "should_fill": branch == "fill",
            "fill_box": fill_box,
        }

    return None


def _try_place_single(grid: np.ndarray, reserved: np.ndarray, color: int) -> bool:
    h, w = grid.shape

    for _ in range(SINGLE_PLACEMENT_ATTEMPTS):
        rect_h = random.randint(RECT_SHORT_MIN, min(RECT_SHORT_MAX, h - 2))
        rect_w = random.randint(RECT_SHORT_MIN, min(RECT_LONG_MAX, w - 2))
        r0 = random.randint(1, h - rect_h - 1)
        c0 = random.randint(1, w - rect_w - 1)
        box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if not _reserve_ok(reserved, box):
            continue
        _paint_rect(grid, box, color)
        _mark_reserved(reserved, box)
        return True

    return False


def _expected_output(grid: np.ndarray, pairs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        if not bool(pair["should_fill"]):
            continue
        r0, c0, r1, c1 = pair["fill_box"]
        out[r0 : r1 + 1, c0 : c1 + 1] = RED
    return out


def _pair_branches(pairs: list[dict[str, object]]) -> tuple[int, int]:
    fill = sum(1 for pair in pairs if bool(pair["should_fill"]))
    skip = len(pairs) - fill
    return fill, skip


def _sample_shape(n_pairs: int, n_singles: int) -> tuple[int, int]:
    min_side = min(GRID_MAX, max(GRID_MIN, 14 + 3 * n_pairs + n_singles))
    return random.randint(min_side, GRID_MAX), random.randint(min_side, GRID_MAX)


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_OPTIONS)
    n_singles = random.randint(0, min(SINGLE_COUNT_MAX, len(OBJECT_COLORS) - n_pairs))
    colors = random.sample(OBJECT_COLORS, n_pairs + n_singles)
    h, w = _sample_shape(n_pairs, n_singles)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    pair_specs = [
        ("h", "fill"),
        ("v", "skip"),
    ]
    while len(pair_specs) < n_pairs:
        pair_specs.append((random.choice(("h", "v")), random.choice(("fill", "skip"))))
    random.shuffle(pair_specs)

    pair_infos: list[dict[str, object]] = []
    for color, (orientation, branch) in zip(colors[:n_pairs], pair_specs, strict=False):
        pair = _try_place_pair(grid, reserved, int(color), orientation, branch)
        if pair is None:
            return None
        pair_infos.append(pair)

    for color in colors[n_pairs:]:
        if not _try_place_single(grid, reserved, int(color)):
            return None

    fill_count, skip_count = _pair_branches(pair_infos)
    if fill_count < 1 or skip_count < 1:
        return None

    expected = _expected_output(grid, pair_infos)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
