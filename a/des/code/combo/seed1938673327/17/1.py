import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIZE_RANGE = (12, 18)
MIN_SPAN = 2
MAX_EXTRA_PAIRS = 2
MAX_ATTEMPTS = 300
PAIR_TYPES = ("hs", "vs", "tb", "lr")


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _classify_pair(a: tuple[int, int], b: tuple[int, int], h: int, w: int) -> str:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2 and r1 in (0, h - 1):
        return "hs"
    if c1 == c2 and c1 in (0, w - 1):
        return "vs"
    if {r1, r2} == {0, h - 1}:
        return "tb"
    if {c1, c2} == {0, w - 1}:
        return "lr"
    raise AssertionError("each color pair must lie on one edge or on opposite edges")


def _rectangle_sort_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    first, second = sorted((a, b))
    return (first[1], first[0], second[1], second[0])


def _bbox_from_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill between same-color edge markers.

    Same-edge pairs draw a border strip on that row or column.
    Opposite-edge pairs draw a rectangle whose corners are the two markers.
    Strips are painted first, then rectangles are painted from left to right by
    the column of the pair's first row-major marker.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    markers: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != bg):
        rr, cc = int(r), int(c)
        assert rr in (0, h - 1) or cc in (0, w - 1), "all markers must lie on the border"
        color = int(g[rr, cc])
        markers.setdefault(color, []).append((rr, cc))

    assert markers, "expected at least one color pair"
    out = g.copy()
    rectangles: list[tuple[tuple[int, int, int, int], int, int, int, int, int]] = []

    for color, pts in markers.items():
        assert len(pts) == 2, "each color must appear exactly twice"
        a, b = pts
        kind = _classify_pair(a, b, h, w)

        if kind == "hs":
            r = a[0]
            c0, c1 = sorted((a[1], b[1]))
            out[r, c0 : c1 + 1] = color
        elif kind == "vs":
            c = a[1]
            r0, r1 = sorted((a[0], b[0]))
            out[r0 : r1 + 1, c] = color
        else:
            r0, c0, r1, c1 = _bbox_from_pair(a, b)
            rectangles.append((_rectangle_sort_key(a, b), r0, c0, r1, c1, color))

    rectangles.sort(key=lambda item: item[0])
    for _key, r0, c0, r1, c1, color in rectangles:
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _gap_pairs(values: set[int]) -> list[tuple[int, int]]:
    ordered = sorted(values)
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b - a >= MIN_SPAN:
                pairs.append((a, b))
    return pairs


def _pick_same_edge_pair(free_values: set[int]) -> tuple[int, int] | None:
    candidates = _gap_pairs(free_values)
    if not candidates:
        return None
    a, b = random.choice(candidates)
    free_values.remove(a)
    free_values.remove(b)
    return a, b


def _pick_tb_pair(
    free_top: set[int], free_bottom: set[int], h: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates = [(t, b) for t in free_top for b in free_bottom if abs(t - b) >= MIN_SPAN]
    if not candidates:
        return None
    top_col, bottom_col = random.choice(candidates)
    free_top.remove(top_col)
    free_bottom.remove(bottom_col)
    return (0, top_col), (h - 1, bottom_col)


def _pick_lr_pair(
    free_left: set[int], free_right: set[int], w: int, first_side: str
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[int, int]] = []
    for left_row in free_left:
        for right_row in free_right:
            if abs(left_row - right_row) < MIN_SPAN:
                continue
            if first_side == "left" and left_row < right_row:
                candidates.append((left_row, right_row))
            if first_side == "right" and right_row < left_row:
                candidates.append((left_row, right_row))
    if not candidates:
        return None
    left_row, right_row = random.choice(candidates)
    free_left.remove(left_row)
    free_right.remove(right_row)
    return (left_row, 0), (right_row, w - 1)


def _branch_counts() -> dict[str, int]:
    counts = {kind: 1 for kind in PAIR_TYPES}
    caps = {"hs": 2, "vs": 2, "tb": 2, "lr": 2}
    for _ in range(random.randint(0, MAX_EXTRA_PAIRS)):
        min_count = min(counts.values())
        choices = [kind for kind in PAIR_TYPES if counts[kind] == min_count and counts[kind] < caps[kind]]
        counts[random.choice(choices)] += 1
    return counts


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _validate_generated_grid(grid: np.ndarray) -> bool:
    h, w = grid.shape
    bg = _bg_color(grid)
    if bg != BG:
        return False

    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    if not (4 <= len(colors) <= 6):
        return False

    kinds: list[str] = []
    rectangles: list[tuple[int, int, int, int]] = []
    rect_keys: list[int] = []
    for color in colors:
        pts = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        if len(pts) != 2:
            return False
        if not all(r in (0, h - 1) or c in (0, w - 1) for r, c in pts):
            return False
        kind = _classify_pair(pts[0], pts[1], h, w)
        kinds.append(kind)
        if kind in ("tb", "lr"):
            rectangles.append(_bbox_from_pair(pts[0], pts[1]))
            rect_keys.append(_rectangle_sort_key(pts[0], pts[1])[0])

    if set(PAIR_TYPES) - set(kinds):
        return False
    if len(rectangles) < 2:
        return False
    if len(rect_keys) != len(set(rect_keys)):
        return False
    if not any(_boxes_overlap(rectangles[i], rectangles[j]) for i in range(len(rectangles)) for j in range(i + 1, len(rectangles))):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*SIZE_RANGE)
    w = random.randint(*SIZE_RANGE)
    counts = _branch_counts()

    free_top = set(range(1, w - 1))
    free_bottom = set(range(1, w - 1))
    free_left = set(range(1, h - 1))
    free_right = set(range(1, h - 1))
    pair_specs: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    for _ in range(counts["tb"]):
        pair = _pick_tb_pair(free_top, free_bottom, h)
        if pair is None:
            return None
        pair_specs.append(("tb", pair[0], pair[1]))

    if counts["lr"] == 1:
        first_side = random.choice(("left", "right"))
        pair = _pick_lr_pair(free_left, free_right, w, first_side)
        if pair is None:
            return None
        pair_specs.append(("lr", pair[0], pair[1]))
    else:
        for first_side in ("left", "right"):
            pair = _pick_lr_pair(free_left, free_right, w, first_side)
            if pair is None:
                return None
            pair_specs.append(("lr", pair[0], pair[1]))

    if counts["hs"] == 1:
        edge = random.choice(("top", "bottom"))
        if edge == "top":
            pair = _pick_same_edge_pair(free_top)
            if pair is None:
                return None
            pair_specs.append(("hs", (0, pair[0]), (0, pair[1])))
        else:
            pair = _pick_same_edge_pair(free_bottom)
            if pair is None:
                return None
            pair_specs.append(("hs", (h - 1, pair[0]), (h - 1, pair[1])))
    else:
        for edge in random.sample(["top", "bottom"], 2):
            if edge == "top":
                pair = _pick_same_edge_pair(free_top)
                if pair is None:
                    return None
                pair_specs.append(("hs", (0, pair[0]), (0, pair[1])))
            else:
                pair = _pick_same_edge_pair(free_bottom)
                if pair is None:
                    return None
                pair_specs.append(("hs", (h - 1, pair[0]), (h - 1, pair[1])))

    if counts["vs"] == 1:
        edge = random.choice(("left", "right"))
        if edge == "left":
            pair = _pick_same_edge_pair(free_left)
            if pair is None:
                return None
            pair_specs.append(("vs", (pair[0], 0), (pair[1], 0)))
        else:
            pair = _pick_same_edge_pair(free_right)
            if pair is None:
                return None
            pair_specs.append(("vs", (pair[0], w - 1), (pair[1], w - 1)))
    else:
        for edge in random.sample(["left", "right"], 2):
            if edge == "left":
                pair = _pick_same_edge_pair(free_left)
                if pair is None:
                    return None
                pair_specs.append(("vs", (pair[0], 0), (pair[1], 0)))
            else:
                pair = _pick_same_edge_pair(free_right)
                if pair is None:
                    return None
                pair_specs.append(("vs", (pair[0], w - 1), (pair[1], w - 1)))

    if len(pair_specs) > len(COLORS):
        return None

    colors = random.sample(COLORS, len(pair_specs))
    random.shuffle(pair_specs)
    grid = np.full((h, w), BG, dtype=int)
    for color, (_kind, a, b) in zip(colors, pair_specs, strict=False):
        grid[a] = color
        grid[b] = color

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
