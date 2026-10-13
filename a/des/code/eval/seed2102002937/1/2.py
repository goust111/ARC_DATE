from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ANCHOR_SIDES = ("top", "left")
OBJECT_COUNT_RANGE = (2, 4)
GRID_SIDE_RANGE = (16, 22)
RECT_SIDE_CHOICES = [2, 3, 4, 5, 6]
MIN_RECT_AREA = 6
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400
SPLIT_ATTEMPTS = 250
LAYOUT_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    values = Counter(int(v) for v in np.asarray(grid).ravel())
    return int(values.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    normalized = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return normalized, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _candidate_dims(area: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        dims.append((int(h), int(w)))
        if h != w:
            dims.append((int(w), int(h)))
    dims.sort()
    return dims


def _solve_rectangle_dims(
    fragment_a: list[tuple[int, int]] | set[tuple[int, int]],
    fragment_b: list[tuple[int, int]] | set[tuple[int, int]],
) -> tuple[int, int]:
    shape_a, ha, wa = _normalize_coords(fragment_a)
    shape_b, hb, wb = _normalize_coords(fragment_b)
    total_area = len(shape_a) + len(shape_b)
    solutions: set[tuple[int, int]] = set()

    for H, W in _candidate_dims(total_area):
        if H < max(ha, hb) or W < max(wa, wb):
            continue

        full_cells = {(r, c) for r in range(H) for c in range(W)}
        for ra in range(H - ha + 1):
            for ca in range(W - wa + 1):
                placed_a = {(r + ra, c + ca) for r, c in shape_a}
                remainder = full_cells - placed_a
                if len(remainder) != len(shape_b):
                    continue
                rem_shape, _, _ = _normalize_coords(remainder)
                if rem_shape == shape_b:
                    solutions.add((int(H), int(W)))

    assert solutions, "no rectangle reconstruction found"
    assert len(solutions) == 1, f"ambiguous rectangle reconstruction: {solutions}"
    return next(iter(solutions))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, reconstruct the solid rectangle they complete, and move it to the same-color edge anchor.

    Input structure:
    - bg = black background. Each non-background color appears in exactly two irregular fragments plus one edge anchor.
    - The two same-color fragments can be translated (without rotation or reflection) to tile a solid rectangle.
    - A single-pixel anchor of that color lies on the top edge or left edge and marks the rectangle's output top-left corner.

    Rule:
    - For each color, find the anchor and the two fragments.
    - Recover the unique rectangle size that the fragments complementarily fill.
    - Draw that solid rectangle with its top-left corner at the anchor position.
    - Clear the scattered fragments; the output contains only the relocated rectangles.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _get_bg(g)
    out = np.full((H, W), bg, dtype=int)

    colors = sorted(int(v) for v in np.unique(g) if int(v) != int(bg))
    assert colors, "expected at least one non-background color"

    for color in colors:
        comps = _components_for_color(g, color)
        anchors = [comp for comp in comps if len(comp) == 1 and (comp[0][0] == 0 or comp[0][1] == 0)]
        assert len(anchors) == 1, f"expected one edge anchor for color {color}"
        anchor = anchors[0][0]
        fragments = [comp for comp in comps if comp is not anchors[0]]
        assert len(fragments) == 2, f"expected two fragments for color {color}"

        rect_h, rect_w = _solve_rectangle_dims(fragments[0], fragments[1])
        r0, c0 = anchor
        assert r0 + rect_h <= H and c0 + rect_w <= W, "anchored rectangle must fit in bounds"
        patch = out[r0 : r0 + rect_h, c0 : c0 + rect_w]
        assert np.all(patch == bg), "anchored rectangles must not overlap"
        patch[:, :] = color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return False

    H, W = mask.shape
    start = tuple(map(int, np.argwhere(mask)[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _grow_connected_subset(h: int, w: int, target_area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}

    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _is_full_rectangle(coords: set[tuple[int, int]]) -> bool:
    _, h, w = _normalize_coords(coords)
    return len(coords) == h * w


def _sample_split(h: int, w: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    full = {(r, c) for r in range(h) for c in range(w)}
    area = h * w

    for _ in range(SPLIT_ATTEMPTS):
        left_area = random.randint(2, area - 2)
        piece_a = _grow_connected_subset(h, w, left_area)
        if piece_a is None:
            continue
        piece_b = full - piece_a
        if not _mask_connected(np.array([[((r, c) in piece_b) for c in range(w)] for r in range(h)], dtype=bool)):
            continue
        if not (_mask_connected(np.array([[((r, c) in piece_a) for c in range(w)] for r in range(h)], dtype=bool))):
            continue
        if _is_full_rectangle(piece_a) and _is_full_rectangle(piece_b):
            continue

        shape_a, _, _ = _normalize_coords(piece_a)
        shape_b, _, _ = _normalize_coords(piece_b)
        try:
            solved = _solve_rectangle_dims(shape_a, shape_b)
        except AssertionError:
            continue
        if solved != (h, w):
            continue
        return shape_a, shape_b

    return None


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> None:
    H, W = blocked.shape
    rr0 = max(0, int(r0) - padding)
    cc0 = max(0, int(c0) - padding)
    rr1 = min(H, int(r0 + h + padding))
    cc1 = min(W, int(c0 + w + padding))
    blocked[rr0:rr1, cc0:cc1] = True


def _can_place_shape(blocked: np.ndarray, shape: frozenset[tuple[int, int]], r0: int, c0: int) -> bool:
    H, W = blocked.shape
    for dr, dc in shape:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if rr < 0 or cc < 0 or rr >= H or cc >= W or bool(blocked[rr, cc]):
            return False
    return True


def _paint_shape(grid: np.ndarray, shape: frozenset[tuple[int, int]], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _mark_shape(blocked: np.ndarray, shape: frozenset[tuple[int, int]], r0: int, c0: int, padding: int) -> None:
    H, W = blocked.shape
    for dr, dc in shape:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        rr0 = max(0, rr - padding)
        cc0 = max(0, cc - padding)
        rr1 = min(H, rr + padding + 1)
        cc1 = min(W, cc + padding + 1)
        blocked[rr0:rr1, cc0:cc1] = True


def _choose_rect_dims(limit_h: int, limit_w: int) -> tuple[int, int]:
    candidates = [
        (h, w)
        for h in RECT_SIDE_CHOICES
        for w in RECT_SIDE_CHOICES
        if h <= limit_h and w <= limit_w and h * w >= MIN_RECT_AREA
    ]
    return random.choice(candidates)


def _place_output_specs(H: int, W: int, colors: list[int]) -> list[dict] | None:
    blocked = np.zeros((H, W), dtype=bool)
    specs: list[dict] = []

    for color in colors:
        side = random.choice(ANCHOR_SIDES)
        if side == "top":
            h, w = _choose_rect_dims(H, W - 1)
            anchors = [(0, c) for c in range(1, W - w + 1)]
        else:
            h, w = _choose_rect_dims(H - 1, W)
            anchors = [(r, 0) for r in range(1, H - h + 1)]

        random.shuffle(anchors)
        placed = False
        for r0, c0 in anchors:
            patch = blocked[r0 : r0 + h, c0 : c0 + w]
            if bool(patch.any()):
                continue
            specs.append({"color": int(color), "anchor": (int(r0), int(c0)), "dims": (int(h), int(w)), "side": side})
            _mark_rect(blocked, r0, c0, h, w, PLACEMENT_PADDING)
            placed = True
            break
        if not placed:
            return None

    return specs


def _place_fragments(H: int, W: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    for spec in specs:
        color = int(spec["color"])
        r0, c0 = spec["anchor"]
        h, w = spec["dims"]
        inp[r0, c0] = color
        expected[r0 : r0 + h, c0 : c0 + w] = color
        _mark_rect(blocked, r0, c0, h, w, PLACEMENT_PADDING)

    fragments: list[tuple[int, frozenset[tuple[int, int]]]] = []
    for spec in specs:
        h, w = spec["dims"]
        split = _sample_split(h, w)
        if split is None:
            return None
        a, b = split
        if random.random() < 0.5:
            a, b = b, a
        fragments.append((int(spec["color"]), a))
        fragments.append((int(spec["color"]), b))

    random.shuffle(fragments)
    blocked[0, :] = True
    blocked[:, 0] = True

    for color, shape in fragments:
        _, sh, sw = _normalize_coords(shape)
        candidates = [(r, c) for r in range(1, H - sh + 1) for c in range(1, W - sw + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if not _can_place_shape(blocked, shape, r0, c0):
                continue
            _paint_shape(inp, shape, r0, c0, color)
            _mark_shape(blocked, shape, r0, c0, PLACEMENT_PADDING)
            placed = True
            break
        if not placed:
            return None

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, n_objects)

    specs = None
    for _ in range(LAYOUT_ATTEMPTS):
        specs = _place_output_specs(H, W, colors)
        if specs is not None:
            break
    if specs is None:
        return None

    placed = _place_fragments(H, W, specs)
    if placed is None:
        return None
    inp, expected = placed

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
