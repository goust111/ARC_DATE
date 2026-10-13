import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

BACKGROUND_COLORS = (AZURE, GREEN, MAGENTA, YELLOW)
BACKGROUND_PAIRS = ((AZURE, YELLOW), (GREEN, MAGENTA))
PAIR_OF = {
    AZURE: YELLOW,
    YELLOW: AZURE,
    GREEN: MAGENTA,
    MAGENTA: GREEN,
}
SHAPE_COLORS = (BLUE, RED, GREY)

QUADRANT_IDS = ("tl", "tr", "bl", "br")
QUADRANT_DIM_RANGE = (6, 11)
SHAPE_SIZE_RANGE = (3, 5)
MASK_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        rows = np.where(np.all(grid == color, axis=1))[0]
        cols = np.where(np.all(grid == color, axis=0))[0]
        if len(rows) == 1 and len(cols) == 1:
            candidates.append((color, int(rows[0]), int(cols[0])))
    assert len(candidates) == 1, "expected one full-row/full-column cross"
    return candidates[0]


def _quadrant_bounds(shape: tuple[int, int], axis_r: int, axis_c: int) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape
    bounds = {
        "tl": (0, axis_r, 0, axis_c),
        "tr": (0, axis_r, axis_c + 1, w),
        "bl": (axis_r + 1, h, 0, axis_c),
        "br": (axis_r + 1, h, axis_c + 1, w),
    }
    for r0, r1, c0, c1 in bounds.values():
        assert r0 < r1 and c0 < c1, "cross must leave four non-empty quadrants"
    return bounds


def _mode_color(block: np.ndarray) -> int:
    colors, counts = np.unique(block, return_counts=True)
    return int(colors[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (int(nr), int(nc))
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == len(cells)


def _extract_shape(block: np.ndarray, bg: int) -> dict:
    mask = block != bg
    assert _is_connected(mask), "shape pixels must form one connected component"
    coords = np.argwhere(mask)
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    rel = [tuple(map(int, cell)) for cell in np.argwhere(cropped)]
    return {
        "kind": "shape",
        "rel": rel,
        "height": int(cropped.shape[0]),
        "width": int(cropped.shape[1]),
    }


def _pair_quadrants(quadrants: list[dict]) -> list[tuple[dict, dict]]:
    by_bg: dict[int, list[dict]] = {}
    for info in quadrants:
        by_bg.setdefault(int(info["bg"]), []).append(info)

    pairs: list[tuple[dict, dict]] = []

    # This supports the literal "same-background" reading of the prompt.
    if len(by_bg) == 2 and all(len(members) == 2 for members in by_bg.values()):
        for members in by_bg.values():
            shape_members = [member for member in members if member["kind"] == "shape"]
            anchor_members = [member for member in members if member["kind"] == "anchor"]
            assert len(shape_members) == len(anchor_members) == 1
            pairs.append((shape_members[0], anchor_members[0]))
        return pairs

    # The generated instances use four distinct quadrant fills arranged into two fixed background pairs.
    assert set(by_bg) == set(BACKGROUND_COLORS), "unexpected quadrant background colors"
    used: set[int] = set()
    for color_a, color_b in BACKGROUND_PAIRS:
        assert color_a not in used and color_b not in used
        first = by_bg[color_a][0]
        second = by_bg[color_b][0]
        shape_members = [member for member in (first, second) if member["kind"] == "shape"]
        anchor_members = [member for member in (first, second) if member["kind"] == "anchor"]
        assert len(shape_members) == len(anchor_members) == 1
        pairs.append((shape_members[0], anchor_members[0]))
        used.update((color_a, color_b))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use quadrant background pairing to choose which source shape is stamped onto which anchor quadrant.

    The grid is split by one full-row/full-column cross. Two quadrants contain a connected
    shape, two contain a single-pixel anchor. Each source shape is copied to its paired
    anchor quadrant, recolored to the source quadrant's background fill, and placed with
    the anchor as the copied shape's top-left reference point.
    """
    g = np.asarray(grid, dtype=int)
    cross_color, axis_r, axis_c = _find_cross(g)
    bounds = _quadrant_bounds(g.shape, axis_r, axis_c)

    quadrants: list[dict] = []
    for qid in QUADRANT_IDS:
        r0, r1, c0, c1 = bounds[qid]
        block = g[r0:r1, c0:c1]
        bg = _mode_color(block)
        fg = np.argwhere(block != bg)
        assert len(fg) > 0, "each quadrant must contain one marker or one shape"

        info = {"id": qid, "bg": int(bg), "bounds": (r0, r1, c0, c1)}
        if len(fg) == 1:
            fr, fc = map(int, fg[0])
            info["kind"] = "anchor"
            info["anchor"] = (r0 + fr, c0 + fc)
        else:
            info.update(_extract_shape(block, bg))
        quadrants.append(info)

    out = g.copy()
    for source, target in _pair_quadrants(quadrants):
        anchor_r, anchor_c = target["anchor"]
        tr0, tr1, tc0, tc1 = target["bounds"]

        out[anchor_r, anchor_c] = target["bg"]
        for dr, dc in source["rel"]:
            rr = anchor_r + int(dr)
            cc = anchor_c + int(dc)
            assert tr0 <= rr < tr1 and tc0 <= cc < tc1, "stamped shape must stay inside its target quadrant"
            current = int(out[rr, cc])
            assert current in (target["bg"], MAROON, cross_color), "target quadrant must be otherwise empty"
            out[rr, cc] = source["bg"]

    return out


def _random_path(h: int, w: int) -> set[tuple[int, int]]:
    moves = ["D"] * (h - 1) + ["R"] * (w - 1)
    random.shuffle(moves)
    r = c = 0
    cells = {(0, 0)}
    for move in moves:
        if move == "D":
            r += 1
        else:
            c += 1
        cells.add((r, c))
    return cells


def _make_shape_mask(h: int, w: int) -> np.ndarray | None:
    path_area = h + w - 1
    upper = min(h * w - 1, max(path_area, (3 * h * w) // 4))
    if upper < path_area:
        return None

    for _ in range(MASK_ATTEMPTS):
        cells = set(_random_path(h, w))
        target_area = random.randint(path_area, upper)

        while len(cells) < target_area:
            frontier = []
            frontier_seen = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in frontier_seen:
                        frontier_seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask[0, 0] and mask[h - 1, w - 1] and _is_connected(mask):
            return mask
    return None


def _fill_quadrants(grid: np.ndarray, bounds: dict[str, tuple[int, int, int, int]], colors: dict[str, int]) -> None:
    for qid, color in colors.items():
        r0, r1, c0, c1 = bounds[qid]
        grid[r0:r1, c0:c1] = int(color)


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    region = grid[top : top + h, left : left + w]
    region[mask] = int(color)


def _oracle_matches(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(inp), expected))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(*QUADRANT_DIM_RANGE)
    bottom_h = random.randint(*QUADRANT_DIM_RANGE)
    left_w = random.randint(*QUADRANT_DIM_RANGE)
    right_w = random.randint(*QUADRANT_DIM_RANGE)

    axis_r = top_h
    axis_c = left_w
    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w

    grid = np.full((h, w), BLACK, dtype=int)
    expected = np.full((h, w), BLACK, dtype=int)
    bounds = _quadrant_bounds((h, w), axis_r, axis_c)

    quadrant_colors = dict(zip(QUADRANT_IDS, random.sample(BACKGROUND_COLORS, len(BACKGROUND_COLORS))))
    _fill_quadrants(grid, bounds, quadrant_colors)
    _fill_quadrants(expected, bounds, quadrant_colors)
    grid[axis_r, :] = ORANGE
    grid[:, axis_c] = ORANGE
    expected[axis_r, :] = ORANGE
    expected[:, axis_c] = ORANGE

    quadrant_of_color = {color: qid for qid, color in quadrant_colors.items()}
    for color_a, color_b in BACKGROUND_PAIRS:
        q_a = quadrant_of_color[color_a]
        q_b = quadrant_of_color[color_b]
        source_q, target_q = (q_a, q_b) if random.random() < 0.5 else (q_b, q_a)

        sr0, sr1, sc0, sc1 = bounds[source_q]
        tr0, tr1, tc0, tc1 = bounds[target_q]
        max_h = min(SHAPE_SIZE_RANGE[1], sr1 - sr0, tr1 - tr0)
        max_w = min(SHAPE_SIZE_RANGE[1], sc1 - sc0, tc1 - tc0)
        if max_h < SHAPE_SIZE_RANGE[0] or max_w < SHAPE_SIZE_RANGE[0]:
            return None

        shape_h = random.randint(SHAPE_SIZE_RANGE[0], max_h)
        shape_w = random.randint(SHAPE_SIZE_RANGE[0], max_w)
        mask = _make_shape_mask(shape_h, shape_w)
        if mask is None:
            return None

        src_top = random.randint(sr0, sr1 - shape_h)
        src_left = random.randint(sc0, sc1 - shape_w)
        anchor_r = random.randint(tr0, tr1 - shape_h)
        anchor_c = random.randint(tc0, tc1 - shape_w)

        shape_color = random.choice(SHAPE_COLORS)
        _paint_mask(grid, src_top, src_left, mask, shape_color)
        _paint_mask(expected, src_top, src_left, mask, shape_color)
        grid[anchor_r, anchor_c] = MAROON

        expected[anchor_r, anchor_c] = quadrant_colors[target_q]
        _paint_mask(expected, anchor_r, anchor_c, mask, quadrant_colors[source_q])

    return grid if _oracle_matches(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
