from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
CYAN = 8
PROBE_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIRS = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _edge_dirs(shape: tuple[int, int], r: int, c: int, color: int, g: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape
    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    if len(dirs) <= 1:
        return dirs

    continuing = []
    for dr, dc in dirs:
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc) and int(g[rr, cc]) == color:
            continuing.append((dr, dc))
    assert len(continuing) == 1, "corner emitters must have one visible inward ray direction"
    return continuing


def _border_cells(shape: tuple[int, int]):
    h, w = shape
    for c in range(w):
        yield 0, c
        if h > 1:
            yield h - 1, c
    for r in range(1, h - 1):
        yield r, 0
        if w > 1:
            yield r, w - 1


def _ray_hits(g: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    hits: dict[tuple[int, int], set[tuple[int, int]]] = {
        (1, 0): set(),
        (-1, 0): set(),
        (0, 1): set(),
        (0, -1): set(),
    }
    for r, c in _border_cells(g.shape):
        color = int(g[r, c])
        if color in (BG, GREY):
            continue

        dirs = _edge_dirs(g.shape, r, c, color, g)
        assert dirs, "colored ray cells must lie on an edge"
        for dr, dc in dirs:
            rr, cc = r, c
            while _in_bounds(g.shape, rr + dr, cc + dc) and int(g[rr + dr, cc + dc]) == color:
                rr += dr
                cc += dc

            ar, ac = rr + dr, cc + dc
            if not _in_bounds(g.shape, ar, ac):
                continue
            ahead = int(g[ar, ac])
            if ahead == GREY:
                continue
            assert ahead == BG, "a stopped probe must face either a grey block or empty hidden-frame cell"
            hits[(dr, dc)].add((ar, ac))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the invisible rectangular frame measured by edge probe rays.

    Input structure:
    - The background is black, with scattered solid grey blocks inside the canvas.
    - Colored edge emitters draw straight same-color rays inward along their row
      or column.
    - A ray may pass through to the opposite edge, stop just before a grey block,
      or stop just outside an unseen axis-aligned rectangular frame.

    Rule:
    - For every edge ray, find its visible terminal cell and inspect the next
      cell in the ray direction.
    - Terminals whose next cell is grey are visible-block decoys, and terminals
      with no next cell are pass-through rays; ignore both.
    - Push every remaining terminal one cell inward to get points on the hidden
      frame, then draw the uniquely determined one-cell-thick cyan rectangle.

    Key insight:
    - The useful probes are exactly the rays that stop before empty space; their
      apparent stop is outside the invisible object, so the real frame boundary
      is one step farther along the ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    hits = _ray_hits(g)
    top = hits[(1, 0)]
    bottom = hits[(-1, 0)]
    left = hits[(0, 1)]
    right = hits[(0, -1)]
    assert all(len(side) >= 2 for side in (top, bottom, left, right)), "each frame side needs at least two probe hits"

    top_rows = {r for r, _c in top}
    bottom_rows = {r for r, _c in bottom}
    left_cols = {c for _r, c in left}
    right_cols = {c for _r, c in right}
    assert len(top_rows) == len(bottom_rows) == len(left_cols) == len(right_cols) == 1

    r0 = top_rows.pop()
    r1 = bottom_rows.pop()
    c0 = left_cols.pop()
    c1 = right_cols.pop()
    assert r0 < r1 and c0 < c1, "probe hits must define a non-degenerate rectangle"

    assert all(r == r0 and c0 <= c <= c1 for r, c in top)
    assert all(r == r1 and c0 <= c <= c1 for r, c in bottom)
    assert all(c == c0 and r0 <= r <= r1 for r, c in left)
    assert all(c == c1 and r0 <= r <= r1 for r, c in right)

    border = np.zeros(g.shape, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    assert np.all(g[border] == BG), "the hidden frame boundary must be visually empty in the input"

    out = g.copy()
    out[border] = CYAN
    return out


def _frame_border(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _paint_ray(grid: np.ndarray, start: tuple[int, int], stop: tuple[int, int], color: int) -> bool:
    sr, sc = start
    er, ec = stop
    dr = 0 if sr == er else (1 if er > sr else -1)
    dc = 0 if sc == ec else (1 if ec > sc else -1)
    if (dr == 0) == (dc == 0):
        return False

    cells = []
    r, c = sr, sc
    while True:
        if not _in_bounds(grid.shape, r, c):
            return False
        value = int(grid[r, c])
        if value not in (BG, color):
            return False
        cells.append((r, c))
        if (r, c) == (er, ec):
            break
        r += dr
        c += dc

    for r, c in cells:
        grid[r, c] = color
    return True


def _ray_spec_for_side(
    side: str, coord: int, h: int, w: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (0, coord), (r0 - 1, coord), (r0, coord)
    if side == "bottom":
        return (h - 1, coord), (r1 + 1, coord), (r1, coord)
    if side == "left":
        return (coord, 0), (coord, c0 - 1), (coord, c0)
    return (coord, w - 1), (coord, c1 + 1), (coord, c1)


def _choose_side_points(lo: int, hi: int, count: int) -> list[int]:
    values = list(range(lo, hi + 1))
    if len(values) <= count:
        return values
    points = {lo, hi} if random.random() < 0.35 and count >= 2 else set()
    while len(points) < count:
        points.add(random.choice(values))
    return sorted(points)


def _place_hidden_probes(
    grid: np.ndarray, color: int, h: int, w: int, r0: int, c0: int, r1: int, c1: int
) -> dict[str, list[int]] | None:
    side_points = {
        "top": _choose_side_points(c0, c1, random.randint(2, min(4, c1 - c0 + 1))),
        "bottom": _choose_side_points(c0, c1, random.randint(2, min(4, c1 - c0 + 1))),
        "left": _choose_side_points(r0, r1, random.randint(2, min(4, r1 - r0 + 1))),
        "right": _choose_side_points(r0, r1, random.randint(2, min(4, r1 - r0 + 1))),
    }
    for side, coords in side_points.items():
        for coord in coords:
            start, stop, boundary = _ray_spec_for_side(side, coord, h, w, r0, c0, r1, c1)
            if int(grid[boundary]) != BG:
                return None
            if not _paint_ray(grid, start, stop, color):
                return None
    return side_points


def _try_place_pass_through(
    grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int
) -> bool:
    h, w = grid.shape
    choices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in list(range(1, c0)) + list(range(c1 + 1, w - 1)):
        choices.append(((0, c), (h - 1, c)))
    for r in list(range(1, r0)) + list(range(r1 + 1, h - 1)):
        choices.append(((r, 0), (r, w - 1)))
    random.shuffle(choices)
    return any(_paint_ray(grid, start, stop, color) for start, stop in choices[:40])


def _block_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _can_place_block(grid: np.ndarray, cells: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if (r, c) in reserved or int(grid[r, c]) != BG:
            return False
    return True


def _bait_candidates(
    grid: np.ndarray, reserved: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int
) -> list[tuple[str, int, int, int, int]]:
    h, w = grid.shape
    candidates = []
    for bh in (1, 2, 3):
        for bw in (1, 2, 3):
            if bh * bw > 5:
                continue
            for top in range(1, h - bh):
                for left in range(1, w - bw):
                    cells = _block_cells(top, left, bh, bw)
                    if not _can_place_block(grid, cells, reserved):
                        continue
                    if top > 1:
                        for c in range(left, left + bw):
                            if top <= r0 or c < c0 or c > c1:
                                candidates.append(("top", top, left, bh, bw))
                                break
                    if top + bh < h - 1:
                        for c in range(left, left + bw):
                            if top + bh - 1 >= r1 or c < c0 or c > c1:
                                candidates.append(("bottom", top, left, bh, bw))
                                break
                    if left > 1:
                        for r in range(top, top + bh):
                            if left <= c0 or r < r0 or r > r1:
                                candidates.append(("left", top, left, bh, bw))
                                break
                    if left + bw < w - 1:
                        for r in range(top, top + bh):
                            if left + bw - 1 >= c1 or r < r0 or r > r1:
                                candidates.append(("right", top, left, bh, bw))
                                break
    random.shuffle(candidates)
    return candidates


def _paint_bait(
    grid: np.ndarray,
    color: int,
    side: str,
    top: int,
    left: int,
    bh: int,
    bw: int,
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    if side == "top":
        c = random.randrange(left, left + bw)
        start, stop = (0, c), (top - 1, c)
    elif side == "bottom":
        c = random.randrange(left, left + bw)
        start, stop = (h - 1, c), (top + bh, c)
    elif side == "left":
        r = random.randrange(top, top + bh)
        start, stop = (r, 0), (r, left - 1)
    else:
        r = random.randrange(top, top + bh)
        start, stop = (r, w - 1), (r, left + bw)

    ray_cells = set()
    sr, sc = start
    er, ec = stop
    dr = 0 if sr == er else (1 if er > sr else -1)
    dc = 0 if sc == ec else (1 if ec > sc else -1)
    r, c = sr, sc
    while True:
        ray_cells.add((r, c))
        if (r, c) == (er, ec):
            break
        r += dr
        c += dc
    if ray_cells & reserved:
        return False
    if not _paint_ray(grid, start, stop, color):
        return False
    for r, c in _block_cells(top, left, bh, bw):
        grid[r, c] = GREY
    return True


def _place_baits(
    grid: np.ndarray, color: int, reserved: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int
) -> int:
    placed = 0
    used_boxes: set[tuple[int, int, int, int]] = set()
    for side, top, left, bh, bw in _bait_candidates(grid, reserved, r0, c0, r1, c1):
        box = (top, left, bh, bw)
        if box in used_boxes:
            continue
        snapshot = grid.copy()
        if _paint_bait(grid, color, side, top, left, bh, bw, reserved):
            placed += 1
            used_boxes.add(box)
            if placed >= 2:
                break
        else:
            grid[:, :] = snapshot
    return placed


def _place_distractor_blocks(grid: np.ndarray, reserved: set[tuple[int, int]], count: int) -> int:
    placed = 0
    h, w = grid.shape
    for _ in range(80):
        if placed >= count:
            break
        bh = random.choice((1, 1, 2, 2, 3))
        bw = random.choice((1, 2, 2, 3))
        if bh * bw > 5:
            continue
        top = random.randint(1, h - bh - 1)
        left = random.randint(1, w - bw - 1)
        cells = _block_cells(top, left, bh, bw)
        if _can_place_block(grid, cells, reserved):
            for r, c in cells:
                grid[r, c] = GREY
            placed += 1
    return placed


def _expected_with_frame(inp: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    out = inp.copy()
    out[r0, c0 : c1 + 1] = CYAN
    out[r1, c0 : c1 + 1] = CYAN
    out[r0 : r1 + 1, c0] = CYAN
    out[r0 : r1 + 1, c1] = CYAN
    return out


def _valid_semantics(inp: np.ndarray, expected: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    hits = _ray_hits(inp)
    if any(len(points) < 2 for points in hits.values()):
        return False
    decoy_count = 0
    passthrough_count = 0
    for r, c in _border_cells(inp.shape):
        color = int(inp[r, c])
        if color in (BG, GREY):
            continue
        for dr, dc in _edge_dirs(inp.shape, r, c, color, inp):
            rr, cc = r, c
            while _in_bounds(inp.shape, rr + dr, cc + dc) and int(inp[rr + dr, cc + dc]) == color:
                rr += dr
                cc += dc
            ar, ac = rr + dr, cc + dc
            if not _in_bounds(inp.shape, ar, ac):
                passthrough_count += 1
            elif int(inp[ar, ac]) == GREY:
                pushed = (ar, ac)
                if not (pushed[0] in (r0, r1) and c0 <= pushed[1] <= c1) and not (
                    pushed[1] in (c0, c1) and r0 <= pushed[0] <= r1
                ):
                    decoy_count += 1
    if decoy_count < 1 or passthrough_count < 1:
        return False
    return bool(np.any(inp != expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 30)
    w = random.randint(17, 30)
    fh = random.randint(6, min(13, h - 7))
    fw = random.randint(6, min(13, w - 7))
    r0 = random.randint(3, h - fh - 3)
    c0 = random.randint(3, w - fw - 3)
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1

    color = random.choice(PROBE_COLORS)
    grid = np.full((h, w), BG, dtype=int)
    reserved = _frame_border(grid.shape, r0, c0, r1, c1)

    if _place_hidden_probes(grid, color, h, w, r0, c0, r1, c1) is None:
        return None
    if not _try_place_pass_through(grid, color, r0, c0, r1, c1):
        return None
    if _place_baits(grid, color, reserved, r0, c0, r1, c1) < 2:
        return None
    _place_distractor_blocks(grid, reserved, random.randint(1, 3))

    expected = _expected_with_frame(grid, r0, c0, r1, c1)
    if _valid_semantics(grid, expected, r0, c0, r1, c1):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-frame probe input")
