from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_ARM = 8
GEN_RETRIES = 500


@dataclass(frozen=True)
class _Cross:
    color: int
    cr: int
    cc: int
    up: int
    down: int
    left: int
    right: int

    @property
    def size(self) -> int:
        return self.up + self.down + self.left + self.right + 1

    @property
    def top(self) -> int:
        return self.cr - self.up

    @property
    def bottom(self) -> int:
        return self.cr + self.down

    @property
    def left_col(self) -> int:
        return self.cc - self.left

    @property
    def right_col(self) -> int:
        return self.cc + self.right


def _endpoints(cross: _Cross) -> tuple[tuple[int, int], ...]:
    return (
        (cross.cr - cross.up, cross.cc),
        (cross.cr + cross.down, cross.cc),
        (cross.cr, cross.cc - cross.left),
        (cross.cr, cross.cc + cross.right),
    )


def _cells(cross: _Cross) -> set[tuple[int, int]]:
    pts = {(r, cross.cc) for r in range(cross.top, cross.bottom + 1)}
    pts.update((cross.cr, c) for c in range(cross.left_col, cross.right_col + 1))
    return pts


def _decode_cross(color: int, points: list[tuple[int, int]]) -> _Cross:
    assert len(points) == 4, "each color must appear as exactly four arm tips"

    row_counts = Counter(r for r, _c in points)
    col_counts = Counter(c for _r, c in points)
    center_rows = [r for r, count in row_counts.items() if count == 2]
    center_cols = [c for c, count in col_counts.items() if count == 2]
    assert len(center_rows) == 1 and len(center_cols) == 1, "tips must define one repeated row and one repeated column"

    cr = int(center_rows[0])
    cc = int(center_cols[0])
    vertical = sorted(r for r, c in points if c == cc)
    horizontal = sorted(c for r, c in points if r == cr)
    assert len(vertical) == 2 and len(horizontal) == 2, "the repeated row/column must contain the two opposite tips"

    top, bottom = vertical
    left, right = horizontal
    assert top < cr < bottom, "vertical tips must lie above and below the implicit center"
    assert left < cc < right, "horizontal tips must lie left and right of the implicit center"

    return _Cross(
        color=int(color),
        cr=cr,
        cc=cc,
        up=cr - top,
        down=bottom - cr,
        left=cc - left,
        right=right - cc,
    )


def _decode_crosses(grid: np.ndarray) -> list[_Cross]:
    crosses: list[_Cross] = []
    colors = [int(c) for c in np.unique(grid) if int(c) != BLACK]
    assert colors, "expected at least one colored tip group"

    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        crosses.append(_decode_cross(color, coords))

    sizes = [cross.size for cross in crosses]
    assert len(set(sizes)) == len(sizes), "cross sizes must be unique for a total layer order"
    return crosses


def _render(crosses: list[_Cross], shape: tuple[int, int], *, descending: bool = True) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(crosses, key=lambda cross: cross.size, reverse=descending)
    for cross in ordered:
        out[cross.top : cross.bottom + 1, cross.cc] = cross.color
        out[cross.cr, cross.left_col : cross.right_col + 1] = cross.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore hidden crosses from same-color arm-tip pixels, layering smaller crosses over larger ones.

    Input structure:
    - The grid has a black background and several non-black color groups.
    - Each non-black color appears exactly four times as the tips of one hidden cross.
    - For each color, two tips share the center column and two other tips share the center row.
    - The resulting full crosses have pairwise distinct total pixel counts.

    Rule:
    - For each color, find the repeated row and repeated column among its four tips.
    - Treat their intersection as the implicit center and draw the full vertical and horizontal arms through the tips.
    - Draw the reconstructed crosses from largest total pixel count to smallest total pixel count.
    - Later, smaller crosses overwrite earlier, larger crosses at every overlapping pixel.

    Key insight:
    - The center is never given directly; it is the row-column intersection implied by the four residual arm tips, and overlap color is determined by cross size rather than by input order or color id.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    crosses = _decode_crosses(g)
    return _render(crosses, g.shape, descending=True)


def _random_arm(max_len: int) -> int:
    return random.randint(1, min(MAX_ARM, int(max_len)))


def _random_cross(color: int, H: int, W: int) -> _Cross:
    cr = random.randint(1, H - 2)
    cc = random.randint(1, W - 2)
    return _Cross(
        color=int(color),
        cr=cr,
        cc=cc,
        up=_random_arm(cr),
        down=_random_arm(H - 1 - cr),
        left=_random_arm(cc),
        right=_random_arm(W - 1 - cc),
    )


def _interior_cells(cross: _Cross, H: int, W: int) -> list[tuple[int, int]]:
    blocked = set(_endpoints(cross)) | {(cross.cr, cross.cc)}
    cells = [
        p
        for p in _cells(cross)
        if p not in blocked and 0 < p[0] < H - 1 and 0 < p[1] < W - 1
    ]
    random.shuffle(cells)
    return cells


def _cross_horizontal_through(color: int, H: int, W: int, r: int, c: int) -> _Cross | None:
    centers = [cc for cc in range(1, W - 1) if cc != c and abs(cc - c) + 1 <= MAX_ARM]
    random.shuffle(centers)
    for cc in centers:
        up = _random_arm(r)
        down = _random_arm(H - 1 - r)
        if c < cc:
            min_left = cc - c + 1
            if min_left > min(MAX_ARM, cc):
                continue
            left = random.randint(min_left, min(MAX_ARM, cc))
            right = _random_arm(W - 1 - cc)
        else:
            min_right = c - cc + 1
            if min_right > min(MAX_ARM, W - 1 - cc):
                continue
            left = _random_arm(cc)
            right = random.randint(min_right, min(MAX_ARM, W - 1 - cc))
        return _Cross(int(color), int(r), int(cc), int(up), int(down), int(left), int(right))
    return None


def _cross_vertical_through(color: int, H: int, W: int, r: int, c: int) -> _Cross | None:
    centers = [cr for cr in range(1, H - 1) if cr != r and abs(cr - r) + 1 <= MAX_ARM]
    random.shuffle(centers)
    for cr in centers:
        left = _random_arm(c)
        right = _random_arm(W - 1 - c)
        if r < cr:
            min_up = cr - r + 1
            if min_up > min(MAX_ARM, cr):
                continue
            up = random.randint(min_up, min(MAX_ARM, cr))
            down = _random_arm(H - 1 - cr)
        else:
            min_down = r - cr + 1
            if min_down > min(MAX_ARM, H - 1 - cr):
                continue
            up = _random_arm(cr)
            down = random.randint(min_down, min(MAX_ARM, H - 1 - cr))
        return _Cross(int(color), int(cr), int(c), int(up), int(down), int(left), int(right))
    return None


def _forced_pair(H: int, W: int, color_a: int, color_b: int) -> tuple[_Cross, _Cross] | None:
    for _ in range(100):
        first = _random_cross(color_a, H, W)
        options = _interior_cells(first, H, W)
        for r, c in options[:12]:
            if random.random() < 0.5:
                second = _cross_horizontal_through(color_b, H, W, r, c)
            else:
                second = _cross_vertical_through(color_b, H, W, r, c)
            if second is not None and first.size != second.size:
                return first, second
    return None


def _basic_geometry_ok(crosses: list[_Cross], shape: tuple[int, int]) -> bool:
    H, W = shape
    sizes = [cross.size for cross in crosses]
    if len(set(sizes)) != len(sizes):
        return False

    endpoints: list[tuple[int, int]] = []
    centers: list[tuple[int, int]] = []
    for cross in crosses:
        if not (0 <= cross.top <= cross.bottom < H and 0 <= cross.left_col <= cross.right_col < W):
            return False
        endpoints.extend(_endpoints(cross))
        centers.append((cross.cr, cross.cc))

    endpoint_set = set(endpoints)
    center_set = set(centers)
    return (
        len(endpoint_set) == len(endpoints)
        and len(center_set) == len(centers)
        and not (endpoint_set & center_set)
    )


def _make_input(crosses: list[_Cross], shape: tuple[int, int]) -> np.ndarray | None:
    if not _basic_geometry_ok(crosses, shape):
        return None

    grid = np.zeros(shape, dtype=int)
    for cross in crosses:
        for r, c in _endpoints(cross):
            if int(grid[r, c]) != BLACK:
                return None
            grid[r, c] = cross.color
    return grid


def _semantically_valid(inp: np.ndarray, crosses: list[_Cross], expected: np.ndarray) -> bool:
    if not _basic_geometry_ok(crosses, inp.shape):
        return False
    if not any(len({cross.up, cross.down, cross.left, cross.right}) > 1 for cross in crosses):
        return False

    endpoint_set = set()
    for cross in crosses:
        endpoint_set.update(_endpoints(cross))
        if int(inp[cross.cr, cross.cc]) != BLACK:
            return False

    cross_cells = [_cells(cross) for cross in crosses]
    overlap = set()
    meaningful_overlap = False
    for i in range(len(cross_cells)):
        for j in range(i + 1, len(cross_cells)):
            shared = cross_cells[i] & cross_cells[j]
            if shared:
                overlap.update(shared)
                if any(cell not in endpoint_set for cell in shared):
                    meaningful_overlap = True
    if not meaningful_overlap:
        return False

    total_cells = sum(len(cells) for cells in cross_cells)
    if len(overlap) > max(10, total_cells // 4):
        return False
    if np.count_nonzero(expected) > int(inp.size * 0.42):
        return False
    if np.array_equal(expected, inp):
        return False
    if np.array_equal(_render(crosses, inp.shape, descending=False), expected):
        return False
    for cross in crosses:
        if not np.any(expected == cross.color):
            return False

    decoded = _decode_crosses(inp)
    if sorted((c.color, c.cr, c.cc, c.up, c.down, c.left, c.right) for c in decoded) != sorted(
        (c.color, c.cr, c.cc, c.up, c.down, c.left, c.right) for c in crosses
    ):
        return False
    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(13, 30)
    W = random.randint(13, 30)
    count = random.choices([2, 3, 4, 5], weights=[2, 4, 4, 1], k=1)[0]
    colors = random.sample(PALETTE, count)

    pair = _forced_pair(H, W, colors[0], colors[1])
    if pair is None:
        return None
    crosses = [pair[0], pair[1]]

    for color in colors[2:]:
        placed = False
        for _ in range(80):
            candidate = _random_cross(color, H, W)
            if _basic_geometry_ok(crosses + [candidate], (H, W)):
                crosses.append(candidate)
                placed = True
                break
        if not placed:
            return None

    random.shuffle(crosses)
    inp = _make_input(crosses, (H, W))
    if inp is None:
        return None

    expected = _render(crosses, inp.shape, descending=True)
    if _semantically_valid(inp, crosses, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
