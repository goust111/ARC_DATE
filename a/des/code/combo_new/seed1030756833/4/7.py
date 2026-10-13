from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
GRAY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
Coord = tuple[int, int]


@dataclass(frozen=True)
class Frame:
    color: int
    marker: int | None
    bbox: tuple[int, int, int, int]

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def area(self) -> int:
        return self.height * self.width


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter(r0: int, c0: int, r1: int, c1: int) -> set[Coord]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _components(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[set[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[Coord] = deque([(sr, sc)])
        comp: set[Coord] = set()
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for nr, nc in _neighbors4(r, c, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _is_isolated_anchor(g: np.ndarray, r: int, c: int) -> bool:
    h, w = g.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) != (r, c) and int(g[rr, cc]) != BLACK:
                return False
    return True


def _find_anchor(g: np.ndarray) -> Coord:
    anchors = [
        (int(r), int(c))
        for r, c in np.argwhere(g == GRAY)
        if _is_isolated_anchor(g, int(r), int(c))
    ]
    assert len(anchors) == 1, "expected exactly one isolated gray anchor pixel"
    return anchors[0]


def _is_side_midpoint(cell: Coord, bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    return ((r == r0 or r == r1) and c == mid_c) or (
        (c == c0 or c == c1) and r == mid_r
    )


def _parse_frame(g: np.ndarray, cells: set[Coord]) -> Frame:
    bbox = _bbox(cells)
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3, "fragments must be rectangular frames"
    assert height % 2 == 1 and width % 2 == 1, "frame centers must be single cells"

    border_cells = _perimeter(r0, c0, r1, c1)
    assert cells == border_cells, "each fragment must be a complete hollow rectangle"

    counts: dict[int, int] = {}
    positions: dict[int, list[Coord]] = {}
    for r, c in cells:
        color = int(g[r, c])
        counts[color] = counts.get(color, 0) + 1
        positions.setdefault(color, []).append((r, c))

    assert 1 <= len(counts) <= 2, "a frame may have at most one marker color"
    border_color = max(counts, key=counts.get)
    marker_color: int | None = None
    if len(counts) == 2:
        marker_colors = [color for color in counts if color != border_color]
        assert len(marker_colors) == 1 and counts[marker_colors[0]] == 1
        marker_color = int(marker_colors[0])
        assert _is_side_midpoint(positions[marker_color][0], bbox)
        assert counts[border_color] == len(border_cells) - 1
    else:
        assert counts[border_color] == len(border_cells)

    return Frame(int(border_color), marker_color, bbox)


def _extract_frames(g: np.ndarray, anchor: Coord) -> list[Frame]:
    mask = g != BLACK
    mask[anchor] = False
    frames = [_parse_frame(g, comp) for comp in _components(mask)]
    assert frames, "expected at least one rectangular fragment"

    colors = [frame.color for frame in frames]
    assert len(colors) == len(set(colors)), "frame border colors must be unique"
    return frames


def _chain_frames(frames: list[Frame]) -> list[Frame]:
    areas = [frame.area for frame in frames]
    assert len(areas) == len(set(areas)), "frame sizes must be strictly ordered"
    current = max(frames, key=lambda frame: frame.area)
    by_color = {frame.color: frame for frame in frames}

    ordered: list[Frame] = []
    seen: set[int] = set()
    while True:
        assert current.color not in seen, "marker chain must not cycle"
        ordered.append(current)
        seen.add(current.color)
        if current.marker is None:
            break
        assert current.marker in by_color, "marker color must match a frame color"
        nxt = by_color[current.marker]
        assert nxt.height <= current.height - 2
        assert nxt.width <= current.width - 2
        current = nxt

    assert len(ordered) == len(frames), "marker chain must use every fragment"
    return ordered


def _draw_clean_frame(out: np.ndarray, center: Coord, height: int, width: int, color: int) -> None:
    cr, cc = center
    r0 = cr - height // 2
    c0 = cc - width // 2
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    h, w = out.shape
    assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Reassemble scattered hollow rectangle fragments into a centered nested stack.

    Input structure:
    - The background is black, with one isolated gray pixel marking the assembly center.
    - Every other non-black connected component is a one-pixel-wide hollow rectangular frame.
    - A frame may have exactly one different-colored marker pixel at the midpoint of one side; that marker color is also the border color of another frame.

    Rule:
    - Start with the largest frame and place it so its geometric center is on the gray anchor.
    - Read that frame's side marker color to choose the next frame by matching border color, and repeat the centered placement inside the previous frame.
    - Stop when the current frame has no marker, and redraw all chosen frames with clean, unmarked borders on a black output canvas.

    Key insight:
    - Size only makes nesting physically possible; the actual layer sequence is the visible color chain from each outer border marker to the next frame's border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    anchor = _find_anchor(g)
    ordered = _chain_frames(_extract_frames(g, anchor))

    out = np.zeros_like(g)
    for frame in ordered:
        _draw_clean_frame(out, anchor, frame.height, frame.width, frame.color)
    return out


# ----------------------------- generator -----------------------------

GEN_RETRIES = 500
PLACE_RETRIES = 300
GRID_SIZE_RANGE = (18, 30)
FRAGMENT_COUNT_RANGE = (3, 5)
MAX_OUTER_DIM = 15


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == 1]


def _nested_lengths(n: int) -> list[int] | None:
    outer_min = 3 + 2 * (n - 1)
    choices = _odd_choices(outer_min, MAX_OUTER_DIM)
    if not choices:
        return None
    lengths = [random.choice(choices)]
    for i in range(1, n):
        remaining = n - i - 1
        lo = 3 + 2 * remaining
        hi = lengths[-1] - 2
        options = _odd_choices(lo, hi)
        if not options:
            return None
        lengths.append(random.choice(options))
    return lengths


def _marker_cell(r0: int, c0: int, height: int, width: int, side: str) -> Coord:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    mid_r = r0 + height // 2
    mid_c = c0 + width // 2
    if side == "top":
        return r0, mid_c
    if side == "bottom":
        return r1, mid_c
    if side == "left":
        return mid_r, c0
    assert side == "right"
    return mid_r, c1


def _draw_input_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    color: int,
    marker: int | None,
    side: str | None,
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if marker is not None:
        assert side is not None
        mr, mc = _marker_cell(r0, c0, height, width, side)
        grid[mr, mc] = int(marker)


def _fits_free(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = used.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return not bool(np.any(used[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _reserve(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = used.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_frames(shape: tuple[int, int], dims: list[tuple[int, int]], anchor: Coord) -> list[Coord] | None:
    h, w = shape
    used = np.zeros(shape, dtype=bool)
    ar, ac = anchor
    used[max(0, ar - 1) : min(h, ar + 2), max(0, ac - 1) : min(w, ac + 2)] = True
    positions: list[Coord | None] = [None] * len(dims)

    for idx in sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True):
        fh, fw = dims[idx]
        candidates = [
            (r, c)
            for r in range(0, h - fh + 1)
            for c in range(0, w - fw + 1)
            if _fits_free(used, r, c, fh, fw)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        _reserve(used, r0, c0, fh, fw)

    assert all(pos is not None for pos in positions)
    return [pos for pos in positions if pos is not None]


def _expected_output(shape: tuple[int, int], anchor: Coord, dims: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for (height, width), color in zip(dims, colors):
        _draw_clean_frame(out, anchor, height, width, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*FRAGMENT_COUNT_RANGE)
    heights = _nested_lengths(n)
    widths = _nested_lengths(n)
    if heights is None or widths is None:
        return None
    dims = list(zip(heights, widths))

    outer_h, outer_w = dims[0]
    min_h = max(GRID_SIZE_RANGE[0], outer_h + 4)
    min_w = max(GRID_SIZE_RANGE[0], outer_w + 4)
    if min_h > GRID_SIZE_RANGE[1] or min_w > GRID_SIZE_RANGE[1]:
        return None
    h = random.randint(min_h, GRID_SIZE_RANGE[1])
    w = random.randint(min_w, GRID_SIZE_RANGE[1])

    anchor_r = random.randint(outer_h // 2, h - outer_h // 2 - 1)
    anchor_c = random.randint(outer_w // 2, w - outer_w // 2 - 1)
    anchor = (anchor_r, anchor_c)

    positions = None
    for _ in range(PLACE_RETRIES):
        positions = _place_frames((h, w), dims, anchor)
        if positions is not None:
            break
    if positions is None:
        return None

    colors = list(random.sample(FRAME_COLORS, n))
    sides = [random.choice(SIDES) for _ in range(n - 1)]

    grid = np.zeros((h, w), dtype=int)
    grid[anchor] = GRAY
    for i, ((fh, fw), (r0, c0), color) in enumerate(zip(dims, positions, colors)):
        marker = colors[i + 1] if i + 1 < n else None
        side = sides[i] if i + 1 < n else None
        _draw_input_frame(grid, r0, c0, fh, fw, color, marker, side)

    expected = _expected_output((h, w), anchor, dims, colors)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
