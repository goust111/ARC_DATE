from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
MAROON = 9
NON_START_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 500
MIN_HEAD_MARGIN = 1
MIN_CHAIN_LEN = 2
MAX_CHAIN_LEN = 5
MIN_LINK_DIST = 4
MAX_LINK_DIST = 9
MIN_GRID = 12
MAX_GRID = 26
MAX_DISTRACTORS = 4

Coord = tuple[int, int]


@dataclass(frozen=True)
class Arrow:
    color: int
    head: Coord
    direction: str
    pixels: tuple[Coord, ...]


def _neighbors8(r: int, c: int) -> Iterable[Coord]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _arrow_pixels(head: Coord, direction: str) -> tuple[Coord, ...]:
    r, c = head
    if direction == "up":
        pts = ((r, c), (r + 1, c - 1), (r + 1, c + 1))
    elif direction == "down":
        pts = ((r, c), (r - 1, c - 1), (r - 1, c + 1))
    elif direction == "left":
        pts = ((r, c), (r - 1, c + 1), (r + 1, c + 1))
    elif direction == "right":
        pts = ((r, c), (r - 1, c - 1), (r + 1, c - 1))
    else:
        raise ValueError(f"unknown direction: {direction}")
    return tuple(sorted(pts))


def _make_arrow(head: Coord, direction: str, color: int) -> Arrow:
    return Arrow(color=color, head=head, direction=direction, pixels=_arrow_pixels(head, direction))


def _extract_components(grid: np.ndarray) -> list[tuple[int, list[Coord]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[Coord]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, pts))
    return comps


def _infer_arrow(color: int, pts: list[Coord]) -> Arrow | None:
    if len(pts) != 3:
        return None

    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})

    if len(rows) == 2:
        top_row, bot_row = rows
        top = [p for p in pts if p[0] == top_row]
        bot = [p for p in pts if p[0] == bot_row]
        if len(top) == 1 and len(bot) == 2:
            hc = top[0][1]
            bcols = sorted(c for _, c in bot)
            if bcols[1] - bcols[0] == 2 and hc == bcols[0] + 1:
                return _make_arrow(top[0], "up", color)
        if len(top) == 2 and len(bot) == 1:
            hc = bot[0][1]
            tcols = sorted(c for _, c in top)
            if tcols[1] - tcols[0] == 2 and hc == tcols[0] + 1:
                return _make_arrow(bot[0], "down", color)

    if len(cols) == 2:
        left_col, right_col = cols
        left = [p for p in pts if p[1] == left_col]
        right = [p for p in pts if p[1] == right_col]
        if len(left) == 1 and len(right) == 2:
            hr = left[0][0]
            rrows = sorted(r for r, _ in right)
            if rrows[1] - rrows[0] == 2 and hr == rrows[0] + 1:
                return _make_arrow(left[0], "left", color)
        if len(left) == 2 and len(right) == 1:
            hr = right[0][0]
            lrows = sorted(r for r, _ in left)
            if lrows[1] - lrows[0] == 2 and hr == lrows[0] + 1:
                return _make_arrow(right[0], "right", color)

    return None


def _parse_arrows(grid: np.ndarray) -> tuple[list[Arrow], int]:
    arrows: list[Arrow] = []
    for color, pts in _extract_components(grid):
        arrow = _infer_arrow(color, pts)
        assert arrow is not None, "every non-background component must be a 3-pixel arrow"
        arrows.append(arrow)
    starts = [idx for idx, arrow in enumerate(arrows) if arrow.color == MAROON]
    assert len(starts) == 1, "expected exactly one maroon start arrow"
    return arrows, starts[0]


def _ray_distance(head: Coord, direction: str, pt: Coord) -> int | None:
    hr, hc = head
    pr, pc = pt
    dr, dc = DIRS[direction]
    step = dr if dr != 0 else dc
    if dr != 0:
        if pc != hc:
            return None
        delta = pr - hr
    else:
        if pr != hr:
            return None
        delta = pc - hc
    if delta * step <= 0:
        return None
    return abs(delta)


def _top_left(arrow: Arrow) -> Coord:
    return min(arrow.pixels)


def _find_next_arrow(current_idx: int, arrows: list[Arrow]) -> tuple[int, int] | None:
    current = arrows[current_idx]
    best: tuple[tuple[int, int, int, int], int] | None = None
    for idx, arrow in enumerate(arrows):
        if idx == current_idx:
            continue
        distances = [_ray_distance(current.head, current.direction, pt) for pt in arrow.pixels]
        valid = [d for d in distances if d is not None]
        if not valid:
            continue
        dist = min(valid)
        tr, tc = _top_left(arrow)
        key = (dist, tr, tc, idx)
        if best is None or key < best[0]:
            best = (key, idx)
    if best is None:
        return None
    return best[1], best[0][0]


def _trace_chain(arrows: list[Arrow], start_idx: int) -> list[tuple[int, int, int]]:
    links: list[tuple[int, int, int]] = []
    current_idx = start_idx
    visited = {start_idx}
    while True:
        nxt = _find_next_arrow(current_idx, arrows)
        if nxt is None:
            break
        next_idx, dist = nxt
        if next_idx in visited:
            break
        links.append((current_idx, next_idx, dist))
        visited.add(next_idx)
        current_idx = next_idx
    return links


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a maroon-start arrow chain and paint the empty cells between successive arrows with the source arrow's color.

    Input structure:
    - The background is black.
    - Each foreground object is a monochrome 3-pixel arrow whose geometry encodes a cardinal direction.
    - Exactly one arrow is maroon; it is the chain's starting arrow.

    Rule:
    - Parse every arrow's head position and pointing direction from its 3-pixel shape.
    - Starting from the maroon arrow, repeatedly look forward along the arrow's ray and choose the nearest arrow having any pixel on that ray.
    - For each consecutive pair in that chain, recolor only the background cells strictly between them with the source arrow's color; arrows not on the chain stay unchanged.

    Key insight:
    - The puzzle requires two linked inferences: first decode each tiny arrow's direction from geometry, then follow the resulting directed nearest-neighbor chain to determine where recoloring happens.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    arrows, start_idx = _parse_arrows(g)
    for src_idx, _dst_idx, dist in _trace_chain(arrows, start_idx):
        src = arrows[src_idx]
        dr, dc = DIRS[src.direction]
        hr, hc = src.head
        for step in range(1, dist):
            rr = hr + dr * step
            cc = hc + dc * step
            if int(out[rr, cc]) == BG:
                out[rr, cc] = src.color
    return out


def _in_bounds(pt: Coord, h: int, w: int) -> bool:
    r, c = pt
    return 0 <= r < h and 0 <= c < w


def _head_runway(head: Coord, direction: str, h: int, w: int) -> int:
    r, c = head
    if direction == "up":
        return r - MIN_HEAD_MARGIN
    if direction == "down":
        return (h - 1 - MIN_HEAD_MARGIN) - r
    if direction == "left":
        return c - MIN_HEAD_MARGIN
    return (w - 1 - MIN_HEAD_MARGIN) - c


def _step(head: Coord, direction: str, amount: int) -> Coord:
    dr, dc = DIRS[direction]
    return head[0] + dr * amount, head[1] + dc * amount


def _interior_ray_cells(head: Coord, direction: str, dist: int) -> list[Coord]:
    dr, dc = DIRS[direction]
    return [(head[0] + dr * k, head[1] + dc * k) for k in range(1, dist)]


def _buffer_cells(pixels: Iterable[Coord], h: int, w: int) -> set[Coord]:
    buf: set[Coord] = set()
    for r, c in pixels:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    buf.add((nr, nc))
    return buf


def _ray_hits_any(head: Coord, direction: str, blocked: set[Coord], h: int, w: int) -> bool:
    dr, dc = DIRS[direction]
    rr, cc = head
    while True:
        rr += dr
        cc += dc
        if not _in_bounds((rr, cc), h, w):
            return False
        if (rr, cc) in blocked:
            return True


def _arrow_fits(arrow: Arrow, h: int, w: int) -> bool:
    return all(_in_bounds(pt, h, w) for pt in arrow.pixels)


def _can_place_arrow(
    arrow: Arrow,
    h: int,
    w: int,
    occupied: set[Coord],
    buffer: set[Coord],
    reserved: set[Coord],
    extra_forbidden: set[Coord],
) -> bool:
    if not _arrow_fits(arrow, h, w):
        return False
    for pt in arrow.pixels:
        if pt in occupied or pt in buffer or pt in reserved or pt in extra_forbidden:
            return False
    return True


def _build_chain_geometry(rng: random.Random, h: int, w: int, chain_len: int) -> list[Arrow] | None:
    def extend(arrows: list[Arrow], occupied: set[Coord], buffer: set[Coord], reserved: set[Coord]) -> list[Arrow] | None:
        if len(arrows) == chain_len:
            return arrows

        prev = arrows[-1]
        max_dist = min(MAX_LINK_DIST, _head_runway(prev.head, prev.direction, h, w))
        if max_dist < MIN_LINK_DIST:
            return None

        dist_options = list(range(MIN_LINK_DIST, max_dist + 1))
        rng.shuffle(dist_options)
        dir_options = list(DIRS)

        for dist in dist_options:
            new_head = _step(prev.head, prev.direction, dist)
            path_cells = _interior_ray_cells(prev.head, prev.direction, dist)
            if any(cell in occupied or cell in reserved for cell in path_cells):
                continue

            rng.shuffle(dir_options)
            is_last = len(arrows) + 1 == chain_len
            for direction in dir_options:
                candidate = _make_arrow(new_head, direction, 1)
                if not _can_place_arrow(candidate, h, w, occupied, buffer, reserved, set()):
                    continue
                if _ray_hits_any(candidate.head, candidate.direction, occupied, h, w):
                    continue
                if _ray_hits_any(candidate.head, candidate.direction, reserved, h, w):
                    continue
                if not is_last and _head_runway(candidate.head, candidate.direction, h, w) < MIN_LINK_DIST:
                    continue

                new_arrows = arrows + [candidate]
                new_occupied = occupied | set(candidate.pixels)
                new_buffer = buffer | _buffer_cells(candidate.pixels, h, w)
                new_reserved = reserved | set(path_cells)
                found = extend(new_arrows, new_occupied, new_buffer, new_reserved)
                if found is not None:
                    return found
        return None

    for _ in range(160):
        first_dir = rng.choice(list(DIRS))
        first_arrow = _make_arrow(
            (
                rng.randint(MIN_HEAD_MARGIN, h - 1 - MIN_HEAD_MARGIN),
                rng.randint(MIN_HEAD_MARGIN, w - 1 - MIN_HEAD_MARGIN),
            ),
            first_dir,
            1,
        )
        if not _arrow_fits(first_arrow, h, w):
            continue
        if chain_len > 1 and _head_runway(first_arrow.head, first_arrow.direction, h, w) < MIN_LINK_DIST:
            continue
        found = extend(
            [first_arrow],
            set(first_arrow.pixels),
            _buffer_cells(first_arrow.pixels, h, w),
            set(),
        )
        if found is not None:
            return found
    return None


def _terminal_forbidden(chain: list[Arrow], h: int, w: int) -> set[Coord]:
    last = chain[-1]
    dr, dc = DIRS[last.direction]
    rr, cc = last.head
    forbidden: set[Coord] = set()
    while True:
        rr += dr
        cc += dc
        if not _in_bounds((rr, cc), h, w):
            break
        forbidden.add((rr, cc))
    return forbidden


def _paint_grid(h: int, w: int, arrows: list[Arrow]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for arrow in arrows:
        for r, c in arrow.pixels:
            grid[r, c] = arrow.color
    return grid


def _colorize_geometry(rng: random.Random, chain: list[Arrow], distractors: list[Arrow]) -> list[Arrow]:
    colored: list[Arrow] = []
    for idx, arrow in enumerate(chain):
        if idx == 0:
            color = MAROON
        else:
            color = rng.choice(NON_START_COLORS)
        colored.append(_make_arrow(arrow.head, arrow.direction, color))
    for arrow in distractors:
        color = rng.choice(NON_START_COLORS)
        colored.append(_make_arrow(arrow.head, arrow.direction, color))
    return colored


def _place_distractors(
    rng: random.Random,
    h: int,
    w: int,
    chain: list[Arrow],
    count: int,
) -> list[Arrow] | None:
    occupied = {pt for arrow in chain for pt in arrow.pixels}
    buffer = _buffer_cells(occupied, h, w)
    links = _trace_chain(chain, 0)
    reserved = set()
    for src_idx, _dst_idx, dist in links:
        reserved.update(_interior_ray_cells(chain[src_idx].head, chain[src_idx].direction, dist))
    extra_forbidden = _terminal_forbidden(chain, h, w)

    distractors: list[Arrow] = []
    for _ in range(count):
        placed = False
        for _ in range(240):
            candidate = _make_arrow(
                (
                    rng.randint(MIN_HEAD_MARGIN, h - 1 - MIN_HEAD_MARGIN),
                    rng.randint(MIN_HEAD_MARGIN, w - 1 - MIN_HEAD_MARGIN),
                ),
                rng.choice(list(DIRS)),
                1,
            )
            if not _can_place_arrow(candidate, h, w, occupied, buffer, reserved, extra_forbidden):
                continue
            distractors.append(candidate)
            occupied.update(candidate.pixels)
            buffer |= _buffer_cells(candidate.pixels, h, w)
            placed = True
            break
        if not placed:
            return None
    return distractors


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)
    chain_len = rng.randint(MIN_CHAIN_LEN, MAX_CHAIN_LEN)
    chain = _build_chain_geometry(rng, h, w, chain_len)
    if chain is None:
        return None

    distractor_count = 0 if rng.random() < 0.5 else rng.randint(1, MAX_DISTRACTORS)
    distractors = _place_distractors(rng, h, w, chain, distractor_count)
    if distractors is None:
        return None

    arrows = _colorize_geometry(rng, chain, distractors)
    grid = _paint_grid(h, w, arrows)

    parsed, start_idx = _parse_arrows(grid)
    if len(parsed) != len(arrows):
        return None
    if parsed[start_idx].color != MAROON:
        return None
    if len(_trace_chain(parsed, start_idx)) != chain_len - 1:
        return None
    if not np.any(main(grid) != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a valid input grid for the arrow-chain recoloring task."""
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
