from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 80
PAD = 1


@dataclass(frozen=True)
class _Bar:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    key_pos: tuple[int, int] | None
    key_color: int | None


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _analyze_bar(grid: np.ndarray, pts: list[tuple[int, int]]) -> _Bar:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    h, w = patch.shape

    assert len(pts) == h * w, "each door bar must be a solid rectangle"
    assert np.all(patch != BG), "door bars cannot contain background holes"

    vals, cnts = np.unique(patch, return_counts=True)
    major_idx = int(cnts.argmax())
    color = int(vals[major_idx])

    key_pos: tuple[int, int] | None = None
    key_color: int | None = None
    if len(vals) == 2:
        assert int(cnts[major_idx]) == h * w - 1, "expected a single-pixel key anomaly"
        key_local = np.argwhere(patch != color)
        assert key_local.shape[0] == 1
        kr, kc = map(int, key_local[0])
        key_pos = (r0 + kr, c0 + kc)
        key_color = int(patch[kr, kc])
    else:
        assert len(vals) == 1, "bars may only contain the frame color or one key pixel"

    return _Bar(
        color=color,
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
        key_pos=key_pos,
        key_color=key_color,
    )


def _fill_vertical_pair(out: np.ndarray, color: int, a: _Bar, b: _Bar) -> None:
    left, right = sorted([a, b], key=lambda bar: bar.c0)
    gap = right.c0 - left.c1 - 1
    assert gap >= 1, "paired vertical bars need a black corridor between them"
    assert left.r0 == right.r0 and left.r1 == right.r1, "vertical pair must share row span"

    key_hits: list[tuple[tuple[int, int], int]] = []
    if left.key_pos is not None and left.key_pos[1] == left.c1:
        key_hits.append((left.key_pos, int(left.key_color)))
    if right.key_pos is not None and right.key_pos[1] == right.c0:
        key_hits.append((right.key_pos, int(right.key_color)))
    assert len(key_hits) == 1, "each pair must expose exactly one key on an inner face"

    (kr, kc), fill = key_hits[0]
    out[kr, kc] = color
    out[left.r0 : left.r1 + 1, left.c1 + 1 : right.c0] = fill


def _fill_horizontal_pair(out: np.ndarray, color: int, a: _Bar, b: _Bar) -> None:
    top, bottom = sorted([a, b], key=lambda bar: bar.r0)
    gap = bottom.r0 - top.r1 - 1
    assert gap >= 1, "paired horizontal bars need a black corridor between them"
    assert top.c0 == bottom.c0 and top.c1 == bottom.c1, "horizontal pair must share column span"

    key_hits: list[tuple[tuple[int, int], int]] = []
    if top.key_pos is not None and top.key_pos[0] == top.r1:
        key_hits.append((top.key_pos, int(top.key_color)))
    if bottom.key_pos is not None and bottom.key_pos[0] == bottom.r0:
        key_hits.append((bottom.key_pos, int(bottom.key_color)))
    assert len(key_hits) == 1, "each pair must expose exactly one key on an inner face"

    (kr, kc), fill = key_hits[0]
    out[kr, kc] = color
    out[top.r1 + 1 : bottom.r0, top.c0 : top.c1 + 1] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the corridor between each same-colored door-bar pair using the hidden key pixel
    on one inner face, then restore the bar to a pure rectangle.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bars = [_analyze_bar(g, pts) for pts in _components4(g != BG)]
    groups: dict[int, list[_Bar]] = defaultdict(list)
    for bar in bars:
        groups[int(bar.color)].append(bar)

    for color, pair in groups.items():
        assert len(pair) == 2, "every frame color must appear on exactly one pair of bars"
        a, b = pair
        if a.r0 == b.r0 and a.r1 == b.r1:
            _fill_vertical_pair(out, int(color), a, b)
        elif a.c0 == b.c0 and a.c1 == b.c1:
            _fill_horizontal_pair(out, int(color), a, b)
        else:
            raise AssertionError("paired bars must align either side-by-side or top-to-bottom")

    return out


def _bbox_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(hh, r0 + h + PAD)
    cc1 = min(ww, c0 + w + PAD)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _paint_vertical_pair(
    grid: np.ndarray,
    out: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    left_w: int,
    gap: int,
    right_w: int,
    frame_color: int,
    fill_color: int,
    key_on_left: bool,
    key_offset: int,
) -> None:
    grid[r0 : r0 + height, c0 : c0 + left_w] = frame_color
    grid[r0 : r0 + height, c0 + left_w + gap : c0 + left_w + gap + right_w] = frame_color
    out[r0 : r0 + height, c0 : c0 + left_w] = frame_color
    out[r0 : r0 + height, c0 + left_w + gap : c0 + left_w + gap + right_w] = frame_color

    if key_on_left:
        key_r = r0 + key_offset
        key_c = c0 + left_w - 1
    else:
        key_r = r0 + key_offset
        key_c = c0 + left_w + gap

    grid[key_r, key_c] = fill_color
    out[key_r, key_c] = frame_color
    out[r0 : r0 + height, c0 + left_w : c0 + left_w + gap] = fill_color


def _paint_horizontal_pair(
    grid: np.ndarray,
    out: np.ndarray,
    r0: int,
    c0: int,
    width: int,
    top_h: int,
    gap: int,
    bottom_h: int,
    frame_color: int,
    fill_color: int,
    key_on_top: bool,
    key_offset: int,
) -> None:
    grid[r0 : r0 + top_h, c0 : c0 + width] = frame_color
    grid[r0 + top_h + gap : r0 + top_h + gap + bottom_h, c0 : c0 + width] = frame_color
    out[r0 : r0 + top_h, c0 : c0 + width] = frame_color
    out[r0 + top_h + gap : r0 + top_h + gap + bottom_h, c0 : c0 + width] = frame_color

    if key_on_top:
        key_r = r0 + top_h - 1
        key_c = c0 + key_offset
    else:
        key_r = r0 + top_h + gap
        key_c = c0 + key_offset

    grid[key_r, key_c] = fill_color
    out[key_r, key_c] = frame_color
    out[r0 + top_h : r0 + top_h + gap, c0 : c0 + width] = fill_color


def _sample_vertical_dims(rng: random.Random, grid_h: int, grid_w: int) -> tuple[int, int, int, int] | None:
    left_w = rng.randint(1, 3)
    right_w = rng.randint(1, 3)
    gap = rng.randint(1, 5)
    max_height = min(11, grid_h - 2)
    min_height = max(4, max(left_w, right_w) + 2)
    if min_height > max_height:
        return None
    height = rng.randint(min_height, max_height)
    if left_w + gap + right_w > grid_w - 2:
        return None
    return height, left_w, gap, right_w


def _sample_horizontal_dims(rng: random.Random, grid_h: int, grid_w: int) -> tuple[int, int, int, int] | None:
    top_h = rng.randint(1, 3)
    bottom_h = rng.randint(1, 3)
    gap = rng.randint(1, 5)
    max_width = min(11, grid_w - 2)
    min_width = max(4, max(top_h, bottom_h) + 2)
    if min_width > max_width:
        return None
    width = rng.randint(min_width, max_width)
    if top_h + gap + bottom_h > grid_h - 2:
        return None
    return width, top_h, gap, bottom_h


def _place_pair(
    rng: random.Random,
    grid: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    frame_color: int,
    fill_color: int,
    orientation: str,
) -> bool:
    h, w = grid.shape

    if orientation == "v":
        dims = _sample_vertical_dims(rng, h, w)
        if dims is None:
            return False
        height, left_w, gap, right_w = dims
        pair_h = height
        pair_w = left_w + gap + right_w
    else:
        dims = _sample_horizontal_dims(rng, h, w)
        if dims is None:
            return False
        width, top_h, gap, bottom_h = dims
        pair_h = top_h + gap + bottom_h
        pair_w = width

    positions = [
        (r0, c0)
        for r0 in range(1, h - pair_h)
        for c0 in range(1, w - pair_w)
        if _bbox_is_free(occupied, r0, c0, pair_h, pair_w)
    ]
    if not positions:
        return False

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        r0, c0 = rng.choice(positions)
        if orientation == "v":
            key_on_left = bool(rng.randint(0, 1) == 0)
            key_offset = rng.randint(0, height - 1)
            _paint_vertical_pair(
                grid,
                out,
                r0,
                c0,
                height,
                left_w,
                gap,
                right_w,
                frame_color,
                fill_color,
                key_on_left,
                key_offset,
            )
        else:
            key_on_top = bool(rng.randint(0, 1) == 0)
            key_offset = rng.randint(0, width - 1)
            _paint_horizontal_pair(
                grid,
                out,
                r0,
                c0,
                width,
                top_h,
                gap,
                bottom_h,
                frame_color,
                fill_color,
                key_on_top,
                key_offset,
            )

        occupied[r0 : r0 + pair_h, c0 : c0 + pair_w] = True
        return True

    return False


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = rng.randint(16, 29)
    grid_w = rng.randint(16, 29)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    out = grid.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    max_pairs = min(4, len(ALL_COLORS) - 1)
    n_pairs = rng.randint(1, max_pairs)
    frame_colors = rng.sample(ALL_COLORS, n_pairs)
    fill_pool = [c for c in ALL_COLORS if c not in frame_colors]
    fill_colors = [rng.choice(fill_pool) for _ in range(n_pairs)]

    orientations = [rng.choice(["v", "h"]) for _ in range(n_pairs)]
    if n_pairs >= 2 and len(set(orientations)) == 1 and rng.random() < 0.6:
        orientations[rng.randrange(n_pairs)] = "h" if orientations[0] == "v" else "v"

    order = list(range(n_pairs))
    rng.shuffle(order)
    for idx in order:
        if not _place_pair(
            rng,
            grid,
            out,
            occupied,
            frame_colors[idx],
            fill_colors[idx],
            orientations[idx],
        ):
            return None

    return grid, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case(rng)
        if case is None:
            continue
        grid, expected = case
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except AssertionError:
            continue
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
