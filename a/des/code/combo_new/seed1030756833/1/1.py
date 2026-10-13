from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
DIRS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class _Frame:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    gap: tuple[int, int]
    direction: str


def _perimeter(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _gap_direction(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> str | None:
    on_top = r == r0 and c0 < c < c1
    on_bottom = r == r1 and c0 < c < c1
    on_left = c == c0 and r0 < r < r1
    on_right = c == c1 and r0 < r < r1
    sides = [on_top, on_bottom, on_left, on_right]
    if sum(bool(x) for x in sides) != 1:
        return None
    if on_top:
        return "up"
    if on_bottom:
        return "down"
    if on_left:
        return "left"
    return "right"


def _find_frames(g: np.ndarray, bg: int = BG) -> list[_Frame]:
    h, w = g.shape
    frames: list[_Frame] = []
    seen = np.zeros((h, w), dtype=bool)

    for sr, sc in zip(*np.where(g != bg), strict=False):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        color = int(g[sr, sc])
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _perimeter(r0, c0, r1, c1)
        border_set = set(border)
        if any(p not in border_set for p in pts):
            continue

        gap_cells = []
        ok = True
        for r, c in border:
            v = int(g[r, c])
            if v == color:
                continue
            if v == bg:
                gap_cells.append((r, c))
            else:
                ok = False
                break
        if not ok or len(gap_cells) != 1:
            continue

        gr, gc = gap_cells[0]
        direction = _gap_direction(gr, gc, r0, c0, r1, c1)
        if direction is None:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        if np.any(interior == color):
            continue
        if not np.any((interior != bg) & (interior != color)):
            continue

        frames.append(_Frame(color, r0, c0, r1, c1, (gr, gc), direction))

    frames.sort(key=lambda f: ((f.r1 - f.r0 + 1) * (f.c1 - f.c0 + 1), f.r0, f.c0), reverse=True)
    selected: list[_Frame] = []
    for cand in frames:
        overlap = False
        for old in selected:
            if not (
                cand.r1 < old.r0
                or old.r1 < cand.r0
                or cand.c1 < old.c0
                or old.c1 < cand.c0
            ):
                overlap = True
                break
        if not overlap:
            selected.append(cand)
    selected.sort(key=lambda f: (f.r0, f.c0))
    return selected


def _moved_sprite_cells(g: np.ndarray, frame: _Frame, bg: int = BG) -> list[tuple[int, int, int]]:
    interior = g[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1]
    mask = (interior != bg) & (interior != frame.color)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return []

    rows = pts[:, 0] + frame.r0 + 1
    cols = pts[:, 1] + frame.c0 + 1
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())

    dr = dc = 0
    if frame.direction == "up":
        dr = -(max_r - frame.r0 + 1)
    elif frame.direction == "down":
        dr = frame.r1 - min_r + 1
    elif frame.direction == "left":
        dc = -(max_c - frame.c0 + 1)
    else:
        dc = frame.c1 - min_c + 1

    moved = []
    for rr, cc in zip(rows.tolist(), cols.tolist(), strict=False):
        moved.append((int(rr) + dr, int(cc) + dc, int(g[int(rr), int(cc)])))
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pop each framed sprite out through the side marked by its one-cell gap.

    Input structure:
    - The background is black. The grid contains several one-pixel-thick rectangular frames drawn in one non-background color.
    - Each frame has exactly one black single-cell gap on one of its four sides, and its interior contains a small sprite whose color differs from the frame color.
    - The space outside the frames is black, with enough room in the gap direction for the sprite to leave the frame.

    Rule:
    - For every rectangular frame, find the unique missing border cell and use its side as the movement direction: top means up, bottom means down, left means left, and right means right.
    - Repair the frame by filling the gap with the frame color, clear the whole frame interior to black, and translate the interior sprite far enough that its bounding box sits completely outside the frame adjacent to that side.
    - Stamp the translated sprite into the outside black space, preserving its original color pattern.

    Key insight: The missing border cell is both the direction marker and the exit; once the side is known, the sprite moves just past that repaired wall.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g, BG)

    moved: list[tuple[int, int, int]] = []
    for frame in frames:
        moved.extend(_moved_sprite_cells(g, frame, BG))
        gr, gc = frame.gap
        out[gr, gc] = frame.color
        out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] = BG

    h, w = out.shape
    for r, c, color in moved:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color

    return out


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _random_sprite(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    h = rng.randint(1, max_h)
    w = rng.randint(1, max_w)
    target = rng.randint(2, min(max(2, h * w), 8))
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    cells = {(r - r0, c - c0) for r, c in cells}
    sh = max(r for r, _ in cells) + 1
    sw = max(c for _, c in cells) + 1
    mask = np.zeros((sh, sw), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _object_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    sprite_rows: np.ndarray,
    sprite_cols: np.ndarray,
    direction: str,
) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    sh = int(sprite_rows.max() - sprite_rows.min() + 1)
    sw = int(sprite_cols.max() - sprite_cols.min() + 1)
    if direction == "up":
        dr, dc = -(int(sprite_rows.max()) - r0 + 1), 0
    elif direction == "down":
        dr, dc = r1 - int(sprite_rows.min()) + 1, 0
    elif direction == "left":
        dr, dc = 0, -(int(sprite_cols.max()) - c0 + 1)
    else:
        dr, dc = 0, c1 - int(sprite_cols.min()) + 1

    for r, c in zip(sprite_rows.tolist(), sprite_cols.tolist(), strict=False):
        cells.add((int(r) + dr, int(c) + dc))

    if direction == "up":
        for r in range(r0 - sh, r0):
            for c in range(int(sprite_cols.min()), int(sprite_cols.max()) + 1):
                cells.add((r, c))
    elif direction == "down":
        for r in range(r1 + 1, r1 + sh + 1):
            for c in range(int(sprite_cols.min()), int(sprite_cols.max()) + 1):
                cells.add((r, c))
    elif direction == "left":
        for r in range(int(sprite_rows.min()), int(sprite_rows.max()) + 1):
            for c in range(c0 - sw, c0):
                cells.add((r, c))
    else:
        for r in range(int(sprite_rows.min()), int(sprite_rows.max()) + 1):
            for c in range(c1 + 1, c1 + sw + 1):
                cells.add((r, c))
    return cells


def _try_place_object(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    rng: random.Random,
    frame_color: int,
    sprite_color: int,
    direction: str,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    sprite = _random_sprite(rng, 4, 4)
    sh, sw = sprite.shape
    ih = rng.randint(max(sh + 1, 3), min(max(sh + 3, 5), 8))
    iw = rng.randint(max(sw + 1, 3), min(max(sw + 3, 5), 8))
    fh, fw = ih + 2, iw + 2

    min_r, max_r = 1, h - fh - 1
    min_c, max_c = 1, w - fw - 1
    if direction == "up":
        min_r = max(min_r, sh)
    elif direction == "down":
        max_r = min(max_r, h - fh - sh)
    elif direction == "left":
        min_c = max(min_c, sw)
    else:
        max_c = min(max_c, w - fw - sw)
    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(80):
        r0 = rng.randint(min_r, max_r)
        c0 = rng.randint(min_c, max_c)
        r1, c1 = r0 + fh - 1, c0 + fw - 1
        spr_r0 = r0 + 1 + rng.randint(0, ih - sh)
        spr_c0 = c0 + 1 + rng.randint(0, iw - sw)
        rr, cc = np.where(sprite)
        sprite_rows = rr + spr_r0
        sprite_cols = cc + spr_c0

        footprint = _object_cells(r0, c0, r1, c1, sprite_rows, sprite_cols, direction)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
            continue
        if footprint & blocked:
            continue

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        if direction == "up":
            gap = (r0, rng.randint(int(sprite_cols.min()), int(sprite_cols.max())))
        elif direction == "down":
            gap = (r1, rng.randint(int(sprite_cols.min()), int(sprite_cols.max())))
        elif direction == "left":
            gap = (rng.randint(int(sprite_rows.min()), int(sprite_rows.max())), c0)
        else:
            gap = (rng.randint(int(sprite_rows.min()), int(sprite_rows.max())), c1)
        grid[gap] = BG

        for r, c in zip(sprite_rows.tolist(), sprite_cols.tolist(), strict=False):
            grid[int(r), int(c)] = sprite_color
        return _expand_cells(footprint, h, w, radius=1)

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 30)
    w = rng.randint(16, 30)
    grid = np.zeros((h, w), dtype=int)
    frame_color = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    sprite_palette = [c for c in range(1, 10) if c != frame_color]

    n_objects = rng.choice([1, 2, 2, 3, 3, 4])
    directions = list(DIRS)
    rng.shuffle(directions)
    while len(directions) < n_objects:
        more = list(DIRS)
        rng.shuffle(more)
        directions.extend(more)

    blocked: set[tuple[int, int]] = set()
    placed = 0
    for i in range(n_objects):
        direction = directions[i]
        sprite_color = rng.choice(sprite_palette)
        new_blocked = _try_place_object(grid, blocked, rng, frame_color, sprite_color, direction)
        if new_blocked is None:
            continue
        blocked |= new_blocked
        placed += 1

    if placed == 0:
        return None

    frames = _find_frames(grid, BG)
    if len(frames) != placed:
        return None
    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to place non-overlapping framed sprites")
