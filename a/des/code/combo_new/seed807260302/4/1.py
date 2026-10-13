from __future__ import annotations

import random
from collections import deque

import numpy as np


def _background_color(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[tuple[int, int]] = deque([(r, c)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(pts)
    return comps


def _ring_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _missing_side(r: int, c: int, h: int, w: int) -> str | None:
    mid_cols = {(w - 1) // 2, w // 2}
    mid_rows = {(h - 1) // 2, h // 2}
    if r == 0 and 0 < c < w - 1 and c in mid_cols:
        return "up"
    if r == h - 1 and 0 < c < w - 1 and c in mid_cols:
        return "down"
    if c == 0 and 0 < r < h - 1 and r in mid_rows:
        return "left"
    if c == w - 1 and 0 < r < h - 1 and r in mid_rows:
        return "right"
    return None


def _find_frames(g: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    frames: list[dict[str, int | str]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(g == color):
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue

            local = np.zeros((h, w), dtype=bool)
            for r, c in pts:
                local[r - r0, c - c0] = True

            ring = _ring_mask(h, w)
            if bool(np.any(local & ~ring)):
                continue
            missing = ring & ~local
            if int(missing.sum()) != 1:
                continue
            mr, mc = map(int, np.argwhere(missing)[0])
            direction = _missing_side(mr, mc, h, w)
            if direction is None:
                continue

            frames.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "direction": direction,
                }
            )
    frames.sort(key=lambda f: (int(f["r0"]), int(f["c0"])))
    return frames


def _find_seed(g: np.ndarray, frame: dict[str, int | str], bg: int) -> dict[str, int]:
    frame_color = int(frame["color"])
    r0, c0 = int(frame["r0"]), int(frame["c0"])
    r1, c1 = int(frame["r1"]), int(frame["c1"])
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (interior != bg) & (interior != frame_color)
    comps = _components(seed_mask)
    assert len(comps) == 1, "each frame must contain exactly one seed object"

    pts = comps[0]
    first_r, first_c = pts[0]
    seed_color = int(interior[first_r, first_c])
    assert all(int(interior[r, c]) == seed_color for r, c in pts), "seed must be monochrome"

    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    sr0, sr1 = min(rows), max(rows)
    sc0, sc1 = min(cols), max(cols)
    patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert bool(np.all(patch == seed_color)), "seed must be a solid rectangle"
    assert int(patch.size) == len(pts), "seed rectangle must not contain holes"

    return {
        "color": seed_color,
        "r0": r0 + 1 + sr0,
        "c0": c0 + 1 + sc0,
        "r1": r0 + 1 + sr1,
        "c1": c0 + 1 + sc1,
    }


def _paint_stair(out: np.ndarray, frame: dict[str, int | str], seed: dict[str, int]) -> None:
    direction = str(frame["direction"])
    ir0, ic0 = int(frame["r0"]) + 1, int(frame["c0"]) + 1
    ir1, ic1 = int(frame["r1"]) - 1, int(frame["c1"]) - 1
    sr0, sc0 = int(seed["r0"]), int(seed["c0"])
    sr1, sc1 = int(seed["r1"]), int(seed["c1"])
    color = int(seed["color"])

    if direction == "up":
        for step, r in enumerate(range(sr0 - 1, ir0 - 1, -1)):
            grow = step
            c0 = max(ic0, sc0 - grow)
            c1 = min(ic1, sc1 + grow)
            out[r, c0 : c1 + 1] = color
    elif direction == "down":
        for step, r in enumerate(range(sr1 + 1, ir1 + 1)):
            grow = step
            c0 = max(ic0, sc0 - grow)
            c1 = min(ic1, sc1 + grow)
            out[r, c0 : c1 + 1] = color
    elif direction == "left":
        for step, c in enumerate(range(sc0 - 1, ic0 - 1, -1)):
            grow = step
            r0 = max(ir0, sr0 - grow)
            r1 = min(ir1, sr1 + grow)
            out[r0 : r1 + 1, c] = color
    elif direction == "right":
        for step, c in enumerate(range(sc1 + 1, ic1 + 1)):
            grow = step
            r0 = max(ir0, sr0 - grow)
            r1 = min(ir1, sr1 + grow)
            out[r0 : r1 + 1, c] = color
    else:
        raise AssertionError(f"unknown frame gap direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow clipped stair fills from each internal seed toward the matching frame gap.

    Input structure:
    - A black background contains one or more same-colored rectangular border frames.
    - Each frame is a hollow rectangle whose border has exactly one single-cell central gap on one side.
    - Inside every frame is exactly one small solid rectangular seed of a non-frame color.

    Rule:
    - Keep the original grid size, background, frame pixels, gap cells, and seed rectangles unchanged.
    - For each frame, read the side containing the gap as the growth direction.
    - From the seed, paint successive rows or columns toward that side in the seed color; each step expands by one cell on both perpendicular sides.
    - Clip every painted row or column to the frame interior and stop at the inner wall before the border.

    Key insight:
    The missing border pixel supplies direction, while the seed supplies both the paint color and the initial stair width or height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    out = g.copy()
    frames = _find_frames(g, bg)
    assert frames, "expected at least one gapped rectangular frame"
    for frame in frames:
        seed = _find_seed(g, frame, bg)
        _paint_stair(out, frame, seed)
    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_FRAME_COLOR = 5
GEN_SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_DIRECTIONS = ["up", "down", "left", "right"]
GEN_ATTEMPTS = 300


def _gap_local(h: int, w: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return 0, w // 2
    if direction == "down":
        return h - 1, w // 2
    if direction == "left":
        return h // 2, 0
    if direction == "right":
        return h // 2, w - 1
    raise ValueError(direction)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, direction: str) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = GEN_FRAME_COLOR
    grid[r1, c0 : c1 + 1] = GEN_FRAME_COLOR
    grid[r0 : r1 + 1, c0] = GEN_FRAME_COLOR
    grid[r0 : r1 + 1, c1] = GEN_FRAME_COLOR
    gr, gc = _gap_local(h, w, direction)
    grid[r0 + gr, c0 + gc] = GEN_BG


def _seed_positions(h: int, w: int, sh: int, sw: int, direction: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for sr in range(2, h - sh - 1):
        for sc in range(2, w - sw - 1):
            top_space = sr - 1
            bottom_space = h - 2 - (sr + sh - 1)
            left_space = sc - 1
            right_space = w - 2 - (sc + sw - 1)
            if min(top_space, bottom_space, left_space, right_space) < 1:
                continue
            if direction == "up" and top_space < 1:
                continue
            if direction == "down" and bottom_space < 1:
                continue
            if direction == "left" and left_space < 1:
                continue
            if direction == "right" and right_space < 1:
                continue
            positions.append((sr, sc))
    return positions


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    rows, cols = occupied.shape
    if r < 1 or c < 1 or r + h >= rows or c + w >= cols:
        return False
    return not bool(occupied[r - 1 : r + h + 1, c - 1 : c + w + 1].any())


def _mark_occupied(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    occupied[r - 1 : r + h + 1, c - 1 : c + w + 1] = True


def _place_one(
    grid: np.ndarray,
    occupied: np.ndarray,
    seed_color: int,
    direction: str,
    rng: random.Random,
) -> bool:
    rows, cols = grid.shape
    size_choices = [7, 9, 11, 13]
    for _ in range(80):
        heights = [s for s in size_choices if s <= rows - 2]
        widths = [s for s in size_choices if s <= cols - 2]
        if not heights or not widths:
            return False
        h = rng.choice(heights)
        w = rng.choice(widths)
        sh = rng.randint(1, min(3, h - 4))
        sw = rng.randint(1, min(3, w - 4))
        seed_positions = _seed_positions(h, w, sh, sw, direction)
        if not seed_positions:
            continue

        top_lefts = [
            (r, c)
            for r in range(1, rows - h)
            for c in range(1, cols - w)
            if _can_place(occupied, r, c, h, w)
        ]
        if not top_lefts:
            continue

        r0, c0 = rng.choice(top_lefts)
        sr, sc = rng.choice(seed_positions)
        _draw_frame(grid, r0, c0, h, w, direction)
        grid[r0 + sr : r0 + sr + sh, c0 + sc : c0 + sc + sw] = int(seed_color)
        _mark_occupied(occupied, r0, c0, h, w)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    object_count = rng.randint(1, 4)
    min_side = 13 if object_count <= 2 else 20
    rows = rng.randint(min_side, 30)
    cols = rng.randint(min_side, 30)
    grid = np.full((rows, cols), GEN_BG, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    colors = rng.sample(GEN_SEED_COLORS, object_count)
    directions = GEN_DIRECTIONS[:]
    rng.shuffle(directions)
    while len(directions) < object_count:
        directions.append(rng.choice(GEN_DIRECTIONS))

    for seed_color, direction in zip(colors, directions[:object_count]):
        if not _place_one(grid, occupied, seed_color, direction, rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stair-fill input")
