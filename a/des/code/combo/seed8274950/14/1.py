from __future__ import annotations

from collections import deque
import random

import numpy as np


TRAIL_BLUE = 1
TRAIL_RED = 2
TRAIL_GREEN = 3

BG_CHOICES = [0, 4, 5, 6, 7, 8, 9]
FG_CHOICES = [4, 5, 6, 7, 8, 9]

GRID_MIN = 10
GRID_MAX = 18
MIN_OBJECTS = 2
MAX_OBJECTS = 4
LANE_GAP = 1
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _count_runs(bits: np.ndarray) -> int:
    runs = 0
    prev = False
    for b in map(bool, bits):
        if b and not prev:
            runs += 1
        prev = b
    return runs


def _connected_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _open_direction(mask: np.ndarray) -> str:
    edges = {
        "up": mask[0, :],
        "down": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }
    run_counts = {side: _count_runs(edge) for side, edge in edges.items()}
    open_sides = [side for side, runs in run_counts.items() if runs >= 2]
    assert len(open_sides) == 1, "expected exactly one open bbox edge"
    open_side = open_sides[0]
    for side, runs in run_counts.items():
        if side == open_side:
            continue
        assert runs == 1, "non-open bbox edges must have one continuous contact segment"
    return open_side


def _trail_color(area: int) -> int:
    if area <= 4:
        return TRAIL_RED
    if area <= 8:
        return TRAIL_BLUE
    return TRAIL_GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each object out through its unique open bbox edge and leave an area-coded trail.

    Input structure:
    - The background fills the grid.
    - Foreground objects are monochrome, 4-connected, and mutually separated.
    - On exactly one bbox edge, the object's contact with that edge is split by
      background gaps; the other three bbox edges have one continuous contact segment.

    Rule:
    - The split edge gives the push direction.
    - Translate the whole object along that direction until it leaves the grid.
    - Paint every in-grid cell touched during that translation with a trail color
      based on object area: <=4 red, 5-8 blue, >=9 green.
    - The original object itself does not remain.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for comp in _connected_components(g, bg):
        direction = _open_direction(comp["mask"])
        trail = _trail_color(int(comp["area"]))

        for r, c in comp["coords"]:
            if direction == "up":
                out[: r + 1, c] = trail
            elif direction == "down":
                out[r:, c] = trail
            elif direction == "left":
                out[r, : c + 1] = trail
            else:
                out[r, c:] = trail

    return out


def _top_open_mask(kind: str) -> np.ndarray:
    # Under the strict edge-segmentation constraint, the smallest valid object has area 5.
    area_lo, area_hi = (5, 8) if kind == "medium" else (9, 18)
    h_range = (2, 4) if kind == "medium" else (3, 6)
    w_range = (3, 5) if kind == "medium" else (3, 6)

    for _ in range(300):
        h = random.randint(*h_range)
        w = random.randint(*w_range)
        peak_mask = np.array([random.random() < 0.5 for _ in range(w)], dtype=bool)
        if peak_mask.sum() < 2 or _count_runs(peak_mask) < 2:
            continue

        heights = np.array([random.randint(1, h - 1) for _ in range(w)], dtype=int)
        heights[peak_mask] = h
        area = int(heights.sum())
        if not (area_lo <= area <= area_hi):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for c, height in enumerate(heights):
            mask[h - height :, c] = True

        if _open_direction(mask) != "up":
            continue
        return mask

    raise RuntimeError(f"failed to build a {kind} top-open object")


def _oriented_mask(direction: str, kind: str) -> np.ndarray:
    base = _top_open_mask(kind)
    if direction == "up":
        return base
    if direction == "down":
        return np.flipud(base)
    if direction == "left":
        return base.T
    if direction == "right":
        return np.fliplr(base.T)
    raise ValueError(direction)


def _random_composition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _place_vertical(
    grid: np.ndarray,
    masks: list[np.ndarray],
    colors: list[int],
    direction: str,
) -> None:
    h, w = grid.shape
    widths = [int(mask.shape[1]) for mask in masks]
    extra = w - sum(widths) - LANE_GAP * (len(masks) + 1)
    assert extra >= 0
    gaps = [LANE_GAP + v for v in _random_composition(extra, len(masks) + 1)]

    left = gaps[0]
    for mask, color, lane_gap in zip(masks, colors, gaps[1:], strict=False):
        mh, mw = mask.shape
        if direction == "up":
            top_lo = max(1, h // 3)
            top_hi = h - mh
        else:
            top_lo = 0
            top_hi = min(h - mh - 1, max(0, (2 * h) // 3 - mh))
        if top_lo > top_hi:
            raise RuntimeError("no valid vertical placement")

        top = random.randint(top_lo, top_hi)
        sub = grid[top : top + mh, left : left + mw]
        sub[mask] = color
        grid[top : top + mh, left : left + mw] = sub
        left += mw + lane_gap


def _place_horizontal(
    grid: np.ndarray,
    masks: list[np.ndarray],
    colors: list[int],
    direction: str,
) -> None:
    h, w = grid.shape
    heights = [int(mask.shape[0]) for mask in masks]
    extra = h - sum(heights) - LANE_GAP * (len(masks) + 1)
    assert extra >= 0
    gaps = [LANE_GAP + v for v in _random_composition(extra, len(masks) + 1)]

    top = gaps[0]
    for mask, color, lane_gap in zip(masks, colors, gaps[1:], strict=False):
        mh, mw = mask.shape
        if direction == "left":
            left_lo = max(1, w // 3)
            left_hi = w - mw
        else:
            left_lo = 0
            left_hi = min(w - mw - 1, max(0, (2 * w) // 3 - mw))
        if left_lo > left_hi:
            raise RuntimeError("no valid horizontal placement")

        left = random.randint(left_lo, left_hi)
        sub = grid[top : top + mh, left : left + mw]
        sub[mask] = color
        grid[top : top + mh, left : left + mw] = sub
        top += mh + lane_gap


def _generate_attempt() -> np.ndarray:
    direction = random.choice(["up", "down", "left", "right"])
    obj_count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    kinds = ["medium", "large"] + [random.choice(["medium", "large"]) for _ in range(obj_count - 2)]
    random.shuffle(kinds)

    masks = [_oriented_mask(direction, kind) for kind in kinds]
    bg = random.choice(BG_CHOICES)
    colors = random.sample([c for c in FG_CHOICES if c != bg], obj_count)

    if direction in ("up", "down"):
        widths = [int(mask.shape[1]) for mask in masks]
        heights = [int(mask.shape[0]) for mask in masks]
        w_min = max(GRID_MIN, sum(widths) + LANE_GAP * (obj_count + 1))
        h_min = max(GRID_MIN, max(heights) + 2)
        if w_min > GRID_MAX or h_min > GRID_MAX:
            raise RuntimeError("vertical layout does not fit")
        h = random.randint(h_min, GRID_MAX)
        w = random.randint(w_min, GRID_MAX)
        grid = np.full((h, w), bg, dtype=int)
        _place_vertical(grid, masks, colors, direction)
    else:
        widths = [int(mask.shape[1]) for mask in masks]
        heights = [int(mask.shape[0]) for mask in masks]
        h_min = max(GRID_MIN, sum(heights) + LANE_GAP * (obj_count + 1))
        w_min = max(GRID_MIN, max(widths) + 2)
        if h_min > GRID_MAX or w_min > GRID_MAX:
            raise RuntimeError("horizontal layout does not fit")
        h = random.randint(h_min, GRID_MAX)
        w = random.randint(w_min, GRID_MAX)
        grid = np.full((h, w), bg, dtype=int)
        _place_horizontal(grid, masks, colors, direction)

    comps = _connected_components(grid, bg)
    if len(comps) != obj_count:
        raise RuntimeError("unexpected component merge")
    if _get_bg(grid) != bg:
        raise RuntimeError("background must remain dominant")

    out = main(grid)
    if out.shape != grid.shape:
        raise RuntimeError("shape changed")
    used_trails = set(map(int, np.unique(out))) - {bg}
    if not ({TRAIL_BLUE, TRAIL_GREEN} <= used_trails):
        raise RuntimeError("need both medium and large trail branches")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _generate_attempt()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
