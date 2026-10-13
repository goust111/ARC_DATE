from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[tuple[int, int]] = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for dr, dc in SIDE_TO_DIR.values():
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append(np.array(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _ray_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    steps: int | None,
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    remaining = steps
    while 0 <= r < h and 0 <= c < w and (remaining is None or remaining > 0):
        cells.append((r, c))
        r += dr
        c += dc
        if remaining is not None:
            remaining -= 1
    return cells


def _detect_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []

    for comp in _components_of_color(grid, GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _perimeter_cells(r0, c0, r1, c1)
        cells = {tuple(map(int, coord)) for coord in comp}
        if len(cells) != len(border) - 1 or not cells <= border:
            continue

        missing = [cell for cell in border if cell not in cells]
        if len(missing) != 1:
            continue

        hole = missing[0]
        hr, hc = hole
        if int(grid[hr, hc]) != BLACK:
            continue
        if (hr in (r0, r1)) and (hc in (c0, c1)):
            continue

        if hr == r0:
            direction = SIDE_TO_DIR["top"]
        elif hr == r1:
            direction = SIDE_TO_DIR["bottom"]
        elif hc == c0:
            direction = SIDE_TO_DIR["left"]
        elif hc == c1:
            direction = SIDE_TO_DIR["right"]
        else:
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or np.any(inner == GREY):
            continue

        seed_mask = (inner != BLACK) & (inner != GREY)
        if not bool(seed_mask.any()):
            continue
        seed_colors = np.unique(inner[seed_mask])
        if seed_colors.size != 1:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "hole": hole,
                "direction": direction,
                "color": int(seed_colors[0]),
                "seed_count": int(seed_mask.sum()),
            }
        )

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["hole"][0], item["hole"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project a same-colored ray outward from each gapped grey frame using the count of interior seed pixels.

    Input structure:
    - The background is black.
    - One or more grey rectangular frames of thickness 1 appear on the grid.
    - Each frame has exactly one single-cell gap on one side of its border.
    - Inside each frame, one or more seed pixels appear in a single non-black, non-grey color.

    Rule:
    - For each grey frame, count the number of interior seed pixels and read their shared color.
    - Use the gap location to determine the outward ray direction: top gap shoots up, bottom shoots down, left shoots left, right shoots right.
    - Starting at the gap cell itself, paint up to that many cells in the outward direction with the seed color.
    - Stop early if the ray would leave the grid or if the next input cell is already non-black.
    - Keep every original frame and every seed pixel unchanged.

    Key insight:
    - Each frame contributes three linked signals: seed color gives the ray color, seed count gives the ray length, and the border gap gives the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    for frame in _detect_frames(g):
        r, c = frame["hole"]
        dr, dc = frame["direction"]
        for _ in range(frame["seed_count"]):
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(g[r, c]) != BLACK:
                break
            out[r, c] = int(frame["color"])
            r += dr
            c += dc
    return out


def _all_bbox_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expanded_bbox_cells(
    r0: int, c0: int, r1: int, c1: int, h: int, w: int, margin: int = 1
) -> set[tuple[int, int]]:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _position_ranges(
    h: int,
    w: int,
    outer_h: int,
    outer_w: int,
    side: str,
    seed_count: int,
    mode: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r_lo, r_hi = 0, h - outer_h
    c_lo, c_hi = 0, w - outer_w
    if r_hi < r_lo or c_hi < c_lo:
        return None

    if side == "top":
        if mode == "free":
            r_lo = max(r_lo, seed_count - 1)
        else:
            r_hi = min(r_hi, seed_count - 2)
    elif side == "bottom":
        if mode == "free":
            r_hi = min(r_hi, h - outer_h + 1 - seed_count)
        else:
            r_lo = max(r_lo, h - outer_h + 2 - seed_count)
    elif side == "left":
        if mode == "free":
            c_lo = max(c_lo, seed_count - 1)
        else:
            c_hi = min(c_hi, seed_count - 2)
    else:  # right
        if mode == "free":
            c_hi = min(c_hi, w - outer_w + 1 - seed_count)
        else:
            c_lo = max(c_lo, w - outer_w + 2 - seed_count)

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (r_lo, r_hi), (c_lo, c_hi)


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        hole = spec["hole"]
        color = int(spec["color"])
        border = _perimeter_cells(r0, c0, r1, c1)
        for cell in border:
            if cell != hole:
                grid[cell] = GREY
        for sr, sc in spec["seeds"]:
            grid[sr, sc] = color
    return grid


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r, c = spec["hole"]
        dr, dc = spec["direction"]
        for _ in range(len(spec["seeds"])):
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                break
            if int(grid[r, c]) != BLACK:
                break
            out[r, c] = int(spec["color"])
            r += dr
            c += dc
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 28)
    w = rng.randint(12, 28)
    max_objects = min(4, len(SEED_COLORS))
    object_count = rng.randint(1, max_objects)

    modes = [rng.choice(("free", "edge"))]
    if object_count >= 2:
        modes = ["free", "edge"]
        while len(modes) < object_count:
            modes.append(rng.choice(("free", "edge")))
        rng.shuffle(modes)

    colors = list(SEED_COLORS)
    rng.shuffle(colors)

    reserved_boxes: set[tuple[int, int]] = set()
    bbox_areas: set[tuple[int, int]] = set()
    protected_black: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for idx in range(object_count):
        mode = modes[idx]
        color = colors[idx]
        placed = False

        for _ in range(PLACEMENT_ATTEMPTS):
            outer_h = rng.randint(4, min(8, h))
            outer_w = rng.randint(4, min(8, w))
            interior_h = outer_h - 2
            interior_w = outer_w - 2
            interior_area = interior_h * interior_w
            if interior_area <= 0:
                continue

            min_seeds = 2 if mode == "edge" else 1
            max_seeds = min(interior_area, 6)
            if max_seeds < min_seeds:
                continue
            seed_count = rng.randint(min_seeds, max_seeds)

            side = rng.choice(SIDES)
            ranges = _position_ranges(h, w, outer_h, outer_w, side, seed_count, mode)
            if ranges is None:
                continue
            (r_lo, r_hi), (c_lo, c_hi) = ranges
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            if side in ("top", "bottom"):
                hole = (r0 if side == "top" else r1, c0 + rng.randint(1, outer_w - 2))
            else:
                hole = (r0 + rng.randint(1, outer_h - 2), c0 if side == "left" else c1)
            direction = SIDE_TO_DIR[side]

            bbox = (r0, c0, r1, c1)
            bbox_area = _all_bbox_cells(*bbox)
            expanded = _expanded_bbox_cells(*bbox, h, w)

            if expanded & reserved_boxes:
                continue
            if bbox_area & protected_black:
                continue

            corridor = _ray_cells(hole, direction, (h, w), seed_count if mode == "free" else None)
            if mode == "free" and len(corridor) != seed_count:
                continue
            if mode == "edge" and len(corridor) >= seed_count:
                continue
            if any(cell in bbox_areas or cell in protected_black for cell in corridor[1:]):
                continue

            interior_cells = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
            ]
            seeds = rng.sample(interior_cells, seed_count)

            specs.append(
                {
                    "bbox": bbox,
                    "hole": hole,
                    "direction": direction,
                    "color": color,
                    "seeds": seeds,
                    "mode": mode,
                }
            )
            reserved_boxes |= expanded
            bbox_areas |= bbox_area
            protected_black |= set(corridor)
            placed = True
            break

        if not placed:
            return None

    grid = _render_input((h, w), specs)
    expected = _expected_output(grid, specs)

    # Keep samples readable and non-degenerate.
    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len({spec["direction"] for spec in specs}) == 1 and len(specs) > 1 and rng.random() < 0.5:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
