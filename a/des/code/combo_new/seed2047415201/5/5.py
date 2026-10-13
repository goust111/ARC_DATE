import random

import numpy as np


CONTROL_COLOR = 7
MAX_SIZE = 30
GEN_RETRIES = 300


def _mode_reading_order(values) -> int:
    counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for idx, value in enumerate(values):
        v = int(value)
        counts[v] = counts.get(v, 0) + 1
        if v not in first_seen:
            first_seen[v] = idx
    if not counts:
        raise ValueError("cannot take mode of an empty collection")
    return max(counts, key=lambda v: (counts[v], -first_seen[v]))


def _boundary_background(g: np.ndarray, marker: tuple[int, int]) -> int:
    h, w = g.shape
    vals = []
    mr, mc = marker
    for c in range(w):
        if (0, c) != (mr, mc):
            vals.append(int(g[0, c]))
        if h > 1 and (h - 1, c) != (mr, mc):
            vals.append(int(g[h - 1, c]))
    for r in range(1, h - 1):
        if (r, 0) != (mr, mc):
            vals.append(int(g[r, 0]))
        if w > 1 and (r, w - 1) != (mr, mc):
            vals.append(int(g[r, w - 1]))
    if vals:
        return _mode_reading_order(vals)
    all_vals = [int(v) for v in g.flat if int(v) != CONTROL_COLOR]
    return _mode_reading_order(all_vals)


def _stripe_bbox(g: np.ndarray, bg: int, marker: tuple[int, int]) -> tuple[int, int, int, int]:
    mask = g != int(bg)
    mask[marker] = False
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise ValueError("expected a non-background striped rectangle")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _classify_stripes(patch: np.ndarray, marker_side: str) -> str:
    h, w = patch.shape
    row_uniform = all(bool(np.all(patch[r, :] == patch[r, 0])) for r in range(h))
    col_uniform = all(bool(np.all(patch[:, c] == patch[0, c])) for c in range(w))

    if row_uniform and not col_uniform:
        return "horizontal"
    if col_uniform and not row_uniform:
        return "vertical"

    row_values = [int(patch[r, 0]) for r in range(h)]
    col_values = [int(patch[0, c]) for c in range(w)]
    row_variety = len(set(row_values))
    col_variety = len(set(col_values))
    if row_uniform and row_variety > 1:
        return "horizontal"
    if col_uniform and col_variety > 1:
        return "vertical"

    if marker_side in ("top", "bottom"):
        return "horizontal"
    if marker_side in ("left", "right"):
        return "vertical"
    raise ValueError("stripe direction is ambiguous")


def _marker_side_and_distance(
    marker: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[str, int]:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[str, int]] = []
    if c0 <= mc <= c1:
        if mr < r0:
            candidates.append(("top", r0 - mr))
        elif mr > r1:
            candidates.append(("bottom", mr - r1))
    if r0 <= mr <= r1:
        if mc < c0:
            candidates.append(("left", c0 - mc))
        elif mc > c1:
            candidates.append(("right", mc - c1))
    if not candidates:
        raise ValueError("control marker must sit outside one side of the striped rectangle")
    return min(candidates, key=lambda item: item[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll a striped rectangular pattern by the distance encoded by an outside orange marker.

    Input structure:
    - A background canvas contains one dense rectangular region filled with horizontal or vertical colored stripes.
    - Exactly one orange single-cell control marker lies outside one side of the striped rectangle.
    - Stripe colors are ordinary pattern colors; the marker is not part of the rectangle.

    Rule:
    - Identify whether the rectangle has horizontal stripes (each row is constant) or vertical stripes (each column is constant).
    - Compute D as the pixel distance from the orange marker to the nearest side of the striped rectangle.
    - Roll horizontal stripes vertically by D cells, or roll vertical stripes horizontally by D cells, preserving the rectangle shape.
    - Clear the orange control marker back to the background color.

    Key insight: The marker's offset from the rectangle is a numeric parameter, while the stripe orientation independently selects the roll axis.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    marker_pts = np.argwhere(g == CONTROL_COLOR)
    if len(marker_pts) != 1:
        raise ValueError("expected exactly one orange control marker")
    marker = tuple(map(int, marker_pts[0]))

    bg = _boundary_background(g, marker)
    bbox = _stripe_bbox(g, bg, marker)
    side, dist = _marker_side_and_distance(marker, bbox)
    r0, c0, r1, c1 = bbox
    patch = g[r0 : r1 + 1, c0 : c1 + 1]

    direction = _classify_stripes(patch, side)
    axis = 0 if direction == "horizontal" else 1

    out = g.copy()
    out[marker] = bg
    out[r0 : r1 + 1, c0 : c1 + 1] = np.roll(patch, shift=int(dist), axis=axis)
    return out


def _paint_stripes(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    colors: list[int],
    orientation: str,
    offset: int,
) -> None:
    if orientation == "horizontal":
        for dr in range(height):
            grid[r0 + dr, c0 : c0 + width] = colors[(dr + offset) % len(colors)]
    else:
        for dc in range(width):
            grid[r0 : r0 + height, c0 + dc] = colors[(dc + offset) % len(colors)]


def _choose_distance(rng: random.Random, color_count: int, max_dist: int) -> int | None:
    candidates = [d for d in range(1, max_dist + 1) if d % color_count != 0]
    if not candidates:
        return None
    return rng.choice(candidates)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["horizontal", "vertical"])
    bg = rng.choice([c for c in range(10) if c != CONTROL_COLOR])
    palette = [c for c in range(10) if c not in (bg, CONTROL_COLOR)]
    rng.shuffle(palette)
    color_count = rng.randint(2, min(4, len(palette)))
    stripe_colors = palette[:color_count]
    offset = rng.randrange(color_count)

    if orientation == "horizontal":
        stripe_h = rng.randint(max(4, color_count + 1), 14)
        stripe_w = rng.randint(4, 18)
        side = rng.choice(["top", "bottom"])
        max_dist = min(8, MAX_SIZE - stripe_h - 2)
        dist = _choose_distance(rng, color_count, max_dist)
        if dist is None:
            return None
        outer = rng.randint(0, 4)
        top = outer + dist if side == "top" else rng.randint(1, 5)
        bottom = outer + dist if side == "bottom" else rng.randint(1, 5)
        left = rng.randint(1, 5)
        right = rng.randint(1, 5)
        h = top + stripe_h + bottom
        w = left + stripe_w + right
        if max(h, w) > MAX_SIZE:
            return None
        r0, c0 = top, left
        marker = (r0 - dist, rng.randint(c0, c0 + stripe_w - 1)) if side == "top" else (
            r0 + stripe_h - 1 + dist,
            rng.randint(c0, c0 + stripe_w - 1),
        )
    else:
        stripe_h = rng.randint(4, 18)
        stripe_w = rng.randint(max(4, color_count + 1), 14)
        side = rng.choice(["left", "right"])
        max_dist = min(8, MAX_SIZE - stripe_w - 2)
        dist = _choose_distance(rng, color_count, max_dist)
        if dist is None:
            return None
        outer = rng.randint(0, 4)
        top = rng.randint(1, 5)
        bottom = rng.randint(1, 5)
        left = outer + dist if side == "left" else rng.randint(1, 5)
        right = outer + dist if side == "right" else rng.randint(1, 5)
        h = top + stripe_h + bottom
        w = left + stripe_w + right
        if max(h, w) > MAX_SIZE:
            return None
        r0, c0 = top, left
        marker = (rng.randint(r0, r0 + stripe_h - 1), c0 - dist) if side == "left" else (
            rng.randint(r0, r0 + stripe_h - 1),
            c0 + stripe_w - 1 + dist,
        )

    grid = np.full((h, w), bg, dtype=int)
    _paint_stripes(grid, r0, c0, stripe_h, stripe_w, stripe_colors, orientation, offset)
    grid[marker] = CONTROL_COLOR

    try:
        output = main(grid)
    except Exception:
        return None
    if np.array_equal(output, grid):
        return None
    if output.ndim != 2 or max(output.shape) > MAX_SIZE:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
