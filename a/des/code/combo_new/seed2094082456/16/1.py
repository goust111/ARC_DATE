import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                if cr > 0 and mask[cr - 1, cc] and not seen[cr - 1, cc]:
                    seen[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not seen[cr + 1, cc]:
                    seen[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not seen[cr, cc - 1]:
                    seen[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not seen[cr, cc + 1]:
                    seen[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _ring_info(
    grid: np.ndarray,
    bg: int,
    color: int,
    points: list[tuple[int, int]],
) -> dict[str, int | tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(height, width)
    if len(points) != int(border.sum()):
        return None
    if not np.all(sub[border] == color):
        return None
    if np.any(sub[~border] == color):
        return None

    interior = sub[1:-1, 1:-1]
    marker_pts = np.argwhere((interior != bg) & (interior != color))
    if len(marker_pts) != 1:
        return None

    mr, mc = marker_pts[0]
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "marker_r": int(r0 + 1 + mr),
        "marker_c": int(c0 + 1 + mc),
        "marker_color": int(interior[mr, mc]),
    }


def _find_rings(grid: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    rings: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            info = _ring_info(grid, bg, color, comp)
            if info is not None:
                rings.append(info)
    rings.sort(key=lambda item: item["bbox"])  # deterministic draw order
    return rings


def _shadow_direction(r0: int, c0: int, r1: int, c1: int, mr: int, mc: int) -> str:
    dists = {
        "up": int(mr - r0),
        "down": int(r1 - mr),
        "left": int(mc - c0),
        "right": int(c1 - mc),
    }
    best = min(dists.values())
    winners = [name for name, dist in dists.items() if dist == best]
    assert len(winners) == 1, f"ambiguous marker direction at {(mr, mc)} in {(r0, c0, r1, c1)}"
    return winners[0]


def _draw_shifted_border(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    dr: int,
    dc: int,
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    for c in range(c0, c1 + 1):
        for rr in (r0, r1):
            nr, nc = rr + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color
    for r in range(r0 + 1, r1):
        for cc in (c0, c1):
            nr, nc = r + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced rectangle with its internal marker color, then cast a border-only shadow in the marker-indicated direction.

    Input structure:
    - The background is the most common color.
    - Foreground fence objects are 1-pixel-thick closed rectangular rings; fences may share one color.
    - Each fence contains exactly one single-pixel marker of a different color somewhere inside the rectangle.
    - Outside the fences, every cell is background.

    Rule:
    - Detect every rectangular ring and its unique interior marker.
    - Fill all background cells strictly inside that ring with the marker's color while keeping the marker itself unchanged.
    - Infer the shadow direction from the marker's nearest fence side: top, bottom, left, or right.
    - Translate the ring outline by one full side length in that direction and draw only the shifted border in the original fence color, clipping anything that leaves the grid.

    Key insight:
    The single marker carries two independent instructions at once: its color specifies the interior fill, and its off-center placement specifies which side receives the translated shadow outline.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rings = _find_rings(g, bg)

    out = g.copy()
    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == bg] = int(ring["marker_color"])

    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        direction = _shadow_direction(
            r0,
            c0,
            r1,
            c1,
            int(ring["marker_r"]),
            int(ring["marker_c"]),
        )
        if direction == "up":
            dr, dc = -height, 0
        elif direction == "down":
            dr, dc = height, 0
        elif direction == "left":
            dr, dc = 0, -width
        else:
            dr, dc = 0, width
        _draw_shifted_border(out, (r0, c0, r1, c1), int(ring["color"]), dr, dc)

    return out


GEN_RETRIES = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
RECT_H_RANGE = (6, 10)
RECT_W_RANGE = (6, 10)
PADDING = 1
CLIP_GRID_PROB = 0.45
COLORS = tuple(range(10))
DIRECTIONS = ("up", "down", "left", "right")


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(height - 1, int(r1))
    cc1 = min(width - 1, int(c1))
    if rr0 > rr1 or cc0 > cc1:
        return None
    return rr0, cc0, rr1, cc1


def _shadow_bbox(
    bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if direction == "up":
        return r0 - height, c0, r1 - height, c1
    if direction == "down":
        return r0 + height, c0, r1 + height, c1
    if direction == "left":
        return r0, c0 - width, r1, c1 - width
    return r0, c0 + width, r1, c1 + width


def _mark_reserved(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = PADDING) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _bbox_overlaps_reserved(
    reserved: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int = PADDING,
) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _marker_position(
    bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return r0 + 1, random.randint(c0 + 2, c1 - 2)
    if direction == "down":
        return r1 - 1, random.randint(c0 + 2, c1 - 2)
    if direction == "left":
        return random.randint(r0 + 2, r1 - 2), c0 + 1
    return random.randint(r0 + 2, r1 - 2), c1 - 1


def _choose_dimensions(grid_h: int, grid_w: int) -> tuple[int, int]:
    max_h = min(RECT_H_RANGE[1], grid_h - 2)
    max_w = min(RECT_W_RANGE[1], grid_w - 2)
    height = random.randint(RECT_H_RANGE[0], max_h)
    width = random.randint(RECT_W_RANGE[0], max_w)
    return height, width


def _place_object(
    grid_h: int,
    grid_w: int,
    reserved: np.ndarray,
    direction: str,
    want_clipped: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    for _ in range(250):
        height, width = _choose_dimensions(grid_h, grid_w)
        r0 = random.randint(0, grid_h - height)
        c0 = random.randint(0, grid_w - width)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        shadow = _shadow_bbox(bbox, direction)
        clipped = _clip_bbox(shadow, grid_h, grid_w)
        if clipped is None:
            continue
        is_clipped = clipped != shadow
        if is_clipped != want_clipped:
            continue
        if _bbox_overlaps_reserved(reserved, bbox):
            continue
        if _bbox_overlaps_reserved(reserved, clipped):
            continue
        marker = _marker_position(bbox, direction)
        _mark_reserved(reserved, bbox)
        _mark_reserved(reserved, clipped)
        return bbox, marker
    return None


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)

    bg = random.choice(COLORS)
    fence = random.choice([c for c in COLORS if c != bg])
    marker_pool = [c for c in COLORS if c not in (bg, fence)]
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    if n_objects > len(marker_pool):
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    directions = random.sample(DIRECTIONS, k=n_objects)
    marker_colors = random.sample(marker_pool, k=n_objects)

    clipped_flags = [False] * n_objects
    if random.random() < CLIP_GRID_PROB:
        clipped_flags[random.randrange(n_objects)] = True

    objects: list[tuple[tuple[int, int, int, int], tuple[int, int], str, int]] = []
    for direction, marker_color, want_clipped in zip(directions, marker_colors, clipped_flags):
        placed = _place_object(grid_h, grid_w, reserved, direction, want_clipped)
        if placed is None:
            return None
        bbox, marker = placed
        objects.append((bbox, marker, direction, marker_color))

    for bbox, marker, _direction, marker_color in objects:
        _draw_ring(grid, bbox, fence)
        mr, mc = marker
        grid[mr, mc] = marker_color

    if _get_bg(grid) != bg:
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-shadow ARC input")
