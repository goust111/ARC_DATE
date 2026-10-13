from __future__ import annotations

import random
from collections import deque

import numpy as np


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _nearest_corner(
    seed: tuple[int, int],
    room_bbox: tuple[int, int, int, int],
) -> str:
    sr, sc = seed
    r0, c0, r1, c1 = room_bbox
    corners = [
        ("tl", r0, c0),
        ("tr", r0, c1),
        ("bl", r1, c0),
        ("br", r1, c1),
    ]
    return min(
        corners,
        key=lambda item: ((sr - item[1]) ** 2 + (sc - item[2]) ** 2, item[0]),
    )[0]


def _aligned_origin(
    room_bbox: tuple[int, int, int, int],
    template_shape: tuple[int, int],
    corner: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = room_bbox
    th, tw = template_shape
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c1 - tw + 1
    if corner == "bl":
        return r1 - th + 1, c0
    if corner == "br":
        return r1 - th + 1, c1 - tw + 1
    raise ValueError(f"unknown corner {corner}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each room seed into a corner-anchored copy of its same-color edge template.

    Input structure:
    - The background is black and a one-pixel-thick gray wall system encloses several rectangular rooms.
    - Each room interior contains exactly one single-pixel colored seed, with one distinct color per room.
    - Outside the wall system, small monochrome connected template shapes appear near the grid edges, one template per seed color.

    Rule:
    - Match every room seed to the template of the same color.
    - Determine which room corner the seed is nearest to: top-left, top-right, bottom-left, or bottom-right.
    - Clear the room interior to background, then place the matched template so that the template bbox corner of the same type aligns to the room interior corner; clip any cells that fall outside the room.
    - Keep the wall system and the external template regions unchanged.

    Key insight:
    - Color identifies which template to use, while the seed's relative position identifies how that template must be corner-aligned inside the room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0
    wall = _most_common_color(non_bg)

    room_components = [
        comp
        for comp in _components_mask(g != wall)
        if not _touches_border(comp, g.shape)
    ]
    assert room_components, "expected enclosed rooms"

    rooms: list[dict[str, object]] = []
    for comp in room_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "room interior must be rectangular"
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(room))) - {bg}
        assert len(colors) == 1, "each room must contain exactly one seed color"
        color = colors.pop()
        seed_positions = np.argwhere(room == color)
        assert len(seed_positions) == 1, "each room must contain exactly one seed"
        sr, sc = map(int, seed_positions[0])
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed": (r0 + sr, c0 + sc),
                "color": color,
            }
        )

    templates: dict[int, np.ndarray] = {}
    for color in sorted(set(int(room["color"]) for room in rooms)):
        components = _components_mask(g == color)
        assert len(components) == 2, "each seed color must appear as one seed and one template"
        seed_like = [comp for comp in components if len(comp) == 1]
        template_like = [comp for comp in components if len(comp) > 1]
        assert len(seed_like) == 1 and len(template_like) == 1
        templates[color] = _component_mask(template_like[0])

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]  # type: ignore[assignment]
        seed = room["seed"]  # type: ignore[assignment]
        color = int(room["color"])
        corner = _nearest_corner(seed, (r0, c0, r1, c1))
        template = templates[color]

        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        base_r, base_c = _aligned_origin((r0, c0, r1, c1), template.shape, corner)
        for dr, dc in np.argwhere(template):
            rr = base_r + int(dr)
            cc = base_c + int(dc)
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                out[rr, cc] = color

    return out


# --------------- generator ---------------

BG = 0
WALL = 5
LAYOUTS = [(2, 2), (2, 3), (3, 2)]
ROOM_SIZE_RANGE = (4, 5)
BASE_MARGIN = 5
MAX_TEMPLATE_DIM = 5
MAX_GENERATION_ATTEMPTS = 300
SIDES = ("top", "bottom", "left", "right")
ANCHORS = ("tl", "tr", "bl", "br")
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _room_intervals(start: int, sizes: list[int]) -> list[tuple[int, int]]:
    intervals = []
    cur = start + 1
    for size in sizes:
        intervals.append((cur, cur + size - 1))
        cur += size + 1
    return intervals


def _draw_wall_system(
    grid: np.ndarray,
    top: int,
    left: int,
    heights: list[int],
    widths: list[int],
) -> list[tuple[int, int, int, int]]:
    system_h = sum(heights) + len(heights) + 1
    system_w = sum(widths) + len(widths) + 1
    bottom = top + system_h - 1
    right = left + system_w - 1

    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL

    cut = top + heights[0] + 1
    for size in heights[1:]:
        grid[cut, left : right + 1] = WALL
        cut += size + 1

    cut = left + widths[0] + 1
    for size in widths[1:]:
        grid[top : bottom + 1, cut] = WALL
        cut += size + 1

    row_intervals = _room_intervals(top, heights)
    col_intervals = _room_intervals(left, widths)
    rooms = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            rooms.append((r0, c0, r1, c1))
    return rooms


def _sample_seed(
    room_bbox: tuple[int, int, int, int],
    anchor: str,
    rng: random.Random,
) -> tuple[int, int]:
    r0, c0, r1, c1 = room_bbox
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    row_depth = min(2, room_h - 1)
    col_depth = min(2, room_w - 1)

    if anchor[0] == "t":
        sr = r0 + rng.randrange(row_depth)
    else:
        sr = r1 - rng.randrange(row_depth)
    if anchor[1] == "l":
        sc = c0 + rng.randrange(col_depth)
    else:
        sc = c1 - rng.randrange(col_depth)
    return sr, sc


def _random_connected_mask(
    height: int,
    width: int,
    anchor: str,
    rng: random.Random,
) -> np.ndarray:
    required = {
        "tl": (0, 0),
        "tr": (0, width - 1),
        "bl": (height - 1, 0),
        "br": (height - 1, width - 1),
    }[anchor]

    min_area = max(2, (height * width) // 3)
    max_area = max(min_area, height * width - (1 if height * width > 4 else 0))

    for _ in range(200):
        target = rng.randint(min_area, max_area)
        cells = {required}
        frontier = [required]
        while len(cells) < target:
            base_r, base_c = rng.choice(frontier)
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = base_r + dr, base_c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier = [cell for cell in frontier if any(
                    0 <= cell[0] + dr < height
                    and 0 <= cell[1] + dc < width
                    and (cell[0] + dr, cell[1] + dc) not in cells
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )]
                if not frontier:
                    break
                continue
            nxt = rng.choice(nbrs)
            cells.add(nxt)
            frontier.append(nxt)
        if len(cells) < 2:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != height - 1 or min(cols) != 0 or max(cols) != width - 1:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask

    raise RuntimeError("failed to generate a connected template mask")


def _sample_template_dims(room_h: int, room_w: int, rng: random.Random) -> tuple[int, int]:
    def sample(room_dim: int) -> int:
        if room_dim == 4 and rng.random() < 0.45:
            return 5
        return rng.randint(2, min(MAX_TEMPLATE_DIM, room_dim))

    return sample(room_h), sample(room_w)


def _reserve_free(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + height > h or c0 + width > w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc1 = min(w, c0 + width + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    occupied[r0 : r0 + height, c0 : c0 + width] = True


def _place_template_rectangles(
    grid_shape: tuple[int, int],
    outer_bbox: tuple[int, int, int, int],
    template_shapes: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    top, left, bottom, right = outer_bbox
    occupied = np.zeros((h, w), dtype=bool)
    occupied[top : bottom + 1, left : right + 1] = True

    side_cycle = list(SIDES) * ((len(template_shapes) + len(SIDES) - 1) // len(SIDES))
    rng.shuffle(side_cycle)
    placements: list[tuple[int, int]] = []

    for idx, (th, tw) in enumerate(template_shapes):
        side_options = [side_cycle[idx]] + [side for side in SIDES if side != side_cycle[idx]]
        placed = None
        for side in side_options:
            for _ in range(80):
                if side == "top":
                    r0 = rng.randint(0, top - th)
                    c0 = rng.randint(0, w - tw)
                elif side == "bottom":
                    r0 = rng.randint(bottom + 1, h - th)
                    c0 = rng.randint(0, w - tw)
                elif side == "left":
                    r0 = rng.randint(0, h - th)
                    c0 = rng.randint(0, left - tw)
                else:
                    r0 = rng.randint(0, h - th)
                    c0 = rng.randint(right + 1, w - tw)

                if _reserve_free(occupied, r0, c0, th, tw):
                    _mark_occupied(occupied, r0, c0, th, tw)
                    placed = (r0, c0)
                    break
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError("failed to place templates without overlap")
        placements.append(placed)

    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray:
    room_rows, room_cols = rng.choice(LAYOUTS)
    heights = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]

    system_h = sum(heights) + room_rows + 1
    system_w = sum(widths) + room_cols + 1
    base_h = system_h + 2 * BASE_MARGIN
    base_w = system_w + 2 * BASE_MARGIN
    if base_h > 30 or base_w > 30:
        return None

    extra_h = rng.randint(0, 30 - base_h)
    extra_w = rng.randint(0, 30 - base_w)
    top = BASE_MARGIN + rng.randint(0, extra_h)
    bottom_margin = BASE_MARGIN + (extra_h - (top - BASE_MARGIN))
    left = BASE_MARGIN + rng.randint(0, extra_w)
    right_margin = BASE_MARGIN + (extra_w - (left - BASE_MARGIN))
    height = system_h + top + bottom_margin
    width = system_w + left + right_margin

    grid = np.full((height, width), BG, dtype=int)
    rooms = _draw_wall_system(grid, top, left, heights, widths)
    outer_bbox = (top, left, top + system_h - 1, left + system_w - 1)

    room_count = len(rooms)
    colors = rng.sample(PALETTE, room_count)
    anchors = list(ANCHORS) * ((room_count + len(ANCHORS) - 1) // len(ANCHORS))
    rng.shuffle(anchors)
    anchors = anchors[:room_count]

    template_specs = []
    for room_bbox, color, anchor in zip(rooms, colors, anchors):
        r0, c0, r1, c1 = room_bbox
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        seed_r, seed_c = _sample_seed(room_bbox, anchor, rng)
        grid[seed_r, seed_c] = color

        th, tw = _sample_template_dims(room_h, room_w, rng)
        mask = _random_connected_mask(th, tw, anchor, rng)
        template_specs.append((color, mask))

    placements = _place_template_rectangles(
        grid.shape,
        outer_bbox,
        [mask.shape for _, mask in template_specs],
        rng,
    )
    for (color, mask), (r0, c0) in zip(template_specs, placements):
        cells = np.argwhere(mask)
        for dr, dc in cells:
            grid[r0 + int(dr), c0 + int(dc)] = color

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
