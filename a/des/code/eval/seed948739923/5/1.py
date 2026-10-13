from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

SEED_BASE_RIGHT = np.array(
    [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ],
    dtype=bool,
)
SEED_MASKS = {
    "right": SEED_BASE_RIGHT,
    "down": np.rot90(SEED_BASE_RIGHT, 3),
    "left": np.rot90(SEED_BASE_RIGHT, 2),
    "up": np.rot90(SEED_BASE_RIGHT, 1),
}
ROTATION_K = {
    "right": 0,
    "down": 3,
    "left": 2,
    "up": 1,
}
TEMPLATE_LIBRARY = tuple(
    np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    for rows in (
        ("111", "010", "011"),
        ("110", "111", "001"),
        ("101", "111", "100"),
        ("1110", "0010", "0111"),
        ("0111", "0100", "1110"),
        ("1100", "0111", "0010"),
        ("1110", "1000", "1110"),
        ("1010", "1110", "0011"),
    )
)

GEN_ATTEMPTS = 400
GEN_ROOM_ATTEMPTS = 120
GEN_TEMPLATE_ATTEMPTS = 200
GEN_GRID_MIN = 22
GEN_GRID_MAX = 36
GEN_EDGE_MARGIN = 5
GEN_ROOM_PAD = 2
GEN_ROOM_H_RANGE = (7, 9)
GEN_ROOM_W_RANGE = (7, 9)
GEN_SPUR_PROB = 0.5


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _tight_mask_from_coords(coords)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = int(color)


def _seed_direction(seed_mask: np.ndarray) -> str:
    for direction, ref in SEED_MASKS.items():
        tight_ref = _tight_mask(ref)
        if seed_mask.shape == tight_ref.shape and np.array_equal(seed_mask, tight_ref):
            return direction
    raise AssertionError("expected a supported 3x3 L-seed sprite")


def _best_centered_placement(region_mask: np.ndarray, sprite_mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(region_mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    sh, sw = sprite_mask.shape
    if sh > r1 - r0 + 1 or sw > c1 - c0 + 1:
        return None

    target_r = (r0 + r1 - sh + 1) / 2.0
    target_c = (c0 + c1 - sw + 1) / 2.0
    best_key: tuple[float, float, int, int] | None = None
    best: tuple[int, int] | None = None

    for top in range(r0, r1 - sh + 2):
        for left in range(c0, c1 - sw + 2):
            patch = region_mask[top : top + sh, left : left + sw]
            if patch.shape != sprite_mask.shape:
                continue
            if not np.all(patch[sprite_mask]):
                continue
            key = (
                float((top - target_r) ** 2 + (left - target_c) ** 2),
                float(abs(top - target_r) + abs(left - target_c)),
                int(top),
                int(left),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (int(top), int(left))
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each enclosed region's seed color to the same-colored outer template,
    rotate that template by the seed's L orientation, and place it back into the region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == WALL), "grey walls must be present"

    passable_components = _components(g != WALL)
    outer_mask = np.zeros(g.shape, dtype=bool)
    region_components: list[np.ndarray] = []
    for comp in passable_components:
        if _touches_border(comp, g.shape):
            outer_mask[comp[:, 0], comp[:, 1]] = True
        else:
            region_components.append(comp)
    assert region_components, "expected at least one enclosed region"

    colors = [int(color) for color in np.unique(g) if int(color) not in {BG, WALL}]
    templates: dict[int, np.ndarray] = {}
    for color in colors:
        outer_parts = _components((g == color) & outer_mask)
        if not outer_parts:
            continue
        assert len(outer_parts) == 1, "each template color must have one outer template"
        templates[color] = _tight_mask_from_coords(outer_parts[0])

    out = g.copy()
    for region in region_components:
        region_mask = np.zeros(g.shape, dtype=bool)
        region_mask[region[:, 0], region[:, 1]] = True

        region_foreground = region_mask & (g != BG) & (g != WALL)
        seed_parts = _components(region_foreground)
        assert len(seed_parts) == 1, "each enclosed region must contain exactly one seed"

        seed_coords = seed_parts[0]
        seed_color = int(g[seed_coords[0, 0], seed_coords[0, 1]])
        assert seed_color in templates, "seed color must have a matching outer template"
        seed_mask = _tight_mask_from_coords(seed_coords)
        direction = _seed_direction(seed_mask)

        rotated_template = np.rot90(templates[seed_color], ROTATION_K[direction])
        placement = _best_centered_placement(region_mask, rotated_template)
        assert placement is not None, "rotated template must fit inside the region"

        out[region_foreground] = BG
        _paint_mask(out, rotated_template, placement[0], placement[1], seed_color)

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _build_room_region(shape: tuple[int, int], top: int, left: int, height: int, width: int) -> np.ndarray:
    h, w = shape
    region = np.zeros((h, w), dtype=bool)
    region[top + 1 : top + height - 1, left + 1 : left + width - 1] = True

    if height >= 8 and width >= 8 and random.random() < GEN_SPUR_PROB:
        side = random.choice(("top", "bottom", "left", "right"))
        if side in {"top", "bottom"}:
            col = random.randint(left + 2, left + width - 3)
            depth = random.randint(1, min(2, height - 5))
            if side == "top":
                region[top + 1 : top + 1 + depth, col] = False
            else:
                region[top + height - 1 - depth : top + height - 1, col] = False
        else:
            row = random.randint(top + 2, top + height - 3)
            depth = random.randint(1, min(2, width - 5))
            if side == "left":
                region[row, left + 1 : left + 1 + depth] = False
            else:
                region[row, left + width - 1 - depth : left + width - 1] = False
    return region


def _draw_room(grid: np.ndarray, region: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top, left : left + width] = WALL
    grid[top + height - 1, left : left + width] = WALL
    grid[top : top + height, left] = WALL
    grid[top : top + height, left + width - 1] = WALL

    full_interior = np.zeros_like(region, dtype=bool)
    full_interior[top + 1 : top + height - 1, left + 1 : left + width - 1] = True
    grid[full_interior & ~region] = WALL


def _candidate_edge_positions(shape: tuple[int, int], mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []

    for left in range(0, w - sw + 1):
        positions.append((0, left))
        positions.append((h - sh, left))
    for top in range(1, h - sh):
        positions.append((top, 0))
        positions.append((top, w - sw))

    random.shuffle(positions)
    return positions


def _can_place_template(grid: np.ndarray, mask: np.ndarray, top: int, left: int, padding: int = 1) -> bool:
    sh, sw = mask.shape
    h, w = grid.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False

    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + sh + padding)
    c1 = min(w, left + sw + padding)
    if np.any(grid[r0:r1, c0:c1] != BG):
        return False
    return True


def _sample_rooms(shape: tuple[int, int], count: int) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    rooms: list[tuple[int, int, int, int]] = []
    if h - GEN_ROOM_H_RANGE[0] - GEN_EDGE_MARGIN < GEN_EDGE_MARGIN:
        return None
    if w - GEN_ROOM_W_RANGE[0] - GEN_EDGE_MARGIN < GEN_EDGE_MARGIN:
        return None
    for _ in range(count):
        placed = False
        for _ in range(GEN_ROOM_ATTEMPTS):
            rh = random.randint(*GEN_ROOM_H_RANGE)
            rw = random.randint(*GEN_ROOM_W_RANGE)
            if h - rh - GEN_EDGE_MARGIN < GEN_EDGE_MARGIN or w - rw - GEN_EDGE_MARGIN < GEN_EDGE_MARGIN:
                continue
            top = random.randint(GEN_EDGE_MARGIN, h - rh - GEN_EDGE_MARGIN)
            left = random.randint(GEN_EDGE_MARGIN, w - rw - GEN_EDGE_MARGIN)
            box = (top, left, top + rh - 1, left + rw - 1)
            if any(_boxes_overlap(box, prev, pad=GEN_ROOM_PAD) for prev in rooms):
                continue
            rooms.append(box)
            placed = True
            break
        if not placed:
            return None
    return rooms


def _generate_templates(colors: list[int]) -> dict[int, np.ndarray]:
    shapes = random.sample(TEMPLATE_LIBRARY, len(colors))
    templates: dict[int, np.ndarray] = {}
    for color, shape in zip(colors, shapes):
        templates[int(color)] = np.rot90(shape, random.randrange(4))
    return templates


def _place_templates(grid: np.ndarray, templates: dict[int, np.ndarray]) -> bool:
    items = sorted(templates.items(), key=lambda item: (-int(np.sum(item[1])), int(item[0])))
    for color, mask in items:
        placed = False
        for top, left in _candidate_edge_positions(grid.shape, mask):
            if not _can_place_template(grid, mask, top, left):
                continue
            _paint_mask(grid, mask, top, left, color)
            placed = True
            break
        if not placed:
            return False
    return True


def _sample_region_colors(colors: list[int], region_count: int) -> list[int]:
    assigned = colors[:]
    while len(assigned) < region_count:
        assigned.append(random.choice(colors))
    random.shuffle(assigned)
    return assigned


def _sample_directions(region_count: int) -> list[str]:
    directions = list(SEED_MASKS)
    random.shuffle(directions)
    return directions[:region_count]


def _attempt() -> np.ndarray | None:
    region_count = random.randint(2, 4)
    template_count = random.randint(2, min(4, region_count))
    lo = max(GEN_GRID_MIN, 18 + 2 * region_count)
    hi = min(GEN_GRID_MAX, 28 + 2 * region_count)
    h = random.randint(lo, hi)
    w = random.randint(lo, hi)

    room_boxes = _sample_rooms((h, w), region_count)
    if room_boxes is None:
        return None

    colors = random.sample(PALETTE, template_count)
    templates = _generate_templates(colors)
    region_colors = _sample_region_colors(colors, region_count)
    directions = _sample_directions(region_count)

    grid = np.full((h, w), BG, dtype=int)
    room_regions: list[np.ndarray] = []
    for top, left, bottom, right in room_boxes:
        region = _build_room_region((h, w), top, left, bottom - top + 1, right - left + 1)
        _draw_room(grid, region, top, left, bottom - top + 1, right - left + 1)
        room_regions.append(region)

    if not _place_templates(grid, templates):
        return None

    for region_mask, color, direction in zip(room_regions, region_colors, directions):
        rotated_template = np.rot90(templates[color], ROTATION_K[direction])
        if _best_centered_placement(region_mask, rotated_template) is None:
            return None
        seed_mask = SEED_MASKS[direction]
        seed_pos = _best_centered_placement(region_mask, seed_mask)
        if seed_pos is None:
            return None
        _paint_mask(grid, seed_mask, seed_pos[0], seed_pos[1], color)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-template region puzzle")
