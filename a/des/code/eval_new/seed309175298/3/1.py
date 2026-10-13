import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey-bordered room with the edge template whose colored anchor matches the room seed after rotation.

    Input structure:
    - bg = black. Several 1-pixel grey rectangular borders enclose room interiors.
    - Each room interior contains background plus one small connected seed shape of a single non-grey color.
    - Along the outer grid edge sit colored template patches whose bounding boxes match room interior sizes; each template contains a same-colored anchor shape plus a second decoration color.

    Rule:
    - Detect every enclosed room interior and read its seed color and seed mask.
    - Detect every edge template as a connected non-background, non-grey patch touching the grid border.
    - For each room, find the unique unused template and rotation where the rotated template has the same shape as the room interior and its seed-colored pixels exactly coincide with the room seed.
    - Replace the room interior with that rotated template, leaving the grey borders and the rest of the grid unchanged.

    Key insight:
    - The seed does not name a color-only fill; it is an anchor shape whose orientation and placement identify both which outside template to use and how much to rotate the whole template before stamping it into the room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _mode_color(g)

    room_components = [
        comp for comp in _components(g != WALL) if not _touches_border(comp, h, w)
    ]
    assert room_components, "expected at least one enclosed room"

    rooms = []
    for comp in room_components:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(v) for v in np.unique(patch) if int(v) not in {bg}]
        assert len(colors) == 1, "each room should contain exactly one seed color"
        seed_color = colors[0]
        seed_mask = patch == seed_color
        assert seed_mask.any()
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": patch.shape,
                "seed_color": seed_color,
                "seed_mask": seed_mask,
            }
        )

    template_mask = (g != bg) & (g != WALL)
    template_components = [
        comp for comp in _components(template_mask) if _touches_border(comp, h, w)
    ]
    assert template_components, "expected at least one border template"

    templates = []
    for comp in template_components:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        templates.append({"bbox": (r0, c0, r1, c1), "patch": patch})

    out = g.copy()
    used_templates: set[int] = set()
    for room in rooms:
        candidates = []
        for idx, template in enumerate(templates):
            if idx in used_templates:
                continue
            patch = template["patch"]
            for k in range(4):
                rotated = np.rot90(patch, k)
                if rotated.shape != room["shape"]:
                    continue
                if not np.any(rotated == room["seed_color"]):
                    continue
                if np.array_equal(rotated == room["seed_color"], room["seed_mask"]):
                    candidates.append((idx, rotated))
        assert len(candidates) == 1, "room-template match should be unique"
        template_idx, rotated_patch = candidates[0]
        used_templates.add(template_idx)
        r0, c0, r1, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rotated_patch

    return out


def _rotation_signature(mask: np.ndarray) -> set[bytes]:
    return {np.rot90(mask, k).tobytes() for k in range(4)}


def _sample_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    max_h = min(max_h, 3)
    max_w = min(max_w, 3)
    for _ in range(SHAPE_RETRIES):
        box_h = rng.randint(2, max_h)
        box_w = rng.randint(2, max_w)
        target = rng.randint(3, min(5, box_h * box_w - 1))
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[rng.randrange(len(frontier))])
        if len(cells) != target:
            continue
        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if len(_rotation_signature(mask)) != 4:
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric seed shape")


def _allowed_offsets(limit: int, size: int) -> list[int]:
    span = limit - size
    if span < 0:
        return []
    if span >= 2:
        return list(range(1, span))
    return list(range(span + 1))


def _embed_seed(room_shape: tuple[int, int], shape: np.ndarray, rng: random.Random) -> np.ndarray:
    room_h, room_w = room_shape
    sh, sw = shape.shape
    row_offsets = _allowed_offsets(room_h, sh)
    col_offsets = _allowed_offsets(room_w, sw)
    if not row_offsets or not col_offsets:
        raise RuntimeError("seed does not fit in room")
    r0 = rng.choice(row_offsets)
    c0 = rng.choice(col_offsets)
    mask = np.zeros(room_shape, dtype=bool)
    mask[r0 : r0 + sh, c0 : c0 + sw] = shape
    return mask


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_template_rect(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = grid_shape
    ph, pw = patch_shape
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides:
        for _ in range(PLACEMENT_RETRIES):
            if side == "top":
                r0, c0 = 0, rng.randint(0, w - pw)
            elif side == "bottom":
                r0, c0 = h - ph, rng.randint(0, w - pw)
            elif side == "left":
                r0, c0 = rng.randint(0, h - ph), 0
            else:
                r0, c0 = rng.randint(0, h - ph), w - pw
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_rect_overlaps(rect, other, pad=1) for other in existing):
                continue
            return rect
    return None


def _place_room_rect(
    grid_shape: tuple[int, int],
    room_shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = grid_shape
    rh, rw = room_shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(1, h - rh - 1)
        c0 = rng.randint(1, w - rw - 1)
        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if any(_rect_overlaps(rect, other, pad=1) for other in existing):
            continue
        return rect
    return None


def _room_spec(rng: random.Random) -> dict:
    required_rotation = rng.randrange(4)
    if required_rotation % 2 == 1:
        size = rng.randint(4, 7)
        room_h = room_w = size
    else:
        room_h = rng.randint(4, 7)
        room_w = rng.randint(4, 7)
    base_shape = _sample_shape(rng, max_h=min(3, room_h - 1), max_w=min(3, room_w - 1))
    seed_rotation = rng.randrange(4)
    seed_shape = np.rot90(base_shape, seed_rotation)
    seed_mask = _embed_seed((room_h, room_w), seed_shape, rng)
    return {
        "interior_shape": (room_h, room_w),
        "seed_mask": seed_mask,
        "required_rotation": required_rotation,
    }


def _make_pair(spec: dict, seed_color: int, decor_color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    room_h, room_w = spec["interior_shape"]
    seed_mask = spec["seed_mask"]
    output_patch = np.full((room_h, room_w), decor_color, dtype=int)
    output_patch[seed_mask] = seed_color
    template_patch = np.rot90(output_patch, -spec["required_rotation"])
    room_input = np.full((room_h, room_w), BG, dtype=int)
    room_input[seed_mask] = seed_color
    return room_input, template_patch, output_patch


def _draw_room(grid: np.ndarray, outer_rect: tuple[int, int, int, int], room_patch: np.ndarray) -> None:
    r0, c0, r1, c1 = outer_rect
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 + 1 : r1, c0 + 1 : c1] = room_patch


def _draw_patch(grid: np.ndarray, rect: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, r1, c1 = rect
    assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    n_rooms = rng.randint(2, 4)
    palette = [c for c in range(10) if c not in {BG, WALL}]
    if len(palette) < 2 * n_rooms:
        return None
    rng.shuffle(palette)
    seed_colors = palette[:n_rooms]
    decor_colors = palette[n_rooms : 2 * n_rooms]

    specs = [_room_spec(rng) for _ in range(n_rooms)]

    template_rects = []
    occupied = []
    for spec in specs:
        rect = _place_template_rect((h, w), spec["interior_shape"], occupied, rng)
        if rect is None:
            return None
        template_rects.append(rect)
        occupied.append(rect)

    room_rects = []
    for spec in specs:
        room_h, room_w = spec["interior_shape"]
        outer_shape = (room_h + 2, room_w + 2)
        rect = _place_room_rect((h, w), outer_shape, occupied, rng)
        if rect is None:
            return None
        room_rects.append(rect)
        occupied.append(rect)

    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()
    for spec, seed_color, decor_color, template_rect, room_rect in zip(
        specs, seed_colors, decor_colors, template_rects, room_rects
    ):
        room_input, template_patch, output_patch = _make_pair(spec, seed_color, decor_color)
        _draw_patch(grid, template_rect, template_patch)
        _draw_patch(expected, template_rect, template_patch)
        _draw_room(grid, room_rect, room_input)
        _draw_room(expected, room_rect, output_patch)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
