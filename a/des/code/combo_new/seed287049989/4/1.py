import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_CW_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -turns % 4)


def _find_rooms(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == wall):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        if len(comp) != 2 * (height + width) - 4:
            continue
        comp_set = set(comp)
        valid = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if ((r, c) in comp_set) != on_border:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            rooms.append((r0, c0, r1, c1))
    return sorted(rooms)


def _extract_templates(
    grid: np.ndarray, bg: int, wall: int, patch_shape: tuple[int, int]
) -> dict[int, np.ndarray]:
    h, w = grid.shape
    templates: dict[int, np.ndarray] = {}
    for color in map(int, np.unique(grid)):
        if color in (bg, wall):
            continue
        comps = _components(grid == color)
        candidates = []
        for comp in comps:
            if _touches_border(comp, h, w):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != patch_shape:
                continue
            candidates.append(comp)
        if len(candidates) != 1:
            continue
        comp = candidates[0]
        r0, c0, _, _ = _bbox(comp)
        patch = np.full(patch_shape, bg, dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = color
        templates[color] = patch
    return templates


def _room_marker(
    room: tuple[int, int, int, int], markers: list[tuple[int, int, int]], h: int, w: int
) -> tuple[str, int]:
    r0, c0, r1, c1 = room
    hits: list[tuple[str, int]] = []
    for mr, mc, color in markers:
        if mr == 0 and r0 == 1 and c0 < mc < c1:
            hits.append(("top", color))
        elif mr == h - 1 and r1 == h - 2 and c0 < mc < c1:
            hits.append(("bottom", color))
        elif mc == 0 and c0 == 1 and r0 < mr < r1:
            hits.append(("left", color))
        elif mc == w - 1 and c1 == w - 2 and r0 < mr < r1:
            hits.append(("right", color))
    assert len(hits) == 1, "each room must have exactly one border marker"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room with the color-matched corner template rotated by its border marker side.

    Input structure:
    - The background is black.
    - Several one-cell-thick gray rectangular frames form closed rooms whose interiors are empty.
    - Single colored marker pixels lie on the outer border, each directly outside one room side.
    - Away from the border, a corner area contains one connected monochrome template per color, separated by black gaps.

    Rule:
    - Detect every gray rectangular room and read its unique border marker.
    - The marker color selects the matching monochrome template of the same color from the corner area.
    - Rotate that template by 0/90/180/270 degrees when the marker is on the top/right/bottom/left side respectively, then copy the rotated patch into the room interior.
    - Leave walls, markers, and the template preview area unchanged.

    Key insight:
    - Each border marker encodes two independent controls at once: its color picks the template, and its side relative to the room picks the rotation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "grid must contain foreground colors"
    wall = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    rooms = _find_rooms(g, wall)
    assert rooms, "expected at least one rectangular room"

    patch_shapes = {(r1 - r0 - 1, c1 - c0 - 1) for r0, c0, r1, c1 in rooms}
    assert len(patch_shapes) == 1, "all rooms must share one interior size"
    patch_shape = next(iter(patch_shapes))

    markers = []
    border_cells = (
        [(0, c) for c in range(w)]
        + [(h - 1, c) for c in range(w)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    for r, c in border_cells:
        color = int(g[r, c])
        if color not in (bg, wall):
            markers.append((r, c, color))
    assert markers, "expected colored border markers"

    templates = _extract_templates(g, bg, wall, patch_shape)
    assert templates, "expected at least one template"

    out = g.copy()
    for room in rooms:
        side, color = _room_marker(room, markers, h, w)
        assert color in templates, "marker color must match a template"
        patch = _rotate_cw(templates[color], SIDE_TO_CW_TURNS[side])
        r0, c0, r1, c1 = room
        out[r0 + 1 : r1, c0 + 1 : c1] = patch

    return out


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _template_signature(mask: np.ndarray) -> bytes:
    rots = [np.rot90(mask, k) for k in range(4)]
    return min(rot.tobytes() for rot in rots)


def _random_template_mask(rng: random.Random, size: int) -> np.ndarray:
    thickness_choices = [1, 2] if size >= 4 else [1]
    max_area = int(0.68 * size * size)
    for _ in range(200):
        thick = rng.choice(thickness_choices)
        mask = np.zeros((size, size), dtype=bool)
        mask[:thick, :] = True
        mask[:, :thick] = True

        for _ in range(rng.randint(0, 2)):
            if rng.random() < 0.5:
                rr = rng.randint(thick, size - 1)
                hh = rng.randint(1, min(2, size - rr))
                ll = rng.randint(thick + 1, size)
                mask[rr : rr + hh, :ll] = True
            else:
                cc = rng.randint(thick, size - 1)
                ww = rng.randint(1, min(2, size - cc))
                ll = rng.randint(thick + 1, size)
                mask[:ll, cc : cc + ww] = True

        mask = _rotate_cw(mask, rng.randrange(4))
        if int(mask.sum()) < size + 2 or int(mask.sum()) > max_area:
            continue
        if len({_rotate_cw(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to create a non-degenerate template")


def _layout_slots(rng: random.Random, count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if count == 2:
        rows, cols = (1, 2) if rng.random() < 0.5 else (2, 1)
    elif count <= 2:
        rows, cols = 1, count
    else:
        rows, cols = 2, 2
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    if count < len(slots):
        slots = rng.sample(slots, count)
    return rows, cols, slots


def _room_bbox_for_side(
    rng: random.Random,
    side: str,
    grid_h: int,
    grid_w: int,
    outer: int,
    reserve_box: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    rr0, rc0, _, _ = reserve_box
    for _ in range(200):
        if side == "top":
            r0 = 1
            c0 = rng.randint(2, grid_w - outer - 2)
        elif side == "bottom":
            r0 = grid_h - outer - 1
            cmax = rc0 - outer - 1
            if cmax < 2:
                return None
            c0 = rng.randint(2, cmax)
        elif side == "left":
            r0 = rng.randint(2, grid_h - outer - 2)
            c0 = 1
        else:
            rmax = rr0 - outer - 1
            if rmax < 2:
                return None
            r0 = rng.randint(2, rmax)
            c0 = grid_w - outer - 1

        box = (r0, c0, r0 + outer - 1, c0 + outer - 1)
        if _boxes_too_close(box, reserve_box, gap=1):
            continue
        if any(_boxes_too_close(box, other, gap=1) for other in existing):
            continue
        return box
    return None


def _marker_position(
    rng: random.Random, side: str, box: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return 0, rng.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1 + 1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), 0
    return rng.randint(r0 + 1, r1 - 1), c1 + 1


def _render_expected(
    grid: np.ndarray,
    rooms: list[dict],
    templates: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["box"]
        color = room["color"]
        turns = SIDE_TO_CW_TURNS[room["side"]]
        patch = np.where(_rotate_cw(templates[color], turns), color, BG)
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    room_size = rng.randint(3, 5)
    room_outer = room_size + 2
    room_count = rng.randint(2, 4)
    room_sides = rng.sample(list(SIDES), room_count)

    template_count = rng.randint(2, min(4, room_count))
    template_colors = rng.sample(list(COLOR_POOL), template_count)

    rows, cols, slots = _layout_slots(rng, template_count)
    reserve_h = rows * room_size + (rows - 1)
    reserve_w = cols * room_size + (cols - 1)

    min_h = max(room_outer + reserve_h + 3, 2 * room_outer + 3)
    min_w = max(room_outer + reserve_w + 3, 2 * room_outer + 3)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    reserve_r0 = grid_h - reserve_h - 1
    reserve_c0 = grid_w - reserve_w - 1
    reserve_box = (reserve_r0, reserve_c0, grid_h - 2, grid_w - 2)

    template_masks: dict[int, np.ndarray] = {}
    seen_templates: set[bytes] = set()
    for color in template_colors:
        for _ in range(100):
            mask = _random_template_mask(rng, room_size)
            sig = _template_signature(mask)
            if sig not in seen_templates:
                seen_templates.add(sig)
                template_masks[color] = mask
                break
        else:
            return None

    slot_order = slots[:]
    rng.shuffle(slot_order)
    for color, (sr, sc) in zip(template_colors, slot_order):
        r0 = reserve_r0 + sr * (room_size + 1)
        c0 = reserve_c0 + sc * (room_size + 1)
        patch = template_masks[color]
        grid[r0 : r0 + room_size, c0 : c0 + room_size][patch] = color

    boxes: list[tuple[int, int, int, int]] = []
    rooms: list[dict] = []
    assignment = template_colors[:]
    while len(assignment) < room_count:
        assignment.append(rng.choice(template_colors))
    rng.shuffle(assignment)

    for side, color in zip(room_sides, assignment):
        box = _room_bbox_for_side(rng, side, grid_h, grid_w, room_outer, reserve_box, boxes)
        if box is None:
            return None
        boxes.append(box)
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = WALL
        grid[r0 + 1 : r1, c0 + 1 : c1] = BG
        mr, mc = _marker_position(rng, side, box)
        grid[mr, mc] = color
        rooms.append({"box": box, "side": side, "color": color})

    expected = _render_expected(grid, rooms, template_masks)

    wall_count = int(np.count_nonzero(grid == WALL))
    largest_other = max(int(np.count_nonzero(grid == color)) for color in template_colors)
    if wall_count <= largest_other:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt(random.Random())
        if sample is None:
            continue
        grid, _ = sample
        return grid
    raise RuntimeError("failed to generate a valid input")
