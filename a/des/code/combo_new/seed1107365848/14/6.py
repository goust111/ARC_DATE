from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Corner order matches the clockwise rotation turns encoded by the puzzle.
CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
CORNER_PATTERNS = (
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 3),
    (1, 3),
    (0, 1, 2),
    (0, 2, 3),
    (0, 1, 3),
)

BOX_OPTIONS = (
    (3, 3),
    (4, 4),
    (2, 4),
    (4, 2),
    (5, 5),
    (3, 5),
    (5, 3),
)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 80


def _corner_origin(name: str, h: int, w: int) -> tuple[int, int]:
    if name == "top_left":
        return 0, 0
    if name == "top_right":
        return 0, w - 2
    if name == "bottom_right":
        return h - 2, w - 2
    if name == "bottom_left":
        return h - 2, 0
    raise AssertionError(name)


def _tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    tight = _tight(np.asarray(mask, dtype=bool))
    return tight.tobytes(), tuple(map(int, tight.shape))


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    base = _mask_key(mask)
    return all(_mask_key(np.rot90(mask, k)) != base for k in (1, 2, 3))


def _connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _random_path(start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = target
    path = [(r, c)]
    while (r, c) != (tr, tc):
        steps = []
        if r < tr:
            steps.append((r + 1, c))
        elif r > tr:
            steps.append((r - 1, c))
        if c < tc:
            steps.append((r, c + 1))
        elif c > tc:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        path.append((r, c))
    return path


def _make_shape(h: int, w: int) -> np.ndarray | None:
    max_area = h * w - 1
    min_area = min(max_area, max(4, h + w - 2))
    if min_area > max_area:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        seeds = {
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        }
        order = list(seeds)
        random.shuffle(order)
        cells = {order[0]}
        for seed in order[1:]:
            target = random.choice(tuple(cells))
            cells.update(_random_path(seed, target))

        lower = max(min_area, len(cells))
        if lower > max_area:
            continue
        target_area = random.randint(lower, max_area)
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area or not _connected(cells):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _is_rotation_asymmetric(mask):
            continue
        return mask

    return None


def _find_corner_markers(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    turns_by_color: dict[int, int] = {}
    marker_mask = np.zeros((h, w), dtype=bool)
    empty_corners = 0

    for turns, name in enumerate(CORNER_NAMES):
        r0, c0 = _corner_origin(name, h, w)
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        color = int(patch[0, 0])
        if bool(np.all(patch == BG)):
            empty_corners += 1
            continue
        if color == BG or not bool(np.all(patch == color)):
            raise AssertionError("corner patches must be empty or solid 2x2 markers")
        if color in turns_by_color:
            raise AssertionError("corner marker colors must be distinct")
        turns_by_color[color] = turns
        marker_mask[r0 : r0 + 2, c0 : c0 + 2] = True

    assert turns_by_color
    assert empty_corners >= 1
    return turns_by_color, marker_mask


def _extract_components(grid: np.ndarray, ignore: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or ignore[r, c] or int(grid[r, c]) == BG:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or ignore[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[pts[:, 0] - int(r0), pts[:, 1] - int(c0)] = True
            components.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    return components


def _place_rotated_mask(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
    turns: int,
) -> None:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    rotated = np.rot90(mask, -turns)
    rh, rw = map(int, rotated.shape)
    row_delta = h - rh
    col_delta = w - rw
    if row_delta % 2 != 0 or col_delta % 2 != 0:
        raise AssertionError("rotation must preserve the bbox center on integer cells")

    nr0 = int(r0 + row_delta // 2)
    nc0 = int(c0 + col_delta // 2)
    nr1 = nr0 + rh
    nc1 = nc0 + rw
    if not (0 <= nr0 and 0 <= nc0 and nr1 <= out.shape[0] and nc1 <= out.shape[1]):
        raise AssertionError("rotated object left the grid")

    region = out[nr0:nr1, nc0:nc1]
    if bool(np.any(region[rotated] != BG)):
        raise AssertionError("rotated objects overlap")
    region[rotated] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each central monochrome object in place using the matching colored corner marker as its instruction.

    Input structure:
    - The background is Black.
    - Some corners contain a solid 2x2 monochrome marker block, with at least one corner left empty.
    - Away from the corners, the foreground is made of small monochrome 4-connected objects.
    - Every central object's color matches exactly one corner marker color.

    Rule:
    - Read the rotation encoded by the marker's corner position: top-left = 0 degrees, top-right = 90 degrees clockwise, bottom-right = 180 degrees, bottom-left = 270 degrees clockwise.
    - For each central object, find the marker with the same color.
    - Rotate that object's tight bounding-box mask by the indicated amount and place it back so the bounding-box center stays fixed.
    - Erase every corner marker to Black in the output.

    Key insight:
    - The corner position, not the color identity, secretly encodes the rotation amount; color only tells which instruction belongs to which object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 4 and g.shape[1] >= 4

    turns_by_color, marker_mask = _find_corner_markers(g)
    components = _extract_components(g, marker_mask)
    assert components

    out = np.zeros_like(g)
    for comp in components:
        color = int(comp["color"])
        if color not in turns_by_color:
            raise AssertionError("every object must match a corner marker color")
        _place_rotated_mask(out, comp["mask"], color, comp["bbox"], turns_by_color[color])
    return out


def _choose_corners() -> tuple[int, ...]:
    return random.choice(CORNER_PATTERNS)


def _choose_box(turns: int) -> tuple[int, int]:
    if turns in (1, 3) and random.random() < 0.6:
        rectangular = [dims for dims in BOX_OPTIONS if dims[0] != dims[1]]
        return random.choice(rectangular)
    return random.choice(BOX_OPTIONS)


def _build_objects(colors_by_turn: dict[int, int]) -> list[dict] | None:
    turns = list(colors_by_turn)
    object_count = random.randint(max(2, len(turns)), min(5, len(turns) + 2))
    assigned_turns = turns[:]
    while len(assigned_turns) < object_count:
        assigned_turns.append(random.choice(turns))
    random.shuffle(assigned_turns)

    objects: list[dict] = []
    for turn in assigned_turns:
        h, w = _choose_box(turn)
        mask = _make_shape(h, w)
        if mask is None:
            return None
        objects.append(
            {
                "turn": int(turn),
                "color": int(colors_by_turn[turn]),
                "bbox_hw": (int(h), int(w)),
                "square": int(max(h, w)),
                "mask": mask,
            }
        )
    return objects


def _render_expected(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in objects:
        r0, c0 = obj["bbox_rc"]
        h, w = obj["bbox_hw"]
        bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        _place_rotated_mask(out, obj["mask"], obj["color"], bbox, obj["turn"])
    return out


def _place_objects(grid: np.ndarray, objects: list[dict]) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)

    for obj in sorted(objects, key=lambda item: item["square"], reverse=True):
        side = int(obj["square"])
        max_r = h - side - 3
        max_c = w - side - 3
        if max_r < 2 or max_c < 2:
            return False

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            sr = random.randint(2, max_r)
            sc = random.randint(2, max_c)
            br0 = max(0, sr - 1)
            bc0 = max(0, sc - 1)
            br1 = min(h, sr + side + 1)
            bc1 = min(w, sc + side + 1)
            if bool(blocked[br0:br1, bc0:bc1].any()):
                continue

            obj["bbox_rc"] = (
                int(sr + (side - obj["bbox_hw"][0]) // 2),
                int(sc + (side - obj["bbox_hw"][1]) // 2),
            )
            blocked[br0:br1, bc0:bc1] = True
            placed = True
            break

        if not placed:
            return False

    for obj in objects:
        r0, c0 = obj["bbox_rc"]
        h0, w0 = obj["bbox_hw"]
        region = grid[r0 : r0 + h0, c0 : c0 + w0]
        region[obj["mask"]] = int(obj["color"])
    return True


def _gen_attempt() -> np.ndarray | None:
    corners = _choose_corners()
    colors = random.sample(PALETTE, len(corners))
    colors_by_turn = {int(turn): int(color) for turn, color in zip(corners, colors)}
    objects = _build_objects(colors_by_turn)
    if objects is None:
        return None

    base = 12 + 2 * len(objects)
    h = random.randint(base, min(30, base + 8))
    w = random.randint(base, min(30, base + 8))

    grid = np.zeros((h, w), dtype=int)
    if not _place_objects(grid, objects):
        return None

    for turn, color in colors_by_turn.items():
        r0, c0 = _corner_origin(CORNER_NAMES[int(turn)], h, w)
        grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)

    expected = _render_expected((h, w), objects)
    if np.array_equal(expected, grid):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-marker sample")
