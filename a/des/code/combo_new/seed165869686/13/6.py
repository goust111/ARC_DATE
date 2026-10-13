import random
from itertools import combinations

import numpy as np


BLACK = 0
GRAY = 5
MAROON = 9
ROOM_SIDES = ("top", "bottom", "left", "right")
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(tuple(cells))
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _is_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    stack = [next(iter(cell_set))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cell_set)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    _, _, r1, c1 = _bbox(shape)
    return r1 + 1, c1 + 1


def _flip_ud(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _ = _shape_dims(shape)
    return tuple(sorted((h - 1 - r, c) for r, c in shape))


def _flip_lr(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _shape_dims(shape)
    return tuple(sorted((r, w - 1 - c) for r, c in shape))


def _enumerate_shapes() -> tuple[tuple[tuple[int, int], ...], ...]:
    universe = [(r, c) for r in range(4) for c in range(4)]
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for size in range(2, 5):
        for combo in combinations(universe, size):
            norm = _normalize(combo)
            if _is_connected(norm):
                shapes.add(norm)
    return tuple(sorted(shapes, key=lambda shape: (len(shape), _shape_dims(shape), shape)))


SHAPES = _enumerate_shapes()


def _room_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) == border:
            rooms.append((r0, c0, r1, c1))
    return sorted(rooms)


def _touching_side(coords: list[tuple[int, int]], inner_h: int, inner_w: int) -> str:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    sides = []
    if min(rows) == 0:
        sides.append("top")
    if max(rows) == inner_h - 1:
        sides.append("bottom")
    if min(cols) == 0:
        sides.append("left")
    if max(cols) == inner_w - 1:
        sides.append("right")
    assert len(sides) == 1, f"expected one touched wall, got {sides}"
    return sides[0]


def _move_object(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    inner_h: int,
    inner_w: int,
    side: str,
    has_marker: bool,
) -> list[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    if side == "left":
        shifted = [(r, c + (inner_w - 1 - max(cols))) for r, c in coords]
    elif side == "right":
        shifted = [(r, c - min(cols)) for r, c in coords]
    elif side == "top":
        shifted = [(r + (inner_h - 1 - max(rows)), c) for r, c in coords]
    else:
        shifted = [(r - min(rows), c) for r, c in coords]

    if not has_marker:
        return sorted(shifted)

    sr0, sc0, sr1, sc1 = _bbox(shifted)
    if side in ("left", "right"):
        shifted = [(sr0 + sr1 - r, c) for r, c in shifted]
    else:
        shifted = [(r, sc0 + sc1 - c) for r, c in shifted]
    return sorted(shifted)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each wall-hugging room object to the opposite wall, with maroon signs flipping its orientation.

    Input structure:
    - The background is black.
    - Several closed rectangular rooms are drawn as one-pixel-thick grey frames.
    - Each room contains one small monochrome 2-4 pixel object touching exactly one inner wall.
    - Some rooms also contain one maroon sign pixel away from the object.

    Rule:
    - For each grey rectangular room, detect which inner wall the object touches.
    - Remove the object from its original cells and translate it to the opposite inner wall, preserving its position along the wall.
    - If the room contains a maroon sign, flip the translated object across the axis perpendicular to the launch direction.
    - Clear the maroon sign to black and keep the grey room border unchanged.

    Key insight:
    - The main rule is an opposite-wall translation inside each enclosed rectangle, while the maroon sign only modifies the object's final orientation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == MAROON] = BLACK

    for r0, c0, r1, c1 in _room_bboxes(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = interior.shape
        object_mask = (interior != BLACK) & (interior != GRAY) & (interior != MAROON)
        object_coords = [tuple(map(int, pt)) for pt in np.argwhere(object_mask)]
        if not object_coords:
            continue

        colors = {int(interior[r, c]) for r, c in object_coords}
        assert len(colors) == 1, "expected one monochrome object per room"
        obj_color = next(iter(colors))
        side = _touching_side(object_coords, inner_h, inner_w)
        moved = _move_object(object_coords, inner_h, inner_w, side, bool((interior == MAROON).any()))

        for rr, cc in object_coords:
            out[r0 + 1 + rr, c0 + 1 + cc] = BLACK
        for rr, cc in moved:
            out[r0 + 1 + rr, c0 + 1 + cc] = obj_color

    return out


def _shape_candidates(inner_h: int, inner_w: int, side: str, has_marker: bool) -> list[tuple[tuple[int, int], ...]]:
    candidates: list[tuple[tuple[int, int], ...]] = []
    for shape in SHAPES:
        sh, sw = _shape_dims(shape)
        if side in ("left", "right"):
            if sh > inner_h - 2 or sw >= inner_w:
                continue
            if has_marker and shape == _flip_ud(shape):
                continue
        else:
            if sw > inner_w - 2 or sh >= inner_h:
                continue
            if has_marker and shape == _flip_lr(shape):
                continue
        candidates.append(shape)
    return candidates


def _place_shape(shape: tuple[tuple[int, int], ...], inner_h: int, inner_w: int, side: str) -> list[tuple[int, int]]:
    sh, sw = _shape_dims(shape)
    if side == "left":
        r0 = random.randint(1, inner_h - sh - 1)
        c0 = 0
    elif side == "right":
        r0 = random.randint(1, inner_h - sh - 1)
        c0 = inner_w - sw
    elif side == "top":
        r0 = 0
        c0 = random.randint(1, inner_w - sw - 1)
    else:
        r0 = inner_h - sh
        c0 = random.randint(1, inner_w - sw - 1)
    return sorted((r0 + r, c0 + c) for r, c in shape)


def _marker_candidates(
    inner_h: int,
    inner_w: int,
    object_coords: list[tuple[int, int]],
    moved_coords: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    obj_set = set(object_coords)
    moved_set = set(moved_coords)
    candidates: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            if (r, c) in obj_set or (r, c) in moved_set:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in object_coords):
                continue
            candidates.append((r, c))
    return candidates


def _draw_room_border(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY


def _room_count() -> int:
    return random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2], k=1)[0]


def _grid_bounds(room_count: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if room_count == 1:
        return (12, 22), (12, 22), (7, 11)
    if room_count == 2:
        return (16, 26), (16, 26), (7, 10)
    if room_count == 3:
        return (20, 30), (18, 30), (7, 9)
    return (22, 30), (22, 30), (7, 9)


def _place_rooms(h: int, w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    bboxes: list[tuple[int, int, int, int]] = []
    for rh, rw in sorted(sizes, key=lambda x: x[0] * x[1], reverse=True):
        candidates = []
        for r0 in range(1, h - rh):
            for c0 in range(1, w - rw):
                if occupied[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1] = True
        bboxes.append((r0, c0, r0 + rh - 1, c0 + rw - 1))
    random.shuffle(bboxes)
    return bboxes


def _marker_flags(room_count: int) -> list[bool]:
    if room_count == 1:
        return [bool(random.getrandbits(1))]
    marker_total = room_count // 2 if room_count % 2 == 0 else random.choice([room_count // 2, room_count // 2 + 1])
    flags = [True] * marker_total + [False] * (room_count - marker_total)
    random.shuffle(flags)
    return flags


def _build_room_spec(
    bbox: tuple[int, int, int, int],
    side: str,
    has_marker: bool,
    color: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    candidates = _shape_candidates(inner_h, inner_w, side, has_marker)
    if not candidates:
        return None

    for _ in range(80):
        shape = random.choice(candidates)
        obj_coords = _place_shape(shape, inner_h, inner_w, side)
        moved_coords = _move_object(obj_coords, inner_h, inner_w, side, has_marker)
        marker = None
        if has_marker:
            marker_options = _marker_candidates(inner_h, inner_w, obj_coords, moved_coords)
            if not marker_options:
                continue
            marker = random.choice(marker_options)
        return {
            "bbox": bbox,
            "side": side,
            "has_marker": has_marker,
            "color": color,
            "object": obj_coords,
            "moved": moved_coords,
            "marker": marker,
        }
    return None


def _render(specs: list[dict[str, object]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BLACK, dtype=int)
    out = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        bbox = spec["bbox"]
        color = int(spec["color"])
        _draw_room_border(inp, bbox)
        _draw_room_border(out, bbox)
        r0, c0, _r1, _c1 = bbox
        for rr, cc in spec["object"]:
            inp[r0 + 1 + rr, c0 + 1 + cc] = color
        if spec["marker"] is not None:
            mr, mc = spec["marker"]
            inp[r0 + 1 + mr, c0 + 1 + mc] = MAROON
        for rr, cc in spec["moved"]:
            out[r0 + 1 + rr, c0 + 1 + cc] = color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    room_count = _room_count()
    h_range, w_range, room_range = _grid_bounds(room_count)
    h = random.randint(*h_range)
    w = random.randint(*w_range)
    sizes = [(random.randint(*room_range), random.randint(*room_range)) for _ in range(room_count)]
    bboxes = _place_rooms(h, w, sizes)
    if bboxes is None:
        return None

    sides = random.sample(ROOM_SIDES, room_count)
    flags = _marker_flags(room_count)
    colors = random.sample(OBJECT_COLORS, room_count)

    specs: list[dict[str, object]] = []
    for bbox, side, has_marker, color in zip(bboxes, sides, flags, colors):
        spec = _build_room_spec(bbox, side, has_marker, color)
        if spec is None:
            return None
        specs.append(spec)

    inp, expected = _render(specs, (h, w))
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
