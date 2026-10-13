import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
FRAGMENT_COLORS = [2, 4, 5, 6, 7, 8, 9]
FRAGMENT_SHAPES = [(1, 1), (1, 3), (3, 1), (3, 3)]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _is_room_union(grid: np.ndarray, pts: list[tuple[int, int]], bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    if r1 - r0 + 1 < 7 or c1 - c0 + 1 < 7:
        return False
    if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in pts):
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    non_bg_non_green = (sub != bg) & (sub != GREEN)
    if int(np.count_nonzero(non_bg_non_green)) != len(pts):
        return False

    inside = sub[1:-1, 1:-1]
    if int(np.count_nonzero(inside == GREEN)) != 1:
        return False
    if np.any((inside != bg) & (inside != GREEN)):
        return False

    perimeter_values = [int(grid[r, c]) for r, c in _perimeter_cells((r0, c0, r1, c1))]
    gap_count = sum(value == bg for value in perimeter_values)
    marker_count = sum(value not in (bg, BLUE) for value in perimeter_values)
    blue_count = sum(value == BLUE for value in perimeter_values)
    if gap_count not in (1, 2):
        return False
    if marker_count != 2 * gap_count:
        return False
    return blue_count > 0


def _extract_room(grid: np.ndarray, pts: list[tuple[int, int]], bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(pts)
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    landmark_local = np.argwhere(inside == GREEN)
    assert len(landmark_local) == 1, "each room must contain one green landmark"
    lr, lc = map(int, landmark_local[0])
    landmark = (r0 + 1 + lr, c0 + 1 + lc)

    marker_cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in _perimeter_cells((r0, c0, r1, c1)):
        value = int(grid[r, c])
        if value not in (bg, BLUE):
            marker_cells[value].append((r, c))
    assert marker_cells, "each room must expose at least one marked door"
    return {
        "bbox": (r0, c0, r1, c1),
        "landmark": landmark,
        "markers": dict(marker_cells),
    }


def _extract_fragment(grid: np.ndarray, pts: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(pts)
    color = int(grid[pts[0]])
    assert color not in (BG, BLUE, GREEN)
    assert all(int(grid[r, c]) == color for r, c in pts), "fragment must be monochrome"
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h <= 3 and w <= 3, "fragment must be at most 3x3"
    assert len(pts) == h * w, "fragment must be a solid rectangle"
    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "area": h * w,
        "color": color,
        "pts": pts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each monochrome fragment to the room whose door markers share its color,
    erase the markers and source fragment, then center the fragment on the room's
    green landmark. Larger fragments are painted first when a room receives more
    than one fragment.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    mask = (g != bg) & (g != GREEN)
    all_components = _components(mask)
    blue_components = [pts for pts in all_components if any(int(g[r, c]) == BLUE for r, c in pts)]
    fragment_components = [pts for pts in all_components if all(int(g[r, c]) != BLUE for r, c in pts)]

    candidates: list[tuple[frozenset[int], dict]] = []
    for i, pts in enumerate(blue_components):
        if _is_room_union(g, pts, bg):
            candidates.append((frozenset({i}), _extract_room(g, pts, bg)))
    for i in range(len(blue_components)):
        for j in range(i + 1, len(blue_components)):
            pts = blue_components[i] + blue_components[j]
            if _is_room_union(g, pts, bg):
                candidates.append((frozenset({i, j}), _extract_room(g, pts, bg)))

    rooms: list[dict] | None = None
    target = set(range(len(blue_components)))

    def search(used: set[int], picked: list[dict]) -> bool:
        nonlocal rooms
        if used == target:
            rooms = picked.copy()
            return True
        next_idx = min(target - used)
        for comp_ids, room in candidates:
            if next_idx not in comp_ids or not comp_ids.isdisjoint(used):
                continue
            picked.append(room)
            if search(used | set(comp_ids), picked):
                return True
            picked.pop()
        return False

    assert search(set(), []), "could not partition blue components into rooms"
    assert rooms is not None

    fragments = [_extract_fragment(g, pts) for pts in fragment_components]

    assert rooms, "expected at least one room"
    assert fragments, "expected at least one fragment"

    color_to_room: dict[int, int] = {}
    for room_idx, room in enumerate(rooms):
        for color in room["markers"]:
            if color in color_to_room:
                assert color_to_room[color] == room_idx, "door colors must map unambiguously"
            color_to_room[color] = room_idx

    out = g.copy()
    room_fragments: dict[int, list[dict]] = defaultdict(list)

    for room in rooms:
        for pts in room["markers"].values():
            for r, c in pts:
                out[r, c] = bg

    for fragment in fragments:
        for r, c in fragment["pts"]:
            out[r, c] = bg
        assert fragment["color"] in color_to_room, "every fragment color must match a door"
        room_fragments[color_to_room[fragment["color"]]].append(fragment)

    for room_idx, room in enumerate(rooms):
        anchor_r, anchor_c = room["landmark"]
        r0, c0, r1, c1 = room["bbox"]
        for fragment in sorted(room_fragments.get(room_idx, []), key=lambda item: item["area"], reverse=True):
            fh, fw = fragment["shape"]
            top = anchor_r - fh // 2
            left = anchor_c - fw // 2
            assert r0 < top and top + fh - 1 < r1
            assert c0 < left and left + fw - 1 < c1
            out[top : top + fh, left : left + fw] = fragment["color"]

    return out


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_box(
    rng: random.Random,
    blocked: np.ndarray,
    box_h: int,
    box_w: int,
    margin: int,
    attempts: int = 200,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    if box_h + 2 * margin > h or box_w + 2 * margin > w:
        return None
    for _ in range(attempts):
        r0 = rng.randint(margin, h - box_h - margin)
        c0 = rng.randint(margin, w - box_w - margin)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        rr0 = max(0, r0 - margin)
        cc0 = max(0, c0 - margin)
        rr1 = min(h - 1, r1 + margin)
        cc1 = min(w - 1, c1 + margin)
        if not blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            return r0, c0
    return None


def _choose_room_shape(rng: random.Random) -> tuple[int, int]:
    return rng.randint(7, 10), rng.randint(7, 11)


def _choose_fragment_shapes(rng: random.Random, door_count: int) -> list[tuple[int, int]]:
    if door_count == 1:
        return [rng.choice(FRAGMENT_SHAPES)]

    areas = rng.sample([1, 3, 9], 2)
    shapes: list[tuple[int, int]] = []
    for area in areas:
        if area == 1:
            shapes.append((1, 1))
        elif area == 3:
            shapes.append(rng.choice([(1, 3), (3, 1)]))
        else:
            shapes.append((3, 3))
    rng.shuffle(shapes)
    return shapes


def _pick_doors(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    colors: list[int],
) -> list[tuple[str, int, int]]:
    r0, c0, r1, c1 = bbox
    sides = rng.sample(["top", "bottom", "left", "right"], len(colors))
    doors: list[tuple[str, int, int]] = []
    for side, color in zip(sides, colors):
        if side in ("top", "bottom"):
            pos = rng.randint(c0 + 2, c1 - 2)
        else:
            pos = rng.randint(r0 + 2, r1 - 2)
        doors.append((side, pos, color))
    return doors


def _draw_room(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    doors: list[tuple[str, int, int]],
    landmark: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    for side, pos, color in doors:
        if side == "top":
            grid[r0, pos] = BG
            grid[r0, pos - 1] = color
            grid[r0, pos + 1] = color
        elif side == "bottom":
            grid[r1, pos] = BG
            grid[r1, pos - 1] = color
            grid[r1, pos + 1] = color
        elif side == "left":
            grid[pos, c0] = BG
            grid[pos - 1, c0] = color
            grid[pos + 1, c0] = color
        else:
            grid[pos, c1] = BG
            grid[pos - 1, c1] = color
            grid[pos + 1, c1] = color
    grid[landmark] = GREEN


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    grid = np.full((h, w), BG, dtype=int)

    room_count = rng.choice([1, 2, 3])
    door_counts = [rng.choice([1, 2]) for _ in range(room_count)]
    total_doors = sum(door_counts)
    if total_doors > len(FRAGMENT_COLORS):
        return None

    room_blocked = np.zeros((h, w), dtype=bool)
    room_specs: list[dict] = []
    for door_count in door_counts:
        room_h, room_w = _choose_room_shape(rng)
        pos = _place_box(rng, room_blocked, room_h, room_w, margin=2, attempts=250)
        if pos is None:
            return None
        r0, c0 = pos
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        room_specs.append({"bbox": bbox, "door_count": door_count})
        _mark_blocked(room_blocked, bbox, pad=2)

    colors = rng.sample(FRAGMENT_COLORS, total_doors)
    color_idx = 0
    fragment_jobs: list[dict] = []
    for room in room_specs:
        bbox = room["bbox"]
        door_count = room["door_count"]
        room_colors = colors[color_idx : color_idx + door_count]
        color_idx += door_count

        fragment_shapes = _choose_fragment_shapes(rng, door_count)
        doors = _pick_doors(rng, bbox, room_colors)

        r0, c0, r1, c1 = bbox
        landmark = (
            rng.randint(r0 + 2, r1 - 2),
            rng.randint(c0 + 2, c1 - 2),
        )
        _draw_room(grid, bbox, doors, landmark)

        room["landmark"] = landmark
        room["colors"] = room_colors
        room["shapes"] = fragment_shapes
        for color, shape in zip(room_colors, fragment_shapes):
            fragment_jobs.append({"color": color, "shape": shape})

    fragment_jobs.sort(key=lambda item: item["shape"][0] * item["shape"][1], reverse=True)
    fragment_blocked = np.zeros((h, w), dtype=bool)
    for room in room_specs:
        _mark_blocked(fragment_blocked, room["bbox"], pad=1)

    for job in fragment_jobs:
        fh, fw = job["shape"]
        pos = _place_box(rng, fragment_blocked, fh, fw, margin=1, attempts=250)
        if pos is None:
            return None
        r0, c0 = pos
        grid[r0 : r0 + fh, c0 : c0 + fw] = job["color"]
        _mark_blocked(fragment_blocked, (r0, c0, r0 + fh - 1, c0 + fw - 1), pad=1)

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
