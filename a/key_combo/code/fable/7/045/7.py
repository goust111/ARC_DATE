from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _bbox_size(box: BBox) -> tuple[int, int]:
    return _bbox_h(box), _bbox_w(box)


def _translate_bbox(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _translate_coords(
    coords: list[tuple[int, int]], *, dr: int = 0, dc: int = 0
) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _axis_center(box: BBox, axis: int) -> float:
    if axis == 0:
        return (box[0] + box[2]) / 2
    return (box[1] + box[3]) / 2


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            box = _bbox_from_coords(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "area": len(coords),
                }
            )

    return components


def _is_solid_rect(grid: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    return obj["area"] == _bbox_h(obj["bbox"]) * _bbox_w(obj["bbox"]) and np.all(
        grid[r0 : r1 + 1, c0 : c1 + 1] == color
    )


def _is_hollow_frame(grid: np.ndarray, bg: int, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    h = _bbox_h(obj["bbox"])
    w = _bbox_w(obj["bbox"])
    if h < 3 or w < 3:
        return False
    if obj["area"] != 2 * h + 2 * w - 4:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(obj["color"])
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.all(patch[border] == color) and np.all(patch[~border] == bg)


def _touching_sides(box: BBox, shape: tuple[int, int]) -> list[str]:
    h, w = shape
    r0, c0, r1, c1 = box
    sides = []
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    return sides


def _paint_cells(out: np.ndarray, cells: list[tuple[int, int]], color: int, owner: np.ndarray, owner_id: int) -> None:
    for r, c in cells:
        assert owner[r, c] == -1, "objects must not overlap after motion"
        out[r, c] = color
        owner[r, c] = owner_id


def _cavity_bbox(box: BBox) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _can_enter_frame(obj: dict, frame: dict, axis: int, direction: int, steps_before_hit: int) -> int | None:
    if obj["is_frame"] or not obj["is_solid_rect"]:
        return None
    if frame["kind"] != "frame" or not frame["empty"]:
        return None

    cavity = frame["cavity_bbox"]
    if _bbox_size(obj["bbox"]) != _bbox_size(cavity):
        return None

    if axis == 0:
        if (obj["bbox"][1], obj["bbox"][3]) != (cavity[1], cavity[3]):
            return None
        shift = cavity[0] - obj["bbox"][0]
    else:
        if (obj["bbox"][0], obj["bbox"][2]) != (cavity[0], cavity[2]):
            return None
        shift = cavity[1] - obj["bbox"][1]

    travel = shift * direction
    if travel <= steps_before_hit:
        return None
    return int(shift)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide every object away from the edge pusher, stopping at blockers or entering an exactly matching empty frame.

    Input structure:
    - The grid has a dominant background and one solid square pusher touching exactly one grid edge.
    - All other foreground components lie on a shared horizontal or vertical motion band.
    - The movable components are solid monochrome blocks and hollow rectangular frames whose interiors are background.

    Rule:
    - Determine the motion axis and direction from the pusher: every other object slides away from it.
    - Resolve objects from farthest to nearest along that axis; each one moves until the next step would leave the grid or hit an already settled obstacle.
    - If the first obstacle is an empty hollow frame and the moving solid block has exactly the frame cavity's bbox size, it enters the cavity and fills it; a filled frame then acts as a solid blocker.

    Key insight:
    - The outcome depends on ordered simulation: far objects settle first, and matching a frame changes that frame from pass-through cavity to solid wall for later objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    components = _find_components(g, bg)
    assert components, "expected foreground objects"
    for obj in components:
        obj["is_frame"] = _is_hollow_frame(g, bg, obj)
        obj["is_solid_rect"] = _is_solid_rect(g, obj)

    pusher_candidates: list[tuple[str, dict]] = []
    for obj in components:
        if obj["is_frame"] or not obj["is_solid_rect"]:
            continue
        if _bbox_h(obj["bbox"]) != _bbox_w(obj["bbox"]):
            continue
        sides = _touching_sides(obj["bbox"], g.shape)
        if len(sides) == 1:
            pusher_candidates.append((sides[0], obj))

    assert len(pusher_candidates) == 1, "expected a unique edge-touching square pusher"
    side, pusher = pusher_candidates[0]
    if side == "left":
        axis, direction = 1, 1
    elif side == "right":
        axis, direction = 1, -1
    elif side == "top":
        axis, direction = 0, 1
    else:
        axis, direction = 0, -1

    movable = [obj for obj in components if obj is not pusher]
    assert any(obj["is_frame"] for obj in movable), "expected at least one hollow frame"
    movable.sort(key=lambda obj: _axis_center(obj["bbox"], axis), reverse=(direction > 0))

    out = np.full_like(g, bg)
    owner = np.full(g.shape, -1, dtype=int)
    obstacles: list[dict] = []

    pusher_id = 0
    _paint_cells(out, pusher["coords"], int(pusher["color"]), owner, pusher_id)
    obstacles.append(
        {
            "kind": "pusher",
            "empty": False,
            "bbox": pusher["bbox"],
            "cavity_bbox": None,
        }
    )

    for obj in movable:
        steps = 0
        hit_ids: set[int] | None = None
        hit_boundary = False

        while True:
            delta = direction * (steps + 1)
            dr = delta if axis == 0 else 0
            dc = delta if axis == 1 else 0
            next_coords = _translate_coords(obj["coords"], dr=dr, dc=dc)

            if any(not (0 <= r < h and 0 <= c < w) for r, c in next_coords):
                hit_boundary = True
                break

            hit_ids = {int(owner[r, c]) for r, c in next_coords if owner[r, c] != -1}
            if hit_ids:
                break

            steps += 1

        entry_shift = None
        frame_id = None
        if not hit_boundary and hit_ids is not None and len(hit_ids) == 1:
            frame_id = next(iter(hit_ids))
            entry_shift = _can_enter_frame(obj, obstacles[frame_id], axis, direction, steps)

        if entry_shift is not None:
            dr = entry_shift if axis == 0 else 0
            dc = entry_shift if axis == 1 else 0
            placed_coords = _translate_coords(obj["coords"], dr=dr, dc=dc)
            frame = obstacles[frame_id]
            cavity = frame["cavity_bbox"]
            assert cavity is not None

            cavity_cells = {
                (r, c)
                for r in range(cavity[0], cavity[2] + 1)
                for c in range(cavity[1], cavity[3] + 1)
            }
            assert set(placed_coords) == cavity_cells, "entering block must exactly fill the frame cavity"
            for r, c in placed_coords:
                assert owner[r, c] == -1, "frame cavity must be empty before entry"
                out[r, c] = int(obj["color"])

            fr0, fc0, fr1, fc1 = frame["bbox"]
            owner[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_id
            frame["kind"] = "filled_frame"
            frame["empty"] = False
            continue

        delta = direction * steps
        dr = delta if axis == 0 else 0
        dc = delta if axis == 1 else 0
        placed_coords = _translate_coords(obj["coords"], dr=dr, dc=dc)
        placed_bbox = _translate_bbox(obj["bbox"], dr, dc)
        new_id = len(obstacles)
        _paint_cells(out, placed_coords, int(obj["color"]), owner, new_id)

        if obj["is_frame"]:
            obstacles.append(
                {
                    "kind": "frame",
                    "empty": True,
                    "bbox": placed_bbox,
                    "cavity_bbox": _cavity_bbox(placed_bbox),
                }
            )
        else:
            obstacles.append(
                {
                    "kind": "solid",
                    "empty": False,
                    "bbox": placed_bbox,
                    "cavity_bbox": None,
                }
            )

    return out


BG = 0
COLORS = list(range(1, 10))
PAIR_COUNT_CHOICES = (2, 2, 3)
PAIR_FLAG_PATTERNS = (
    (True, False),
    (False, True),
    (True, True),
    (False, False),
)
MAX_GENERATION_TRIES = 400


def _draw_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _choose_match_flags(pair_count: int) -> list[bool]:
    if pair_count == 2:
        return list(random.choice(PAIR_FLAG_PATTERNS))

    flags = [random.choice((False, True)) for _ in range(pair_count)]
    if random.random() < 0.7:
        if all(flags):
            flags[random.randrange(pair_count)] = False
        elif not any(flags):
            flags[random.randrange(pair_count)] = True
    return flags


def _apply_orientation(grid: np.ndarray) -> np.ndarray:
    variant = random.randrange(4)
    if variant == 0:
        return grid
    if variant == 1:
        return np.fliplr(grid).copy()
    if variant == 2:
        return grid.T.copy()
    return np.fliplr(grid).T.copy()


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        pair_count = random.choice(PAIR_COUNT_CHOICES)
        band_h = random.randint(2, 4)
        pusher_size = band_h
        match_flags = _choose_match_flags(pair_count)

        items: list[tuple[str, int, int | None]] = []
        for is_match in match_flags:
            cavity_w = random.randint(2, 5)
            if is_match:
                solid_w = cavity_w
            else:
                non_match = [w for w in range(1, 6) if w != cavity_w]
                solid_w = random.choice(non_match)
            items.append(("solid", solid_w, None))
            items.append(("frame", cavity_w + 2, cavity_w))

        left_gap = random.randint(1, 4)
        right_gap = random.randint(1, 5)
        inner_gaps = [random.randint(1, 4) for _ in range(len(items) - 1)]
        total_width = pusher_size + left_gap + right_gap + sum(size for _, size, _ in items) + sum(inner_gaps)
        if total_width > 30:
            continue

        outer_h = band_h + 2
        height = random.randint(outer_h + 2, min(outer_h + 10, 18))
        band_r = random.randint(1, height - band_h - 1)

        grid = np.zeros((height, total_width), dtype=int)
        color_count = 1 + len(items)
        palette = random.sample(COLORS, color_count)
        pusher_color = palette[0]
        item_colors = palette[1:]

        _draw_rect(grid, band_r, 0, pusher_size, pusher_size, pusher_color)

        x = pusher_size + left_gap
        for idx, (kind, size, cavity_w) in enumerate(items):
            color = item_colors[idx]
            if kind == "solid":
                _draw_rect(grid, band_r, x, band_h, size, color)
                x += size
            else:
                _draw_frame(grid, band_r - 1, x, outer_h, size, color)
                assert cavity_w is not None
                x += size
            if idx + 1 < len(items):
                x += inner_gaps[idx]

        assert x + right_gap == total_width

        oriented = _apply_orientation(grid)
        try:
            out = main(oriented)
        except AssertionError:
            continue

        if np.array_equal(oriented, out):
            continue
        if max(oriented.shape) > 30 or max(out.shape) > 30:
            continue

        return oriented

    raise RuntimeError("failed to generate a valid input")
