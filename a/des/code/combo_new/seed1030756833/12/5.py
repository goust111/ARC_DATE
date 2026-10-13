from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = list(range(1, 10))
DIRECTIONS = ("above", "below", "left", "right")
VERTICAL_DIRECTIONS = ("above", "below")
HORIZONTAL_DIRECTIONS = ("left", "right")

GRID_SIZE_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (3, 5)
OBJECT_AREA_VALUES = list(range(3, 14))
MAX_OBJECT_BBOX = 5
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 500
MASK_ATTEMPTS = 700

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _axis_from_anchor(anchor: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = float(anchor[0]) - center_r
    dc = float(anchor[1]) - center_c
    assert abs(dr) != abs(dc), "anchor direction must choose a unique axis"
    if abs(dr) > abs(dc):
        return "vertical"
    return "horizontal"


def _extract_flipped_objects(grid: np.ndarray) -> list[dict]:
    comps_by_color: dict[int, list[dict]] = {}
    for comp in _find_components_by_color(grid):
        comps_by_color.setdefault(int(comp["color"]), []).append(comp)

    assert comps_by_color, "expected colored objects and anchors"
    objects: list[dict] = []
    for color, comps in comps_by_color.items():
        singles = [comp for comp in comps if int(comp["area"]) == 1]
        sprites = [comp for comp in comps if int(comp["area"]) > 1]
        assert len(singles) == 1 and len(sprites) == 1, "each color must have one object and one single-pixel anchor"

        anchor_coord = tuple(map(int, singles[0]["coords"][0]))
        sprite = sprites[0]
        mask = _component_mask(sprite)
        assert max(mask.shape) <= MAX_OBJECT_BBOX, "object bbox must be at most 5x5"

        axis = _axis_from_anchor(anchor_coord, sprite["bbox"])
        flipped = np.flipud(mask) if axis == "vertical" else np.fliplr(mask)
        r0, c0, _r1, _c1 = sprite["bbox"]
        objects.append(
            {
                "color": int(color),
                "mask": flipped.copy(),
                "area": int(sprite["area"]),
                "origin": (int(r0), int(c0)),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Flip same-colored anchored objects and stack them at the center by area.

    Input structure:
    - The background is black.
    - Each non-black color forms exactly one 4-connected multi-pixel object and
      one separate same-colored single-pixel anchor.
    - Object bounding boxes are no larger than 5x5, and anchors are isolated
      from their matching objects.

    Rule:
    - For each color, compare the anchor to the center of the matching object's
      bounding box and choose the dominant row or column direction.
    - Anchors above or below the object cause an up-down flip; anchors left or
      right of the object cause a left-right flip.
    - Sort the flipped objects by decreasing pixel area and draw them in that
      order on a new black square canvas, aligning each bounding-box center to
      the canvas center. Later objects overwrite earlier ones.
    - The output side length is twice the largest object bbox side plus one.

    Key insight:
    The isolated same-color pixel is a directional anchor, not noise; its
    relative position selects the mirror axis, and the transformed objects are
    repacked into one centered area-ordered stack rather than left in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "expected a non-empty grid"

    objects = _extract_flipped_objects(g)
    max_side = max(max(obj["mask"].shape) for obj in objects)
    side = int(2 * max_side + 1)
    assert side <= 30, "output must stay within ARC bounds"

    out = np.zeros((side, side), dtype=int)
    center = side // 2
    ordered = sorted(objects, key=lambda obj: (-int(obj["area"]), obj["origin"]))

    for obj in ordered:
        mask = obj["mask"]
        h, w = map(int, mask.shape)
        r0 = center - h // 2
        c0 = center - w // 2
        patch = out[r0 : r0 + h, c0 : c0 + w]
        patch[mask] = int(obj["color"])

    return out


# --------------- generator ---------------


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _is_asymmetric_for_axis(mask: np.ndarray, axis: str) -> bool:
    if axis == "vertical":
        return not np.array_equal(mask, np.flipud(mask))
    assert axis == "horizontal"
    return not np.array_equal(mask, np.fliplr(mask))


def _random_connected_mask(area: int, axis: str) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        cells = {(random.randrange(MAX_OBJECT_BBOX), random.randrange(MAX_OBJECT_BBOX))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < MAX_OBJECT_BBOX and 0 <= nc < MAX_OBJECT_BBOX and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        full = np.zeros((MAX_OBJECT_BBOX, MAX_OBJECT_BBOX), dtype=bool)
        for r, c in cells:
            full[r, c] = True
        mask = _trim_mask(full)
        h, w = map(int, mask.shape)
        if h < 2 or w < 2:
            continue
        if int(mask.sum()) == h * w:
            continue
        if axis == "vertical" and h < 2:
            continue
        if axis == "horizontal" and w < 2:
            continue
        if not _is_asymmetric_for_axis(mask, axis):
            continue
        return mask
    return None


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(r0), int(c0)])


def _touches_existing(occupied: np.ndarray, coords: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        r0, r1 = max(0, rr - 1), min(h, rr + 2)
        c0, c1 = max(0, cc - 1), min(w, cc + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return True
    return False


def _anchor_touches_object(anchor: tuple[int, int], coords: np.ndarray) -> bool:
    ar, ac = anchor
    return bool(np.any((np.abs(coords[:, 0] - ar) <= 1) & (np.abs(coords[:, 1] - ac) <= 1)))


def _direction_from_anchor(anchor: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = float(anchor[0]) - center_r
    dc = float(anchor[1]) - center_c
    if abs(dr) > abs(dc):
        return "above" if dr < 0 else "below"
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    return "tie"


def _sample_anchor(
    shape: tuple[int, int], r0: int, c0: int, mask: np.ndarray, direction: str
) -> tuple[int, int] | None:
    h, w = map(int, mask.shape)
    grid_h, grid_w = shape
    center_r = r0 + (h - 1) / 2.0
    center_c = c0 + (w - 1) / 2.0

    for _ in range(40):
        if direction == "above":
            max_gap = min(7, r0)
            if max_gap < 2:
                return None
            ar = r0 - random.randint(2, max_gap)
            ac = int(round(center_c)) + random.randint(-2, 2)
        elif direction == "below":
            max_gap = min(7, grid_h - r0 - h)
            if max_gap < 2:
                return None
            ar = r0 + h - 1 + random.randint(2, max_gap)
            ac = int(round(center_c)) + random.randint(-2, 2)
        elif direction == "left":
            max_gap = min(7, c0)
            if max_gap < 2:
                return None
            ar = int(round(center_r)) + random.randint(-2, 2)
            ac = c0 - random.randint(2, max_gap)
        else:
            max_gap = min(7, grid_w - c0 - w)
            if max_gap < 2:
                return None
            ar = int(round(center_r)) + random.randint(-2, 2)
            ac = c0 + w - 1 + random.randint(2, max_gap)

        if 0 <= ar < grid_h and 0 <= ac < grid_w:
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _direction_from_anchor((ar, ac), bbox) == direction:
                return int(ar), int(ac)
    return None


def _paint_object(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = _mask_coords(mask, r0, c0)
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _place_specs(specs: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    order = sorted(specs, key=lambda spec: int(spec["mask"].sum()), reverse=True)

    for spec in order:
        mask = spec["mask"]
        h, w = map(int, mask.shape)
        color = int(spec["color"])
        placed = False

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, shape[0] - h)
            c0 = random.randint(0, shape[1] - w)
            anchor = _sample_anchor(shape, r0, c0, mask, spec["direction"])
            if anchor is None:
                continue

            obj_coords = _mask_coords(mask, r0, c0)
            anchor_coords = np.asarray([anchor], dtype=int)
            if _anchor_touches_object(anchor, obj_coords):
                continue
            if _touches_existing(occupied, obj_coords) or _touches_existing(occupied, anchor_coords):
                continue

            _paint_object(grid, mask, r0, c0, color)
            grid[anchor] = color
            occupied[obj_coords[:, 0], obj_coords[:, 1]] = True
            occupied[anchor] = True
            placed = True
            break

        if not placed:
            return None
    return grid


def _sample_specs() -> list[dict] | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, count)
    areas = random.sample(OBJECT_AREA_VALUES, count)

    directions = [random.choice(VERTICAL_DIRECTIONS), random.choice(HORIZONTAL_DIRECTIONS)]
    directions.extend(random.choice(DIRECTIONS) for _ in range(count - len(directions)))
    random.shuffle(directions)

    specs: list[dict] = []
    for color, area, direction in zip(colors, areas, directions):
        axis = "vertical" if direction in VERTICAL_DIRECTIONS else "horizontal"
        mask = _random_connected_mask(int(area), axis)
        if mask is None:
            return None
        specs.append({"color": int(color), "area": int(area), "direction": direction, "mask": mask})
    return specs


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    grid = _place_specs(specs, (h, w))
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if grid.ndim != 2 or out.ndim != 2:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored mirror-stack input")
