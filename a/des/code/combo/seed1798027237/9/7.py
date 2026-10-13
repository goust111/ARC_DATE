from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _border_color(grid: np.ndarray) -> int:
    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[1:-1, 0]
    right = grid[1:-1, -1]
    border = np.concatenate([top, bottom, left, right])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _internal_voids(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    voids: list[list[tuple[int, int]]] = []
    for comp in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        voids.append(comp)
    return voids


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            mask, bbox = _coords_to_mask(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "mask": mask,
                    "bbox": bbox,
                    "area": int(len(coords)),
                }
            )
    return comps


def _annotate_holes(comp: dict[str, object]) -> list[dict[str, object]]:
    mask = np.asarray(comp["mask"], dtype=bool)
    r0, c0, _r1, _c1 = comp["bbox"]
    holes: list[dict[str, object]] = []
    for void in _internal_voids(mask):
        hole_mask, hole_bbox_rel = _coords_to_mask(void)
        hr0, hc0, hr1, hc1 = hole_bbox_rel
        abs_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in void]
        holes.append(
            {
                "coords": abs_coords,
                "mask": hole_mask,
                "bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
                "area": int(len(void)),
            }
        )
    return holes


def _unique_matching_rotation(piece_mask: np.ndarray, hole_mask: np.ndarray) -> int:
    matches: list[int] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(piece_mask, k)
        key = (tuple(map(int, rotated.shape)), rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        if rotated.shape == hole_mask.shape and np.array_equal(rotated, hole_mask):
            matches.append(int(k))
    assert len(matches) == 1
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each template hole with the uniquely area-matched fragment after rotating it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_color(g)
    comps = _monochrome_components(g, bg)
    annotated = [{**comp, "holes": _annotate_holes(comp)} for comp in comps]
    candidates = [comp for comp in annotated if comp["holes"]]
    assert candidates

    template = max(candidates, key=lambda comp: int(comp["area"]))
    pieces = [comp for comp in annotated if comp is not template]
    holes = list(template["holes"])
    assert holes and len(pieces) == len(holes)

    hole_by_area = {int(hole["area"]): hole for hole in holes}
    assert len(hole_by_area) == len(holes)
    assert len({int(piece["area"]) for piece in pieces}) == len(pieces)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = bg

    for piece in pieces:
        area = int(piece["area"])
        assert area in hole_by_area
        hole = hole_by_area[area]
        piece_mask = np.asarray(piece["mask"], dtype=bool)
        hole_mask = np.asarray(hole["mask"], dtype=bool)
        rot_k = _unique_matching_rotation(piece_mask, hole_mask)
        rotated = np.rot90(piece_mask, rot_k)
        assert np.array_equal(rotated, hole_mask)

        r0, c0, r1, c1 = hole["bbox"]
        patch = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        patch[rotated] = int(piece["color"])

    return out


# --------------- generator ---------------

BACKGROUND_COLOR = 0
TEMPLATE_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
TEMPLATE_HEIGHT_MAX = 18
TEMPLATE_WIDTH_MAX = 18
TEMPLATE_MIN_SIDE = 10
HOLE_COUNT_RANGE = (2, 4)
HOLE_AREA_RANGE = (4, 9)

SHAPE_BBOX_MAX = 5
SHAPE_ATTEMPTS = 250
TEMPLATE_PLACEMENT_ATTEMPTS = 200
PIECE_PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _unique_rotation_count(mask: np.ndarray) -> int:
    return len({_rotation_key(np.rot90(mask, k)) for k in range(4)})


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _random_connected_mask(area: int) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(2, SHAPE_BBOX_MAX + 1)
        for w in range(2, SHAPE_BBOX_MAX + 1)
        if area <= h * w <= area + 4
    ]
    if not dims:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        h, w = random.choice(dims)
        mask = np.zeros((h, w), dtype=bool)
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        mask[r0, c0] = True
        frontier = set()

        def add_frontier(rr: int, cc: int) -> None:
            for nr, nc in _neighbors4(rr, cc, h, w):
                if not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))

        add_frontier(r0, c0)
        while int(mask.sum()) < area and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            mask[rr, cc] = True
            add_frontier(rr, cc)

        if int(mask.sum()) != area:
            continue

        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        mask = _coords_to_mask(coords)[0]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if bool(np.all(mask)):
            continue
        if _internal_voids(mask):
            continue
        if _unique_rotation_count(mask) != 4:
            continue
        return mask
    return None


def _sample_hole_masks() -> list[np.ndarray] | None:
    hole_count = random.randint(HOLE_COUNT_RANGE[0], HOLE_COUNT_RANGE[1])
    areas = random.sample(
        range(HOLE_AREA_RANGE[0], HOLE_AREA_RANGE[1] + 1),
        hole_count,
    )
    masks: list[np.ndarray] = []
    for area in areas:
        mask = _random_connected_mask(int(area))
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _sample_template_dims(hole_masks: list[np.ndarray]) -> tuple[int, int] | None:
    total_area = sum(int(mask.sum()) for mask in hole_masks)
    max_h = max(int(mask.shape[0]) for mask in hole_masks)
    max_w = max(int(mask.shape[1]) for mask in hole_masks)
    base = int(np.ceil(np.sqrt(total_area * 3.0))) + 1

    min_h = max(TEMPLATE_MIN_SIDE, max_h + 2, base - 1)
    min_w = max(TEMPLATE_MIN_SIDE, max_w + 2, base)
    if min_h > TEMPLATE_HEIGHT_MAX or min_w > TEMPLATE_WIDTH_MAX:
        return None
    return (
        random.randint(min_h, TEMPLATE_HEIGHT_MAX),
        random.randint(min_w, TEMPLATE_WIDTH_MAX),
    )


def _place_holes_in_template(
    template_h: int,
    template_w: int,
    hole_masks: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((template_h, template_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(hole_masks)
    order = sorted(range(len(hole_masks)), key=lambda i: int(hole_masks[i].sum()), reverse=True)

    for idx in order:
        mask = hole_masks[idx]
        h, w = map(int, mask.shape)
        max_r0 = template_h - h - 1
        max_c0 = template_w - w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None

        dilated = _dilate8(occupied)
        placed = False
        for _ in range(TEMPLATE_PLACEMENT_ATTEMPTS):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            if np.any(dilated[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            occupied[r0 : r0 + h, c0 : c0 + w] |= mask
            positions[idx] = (int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None

    assert all(pos is not None for pos in positions)
    return [(int(pos[0]), int(pos[1])) for pos in positions]


def _sample_grid_dims(template_h: int, template_w: int) -> tuple[int, int] | None:
    min_h = max(GRID_HEIGHT_RANGE[0], template_h + 6)
    min_w = max(GRID_WIDTH_RANGE[0], template_w + 6)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None
    return (
        random.randint(min_h, GRID_HEIGHT_RANGE[1]),
        random.randint(min_w, GRID_WIDTH_RANGE[1]),
    )


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    grid[int(top) : int(top) + h, int(left) : int(left) + w][mask] = int(color)


def _build_task() -> tuple[np.ndarray, np.ndarray] | None:
    hole_masks = _sample_hole_masks()
    if hole_masks is None:
        return None

    dims = _sample_template_dims(hole_masks)
    if dims is None:
        return None
    template_h, template_w = dims

    hole_positions = _place_holes_in_template(template_h, template_w, hole_masks)
    if hole_positions is None or len(hole_positions) != len(hole_masks):
        return None

    grid_dims = _sample_grid_dims(template_h, template_w)
    if grid_dims is None:
        return None
    grid_h, grid_w = grid_dims

    grid = np.full((grid_h, grid_w), BACKGROUND_COLOR, dtype=int)
    expected = np.full((grid_h, grid_w), BACKGROUND_COLOR, dtype=int)

    template_r = random.randint(2, grid_h - template_h - 2)
    template_c = random.randint(2, grid_w - template_w - 2)

    template_mask = np.ones((template_h, template_w), dtype=bool)
    for mask, (r0, c0) in zip(hole_masks, hole_positions, strict=False):
        h, w = map(int, mask.shape)
        template_mask[r0 : r0 + h, c0 : c0 + w][mask] = False

    _paint_mask(grid, template_mask, template_r, template_c, TEMPLATE_COLOR)
    _paint_mask(expected, template_mask, template_r, template_c, TEMPLATE_COLOR)

    colors = random.sample(PIECE_COLORS, len(hole_masks))
    pieces = []
    for color, mask, (r0, c0) in zip(colors, hole_masks, hole_positions, strict=False):
        abs_r = int(template_r + r0)
        abs_c = int(template_c + c0)
        pieces.append({"color": int(color), "mask": mask, "hole_top": abs_r, "hole_left": abs_c})
        _paint_mask(expected, mask, abs_r, abs_c, int(color))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[template_r : template_r + template_h, template_c : template_c + template_w] = True
    order = list(range(len(pieces)))
    order.sort(key=lambda i: int(pieces[i]["mask"].sum()), reverse=True)

    for idx in order:
        piece = pieces[idx]
        rot_k = random.randint(0, 3)
        piece_mask = np.rot90(np.asarray(piece["mask"], dtype=bool), rot_k)
        h, w = map(int, piece_mask.shape)
        placed = False
        for _ in range(PIECE_PLACEMENT_ATTEMPTS):
            dilated = _dilate8(occupied)
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            if np.any(dilated[r0 : r0 + h, c0 : c0 + w] & piece_mask):
                continue
            _paint_mask(grid, piece_mask, r0, c0, int(piece["color"]))
            occupied[r0 : r0 + h, c0 : c0 + w] |= piece_mask
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_task()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
