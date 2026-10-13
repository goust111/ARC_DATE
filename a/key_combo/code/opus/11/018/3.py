from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Corner = str

_CORNERS: tuple[Corner, ...] = ("TL", "TR", "BR", "BL")
_NEXT_CORNER: dict[Corner, Corner] = {
    "TL": "TR",
    "TR": "BR",
    "BR": "BL",
    "BL": "TL",
}

_PLACEMENT_ORDER: tuple[Corner, ...] = ("TL", "TR", "BR", "BL")

_BG = 0
_CORNER_COLORS = (1, 2, 3, 4)
_DISTRACTOR_COLOR = 5
_AREA_CHOICES = (3, 4, 5, 6, 7, 8)
_GRID_H_RANGE = (14, 18)
_GRID_W_RANGE = (14, 18)
_CORNER_DIM_RANGE = (2, 4)
_DISTRACTOR_AREA_RANGE = (1, 4)
_DISTRACTOR_DIM_RANGE = (1, 3)
_DISTRACTOR_COUNT_OPTIONS = (1, 2)
_MAX_GENERATION_ATTEMPTS = 500
_GROW_ATTEMPTS = 80
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _bbox_corner(bbox: tuple[int, int, int, int], height: int, width: int) -> Corner | None:
    r0, c0, r1, c1 = bbox
    corners: list[Corner] = []
    if r0 == 0 and c0 == 0:
        corners.append("TL")
    if r0 == 0 and c1 == width - 1:
        corners.append("TR")
    if r1 == height - 1 and c1 == width - 1:
        corners.append("BR")
    if r1 == height - 1 and c0 == 0:
        corners.append("BL")
    assert len(corners) <= 1, "each component should match at most one true corner"
    return corners[0] if corners else None


def _corner_anchor(corner: Corner, height: int, width: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, width - obj_w
    if corner == "BR":
        return height - obj_h, width - obj_w
    if corner == "BL":
        return height - obj_h, 0
    raise AssertionError(f"unknown corner: {corner}")


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _rotate_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask, k=-(quarter_turns % 4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each corner-anchored object clockwise into the next corner, rotating by global size rank.

    Input structure:
    - The background is black, with exactly four single-color 4-connected corner objects and optional gray interior distractors.
    - Each relevant object has a bounding box that touches the two borders of one true grid corner.
    - The four corner objects use distinct non-gray colors and have pairwise distinct pixel counts.

    Rule:
    - Ignore every component that does not touch two adjacent borders at a true corner.
    - Rank the four corner objects globally by pixel count from smallest to largest.
    - Copy each corner object to the next clockwise corner (TL->TR, TR->BR, BR->BL, BL->TL), but rotate the copied mask clockwise by 0, 90, 180, 270 degrees for ranks 1, 2, 3, 4 respectively before corner-aligning its bounding box.
    - Keep the original objects and distractors unchanged; copied pixels overwrite whatever is already in the destination corner.

    Key insight:
    - The destination corner depends only on perimeter order, but the rotation depends on a separate global comparison across all four corner-object sizes, so detection, ranking, rotation, and placement must be chained together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _get_bg(g)

    corner_objects: dict[Corner, dict] = {}
    for comp in _find_components(g, bg=bg):
        corner = _bbox_corner(comp["bbox"], height, width)
        if corner is None:
            continue
        assert corner not in corner_objects, "expected exactly one corner object per corner"
        corner_objects[corner] = comp

    assert set(corner_objects) == set(_CORNERS), "expected one participating object at each corner"
    areas = [int(comp["area"]) for comp in corner_objects.values()]
    assert len(set(areas)) == 4, "corner objects must have unique sizes"

    ranks_by_corner = {
        corner: rank
        for rank, (corner, _comp) in enumerate(
            sorted(corner_objects.items(), key=lambda item: int(item[1]["area"]))
        )
    }

    out = g.copy()
    for src_corner in _PLACEMENT_ORDER:
        comp = corner_objects[src_corner]
        color = int(comp["color"])
        mask = _component_mask(comp)
        rot_mask = _rotate_cw(mask, ranks_by_corner[src_corner])
        dst_corner = _NEXT_CORNER[src_corner]
        dst_r0, dst_c0 = _corner_anchor(dst_corner, height, width, *rot_mask.shape)
        rows, cols = np.nonzero(rot_mask)
        out[dst_r0 + rows, dst_c0 + cols] = color

    return out


def _source_local_corner(corner: Corner, h: int, w: int) -> tuple[int, int]:
    r = 0 if corner in ("TL", "TR") else h - 1
    c = 0 if corner in ("TL", "BL") else w - 1
    return r, c


def _step_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for dr, dc in _N4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _spanning_l_mask(corner: Corner, h: int, w: int) -> np.ndarray:
    sr, sc = _source_local_corner(corner, h, w)
    er, ec = h - 1 - sr, w - 1 - sc
    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        for r in _step_range(sr, er):
            mask[r, sc] = True
        for c in _step_range(sc, ec):
            mask[er, c] = True
    else:
        for c in _step_range(sc, ec):
            mask[sr, c] = True
        for r in _step_range(sr, er):
            mask[r, ec] = True
    return mask


def _generate_corner_mask(corner: Corner, area: int) -> np.ndarray:
    dim_min, dim_max = _CORNER_DIM_RANGE
    candidates = [
        (h, w)
        for h in range(dim_min, dim_max + 1)
        for w in range(dim_min, dim_max + 1)
        if h * w > area and h + w - 1 <= area
    ]
    random.shuffle(candidates)
    for h, w in candidates:
        for _ in range(_GROW_ATTEMPTS):
            mask = _spanning_l_mask(corner, h, w)
            while int(mask.sum()) < area:
                choices = _frontier(mask)
                if not choices:
                    break
                rr, cc = random.choice(choices)
                mask[rr, cc] = True
            if int(mask.sum()) != area:
                continue
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            return mask
    raise RuntimeError(f"failed to generate corner mask for area={area}")


def _stamp_mask(canvas: np.ndarray, mask: np.ndarray, color: int, corner: Corner) -> None:
    r0, c0 = _corner_anchor(corner, canvas.shape[0], canvas.shape[1], *mask.shape)
    rows, cols = np.nonzero(mask)
    canvas[r0 + rows, c0 + cols] = color


def _future_stamp_cells(objects: dict[Corner, dict], height: int, width: int) -> set[tuple[int, int]]:
    ranks_by_corner = {
        corner: rank
        for rank, (corner, _obj) in enumerate(sorted(objects.items(), key=lambda item: int(item[1]["area"])))
    }
    cells: set[tuple[int, int]] = set()
    for corner, obj in objects.items():
        rot_mask = _rotate_cw(obj["mask"], ranks_by_corner[corner])
        dst_corner = _NEXT_CORNER[corner]
        r0, c0 = _corner_anchor(dst_corner, height, width, *rot_mask.shape)
        for rr, cc in np.argwhere(rot_mask):
            cells.add((r0 + int(rr), c0 + int(cc)))
    return cells


def _generate_blob_mask(area: int) -> np.ndarray:
    dmin, dmax = _DISTRACTOR_DIM_RANGE
    candidates = [
        (h, w)
        for h in range(dmin, dmax + 1)
        for w in range(dmin, dmax + 1)
        if h * w >= area
    ]
    random.shuffle(candidates)
    for h, w in candidates:
        for _ in range(_GROW_ATTEMPTS):
            mask = np.zeros((h, w), dtype=bool)
            sr, sc = random.randrange(h), random.randrange(w)
            mask[sr, sc] = True
            while int(mask.sum()) < area:
                choices = _frontier(mask)
                if not choices:
                    break
                rr, cc = random.choice(choices)
                mask[rr, cc] = True
            if int(mask.sum()) != area:
                continue
            rows, cols = np.nonzero(mask)
            return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    raise RuntimeError(f"failed to generate distractor mask for area={area}")


def _can_place_patch(
    occupied: np.ndarray,
    patch_mask: np.ndarray,
    r0: int,
    c0: int,
    *,
    require_interior: bool,
) -> bool:
    h, w = occupied.shape
    ph, pw = patch_mask.shape
    rows, cols = np.nonzero(patch_mask)
    patch_cells = {(r0 + int(r), c0 + int(c)) for r, c in zip(rows, cols)}
    if require_interior and any(r in (0, h - 1) or c in (0, w - 1) for r, c in patch_cells):
        return False
    if any(occupied[r, c] for r, c in patch_cells):
        return False
    for r, c in patch_cells:
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in patch_cells and occupied[nr, nc]:
                return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch_mask: np.ndarray, color: int, require_interior: bool) -> bool:
    h, w = grid.shape
    ph, pw = patch_mask.shape
    choices: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if _can_place_patch(occupied, patch_mask, r0, c0, require_interior=require_interior):
                choices.append((r0, c0))
    if not choices:
        return False
    r0, c0 = random.choice(choices)
    rows, cols = np.nonzero(patch_mask)
    grid[r0 + rows, c0 + cols] = color
    occupied[r0 + rows, c0 + cols] = True
    return True


def _build_corner_objects() -> dict[Corner, dict]:
    colors = list(_CORNER_COLORS)
    random.shuffle(colors)
    areas = random.sample(_AREA_CHOICES, 4)
    random.shuffle(areas)
    objects: dict[Corner, dict] = {}
    for corner, color, area in zip(_CORNERS, colors, areas):
        objects[corner] = {
            "corner": corner,
            "color": color,
            "area": area,
            "mask": _generate_corner_mask(corner, area),
        }
    return objects


def _render_input(height: int, width: int, objects: dict[Corner, dict]) -> np.ndarray:
    grid = np.full((height, width), _BG, dtype=int)
    for corner in _CORNERS:
        obj = objects[corner]
        _stamp_mask(grid, obj["mask"], int(obj["color"]), corner)
    return grid


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*_GRID_H_RANGE)
    width = random.randint(*_GRID_W_RANGE)
    objects = _build_corner_objects()
    inp = _render_input(height, width, objects)

    occupied = inp != _BG
    reserved = occupied.copy()
    for r, c in _future_stamp_cells(objects, height, width):
        reserved[r, c] = True

    distractor_count = random.choice(_DISTRACTOR_COUNT_OPTIONS)
    for _ in range(distractor_count):
        area = random.randint(*_DISTRACTOR_AREA_RANGE)
        patch = _generate_blob_mask(area)
        if not _place_patch(inp, reserved, patch, _DISTRACTOR_COLOR, require_interior=True):
            return None

    try:
        out = main(inp)
    except AssertionError:
        return None

    if np.array_equal(inp, out):
        return None
    if _get_bg(inp) != _BG:
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
