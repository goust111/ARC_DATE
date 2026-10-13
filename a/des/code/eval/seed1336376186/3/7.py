from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
FILL_COLORS = (1, 2, 3, 4, 6, 8, 9)

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
OBJECT_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (7, 11)
RECT_W_RANGE = (7, 11)
L_H_RANGE = (8, 12)
L_W_RANGE = (8, 12)
DEFECT_COUNT_RANGE = (1, 4)
LOCAL_RETRIES = 200
PLACEMENT_RETRIES = 250
GENERATE_RETRIES = 300
OVERLAP_PROB = 0.35

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOTION_CHOICES = (
    ("vertical", -1),
    ("vertical", 1),
    ("horizontal", -1),
    ("horizontal", 1),
)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _outside_mask(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    outside = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _symmetry_candidates(patch: np.ndarray, fill: int) -> list[tuple[int, int]]:
    occ = (patch == fill) | (patch == ORANGE)
    orange = np.argwhere(patch == ORANGE)
    if orange.size == 0:
        return []

    h, w = patch.shape
    candidates: list[tuple[int, int]] = []

    if np.array_equal(occ, occ[:, ::-1]):
        signs = set()
        ok = True
        for r, c in orange:
            mc = w - 1 - int(c)
            if mc == int(c) or int(patch[int(r), mc]) != fill:
                ok = False
                break
            signs.add(1 if int(c) > mc else -1)
        if ok and len(signs) == 1:
            candidates.append((0, len(orange) * next(iter(signs))))

    if np.array_equal(occ, occ[::-1, :]):
        signs = set()
        ok = True
        for r, c in orange:
            mr = h - 1 - int(r)
            if mr == int(r) or int(patch[mr, int(c)]) != fill:
                ok = False
                break
            signs.add(1 if int(r) > mr else -1)
        if ok and len(signs) == 1:
            candidates.append((len(orange) * next(iter(signs)), 0))

    return candidates


def _infer_motion(patch: np.ndarray, fill: int) -> tuple[int, int]:
    candidates = _symmetry_candidates(patch, fill)
    assert len(candidates) == 1, "object must have a unique valid symmetry defect direction"
    return candidates[0]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    grey_comps = _components4(grid == GREY)
    assert grey_comps
    h, w = grid.shape
    objects: list[dict] = []

    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(h - 1, r1 + 1)
        ec1 = min(w - 1, c1 + 1)

        expanded = grid[er0 : er1 + 1, ec0 : ec1 + 1]
        grey_local = np.zeros_like(expanded, dtype=bool)
        for r, c in comp:
            grey_local[r - er0, c - ec0] = True

        outside = _outside_mask(~grey_local)
        enclosed = (~grey_local) & (~outside)
        enclosed_exact = enclosed[r0 - er0 : r1 - er0 + 1, c0 - ec0 : c1 - ec0 + 1]

        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        inside_values = {
            int(v)
            for v in patch[enclosed_exact]
            if int(v) not in (BG, ORANGE)
        }
        assert len(inside_values) == 1, "each shape must contain exactly one true fill color"
        fill = next(iter(inside_values))
        dr, dc = _infer_motion(patch, fill)

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "fill": fill,
                "move": (dr, dc),
            }
        )
    return objects


def _paint_order(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    return c1, r1, c0, r0


def _paste_stamp(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(out.shape[0], top + h)
    cc1 = min(out.shape[1], left + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return

    src_r0 = rr0 - top
    src_c0 = cc0 - left
    src_r1 = src_r0 + (rr1 - rr0)
    src_c1 = src_c0 + (cc1 - cc0)
    src = patch[src_r0:src_r1, src_c0:src_c1]
    mask = src != BG
    target = out[rr0:rr1, cc0:cc1]
    target[mask] = src[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each grey-framed shape toward the side containing its orange symmetry defects.

    The enclosed pattern uses a single fill color plus orange defect pixels. The pattern is
    symmetric across exactly one axis in the object's bounding box once orange is treated as
    the fill color. The side of that axis containing all orange pixels gives the translation
    direction, and the number of orange pixels gives the translation distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(np.min(g)) >= 0

    objects = _extract_objects(g)
    out = np.full_like(g, BG)

    for obj in sorted(objects, key=_paint_order, reverse=True):
        patch = obj["patch"].copy()
        patch[patch == ORANGE] = int(obj["fill"])
        r0, c0, _r1, _c1 = obj["bbox"]
        dr, dc = obj["move"]
        _paste_stamp(out, patch, r0 + dr, c0 + dc)

    return out


def _interior_and_border(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = region.shape
    interior = np.zeros_like(region, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not region[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                continue
            if region[r - 1, c] and region[r + 1, c] and region[r, c - 1] and region[r, c + 1]:
                interior[r, c] = True
    border = region & (~interior)
    return interior, border


def _make_region(shape_type: str, h: int, w: int, rng: random.Random) -> np.ndarray:
    if shape_type == "rectangle":
        return np.ones((h, w), dtype=bool)

    region = np.ones((h, w), dtype=bool)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    cut_h = rng.randint(2, h - 3)
    cut_w = rng.randint(2, w - 3)
    if h - cut_h < 3 or w - cut_w < 3:
        return region

    if corner == "tl":
        region[:cut_h, :cut_w] = False
    elif corner == "tr":
        region[:cut_h, w - cut_w :] = False
    elif corner == "bl":
        region[h - cut_h :, :cut_w] = False
    else:
        region[h - cut_h :, w - cut_w :] = False
    return region


def _support_mask(interior: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "vertical":
        return interior & interior[:, ::-1]
    return interior & interior[::-1, :]


def _pair_lists(mask: np.ndarray, orientation: str) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    h, w = mask.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    axes: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if orientation == "vertical":
                mr, mc = r, w - 1 - c
            else:
                mr, mc = h - 1 - r, c
            if not mask[mr, mc]:
                continue
            if (r, c) == (mr, mc):
                axes.append((r, c))
            elif (r, c) < (mr, mc):
                pairs.append(((r, c), (mr, mc)))
    return pairs, axes


def _is_symmetric(mask: np.ndarray, orientation: str) -> bool:
    if orientation == "vertical":
        return np.array_equal(mask, mask[:, ::-1])
    return np.array_equal(mask, mask[::-1, :])


def _side_cells(mask: np.ndarray, orientation: str, sign: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        if orientation == "vertical":
            delta = (2 * int(c)) - (w - 1)
        else:
            delta = (2 * int(r)) - (h - 1)
        if delta == 0:
            continue
        if (delta > 0) == (sign > 0):
            cells.append((int(r), int(c)))
    return cells


def _make_pattern(
    interior: np.ndarray,
    orientation: str,
    sign: int,
    defect_count: int,
    rng: random.Random,
) -> np.ndarray | None:
    support = _support_mask(interior, orientation)
    pairs, axes = _pair_lists(support, orientation)
    if len(pairs) < defect_count + 1:
        return None

    min_pairs = max(2, defect_count + 1)
    max_pairs = min(len(pairs), max(min_pairs, len(pairs) - rng.randint(0, max(0, len(pairs) // 3))))
    if max_pairs < min_pairs:
        return None
    pair_count = rng.randint(min_pairs, max_pairs)
    chosen_pairs = rng.sample(pairs, pair_count)

    occ = np.zeros_like(interior, dtype=bool)
    for a, b in chosen_pairs:
        occ[a] = True
        occ[b] = True

    axis_take = 0
    if axes:
        axis_take = rng.randint(0, min(len(axes), max(1, len(axes) // 2)))
    if axis_take:
        for cell in rng.sample(axes, axis_take):
            occ[cell] = True

    if int(np.count_nonzero(occ)) <= defect_count:
        return None
    if not _is_symmetric(occ, orientation):
        return None
    other = "horizontal" if orientation == "vertical" else "vertical"
    if _is_symmetric(occ, other):
        return None

    side = _side_cells(occ, orientation, sign)
    if len(side) < defect_count:
        return None
    return occ


def _local_patch_from_pattern(
    border: np.ndarray,
    occ: np.ndarray,
    fill: int,
    orientation: str,
    sign: int,
    defect_count: int,
    rng: random.Random,
) -> np.ndarray | None:
    side = _side_cells(occ, orientation, sign)
    if len(side) < defect_count:
        return None

    defects = set(rng.sample(side, defect_count))
    patch = np.full(border.shape, BG, dtype=int)
    patch[border] = GREY
    patch[occ] = fill
    for r, c in defects:
        patch[r, c] = ORANGE

    if np.count_nonzero(patch == fill) == 0:
        return None
    motion = _symmetry_candidates(patch, fill)
    expected = [(0, defect_count * sign)] if orientation == "vertical" else [(defect_count * sign, 0)]
    if motion != expected:
        return None
    return patch


def _generate_local_object(
    fill: int,
    shape_type: str,
    orientation: str,
    sign: int,
    defect_count: int,
    rng: random.Random,
) -> dict | None:
    h_range = RECT_H_RANGE if shape_type == "rectangle" else L_H_RANGE
    w_range = RECT_W_RANGE if shape_type == "rectangle" else L_W_RANGE

    for _ in range(LOCAL_RETRIES):
        h = rng.randint(h_range[0], h_range[1])
        w = rng.randint(w_range[0], w_range[1])
        region = _make_region(shape_type, h, w, rng)
        interior, border = _interior_and_border(region)
        if np.count_nonzero(interior) < 10:
            continue

        occ = _make_pattern(interior, orientation, sign, defect_count, rng)
        if occ is None:
            continue
        patch = _local_patch_from_pattern(border, occ, fill, orientation, sign, defect_count, rng)
        if patch is None:
            continue

        dr = defect_count * sign if orientation == "horizontal" else 0
        dc = defect_count * sign if orientation == "vertical" else 0
        return {
            "patch": patch,
            "fill": fill,
            "shape_type": shape_type,
            "move": (dr, dc),
        }
    return None


def _bbox_from_pos(top: int, left: int, patch: np.ndarray) -> tuple[int, int, int, int]:
    h, w = patch.shape
    return top, left, top + h - 1, left + w - 1


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _non_bg_coords(obj: dict, top: int, left: int, moved: bool) -> set[tuple[int, int]]:
    patch = obj["patch"].copy()
    patch[patch == ORANGE] = obj["fill"]
    if moved:
        top += obj["move"][0]
        left += obj["move"][1]
    coords = set()
    for r, c in np.argwhere(patch != BG):
        coords.add((top + int(r), left + int(c)))
    return coords


def _place_objects(
    objects: list[dict],
    h: int,
    w: int,
    want_overlap: bool,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        positions: list[tuple[int, int]] = []
        ok = True
        for obj in objects:
            ph, pw = obj["patch"].shape
            dr, dc = obj["move"]
            min_r = 1 + max(0, -dr)
            max_r = h - ph - 2 - max(0, dr)
            min_c = 1 + max(0, -dc)
            max_c = w - pw - 2 - max(0, dc)
            if min_r > max_r or min_c > max_c:
                ok = False
                break
            positions.append((rng.randint(min_r, max_r), rng.randint(min_c, max_c)))
        if not ok:
            continue

        orig_boxes = [_bbox_from_pos(r, c, obj["patch"]) for obj, (r, c) in zip(objects, positions)]
        if any(_intersects(orig_boxes[i], orig_boxes[j], gap=1) for i in range(len(orig_boxes)) for j in range(i + 1, len(orig_boxes))):
            continue

        overlap_pairs = 0
        for i in range(len(objects)):
            coords_i = _non_bg_coords(objects[i], positions[i][0], positions[i][1], moved=True)
            for j in range(i + 1, len(objects)):
                coords_j = _non_bg_coords(objects[j], positions[j][0], positions[j][1], moved=True)
                inter = coords_i & coords_j
                if inter:
                    overlap_pairs += 1
        if want_overlap:
            if overlap_pairs != 1:
                continue
        else:
            if overlap_pairs != 0:
                continue
        return positions
    return None


def _render_input(objects: list[dict], positions: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj, (r0, c0) in zip(objects, positions):
        _paste_stamp(grid, obj["patch"], r0, c0)
    return grid


def _render_output(objects: list[dict], positions: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    merged = []
    for obj, (r0, c0) in zip(objects, positions):
        h, w = obj["patch"].shape
        merged.append(
            {
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "patch": obj["patch"],
                "fill": obj["fill"],
                "move": obj["move"],
            }
        )

    for obj in sorted(merged, key=_paint_order, reverse=True):
        patch = obj["patch"].copy()
        patch[patch == ORANGE] = obj["fill"]
        r0, c0, _r1, _c1 = obj["bbox"]
        dr, dc = obj["move"]
        _paste_stamp(out, patch, r0 + dr, c0 + dc)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    want_overlap = count >= 2 and rng.random() < OVERLAP_PROB

    try:
        fills = rng.sample(FILL_COLORS, count)
    except ValueError:
        return None

    objects: list[dict] = []
    for i in range(count):
        fill = fills[i]
        shape_type = rng.choice(("rectangle", "lshape"))
        orientation, sign = rng.choice(MOTION_CHOICES)
        defect_count = rng.randint(DEFECT_COUNT_RANGE[0], DEFECT_COUNT_RANGE[1])
        obj = _generate_local_object(fill, shape_type, orientation, sign, defect_count, rng)
        if obj is None:
            return None
        objects.append(obj)

    positions = _place_objects(objects, h, w, want_overlap, rng)
    if positions is None:
        return None

    inp = _render_input(objects, positions, (h, w))
    expected = _render_output(objects, positions, (h, w))
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
