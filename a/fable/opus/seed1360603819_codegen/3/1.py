import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]
SIDES = ["left", "right", "top", "bottom"]
HOLE_POOL = [0, 1, 2, 3, 4]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    neighbor_fn = _neighbors8 if connectivity == 8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbor_fn(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _count_holes(mask: np.ndarray) -> int:
    holes = 0
    bg = ~mask
    h, w = mask.shape
    for comp in _components(bg, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _body_tail_side(body_points: list[tuple[int, int]], tail_points: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(body_points)
    if all(c < c0 for _, c in tail_points):
        return "left"
    if all(c > c1 for _, c in tail_points):
        return "right"
    if all(r < r0 for r, _ in tail_points):
        return "top"
    if all(r > r1 for r, _ in tail_points):
        return "bottom"
    raise AssertionError("tail must lie on exactly one side of the body")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey bodies to a left-side legend by hole count, then slide each kept
    body+tail object away from its azure tail until it reaches the grid edge.
    Only the body is recolored and rotated 180 degrees; the tail stays azure.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fg_components = _components(g != BLACK, connectivity=8)

    legend_map: dict[int, int] = {}
    legend_components: list[list[tuple[int, int]]] = []
    object_components: list[list[tuple[int, int]]] = []

    for pts in fg_components:
        colors = {int(g[r, c]) for r, c in pts}
        if GREY in colors:
            object_components.append(pts)
        else:
            legend_components.append(pts)
            mask = _coords_to_mask(pts)
            hole_count = _count_holes(mask)
            visible_colors = colors - {BLACK}
            assert len(visible_colors) == 1, "legend component must be monochrome"
            color = int(next(iter(visible_colors)))
            if hole_count in legend_map and legend_map[hole_count] != color:
                raise AssertionError("hole-count legend must be unambiguous")
            legend_map[hole_count] = color

    out = np.full_like(g, BLACK)

    for pts in legend_components:
        for r, c in pts:
            out[r, c] = g[r, c]

    for pts in object_components:
        body_points = [(r, c) for r, c in pts if int(g[r, c]) == GREY]
        tail_points = [(r, c) for r, c in pts if int(g[r, c]) == AZURE]
        assert body_points and tail_points, "each composite object must contain body and tail"

        side = _body_tail_side(body_points, tail_points)
        body_mask = _coords_to_mask(body_points)
        hole_count = _count_holes(body_mask)
        if hole_count not in legend_map:
            continue
        target_color = int(legend_map[hole_count])

        obj_r0, obj_c0, obj_r1, obj_c1 = _bbox(pts)
        body_r0, body_c0, body_r1, body_c1 = _bbox(body_points)

        dr = 0
        dc = 0
        if side == "left":
            dc = (w - 1) - obj_c1
        elif side == "right":
            dc = -obj_c0
        elif side == "top":
            dr = (h - 1) - obj_r1
        else:
            dr = -obj_r0

        rotated_body = np.rot90(body_mask, 2)
        for rr, cc in np.argwhere(rotated_body):
            out[body_r0 + dr + int(rr), body_c0 + dc + int(cc)] = target_color
        for r, c in tail_points:
            out[r + dr, c + dc] = AZURE

    return out


def _normalize_coords(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(points)
    return [(r - r0, c - c0) for r, c in points]


def _mask_from_coords(points: list[tuple[int, int]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 0), dtype=bool)
    pts = _normalize_coords(points)
    r1 = max(r for r, _ in pts)
    c1 = max(c for _, c in pts)
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return mask


def _make_holey_square(hole_count: int, color: int, rng: random.Random) -> np.ndarray:
    side = 5
    patch = np.full((side, side), color, dtype=int)
    candidates = [(r, c) for r in range(1, side - 1, 2) for c in range(1, side - 1, 2)]
    chosen = rng.sample(candidates, hole_count)
    for r, c in chosen:
        patch[r, c] = BLACK
    return patch


def _make_body_mask(hole_count: int, tail_side: str, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        core_h = rng.choice([5, 7])
        core_w = rng.choice([5, 7])
        candidates = [(r, c) for r in range(1, core_h - 1, 2) for c in range(1, core_w - 1, 2)]
        if len(candidates) < hole_count:
            continue

        cells = {(r, c) for r in range(core_h) for c in range(core_w)}
        for r, c in rng.sample(candidates, hole_count):
            cells.remove((r, c))

        protrusion_side = rng.choice([side for side in SIDES if side != tail_side])
        if protrusion_side in ("left", "right"):
            span = rng.randint(1, min(2, core_h))
            start = rng.randint(0, core_h - span)
            extra_col = -1 if protrusion_side == "left" else core_w
            for rr in range(start, start + span):
                cells.add((rr, extra_col))
        else:
            span = rng.randint(1, min(2, core_w))
            start = rng.randint(0, core_w - span)
            extra_row = -1 if protrusion_side == "top" else core_h
            for cc in range(start, start + span):
                cells.add((extra_row, cc))

        mask = _mask_from_coords(list(cells))
        if len(_components(mask, connectivity=8)) != 1:
            continue
        if _count_holes(mask) != hole_count:
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to create asymmetric body mask")


def _build_object_descriptor(
    hole_count: int,
    tail_side: str,
    target_color: int | None,
    rng: random.Random,
) -> dict:
    body_mask = _make_body_mask(hole_count, tail_side, rng)
    bh, bw = body_mask.shape
    if tail_side in ("left", "right"):
        tail_len = rng.randint(1, min(3, bh))
        start = rng.randint(0, bh - tail_len)
        tail = [
            (start + i, -1 if tail_side == "left" else bw)
            for i in range(tail_len)
        ]
    else:
        tail_len = rng.randint(1, min(3, bw))
        start = rng.randint(0, bw - tail_len)
        tail = [
            (-1 if tail_side == "top" else bh, start + i)
            for i in range(tail_len)
        ]

    body_coords = [(r, c) for r in range(bh) for c in range(bw) if body_mask[r, c]]
    all_coords = body_coords + tail
    norm_all = _normalize_coords(all_coords)
    body_set = set(body_coords)
    tail_set = set(tail)

    comp_h = max(r for r, _ in norm_all) + 1
    comp_w = max(c for _, c in norm_all) + 1
    input_patch = np.full((comp_h, comp_w), BLACK, dtype=int)
    tail_patch = np.zeros((comp_h, comp_w), dtype=bool)

    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)

    body_points_norm: list[tuple[int, int]] = []
    tail_points_norm: list[tuple[int, int]] = []
    for r, c in body_set:
        rr = r - min_r
        cc = c - min_c
        body_points_norm.append((rr, cc))
        input_patch[rr, cc] = GREY
    for r, c in tail_set:
        rr = r - min_r
        cc = c - min_c
        tail_points_norm.append((rr, cc))
        input_patch[rr, cc] = AZURE
        tail_patch[rr, cc] = True

    br0, bc0, br1, bc1 = _bbox(body_points_norm)
    output_patch = None
    if target_color is not None:
        output_patch = np.full((comp_h, comp_w), BLACK, dtype=int)
        output_patch[tail_patch] = AZURE
        rotated = np.rot90(body_mask, 2)
        for rr, cc in np.argwhere(rotated):
            output_patch[br0 + int(rr), bc0 + int(cc)] = target_color

    return {
        "hole_count": hole_count,
        "tail_side": tail_side,
        "input_patch": input_patch,
        "output_patch": output_patch,
        "shape": input_patch.shape,
        "bbox": (0, 0, comp_h - 1, comp_w - 1),
    }


def _rect(h: int, w: int, top: int, left: int) -> tuple[int, int, int, int]:
    return top, left, top + h - 1, left + w - 1


def _overlap_1d(a0: int, a1: int, b0: int, b1: int, pad: int) -> bool:
    return not (a1 + pad < b0 or b1 + pad < a0)


def _rect_conflicts(
    rect: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    pad: int,
) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if _overlap_1d(r0, r1, or0, or1, pad) and _overlap_1d(c0, c1, oc0, oc1, pad):
            return True
    return False


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    sub[mask] = patch[mask]


def _edge_position(
    tail_side: str,
    patch_shape: tuple[int, int],
    h: int,
    w: int,
    variable: int,
) -> tuple[int, int]:
    ph, pw = patch_shape
    if tail_side == "left":
        return variable, w - pw
    if tail_side == "right":
        return variable, 0
    if tail_side == "top":
        return h - ph, variable
    return 0, variable


def _valid_initial_deltas(
    tail_side: str,
    final_top: int,
    final_left: int,
    patch_shape: tuple[int, int],
    h: int,
    w: int,
) -> list[int]:
    ph, pw = patch_shape
    if tail_side == "left":
        return list(range(2, final_left))
    if tail_side == "right":
        return list(range(2, w - pw - final_left))
    if tail_side == "top":
        return list(range(2, final_top))
    return list(range(2, h - ph - final_top))


def _shift_for_input(tail_side: str, final_top: int, final_left: int, delta: int) -> tuple[int, int]:
    if tail_side == "left":
        return final_top, final_left - delta
    if tail_side == "right":
        return final_top, final_left + delta
    if tail_side == "top":
        return final_top - delta, final_left
    return final_top + delta, final_left


def _build_expected(
    h: int,
    w: int,
    legend_items: list[tuple[np.ndarray, int, int]],
    valid_objects: list[dict],
) -> np.ndarray:
    out = np.full((h, w), BLACK, dtype=int)
    for patch, top, left in legend_items:
        _paste_patch(out, patch, top, left)
    for desc in valid_objects:
        _paste_patch(out, desc["output_patch"], desc["final_top"], desc["final_left"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(28, 34)
    w = rng.randint(28, 36)

    legend_count = rng.randint(2, 4)
    legend_holes = rng.sample(HOLE_POOL, legend_count)
    legend_colors = rng.sample(PALETTE, legend_count)
    legend_map = dict(zip(legend_holes, legend_colors))

    legend_items: list[tuple[np.ndarray, int, int]] = []
    final_rects: list[tuple[int, int, int, int]] = []
    input_rects: list[tuple[int, int, int, int]] = []

    legend_left = 1
    for idx, hole_count in enumerate(legend_holes):
        patch = _make_holey_square(hole_count, legend_map[hole_count], rng)
        top = 1 + idx * 6
        left = legend_left
        if top + patch.shape[0] > h:
            return None
        rect = _rect(patch.shape[0], patch.shape[1], top, left)
        legend_items.append((patch, top, left))
        final_rects.append(rect)
        input_rects.append(rect)

    object_count = rng.randint(2, 5)
    include_invalid = rng.random() < 0.5
    invalid_count = rng.randint(1, object_count - 1) if include_invalid else 0
    statuses = [False] * invalid_count + [True] * (object_count - invalid_count)
    rng.shuffle(statuses)

    objects: list[dict] = []
    invalid_holes = [hc for hc in HOLE_POOL if hc not in legend_map]
    if include_invalid and not invalid_holes:
        return None

    for is_valid in statuses:
        tail_side = rng.choice(SIDES)
        hole_count = rng.choice(legend_holes) if is_valid else rng.choice(invalid_holes)
        target_color = legend_map.get(hole_count)
        objects.append(_build_object_descriptor(hole_count, tail_side, target_color, rng))

    valid_objects = [desc for desc, is_valid in zip(objects, statuses) if is_valid]
    invalid_objects = [desc for desc, is_valid in zip(objects, statuses) if not is_valid]

    valid_objects.sort(
        key=lambda d: (
            0 if d["tail_side"] in ("right", "bottom") else 1,
            -(d["shape"][0] * d["shape"][1]),
        )
    )

    for desc in valid_objects:
        ph, pw = desc["shape"]
        variable_max = h - ph if desc["tail_side"] in ("left", "right") else w - pw
        choices = list(range(variable_max + 1))
        rng.shuffle(choices)
        placed = False
        for var in choices:
            top, left = _edge_position(desc["tail_side"], desc["shape"], h, w, var)
            rect = _rect(ph, pw, top, left)
            if _rect_conflicts(rect, final_rects, pad=1):
                continue
            desc["final_top"] = top
            desc["final_left"] = left
            final_rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    for desc in sorted(valid_objects, key=lambda d: -(d["shape"][0] * d["shape"][1])):
        ph, pw = desc["shape"]
        deltas = _valid_initial_deltas(
            desc["tail_side"],
            desc["final_top"],
            desc["final_left"],
            desc["shape"],
            h,
            w,
        )
        rng.shuffle(deltas)
        placed = False
        for delta in deltas:
            top, left = _shift_for_input(desc["tail_side"], desc["final_top"], desc["final_left"], delta)
            rect = _rect(ph, pw, top, left)
            if _rect_conflicts(rect, input_rects, pad=1):
                continue
            desc["input_top"] = top
            desc["input_left"] = left
            input_rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    for desc in sorted(invalid_objects, key=lambda d: -(d["shape"][0] * d["shape"][1])):
        ph, pw = desc["shape"]
        rows = list(range(2, h - ph - 1))
        cols = list(range(2, w - pw - 1))
        rng.shuffle(rows)
        rng.shuffle(cols)
        placed = False
        for top in rows:
            for left in cols:
                rect = _rect(ph, pw, top, left)
                if _rect_conflicts(rect, input_rects, pad=1):
                    continue
                desc["input_top"] = top
                desc["input_left"] = left
                input_rects.append(rect)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    grid = np.full((h, w), BLACK, dtype=int)
    for patch, top, left in legend_items:
        _paste_patch(grid, patch, top, left)
    for desc in valid_objects + invalid_objects:
        _paste_patch(grid, desc["input_patch"], desc["input_top"], desc["input_left"])

    expected = _build_expected(h, w, legend_items, valid_objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
