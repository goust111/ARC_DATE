import random
from collections import Counter, deque

import numpy as np


DIRS = ("up", "down", "left", "right")
DIR_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        pts.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r) + dr, int(c) + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                out[int(r), int(c)] = True
                break
    return out


def _has_hole(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any((~mask) & (~outside)))


def _shape_has_interior(mask: np.ndarray) -> bool:
    boundary = _boundary_mask(mask)
    return bool(np.any(mask & (~boundary)))


def _frame_pattern(height: int, width: int, opening: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if opening == "up":
        mask[0, 1:-1] = False
    elif opening == "down":
        mask[-1, 1:-1] = False
    elif opening == "left":
        mask[1:-1, 0] = False
    elif opening == "right":
        mask[1:-1, -1] = False
    else:
        raise ValueError(opening)
    return mask


def _match_frame(local_mask: np.ndarray) -> str | None:
    h, w = local_mask.shape
    if h < 3 or w < 3:
        return None
    for opening in DIRS:
        if np.array_equal(local_mask, _frame_pattern(h, w, opening)):
            return opening
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, int | str]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components(g == int(color)):
            if len(comp) < 6:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                local[r - r0, c - c0] = True
            opening = _match_frame(local)
            if opening is None:
                continue
            frames.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "color": int(color),
                    "opening": opening,
                }
            )
    frames.sort(key=lambda item: (int(item["r0"]), int(item["c0"]), int(item["color"])))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each inner object through the wall opening by the count of inner wall-color pixels.

    Input structure:
    - bg = most common color.
    - Each wall is a 1px hollow rectangle missing exactly one side.
    - A differently colored connected object sits inside each wall.
    - Wall-colored ammo pixels are scattered in the interior background.

    Rule:
    - Opening direction determines translation direction.
    - Ammo count N determines translation distance.
    - Remove the original object, then redraw only its 1px boundary at the translated position.
    - The translated boundary uses the wall color.
    - Walls and ammo remain unchanged. Out-of-bounds pixels are clipped.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one 3-sided frame"

    out = g.copy()
    paints: list[tuple[int, int, int]] = []
    erases: list[tuple[int, int]] = []

    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        wall_color = int(frame["color"])
        opening = str(frame["opening"])

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ammo_count = int(np.count_nonzero(interior == wall_color))
        assert ammo_count > 0, "each frame must contain at least one ammo pixel"

        object_mask = (interior != bg) & (interior != wall_color)
        assert np.any(object_mask), "each frame must contain one interior object"
        comps = _components(object_mask)
        assert len(comps) == 1, "interior object must be a single connected component"

        or0, oc0, or1, oc1 = _bbox(comps[0])
        object_crop = object_mask[or0 : or1 + 1, oc0 : oc1 + 1]
        outline = _boundary_mask(object_crop)
        dr, dc = DIR_VEC[opening]

        for rr, cc in comps[0]:
            erases.append((r0 + 1 + rr, c0 + 1 + cc))

        for rr, cc in np.argwhere(outline):
            nr = r0 + 1 + or0 + int(rr) + dr * ammo_count
            nc = c0 + 1 + oc0 + int(cc) + dc * ammo_count
            if 0 <= nr < h and 0 <= nc < w:
                paints.append((int(nr), int(nc), wall_color))

    for r, c in erases:
        out[r, c] = bg
    for r, c, color in paints:
        out[r, c] = color
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
COUNT_OPTIONS = (1, 2, 3)
COUNT_WEIGHTS = (2, 4, 3)
COLOR_POOL = tuple(range(10))
PLACEMENT_PAD = 1
STRUCTURE_TRIES = 120
SHAPE_TRIES = 120
PLAN_TRIES = 80


def _randint(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(lo, hi))


def _random_shape(rng: random.Random, require_interior: bool) -> np.ndarray:
    for _ in range(SHAPE_TRIES):
        if require_interior:
            kind = rng.choice(["rect", "notch", "thick_l"])
        else:
            kind = rng.choice(["rect", "notch", "thick_l", "t"])

        if kind == "rect":
            h_min = 3 if require_interior else 2
            w_min = 3 if require_interior else 2
            h = _randint(rng, h_min, 5)
            w = _randint(rng, w_min, 5)
            mask = np.ones((h, w), dtype=bool)
        elif kind == "notch":
            h = _randint(rng, 3, 5)
            w = _randint(rng, 3, 5)
            mask = np.ones((h, w), dtype=bool)
            nh = _randint(rng, 1, h - 1)
            nw = _randint(rng, 1, w - 1)
            corner = rng.choice(["tl", "tr", "bl", "br"])
            if corner == "tl":
                mask[:nh, :nw] = False
            elif corner == "tr":
                mask[:nh, w - nw :] = False
            elif corner == "bl":
                mask[h - nh :, :nw] = False
            else:
                mask[h - nh :, w - nw :] = False
        elif kind == "thick_l":
            h = _randint(rng, 3, 5)
            w = _randint(rng, 3, 5)
            t_max = min(2, h - 1, w - 1)
            t_min = 2 if require_interior and t_max >= 2 else 1
            t = _randint(rng, t_min, t_max)
            mask = np.zeros((h, w), dtype=bool)
            mask[:, :t] = True
            mask[h - t :, :] = True
        else:
            h = _randint(rng, 3, 5)
            w = _randint(rng, 3, 5)
            bar_t = _randint(rng, 1, min(2, h - 1))
            stem_w = _randint(rng, 1, min(2, w))
            mask = np.zeros((h, w), dtype=bool)
            mask[:bar_t, :] = True
            c0 = (w - stem_w) // 2
            mask[:, c0 : c0 + stem_w] = True

        mask = np.rot90(mask, k=_randint(rng, 0, 3))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.3:
            mask = np.flipud(mask)
        mask = _crop_mask(mask)
        if mask.sum() < 3:
            continue
        if len(_components(mask)) != 1:
            continue
        if _has_hole(mask):
            continue
        if require_interior and not _shape_has_interior(mask):
            continue
        return mask.astype(bool)
    raise RuntimeError("failed to create a valid shape")


def _local_object_mask(
    interior_h: int, interior_w: int, top: int, left: int, shape: np.ndarray
) -> np.ndarray:
    mask = np.zeros((interior_h, interior_w), dtype=bool)
    sh, sw = shape.shape
    mask[top : top + sh, left : left + sw] = shape
    return mask


def _candidate_ammo_cells(
    frame_mask: np.ndarray, object_mask: np.ndarray
) -> list[tuple[int, int]]:
    interior_h, interior_w = object_mask.shape
    cands: list[tuple[int, int]] = []
    for r in range(interior_h):
        for c in range(interior_w):
            if object_mask[r, c]:
                continue
            fr, fc = r + 1, c + 1
            touches_wall = False
            for dr, dc in NEIGHBORS4:
                nr, nc = fr + dr, fc + dc
                if 0 <= nr < frame_mask.shape[0] and 0 <= nc < frame_mask.shape[1] and frame_mask[nr, nc]:
                    touches_wall = True
                    break
            if not touches_wall:
                cands.append((r, c))
    return cands


def _plan_structure(rng: random.Random, shape: np.ndarray, opening: str) -> dict[str, object] | None:
    sh, sw = shape.shape
    for _ in range(PLAN_TRIES):
        if opening == "right":
            top_gap = _randint(rng, 1, 4)
            bottom_gap = _randint(rng, 1, 4)
            back_gap = _randint(rng, 1, 5)
            open_gap = _randint(rng, 0, 2)
            interior_h = top_gap + sh + bottom_gap
            interior_w = back_gap + sw + open_gap
            obj_top, obj_left = top_gap, back_gap
            min_exit = sw + open_gap + 1
        elif opening == "left":
            top_gap = _randint(rng, 1, 4)
            bottom_gap = _randint(rng, 1, 4)
            open_gap = _randint(rng, 0, 2)
            back_gap = _randint(rng, 1, 5)
            interior_h = top_gap + sh + bottom_gap
            interior_w = open_gap + sw + back_gap
            obj_top, obj_left = top_gap, open_gap
            min_exit = sw + open_gap + 1
        elif opening == "down":
            left_gap = _randint(rng, 1, 4)
            right_gap = _randint(rng, 1, 4)
            back_gap = _randint(rng, 1, 5)
            open_gap = _randint(rng, 0, 2)
            interior_h = back_gap + sh + open_gap
            interior_w = left_gap + sw + right_gap
            obj_top, obj_left = back_gap, left_gap
            min_exit = sh + open_gap + 1
        else:
            left_gap = _randint(rng, 1, 4)
            right_gap = _randint(rng, 1, 4)
            open_gap = _randint(rng, 0, 2)
            back_gap = _randint(rng, 1, 5)
            interior_h = open_gap + sh + back_gap
            interior_w = left_gap + sw + right_gap
            obj_top, obj_left = open_gap, left_gap
            min_exit = sh + open_gap + 1

        frame_h, frame_w = interior_h + 2, interior_w + 2
        frame_mask = _frame_pattern(frame_h, frame_w, opening)
        object_mask = _local_object_mask(interior_h, interior_w, obj_top, obj_left, shape)
        ammo_candidates = _candidate_ammo_cells(frame_mask, object_mask)
        if len(ammo_candidates) < min_exit:
            continue

        max_exit = min(len(ammo_candidates), min_exit + 5)
        if max_exit < min_exit:
            continue
        distance = _randint(rng, min_exit, max_exit)
        rng.shuffle(ammo_candidates)
        ammo = ammo_candidates[:distance]

        obj_r0 = 1 + obj_top
        obj_c0 = 1 + obj_left
        obj_r1 = obj_r0 + sh - 1
        obj_c1 = obj_c0 + sw - 1
        dr, dc = DIR_VEC[opening]
        land_r0 = obj_r0 + dr * distance
        land_c0 = obj_c0 + dc * distance
        land_r1 = obj_r1 + dr * distance
        land_c1 = obj_c1 + dc * distance
        span_r0 = min(0, land_r0)
        span_c0 = min(0, land_c0)
        span_r1 = max(frame_h - 1, land_r1)
        span_c1 = max(frame_w - 1, land_c1)

        return {
            "shape": shape,
            "opening": opening,
            "distance": int(distance),
            "ammo": ammo,
            "object_top": int(obj_top),
            "object_left": int(obj_left),
            "interior_h": int(interior_h),
            "interior_w": int(interior_w),
            "frame_h": int(frame_h),
            "frame_w": int(frame_w),
            "frame_mask": frame_mask,
            "span": (int(span_r0), int(span_c0), int(span_r1), int(span_c1)),
        }
    return None


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_structure(
    rng: random.Random,
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    plan: dict[str, object],
    bg: int,
    wall_color: int,
    object_color: int,
) -> bool:
    h, w = grid.shape
    span_r0, span_c0, span_r1, span_c1 = plan["span"]  # type: ignore[misc]
    lo_r = PLACEMENT_PAD - int(span_r0)
    hi_r = h - 1 - PLACEMENT_PAD - int(span_r1)
    lo_c = PLACEMENT_PAD - int(span_c0)
    hi_c = w - 1 - PLACEMENT_PAD - int(span_c1)
    if lo_r > hi_r or lo_c > hi_c:
        return False

    frame_h = int(plan["frame_h"])
    frame_w = int(plan["frame_w"])
    frame_mask = np.asarray(plan["frame_mask"], dtype=bool)
    shape = np.asarray(plan["shape"], dtype=bool)
    obj_top = int(plan["object_top"])
    obj_left = int(plan["object_left"])
    ammo = list(plan["ammo"])  # type: ignore[misc]

    for _ in range(STRUCTURE_TRIES):
        top = _randint(rng, lo_r, hi_r)
        left = _randint(rng, lo_c, hi_c)
        new_rect = (
            top + int(span_r0) - PLACEMENT_PAD,
            left + int(span_c0) - PLACEMENT_PAD,
            top + int(span_r1) + PLACEMENT_PAD,
            left + int(span_c1) + PLACEMENT_PAD,
        )
        if any(_rects_intersect(new_rect, old) for old in reserved):
            continue

        local = np.full((frame_h, frame_w), bg, dtype=int)
        local[frame_mask] = wall_color
        sh, sw = shape.shape
        obj_slice = local[1 + obj_top : 1 + obj_top + sh, 1 + obj_left : 1 + obj_left + sw]
        obj_slice[shape] = object_color
        for r, c in ammo:
            local[1 + int(r), 1 + int(c)] = wall_color

        sub = grid[top : top + frame_h, left : left + frame_w]
        if np.any((sub != bg) & (local != bg)):
            continue
        sub[local != bg] = local[local != bg]
        reserved.append(new_rect)
        return True
    return False


def _choose_count(rng: random.Random, h: int, w: int) -> int:
    max_count = 3
    if min(h, w) < 22:
        max_count = 2
    options = [n for n in COUNT_OPTIONS if n <= max_count]
    weights = [COUNT_WEIGHTS[COUNT_OPTIONS.index(n)] for n in options]
    return int(rng.choices(options, weights=weights, k=1)[0])


def _valid_generated_grid(grid: np.ndarray, expected_count: int) -> bool:
    try:
        bg = get_bg(grid)
        frames = _find_frames(grid, bg)
        if len(frames) != expected_count:
            return False
        dirs = Counter(str(frame["opening"]) for frame in frames)
        if sum(dirs.values()) != expected_count:
            return False
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = _randint(rng, *GRID_H_RANGE)
    w = _randint(rng, *GRID_W_RANGE)
    bg = int(rng.choice(COLOR_POOL))
    count = _choose_count(rng, h, w)
    available = [c for c in COLOR_POOL if c != bg]
    if len(available) < 2 * count:
        return None
    chosen_colors = rng.sample(available, 2 * count)
    wall_colors = chosen_colors[::2]
    object_colors = chosen_colors[1::2]
    directions = rng.sample(list(DIRS), count)

    grid = np.full((h, w), bg, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []
    have_interior_shape = False

    for idx in range(count):
        require_interior = not have_interior_shape
        shape = _random_shape(rng, require_interior=require_interior)
        if _shape_has_interior(shape):
            have_interior_shape = True
        plan = _plan_structure(rng, shape, directions[idx])
        if plan is None:
            return None
        placed = _place_structure(
            rng=rng,
            grid=grid,
            reserved=reserved,
            plan=plan,
            bg=bg,
            wall_color=int(wall_colors[idx]),
            object_color=int(object_colors[idx]),
        )
        if not placed:
            return None

    if not have_interior_shape:
        return None
    if not _valid_generated_grid(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
