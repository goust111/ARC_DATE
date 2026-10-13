import random

import numpy as np


BLACK = 0
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (20, 32)
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
CORE_H_HOLLOW_RANGE = (5, 9)
CORE_W_HOLLOW_RANGE = (5, 10)
CORE_H_SOLID_RANGE = (3, 8)
CORE_W_SOLID_RANGE = (3, 9)
PROTRUSION_THICKNESS_RANGE = (1, 2)
PROTRUSION_COUNT_RANGE = (0, 2)
HOLE_COUNT_RANGE = (1, 3)
HOLE_BBOX_H_RANGE = (1, 4)
HOLE_BBOX_W_RANGE = (1, 4)
PLACEMENT_MARGIN = 1
GENERATION_RETRIES = 400
OBJECT_RETRIES = 120
PLACEMENT_RETRIES = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _component_holes(grid: np.ndarray, color: int, coords: np.ndarray) -> list[dict]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    obj_mask = patch == int(color)
    bg_mask = patch == BLACK
    holes: list[dict] = []

    for hole_coords in _components(bg_mask):
        if np.any(hole_coords[:, 0] == 0) or np.any(hole_coords[:, 0] == obj_mask.shape[0] - 1):
            continue
        if np.any(hole_coords[:, 1] == 0) or np.any(hole_coords[:, 1] == obj_mask.shape[1] - 1):
            continue

        hr0, hc0, hr1, hc1 = _bbox(hole_coords)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        global_coords = hole_coords.copy()
        global_coords[:, 0] += r0
        global_coords[:, 1] += c0
        for rr, cc in hole_coords:
            mask[int(rr) - hr0, int(cc) - hc0] = True

        # Every side-neighbor around a true hole must belong to the enclosing object.
        for rr, cc in hole_coords:
            for dr, dc in DIR4:
                nr = int(rr) + dr
                nc = int(cc) + dc
                if hr0 <= nr <= hr1 and hc0 <= nc <= hc1 and mask[nr - hr0, nc - hc0]:
                    continue
                assert obj_mask[nr, nc], "enclosed hole must be surrounded by object pixels"

        holes.append(
            {
                "coords": global_coords,
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "mask": mask,
            }
        )

    holes.sort(key=lambda hole: (int(hole["bbox"][1]), int(hole["bbox"][0])))
    return holes


def _foreground_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == BLACK:
            continue
        for coords in _components(grid == color):
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    comps.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["color"])))
    return comps


def _copy_extent(holes: list[dict]) -> tuple[int, int]:
    if not holes:
        return 0, 0
    height = max(int(hole["mask"].shape[0]) for hole in holes)
    width = sum(int(hole["mask"].shape[1]) for hole in holes) + max(0, len(holes) - 1)
    return height, width


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each enclosed black hole from a foreground object, fill the hole in place,
    and redraw the hole shapes in red directly below the object's bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    h, w = g.shape

    for comp in _foreground_components(g):
        color = int(comp["color"])
        r0, c0, r1, _c1 = comp["bbox"]
        holes = _component_holes(g, color, comp["coords"])
        if not holes:
            continue

        for hole in holes:
            coords = hole["coords"]
            out[coords[:, 0], coords[:, 1]] = color

        cursor = int(c0)
        top = int(r1 + 1)
        for hole in holes:
            mask = hole["mask"]
            hh, hw = map(int, mask.shape)
            assert top + hh <= h, "not enough rows to place hole copy"
            assert cursor + hw <= w, "not enough columns to place hole copy"
            target = out[top : top + hh, cursor : cursor + hw]
            assert np.all(target[mask] == BLACK), "hole copy area must start as background"
            target[mask] = RED
            cursor += hw + 1

    return out


def _grow_connected_mask(height: int, width: int, target: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (rng.randrange(height), rng.randrange(width))
    mask[seed] = True
    cells = {seed}
    while len(cells) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nxt = rng.choice(tuple(frontier))
        cells.add(nxt)
        mask[nxt] = True
    return _normalize_mask(mask)


def _random_hole_mask(max_h: int, max_w: int, rng: random.Random) -> np.ndarray:
    raw_h = rng.randint(1, max_h)
    raw_w = rng.randint(1, max_w)
    area = raw_h * raw_w
    target = rng.randint(1, area)
    return _grow_connected_mask(raw_h, raw_w, target, rng)


def _add_protrusions(mask: np.ndarray, core_top: int, core_left: int, core_h: int, core_w: int, rng: random.Random) -> None:
    n_protrusions = rng.randint(*PROTRUSION_COUNT_RANGE)
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[:n_protrusions]:
        thickness = rng.randint(*PROTRUSION_THICKNESS_RANGE)
        if side in ("top", "bottom"):
            if core_w < 2:
                continue
            span = rng.randint(2, core_w)
            start = rng.randint(0, core_w - span)
            rr = slice(core_top - thickness, core_top) if side == "top" else slice(core_top + core_h, core_top + core_h + thickness)
            cc = slice(core_left + start, core_left + start + span)
        else:
            if core_h < 2:
                continue
            span = rng.randint(2, core_h)
            start = rng.randint(0, core_h - span)
            rr = slice(core_top + start, core_top + start + span)
            cc = slice(core_left - thickness, core_left) if side == "left" else slice(core_left + core_w, core_left + core_w + thickness)
        mask[rr, cc] = True


def _make_object(has_holes: bool, rng: random.Random) -> dict | None:
    for _ in range(OBJECT_RETRIES):
        if has_holes:
            core_h = rng.randint(*CORE_H_HOLLOW_RANGE)
            core_w = rng.randint(*CORE_W_HOLLOW_RANGE)
        else:
            core_h = rng.randint(*CORE_H_SOLID_RANGE)
            core_w = rng.randint(*CORE_W_SOLID_RANGE)

        canvas_h = core_h + 6
        canvas_w = core_w + 6
        core_top = 3
        core_left = 3
        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        mask[core_top : core_top + core_h, core_left : core_left + core_w] = True
        _add_protrusions(mask, core_top, core_left, core_h, core_w, rng)

        holes: list[dict] = []
        signatures: set[tuple[tuple[int, int], bytes]] = set()
        occupancy = np.zeros((core_h, core_w), dtype=bool)
        if has_holes:
            max_holes = min(HOLE_COUNT_RANGE[1], max(1, ((core_h - 2) * (core_w - 2)) // 6))
            if max_holes < 1:
                continue
            hole_count = rng.randint(HOLE_COUNT_RANGE[0], max_holes)
            for _hole_index in range(hole_count):
                placed = False
                for _ in range(60):
                    hole_mask = _random_hole_mask(min(HOLE_BBOX_H_RANGE[1], core_h - 2), min(HOLE_BBOX_W_RANGE[1], core_w - 2), rng)
                    hh, hw = map(int, hole_mask.shape)
                    if core_h - hh - 1 <= 0 or core_w - hw - 1 <= 0:
                        continue
                    sig = _hole_signature(hole_mask)
                    if sig in signatures:
                        continue

                    local_top = rng.randint(1, core_h - hh - 1)
                    local_left = rng.randint(1, core_w - hw - 1)
                    pad_top = local_top - 1
                    pad_left = local_left - 1
                    pad_bottom = local_top + hh + 1
                    pad_right = local_left + hw + 1
                    if occupancy[pad_top:pad_bottom, pad_left:pad_right].any():
                        continue

                    mask[
                        core_top + local_top : core_top + local_top + hh,
                        core_left + local_left : core_left + local_left + hw,
                    ][hole_mask] = False
                    occupancy[pad_top:pad_bottom, pad_left:pad_right] = True
                    holes.append({"top": local_top, "left": local_left, "mask": hole_mask})
                    signatures.add(sig)
                    placed = True
                    break
                if not placed:
                    break
            if len(holes) != hole_count:
                continue

        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
        if len(_components(cropped)) != 1:
            continue

        copy_h, copy_w = _copy_extent(holes)
        return {
            "mask": cropped,
            "has_holes": has_holes,
            "copy_h": int(copy_h),
            "copy_w": int(copy_w),
        }
    return None


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    obj: dict,
    color: int,
    rng: random.Random,
) -> bool:
    mask = obj["mask"]
    mask_h, mask_w = map(int, mask.shape)
    copy_h = int(obj["copy_h"])
    copy_w = int(obj["copy_w"])
    reserve_h = mask_h + copy_h
    reserve_w = max(mask_w, copy_w)

    h, w = grid.shape
    min_top = PLACEMENT_MARGIN
    min_left = PLACEMENT_MARGIN
    max_top = h - reserve_h - PLACEMENT_MARGIN
    max_left = w - reserve_w - PLACEMENT_MARGIN
    if max_top < min_top or max_left < min_left:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(min_top, max_top)
        left = rng.randint(min_left, max_left)

        pad_top = top - PLACEMENT_MARGIN
        pad_left = left - PLACEMENT_MARGIN
        pad_bottom = top + reserve_h + PLACEMENT_MARGIN
        pad_right = left + reserve_w + PLACEMENT_MARGIN
        if blocked[pad_top:pad_bottom, pad_left:pad_right].any():
            continue

        target = grid[top : top + mask_h, left : left + mask_w]
        if np.any(target[mask] != BLACK):
            continue

        target[mask] = int(color)
        blocked[pad_top:pad_bottom, pad_left:pad_right] = True
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    colors = rng.sample(OBJECT_COLORS, n_objects)
    hollow_flags = [rng.random() < 0.5 for _ in range(n_objects)]
    if not any(hollow_flags):
        hollow_flags[rng.randrange(n_objects)] = True

    order = list(range(n_objects))
    rng.shuffle(order)
    for idx in order:
        obj = _make_object(hollow_flags[idx], rng)
        if obj is None:
            return None
        if not _place_object(grid, blocked, obj, colors[idx], rng):
            return None

    # Validate against the transformation assumptions.
    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert np.any(out == RED), "generated sample must include at least one copied hole"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
