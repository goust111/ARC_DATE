import random
from collections import deque

import numpy as np


BG = 8
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _tight_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _is_solid_rect(mask: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1) and bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _seal_side(mask: np.ndarray, side: str) -> np.ndarray:
    sealed = mask.copy()
    if side == "up":
        sealed[0, :] = True
    elif side == "down":
        sealed[-1, :] = True
    elif side == "left":
        sealed[:, 0] = True
    else:
        sealed[:, -1] = True
    return sealed


def _extract_cavity(mask: np.ndarray) -> tuple[str, np.ndarray]:
    h, w = mask.shape
    candidates: list[tuple[str, np.ndarray]] = []
    for side in ("up", "down", "left", "right"):
        sealed = _seal_side(mask, side)
        inner = [comp for comp in _components(~sealed) if not _touches_edge(comp, h, w)]
        if len(inner) != 1:
            continue
        cavity = np.zeros((h, w), dtype=bool)
        cavity[inner[0][:, 0], inner[0][:, 1]] = True
        candidates.append((side, cavity))
    assert len(candidates) == 1, "mold must have exactly one opening side"
    return candidates[0]


def _slide(coords: np.ndarray, side: str, wall_mask: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape
    dr, dc = DIRS[side]
    steps = 0
    while True:
        rr = coords[:, 0] + dr * (steps + 1)
        cc = coords[:, 1] + dc * (steps + 1)
        if np.any(rr < 0) or np.any(rr >= h) or np.any(cc < 0) or np.any(cc >= w):
            break
        if np.any(wall_mask[rr, cc]):
            break
        steps += 1
    moved = coords.copy()
    moved[:, 0] += dr * steps
    moved[:, 1] += dc * steps
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    wall_mask = g == WALL

    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, WALL)]
    for color in colors:
        color_mask = g == color
        comps = _components(color_mask)
        assert len(comps) == 2, "each color must have one mold and one material block"

        solid = [comp for comp in comps if _is_solid_rect(color_mask, comp)]
        assert len(solid) == 1, "expected a unique solid rectangle"
        material = solid[0]
        mold = comps[0] if comps[0] is not material else comps[1]

        mr0, mc0, mr1, mc1 = _bbox(material)
        mold_r0, mold_c0, mold_r1, mold_c1 = _bbox(mold)

        mold_mask = np.zeros((mold_r1 - mold_r0 + 1, mold_c1 - mold_c0 + 1), dtype=bool)
        mold_mask[mold[:, 0] - mold_r0, mold[:, 1] - mold_c0] = True
        side, cavity = _extract_cavity(mold_mask)

        cut_mask, _ = _tight_mask(cavity)
        cut_coords = np.argwhere(cut_mask)
        mat_h = mr1 - mr0 + 1
        mat_w = mc1 - mc0 + 1
        assert cut_mask.shape[0] <= mat_h and cut_mask.shape[1] <= mat_w
        out[mr0 + cut_coords[:, 0], mc0 + cut_coords[:, 1]] = BG

        cavity_coords = np.argwhere(cavity)
        cavity_coords[:, 0] += mold_r0
        cavity_coords[:, 1] += mold_c0
        moved = _slide(cavity_coords, side, wall_mask)
        out[moved[:, 0], moved[:, 1]] = color

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _add_dent(mask: np.ndarray, side: str, start: int, end: int, depth: int) -> None:
    h, w = mask.shape
    if side == "up":
        mask[1 : depth + 1, start] = True
        mask[1 : depth + 1, end] = True
        mask[depth, start : end + 1] = True
    elif side == "down":
        row = h - 1 - depth
        mask[row : h - 1, start] = True
        mask[row : h - 1, end] = True
        mask[row, start : end + 1] = True
    elif side == "left":
        mask[start, 1 : depth + 1] = True
        mask[end, 1 : depth + 1] = True
        mask[start : end + 1, depth] = True
    else:
        col = w - 1 - depth
        mask[start, col : w - 1] = True
        mask[end, col : w - 1] = True
        mask[start : end + 1, col] = True


def _generate_mold(rng: random.Random, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        h = rng.randint(6, 9)
        w = rng.randint(6, 9)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        dent_sides = [name for name in ("up", "down", "left", "right") if name != side]
        rng.shuffle(dent_sides)
        dent_count = rng.randint(0, min(3, len(dent_sides)))
        for dent_side in dent_sides[:dent_count]:
            if dent_side in ("up", "down"):
                if w < 6 or h < 6:
                    continue
                start = rng.randint(1, w - 4)
                end = rng.randint(start + 2, w - 2)
                depth = rng.randint(1, min(3, h - 3))
            else:
                if h < 6 or w < 6:
                    continue
                start = rng.randint(1, h - 4)
                end = rng.randint(start + 2, h - 2)
                depth = rng.randint(1, min(3, w - 3))
            _add_dent(mask, dent_side, start, end, depth)

        open_mask = mask.copy()
        if side == "up":
            open_mask[0, 1:-1] = False
        elif side == "down":
            open_mask[-1, 1:-1] = False
        elif side == "left":
            open_mask[1:-1, 0] = False
        else:
            open_mask[1:-1, -1] = False

        if len(_components(open_mask)) != 1:
            continue
        try:
            found_side, cavity = _extract_cavity(open_mask)
        except AssertionError:
            continue
        if found_side != side:
            continue
        tight, _ = _tight_mask(cavity)
        if int(cavity.sum()) < 4:
            continue
        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        return open_mask, cavity
    return None


def _material_size(rng: random.Random, shape_h: int, shape_w: int, area: int) -> tuple[int, int]:
    mh = shape_h + rng.randint(0, 2)
    mw = shape_w + rng.randint(0, 2)
    if mh * mw <= area:
        if rng.random() < 0.5:
            mh += 1
        else:
            mw += 1
    return mh, mw


def _bundle_mask(
    grid_shape: tuple[int, int],
    mold_mask: np.ndarray,
    mold_pos: tuple[int, int],
    material_bbox: tuple[int, int, int, int],
    sweep_bbox: tuple[int, int, int, int],
    wall_cells: list[tuple[int, int]],
) -> np.ndarray:
    h, w = grid_shape
    bundle = np.zeros((h, w), dtype=bool)

    mr0, mc0 = mold_pos
    mh, mw = mold_mask.shape
    bundle[mr0 : mr0 + mh, mc0 : mc0 + mw] |= mold_mask

    ar0, ac0, ar1, ac1 = material_bbox
    bundle[ar0 : ar1 + 1, ac0 : ac1 + 1] = True

    sr0, sc0, sr1, sc1 = sweep_bbox
    bundle[sr0 : sr1 + 1, sc0 : sc1 + 1] = True

    for r, c in wall_cells:
        bundle[r, c] = True
    return bundle


def _sweep_bbox(
    cavity_bbox: tuple[int, int, int, int], side: str, steps: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = cavity_bbox
    dr, dc = DIRS[side]
    nr0 = r0 + dr * steps
    nc0 = c0 + dc * steps
    nr1 = r1 + dr * steps
    nc1 = c1 + dc * steps
    return min(r0, nr0), min(c0, nc0), max(r1, nr1), max(c1, nc1)


def _try_place_pair(
    grid: np.ndarray, reserved: np.ndarray, rng: random.Random, color: int
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = grid.shape
    side = rng.choice(("up", "down", "left", "right"))
    stop_with_wall = bool(rng.randint(0, 1))
    mold_data = _generate_mold(rng, side)
    if mold_data is None:
        return None
    mold_mask, cavity = mold_data
    cavity_tight, _ = _tight_mask(cavity)
    cut_area = int(cavity.sum())
    mat_h, mat_w = _material_size(rng, cavity_tight.shape[0], cavity_tight.shape[1], cut_area)

    mold_h, mold_w = mold_mask.shape
    cavity_local = np.argwhere(cavity)
    cav_r0, cav_c0, cav_r1, cav_c1 = _bbox(cavity_local)

    for _ in range(160):
        mold_r0 = rng.randint(0, h - mold_h)
        mold_c0 = rng.randint(0, w - mold_w)
        start_bbox = (
            mold_r0 + cav_r0,
            mold_c0 + cav_c0,
            mold_r0 + cav_r1,
            mold_c0 + cav_c1,
        )

        if side == "up":
            max_steps = start_bbox[0]
        elif side == "down":
            max_steps = h - 1 - start_bbox[2]
        elif side == "left":
            max_steps = start_bbox[1]
        else:
            max_steps = w - 1 - start_bbox[3]
        if max_steps < 2:
            continue

        if stop_with_wall:
            if max_steps < 3:
                continue
            steps = rng.randint(2, max_steps - 1)
        else:
            steps = max_steps

        final_bbox = (
            start_bbox[0] + DIRS[side][0] * steps,
            start_bbox[1] + DIRS[side][1] * steps,
            start_bbox[2] + DIRS[side][0] * steps,
            start_bbox[3] + DIRS[side][1] * steps,
        )

        wall_cells: list[tuple[int, int]] = []
        if stop_with_wall:
            if side == "up":
                wr = final_bbox[0] - 1
                wall_cells = [(wr, c) for c in range(final_bbox[1], final_bbox[3] + 1)]
            elif side == "down":
                wr = final_bbox[2] + 1
                wall_cells = [(wr, c) for c in range(final_bbox[1], final_bbox[3] + 1)]
            elif side == "left":
                wc = final_bbox[1] - 1
                wall_cells = [(r, wc) for r in range(final_bbox[0], final_bbox[2] + 1)]
            else:
                wc = final_bbox[3] + 1
                wall_cells = [(r, wc) for r in range(final_bbox[0], final_bbox[2] + 1)]
            if any(not (0 <= r < h and 0 <= c < w) for r, c in wall_cells):
                continue

        pre_bundle = np.zeros((h, w), dtype=bool)
        pre_bundle[mold_r0 : mold_r0 + mold_h, mold_c0 : mold_c0 + mold_w] |= mold_mask
        sr0, sc0, sr1, sc1 = _sweep_bbox(start_bbox, side, steps)
        pre_bundle[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
        for r, c in wall_cells:
            pre_bundle[r, c] = True
        if np.any(_dilate(pre_bundle) & reserved):
            continue

        local_forbidden = _dilate(pre_bundle)
        mat_positions: list[tuple[int, int]] = []
        for ar0 in range(h - mat_h + 1):
            for ac0 in range(w - mat_w + 1):
                rect = np.zeros((h, w), dtype=bool)
                rect[ar0 : ar0 + mat_h, ac0 : ac0 + mat_w] = True
                if np.any(rect & local_forbidden):
                    continue
                if np.any(_dilate(rect) & reserved):
                    continue
                mat_positions.append((ar0, ac0))
        if not mat_positions:
            continue

        mat_r0, mat_c0 = rng.choice(mat_positions)
        material_bbox = (mat_r0, mat_c0, mat_r0 + mat_h - 1, mat_c0 + mat_w - 1)
        bundle = _bundle_mask(
            (h, w), mold_mask, (mold_r0, mold_c0), material_bbox, (sr0, sc0, sr1, sc1), wall_cells
        )
        if np.any(_dilate(bundle) & reserved):
            continue

        pair_reserved = _dilate(bundle)
        new_grid = grid.copy()
        new_reserved = reserved.copy()

        new_grid[mold_r0 : mold_r0 + mold_h, mold_c0 : mold_c0 + mold_w][mold_mask] = color
        new_grid[mat_r0 : mat_r0 + mat_h, mat_c0 : mat_c0 + mat_w] = color
        for r, c in wall_cells:
            new_grid[r, c] = WALL
        new_reserved |= pair_reserved
        return new_grid, new_reserved

    return None


def _add_extra_walls(grid: np.ndarray, reserved: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    new_grid = grid.copy()
    new_reserved = reserved.copy()
    target = rng.randint(1, 3)
    added = 0
    for _ in range(120):
        horizontal = bool(rng.randint(0, 1))
        length = rng.randint(3, 7)
        if horizontal:
            if length > w:
                continue
            r = rng.randint(0, h - 1)
            c = rng.randint(0, w - length)
            seg = [(r, cc) for cc in range(c, c + length)]
        else:
            if length > h:
                continue
            r = rng.randint(0, h - length)
            c = rng.randint(0, w - 1)
            seg = [(rr, c) for rr in range(r, r + length)]
        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in seg:
            mask[rr, cc] = True
        if np.any(_dilate(mask) & new_reserved):
            continue
        new_grid[mask] = WALL
        new_reserved |= _dilate(mask)
        added += 1
        if added >= target:
            break
    return new_grid, new_reserved


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    pair_count = rng.choice((1, 2, 2, 3))
    colors = rng.sample(PALETTE, pair_count)
    for color in colors:
        placed = _try_place_pair(grid, reserved, rng, color)
        if placed is None:
            return None
        grid, reserved = placed

    if not np.any(grid == WALL) or rng.random() < 0.5:
        grid, reserved = _add_extra_walls(grid, reserved, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
