from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
TRAIL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
GEN_RETRIES = 500
PLACE_RETRIES = 200


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _arrow_cells(r0: int, c0: int, size: int, direction: str, variant: int) -> list[tuple[int, int]]:
    if direction == "up":
        wing_cols = [c0, c0 + size - 1] if size == 3 else [c0, c0 + 1]
        tip_col = c0 + 1 if size == 3 else c0 + variant
        return [(r0 - 1, wing_cols[0]), (r0 - 1, wing_cols[1]), (r0 - 2, tip_col)]
    if direction == "down":
        wing_cols = [c0, c0 + size - 1] if size == 3 else [c0, c0 + 1]
        tip_col = c0 + 1 if size == 3 else c0 + variant
        return [(r0 + size, wing_cols[0]), (r0 + size, wing_cols[1]), (r0 + size + 1, tip_col)]
    if direction == "left":
        wing_rows = [r0, r0 + size - 1] if size == 3 else [r0, r0 + 1]
        tip_row = r0 + 1 if size == 3 else r0 + variant
        return [(wing_rows[0], c0 - 1), (wing_rows[1], c0 - 1), (tip_row, c0 - 2)]
    if direction == "right":
        wing_rows = [r0, r0 + size - 1] if size == 3 else [r0, r0 + 1]
        tip_row = r0 + 1 if size == 3 else r0 + variant
        return [(wing_rows[0], c0 + size), (wing_rows[1], c0 + size), (tip_row, c0 + size + 1)]
    raise ValueError(f"unknown direction: {direction}")


def _final_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], direction: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    if direction == "up":
        return 0, c0, bh - 1, c1
    if direction == "down":
        return h - bh, c0, h - 1, c1
    if direction == "left":
        return r0, 0, r1, bw - 1
    if direction == "right":
        return r0, w - bw, r1, w - 1
    raise ValueError(f"unknown direction: {direction}")


def _paint_trajectory(out: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> None:
    r0, c0, r1, c1 = bbox
    h, w = out.shape
    if direction == "up":
        out[0 : r1 + 1, c0 : c1 + 1] = TRAIL
    elif direction == "down":
        out[r0:h, c0 : c1 + 1] = TRAIL
    elif direction == "left":
        out[r0 : r1 + 1, 0 : c1 + 1] = TRAIL
    elif direction == "right":
        out[r0 : r1 + 1, c0:w] = TRAIL
    else:
        raise ValueError(f"unknown direction: {direction}")


def _detect_arrow_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    h, w = grid.shape
    candidates: list[tuple[str, set[tuple[int, int]]]] = []

    if r0 >= 2:
        wings = [(r0 - 1, c) for c in range(c0, c1 + 1) if int(grid[r0 - 1, c]) == TRAIL]
        tips = [(r0 - 2, c) for c in range(c0, c1 + 1) if int(grid[r0 - 2, c]) == TRAIL]
        if len(wings) == 2 and len(tips) == 1:
            wing_cs = sorted(c for _, c in wings)
            tip_c = tips[0][1]
            if wing_cs[0] <= tip_c <= wing_cs[-1]:
                candidates.append(("up", set(wings + tips)))

    if r1 + 2 < h:
        wings = [(r1 + 1, c) for c in range(c0, c1 + 1) if int(grid[r1 + 1, c]) == TRAIL]
        tips = [(r1 + 2, c) for c in range(c0, c1 + 1) if int(grid[r1 + 2, c]) == TRAIL]
        if len(wings) == 2 and len(tips) == 1:
            wing_cs = sorted(c for _, c in wings)
            tip_c = tips[0][1]
            if wing_cs[0] <= tip_c <= wing_cs[-1]:
                candidates.append(("down", set(wings + tips)))

    if c0 >= 2:
        wings = [(r, c0 - 1) for r in range(r0, r1 + 1) if int(grid[r, c0 - 1]) == TRAIL]
        tips = [(r, c0 - 2) for r in range(r0, r1 + 1) if int(grid[r, c0 - 2]) == TRAIL]
        if len(wings) == 2 and len(tips) == 1:
            wing_rs = sorted(r for r, _ in wings)
            tip_r = tips[0][0]
            if wing_rs[0] <= tip_r <= wing_rs[-1]:
                candidates.append(("left", set(wings + tips)))

    if c1 + 2 < w:
        wings = [(r, c1 + 1) for r in range(r0, r1 + 1) if int(grid[r, c1 + 1]) == TRAIL]
        tips = [(r, c1 + 2) for r in range(r0, r1 + 1) if int(grid[r, c1 + 2]) == TRAIL]
        if len(wings) == 2 and len(tips) == 1:
            wing_rs = sorted(r for r, _ in wings)
            tip_r = tips[0][0]
            if wing_rs[0] <= tip_r <= wing_rs[-1]:
                candidates.append(("right", set(wings + tips)))

    assert len(candidates) == 1, f"expected exactly one arrow for block at {bbox}, got {candidates}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored block to the edge indicated by its adjacent grey arrow and leave a grey motion trail.

    Input structure:
    - The background is black.
    - Several small solid colored squares appear in the interior; each one is either 2x2 or 3x3.
    - Every colored square has one adjacent grey 3-pixel arrow on one side.
    - The arrow has two wing pixels flush with the block edge and one tip pixel one step farther away, so it points up, down, left, or right.

    Rule:
    - Remove every grey arrow from the scene.
    - For each colored square, infer the push direction from its arrow.
    - Slide the whole square in that direction until its front edge reaches the grid boundary.
    - Fill the full swept stripe from the source position to the destination with grey.
    - If multiple swept stripes overlap, later blocks in row-major source order overwrite earlier stripes; finally paint each destination square in its original color.

    Key insight:
    - The arrow does not move with the block; it only encodes direction, and the real transformation is the grey stripe swept out by the block's translation to the boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    block_mask = (g != BG) & (g != TRAIL)
    blocks = []
    for comp in _components(block_mask, connectivity=4):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each block must be monochrome"
        color = next(iter(colors))
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        assert (bh, bw) in {(2, 2), (3, 3)}, f"unexpected block shape: {(bh, bw)}"
        assert len(comp) == bh * bw, "block must be a solid square"
        blocks.append({"bbox": bbox, "color": color})

    assert blocks, "expected at least one colored block"
    blocks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == TRAIL)}
    used_arrow_cells: set[tuple[int, int]] = set()

    for block in blocks:
        direction, arrow_cells = _detect_arrow_direction(g, block["bbox"])
        assert used_arrow_cells.isdisjoint(arrow_cells), "arrows must be distinct"
        used_arrow_cells |= arrow_cells
        block["direction"] = direction

    assert used_arrow_cells == grey_cells, "every grey pixel must belong to exactly one arrow"

    out = np.full_like(g, BG)
    for block in blocks:
        _paint_trajectory(out, block["bbox"], block["direction"])

    for block in blocks:
        fr0, fc0, fr1, fc1 = _final_bbox(block["bbox"], g.shape, block["direction"])
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = int(block["color"])

    return out


def _input_cells(obj: dict) -> list[tuple[int, int]]:
    r0, c0, size = obj["r0"], obj["c0"], obj["size"]
    cells = [(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)]
    cells.extend(_arrow_cells(r0, c0, size, obj["direction"], obj["variant"]))
    return cells


def _trajectory_mask(obj: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    bbox = (obj["r0"], obj["c0"], obj["r0"] + obj["size"] - 1, obj["c0"] + obj["size"] - 1)
    _paint_trajectory(out, bbox, obj["direction"])
    return out


def _dest_mask(obj: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    bbox = (obj["r0"], obj["c0"], obj["r0"] + obj["size"] - 1, obj["c0"] + obj["size"] - 1)
    r0, c0, r1, c1 = _final_bbox(bbox, shape, obj["direction"])
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _render_oracle(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    ordered = sorted(objects, key=lambda obj: (obj["r0"], obj["c0"]))
    for obj in ordered:
        bbox = (obj["r0"], obj["c0"], obj["r0"] + obj["size"] - 1, obj["c0"] + obj["size"] - 1)
        _paint_trajectory(out, bbox, obj["direction"])
    for obj in ordered:
        bbox = (obj["r0"], obj["c0"], obj["r0"] + obj["size"] - 1, obj["c0"] + obj["size"] - 1)
        r0, c0, r1, c1 = _final_bbox(bbox, shape, obj["direction"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(obj["color"])
    return out


def _ranges_for_position(h: int, w: int, size: int, direction: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if direction == "up":
        r_range = (2, h - size - 1)
        c_range = (1, w - size - 2)
    elif direction == "down":
        r_range = (1, h - size - 2)
        c_range = (1, w - size - 2)
    elif direction == "left":
        r_range = (1, h - size - 2)
        c_range = (2, w - size - 1)
    elif direction == "right":
        r_range = (1, h - size - 2)
        c_range = (1, w - size - 2)
    else:
        raise ValueError(f"unknown direction: {direction}")

    if r_range[0] > r_range[1] or c_range[0] > c_range[1]:
        return None
    return r_range, c_range


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved[nr, nc] = True


def _can_place(reserved: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if reserved[r, c]:
            return False
    return True


def _choose_directions(rng: random.Random, count: int, need_cross: bool) -> list[str]:
    dirs: list[str] = []
    if need_cross and count >= 2:
        dirs.append(rng.choice(("up", "down")))
        dirs.append(rng.choice(("left", "right")))
    while len(dirs) < count:
        dirs.append(rng.choice(DIRS))
    if count >= 2 and len(set(dirs)) == 1:
        dirs[-1] = rng.choice([d for d in DIRS if d != dirs[0]])
    rng.shuffle(dirs)
    return dirs


def _choose_sizes(rng: random.Random, count: int) -> list[int]:
    sizes = [rng.choice((2, 3)) for _ in range(count)]
    if count >= 2 and len(set(sizes)) == 1 and rng.random() < 0.6:
        sizes[-1] = 5 - sizes[0]
    return sizes


def _destinations_disjoint(objects: list[dict], shape: tuple[int, int]) -> bool:
    masks = [_dest_mask(obj, shape) for obj in objects]
    acc = np.zeros(shape, dtype=int)
    for mask in masks:
        acc += mask.astype(int)
    return bool(np.all(acc <= 1))


def _has_visible_crossing(objects: list[dict], shape: tuple[int, int]) -> bool:
    traj_masks = [_trajectory_mask(obj, shape) for obj in objects]
    dest_masks = [_dest_mask(obj, shape) for obj in objects]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            inter = traj_masks[i] & traj_masks[j] & ~dest_masks[i] & ~dest_masks[j]
            if np.any(inter):
                return True
    return False


def _build_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        r0, c0, size, color = obj["r0"], obj["c0"], obj["size"], obj["color"]
        grid[r0 : r0 + size, c0 : c0 + size] = color
        for r, c in _arrow_cells(r0, c0, size, obj["direction"], obj["variant"]):
            grid[r, c] = TRAIL
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 5)
    need_cross = rng.random() < 0.5

    min_side = 12 + 2 * count
    h = rng.randint(min_side, min(26, min_side + 4))
    w = rng.randint(min_side, min(26, min_side + 4))

    colors = rng.sample(PALETTE, count)
    directions = _choose_directions(rng, count, need_cross)
    sizes = _choose_sizes(rng, count)

    skeletons = []
    for color, direction, size in zip(colors, directions, sizes):
        skeletons.append(
            {
                "color": color,
                "direction": direction,
                "size": size,
                "variant": 0 if size == 3 else rng.randint(0, 1),
            }
        )

    skeletons.sort(key=lambda obj: obj["size"], reverse=True)
    reserved = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for spec in skeletons:
        ranges = _ranges_for_position(h, w, spec["size"], spec["direction"])
        if ranges is None:
            return None
        (r_lo, r_hi), (c_lo, c_hi) = ranges
        placed = False
        for _ in range(PLACE_RETRIES):
            obj = dict(spec)
            obj["r0"] = rng.randint(r_lo, r_hi)
            obj["c0"] = rng.randint(c_lo, c_hi)
            cells = _input_cells(obj)
            if not _can_place(reserved, cells):
                continue
            objects.append(obj)
            _mark_reserved(reserved, cells)
            placed = True
            break
        if not placed:
            return None

    if not _destinations_disjoint(objects, (h, w)):
        return None
    if need_cross and not _has_visible_crossing(objects, (h, w)):
        return None

    grid = _build_input((h, w), objects)
    oracle = _render_oracle((h, w), objects)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, oracle):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
