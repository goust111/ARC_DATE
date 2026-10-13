from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, object_interior, paste


BLACK = Color.BLACK
GRAY = Color.GREY
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")
DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400


def _arrow_mask(direction: str, shaft_len: int) -> np.ndarray:
    shaft_len = int(shaft_len)
    if shaft_len < 1:
        raise ValueError("shaft_len must be >= 1")

    if direction == "right":
        mask = np.zeros((5, shaft_len + 3), dtype=bool)
        mask[2, :shaft_len] = True
        mask[:, shaft_len] = True
        mask[1:4, shaft_len + 1] = True
        mask[2, shaft_len + 2] = True
        return mask
    if direction == "left":
        mask = np.zeros((5, shaft_len + 3), dtype=bool)
        mask[2, shaft_len + 2] = True
        mask[1:4, shaft_len + 1] = True
        mask[:, shaft_len] = True
        mask[2, :shaft_len] = True
        return np.fliplr(mask)
    if direction == "down":
        mask = np.zeros((shaft_len + 3, 5), dtype=bool)
        mask[:shaft_len, 2] = True
        mask[shaft_len, :] = True
        mask[shaft_len + 1, 1:4] = True
        mask[shaft_len + 2, 2] = True
        return mask
    if direction == "up":
        mask = np.zeros((shaft_len + 3, 5), dtype=bool)
        mask[shaft_len + 2, 2] = True
        mask[shaft_len + 1, 1:4] = True
        mask[shaft_len, :] = True
        mask[:shaft_len, 2] = True
        return np.flipud(mask)
    raise ValueError(f"invalid direction: {direction}")


def _parse_hole(hole_mask: np.ndarray) -> tuple[str, str | None, int | None]:
    h, w = hole_mask.shape
    if h == w and np.all(hole_mask):
        return "square", None, None

    if h == 5 and w >= 4:
        shaft_len = w - 3
        for direction in ("left", "right"):
            if np.array_equal(hole_mask, _arrow_mask(direction, shaft_len)):
                return "arrow", direction, shaft_len
    if w == 5 and h >= 4:
        shaft_len = h - 3
        for direction in ("up", "down"):
            if np.array_equal(hole_mask, _arrow_mask(direction, shaft_len)):
                return "arrow", direction, shaft_len
    raise AssertionError("unrecognized hole shape")


def _object_hits_on_ray(
    objects: list[dict],
    *,
    direction: str,
    axis: int,
    block_bbox: tuple[int, int, int, int],
) -> list[tuple[int, dict]]:
    r0, c0, r1, c1 = map(int, block_bbox)
    hits: list[tuple[int, dict]] = []

    for obj in objects:
        coords = obj["coords"]
        if direction in ("left", "right"):
            on_axis = coords[coords[:, 0] == axis]
            if on_axis.size == 0:
                continue
            cols = on_axis[:, 1]
            if direction == "right":
                near = int(cols.min())
                if near <= c1:
                    continue
                dist = near - c1
            else:
                near = int(cols.max())
                if near >= c0:
                    continue
                dist = c0 - near
        else:
            on_axis = coords[coords[:, 1] == axis]
            if on_axis.size == 0:
                continue
            rows = on_axis[:, 0]
            if direction == "down":
                near = int(rows.min())
                if near <= r1:
                    continue
                dist = near - r1
            else:
                near = int(rows.max())
                if near >= r0:
                    continue
                dist = r0 - near
        hits.append((dist, obj))

    hits.sort(key=lambda item: item[0])
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each gray block's internal hole as either a passive square or an arrow instruction.

    Input structure:
    - The background is black.
    - Foreground includes several solid gray blocks, each with exactly one enclosed black hole.
    - Other foreground components are isolated colored objects elsewhere on the grid.

    Rule:
    - A square hole means its gray block stays unchanged.
    - An arrow-shaped hole encodes a direction and a shaft length k; cast a ray from the block along that axis and count colored objects hit in order.
    - Recolor that block's gray cells with the k-th hit object's color, keep the hole black, and erase the selected target object from the grid.

    Key insight:
    - The hole is not damage to repair but a glyph to read: square holes are inert, while arrow orientation gives the ray and shaft length gives which along-ray object is absorbed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BLACK

    gray_blocks = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[GRAY],
        return_format="dict",
    )
    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=OBJECT_COLORS,
        return_format="dict",
    )

    out = g.copy()
    chosen_targets: list[dict] = []

    for block in gray_blocks:
        br0, bc0, br1, bc1 = map(int, block["bbox"])
        block_crop = crop_bbox(g, block["bbox"])
        hole_mask = (block_crop == bg) & object_interior(block_crop, background=bg)
        hole_coords = np.argwhere(hole_mask)
        assert hole_coords.size > 0, "gray block must contain one enclosed hole"

        hr0 = int(hole_coords[:, 0].min())
        hc0 = int(hole_coords[:, 1].min())
        hr1 = int(hole_coords[:, 0].max())
        hc1 = int(hole_coords[:, 1].max())
        hole_crop = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]
        hole_kind, direction, shaft_len = _parse_hole(hole_crop)
        if hole_kind == "square":
            continue

        assert direction is not None and shaft_len is not None
        if direction in ("left", "right"):
            axis = br0 + hr0 + hole_crop.shape[0] // 2
        else:
            axis = bc0 + hc0 + hole_crop.shape[1] // 2

        hits = _object_hits_on_ray(objects, direction=direction, axis=axis, block_bbox=block["bbox"])
        assert len(hits) >= shaft_len, "ray must have at least k objects"
        target = hits[shaft_len - 1][1]
        chosen_targets.append(target)
        out[block["coords"][:, 0], block["coords"][:, 1]] = int(target["color"])

    target_ids = {id(target) for target in chosen_targets}
    assert len(target_ids) == len(chosen_targets), "selected targets must be unique"
    for target in chosen_targets:
        out[target["coords"][:, 0], target["coords"][:, 1]] = bg

    return out


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _patch_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(patch != BLACK)
    return [(int(top + r), int(left + c)) for r, c in coords]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _square_block_patch() -> np.ndarray:
    hole_side = random.randint(1, 3)
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    patch = np.full((top + hole_side + bottom, left + hole_side + right), GRAY, dtype=int)
    patch[top : top + hole_side, left : left + hole_side] = BLACK
    return patch


def _arrow_group_spec(
    used_target_colors: set[int],
    force_extra: bool,
) -> dict:
    direction = random.choice(DIRECTIONS)
    shaft_len = random.randint(1, 3)
    arrow = _arrow_mask(direction, shaft_len)
    mt = random.randint(1, 2)
    mb = random.randint(1, 2)
    ml = random.randint(1, 2)
    mr = random.randint(1, 2)
    patch = np.full((mt + arrow.shape[0] + mb, ml + arrow.shape[1] + mr), GRAY, dtype=int)
    patch[mt : mt + arrow.shape[0], ml : ml + arrow.shape[1]][arrow] = BLACK

    axis_offset = mt + arrow.shape[0] // 2 if direction in ("left", "right") else ml + arrow.shape[1] // 2
    if force_extra:
        object_count = random.randint(shaft_len + 1, shaft_len + 2)
    else:
        object_count = random.randint(shaft_len, shaft_len + 2)
    distances = []
    cur = random.randint(2, 4)
    for _ in range(object_count):
        distances.append(cur)
        cur += random.randint(2, 4)

    colors = random.sample(OBJECT_COLORS, k=min(object_count, len(OBJECT_COLORS)))
    if len(colors) < object_count:
        while len(colors) < object_count:
            colors.append(random.choice(OBJECT_COLORS))
    target_options = [c for c in OBJECT_COLORS if c not in used_target_colors]
    if target_options:
        colors[shaft_len - 1] = random.choice(target_options)

    return {
        "direction": direction,
        "shaft_len": shaft_len,
        "patch": patch,
        "axis_offset": axis_offset,
        "distances": distances,
        "colors": colors,
    }


def _ray_cells(shape: tuple[int, int], top: int, left: int, patch: np.ndarray, direction: str, axis_offset: int) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch.shape
    if direction == "right":
        axis_r = top + axis_offset
        return [(axis_r, c) for c in range(left + pw, w)]
    if direction == "left":
        axis_r = top + axis_offset
        return [(axis_r, c) for c in range(left - 1, -1, -1)]
    if direction == "down":
        axis_c = left + axis_offset
        return [(r, axis_c) for r in range(top + ph, h)]
    axis_c = left + axis_offset
    return [(r, axis_c) for r in range(top - 1, -1, -1)]


def _arrow_object_positions(top: int, left: int, patch: np.ndarray, direction: str, axis_offset: int, distances: list[int]) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    positions = []
    if direction == "right":
        axis_r = top + axis_offset
        boundary = left + pw - 1
        positions = [(axis_r, boundary + d) for d in distances]
    elif direction == "left":
        axis_r = top + axis_offset
        boundary = left
        positions = [(axis_r, boundary - d) for d in distances]
    elif direction == "down":
        axis_c = left + axis_offset
        boundary = top + ph - 1
        positions = [(boundary + d, axis_c) for d in distances]
    else:
        axis_c = left + axis_offset
        boundary = top
        positions = [(boundary - d, axis_c) for d in distances]
    return positions


def _place_arrow_group(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    ray_forbidden: np.ndarray,
    used_target_colors: set[int],
    force_extra: bool,
) -> bool:
    h, w = grid.shape
    spec = _arrow_group_spec(used_target_colors, force_extra)
    patch = spec["patch"]
    ph, pw = patch.shape
    direction = spec["direction"]
    axis_offset = spec["axis_offset"]
    distances = spec["distances"]
    colors = spec["colors"]
    farthest = max(distances)

    top_min, top_max = 0, h - ph
    left_min, left_max = 0, w - pw
    if direction == "right":
        left_max = w - pw - farthest
    elif direction == "left":
        left_min = farthest
    elif direction == "down":
        top_max = h - ph - farthest
    else:
        top_min = farthest
    if top_min > top_max or left_min > left_max:
        return False

    for _ in range(120):
        top = random.randint(top_min, top_max)
        left = random.randint(left_min, left_max)
        block_cells = _patch_cells(patch, top, left)
        obj_positions = _arrow_object_positions(top, left, patch, direction, axis_offset, distances)
        all_new = block_cells + obj_positions

        if any((not _in_bounds(grid.shape, r, c)) or blocked[r, c] or ray_forbidden[r, c] for r, c in all_new):
            continue
        line = _ray_cells(grid.shape, top, left, patch, direction, axis_offset)
        obj_set = set(obj_positions)
        if any(occupied[r, c] or ray_forbidden[r, c] for r, c in line if (r, c) not in obj_set):
            continue

        paste(grid, patch, top, left, transparent=BLACK)
        for (r, c), color in zip(obj_positions, colors):
            grid[r, c] = color
        for r, c in block_cells:
            occupied[r, c] = True
        for r, c in obj_positions:
            occupied[r, c] = True
        _mark_blocked(blocked, all_new)
        for r, c in line:
            ray_forbidden[r, c] = True
        used_target_colors.add(colors[spec["shaft_len"] - 1])
        return True
    return False


def _place_square_block(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    ray_forbidden: np.ndarray,
) -> bool:
    patch = _square_block_patch()
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False

    for _ in range(120):
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        cells = _patch_cells(patch, top, left)
        if any(blocked[r, c] or ray_forbidden[r, c] for r, c in cells):
            continue
        paste(grid, patch, top, left, transparent=BLACK)
        for r, c in cells:
            occupied[r, c] = True
        _mark_blocked(blocked, cells)
        return True
    return False


def _small_object_patch() -> np.ndarray:
    kind = random.choice(["single", "h2", "v2", "l3"])
    if kind == "single":
        return np.array([[1]], dtype=int)
    if kind == "h2":
        return np.array([[1, 1]], dtype=int)
    if kind == "v2":
        return np.array([[1], [1]], dtype=int)
    return np.array([[1, 0], [1, 1]], dtype=int)


def _place_extra_objects(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    ray_forbidden: np.ndarray,
) -> None:
    extra_count = random.randint(0, 3)
    h, w = grid.shape
    for _ in range(extra_count):
        patch = _small_object_patch()
        ph, pw = patch.shape
        color = random.choice(OBJECT_COLORS)
        placed = False
        for _ in range(80):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(patch)]
            if any(blocked[r, c] or ray_forbidden[r, c] for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = color
                occupied[r, c] = True
            _mark_blocked(blocked, cells)
            placed = True
            break
        if not placed:
            continue


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BLACK, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        blocked = np.zeros((h, w), dtype=bool)
        ray_forbidden = np.zeros((h, w), dtype=bool)
        used_target_colors: set[int] = set()

        arrow_count = random.randint(1, 3)
        square_count = random.randint(1, 2)
        need_extra = True
        ok = True

        order = ["arrow"] * arrow_count + ["square"] * square_count
        random.shuffle(order)
        for kind in order:
            if kind == "arrow":
                if not _place_arrow_group(
                    grid,
                    occupied,
                    blocked,
                    ray_forbidden,
                    used_target_colors,
                    force_extra=need_extra,
                ):
                    ok = False
                    break
                if need_extra:
                    need_extra = False
            else:
                if not _place_square_block(grid, occupied, blocked, ray_forbidden):
                    ok = False
                    break
        if not ok:
            continue

        _place_extra_objects(grid, occupied, blocked, ray_forbidden)

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
