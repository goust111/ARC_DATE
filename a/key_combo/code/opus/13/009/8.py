from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, paste


BG = 5
BLACK = 0
CTRL_COLORS = (1, 2)
TILE_COLORS = (3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
TRANSFORMS = [(k, flip) for k in range(4) for flip in (False, True)]
GEN_MAX_ATTEMPTS = 400


def _apply_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -k)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_control_sides(grid: np.ndarray) -> dict[str, int]:
    sides = {
        "top": grid[0, 1:-1],
        "bottom": grid[-1, 1:-1],
        "left": grid[1:-1, 0],
        "right": grid[1:-1, -1],
    }
    controls: dict[str, int] = {}
    for side, vals in sides.items():
        unique = {int(v) for v in vals}
        if len(unique) == 1:
            color = unique.pop()
            if color in CTRL_COLORS:
                controls[side] = color
    if len(controls) != 2:
        raise ValueError("expected exactly two control sides")
    return controls


def _extract_object(grid: np.ndarray, control_colors: set[int]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = np.isin(grid, list(control_colors))
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    bbox = _bbox_from_mask(mask)
    return bbox, crop_bbox(grid, bbox)


def _side_arrays(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }


def _guide_sides(arr: np.ndarray) -> dict[str, int]:
    guides: dict[str, int] = {}
    for side, vals in _side_arrays(arr).items():
        vals = [int(v) for v in vals]
        non_bg = [v for v in vals if v != BG]
        if not non_bg:
            continue
        if len(set(non_bg)) != 1:
            continue
        color = non_bg[0]
        if color not in CTRL_COLORS:
            continue
        if len(non_bg) >= len(vals) - 1:
            guides[side] = color
    return guides


def _crop_to_tile(aligned_obj: np.ndarray, controls: dict[str, int]) -> np.ndarray:
    r0 = 1 if "top" in controls else 0
    r1 = aligned_obj.shape[0] - (1 if "bottom" in controls else 0)
    c0 = 1 if "left" in controls else 0
    c1 = aligned_obj.shape[1] - (1 if "right" in controls else 0)
    return aligned_obj[r0:r1, c0:c1].copy()


def _core_anchor_from_bbox(obj_bbox: tuple[int, int, int, int], controls: dict[str, int]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = obj_bbox
    return int(r0 + ("top" in controls)), int(c0 + ("left" in controls))


def _periodic_inner(shape: tuple[int, int], tile: np.ndarray, anchor_r: int, anchor_c: int) -> np.ndarray:
    h, w = shape
    rows = (np.arange(1, h - 1) - int(anchor_r)) % tile.shape[0]
    cols = (np.arange(1, w - 1) - int(anchor_c)) % tile.shape[1]
    inner = tile[np.ix_(rows, cols)]
    out = np.full(shape, BG, dtype=int)
    out[1:-1, 1:-1] = inner
    return out


def _visible_conflicts(
    grid: np.ndarray,
    tile: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    obj_bbox: tuple[int, int, int, int],
) -> int:
    pred = _periodic_inner(grid.shape, tile, anchor_r, anchor_c)
    r0, c0, r1, c1 = obj_bbox
    mask = (grid != BG) & (grid != BLACK)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return int(np.count_nonzero(pred[mask] != grid[mask]))


def _solve_components(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    controls = _detect_control_sides(grid)
    obj_bbox, obj = _extract_object(grid, set(controls.values()))
    anchor_r, anchor_c = _core_anchor_from_bbox(obj_bbox, controls)

    candidates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for k, flip in TRANSFORMS:
        aligned = _apply_transform(obj, k, flip)
        if _guide_sides(aligned) != controls:
            continue
        tile = _crop_to_tile(aligned, controls)
        conflicts = _visible_conflicts(grid, tile, anchor_r, anchor_c, obj_bbox)
        candidates.append((conflicts, k, aligned, tile))

    if not candidates:
        raise ValueError("no aligned orientation matches the control frame")

    candidates.sort(key=lambda item: item[0])
    best_conflicts, _k, _aligned, best_tile = candidates[0]
    if best_conflicts != 0:
        raise ValueError("visible periodic cells do not support any orientation")
    return best_tile, (anchor_r, anchor_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the guided central object to the border-indicated orientation, crop it into a tile, and restore the full periodic interior.

    Input structure:
    - The grid has a grey background border ring, with one solid colored control strip on one horizontal edge and one on one vertical edge.
    - A single interior square object contains a patterned core plus two 1-pixel guide strips whose colors match the control strips.
    - Outside that object, the non-border interior is a translational tiling of the correctly oriented core, except some cells are replaced by black.

    Rule:
    - Read the two border control sides and find the interior object marked by the same two guide colors.
    - Enumerate dihedral transforms of that object until the guide-strip colors face the same sides as the matching border controls, then crop those guide strips away.
    - Use the cropped core as the periodic tile, anchored at the central core position implied by the aligned object, and fill the whole non-border interior with that tiling.

    Key insight:
    - The colored guide strips encode the tile's orientation, and only the correctly oriented crop is consistent with the visible periodic pixels surrounding the center.
    """
    g = np.asarray(grid, dtype=int)
    tile, (anchor_r, anchor_c) = _solve_components(g)
    out = g.copy()
    out[1:-1, 1:-1] = _periodic_inner(g.shape, tile, anchor_r, anchor_c)[1:-1, 1:-1]
    return out


def _random_tile(size: int) -> np.ndarray | None:
    palette = random.sample(TILE_COLORS, random.randint(3, min(5, len(TILE_COLORS))))
    base = palette[0]
    tile = np.full((size, size), base, dtype=int)

    for color in palette[1:]:
        kind = random.choice(("line", "rect", "dot", "diag"))
        if kind == "line":
            if random.random() < 0.5:
                row = random.randrange(size)
                c0 = random.randrange(size)
                c1 = random.randrange(c0, size)
                tile[row, c0 : c1 + 1] = color
            else:
                col = random.randrange(size)
                r0 = random.randrange(size)
                r1 = random.randrange(r0, size)
                tile[r0 : r1 + 1, col] = color
        elif kind == "rect":
            h = random.randint(1, min(3, size))
            w = random.randint(1, min(3, size))
            r0 = random.randrange(size - h + 1)
            c0 = random.randrange(size - w + 1)
            tile[r0 : r0 + h, c0 : c0 + w] = color
        elif kind == "diag":
            offset = random.randrange(size)
            for r in range(size):
                c = (r + offset) % size
                tile[r, c] = color
        else:
            for _ in range(random.randint(1, size)):
                tile[random.randrange(size), random.randrange(size)] = color

    extras = random.randint(size, size * 2)
    for _ in range(extras):
        tile[random.randrange(size), random.randrange(size)] = random.choice(palette)

    if len(np.unique(tile)) < 3:
        return None
    if any(np.array_equal(tile, _apply_transform(tile, k, flip)) for k, flip in TRANSFORMS[1:]):
        return None
    for period in range(1, size):
        if np.array_equal(tile, np.roll(tile, period, axis=0)):
            return None
        if np.array_equal(tile, np.roll(tile, period, axis=1)):
            return None
    return tile


def _attach_guides(tile: np.ndarray, controls: dict[str, int]) -> np.ndarray:
    size = tile.shape[0]
    out = np.full((size + 1, size + 1), BG, dtype=int)
    r0 = 1 if "top" in controls else 0
    c0 = 1 if "left" in controls else 0
    paste(out, tile, r0, c0)
    if "top" in controls:
        out[0, c0 : c0 + size] = controls["top"]
    if "bottom" in controls:
        out[-1, c0 : c0 + size] = controls["bottom"]
    if "left" in controls:
        out[r0 : r0 + size, 0] = controls["left"]
    if "right" in controls:
        out[r0 : r0 + size, -1] = controls["right"]
    return out


def _paint_controls(grid: np.ndarray, controls: dict[str, int]) -> None:
    if "top" in controls:
        grid[0, :] = controls["top"]
    if "bottom" in controls:
        grid[-1, :] = controls["bottom"]
    if "left" in controls:
        grid[:, 0] = controls["left"]
    if "right" in controls:
        grid[:, -1] = controls["right"]


def _choose_controls() -> dict[str, int]:
    horizontal = random.choice(("top", "bottom"))
    vertical = random.choice(("left", "right"))
    colors = list(CTRL_COLORS)
    random.shuffle(colors)
    return {horizontal: colors[0], vertical: colors[1]}


def _centered_location(height: int, width: int, obj_size: int) -> tuple[int, int]:
    r_min, r_max = 2, height - obj_size - 2
    c_min, c_max = 2, width - obj_size - 2
    if r_min > r_max or c_min > c_max:
        raise ValueError("object does not fit")
    r = max(r_min, min(r_max, (height - obj_size) // 2 + random.randint(-2, 2)))
    c = max(c_min, min(c_max, (width - obj_size) // 2 + random.randint(-2, 2)))
    return int(r), int(c)


def _residue_groups(
    shape: tuple[int, int],
    tile_size: int,
    anchor_r: int,
    anchor_c: int,
    obj_bbox: tuple[int, int, int, int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    r0, c0, r1, c1 = obj_bbox
    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            residue = ((r - anchor_r) % tile_size, (c - anchor_c) % tile_size)
            groups.setdefault(residue, []).append((r, c))
    return groups


def _add_occlusions(inp: np.ndarray, tile_size: int, anchor_r: int, anchor_c: int, obj_bbox: tuple[int, int, int, int]) -> bool:
    groups = _residue_groups(inp.shape, tile_size, anchor_r, anchor_c, obj_bbox)
    quotas = {res: len(cells) - 1 for res, cells in groups.items() if len(cells) >= 2}
    if sum(quotas.values()) < 10:
        return False

    target = random.randint(10, 15)
    chosen: list[tuple[int, int]] = []
    used = {res: 0 for res in quotas}
    pool = [(res, cell) for res, cells in groups.items() for cell in cells if res in quotas]
    random.shuffle(pool)
    for res, cell in pool:
        if len(chosen) >= target:
            break
        if used[res] >= quotas[res]:
            continue
        chosen.append(cell)
        used[res] += 1

    if len(chosen) < target:
        return False

    for r, c in chosen:
        inp[r, c] = BLACK
    return True


def _candidate_scores(
    inp: np.ndarray,
    obj_bbox: tuple[int, int, int, int],
    controls: dict[str, int],
) -> list[int]:
    anchor_r, anchor_c = _core_anchor_from_bbox(obj_bbox, controls)
    obj = crop_bbox(inp, obj_bbox)
    scores = []
    for k, flip in TRANSFORMS:
        aligned = _apply_transform(obj, k, flip)
        if _guide_sides(aligned) != controls:
            continue
        tile = _crop_to_tile(aligned, controls)
        scores.append(_visible_conflicts(inp, tile, anchor_r, anchor_c, obj_bbox))
    return scores


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(15, 22)
    width = random.randint(15, 22)
    controls = _choose_controls()
    tile_size = random.randint(3, 6)
    tile = _random_tile(tile_size)
    if tile is None:
        return None

    aligned_obj = _attach_guides(tile, controls)
    transform = random.choice(TRANSFORMS)
    input_obj = _inverse_transform(aligned_obj, *transform)
    obj_size = input_obj.shape[0]
    obj_r, obj_c = _centered_location(height, width, obj_size)
    obj_bbox = (obj_r, obj_c, obj_r + obj_size - 1, obj_c + obj_size - 1)
    anchor_r, anchor_c = _core_anchor_from_bbox(obj_bbox, controls)

    out = _periodic_inner((height, width), tile, anchor_r, anchor_c)
    _paint_controls(out, controls)

    inp = out.copy()
    paste(inp, input_obj, obj_r, obj_c)
    if not _add_occlusions(inp, tile_size, anchor_r, anchor_c, obj_bbox):
        return None

    scores = _candidate_scores(inp, obj_bbox, controls)
    if not scores or min(scores) != 0 or scores.count(0) != 1:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
