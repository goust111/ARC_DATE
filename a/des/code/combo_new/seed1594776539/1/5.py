from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DIR4: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300
PAIR_COUNT_CHOICES = (1, 2, 3)
COLORS = tuple(range(1, 10))
HOLE_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
)


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    bg = ~mask
    reachable = np.zeros((h, w), dtype=bool)
    stack: List[Tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                stack.append((nr, nc))
    return bg & (~reachable)


def _mask_key(mask: np.ndarray) -> Tuple[Tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _describe_object(grid: np.ndarray, coords: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    mask, bbox = _mask_from_coords(coords)
    r0, c0, _r1, _c1 = bbox
    colors = {int(grid[r, c]) for r, c in coords}
    assert 0 not in colors and len(colors) == 1, "each object must be monochrome and non-black"
    holes = _hole_mask(mask)
    return {
        "bbox": bbox,
        "color": int(next(iter(colors))),
        "mask": mask,
        "holes": holes,
        "filled_key": _mask_key(mask | holes),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-silhouette objects of different colors and fill each hollow object's holes with its solid partner's color.

    Input structure:
    - The background is black.
    - The foreground consists of separated monochrome 4-connected objects.
    - Each silhouette appears exactly twice in different colors; within a pair,
      one object is solid while the other contains one or more enclosed black holes.

    Rule:
    - Extract every non-black 4-connected object and detect any black regions that
      are fully enclosed inside its bounding box.
    - Use the object's hole-filled silhouette as its normalized shape key.
    - The two differently colored objects with the same silhouette form a pair.
    - In each pair, copy the solid object's color into every enclosed hole cell of
      the hollow object. Leave all non-hole pixels unchanged.

    Key insight:
    Colors do not determine the matching; the real match is the shared outer
    silhouette, and only the hollow member changes by receiving the solid
    member's color inside its enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != 0)
    assert comps, "expected at least one foreground object"

    groups: Dict[Tuple[Tuple[int, int], bytes], List[Dict[str, object]]] = {}
    for coords in comps:
        obj = _describe_object(g, coords)
        groups.setdefault(obj["filled_key"], []).append(obj)

    out = g.copy()
    for objs in groups.values():
        assert len(objs) == 2, "every silhouette should appear in exactly one pair"
        hollow = [obj for obj in objs if bool(np.any(obj["holes"]))]
        solid = [obj for obj in objs if not bool(np.any(obj["holes"]))]
        assert len(hollow) == 1 and len(solid) == 1, "each pair must contain one hollow and one solid object"

        hollow_obj = hollow[0]
        solid_color = int(solid[0]["color"])
        r0, c0, _r1, _c1 = hollow_obj["bbox"]
        holes = hollow_obj["holes"]
        for hr, hc in np.argwhere(holes):
            out[r0 + int(hr), c0 + int(hc)] = solid_color

    return out


def _shape_with_holes(
    rng: random.Random,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    for _ in range(80):
        core_h = rng.randint(5, 8)
        core_w = rng.randint(5, 8)
        temp = np.zeros((core_h + 4, core_w + 4), dtype=bool)
        off = 2
        temp[off : off + core_h, off : off + core_w] = True

        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        for side in sides[: rng.randint(0, 3)]:
            depth = rng.randint(1, 2)
            if side in ("top", "bottom"):
                span = rng.randint(2, core_w)
                start = rng.randint(0, core_w - span)
                c0 = off + start
                c1 = c0 + span
                if side == "top":
                    temp[off - depth : off, c0:c1] = True
                else:
                    temp[off + core_h : off + core_h + depth, c0:c1] = True
            else:
                span = rng.randint(2, core_h)
                start = rng.randint(0, core_h - span)
                r0 = off + start
                r1 = r0 + span
                if side == "left":
                    temp[r0:r1, off - depth : off] = True
                else:
                    temp[r0:r1, off + core_w : off + core_w + depth] = True

        interior_area = (core_h - 2) * (core_w - 2)
        max_holes = min(3, max(1, interior_area // 6))
        hole_count = rng.choice(tuple(range(1, max_holes + 1)))
        hole_mask = np.zeros_like(temp, dtype=bool)
        hole_cells: List[Tuple[int, int]] = []

        ok = True
        for _hole_idx in range(hole_count):
            candidates = [shape for shape in HOLE_SHAPES if shape.shape[0] <= core_h - 2 and shape.shape[1] <= core_w - 2]
            rng.shuffle(candidates)
            placed = False
            for shape in candidates:
                sh, sw = shape.shape
                for _ in range(60):
                    rr = rng.randint(off + 1, off + core_h - sh - 1)
                    cc = rng.randint(off + 1, off + core_w - sw - 1)
                    cells = [(rr + dr, cc + dc) for dr, dc in np.argwhere(shape)]
                    if any(
                        max(abs(r - er), abs(c - ec)) <= 1
                        for r, c in cells
                        for er, ec in hole_cells
                    ):
                        continue
                    for r, c in cells:
                        hole_mask[r, c] = True
                    hole_cells.extend(cells)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        hollow = temp & (~hole_mask)
        if not _is_connected(hollow):
            continue

        rows, cols = np.where(temp)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        solid_crop = temp[r0 : r1 + 1, c0 : c1 + 1]
        holes_crop = hole_mask[r0 : r1 + 1, c0 : c1 + 1]
        detected = _hole_mask(solid_crop & (~holes_crop))
        if not np.array_equal(detected, holes_crop):
            continue
        if len(_components(holes_crop)) != hole_count:
            continue
        if max(solid_crop.shape) > 10:
            continue
        return solid_crop, holes_crop
    return None


def _place_patch(
    rng: random.Random,
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    reserved: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> bool:
    h, w = patch_in.shape
    gh, gw = grid_in.shape
    if h + 2 > gh or w + 2 > gw:
        return False
    for _ in range(240):
        r0 = rng.randint(1, gh - h - 1)
        c0 = rng.randint(1, gw - w - 1)
        if np.any(reserved[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1]):
            continue
        mask_in = patch_in != 0
        mask_out = patch_out != 0
        grid_in[r0 : r0 + h, c0 : c0 + w][mask_in] = patch_in[mask_in]
        grid_out[r0 : r0 + h, c0 : c0 + w][mask_out] = patch_out[mask_out]
        reserved[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    pair_specs = []
    used_keys = set()

    for _ in range(pair_count):
        shape = _shape_with_holes(rng)
        if shape is None:
            return None
        solid_mask, holes_mask = shape
        key = _mask_key(solid_mask)
        if key in used_keys:
            return None
        used_keys.add(key)
        pair_specs.append((solid_mask, holes_mask))

    colors = rng.sample(COLORS, 2 * pair_count)
    objects = []
    for idx, (solid_mask, holes_mask) in enumerate(pair_specs):
        h, w = solid_mask.shape
        solid_color = colors[2 * idx]
        hollow_color = colors[2 * idx + 1]

        solid_patch = np.where(solid_mask, solid_color, 0).astype(int)
        hollow_pixels = solid_mask & (~holes_mask)
        hollow_in = np.where(hollow_pixels, hollow_color, 0).astype(int)
        hollow_out = hollow_in.copy()
        hollow_out[holes_mask] = solid_color

        solid_obj = {"in": solid_patch, "out": solid_patch, "size": (h, w)}
        hollow_obj = {"in": hollow_in, "out": hollow_out, "size": (h, w)}
        pair_objs = [solid_obj, hollow_obj]
        rng.shuffle(pair_objs)
        objects.extend(pair_objs)

    total_area = sum((obj["size"][0] + 2) * (obj["size"][1] + 2) for obj in objects)
    max_h = max(obj["size"][0] for obj in objects)
    max_w = max(obj["size"][1] for obj in objects)
    base_h = max(12, max_h + 2, int(np.ceil(np.sqrt(total_area * 0.8))))
    base_w = max(12, max_w + 2, int(np.ceil(np.sqrt(total_area * 1.2))))
    if base_h > 30 or base_w > 30:
        return None
    grid_h = rng.randint(base_h, min(30, base_h + 8))
    grid_w = rng.randint(base_w, min(30, base_w + 8))
    if grid_h > 30 or grid_w > 30:
        return None

    grid_in = np.zeros((grid_h, grid_w), dtype=int)
    grid_out = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    rng.shuffle(objects)
    for obj in objects:
        if not _place_patch(rng, grid_in, grid_out, reserved, obj["in"], obj["out"]):
            return None

    try:
        actual = main(grid_in)
    except Exception:
        return None
    if not np.array_equal(actual, grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
