import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
GRID_H_RANGE = (12, 23)
GRID_W_RANGE = (12, 23)
OBJECT_COUNT_RANGE = (3, 5)
OBJECT_SIDE_RANGE = (4, 8)
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return [np.asarray(comp, dtype=int) for comp in comps]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for coords in _components(grid == color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored object by the shadow of its occupied row-column footprint, with smaller footprints on top.

    Input structure:
    - Black background with several non-overlapping monochrome colored objects.
    - Each object has a clear axis-aligned row range and column range; colors distinguish the objects.
    - Object footprint areas are intended to be different, so overlap priority is unambiguous.

    Rule:
    - For each non-black object, find its bounding row range and column range.
    - The object's shadow is the Cartesian product of that row range and column range, excluding the object's own colored cells.
    - Paint all shadows on a black canvas; when shadows overlap, the object with the smaller row-range x column-range area is visible.
    - Remove all original object cells after shadow painting, leaving black at every original object position.

    Key insight:
    - The projection is a whole row-range by column-range footprint rather than a one-direction ray.
    - Overlaps are resolved by size with the smaller object drawn above larger objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    out = np.full_like(g, BLACK)
    original = g != BLACK

    # Draw large footprints first, so smaller footprints overwrite them at overlaps.
    for obj in sorted(objects, key=lambda item: (-item["area"], item["bbox"][0], item["bbox"][1])):
        r0, c0, r1, c1 = obj["bbox"]
        shadow = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_r = obj["coords"][:, 0] - r0
        local_c = obj["coords"][:, 1] - c0
        shadow[local_r, local_c] = False

        window = out[r0 : r1 + 1, c0 : c1 + 1]
        window[shadow] = int(obj["color"])

    out[original] = BLACK
    return out


def _corner_mask(h: int, w: int, orientation: str, thick_r: int, thick_c: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if orientation in ("TL", "TR"):
        mask[:thick_r, :] = True
    else:
        mask[h - thick_r :, :] = True

    if orientation in ("TL", "BL"):
        mask[:, :thick_c] = True
    else:
        mask[:, w - thick_c :] = True
    return mask


def _make_shape(used_areas: set[int], max_h: int, max_w: int) -> tuple[np.ndarray, int] | None:
    for _ in range(100):
        h = random.randint(OBJECT_SIDE_RANGE[0], min(OBJECT_SIDE_RANGE[1], max_h))
        w = random.randint(OBJECT_SIDE_RANGE[0], min(OBJECT_SIDE_RANGE[1], max_w))
        area = h * w
        if area in used_areas:
            continue
        thick_r = random.randint(1, min(2, h - 2))
        thick_c = random.randint(1, min(2, w - 2))
        orientation = random.choice(("TL", "TR", "BL", "BR"))
        mask = _corner_mask(h, w, orientation, thick_r, thick_c)
        if 0 < int(mask.sum()) < area:
            used_areas.add(area)
            return mask, area
    return None


def _actual_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    rs, cs = np.where(mask)
    return {(int(r + r0), int(c + c0)) for r, c in zip(rs, cs)}


def _shadow_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    h, w = mask.shape
    cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    return cells - _actual_cells(mask, r0, c0)


def _has_visible_shadow_overlap(
    mask: np.ndarray,
    r0: int,
    c0: int,
    actual: set[tuple[int, int]],
    placed: list[dict],
) -> bool:
    shadow = _shadow_cells(mask, r0, c0)
    original = set(actual)
    for obj in placed:
        original |= obj["actual"]
    for obj in placed:
        if (shadow & obj["shadow"]) - original:
            return True
    return False


def _choose_position(
    mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    occupied: set[tuple[int, int]],
    placed: list[dict],
    require_overlap: bool,
) -> tuple[int, int] | None:
    h, w = mask.shape
    candidates = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        actual = _actual_cells(mask, r0, c0)
        if actual & occupied:
            continue
        if require_overlap and not _has_visible_shadow_overlap(mask, r0, c0, actual, placed):
            continue
        return r0, c0
    return None


def _visible_overlap_ok(grid: np.ndarray, out: np.ndarray, placed: list[dict]) -> bool:
    original = set()
    for obj in placed:
        original |= obj["actual"]
    for r, c in original:
        if int(out[r, c]) != BLACK:
            return False

    by_cell: dict[tuple[int, int], list[dict]] = {}
    for obj in placed:
        for cell in obj["shadow"] - original:
            by_cell.setdefault(cell, []).append(obj)

    overlap_cells = [cell for cell, objs in by_cell.items() if len(objs) >= 2]
    if not overlap_cells:
        return False

    for cell in overlap_cells:
        winner = min(by_cell[cell], key=lambda obj: (obj["area"], obj["r0"], obj["c0"]))
        if int(out[cell]) != int(winner["color"]):
            return False

    colors = {int(obj["color"]) for obj in placed}
    return all(bool(np.any(out == color)) for color in colors) and bool(np.any(grid != out))


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, n_objects)

    shapes: list[tuple[np.ndarray, int, int]] = []
    used_areas: set[int] = set()
    for color in colors:
        made = _make_shape(used_areas, min(OBJECT_SIDE_RANGE[1], grid_h - 1), min(OBJECT_SIDE_RANGE[1], grid_w - 1))
        if made is None:
            return None
        mask, area = made
        shapes.append((mask, area, int(color)))
    random.shuffle(shapes)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for i, (mask, area, color) in enumerate(shapes):
        require_overlap = i == 1 or (i > 1 and random.random() < 0.6)
        pos = _choose_position(mask, grid_h, grid_w, occupied, placed, require_overlap=require_overlap)
        if pos is None and require_overlap:
            pos = _choose_position(mask, grid_h, grid_w, occupied, placed, require_overlap=False)
        if pos is None:
            return None

        r0, c0 = pos
        actual = _actual_cells(mask, r0, c0)
        shadow = _shadow_cells(mask, r0, c0)
        for r, c in actual:
            grid[r, c] = color
        occupied |= actual
        placed.append(
            {
                "mask": mask,
                "area": int(area),
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "actual": actual,
                "shadow": shadow,
            }
        )

    try:
        out = main(grid)
    except Exception:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if _visible_overlap_ok(grid, out, placed):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-overlap input")
