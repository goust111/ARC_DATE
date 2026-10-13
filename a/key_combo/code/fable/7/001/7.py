import itertools
import math
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
WORK_SHAPE = (30, 30)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _components_mask(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or bg_seen[r, c]:
                continue
            q = deque([(r, c)])
            bg_seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or bg_seen[nr, nc]:
                        continue
                    bg_seen[nr, nc] = True
                    touches_border |= nr in (0, h - 1) or nc in (0, w - 1)
                    q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _find_container(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for comp in _find_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if comp["area"] != 2 * (h + w) - 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == comp["color"]):
            continue
        if not np.all(sub[~border] == bg):
            continue
        candidates.append(comp)
    assert len(candidates) == 1, "expected exactly one hollow rectangular container"
    return candidates[0]


def _analyze_objects(grid: np.ndarray, container: dict, bg: int) -> tuple[list[dict], dict]:
    cr0, cc0, cr1, cc1 = container["bbox"]
    inner_h = cr1 - cr0 - 1
    inner_w = cc1 - cc0 - 1
    objects = []
    for comp in _find_components(grid, bg):
        if comp["bbox"] == container["bbox"] and comp["color"] == container["color"]:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rr = comp["coords"][:, 0] - r0
        cc = comp["coords"][:, 1] - c0
        sub[rr, cc] = True
        holes = _count_holes(sub)
        objects.append(
            {
                "color": comp["color"],
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "area": comp["area"],
                "holes": holes,
                "layer_rows": comp["area"] // inner_w,
            }
        )
    hole_counts = [obj["holes"] for obj in objects]
    assert objects, "expected at least one porous object"
    assert len(set(hole_counts)) == len(hole_counts), "hole counts must be unique"
    total_area = sum(obj["area"] for obj in objects)
    assert total_area == inner_h * inner_w, "solid areas must exactly fill the container interior"
    assert all(obj["area"] % inner_w == 0 for obj in objects), "each solid area must be a multiple of the inner width"
    return objects, {"inner_h": inner_h, "inner_w": inner_w}


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Melt the scattered porous objects into horizontal sediment layers inside the hollow container.

    Input structure:
    - The dominant background contains one one-pixel-thick hollow rectangular container.
    - Outside the container are several disconnected monochrome objects, each with a distinct color.
    - Every object encloses a unique number of background holes, and each object's solid area is a multiple of the container's inner width.

    Rule:
    - Remove every object outside the container.
    - Keep the container border unchanged and fill its interior with solid horizontal color bands.
    - Order the bands from bottom to top by increasing hole count; each band's height is that color's solid area divided by the container's inner width.

    Key insight:
    - Object shape and position are irrelevant after recognition: hole count controls vertical order, while solid area alone controls layer thickness.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    bg = _get_bg(grid)
    container = _find_container(grid, bg)
    objects, dims = _analyze_objects(grid, container, bg)
    cr0, cc0, cr1, cc1 = container["bbox"]

    out = np.full_like(grid, bg)
    out[cr0 : cr1 + 1, cc0] = container["color"]
    out[cr0 : cr1 + 1, cc1] = container["color"]
    out[cr0, cc0 : cc1 + 1] = container["color"]
    out[cr1, cc0 : cc1 + 1] = container["color"]

    fill_bottom = cr1 - 1
    for obj in sorted(objects, key=lambda item: item["holes"]):
        rows = obj["layer_rows"]
        r0 = fill_bottom - rows + 1
        out[r0 : fill_bottom + 1, cc0 + 1 : cc1] = obj["color"]
        fill_bottom = r0 - 1

    assert fill_bottom == cr0
    assert dims["inner_h"] <= 28 and dims["inner_w"] <= 28
    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _sample_config(rng: np.random.Generator) -> tuple[int, list[int], list[int]]:
    for _ in range(400):
        n_objects = int(rng.integers(2, 5))
        inner_w = int(rng.integers(5, 10))
        inner_h = int(rng.integers(max(5, n_objects + 2), 13))
        layer_rows = _random_composition(inner_h, n_objects, rng)
        capacities = [min(5, (rows * inner_w + 1) // 9) for rows in layer_rows]
        if sum(cap > 0 for cap in capacities) < n_objects:
            continue

        valid_assignments = []
        for holes in itertools.permutations(range(1, 6), n_objects):
            if any(h > cap for h, cap in zip(holes, capacities)):
                continue
            if not any(
                (layer_rows[i] - layer_rows[j]) * (holes[i] - holes[j]) < 0
                for i in range(n_objects)
                for j in range(i + 1, n_objects)
            ):
                continue
            valid_assignments.append(list(holes))
        if not valid_assignments:
            continue
        hole_counts = valid_assignments[int(rng.integers(len(valid_assignments)))]
        return inner_w, layer_rows, hole_counts
    raise RuntimeError("failed to sample a non-degenerate layer / hole configuration")


def _build_porous_mask(hole_count: int, area: int, rng: np.random.Generator) -> np.ndarray:
    width = 4 * hole_count - 1
    base = np.zeros((3, width), dtype=bool)
    connector_row = 0 if bool(rng.integers(2)) else 2

    for idx in range(hole_count):
        c0 = 4 * idx
        base[0, c0 : c0 + 3] = True
        base[1, c0] = True
        base[1, c0 + 2] = True
        base[2, c0 : c0 + 3] = True
        if idx > 0:
            base[connector_row, c0 - 1] = True

    extra = area - int(base.sum())
    assert extra >= 0
    if extra == 0:
        mask = base
    else:
        feasible_widths = [
            attach_w
            for attach_w in range(1, min(width, extra) + 1)
            if math.ceil(extra / attach_w) <= 8
        ]
        if not feasible_widths:
            feasible_widths = list(range(1, min(width, extra) + 1))
        attach_w = int(rng.choice(feasible_widths))
        added_rows = int(math.ceil(extra / attach_w))
        start = int(rng.integers(0, width - attach_w + 1))
        fill_from_left = bool(rng.integers(2))

        if connector_row == 0:
            mask = np.zeros((3 + added_rows, width), dtype=bool)
            mask[added_rows:, :] = base
            remain = extra
            for step in range(added_rows):
                row = added_rows - 1 - step
                take = min(attach_w, remain)
                if fill_from_left:
                    mask[row, start : start + take] = True
                else:
                    mask[row, start + attach_w - take : start + attach_w] = True
                remain -= take
        else:
            mask = np.zeros((3 + added_rows, width), dtype=bool)
            mask[:3, :] = base
            remain = extra
            for step in range(added_rows):
                row = 3 + step
                take = min(attach_w, remain)
                if fill_from_left:
                    mask[row, start : start + take] = True
                else:
                    mask[row, start + attach_w - take : start + attach_w] = True
                remain -= take

    k = int(rng.integers(4))
    mask = np.rot90(mask, k=k)
    if bool(rng.integers(2)):
        mask = np.fliplr(mask)
    if bool(rng.integers(2)):
        mask = np.flipud(mask)

    assert int(mask.sum()) == area
    assert _count_holes(mask) == hole_count
    return mask


def _is_rectangular_frame_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(mask[border]) and np.all(~mask[~border]))


def _can_place(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, gap: int = 1) -> bool:
    h, w = mask.shape
    hh, ww = canvas.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(hh, top + h + gap)
    c1 = min(ww, left + w + gap)
    return not np.any(canvas[r0:r1, c0:c1] != BG)


def _place(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = canvas[top : top + h, left : left + w]
    sub[mask] = int(color)


def _generate_case(return_meta: bool = False, seed: int | None = None):
    rng = np.random.default_rng(seed)

    for _ in range(GEN_RETRIES):
        inner_w, layer_rows, hole_counts = _sample_config(rng)
        inner_h = sum(layer_rows)
        container_h = inner_h + 2
        container_w = inner_w + 2
        areas = [inner_w * rows for rows in layer_rows]
        colors = rng.choice(np.array(PALETTE, dtype=int), size=len(areas) + 1, replace=False).tolist()
        container_color = int(colors[0])
        object_colors = [int(c) for c in colors[1:]]

        objects = []
        ok = True
        for color, holes, rows, area in zip(object_colors, hole_counts, layer_rows, areas):
            mask = _build_porous_mask(int(holes), int(area), rng)
            if max(mask.shape) > 22 or _is_rectangular_frame_mask(mask):
                ok = False
                break
            objects.append(
                {
                    "color": int(color),
                    "holes": int(holes),
                    "layer_rows": int(rows),
                    "area": int(area),
                    "mask": mask,
                }
            )
        if not ok:
            continue

        canvas = np.zeros(WORK_SHAPE, dtype=int)
        cr0 = int(rng.integers(2, WORK_SHAPE[0] - container_h - 1))
        cc0 = int(rng.integers(2, WORK_SHAPE[1] - container_w - 1))
        canvas[cr0, cc0 : cc0 + container_w] = container_color
        canvas[cr0 + container_h - 1, cc0 : cc0 + container_w] = container_color
        canvas[cr0 : cr0 + container_h, cc0] = container_color
        canvas[cr0 : cr0 + container_h, cc0 + container_w - 1] = container_color

        placed = []
        for obj in rng.permutation(objects):
            mask = obj["mask"]
            h, w = mask.shape
            for _ in range(300):
                top = int(rng.integers(0, WORK_SHAPE[0] - h + 1))
                left = int(rng.integers(0, WORK_SHAPE[1] - w + 1))
                if not _can_place(canvas, mask, top, left, gap=1):
                    continue
                _place(canvas, mask, top, left, obj["color"])
                placed.append({**obj, "top": top, "left": left, "bbox": (top, left, top + h - 1, left + w - 1)})
                break
            else:
                ok = False
                break
        if not ok:
            continue

        coords = np.argwhere(canvas != BG)
        r0, c0, r1, c1 = _bbox_coords(coords)
        top_margin = int(rng.integers(0, min(3, r0) + 1))
        left_margin = int(rng.integers(0, min(3, c0) + 1))
        bottom_margin = int(rng.integers(0, min(3, WORK_SHAPE[0] - 1 - r1) + 1))
        right_margin = int(rng.integers(0, min(3, WORK_SHAPE[1] - 1 - c1) + 1))
        final = canvas[r0 - top_margin : r1 + bottom_margin + 1, c0 - left_margin : c1 + right_margin + 1].copy()

        try:
            out = main(final)
        except AssertionError:
            continue
        if max(final.shape) > 30 or max(out.shape) > 30:
            continue

        meta = {
            "container_color": container_color,
            "layer_rows_by_color": {obj["color"]: obj["layer_rows"] for obj in objects},
            "holes_by_color": {obj["color"]: obj["holes"] for obj in objects},
            "areas_by_color": {obj["color"]: obj["area"] for obj in objects},
            "inner_shape": (inner_h, inner_w),
            "objects": objects,
        }
        return (final, meta) if return_meta else final

    raise RuntimeError("failed to generate a valid porous-sediment task instance")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
