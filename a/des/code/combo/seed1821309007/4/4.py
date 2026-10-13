from __future__ import annotations

from collections import Counter
import random

import numpy as np


ORANGE = 7
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 8, 9)
GEN_ATTEMPTS = 400


def _components_by_color(mask: np.ndarray, colors: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(colors[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and mask[nr, nc]
                        and int(colors[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _is_rectangular_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    coords = component["coords"]
    if component["area"] != 2 * h + 2 * w - 4:
        return False
    for rr, cc in coords:
        if rr not in (r0, r1) and cc not in (c0, c1):
            return False
    return True


def _find_regions(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components_by_color(grid == color, grid)
        if comps and all(_is_rectangular_frame(comp) for comp in comps):
            total = sum(comp["area"] for comp in comps)
            candidates.append((total, color, [comp["bbox"] for comp in comps]))
    assert len(candidates) == 1, "expected exactly one frame color"
    _total, wall, bboxes = candidates[0]
    bboxes.sort()
    return wall, bboxes


def _paint_ring(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort each enclosed region's inner objects by area descending, then redraw them
    as concentric rectangular rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    wall, regions = _find_regions(g, bg)
    out = g.copy()

    for r0, c0, r1, c1 in regions:
        ir0, ic0 = r0 + 1, c0 + 1
        ir1, ic1 = r1 - 1, c1 - 1
        inner = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        comps = _components_by_color((inner != bg) & (inner != wall), inner)
        assert comps, "each region must contain at least one object"
        comps.sort(key=lambda comp: (-int(comp["area"]), *comp["bbox"]))

        max_layers = (min(ir1 - ir0 + 1, ic1 - ic0 + 1) + 1) // 2
        assert len(comps) <= max_layers, "too many objects for concentric encoding"

        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg
        for depth, comp in enumerate(comps):
            rr0 = ir0 + depth
            cc0 = ic0 + depth
            rr1 = ir1 - depth
            cc1 = ic1 - depth
            color = int(comp["color"])
            if depth == len(comps) - 1:
                out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
            else:
                _paint_ring(out, rr0, cc0, rr1, cc1, color)

    return out


def _border_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    halo = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((r + dr, c + dc))
    return halo


def _random_connected_shape(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(80):
        cells = {(0, 0)}
        frontier = [(0, 0)]
        while len(cells) < area:
            base_r, base_c = random.choice(frontier)
            opts = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (base_r + dr, base_c + dc)
                if nxt not in cells:
                    opts.append(nxt)
            if not opts:
                frontier = list(cells)
                continue
            cells.add(random.choice(opts))
            frontier = list(cells)

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > max_h or w > max_w:
            continue
        patch = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            patch[r - r0, c - c0] = True
        return patch
    return None


def _sample_object_areas(inner_h: int, inner_w: int, n_obj: int) -> list[int] | None:
    max_area = max(1, inner_h * inner_w // 3)
    candidates = list(range(1, max_area + 1))
    random.shuffle(candidates)
    areas: list[int] = []
    for area in candidates:
        if len(areas) == n_obj:
            break
        if area not in areas:
            areas.append(area)
    if len(areas) != n_obj:
        return None
    areas.sort(reverse=True)
    return areas


def _place_objects(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    colors: list[int],
    areas: list[int],
) -> bool:
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    order = list(range(len(colors)))
    random.shuffle(order)
    for idx in order:
        area = int(areas[idx])
        shape = _random_connected_shape(area, inner_h, inner_w)
        if shape is None:
            return False
        sh, sw = shape.shape

        positions = [
            (r, c)
            for r in range(inner_h - sh + 1)
            for c in range(inner_w - sw + 1)
        ]
        random.shuffle(positions)

        placed = False
        for r, c in positions:
            coords = {
                (top + r + rr, left + c + cc)
                for rr in range(sh)
                for cc in range(sw)
                if shape[rr, cc]
            }
            if coords & blocked:
                continue
            for rr, cc in coords:
                grid[rr, cc] = colors[idx]
            occupied |= coords
            blocked |= _border_coords(coords)
            placed = True
            break
        if not placed:
            return False
    return True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _non_overlapping_boxes(h: int, w: int, boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, bh, bw) in enumerate(boxes):
        a0, a1 = r0 - 1, r0 + bh
        b0, b1 = c0 - 1, c0 + bw
        if a0 < 0 or b0 < 0 or a1 >= h or b1 >= w:
            return False
        for rr0, cc0, rh, rw in boxes[i + 1 :]:
            if not (r0 + bh < rr0 - 1 or rr0 + rh < r0 - 1 or c0 + bw < cc0 - 1 or cc0 + rw < c0 - 1):
                return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), ORANGE, dtype=int)

    n_regions = random.randint(1, 3)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_regions):
        inner_h = random.randint(3, 10)
        inner_w = random.randint(3, 10)
        boxes.append((0, 0, inner_h + 2, inner_w + 2))

    placed_boxes: list[tuple[int, int, int, int]] = []
    for _bh in boxes:
        pass

    for bh_box in boxes:
        bh, bw = bh_box[2], bh_box[3]
        choices = [
            (r0, c0)
            for r0 in range(1, h - bh)
            for c0 in range(1, w - bw)
        ]
        random.shuffle(choices)
        for r0, c0 in choices:
            trial = placed_boxes + [(r0, c0, bh, bw)]
            if _non_overlapping_boxes(h, w, trial):
                placed_boxes = trial
                break
        else:
            return None

    branch_force = random.choice(("full-depth", "shallow", "mixed"))
    for region_index, (r0, c0, bh, bw) in enumerate(placed_boxes):
        _draw_frame(grid, r0, c0, bh, bw)
        inner_h = bh - 2
        inner_w = bw - 2
        max_layers = (min(inner_h, inner_w) + 1) // 2
        if branch_force == "full-depth" and region_index == 0:
            n_obj = max_layers
        elif branch_force == "shallow" and region_index == 0 and max_layers >= 2:
            n_obj = random.randint(1, max_layers - 1)
        else:
            n_obj = random.randint(1, max_layers)

        colors = random.sample(PALETTE, n_obj)
        areas = _sample_object_areas(inner_h, inner_w, n_obj)
        if areas is None:
            return None
        if not _place_objects(grid, r0 + 1, c0 + 1, inner_h, inner_w, colors, areas):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
