from __future__ import annotations

import random

import numpy as np


GREY = 5
COLOR_VALUES = tuple(range(10))
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_plus(cells: list[tuple[int, int]]) -> bool:
    pts = {(int(r), int(c)) for r, c in cells}
    if len(pts) != 5:
        return False
    for r, c in pts:
        if {
            (r, c),
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        } == pts:
            return True
    return False


def _paint_mask(out: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    region = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    region[mask] = color


def _extract_shapes(grid: np.ndarray, bg: int) -> list[dict]:
    colors = sorted(set(map(int, grid.flatten())) - {bg})
    shapes: list[dict] = []
    for color in colors:
        comps = _components_mask(grid == color)
        if not comps:
            continue
        largest = max(comps, key=len)
        r0, c0, r1, c1 = _bbox(largest)
        if len(largest) < 8 or (r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 4:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch == color
        holes = []
        for comp in _components_mask(~mask):
            if any(
                rr == 0 or cc == 0 or rr == mask.shape[0] - 1 or cc == mask.shape[1] - 1
                for rr, cc in comp
            ):
                continue
            markers: dict[int, list[tuple[int, int]]] = {}
            for rr, cc in comp:
                value = int(patch[rr, cc])
                if value != bg and value != color:
                    markers.setdefault(value, []).append((int(rr), int(cc)))
            holes.append({"cells": comp, "markers": markers})

        shapes.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "holes": holes,
            }
        )
    return shapes


def _stack_order(shapes: list[dict], base_color: int) -> list[int]:
    shape_by_color = {int(shape["color"]): shape for shape in shapes}
    moving_colors = [color for color in shape_by_color if color != base_color]
    above: dict[int, set[int]] = {color: set() for color in moving_colors}
    indegree: dict[int, int] = {color: 0 for color in moving_colors}

    for shape in shapes:
        upper = int(shape["color"])
        if upper == base_color:
            continue
        for hole in shape["holes"]:
            for marker_color, cells in hole["markers"].items():
                marker_color = int(marker_color)
                if marker_color == upper or marker_color == base_color:
                    continue
                if marker_color not in above or _is_plus(cells):
                    continue
                if upper not in above[marker_color]:
                    above[marker_color].add(upper)
                    indegree[upper] += 1

    ready = [color for color in moving_colors if indegree[color] == 0]
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "marker relations must determine a unique stack order"
        color = ready.pop()
        order.append(color)
        newly_ready = []
        for nxt in sorted(above[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                newly_ready.append(nxt)
        ready.extend(newly_ready)

    assert len(order) == len(moving_colors), "stack relation graph must cover every non-base shape"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack scattered closed shapes onto the marked base using hole-contained color markers.

    Input structure:
    - Background cells surround several disconnected colored closed shapes.
    - Exactly one shape contains a small 5-cell plus marker inside one of its holes; that
      shape is the base and stays in place.
    - Other shapes may contain single colored pixels inside their holes. If shape A contains
      shape B's color, then A must be stacked above B.

    Rule:
    - Reconstruct the bottom-to-top stack order among all non-base shapes from the hole markers.
    - Erase every non-base shape from its original position.
    - Keep the base at its original location, removing the plus marker.
    - Repaint each non-base shape, from bottom to top, so its bounding-box center aligns with
      the base bounding-box center. Later shapes overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    shapes = _extract_shapes(g, bg)
    assert len(shapes) >= 3, "expected at least one base and two movable shapes"

    base_candidates = []
    for shape in shapes:
        plus_colors = []
        for hole in shape["holes"]:
            for marker_color, cells in hole["markers"].items():
                if _is_plus(cells):
                    plus_colors.append(int(marker_color))
        if plus_colors:
            shape["cross_color"] = plus_colors[0]
            base_candidates.append(shape)

    assert len(base_candidates) == 1, "expected exactly one base shape with a plus marker"
    base = base_candidates[0]
    order = _stack_order(shapes, int(base["color"]))

    out = np.full_like(g, bg)
    br0, bc0, br1, bc1 = base["bbox"]
    _paint_mask(out, base["mask"], int(base["color"]), int(br0), int(bc0))

    base_center_r2 = int(br0 + br1)
    base_center_c2 = int(bc0 + bc1)
    for color in order:
        shape = next(shape for shape in shapes if int(shape["color"]) == color)
        mask = shape["mask"]
        top = (base_center_r2 - (mask.shape[0] - 1)) // 2
        left = (base_center_c2 - (mask.shape[1] - 1)) // 2
        assert 0 <= top and top + mask.shape[0] <= g.shape[0]
        assert 0 <= left and left + mask.shape[1] <= g.shape[1]
        _paint_mask(out, mask, color, int(top), int(left))

    return out


GEN_ATTEMPTS = 400
GRID_MAX = 30
ODD_SIZES = (5, 7, 9, 11)
EMPTY = "empty"
MARKER = "marker"
CROSS = "cross"


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _candidate_holes(h: int, w: int) -> list[tuple[int, int, int, int]]:
    starts = [
        (1, 1),
        (1, w - 4),
        (h - 4, 1),
        (h - 4, w - 4),
        ((h - 3) // 2, (w - 3) // 2),
        (1, (w - 3) // 2),
        (h - 4, (w - 3) // 2),
        ((h - 3) // 2, 1),
        ((h - 3) // 2, w - 4),
    ]
    rects = []
    seen = set()
    for r0, c0 in starts:
        rect = (int(r0), int(c0), int(r0 + 2), int(c0 + 2))
        if rect[0] < 1 or rect[1] < 1 or rect[2] >= h - 1 or rect[3] >= w - 1:
            continue
        if rect in seen:
            continue
        seen.add(rect)
        rects.append(rect)
    return rects


def _hole_capacity(h: int, w: int) -> int:
    rects = _candidate_holes(h, w)
    best = 0
    for _ in range(60):
        random.shuffle(rects)
        chosen: list[tuple[int, int, int, int]] = []
        for rect in rects:
            if any(_rects_too_close(rect, other) for other in chosen):
                continue
            chosen.append(rect)
        best = max(best, len(chosen))
    return best


def _choose_holes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    if count == 0:
        return []
    rects = _candidate_holes(h, w)
    for _ in range(120):
        random.shuffle(rects)
        chosen: list[tuple[int, int, int, int]] = []
        for rect in rects:
            if any(_rects_too_close(rect, other) for other in chosen):
                continue
            chosen.append(rect)
            if len(chosen) == count:
                return chosen
    return None


def _sample_dims(max_h: int, max_w: int, hole_count: int) -> tuple[int, int] | None:
    options = []
    for h in ODD_SIZES:
        for w in ODD_SIZES:
            if h > max_h or w > max_w:
                continue
            if _hole_capacity(h, w) < hole_count:
                continue
            options.append((h, w))
    return random.choice(options) if options else None


def _render_shape(
    color: int,
    bg: int,
    h: int,
    w: int,
    payloads: list[tuple[str, int | None]],
) -> np.ndarray | None:
    hole_rects = _choose_holes(h, w, len(payloads))
    if hole_rects is None:
        return None

    patch = np.full((h, w), color, dtype=int)
    random.shuffle(payloads)
    for rect, payload in zip(hole_rects, payloads, strict=False):
        r0, c0, r1, c1 = rect
        patch[r0 : r1 + 1, c0 : c1 + 1] = bg
        kind, value = payload
        if kind == EMPTY:
            continue
        if kind == MARKER:
            rr = random.randint(r0, r1)
            cc = random.randint(c0, c1)
            patch[rr, cc] = int(value)
            continue
        assert kind == CROSS
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        for rr, cc2 in [
            (cr, cc),
            (cr - 1, cc),
            (cr + 1, cc),
            (cr, cc - 1),
            (cr, cc + 1),
        ]:
            patch[rr, cc2] = int(value)
    return patch


def _build_specs(bg: int) -> tuple[list[dict], list[int]] | None:
    usable = [color for color in COLOR_VALUES if color not in (bg, GREY)]
    n_nonbase = random.choice([2, 3])
    picked = random.sample(usable, n_nonbase + 1)
    base_color = picked[0]
    order = picked[1:]
    random.shuffle(order)

    base_holes = random.choice([1, 2, 3])
    base_dims = _sample_dims(11, 11, base_holes)
    if base_dims is None:
        return None
    base_payloads = [(CROSS, GREY)] + [(EMPTY, None)] * (base_holes - 1)

    specs = [{"name": "base", "color": base_color, "dims": base_dims, "payloads": base_payloads}]
    for idx, color in enumerate(order):
        refs: list[int] = []
        if idx > 0:
            refs.append(order[idx - 1])
            earlier = order[: idx - 1]
            random.shuffle(earlier)
            if earlier and random.random() < 0.5:
                refs.append(earlier[0])
        extra_empty = 0 if len(refs) >= 3 else random.choice([0, 1])
        if idx == 0:
            refs = []
            extra_empty = random.choice([0, 1, 2])
        payloads = [(MARKER, ref) for ref in refs] + [(EMPTY, None)] * extra_empty
        dims = _sample_dims(base_dims[0], base_dims[1], len(payloads))
        if dims is None:
            return None
        specs.append({"name": f"shape_{idx}", "color": color, "dims": dims, "payloads": payloads})

    return specs, order


def _place_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    min_h = max(max_h + 4, 24)
    min_w = max(max_w + 4, 24)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    grid = np.full((height, width), bg, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for patch in sorted(patches, key=lambda arr: int(arr.shape[0] * arr.shape[1]), reverse=True):
        ph, pw = map(int, patch.shape)
        for _ in range(200):
            top = random.randint(1, height - ph - 1)
            left = random.randint(1, width - pw - 1)
            box = (top, left, top + ph - 1, left + pw - 1)
            if any(_rects_too_close(box, other) for other in placed_boxes):
                continue
            window = grid[top : top + ph, left : left + pw]
            mask = patch != bg
            window[mask] = patch[mask]
            placed_boxes.append(box)
            break
        else:
            return None
    return grid


def _make_example() -> np.ndarray | None:
    bg_choices = [color for color in COLOR_VALUES if color != GREY]
    bg = random.choice(bg_choices)
    spec_info = _build_specs(bg)
    if spec_info is None:
        return None
    specs, _order = spec_info

    patches = []
    for spec in specs:
        h, w = spec["dims"]
        patch = _render_shape(int(spec["color"]), bg, int(h), int(w), list(spec["payloads"]))
        if patch is None:
            return None
        patches.append(patch)

    grid = _place_patches(patches, bg)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    other_counts = [int(np.sum(grid == color)) for color in np.unique(grid) if int(color) != bg]
    if other_counts and int(np.sum(grid == bg)) <= max(other_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_example()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
