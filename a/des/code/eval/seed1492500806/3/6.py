from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
FRAME_THICKNESS = 3
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_array(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _touches_boundary(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _border_band_mask(height: int, width: int, thickness: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    mask[thickness : height - thickness, thickness : width - thickness] = False
    return mask


def _infer_hole_side(hole_cells_local: list[tuple[int, int]], height: int, width: int) -> str | None:
    if not hole_cells_local:
        return None
    top = all(r < FRAME_THICKNESS - 1 for r, _ in hole_cells_local) and any(r == 0 for r, _ in hole_cells_local)
    bottom = all(r >= height - (FRAME_THICKNESS - 1) for r, _ in hole_cells_local) and any(
        r == height - 1 for r, _ in hole_cells_local
    )
    left = all(c < FRAME_THICKNESS - 1 for _, c in hole_cells_local) and any(c == 0 for _, c in hole_cells_local)
    right = all(c >= width - (FRAME_THICKNESS - 1) for _, c in hole_cells_local) and any(
        c == width - 1 for _, c in hole_cells_local
    )
    hits = [name for name, ok in (("top", top), ("bottom", bottom), ("left", left), ("right", right)) if ok]
    if len(hits) != 1:
        return None
    return hits[0]


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return row_gap + col_gap


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _direction_from_arrow(
    key_bbox: tuple[int, int, int, int], arrow_bbox: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    kr0, kc0, kr1, kc1 = key_bbox
    ar0, ac0, ar1, ac1 = arrow_bbox
    if ac1 < kc0 and _ranges_overlap(ar0, ar1, kr0, kr1):
        return 0, -1
    if ac0 > kc1 and _ranges_overlap(ar0, ar1, kr0, kr1):
        return 0, 1
    if ar1 < kr0 and _ranges_overlap(ac0, ac1, kc0, kc1):
        return -1, 0
    if ar0 > kr1 and _ranges_overlap(ac0, ac1, kc0, kc1):
        return 1, 0
    return None


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _find_safes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    safes: list[dict] = []
    for comp in _components(g != GREY):
        if _touches_boundary(comp, h, w):
            continue
        ir0, ic0, ir1, ic1 = _bbox(comp)
        interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        if len(comp) != int(interior.size):
            continue
        if np.any(interior == GREY) or np.any(interior == MAROON):
            continue
        colors = set(map(int, np.unique(interior))) - {BLACK}
        if len(colors) != 1:
            continue
        color = int(next(iter(colors)))
        or0 = ir0 - FRAME_THICKNESS
        oc0 = ic0 - FRAME_THICKNESS
        or1 = ir1 + FRAME_THICKNESS
        oc1 = ic1 + FRAME_THICKNESS
        if or0 < 0 or oc0 < 0 or or1 >= h or oc1 >= w:
            continue
        outer = g[or0 : or1 + 1, oc0 : oc1 + 1]
        band = _border_band_mask(outer.shape[0], outer.shape[1], FRAME_THICKNESS)
        band_colors = set(map(int, np.unique(outer[band])))
        if not band_colors.issubset({BLACK, GREY}):
            continue
        hole_mask = band & (outer == BLACK)
        hole_cells_local = [tuple(map(int, rc)) for rc in np.argwhere(hole_mask)]
        if not hole_cells_local:
            continue
        side = _infer_hole_side(hole_cells_local, outer.shape[0], outer.shape[1])
        if side is None:
            continue
        if len(_components(hole_mask)) != 1:
            continue
        pattern_cells = [(r, c) for r, c in comp if int(g[r, c]) == color]
        if not pattern_cells:
            continue
        hole_cells_global = [(or0 + r, oc0 + c) for r, c in hole_cells_local]
        safes.append(
            {
                "color": color,
                "interior_bbox": (ir0, ic0, ir1, ic1),
                "outer_bbox": (or0, oc0, or1, oc1),
                "hole_cells": hole_cells_global,
                "hole_shape": _normalize(hole_cells_local),
                "pattern_cells": pattern_cells,
                "side": side,
            }
        )
    assert safes, "expected at least one safe"
    return safes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the safe with the same hidden color and the same lockhole
    shape, then release the hidden pattern in the maroon-indicated direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    safes = _find_safes(g)

    relevant_colors = {safe["color"] for safe in safes}
    color_components: dict[int, list[list[tuple[int, int]]]] = {}
    for color in relevant_colors:
        color_components[color] = _components(g == color)

    arrow_components = _components(g == MAROON)
    assert len(arrow_components) >= len(safes), "expected one arrow per safe"
    used_arrows: set[int] = set()

    out = g.copy()
    for safe in safes:
        color = int(safe["color"])
        ir0, ic0, ir1, ic1 = safe["interior_bbox"]
        or0, oc0, or1, oc1 = safe["outer_bbox"]
        box_h = or1 - or0 + 1
        box_w = oc1 - oc0 + 1

        key_candidates = []
        for comp in color_components[color]:
            cr0, cc0, cr1, cc1 = _bbox(comp)
            inside_this_safe = ir0 <= cr0 and cr1 <= ir1 and ic0 <= cc0 and cc1 <= ic1
            if inside_this_safe:
                continue
            key_candidates.append(comp)
        assert len(key_candidates) == 1, f"expected exactly one external key for color {color}"
        key_cells = key_candidates[0]
        assert _normalize(key_cells) == safe["hole_shape"], "key shape must match the hole shape"

        key_bbox = _bbox(key_cells)
        arrow_options = []
        for idx, arrow_cells in enumerate(arrow_components):
            if idx in used_arrows:
                continue
            arrow_bbox = _bbox(arrow_cells)
            direction = _direction_from_arrow(key_bbox, arrow_bbox)
            if direction is None:
                continue
            arrow_options.append(((_rect_gap(key_bbox, arrow_bbox)), idx, direction, arrow_cells))
        assert arrow_options, f"expected an arrow adjacent to key color {color}"
        arrow_options.sort(key=lambda item: (item[0], item[1]))
        _, arrow_idx, direction, arrow_cells = arrow_options[0]
        used_arrows.add(int(arrow_idx))

        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BLACK
        _paint_cells(out, key_cells, BLACK)
        _paint_cells(out, arrow_cells, BLACK)
        _paint_cells(out, safe["hole_cells"], color)

        dr, dc = direction
        shift = box_h if dr != 0 else box_w
        translated = [(r + dr * shift, c + dc * shift) for r, c in safe["pattern_cells"]]
        for r, c in translated:
            assert 0 <= r < h and 0 <= c < w, "released pattern must stay in bounds"
            out[r, c] = color

    return out


HOLE_TEMPLATES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 1, 0]], dtype=bool),
]


def _shape_from_template(template: np.ndarray, side: str, outer_h: int, outer_w: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((outer_h, outer_w), dtype=bool)
    if side in ("top", "bottom"):
        th, tw = template.shape
        if tw + 4 > outer_w:
            return None
        start_c = rng.randint(2, outer_w - tw - 2)
        start_r = 0 if side == "top" else outer_h - th
        mask[start_r : start_r + th, start_c : start_c + tw] = template
    else:
        rotated = np.rot90(template)
        th, tw = rotated.shape
        if th + 4 > outer_h:
            return None
        start_r = rng.randint(2, outer_h - th - 2)
        start_c = 0 if side == "left" else outer_w - tw
        mask[start_r : start_r + th, start_c : start_c + tw] = rotated
    return mask


def _tight_shape(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_array(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    cells = [tuple(map(int, rc)) for rc in np.argwhere(cropped)]
    return cropped, cells


def _random_connected_cells(height: int, width: int, area: int, rng: random.Random) -> list[tuple[int, int]] | None:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier = {start}
    while len(cells) < area and frontier:
        base = rng.choice(tuple(frontier))
        candidates = []
        for nr, nc in ((base[0] - 1, base[1]), (base[0] + 1, base[1]), (base[0], base[1] - 1), (base[0], base[1] + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                candidates.append((nr, nc))
        if not candidates:
            frontier.remove(base)
            continue
        nxt = rng.choice(candidates)
        cells.add(nxt)
        frontier.add(nxt)
    if len(cells) != area:
        return None
    return sorted(cells)


def _random_pattern(interior_h: int, interior_w: int, rng: random.Random) -> np.ndarray | None:
    max_area = min(8, interior_h * interior_w - 1)
    min_area = min(3, max_area)
    if max_area < 2:
        return None
    for _ in range(40):
        area = rng.randint(min_area, max_area)
        cells = _random_connected_cells(interior_h, interior_w, area, rng)
        if cells is None:
            continue
        mask = np.zeros((interior_h, interior_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _bbox_in_bounds(bbox: tuple[int, int, int, int], height: int, width: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width


def _region_free(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(h - 1, r1 + pad)
    c1 = min(w - 1, c1 + pad)
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(h - 1, r1 + pad)
    c1 = min(w - 1, c1 + pad)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _safe_landing_bbox(
    top: int, left: int, outer_h: int, outer_w: int, direction: tuple[int, int]
) -> tuple[int, int, int, int]:
    ir0 = top + FRAME_THICKNESS
    ic0 = left + FRAME_THICKNESS
    ir1 = top + outer_h - FRAME_THICKNESS - 1
    ic1 = left + outer_w - FRAME_THICKNESS - 1
    dr, dc = direction
    shift = outer_h if dr != 0 else outer_w
    return ir0 + dr * shift, ic0 + dc * shift, ir1 + dr * shift, ic1 + dc * shift


def _arrow_cells_for_key(
    key_top: int,
    key_left: int,
    key_h: int,
    key_w: int,
    direction: tuple[int, int],
    arrow_len: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    if dc != 0:
        col = key_left + key_w + 1 if dc > 0 else key_left - 2
        start_r = key_top + (key_h - arrow_len) // 2
        return [(start_r + k, col) for k in range(arrow_len)]
    row = key_top + key_h + 1 if dr > 0 else key_top - 2
    start_c = key_left + (key_w - arrow_len) // 2
    return [(row, start_c + k) for k in range(arrow_len)]


def _expected_output(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        ir0, ic0, ir1, ic1 = obj["interior_bbox"]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BLACK
        _paint_cells(out, obj["key_cells"], BLACK)
        _paint_cells(out, obj["arrow_cells"], BLACK)
        _paint_cells(out, obj["hole_cells"], obj["color"])
        dr, dc = obj["direction"]
        shift = obj["outer_h"] if dr != 0 else obj["outer_w"]
        translated = [(r + dr * shift, c + dc * shift) for r, c in obj["pattern_cells"]]
        _paint_cells(out, translated, obj["color"])
    return out


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(30, 38)
    width = rng.randint(30, 38)
    safe_count = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, safe_count)

    directions = DIRS[:]
    rng.shuffle(directions)
    directions = directions[:safe_count]

    sides = SIDES[:]
    rng.shuffle(sides)
    while len(sides) < safe_count:
        pool = SIDES[:]
        rng.shuffle(pool)
        sides.extend(pool)
    sides = sides[:safe_count]

    specs = []
    for idx in range(safe_count):
        outer_h = rng.randint(9, 13)
        outer_w = rng.randint(9, 13)
        interior_h = outer_h - 2 * FRAME_THICKNESS
        interior_w = outer_w - 2 * FRAME_THICKNESS
        template_choices = [tpl for tpl in HOLE_TEMPLATES if _shape_from_template(tpl, sides[idx], outer_h, outer_w, rng) is not None]
        if not template_choices:
            return None
        template = rng.choice(template_choices)
        hole_mask = _shape_from_template(template, sides[idx], outer_h, outer_w, rng)
        if hole_mask is None:
            return None
        key_mask, key_cells_local = _tight_shape(hole_mask)
        pattern_mask = _random_pattern(interior_h, interior_w, rng)
        if pattern_mask is None:
            return None
        specs.append(
            {
                "color": colors[idx],
                "direction": directions[idx],
                "side": sides[idx],
                "outer_h": outer_h,
                "outer_w": outer_w,
                "interior_h": interior_h,
                "interior_w": interior_w,
                "hole_mask": hole_mask,
                "key_mask": key_mask,
                "key_cells_local": key_cells_local,
                "pattern_mask": pattern_mask,
            }
        )

    specs.sort(
        key=lambda spec: spec["outer_h"] * spec["outer_w"]
        + (spec["interior_h"] * spec["interior_w"]),
        reverse=True,
    )

    blocked = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []
    for spec in specs:
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        direction = spec["direction"]
        placed_here = False
        for _ in range(400):
            top = rng.randint(1, height - outer_h - 1)
            left = rng.randint(1, width - outer_w - 1)
            safe_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
            landing_bbox = _safe_landing_bbox(top, left, outer_h, outer_w, direction)
            if not _bbox_in_bounds(landing_bbox, height, width):
                continue
            if not _region_free(blocked, safe_bbox, pad=1):
                continue
            if not _region_free(blocked, landing_bbox, pad=1):
                continue
            _mark_blocked(blocked, safe_bbox, pad=1)
            _mark_blocked(blocked, landing_bbox, pad=1)

            hole_cells = [(top + r, left + c) for r, c in np.argwhere(spec["hole_mask"])]
            ir0 = top + FRAME_THICKNESS
            ic0 = left + FRAME_THICKNESS
            ir1 = top + outer_h - FRAME_THICKNESS - 1
            ic1 = left + outer_w - FRAME_THICKNESS - 1
            pattern_cells = [(ir0 + r, ic0 + c) for r, c in np.argwhere(spec["pattern_mask"])]
            placed.append(
                {
                    **spec,
                    "top": top,
                    "left": left,
                    "safe_bbox": safe_bbox,
                    "landing_bbox": landing_bbox,
                    "hole_cells": hole_cells,
                    "interior_bbox": (ir0, ic0, ir1, ic1),
                    "pattern_cells": pattern_cells,
                }
            )
            placed_here = True
            break
        if not placed_here:
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    for obj in placed:
        top = obj["top"]
        left = obj["left"]
        outer_h = obj["outer_h"]
        outer_w = obj["outer_w"]
        band = _border_band_mask(outer_h, outer_w, FRAME_THICKNESS)
        grid[top : top + outer_h, left : left + outer_w][band] = GREY
        for r, c in obj["hole_cells"]:
            grid[r, c] = BLACK
        for r, c in obj["pattern_cells"]:
            grid[r, c] = obj["color"]

    for obj in placed:
        key_h, key_w = obj["key_mask"].shape
        arrow_len = rng.randint(1, 2)
        placed_key = False
        for _ in range(500):
            key_top = rng.randint(1, height - key_h - 2)
            key_left = rng.randint(1, width - key_w - 2)
            key_bbox = (key_top, key_left, key_top + key_h - 1, key_left + key_w - 1)
            arrow_cells = _arrow_cells_for_key(key_top, key_left, key_h, key_w, obj["direction"], arrow_len)
            arrow_bbox = _bbox(arrow_cells)
            union_bbox = (
                min(key_bbox[0], arrow_bbox[0]),
                min(key_bbox[1], arrow_bbox[1]),
                max(key_bbox[2], arrow_bbox[2]),
                max(key_bbox[3], arrow_bbox[3]),
            )
            if not _bbox_in_bounds(key_bbox, height, width) or not _bbox_in_bounds(arrow_bbox, height, width):
                continue
            if not _region_free(blocked, union_bbox, pad=1):
                continue
            _mark_blocked(blocked, union_bbox, pad=1)
            key_cells = [(key_top + r, key_left + c) for r, c in obj["key_cells_local"]]
            obj["key_cells"] = key_cells
            obj["arrow_cells"] = arrow_cells
            placed_key = True
            break
        if not placed_key:
            return None

    for obj in placed:
        _paint_cells(grid, obj["key_cells"], obj["color"])
        _paint_cells(grid, obj["arrow_cells"], MAROON)

    expected = _expected_output(grid, placed)
    return grid, expected


def generate_input() -> np.ndarray:
    for attempt in range(500):
        rng = random.Random((random.getrandbits(32) << 10) ^ attempt)
        built = _build_case(rng)
        if built is None:
            continue
        grid, expected = built
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
