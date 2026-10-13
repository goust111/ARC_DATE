import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
SIDES_3 = [
    ("top", "bottom", "left"),
    ("top", "bottom", "right"),
    ("top", "left", "right"),
    ("bottom", "left", "right"),
]
SIDES_4 = [("top", "bottom", "left", "right")]
SIZE_CHOICES = [
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
]
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude=None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _expand_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _marker_direction(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    if mr < r0 and c0 <= mc <= c1:
        return "up"
    if mr > r1 and c0 <= mc <= c1:
        return "down"
    if mc < c0 and r0 <= mr <= r1:
        return "left"
    if mc > c1 and r0 <= mr <= r1:
        return "right"
    return None


def _stretch_to_target(bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    direction = _marker_direction(marker, bbox)
    assert direction is not None
    if direction == "up":
        return mr, c0, r1, c1
    if direction == "down":
        return r0, c0, mr, c1
    if direction == "left":
        return r0, mc, r1, c1
    return r0, c0, r1, mc


def _is_solid_rect(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = []
    for color in _colors_present(g, exclude={bg}):
        comps = _components_mask(g == color)
        comp_infos = []
        for coords in comps:
            bbox = _bbox(coords)
            comp_infos.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(coords.shape[0]),
                }
            )
        rect = max(comp_infos, key=lambda info: info["area"])
        assert rect["area"] >= 4
        assert _is_solid_rect(rect["coords"], rect["bbox"])
        marker_candidates = []
        extra_cells = []
        for info in comp_infos:
            if info is rect:
                continue
            assert info["area"] == 1
            rc = tuple(map(int, info["coords"][0]))
            if _marker_direction(rc, rect["bbox"]) is not None:
                marker_candidates.append(rc)
            else:
                extra_cells.append(rc)
        assert len(marker_candidates) == 1
        marker = marker_candidates[0]
        objects.append(
            {
                "color": int(color),
                "bbox": rect["bbox"],
                "area": rect["area"],
                "marker": marker,
                "direction": _marker_direction(marker, rect["bbox"]),
                "total": int((g == color).sum()),
                "extras": extra_cells,
            }
        )
    totals = [obj["total"] for obj in objects]
    assert len(set(totals)) == len(totals)
    return objects


def _stretch_with_blocked(
    blocked: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    assert not blocked[r0 : r1 + 1, c0 : c1 + 1].any()
    direction = _marker_direction(marker, bbox)
    assert direction is not None
    mr, mc = marker
    if direction == "up":
        nr0 = r0
        while nr0 > mr and not blocked[nr0 - 1, c0 : c1 + 1].any():
            nr0 -= 1
        return nr0, c0, r1, c1
    if direction == "down":
        nr1 = r1
        while nr1 < mr and not blocked[nr1 + 1, c0 : c1 + 1].any():
            nr1 += 1
        return r0, c0, nr1, c1
    if direction == "left":
        nc0 = c0
        while nc0 > mc and not blocked[r0 : r1 + 1, nc0 - 1].any():
            nc0 -= 1
        return r0, nc0, r1, c1
    nc1 = c1
    while nc1 < mc and not blocked[r0 : r1 + 1, nc1 + 1].any():
        nc1 += 1
    return r0, c0, r1, nc1


def _solve(grid: np.ndarray, order_override=None, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG
    objects = _find_objects(g)
    order = (
        list(order_override)
        if order_override is not None
        else [obj["color"] for obj in sorted(objects, key=lambda obj: obj["total"])]
    )
    object_by_color = {obj["color"]: obj for obj in objects}
    out = np.full_like(g, bg)
    blocked = np.zeros(g.shape, dtype=bool)
    meta = []
    for color in order:
        obj = object_by_color[int(color)]
        final_bbox = _stretch_with_blocked(blocked, obj["bbox"], obj["marker"])
        _paint_rect(out, final_bbox, obj["color"])
        _expand_blocked(blocked, final_bbox)
        meta.append(
            {
                **obj,
                "naive_bbox": _stretch_to_target(obj["bbox"], obj["marker"]),
                "final_bbox": final_bbox,
            }
        )
    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch several marked solid rectangles in rarity order while preserving a one-cell buffer between processed rectangles.

    Input structure:
    - The background is black.
    - Each non-black color has one solid axis-aligned rectangle, one same-color singleton marker aligned with exactly one side extension, and possibly a few detached same-color tally singletons.
    - The tally singletons are not row- or column-aligned with that rectangle, so the aligned singleton remains the unique marker for that color.
    - Different colors have different total pixel counts across the whole grid.

    Rule:
    - For each color, count every pixel of that color in the grid to get its rarity score.
    - Process the colors from smallest total count to largest total count.
    - For the current rectangle, use the aligned singleton to choose the stretch direction and its target row or column.
    - Extend that one side step by step toward the marker, but stop before the rectangle would enter any cell occupied by an already processed rectangle or its one-cell buffer zone.
    - Redraw only the stretched rectangles on a black canvas; markers and detached tally singletons disappear.

    Key insight:
    - The same-color detached tally pixels affect only processing order, and that order changes which rectangle claims contested space first because every finished rectangle creates a one-cell moat that blocks later stretches.
    """
    return _solve(np.asarray(grid, dtype=int))


def _choose_sides() -> tuple[str, ...]:
    if random.random() < 0.6:
        return random.choice(SIDES_4)
    return random.choice(SIDES_3)


def _choose_layout(h: int, w: int, sides: tuple[str, ...]) -> dict | None:
    center_r = random.randint(h // 2 - 1, h // 2 + 1)
    center_c = random.randint(w // 2 - 1, w // 2 + 1)
    layout = {}

    for side in sides:
        rh, rw = random.choice(SIZE_CHOICES)
        if side == "top":
            c_lo = max(1, center_c - rw + 1)
            c_hi = min(center_c, w - rw - 2)
            if c_lo > c_hi:
                return None
            r0 = random.randint(1, 2)
            c0 = random.randint(c_lo, c_hi)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            m_lo = max(r1 + 3, center_r + 1)
            m_hi = h - 4
            if m_lo > m_hi:
                return None
            marker = (random.randint(m_lo, m_hi), random.randint(c0, c1))
        elif side == "bottom":
            c_lo = max(1, center_c - rw + 1)
            c_hi = min(center_c, w - rw - 2)
            if c_lo > c_hi:
                return None
            r1 = h - 1 - random.randint(1, 2)
            r0 = r1 - rh + 1
            c0 = random.randint(c_lo, c_hi)
            c1 = c0 + rw - 1
            m_lo = 3
            m_hi = min(r0 - 3, center_r - 1)
            if m_lo > m_hi:
                return None
            marker = (random.randint(m_lo, m_hi), random.randint(c0, c1))
        elif side == "left":
            r_lo = max(1, center_r - rh + 1)
            r_hi = min(center_r, h - rh - 2)
            if r_lo > r_hi:
                return None
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(1, 2)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            m_lo = max(c1 + 3, center_c + 1)
            m_hi = w - 4
            if m_lo > m_hi:
                return None
            marker = (random.randint(r0, r1), random.randint(m_lo, m_hi))
        else:
            r_lo = max(1, center_r - rh + 1)
            r_hi = min(center_r, h - rh - 2)
            if r_lo > r_hi:
                return None
            r0 = random.randint(r_lo, r_hi)
            c1 = w - 1 - random.randint(1, 2)
            c0 = c1 - rw + 1
            r1 = r0 + rh - 1
            m_lo = 3
            m_hi = min(c0 - 3, center_c - 1)
            if m_lo > m_hi:
                return None
            marker = (random.randint(r0, r1), random.randint(m_lo, m_hi))

        layout[side] = {"bbox": (r0, c0, r1, c1), "marker": marker, "area": rh * rw}

    occupied = np.zeros((h, w), dtype=bool)
    for spec in layout.values():
        r0, c0, r1, c1 = spec["bbox"]
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        mr, mc = spec["marker"]
        if occupied[mr, mc]:
            return None
        occupied[mr, mc] = True
    return layout


def _choose_bonuses(areas: list[int]) -> list[int] | None:
    base = [area + 1 for area in areas]
    area_order = tuple(np.argsort(base, kind="mergesort"))
    for _ in range(200):
        bonuses = [random.randint(0, 8) for _ in base]
        totals = [b + bonus for b, bonus in zip(base, bonuses)]
        if len(set(totals)) != len(totals):
            continue
        total_order = tuple(np.argsort(totals, kind="mergesort"))
        if total_order != area_order:
            return bonuses
    return None


def _free_singleton_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG:
                continue
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(h - 1, r + 1)
            c1 = min(w - 1, c + 1)
            if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BG):
                continue
            cells.append((r, c))
    return cells


def _place_tallies(grid: np.ndarray, color_specs: list[dict]) -> bool:
    for spec in color_specs:
        r0, c0, r1, c1 = spec["bbox"]
        count = spec["bonus"]
        for _ in range(count):
            cells = [
                (r, c)
                for r, c in _free_singleton_cells(grid)
                if not (r0 <= r <= r1 or c0 <= c <= c1)
            ]
            if not cells:
                return False
            r, c = random.choice(cells)
            grid[r, c] = spec["color"]
    return True


def _naive_overlap(meta: list[dict]) -> bool:
    for i, obj_a in enumerate(meta):
        ar0, ac0, ar1, ac1 = obj_a["naive_bbox"]
        for obj_b in meta[i + 1 :]:
            br0, bc0, br1, bc1 = obj_b["naive_bbox"]
            if max(ar0, br0) <= min(ar1, br1) and max(ac0, bc0) <= min(ac1, bc1):
                return True
    return False


def _reverse_order_changes(grid: np.ndarray, meta: list[dict]) -> bool:
    order = [obj["color"] for obj in sorted(meta, key=lambda obj: obj["total"])]
    rev = list(reversed(order))
    try:
        alt = _solve(grid, order_override=rev)
    except Exception:
        return True
    return not np.array_equal(_solve(grid, order_override=order), alt)


def _valid_scene(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid, return_meta=True)
    except Exception:
        return False
    if np.array_equal(grid, out):
        return False
    if not _naive_overlap(meta):
        return False
    if not any(obj["final_bbox"] != obj["naive_bbox"] for obj in meta):
        return False
    if not _reverse_order_changes(grid, meta):
        return False
    areas = [obj["area"] for obj in meta]
    totals = [obj["total"] for obj in meta]
    if tuple(np.argsort(areas, kind="mergesort")) == tuple(np.argsort(totals, kind="mergesort")):
        return False
    return True


def _attempt() -> np.ndarray | None:
    h = random.randint(17, 24)
    w = random.randint(17, 24)
    sides = _choose_sides()
    layout = _choose_layout(h, w, sides)
    if layout is None:
        return None

    colors = random.sample(PALETTE, len(sides))
    areas = [layout[side]["area"] for side in sides]
    bonuses = _choose_bonuses(areas)
    if bonuses is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    color_specs = []
    for side, color, bonus in zip(sides, colors, bonuses):
        bbox = layout[side]["bbox"]
        marker = layout[side]["marker"]
        _paint_rect(grid, bbox, color)
        grid[marker] = color
        color_specs.append(
            {
                "side": side,
                "color": color,
                "bbox": bbox,
                "marker": marker,
                "bonus": bonus,
            }
        )

    random.shuffle(color_specs)
    if not _place_tallies(grid, color_specs):
        return None
    if not _valid_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
