import random

import numpy as np


BG = 0
DIVIDER = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ROTATION = "rotation"
NONE = "none"
ORDERED_CATEGORIES = [HORIZONTAL, VERTICAL, ROTATION, NONE]

MAX_SHAPE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    h_sym = bool(np.array_equal(mask, np.flipud(mask)))
    v_sym = bool(np.array_equal(mask, np.fliplr(mask)))
    r_sym = bool(np.array_equal(mask, np.rot90(mask, 2)))
    return h_sym, v_sym, r_sym


def _mask_category_or_none(mask: np.ndarray) -> str | None:
    h_sym, v_sym, r_sym = _symmetry_flags(mask)
    if h_sym and not v_sym:
        return HORIZONTAL
    if v_sym and not h_sym:
        return VERTICAL
    if r_sym and not h_sym and not v_sym:
        return ROTATION
    if not h_sym and not v_sym and not r_sym:
        return NONE
    return None


def _classify_mask(mask: np.ndarray) -> str:
    category = _mask_category_or_none(mask)
    if category is not None:
        return category
    raise AssertionError("object shape does not match one of the four allowed symmetry classes")


def _find_divider(grid: np.ndarray, bg: int = BG) -> tuple[int, int, int]:
    h, w = grid.shape
    full_rows = [(r, int(grid[r, 0])) for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    full_cols = [(c, int(grid[0, c])) for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    matches: list[tuple[int, int, int]] = []
    for r, color_r in full_rows:
        for c, color_c in full_cols:
            if color_r == color_c:
                matches.append((int(r), int(c), int(color_r)))
    assert len(matches) == 1, "expected exactly one matching full-row/full-column divider"
    return matches[0]


def _extract_objects(grid: np.ndarray, mid_r: int, mid_c: int, bg: int = BG) -> list[dict]:
    obj_mask = grid != bg
    obj_mask[mid_r, :] = False
    obj_mask[:, mid_c] = False

    objects: list[dict] = []
    for comp in _components(obj_mask):
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        mask = np.zeros_like(patch, dtype=bool)
        for r, c in comp:
            patch[r - r0, c - c0] = int(grid[r, c])
            mask[r - r0, c - c0] = True
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": mask,
                "category": _classify_mask(mask),
            }
        )
    return objects


def _paste_patch(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int = BG) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= out.shape[0] and c0 + w <= out.shape[1]
    rows, cols = np.where(patch != bg)
    for dr, dc in zip(rows.tolist(), cols.tolist(), strict=False):
        rr = r0 + dr
        cc = c0 + dc
        assert out[rr, cc] == bg
        out[rr, cc] = int(patch[dr, dc])


def _pack_strip(out: np.ndarray, objects: list[dict], bounds: tuple[int, int, int, int], align_top: bool, align_right: bool) -> None:
    if not objects:
        return
    r0, r1, c0, c1 = bounds
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    required_width = sum(obj["patch"].shape[1] for obj in objects) + max(0, len(objects) - 1)
    required_height = max(obj["patch"].shape[0] for obj in objects)
    assert required_width <= width and required_height <= height

    if align_right:
        cur = c1
        for obj in objects:
            patch = obj["patch"]
            h, w = patch.shape
            rr = r0 if align_top else r1 - h + 1
            cc = cur - w + 1
            _paste_patch(out, patch, rr, cc)
            cur = cc - 2
    else:
        cur = c0
        for obj in objects:
            patch = obj["patch"]
            h, w = patch.shape
            rr = r0 if align_top else r1 - h + 1
            _paste_patch(out, patch, rr, cur)
            cur += w + 1


def main(grid: np.ndarray) -> np.ndarray:
    """Classify each foreground object's symmetry type and move it into the matching quadrant.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column form a cross that splits the grid into four quadrants.
    - Several 4-connected multicolor foreground objects lie away from the divider, and each object's shape belongs to exactly one class: horizontal-reflection only, vertical-reflection only, 180-degree rotation only, or no symmetry.

    Rule:
    - Detect each object's symmetry from its occupied-cell mask, ignoring its internal colors.
    - Move horizontal-reflection objects to the top-left quadrant, vertical-reflection objects to the top-right quadrant, rotation-only objects to the bottom-left quadrant, and non-symmetric objects to the bottom-right quadrant.
    - Keep each object's colors and shape unchanged, clear its original location, and preserve the grey cross.
    - Within each target quadrant, place objects in original row-major order along the quadrant edge nearest the corner, with a one-cell gap.

    Key insight:
    - The symmetry type is a label used for sorting objects into quadrant bins; the divider is only a persistent layout guide, not a symmetry axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = BG
    mid_r, mid_c, divider_color = _find_divider(g, bg)
    assert divider_color != bg

    objects = _extract_objects(g, mid_r, mid_c, bg)
    assert objects, "expected at least one foreground object"

    grouped = {category: [] for category in ORDERED_CATEGORIES}
    for obj in objects:
        grouped[obj["category"]].append(obj)
    for category in ORDERED_CATEGORIES:
        grouped[category].sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

    out = np.full((h, w), bg, dtype=int)
    out[mid_r, :] = divider_color
    out[:, mid_c] = divider_color

    _pack_strip(out, grouped[HORIZONTAL], (0, mid_r - 1, 0, mid_c - 1), align_top=True, align_right=False)
    _pack_strip(out, grouped[VERTICAL], (0, mid_r - 1, mid_c + 1, w - 1), align_top=True, align_right=True)
    _pack_strip(out, grouped[ROTATION], (mid_r + 1, h - 1, 0, mid_c - 1), align_top=False, align_right=False)
    _pack_strip(out, grouped[NONE], (mid_r + 1, h - 1, mid_c + 1, w - 1), align_top=False, align_right=True)
    return out


def _seed_cells(category: str, h: int, w: int) -> set[tuple[int, int]]:
    if category == HORIZONTAL:
        c = random.randrange(w)
        if h % 2 == 1:
            return {(h // 2, c)}
        return {(h // 2 - 1, c), (h // 2, c)}
    if category == VERTICAL:
        r = random.randrange(h)
        if w % 2 == 1:
            return {(r, w // 2)}
        return {(r, w // 2 - 1), (r, w // 2)}
    if category == ROTATION:
        if h % 2 == 1 and w % 2 == 1:
            return {(h // 2, w // 2)}
        if h % 2 == 0 and w % 2 == 0:
            return {
                (h // 2 - 1, w // 2 - 1),
                (h // 2 - 1, w // 2),
                (h // 2, w // 2 - 1),
                (h // 2, w // 2),
            }
        if h % 2 == 1:
            return {(h // 2, w // 2 - 1), (h // 2, w // 2)}
        return {(h // 2 - 1, w // 2), (h // 2, w // 2)}
    raise AssertionError("bad category")


def _orbit_cells(category: str, r: int, c: int, h: int, w: int) -> set[tuple[int, int]]:
    if category == HORIZONTAL:
        return {(r, c), (h - 1 - r, c)}
    if category == VERTICAL:
        return {(r, c), (r, w - 1 - c)}
    if category == ROTATION:
        return {(r, c), (h - 1 - r, w - 1 - c)}
    raise AssertionError("bad category")


def _grow_connected_mask(target_category: str) -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        cells = _seed_cells(target_category, h, w)
        max_area = min(10, h * w)
        growth_steps = random.randint(2, 6)
        for _step in range(growth_steps):
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break

            options: list[tuple[list[tuple[int, int]], int]] = []
            seen: set[tuple[tuple[int, int], ...]] = set()
            for nr, nc in frontier:
                orbit = _orbit_cells(target_category, nr, nc, h, w)
                new_cells = sorted(orbit - cells)
                if not new_cells or len(cells) + len(new_cells) > max_area:
                    continue
                key = tuple(new_cells)
                if key in seen:
                    continue
                seen.add(key)
                touch = 0
                for rr, cc in new_cells:
                    touch += sum((rr + dr, cc + dc) in cells for dr, dc in NEIGHBORS4)
                options.append((new_cells, max(1, touch)))
            if not options:
                break
            chosen = random.choices([cells_ for cells_, _ in options], weights=[score for _, score in options], k=1)[0]
            cells.update(chosen)

        mask = _mask_from_cells(cells, h, w)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 4:
            continue
        if len(_components(mask)) != 1:
            continue
        if _mask_category_or_none(mask) == target_category:
            return mask
    raise RuntimeError(f"failed to generate a {target_category} mask")


def _grow_nonsymmetric_mask() -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        target_area = random.randint(4, min(10, h * w - 1))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier: set[tuple[int, int]] = set()
        for dr, dc in NEIGHBORS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target_area and frontier:
            options = sorted(
                frontier,
                key=lambda cell: sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in NEIGHBORS4),
                reverse=True,
            )
            cell = random.choice(options[: min(4, len(options))])
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in NEIGHBORS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        mask = _mask_from_cells(cells, h, w)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 4:
            continue
        if len(_components(mask)) != 1:
            continue
        if _mask_category_or_none(mask) == NONE:
            return mask
    raise RuntimeError("failed to generate a non-symmetric mask")


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    palette_size = random.randint(2, min(3, len(cells)))
    palette = random.sample(OBJECT_COLORS, palette_size)
    color_offset = random.randrange(palette_size)
    for idx, (r, c) in enumerate(cells):
        color = palette[(2 * r + c + color_offset) % palette_size]
        patch[r, c] = color
        if idx < palette_size:
            patch[r, c] = palette[idx]
    return patch


def _sample_object(category: str) -> dict:
    mask = _grow_nonsymmetric_mask() if category == NONE else _grow_connected_mask(category)
    return {"category": category, "patch": _colorize_mask(mask)}


def _group_width(objects: list[dict]) -> int:
    if not objects:
        return 0
    return sum(obj["patch"].shape[1] for obj in objects) + len(objects) - 1


def _group_height(objects: list[dict]) -> int:
    if not objects:
        return 0
    return max(obj["patch"].shape[0] for obj in objects)


def _choose_axis_sizes(upper_req: int, lower_req: int) -> tuple[int, int] | None:
    for _ in range(60):
        upper = upper_req + random.randint(2, 4)
        lower = lower_req + random.randint(2, 4)
        if upper + 1 + lower <= 30:
            return upper, lower
    return None


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, mid_r: int, mid_c: int) -> bool:
    if r0 <= mid_r < r0 + h or c0 <= mid_c < c0 + w:
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_input_objects(h: int, w: int, mid_r: int, mid_c: int, objects: list[dict]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    grid[mid_r, :] = DIVIDER
    grid[:, mid_c] = DIVIDER

    occupied = np.zeros((h, w), dtype=bool)
    occupied[mid_r, :] = True
    occupied[:, mid_c] = True

    order = list(objects)
    random.shuffle(order)
    order.sort(key=lambda obj: obj["patch"].shape[0] * obj["patch"].shape[1], reverse=True)

    for obj in order:
        patch = obj["patch"]
        ph, pw = patch.shape
        candidates = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if _can_place_box(occupied, r0, c0, ph, pw, mid_r, mid_c):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        r0, c0 = chosen
        _paste_patch(grid, patch, r0, c0)
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True
    return grid


def _gen_attempt() -> np.ndarray | None:
    grouped = {category: [_sample_object(category) for _ in range(random.randint(1, 2))] for category in ORDERED_CATEGORIES}
    all_objects = [obj for category in ORDERED_CATEGORIES for obj in grouped[category]]

    left_req = max(_group_width(grouped[HORIZONTAL]), _group_width(grouped[ROTATION]))
    right_req = max(_group_width(grouped[VERTICAL]), _group_width(grouped[NONE]))
    top_req = max(_group_height(grouped[HORIZONTAL]), _group_height(grouped[VERTICAL]))
    bottom_req = max(_group_height(grouped[ROTATION]), _group_height(grouped[NONE]))

    widths = _choose_axis_sizes(left_req, right_req)
    heights = _choose_axis_sizes(top_req, bottom_req)
    if widths is None or heights is None:
        return None

    left_w, right_w = widths
    top_h, bottom_h = heights
    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w
    if h > 30 or w > 30:
        return None

    grid = _place_input_objects(h, w, top_h, left_w, all_objects)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    found = _extract_objects(grid, top_h, left_w, BG)
    if len(found) != len(all_objects):
        return None
    category_counts = {category: 0 for category in ORDERED_CATEGORIES}
    for obj in found:
        category_counts[obj["category"]] += 1
    if any(category_counts[category] != len(grouped[category]) for category in ORDERED_CATEGORIES):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
