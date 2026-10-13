from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
TEMPLATE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (12, 25)
GRID_WIDTH_RANGE = (12, 25)
TEMPLATE_COUNT_RANGE = (2, 5)
TEMPLATE_DIMS = (
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 5),
    (5, 6),
)
MIN_FILL_FRACTION = 0.35
MAX_FILL_FRACTION = 0.65
PLACEMENT_GAP = 1
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        for coords in _components_mask(grid == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    mask[rr, cc] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return np.array_equal(mask, np.fliplr(mask))


def _symmetrized_mask(mask: np.ndarray) -> np.ndarray:
    return np.logical_or(mask, np.fliplr(mask))


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _draw_random_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> None:
    r, c = start
    tr, tc = goal
    mask[r, c] = True
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        mask[r, c] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((nr, nc))
    return list(out)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask)
    return len(comps) == 1


def _generate_template_mask(h: int, w: int) -> np.ndarray | None:
    total = h * w
    min_area = max(4, int(np.ceil(total * MIN_FILL_FRACTION)))
    max_area = min(total - 1, int(np.floor(total * MAX_FILL_FRACTION)))
    if min_area > max_area:
        return None

    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)
        for anchor in anchors:
            _draw_random_path(mask, seed, anchor)
        base_area = int(mask.sum())
        if base_area > max_area:
            continue

        target = random.randint(max(min_area, base_area), max_area)
        while int(mask.sum()) < target:
            frontier = _frontier(mask)
            if not frontier:
                break
            candidates = []
            for cell in frontier:
                trial = mask.copy()
                trial[cell] = True
                if _touches_all_sides(trial):
                    candidates.append(cell)
            if not candidates:
                candidates = frontier
            mask[random.choice(candidates)] = True

        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if _is_lr_symmetric(mask):
            continue
        sym = _symmetrized_mask(mask)
        if not _is_connected(sym):
            continue
        return mask
    return None


def _choose_dims(count: int) -> list[tuple[int, int]]:
    dims = [random.choice(TEMPLATE_DIMS), random.choice(TEMPLATE_DIMS)]
    while len(dims) < count:
        dims.append(random.choice(TEMPLATE_DIMS))
    random.shuffle(dims)
    return dims


def _build_templates() -> list[dict] | None:
    count = random.randint(*TEMPLATE_COUNT_RANGE)
    colors = random.sample(TEMPLATE_COLORS, count)
    dims = _choose_dims(count)

    templates = []
    used_orig = set()
    used_sym = set()
    for color, (h, w) in zip(colors, dims):
        chosen = None
        for _ in range(200):
            mask = _generate_template_mask(h, w)
            if mask is None:
                continue
            sym = _symmetrized_mask(mask)
            orig_key = _mask_key(mask)
            sym_key = _mask_key(sym)
            if orig_key in used_orig or sym_key in used_sym:
                continue
            chosen = {
                "color": int(color),
                "mask": mask,
                "sym": sym,
                "orig_key": orig_key,
                "sym_key": sym_key,
                "orig_area": int(mask.sum()),
                "sym_area": int(sym.sum()),
                "dims": tuple(mask.shape),
            }
            break
        if chosen is None:
            return None
        used_orig.add(chosen["orig_key"])
        used_sym.add(chosen["sym_key"])
        templates.append(chosen)

    repeated_dims = any(
        templates[i]["sym"].shape == templates[j]["sym"].shape
        for i in range(len(templates))
        for j in range(i + 1, len(templates))
    )
    repeated_area = any(
        templates[i]["sym_area"] == templates[j]["sym_area"]
        for i in range(len(templates))
        for j in range(i + 1, len(templates))
    )
    if not (repeated_dims or repeated_area):
        return None
    return templates


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    hh, ww = blocked.shape
    blocked[
        max(0, r0 - PLACEMENT_GAP) : min(hh, r0 + h + PLACEMENT_GAP),
        max(0, c0 - PLACEMENT_GAP) : min(ww, c0 + w + PLACEMENT_GAP),
    ] = True


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = dst[r0 : r0 + h, c0 : c0 + w]
    region[patch != BG] = patch[patch != BG]


def _place_objects(height: int, width: int, objects: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    order = objects[:]
    random.shuffle(order)
    order.sort(key=lambda obj: obj["patch_in"].shape[0] * obj["patch_in"].shape[1], reverse=True)

    for obj in order:
        patch_in = obj["patch_in"]
        patch_out = obj["patch_out"]
        ph, pw = patch_in.shape
        candidates = [
            (r0, c0)
            for r0 in range(height - ph + 1)
            for c0 in range(width - pw + 1)
            if _can_place(blocked, r0, c0, ph, pw)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _paste(inp, patch_in, r0, c0)
        _paste(out, patch_out, r0, c0)
        _mark_blocked(blocked, r0, c0, ph, pw)
    return inp, out


def _build_objects(templates: list[dict]) -> list[dict]:
    objects = []
    for tpl in templates:
        objects.append(
            {
                "kind": "template",
                "patch_in": _paint_patch(tpl["mask"], tpl["color"]),
                "patch_out": _paint_patch(tpl["mask"], tpl["color"]),
            }
        )
        objects.append(
            {
                "kind": "placeholder",
                "patch_in": _paint_patch(tpl["sym"], BLUE),
                "patch_out": _paint_patch(tpl["sym"], tpl["color"]),
            }
        )
    return objects


def _validate_puzzle(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = _get_bg(inp)
    comps = _find_components(inp, bg)
    blue = [comp for comp in comps if comp["color"] == BLUE]
    templates = [comp for comp in comps if comp["color"] != BLUE]
    if len(blue) < 2 or len(blue) != len(templates):
        return False

    direct_keys = set()
    sym_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    sym_shapes = []
    for comp in templates:
        mask = _mask_from_coords(comp["coords"])
        if _is_lr_symmetric(mask):
            return False
        sym = _symmetrized_mask(mask)
        if not _is_connected(sym):
            return False
        direct_keys.add(_mask_key(mask))
        sym_key = _mask_key(sym)
        if sym_key in sym_to_color:
            return False
        sym_to_color[sym_key] = int(comp["color"])
        sym_shapes.append(sym.shape)

    if not any(
        sym_shapes[i] == sym_shapes[j]
        for i in range(len(sym_shapes))
        for j in range(i + 1, len(sym_shapes))
    ):
        return False

    seen_blue = set()
    for comp in blue:
        mask = _mask_from_coords(comp["coords"])
        if not _is_lr_symmetric(mask):
            return False
        key = _mask_key(mask)
        if key not in sym_to_color:
            return False
        if key in direct_keys:
            return False
        if key in seen_blue:
            return False
        seen_blue.add(key)

    if len(seen_blue) != len(blue):
        return False
    if not np.any(inp != expected):
        return False
    if not np.any(expected == 2) and not np.any(expected == 3) and not np.any(expected == 4):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each blue placeholder with the color of the asymmetric template whose left-right symmetrized mask matches it.

    Input structure:
    - A black background contains several separated 4-connected objects.
    - Blue objects are placeholders and are each left-right symmetric inside their own tight bounding boxes.
    - Every non-blue object is a colored asymmetric template.

    Rule:
    - For each colored template, crop its tight bbox and take the union of its mask with its left-right mirror inside that bbox.
    - Compare each blue object's normalized mask against those symmetrized template masks.
    - Every blue object matches exactly one template after this symmetrization; recolor the whole blue object to that template's color and leave templates and background unchanged.

    Key insight:
    - Direct shape matching is guaranteed to fail because the placeholders already show the mirrored union, while the visible templates are intentionally asymmetric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected foreground objects"

    sym_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    direct_template_keys = set()
    blue_components = []

    for comp in comps:
        mask = _mask_from_coords(comp["coords"])
        if comp["color"] == BLUE:
            assert _is_lr_symmetric(mask), "blue placeholders must be left-right symmetric"
            blue_components.append(comp)
            continue

        assert not _is_lr_symmetric(mask), "templates must be asymmetric before mirroring"
        sym = _symmetrized_mask(mask)
        assert _is_connected(sym), "symmetrized template must stay 4-connected"
        direct_template_keys.add(_mask_key(mask))
        sym_key = _mask_key(sym)
        assert sym_key not in sym_to_color, "symmetrized template masks must be unique"
        sym_to_color[sym_key] = int(comp["color"])

    assert blue_components, "expected blue placeholder objects"
    assert sym_to_color, "expected non-blue template objects"

    out = g.copy()
    for comp in blue_components:
        mask = _mask_from_coords(comp["coords"])
        key = _mask_key(mask)
        assert key in sym_to_color, "each placeholder must match one symmetrized template"
        assert key not in direct_template_keys, "direct template matching must fail"
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = sym_to_color[key]

    return out


def _attempt_generate() -> np.ndarray | None:
    templates = _build_templates()
    if templates is None:
        return None

    objects = _build_objects(templates)
    min_h = max(obj["patch_in"].shape[0] for obj in objects)
    min_w = max(obj["patch_in"].shape[1] for obj in objects)
    height = random.randint(max(GRID_HEIGHT_RANGE[0], min_h + 2), GRID_HEIGHT_RANGE[1] - 1)
    width = random.randint(max(GRID_WIDTH_RANGE[0], min_w + 2), GRID_WIDTH_RANGE[1] - 1)

    placed = _place_objects(height, width, objects)
    if placed is None:
        return None
    inp, expected = placed

    if not _validate_puzzle(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-union matching puzzle")
