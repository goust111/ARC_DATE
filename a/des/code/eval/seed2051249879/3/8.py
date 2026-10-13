import numpy as np


GREY = 5
MAX_GENERATION_ATTEMPTS = 300
COLOR_VALUES = tuple(range(10))
GRID_HEIGHT_RANGE = (10, 19)
GRID_WIDTH_RANGE = (10, 19)
OBJECT_COUNT_RANGE = (3, 6)
OVERLAP_SIZE_RANGE = (2, 5)
SEPARATED_SIZE_RANGE = (2, 4)

OVERLAP_OFFSETS = [
    (dy, dx)
    for dy in range(-2, 3)
    for dx in range(-2, 3)
    if (dy, dx) != (0, 0)
]
SEPARATED_OFFSETS = [
    (dy, dx)
    for dy in range(-4, 5)
    for dx in range(-4, 5)
    if (dy, dx) != (0, 0) and max(abs(dy), abs(dx)) >= 3
]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in (
                    (cr - 1, cc),
                    (cr + 1, cc),
                    (cr, cc - 1),
                    (cr, cc + 1),
                ):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, GREY):
            continue
        for coords in _components(grid == color):
            rs = coords[:, 0]
            cs = coords[:, 1]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (
                        int(rs.min()),
                        int(cs.min()),
                        int(rs.max()),
                        int(cs.max()),
                    ),
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _object_mask(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for obj in objects:
        coords = obj["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _visible_shadow_coords(
    coords: np.ndarray,
    dy: int,
    dx: int,
    shape: tuple[int, int],
    object_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    rr = coords[:, 0] + dy
    cc = coords[:, 1] + dx
    inside = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
    if not np.any(inside):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    rr = rr[inside]
    cc = cc[inside]
    visible = ~object_mask[rr, cc]
    return rr[visible], cc[visible]


def _evaluate_offset(
    grid: np.ndarray,
    objects: list[dict],
    object_mask: np.ndarray,
    grey_mask: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[int, ...]] | None:
    union = np.zeros_like(grey_mask, dtype=bool)
    full_shadowed: list[int] = []
    missing_shadow: list[int] = []
    informative = 0

    for idx, obj in enumerate(objects):
        rr, cc = _visible_shadow_coords(obj["coords"], dy, dx, grid.shape, object_mask)
        if rr.size == 0:
            continue
        informative += 1
        grey_hits = grey_mask[rr, cc]
        grey_count = int(np.count_nonzero(grey_hits))
        if grey_count == 0:
            missing_shadow.append(idx)
            continue
        if grey_count != int(rr.size):
            return None
        union[rr, cc] = True
        full_shadowed.append(idx)

    if not full_shadowed:
        return None
    if not np.array_equal(union, grey_mask):
        return None

    score = (len(full_shadowed), informative, int(np.count_nonzero(union)))
    return score, tuple(full_shadowed), tuple(missing_shadow)


def _rank_offsets(grid: np.ndarray, objects: list[dict]) -> list[tuple[tuple[int, int, int], int, int]]:
    grey_mask = grid == GREY
    if not np.any(grey_mask) or not objects:
        return []

    object_mask = _object_mask(grid.shape, objects)
    h, w = grid.shape
    ranked: list[tuple[tuple[int, int, int], int, int]] = []

    for dy in range(-(h - 1), h):
        for dx in range(-(w - 1), w):
            if dy == 0 and dx == 0:
                continue
            result = _evaluate_offset(grid, objects, object_mask, grey_mask, dy, dx)
            if result is None:
                continue
            score, _full_shadowed, _missing_shadow = result
            ranked.append((score, dy, dx))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _infer_shadow_offset(grid: np.ndarray, objects: list[dict]) -> tuple[int, int] | None:
    ranked = _rank_offsets(grid, objects)
    if not ranked:
        return None
    _score, dy, dx = ranked[0]
    return int(dy), int(dx)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the common grey-shadow offset and add missing shadows for objects that lack one.

    Input structure
    - The background is the dominant color.
    - Non-grey foreground objects are small solid colored shapes.
    - Some objects already have a grey shadow: the same shape shifted by one
      common offset vector and drawn beneath the object.

    Rule
    - Infer the shared shadow offset from the objects that already show a shadow.
    - For each object without a shadow, draw the same grey shifted copy beneath it.
    - Existing shadows stay unchanged.
    - Clipped or self-overlapped shadow parts stay clipped/hidden.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _bg_color(g)
    objects = _extract_objects(g, bg)
    if not objects or not np.any(g == GREY):
        return g.copy()

    offset = _infer_shadow_offset(g, objects)
    if offset is None:
        return g.copy()
    dy, dx = offset

    object_mask = _object_mask(g.shape, objects)
    grey_mask = g == GREY
    out = g.copy()

    for obj in objects:
        rr, cc = _visible_shadow_coords(obj["coords"], dy, dx, g.shape, object_mask)
        if rr.size == 0:
            continue
        if not np.any(grey_mask[rr, cc]):
            out[rr, cc] = GREY

    return out


def _rect_cell_set(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _shifted_cell_set(
    cells: set[tuple[int, int]],
    dy: int,
    dx: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    shifted = set()
    for r, c in cells:
        rr = r + dy
        cc = c + dx
        if 0 <= rr < h and 0 <= cc < w:
            shifted.add((rr, cc))
    return shifted


def _sample_size(
    rng: np.random.Generator,
    dy: int,
    dx: int,
    overlap_mode: bool,
    require_overlap: bool,
) -> tuple[int, int]:
    if overlap_mode:
        h = int(rng.integers(*OVERLAP_SIZE_RANGE))
        w = int(rng.integers(*OVERLAP_SIZE_RANGE))
        if require_overlap:
            h = max(h, abs(dy) + 1 if dy != 0 else 2)
            w = max(w, abs(dx) + 1 if dx != 0 else 2)
            h = min(h, OVERLAP_SIZE_RANGE[1] - 1)
            w = min(w, OVERLAP_SIZE_RANGE[1] - 1)
        return h, w

    h = int(rng.integers(*SEPARATED_SIZE_RANGE))
    w = int(rng.integers(*SEPARATED_SIZE_RANGE))
    return h, w


def _place_objects(
    rng: np.random.Generator,
    shape: tuple[int, int],
    dy: int,
    dx: int,
    specs: list[dict],
) -> list[dict] | None:
    h, w = shape
    occupied_bodies: set[tuple[int, int]] = set()
    occupied_shadows: set[tuple[int, int]] = set()
    placed: list[dict | None] = [None] * len(specs)

    order = sorted(
        range(len(specs)),
        key=lambda idx: (
            specs[idx]["must_crop"],
            specs[idx]["need_overlap"],
            specs[idx]["h"] * specs[idx]["w"],
        ),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        positions = [
            (r0, c0)
            for r0 in range(h - spec["h"] + 1)
            for c0 in range(w - spec["w"] + 1)
        ]
        if not positions:
            return None
        rng.shuffle(positions)

        placed_here = None
        for r0, c0 in positions:
            body = _rect_cell_set(r0, c0, spec["h"], spec["w"])
            shadow = _shifted_cell_set(body, dy, dx, shape)
            if not shadow:
                continue

            cropped = len(shadow) < spec["h"] * spec["w"]
            if spec["must_crop"] and not cropped:
                continue
            if spec["must_not_crop"] and cropped:
                continue

            overlap = bool(body & shadow)
            if spec["need_overlap"] and not overlap:
                continue
            if spec["forbid_overlap"] and overlap:
                continue

            visible = shadow - body
            if not visible:
                continue

            if body & occupied_bodies:
                continue
            if body & occupied_shadows:
                continue
            if shadow & occupied_bodies:
                continue
            if shadow & occupied_shadows:
                continue

            placed_here = {
                "color": spec["color"],
                "h": spec["h"],
                "w": spec["w"],
                "r0": r0,
                "c0": c0,
                "body": body,
                "shadow": shadow,
                "shadowed": spec["shadowed"],
            }
            occupied_bodies |= body
            occupied_shadows |= shadow
            break

        if placed_here is None:
            return None
        placed[idx] = placed_here

    return [obj for obj in placed if obj is not None]


def _render_scene(shape: tuple[int, int], bg: int, objects: list[dict], include_all_shadows: bool) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for obj in objects:
        if include_all_shadows or obj["shadowed"]:
            for r, c in obj["shadow"]:
                grid[r, c] = GREY
    for obj in objects:
        for r, c in obj["body"]:
            grid[r, c] = obj["color"]
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    overlap_mode = bool(rng.integers(0, 2))
    dy, dx = (OVERLAP_OFFSETS if overlap_mode else SEPARATED_OFFSETS)[
        int(rng.integers(len(OVERLAP_OFFSETS if overlap_mode else SEPARATED_OFFSETS)))
    ]

    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))
    shape = (h, w)

    bg_choices = [color for color in COLOR_VALUES if color != GREY]
    bg = int(rng.choice(bg_choices))
    object_count = int(rng.integers(*OBJECT_COUNT_RANGE))

    color_choices = [color for color in COLOR_VALUES if color not in (bg, GREY)]
    if len(color_choices) < object_count:
        return None
    colors = list(map(int, rng.choice(color_choices, size=object_count, replace=False)))

    shadowed_count = int(rng.integers(2, object_count))
    shadowed_ids = set(map(int, rng.choice(object_count, size=shadowed_count, replace=False)))

    crop_mode = bool(rng.integers(0, 2))
    crop_idx = int(rng.choice(sorted(shadowed_ids))) if crop_mode else None
    overlap_idx = int(rng.choice(sorted(shadowed_ids))) if overlap_mode else None

    specs: list[dict] = []
    for idx in range(object_count):
        require_overlap = overlap_mode and idx == overlap_idx
        obj_h, obj_w = _sample_size(rng, dy, dx, overlap_mode, require_overlap)
        specs.append(
            {
                "color": colors[idx],
                "h": obj_h,
                "w": obj_w,
                "shadowed": idx in shadowed_ids,
                "must_crop": idx == crop_idx,
                "must_not_crop": idx != crop_idx,
                "need_overlap": require_overlap,
                "forbid_overlap": not overlap_mode,
            }
        )

    objects = _place_objects(rng, shape, dy, dx, specs)
    if objects is None:
        return None

    inp = _render_scene(shape, bg, objects, include_all_shadows=False)
    expected = _render_scene(shape, bg, objects, include_all_shadows=True)
    if np.array_equal(inp, expected):
        return None

    parsed_objects = _extract_objects(inp, bg)
    ranked = _rank_offsets(inp, parsed_objects)
    if not ranked:
        return None
    best_score = ranked[0][0]
    best_offsets = [(cand_dy, cand_dx) for score, cand_dy, cand_dx in ranked if score == best_score]
    if len(best_offsets) != 1 or best_offsets[0] != (dy, dx):
        return None

    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid shadow-completion input")
