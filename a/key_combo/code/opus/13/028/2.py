import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RANK_COLORS = (RED, GREEN, YELLOW)
AREA_CHOICES = (4, 5, 6, 7, 8)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_key(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    shifted = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _key_to_mask(key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in key) + 1
    w = max(c for _, c in key) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[r, c] = True
    return mask


def _shape_variants(key: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _key_to_mask(key)
    variants: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    cur = mask
    for _ in range(4):
        for candidate in (cur, np.fliplr(cur)):
            variant = _coords_key(np.argwhere(candidate))
            if variant not in seen:
                seen.add(variant)
                variants.append(variant)
        cur = np.rot90(cur)
    return variants


def _canonical_shape(coords) -> tuple[tuple[int, int], ...]:
    key = _coords_key(coords)
    return min(_shape_variants(key))


def _is_connected_mask(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _build_shape_library() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(4) for c in range(4)]
    library: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for area in AREA_CHOICES:
        shapes = []
        seen = set()
        for combo in itertools.combinations(cells, area):
            key = _coords_key(combo)
            mask = _key_to_mask(key)
            hh, ww = mask.shape
            if hh < 2 or ww < 2:
                continue
            if hh * ww == area:
                continue
            if not _is_connected_mask(mask):
                continue
            canon = _canonical_shape(combo)
            if canon in seen:
                continue
            if len(_shape_variants(canon)) < 2:
                continue
            seen.add(canon)
            shapes.append(canon)
        if not shapes:
            raise RuntimeError(f"no usable shapes for area {area}")
        library[area] = shapes
    return library


SHAPE_LIBRARY = _build_shape_library()
SHAPE_VARIANTS = {
    shape: _shape_variants(shape)
    for shapes in SHAPE_LIBRARY.values()
    for shape in shapes
}


def _analyze_grid(grid: np.ndarray):
    comps = _components(grid != BLACK)
    objects = []
    for coords in comps:
        values = {int(grid[r, c]) for r, c in coords}
        if len(values) != 1:
            raise AssertionError("each object must be monochromatic")
        color = values.pop()
        bbox = _bbox_from_coords(coords)
        objects.append(
            {
                "coords": coords,
                "color": color,
                "bbox": bbox,
                "area": int(coords.shape[0]),
                "shape": _canonical_shape(coords),
            }
        )
    templates = [obj for obj in objects if obj["color"] == GREY]
    placeholders = [obj for obj in objects if obj["color"] == BLUE]
    return objects, templates, placeholders


def _patch_from_key(key: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    mask = _key_to_mask(key)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _separated(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        b1[2] + gap < b2[0]
        or b2[2] + gap < b1[0]
        or b1[3] + gap < b2[1]
        or b2[3] + gap < b1[1]
    )


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    starts = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    rng.shuffle(starts)
    for r, c in starts:
        bbox = (r, c, r + h - 1, c + w - 1)
        if not all(_separated(bbox, other, gap=1) for other in bboxes):
            continue
        grid[r : r + h, c : c + w][patch != BLACK] = patch[patch != BLACK]
        bboxes.append(bbox)
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor blue placeholders by the area rank of their matched grey templates.

    Input structure:
    - A black background contains exactly three grey template objects and exactly
      three blue placeholder objects, all 4-connected and pairwise separated.
    - Every object is a small irregular polyomino with 4-8 pixels and bounding
      box at most 4x4.
    - Each blue placeholder matches exactly one grey template by normalized shape
      after allowing rotations and mirror reflections.

    Rule:
    - Find all 4-connected foreground components and split them into grey
      templates and blue placeholders.
    - Match each placeholder to the unique template with the same normalized
      shape up to the 8 dihedral transforms, then read that template's area.
    - Rank the three placeholders by matched template area from smallest to
      largest and recolor them red, green, and yellow respectively.

    Key insight:
    - Shape matching identifies which template each placeholder refers to, but
      the final color comes from the matched template's global area rank.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    objects, templates, placeholders = _analyze_grid(g)
    if len(objects) != 6 or len(templates) != 3 or len(placeholders) != 3:
        raise AssertionError("expected exactly 3 templates and 3 placeholders")

    shape_to_area: dict[tuple[tuple[int, int], ...], int] = {}
    for obj in templates:
        if not (4 <= obj["area"] <= 8):
            raise AssertionError("template area out of range")
        if obj["shape"] in shape_to_area:
            raise AssertionError("template shapes must be unique up to symmetry")
        shape_to_area[obj["shape"]] = int(obj["area"])
    if len(set(shape_to_area.values())) != 3:
        raise AssertionError("template areas must be distinct")

    ranked = []
    for obj in placeholders:
        area = shape_to_area.get(obj["shape"])
        if area is None:
            raise AssertionError("placeholder has no unique template match")
        ranked.append((area, obj))
    if len({area for area, _ in ranked}) != 3:
        raise AssertionError("placeholder matches must inherit distinct template areas")

    out = g.copy()
    for new_color, (_area, obj) in zip(RANK_COLORS, sorted(ranked, key=lambda x: x[0])):
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid_h = int(rng.integers(18, 25))
        grid_w = int(rng.integers(18, 25))
        grid = np.zeros((grid_h, grid_w), dtype=int)

        areas = sorted(rng.choice(AREA_CHOICES, size=3, replace=False).tolist())
        chosen_shapes = [
            SHAPE_LIBRARY[area][int(rng.integers(len(SHAPE_LIBRARY[area])))]
            for area in areas
        ]

        template_keys = []
        placeholder_keys = []
        orientation_diff = False
        for shape in chosen_shapes:
            variants = SHAPE_VARIANTS[shape]
            template_key = variants[int(rng.integers(len(variants)))]
            placeholder_key = variants[int(rng.integers(len(variants)))]
            if placeholder_key != template_key:
                orientation_diff = True
            template_keys.append(template_key)
            placeholder_keys.append(placeholder_key)

        if not orientation_diff:
            idx = int(rng.integers(3))
            variants = SHAPE_VARIANTS[chosen_shapes[idx]]
            alt = [variant for variant in variants if variant != template_keys[idx]]
            if alt:
                placeholder_keys[idx] = alt[int(rng.integers(len(alt)))]

        objects = []
        for template_key, placeholder_key in zip(template_keys, placeholder_keys):
            objects.append(_patch_from_key(template_key, GREY))
            objects.append(_patch_from_key(placeholder_key, BLUE))
        rng.shuffle(objects)

        bboxes: list[tuple[int, int, int, int]] = []
        ok = True
        for patch in objects:
            if not _place_patch(grid, patch, bboxes, rng):
                ok = False
                break
        if not ok:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        out_objects, out_templates, _out_placeholders = _analyze_grid(out)
        recolored = [obj for obj in out_objects if obj["color"] in RANK_COLORS]
        if len(out_objects) != 6 or len(out_templates) != 3 or len(recolored) != 3:
            continue
        if {obj["color"] for obj in recolored} != set(RANK_COLORS):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
