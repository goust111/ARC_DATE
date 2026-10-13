from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_GRID_MIN = 20
GEN_GRID_MAX = 30
GEN_OBJECT_COUNT_MIN = 2
GEN_OBJECT_COUNT_MAX = 4
GEN_HALF_SIZE_MIN = 2
GEN_HALF_SIZE_MAX = 4
GEN_HALF_AREA_MIN = 3
GEN_PLACEMENT_GAP = 1
GEN_RETRY_LIMIT = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalized_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _normalized_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalized_key_from_coords(np.argwhere(mask))


def _component_mask(coords: np.ndarray) -> np.ndarray:
    shifted = coords - coords.min(axis=0)
    h, w = shifted.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "mask": _component_mask(coords),
                }
            )
    return comps


def _mirror_complete(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return np.concatenate([np.fliplr(mask), mask], axis=1)
    if side == "right":
        return np.concatenate([mask, np.fliplr(mask)], axis=1)
    if side == "top":
        return np.concatenate([np.flipud(mask), mask], axis=0)
    if side == "bottom":
        return np.concatenate([mask, np.flipud(mask)], axis=0)
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored half-template by edge reflection, then recolor the same-shaped grey placeholder.

    Input structure:
    - A black background contains several grey placeholder objects and several non-grey colored template objects.
    - Each colored template is only one half of a larger axis-symmetric object.
    - Each placeholder shows one full object shape, and template/placeholder objects are 4-connected and separated.

    Rule:
    - For every colored template, crop its tight bounding box and consider the four possible completions formed by mirroring it outward across the top, bottom, left, or right bbox edge.
    - Exactly one of those completed shapes matches one grey placeholder by normalized 4-connected shape equality under translation only, with no rotation.
    - Recolor the entire matched grey placeholder to the template's color, leaving templates and background unchanged.

    Key insight:
    - The visible colored objects are incomplete, so direct shape matching fails; the correct match appears only after mentally mirroring the half-template across the correct bbox edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert GREY in colors, "expected grey placeholders"
    template_colors = [color for color in colors if color != GREY]
    assert template_colors, "expected colored templates"

    placeholders = _components_of_color(g, GREY)
    assert placeholders, "expected at least one grey placeholder"
    placeholder_by_key: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}
    for comp in placeholders:
        key = _normalized_key_from_coords(comp["coords"])
        assert key not in placeholder_by_key, "placeholder shapes must be unique"
        placeholder_by_key[key] = comp

    out = g.copy()
    matched_placeholder_keys: set[tuple[tuple[int, int], ...]] = set()

    for color in template_colors:
        for comp in _components_of_color(g, color):
            hits: list[tuple[str, tuple[tuple[int, int], ...]]] = []
            for side in SIDES:
                full_mask = _mirror_complete(comp["mask"], side)
                key = _normalized_key_from_mask(full_mask)
                if key in placeholder_by_key:
                    hits.append((side, key))
            assert len(hits) == 1, "each template must match exactly one placeholder after completion"
            _, placeholder_key = hits[0]
            assert placeholder_key not in matched_placeholder_keys, "placeholder already claimed by another template"
            matched_placeholder_keys.add(placeholder_key)
            coords = placeholder_by_key[placeholder_key]["coords"]
            out[coords[:, 0], coords[:, 1]] = int(color)

    assert len(matched_placeholder_keys) == len(placeholders), "every grey placeholder must be recolored"
    assert np.any(out != g), "expected at least one recolor"
    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < h and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < w and mask[r, c + 1])
            ):
                cells.append((r, c))
    return cells


def _grow_connected_mask(h: int, w: int, count: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < count:
        choices = _frontier(mask)
        if not choices:
            break
        mask[random.choice(choices)] = True
    return mask


def _touches_side(mask: np.ndarray, side: str) -> bool:
    if side == "top":
        return bool(mask[0, :].any())
    if side == "bottom":
        return bool(mask[-1, :].any())
    if side == "left":
        return bool(mask[:, 0].any())
    if side == "right":
        return bool(mask[:, -1].any())
    raise ValueError(f"unknown side: {side}")


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(coords[0])}
    stack = [tuple(coords[0])]
    cells = {tuple(rc) for rc in coords.tolist()}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _random_half_mask(side: str) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(GEN_HALF_SIZE_MIN, GEN_HALF_SIZE_MAX)
        w = random.randint(GEN_HALF_SIZE_MIN, GEN_HALF_SIZE_MAX)
        max_area = h * w - 1
        if max_area < GEN_HALF_AREA_MIN:
            continue
        area = random.randint(GEN_HALF_AREA_MIN, max_area)
        mask = _grow_connected_mask(h, w, area)
        if int(mask.sum()) != area:
            continue
        if not _touches_all_edges(mask):
            continue
        if not _touches_side(mask, side):
            continue
        if not _is_connected(mask):
            continue
        variant_keys = [_normalized_key_from_mask(_mirror_complete(mask, candidate)) for candidate in SIDES]
        if len(set(variant_keys)) != 4:
            continue
        intended = _mirror_complete(mask, side)
        if not _is_connected(intended):
            continue
        return mask
    return None


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _count_true(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _place_objects(
    rows: int,
    cols: int,
    objects: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    occupied_boxes = np.zeros((rows, cols), dtype=bool)
    placed: list[dict[str, object]] = []

    ordered = sorted(objects, key=lambda obj: obj["patch"].shape[0] * obj["patch"].shape[1], reverse=True)
    for obj in ordered:
        patch = obj["patch"]
        h, w = patch.shape
        options: list[tuple[int, int]] = []
        for r0 in range(rows - h + 1):
            for c0 in range(cols - w + 1):
                rr0 = max(0, r0 - GEN_PLACEMENT_GAP)
                cc0 = max(0, c0 - GEN_PLACEMENT_GAP)
                rr1 = min(rows, r0 + h + GEN_PLACEMENT_GAP)
                cc1 = min(cols, c0 + w + GEN_PLACEMENT_GAP)
                if not occupied_boxes[rr0:rr1, cc0:cc1].any():
                    options.append((r0, c0))
        if not options:
            return None

        r0, c0 = random.choice(options)
        occupied_boxes[r0 : r0 + h, c0 : c0 + w] = True
        placed.append({**obj, "r0": r0, "c0": c0})
    return placed


def _render_input_and_output(
    rows: int,
    cols: int,
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray] | None:
    objects: list[dict[str, object]] = []
    for spec in specs:
        half_patch = _shape_patch(spec["half_mask"], spec["color"])
        placeholder_patch = _shape_patch(spec["full_mask"], GREY)
        recolored_patch = _shape_patch(spec["full_mask"], spec["color"])
        objects.append(
            {
                "kind": "template",
                "spec_id": spec["spec_id"],
                "patch": half_patch,
                "output_patch": half_patch,
            }
        )
        objects.append(
            {
                "kind": "placeholder",
                "spec_id": spec["spec_id"],
                "patch": placeholder_patch,
                "output_patch": recolored_patch,
            }
        )

    placed = _place_objects(rows, cols, objects)
    if placed is None:
        return None

    inp = np.zeros((rows, cols), dtype=int)
    out = np.zeros((rows, cols), dtype=int)
    for obj in placed:
        patch = obj["patch"]
        output_patch = obj["output_patch"]
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        mask = patch != BLACK
        inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
        out[r0 : r0 + output_patch.shape[0], c0 : c0 + output_patch.shape[1]][output_patch != BLACK] = output_patch[
            output_patch != BLACK
        ]
    return inp, out


def _specs_have_unique_semantics(specs: list[dict[str, object]]) -> bool:
    intended_keys = {spec["full_key"] for spec in specs}
    if len(intended_keys) != len(specs):
        return False

    for spec in specs:
        hit_count = 0
        for side in SIDES:
            key = _normalized_key_from_mask(_mirror_complete(spec["half_mask"], side))
            if key in intended_keys:
                hit_count += 1
        if hit_count != 1:
            return False

    areas = [_count_true(spec["full_mask"]) for spec in specs]
    bboxes = [tuple(spec["full_mask"].shape) for spec in specs]
    half_boxes = [tuple(spec["half_mask"].shape) for spec in specs]
    return (
        len(set(areas)) < len(areas)
        or len(set(bboxes)) < len(bboxes)
        or len(set(half_boxes)) < len(half_boxes)
    )


def _scene_matches_rule(inp: np.ndarray, expected: np.ndarray) -> bool:
    bg = _get_bg(inp)
    if bg != BLACK:
        return False

    colors = [int(v) for v in np.unique(inp) if int(v) != bg]
    if GREY not in colors:
        return False
    template_colors = [color for color in colors if color != GREY]
    if not template_colors:
        return False

    placeholders = _components_of_color(inp, GREY)
    if len(placeholders) != len(template_colors):
        return False

    placeholder_by_key: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}
    for comp in placeholders:
        key = _normalized_key_from_coords(comp["coords"])
        if key in placeholder_by_key:
            return False
        placeholder_by_key[key] = comp

    rebuilt = inp.copy()
    matched: set[tuple[tuple[int, int], ...]] = set()
    for color in template_colors:
        comps = _components_of_color(inp, color)
        if len(comps) != 1:
            return False
        comp = comps[0]
        hits: list[tuple[tuple[int, int], ...]] = []
        for side in SIDES:
            key = _normalized_key_from_mask(_mirror_complete(comp["mask"], side))
            if key in placeholder_by_key:
                hits.append(key)
        if len(hits) != 1:
            return False
        key = hits[0]
        if key in matched:
            return False
        matched.add(key)
        coords = placeholder_by_key[key]["coords"]
        rebuilt[coords[:, 0], coords[:, 1]] = int(color)

    if len(matched) != len(placeholders):
        return False
    if np.any(rebuilt == GREY):
        return False
    if not np.array_equal(rebuilt, expected):
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _choose_sides(k: int) -> list[str]:
    base = random.sample(list(SIDES), k=min(k, len(SIDES)))
    while len(base) < k:
        base.append(random.choice(SIDES))
    random.shuffle(base)
    return base


def _build_specs() -> list[dict[str, object]] | None:
    k = random.randint(GEN_OBJECT_COUNT_MIN, GEN_OBJECT_COUNT_MAX)
    sides = _choose_sides(k)
    colors = random.sample(TEMPLATE_COLORS, k)

    specs: list[dict[str, object]] = []
    for spec_id, (side, color) in enumerate(zip(sides, colors)):
        half_mask = _random_half_mask(side)
        if half_mask is None:
            return None
        full_mask = _mirror_complete(half_mask, side)
        specs.append(
            {
                "spec_id": spec_id,
                "side": side,
                "color": int(color),
                "half_mask": half_mask,
                "full_mask": full_mask,
                "full_key": _normalized_key_from_mask(full_mask),
            }
        )

    if not _specs_have_unique_semantics(specs):
        return None
    return specs


def _attempt() -> np.ndarray | None:
    specs = _build_specs()
    if specs is None:
        return None

    rows = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    cols = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    rendered = _render_input_and_output(rows, cols, specs)
    if rendered is None:
        return None
    inp, expected = rendered

    if not _scene_matches_rule(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-template ARC input")
