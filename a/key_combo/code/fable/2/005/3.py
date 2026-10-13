import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, paste


BLACK = Color.BLACK
GREY = Color.GREY

_VARIANTS = tuple(range(8))
_GEN_ATTEMPTS = 400
_SHAPE_ATTEMPTS = 500
_PLACE_ATTEMPTS = 300


def _apply_variant(arr: np.ndarray, variant: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    if int(variant) >= 4:
        out = np.fliplr(out)
    return np.rot90(out, int(variant) % 4)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), tuple(int(x) for x in m.astype(np.uint8).ravel())


def _variant_records(patch: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    seen = {}
    for variant in _VARIANTS:
        patch_v = _apply_variant(patch, variant)
        mask_v = patch_v != BLACK
        key = _mask_key(mask_v)
        seen.setdefault(key, (variant, patch_v, mask_v))
    return list(seen.values())


def _find_separator_row(grid: np.ndarray) -> int:
    candidates = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BLACK and np.all(grid[r] == grid[r, 0])
    ]
    assert len(candidates) == 1, "expected one full-width non-black separator row"
    return int(candidates[0])


def _split_panels(grid: np.ndarray) -> tuple[int, int, np.ndarray, int, np.ndarray]:
    sep_r = _find_separator_row(grid)
    top = grid[:sep_r]
    bottom = grid[sep_r + 1 :]
    top_has_grey = bool(np.any(top == GREY))
    bottom_has_grey = bool(np.any(bottom == GREY))
    assert top_has_grey ^ bottom_has_grey, "expected exactly one grey work panel"
    if top_has_grey:
        return sep_r, 0, top, sep_r + 1, bottom
    return sep_r, sep_r + 1, bottom, 0, top


def _border_reachable_black(panel: np.ndarray) -> np.ndarray:
    h, w = panel.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if panel[r, c] == BLACK and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if panel[r, c] == BLACK and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and panel[nr, nc] == BLACK and not reachable[nr, nc]:
                reachable[nr, nc] = True
                queue.append((nr, nc))
    return reachable


def _extract_holes(panel: np.ndarray) -> list[dict]:
    reachable = _border_reachable_black(panel)
    holes_mask = (panel == BLACK) & (~reachable)
    holes = []
    for coords in components_mask(holes_mask, connectivity=4):
        box = bbox_coords(coords)
        local_mask = crop_bbox(holes_mask, box)
        holes.append(
            {
                "bbox": box,
                "coords": coords,
                "mask": local_mask,
            }
        )
    holes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return holes


def _extract_templates(panel: np.ndarray) -> list[dict]:
    templates = []
    for coords in components_mask(panel != BLACK, connectivity=4):
        box = bbox_coords(coords)
        patch = crop_bbox(panel, box)
        templates.append(
            {
                "bbox": box,
                "patch": patch,
                "mask": patch != BLACK,
                "variants": _variant_records(patch),
            }
        )
    templates.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return templates


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep_r, work_r0, work_panel, _catalog_r0, catalog_panel = _split_panels(g)
    assert np.any(work_panel == GREY), "work panel must contain grey blocks"
    assert not np.any(catalog_panel == GREY), "catalog panel must not contain grey"

    holes = _extract_holes(work_panel)
    templates = _extract_templates(catalog_panel)
    assert holes, "expected at least one enclosed hole"
    assert len(templates) > len(holes), "expected unmatched catalog distractors"

    out = g.copy()
    for hole in holes:
        matches = []
        for template_index, template in enumerate(templates):
            for variant, patch_v, mask_v in template["variants"]:
                if mask_v.shape == hole["mask"].shape and np.array_equal(mask_v, hole["mask"]):
                    matches.append((template_index, variant, patch_v))
        assert len(matches) == 1, "each hole must match exactly one template pose"
        _template_index, _variant, patch_v = matches[0]
        r0, c0, _r1, _c1 = hole["bbox"]
        paste(out, patch_v, work_r0 + r0, c0, transparent=BLACK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey block's enclosed black hole to a catalog silhouette and stamp the same multicolor template pose into the hole.

    Input structure:
    - One full-width separator row splits the grid into a grey work panel and a multicolor catalog panel.
    - The work panel contains several disconnected grey blocks; each block encloses exactly one black hole unreachable from the panel border.
    - The catalog panel contains several solid multicolor template sprites on black, with distinct silhouettes and some unmatched distractors.

    Rule:
    - Flood from the work-panel border through black cells to mark the exterior background.
    - Treat every remaining 4-connected black component as a hole, normalize its mask, and compare it against every catalog silhouette under all rotations and reflections.
    - For the unique matching template and pose, stamp the full multicolor template transformed by that same pose into the hole.
    - Leave the grey blocks, separator row, and catalog panel unchanged.

    Key insight:
    - The decisive shape is negative space: the black cavity inside each grey block identifies both which catalog template to use and which orientation that template must take.
    """
    return _solve(np.asarray(grid, dtype=int))


def _shape_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return bool(np.any(empty & (~seen)))


def _canonical_shape_key(mask: np.ndarray) -> tuple:
    return min(_mask_key(_apply_variant(mask, variant)) for variant in _VARIANTS)


def _random_connected_shape(rng: np.random.Generator, used_keys: set[tuple]) -> np.ndarray | None:
    for _ in range(_SHAPE_ATTEMPTS):
        target = int(rng.integers(5, 8))
        cells = {(0, 0)}
        while len(cells) < target:
            base_r, base_c = list(cells)[int(rng.integers(len(cells)))]
            candidates = []
            for nr, nc in ((base_r - 1, base_c), (base_r + 1, base_c), (base_r, base_c - 1), (base_r, base_c + 1)):
                if (nr, nc) not in cells:
                    candidates.append((nr, nc))
            if not candidates:
                break
            cells.add(candidates[int(rng.integers(len(candidates)))])

        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        h, w = coords.max(axis=0) + 1
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue

        mask = np.zeros((int(h), int(w)), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if _shape_has_hole(mask):
            continue
        if len({_mask_key(_apply_variant(mask, variant)) for variant in _VARIANTS}) != 8:
            continue
        key = _canonical_shape_key(mask)
        if key in used_keys:
            continue
        used_keys.add(key)
        return mask
    return None


def _random_colored_template(mask: np.ndarray, rng: np.random.Generator, palette: list[int]) -> np.ndarray:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    rng.shuffle(coords)
    color_count = min(len(coords), int(rng.integers(2, 5)))
    colors = list(map(int, rng.choice(palette, size=color_count, replace=False)))
    patch = np.zeros(mask.shape, dtype=int)
    accent_count = min(len(coords), max(1, color_count - 1))
    for index, (r, c) in enumerate(coords):
        if index < accent_count:
            patch[r, c] = colors[index]
        else:
            patch[r, c] = colors[int(rng.integers(color_count))]
    if len({int(patch[r, c]) for r, c in coords}) < 2:
        patch[coords[0]] = colors[0]
        patch[coords[-1]] = colors[1]
    return patch


def _place_bbox(occupied: np.ndarray, box_h: int, box_w: int, rng: np.random.Generator, margin: int, gap: int) -> tuple[int, int] | None:
    h, w = occupied.shape
    if box_h + 2 * margin > h or box_w + 2 * margin > w:
        return None
    for _ in range(_PLACE_ATTEMPTS):
        r = int(rng.integers(margin, h - box_h - margin + 1))
        c = int(rng.integers(margin, w - box_w - margin + 1))
        r0 = max(0, r - gap)
        c0 = max(0, c - gap)
        r1 = min(h, r + box_h + gap)
        c1 = min(w, c + box_w + gap)
        if np.any(occupied[r0:r1, c0:c1]):
            continue
        occupied[r:r + box_h, c:c + box_w] = True
        return r, c
    return None


def _build_work_block(hole_mask: np.ndarray, border: int) -> np.ndarray:
    h, w = hole_mask.shape
    block = np.full((h + 2 * border, w + 2 * border), GREY, dtype=int)
    block[border:border + h, border:border + w][hole_mask] = BLACK
    return block


def _match_variants_for_generation(templates: list[dict], holes: list[np.ndarray]) -> list[tuple[int, int]]:
    matches = []
    used = set()
    for hole_mask in holes:
        options = []
        for template_index, template in enumerate(templates):
            for variant, _patch_v, mask_v in template["variants"]:
                if mask_v.shape == hole_mask.shape and np.array_equal(mask_v, hole_mask):
                    options.append((template_index, variant))
        assert len(options) == 1, "generated hole must have a unique template pose"
        template_index, variant = options[0]
        assert template_index not in used, "generated holes must use distinct templates"
        used.add(template_index)
        matches.append((template_index, variant))
    return matches


def _validate_generated(grid: np.ndarray, expected: np.ndarray, matched_count: int, template_count: int) -> None:
    sep_r, work_r0, work_panel, _catalog_r0, catalog_panel = _split_panels(grid)
    assert 0 < sep_r < grid.shape[0] - 1
    assert np.all(grid[sep_r] == grid[sep_r, 0])
    assert set(np.unique(work_panel)).issubset({BLACK, GREY})
    templates = _extract_templates(catalog_panel)
    holes = _extract_holes(work_panel)
    assert len(templates) == template_count
    assert len(holes) == matched_count
    assert len(templates) > len(holes)

    grey_components = components_mask(work_panel == GREY, connectivity=4)
    assert len(grey_components) == matched_count
    grey_boxes = [bbox_coords(coords) for coords in grey_components]
    for hole in holes:
        hr0, hc0, hr1, hc1 = hole["bbox"]
        container_count = sum(
            1
            for gr0, gc0, gr1, gc1 in grey_boxes
            if gr0 < hr0 and gc0 < hc0 and gr1 > hr1 and gc1 > hc1
        )
        assert container_count == 1, "each hole must lie inside exactly one grey block"

    matches = []
    for hole in holes:
        options = []
        for template_index, template in enumerate(templates):
            for variant, _patch_v, mask_v in template["variants"]:
                if mask_v.shape == hole["mask"].shape and np.array_equal(mask_v, hole["mask"]):
                    options.append((template_index, variant))
        assert len(options) == 1
        matches.append(options[0])
    assert len({template_index for template_index, _variant in matches}) == matched_count
    assert np.array_equal(main(grid), expected)


def _generate_instance() -> tuple[np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng()
    width = int(rng.integers(22, 30))
    catalog_h = int(rng.integers(7, 10))
    work_h = int(rng.integers(13, 20))
    if catalog_h + work_h + 1 > 30:
        work_h = 29 - catalog_h
    matched_count = int(rng.integers(2, 4))
    distractor_count = int(rng.integers(1, 3))
    template_count = matched_count + distractor_count

    used_shape_keys = set()
    templates = []
    for _ in range(template_count):
        shape = _random_connected_shape(rng, used_shape_keys)
        if shape is None:
            raise RuntimeError("failed to generate enough unique asymmetric shapes")
        templates.append({"mask": shape})

    catalog_side_top = bool(rng.integers(2))
    sep_color = int(rng.choice([1, 2, 3, 4, 6, 7, 8, 9]))
    palette = [color for color in [1, 2, 3, 4, 6, 7, 8, 9] if color != sep_color]
    for template in templates:
        template["patch"] = _random_colored_template(template["mask"], rng, palette)
        template["variants"] = _variant_records(template["patch"])

    if matched_count >= 2:
        matched_indices = list(map(int, rng.choice(template_count, size=matched_count, replace=False)))
    else:
        matched_indices = [int(rng.integers(template_count))]

    transform_choices = list(range(matched_count))
    rng.shuffle(transform_choices)
    chosen_variants = []
    if matched_count >= 2:
        chosen_variants.append(int(rng.integers(0, 4)))
        chosen_variants.append(int(rng.integers(4, 8)))
        while len(chosen_variants) < matched_count:
            chosen_variants.append(int(rng.integers(0, 8)))
        rng.shuffle(chosen_variants)
    else:
        chosen_variants = [int(rng.integers(0, 8))]

    holes = []
    matched_specs = []
    for template_index, variant in zip(matched_indices, chosen_variants):
        hole_mask = _apply_variant(templates[template_index]["mask"], variant)
        matched_specs.append((template_index, variant))
        holes.append(hole_mask)

    _match_variants_for_generation(templates, holes)

    catalog_panel = np.zeros((catalog_h, width), dtype=int)
    catalog_occ = np.zeros((catalog_h, width), dtype=bool)
    catalog_order = sorted(range(template_count), key=lambda idx: templates[idx]["patch"].size, reverse=True)
    for template_index in catalog_order:
        patch = templates[template_index]["patch"]
        pos = _place_bbox(catalog_occ, patch.shape[0], patch.shape[1], rng, margin=1, gap=1)
        if pos is None:
            raise RuntimeError("failed to place catalog template")
        r, c = pos
        paste(catalog_panel, patch, r, c, transparent=BLACK)
        templates[template_index]["catalog_pos"] = (r, c)

    work_panel = np.zeros((work_h, width), dtype=int)
    work_occ = np.zeros((work_h, width), dtype=bool)
    expected_panel = work_panel.copy()
    work_order = sorted(
        range(matched_count),
        key=lambda idx: holes[idx].shape[0] * holes[idx].shape[1],
        reverse=True,
    )
    for hole_index in work_order:
        hole_mask = holes[hole_index]
        border = int(rng.integers(1, 3))
        block = _build_work_block(hole_mask, border)
        pos = _place_bbox(work_occ, block.shape[0], block.shape[1], rng, margin=1, gap=2)
        if pos is None:
            raise RuntimeError("failed to place grey work block")
        r, c = pos
        paste(work_panel, block, r, c)
        template_index, variant = matched_specs[hole_index]
        stamped = _apply_variant(templates[template_index]["patch"], variant)
        paste(expected_panel, block, r, c)
        paste(expected_panel, stamped, r + border, c + border, transparent=BLACK)

    separator = np.full((1, width), sep_color, dtype=int)
    if catalog_side_top:
        grid = np.vstack([catalog_panel, separator, work_panel])
        expected = np.vstack([catalog_panel, separator, expected_panel])
    else:
        grid = np.vstack([work_panel, separator, catalog_panel])
        expected = np.vstack([expected_panel, separator, catalog_panel])

    return grid, expected, matched_count, template_count


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        try:
            grid, expected, matched_count, template_count = _generate_instance()
        except RuntimeError:
            continue
        try:
            _validate_generated(grid, expected, matched_count, template_count)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid negative-space matching instance")
