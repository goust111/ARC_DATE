import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
TEMPLATE_PALETTE = (1, 2, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 300


def _neighbors(r: int, c: int, connectivity: int):
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _touches_mask(coords: np.ndarray, mask: np.ndarray, connectivity: int = 8) -> bool:
    h, w = mask.shape
    for r, c in coords:
        for nr, nc in _neighbors(int(r), int(c), connectivity):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _largest_component(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    comps = _components(mask, connectivity)
    assert comps, "expected at least one component"
    return max(comps, key=len)


def _catalog_mask(grid: np.ndarray) -> np.ndarray:
    coords = _largest_component(grid == GREY, connectivity=4)
    mask = np.zeros_like(grid, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _extract_templates(grid: np.ndarray):
    catalog = _catalog_mask(grid)
    templates = []
    for coords in _components((grid != BLACK) & (~catalog), connectivity=8):
        if not _touches_mask(coords, catalog, connectivity=8):
            continue
        box = _bbox(coords)
        patch = _crop(grid, box)
        visible = (patch != BLACK) & (patch != GREY)
        colors = tuple(sorted(int(x) for x in np.unique(patch[visible])))
        templates.append(
            {
                "bbox": box,
                "patch": patch,
                "visible": visible,
                "colors": colors,
                "color_count": len(colors),
            }
        )
    assert 2 <= len(templates) <= 3, "expected 2-3 catalog templates"
    return templates, catalog


def _is_filled_rectangle(rel_coords: list[tuple[int, int]]) -> bool:
    if not rel_coords:
        return False
    rows = [r for r, _c in rel_coords]
    cols = [c for _r, c in rel_coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return len(rel_coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _object_match_candidates(
    grid: np.ndarray,
    coords: np.ndarray,
    templates: list[dict],
) -> list[tuple[int, int, int]]:
    matches = []
    H, W = grid.shape
    object_cells = [(int(r), int(c), int(grid[int(r), int(c)])) for r, c in coords]
    for template_index, template in enumerate(templates):
        patch = template["patch"]
        visible = template["visible"]
        h, w = patch.shape
        candidate_anchors = set()
        for r, c, color in object_cells:
            for tr, tc in np.argwhere(visible):
                if int(patch[tr, tc]) == color:
                    candidate_anchors.add((r - int(tr), c - int(tc)))
        for anchor_r, anchor_c in candidate_anchors:
            if anchor_r < 0 or anchor_c < 0 or anchor_r + h > H or anchor_c + w > W:
                continue
            matched = []
            extra = []
            ok = True
            for r, c, color in object_cells:
                tr = r - anchor_r
                tc = c - anchor_c
                if not (0 <= tr < h and 0 <= tc < w):
                    ok = False
                    break
                if visible[tr, tc] and int(patch[tr, tc]) == color:
                    matched.append((r, c))
                else:
                    extra.append((r, c, color))
            if not ok or not matched:
                continue
            if extra:
                extra_colors = {color for _r, _c, color in extra}
                if len(extra_colors) != 1:
                    continue
                rel = [(r - anchor_r, c - anchor_c) for r, c, _color in extra]
                if not _is_filled_rectangle(rel):
                    continue
            matches.append((template_index, int(anchor_r), int(anchor_c)))
    unique = sorted(set(matches))
    return unique


def _stamp_template(out: np.ndarray, template: dict, anchor_r: int, anchor_c: int) -> tuple[int, int, int, int]:
    patch = template["patch"]
    visible = template["visible"]
    h, w = patch.shape
    for tr in range(h):
        for tc in range(w):
            if visible[tr, tc]:
                out[anchor_r + tr, anchor_c + tc] = int(patch[tr, tc])
    return int(anchor_r), int(anchor_c), int(anchor_r + h - 1), int(anchor_c + w - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete workspace sprite fragments from the catalog, then draw green bars below them whose heights equal the completed sprite color counts.

    Input structure:
    - A grey connected catalog strip occupies the upper part of the grid, with 2-3 complete multicolor template sprites on it.
    - The lower black workspace contains 2-3 non-black objects that are either cropped views of a template or template views partially hidden by a single-color rectangle.
    - The workspace may also contain one distractor object that does not match any catalog template.

    Rule:
    - Detect the grey catalog background and extract the complete template sprites touching it.
    - For each workspace object, search all template alignments and keep it only if exactly one template/alignment explains the visible template-colored cells, allowing one solid-color rectangular occluder.
    - Stamp the full matched template back into the workspace, treat grey cells in the template bbox as transparent, then count the distinct non-black, non-grey colors in the completed sprite and draw a solid green bar directly below its bbox with that height and the same width.
    - Leave unmatched distractor objects unchanged and preserve the catalog region.

    Key insight:
    - The bar height depends on the completed sprite rather than the fragment, so template matching and completion must happen before the counting step.
    """
    g = np.asarray(grid, dtype=int)
    templates, catalog = _extract_templates(g)

    out = g.copy()
    workspace_mask = (g != BLACK) & (~catalog)
    matched = []
    for coords in _components(workspace_mask, connectivity=8):
        if _touches_mask(coords, catalog, connectivity=8):
            continue
        candidates = _object_match_candidates(g, coords, templates)
        if len(candidates) != 1:
            continue
        template_index, anchor_r, anchor_c = candidates[0]
        box = _stamp_template(out, templates[template_index], anchor_r, anchor_c)
        matched.append((templates[template_index], box))

    for template, (r0, c0, r1, c1) in matched:
        bar_h = int(template["color_count"])
        out[r1 + 1 : r1 + 1 + bar_h, c0 : c1 + 1] = GREEN

    return out


def _random_template(existing_color_sets: set[tuple[int, ...]]) -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        k = random.randint(2, min(4, h * w))
        colors = sorted(random.sample(TEMPLATE_PALETTE, k))
        if tuple(colors) in existing_color_sets:
            continue
        patch = np.array([random.choice(colors) for _ in range(h * w)], dtype=int).reshape(h, w)
        if len(set(int(x) for x in patch.flat)) != k:
            continue
        if len(set(int(x) for x in patch[0])) == 1 and len(set(int(x) for x in patch[:, 0])) == 1:
            continue
        return patch
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _local_object_coords(patch: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    coords[:, 0] += top
    coords[:, 1] += left
    return coords


def _make_crop_variant(template: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    h, w = template.shape
    choices = []
    for r0 in range(h):
        for c0 in range(w):
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    if r0 == 0 and c0 == 0 and r1 == h - 1 and c1 == w - 1:
                        continue
                    if (r1 - r0 + 1) * (c1 - c0 + 1) < 2:
                        continue
                    choices.append((r0, c0, r1, c1))
    random.shuffle(choices)
    if not choices:
        return None
    r0, c0, r1, c1 = choices[0]
    return template[r0 : r1 + 1, c0 : c1 + 1].copy(), r0, c0


def _make_mask_variant(template: np.ndarray, all_template_colors: set[int]) -> tuple[np.ndarray, int, int] | None:
    h, w = template.shape
    cover_choices = []
    for r0 in range(h):
        for c0 in range(w):
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area >= h * w:
                        continue
                    if h * w - area < 2:
                        continue
                    cover_choices.append((r0, c0, r1, c1))
    random.shuffle(cover_choices)
    available = [c for c in TEMPLATE_PALETTE if c not in set(int(x) for x in template.flat)]
    if not available:
        available = [c for c in TEMPLATE_PALETTE if c not in all_template_colors]
    if not available:
        return None
    for r0, c0, r1, c1 in cover_choices[:40]:
        mask_color = random.choice(available)
        patch = template.copy()
        patch[r0 : r1 + 1, c0 : c1 + 1] = mask_color
        return patch, 0, 0
    return None


def _pick_partial_variant(
    template: np.ndarray,
    method: str,
    anchor_r: int,
    anchor_c: int,
    templates_for_match: list[dict],
    all_template_colors: set[int],
) -> tuple[np.ndarray, int, int] | None:
    makers = [_make_crop_variant] if method == "crop" else []
    if method == "mask":
        makers = [lambda x: _make_mask_variant(x, all_template_colors)]
    if method == "mixed":
        makers = [_make_crop_variant, lambda x: _make_mask_variant(x, all_template_colors)]

    variants = []
    if method == "crop":
        for _ in range(40):
            variant = _make_crop_variant(template)
            if variant is not None:
                variants.append(variant)
    elif method == "mask":
        for _ in range(40):
            variant = _make_mask_variant(template, all_template_colors)
            if variant is not None:
                variants.append(variant)
    else:
        for maker in makers:
            for _ in range(20):
                variant = maker(template)
                if variant is not None:
                    variants.append(variant)
    random.shuffle(variants)

    blank_h = max(anchor_r + template.shape[0] + 5, 20)
    blank_w = max(anchor_c + template.shape[1] + 5, 20)
    scratch = np.zeros((blank_h, blank_w), dtype=int)

    for patch, off_r, off_c in variants:
        coords = _local_object_coords(patch, anchor_r + off_r, anchor_c + off_c)
        scratch.fill(0)
        scratch[coords[:, 0], coords[:, 1]] = patch[patch != BLACK]
        matches = _object_match_candidates(scratch, coords, templates_for_match)
        if len(matches) == 1:
            return patch, off_r, off_c
    return None


def _random_distractor_patch() -> np.ndarray:
    h = random.randint(2, 3)
    w = random.randint(2, 3)
    patch = np.zeros((h, w), dtype=int)
    colors = random.sample(TEMPLATE_PALETTE, random.randint(1, min(3, h * w)))
    shape_type = random.choice(["rect", "lshape", "diag"])
    if shape_type == "rect":
        patch[:, :] = random.choice(colors)
        if len(colors) > 1:
            patch[0, 0] = colors[-1]
    elif shape_type == "lshape":
        patch[:, 0] = random.choice(colors)
        patch[-1, :] = random.choice(colors)
    else:
        for i in range(min(h, w)):
            patch[i, i] = random.choice(colors)
            if i + 1 < w:
                patch[i, i + 1] = random.choice(colors)
    if not ((patch != BLACK).any() and len(_components(patch != BLACK, connectivity=8)) == 1):
        return _random_distractor_patch()
    return patch


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(15, 18)
    W = random.randint(15, 20)
    catalog_h = random.randint(4, 5)

    input_grid = np.zeros((H, W), dtype=int)
    output_grid = np.zeros((H, W), dtype=int)
    input_grid[:catalog_h, :] = GREY
    output_grid[:catalog_h, :] = GREY

    template_count = random.randint(2, 3)
    templates = []
    existing_color_sets = set()
    total_width = 0
    for _ in range(template_count):
        patch = _random_template(existing_color_sets)
        if patch is None:
            return None
        existing_color_sets.add(tuple(sorted(int(x) for x in np.unique(patch))))
        templates.append(patch)
        total_width += patch.shape[1]
    gaps = [random.randint(2, 3) for _ in range(template_count - 1)]
    if total_width + sum(gaps) + 2 > W:
        return None

    x = random.randint(1, W - total_width - sum(gaps) - 1)
    catalog_templates = []
    for i, patch in enumerate(templates):
        h, w = patch.shape
        y = random.randint(0, catalog_h - h)
        input_grid[y : y + h, x : x + w] = patch
        output_grid[y : y + h, x : x + w] = patch
        catalog_templates.append(
            {
                "bbox": (y, x, y + h - 1, x + w - 1),
                "patch": patch.copy(),
                "visible": np.ones_like(patch, dtype=bool),
                "colors": tuple(sorted(int(v) for v in np.unique(patch))),
                "color_count": len(set(int(v) for v in np.unique(patch))),
            }
        )
        if i < len(gaps):
            x += w + gaps[i]

    all_template_colors = set()
    for template in templates:
        all_template_colors.update(int(x) for x in np.unique(template))

    occupied = np.zeros((H, W), dtype=bool)
    occupied[: catalog_h + 1, :] = True

    matched_count = random.randint(2, 3)
    methods = ["crop", "mask"]
    if matched_count == 3:
        methods.append(random.choice(["crop", "mask"]))
    random.shuffle(methods)

    for index in range(matched_count):
        template_index = random.randrange(template_count)
        template = templates[template_index]
        h, w = template.shape
        color_count = len(set(int(v) for v in np.unique(template)))
        placed = False
        for _ in range(80):
            r0 = random.randint(catalog_h + 2, H - h - 4)
            c0 = random.randint(0, W - w)
            if not _can_place(occupied, r0, c0, h + color_count, w, pad=1):
                continue
            variant = _pick_partial_variant(
                template,
                methods[index],
                r0,
                c0,
                catalog_templates,
                all_template_colors,
            )
            if variant is None:
                continue
            patch, off_r, off_c = variant
            output_grid[r0 : r0 + h, c0 : c0 + w] = template
            output_grid[r0 + h : r0 + h + color_count, c0 : c0 + w] = GREEN
            input_grid[r0 + off_r : r0 + off_r + patch.shape[0], c0 + off_c : c0 + off_c + patch.shape[1]] = patch
            _reserve(occupied, r0, c0, h + color_count, w)
            placed = True
            break
        if not placed:
            return None

    if random.random() < 0.7:
        for _ in range(60):
            patch = _random_distractor_patch()
            h, w = patch.shape
            r0 = random.randint(catalog_h + 2, H - h - 1)
            c0 = random.randint(0, W - w)
            if not _can_place(occupied, r0, c0, h, w, pad=1):
                continue
            scratch = np.zeros((max(H, 20), max(W, 20)), dtype=int)
            coords = _local_object_coords(patch, r0, c0)
            scratch[coords[:, 0], coords[:, 1]] = patch[patch != BLACK]
            if _object_match_candidates(scratch, coords, catalog_templates):
                continue
            input_grid[r0 : r0 + h, c0 : c0 + w] = patch
            output_grid[r0 : r0 + h, c0 : c0 + w] = patch
            _reserve(occupied, r0, c0, h, w)
            break

    try:
        produced = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
