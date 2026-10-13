from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8
TEMPLATE_H = 4
TEMPLATE_W = 4
GEN_RETRIES = 400
PALETTE = [1, 2, 3, 4, 6, 7, 9]


# Base family: 4 templates = left-feature x right-feature.
_CORE_COORDS = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 2),
)
_LEFT_FEATURES = {0: ((2, 0),), 1: ((0, 0),)}
_RIGHT_FEATURES = {0: ((1, 3),), 1: ((3, 3),)}
_LEFT_VISIBLE_SHARED = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 2))
_RIGHT_VISIBLE_SHARED = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3))
_COORD_COLORS = {
    (0, 1): 0,
    (1, 0): 1,
    (1, 1): 2,
    (1, 2): 0,
    (2, 1): 1,
    (2, 2): 2,
    (2, 3): 0,
    (3, 2): 1,
    (2, 0): 2,
    (0, 0): 2,
    (1, 3): 1,
    (3, 3): 1,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete paired partial sprites by selecting the only catalog template consistent with a cross-panel mirror constraint.

    Input structure:
    - A grey catalog strip at the top contains four full multicolor templates on a grey background.
    - Below the strip, a solid teal separator column splits the remaining area into equal black left and right panels.
    - Each panel contains two partial multicolor sprites; right-panel sprites are horizontal-mirror counterparts of the left-panel placements.

    Rule:
    - Extract the four catalog templates from the top strip.
    - For each left-panel partial sprite, enumerate every catalog template and anchor where all visible cells are covered with matching colors.
    - Mirror each candidate template across the separator into the right panel and keep only candidates whose mirrored template also covers the visible right-panel cells at the corresponding mirrored location.
    - The mirror consistency leaves exactly one valid template per position pair; stamp that full template into the left panel and its horizontal mirror into the right panel.
    - Remove the catalog strip and separator column from the output.

    Key insight:
    - Neither side identifies the template alone: the left partial constrains one subset of template structure, the right mirrored partial constrains another, and only their joint mirror-consistent intersection is unique.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    catalog_h = _catalog_height(g)
    catalog = g[:catalog_h, :]
    lower = g[catalog_h:, :]
    sep_col = _separator_col(lower)
    left = lower[:, :sep_col].copy()
    right = lower[:, sep_col + 1 :].copy()
    assert left.shape == right.shape and left.size > 0

    templates = _extract_catalog_templates(catalog)
    assert len(templates) == 4, "expected four catalog templates"
    mirrored = [np.fliplr(t) for t in templates]

    left_objects = _components(left != BLACK)
    assert len(left_objects) == 2, "expected two left-panel partial sprites"

    out_left = left.copy()
    out_right = right.copy()
    for obj in left_objects:
        candidates = []
        for tid, template in enumerate(templates):
            for anchor in _candidate_anchors(left, obj, template):
                mirror_anchor = _mirrored_anchor(anchor, template.shape[1], left.shape[1])
                if _region_matches_partial(right, mirrored[tid], mirror_anchor):
                    candidates.append((tid, anchor))
        uniq = {(tid, anchor) for tid, anchor in candidates}
        assert len(uniq) == 1, f"mirror constraint should disambiguate uniquely, got {uniq}"
        tid, anchor = uniq.pop()
        _stamp(out_left, templates[tid], anchor)
        _stamp(out_right, mirrored[tid], _mirrored_anchor(anchor, templates[tid].shape[1], left.shape[1]))

    return np.concatenate([out_left, out_right], axis=1)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-disambiguation ARC input")


def _catalog_height(grid: np.ndarray) -> int:
    for r in range(grid.shape[0]):
        if np.any(grid[r] == BLACK):
            assert r > 0, "catalog strip must be non-empty"
            return r
    raise AssertionError("failed to locate the catalog strip")


def _separator_col(lower: np.ndarray) -> int:
    candidates = [
        c
        for c in range(lower.shape[1])
        if lower[0, c] != BLACK and np.all(lower[:, c] == lower[0, c])
    ]
    assert len(candidates) == 1, f"expected a unique separator column, got {candidates}"
    return int(candidates[0])


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
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_catalog_templates(catalog: np.ndarray) -> list[np.ndarray]:
    comps = _components(catalog != GREY)
    templates = []
    for coords in comps:
        box = _bbox(coords)
        patch = _crop(catalog, box)
        templates.append(np.where(patch == GREY, BLACK, patch))
    templates.sort(key=lambda t: tuple(t.shape) + tuple(t.flatten()))
    return templates


def _candidate_anchors(panel: np.ndarray, obj_coords: np.ndarray, template: np.ndarray) -> list[tuple[int, int]]:
    obj_cells = [(int(r), int(c), int(panel[r, c])) for r, c in obj_coords]
    color_positions: dict[int, list[tuple[int, int]]] = {}
    for tr, tc in zip(*np.nonzero(template)):
        color_positions.setdefault(int(template[tr, tc]), []).append((int(tr), int(tc)))

    anchors = set()
    h, w = template.shape
    for r, c, val in obj_cells:
        for tr, tc in color_positions.get(val, []):
            anchor = (r - tr, c - tc)
            ar, ac = anchor
            if ar < 0 or ac < 0 or ar + h > panel.shape[0] or ac + w > panel.shape[1]:
                continue
            ok = True
            for rr, cc, color in obj_cells:
                lr = rr - ar
                lc = cc - ac
                if not (0 <= lr < h and 0 <= lc < w and int(template[lr, lc]) == color):
                    ok = False
                    break
            if ok:
                anchors.add(anchor)
    return sorted(anchors)


def _mirrored_anchor(anchor: tuple[int, int], width: int, panel_width: int) -> tuple[int, int]:
    r0, c0 = anchor
    return int(r0), int(panel_width - c0 - width)


def _region_matches_partial(panel: np.ndarray, template: np.ndarray, anchor: tuple[int, int]) -> bool:
    r0, c0 = anchor
    h, w = template.shape
    if r0 < 0 or c0 < 0 or r0 + h > panel.shape[0] or c0 + w > panel.shape[1]:
        return False
    patch = panel[r0 : r0 + h, c0 : c0 + w]
    visible = patch != BLACK
    return bool(visible.any()) and np.array_equal(patch[visible], template[visible])


def _stamp(panel: np.ndarray, template: np.ndarray, anchor: tuple[int, int]) -> None:
    r0, c0 = anchor
    h, w = template.shape
    patch = panel[r0 : r0 + h, c0 : c0 + w]
    mask = template != BLACK
    patch[mask] = template[mask]


def _transform_coord(coord: tuple[int, int], flip_vertical: bool) -> tuple[int, int]:
    r, c = coord
    if flip_vertical:
        return TEMPLATE_H - 1 - r, c
    return r, c


def _build_template_family(colors: list[int], flip_vertical: bool) -> list[dict]:
    family = []
    for left_idx in (0, 1):
        for right_idx in (0, 1):
            template = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=int)
            full_coords = list(_CORE_COORDS) + list(_LEFT_FEATURES[left_idx]) + list(_RIGHT_FEATURES[right_idx])
            for coord in full_coords:
                tr, tc = _transform_coord(coord, flip_vertical)
                template[tr, tc] = colors[_COORD_COLORS[coord]]

            left_mask = np.zeros_like(template, dtype=bool)
            for coord in list(_LEFT_VISIBLE_SHARED) + list(_LEFT_FEATURES[left_idx]):
                tr, tc = _transform_coord(coord, flip_vertical)
                left_mask[tr, tc] = True

            right_mask = np.zeros_like(template, dtype=bool)
            for coord in list(_RIGHT_VISIBLE_SHARED) + list(_RIGHT_FEATURES[right_idx]):
                tr, tc = _transform_coord(coord, flip_vertical)
                right_mask[tr, tc] = True

            family.append(
                {
                    "id": 2 * left_idx + right_idx,
                    "grid": template,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "left_mask": left_mask,
                    "right_mask": right_mask,
                }
            )
    return family


def _partial_from_mask(template: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    return np.where(visible_mask, template, BLACK)


def _choose_template_ids() -> list[int]:
    return random.choice(([0, 3], [1, 2]))


def _place_catalog_row(full_width: int, catalog_h: int, family: list[dict]) -> np.ndarray:
    catalog = np.full((catalog_h, full_width), GREY, dtype=int)
    order = family[:]
    random.shuffle(order)
    spare = full_width - (4 * TEMPLATE_W + 3)
    gap_bonus = [0, 0, 0]
    for _ in range(spare):
        gap_bonus[random.randrange(3)] += 1
    gaps = [1 + bonus for bonus in gap_bonus]
    used_width = 4 * TEMPLATE_W + sum(gaps)
    margin_left = random.randint(0, full_width - used_width)
    top = random.randint(0, catalog_h - TEMPLATE_H)
    c = margin_left
    for idx, entry in enumerate(order):
        patch = np.where(entry["grid"] != BLACK, entry["grid"], GREY)
        catalog[top : top + TEMPLATE_H, c : c + TEMPLATE_W] = patch
        c += TEMPLATE_W
        if idx < 3:
            c += gaps[idx]
    return catalog


def _sample_anchors(panel_h: int, panel_w: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(panel_h - TEMPLATE_H + 1) for c in range(panel_w - TEMPLATE_W + 1)]
    pairs = []
    for idx, (r0, c0) in enumerate(positions):
        for r1, c1 in positions[idx + 1 :]:
            separated = (
                r0 + TEMPLATE_H + 1 <= r1
                or r1 + TEMPLATE_H + 1 <= r0
                or c0 + TEMPLATE_W + 1 <= c1
                or c1 + TEMPLATE_W + 1 <= c0
            )
            if separated:
                pairs.append([(r0, c0), (r1, c1)])
    if not pairs:
        raise RuntimeError("failed to place two non-overlapping template boxes")
    return random.choice(pairs)


def _generate_attempt() -> np.ndarray | None:
    panel_h = random.randint(9, 12)
    panel_w = random.randint(10, 12)
    catalog_h = random.randint(5, 6)
    full_w = 2 * panel_w + 1
    if full_w < 4 * TEMPLATE_W + 3:
        return None

    colors = random.sample(PALETTE, 3)
    flip_vertical = bool(random.getrandbits(1))
    family = _build_template_family(colors, flip_vertical)
    family_by_id = {entry["id"]: entry for entry in family}

    left = np.zeros((panel_h, panel_w), dtype=int)
    right = np.zeros((panel_h, panel_w), dtype=int)
    expected_left = np.zeros_like(left)
    expected_right = np.zeros_like(right)

    ids = _choose_template_ids()
    random.shuffle(ids)
    anchors = _sample_anchors(panel_h, panel_w)
    random.shuffle(anchors)

    for tid, anchor in zip(ids, anchors):
        entry = family_by_id[tid]
        template = entry["grid"]
        left_partial = _partial_from_mask(template, entry["left_mask"])
        right_partial = np.fliplr(_partial_from_mask(template, entry["right_mask"]))
        mirror_anchor = _mirrored_anchor(anchor, TEMPLATE_W, panel_w)
        _stamp(left, left_partial, anchor)
        _stamp(right, right_partial, mirror_anchor)
        _stamp(expected_left, template, anchor)
        _stamp(expected_right, np.fliplr(template), mirror_anchor)

    catalog = _place_catalog_row(full_w, catalog_h, family)
    lower = np.concatenate([left, np.full((panel_h, 1), TEAL, dtype=int), right], axis=1)
    grid = np.concatenate([catalog, lower], axis=0)

    if not _validate_generated_instance(grid, family_by_id, expected_left, expected_right):
        return None
    return grid


def _validate_generated_instance(
    grid: np.ndarray,
    family_by_id: dict[int, dict],
    expected_left: np.ndarray,
    expected_right: np.ndarray,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    out = main(grid)
    expected = np.concatenate([expected_left, expected_right], axis=1)
    if not np.array_equal(out, expected):
        return False

    catalog_h = _catalog_height(grid)
    lower = grid[catalog_h:, :]
    sep = _separator_col(lower)
    left = lower[:, :sep]
    right = lower[:, sep + 1 :]

    templates = _extract_catalog_templates(grid[:catalog_h, :])
    if len(templates) != 4:
        return False
    mirrored = [np.fliplr(t) for t in templates]

    left_objects = _components(left != BLACK)
    right_objects = _components(right != BLACK)
    if len(left_objects) != 2 or len(right_objects) != 2:
        return False

    for obj in left_objects:
        left_template_ids = set()
        valid_joint = []
        for tid, template in enumerate(templates):
            anchors = _candidate_anchors(left, obj, template)
            if anchors:
                left_template_ids.add(tid)
            for anchor in anchors:
                mirror_anchor = _mirrored_anchor(anchor, template.shape[1], left.shape[1])
                if _region_matches_partial(right, mirrored[tid], mirror_anchor):
                    valid_joint.append((tid, anchor))
        if len(left_template_ids) < 2:
            return False
        if len(set(tid for tid, _ in valid_joint)) != 1 or len(set(valid_joint)) != 1:
            return False

    for obj in right_objects:
        right_template_ids = set()
        for tid, template in enumerate(mirrored):
            if _candidate_anchors(right, obj, template):
                right_template_ids.add(tid)
        if len(right_template_ids) < 2:
            return False

    family_templates = [entry["grid"] for entry in family_by_id.values()]
    extracted_sorted = sorted(tuple(t.flatten()) for t in templates)
    family_sorted = sorted(tuple(t.flatten()) for t in family_templates)
    return extracted_sorted == family_sorted
