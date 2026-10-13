from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    return coords - coords.min(axis=0)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    norm = _normalize_coords(coords)
    h, w = norm.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[norm[:, 0], norm[:, 1]] = True
    return mask


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    norm = _normalize_coords(coords)
    h, w = norm.max(axis=0) + 1
    cells = tuple(sorted((int(r), int(c)) for r, c in norm))
    return int(h), int(w), cells


def _unique_oriented_masks(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            coords = np.argwhere(candidate)
            normalized = _mask_from_coords(coords)
            if not any(np.array_equal(normalized, prev) for prev in out):
                out.append(normalized)
    return out


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(patch)
    dst[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components(grid == int(color)):
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _holes_in_sprite(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    bg_comps = _components(patch == BLACK)
    holes = [coords for coords in bg_comps if not _touches_border(coords, patch.shape)]
    return holes


def _sprite_instances(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    comps = _find_color_components(grid, GREY)
    assert 4 <= len(comps) <= 6, "expected 4-6 grey sprites"
    instances = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch == GREY
        instances.append((r0, c0, mask))
    ref = instances[0][2]
    assert all(mask.shape == ref.shape and np.array_equal(mask, ref) for _r, _c, mask in instances), (
        "all grey components must share one sprite shape"
    )
    return instances


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a rectangular lattice of identical gray sprites, then fill every sprite's internal holes using colors from matching external marker shapes.

    Input structure:
    - A black background contains 4-6 identical gray 4-connected sprites on a rectangular row/column lattice, with at least one occupied full row and full column and at least two empty lattice slots.
    - Each sprite has exactly two internal black hole components inside its bounding box, and those hole shapes and positions are identical across sprites.
    - Outside the lattice area are 3-4 small colored marker components; exactly two have normalized shapes matching the two sprite-hole shapes, while the others are distractors.

    Rule:
    - Recover the common sprite mask and the lattice row/column starts from the existing gray components.
    - Fill every missing Cartesian-product lattice slot with a gray copy of that sprite.
    - Extract the two internal hole shapes from a sprite bbox, normalize them, and match them against the normalized external marker shapes.
    - Use the color of each matching marker to fill the corresponding hole in every sprite, including newly completed ones.
    - Remove all external marker shapes, leaving only the completed lattice with colored holes.

    Key insight:
    - The lattice completion step must happen first because the newly created sprites also contain holes, and the hole colors come only from normalized shape matching rather than marker position or marker color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    sprites = _sprite_instances(g)
    rows = sorted({int(r) for r, _c, _mask in sprites})
    cols = sorted({int(c) for _r, c, _mask in sprites})
    occupied = {(int(r), int(c)) for r, c, _mask in sprites}
    assert len(rows) >= 2 and len(cols) >= 2
    assert any(all((r, c) in occupied for c in cols) for r in rows), "missing complete sprite row"
    assert any(all((r, c) in occupied for r in rows) for c in cols), "missing complete sprite column"

    sprite_mask = sprites[0][2]
    sprite_h, sprite_w = sprite_mask.shape
    missing = [(r, c) for r in rows for c in cols if (r, c) not in occupied]
    assert len(missing) >= 2, "expected at least two empty lattice slots"

    ref_r, ref_c, _ = sprites[0]
    ref_box = (ref_r, ref_c, ref_r + sprite_h - 1, ref_c + sprite_w - 1)
    hole_comps = _holes_in_sprite(g, ref_box)
    assert len(hole_comps) == 2, "expected exactly two sprite holes"
    hole_keys = [_shape_key(coords) for coords in hole_comps]
    assert hole_keys[0] != hole_keys[1], "sprite holes must have different shapes"
    hole_locals = {key: coords.copy() for key, coords in zip(hole_keys, hole_comps)}

    marker_components = []
    for color in MARKER_COLORS:
        marker_components.extend(_find_color_components(g, color))

    fills: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    match_counts: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    marker_coords = []
    for comp in marker_components:
        marker_coords.append(comp["coords"])
        key = _shape_key(comp["coords"])
        if key in hole_locals:
            fills[key] = int(comp["color"])
            match_counts[key] = match_counts.get(key, 0) + 1
    assert set(fills) == set(hole_locals), "expected one matching marker per hole shape"
    assert all(match_counts.get(key, 0) == 1 for key in hole_locals), "each hole shape must have exactly one marker"
    assert len(set(fills.values())) == 2, "hole colors must be distinct"

    out = g.copy()
    for coords in marker_coords:
        out[coords[:, 0], coords[:, 1]] = BLACK

    for r, c in missing:
        _paint_patch(out, sprite_mask, r, c, GREY)

    for r in rows:
        for c in cols:
            for key, local_coords in hole_locals.items():
                color = fills[key]
                out[r + local_coords[:, 0], c + local_coords[:, 1]] = color

    return out


def _build_shape_catalog() -> list[np.ndarray]:
    base = [
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ]
    shapes: list[np.ndarray] = []
    for mask in base:
        for variant in _unique_oriented_masks(mask):
            if not any(np.array_equal(variant, seen) for seen in shapes):
                shapes.append(variant)
    return shapes


SHAPE_CATALOG = _build_shape_catalog()


@dataclass(frozen=True)
class SpriteTemplate:
    body_mask: np.ndarray
    hole_masks: tuple[np.ndarray, np.ndarray]
    hole_locals: tuple[np.ndarray, np.ndarray]
    same_area: bool


def _build_sprite_templates() -> list[SpriteTemplate]:
    templates: list[SpriteTemplate] = []
    seen = set()
    for h in range(4, 7):
        for w in range(4, 7):
            if h * w > 30:
                continue
            outer = np.ones((h, w), dtype=bool)
            for i, shape_a in enumerate(SHAPE_CATALOG):
                for j, shape_b in enumerate(SHAPE_CATALOG):
                    if i >= j:
                        continue
                    area_a = int(shape_a.sum())
                    area_b = int(shape_b.sum())
                    body_area = h * w - area_a - area_b
                    if not (10 <= body_area <= 20):
                        continue
                    ha, wa = shape_a.shape
                    hb, wb = shape_b.shape
                    for ra in range(1, h - ha):
                        for ca in range(1, w - wa):
                            mask_a = np.zeros((h, w), dtype=bool)
                            mask_a[ra : ra + ha, ca : ca + wa] = shape_a
                            for rb in range(1, h - hb):
                                for cb in range(1, w - wb):
                                    mask_b = np.zeros((h, w), dtype=bool)
                                    mask_b[rb : rb + hb, cb : cb + wb] = shape_b
                                    holes = mask_a | mask_b
                                    if np.any(mask_a & mask_b):
                                        continue
                                    hole_parts = _components(holes)
                                    if len(hole_parts) != 2:
                                        continue
                                    if any(_touches_border(coords, holes.shape) for coords in hole_parts):
                                        continue
                                    body = outer & ~holes
                                    if len(_components(body)) != 1:
                                        continue
                                    body_coords = np.argwhere(body)
                                    local_a = np.argwhere(mask_a)
                                    local_b = np.argwhere(mask_b)
                                    key = (
                                        _shape_key(body_coords),
                                        tuple(sorted((_shape_key(local_a), _shape_key(local_b)))),
                                    )
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    templates.append(
                                        SpriteTemplate(
                                            body_mask=body.copy(),
                                            hole_masks=(_mask_from_coords(local_a), _mask_from_coords(local_b)),
                                            hole_locals=(local_a.copy(), local_b.copy()),
                                            same_area=area_a == area_b,
                                        )
                                    )
    assert templates, "failed to build sprite templates"
    return templates


SPRITE_TEMPLATES = _build_sprite_templates()
SAME_AREA_TEMPLATES = [tpl for tpl in SPRITE_TEMPLATES if tpl.same_area]
DIFF_AREA_TEMPLATES = [tpl for tpl in SPRITE_TEMPLATES if not tpl.same_area]


def _choose_template() -> SpriteTemplate:
    pool = SAME_AREA_TEMPLATES if random.random() < 0.5 else DIFF_AREA_TEMPLATES
    return random.choice(pool)


def _transform_template(template: SpriteTemplate) -> tuple[np.ndarray, list[np.ndarray]]:
    sprite = template.body_mask.copy()
    holes = [mask.copy() for mask in template.hole_masks]
    turns = random.randrange(4)
    flip = bool(random.randrange(2))
    if flip:
        sprite = np.fliplr(sprite)
        holes = [np.fliplr(mask) for mask in holes]
    if turns:
        sprite = np.rot90(sprite, turns)
        holes = [np.rot90(mask, turns) for mask in holes]
    holes = [_mask_from_coords(np.argwhere(mask)) for mask in holes]
    return sprite, holes


def _sample_lattice(sprite_shape: tuple[int, int]) -> tuple[int, int, list[int], list[int], int, int]:
    h, w = sprite_shape
    lattice_options = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)]
    n_rows, n_cols = random.choice(lattice_options)
    step_r = h + random.randint(2, 4)
    step_c = w + random.randint(2, 4)
    rows = [i * step_r for i in range(n_rows)]
    cols = [j * step_c for j in range(n_cols)]
    lattice_h = rows[-1] + h
    lattice_w = cols[-1] + w
    return n_rows, n_cols, rows, cols, lattice_h, lattice_w


def _positions_from_lattice(
    top: int,
    left: int,
    rows: list[int],
    cols: list[int],
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    abs_rows = [top + r for r in rows]
    abs_cols = [left + c for c in cols]
    row_idx = random.randrange(len(rows))
    col_idx = random.randrange(len(cols))
    occupied = {(abs_rows[row_idx], c) for c in abs_cols} | {(r, abs_cols[col_idx]) for r in abs_rows}
    return abs_rows, abs_cols, sorted(occupied)


def _bbox_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _choose_distractor_shapes(hole_keys: list[tuple[int, int, tuple[tuple[int, int], ...]]]) -> list[np.ndarray]:
    distractors: list[np.ndarray] = []
    num_distractors = random.choice((1, 2))

    rotated_added = False
    if num_distractors == 2 or random.random() < 0.55:
        for key in hole_keys:
            mask = _mask_from_coords(np.array(key[2], dtype=int))
            for candidate in _unique_oriented_masks(mask):
                cand_key = _shape_key(np.argwhere(candidate))
                if cand_key != key and cand_key not in hole_keys:
                    distractors.append(candidate)
                    rotated_added = True
                    break
            if rotated_added:
                break

    while len(distractors) < num_distractors:
        candidate = random.choice(SHAPE_CATALOG)
        cand_key = _shape_key(np.argwhere(candidate))
        if cand_key in hole_keys:
            continue
        if any(np.array_equal(candidate, seen) for seen in distractors):
            continue
        distractors.append(candidate)

    return distractors


def _place_markers(
    grid_shape: tuple[int, int],
    lattice_box: tuple[int, int, int, int],
    markers: list[np.ndarray],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = []
    order = list(range(len(markers)))
    random.shuffle(order)
    ordered_markers = [markers[idx] for idx in order]

    chosen_positions: list[tuple[int, int]] = []
    for marker in ordered_markers:
        mh, mw = marker.shape
        candidates = []
        for r in range(h - mh + 1):
            for c in range(w - mw + 1):
                box = (r, c, r + mh - 1, c + mw - 1)
                if not _bbox_far(box, lattice_box, pad=1):
                    continue
                if any(not _bbox_far(box, other, pad=1) for other in placed_boxes):
                    continue
                candidates.append((r, c, box))
        if not candidates:
            raise ValueError("no room for markers")
        r, c, box = random.choice(candidates)
        chosen_positions.append((r, c))
        placed_boxes.append(box)

    reverse = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    positions = [chosen_positions[reverse[idx]] for idx in range(len(markers))]
    return positions


def _render_expected(
    grid_shape: tuple[int, int],
    sprite_mask: np.ndarray,
    hole_locals: list[np.ndarray],
    rows: list[int],
    cols: list[int],
    fills: list[int],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for r in rows:
        for c in cols:
            _paint_patch(out, sprite_mask, r, c, GREY)
            for coords, color in zip(hole_locals, fills):
                out[r + coords[:, 0], c + coords[:, 1]] = int(color)
    return out


def _render_input(
    grid_shape: tuple[int, int],
    sprite_mask: np.ndarray,
    occupied: list[tuple[int, int]],
    markers: list[np.ndarray],
    marker_colors: list[int],
    marker_positions: list[tuple[int, int]],
) -> np.ndarray:
    inp = np.zeros(grid_shape, dtype=int)
    for r, c in occupied:
        _paint_patch(inp, sprite_mask, r, c, GREY)
    for patch, color, (r, c) in zip(markers, marker_colors, marker_positions):
        _paint_patch(inp, patch, r, c, color)
    return inp


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    marker_shapes: list[np.ndarray],
    marker_colors: list[int],
    marker_positions: list[tuple[int, int]],
) -> bool:
    if inp.shape != expected.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False

    grey = _sprite_instances(inp)
    rows = sorted({int(r) for r, _c, _mask in grey})
    cols = sorted({int(c) for _r, c, _mask in grey})
    occupied = {(int(r), int(c)) for r, c, _mask in grey}
    if len([(r, c) for r in rows for c in cols if (r, c) not in occupied]) < 2:
        return False
    if not any(all((r, c) in occupied for c in cols) for r in rows):
        return False
    if not any(all((r, c) in occupied for r in rows) for c in cols):
        return False

    ref_r, ref_c, ref_mask = grey[0]
    holes = _holes_in_sprite(inp, (ref_r, ref_c, ref_r + ref_mask.shape[0] - 1, ref_c + ref_mask.shape[1] - 1))
    if len(holes) != 2:
        return False
    hole_keys = {_shape_key(hole) for hole in holes}

    matched = 0
    seen_colors = set()
    for patch, color, (r, c) in zip(marker_shapes, marker_colors, marker_positions):
        if len(np.argwhere(patch)) < 2:
            return False
        seen_colors.add(color)
        coords = np.argwhere(patch) + np.array([r, c])
        if _shape_key(coords) in hole_keys:
            matched += 1
    if matched != 2 or len(seen_colors) != len(marker_colors):
        return False

    got = main(inp)
    if not np.array_equal(got, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        sprite_mask, hole_masks = _transform_template(_choose_template())
        sprite_h, sprite_w = sprite_mask.shape
        hole_locals = _components(~sprite_mask)
        if len(hole_locals) != 2:
            continue
        hole_marker_masks = [_mask_from_coords(coords) for coords in hole_locals]
        hole_keys = [_shape_key(np.argwhere(mask)) for mask in hole_marker_masks]
        if len(set(hole_keys)) != 2:
            continue

        n_rows, n_cols, row_offsets, col_offsets, lattice_h, lattice_w = _sample_lattice((sprite_h, sprite_w))
        min_h = max(20, lattice_h + 6)
        min_w = max(20, lattice_w + 6)
        if min_h > 30 or min_w > 30:
            continue
        grid_h = random.randint(min_h, min(30, lattice_h + 12))
        grid_w = random.randint(min_w, min(30, lattice_w + 12))

        top = random.randint(1, grid_h - lattice_h - 1)
        left = random.randint(1, grid_w - lattice_w - 1)
        rows, cols, occupied = _positions_from_lattice(top, left, row_offsets, col_offsets)
        lattice_box = (rows[0], cols[0], rows[-1] + sprite_h - 1, cols[-1] + sprite_w - 1)

        fill_colors = random.sample(list(MARKER_COLORS), 2)
        matched_markers = [mask.copy() for mask in hole_marker_masks]
        distractors = _choose_distractor_shapes(hole_keys)
        all_markers = matched_markers + distractors
        marker_colors = fill_colors + random.sample([c for c in MARKER_COLORS if c not in fill_colors], len(distractors))

        try:
            marker_positions = _place_markers((grid_h, grid_w), lattice_box, all_markers)
        except ValueError:
            continue

        expected = _render_expected((grid_h, grid_w), sprite_mask, hole_locals, rows, cols, fill_colors)
        inp = _render_input((grid_h, grid_w), sprite_mask, occupied, all_markers, marker_colors, marker_positions)

        if not _validate_instance(inp, expected, all_markers, marker_colors, marker_positions):
            continue
        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
