from __future__ import annotations

import random

import numpy as np


BLACK = 0
_PAIR_HIDDEN_STATUSES = {"left_hidden", "right_hidden"}
_GEN_ATTEMPTS = 500
_GEN_GRID_DIMS = tuple(range(14, 31, 2))
_GEN_BG_COLOR_COUNT = (3, 5)
_GEN_BG_BLOCK = (1, 3)
_GEN_SHAPE_HEIGHT = (5, 10)
_GEN_SHAPE_WIDTH_MIN = 5
_GEN_SHAPE_WIDTH_MAX = 11
_GEN_QUADRANTS = ("tl", "tr", "bl", "br")
_GEN_COLORS = tuple(range(1, 10))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _orbit_coords(r: int, c: int, height: int, width: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (int(r), int(c)),
                (int(r), int(width - 1 - c)),
                (int(height - 1 - r), int(c)),
                (int(height - 1 - r), int(width - 1 - c)),
            }
        )
    )


def _unique_mode(values: list[int]) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert winners.size == 1, "expected a unique orbit majority"
    return int(winners[0])


def _restore_background(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    restored = np.empty_like(grid, dtype=int)
    seen: set[tuple[int, int]] = set()
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            orbit = _orbit_coords(r, c, height, width)
            color = _unique_mode([int(grid[rr, cc]) for rr, cc in orbit])
            for rr, cc in orbit:
                restored[rr, cc] = color
                seen.add((rr, cc))
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the overwritten symmetric intruder, then fill its invisible cells by the intruder's own left-right symmetry.

    Input structure:
    - The canvas shows a non-black multicolor background that is symmetric under both left-right and up-down reflection.
    - A separate intruder shape, itself left-right symmetric, overwrote one bounded region; some overwritten cells stayed invisible because their overwrite color matched the background underneath.
    - No background symmetry orbit has more than one overwritten cell, so the original background can be recovered by orbit voting.

    Rule:
    - For each cell, group it with its horizontal mirror, vertical mirror, and 180-degree partner, and restore the background color by majority vote within that orbit.
    - Mark every cell whose current value differs from the restored background; these visible residue cells form the intruder with holes.
    - Take the residue bbox, treat its vertical midline as the intruder's symmetry axis, and copy each visible residue color to its mirror position to refill the invisible holes.
    - Return only that completed intruder bbox, keeping intruder cells colored and leaving non-intruder cells black.

    Key insight:
    - Two different symmetries act in sequence: the canvas's biaxial symmetry reveals where the overwrite happened, then the intruder's own left-right symmetry recovers the cells hidden by same-color overwrites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    restored = _restore_background(g)
    residue_mask = g != restored
    box = _bbox(residue_mask)
    assert box is not None, "expected at least one visible overwritten residue"

    residue = _crop_bbox(g, box)
    residue_local = _crop_bbox(residue_mask.astype(int), box).astype(bool)
    out = np.zeros_like(residue, dtype=int)
    width = residue.shape[1]

    for r, c in np.argwhere(residue_local):
        rr = int(r)
        cc = int(c)
        mirror_c = width - 1 - cc
        color = int(residue[rr, cc])
        if out[rr, cc] not in (BLACK, color):
            raise AssertionError("visible residue conflicts with inferred intruder color")
        if out[rr, mirror_c] not in (BLACK, color):
            raise AssertionError("mirror completion conflicts with visible residue")
        out[rr, cc] = color
        out[rr, mirror_c] = color

    assert np.array_equal(out, out[:, ::-1]), "completed intruder must be left-right symmetric"
    return out


def _gen_symmetric_background(height: int, width: int, palette: list[int]) -> np.ndarray:
    half_h, half_w = height // 2, width // 2
    block_h = random.randint(*_GEN_BG_BLOCK)
    block_w = random.randint(*_GEN_BG_BLOCK)
    coarse_h = (half_h + block_h - 1) // block_h
    coarse_w = (half_w + block_w - 1) // block_w
    coarse_cells = coarse_h * coarse_w

    values = list(palette)
    while len(values) < coarse_cells:
        values.append(random.choice(palette))
    random.shuffle(values)
    coarse = np.asarray(values[:coarse_cells], dtype=int).reshape(coarse_h, coarse_w)

    quadrant = np.kron(coarse, np.ones((block_h, block_w), dtype=int))[:half_h, :half_w]
    top = np.concatenate([quadrant, quadrant[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


def _gen_profile(height: int, width: int) -> list[int]:
    max_left = (width - 1) // 2
    current = random.randint(0, max_left)
    lefts = [current]
    for _ in range(1, height):
        steps = [delta for delta in (-1, 0, 1) if 0 <= current + delta <= max_left]
        current += random.choice(steps)
        lefts.append(current)

    if 0 not in lefts:
        lefts[random.randrange(height)] = 0
    if max_left > 0 and max(lefts) == 0:
        lefts[random.randrange(height)] = random.randint(1, max_left)
    return lefts


def _mask_from_profile(lefts: list[int], width: int) -> np.ndarray:
    mask = np.zeros((len(lefts), width), dtype=bool)
    for r, left in enumerate(lefts):
        mask[r, left : width - left] = True
    return mask


def _pick_intruder_dims(half_h: int, half_w: int) -> tuple[int, int] | None:
    max_h = min(_GEN_SHAPE_HEIGHT[1], half_h)
    max_w = min(_GEN_SHAPE_WIDTH_MAX, half_w)
    if max_h < _GEN_SHAPE_HEIGHT[0] or max_w < _GEN_SHAPE_WIDTH_MIN:
        return None

    height = random.randint(_GEN_SHAPE_HEIGHT[0], max_h)
    width_choices = list(range(_GEN_SHAPE_WIDTH_MIN, max_w + 1))
    parity = random.randrange(2)
    parity_choices = [width for width in width_choices if width % 2 == parity]
    width = random.choice(parity_choices or width_choices)
    return height, width


def _pick_bbox_origin(height: int, width: int, shape_h: int, shape_w: int, quadrant: str) -> tuple[int, int]:
    half_h, half_w = height // 2, width // 2
    row_base = 0 if quadrant in {"tl", "tr"} else half_h
    col_base = 0 if quadrant in {"tl", "bl"} else half_w
    r0 = row_base + random.randint(0, half_h - shape_h)
    c0 = col_base + random.randint(0, half_w - shape_w)
    return r0, c0


def _build_units(mask: np.ndarray, bg_crop: np.ndarray, palette: list[int]) -> list[dict]:
    units: list[dict] = []
    width = mask.shape[1]
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        assert cols.size > 0
        left = int(cols[0])
        right = int(cols[-1])
        c = left
        while c <= right:
            mirror_c = width - 1 - c
            if c > mirror_c:
                break
            if c == mirror_c:
                bg = int(bg_crop[r, c])
                options = [("center", color) for color in palette if color != bg]
            else:
                bg_left = int(bg_crop[r, c])
                bg_right = int(bg_crop[r, mirror_c])
                options = [("both", color) for color in palette if color != bg_left and color != bg_right]
                if bg_left != bg_right:
                    options.append(("left_hidden", bg_left))
                    options.append(("right_hidden", bg_right))
            assert options, "every occupied symmetry unit needs at least one legal coloring"
            units.append({"row": r, "left": c, "right": mirror_c, "options": options})
            c += 1
    return units


def _shape_semantics_ok(shape: np.ndarray, mask: np.ndarray, bg_crop: np.ndarray) -> bool:
    visible = (shape != BLACK) & (shape != bg_crop)
    hidden = (shape != BLACK) & (shape == bg_crop)
    if not np.any(hidden):
        return False
    if not np.any((shape[:, 0] != BLACK) & (shape[:, 0] != bg_crop[:, 0])):
        return False
    if not np.any((shape[:, -1] != BLACK) & (shape[:, -1] != bg_crop[:, -1])):
        return False

    width = shape.shape[1]
    for r in range(shape.shape[0]):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            continue
        if not np.any(visible[r] & mask[r]):
            return False
        left = int(cols[0])
        right = int(cols[-1])
        c = left
        while c <= right:
            mirror_c = width - 1 - c
            if c > mirror_c:
                break
            if shape[r, c] != shape[r, mirror_c]:
                return False
            if c == mirror_c:
                if shape[r, c] == int(bg_crop[r, c]):
                    return False
            else:
                left_visible = shape[r, c] != int(bg_crop[r, c])
                right_visible = shape[r, mirror_c] != int(bg_crop[r, mirror_c])
                if not (left_visible or right_visible):
                    return False
            c += 1
    return True


def _render_intruder(mask: np.ndarray, bg_crop: np.ndarray, palette: list[int]) -> np.ndarray | None:
    units = _build_units(mask, bg_crop, palette)
    boundary_both = [
        i
        for i, unit in enumerate(units)
        if unit["left"] == 0
        and unit["right"] == mask.shape[1] - 1
        and any(status == "both" for status, _ in unit["options"])
    ]
    hidden_units = [
        i
        for i, unit in enumerate(units)
        if any(status in _PAIR_HIDDEN_STATUSES for status, _ in unit["options"])
    ]
    if not boundary_both or not hidden_units:
        return None

    for _ in range(200):
        assignments: dict[int, tuple[str, int]] = {}

        boundary_idx = random.choice(boundary_both)
        boundary_options = [opt for opt in units[boundary_idx]["options"] if opt[0] == "both"]
        assignments[boundary_idx] = random.choice(boundary_options)

        hidden_candidates = [idx for idx in hidden_units if idx != boundary_idx]
        if not hidden_candidates:
            continue
        hidden_idx = random.choice(hidden_candidates)
        hidden_options = [opt for opt in units[hidden_idx]["options"] if opt[0] in _PAIR_HIDDEN_STATUSES]
        assignments[hidden_idx] = random.choice(hidden_options)

        for idx, unit in enumerate(units):
            if idx in assignments:
                continue
            options = unit["options"]
            hole_options = [opt for opt in options if opt[0] in _PAIR_HIDDEN_STATUSES]
            both_options = [opt for opt in options if opt[0] == "both"]
            if unit["left"] == unit["right"]:
                assignments[idx] = random.choice(options)
            elif hole_options and random.random() < 0.4:
                assignments[idx] = random.choice(hole_options)
            elif both_options:
                assignments[idx] = random.choice(both_options)
            else:
                assignments[idx] = random.choice(options)

        shape = np.zeros(mask.shape, dtype=int)
        for idx, unit in enumerate(units):
            _status, color = assignments[idx]
            r = int(unit["row"])
            c0 = int(unit["left"])
            c1 = int(unit["right"])
            shape[r, c0] = int(color)
            shape[r, c1] = int(color)

        if _shape_semantics_ok(shape, mask, bg_crop):
            return shape
    return None


def _overlay_intruder(background: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> np.ndarray:
    out = background.copy()
    patch = out[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]]
    overwrite = shape != BLACK
    patch[overwrite] = shape[overwrite]
    return out


def _validate_instance(
    grid: np.ndarray,
    background: np.ndarray,
    expected: np.ndarray,
    r0: int,
    c0: int,
) -> bool:
    restored = _restore_background(grid)
    if not np.array_equal(restored, background):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    residue_mask = grid != restored
    box = _bbox(residue_mask)
    expected_box = (r0, c0, r0 + expected.shape[0] - 1, c0 + expected.shape[1] - 1)
    if box != expected_box:
        return False

    orbit_reprs: set[tuple[tuple[int, int], ...]] = set()
    for local_r, local_c in np.argwhere(expected != BLACK):
        orbit = _orbit_coords(r0 + int(local_r), c0 + int(local_c), *background.shape)
        if orbit in orbit_reprs:
            return False
        orbit_reprs.add(orbit)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(_GEN_GRID_DIMS)
    width = random.choice(_GEN_GRID_DIMS)
    dims = _pick_intruder_dims(height // 2, width // 2)
    if dims is None:
        return None
    shape_h, shape_w = dims

    palette_size = random.randint(*_GEN_BG_COLOR_COUNT)
    palette = random.sample(_GEN_COLORS, palette_size)
    background = _gen_symmetric_background(height, width, palette)

    profile = _gen_profile(shape_h, shape_w)
    mask = _mask_from_profile(profile, shape_w)
    quadrant = random.choice(_GEN_QUADRANTS)
    r0, c0 = _pick_bbox_origin(height, width, shape_h, shape_w, quadrant)
    bg_crop = background[r0 : r0 + shape_h, c0 : c0 + shape_w]

    shape = _render_intruder(mask, bg_crop, palette)
    if shape is None:
        return None

    grid = _overlay_intruder(background, shape, r0, c0)
    if not _validate_instance(grid, background, shape, r0, c0):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_ATTEMPTS} attempts")
