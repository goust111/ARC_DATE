from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    out: list[np.ndarray] = []

    for start_r in range(height):
        for start_c in range(width):
            if not bool(mask[start_r, start_c]) or bool(seen[start_r, start_c]):
                continue

            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and bool(mask[nr, nc])
                        and not bool(seen[nr, nc])
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "shape mask must be non-empty"
    return _coords_to_mask(coords)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped = _crop_mask(np.asarray(mask, dtype=bool))
    coords = np.argwhere(cropped)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _rotation_keys(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    base = _crop_mask(np.asarray(mask, dtype=bool))
    return [_shape_key(np.rot90(base, k)) for k in range(4)]


def _rot_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_rotation_keys(mask))


def _dihedral_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    base = _crop_mask(np.asarray(mask, dtype=bool))
    forms: list[tuple[tuple[int, int], ...]] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        forms.append(_shape_key(rotated))
        forms.append(_shape_key(np.fliplr(rotated)))
    return min(forms)


def _enclosed_grey_holes(g: np.ndarray) -> list[dict[str, np.ndarray]]:
    height, width = g.shape
    holes: list[dict[str, np.ndarray]] = []

    for coords in _components(g == BLACK):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == height - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == width - 1):
            continue

        comp = np.zeros((height, width), dtype=bool)
        comp[coords[:, 0], coords[:, 1]] = True
        boundary_colors: set[int] = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and not bool(comp[nr, nc]):
                    boundary_colors.add(int(g[nr, nc]))

        if boundary_colors == {GREY}:
            holes.append({"coords": coords, "mask": _coords_to_mask(coords)})

    assert holes, "expected at least one grey-enclosed black hole"
    return holes


def _key_sprites(g: np.ndarray) -> list[dict[str, object]]:
    keys: list[dict[str, object]] = []
    for color in sorted(set(map(int, g.ravel())) - {BLACK, GREY}):
        comps = _components(g == int(color))
        assert len(comps) == 1, "each key color must form exactly one sprite"
        keys.append({"color": int(color), "mask": _coords_to_mask(comps[0])})

    assert keys, "expected at least one colored key sprite"
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled black keyhole whose negative-space shape matches a colored key up to rotation.

    Input structure:
    - The black background contains grey contour walls, each enclosing one black keyhole region.
    - Away from the walls are several single-color 4-connected key sprites; key colors are unique.
    - Some holes are rotations of key shapes, while other holes are unmatched or mirrored distractors.

    Rule:
    - Extract every black component fully enclosed by grey wall cells as a keyhole shape.
    - Extract every colored non-grey sprite as a key shape, and compare shapes up to translation and 90-degree rotations only.
    - Fill each matching keyhole with the color of its unique matching key; leave non-matching and mirrored-only holes black.
    - Preserve all grey wall pixels and the colored key sprites themselves.

    Key insight:
    - The shape to match is the wall's negative space, not the wall outline; rotations are allowed, but mirror images are not equivalent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    keys = _key_sprites(g)
    key_by_shape: dict[tuple[tuple[int, int], ...], int] = {}
    for key in keys:
        shape = _rot_key(np.asarray(key["mask"], dtype=bool))
        assert shape not in key_by_shape, "key shapes must be unique up to rotation"
        key_by_shape[shape] = int(key["color"])

    out = g.copy()
    for hole in _enclosed_grey_holes(g):
        shape = _rot_key(np.asarray(hole["mask"], dtype=bool))
        if shape in key_by_shape:
            coords = np.asarray(hole["coords"], dtype=int)
            out[coords[:, 0], coords[:, 1]] = key_by_shape[shape]

    return out


# --------------- generator ---------------

MIN_GRID_SIZE = 24
MAX_GRID_SIZE = 30
KEY_COUNT_RANGE = (2, 4)
SHAPE_H_RANGE = (3, 5)
SHAPE_W_RANGE = (3, 6)
SHAPE_AREA_RANGE = (5, 10)
GENERATION_ATTEMPTS = 800
SHAPE_ATTEMPTS = 400
PLACE_ATTEMPTS = 1000
PAD = 1
MARGIN = 1


def _random_connected_mask() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*SHAPE_H_RANGE)
        width = random.randint(*SHAPE_W_RANGE)
        max_area = height * width
        area = random.randint(SHAPE_AREA_RANGE[0], min(SHAPE_AREA_RANGE[1], max_area - 1))

        mask = np.zeros((height, width), dtype=bool)
        r = random.randrange(height)
        c = random.randrange(width)
        mask[r, c] = True
        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr = int(rr) + dr
                    nc = int(cc) + dc
                    if 0 <= nr < height and 0 <= nc < width and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        cropped = _crop_mask(mask)
        if cropped.shape[0] >= 2 and cropped.shape[1] >= 2 and int(cropped.sum()) >= SHAPE_AREA_RANGE[0]:
            return cropped

    raise RuntimeError("failed to sample a connected shape")


def _is_good_key_shape(mask: np.ndarray) -> bool:
    if len(set(_rotation_keys(mask))) != 4:
        return False
    return _rot_key(mask) != _rot_key(np.fliplr(mask))


def _sample_key_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    used_dihedral: set[tuple[tuple[int, int], ...]] = set()

    for _idx in range(count):
        for _attempt in range(SHAPE_ATTEMPTS):
            shape = _random_connected_mask()
            if not _is_good_key_shape(shape):
                continue
            dkey = _dihedral_key(shape)
            if dkey in used_dihedral:
                continue
            used_dihedral.add(dkey)
            shapes.append(shape)
            break
        else:
            return None
    return shapes


def _sample_unmatched_shape(key_rot_keys: set[tuple[tuple[int, int], ...]]) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        shape = _random_connected_mask()
        if _rot_key(shape) in key_rot_keys:
            continue
        if _rot_key(np.fliplr(shape)) in key_rot_keys:
            continue
        return shape
    return None


def _wall_patch(hole_mask: np.ndarray) -> np.ndarray:
    hole = _crop_mask(hole_mask)
    height, width = hole.shape
    patch = np.zeros((height + 2, width + 2), dtype=int)

    for r, c in np.argwhere(hole):
        pr = int(r) + 1
        pc = int(c) + 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = pr + dr
                nc = pc + dc
                if 0 <= nr < height + 2 and 0 <= nc < width + 2:
                    patch[nr, nc] = GREY

    for r, c in np.argwhere(hole):
        patch[int(r) + 1, int(c) + 1] = BLACK
    return patch


def _key_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(_crop_mask(mask), int(color), BLACK).astype(int)


def _can_place(reserved: np.ndarray, r: int, c: int, height: int, width: int) -> bool:
    grid_h, grid_w = reserved.shape
    r0 = max(0, r - PAD)
    c0 = max(0, c - PAD)
    r1 = min(grid_h, r + height + PAD)
    c1 = min(grid_w, c + width + PAD)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve(reserved: np.ndarray, r: int, c: int, height: int, width: int) -> None:
    grid_h, grid_w = reserved.shape
    r0 = max(0, r - PAD)
    c0 = max(0, c - PAD)
    r1 = min(grid_h, r + height + PAD)
    c1 = min(grid_w, c + width + PAD)
    reserved[r0:r1, c0:c1] = True


def _paste_nonblack(dst: np.ndarray, patch: np.ndarray, r: int, c: int) -> None:
    mask = patch != BLACK
    view = dst[r : r + patch.shape[0], c : c + patch.shape[1]]
    view[mask] = patch[mask]


def _place_patch(grid: np.ndarray, expected: np.ndarray, reserved: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    grid_h, grid_w = grid.shape
    height, width = patch.shape
    if height + 2 * MARGIN > grid_h or width + 2 * MARGIN > grid_w:
        return None

    for _ in range(PLACE_ATTEMPTS):
        r = random.randint(MARGIN, grid_h - height - MARGIN)
        c = random.randint(MARGIN, grid_w - width - MARGIN)
        if not _can_place(reserved, r, c, height, width):
            continue
        _paste_nonblack(grid, patch, r, c)
        _paste_nonblack(expected, patch, r, c)
        _reserve(reserved, r, c, height, width)
        return r, c
    return None


def _paint_hole(expected: np.ndarray, hole_mask: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(_crop_mask(hole_mask)):
        expected[top + 1 + int(r), left + 1 + int(c)] = int(color)


def _build_hole_specs(
    key_shapes: list[np.ndarray],
    key_colors: list[int],
) -> list[dict[str, object]] | None:
    key_rot_keys = {_rot_key(shape) for shape in key_shapes}
    key_count = len(key_shapes)
    match_count = random.randint(1, min(key_count, 3))
    match_indices = random.sample(range(key_count), match_count)

    specs: list[dict[str, object]] = []
    for pos, idx in enumerate(match_indices):
        turns = random.choice((1, 2, 3)) if pos == 0 else random.randrange(4)
        specs.append(
            {
                "mask": np.rot90(key_shapes[idx], turns).copy(),
                "fill": int(key_colors[idx]),
                "kind": "rotated_match" if turns else "match",
            }
        )

    mirror_idx = random.choice(match_indices)
    mirror_turns = random.randrange(4)
    mirror_mask = np.rot90(np.fliplr(key_shapes[mirror_idx]), mirror_turns).copy()
    if _rot_key(mirror_mask) in key_rot_keys:
        return None
    specs.append({"mask": mirror_mask, "fill": None, "kind": "mirror_decoy"})

    unmatched = _sample_unmatched_shape(key_rot_keys)
    if unmatched is None:
        return None
    specs.append({"mask": unmatched, "fill": None, "kind": "unmatched"})

    if random.random() < 0.35:
        extra = _sample_unmatched_shape(key_rot_keys)
        if extra is not None:
            specs.append({"mask": extra, "fill": None, "kind": "unmatched"})

    random.shuffle(specs)
    return specs


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    hole_specs: list[dict[str, object]],
    key_shapes: list[np.ndarray],
) -> bool:
    if np.array_equal(inp, expected):
        return False

    fills = [spec for spec in hole_specs if spec["fill"] is not None]
    decoys = [spec for spec in hole_specs if spec["fill"] is None]
    if not fills or not decoys:
        return False
    if not any(spec["kind"] == "mirror_decoy" for spec in hole_specs):
        return False
    if not any(spec["kind"] == "rotated_match" for spec in hole_specs):
        return False

    key_rot_keys = {_rot_key(shape) for shape in key_shapes}
    for spec in decoys:
        if _rot_key(np.asarray(spec["mask"], dtype=bool)) in key_rot_keys:
            return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    key_count = random.randint(*KEY_COUNT_RANGE)
    key_colors = random.sample(KEY_COLORS, key_count)
    key_shapes = _sample_key_shapes(key_count)
    if key_shapes is None:
        return None

    hole_specs = _build_hole_specs(key_shapes, key_colors)
    if hole_specs is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    wall_specs = sorted(hole_specs, key=lambda spec: int(np.asarray(spec["mask"], dtype=bool).sum()), reverse=True)
    for spec in wall_specs:
        patch = _wall_patch(np.asarray(spec["mask"], dtype=bool))
        placed = _place_patch(inp, expected, reserved, patch)
        if placed is None:
            return None
        if spec["fill"] is not None:
            _paint_hole(expected, np.asarray(spec["mask"], dtype=bool), placed[0], placed[1], int(spec["fill"]))

    key_items = list(zip(key_shapes, key_colors))
    random.shuffle(key_items)
    key_items.sort(key=lambda item: int(item[0].sum()), reverse=True)
    for shape, color in key_items:
        patch = _key_patch(shape, color)
        if _place_patch(inp, expected, reserved, patch) is None:
            return None

    if not _semantic_checks(inp, expected, hole_specs, key_shapes):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-and-keyhole input")
