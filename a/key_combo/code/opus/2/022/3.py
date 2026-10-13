from __future__ import annotations

import random

import numpy as np


BLACK = 0
TEAL = 8
SPRITE_COLORS = (1, 2, 3, 4, 7)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _unique_variants(arrays: list[np.ndarray]) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for arr in arrays:
        candidate = np.ascontiguousarray(arr.astype(bool))
        key = (candidate.shape, candidate.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _dihedral_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return _unique_variants(variants)


VERTICAL_BASES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]
HORIZONTAL_BASES = [
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
]
ROT180_BASES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
]
ASYM_BASES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
]

VERTICAL_VARIANTS = _unique_variants([base for arr in VERTICAL_BASES for base in (arr, np.flipud(arr))])
HORIZONTAL_VARIANTS = _unique_variants([base for arr in HORIZONTAL_BASES for base in (arr, np.fliplr(arr))])
ROT180_VARIANTS = _unique_variants([variant for arr in ROT180_BASES for variant in _dihedral_variants(arr)])
ASYM_VARIANTS = _unique_variants([variant for arr in ASYM_BASES for variant in _dihedral_variants(arr)])
ALL_SYM_VARIANTS = VERTICAL_VARIANTS + HORIZONTAL_VARIANTS + ROT180_VARIANTS


def _is_symmetric(sprite_mask: np.ndarray) -> bool:
    return (
        np.array_equal(sprite_mask, np.fliplr(sprite_mask))
        or np.array_equal(sprite_mask, np.flipud(sprite_mask))
        or np.array_equal(sprite_mask, np.rot90(sprite_mask, 2))
    )


def _cavity_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    cavities = []
    for coords in _connected_components(grid != TEAL):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords):
            continue
        cavities.append(coords)
    cavities.sort(key=lambda comp: _bbox_coords(comp)[:2])
    return cavities


def _analyze_cavity(grid: np.ndarray, cavity: list[tuple[int, int]]) -> dict:
    colors = {int(grid[r, c]) for r, c in cavity if int(grid[r, c]) != BLACK}
    assert len(colors) == 1, "each cavity must contain exactly one sprite color"
    color = next(iter(colors))

    sprite_mask_full = np.zeros(grid.shape, dtype=bool)
    sprite_coords = []
    for r, c in cavity:
        if int(grid[r, c]) == color:
            sprite_mask_full[r, c] = True
            sprite_coords.append((r, c))

    assert 3 <= len(sprite_coords) <= 6, "sprite area must be 3..6"
    assert len(_connected_components(sprite_mask_full)) == 1, "sprite must be 4-connected"

    sprite_bbox = _bbox_coords(sprite_coords)
    sprite_mask = _crop_bbox(sprite_mask_full, sprite_bbox)
    return {
        "bbox": _bbox_coords(cavity),
        "color": color,
        "area": len(sprite_coords),
        "sprite_bbox": sprite_bbox,
        "sprite_mask": sprite_mask,
        "symmetric": _is_symmetric(sprite_mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill symmetric sprites' chambers and erase asymmetric sprites from the teal lattice.

    Input structure:
    - A teal one-pixel lattice with an outer border partitions the grid into closed rectangular chambers.
    - Each chamber contains black interior cells plus exactly one monochrome 4-connected sprite of size 3 to 6.
    - Sprite colors come from the non-teal ARC palette and may repeat across different chambers.

    Rule:
    - Find each chamber enclosed by the teal lattice and isolate the sprite inside it.
    - Crop the sprite to its own bounding box and test for vertical reflection, horizontal reflection, or 180-degree rotational symmetry.
    - If the sprite passes any of those symmetry tests, fill the entire chamber interior with the sprite's color; otherwise clear the entire chamber interior to black.

    Key insight:
    - Symmetry is evaluated on each sprite's cropped shape, and that local symmetry result gates a separate fill-versus-erase action on the whole chamber.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cavities = _cavity_components(g)
    assert cavities, "expected enclosed chambers"

    out = g.copy()
    for cavity in cavities:
        info = _analyze_cavity(g, cavity)
        fill_color = info["color"] if info["symmetric"] else BLACK
        for r, c in cavity:
            out[r, c] = fill_color
    return out


def _random_vertical_shape() -> np.ndarray:
    return random.choice(VERTICAL_VARIANTS).copy()


def _random_horizontal_shape() -> np.ndarray:
    return random.choice(HORIZONTAL_VARIANTS).copy()


def _random_rot180_shape() -> np.ndarray:
    return random.choice(ROT180_VARIANTS).copy()


def _random_symmetric_shape() -> np.ndarray:
    return random.choice(ALL_SYM_VARIANTS).copy()


def _random_asymmetric_shape() -> np.ndarray:
    return random.choice(ASYM_VARIANTS).copy()


def _build_lattice(chamber_rows: int, chamber_cols: int, chamber_size: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = chamber_rows * (chamber_size + 1) + 1
    w = chamber_cols * (chamber_size + 1) + 1
    grid = np.full((h, w), BLACK, dtype=int)

    for r in range(0, h, chamber_size + 1):
        grid[r, :] = TEAL
    for c in range(0, w, chamber_size + 1):
        grid[:, c] = TEAL

    chambers = []
    for rr in range(chamber_rows):
        for cc in range(chamber_cols):
            r0 = rr * (chamber_size + 1) + 1
            c0 = cc * (chamber_size + 1) + 1
            chambers.append((r0, c0, r0 + chamber_size - 1, c0 + chamber_size - 1))
    return grid, chambers


def _place_sprite(grid: np.ndarray, chamber: tuple[int, int, int, int], shape: np.ndarray, color: int) -> tuple[int, int]:
    r0, c0, r1, c1 = chamber
    chamber_h = r1 - r0 + 1
    chamber_w = c1 - c0 + 1
    shape_h, shape_w = shape.shape
    dr = random.randint(0, chamber_h - shape_h)
    dc = random.randint(0, chamber_w - shape_w)

    for sr, sc in np.argwhere(shape):
        grid[r0 + dr + int(sr), c0 + dc + int(sc)] = color
    return dr, dc


def _render_expected(grid: np.ndarray, metadata: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in metadata:
        r0, c0, r1, c1 = item["chamber"]
        out[r0 : r1 + 1, c0 : c1 + 1] = item["color"] if item["symmetric"] else BLACK
    return out


def _shape_area(shape: np.ndarray) -> int:
    return int(np.count_nonzero(shape))


def _validate_instance(grid: np.ndarray, expected: np.ndarray, metadata: list[dict]) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    cavities = _cavity_components(grid)
    if len(cavities) != len(metadata):
        return False

    status_counts = {True: 0, False: 0}
    colors_by_status = {True: set(), False: set()}
    areas_by_status = {True: set(), False: set()}
    offsets = set()

    for cavity, item in zip(cavities, metadata):
        info = _analyze_cavity(grid, cavity)
        if info["bbox"] != item["chamber"]:
            return False
        if info["color"] != item["color"]:
            return False
        if info["symmetric"] != item["symmetric"]:
            return False

        status_counts[item["symmetric"]] += 1
        colors_by_status[item["symmetric"]].add(item["color"])
        areas_by_status[item["symmetric"]].add(item["area"])
        offsets.add(item["offset"])

    if status_counts[True] < 3 or status_counts[False] < 3:
        return False
    if not (colors_by_status[True] & colors_by_status[False]):
        return False
    if not (areas_by_status[True] & areas_by_status[False]):
        return False
    if len(offsets) < 2:
        return False
    return True


def _generate_statuses(count: int) -> list[bool]:
    n_sym = random.randint(3, count - 3)
    statuses = [True] * n_sym + [False] * (count - n_sym)
    random.shuffle(statuses)
    return statuses


def _generate_shapes_and_colors(statuses: list[bool]) -> tuple[list[np.ndarray], list[int]]:
    count = len(statuses)
    shapes: list[np.ndarray | None] = [None] * count
    colors: list[int | None] = [None] * count

    sym_indices = [i for i, status in enumerate(statuses) if status]
    asym_indices = [i for i, status in enumerate(statuses) if not status]
    random.shuffle(sym_indices)
    random.shuffle(asym_indices)

    bridge_color = random.choice(SPRITE_COLORS)
    bridge_sym = sym_indices[0]
    horiz_sym = sym_indices[1]
    rot_sym = sym_indices[2]
    bridge_asym = asym_indices[0]

    shapes[bridge_sym] = VERTICAL_BASES[0].copy()
    shapes[horiz_sym] = HORIZONTAL_BASES[0].copy()
    shapes[rot_sym] = ROT180_BASES[0].copy()
    shapes[bridge_asym] = ASYM_BASES[1].copy()

    colors[bridge_sym] = bridge_color
    colors[bridge_asym] = bridge_color

    for idx in sym_indices[3:]:
        shapes[idx] = _random_symmetric_shape()
    for idx in asym_indices[1:]:
        shapes[idx] = _random_asymmetric_shape()

    for idx in range(count):
        if colors[idx] is None:
            colors[idx] = random.choice(SPRITE_COLORS)
        if shapes[idx] is None:
            shapes[idx] = _random_symmetric_shape() if statuses[idx] else _random_asymmetric_shape()

    return [shape.copy() for shape in shapes], [int(color) for color in colors]


def _gen_attempt() -> np.ndarray | None:
    chamber_rows = random.choice((3, 4))
    chamber_cols = random.choice((3, 4))
    chamber_size = random.choice((3, 4))

    grid, chambers = _build_lattice(chamber_rows, chamber_cols, chamber_size)
    statuses = _generate_statuses(len(chambers))
    shapes, colors = _generate_shapes_and_colors(statuses)

    metadata = []
    for chamber, symmetric, shape, color in zip(chambers, statuses, shapes, colors):
        if shape.shape[0] > chamber_size or shape.shape[1] > chamber_size:
            return None
        offset = _place_sprite(grid, chamber, shape, color)
        metadata.append(
            {
                "chamber": chamber,
                "symmetric": symmetric,
                "shape": shape,
                "color": color,
                "area": _shape_area(shape),
                "offset": offset,
            }
        )

    expected = _render_expected(grid, metadata)
    if not _validate_instance(grid, expected, metadata):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
