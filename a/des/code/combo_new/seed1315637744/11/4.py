from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 9
BODY_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
TRANSFORMS = (
    "flip_ud",
    "rot_cw",
    "identity",
    "rot_ccw",
    "rot_180",
    "flip_lr",
)
SLOT_STEP = 9
MAX_ATTEMPTS = 300


def _find_components(mask: np.ndarray) -> list[dict]:
    """Return 8-connected components of a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS8:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _apply_transform(patch: np.ndarray, transform: str) -> np.ndarray:
    patch = np.asarray(patch)
    if transform == "identity":
        return patch.copy()
    if transform == "flip_ud":
        return np.flipud(patch)
    if transform == "flip_lr":
        return np.fliplr(patch)
    if transform == "rot_cw":
        assert patch.shape[0] == patch.shape[1], "90-degree rotation requires a square bbox"
        return np.rot90(patch, -1)
    if transform == "rot_ccw":
        assert patch.shape[0] == patch.shape[1], "90-degree rotation requires a square bbox"
        return np.rot90(patch, 1)
    assert transform == "rot_180"
    return np.rot90(patch, 2)


def _marker_positions_for_transform(size: int, transform: str) -> tuple[tuple[int, int], ...]:
    mid = size // 2
    mapping = {
        "flip_ud": ((0, mid),),
        "rot_cw": ((mid, size - 1),),
        "identity": ((size - 1, mid),),
        "rot_ccw": ((mid, 0),),
        "rot_180": ((0, 0), (size - 1, size - 1)),
        "flip_lr": ((size - 1, 0), (0, size - 1)),
    }
    return mapping[transform]


def _infer_transform(marker_coords: list[tuple[int, int]], shape: tuple[int, int]) -> str:
    h, w = map(int, shape)
    markers = {(int(r), int(c)) for r, c in marker_coords}

    if len(markers) == 1:
        (r, c) = next(iter(markers))
        if w % 2 == 1 and (r, c) == (0, w // 2):
            return "flip_ud"
        if h % 2 == 1 and (r, c) == (h // 2, w - 1):
            return "rot_cw"
        if w % 2 == 1 and (r, c) == (h - 1, w // 2):
            return "identity"
        if h % 2 == 1 and (r, c) == (h // 2, 0):
            return "rot_ccw"
    elif len(markers) == 2:
        if markers == {(0, 0), (h - 1, w - 1)}:
            return "rot_180"
        if markers == {(h - 1, 0), (0, w - 1)}:
            return "flip_lr"

    raise AssertionError(f"unrecognized marker layout: {sorted(markers)} for shape {shape}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each marked monochrome shape in place according to the maroon control pixels on its boundary.

    Input structure:
    - The background is black.
    - Each foreground object is a monochrome shape together with one or two maroon marker pixels attached on the shape boundary.
    - Different shapes are disconnected from one another, and each object's tight bounding box is the frame used for its transformation.

    Rule:
    - For each connected object, treat maroon pixels as control markers rather than body color.
    - A top-middle marker means vertical flip, a right-middle marker means clockwise 90-degree rotation, a bottom-middle marker means no geometric change, a left-middle marker means counterclockwise 90-degree rotation.
    - Two markers on top-left and bottom-right mean 180-degree rotation; two markers on bottom-left and top-right mean horizontal flip.
    - Replace marker pixels with the object's own color, apply the indicated transform about the bounding-box center, and redraw the transformed shape back into the same bounding box.

    Key insight:
    - The maroon pixels encode an operation by where they sit on the object's boundary, so solving the task means separating signal pixels from the true body before transforming the whole bbox patch.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for component in _find_components(g != BG_COLOR):
        r0, c0, r1, c1 = map(int, component["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        patch = np.full((h, w), BG_COLOR, dtype=int)

        colors = {int(g[r, c]) for r, c in component["coords"]}
        assert MARKER_COLOR in colors, "each shape must contain maroon control markers"
        body_colors = colors - {MARKER_COLOR}
        assert len(body_colors) == 1, "each shape body must be monochrome"
        body_color = int(next(iter(body_colors)))

        marker_coords: list[tuple[int, int]] = []
        for r, c in component["coords"]:
            rr = int(r - r0)
            cc = int(c - c0)
            if int(g[r, c]) == MARKER_COLOR:
                marker_coords.append((rr, cc))
            patch[rr, cc] = body_color

        assert 1 <= len(marker_coords) <= 2, "expected one or two markers per object"
        transform = _infer_transform(marker_coords, patch.shape)
        transformed = _apply_transform(patch, transform)
        assert transformed.shape == patch.shape, "transformed patch must stay in its bbox"

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[:] = BG_COLOR
        region[transformed != BG_COLOR] = body_color

    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected_4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_boundary_cell(size: int, side: str, banned: set[tuple[int, int]]) -> tuple[int, int]:
    if side == "top":
        choices = [(0, c) for c in range(size)]
    elif side == "bottom":
        choices = [(size - 1, c) for c in range(size)]
    elif side == "left":
        choices = [(r, 0) for r in range(size)]
    else:
        choices = [(r, size - 1) for r in range(size)]

    choices = [cell for cell in choices if cell not in banned]
    if not choices:
        choices = [(0, 0)]
    return random.choice(choices)


def _connect_mask(size: int, anchors: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    order = anchors[:]
    random.shuffle(order)
    mask[order[0]] = True

    for target in order[1:]:
        existing = list(zip(*np.where(mask)))
        r, c = random.choice(existing)
        tr, tc = target
        while (r, c) != (tr, tc):
            options = []
            if r != tr:
                options.append((r + (1 if tr > r else -1), c))
            if c != tc:
                options.append((r, c + (1 if tc > c else -1)))
            random.shuffle(options)
            r, c = options[0]
            mask[r, c] = True

    return mask


def _grow_mask(mask: np.ndarray, target_area: int) -> np.ndarray:
    size = mask.shape[0]
    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break

        candidates = list(frontier)
        boundary_cells = [cell for cell in candidates if 0 in cell or size - 1 in cell]
        if boundary_cells and random.random() < 0.35:
            pick = random.choice(boundary_cells)
        else:
            pick = random.choice(candidates)
        mask[pick] = True

    return mask


def _make_shape_mask(size: int, marker_cells: tuple[tuple[int, int], ...], transform: str) -> np.ndarray | None:
    for _ in range(200):
        anchors = set(marker_cells)
        if not any(r == 0 for r, _ in anchors):
            anchors.add(_random_boundary_cell(size, "top", anchors))
        if not any(r == size - 1 for r, _ in anchors):
            anchors.add(_random_boundary_cell(size, "bottom", anchors))
        if not any(c == 0 for _, c in anchors):
            anchors.add(_random_boundary_cell(size, "left", anchors))
        if not any(c == size - 1 for _, c in anchors):
            anchors.add(_random_boundary_cell(size, "right", anchors))

        extra_count = random.randint(1, 3)
        all_cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in anchors]
        random.shuffle(all_cells)
        for cell in all_cells[:extra_count]:
            anchors.add(cell)

        mask = _connect_mask(size, list(anchors))
        min_area = max(int(mask.sum()) + 2, size * size // 3)
        max_area = size * size - 2
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        mask = _grow_mask(mask, target_area)

        if not _touches_all_sides(mask):
            continue
        if not _is_connected_4(mask):
            continue
        if any(not mask[r, c] for r, c in marker_cells):
            continue
        if int(mask.sum()) >= size * size - 1:
            continue

        body = mask.astype(int)
        if transform != "identity" and np.array_equal(body, _apply_transform(body, transform)):
            continue
        return mask

    return None


def _build_object_patch(color: int, transform: str) -> tuple[np.ndarray, np.ndarray] | None:
    size = random.choice((5, 5, 7))
    marker_cells = _marker_positions_for_transform(size, transform)
    mask = _make_shape_mask(size, marker_cells, transform)
    if mask is None:
        return None

    clean_patch = np.where(mask, color, BG_COLOR).astype(int)
    input_patch = clean_patch.copy()
    for r, c in marker_cells:
        input_patch[r, c] = MARKER_COLOR
    return input_patch, clean_patch


def _slot_origins(grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    base_h = SLOT_STEP * 3
    base_w = SLOT_STEP * 3
    pad_top = random.randint(0, grid_h - base_h)
    pad_left = random.randint(0, grid_w - base_w)
    return [
        (pad_top + slot_r * SLOT_STEP, pad_left + slot_c * SLOT_STEP)
        for slot_r in range(3)
        for slot_c in range(3)
    ]


def _place_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = canvas[top : top + h, left : left + w]
    assert region.shape == patch.shape
    assert not np.any((region != BG_COLOR) & (patch != BG_COLOR))
    region[patch != BG_COLOR] = patch[patch != BG_COLOR]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(27, 30)
    grid_w = random.randint(27, 30)
    canvas = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    expected = np.full_like(canvas, BG_COLOR)

    object_count = random.randint(6, 8)
    transforms = list(TRANSFORMS)
    while len(transforms) < object_count:
        transforms.append(random.choice(TRANSFORMS))
    random.shuffle(transforms)

    colors = random.sample(BODY_COLORS, object_count)
    slots = _slot_origins(grid_h, grid_w)
    slot_indices = random.sample(range(len(slots)), object_count)

    for slot_index, color, transform in zip(slot_indices, colors, transforms):
        built = _build_object_patch(int(color), transform)
        if built is None:
            return None
        input_patch, clean_patch = built

        slot_top, slot_left = slots[slot_index]
        size = input_patch.shape[0]
        inner_offset = random.randint(1, SLOT_STEP - size - 1)
        top = int(slot_top + inner_offset)
        left = int(slot_left + random.randint(1, SLOT_STEP - size - 1))

        _place_patch(canvas, input_patch, top, left)
        _place_patch(expected, _apply_transform(clean_patch, transform), top, left)

    if np.array_equal(canvas, expected):
        return None
    if not np.array_equal(main(canvas), expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marked-transform sample")
