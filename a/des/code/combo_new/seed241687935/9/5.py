from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

HORIZONTAL_ONLY = "horizontal_only"
VERTICAL_ONLY = "vertical_only"
BOTH_AXES = "both_axes"
NO_SYMMETRY = "no_symmetry"

OBJECT_KINDS = (HORIZONTAL_ONLY, VERTICAL_ONLY, BOTH_AXES, NO_SYMMETRY)
KIND_TO_COLOR = {
    HORIZONTAL_ONLY: BLUE,
    VERTICAL_ONLY: RED,
    BOTH_AXES: YELLOW,
    NO_SYMMETRY: MAGENTA,
}

GRID_SIZE_RANGE = (20, 30)
EXTRA_OBJECTS_RANGE = (0, 3)
SHAPE_SIZE_CHOICES = (3, 5)
MAX_SHAPE_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[arr[:, 0] - r0, arr[:, 1] - c0] = True
            components.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "area": int(local.sum()),
                }
            )

    return components


def _legend_role(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    area = int(mask.sum())
    if area == 1:
        return NO_SYMMETRY
    if h == 1 and w >= 2 and area == w:
        return HORIZONTAL_ONLY
    if w == 1 and h >= 2 and area == h:
        return VERTICAL_ONLY
    if h % 2 == 1 and w % 2 == 1 and h >= 3 and w >= 3:
        cross = np.zeros_like(mask, dtype=bool)
        cross[h // 2, :] = True
        cross[:, w // 2] = True
        if np.array_equal(mask, cross):
            return BOTH_AXES
    return None


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    if horizontal and vertical:
        return BOTH_AXES
    if horizontal:
        return HORIZONTAL_ONLY
    if vertical:
        return VERTICAL_ONLY
    return NO_SYMMETRY


def _infer_palette(g: np.ndarray) -> tuple[dict[str, int], int]:
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    by_color = {color: _find_components(g == color) for color in colors}

    role_to_color: dict[str, int] = {}
    legend_colors: set[int] = set()
    for color, comps in by_color.items():
        if len(comps) != 1:
            continue
        role = _legend_role(comps[0]["mask"])
        if role is None:
            continue
        assert role not in role_to_color, "legend roles must be unique"
        role_to_color[role] = int(color)
        legend_colors.add(int(color))

    assert set(role_to_color) == set(OBJECT_KINDS), "expected point, line, and cross legend markers"
    object_colors = [color for color in colors if color not in legend_colors]
    assert len(object_colors) == 1, "expected exactly one object color outside the legend"
    return role_to_color, int(object_colors[0])


def _draw_marker(out: np.ndarray, bbox: tuple[int, int, int, int], kind: str, color: int) -> None:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if kind == HORIZONTAL_ONLY:
        out[center_r, c0 : c1 + 1] = color
    elif kind == VERTICAL_ONLY:
        out[r0 : r1 + 1, center_c] = color
    elif kind == BOTH_AXES:
        out[center_r, c0 : c1 + 1] = color
        out[r0 : r1 + 1, center_c] = color
    else:
        out[center_r, center_c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each green shape's reflection symmetry and overlay the corresponding axis or center marker from the corner legend.

    Input structure:
    - The background is black.
    - Several small non-overlapping objects are drawn in one shared foreground color.
    - A corner legend contains four separate markers: a horizontal segment, a vertical segment, a cross, and a single point, each in a different color.

    Rule:
    - Infer the legend mapping from each marker's visible shape rather than from its color id.
    - Treat the remaining non-background color as the object color and split it into 4-connected components.
    - For each object, test whether its tight bounding-box mask is horizontally symmetric, vertically symmetric, both, or neither.
    - Keep the input unchanged except for overlaying the matching legend color: a full-width horizontal line, a full-height vertical line, a full cross, or a single center pixel.

    Key insight:
    - The corner markers are not examples to copy literally; they define a symbolic mapping from visible marker shape to symmetry type, and that mapping must then be applied to each green object's own bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    role_to_color, object_color = _infer_palette(g)
    components = _find_components(g == object_color)
    assert components, "expected at least one object component"

    out = g.copy()
    for comp in components:
        kind = _symmetry_kind(comp["mask"])
        _draw_marker(out, comp["bbox"], kind, role_to_color[kind])
    return out


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _grow_connected(h: int, w: int, start: tuple[int, int], target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    occupied = {start}
    frontier = set(_neighbors(start[0], start[1], h, w))
    mask[start] = True

    while len(occupied) < target_area:
        if not frontier:
            return np.zeros((h, w), dtype=bool)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in occupied:
            continue
        occupied.add(cell)
        mask[cell] = True
        for nbr in _neighbors(cell[0], cell[1], h, w):
            if nbr not in occupied:
                frontier.add(nbr)
    return mask


def _mirror_horizontal(base: np.ndarray) -> np.ndarray:
    top_h, w = base.shape
    full_h = 2 * top_h - 1
    out = np.zeros((full_h, w), dtype=bool)
    out[:top_h, :] = base
    out[top_h:, :] = np.flipud(base[:-1, :])
    return out


def _mirror_vertical(base: np.ndarray) -> np.ndarray:
    h, left_w = base.shape
    full_w = 2 * left_w - 1
    out = np.zeros((h, full_w), dtype=bool)
    out[:, :left_w] = base
    out[:, left_w:] = np.fliplr(base[:, :-1])
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    comps = _find_components(mask)
    return len(comps) == 1


def _random_area(h: int, w: int) -> int:
    max_area = h * w - 1
    min_area = max(4, (h * w + 2) // 3)
    return random.randint(min_area, max_area)


def _generate_shape(kind: str) -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.choice(SHAPE_SIZE_CHOICES)
        w = random.choice(SHAPE_SIZE_CHOICES)

        if kind == BOTH_AXES:
            half_h = h // 2 + 1
            half_w = w // 2 + 1
            area = random.randint(3, half_h * half_w)
            base = _grow_connected(half_h, half_w, (half_h - 1, half_w - 1), area)
            mask = _mirror_vertical(_mirror_horizontal(base))
        elif kind == HORIZONTAL_ONLY:
            half_h = h // 2 + 1
            area = _random_area(half_h, w)
            start = (half_h - 1, random.randrange(w))
            base = _grow_connected(half_h, w, start, area)
            mask = _mirror_horizontal(base)
        elif kind == VERTICAL_ONLY:
            half_w = w // 2 + 1
            area = _random_area(h, half_w)
            start = (random.randrange(h), half_w - 1)
            base = _grow_connected(h, half_w, start, area)
            mask = _mirror_vertical(base)
        else:
            area = _random_area(h, w)
            start = (random.randrange(h), random.randrange(w))
            mask = _grow_connected(h, w, start, area)

        if not mask.any():
            continue
        if mask.all():
            continue
        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if _legend_role(mask) is not None:
            continue
        if _symmetry_kind(mask) != kind:
            continue
        return mask

    raise RuntimeError(f"failed to generate a {kind} shape")


def _legend_patch() -> np.ndarray:
    patch = np.zeros((7, 7), dtype=int)
    patch[1, 0:3] = BLUE
    patch[0:3, 5] = RED
    patch[4, 0:3] = YELLOW
    patch[3:6, 1] = YELLOW
    patch[5, 5] = MAGENTA
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, corner: str) -> tuple[int, int]:
    h, w = grid.shape
    ph, pw = patch.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return r0, c0


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _render_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> tuple[int, int, int, int]:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = color
    return top, left, top + h - 1, left + w - 1


def _place_objects(
    shape_specs: list[tuple[str, np.ndarray]],
    grid_shape: tuple[int, int],
    blocked: np.ndarray,
) -> list[tuple[str, np.ndarray, tuple[int, int, int, int]]]:
    h, w = grid_shape
    placed: list[tuple[str, np.ndarray, tuple[int, int, int, int]]] = []

    for kind, mask in shape_specs:
        mh, mw = mask.shape
        candidates = []
        for top in range(h - mh + 1):
            for left in range(w - mw + 1):
                if blocked[top : top + mh, left : left + mw].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return []

        top, left = random.choice(candidates)
        bbox = (top, left, top + mh - 1, left + mw - 1)
        _reserve(blocked, *bbox)
        placed.append((kind, mask, bbox))

    return placed


def _expected_output(inp: np.ndarray, placed: list[tuple[str, np.ndarray, tuple[int, int, int, int]]]) -> np.ndarray:
    out = inp.copy()
    for kind, _mask, bbox in placed:
        _draw_marker(out, bbox, kind, KIND_TO_COLOR[kind])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    patch = _legend_patch()
    if h < patch.shape[0] or w < patch.shape[1]:
        return None

    blocked = np.zeros_like(grid, dtype=bool)
    corner = random.choice(("tl", "tr", "bl", "br"))
    patch_r0, patch_c0 = _place_patch(grid, patch, corner)
    _reserve(blocked, patch_r0, patch_c0, patch_r0 + patch.shape[0] - 1, patch_c0 + patch.shape[1] - 1)

    kinds = list(OBJECT_KINDS)
    extra_count = random.randint(*EXTRA_OBJECTS_RANGE)
    kinds.extend(random.choice(OBJECT_KINDS) for _ in range(extra_count))
    random.shuffle(kinds)

    shape_specs = [(kind, _generate_shape(kind)) for kind in kinds]
    shape_specs.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    placed = _place_objects(shape_specs, grid.shape, blocked)
    if not placed:
        return None

    for kind, mask, bbox in placed:
        r0, c0, _r1, _c1 = bbox
        _render_object(grid, mask, r0, c0, GREEN)

    expected = _expected_output(grid, placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-marker ARC input")
