from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


BG = 0
RED = 2
CONTROL = 8
INPUT_COLORS = (1, 3, 4, 6, 7, 9)
CORNERS = ("tl", "tr", "bl", "br")
LEFT_CORNERS = {"tl", "bl"}
TOP_CORNERS = {"tl", "tr"}


def get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_four_connected(mask: np.ndarray) -> bool:
    return len(_components_from_mask(mask)) == 1


@dataclass(frozen=True)
class ParsedObject:
    coords: np.ndarray
    bbox: BBox
    base_color: int
    control: Coord
    corner: str


def _corner_from_control(b: BBox, control: Coord) -> str:
    r0, c0, r1, c1 = b
    corners = {
        (r0, c0): "tl",
        (r0, c1): "tr",
        (r1, c0): "bl",
        (r1, c1): "br",
    }
    if control not in corners:
        raise AssertionError("control pixel must occupy a bbox corner")
    return corners[control]


def _parse_objects(grid: np.ndarray, bg: int) -> list[ParsedObject]:
    objects: list[ParsedObject] = []
    for coords in _components_from_mask(grid != bg):
        values = grid[coords[:, 0], coords[:, 1]]
        control_coords = coords[values == CONTROL]
        if control_coords.shape[0] != 1:
            raise AssertionError("each object must contain exactly one control pixel")
        base_colors = {int(v) for v in values if int(v) != CONTROL}
        if len(base_colors) != 1:
            raise AssertionError("each object must be monochrome apart from its control")
        b = bbox_coords(coords)
        control = tuple(map(int, control_coords[0]))
        objects.append(
            ParsedObject(
                coords=coords,
                bbox=b,
                base_color=base_colors.pop(),
                control=control,
                corner=_corner_from_control(b, control),
            )
        )
    if not objects:
        raise AssertionError("expected at least one foreground object")
    return sorted(objects, key=lambda obj: (obj.bbox[0], obj.bbox[1], obj.base_color))


def _shear_pixels(
    obj: ParsedObject,
    shape: tuple[int, int],
    *,
    anchor_policy: str = "control",
    apply_shear: bool = True,
) -> list[tuple[int, int, int]]:
    h, w = shape
    r0, _c0, r1, _c1 = obj.bbox
    if anchor_policy == "control":
        anchor_r = r0 if obj.corner in TOP_CORNERS else r1
    elif anchor_policy == "top":
        anchor_r = r0
    elif anchor_policy == "bottom":
        anchor_r = r1
    else:
        raise ValueError(f"unknown anchor policy: {anchor_policy}")

    direction = -1 if obj.corner in LEFT_CORNERS else 1
    out: list[tuple[int, int, int]] = []
    for r, c in obj.coords:
        dc = direction * abs(int(r) - int(anchor_r)) if apply_shear else 0
        nr = int(r)
        nc = int(c) + int(dc)
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc, int(c)))
    if not out:
        raise AssertionError("shear clipped away an entire object")
    return out


def _paint_object(
    canvas: np.ndarray,
    obj: ParsedObject,
    pixels: list[tuple[int, int, int]],
    *,
    half_policy: str = "control",
    strict_collision: bool = True,
) -> None:
    placed = np.array([(r, c) for r, c, _src_c in pixels], dtype=int)
    b = bbox_coords(placed)
    _r0, c0, _r1, c1 = b
    current_mid = (int(c0) + int(c1)) // 2
    orig_mid = (int(obj.bbox[1]) + int(obj.bbox[3])) // 2

    if half_policy == "control":
        make_red_left = obj.corner in LEFT_CORNERS
    elif half_policy == "left":
        make_red_left = True
    elif half_policy == "right":
        make_red_left = False
    elif half_policy == "original":
        make_red_left = obj.corner in LEFT_CORNERS
    else:
        raise ValueError(f"unknown half policy: {half_policy}")

    for r, c, src_c in pixels:
        if half_policy == "original":
            in_left = int(src_c) <= int(orig_mid)
        else:
            in_left = int(c) <= int(current_mid)
        color = RED if (in_left == make_red_left) else obj.base_color
        if canvas[r, c] != BG:
            if strict_collision:
                raise AssertionError("transformed objects must not overlap")
            canvas[r, c] = -1
            continue
        canvas[r, c] = int(color)


def _render(
    grid: np.ndarray,
    *,
    anchor_policy: str = "control",
    half_policy: str = "control",
    apply_shear: bool = True,
    strict_collision: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full(g.shape, bg, dtype=int)
    for obj in _parse_objects(g, bg):
        pixels = _shear_pixels(obj, g.shape, anchor_policy=anchor_policy, apply_shear=apply_shear)
        _paint_object(out, obj, pixels, half_policy=half_policy, strict_collision=strict_collision)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shear each control-marked object from its indicated corner, then recolor the matching half of the sheared bbox red.

    Input structure:
    - Black background with 2 or 3 separated 4-connected foreground objects.
    - Each object is monochrome except for one cyan control pixel that replaces one corner cell of the object's bbox.
    - The control corner can be upper-left, upper-right, lower-left, or lower-right.

    Rule:
    - Treat the cyan cell as part of its object's original color.
    - Use the control corner to choose the anchor row and horizontal shear direction: upper corners anchor the top row, lower corners anchor the bottom row; left corners shear left, right corners shear right; each row shifts by its distance from the anchor row.
    - After shearing, compute that object's new bbox and recolor the control-side half of that new bbox red, keeping the opposite half in the object's original color.

    Key insight:
    - The same corner marker jointly encodes three things: which row stays fixed during shear, which horizontal direction the shear goes, and which half of the sheared bbox is recolored.
    """
    return _render(grid)


def _mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_TEMPLATES: dict[int, tuple[np.ndarray, ...]] = {
    4: (
        _mask(("1111", "1100", "0110", "0011")),
        _mask(("1110", "1010", "1111", "0011")),
        _mask(("1111", "1001", "1001", "1111")),
        _mask(("1100", "1110", "1011", "0011")),
        _mask(("1110", "0010", "1111", "1001")),
    ),
    6: (
        _mask(("1111", "1100", "1110", "0110", "0011", "0011")),
        _mask(("1111", "1001", "1111", "1100", "0110", "0011")),
        _mask(("1111", "1001", "1001", "1001", "1001", "1111")),
        _mask(("1110", "1010", "1111", "0111", "0011", "0010")),
        _mask(("1100", "1110", "1010", "1111", "0011", "0011")),
        _mask(("1111", "0011", "0111", "1110", "1100", "1100")),
    ),
}


for _height, _templates in BASE_TEMPLATES.items():
    for _template in _templates:
        assert _template.shape == (_height, 4)
        assert _template[0, 0]
        assert int(_template.sum()) in range(8, 21)
        assert _is_four_connected(_template)
        assert bbox_coords(np.argwhere(_template)) == (0, 0, _height - 1, 3)
        assert not np.all(_template)


def _orient_mask(mask: np.ndarray, corner: str) -> np.ndarray:
    out = mask
    if corner in {"tr", "br"}:
        out = np.fliplr(out)
    if corner in {"bl", "br"}:
        out = np.flipud(out)
    return out.copy()


def _grid_signature(grid: np.ndarray) -> bytes:
    return grid.tobytes() + repr(tuple(int(x) for x in grid.shape)).encode()


def _random_composition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _choose_corners(n_objects: int) -> list[str]:
    if n_objects == 2:
        pair = random.choice((("tl", "br"), ("tr", "bl")))
        corners = list(pair)
        random.shuffle(corners)
        return corners

    while True:
        corners = random.sample(CORNERS, 3)
        if any(c in TOP_CORNERS for c in corners) and any(c not in TOP_CORNERS for c in corners):
            if any(c in LEFT_CORNERS for c in corners) and any(c not in LEFT_CORNERS for c in corners):
                return corners


def _choose_row_positions(heights: list[int], total_rows: int) -> list[int]:
    min_slots = [1] + [3] * (len(heights) - 1) + [1]
    min_rows = sum(heights) + sum(min_slots)
    extra = int(total_rows) - int(min_rows)
    if extra < 0:
        raise ValueError("row budget too small")
    extra_slots = _random_composition(extra, len(min_slots))
    slots = [base + add for base, add in zip(min_slots, extra_slots)]

    rows: list[int] = []
    cursor = slots[0]
    for idx, height in enumerate(heights):
        rows.append(int(cursor))
        cursor += int(height)
        if idx < len(heights) - 1:
            cursor += int(slots[idx + 1])
    return rows


def _safe_column_bounds(width: int, height: int, corner: str) -> tuple[int, int]:
    shift = int(height) - 1
    if corner in LEFT_CORNERS:
        return shift, width - 4
    return 0, width - 4 - shift


def _clipped_column_bounds(width: int, height: int, corner: str) -> tuple[int, int]:
    shift = int(height) - 1
    if corner in LEFT_CORNERS:
        return 0, max(0, shift - 1)
    lo = max(0, width - 4 - shift + 1)
    return lo, width - 4


def _place_patch(grid: np.ndarray, mask: np.ndarray, row: int, col: int, color: int, corner: str) -> None:
    grid[row : row + mask.shape[0], col : col + mask.shape[1]][mask] = int(color)
    control = {
        "tl": (row, col),
        "tr": (row, col + mask.shape[1] - 1),
        "bl": (row + mask.shape[0] - 1, col),
        "br": (row + mask.shape[0] - 1, col + mask.shape[1] - 1),
    }[corner]
    grid[control] = CONTROL


def _object_nondegenerate(inp: np.ndarray, out: np.ndarray, obj: ParsedObject) -> bool:
    pixels = _shear_pixels(obj, inp.shape)
    placed = np.array([(r, c) for r, c, _src_c in pixels], dtype=int)
    if bbox_coords(placed) == obj.bbox:
        return False

    b = bbox_coords(placed)
    mid = (b[1] + b[3]) // 2
    left_count = sum(int(c) <= int(mid) for _r, c, _src_c in pixels)
    right_count = len(pixels) - left_count
    recolor_count = left_count if obj.corner in LEFT_CORNERS else right_count
    keep_count = len(pixels) - recolor_count
    if recolor_count == 0 or keep_count == 0:
        return False

    orig_variant = np.full(inp.shape, BG, dtype=int)
    _paint_object(orig_variant, obj, pixels, half_policy="original", strict_collision=False)
    true_variant = np.full(inp.shape, BG, dtype=int)
    _paint_object(true_variant, obj, pixels, half_policy="control", strict_collision=False)
    if np.array_equal(orig_variant, true_variant):
        return False

    if not np.any(out == RED):
        return False
    return True


def _semantic_checks(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    bg = get_bg(inp)
    if bg != BG:
        return False

    try:
        objects = _parse_objects(inp, bg)
        out = main(inp)
    except AssertionError:
        return False

    if len(objects) not in {2, 3}:
        return False

    row_sides = {obj.corner in TOP_CORNERS for obj in objects}
    col_sides = {obj.corner in LEFT_CORNERS for obj in objects}
    if row_sides != {False, True} or col_sides != {False, True}:
        return False

    used_colors = {obj.base_color for obj in objects}
    if len(used_colors) != len(objects):
        return False

    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        if (r1 - r0 + 1) not in {4, 6} or (c1 - c0 + 1) != 4:
            return False
        area = int(obj.coords.shape[0])
        if not (8 <= area <= 20):
            return False
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = obj.coords - np.array([r0, c0])
        patch[rel[:, 0], rel[:, 1]] = True
        if np.all(patch):
            return False
        if not _object_nondegenerate(inp, out, obj):
            return False

    variants = (
        _render(inp, apply_shear=False, strict_collision=False),
        _render(inp, anchor_policy="top", strict_collision=False),
        _render(inp, anchor_policy="bottom", strict_collision=False),
        _render(inp, half_policy="left", strict_collision=False),
        _render(inp, half_policy="right", strict_collision=False),
        _render(inp, half_policy="original", strict_collision=False),
    )
    if any(np.array_equal(out, alt) for alt in variants):
        return False

    return True


def _attempt_input() -> np.ndarray | None:
    n_objects = 2 if random.random() < 0.65 else 3
    corners = _choose_corners(n_objects)
    random.shuffle(corners)
    heights = [random.choice((4, 6)) for _ in range(n_objects)]
    colors = random.sample(INPUT_COLORS, n_objects)
    masks = [_orient_mask(random.choice(BASE_TEMPLATES[h]), corner) for h, corner in zip(heights, corners)]

    min_rows = sum(heights) + 3 * (n_objects - 1) + 2
    if n_objects == 2:
        rows = random.randint(max(14, min_rows), min(20, max(14, min_rows + 4)))
    else:
        rows = random.randint(max(19, min_rows), min(28, max(19, min_rows + 4)))
    cols = random.randint(14, 18)

    try:
        row_positions = _choose_row_positions(heights, rows)
    except ValueError:
        return None

    clip_index = random.randrange(n_objects) if random.random() < 0.2 else None
    col_positions: list[int] = []
    for idx, (height, corner) in enumerate(zip(heights, corners)):
        if idx == clip_index:
            lo, hi = _clipped_column_bounds(cols, height, corner)
            if lo > hi:
                lo, hi = _safe_column_bounds(cols, height, corner)
        else:
            lo, hi = _safe_column_bounds(cols, height, corner)
        if lo > hi:
            return None
        col_positions.append(random.randint(lo, hi))

    grid = np.zeros((rows, cols), dtype=int)
    for mask, row, col, color, corner in zip(masks, row_positions, col_positions, colors, corners):
        _place_patch(grid, mask, row, col, color, corner)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    seen: set[bytes] = set()
    for _ in range(500):
        grid = _attempt_input()
        if grid is None:
            continue
        seen.add(_grid_signature(grid))
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
