from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GUIDE_COLORS = {BLUE, RED}
SIDE_ORDER = ("top", "right", "bottom", "left")
ADJACENT_SIDE_SETS = {
    frozenset(("top", "left")),
    frozenset(("top", "right")),
    frozenset(("bottom", "left")),
    frozenset(("bottom", "right")),
}
D4_TRANSFORMS = tuple((k, flip) for k in range(4) for flip in (False, True))
HARD_TRANSFORMS = ((1, False), (1, True), (3, False), (3, True))
CORE_SIZE_RANGE = range(4, 9)
GRID_SIZE_RANGE = range(11, 17)
MAX_GENERATION_ATTEMPTS = 400


def _side_values(arr: np.ndarray, side: str, *, exclude_corners: bool) -> np.ndarray:
    if side == "top":
        vals = arr[0, :]
    elif side == "bottom":
        vals = arr[-1, :]
    elif side == "left":
        vals = arr[:, 0]
    else:
        vals = arr[:, -1]
    if exclude_corners and vals.size >= 3:
        return vals[1:-1]
    return vals


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _sides_are_adjacent(sides: list[str] | tuple[str, ...] | set[str]) -> bool:
    return frozenset(sides) in ADJACENT_SIDE_SETS


def _detect_control_sides(grid: np.ndarray) -> dict[str, int]:
    controls: dict[str, int] = {}
    for side in SIDE_ORDER:
        vals = _side_values(grid, side, exclude_corners=True)
        unique = {int(v) for v in vals}
        if len(unique) == 1:
            color = unique.pop()
            if color != BG:
                controls[side] = color
    assert len(controls) == 2, "expected exactly two colored border controls"
    assert len(set(controls.values())) == 2, "control colors must be distinct"
    assert _sides_are_adjacent(list(controls)), "control sides must be adjacent"
    return controls


def _extract_object(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid != BG)
    interior = [comp for comp in comps if not _touches_border(comp, grid.shape)]
    assert len(interior) == 1, "expected exactly one interior object"
    r0, c0, r1, c1 = _bbox(interior[0])
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _detect_guide_sides(obj: np.ndarray, guide_colors: set[int]) -> dict[str, int]:
    guides: dict[str, int] = {}
    for side in SIDE_ORDER:
        vals = _side_values(obj, side, exclude_corners=False)
        counts = {color: int(np.count_nonzero(vals == color)) for color in guide_colors}
        color, count = max(counts.items(), key=lambda item: item[1])
        if count >= vals.size - 1:
            guides[side] = color
    assert len(guides) == 2, "expected exactly two guide sides"
    assert len(set(guides.values())) == 2, "guide colors must be distinct"
    assert _sides_are_adjacent(list(guides)), "guide sides must be adjacent"
    return guides


def _apply_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _crop_guides(aligned_obj: np.ndarray, guide_sides: dict[str, int]) -> np.ndarray:
    r0 = 1 if "top" in guide_sides else 0
    r1 = aligned_obj.shape[0] - (1 if "bottom" in guide_sides else 0)
    c0 = 1 if "left" in guide_sides else 0
    c1 = aligned_obj.shape[1] - (1 if "right" in guide_sides else 0)
    core = aligned_obj[r0:r1, c0:c1].copy()
    assert core.size > 0
    return core


def _scale2(arr: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def _aligned_candidates(obj: np.ndarray, desired: dict[str, int]) -> list[np.ndarray]:
    matches = []
    for k, flip in D4_TRANSFORMS:
        transformed = _apply_transform(obj, k, flip)
        if _detect_guide_sides(transformed, set(desired.values())) == desired:
            matches.append(transformed)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate/reflect the guided central rectangle to match the border controls, crop the guides, and scale the core by 2.

    Input structure:
    - Black background with exactly two adjacent non-black border edges, each a solid control color.
    - One interior rectangular object whose body is green and whose two adjacent 1-pixel guide edges use those control colors.

    Rule:
    - Read which border side belongs to each control color.
    - Apply the unique dihedral transform to the interior object so each guide color faces the matching border side.
    - Crop away the two matched guide strips and 2x-scale the remaining rectangular core pixel-for-pixel.

    Key insight:
    - The colored guide edges are temporary orientation metadata: they determine the only valid transform, then must be removed before scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    desired = _detect_control_sides(g)
    obj = _extract_object(g)
    matches = _aligned_candidates(obj, desired)
    assert len(matches) == 1, "expected a unique matching dihedral transform"
    return _scale2(_crop_guides(matches[0], desired))


def _paint_border_side(grid: np.ndarray, side: str, color: int) -> None:
    if side == "top":
        grid[0, :] = color
    elif side == "bottom":
        grid[-1, :] = color
    elif side == "left":
        grid[:, 0] = color
    else:
        grid[:, -1] = color


def _build_aligned_object(core_h: int, core_w: int, desired: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in desired else 0
    bottom = 1 if "bottom" in desired else 0
    left = 1 if "left" in desired else 0
    right = 1 if "right" in desired else 0
    obj = np.full((core_h + top + bottom, core_w + left + right), GREEN, dtype=int)
    for side in SIDE_ORDER:
        if side in desired:
            _paint_border_side(obj, side, desired[side])
    return obj


def _expected_output_from_core(core_h: int, core_w: int) -> np.ndarray:
    return np.full((core_h * 2, core_w * 2), GREEN, dtype=int)


def _place_object(grid: np.ndarray, obj: np.ndarray) -> None:
    h, w = obj.shape
    gh, gw = grid.shape
    r0 = random.randint(2, gh - h - 2)
    c0 = random.randint(2, gw - w - 2)
    grid[r0 : r0 + h, c0 : c0 + w] = obj


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    desired = _detect_control_sides(grid)
    obj = _extract_object(grid)
    input_guides = _detect_guide_sides(obj, set(desired.values()))
    matches = _aligned_candidates(obj, desired)
    if len(matches) != 1:
        return False
    aligned = matches[0]
    if not np.array_equal(_scale2(_crop_guides(aligned, desired)), expected):
        return False
    naive = _scale2(_crop_guides(obj, input_guides))
    if np.array_equal(naive, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    core_h = random.choice(tuple(CORE_SIZE_RANGE))
    core_w = random.choice(tuple(v for v in CORE_SIZE_RANGE if v != core_h))

    side_pair = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )
    colors = [BLUE, RED]
    random.shuffle(colors)
    desired = {side_pair[0]: colors[0], side_pair[1]: colors[1]}

    aligned_obj = _build_aligned_object(core_h, core_w, desired)
    transform = random.choice(HARD_TRANSFORMS)
    input_obj = _inverse_transform(aligned_obj, *transform)

    grid_h_choices = [n for n in GRID_SIZE_RANGE if n >= input_obj.shape[0] + 5]
    grid_w_choices = [n for n in GRID_SIZE_RANGE if n >= input_obj.shape[1] + 5]
    if not grid_h_choices or not grid_w_choices:
        return None

    grid = np.zeros((random.choice(grid_h_choices), random.choice(grid_w_choices)), dtype=int)
    for side in SIDE_ORDER:
        if side in desired:
            _paint_border_side(grid, side, desired[side])
    _place_object(grid, input_obj)

    expected = _expected_output_from_core(core_h, core_w)
    if not _semantic_checks(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
