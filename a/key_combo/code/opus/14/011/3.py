from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
ORANGE = 7
ACCENT_COLORS = (3, 4, 6, 8, 9)
SIDE_ORDER = ("top", "right", "bottom", "left")


TRANSFORMS = {
    "id": lambda a: a.copy(),
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "flip_lr": lambda a: np.fliplr(a),
    "flip_ud": lambda a: np.flipud(a),
    "transpose": lambda a: a.T.copy(),
    "anti_transpose": lambda a: np.fliplr(np.rot90(a, 1)),
}

SIDE_MAPS = {
    "id": {"top": "top", "right": "right", "bottom": "bottom", "left": "left"},
    "rot90": {"top": "left", "right": "top", "bottom": "right", "left": "bottom"},
    "rot180": {"top": "bottom", "right": "left", "bottom": "top", "left": "right"},
    "rot270": {"top": "right", "right": "bottom", "bottom": "left", "left": "top"},
    "flip_lr": {"top": "top", "right": "left", "bottom": "bottom", "left": "right"},
    "flip_ud": {"top": "bottom", "right": "right", "bottom": "top", "left": "left"},
    "transpose": {"top": "left", "right": "bottom", "bottom": "right", "left": "top"},
    "anti_transpose": {"top": "right", "right": "top", "bottom": "left", "left": "bottom"},
}

ROTATION_CODES = ("id", "rot90", "rot180", "rot270")
REFLECTION_CODES = ("flip_lr", "flip_ud", "transpose", "anti_transpose")
GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _line_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, tuple[int, ...]]]:
    lines = pattern if axis == 0 else pattern.T
    runs: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or not np.array_equal(lines[i], lines[start]):
            runs.append((start, i, tuple(map(int, lines[start]))))
            start = i
    return runs


def _stretch_run_index(runs: list[tuple[int, int, tuple[int, ...]]]) -> int:
    assert len(runs) >= 3, "expected boundary runs plus one internal stretch run"
    internal = list(range(1, len(runs) - 1))
    complexities = [len(set(line)) for _s, _e, line in runs]
    max_complexity = max(complexities[i] for i in internal)
    candidates = [i for i in internal if complexities[i] == max_complexity]
    if len(candidates) == 1:
        return candidates[0]

    lengths = [end - start for start, end, _line in runs]
    max_length = max(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == max_length]
    assert len(candidates) == 1, "expected a unique internal stretch run"
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed = sum(end - start for i, (start, end, _line) in enumerate(runs) if i != stretch_i)
    stretch_len = int(target_length) - fixed
    assert stretch_len > 0, "target is too small for the fixed boundary runs"

    indices: list[int] = []
    for i, (start, end, _line) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch_len)
        else:
            indices.extend(range(start, end))
    assert len(indices) == int(target_length)
    return indices


def _stretch_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = _axis_indices(pattern, 0, int(shape[0]))
    cols = _axis_indices(pattern, 1, int(shape[1]))
    return pattern[np.ix_(rows, cols)]


def _apply_transform(pattern: np.ndarray, code: str) -> np.ndarray:
    return TRANSFORMS[code](pattern)


def _mapped_side(side: str, code: str) -> str:
    return SIDE_MAPS[code][side]


def _adjacent_sides(a: str, b: str) -> bool:
    ia = SIDE_ORDER.index(a)
    ib = SIDE_ORDER.index(b)
    return (ia - ib) % 4 in (1, 3)


def _detect_guides(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, str] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    guides: dict[int, str] = {}

    checks = (
        (RED, "top", r0 > 0 and np.all(grid[r0 - 1, c0 : c1 + 1] == RED)),
        (RED, "bottom", r1 + 1 < h and np.all(grid[r1 + 1, c0 : c1 + 1] == RED)),
        (RED, "left", c0 > 0 and np.all(grid[r0 : r1 + 1, c0 - 1] == RED)),
        (RED, "right", c1 + 1 < w and np.all(grid[r0 : r1 + 1, c1 + 1] == RED)),
        (BLUE, "top", r0 > 0 and np.all(grid[r0 - 1, c0 : c1 + 1] == BLUE)),
        (BLUE, "bottom", r1 + 1 < h and np.all(grid[r1 + 1, c0 : c1 + 1] == BLUE)),
        (BLUE, "left", c0 > 0 and np.all(grid[r0 : r1 + 1, c0 - 1] == BLUE)),
        (BLUE, "right", c1 + 1 < w and np.all(grid[r0 : r1 + 1, c1 + 1] == BLUE)),
    )
    for color, side, ok in checks:
        if ok:
            if color in guides:
                return None
            guides[color] = side

    if set(guides) != {RED, BLUE}:
        return None
    if not _adjacent_sides(guides[RED], guides[BLUE]):
        return None
    return guides


def _extract_objects(grid: np.ndarray) -> tuple[dict, list[dict]]:
    objects: list[dict] = []
    for coords in _components_of_color(grid, ORANGE):
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BG):
            continue
        if [int(patch[0, 0]), int(patch[0, -1]), int(patch[-1, 0]), int(patch[-1, -1])] != [ORANGE] * 4:
            continue
        guides = _detect_guides(grid, bbox)
        if guides is None:
            continue
        objects.append(
            {
                "bbox": bbox,
                "patch": patch.copy(),
                "guides": guides,
                "solid": bool(np.all(patch == ORANGE)),
            }
        )

    exemplars = [obj for obj in objects if not obj["solid"]]
    targets = [obj for obj in objects if obj["solid"]]
    assert len(exemplars) == 1, "expected exactly one multicolor exemplar"
    assert 2 <= len(targets) <= 3, "expected two or three guided target rectangles"
    return exemplars[0], targets


def _transform_for_target(src_guides: dict[int, str], dst_guides: dict[int, str]) -> str:
    matches = [
        code
        for code in TRANSFORMS
        if _mapped_side(src_guides[RED], code) == dst_guides[RED]
        and _mapped_side(src_guides[BLUE], code) == dst_guides[BLUE]
    ]
    assert len(matches) == 1, "guide colors must determine a unique dihedral transform"
    return matches[0]


def _erase_guides(grid: np.ndarray, bbox: tuple[int, int, int, int], guides: dict[int, str]) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    for side in guides.values():
        if side == "top" and r0 > 0:
            grid[r0 - 1, c0 : c1 + 1] = BG
        elif side == "bottom" and r1 + 1 < h:
            grid[r1 + 1, c0 : c1 + 1] = BG
        elif side == "left" and c0 > 0:
            grid[r0 : r1 + 1, c0 - 1] = BG
        elif side == "right" and c1 + 1 < w:
            grid[r0 : r1 + 1, c1 + 1] = BG

    side_set = set(guides.values())
    corner = None
    if side_set == {"top", "left"}:
        corner = (r0 - 1, c0 - 1)
    elif side_set == {"top", "right"}:
        corner = (r0 - 1, c1 + 1)
    elif side_set == {"bottom", "left"}:
        corner = (r1 + 1, c0 - 1)
    elif side_set == {"bottom", "right"}:
        corner = (r1 + 1, c1 + 1)
    if corner is not None:
        rr, cc = corner
        if 0 <= rr < h and 0 <= cc < w:
            grid[rr, cc] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate/reflect a guided exemplar to each target guide orientation, stretch it to each target rectangle, then remove all guides.

    Input structure:
    - A black background contains one multicolor orange-corner exemplar rectangle and two or three solid orange target rectangles.
    - Every rectangle has one red guide strip and one blue guide strip attached just outside two adjacent sides.
    - The exemplar core is fully filled with colors, while each target core is a solid orange rectangle.

    Rule:
    - Identify the exemplar core by its multicolor interior and read which sides carry its red and blue guides.
    - For each target, read that target's red/blue guide sides, find the unique dihedral transform that maps the exemplar guide orientation to the target guide orientation, and apply that transform to the exemplar core.
    - Stretch the transformed exemplar elastically to the target size by preserving every boundary row/column run and expanding only the unique internal stretch run in each axis.
    - Replace each target's solid orange core with the stretched pattern, erase every guide strip, and erase the exemplar itself.

    Key insight:
    - The guide colors determine orientation and the run-length structure determines size, so the correct answer requires applying both the orientation transform and the elastic stretch before removing guides.
    """
    g = np.asarray(grid, dtype=int)
    exemplar, targets = _extract_objects(g)

    out = g.copy()
    exemplar_core = exemplar["patch"]
    exemplar_guides = exemplar["guides"]

    for target in targets:
        code = _transform_for_target(exemplar_guides, target["guides"])
        transformed = _apply_transform(exemplar_core, code)
        stretched = _stretch_pattern(transformed, target["patch"].shape)
        r0, c0, r1, c1 = target["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = stretched

    r0, c0, r1, c1 = exemplar["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    _erase_guides(out, exemplar["bbox"], exemplar["guides"])
    for target in targets:
        _erase_guides(out, target["bbox"], target["guides"])
    return out


def _build_base_core() -> np.ndarray:
    a, b, c = random.sample(ACCENT_COLORS, 3)
    row_stretch = random.randint(1, 2)
    row_fixed = random.randint(1, 2)
    col_stretch = random.randint(1, 2)
    col_fixed = random.randint(1, 2)
    seed = np.array(
        [
            [ORANGE, ORANGE, ORANGE, ORANGE],
            [ORANGE, a, b, ORANGE],
            [ORANGE, c, ORANGE, ORANGE],
            [ORANGE, ORANGE, ORANGE, ORANGE],
        ],
        dtype=int,
    )
    core = np.repeat(np.repeat(seed, [1, row_stretch, row_fixed, 1], axis=0), [1, col_stretch, col_fixed, 1], axis=1)
    core = _apply_transform(core, random.choice(tuple(TRANSFORMS)))
    assert len(np.unique(core)) == 4
    assert 0 < _stretch_run_index(_line_runs(core, 0)) < len(_line_runs(core, 0)) - 1
    assert 0 < _stretch_run_index(_line_runs(core, 1)) < len(_line_runs(core, 1)) - 1
    return core


def _attach_guides(core: np.ndarray, red_side: str, blue_side: str) -> np.ndarray:
    assert _adjacent_sides(red_side, blue_side)
    h, w = core.shape
    top = 1 if "top" in (red_side, blue_side) else 0
    bottom = 1 if "bottom" in (red_side, blue_side) else 0
    left = 1 if "left" in (red_side, blue_side) else 0
    right = 1 if "right" in (red_side, blue_side) else 0

    patch = np.zeros((h + top + bottom, w + left + right), dtype=int)
    r0 = top
    c0 = left
    patch[r0 : r0 + h, c0 : c0 + w] = core

    if red_side == "top":
        patch[r0 - 1, c0 : c0 + w] = RED
    elif red_side == "bottom":
        patch[r0 + h, c0 : c0 + w] = RED
    elif red_side == "left":
        patch[r0 : r0 + h, c0 - 1] = RED
    else:
        patch[r0 : r0 + h, c0 + w] = RED

    if blue_side == "top":
        patch[r0 - 1, c0 : c0 + w] = BLUE
    elif blue_side == "bottom":
        patch[r0 + h, c0 : c0 + w] = BLUE
    elif blue_side == "left":
        patch[r0 : r0 + h, c0 - 1] = BLUE
    else:
        patch[r0 : r0 + h, c0 + w] = BLUE

    side_set = {red_side, blue_side}
    if side_set == {"top", "left"}:
        patch[r0 - 1, c0 - 1] = BG
    elif side_set == {"top", "right"}:
        patch[r0 - 1, c0 + w] = BG
    elif side_set == {"bottom", "left"}:
        patch[r0 + h, c0 - 1] = BG
    elif side_set == {"bottom", "right"}:
        patch[r0 + h, c0 + w] = BG
    return patch


def _choose_target_codes(count: int) -> list[str]:
    if count == 2:
        return [random.choice(ROTATION_CODES[1:]), random.choice(REFLECTION_CODES)]

    first = random.choice(ROTATION_CODES)
    second = random.choice(tuple(code for code in ROTATION_CODES if code != first))
    third = random.choice(REFLECTION_CODES)
    return [first, second, third]


def _guide_sides_for_code(code: str) -> tuple[str, str]:
    return _mapped_side("top", code), _mapped_side("left", code)


def _candidate_target_shapes(core_shape: tuple[int, int], count: int) -> list[tuple[int, int]]:
    h0, w0 = core_shape
    heights = range(h0 + 1, min(8, h0 + 4) + 1)
    widths = range(w0 + 1, min(9, w0 + 5) + 1)
    candidates = [(h, w) for h in heights for w in widths]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for shape in candidates:
        if shape not in chosen:
            chosen.append(shape)
        if len(chosen) == count:
            return chosen
    raise AssertionError("not enough distinct target shapes")


def _place_patches(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
    *,
    border: int = 1,
    padding: int = 2,
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        ph, pw = patches[idx].shape
        spots = []
        for r0 in range(border, h - ph - border + 1):
            for c0 in range(border, w - pw - border + 1):
                rr0 = max(0, r0 - padding)
                cc0 = max(0, c0 - padding)
                rr1 = min(h, r0 + ph + padding)
                cc1 = min(w, c0 + pw + padding)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    spots.append((r0, c0))
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        placements[idx] = (r0, c0)
        rr0 = max(0, r0 - padding)
        cc0 = max(0, c0 - padding)
        rr1 = min(h, r0 + ph + padding)
        cc1 = min(w, c0 + pw + padding)
        occupied[rr0:rr1, cc0:cc1] = True

    return [pos for pos in placements if pos is not None]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _explicit_output(
    grid_shape: tuple[int, int],
    exemplar_pos: tuple[int, int],
    exemplar_patch: np.ndarray,
    target_specs: list[dict],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for spec in target_specs:
        r0, c0 = spec["core_pos"]
        pattern = spec["expected"]
        h, w = pattern.shape
        out[r0 : r0 + h, c0 : c0 + w] = pattern
    _ = exemplar_pos, exemplar_patch
    return out


def _validate_semantics(grid: np.ndarray, exemplar_core: np.ndarray, target_specs: list[dict], expected: np.ndarray) -> bool:
    try:
        exemplar, targets = _extract_objects(grid)
        if exemplar["guides"] != {RED: "top", BLUE: "left"}:
            return False
        if len(targets) != len(target_specs):
            return False

        seen_pairs = set()
        seen_patterns = set()
        for spec in target_specs:
            pair = (spec["red_side"], spec["blue_side"])
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)

            shape = spec["shape"]
            code = spec["code"]
            variants = []
            for other in TRANSFORMS:
                variant = _stretch_pattern(_apply_transform(exemplar_core, other), shape)
                variants.append((other, variant))
            matches = [name for name, variant in variants if np.array_equal(variant, spec["expected"])]
            if matches != [code]:
                return False
            if code != "id" and np.array_equal(spec["expected"], _stretch_pattern(exemplar_core, shape)):
                return False
            sig = (spec["expected"].shape, spec["expected"].tobytes())
            if sig in seen_patterns:
                return False
            seen_patterns.add(sig)

        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False
        if np.array_equal(solved, grid):
            return False
        return True
    except Exception:
        return False


def _generate_one() -> np.ndarray | None:
    exemplar_core = _build_base_core()
    target_count = random.choice((2, 3))
    target_codes = _choose_target_codes(target_count)
    target_shapes = _candidate_target_shapes(exemplar_core.shape, target_count)

    exemplar_patch = _attach_guides(exemplar_core, "top", "left")
    patches = [exemplar_patch]
    target_specs: list[dict] = []
    for code, shape in zip(target_codes, target_shapes):
        red_side, blue_side = _guide_sides_for_code(code)
        target_patch = _attach_guides(np.full(shape, ORANGE, dtype=int), red_side, blue_side)
        expected = _stretch_pattern(_apply_transform(exemplar_core, code), shape)
        target_specs.append(
            {
                "code": code,
                "shape": shape,
                "red_side": red_side,
                "blue_side": blue_side,
                "patch": target_patch,
                "expected": expected,
            }
        )
        patches.append(target_patch)

    for _ in range(80):
        grid_h = random.randint(18, 24)
        grid_w = random.randint(18, 24)
        placements = _place_patches((grid_h, grid_w), patches)
        if placements is None:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        exemplar_pos = placements[0]
        _paste(grid, exemplar_patch, *exemplar_pos)
        for spec, pos in zip(target_specs, placements[1:]):
            _paste(grid, spec["patch"], *pos)
            core_top = pos[0] + (1 if "top" in (spec["red_side"], spec["blue_side"]) else 0)
            core_left = pos[1] + (1 if "left" in (spec["red_side"], spec["blue_side"]) else 0)
            spec["core_pos"] = (core_top, core_left)

        expected = _explicit_output(grid.shape, exemplar_pos, exemplar_patch, target_specs)
        if _validate_semantics(grid, exemplar_core, target_specs, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guided elastic-stretch instance")
