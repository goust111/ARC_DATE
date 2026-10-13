import random

import numpy as np


MAROON = 9
SYMMETRY_TYPES = ("horizontal", "vertical", "rot180")

# --------------- solver ---------------


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _partner(r: int, c: int, h: int, w: int, kind: str) -> tuple[int, int]:
    if kind == "horizontal":
        return h - 1 - r, c
    if kind == "vertical":
        return r, w - 1 - c
    if kind == "rot180":
        return h - 1 - r, w - 1 - c
    raise ValueError(f"unknown symmetry kind: {kind}")


def _transform(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "horizontal":
        return np.flipud(arr)
    if kind == "vertical":
        return np.fliplr(arr)
    if kind == "rot180":
        return np.rot90(arr, 2)
    raise ValueError(f"unknown symmetry kind: {kind}")


def _rarest_non_bg_color(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(colors) > 0, "expected at least one non-background color"
    min_count = int(counts.min())
    rare = colors[counts == min_count]
    assert len(rare) == 1, "distractor color must be the unique rarest non-background color"
    return int(rare[0])


def _evaluate_candidate(crop: np.ndarray, damage_color: int, kind: str) -> dict[tuple[int, int], int] | None:
    h, w = crop.shape
    repaired: dict[tuple[int, int], int] = {}
    damage_positions = {tuple(map(int, rc)) for rc in np.argwhere(crop == damage_color)}
    assert damage_positions, "expected at least one damaged pixel"

    for r in range(h):
        for c in range(w):
            rr, cc = _partner(r, c, h, w, kind)
            if (rr, cc) < (r, c):
                continue

            v1 = int(crop[r, c])
            v2 = int(crop[rr, cc])
            if (rr, cc) == (r, c):
                if v1 == damage_color:
                    return None
                continue

            if v1 == damage_color and v2 == damage_color:
                return None
            if v1 == damage_color:
                repaired[(r, c)] = v2
                continue
            if v2 == damage_color:
                repaired[(rr, cc)] = v1
                continue
            if v1 != v2:
                return None

    if set(repaired) != damage_positions:
        return None
    return repaired


def _infer_damage_positions(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int], list[tuple[str, dict[tuple[int, int], int]]]]:
    bg = _most_common_color(grid)
    damage_color = _rarest_non_bg_color(grid, bg)
    r0, c0, r1, c1 = _bbox(grid != bg)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]

    candidates: list[tuple[str, dict[tuple[int, int], int]]] = []
    for kind in SYMMETRY_TYPES:
        repaired = _evaluate_candidate(crop, damage_color, kind)
        if repaired is not None:
            candidates.append((kind, repaired))
    assert candidates, "no symmetry candidate explains the damaged pattern"
    return bg, damage_color, (r0, c0, r1, c1), candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify which pixels were corrupted in a symmetric pattern and mark those repaired sites in maroon.

    Input structure:
    - The background is the most common color.
    - One multicolor foreground pattern occupies a single bounding box and follows exactly one global symmetry: horizontal reflection, vertical reflection, or 180-degree rotation.
    - A distractor color replaces several original pattern pixels; this distractor is the unique least frequent non-background color.

    Rule:
    - Find the distractor color as the rarest non-background color.
    - Crop to the foreground bounding box and test which symmetry lets every non-distractor cell agree with its partner while every distractor cell can be repaired from its partner.
    - Treat those distractor positions as the repaired pixels implied by that symmetry.
    - Return the original grid, but recolor every repaired position to maroon instead of restoring its hidden original color.

    Key insight:
    - The rare color points to suspicious cells, but only the surviving symmetric structure confirms which global symmetry is present and which cells were truly damaged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, damage_color, (r0, c0, _r1, _c1), candidates = _infer_damage_positions(g)
    damage_sets = [set(repaired) for _kind, repaired in candidates]
    assert all(damage_sets[0] == item for item in damage_sets), "all valid symmetries must agree on damaged sites"

    out = g.copy()
    for r, c in damage_sets[0]:
        out[r0 + r, c0 + c] = MAROON
    assert not np.any(out == damage_color), "all distractor pixels should be replaced in the output"
    return out


# --------------- generator ---------------

INPUT_COLORS = tuple(range(9))
PATTERN_SIDE_RANGE = (7, 15)
PATTERN_COLOR_COUNT_RANGE = (2, 4)
FEATURE_COUNT_RANGE = (5, 9)
MARGIN_RANGE = (1, 6)
MIN_PATTERN_CELLS = 18
MAX_DAMAGE_CELLS = 4
GEN_ATTEMPTS = 500


def _random_feature_cells(h: int, w: int) -> set[tuple[int, int]]:
    kind = random.choices(("rect", "line", "lshape"), weights=(5, 2, 2), k=1)[0]

    if kind == "rect":
        fh = random.randint(1, min(4, h))
        fw = random.randint(1, min(4, w))
        r0 = random.randint(0, h - fh)
        c0 = random.randint(0, w - fw)
        return {(r, c) for r in range(r0, r0 + fh) for c in range(c0, c0 + fw)}

    if kind == "line":
        if random.random() < 0.5:
            length = random.randint(2, min(6, w))
            r = random.randrange(h)
            c0 = random.randint(0, w - length)
            return {(r, c) for c in range(c0, c0 + length)}
        length = random.randint(2, min(6, h))
        r0 = random.randint(0, h - length)
        c = random.randrange(w)
        return {(r, c) for r in range(r0, r0 + length)}

    arm_h = random.randint(2, min(5, h))
    arm_w = random.randint(2, min(5, w))
    r0 = random.randint(0, h - arm_h)
    c0 = random.randint(0, w - arm_w)
    orient = random.randrange(4)
    if orient == 0:
        return {(r0 + dr, c0) for dr in range(arm_h)} | {(r0, c0 + dc) for dc in range(arm_w)}
    if orient == 1:
        return {(r0 + dr, c0 + arm_w - 1) for dr in range(arm_h)} | {(r0, c0 + dc) for dc in range(arm_w)}
    if orient == 2:
        return {(r0 + arm_h - 1, c0 + dc) for dc in range(arm_w)} | {(r0 + dr, c0) for dr in range(arm_h)}
    return {(r0 + arm_h - 1, c0 + dc) for dc in range(arm_w)} | {
        (r0 + dr, c0 + arm_w - 1) for dr in range(arm_h)
    }


def _paint_symmetric(canvas: np.ndarray, cells: set[tuple[int, int]], color: int, kind: str) -> None:
    h, w = canvas.shape
    for r, c in cells:
        rr, cc = _partner(r, c, h, w, kind)
        canvas[r, c] = color
        canvas[rr, cc] = color


def _exact_symmetry_kinds(pattern: np.ndarray, bg: int) -> set[str]:
    mask = pattern != bg
    assert mask.any()
    r0, c0, r1, c1 = _bbox(mask)
    crop = pattern[r0 : r1 + 1, c0 : c1 + 1]
    return {kind for kind in SYMMETRY_TYPES if np.array_equal(crop, _transform(crop, kind))}


def _build_pattern(bg: int, kind: str) -> np.ndarray | None:
    h = random.randint(*PATTERN_SIDE_RANGE)
    w = random.randint(*PATTERN_SIDE_RANGE)
    pattern_colors = random.sample(
        [c for c in INPUT_COLORS if c not in {bg, MAROON}],
        random.randint(*PATTERN_COLOR_COUNT_RANGE),
    )

    canvas = np.full((h, w), bg, dtype=int)
    for _ in range(random.randint(*FEATURE_COUNT_RANGE)):
        _paint_symmetric(canvas, _random_feature_cells(h, w), random.choice(pattern_colors), kind)

    mask = canvas != bg
    if np.count_nonzero(mask) < MIN_PATTERN_CELLS:
        return None
    r0, c0, r1, c1 = _bbox(mask)
    pattern = canvas[r0 : r1 + 1, c0 : c1 + 1].copy()
    if min(pattern.shape) < 5:
        return None
    if _exact_symmetry_kinds(pattern, bg) != {kind}:
        return None
    return pattern


def _nonfixed_orbits(pattern: np.ndarray, bg: int, kind: str) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    h, w = pattern.shape
    orbits: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    seen: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if pattern[r, c] == bg:
                continue
            rr, cc = _partner(r, c, h, w, kind)
            if (rr, cc) == (r, c):
                continue
            if pattern[rr, cc] != pattern[r, c]:
                return []
            rep = min((r, c), (rr, cc))
            if rep in seen:
                continue
            seen.add(rep)
            orbits.append((rep, max((r, c), (rr, cc)), int(pattern[r, c])))
    return orbits


def _inject_damage(pattern: np.ndarray, bg: int, kind: str) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    colors, counts = np.unique(pattern[pattern != bg], return_counts=True)
    if len(colors) < 2:
        return None

    orbit_list = _nonfixed_orbits(pattern, bg, kind)
    if not orbit_list:
        return None

    min_count = int(counts.min())
    max_damage = min(MAX_DAMAGE_CELLS, len(orbit_list), (min_count - 1) // 2)
    if max_damage < 1:
        return None

    damage_color_candidates = [c for c in INPUT_COLORS if c not in set(map(int, colors)) | {bg, MAROON}]
    if not damage_color_candidates:
        return None
    damage_color = random.choice(damage_color_candidates)

    damaged = pattern.copy()
    chosen = random.sample(orbit_list, random.randint(1, max_damage))
    damage_positions: set[tuple[int, int]] = set()
    for a, b, _color in chosen:
        pos = random.choice((a, b))
        damaged[pos] = damage_color
        damage_positions.add(pos)

    new_colors, new_counts = np.unique(damaged[damaged != bg], return_counts=True)
    min_after = int(new_counts.min())
    rare = new_colors[new_counts == min_after]
    if len(rare) != 1 or int(rare[0]) != damage_color:
        return None

    return damaged, damage_positions


def _embed(pattern: np.ndarray, bg: int) -> np.ndarray | None:
    ph, pw = pattern.shape
    max_top = min(MARGIN_RANGE[1], 30 - ph - 1)
    max_left = min(MARGIN_RANGE[1], 30 - pw - 1)
    if max_top < MARGIN_RANGE[0] or max_left < MARGIN_RANGE[0]:
        return None

    top = random.randint(MARGIN_RANGE[0], max_top)
    bottom = random.randint(MARGIN_RANGE[0], min(MARGIN_RANGE[1], 30 - ph - top))
    left = random.randint(MARGIN_RANGE[0], max_left)
    right = random.randint(MARGIN_RANGE[0], min(MARGIN_RANGE[1], 30 - pw - left))

    h = ph + top + bottom
    w = pw + left + right
    grid = np.full((h, w), bg, dtype=int)
    grid[top : top + ph, left : left + pw] = pattern
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(INPUT_COLORS)
    kind = random.choice(SYMMETRY_TYPES)

    pattern = _build_pattern(bg, kind)
    if pattern is None:
        return None
    injected = _inject_damage(pattern, bg, kind)
    if injected is None:
        return None
    damaged_pattern, local_damage_positions = injected

    grid = _embed(damaged_pattern, bg)
    if grid is None:
        return None

    bg_count = int(np.count_nonzero(grid == bg))
    non_bg_colors, non_bg_counts = np.unique(grid[grid != bg], return_counts=True)
    if bg_count <= int(non_bg_counts.max()):
        return None

    solver_bg, damage_color, bbox, candidates = _infer_damage_positions(grid)
    if solver_bg != bg:
        return None
    if len(candidates) != 1 or candidates[0][0] != kind:
        return None

    r0, c0, _r1, _c1 = bbox
    expected_damage_positions = {(r0 + r, c0 + c) for r, c in local_damage_positions}
    actual_damage_positions = {(r0 + r, c0 + c) for r, c in candidates[0][1]}
    if expected_damage_positions != actual_damage_positions:
        return None

    expected = grid.copy()
    for r, c in expected_damage_positions:
        expected[r, c] = MAROON
    if not np.array_equal(main(grid), expected):
        return None
    if damage_color == MAROON:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-repair input")
