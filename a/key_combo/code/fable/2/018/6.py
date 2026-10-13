from __future__ import annotations

from collections import Counter
import random

import numpy as np


ALL_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _orbit_coords(box: tuple[int, int, int, int], r: int, c: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    rr = r0 + r1 - int(r)
    cc = c0 + c1 - int(c)
    orbit = {(int(r), int(c)), (int(r), int(cc)), (int(rr), int(c)), (int(rr), int(cc))}
    return sorted(orbit)


def _iter_orbits(box: tuple[int, int, int, int]):
    r0, c0, r1, c1 = box
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            orbit = tuple(_orbit_coords(box, r, c))
            if orbit in seen:
                continue
            seen.add(orbit)
            yield list(orbit)


def _is_symmetric_in_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    for orbit in _iter_orbits(box):
        values = {int(grid[r, c]) for r, c in orbit}
        if len(values) != 1:
            return False
    return True


def _mirror_quadrant(quadrant: np.ndarray) -> np.ndarray:
    top = np.concatenate([quadrant, quadrant[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


def _build_quadrant(qh: int, qw: int, palette: list[int]) -> np.ndarray:
    quadrant = np.full((qh, qw), random.choice(palette), dtype=int)
    for _ in range(random.randint(4, 8)):
        color = random.choice(palette)
        h = random.randint(1, max(1, qh // 2 + 1))
        w = random.randint(1, max(1, qw // 2 + 1))
        r0 = random.randint(0, qh - h)
        c0 = random.randint(0, qw - w)
        quadrant[r0 : r0 + h, c0 : c0 + w] = color

    for color in palette:
        if not np.any(quadrant == color):
            quadrant[random.randrange(qh), random.randrange(qw)] = color

    if len(np.unique(quadrant)) < 2:
        quadrant[random.randrange(qh), random.randrange(qw)] = palette[-1]

    if random.random() < 0.5:
        quadrant = quadrant[:, ::-1]
    quadrant = np.rot90(quadrant, random.randrange(4))
    return np.asarray(quadrant, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """Resolve each broken 4-cell reflection orbit by erasing old-color defects or spreading brand-new colors.

    Input structure:
    - A dominant background surrounds one tight multicolor pattern; the non-background bounding box defines the intended horizontal and vertical mirror axes.
    - The clean pattern would be symmetric across both bbox center lines.
    - Some 4-cell reflection orbits are broken by exactly one minority-color defect pixel.

    Rule:
    - Compute the non-background bbox and group its cells into orbits under horizontal and vertical reflection about that bbox center.
    - For every non-uniform 4-cell orbit with a 3-to-1 color split, identify the unique minority color.
    - If that minority color appears elsewhere outside the orbit, repaint the orbit with the majority color; otherwise repaint all four cells with the minority color.

    Key insight:
    - The local defect signature is always the same, but the fix depends on palette membership: existing colors are corrected away, while truly new colors take over their whole symmetry orbit.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = _get_bg(g)
    box = _bbox(g != bg)
    if box is None:
        return out

    totals = Counter(map(int, g.flat))
    for orbit in _iter_orbits(box):
        if len(orbit) != 4:
            continue
        values = [int(g[r, c]) for r, c in orbit]
        counts = Counter(values)
        if len(counts) == 1:
            continue
        if sorted(counts.values()) != [1, 3]:
            continue

        minority = next(color for color, count in counts.items() if count == 1)
        majority = next(color for color, count in counts.items() if count == 3)
        outside_count = totals[minority] - sum(1 for value in values if value == minority)
        fill = majority if outside_count > 0 else minority
        for r, c in orbit:
            out[r, c] = fill

    return out


def _make_case():
    bg = random.choice(ALL_COLORS)
    remaining = [color for color in ALL_COLORS if color != bg]
    palette_size = random.randint(3, 5)
    palette = random.sample(remaining, palette_size)
    new_pool = [color for color in remaining if color not in palette]

    qh = random.randint(3, 6)
    qw = random.randint(3, 6)
    quadrant = _build_quadrant(qh, qw, palette)
    qh, qw = map(int, quadrant.shape)
    patch = _mirror_quadrant(quadrant)
    patch_h, patch_w = map(int, patch.shape)

    extra_h = random.randint(6, 30 - patch_h)
    extra_w = random.randint(6, 30 - patch_w)
    top = random.randint(1, extra_h - 1)
    left = random.randint(1, extra_w - 1)
    bottom = extra_h - top
    right = extra_w - left

    canvas_h = patch_h + top + bottom
    canvas_w = patch_w + left + right
    clean = np.full((canvas_h, canvas_w), bg, dtype=int)
    clean[top : top + patch_h, left : left + patch_w] = patch

    box = (top, left, top + patch_h - 1, left + patch_w - 1)
    reps = [(top + r, left + c) for r in range(qh) for c in range(qw)]
    random.shuffle(reps)

    new_count = random.randint(1, min(2, len(new_pool)))
    old_count = random.randint(1, min(3, len(reps) - new_count))
    chosen_reps = reps[: old_count + new_count]
    branches = ["repair"] * old_count + ["spread"] * new_count
    random.shuffle(branches)

    clean_counts = Counter(map(int, clean.flat))
    inp = clean.copy()
    out = clean.copy()
    plans = []
    used_new_colors: set[int] = set()

    for rep, branch in zip(chosen_reps, branches):
        orbit = _orbit_coords(box, *rep)
        majority = int(clean[orbit[0]])
        defect_cell = random.choice(orbit)

        if branch == "repair":
            candidates = [color for color in palette if color != majority and clean_counts[color] >= 8]
            if not candidates:
                candidates = [color for color in palette if color != majority]
            defect_color = random.choice(candidates)
        else:
            available_new = [color for color in new_pool if color not in used_new_colors]
            defect_color = random.choice(available_new)
            used_new_colors.add(defect_color)
            for r, c in orbit:
                out[r, c] = defect_color

        inp[defect_cell] = defect_color
        plans.append(
            {
                "branch": branch,
                "orbit": orbit,
                "defect_cell": defect_cell,
                "majority": majority,
                "defect_color": defect_color,
            }
        )

    return inp, out, clean, box, plans, bg


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    clean: np.ndarray,
    box: tuple[int, int, int, int],
    plans: list[dict],
    bg: int,
) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if inp.shape != out.shape or inp.shape != clean.shape:
        return False
    if _get_bg(inp) != bg:
        return False
    if _bbox(inp != bg) != box or _bbox(out != bg) != box:
        return False
    if not _is_symmetric_in_bbox(clean, box):
        return False
    if not _is_symmetric_in_bbox(out, box):
        return False

    expected_orbits = {tuple(plan["orbit"]) for plan in plans}
    found_broken: set[tuple[tuple[int, int], ...]] = set()
    for orbit in _iter_orbits(box):
        if len(orbit) != 4:
            continue
        values = [int(inp[r, c]) for r, c in orbit]
        counts = Counter(values)
        if len(counts) == 1:
            continue
        if sorted(counts.values()) != [1, 3]:
            return False
        found_broken.add(tuple(orbit))

    if found_broken != expected_orbits:
        return False

    for plan in plans:
        orbit = plan["orbit"]
        values = [int(inp[r, c]) for r, c in orbit]
        counts = Counter(values)
        if sorted(counts.values()) != [1, 3]:
            return False

        minority = next(color for color, count in counts.items() if count == 1)
        majority = next(color for color, count in counts.items() if count == 3)
        if minority != plan["defect_color"] or majority != plan["majority"]:
            return False

        outside_input = int(np.count_nonzero(inp == minority)) - sum(int(inp[r, c] == minority) for r, c in orbit)
        outside_clean = int(np.count_nonzero(clean == minority)) - sum(int(clean[r, c] == minority) for r, c in orbit)
        out_values = {int(out[r, c]) for r, c in orbit}

        if plan["branch"] == "repair":
            if outside_input <= 0 or outside_clean <= 0:
                return False
            if out_values != {majority}:
                return False
        else:
            if outside_input != 0 or int(np.count_nonzero(clean == minority)) != 0:
                return False
            if out_values != {minority}:
                return False

    if not np.array_equal(main(inp), out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _make_case()
        if _validate_case(*case):
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
