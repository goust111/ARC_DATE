import itertools
import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _swap_color(color: int, color_a: int, color_b: int) -> int:
    if color == color_a:
        return color_b
    if color == color_b:
        return color_a
    return color


def _extract_patch(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    bg = _most_common_color(grid)
    box = _bbox(grid != bg)
    assert box is not None, "expected a non-background pattern"
    r0, c0, r1, c1 = box
    return bg, box, np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _evaluate_candidate(
    patch: np.ndarray, hole_color: int, color_a: int, color_b: int
) -> dict | None:
    h, w = map(int, patch.shape)
    visible = patch != hole_color
    if not np.any(~visible):
        return None

    if color_a not in patch[visible] or color_b not in patch[visible]:
        return None

    fills: dict[tuple[int, int], int] = {}
    evidence = 0
    swap_support = 0

    for r in range(h):
        hr = h - 1 - r
        for c in range(w):
            vc = w - 1 - c
            value = int(patch[r, c])

            if value != hole_color:
                partners = [
                    (hr, c, False),
                    (r, vc, True),
                    (hr, vc, True),
                ]
                for rr, cc, do_swap in partners:
                    other = int(patch[rr, cc])
                    if other == hole_color:
                        continue
                    expected = _swap_color(other, color_a, color_b) if do_swap else other
                    evidence += 1
                    if value != expected:
                        return None
                    if do_swap and (value in (color_a, color_b) or other in (color_a, color_b)):
                        swap_support += 1
            else:
                options = []
                for rr, cc, do_swap in (
                    (hr, c, False),
                    (r, vc, True),
                    (hr, vc, True),
                ):
                    other = int(patch[rr, cc])
                    if other == hole_color:
                        continue
                    options.append(_swap_color(other, color_a, color_b) if do_swap else other)
                if not options:
                    return None
                if len(set(options)) != 1:
                    return None
                fills[(r, c)] = options[0]
                evidence += len(options)
                if options[0] in (color_a, color_b):
                    swap_support += 1

    if swap_support == 0:
        return None

    return {
        "hole_color": hole_color,
        "color_a": color_a,
        "color_b": color_b,
        "fills": fills,
        "score": (evidence, swap_support, -np.count_nonzero(visible)),
    }


def _infer_pattern_spec(patch: np.ndarray) -> dict:
    colors = sorted(map(int, np.unique(patch)))
    candidates = []
    for hole_color in colors:
        others = [color for color in colors if color != hole_color]
        for color_a, color_b in itertools.combinations(others, 2):
            candidate = _evaluate_candidate(patch, hole_color, color_a, color_b)
            if candidate is not None:
                candidates.append(candidate)

    assert candidates, "failed to infer the occluding color and swapped color pair"
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1]["score"] == best["score"]:
        raise AssertionError("ambiguous pattern specification")
    return best


def _restore_patch(patch: np.ndarray, spec: dict) -> np.ndarray:
    out = patch.copy()
    for (r, c), color in spec["fills"].items():
        out[r, c] = color
    return out


def _orbit_members(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    members = {
        (r, c),
        (h - 1 - r, c),
        (r, w - 1 - c),
        (h - 1 - r, w - 1 - c),
    }
    return sorted(members)


def _all_orbits(h: int, w: int) -> list[list[tuple[int, int]]]:
    seen = set()
    orbits = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = _orbit_members(h, w, r, c)
            for cell in orbit:
                seen.add(cell)
            orbits.append(orbit)
    return orbits


def _build_symmetric_patch(
    patch_h: int, patch_w: int, color_a: int, color_b: int, fixed_colors: list[int]
) -> np.ndarray:
    patch = np.zeros((patch_h, patch_w), dtype=int)
    rep_rows = range((patch_h + 1) // 2)
    rep_cols = range((patch_w + 1) // 2)

    swappable_reps = [(r, c) for r in rep_rows for c in rep_cols if c != patch_w - 1 - c]
    assert len(swappable_reps) >= 2

    assignments: dict[tuple[int, int], int] = {}
    a_pos, b_pos = random.sample(swappable_reps, 2)
    assignments[a_pos] = color_a
    assignments[b_pos] = color_b

    remaining_reps = [pos for pos in [(r, c) for r in rep_rows for c in rep_cols] if pos not in assignments]
    random.shuffle(remaining_reps)
    for color in fixed_colors:
        if not remaining_reps:
            break
        for idx, pos in enumerate(remaining_reps):
            r, c = pos
            if c == patch_w - 1 - c:
                assignments[pos] = color
                remaining_reps.pop(idx)
                break
        else:
            pos = remaining_reps.pop()
            assignments[pos] = color

    palette = [color_a, color_b, *fixed_colors]
    for r in rep_rows:
        for c in rep_cols:
            if (r, c) in assignments:
                base_color = assignments[(r, c)]
            elif c == patch_w - 1 - c:
                base_color = random.choice(fixed_colors)
            else:
                base_color = random.choice(palette)

            patch[r, c] = base_color
            patch[patch_h - 1 - r, c] = base_color
            swapped = _swap_color(base_color, color_a, color_b)
            patch[r, patch_w - 1 - c] = swapped
            patch[patch_h - 1 - r, patch_w - 1 - c] = swapped

    assert np.any(patch == color_a)
    assert np.any(patch == color_b)
    for color in fixed_colors:
        assert np.any(patch == color)
    return patch


def _sample_hole_mask(patch: np.ndarray, color_a: int, color_b: int) -> np.ndarray | None:
    h, w = map(int, patch.shape)
    orbits = _all_orbits(h, w)
    random.shuffle(orbits)
    mask = np.zeros((h, w), dtype=bool)

    target_min = max(3, (h * w) // 8)
    target_max = max(target_min, (h * w) // 3)
    target = random.randint(target_min, target_max)

    for orbit in orbits:
        size = len(orbit)
        if size == 1:
            continue
        if random.random() < 0.45:
            continue

        hide_count = random.randint(1, size - 1)
        if size == 4 and random.random() < 0.35:
            hide_count = min(size - 1, hide_count + 1)
        chosen = random.sample(orbit, hide_count)
        for r, c in chosen:
            mask[r, c] = True
        if int(mask.sum()) >= target:
            break

    if int(mask.sum()) < target_min:
        candidates = [orbit for orbit in orbits if len(orbit) > 1]
        for orbit in candidates:
            visible_members = [(r, c) for r, c in orbit if not mask[r, c]]
            if len(visible_members) <= 1:
                continue
            r, c = random.choice(visible_members)
            mask[r, c] = True
            if int(mask.sum()) >= target_min:
                break

    if int(mask.sum()) < target_min or int(mask.sum()) > target_max:
        return None

    for orbit in orbits:
        hidden = sum(mask[r, c] for r, c in orbit)
        if hidden == len(orbit):
            return None

    visible_values = patch[~mask]
    if color_a not in visible_values or color_b not in visible_values:
        return None

    return mask


def _gen_attempt() -> np.ndarray | None:
    bg_color = random.randrange(10)
    remaining = [color for color in range(10) if color != bg_color]
    random.shuffle(remaining)

    hole_color = remaining.pop()
    color_a = remaining.pop()
    color_b = remaining.pop()
    extra_count = random.randint(1, min(3, len(remaining)))
    fixed_colors = remaining[:extra_count]

    patch_h = random.randint(5, 11)
    patch_w = random.randint(6, 12)
    canvas_h = random.randint(patch_h + 4, min(30, patch_h + 12))
    canvas_w = random.randint(patch_w + 4, min(30, patch_w + 12))

    patch = _build_symmetric_patch(patch_h, patch_w, color_a, color_b, fixed_colors)
    hole_mask = _sample_hole_mask(patch, color_a, color_b)
    if hole_mask is None:
        return None

    damaged_patch = patch.copy()
    damaged_patch[hole_mask] = hole_color

    canvas = np.full((canvas_h, canvas_w), bg_color, dtype=int)
    max_r = canvas_h - patch_h
    max_c = canvas_w - patch_w
    top = random.randint(0, max_r)
    left = random.randint(0, max_c)
    canvas[top : top + patch_h, left : left + patch_w] = damaged_patch

    try:
        bg, box, cropped = _extract_patch(canvas)
        if bg != bg_color or box != (top, left, top + patch_h - 1, left + patch_w - 1):
            return None
        spec = _infer_pattern_spec(cropped)
        restored = _restore_patch(cropped, spec)
    except AssertionError:
        return None

    if spec["hole_color"] != hole_color:
        return None
    if {spec["color_a"], spec["color_b"]} != {color_a, color_b}:
        return None
    if not np.array_equal(restored, patch):
        return None

    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the damaged rectangular motif by combining ordinary vertical reflection with left-right reflection that swaps one color pair.

    Input structure:
    - The grid has a uniform background and one dense rectangular motif whose damaged cells are overwritten by one occluding color.
    - Removing the occluding color reveals a pattern that is symmetric across its horizontal midline with colors unchanged.
    - The same motif is symmetric across its vertical midline after swapping exactly two motif colors while keeping all others fixed.

    Rule:
    - Find the motif bounding box, then infer the unique occluding color and the unique swapped color pair that make all visible cells mutually consistent.
    - For each occluded cell, read any visible horizontal, vertical, or diagonal orbit mate inside the motif.
    - Horizontal mates copy their color directly; vertical and diagonal mates contribute after applying the same two-color swap once.

    Key insight:
    - The hidden symmetry is not purely geometric: left-right reflection must be composed with a specific two-color permutation, while top-bottom reflection is ordinary.
    """
    grid = np.asarray(grid, dtype=int)
    _, (r0, c0, r1, c1), patch = _extract_patch(grid)
    spec = _infer_pattern_spec(patch)
    repaired_patch = _restore_patch(patch, spec)

    out = grid.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = repaired_patch
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-swap repair input")
