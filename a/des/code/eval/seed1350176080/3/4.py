from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mirror_pos(r: int, c: int, shape: tuple[int, int], axis: str) -> tuple[int, int]:
    h, w = shape
    if axis == "vertical":
        return r, w - 1 - c
    return h - 1 - r, c


def _reflect(patch: np.ndarray, axis: str) -> np.ndarray:
    return np.fliplr(patch) if axis == "vertical" else np.flipud(patch)


def _axis_cells(shape: tuple[int, int], axis: str) -> list[tuple[int, int]]:
    h, w = shape
    if axis == "vertical":
        if w % 2 == 0:
            return []
        return [(r, w // 2) for r in range(h)]
    if h % 2 == 0:
        return []
    return [(h // 2, c) for c in range(w)]


def _component_hypothesis(
    patch: np.ndarray, bg: int, axis: str
) -> tuple[str, tuple[int, int], int | None] | None:
    counts = Counter(int(v) for v in patch.flat if int(v) != bg)
    inconsistent: list[list[tuple[int, int]]] = []
    axis_fg_cells: list[tuple[int, int]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    h, w = patch.shape

    for r in range(h):
        for c in range(w):
            mr, mc = _mirror_pos(r, c, patch.shape, axis)
            a = (r, c)
            b = (mr, mc)
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            orbit = [key[0]] if key[0] == key[1] else [key[0], key[1]]
            colors = {int(patch[rr, cc]) for rr, cc in orbit}
            if len(orbit) == 1 and int(patch[orbit[0]]) != bg:
                axis_fg_cells.append(orbit[0])
            if len(colors) > 1:
                inconsistent.append(orbit)
                if len(inconsistent) > 1:
                    return None

    if len(inconsistent) == 1:
        orbit = inconsistent[0]
        if len(orbit) != 2:
            return None
        candidates: list[tuple[str, tuple[int, int], int]] = []
        for idx, cell in enumerate(orbit):
            color = int(patch[cell])
            other = orbit[1 - idx]
            other_color = int(patch[other])
            if color == bg or other_color == bg:
                continue
            if counts.get(color, 0) != 1:
                continue
            trial = patch.copy()
            trial[cell] = other_color
            if np.array_equal(trial, _reflect(trial, axis)):
                candidates.append(("repair", cell, other_color))
        if len(candidates) == 1:
            return candidates[0]
        return None

    if not np.array_equal(patch, _reflect(patch, axis)):
        return None

    unique_axis_cells = [cell for cell in axis_fg_cells if counts.get(int(patch[cell]), 0) == 1]
    if len(unique_axis_cells) == 1:
        return "delete", unique_axis_cells[0], None
    return None


def _component_plans(
    patch: np.ndarray, bg: int
) -> list[tuple[str, str, tuple[int, int], int | None]]:
    plans = []
    for axis in ("vertical", "horizontal"):
        hypothesis = _component_hypothesis(patch, bg, axis)
        if hypothesis is not None:
            action, cell, new_color = hypothesis
            plans.append((axis, action, cell, new_color))
    return plans


def _component_plan(
    patch: np.ndarray, bg: int
) -> tuple[str, str, tuple[int, int], int | None]:
    plans = _component_plans(patch, bg)
    assert len(plans) == 1, "component must admit exactly one symmetry-based repair plan"
    return plans[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the single defective pixel in each symmetric component.

    Components are 8-connected non-background objects. Each object is intended
    to be symmetric either across a vertical axis (left-right reflection) or a
    horizontal axis (up-down reflection), but one pixel has been recolored.
    If the defective pixel lies on the symmetry axis, erase the whole object.
    Otherwise recolor that pixel to match its reflected partner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    out = g.copy()

    for coords in _components8(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = int(g[r, c])

        _axis, action, (rr, cc), new_color = _component_plan(patch, bg)
        if action == "delete":
            for r, c in coords:
                out[r, c] = bg
        else:
            out[r0 + rr, c0 + cc] = int(new_color)

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 5)
PATCH_SIZE_CHOICES = [3, 4, 5, 6, 7]
ODD_PATCH_SIZES = [3, 5, 7]
PATTERN_COLOR_COUNTS = [2, 3]
PALETTE = list(range(10))
AXES = ("vertical", "horizontal")
BRANCHES = ("repair", "delete")
BASE_PATCH_ATTEMPTS = 80
COMPONENT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400
MAX_COVERAGE = 0.34


def _balanced_labels(n: int, a: str, b: str) -> list[str]:
    if n % 2 == 0:
        labels = [a] * (n // 2) + [b] * (n // 2)
    else:
        major = random.choice((a, b))
        minor = b if major == a else a
        labels = [major] * ((n + 1) // 2) + [minor] * (n // 2)
    random.shuffle(labels)
    return labels


def _random_patch(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = shape
    return np.array([[random.choice(colors) for _ in range(w)] for _ in range(h)], dtype=int)


def _build_base_patch(axis: str, h: int, w: int, colors: list[int]) -> np.ndarray | None:
    other_axis = "horizontal" if axis == "vertical" else "vertical"
    for _ in range(BASE_PATCH_ATTEMPTS):
        if axis == "vertical":
            half = w // 2
            left = _random_patch((h, half), colors) if half else np.zeros((h, 0), dtype=int)
            if w % 2:
                center = _random_patch((h, 1), colors)
                patch = np.concatenate([left, center, np.fliplr(left)], axis=1)
            else:
                patch = np.concatenate([left, np.fliplr(left)], axis=1)
        else:
            half = h // 2
            top = _random_patch((half, w), colors) if half else np.zeros((0, w), dtype=int)
            if h % 2:
                center = _random_patch((1, w), colors)
                patch = np.concatenate([top, center, np.flipud(top)], axis=0)
            else:
                patch = np.concatenate([top, np.flipud(top)], axis=0)

        if len(np.unique(patch)) < 2:
            continue
        counts = Counter(int(v) for v in patch.flat)
        if min(counts.values()) < 2:
            continue
        if np.array_equal(patch, _reflect(patch, other_axis)):
            continue
        return patch
    return None


def _choose_dims(axis: str, branch: str) -> tuple[int, int]:
    if axis == "vertical":
        h = random.choice(PATCH_SIZE_CHOICES)
        w = random.choice(ODD_PATCH_SIZES if branch == "delete" else PATCH_SIZE_CHOICES)
    else:
        h = random.choice(ODD_PATCH_SIZES if branch == "delete" else PATCH_SIZE_CHOICES)
        w = random.choice(PATCH_SIZE_CHOICES)
    return h, w


def _defect_candidates(base: np.ndarray, axis: str, branch: str) -> list[tuple[int, int]]:
    counts = Counter(int(v) for v in base.flat)
    axis_set = set(_axis_cells(base.shape, axis))
    candidates: list[tuple[int, int]] = []
    for r in range(base.shape[0]):
        for c in range(base.shape[1]):
            on_axis = (r, c) in axis_set
            if branch == "delete" and not on_axis:
                continue
            if branch == "repair" and on_axis:
                continue
            if counts[int(base[r, c])] >= 3:
                candidates.append((r, c))
    return candidates


def _make_component_patch(axis: str, branch: str, bg: int) -> np.ndarray | None:
    usable = [c for c in PALETTE if c != bg]
    for _ in range(COMPONENT_ATTEMPTS):
        h, w = _choose_dims(axis, branch)
        n_colors = random.choice(PATTERN_COLOR_COUNTS)
        base_colors = random.sample(usable, n_colors)
        base = _build_base_patch(axis, h, w, base_colors)
        if base is None:
            continue

        candidates = _defect_candidates(base, axis, branch)
        if not candidates:
            continue

        defect = random.choice(candidates)
        defect_colors = [c for c in usable if c not in base_colors]
        defect_color = random.choice(defect_colors)
        patch = base.copy()
        patch[defect] = defect_color

        plans = _component_plans(patch, bg)
        if len(plans) != 1:
            continue
        plan = plans[0]
        if plan[0] != axis or plan[1] != branch or plan[2] != defect:
            continue
        if branch == "repair":
            mr, mc = _mirror_pos(defect[0], defect[1], patch.shape, axis)
            if plan[3] != int(base[mr, mc]):
                continue
        return patch
    return None


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        r_lo = max(0, r0 - 1)
        r_hi = min(h, r0 + ph + 1)
        c_lo = max(0, c0 - 1)
        c_hi = min(w, c0 + pw + 1)
        if blocked[r_lo:r_hi, c_lo:c_hi].any():
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        blocked[r_lo:r_hi, c_lo:c_hi] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(PALETTE)
    count = random.randint(*OBJECT_COUNT_RANGE)

    axes = _balanced_labels(count, AXES[0], AXES[1])
    branches = _balanced_labels(count, BRANCHES[0], BRANCHES[1])

    patches: list[np.ndarray] = []
    for axis, branch in zip(axes, branches, strict=False):
        patch = _make_component_patch(axis, branch, bg)
        if patch is None:
            return None
        patches.append(patch)

    total_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    if total_area / float(h * w) > MAX_COVERAGE:
        return None

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        if not _place_patch(grid, blocked, patches[idx]):
            return None

    values, counts = np.unique(grid, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts, strict=False)}
    bg_count = freq.pop(bg, 0)
    if not freq or bg_count <= max(freq.values()):
        return None

    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
