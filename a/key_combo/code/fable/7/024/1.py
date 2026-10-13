from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"].astype(int, copy=False)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _make_component_mask(shape: tuple[int, int], component: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    coords = component["coords"].astype(int, copy=False)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _row_lefts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(mask.any(axis=1))
    assert rows.size > 0, "fragment must occupy at least one row"
    lefts = np.array([int(np.flatnonzero(mask[r])[0]) for r in rows], dtype=int)
    return rows.astype(int), lefts


def _candidate_positions(
    fragment_mask: np.ndarray,
    wall_mask: np.ndarray,
    occupied_mask: np.ndarray,
) -> list[tuple[int, int]]:
    wall_right = np.full(wall_mask.shape[0], -1, dtype=int)
    for r in range(wall_mask.shape[0]):
        cols = np.flatnonzero(wall_mask[r])
        if cols.size:
            wall_right[r] = int(cols[-1])

    rows, lefts = _row_lefts(fragment_mask)
    frag_coords = np.argwhere(fragment_mask).astype(int, copy=False)
    frag_h, frag_w = map(int, fragment_mask.shape)
    grid_h, grid_w = map(int, wall_mask.shape)
    candidates: list[tuple[int, int]] = []

    for top in range(grid_h - frag_h + 1):
        seam_cols = wall_right[top + rows] + 1 - lefts
        if np.any(wall_right[top + rows] < 0):
            continue
        if np.any(seam_cols != seam_cols[0]):
            continue

        left = int(seam_cols[0])
        if left < 0 or left + frag_w > grid_w:
            continue

        shifted = frag_coords + np.array([top, left], dtype=int)
        if np.any(shifted[:, 0] < 0) or np.any(shifted[:, 0] >= grid_h):
            continue
        if np.any(shifted[:, 1] < 0) or np.any(shifted[:, 1] >= grid_w):
            continue
        if np.any(occupied_mask[shifted[:, 0], shifted[:, 1]]):
            continue
        candidates.append((top, left))

    return candidates


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Translate each scattered fragment onto the wall only at its unique seamless lock-fit, leaving unfittable fragments untouched.

    Input structure:
    - The background is the most common color.
    - Exactly one monochrome component touches the left border; it is a thick vertical wall whose right contour is jagged by row.
    - The remaining monochrome components are disconnected colored fragments with their own jagged left contours.

    Rule:
    - For each fragment, search all row and column translations for positions where every occupied fragment row has its leftmost cell exactly one column to the right of the wall's rightmost cell on the corresponding row.
    - Keep only translations with no overlap against the wall or fragments already fitted onto it; if a fragment has exactly one such position, move it there and clear its original location.
    - If a fragment has no valid fit, or more than one valid fit, leave it where it started.

    Key insight:
    - The wall and each fragment are matched by row-wise contour differences, not by pushing horizontally: a true fit is the unique translation where the fragment's entire left profile locks flush against the wall without overlap.
    """
    g = np.asarray(input_grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected a wall and at least one fragment"

    wall_candidates = [comp for comp in components if np.any(comp["coords"][:, 1] == 0)]
    assert len(wall_candidates) == 1, "expected a unique left-border wall"
    wall = wall_candidates[0]
    wall_mask = _make_component_mask(g.shape, wall)

    fragments = [comp for comp in components if comp is not wall]
    out = g.copy()
    placed_mask = wall_mask.copy()

    fragments.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    for comp in fragments:
        fragment_mask = _component_mask(comp)
        candidates = _candidate_positions(fragment_mask, wall_mask, placed_mask)
        if len(candidates) != 1:
            continue

        top, left = candidates[0]
        coords = comp["coords"].astype(int, copy=False)
        out[coords[:, 0], coords[:, 1]] = bg
        patch = np.full(fragment_mask.shape, bg, dtype=int)
        patch[fragment_mask] = int(comp["color"])
        paste(out, patch, top, left, transparent=bg)

        placed_coords = np.argwhere(fragment_mask).astype(int, copy=False) + np.array([top, left], dtype=int)
        placed_mask[placed_coords[:, 0], placed_coords[:, 1]] = True

    return out


GEN_MAX_ATTEMPTS = 500
GEN_WALL_MIN = 2
GEN_WALL_MAX = 7
GEN_WALL_STEP_CHOICES = (-1, 0, 1)
GEN_HEIGHT_RANGE = (12, 24)
GEN_WIDTH_RANGE = (18, 30)
GEN_FIT_HEIGHTS = (2, 3, 4, 5)
GEN_UNFIT_HEIGHTS = (1, 2, 3, 4, 5)
GEN_FRAGMENT_WIDTH_RANGE = (2, 5)
GEN_SCATTER_PADDING = 1
GEN_VISUAL_GAP = 4


def _normalize_profile(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    base = min(int(v) for v in values)
    return tuple(int(v) - base for v in values)


def _make_wall_profile(height: int) -> list[int]:
    profile = [random.randint(GEN_WALL_MIN + 1, GEN_WALL_MAX - 1)]
    for _ in range(height - 1):
        nxt = profile[-1] + random.choice(GEN_WALL_STEP_CHOICES)
        profile.append(max(GEN_WALL_MIN, min(GEN_WALL_MAX, nxt)))
    if len(set(profile)) == 1:
        idx = random.randrange(height)
        profile[idx] = max(GEN_WALL_MIN, min(GEN_WALL_MAX, profile[idx] + random.choice((-1, 1))))
    return profile


def _wall_grid(shape: tuple[int, int], profile: list[int], color: int, bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for r, right in enumerate(profile):
        grid[r, : int(right) + 1] = int(color)
    return grid


def _window_occurrences(profile: list[int], size: int) -> Counter:
    return Counter(
        _normalize_profile(tuple(profile[start : start + size]))
        for start in range(len(profile) - size + 1)
    )


def _build_fragment_from_left_profile(left_profile: tuple[int, ...]) -> np.ndarray:
    widths = [random.randint(GEN_FRAGMENT_WIDTH_RANGE[0], GEN_FRAGMENT_WIDTH_RANGE[1]) for _ in left_profile]
    frag_w = max(int(left) + int(width) for left, width in zip(left_profile, widths))
    mask = np.zeros((len(left_profile), frag_w), dtype=bool)
    for r, (left, width) in enumerate(zip(left_profile, widths)):
        mask[r, int(left) : int(left) + int(width)] = True
    return mask


def _choose_fit_windows(profile: list[int], count: int) -> list[tuple[int, np.ndarray]]:
    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for size in GEN_FIT_HEIGHTS:
        if size > len(profile):
            continue
        counts = _window_occurrences(profile, size)
        for start in range(len(profile) - size + 1):
            signature = _normalize_profile(tuple(profile[start : start + size]))
            if counts[signature] == 1:
                candidates.append((start, size, signature))

    random.shuffle(candidates)
    picked: list[tuple[int, np.ndarray]] = []
    used_rows: set[int] = set()
    for start, size, signature in candidates:
        rows = set(range(start - 1, start + size + 1))
        if rows & used_rows:
            continue
        used_rows |= rows
        picked.append((start, _build_fragment_from_left_profile(signature)))
        if len(picked) == count:
            break

    return picked


def _make_unfit_fragment(wall_mask: np.ndarray) -> np.ndarray:
    for _ in range(200):
        height = random.choice(GEN_UNFIT_HEIGHTS)
        profile = [0]
        for _ in range(height - 1):
            profile.append(max(0, profile[-1] + random.choice(GEN_WALL_STEP_CHOICES)))
        if len(set(profile)) == 1 and height > 1:
            profile[random.randrange(height)] += 1
        profile = list(_normalize_profile(profile))
        fragment = _build_fragment_from_left_profile(tuple(profile))
        occupied = wall_mask.copy()
        if not _candidate_positions(fragment, wall_mask, occupied):
            return fragment
    raise RuntimeError("failed to construct an unfittable fragment")


def _scatter_fragment(
    grid: np.ndarray,
    fragment: np.ndarray,
    color: int,
    bg: int,
    scatter_c0: int,
) -> tuple[int, int]:
    patch = np.full(fragment.shape, bg, dtype=int)
    patch[fragment] = int(color)

    sub = grid[:, scatter_c0:].copy()
    r0, c0 = random_free_location(
        sub,
        patch,
        transparent=bg,
        padding=GEN_SCATTER_PADDING,
        padding_connectivity=8,
    )
    paste(grid, patch, int(r0), int(c0) + scatter_c0, transparent=bg)
    return int(r0), int(c0) + scatter_c0


def _paint_fragment(grid: np.ndarray, fragment: np.ndarray, color: int, top: int, left: int, bg: int) -> None:
    patch = np.full(fragment.shape, bg, dtype=int)
    patch[fragment] = int(color)
    paste(grid, patch, int(top), int(left), transparent=bg)


def _generate_case(with_meta: bool = False):
    for _ in range(GEN_MAX_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        bg, wall_color = random.sample(range(10), 2)
        profile = _make_wall_profile(height)
        wall_grid = _wall_grid((height, width), profile, wall_color, bg)
        wall_mask = wall_grid == wall_color

        max_fit_count = min(4, max(1, height // 4))
        fit_count = random.randint(1, max_fit_count)
        fit_specs = _choose_fit_windows(profile, fit_count)
        if len(fit_specs) < fit_count:
            continue

        target_right = 0
        fit_fragments: list[dict] = []
        for start, fragment in fit_specs:
            left_profile = _row_lefts(fragment)[1]
            seam_col = min(profile[start : start + fragment.shape[0]]) + 1
            target_left = seam_col
            target_right = max(target_right, target_left + fragment.shape[1] - 1)
            fit_fragments.append(
                {
                    "start": int(start),
                    "fragment": fragment,
                    "target_left": int(target_left),
                }
            )

        scatter_c0 = target_right + GEN_VISUAL_GAP
        if scatter_c0 >= width - 2:
            continue
        scatter_width = width - scatter_c0
        if scatter_width < max(int(spec["fragment"].shape[1]) for spec in fit_fragments):
            continue

        color_pool = [c for c in range(10) if c not in {bg, wall_color}]
        needed_colors = fit_count + random.choice((0, 1, 1, 2))
        if needed_colors > len(color_pool):
            continue
        random.shuffle(color_pool)

        out = wall_grid.copy()
        inp = wall_grid.copy()
        fit_checks: list[tuple[np.ndarray, tuple[int, int]]] = []
        unfit_checks: list[np.ndarray] = []
        meta = {
            "bg": bg,
            "wall_color": wall_color,
            "fit": [],
            "unfit": [],
            "wall_profile": tuple(profile),
        }

        color_index = 0
        try:
            for spec in fit_fragments:
                color = color_pool[color_index]
                color_index += 1
                fragment = spec["fragment"]
                top = spec["start"]
                left = spec["target_left"]
                _paint_fragment(out, fragment, color, top, left, bg)
                src_top, src_left = _scatter_fragment(inp, fragment, color, bg, scatter_c0)
                fit_checks.append((fragment.copy(), (top, left)))
                meta["fit"].append(
                    {
                        "color": color,
                        "target": (top, left),
                        "source": (src_top, src_left),
                        "shape": tuple(map(int, fragment.shape)),
                    }
                )
        except ValueError:
            continue

        unfit_count = needed_colors - fit_count
        ok = True
        for _ in range(unfit_count):
            color = color_pool[color_index]
            color_index += 1
            fragment = _make_unfit_fragment(wall_mask)
            try:
                src_top, src_left = _scatter_fragment(inp, fragment, color, bg, scatter_c0)
            except ValueError:
                ok = False
                break
            _paint_fragment(out, fragment, color, src_top, src_left, bg)
            unfit_checks.append(fragment.copy())
            meta["unfit"].append(
                {
                    "color": color,
                    "source": (src_top, src_left),
                    "shape": tuple(map(int, fragment.shape)),
                }
            )
        if not ok:
            continue

        semantic_ok = True
        occupied = wall_mask.copy()
        for fragment, target in fit_checks:
            cands_wall = _candidate_positions(fragment, wall_mask, wall_mask.copy())
            cands_occ = _candidate_positions(fragment, wall_mask, occupied)
            if cands_wall != [target] or cands_occ != [target]:
                semantic_ok = False
                break
            shifted = np.argwhere(fragment).astype(int, copy=False) + np.array(target, dtype=int)
            occupied[shifted[:, 0], shifted[:, 1]] = True
        if not semantic_ok:
            continue
        if any(_candidate_positions(fragment, wall_mask, wall_mask.copy()) for fragment in unfit_checks):
            continue

        produced = main(inp)
        if not np.array_equal(produced, out):
            continue
        if produced.shape != inp.shape or max(produced.shape) > 30:
            continue
        if with_meta:
            return inp, out, meta
        return inp
    raise RuntimeError("failed to generate a valid lock-fit puzzle")


def generate_input() -> np.ndarray:
    return _generate_case(with_meta=False)
