from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
CUP_COLORS = (1, 2, 3, 4, 7)
OPEN_SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_perimeter_size(height: int, width: int) -> int:
    return 2 * int(height) + 2 * int(width) - 4


def _gap_projection(side: str, gap_positions: list[int], height: int, width: int) -> int:
    if side in ("top", "bottom"):
        return sum(1 for pos in gap_positions if 1 <= int(pos) <= int(width) - 2)
    return sum(1 for pos in gap_positions if 1 <= int(pos) <= int(height) - 2)


def _contiguous(sorted_positions: list[int]) -> bool:
    return all(b - a == 1 for a, b in zip(sorted_positions, sorted_positions[1:]))


def _analyze_cup(grid: np.ndarray, coords: np.ndarray, color: int) -> dict | None:
    """Validate one monochromatic component as a cup and compute its effective area."""
    if int(coords.shape[0]) <= 1:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 4 or width < 4:
        return None

    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = crop == int(color)
    if np.any(mask[1:-1, 1:-1]):
        return None

    side_missing = {
        "top": [c for c in range(width) if not mask[0, c]],
        "bottom": [c for c in range(width) if not mask[height - 1, c]],
        "left": [r for r in range(height) if not mask[r, 0]],
        "right": [r for r in range(height) if not mask[r, width - 1]],
    }
    candidate_sides = [side for side, missing in side_missing.items() if missing]
    if len(candidate_sides) != 1:
        return None

    gap_side = candidate_sides[0]
    gap_positions = sorted(int(x) for x in side_missing[gap_side])
    if not _contiguous(gap_positions):
        return None
    if len(gap_positions) > 3:
        return None

    for side, missing in side_missing.items():
        if side != gap_side and missing:
            return None

    if int(coords.shape[0]) != _full_perimeter_size(height, width) - len(gap_positions):
        return None

    effective_area = (height - 2) * (width - 2) - _gap_projection(gap_side, gap_positions, height, width)
    if effective_area < 0:
        effective_area = 0

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "left": int(c0),
        "effective_area": int(effective_area),
        "gap_side": gap_side,
        "gap_size": int(len(gap_positions)),
    }


def _cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BACKGROUND):
        for coords in _component_coords(grid == color):
            info = _analyze_cup(grid, coords, color)
            if info is not None:
                cups.append(info)
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the cup outlines whose leak-adjusted interior area is largest, then summarize them as ordered color bars.

    Input structure:
    - The grid has a black background with 3-5 monochromatic rectangular cup outlines, each a 1-pixel-thick frame with one contiguous gap on exactly one side.
    - Different cups use different colors, and isolated single pixels may appear as same-color distractors.
    - Cups are separated, while the gap size and opening side vary across cups.

    Rule:
    - Ignore singleton distractors and identify each cup from its monochromatic 4-connected outline component.
    - For each cup, use its bounding box and missing border segment to compute the effective filled interior area: start from the sealed rectangle interior and subtract the interior cells directly exposed through the opening.
    - Find the maximum effective area, keep every cup with that value, sort the kept cups by bounding-box left edge, and output a solid vertical bar of that cup color for each kept cup.

    Key insight:
    - The visible outline size is not enough: the opening width determines how much of the interior "leaks", so the largest surviving bar height comes from leak-adjusted interior area rather than from outline area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cups = _cups(g)
    assert cups, "expected at least one valid cup"

    max_area = max(int(cup["effective_area"]) for cup in cups)
    kept = [cup for cup in cups if int(cup["effective_area"]) == max_area]
    kept.sort(key=lambda cup: int(cup["left"]))

    out = np.empty((max_area, len(kept)), dtype=int)
    for c, cup in enumerate(kept):
        out[:, c] = int(cup["color"])
    return out


def _cup_area(height: int, width: int, gap_size: int) -> int:
    return (int(height) - 2) * (int(width) - 2) - int(gap_size)


def _feasible_sides(height: int, width: int, gap_size: int) -> list[str]:
    sides = []
    if int(width) - 2 >= int(gap_size):
        sides.extend(["top", "bottom"])
    if int(height) - 2 >= int(gap_size):
        sides.extend(["left", "right"])
    return sides


def _render_cup(height: int, width: int, color: int, side: str, gap_size: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.zeros((int(height), int(width)), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)

    if side in ("top", "bottom"):
        start = int(rng.integers(1, int(width) - int(gap_size)))
        row = 0 if side == "top" else int(height) - 1
        patch[row, start : start + int(gap_size)] = BACKGROUND
    else:
        start = int(rng.integers(1, int(height) - int(gap_size)))
        col = 0 if side == "left" else int(width) - 1
        patch[start : start + int(gap_size), col] = BACKGROUND
    return patch


def _sample_lower_area_spec(max_area: int, color: int, rng: np.random.Generator) -> dict:
    for _ in range(200):
        height = int(rng.integers(4, 7))
        width = int(rng.integers(4, 7))
        max_gap = min(3, (width - 2) if rng.random() < 0.5 else (height - 2))
        max_gap = min(max_gap, max(width - 2, height - 2))
        gap_size = int(rng.integers(1, min(3, max(width - 2, height - 2)) + 1))
        side = random.choice(_feasible_sides(height, width, gap_size))
        area = _cup_area(height, width, gap_size)
        if 0 <= area < int(max_area):
            return {
                "height": height,
                "width": width,
                "color": int(color),
                "side": side,
                "gap_size": gap_size,
                "effective_area": area,
            }
    raise RuntimeError("failed to sample non-maximum cup")


def _candidate_positions(
    occupied: np.ndarray,
    patch_shape: tuple[int, int],
    *,
    required_lefts: set[int] | None = None,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = map(int, patch_shape)
    left_block = required_lefts or set()
    positions: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if c0 in left_block:
                continue
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    mask = patch != BACKGROUND
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    rr0 = max(0, r0 - 1)
    rr1 = min(grid.shape[0], r0 + ph + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(grid.shape[1], c0 + pw + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _sample_specs(rng: np.random.Generator) -> list[dict]:
    n_cups = int(rng.integers(3, 6))
    n_keep = int(rng.integers(1, min(3, n_cups - 1) + 1))
    colors = list(rng.choice(np.array(CUP_COLORS, dtype=int), size=n_cups, replace=False))

    pair_height = int(rng.integers(5, 7))
    pair_width = int(rng.integers(5, 7))
    max_gap = min(3, max(pair_height - 2, pair_width - 2))
    gap_small = int(rng.integers(1, max_gap))
    gap_large = int(rng.integers(gap_small + 1, max_gap + 1))

    max_area = _cup_area(pair_height, pair_width, gap_small)

    specs = [
        {
            "height": pair_height,
            "width": pair_width,
            "color": int(colors[0]),
            "side": None,
            "gap_size": gap_small,
            "effective_area": max_area,
        },
        {
            "height": pair_height,
            "width": pair_width,
            "color": int(colors[1]),
            "side": None,
            "gap_size": gap_large,
            "effective_area": _cup_area(pair_height, pair_width, gap_large),
        },
    ]

    for idx in range(2, 1 + n_keep):
        specs.append(
            {
                "height": pair_height,
                "width": pair_width,
                "color": int(colors[idx]),
                "side": None,
                "gap_size": gap_small,
                "effective_area": max_area,
            }
        )

    for idx in range(1 + n_keep, n_cups):
        specs.append(_sample_lower_area_spec(max_area, int(colors[idx]), rng))

    chosen_sides: list[str] = []
    for spec in specs:
        feasible = _feasible_sides(int(spec["height"]), int(spec["width"]), int(spec["gap_size"]))
        spec["side"] = random.choice(feasible)
        chosen_sides.append(str(spec["side"]))

    if len(set(chosen_sides)) == 1:
        spec = specs[-1]
        feasible = [side for side in _feasible_sides(int(spec["height"]), int(spec["width"]), int(spec["gap_size"])) if side != spec["side"]]
        if feasible:
            spec["side"] = random.choice(feasible)

    random.shuffle(specs)
    return specs


def _expected_output(specs: list[dict]) -> np.ndarray:
    max_area = max(int(spec["effective_area"]) for spec in specs)
    kept = [spec for spec in specs if int(spec["effective_area"]) == max_area]
    kept.sort(key=lambda spec: int(spec["left"]))
    out = np.empty((max_area, len(kept)), dtype=int)
    for c, spec in enumerate(kept):
        out[:, c] = int(spec["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _sample_specs(rng)
    grid_h = int(rng.integers(16, 23))
    grid_w = int(rng.integers(16, 23))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    max_area = max(int(spec["effective_area"]) for spec in specs)
    max_lefts: set[int] = set()

    specs.sort(key=lambda spec: (-int(spec["height"]) * int(spec["width"]), -int(spec["effective_area"])))
    for spec in specs:
        patch = _render_cup(
            int(spec["height"]),
            int(spec["width"]),
            int(spec["color"]),
            str(spec["side"]),
            int(spec["gap_size"]),
            rng,
        )
        left_constraints = max_lefts if int(spec["effective_area"]) == max_area else None
        positions = _candidate_positions(occupied, patch.shape, required_lefts=left_constraints)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(0, len(positions)))]
        _place_patch(grid, occupied, patch, r0, c0)
        spec["left"] = int(c0)
        if int(spec["effective_area"]) == max_area:
            max_lefts.add(int(c0))

    n_noise = int(rng.integers(2, 4))
    used_noise: set[tuple[int, int]] = set()
    for _ in range(n_noise):
        choices = [
            (r, c)
            for r in range(grid_h)
            for c in range(grid_w)
            if not occupied[max(0, r - 1) : min(grid_h, r + 2), max(0, c - 1) : min(grid_w, c + 2)].any()
            and (r, c) not in used_noise
        ]
        if not choices:
            return None
        r, c = choices[int(rng.integers(0, len(choices)))]
        grid[r, c] = int(specs[int(rng.integers(0, len(specs)))]["color"])
        occupied[max(0, r - 1) : min(grid_h, r + 2), max(0, c - 1) : min(grid_w, c + 2)] = True
        used_noise.add((r, c))

    expected = _expected_output(specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cup-area puzzle input")
