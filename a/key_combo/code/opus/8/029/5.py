from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAROON = 9

PALETTE = (BLUE, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _orbit(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    assert h == w, "expected a square grid"
    return sorted(
        {
            (int(r), int(c)),
            (h - 1 - int(r), int(c)),
            (int(r), w - 1 - int(c)),
            (h - 1 - int(r), w - 1 - int(c)),
        }
    )


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _is_center_symmetric(grid: np.ndarray) -> bool:
    return bool(np.array_equal(grid, np.flipud(grid)) and np.array_equal(grid, np.fliplr(grid)))


def _attempt_repair(grid: np.ndarray, hole_color: int) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int).copy()
    hole_color = int(hole_color)
    if not np.any(g == hole_color):
        return None

    while True:
        changed = False
        for r, c in np.argwhere(g == hole_color):
            rr = int(r)
            cc = int(c)
            values = {
                int(g[orow, ocol])
                for orow, ocol in _orbit(g.shape, rr, cc)
                if int(g[orow, ocol]) != hole_color
            }
            if len(values) > 1:
                return None
            if len(values) == 1:
                g[rr, cc] = values.pop()
                changed = True
        if not changed:
            break

    if np.any(g == hole_color):
        return None
    if not _is_center_symmetric(g):
        return None
    return g


def _infer_hole_color_and_repaired(grid: np.ndarray) -> tuple[int, np.ndarray]:
    candidates: list[tuple[int, np.ndarray]] = []
    for color in map(int, np.unique(grid)):
        repaired = _attempt_repair(grid, color)
        if repaired is not None:
            candidates.append((int(color), repaired))
    assert len(candidates) == 1, "expected exactly one repairable occlusion color"
    return candidates[0]


def _guide_regions(repaired: np.ndarray) -> tuple[list[tuple[int, int]], list[np.ndarray], list[int]]:
    regions = _components_mask(repaired == BLACK)
    region_id = -np.ones(repaired.shape, dtype=int)
    for idx, coords in enumerate(regions):
        region_id[coords[:, 0], coords[:, 1]] = idx

    guides = [tuple(map(int, rc)) for rc in np.argwhere(repaired == RED)]
    assert guides, "expected repaired guide pixels"

    touched: list[int] = []
    for r, c in guides:
        adjacent = {
            int(region_id[nr, nc])
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < repaired.shape[0] and 0 <= nc < repaired.shape[1] and int(repaired[nr, nc]) == BLACK
        }
        assert len(adjacent) == 1, "each guide must touch one black region"
        touched.append(adjacent.pop())

    assert len(set(touched)) == len(touched), "a black region must not have multiple guides"
    return guides, regions, touched


def _apply_rule_from_repaired(repaired: np.ndarray) -> np.ndarray:
    out = np.asarray(repaired, dtype=int).copy()
    guides, regions, touched = _guide_regions(out)
    for region_idx in touched:
        coords = regions[region_idx]
        out[coords[:, 0], coords[:, 1]] = RED
    for r, c in guides:
        out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the symmetric wall pattern, then let the revealed red guide pixels repaint their adjacent black chambers before disappearing.

    Input structure:
    - The grid is square, with black background regions separated by a horizontally and vertically symmetric foreground wall pattern.
    - The wall pattern uses blue, green, and/or yellow, and some wall cells are covered by maroon occlusion pixels.
    - After the occlusions are repaired, one or more red guide pixels lie on the wall and each guide touches exactly one 4-connected black chamber.

    Rule:
    - Identify the maroon-like occlusion color as the unique color whose cells can all be replaced from their horizontal, vertical, and 180-degree symmetry orbit.
    - Repair every occluded cell with the unique visible color from that orbit, producing the fully symmetric wall pattern.
    - Find every red guide pixel in the repaired pattern, recolor its unique adjacent black chamber to red, and then erase the guide pixel itself back to black.

    Key insight:
    - Symmetry repair is not the final goal: it reveals hidden red guide pixels whose location, not their pre-occlusion placeholder, determines which background chambers must be repainted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    hole_color, repaired = _infer_hole_color_and_repaired(g)
    assert hole_color == MAROON, "expected maroon occlusions"
    return _apply_rule_from_repaired(repaired)


def _wall_rows(n: int, divider_count: int) -> list[int]:
    mid = n // 2
    candidates = list(range(2, mid, 2))
    top = sorted(random.sample(candidates, divider_count))
    rows = sorted([0, n - 1, *top, *[n - 1 - r for r in top]])
    return rows


def _band_intervals(wall_rows: list[int]) -> list[tuple[int, int]]:
    return [(wall_rows[i] + 1, wall_rows[i + 1] - 1) for i in range(len(wall_rows) - 1)]


def _build_full_pattern() -> tuple[np.ndarray, dict]:
    divider_count = random.choice([1, 2])
    n_choices = [11, 13, 15] if divider_count == 1 else [13, 15]
    n = random.choice(n_choices)
    mid = n // 2

    colors = list(random.sample(PALETTE, random.choice([2, 3])))
    frame_color = colors[0]
    spine_color = colors[1] if len(colors) >= 2 else colors[0]
    divider_color = colors[2] if len(colors) >= 3 else random.choice(colors[:2])

    wall_rows = _wall_rows(n, divider_count)
    bands = _band_intervals(wall_rows)

    full = np.zeros((n, n), dtype=int)
    full[0, :] = frame_color
    full[-1, :] = frame_color
    full[:, 0] = frame_color
    full[:, -1] = frame_color
    full[:, mid] = spine_color
    for row in wall_rows[1:-1]:
        full[row, :] = divider_color

    guide_mode = random.choices(["axis_pair", "quartet"], weights=[3, 5])[0]
    if guide_mode == "axis_pair":
        guide_rows = [mid]
    else:
        top_bands = [band for band in bands if band[1] < mid]
        start, end = random.choice(top_bands)
        row = (start + end) // 2
        guide_rows = [row, n - 1 - row]

    guides = {(row, 0) for row in guide_rows} | {(row, n - 1) for row in guide_rows}
    for r, c in guides:
        full[r, c] = RED

    transpose = bool(random.getrandbits(1))
    if transpose:
        full = full.T.copy()
        guides = {(c, r) for r, c in guides}

    metadata = {
        "divider_count": divider_count,
        "guide_mode": guide_mode,
        "guides": sorted(guides),
        "transpose": transpose,
    }
    return full, metadata


def _occlude(full: np.ndarray, guides: list[tuple[int, int]]) -> tuple[np.ndarray, set[tuple[int, int]]]:
    inp = full.copy()
    guide_set = set(guides)
    if len(guides) == 2:
        hidden_count = 1
    else:
        hidden_count = random.choice([1, 2])

    hidden_guides = set(random.sample(guides, hidden_count))
    for r, c in hidden_guides:
        inp[r, c] = MAROON

    non_guide_walls = [tuple(map(int, rc)) for rc in np.argwhere((full != BLACK) & (full != RED))]
    random.shuffle(non_guide_walls)
    target_extra = random.randint(4, 10)
    extras = 0
    for r, c in non_guide_walls:
        if (r, c) in guide_set:
            continue
        trial = inp.copy()
        trial[r, c] = MAROON
        repaired = _attempt_repair(trial, MAROON)
        if repaired is None or not np.array_equal(repaired, full):
            continue
        inp = trial
        extras += 1
        if extras >= target_extra:
            break

    assert extras >= 4, "expected several non-guide occlusions"
    return inp, hidden_guides


def _validate_generated(inp: np.ndarray, full: np.ndarray, hidden_guides: set[tuple[int, int]]) -> None:
    hole_color, repaired = _infer_hole_color_and_repaired(inp)
    assert hole_color == MAROON
    assert np.array_equal(repaired, full)

    guides, regions, touched = _guide_regions(full)
    assert len(hidden_guides) in {1, 2}
    assert hidden_guides <= set(guides)
    assert all(int(inp[r, c]) == MAROON for r, c in hidden_guides)
    assert any(int(inp[r, c]) == RED for r, c in guides if (r, c) not in hidden_guides)
    assert len(regions) >= len(touched) + 1, "need non-target background regions as counterexamples"
    assert all(len(regions[idx]) >= 2 for idx in touched), "target regions should be visible chambers"
    assert int(np.count_nonzero(inp == MAROON)) >= len(hidden_guides) + 4

    expected = _apply_rule_from_repaired(full)
    actual = main(inp)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(expected, full), "guides must disappear or recolor something"
    assert not np.array_equal(inp, expected), "the task must not be identity"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        full, metadata = _build_full_pattern()
        inp, hidden_guides = _occlude(full, metadata["guides"])
        try:
            _validate_generated(inp, full, hidden_guides)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
