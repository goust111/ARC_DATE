from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = {BLUE, RED, GREEN, YELLOW}
CORNER_ORDER = ("nw", "ne", "se", "sw")
BASE_CW = [BLUE, RED, YELLOW, GREEN]
BASE_CCW = [BLUE, GREEN, YELLOW, RED]
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate_colors_right(seq: list[int], shift: int) -> list[int]:
    shift %= 4
    return seq[-shift:] + seq[:-shift] if shift else seq[:]


def _chirality_from_corners(corner_colors: list[int]) -> str | None:
    for shift in range(4):
        if corner_colors == _rotate_colors_right(BASE_CW, shift):
            return "cw"
        if corner_colors == _rotate_colors_right(BASE_CCW, shift):
            return "ccw"
    return None


def _rotate_cells_about_center(cells: np.ndarray, turns_ccw: int) -> np.ndarray:
    turns_ccw %= 4
    if turns_ccw == 0:
        return cells.copy()

    r0, c0, r1, c1 = _bbox_coords(cells)
    rsum = r0 + r1
    csum = c0 + c1
    out: list[tuple[int, int]] = []
    for r, c in cells:
        dr2 = 2 * int(r) - rsum
        dc2 = 2 * int(c) - csum
        for _ in range(turns_ccw):
            dr2, dc2 = -dc2, dr2
        nr_num = rsum + dr2
        nc_num = csum + dc2
        assert nr_num % 2 == 0 and nc_num % 2 == 0, "sprite center parity must support quarter-turn rotation"
        out.append((nr_num // 2, nc_num // 2))
    rotated = np.array(out, dtype=int)
    assert len({(int(r), int(c)) for r, c in rotated.tolist()}) == len(out)
    return rotated


def _find_valid_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    grey_components = [
        {"id": idx, "cells": cells, "bbox": _bbox_coords(cells)}
        for idx, cells in enumerate(_components(g == GREY, connectivity=8))
    ]

    candidates: list[dict] = []
    for r0 in range(h):
        for r1 in range(r0 + 2, h):
            side = r1 - r0
            for c0 in range(w - side):
                c1 = c0 + side
                corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
                corner_colors = [int(g[r, c]) for r, c in corners]
                if set(corner_colors) != MARKER_COLORS:
                    continue
                chirality = _chirality_from_corners(corner_colors)
                if chirality is None:
                    continue

                interior = g[r0 + 1 : r1, c0 + 1 : c1]
                if np.any((interior != BLACK) & (interior != GREY)):
                    continue

                intersecting: list[dict] = []
                fully_inside: list[dict] = []
                for comp in grey_components:
                    inside = [
                        r0 < int(r) < r1 and c0 < int(c) < c1
                        for r, c in comp["cells"]
                    ]
                    if any(inside):
                        intersecting.append(comp)
                    if all(inside):
                        fully_inside.append(comp)
                if len(intersecting) != 1 or len(fully_inside) != 1:
                    continue
                if intersecting[0]["id"] != fully_inside[0]["id"]:
                    continue

                blue_index = corner_colors.index(BLUE)
                turns_ccw = blue_index if chirality == "ccw" else (-blue_index) % 4
                candidates.append(
                    {
                        "component_id": fully_inside[0]["id"],
                        "cells": fully_inside[0]["cells"],
                        "bbox": (r0, c0, r1, c1),
                        "turns_ccw": turns_ccw,
                        "side": side,
                    }
                )

    best_by_component: dict[int, dict] = {}
    for cand in sorted(candidates, key=lambda item: (item["side"], item["bbox"])):
        best_by_component.setdefault(cand["component_id"], cand)
    return list(best_by_component.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each enclosed grey sprite according to the phase and chirality of its four corner markers.

    Input structure:
    - The grid has a black background with grey sprites and scattered single-pixel blue, red, yellow, and green dots.
    - Some quartets of colored dots occupy the four corners of an axis-aligned square, one color per corner.
    - A valid corner quartet strictly encloses exactly one grey sprite; extra colored dots and unframed grey sprites may also appear.

    Rule:
    - Preserve the whole grid unless a four-corner marker square is valid.
    - For each valid square, read the blue corner's phase relative to the baseline layout to get the quarter-turn amount, and read the clockwise/counterclockwise color winding to get the turn direction.
    - Rotate the enclosed grey sprite about its own center by that instruction, redraw it in place, and leave all marker dots, decoys, and untouched sprites unchanged.

    Key insight:
    - The corner markers encode a two-part rotation command: blue's corner gives the angle, while the color winding decides whether that angle is applied clockwise or counterclockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for group in _find_valid_groups(g):
        original = group["cells"]
        rotated = _rotate_cells_about_center(original, group["turns_ccw"])
        r0, c0, r1, c1 = group["bbox"]
        assert np.all((r0 < rotated[:, 0]) & (rotated[:, 0] < r1))
        assert np.all((c0 < rotated[:, 1]) & (rotated[:, 1] < c1))
        out[original[:, 0], original[:, 1]] = BLACK
        out[rotated[:, 0], rotated[:, 1]] = GREY
    return out


SPRITE_TEMPLATES = [
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1, 1, 0], [0, 0, 1, 0, 1], [0, 1, 1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 1, 0]], dtype=int),
    np.array([[1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]], dtype=int),
    np.array([[1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 0, 1, 1]], dtype=int),
]


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    cur = mask.copy()
    for _ in range(4):
        variants.append(_normalize_mask(cur))
        variants.append(_normalize_mask(np.fliplr(cur)))
        cur = np.rot90(cur)
    return variants


def _is_non_symmetric(mask: np.ndarray) -> bool:
    base = _normalize_mask(mask)
    same = 0
    for variant in _mask_variants(base):
        if variant.shape == base.shape and np.array_equal(variant, base):
            same += 1
    return same == 1


assert all(_is_non_symmetric(mask.astype(bool)) for mask in SPRITE_TEMPLATES)


def _sample_sprite_mask() -> np.ndarray:
    mask = random.choice(SPRITE_TEMPLATES).copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, k=random.randrange(4))
    mask = _normalize_mask(mask.astype(bool))
    assert mask.shape[0] % 2 == mask.shape[1] % 2
    return mask


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _corner_sequence(chirality: str, phase: int) -> list[int]:
    base = BASE_CW if chirality == "cw" else BASE_CCW
    return _rotate_colors_right(base, phase)


def _build_target(
    grid: np.ndarray,
    blocked: np.ndarray,
    turns_ccw: int,
) -> dict | None:
    h, w = grid.shape
    mask = _sample_sprite_mask()
    ph, pw = mask.shape
    chirality = random.choice(["cw", "ccw"])
    phase = turns_ccw if chirality == "ccw" else (-turns_ccw) % 4
    corner_colors = _corner_sequence(chirality, phase)

    min_side = max(ph, pw) + 3
    max_side = min(h, w) - 3
    if min_side > max_side:
        return None

    for _ in range(120):
        side = random.randint(min_side, min(max_side, min_side + 8))
        r0 = random.randint(1, h - side - 2)
        c0 = random.randint(1, w - side - 2)
        r1, c1 = r0 + side, c0 + side
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue

        top_lo, top_hi = r0 + 2, r1 - ph - 1
        left_lo, left_hi = c0 + 2, c1 - pw - 1
        if top_lo > top_hi or left_lo > left_hi:
            continue
        sr = random.randint(top_lo, top_hi)
        sc = random.randint(left_lo, left_hi)

        cells = np.argwhere(mask) + np.array([sr, sc])
        rotated = _rotate_cells_about_center(cells, turns_ccw)
        if not np.all((r0 < rotated[:, 0]) & (rotated[:, 0] < r1)):
            continue
        if not np.all((c0 < rotated[:, 1]) & (rotated[:, 1] < c1)):
            continue

        grid[cells[:, 0], cells[:, 1]] = GREY
        corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
        for (rr, cc), color in zip(corners, corner_colors, strict=True):
            grid[rr, cc] = color
        _reserve(blocked, r0, c0, r1, c1, pad=1)
        return {
            "input_cells": cells,
            "output_cells": rotated,
            "bbox": (r0, c0, r1, c1),
            "turns_ccw": turns_ccw,
        }
    return None


def _place_bystander(grid: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = grid.shape
    mask = _sample_sprite_mask()
    ph, pw = mask.shape
    for _ in range(100):
        r0 = random.randint(1, h - ph - 2)
        c0 = random.randint(1, w - pw - 2)
        r1, c1 = r0 + ph - 1, c0 + pw - 1
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue
        cells = np.argwhere(mask) + np.array([r0, c0])
        grid[cells[:, 0], cells[:, 1]] = GREY
        _reserve(blocked, r0, c0, r1, c1, pad=1)
        return True
    return False


def _place_partial_square_decoy(grid: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(80):
        side = random.randint(4, min(h, w) - 3)
        r0 = random.randint(1, h - side - 2)
        c0 = random.randint(1, w - side - 2)
        r1, c1 = r0 + side, c0 + side
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue
        corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
        chosen = random.sample(corners, k=random.randint(2, 3))
        colors = random.sample(list(MARKER_COLORS), k=len(chosen))
        for (rr, cc), color in zip(chosen, colors, strict=True):
            grid[rr, cc] = color
        _reserve(blocked, r0, c0, r1, c1, pad=0)
        return True
    return False


def _place_rectangle_decoy(grid: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(80):
        height = random.randint(3, h - 4)
        width = random.randint(3, w - 4)
        if height == width:
            continue
        r0 = random.randint(1, h - height - 2)
        c0 = random.randint(1, w - width - 2)
        r1, c1 = r0 + height, c0 + width
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue
        corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
        colors = random.sample(list(MARKER_COLORS), k=4)
        for (rr, cc), color in zip(corners, colors, strict=True):
            grid[rr, cc] = color
        _reserve(blocked, r0, c0, r1, c1, pad=0)
        return True
    return False


def _place_noise_dots(grid: np.ndarray, blocked: np.ndarray, count: int) -> bool:
    h, w = grid.shape
    choices = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c] and grid[r, c] == BLACK]
    if len(choices) < count:
        return False
    for r, c in random.sample(choices, k=count):
        grid[r, c] = random.choice(tuple(MARKER_COLORS))
        blocked[r, c] = True
    return True


def _expected_from_targets(grid: np.ndarray, targets: list[dict]) -> np.ndarray:
    out = grid.copy()
    for target in targets:
        inp = target["input_cells"]
        out[inp[:, 0], inp[:, 1]] = BLACK
        rot = target["output_cells"]
        out[rot[:, 0], rot[:, 1]] = GREY
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    target_count = random.randint(1, 3)
    turns_plan = [random.randrange(4) for _ in range(target_count)]
    if all(turn == 0 for turn in turns_plan):
        turns_plan[random.randrange(target_count)] = random.choice([1, 2, 3])
    random.shuffle(turns_plan)

    targets: list[dict] = []
    for turns_ccw in turns_plan:
        target = _build_target(grid, blocked, turns_ccw)
        if target is None:
            return None
        targets.append(target)

    for _ in range(random.randint(0, 2)):
        if not _place_bystander(grid, blocked):
            return None

    decoy_builders = [_place_partial_square_decoy, _place_rectangle_decoy]
    random.shuffle(decoy_builders)
    decoy_total = random.randint(1, 3)
    placed_decoys = 0
    for builder in decoy_builders:
        if placed_decoys >= decoy_total:
            break
        if builder(grid, blocked):
            placed_decoys += 1
    while placed_decoys < decoy_total:
        builder = random.choice(decoy_builders)
        if not builder(grid, blocked):
            break
        placed_decoys += 1
    if placed_decoys == 0:
        return None

    if not _place_noise_dots(grid, blocked, count=random.randint(1, 5)):
        return None

    expected = _expected_from_targets(grid, targets)
    k = random.randrange(4)
    if k:
        grid = np.rot90(grid, k=k)
        expected = np.rot90(expected, k=k)

    groups = _find_valid_groups(grid)
    if len(groups) != target_count:
        return None
    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 3")
