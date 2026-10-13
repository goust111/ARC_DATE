from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)


def _full_separator_indices(g: np.ndarray) -> tuple[int, int]:
    maroon_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == MAROON)]
    maroon_cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == MAROON)]
    assert len(maroon_rows) == 1, "expected one complete maroon horizontal separator"
    assert len(maroon_cols) == 1, "expected one complete maroon vertical separator"

    split_r = maroon_rows[0]
    split_c = maroon_cols[0]
    assert split_r == g.shape[0] - split_r - 1, "subregions must have equal heights"
    assert split_c == g.shape[1] - split_c - 1, "subregions must have equal widths"
    return int(split_r), int(split_c)


def _subregions(g: np.ndarray, split_r: int, split_c: int) -> list[np.ndarray]:
    return [
        g[:split_r, :split_c],
        g[:split_r, split_c + 1 :],
        g[split_r + 1 :, :split_c],
        g[split_r + 1 :, split_c + 1 :],
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor four grey templates from partitioned panels and overlay them on one canvas.

    Input structure:
    - A black grid is divided by one full-width and one full-height one-pixel
      maroon separator into four equal rectangular subregions.
    - Each subregion contains one connected grey template shape.
    - Each subregion also contains exactly one isolated non-black, non-grey,
      non-maroon indicator pixel that gives that template's output color.

    Rule:
    - Identify the four subregions in reading order: top-left, top-right,
      bottom-left, bottom-right.
    - In each subregion, take all grey pixels as the template and take the
      single other colored pixel as its paint color.
    - Create a black output canvas the size of one subregion and draw the
      recolored templates at their same relative offsets, in the same reading
      order, so later templates overwrite earlier ones.

    Key insight:
    The grey pixels are shape placeholders; the isolated colored pixel in the
    same panel supplies the real color, and all four local panels collapse into
    one ordered overlay.
    """
    g = np.asarray(grid, dtype=int)
    split_r, split_c = _full_separator_indices(g)
    out = np.full((split_r, split_c), BLACK, dtype=int)

    for sub in _subregions(g, split_r, split_c):
        grey_mask = sub == GREY
        assert bool(grey_mask.any()), "each subregion must contain a grey template"

        indicator_mask = (sub != BLACK) & (sub != GREY) & (sub != MAROON)
        indicator = np.argwhere(indicator_mask)
        assert indicator.shape[0] == 1, "each subregion must contain one indicator pixel"
        color = int(sub[int(indicator[0, 0]), int(indicator[0, 1])])

        out[grey_mask] = color

    return out


# --------------- generator ---------------

SUB_H_RANGE = (6, 14)
SUB_W_RANGE = (6, 14)
SHAPE_AREA_RANGE = (4, 11)
PATCH_SIDE_MAX = 7
GENERATION_ATTEMPTS = 300
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.asarray(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    pts = [(int(r), int(c)) for r, c in coords]
    variants = []
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                transformed = []
                for r, c in pts:
                    a, b = (c, r) if swap else (r, c)
                    transformed.append((sr * a, sc * b))
                variants.append(_normalize_coords(np.array(transformed, dtype=int)))
    return min(variants)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_connected_patch(sub_h: int, sub_w: int, used_shapes: set[tuple[tuple[int, int], ...]]) -> np.ndarray | None:
    max_area = min(SHAPE_AREA_RANGE[1], (int(sub_h) * int(sub_w)) // 2)
    if max_area < SHAPE_AREA_RANGE[0]:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        patch_h = random.randint(2, min(PATCH_SIDE_MAX, int(sub_h)))
        patch_w = random.randint(2, min(PATCH_SIDE_MAX, int(sub_w)))
        target_area = random.randint(SHAPE_AREA_RANGE[0], min(max_area, patch_h * patch_w))

        mask = np.zeros((patch_h, patch_w), dtype=bool)
        r = random.randrange(patch_h)
        c = random.randrange(patch_w)
        mask[r, c] = True
        frontier: set[tuple[int, int]] = set()

        def add_frontier(rr: int, cc: int) -> None:
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < patch_h and 0 <= nc < patch_w and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))

        add_frontier(r, c)
        while int(mask.sum()) < target_area and frontier:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            mask[nr, nc] = True
            add_frontier(nr, nc)

        if int(mask.sum()) != target_area:
            continue

        cropped = _crop_mask(mask)
        if cropped.shape[0] == 1 and cropped.shape[1] == 1:
            continue
        if int(cropped.sum()) > max_area:
            continue

        key = _canonical_shape(cropped)
        if key in used_shapes:
            continue
        used_shapes.add(key)
        return cropped

    return None


def _indicator_candidates(shape_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape_mask.shape
    blocked = shape_mask.copy()
    for r, c in np.argwhere(shape_mask):
        r = int(r)
        c = int(c)
        for dr, dc in NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True
    return [(int(r), int(c)) for r, c in np.argwhere(~blocked)]


def _candidate_placements(
    patch: np.ndarray,
    sub_h: int,
    sub_w: int,
    require_overlap_with: np.ndarray | None,
) -> list[tuple[int, int, np.ndarray, list[tuple[int, int]]]]:
    ph, pw = map(int, patch.shape)
    choices = []
    for r0 in range(int(sub_h) - ph + 1):
        for c0 in range(int(sub_w) - pw + 1):
            mask = np.zeros((int(sub_h), int(sub_w)), dtype=bool)
            mask[r0 : r0 + ph, c0 : c0 + pw] = patch
            if require_overlap_with is not None and not bool(np.any(mask & require_overlap_with)):
                continue
            candidates = _indicator_candidates(mask)
            if candidates:
                choices.append((int(r0), int(c0), mask, candidates))
    return choices


def _compose_grid(panels: list[np.ndarray], sub_h: int, sub_w: int) -> np.ndarray:
    grid = np.full((2 * int(sub_h) + 1, 2 * int(sub_w) + 1), BLACK, dtype=int)
    grid[int(sub_h), :] = MAROON
    grid[:, int(sub_w)] = MAROON

    positions = [
        (0, 0),
        (0, int(sub_w) + 1),
        (int(sub_h) + 1, 0),
        (int(sub_h) + 1, int(sub_w) + 1),
    ]
    for panel, (r0, c0) in zip(panels, positions, strict=False):
        grid[r0 : r0 + int(sub_h), c0 : c0 + int(sub_w)] = panel
    return grid


def _expected_from_masks(masks: list[np.ndarray], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for mask, color in zip(masks, colors, strict=False):
        out[mask] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    sub_h = random.randint(*SUB_H_RANGE)
    sub_w = random.randint(*SUB_W_RANGE)
    colors = random.sample(COLOR_POOL, 4)
    force_overlap = random.random() < 0.55

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    masks: list[np.ndarray] = []
    panels: list[np.ndarray] = []
    union = np.zeros((sub_h, sub_w), dtype=bool)

    for i in range(4):
        patch = _random_connected_patch(sub_h, sub_w, used_shapes)
        if patch is None:
            return None

        require_overlap = union if force_overlap and i == 1 else None

        choices = _candidate_placements(patch, sub_h, sub_w, require_overlap)
        if not choices:
            choices = _candidate_placements(patch, sub_h, sub_w, None)
        if not choices:
            return None

        _r0, _c0, mask, candidates = random.choice(choices)
        panel = np.full((sub_h, sub_w), BLACK, dtype=int)
        panel[mask] = GREY
        ir, ic = random.choice(candidates)
        panel[ir, ic] = colors[i]

        masks.append(mask)
        panels.append(panel)
        union |= mask

    expected = _expected_from_masks(masks, colors, (sub_h, sub_w))
    if not all(int(color) in set(map(int, np.unique(expected))) for color in colors):
        return None

    overlap_count = np.zeros((sub_h, sub_w), dtype=int)
    for mask in masks:
        overlap_count += mask.astype(int)
    if force_overlap and int(overlap_count.max()) < 2:
        return None

    grid = _compose_grid(panels, sub_h, sub_w)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
