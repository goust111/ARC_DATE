from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIZES = (16, 20, 24, 28)
SPRITE_DIMS = (2, 4, 6)
CONNECTIVITY4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CONNECTIVITY8 = CONNECTIVITY4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PLACE_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if int(connectivity) == 4:
        return CONNECTIVITY4
    if int(connectivity) == 8:
        return CONNECTIVITY8
    raise ValueError("connectivity must be 4 or 8")


def _component_from(
    mask: np.ndarray,
    start: tuple[int, int],
    seen: np.ndarray,
    *,
    connectivity: int,
) -> np.ndarray:
    h, w = mask.shape
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True
    coords: list[tuple[int, int]] = []
    steps = _neighbors(connectivity)
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for dr, dc in steps:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return np.array(coords, dtype=int)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g) if bg is None else int(bg)
    color_list = sorted(set(map(int, np.unique(g))) - {bg}) if colors is None else [int(c) for c in colors if int(c) != bg]

    out: list[dict] = []
    if monochromatic:
        for color in color_list:
            mask = g == color
            seen = np.zeros_like(mask, dtype=bool)
            starts = np.argwhere(mask)
            for r, c in starts:
                if seen[r, c]:
                    continue
                coords = _component_from(mask, (int(r), int(c)), seen, connectivity=connectivity)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    mask = g != bg
    seen = np.zeros_like(mask, dtype=bool)
    starts = np.argwhere(mask)
    for r, c in starts:
        if seen[r, c]:
            continue
        coords = _component_from(mask, (int(r), int(c)), seen, connectivity=connectivity)
        out.append(
            {
                "color": None,
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _quadrant_bboxes(shape: tuple[int, int]) -> dict[int, tuple[int, int, int, int]]:
    h, w = map(int, shape)
    mid_r = h // 2
    mid_c = w // 2
    return {
        0: (0, 0, mid_r - 1, mid_c - 1),
        1: (0, mid_c, mid_r - 1, w - 1),
        2: (mid_r, 0, h - 1, mid_c - 1),
        3: (mid_r, mid_c, h - 1, w - 1),
    }


def _centered_topleft(
    outer_bbox: tuple[int, int, int, int],
    inner_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, outer_bbox)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    inner_h, inner_w = map(int, inner_shape)
    if inner_h > outer_h or inner_w > outer_w:
        raise ValueError("inner shape does not fit target quadrant")
    if (outer_h - inner_h) % 2 != 0 or (outer_w - inner_w) % 2 != 0:
        raise ValueError("center alignment is not integral")
    return r0 + (outer_h - inner_h) // 2, c0 + (outer_w - inner_w) // 2


def _find_guide(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    candidates: list[tuple[tuple[int, int, int, int], dict[int, int]]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False):
        b = comp["bbox"]
        if int(comp["area"]) != 4 or bbox_h(b) != 2 or bbox_w(b) != 2:
            continue
        block = crop_bbox(grid, b)
        if block.shape != (2, 2) or np.any(block == bg):
            continue
        values = list(map(int, block.flatten()))
        if len(set(values)) != 4:
            continue
        candidates.append(
            (
                b,
                {
                    int(block[0, 0]): 0,
                    int(block[0, 1]): 1,
                    int(block[1, 0]): 2,
                    int(block[1, 1]): 3,
                },
            )
        )
    assert len(candidates) == 1, "expected exactly one 2x2 four-color guide block"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Move each listed sprite to the centered quadrant named by the 2x2 color guide.

    Input structure:
    - The background is the dominant color.
    - One 4-connected multicolor component is a 2x2 guide block whose four cells all have distinct colors.
    - All other foreground objects are separated small monochrome sprites, with no shared color meaning except their visible paint.

    Rule:
    - Remove the guide block and clear every sprite from its original position.
    - Split the full canvas into top-left, top-right, bottom-left, and bottom-right quadrants.
    - For each sprite whose color appears in the guide, move that same-color sprite to the corresponding quadrant and center its bbox inside that quadrant; discard sprites whose colors are absent from the guide.

    Key insight:
    - The guide colors are object identities, not repaint instructions: the 2x2 guide is an address table telling each sprite where to go.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    guide_bbox, color_to_quadrant = _find_guide(g, bg)
    guide_r0, guide_c0, guide_r1, guide_c1 = guide_bbox

    cleaned = g.copy()
    cleaned[guide_r0 : guide_r1 + 1, guide_c0 : guide_c1 + 1] = bg

    sprites = find_components(cleaned, bg=bg, connectivity=4, monochromatic=True)
    seen_colors: set[int] = set()
    quadrants = _quadrant_bboxes(g.shape)

    out = np.full_like(g, bg)
    for sprite in sprites:
        color = int(sprite["color"])
        assert color not in seen_colors, "expected distinct sprite colors"
        seen_colors.add(color)
        if color not in color_to_quadrant:
            continue

        bbox = sprite["bbox"]
        patch = crop_bbox(cleaned, bbox)
        target_bbox = quadrants[color_to_quadrant[color]]
        target_r0, target_c0 = _centered_topleft(target_bbox, patch.shape)
        target_view = out[target_r0 : target_r0 + patch.shape[0], target_c0 : target_c0 + patch.shape[1]]
        assert not np.any((patch != bg) & (target_view != bg)), "centered sprites should not overlap"
        paste(out, patch, target_r0, target_c0, transparent=bg)

    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(map(int, np.argwhere(mask)[0]))
    coords = _component_from(mask.astype(bool), start, seen, connectivity=4)
    return int(coords.shape[0]) == int(np.count_nonzero(mask))


def _random_sprite_mask(max_h: int, max_w: int) -> np.ndarray:
    h_choices = [d for d in SPRITE_DIMS if d <= int(max_h)]
    w_choices = [d for d in SPRITE_DIMS if d <= int(max_w)]
    if not h_choices or not w_choices:
        raise ValueError("quadrant too small for sprite library")
    height = random.choice(h_choices)
    width = random.choice(w_choices)

    mask = np.ones((height, width), dtype=bool)
    min_area = max(3, (height * width + 1) // 2)
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    remove_budget = random.randint(0, max(1, (height * width) // 3))
    removed = 0

    for r, c in cells:
        if removed >= remove_budget or int(mask.sum()) <= min_area:
            break
        trial = mask.copy()
        trial[r, c] = False
        if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any() or not trial[:, -1].any():
            continue
        if not _mask_is_connected(trial):
            continue
        mask = trial
        removed += 1

    return mask.astype(int)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, *, pad: int = 1) -> bool:
    h0, w0 = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > h0 or c0 + w > w0:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, *, pad: int = 1) -> None:
    h0, w0 = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    *,
    forbidden_topleft: tuple[int, int] | None = None,
) -> tuple[int, int]:
    h, w = patch.shape
    height, width = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        r0 = random.randint(0, height - h)
        c0 = random.randint(0, width - w)
        if forbidden_topleft is not None and (r0, c0) == forbidden_topleft:
            continue
        if not _can_place(occupied, r0, c0, h, w, pad=1):
            continue
        paste(grid, patch, r0, c0, transparent=BG)
        _reserve(occupied, r0, c0, h, w, pad=1)
        return r0, c0
    raise RuntimeError("failed to place patch")


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    height = random.choice(GRID_SIZES)
    width = random.choice(GRID_SIZES)
    qh = height // 2
    qw = width // 2
    grid = np.full((height, width), BG, dtype=int)
    expected = np.full_like(grid, BG)
    occupied = np.zeros((height, width), dtype=bool)

    guide_colors = random.sample(PALETTE, 4)
    guide_layout = random.sample(guide_colors, 4)
    guide_block = np.array(
        [[guide_layout[0], guide_layout[1]], [guide_layout[2], guide_layout[3]]],
        dtype=int,
    )
    color_to_quadrant = {
        int(guide_block[0, 0]): 0,
        int(guide_block[0, 1]): 1,
        int(guide_block[1, 0]): 2,
        int(guide_block[1, 1]): 3,
    }

    mode = random.choice(("all", "missing", "decoy", "both"))
    if mode == "all":
        guided_colors = list(guide_colors)
        decoy_count = 0
    elif mode == "missing":
        guided_colors = random.sample(guide_colors, random.randint(2, 3))
        decoy_count = 0
    elif mode == "decoy":
        guided_colors = list(guide_colors)
        decoy_count = random.randint(1, 2)
    else:
        guided_colors = random.sample(guide_colors, random.randint(2, 3))
        decoy_count = random.randint(1, 2)

    remaining_colors = [c for c in PALETTE if c not in guide_colors]
    decoy_colors = random.sample(remaining_colors, decoy_count)
    sprite_colors = guided_colors + decoy_colors
    random.shuffle(sprite_colors)

    guide_patch = guide_block.copy()
    guide_h, guide_w = guide_patch.shape
    for _ in range(PLACE_ATTEMPTS):
        guide_r0 = random.randint(0, height - guide_h)
        guide_c0 = random.randint(0, width - guide_w)
        if not _can_place(occupied, guide_r0, guide_c0, guide_h, guide_w, pad=1):
            continue
        paste(grid, guide_patch, guide_r0, guide_c0)
        _reserve(occupied, guide_r0, guide_c0, guide_h, guide_w, pad=1)
        break
    else:
        raise RuntimeError("failed to place guide")

    quadrants = _quadrant_bboxes(grid.shape)
    placed_guided = 0
    for color in sprite_colors:
        mask = _random_sprite_mask(qh - 2, qw - 2)
        patch = np.where(mask, int(color), BG)
        target_bbox = quadrants[color_to_quadrant[color]] if color in color_to_quadrant else None
        forbidden_topleft = _centered_topleft(target_bbox, patch.shape) if target_bbox is not None else None
        r0, c0 = _place_patch(grid, occupied, patch, forbidden_topleft=forbidden_topleft)

        if color in color_to_quadrant:
            target_r0, target_c0 = forbidden_topleft
            paste(expected, patch, target_r0, target_c0, transparent=BG)
            placed_guided += 1

    assert placed_guided >= 1
    assert np.count_nonzero(expected) > 0
    assert not np.array_equal(grid, expected)

    guide_comp, guide_map = _find_guide(grid, BG)
    assert guide_comp == (guide_r0, guide_c0, guide_r0 + 1, guide_c0 + 1)
    assert guide_map == color_to_quadrant

    out = main(grid)
    assert np.array_equal(out, expected), "semantic oracle mismatch"
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, _expected = _build_case()
            return grid
        except Exception:
            continue
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
