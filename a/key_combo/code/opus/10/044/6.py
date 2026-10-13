from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4

QUADRANTS = ("tl", "tr", "bl", "br")
_PATTERN_BAG: list[tuple[str, ...]] = []
_BOX_BAG: list[tuple[int, int]] = []
_SIZE_BAG: list[int] = []


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_inner_edge(mask: np.ndarray, quadrant: str) -> bool:
    if quadrant == "tl":
        return bool(mask[-1, :].any() or mask[:, -1].any())
    if quadrant == "tr":
        return bool(mask[-1, :].any() or mask[:, 0].any())
    if quadrant == "bl":
        return bool(mask[0, :].any() or mask[:, -1].any())
    if quadrant == "br":
        return bool(mask[0, :].any() or mask[:, 0].any())
    raise ValueError(f"unknown quadrant {quadrant}")


def _transform_patch(mask: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tl":
        return np.rot90(mask, 2)
    if quadrant == "tr":
        return np.fliplr(mask)
    if quadrant == "bl":
        return np.flipud(mask)
    if quadrant == "br":
        return mask.copy()
    raise ValueError(f"unknown quadrant {quadrant}")


def _place_corner(out: np.ndarray, mask: np.ndarray, quadrant: str) -> None:
    h, w = mask.shape
    if quadrant == "tl":
        r0, c0 = 0, 0
    elif quadrant == "tr":
        r0, c0 = 0, out.shape[1] - w
    elif quadrant == "bl":
        r0, c0 = out.shape[0] - h, 0
    elif quadrant == "br":
        r0, c0 = out.shape[0] - h, out.shape[1] - w
    else:
        raise ValueError(f"unknown quadrant {quadrant}")

    region = out[r0 : r0 + h, c0 : c0 + w]
    region[mask] = BLUE


def _detect_center_lines(grid: np.ndarray) -> tuple[int, int]:
    red_rows = np.where(np.all(grid == RED, axis=1))[0]
    red_cols = np.where(np.all(grid == RED, axis=0))[0]
    assert len(red_rows) == 1 and len(red_cols) == 1, "expected one full red row and one full red column"
    center_r = int(red_rows[0])
    center_c = int(red_cols[0])
    assert center_r == grid.shape[0] // 2 and center_c == grid.shape[1] // 2, "cross must be centered"
    return center_r, center_c


def _extract_quadrant_patch(grid: np.ndarray, center_r: int, center_c: int, quadrant: str) -> np.ndarray | None:
    if quadrant == "tl":
        sub = grid[:center_r, :center_c] == BLUE
    elif quadrant == "tr":
        sub = grid[:center_r, center_c + 1 :] == BLUE
    elif quadrant == "bl":
        sub = grid[center_r + 1 :, :center_c] == BLUE
    elif quadrant == "br":
        sub = grid[center_r + 1 :, center_c + 1 :] == BLUE
    else:
        raise ValueError(f"unknown quadrant {quadrant}")

    box = _bbox(sub)
    if box is None:
        return None
    return _crop(sub, box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the centered sprite into quadrant fragments, slide them diagonally to the corners, and mirror them into a four-corner mosaic.

    Input structure:
    - The grid is an odd-sized 2D canvas with black background, four yellow corner markers, and a full red horizontal and vertical line crossing at the center.
    - Blue sprite pixels appear only off the red cross and occupy up to four quadrant fragments around the center.
    - The red cross indicates the cut lines; any sprite pixels on those lines belong to the cut and are not part of any fragment.

    Rule:
    - Identify the centered red cross and split the blue pixels into the four strict quadrants around it.
    - Erase the entire input scene, including the red cross and yellow corner markers.
    - For each nonempty quadrant fragment, slide its bounding box diagonally until it is flush with the matching grid corner.
    - After sliding, transform the fragment so it faces the center: top-left rotates 180 degrees, top-right flips left-right, bottom-left flips up-down, and bottom-right stays unchanged.

    Key insight:
    - The output is determined by chaining three steps on the same cut fragments: quadrant-based cutting, quadrant-to-corner diagonal sliding, and a quadrant-specific mirror transform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1, "expected an odd-sized 2D grid"
    assert int(g[0, 0]) == YELLOW and int(g[0, -1]) == YELLOW and int(g[-1, 0]) == YELLOW and int(g[-1, -1]) == YELLOW

    center_r, center_c = _detect_center_lines(g)

    out = np.full_like(g, BG)
    for quadrant in QUADRANTS:
        patch = _extract_quadrant_patch(g, center_r, center_c, quadrant)
        if patch is None:
            continue
        _place_corner(out, _transform_patch(patch, quadrant), quadrant)
    return out


def _local_masks_by_area(height: int, width: int, quadrant: str) -> dict[int, list[np.ndarray]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    out: dict[int, list[np.ndarray]] = {}
    for bits in range(1, 1 << (height * width)):
        mask = np.zeros((height, width), dtype=bool)
        for i, (r, c) in enumerate(cells):
            if bits & (1 << i):
                mask[r, c] = True
        if not _is_connected(mask):
            continue
        if not _touches_inner_edge(mask, quadrant):
            continue
        out.setdefault(int(mask.sum()), []).append(mask)
    return out


_LOCAL_MASK_CACHE = {
    (h, w, q): _local_masks_by_area(h, w, q)
    for h, w in ((2, 2), (2, 3), (3, 2))
    for q in QUADRANTS
}


def _draw_from_bag(bag: list, values: list) -> object:
    if not bag:
        bag.extend(random.sample(values, k=len(values)))
    return bag.pop()


def _choose_active_quadrants() -> tuple[str, ...]:
    patterns = [
        ("tl", "tr", "bl", "br"),
        ("tl", "tr", "bl", "br"),
        ("tl", "tr", "bl", "br"),
        ("tl", "tr", "bl", "br"),
        ("tr", "bl", "br"),
        ("tl", "bl", "br"),
        ("tl", "tr", "br"),
        ("tl", "tr", "bl"),
    ]
    return tuple(_draw_from_bag(_PATTERN_BAG, patterns))


def _choose_box() -> tuple[int, int]:
    return tuple(_draw_from_bag(_BOX_BAG, [(2, 2), (2, 3), (3, 2)]))


def _choose_size() -> int:
    return int(_draw_from_bag(_SIZE_BAG, [13, 15, 17, 19]))


def _partition_total(total: int, active: tuple[str, ...], height: int, width: int) -> dict[str, int]:
    counts = {quadrant: 1 for quadrant in active}
    room = {quadrant: height * width - 1 for quadrant in active}
    extra = total - len(active)
    while extra > 0:
        choices = [quadrant for quadrant in active if room[quadrant] > 0]
        if not choices:
            raise ValueError("not enough room to distribute visible cells")
        quadrant = random.choice(choices)
        counts[quadrant] += 1
        room[quadrant] -= 1
        extra -= 1
    return counts


def _assemble_visible_mask(size: int, half_h: int, half_w: int, local_masks: dict[str, np.ndarray]) -> np.ndarray:
    center = size // 2
    visible = np.zeros((size, size), dtype=bool)
    offsets = {
        "tl": (center - half_h, center - half_w),
        "tr": (center - half_h, center + 1),
        "bl": (center + 1, center - half_w),
        "br": (center + 1, center + 1),
    }
    for quadrant, mask in local_masks.items():
        r0, c0 = offsets[quadrant]
        visible[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] |= mask
    return visible


def _latent_sprite_mask(visible: np.ndarray, center: int, half_h: int, half_w: int) -> np.ndarray:
    latent = visible.copy()
    latent[center - half_h : center + half_h + 1, center] = True
    latent[center, center - half_w : center + half_w + 1] = True
    return latent


def _is_degenerate(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _render_input(size: int, visible: np.ndarray) -> np.ndarray:
    center = size // 2
    grid = np.full((size, size), BG, dtype=int)
    grid[visible] = BLUE
    grid[center, :] = RED
    grid[:, center] = RED
    grid[0, 0] = YELLOW
    grid[0, -1] = YELLOW
    grid[-1, 0] = YELLOW
    grid[-1, -1] = YELLOW
    return grid


def _expected_output(size: int, local_masks: dict[str, np.ndarray]) -> np.ndarray:
    out = np.full((size, size), BG, dtype=int)
    for quadrant, local in local_masks.items():
        box = _bbox(local)
        if box is None:
            continue
        patch = _crop(local, box)
        _place_corner(out, _transform_patch(patch, quadrant), quadrant)
    return out


def _validate_input_semantics(grid: np.ndarray, active: tuple[str, ...], half_h: int, half_w: int) -> None:
    assert grid.shape[0] == grid.shape[1]
    assert grid.shape[0] % 2 == 1 and grid.shape[0] <= 30
    center = grid.shape[0] // 2
    assert int(grid[0, 0]) == YELLOW and int(grid[0, -1]) == YELLOW and int(grid[-1, 0]) == YELLOW and int(grid[-1, -1]) == YELLOW
    center_r, center_c = _detect_center_lines(grid)
    assert center_r == center and center_c == center

    visible = grid == BLUE
    assert not visible[center, :].any() and not visible[:, center].any(), "visible sprite pixels cannot sit on the cut lines"

    latent = _latent_sprite_mask(visible, center, half_h, half_w)
    assert _is_connected(latent), "latent sprite must be 4-connected"
    assert int(latent.sum()) >= 12 and int(latent.sum()) <= 20, "latent sprite pixel count must stay in the target band"

    box = _bbox(latent)
    assert box == (center - half_h, center - half_w, center + half_h, center + half_w), "sprite bbox must stay centered"
    centered_patch = latent[box[0] : box[2] + 1, box[1] : box[3] + 1]
    assert not _is_degenerate(centered_patch), "sprite must be asymmetric"

    quadrants = {
        "tl": visible[:center, :center],
        "tr": visible[:center, center + 1 :],
        "bl": visible[center + 1 :, :center],
        "br": visible[center + 1 :, center + 1 :],
    }
    for quadrant, submask in quadrants.items():
        if quadrant in active:
            assert submask.any(), f"quadrant {quadrant} must contain a fragment"
            assert _is_connected(submask), f"quadrant {quadrant} fragment must be connected"
            assert _touches_inner_edge(submask, quadrant), f"quadrant {quadrant} must touch the cut-adjacent edge"
        else:
            assert not submask.any(), f"quadrant {quadrant} should be empty in this branch"


def _validate_output_semantics(out: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(out, expected)
    colors = set(map(int, np.unique(out)))
    assert colors.issubset({BG, BLUE}), "output should contain only background and blue fragments"


def _generation_attempt() -> np.ndarray | None:
    size = _choose_size()
    half_h, half_w = _choose_box()
    active = _choose_active_quadrants()

    hidden_cross = 2 * half_h + 2 * half_w + 1
    min_visible = max(len(active) + 1, 4, 12 - hidden_cross)
    max_visible = min(len(active) * half_h * half_w, 20 - hidden_cross)
    if min_visible > max_visible:
        return None

    counts = _partition_total(random.randint(min_visible, max_visible), active, half_h, half_w)
    local_masks: dict[str, np.ndarray] = {}
    for quadrant in active:
        choices = _LOCAL_MASK_CACHE[(half_h, half_w, quadrant)].get(counts[quadrant], [])
        if not choices:
            return None
        local_masks[quadrant] = random.choice(choices).copy()

    visible = _assemble_visible_mask(size, half_h, half_w, local_masks)
    latent = _latent_sprite_mask(visible, size // 2, half_h, half_w)
    box = _bbox(latent)
    assert box is not None
    centered_patch = latent[box[0] : box[2] + 1, box[1] : box[3] + 1]
    if _is_degenerate(centered_patch):
        return None

    grid = _render_input(size, visible)
    _validate_input_semantics(grid, active, half_h, half_w)

    expected = _expected_output(size, local_masks)
    actual = main(grid)
    _validate_output_semantics(actual, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
