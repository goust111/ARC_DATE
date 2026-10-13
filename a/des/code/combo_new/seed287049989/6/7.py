from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = tuple(color for color in range(1, 10) if color != FRAME)
QUADRANTS = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 500
PATCH_ATTEMPTS = 250
TILE_SIZE_CHOICES = (2, 3, 4, 5)


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).reshape(-1)
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    height, width = shape
    return any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in coords)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _rotate_for_quadrant(patch: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tl":
        return patch.copy()
    if quadrant == "tr":
        return np.rot90(patch, -1).copy()
    if quadrant == "br":
        return np.rot90(patch, 2).copy()
    if quadrant == "bl":
        return np.rot90(patch, 1).copy()
    raise ValueError(quadrant)


def _inverse_rotate_for_quadrant(patch: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tl":
        return patch.copy()
    if quadrant == "tr":
        return np.rot90(patch, 1).copy()
    if quadrant == "br":
        return np.rot90(patch, 2).copy()
    if quadrant == "bl":
        return np.rot90(patch, -1).copy()
    raise ValueError(quadrant)


def _quadrant_of_bbox(box: tuple[int, int, int, int], center_r: float, center_c: float) -> str:
    r0, c0, r1, c1 = box
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0
    if mid_r < center_r and mid_c < center_c:
        return "tl"
    if mid_r < center_r and mid_c > center_c:
        return "tr"
    if mid_r > center_r and mid_c < center_c:
        return "bl"
    if mid_r > center_r and mid_c > center_c:
        return "br"
    raise AssertionError("component center must lie in a unique quadrant")


def _frame_info(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    bg = _most_common_color(grid)
    enclosed = [comp for comp in _components(grid == bg) if not _touches_edge(comp, grid.shape)]
    assert len(enclosed) == 1, "expected one enclosed black frame interior"

    ir0, ic0, ir1, ic1 = _bbox(enclosed[0])
    assert np.all(grid[ir0 : ir1 + 1, ic0 : ic1 + 1] == bg), "frame interior must be a solid rectangle"
    assert (ir1 - ir0 + 1) % 2 == 0 and (ic1 - ic0 + 1) % 2 == 0, "frame interior must split evenly"
    assert ir0 > 0 and ic0 > 0 and ir1 + 1 < grid.shape[0] and ic1 + 1 < grid.shape[1]

    fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    ring_colors = set(int(v) for v in grid[fr0, fc0 : fc1 + 1])
    ring_colors.update(int(v) for v in grid[fr1, fc0 : fc1 + 1])
    ring_colors.update(int(v) for v in grid[fr0 : fr1 + 1, fc0])
    ring_colors.update(int(v) for v in grid[fr0 : fr1 + 1, fc1])
    ring_colors.discard(bg)
    assert len(ring_colors) == 1, "frame border must have one color"
    frame_color = ring_colors.pop()
    return bg, frame_color, ir0, ic0, ir1, ic1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the four quadrant pattern blocks by quadrant-specific angles and assemble them inside the central frame.

    Input structure:
    - The background is black.
    - A grey rectangular frame encloses a black interior near the center, and grey horizontal and vertical guide lines divide the outer area into four quadrants.
    - Each outer quadrant contains exactly one isolated colored pattern block, with its tight bounding box matching one quarter-tile of the frame interior after the required rotation.

    Rule:
    - Find the unique enclosed black rectangle to recover the frame interior, then split that interior into equal top-left, top-right, bottom-left, and bottom-right quarter-tiles.
    - Remove the four outer pattern blocks from their original positions.
    - Place the top-left block unchanged, rotate the top-right block 90 degrees clockwise, rotate the bottom-right block 180 degrees, and rotate the bottom-left block 270 degrees clockwise before placing them into the matching frame quadrants.
    - Keep the grey frame border and grey guide lines unchanged.

    Key insight:
    - The source quadrant itself encodes the rotation amount, so the solver must infer a position-to-rotation mapping rather than merely translating the four blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg, frame_color, ir0, ic0, ir1, ic1 = _frame_info(g)
    tile_h = (ir1 - ir0 + 1) // 2
    tile_w = (ic1 - ic0 + 1) // 2
    center_r = (ir0 + ir1) / 2.0
    center_c = (ic0 + ic1) / 2.0

    pattern_mask = (g != bg) & (g != frame_color)
    comps = _components(pattern_mask)
    assert len(comps) == 4, "expected exactly four source pattern blocks"

    patches: dict[str, np.ndarray] = {}
    for comp in comps:
        box = _bbox(comp)
        quadrant = _quadrant_of_bbox(box, center_r, center_c)
        assert quadrant not in patches, "expected one source block per quadrant"
        patch = _crop_component(g, comp, bg)
        rotated = _rotate_for_quadrant(patch, quadrant)
        assert rotated.shape == (tile_h, tile_w), "rotated pattern must fit its destination tile"
        patches[quadrant] = rotated

    assert set(patches) == set(QUADRANTS)

    out = g.copy()
    out[pattern_mask] = bg
    placements = {
        "tl": (ir0, ic0),
        "tr": (ir0, ic0 + tile_w),
        "bl": (ir0 + tile_h, ic0),
        "br": (ir0 + tile_h, ic0 + tile_w),
    }
    for quadrant, (r0, c0) in placements.items():
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = patches[quadrant]
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_has_hole(mask: np.ndarray) -> bool:
    return any(not _touches_edge(comp, mask.shape) for comp in _components(~mask))


def _random_connected_mask(height: int, width: int, forbidden_turns: tuple[int, ...]) -> np.ndarray | None:
    total = height * width
    min_cells = max(3, (total + 1) // 2)
    max_cells = total - 1
    if min_cells > max_cells:
        min_cells = max_cells
    for _ in range(PATCH_ATTEMPTS):
        target = random.randint(min_cells, max_cells)
        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _touches_all_sides(mask):
            continue
        if _mask_has_hole(mask):
            continue
        if np.all(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, turns)) for turns in forbidden_turns):
            continue
        return mask
    return None


def _make_colored_patch(height: int, width: int, color: int, forbidden_turns: tuple[int, ...]) -> np.ndarray | None:
    mask = _random_connected_mask(height, width, forbidden_turns)
    if mask is None:
        return None
    patch = np.full((height, width), BG, dtype=int)
    patch[mask] = color
    return patch


def _rand_between(lo: int, hi: int) -> int:
    assert lo <= hi
    return random.randint(lo, hi)


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    tile_h = random.choice(TILE_SIZE_CHOICES)
    tile_w = random.choice(TILE_SIZE_CHOICES)
    colors = random.sample(PALETTE, 4)

    dest_patches: dict[str, np.ndarray] = {}
    for quadrant, color in zip(QUADRANTS, colors):
        if quadrant in {"tr", "bl"}:
            forbidden = (1,)
        elif quadrant == "br":
            forbidden = (2,)
        else:
            forbidden = ()
        patch = _make_colored_patch(tile_h, tile_w, color, forbidden)
        if patch is None:
            return None
        dest_patches[quadrant] = patch

    source_patches = {
        quadrant: _inverse_rotate_for_quadrant(dest_patches[quadrant], quadrant)
        for quadrant in QUADRANTS
    }

    outer_h = 2 * tile_h + 2
    outer_w = 2 * tile_w + 2
    top_min = max(source_patches["tl"].shape[0], source_patches["tr"].shape[0]) + 1
    bottom_min = max(source_patches["bl"].shape[0], source_patches["br"].shape[0]) + 1
    left_min = max(source_patches["tl"].shape[1], source_patches["bl"].shape[1]) + 1
    right_min = max(source_patches["tr"].shape[1], source_patches["br"].shape[1]) + 1

    max_top = min(8, 30 - outer_h - bottom_min)
    if top_min > max_top:
        return None
    top_space = _rand_between(top_min, max_top)

    max_bottom = min(8, 30 - outer_h - top_space)
    if bottom_min > max_bottom:
        return None
    bottom_space = _rand_between(bottom_min, max_bottom)

    max_left = min(8, 30 - outer_w - right_min)
    if left_min > max_left:
        return None
    left_space = _rand_between(left_min, max_left)

    max_right = min(8, 30 - outer_w - left_space)
    if right_min > max_right:
        return None
    right_space = _rand_between(right_min, max_right)

    height = top_space + outer_h + bottom_space
    width = left_space + outer_w + right_space
    if height > 30 or width > 30:
        return None

    fr0, fc0 = top_space, left_space
    fr1, fc1 = fr0 + outer_h - 1, fc0 + outer_w - 1
    ir0, ic0 = fr0 + 1, fc0 + 1

    inp = np.full((height, width), BG, dtype=int)
    inp[fr0, fc0 : fc1 + 1] = FRAME
    inp[fr1, fc0 : fc1 + 1] = FRAME
    inp[fr0 : fr1 + 1, fc0] = FRAME
    inp[fr0 : fr1 + 1, fc1] = FRAME

    line_r = fr0 + tile_h + random.randint(0, 1)
    line_c = fc0 + tile_w + random.randint(0, 1)
    inp[line_r, : fc0 + 1] = FRAME
    inp[line_r, fc1:] = FRAME
    inp[: fr0 + 1, line_c] = FRAME
    inp[fr1:, line_c] = FRAME

    corners = {
        "tl": ((0, fr0), (0, fc0)),
        "tr": ((0, fr0), (fc1 + 1, width)),
        "bl": ((fr1 + 1, height), (0, fc0)),
        "br": ((fr1 + 1, height), (fc1 + 1, width)),
    }
    for quadrant, patch in source_patches.items():
        ph, pw = patch.shape
        (rr0, rr1), (cc0, cc1) = corners[quadrant]
        if rr1 - rr0 < ph or cc1 - cc0 < pw:
            return None
        r0 = random.randint(rr0, rr1 - ph)
        c0 = random.randint(cc0, cc1 - pw)
        _place_patch(inp, patch, r0, c0)

    out = inp.copy()
    out[(out != BG) & (out != FRAME)] = BG
    placements = {
        "tl": (ir0, ic0),
        "tr": (ir0, ic0 + tile_w),
        "bl": (ir0 + tile_h, ic0),
        "br": (ir0 + tile_h, ic0 + tile_w),
    }
    for quadrant, (r0, c0) in placements.items():
        _place_patch(out, dest_patches[quadrant], r0, c0)

    if np.array_equal(main(inp), out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _build_case()
        if pair is not None:
            return pair[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
