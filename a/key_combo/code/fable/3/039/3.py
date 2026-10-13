from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TERRAIN = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _legend_side_and_order(grid: np.ndarray) -> tuple[str, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, str, list[int]]] = []
    for side, cells in (
        ("left", [int(grid[r, 0]) for r in range(h)]),
        ("right", [int(grid[r, w - 1]) for r in range(h)]),
    ):
        colors = [value for value in cells if value not in (BG, TERRAIN)]
        if not colors:
            continue
        if len(colors) != len(set(colors)):
            continue
        candidates.append((len(colors), side, colors))
    assert len(candidates) == 1, "expected exactly one edge legend"
    _count, side, order = candidates[0]
    assert len(order) >= 2, "expected a multi-color legend"
    return side, order


def _extract_sprites(grid: np.ndarray, side: str, order: list[int]) -> dict[int, np.ndarray]:
    h, w = grid.shape
    sprites: dict[int, np.ndarray] = {}
    for color in order:
        comps = _components(grid == color)
        assert len(comps) == 2, "each color should appear as one legend cell and one sprite"
        legend_idx = None
        for idx, comp in enumerate(comps):
            rows = comp[:, 0]
            cols = comp[:, 1]
            touches = False
            if side == "left":
                touches = bool(np.any(cols == 0))
            else:
                touches = bool(np.any(cols == w - 1))
            if touches:
                assert comp.shape[0] == 1, "legend entries must be single pixels"
                legend_idx = idx
                break
        assert legend_idx is not None, "could not locate legend component"
        sprite = comps[1 - legend_idx]
        assert not np.any(sprite[:, 0] == 0) and not np.any(sprite[:, 0] == h - 1)
        assert not np.any(sprite[:, 1] == 0) and not np.any(sprite[:, 1] == w - 1)
        sprites[int(color)] = sprite
    return sprites


def _drop_component(coords: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    h, _w = occupied.shape
    shift = 0
    while True:
        moved = coords + np.array([shift + 1, 0], dtype=int)
        if int(moved[:, 0].max()) >= h:
            break
        if bool(occupied[moved[:, 0], moved[:, 1]].any()):
            break
        shift += 1
    return coords + np.array([shift, 0], dtype=int)


def _simulate(
    grid: np.ndarray,
    side: str,
    order: list[int],
    sprites: dict[int, np.ndarray],
    *,
    landed_block: bool,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    out = np.array(grid, dtype=int, copy=True)
    for color in order:
        coords = sprites[color]
        out[coords[:, 0], coords[:, 1]] = BG

    fixed = (out != BG)
    if not landed_block:
        fixed = fixed.copy()

    landed: dict[int, np.ndarray] = {}
    for color in order:
        occupied = fixed.copy() if landed_block else fixed
        final_coords = _drop_component(sprites[color], occupied)
        landed[color] = final_coords
        out[final_coords[:, 0], final_coords[:, 1]] = color
        if landed_block:
            fixed[final_coords[:, 0], final_coords[:, 1]] = True
    return out, landed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop floating sprites one by one according to the edge color legend, with earlier landings blocking later ones.

    Input structure:
    - The grid has a black background, an irregular gray terrain anchored at the bottom, and several floating monochrome sprites.
    - One vertical grid edge contains a single-pixel color legend; each legend color matches exactly one floating sprite color.
    - Sprites are pairwise separated, and the legend pixels are detached from the interior sprites.

    Rule:
    - Read the edge legend from top to bottom to obtain the falling order.
    - Remove all floating sprites from the scene, then drop the matching sprite for each legend color straight downward without rotation.
    - A sprite stops at the last position before it would hit the gray terrain, the grid bottom, or any sprite that has already landed; terrain and legend stay unchanged.

    Key insight:
    - The colored edge strip is not decoration but the time axis of a sequential gravity simulation, so previously landed sprites become new obstacles for later ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == TERRAIN), "expected gray terrain"

    side, order = _legend_side_and_order(g)
    sprites = _extract_sprites(g, side, order)
    out, _landed = _simulate(g, side, order, sprites, landed_block=True)
    return out


def _random_walk_profile(width: int, max_height: int) -> list[int]:
    heights = [random.randint(1, max_height)]
    for _ in range(width - 1):
        prev = heights[-1]
        step = random.choice([-2, -1, 0, 1, 2])
        heights.append(min(max(prev + step, 1), max_height))
    if len(set(heights)) == 1:
        idx = random.randrange(width)
        heights[idx] = min(max_height, heights[idx] + 1)
    return heights


def _make_polyomino(target_area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(sorted(frontier)))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        patch[r, c] = True
    return patch


def _bbox_from_patch(patch: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = patch.shape
    return top, left, top + h - 1, left + w - 1


def _boxes_are_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(patch)
    grid[top + coords[:, 0], left + coords[:, 1]] = int(color)
    return top + coords[:, 0], left + coords[:, 1]


def _terrain_clearance(terrain_top: np.ndarray, left: int, width: int) -> int:
    cols = range(left, left + width)
    return int(min(terrain_top[c] for c in cols))


def _build_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(12, 20)
    side = random.choice(["left", "right"])
    legend_c = 0 if side == "left" else w - 1
    active_cols = [c for c in range(w) if c != legend_c]

    terrain_max = max(3, min(7, h // 3))
    profile = _random_walk_profile(len(active_cols), terrain_max)
    terrain_top = np.full(w, h, dtype=int)

    grid = np.full((h, w), BG, dtype=int)
    for idx, c in enumerate(active_cols):
        height = profile[idx]
        top = h - height
        grid[top:, c] = TERRAIN
        terrain_top[c] = top

    n_sprites = random.randint(3, 5)
    colors = random.sample(SPRITE_COLORS, n_sprites)
    order = random.sample(colors, k=n_sprites)

    latest_legend_start = min(h - n_sprites - 2, max(1, h - terrain_max - n_sprites - 2))
    if latest_legend_start < 1:
        return None
    legend_r0 = random.randint(1, latest_legend_start)
    for offset, color in enumerate(order):
        grid[legend_r0 + offset, legend_c] = color

    center = random.randint(3, w - 4) if w >= 8 else w // 2
    if legend_c == 0:
        center = max(center, 3)
    else:
        center = min(center, w - 4)

    sprite_specs: list[tuple[int, np.ndarray]] = []
    for color in colors:
        area = random.randint(3, 6)
        patch = _make_polyomino(area)
        sprite_specs.append((color, patch))

    placed_boxes: list[tuple[int, int, int, int]] = []
    overlap_seed = 0
    for idx, (color, patch) in enumerate(sprite_specs):
        ph, pw = patch.shape
        min_left = 1
        max_left = w - pw - 2
        if side == "left":
            min_left = 2
        else:
            max_left = w - pw - 3
        if min_left > max_left:
            return None

        if idx < 2:
            left_low = max(min_left, center - pw + overlap_seed - 1)
            left_high = min(max_left, center + overlap_seed)
        else:
            left_low = max(min_left, center - 3)
            left_high = min(max_left, center + 2)
        if left_low > left_high:
            left_low, left_high = min_left, max_left
        left = random.randint(left_low, left_high)

        clearance = _terrain_clearance(terrain_top, left, pw)
        max_top = clearance - ph - 2
        if max_top < 1:
            return None
        top_low = 1
        top_high = min(max_top, h // 2 + idx)
        if top_high < top_low:
            return None

        placed = False
        for _ in range(60):
            top = random.randint(top_low, top_high)
            box = _bbox_from_patch(patch, top, left)
            if any(not _boxes_are_separate(box, other, margin=1) for other in placed_boxes):
                continue
            _paint_patch(grid, patch, top, left, color)
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
        overlap_seed += 1

    if len(set(profile)) == 1:
        return None

    out = main(grid)
    seq_side, seq_order = _legend_side_and_order(grid)
    sprites = _extract_sprites(grid, seq_side, seq_order)
    terrain_only, landed_only = _simulate(grid, seq_side, seq_order, sprites, landed_block=False)
    terrain_and_stack, landed_stack = _simulate(grid, seq_side, seq_order, sprites, landed_block=True)

    if np.array_equal(terrain_only, terrain_and_stack):
        return None
    if np.array_equal(grid, out):
        return None
    if not np.array_equal(out, terrain_and_stack):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    changed = 0
    for color in seq_order:
        if not np.array_equal(landed_only[color], landed_stack[color]):
            changed += 1
    if changed == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _build_attempt()
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sequential-drop input")
