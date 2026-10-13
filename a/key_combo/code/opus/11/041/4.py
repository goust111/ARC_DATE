from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLOCK_SIZE = 5
TASK_COLORS = (1, 2, 3)
GEN_RETRIES = 600


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[r, c] = True
    return out


def centered_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    off_r = (BLOCK_SIZE - h) // 2
    off_c = (BLOCK_SIZE - w) // 2
    out[off_r : off_r + h, off_c : off_c + w] = mask
    return out


def directional_diff(mask: np.ndarray) -> np.ndarray:
    mirrored = np.fliplr(mask)
    return mask & ~mirrored


def _analyze_grid(grid: np.ndarray) -> list[dict[str, object]]:
    bg = get_bg(grid)
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != bg)
    assert len(colors) == 3, "expected exactly three non-background colors"

    entries: list[dict[str, object]] = []
    h, _w = grid.shape
    for color in colors:
        comps = component_coords(grid == color)
        assert len(comps) == 2, f"color {color} should have exactly two components"
        block = None
        sprite = None
        for coords in comps:
            bbox = bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            area = len(coords)
            patch = crop_bbox(grid, bbox)
            if hh == BLOCK_SIZE and ww == BLOCK_SIZE and area == BLOCK_SIZE * BLOCK_SIZE and np.all(patch == color):
                block = {"bbox": bbox, "coords": coords}
            else:
                sprite = {
                    "bbox": bbox,
                    "coords": coords,
                    "area": area,
                    "mask": patch == color,
                }
        assert block is not None and sprite is not None, f"failed to classify color {color}"
        sr0, _sc0, sr1, _sc1 = sprite["bbox"]  # type: ignore[index]
        br0, _bc0, _br1, _bc1 = block["bbox"]  # type: ignore[index]
        assert sr1 < h // 2, "sprite should be in upper half"
        assert br0 >= h // 2, "block should be in lower half"
        entries.append({"color": color, "block": block, "sprite": sprite})
    return entries


def _render_with_mode(grid: np.ndarray, entries: list[dict[str, object]], mode: str) -> np.ndarray:
    bg = get_bg(grid)
    out = np.array(grid, dtype=int, copy=True)
    for entry in entries:
        block = entry["block"]  # type: ignore[assignment]
        sprite = entry["sprite"]  # type: ignore[assignment]
        color = int(entry["color"])  # type: ignore[arg-type]
        block_bbox = block["bbox"]
        sprite_bbox = sprite["bbox"]
        sprite_mask = np.array(sprite["mask"], dtype=bool, copy=True)

        if mode == "correct":
            hole_mask = directional_diff(sprite_mask)
        elif mode == "original":
            hole_mask = sprite_mask
        elif mode == "mirrored":
            hole_mask = np.fliplr(sprite_mask)
        elif mode == "symmetric":
            hole_mask = sprite_mask ^ np.fliplr(sprite_mask)
        elif mode == "none":
            hole_mask = np.zeros_like(sprite_mask, dtype=bool)
        else:
            raise ValueError(f"unknown mode: {mode}")

        sr0, sc0, sr1, sc1 = sprite_bbox
        sprite_patch = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
        sprite_patch[sprite_mask] = bg

        br0, bc0, br1, bc1 = block_bbox
        block_patch = np.full((BLOCK_SIZE, BLOCK_SIZE), color, dtype=int)
        centered = centered_mask(hole_mask)
        block_patch[centered] = bg
        out[br0 : br1 + 1, bc0 : bc1 + 1] = block_patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract left-right self-mirror differences from each sprite and carve them into the same-colored solid block.

    Input structure:
    - Black background with exactly three non-background colors, each appearing in two disconnected components.
    - In the upper half, one component per color is a 4-connected irregular sprite with area 6-10 and bbox at most 5x5.
    - In the lower half, one component per color is a solid 5x5 monochrome square of the same color.

    Rule:
    - For each upper sprite, crop its tight bbox and horizontally mirror the sprite inside that bbox.
    - Mark only the bbox cells where the original sprite is colored but the mirrored version is empty.
    - Center that difference mask inside the matching 5x5 block and turn those block cells to black.
    - Erase all sprite pixels to black and leave everything else unchanged.

    Key insight:
    - The hole stencil is not the sprite itself but the one-sided mismatch between the sprite and its own horizontal reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    entries = _analyze_grid(g)
    for entry in entries:
        sprite_mask = np.array(entry["sprite"]["mask"], dtype=bool)  # type: ignore[index]
        diff = directional_diff(sprite_mask)
        assert diff.any(), "sprites must be horizontally asymmetric"
    return _render_with_mode(g, entries, "correct")


def _l_path(a: tuple[int, int], b: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    cells = [(r0, c0)]
    horiz_first = rng.choice([False, True])
    if horiz_first:
        step = 1 if c1 >= c0 else -1
        for cc in range(c0 + step, c1 + step, step):
            cells.append((r0, cc))
        step = 1 if r1 >= r0 else -1
        for rr in range(r0 + step, r1 + step, step):
            cells.append((rr, c1))
    else:
        step = 1 if r1 >= r0 else -1
        for rr in range(r0 + step, r1 + step, step):
            cells.append((rr, c0))
        step = 1 if c1 >= c0 else -1
        for cc in range(c0 + step, c1 + step, step):
            cells.append((r1, cc))
    return cells


def _choose_dims(profile: str, rng: random.Random) -> tuple[int, int]:
    dims: list[tuple[int, int]] = []
    for h in range(2, 6):
        for w in range(2, 6):
            if h * w < 7:
                continue
            if profile == "odd_center" and w not in (3, 5):
                continue
            if profile == "even_width" and w not in (2, 4):
                continue
            if profile == "max_side" and h != 5 and w != 5:
                continue
            dims.append((h, w))
    return rng.choice(dims)


def _random_sprite_mask(profile: str, rng: random.Random) -> np.ndarray:
    require_center = profile == "odd_center"
    for _ in range(400):
        h, w = _choose_dims(profile, rng)
        edge_points = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        order = list(dict.fromkeys(edge_points))
        rng.shuffle(order)
        cells = {order[0]}
        current = order[0]
        for point in order[1:]:
            cells.update(_l_path(current, point, rng))
            current = point
        if len(cells) > 10 or len(cells) >= h * w:
            continue

        target_area = rng.randint(max(6, len(cells)), min(10, h * w - 1))
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        comps = component_coords(mask)
        if len(comps) != 1:
            continue
        rr0, cc0, rr1, cc1 = bbox_from_coords(comps[0])
        if (rr1 - rr0 + 1, cc1 - cc0 + 1) != (h, w):
            continue
        if require_center and not mask[:, w // 2].any():
            continue

        diff = directional_diff(mask)
        if diff.sum() < 2:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError(f"failed to sample sprite for profile {profile}")


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _sample_block_positions(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    row_lo = h // 2
    row_hi = h - BLOCK_SIZE - 1
    col_lo = 1
    col_hi = w - BLOCK_SIZE - 1
    for _ in range(500):
        rects: list[tuple[int, int, int, int]] = []
        poses: list[tuple[int, int]] = []
        for _inner in range(300):
            if len(poses) == 3:
                break
            r0 = rng.randint(row_lo, row_hi)
            c0 = rng.randint(col_lo, col_hi)
            rect = (r0, c0, r0 + BLOCK_SIZE - 1, c0 + BLOCK_SIZE - 1)
            if all(_rects_separated(rect, prev, gap=1) for prev in rects):
                rects.append(rect)
                poses.append((r0, c0))
        if len(poses) == 3:
            return poses
    raise RuntimeError("failed to place blocks")


def _sample_sprite_positions(
    h: int, w: int, shapes: list[np.ndarray], rng: random.Random
) -> list[tuple[int, int]]:
    top_limit = h // 2 - 1
    for _ in range(500):
        rects: list[tuple[int, int, int, int]] = []
        poses: list[tuple[int, int]] = []
        ok = True
        for mask in shapes:
            sh, sw = mask.shape
            row_lo = 1
            row_hi = top_limit - sh + 1
            col_lo = 1
            col_hi = w - sw - 1
            if row_lo > row_hi or col_lo > col_hi:
                ok = False
                break
            placed = False
            for _inner in range(120):
                r0 = rng.randint(row_lo, row_hi)
                c0 = rng.randint(col_lo, col_hi)
                rect = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if all(_rects_separated(rect, prev, gap=1) for prev in rects):
                    rects.append(rect)
                    poses.append((r0, c0))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and len(poses) == 3:
            return poses
    raise RuntimeError("failed to place sprites")


def _build_input(
    h: int,
    w: int,
    sprite_masks: list[np.ndarray],
    sprite_positions: list[tuple[int, int]],
    block_positions: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for color, (r0, c0) in zip(colors, block_positions):
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = color
    for color, mask, (r0, c0) in zip(colors, sprite_masks, sprite_positions):
        sh, sw = mask.shape
        patch = np.zeros((sh, sw), dtype=int)
        patch[mask] = color
        grid[r0 : r0 + sh, c0 : c0 + sw] = np.maximum(grid[r0 : r0 + sh, c0 : c0 + sw], patch)
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    entries = _analyze_grid(grid)
    correct = _render_with_mode(grid, entries, "correct")
    assert np.array_equal(main(grid), correct)

    centered_diffs = []
    for entry in entries:
        sprite = entry["sprite"]  # type: ignore[assignment]
        mask = np.array(sprite["mask"], dtype=bool, copy=True)
        diff = directional_diff(mask)
        assert 2 <= int(diff.sum()) <= 6
        assert not np.array_equal(mask, diff)
        assert not np.array_equal(np.fliplr(mask), diff)
        centered_diffs.append(centered_mask(diff))

    for i in range(len(centered_diffs)):
        for j in range(i + 1, len(centered_diffs)):
            assert not np.array_equal(centered_diffs[i], centered_diffs[j]), "hole patterns should differ"

    alt_modes = ("none", "original", "mirrored", "symmetric")
    for mode in alt_modes:
        alt = _render_with_mode(grid, entries, mode)
        assert not np.array_equal(correct, alt), f"degenerate instance: mode {mode} matches correct output"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(19, 23)
    w = rng.randint(19, 23)

    profiles = ["odd_center", "even_width", "max_side"]
    rng.shuffle(profiles)
    sprite_masks = [_random_sprite_mask(profile, rng) for profile in profiles]
    diff_areas = [int(directional_diff(mask).sum()) for mask in sprite_masks]
    if len(set(diff_areas)) < 2:
        return None

    try:
        sprite_positions = _sample_sprite_positions(h, w, list(sprite_masks), rng)
        block_positions = _sample_block_positions(h, w, rng)
    except RuntimeError:
        return None

    colors = list(TASK_COLORS)
    rng.shuffle(colors)
    grid = _build_input(h, w, sprite_masks, sprite_positions, block_positions, colors)

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
