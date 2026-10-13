import random
from collections import Counter

import numpy as np
from scipy.ndimage import label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
BG = 0
MAX_GEN_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    lab, n = label(mask.astype(bool), structure=STRUCT4)
    return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]


def _coords_touch_bbox(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotation_matches(src: np.ndarray, dst: np.ndarray) -> list[int]:
    matches = []
    for k in range(4):
        rot = np.rot90(src, k)
        if rot.shape == dst.shape and np.array_equal(rot, dst):
            matches.append(k)
    return matches


def _rotation_signature(mask: np.ndarray) -> tuple:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((rot.shape, tuple(map(int, rot.ravel()))))
    return min(variants)


def _is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask)
    return len(comps) == 1 and int(comps[0].shape[0]) == int(mask.sum())


def _has_internal_hole(mask: np.ndarray) -> bool:
    for coords in components_mask(~mask):
        if not _coords_touch_bbox(coords, mask.shape):
            return True
    return False


def _is_non_symmetric(mask: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rot = np.rot90(mask, k)
        if rot.shape == mask.shape and np.array_equal(rot, mask):
            return False
    return True


def _is_chiral(mask: np.ndarray) -> bool:
    mirrored = np.fliplr(mask)
    return not _rotation_matches(mask, mirrored)


def _find_frames_and_holes(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    holes: list[dict] = []
    frame_mask = np.zeros_like(grid, dtype=bool)
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        lab, n = label(grid == color, structure=STRUCT4)
        for k in range(1, int(n) + 1):
            comp = lab == k
            box = bbox(comp)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            crop = comp[r0 : r1 + 1, c0 : c1 + 1]
            if crop.shape[0] < 3 or crop.shape[1] < 3:
                continue
            if not (crop[0, :].all() and crop[-1, :].all() and crop[:, 0].all() and crop[:, -1].all()):
                continue
            enclosed = []
            for coords in components_mask(~crop):
                if not _coords_touch_bbox(coords, crop.shape):
                    enclosed.append(coords)
            if len(enclosed) != 1:
                continue
            hole_mask = np.zeros(crop.shape, dtype=bool)
            hole_mask[tuple(enclosed[0].T)] = True
            hole_box_local = bbox(hole_mask)
            assert hole_box_local is not None
            hr0, hc0, hr1, hc1 = hole_box_local
            hole_crop = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1].copy()
            holes.append(
                {
                    "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                    "mask": hole_crop,
                }
            )
            frame_mask |= comp
    assert holes, "expected at least one framed hole"
    return holes, frame_mask


def _find_sprites(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> list[dict]:
    fg = (grid != bg) & ~frame_mask
    lab, n = label(fg, structure=STRUCT4)
    sprites: list[dict] = []
    for k in range(1, int(n) + 1):
        comp = lab == k
        box = bbox(comp)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        mask = comp[r0 : r1 + 1, c0 : c1 + 1].copy()
        crop = np.full(mask.shape, bg, dtype=int)
        crop[mask] = grid[r0 : r1 + 1, c0 : c1 + 1][mask]
        sprites.append(
            {
                "bbox": box,
                "mask": mask,
                "crop": crop,
                "coords": np.argwhere(comp),
            }
        )
    assert sprites, "expected at least one free sprite"
    return sprites


def _solve_assignment(sprites: list[dict], holes: list[dict]) -> list[tuple[int, int]]:
    assert len(sprites) == len(holes), "sprite/hole count mismatch"
    options: dict[int, list[tuple[int, int]]] = {}
    for i, sprite in enumerate(sprites):
        cur = []
        for j, hole in enumerate(holes):
            for k in _rotation_matches(sprite["mask"], hole["mask"]):
                cur.append((j, k))
        assert cur, "every sprite must match at least one hole by rotation"
        options[i] = cur

    order = sorted(options, key=lambda idx: len(options[idx]))
    used_holes: set[int] = set()
    assignment: list[tuple[int, int] | None] = [None] * len(sprites)
    solutions: list[list[tuple[int, int]]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append([item for item in assignment if item is not None])
            return
        sprite_idx = order[pos]
        for hole_idx, rot_k in options[sprite_idx]:
            if hole_idx in used_holes:
                continue
            used_holes.add(hole_idx)
            assignment[sprite_idx] = (hole_idx, rot_k)
            dfs(pos + 1)
            assignment[sprite_idx] = None
            used_holes.remove(hole_idx)

    dfs(0)
    assert len(solutions) == 1, "rotation-only pairing must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each free multicolor sprite into its uniquely matching framed hole, fill the hole, and erase the source sprite.

    Input structure:
    - A black background contains several single-color rectangular lock bodies whose outer bbox boundary is solid and whose interior encloses one irregular background hole.
    - Outside those lock bodies are the same number of disconnected multicolor sprites on black background.
    - Each sprite shape matches exactly one hole shape under rotation only; mirror matches are traps and do not count.

    Rule:
    - Detect every enclosed background hole inside a monochrome rectangular lock body.
    - Detect every free multicolor sprite outside the locks.
    - Pair sprites to holes using only 0/90/180/270-degree rotations, then rotate each sprite's full color pattern by the same angle needed to align the shapes.
    - Fill the matched hole with that rotated pattern and erase the original free sprite cells to background, while leaving the lock bodies unchanged.

    Key insight:
    - The relevant match is between a positive sprite shape and a negative enclosed hole shape, and only rotational congruence is legal even when a distracting mirror-shaped hole exists.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    holes, frame_mask = _find_frames_and_holes(g, bg)
    sprites = _find_sprites(g, bg, frame_mask)
    assignment = _solve_assignment(sprites, holes)

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for sprite, (hole_idx, rot_k) in zip(sprites, assignment):
        hole = holes[hole_idx]
        rotated_mask = np.rot90(sprite["mask"], rot_k)
        rotated_crop = np.rot90(sprite["crop"], rot_k)
        assert rotated_mask.shape == hole["mask"].shape
        assert np.array_equal(rotated_mask, hole["mask"])
        r0, c0, r1, c1 = hole["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[rotated_mask] = rotated_crop[rotated_mask]

    return out


def _grow_connected_shape(height: int, width: int, area: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    box = bbox(mask)
    assert box is not None
    return _crop(mask, box)


def _random_shape(*, require_chiral: bool) -> np.ndarray:
    for _ in range(200):
        height = random.randint(3, 5)
        width = random.randint(3, 6)
        max_area = height * width - 1
        min_area = max(5, (height * width + 1) // 2)
        area = random.randint(min_area, max_area)
        mask = _grow_connected_shape(height, width, area)
        if mask is None:
            continue
        if min(mask.shape) < 3 or max(mask.shape) > 6:
            continue
        if not _is_connected(mask):
            continue
        if _has_internal_hole(mask):
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if not _is_non_symmetric(mask):
            continue
        if require_chiral and not _is_chiral(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a valid hole shape")


def _random_colored_patch(mask: np.ndarray, forbidden_color: int) -> np.ndarray:
    cells = [tuple(cell) for cell in np.argwhere(mask)]
    palette = [color for color in range(1, 10) if color != forbidden_color]
    random.shuffle(palette)
    color_count = min(len(cells), random.randint(2, min(4, len(cells))))
    colors = palette[:color_count]
    patch = np.full(mask.shape, BG, dtype=int)
    random.shuffle(cells)
    for (r, c), color in zip(cells, colors):
        patch[r, c] = color
    for r, c in cells[color_count:]:
        patch[r, c] = random.choice(colors)
    return patch


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, *, gap: int) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = grid.shape
    candidates = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            rr0 = max(0, r0 - gap)
            cc0 = max(0, c0 - gap)
            rr1 = min(H, r0 + h + gap)
            cc1 = min(W, c0 + w + gap)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + h, c0 : c0 + w] = patch
    occupied[r0 : r0 + h, c0 : c0 + w] = True
    return r0, c0


def _has_exact_mirror_trap(sprite_shapes: list[np.ndarray], hole_shapes: list[np.ndarray]) -> bool:
    for i, sprite_shape in enumerate(sprite_shapes):
        mirrored = np.fliplr(sprite_shape)
        for j, hole_shape in enumerate(hole_shapes):
            if i != j and mirrored.shape == hole_shape.shape and np.array_equal(mirrored, hole_shape):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(2, 4)

    base = _random_shape(require_chiral=True)
    hole_shapes = [np.rot90(base, random.randrange(4)).copy()]
    hole_shapes.append(np.fliplr(hole_shapes[0]).copy())

    seen = {_rotation_signature(hole_shapes[0]), _rotation_signature(hole_shapes[1])}
    while len(hole_shapes) < object_count:
        extra = np.rot90(_random_shape(require_chiral=False), random.randrange(4)).copy()
        sig = _rotation_signature(extra)
        if sig in seen:
            continue
        seen.add(sig)
        hole_shapes.append(extra)

    frame_colors = random.sample(range(1, 10), object_count)

    sprite_rotations = [0]
    sprite_rotations.extend(random.randrange(4) for _ in range(object_count - 1))
    if object_count > 1 and all(rot == 0 for rot in sprite_rotations):
        sprite_rotations[random.randrange(1, object_count)] = random.choice((1, 2, 3))

    frame_patches = []
    sprite_patches = []
    sprite_shapes = []
    for i, hole_shape in enumerate(hole_shapes):
        hh, hw = hole_shape.shape
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        frame = np.full((top + hh + bottom, left + hw + right), frame_colors[i], dtype=int)
        frame[top : top + hh, left : left + hw][hole_shape] = BG
        frame_patches.append(frame)

        aligned = _random_colored_patch(hole_shape, frame_colors[i])
        shown = np.rot90(aligned, sprite_rotations[i]).copy()
        sprite_patches.append(shown)
        sprite_shapes.append(np.rot90(hole_shape, sprite_rotations[i]).copy())

    total_h = sum(p.shape[0] for p in frame_patches + sprite_patches)
    total_w = sum(p.shape[1] for p in frame_patches + sprite_patches)
    H = random.randint(max(18, min(24, total_h // 2 + 8)), 30)
    W = random.randint(max(18, min(24, total_w // 2 + 8)), 30)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    assets = [("frame", patch) for patch in frame_patches] + [("sprite", patch) for patch in sprite_patches]
    random.shuffle(assets)
    for _kind, patch in sorted(assets, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True):
        pos = _place_patch(grid, occupied, patch, gap=1)
        if pos is None:
            return None

    if not _has_exact_mirror_trap(sprite_shapes, hole_shapes):
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key input")
