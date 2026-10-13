import random
from collections import Counter, deque

import numpy as np


PALETTE = tuple(range(10))
NONZERO_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400
COMPONENT_RETRIES = 120
PLACEMENT_RETRIES = 160
POLYOMINO_RETRIES = 160


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each holed object onto the uniquely matching key sprite and recolor it to the key's color.

    Input structure:
    - The dominant color is the background. Several monochrome foreground objects each contain exactly one enclosed background hole.
    - Separate monochrome sprite components are scattered elsewhere; every hole shape matches exactly one sprite under a 90-degree rotation, and extra distractor sprites may also appear.
    - Hole shapes are pairwise distinct up to rotation, so the lock-to-key pairing is unique from shape alone.

    Rule:
    - Extract every holed object's enclosed background hole as a normalized boolean mask.
    - For each hole, find the unique sprite whose normalized mask equals the hole after some 0/90/180/270-degree rotation; that rotation is also applied to the whole object.
    - Translate the rotated object so that its rotated hole cells exactly cover the matched sprite cells, erase the object's original position to background, and recolor every moved object pixel to the matched sprite's color.

    Key insight:
    - The matching is between the hole's negative space and a sprite's positive shape under quarter-turn rotations; the lock moves to the key, not the other way around.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg)
    assert components, "expected at least one foreground component"

    locks = []
    sprites = []
    for comp in components:
        hole_masks = enclosed_holes(comp["mask"])
        if hole_masks:
            assert len(hole_masks) == 1, "each lock must contain exactly one enclosed hole"
            hole_full = hole_masks[0]
            hole_shape = trim_mask(hole_full)
            locks.append(
                {
                    "color": comp["color"],
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "mask": comp["mask"],
                    "hole_full": hole_full,
                    "hole_shape": hole_shape,
                }
            )
        else:
            sprites.append(
                {
                    "color": comp["color"],
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "mask": trim_mask(comp["mask"]),
                }
            )

    assert locks, "expected at least one holed object"
    assert len(sprites) >= len(locks), "expected at least as many sprites as holed objects"

    out = g.copy()
    for lock in locks:
        rr, cc = lock["coords"][:, 0], lock["coords"][:, 1]
        out[rr, cc] = bg

    used_sprites = set()
    occupied = (out != bg)
    for lock in locks:
        matches = []
        for sprite_index, sprite in enumerate(sprites):
            for rot in range(4):
                if same_mask(np.rot90(lock["hole_shape"], rot), sprite["mask"]):
                    matches.append((sprite_index, rot))
        assert len(matches) == 1, "each hole must match exactly one sprite under rotation"

        sprite_index, rot = matches[0]
        assert sprite_index not in used_sprites, "a sprite cannot unlock multiple objects"
        used_sprites.add(sprite_index)

        sprite = sprites[sprite_index]
        rotated_mask = np.rot90(lock["mask"], rot)
        rotated_hole_full = np.rot90(lock["hole_full"], rot)
        hole_bbox = mask_bbox(rotated_hole_full)
        assert hole_bbox is not None, "rotated hole must remain non-empty"

        sr0, sc0, _, _ = sprite["bbox"]
        hr0, hc0, _, _ = hole_bbox
        place_r0 = sr0 - hr0
        place_c0 = sc0 - hc0
        place_r1 = place_r0 + rotated_mask.shape[0] - 1
        place_c1 = place_c0 + rotated_mask.shape[1] - 1
        H, W = g.shape
        assert 0 <= place_r0 <= place_r1 < H and 0 <= place_c0 <= place_c1 < W, "moved object exceeds grid bounds"

        hole_cells = np.zeros_like(g, dtype=bool)
        hole_coords = np.argwhere(rotated_hole_full)
        for dr, dc in hole_coords:
            hole_cells[place_r0 + int(dr), place_c0 + int(dc)] = True

        sprite_mask_on_grid = np.zeros_like(g, dtype=bool)
        sprite_coords = sprite["coords"]
        sprite_mask_on_grid[sprite_coords[:, 0], sprite_coords[:, 1]] = True
        assert np.array_equal(hole_cells, sprite_mask_on_grid), "translated hole must cover the matched sprite exactly"

        for dr, dc in np.argwhere(rotated_mask):
            r = place_r0 + int(dr)
            c = place_c0 + int(dc)
            assert not occupied[r, c], "moved objects and untouched sprites must not overlap"
            out[r, c] = sprite["color"]
            occupied[r, c] = True

    assert np.any(out != g), "expected a visible transformation"
    return out


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = bbox_coords(coords_arr)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords_arr[:, 0] - r0, coords_arr[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return components


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def trim_mask(mask: np.ndarray) -> np.ndarray:
    box = mask_bbox(mask)
    assert box is not None, "mask must contain at least one cell"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    bg = ~mask
    seen = np.zeros((H, W), dtype=bool)
    holes = []
    for r in range(H):
        for c in range(W):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == H - 1 or cc == 0 or cc == W - 1:
                    touches_edge = True
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if touches_edge:
                continue
            hole = np.zeros_like(mask, dtype=bool)
            for rr, cc in coords:
                hole[rr, cc] = True
            holes.append(hole)
    return holes


def canonical_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(shape_code(trim_mask(np.rot90(mask, rot))) for rot in range(4))


def shape_code(mask: np.ndarray) -> tuple[str, ...]:
    return (f"{mask.shape[0]}x{mask.shape[1]}",) + tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def generate_random_polyomino(min_cells: int = 4, max_cells: int = 6) -> np.ndarray:
    for _ in range(POLYOMINO_RETRIES):
        target = random.randint(min_cells, max_cells)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))
        coords = np.array(sorted(cells), dtype=int)
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        mask = trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.all():
            continue
        if any(same_mask(mask, trim_mask(np.rot90(mask, rot))) for rot in (1, 2, 3)):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric polyomino")


def build_lock_patch(hole_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hole_h, hole_w = hole_mask.shape
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    base_h = top + hole_h + bottom
    base_w = left + hole_w + right

    canvas = np.zeros((base_h + 4, base_w + 4), dtype=bool)
    hole_canvas = np.zeros_like(canvas)
    r0 = 2
    c0 = 2
    canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True
    hole_slice = hole_canvas[r0 + top : r0 + top + hole_h, c0 + left : c0 + left + hole_w]
    hole_slice[:, :] = hole_mask
    canvas[hole_canvas] = False

    for side in random.sample(("top", "bottom", "left", "right"), k=random.randint(1, 2)):
        if side in ("top", "bottom"):
            length = random.randint(1, 2)
            span = random.randint(1, min(2, base_w))
            start = c0 + random.randint(0, base_w - span)
            if side == "top":
                canvas[r0 - length : r0, start : start + span] = True
            else:
                canvas[r0 + base_h : r0 + base_h + length, start : start + span] = True
        else:
            length = random.randint(1, 2)
            span = random.randint(1, min(2, base_h))
            start = r0 + random.randint(0, base_h - span)
            if side == "left":
                canvas[start : start + span, c0 - length : c0] = True
            else:
                canvas[start : start + span, c0 + base_w : c0 + base_w + length] = True

    box = mask_bbox(canvas)
    assert box is not None
    br0, bc0, br1, bc1 = box
    return canvas[br0 : br1 + 1, bc0 : bc1 + 1].copy(), hole_canvas[br0 : br1 + 1, bc0 : bc1 + 1].copy()


def rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def try_place_box(box_h: int, box_w: int, zone: tuple[int, int, int, int], used: list[tuple[int, int, int, int]], pad: int = 1):
    zr0, zc0, zr1, zc1 = zone
    if box_h > zr1 - zr0 + 1 or box_w > zc1 - zc0 + 1:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(zr0, zr1 - box_h + 1)
        c0 = random.randint(zc0, zc1 - box_w + 1)
        rect = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(not rects_intersect(rect, other, pad=pad) for other in used):
            used.append(rect)
            return rect
    return None


def choose_zones(H: int, W: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    mode = random.choice(("lr", "rl", "tb", "bt"))
    if mode == "lr":
        mid = W // 2
        return (1, 1, H - 2, mid - 2), (1, mid + 1, H - 2, W - 2)
    if mode == "rl":
        mid = W // 2
        return (1, mid + 1, H - 2, W - 2), (1, 1, H - 2, mid - 2)
    if mode == "tb":
        mid = H // 2
        return (1, 1, mid - 2, W - 2), (mid + 1, 1, H - 2, W - 2)
    mid = H // 2
    return (mid + 1, 1, H - 2, W - 2), (1, 1, mid - 2, W - 2)


def component_center(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(22, 30)
    W = random.randint(22, 30)
    bg = random.choice(PALETTE)
    colors = [c for c in PALETTE if c != bg]

    lock_count = random.randint(2, 3)
    distractor_count = random.randint(1, 2)
    total_needed = 2 * lock_count + distractor_count
    if total_needed > len(colors):
        return None
    random.shuffle(colors)

    hole_signatures = set()
    locks = []
    for _ in range(lock_count):
        hole_mask = None
        for _ in range(COMPONENT_RETRIES):
            candidate = generate_random_polyomino()
            sig = canonical_signature(candidate)
            if sig not in hole_signatures:
                hole_mask = candidate
                hole_signatures.add(sig)
                break
        if hole_mask is None:
            return None

        base_mask, base_hole = build_lock_patch(hole_mask)
        obj_rot = random.randint(0, 3)
        move_rot = random.randint(0, 3)
        input_mask = np.rot90(base_mask, obj_rot)
        final_mask = np.rot90(input_mask, move_rot)
        final_hole = np.rot90(np.rot90(base_hole, obj_rot), move_rot)
        sprite_shape = trim_mask(final_hole)
        locks.append(
            {
                "input_mask": input_mask,
                "final_mask": final_mask,
                "final_hole": final_hole,
                "sprite_shape": sprite_shape,
            }
        )

    distractors = []
    distractor_signatures = set()
    for _ in range(distractor_count):
        distractor_mask = None
        for _ in range(COMPONENT_RETRIES):
            candidate = generate_random_polyomino(min_cells=3, max_cells=6)
            sig = canonical_signature(candidate)
            if sig not in hole_signatures and sig not in distractor_signatures:
                distractor_mask = candidate
                distractor_signatures.add(sig)
                break
        if distractor_mask is None:
            return None
        distractors.append(distractor_mask)

    origin_zone, target_zone = choose_zones(H, W)
    input_used = []
    target_used = []
    final_rects = []
    lock_rects = []
    sprite_coords_by_lock = []

    for index, lock in enumerate(locks):
        final_h, final_w = lock["final_mask"].shape
        rect = try_place_box(final_h, final_w, target_zone, target_used, pad=1)
        if rect is None:
            return None
        final_rects.append(rect)
        fr0, fc0, _, _ = rect
        hole_coords = np.argwhere(lock["final_hole"])
        sprite_coords = hole_coords + np.array([fr0, fc0])
        sprite_coords_by_lock.append(sprite_coords)

    for lock in locks:
        inp_h, inp_w = lock["input_mask"].shape
        rect = try_place_box(inp_h, inp_w, origin_zone, input_used, pad=1)
        if rect is None:
            return None
        lock_rects.append(rect)

    distractor_rects = []
    for mask in distractors:
        rect = try_place_box(mask.shape[0], mask.shape[1], origin_zone, input_used, pad=1)
        if rect is None:
            return None
        distractor_rects.append(rect)

    grid = np.full((H, W), bg, dtype=int)
    random.shuffle(colors)
    lock_colors = colors[:lock_count]
    sprite_colors = colors[lock_count : 2 * lock_count]
    distractor_colors = colors[2 * lock_count : total_needed]

    lock_components = []
    all_sprite_components = []

    for index, lock in enumerate(locks):
        r0, c0, _, _ = lock_rects[index]
        mask = lock["input_mask"]
        color = lock_colors[index]
        coords = []
        for dr, dc in np.argwhere(mask):
            rr = r0 + int(dr)
            cc = c0 + int(dc)
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
            coords.append((rr, cc))
        lock_components.append(np.array(coords, dtype=int))

    for index, sprite_coords in enumerate(sprite_coords_by_lock):
        color = sprite_colors[index]
        coords = []
        for rr, cc in sprite_coords:
            rr = int(rr)
            cc = int(cc)
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
            coords.append((rr, cc))
        all_sprite_components.append(
            {
                "matched": True,
                "lock_index": index,
                "coords": np.array(coords, dtype=int),
            }
        )

    for index, mask in enumerate(distractors):
        r0, c0, _, _ = distractor_rects[index]
        color = distractor_colors[index]
        coords = []
        for dr, dc in np.argwhere(mask):
            rr = r0 + int(dr)
            cc = c0 + int(dc)
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
            coords.append((rr, cc))
        all_sprite_components.append(
            {
                "matched": False,
                "lock_index": None,
                "coords": np.array(coords, dtype=int),
            }
        )

    for lock_index, lock_coords in enumerate(lock_components):
        lock_center = component_center(lock_coords)
        distances = []
        for sprite in all_sprite_components:
            sprite_center = component_center(sprite["coords"])
            dist = (lock_center[0] - sprite_center[0]) ** 2 + (lock_center[1] - sprite_center[1]) ** 2
            distances.append((dist, sprite["matched"], sprite["lock_index"]))
        distances.sort()
        if distances[0][1] and distances[0][2] == lock_index:
            return None

    if max(grid.shape) > 30:
        return None
    return grid
