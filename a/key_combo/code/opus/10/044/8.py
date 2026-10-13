from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 5
COLOR_SWAP = {1: 4, 4: 1, 2: 3, 3: 2}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(bool).tobytes()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    colors: list[int] | None = None,
) -> list[dict]:
    if colors is None:
        colors = sorted(int(v) for v in np.unique(grid) if int(v) != int(bg))
    comps: list[dict] = []
    for color in colors:
        for coords in components_mask(grid == int(color), connectivity=connectivity):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if mask is None:
        view[:] = patch
    else:
        use = np.asarray(mask, dtype=bool)
        view[use] = patch[use]


def orientation_table(mask: np.ndarray) -> dict[tuple[tuple[int, int], bytes], tuple[int, np.ndarray]]:
    out: dict[tuple[tuple[int, int], bytes], tuple[int, np.ndarray]] = {}
    base = np.asarray(mask, dtype=bool)
    for flipped in (0, 1):
        cur = np.flipud(base) if flipped else base
        for k in range(4):
            variant = np.rot90(cur, k)
            key = mask_key(variant)
            cost = int(flipped + k)
            prev = out.get(key)
            if prev is None or cost < prev[0]:
                out[key] = (cost, variant.copy())
    return out


def distinct_orientations(mask: np.ndarray) -> list[np.ndarray]:
    return [variant for _cost, variant in orientation_table(mask).values()]


def canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(orientation_table(mask))


def min_transform_cost(src_mask: np.ndarray, dst_mask: np.ndarray) -> int | None:
    return orientation_table(src_mask).get(mask_key(dst_mask), (None, None))[0]


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    grey_components = find_components(grid, bg=bg, connectivity=4, colors=[FRAME])
    if grey_components:
        candidates = grey_components
    else:
        candidates = find_components(grid, bg=bg, connectivity=4)
    assert candidates, "expected a frame component"
    max_area = max(comp["area"] for comp in candidates)
    frame_candidates = [comp for comp in candidates if comp["area"] == max_area]
    assert len(frame_candidates) == 1, "expected a unique largest frame"
    return frame_candidates[0]


def _find_holes(grid: np.ndarray, frame_box: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, c0, _r1, _c1 = frame_box
    sub = crop_bbox(grid, frame_box)
    holes: list[dict] = []
    for coords in components_mask(sub == int(bg), connectivity=4):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 1] == 0) or np.any(coords[:, 0] == sub.shape[0] - 1) or np.any(coords[:, 1] == sub.shape[1] - 1):
            continue
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "bbox_local": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "bbox_abs": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
                "mask": mask,
                "coords_abs": coords + np.array([r0, c0]),
                "canonical": canonical_key(mask),
            }
        )
    holes.sort(key=lambda hole: hole["bbox_abs"])
    return holes


def _find_sprites(grid: np.ndarray, frame_box: tuple[int, int, int, int], bg: int) -> list[dict]:
    outside = np.array(grid, copy=True)
    r0, c0, r1, c1 = frame_box
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprites = []
    for comp in find_components(outside, bg=bg, connectivity=4):
        sr0, sc0, sr1, sc1 = comp["bbox"]
        rel = comp["coords"] - np.array([sr0, sc0])
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        sprites.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "mask": mask,
                "canonical": canonical_key(mask),
            }
        )
    sprites.sort(key=lambda sprite: sprite["bbox"])
    return sprites


def _assign_matches(holes: list[dict], sprites: list[dict]) -> list[dict]:
    matches = []
    used_holes = set()
    for sprite in sprites:
        candidates = []
        for hole_index, hole in enumerate(holes):
            cost = min_transform_cost(sprite["mask"], hole["mask"])
            if cost is not None:
                candidates.append((hole_index, cost))
        assert len(candidates) == 1, "each sprite should match exactly one hole"
        hole_index, cost = candidates[0]
        assert hole_index not in used_holes, "holes should match bijectively"
        used_holes.add(hole_index)
        fill_color = COLOR_SWAP.get(sprite["color"], sprite["color"]) if cost % 2 else sprite["color"]
        matches.append(
            {
                "hole_index": hole_index,
                "sprite": sprite,
                "cost": int(cost),
                "fill_color": int(fill_color),
            }
        )
    assert len(used_holes) == len(holes), "every hole should receive exactly one sprite"
    return matches


def _analyze(grid: np.ndarray) -> tuple[int, dict, list[dict], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame = _find_frame(g, bg)
    holes = _find_holes(g, frame["bbox"], bg)
    sprites = _find_sprites(g, frame["bbox"], bg)
    assert len(holes) == len(sprites) and len(holes) in (3, 4)
    assert len({hole["canonical"] for hole in holes}) == len(holes), "hole shapes must be distinct"
    matches = _assign_matches(holes, sprites)
    assert any(match["cost"] % 2 == 0 for match in matches)
    assert any(match["cost"] % 2 == 1 for match in matches)
    return bg, frame, holes, sprites, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame hole with its matching outside sprite, flipping the fill color when the match needs an odd number of basic transforms.

    Input structure:
    - The background is black, with one large grey rectangular frame near the center.
    - Inside the frame are 3 or 4 black 4-connected holes of distinct irregular shapes.
    - Outside the frame are 3 or 4 monochrome sprites whose shapes match the holes up to rotation and flip.

    Rule:
    - Keep the whole input canvas size, the grey frame, and all outside sprites unchanged.
    - Match each outside sprite to the unique hole with the same shape under the 8 orientation variants generated by quarter-turn rotations and one vertical flip.
    - Let T be the minimum number of basic operations for that sprite-to-hole match, counting each 90-degree rotation as 1 and the optional flip as 1.
    - If T is even, fill the hole with the sprite's own color; if T is odd, first apply the fixed swap 2<->3 and 1<->4, then fill the hole.

    Key insight:
    - The sprite shape identifies which hole to fill, but the decisive color rule comes from the parity of the cheapest matching transformation rather than from shape or color alone.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, _frame, holes, _sprites, matches = _analyze(g)
    for match in matches:
        hole = holes[match["hole_index"]]
        r0, c0, _r1, _c1 = hole["bbox_abs"]
        patch = np.full(hole["mask"].shape, int(match["fill_color"]), dtype=int)
        paste(out, patch, r0, c0, mask=hole["mask"])
    return out


def _shape_is_asymmetric(mask: np.ndarray) -> bool:
    return len(orientation_table(mask)) == 8


def _random_shape() -> np.ndarray | None:
    for _ in range(120):
        area = random.randint(3, 5)
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if h * w < area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {
            (start[0] + dr, start[1] + dc)
            for dr, dc in DIRS4
            if 0 <= start[0] + dr < h and 0 <= start[1] + dc < w
        }
        while int(mask.sum()) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            r, c = cell
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) != area:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) == mask.shape[0] * mask.shape[1]:
            continue
        if not _shape_is_asymmetric(mask):
            continue
        return mask
    return None


def _sample_distinct_shapes(count: int) -> list[np.ndarray] | None:
    shapes = []
    seen = set()
    for _ in range(500):
        shape = _random_shape()
        if shape is None:
            continue
        key = canonical_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _reserve_cells(mask: np.ndarray, coords: list[tuple[int, int]], padding: int) -> bool:
    h, w = mask.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - padding)
        r1 = min(h, r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(w, c + padding + 1)
        if np.any(mask[r0:r1, c0:c1]):
            return False
    for r, c in coords:
        r0 = max(0, r - padding)
        r1 = min(h, r + padding + 1)
        c0 = max(0, c - padding)
        c1 = min(w, c + padding + 1)
        mask[r0:r1, c0:c1] = True
    return True


def _place_holes(frame_h: int, frame_w: int, shapes: list[np.ndarray]) -> list[dict] | None:
    reserved = np.zeros((frame_h, frame_w), dtype=bool)
    placements = []
    for base_shape in shapes:
        hole_shape = random.choice(distinct_orientations(base_shape))
        sh, sw = hole_shape.shape
        candidates = [
            (r, c)
            for r in range(1, frame_h - sh)
            for c in range(1, frame_w - sw)
        ]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates[:250]:
            coords = [(r0 + rr, c0 + cc) for rr, cc in np.argwhere(hole_shape)]
            if _reserve_cells(reserved.copy(), coords, padding=1):
                _reserve_cells(reserved, coords, padding=1)
                placements.append({"base": base_shape, "mask": hole_shape, "top_left": (int(r0), int(c0))})
                placed = True
                break
        if not placed:
            return None
    return placements


def _cost_plan(count: int) -> list[int]:
    if count == 4:
        plan = [0, 1, 2, random.choice([3, 4])]
    else:
        plan = random.choice([[0, 1, 2], [1, 2, 3], [0, 1, 4], [0, 2, 3]])
    random.shuffle(plan)
    return plan


def _pick_sprite_mask(hole_mask: np.ndarray, desired_cost: int) -> np.ndarray | None:
    candidates = [variant for variant in distinct_orientations(hole_mask) if min_transform_cost(variant, hole_mask) == desired_cost]
    if not candidates:
        return None
    return random.choice(candidates).copy()


def _pick_colors(count: int) -> list[int]:
    colors = [1, 2, 3, 4]
    if count == 4:
        random.shuffle(colors)
        return colors
    return random.sample(colors, count)


def _paint_patch(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = np.full(mask.shape, int(color), dtype=int)
    paste(grid, patch, int(r0), int(c0), mask=mask)


def _can_place_sprite(
    reserved: np.ndarray,
    sprite_mask: np.ndarray,
    r0: int,
    c0: int,
    grid_shape: tuple[int, int],
) -> bool:
    h, w = sprite_mask.shape
    gh, gw = grid_shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    coords = [(r0 + rr, c0 + cc) for rr, cc in np.argwhere(sprite_mask)]
    return _reserve_cells(reserved.copy(), coords, padding=1)


def _place_sprites(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    sprite_specs: list[dict],
) -> bool:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = frame_box
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r1 + 2)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c1 + 2)
    reserved[rr0:rr1, cc0:cc1] = True

    for spec in sprite_specs:
        sprite_mask = spec["sprite_mask"]
        sh, sw = sprite_mask.shape
        candidates = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        random.shuffle(candidates)
        placed = False
        for sr, sc in candidates[:400]:
            if not _can_place_sprite(reserved, sprite_mask, sr, sc, grid.shape):
                continue
            coords = [(sr + rr, sc + cc) for rr, cc in np.argwhere(sprite_mask)]
            _reserve_cells(reserved, coords, padding=1)
            _paint_patch(grid, sprite_mask, sr, sc, spec["color"])
            spec["sprite_top_left"] = (int(sr), int(sc))
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected(
    grid: np.ndarray,
    frame_origin: tuple[int, int],
    hole_specs: list[dict],
) -> np.ndarray:
    expected = np.array(grid, copy=True)
    fr, fc = frame_origin
    for spec in hole_specs:
        r0, c0 = spec["top_left"]
        fill = spec["fill_color"]
        patch = np.full(spec["mask"].shape, int(fill), dtype=int)
        paste(expected, patch, int(fr + r0), int(fc + c0), mask=spec["mask"])
    return expected


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, hole_specs: list[dict]) -> bool:
    bg, frame, holes, sprites, matches = _analyze(grid)
    if bg != BG or int(frame["color"]) != FRAME:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    intended = {}
    for spec in hole_specs:
        key = canonical_key(spec["mask"])
        intended[key] = {"cost": int(spec["cost"]), "fill_color": int(spec["fill_color"]), "color": int(spec["color"])}

    for match in matches:
        hole = holes[match["hole_index"]]
        info = intended.get(hole["canonical"])
        if info is None:
            return False
        if int(match["cost"]) != info["cost"]:
            return False
        if int(match["fill_color"]) != info["fill_color"]:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 24)
    grid_w = random.randint(18, 24)
    frame_h = random.randint(10, min(13, grid_h - 6))
    frame_w = random.randint(10, min(13, grid_w - 6))
    count = random.choice([3, 4])
    shapes = _sample_distinct_shapes(count)
    if shapes is None:
        return None

    hole_specs = _place_holes(frame_h, frame_w, shapes)
    if hole_specs is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_top = max(2, min(grid_h - frame_h - 2, grid_h // 2 - frame_h // 2 + random.randint(-1, 1)))
    frame_left = max(2, min(grid_w - frame_w - 2, grid_w // 2 - frame_w // 2 + random.randint(-1, 1)))
    frame_box = (int(frame_top), int(frame_left), int(frame_top + frame_h - 1), int(frame_left + frame_w - 1))
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = FRAME

    costs = _cost_plan(count)
    colors = _pick_colors(count)
    random.shuffle(hole_specs)

    sprite_specs = []
    odd_count = 0
    even_count = 0
    for spec, cost, color in zip(hole_specs, costs, colors):
        sprite_mask = _pick_sprite_mask(spec["mask"], cost)
        if sprite_mask is None:
            return None
        spec["cost"] = int(cost)
        spec["color"] = int(color)
        spec["fill_color"] = int(COLOR_SWAP[color] if cost % 2 else color)
        odd_count += int(cost % 2 == 1)
        even_count += int(cost % 2 == 0)
        sprite_specs.append({"sprite_mask": sprite_mask, "color": int(color)})

    if odd_count == 0 or even_count == 0:
        return None

    for spec in hole_specs:
        rr, cc = spec["top_left"]
        _paint_patch(grid, spec["mask"], frame_top + rr, frame_left + cc, BG)

    if not _place_sprites(grid, frame_box, sprite_specs):
        return None

    expected = _build_expected(grid, (frame_top, frame_left), hole_specs)
    if not _semantic_checks(grid, expected, hole_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
