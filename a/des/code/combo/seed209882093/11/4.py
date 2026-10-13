from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
COLORS = list(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
PAIR_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (22, 34)
GRID_W_RANGE = (22, 34)
SHAPE_AREA_RANGE = (3, 8)
SHAPE_MAX_DIM = 5
CONTAINER_MARGIN_RANGE = (1, 3)
CONTAINER_PAD_RANGE = (0, 2)
PROTRUSION_COUNT_RANGE = (0, 4)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_SHAPE_ATTEMPTS = 200
MAX_CONTAINER_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
    else:
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0 = min(rows)
        c0 = min(cols)
        r1 = max(rows)
        c1 = max(cols)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(
    coords: list[tuple[int, int]] | np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    arr = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1), arr


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    base = mask.astype(bool)
    seen: set[tuple[tuple[int, int], bytes]] = set()
    variants: list[np.ndarray] = []
    for flip in (False, True):
        cur = np.fliplr(base) if flip else base
        for k in range(4):
            variant = np.rot90(cur, k)
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, variant.shape)), variant.tobytes()) for variant in _orientations(mask))


def _matching_orientation(src_mask: np.ndarray, dst_mask: np.ndarray) -> np.ndarray | None:
    for variant in _orientations(src_mask):
        if variant.shape == dst_mask.shape and np.array_equal(variant, dst_mask):
            return variant
    return None


def _hole_regions(sprite_mask: np.ndarray) -> list[dict[str, object]]:
    inv = ~sprite_mask
    h, w = sprite_mask.shape
    holes: list[dict[str, object]] = []
    for comp in _components(inv):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        hole_mask, hole_bbox, hole_coords = _coords_to_mask(comp)
        holes.append({"mask": hole_mask, "bbox": hole_bbox, "coords": hole_coords})
    return holes


def _find_sprites(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    sprites: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == BG or seen[r, c]:
                continue
            color = int(g[r, c])
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if g[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            mask, bbox_data, coord_arr = _coords_to_mask(coords)
            sprites.append(
                {
                    "color": color,
                    "mask": mask,
                    "bbox": bbox_data,
                    "coords": coord_arr,
                    "holes": _hole_regions(mask),
                }
            )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each solid insert sprite into the hole of a container sprite whose
    enclosed black region has the same shape up to rotation or reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprites = _find_sprites(g)
    assert sprites

    containers: list[dict[str, object]] = []
    inserts: list[dict[str, object]] = []

    for sprite in sprites:
        holes = sprite["holes"]
        if len(holes) == 0:
            inserts.append(sprite)
            continue
        assert len(holes) == 1, "each sprite may contain at most one hole"
        hole = holes[0]
        sr0, sc0, _sr1, _sc1 = sprite["bbox"]
        hr0, hc0, hr1, hc1 = hole["bbox"]
        container = dict(sprite)
        container["hole_mask"] = hole["mask"]
        container["hole_bbox_global"] = (
            int(sr0 + hr0),
            int(sc0 + hc0),
            int(sr0 + hr1),
            int(sc0 + hc1),
        )
        containers.append(container)

    assert containers
    assert len(containers) == len(inserts)

    containers_by_key: dict[tuple[tuple[int, int], bytes], list[dict[str, object]]] = defaultdict(list)
    inserts_by_key: dict[tuple[tuple[int, int], bytes], list[dict[str, object]]] = defaultdict(list)
    for container in containers:
        containers_by_key[_canonical_key(container["hole_mask"])].append(container)
    for insert in inserts:
        inserts_by_key[_canonical_key(insert["mask"])].append(insert)

    assert set(containers_by_key) == set(inserts_by_key)

    out = g.copy()
    for key in sorted(containers_by_key, key=lambda item: (item[0], item[1])):
        cur_containers = sorted(
            containers_by_key[key],
            key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]),
        )
        cur_inserts = sorted(
            inserts_by_key[key],
            key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]),
        )
        assert len(cur_containers) == len(cur_inserts)
        for container, insert in zip(cur_containers, cur_inserts):
            coords = insert["coords"]
            out[coords[:, 0], coords[:, 1]] = BG
            oriented = _matching_orientation(insert["mask"], container["hole_mask"])
            assert oriented is not None
            hr0, hc0, hr1, hc1 = container["hole_bbox_global"]
            patch = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
            patch[oriented] = int(insert["color"])

    return out


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_shape(used_keys: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target_area = random.randint(*SHAPE_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    frontier.add((r + dr, c + dc))
            frontier.difference_update(cells)
            cells.add(random.choice(tuple(frontier)))
        mask = _mask_from_cells(cells)
        if mask.shape[0] > SHAPE_MAX_DIM or mask.shape[1] > SHAPE_MAX_DIM:
            continue
        if _hole_regions(mask):
            continue
        key = _canonical_key(mask)
        if key in used_keys:
            continue
        return mask
    return None


def _choose_oriented_pair(base_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variants = _orientations(base_mask)
    insert_mask = random.choice(variants).copy()
    if len(variants) > 1 and random.random() < 0.75:
        alternatives = [variant for variant in variants if not np.array_equal(variant, insert_mask)]
        hole_mask = random.choice(alternatives).copy() if alternatives else insert_mask.copy()
    else:
        hole_mask = random.choice(variants).copy()
    return insert_mask, hole_mask


def _make_container_mask(hole_mask: np.ndarray) -> np.ndarray | None:
    hole_h, hole_w = hole_mask.shape
    for _ in range(MAX_CONTAINER_ATTEMPTS):
        top_margin = random.randint(*CONTAINER_MARGIN_RANGE)
        bottom_margin = random.randint(*CONTAINER_MARGIN_RANGE)
        left_margin = random.randint(*CONTAINER_MARGIN_RANGE)
        right_margin = random.randint(*CONTAINER_MARGIN_RANGE)
        pad_top = random.randint(*CONTAINER_PAD_RANGE)
        pad_bottom = random.randint(*CONTAINER_PAD_RANGE)
        pad_left = random.randint(*CONTAINER_PAD_RANGE)
        pad_right = random.randint(*CONTAINER_PAD_RANGE)

        base_h = hole_h + top_margin + bottom_margin
        base_w = hole_w + left_margin + right_margin
        canvas_h = base_h + pad_top + pad_bottom
        canvas_w = base_w + pad_left + pad_right

        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        base_r0 = pad_top
        base_c0 = pad_left
        mask[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

        hole_r0 = base_r0 + top_margin
        hole_c0 = base_c0 + left_margin
        hole_region = mask[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w]
        hole_region[hole_mask] = False

        for _ in range(random.randint(*PROTRUSION_COUNT_RANGE)):
            sides = []
            if pad_top > 0:
                sides.append("top")
            if pad_bottom > 0:
                sides.append("bottom")
            if pad_left > 0:
                sides.append("left")
            if pad_right > 0:
                sides.append("right")
            if not sides:
                break
            side = random.choice(sides)
            if side == "top":
                depth = random.randint(1, pad_top)
                span = random.randint(1, base_w)
                offset = random.randint(0, base_w - span)
                mask[base_r0 - depth : base_r0, base_c0 + offset : base_c0 + offset + span] = True
            elif side == "bottom":
                depth = random.randint(1, pad_bottom)
                span = random.randint(1, base_w)
                offset = random.randint(0, base_w - span)
                mask[
                    base_r0 + base_h : base_r0 + base_h + depth,
                    base_c0 + offset : base_c0 + offset + span,
                ] = True
            elif side == "left":
                depth = random.randint(1, pad_left)
                span = random.randint(1, base_h)
                offset = random.randint(0, base_h - span)
                mask[base_r0 + offset : base_r0 + offset + span, base_c0 - depth : base_c0] = True
            else:
                depth = random.randint(1, pad_right)
                span = random.randint(1, base_h)
                offset = random.randint(0, base_h - span)
                mask[
                    base_r0 + offset : base_r0 + offset + span,
                    base_c0 + base_w : base_c0 + base_w + depth,
                ] = True

        mask = _crop_true(mask)
        holes = _hole_regions(mask)
        if len(holes) != 1:
            continue
        if np.array_equal(holes[0]["mask"], hole_mask):
            return mask
    return None


def _reserve(blocked: np.ndarray, r: int, c: int, gap: int) -> None:
    h, w = blocked.shape
    r0 = max(0, r - gap)
    r1 = min(h, r + gap + 1)
    c0 = max(0, c - gap)
    c1 = min(w, c + gap + 1)
    blocked[r0:r1, c0:c1] = True


def _place_mask(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        gr = int(top + rr)
        gc = int(left + cc)
        grid[gr, gc] = int(color)
        _reserve(blocked, gr, gc, PLACEMENT_GAP)


def _candidate_positions(blocked: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    grid_h, grid_w = blocked.shape
    mask_h, mask_w = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(grid_h - mask_h + 1):
        for left in range(grid_w - mask_w + 1):
            if not blocked[top : top + mask_h, left : left + mask_w][mask].any():
                candidates.append((top, left))
    return candidates


def _compose_scene(sprites: list[dict[str, object]]) -> np.ndarray | None:
    total_area = sum(int(np.sum(sprite["mask"])) for sprite in sprites)
    max_h = max(int(sprite["mask"].shape[0]) for sprite in sprites)
    max_w = max(int(sprite["mask"].shape[1]) for sprite in sprites)

    min_h = max(GRID_H_RANGE[0], max_h + 2, int(np.ceil(np.sqrt(total_area * 2.8))))
    min_w = max(GRID_W_RANGE[0], max_w + 2, int(np.ceil(np.sqrt(total_area * 2.8))))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(sprites)
    random.shuffle(order)
    for sprite in order:
        mask = sprite["mask"]
        candidates = _candidate_positions(blocked, mask)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _place_mask(grid, blocked, mask, int(sprite["color"]), int(top), int(left))
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    if 2 * pair_count > len(COLORS):
        return None

    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    pair_specs: list[dict[str, object]] = []

    for _ in range(pair_count):
        base_mask = _random_shape(used_keys)
        if base_mask is None:
            return None
        used_keys.add(_canonical_key(base_mask))
        insert_mask, hole_mask = _choose_oriented_pair(base_mask)
        container_mask = _make_container_mask(hole_mask)
        if container_mask is None:
            return None
        pair_specs.append(
            {
                "insert_mask": insert_mask,
                "container_mask": container_mask,
            }
        )

    palette = random.sample(COLORS, 2 * pair_count)
    sprites: list[dict[str, object]] = []
    for idx, spec in enumerate(pair_specs):
        container_color = palette[2 * idx]
        insert_color = palette[2 * idx + 1]
        sprites.append({"mask": spec["container_mask"], "color": container_color})
        sprites.append({"mask": spec["insert_mask"], "color": insert_color})

    grid = _compose_scene(sprites)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    in_sprites = _find_sprites(grid)
    out_sprites = _find_sprites(out)
    if len(in_sprites) != len(out_sprites):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
