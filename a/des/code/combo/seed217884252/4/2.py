from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GROUP_CONFIGS = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 2, 2)]
TWO_MEMBER_SCALES = [(1, 2), (1, 3), (2, 3)]
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    return (
        int(mask.shape[0]),
        int(mask.shape[1]),
        tuple((int(r), int(c)) for r, c in coords),
    )


def _uniform_block_reduce(mask: np.ndarray, scale: int) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    if scale <= 1 or h % scale != 0 or w % scale != 0:
        return None
    blocks = mask.reshape(h // scale, scale, w // scale, scale)
    block_any = blocks.any(axis=(1, 3))
    block_all = blocks.all(axis=(1, 3))
    if not np.array_equal(block_any, block_all):
        return None
    return np.asarray(block_any, dtype=bool)


def _shape_key_and_scale(mask: np.ndarray) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], int]:
    mask = np.asarray(mask, dtype=bool)
    best_mask = mask
    best_scale = 1
    for scale in range(min(mask.shape), 1, -1):
        reduced = _uniform_block_reduce(mask, scale)
        if reduced is not None:
            best_mask = reduced
            best_scale = int(scale)
            break
    return _mask_key(best_mask), int(best_scale)


def _extract_control_color(region: np.ndarray, obj_mask: np.ndarray) -> int | None:
    colors = Counter(int(region[r, c]) for r, c in np.argwhere(obj_mask))
    ranked = colors.most_common()
    if len(ranked) == 1:
        return None
    assert ranked[0][1] > ranked[1][1]
    dominant = int(ranked[0][0])
    other_colors = [int(color) for color in colors if int(color) != dominant]
    assert len(other_colors) == 1
    control = other_colors[0]
    assert colors[control] == 1
    return int(control)


def _objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    objects: list[dict] = []
    for coords in comps:
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        obj_mask = region != BG
        shape_key, scale = _shape_key_and_scale(obj_mask)
        control_color = _extract_control_color(region, obj_mask)
        objects.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "shape_key": shape_key,
                "scale": int(scale),
                "control_color": control_color,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find shape groups, read each group's embedded control color, keep only the largest scaled member.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    groups: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[dict]] = defaultdict(list)
    for obj in _objects(g):
        groups[obj["shape_key"]].append(obj)

    out = np.zeros_like(g)
    for members in groups.values():
        control_members = [member for member in members if member["control_color"] is not None]
        assert len(control_members) == 1
        target_color = int(control_members[0]["control_color"])

        winner = max(members, key=lambda member: int(member["area"]))
        max_area = int(winner["area"])
        assert sum(int(member["area"]) == max_area for member in members) == 1

        coords = np.asarray(winner["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = target_color
    return out


def _random_polyomino(area: int, h: int, w: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.asarray(out, dtype=bool)


def _random_primitive_mask(existing_keys: set[tuple[int, int, tuple[tuple[int, int], ...]]]) -> np.ndarray | None:
    for _ in range(200):
        area = random.randint(4, 6)
        canvas_h = random.randint(3, 5)
        canvas_w = random.randint(3, 5)
        raw = _random_polyomino(area, canvas_h, canvas_w)
        if raw is None:
            continue
        mask = _tight_crop(raw)
        h, w = map(int, mask.shape)
        if not (2 <= h <= 4 and 2 <= w <= 4):
            continue
        if len(_components(mask)) != 1:
            continue
        mask = _random_transform(mask)
        if _shape_key_and_scale(mask)[1] != 1:
            continue
        key = _mask_key(mask)
        if key in existing_keys:
            continue
        return np.asarray(mask, dtype=bool)
    return None


def _choose_control_cell(mask: np.ndarray) -> tuple[int, int]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    scored: list[tuple[int, int, int, int]] = []
    h, w = map(int, mask.shape)
    for r, c in coords:
        degree = 0
        for dr, dc in DIRS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                degree += 1
        interior = int(0 < r < h - 1 and 0 < c < w - 1)
        scored.append((int(interior), int(degree), int(r), int(c)))
    best_interior = max(item[0] for item in scored)
    best_degree = max(item[1] for item in scored if item[0] == best_interior)
    candidates = [(r, c) for interior, degree, r, c in scored if interior == best_interior and degree == best_degree]
    return random.choice(candidates)


def _make_sprite(base_mask: np.ndarray, scale: int, fill_color: int, control_color: int | None) -> np.ndarray:
    scaled_mask = np.kron(np.asarray(base_mask, dtype=int), np.ones((scale, scale), dtype=int)).astype(bool)
    sprite = np.zeros(scaled_mask.shape, dtype=int)
    sprite[scaled_mask] = int(fill_color)
    if control_color is not None:
        r, c = _choose_control_cell(scaled_mask)
        sprite[int(r), int(c)] = int(control_color)
    return sprite


def _sample_grid_shape(sprites: list[np.ndarray]) -> tuple[int, int] | None:
    padded_area = sum((int(sprite.shape[0]) + 2) * (int(sprite.shape[1]) + 2) for sprite in sprites)
    if padded_area > 720:
        return None

    min_h = max(int(max(sprite.shape[0] for sprite in sprites)) + 2, 16)
    min_w = max(int(max(sprite.shape[1] for sprite in sprites)) + 2, 16)
    min_area = int(np.ceil(padded_area * 1.18))

    choices: list[tuple[int, int]] = []
    for h in range(min_h, 31):
        for w in range(min_w, 31):
            area = int(h * w)
            if min_area <= area <= min(900, min_area + 260):
                choices.append((int(h), int(w)))
    if not choices:
        return None
    return random.choice(choices)


def _place_sprites(sprites: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(sprites)), key=lambda idx: (int(sprites[idx].shape[0] * sprites[idx].shape[1]), int(idx)), reverse=True)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        sprite = np.asarray(sprites[idx], dtype=int)
        sh, sw = map(int, sprite.shape)
        candidates: list[tuple[int, int]] = []
        for top in range(1, h - sh):
            for left in range(1, w - sw):
                r0 = int(top - 1)
                r1 = int(top + sh + 1)
                c0 = int(left - 1)
                c1 = int(left + sw + 1)
                if blocked[r0:r1, c0:c1].any():
                    continue
                candidates.append((int(top), int(left)))
        if not candidates:
            return None
        placements[idx] = random.choice(candidates)
        top, left = placements[idx]
        mask = sprite != BG
        grid[top : top + sh, left : left + sw][mask] = sprite[mask]
        blocked[top - 1 : top + sh + 1, left - 1 : left + sw + 1] = True

    return grid


def _group_sprites() -> list[np.ndarray] | None:
    group_sizes = random.choice(GROUP_CONFIGS)
    group_count = len(group_sizes)
    total_objects = int(sum(group_sizes))
    total_colors = int(total_objects + group_count)
    assert total_colors <= 9

    palette = list(range(1, 10))
    random.shuffle(palette)
    object_colors = palette[:total_objects]
    control_colors = palette[total_objects:total_colors]

    seen_keys: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    sprites: list[np.ndarray] = []
    color_idx = 0

    for group_idx, member_count in enumerate(group_sizes):
        base_mask = _random_primitive_mask(seen_keys)
        if base_mask is None:
            return None
        seen_keys.add(_mask_key(base_mask))

        if member_count == 2:
            scales = list(random.choice(TWO_MEMBER_SCALES))
        else:
            scales = [1, 2, 3]

        marked_idx = random.randrange(member_count)
        for member_idx, scale in enumerate(scales):
            fill_color = int(object_colors[color_idx])
            color_idx += 1
            control_color = int(control_colors[group_idx]) if member_idx == marked_idx else None
            sprites.append(_make_sprite(base_mask, int(scale), fill_color, control_color))

    return sprites


def _attempt_input() -> np.ndarray | None:
    sprites = _group_sprites()
    if sprites is None:
        return None

    grid_shape = _sample_grid_shape(sprites)
    if grid_shape is None:
        return None

    grid = _place_sprites(sprites, grid_shape)
    if grid is None:
        return None

    if np.count_nonzero(grid) * 3 > grid.size:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
