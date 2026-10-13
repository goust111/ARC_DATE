from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_SEEN_OUTPUT_KEYS: set[tuple[tuple[int, int], bytes]] = set()

_FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GRID_H_RANGE = (16, 23)
_GRID_W_RANGE = (16, 24)
_UNIQUE_COUNT_CHOICES = (3, 4, 5)
_MAX_GENERATION_ATTEMPTS = 600
_MAX_SPRITE_ATTEMPTS = 200
_MAX_PLACEMENT_ATTEMPTS = 120


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_component_crop(
    grid: np.ndarray, coords: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return out


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            if g[r, c] == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _FOUR_NEIGHBORS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] != 0:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"coords": coords, "bbox": _bbox_from_coords(coords), "area": int(len(cells))})
    return comps


def _sprite_key(sprite: np.ndarray) -> tuple[tuple[int, int], bytes]:
    s = np.asarray(sprite, dtype=int)
    return tuple(map(int, s.shape)), s.tobytes()


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, int]:
    comps = _find_components(grid)
    assert 5 <= len(comps) <= 8, "expected 5-8 foreground sprites"

    counts: Counter = Counter()
    crops: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for comp in comps:
        crop = _extract_component_crop(grid, comp["coords"], comp["bbox"])
        key = _sprite_key(crop)
        counts[key] += 1
        crops[key] = crop

    duplicate_keys = [key for key, count in counts.items() if count == 2]
    unique_keys = [key for key, count in counts.items() if count == 1]
    assert len(duplicate_keys) == 1, "expected exactly one crop pattern to occur twice"
    assert sum(count > 2 for count in counts.values()) == 0, "no pattern may occur more than twice"
    assert 3 <= len(unique_keys) <= 5, "expected 3-5 unique distractor sprites"
    return crops[duplicate_keys[0]].copy(), len(unique_keys)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the only repeated sprite crop and use the number of non-repeated sprites to set the width of an alternating mirrored tiling.

    Input structure:
    - The background is black.
    - The foreground consists of 5-8 separated 4-connected multicolor sprites.
    - Exactly one tight bounding-box crop appears twice, while every other sprite crop appears once.

    Rule:
    - Crop each foreground sprite to its tight bounding box, keeping any black holes inside the box.
    - Identify the single crop pattern that occurs exactly twice; call it T.
    - Count the crops that occur only once; call that number N.
    - Mirror T left-to-right to get T', then concatenate N tiles horizontally, using T' in positions 1,3,5,... and T in positions 2,4,6,....

    Key insight:
    - The discarded sprites are not irrelevant noise: their count determines how many mirrored-or-original copies of the repeated sprite must be stitched together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile, n_unique = _analyze_grid(g)
    mirrored = np.fliplr(tile)
    parts = [mirrored.copy() if i % 2 == 0 else tile.copy() for i in range(n_unique)]
    return np.concatenate(parts, axis=1)


def _opposite_corner_walk_options(h: int, w: int) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    return [
        ((0, 0), (h - 1, w - 1), [(1, 0), (0, 1)]),
        ((0, w - 1), (h - 1, 0), [(1, 0), (0, -1)]),
        ((h - 1, 0), (0, w - 1), [(-1, 0), (0, 1)]),
        ((h - 1, w - 1), (0, 0), [(-1, 0), (0, -1)]),
    ]


def _random_connected_mask(h: int, w: int, k: int) -> np.ndarray:
    start, end, moves = _opposite_corner_walk_options(h, w)[int(_RNG.integers(0, 4))]
    r, c = start
    steps = [moves[0]] * abs(end[0] - r) + [moves[1]] * abs(end[1] - c)
    _RNG.shuffle(steps)

    cells = {(r, c)}
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))

    while len(cells) < k:
        frontier = []
        for rr, cc in cells:
            for dr, dc in _FOUR_NEIGHBORS:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(_RNG.integers(0, len(frontier)))]
        cells.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    return mask


def _paint_mask(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    _RNG.shuffle(coords)
    palette = colors.copy()
    _RNG.shuffle(palette)

    out = np.zeros(mask.shape, dtype=int)
    need = len(palette)
    for (r, c), color in zip(coords[:need], palette):
        out[int(r), int(c)] = int(color)
    for r, c in coords[need:]:
        out[int(r), int(c)] = int(palette[int(_RNG.integers(0, len(palette)))])
    return out


def _random_sprite(
    *,
    force_shape: tuple[int, int] | None = None,
    forbidden_keys: set[tuple[tuple[int, int], bytes]] | None = None,
    require_non_symmetric: bool = False,
) -> np.ndarray | None:
    forbidden = forbidden_keys or set()
    feasible_shapes = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if h + w - 1 <= 8
    ]
    if force_shape is not None:
        feasible_shapes = [tuple(map(int, force_shape))]

    for _ in range(_MAX_SPRITE_ATTEMPTS):
        if not feasible_shapes:
            return None
        h, w = feasible_shapes[int(_RNG.integers(0, len(feasible_shapes)))]
        min_fg = max(3, h + w - 1)
        max_fg = min(8, h * w - 1)
        if min_fg > max_fg:
            continue
        k = int(_RNG.integers(min_fg, max_fg + 1))
        mask = _random_connected_mask(h, w, k)
        if mask.sum() != k:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        color_count = 2 if k == 3 else int(_RNG.integers(2, min(3, k) + 1))
        colors = _RNG.choice(np.arange(1, 10), size=color_count, replace=False)
        sprite = _paint_mask(mask, colors)
        if require_non_symmetric and np.array_equal(sprite, np.fliplr(sprite)):
            continue
        key = _sprite_key(sprite)
        if key in forbidden:
            continue
        return sprite
    return None


def _mutate_sprite(sprite: np.ndarray, forbidden_keys: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    h, w = sprite.shape
    mask = sprite != 0
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    bg_coords = [tuple(map(int, rc)) for rc in np.argwhere(~mask)]
    colors = np.unique(sprite[mask])

    for _ in range(_MAX_SPRITE_ATTEMPTS):
        out = sprite.copy()
        changed = False

        if bg_coords and len(fg_coords) < 8 and _RNG.random() < 0.5:
            br, bc = bg_coords[int(_RNG.integers(0, len(bg_coords)))]
            for dr, dc in _FOUR_NEIGHBORS:
                nr = br + dr
                nc = bc + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr, nc] != 0:
                    out[br, bc] = int(colors[int(_RNG.integers(0, len(colors)))])
                    changed = True
                    break
        else:
            candidates = []
            for fr, fc in fg_coords:
                if fr in (0, h - 1) and mask[(0 if fr == h - 1 else h - 1), :].sum() == 0:
                    continue
                if fc in (0, w - 1) and mask[:, (0 if fc == w - 1 else w - 1)].sum() == 0:
                    continue
                candidates.append((fr, fc))
            if candidates:
                fr, fc = candidates[int(_RNG.integers(0, len(candidates)))]
                new_colors = [int(c) for c in colors if int(c) != int(out[fr, fc])]
                if new_colors:
                    out[fr, fc] = int(new_colors[int(_RNG.integers(0, len(new_colors)))])
                    changed = True

        if not changed:
            continue
        if np.count_nonzero(out) < 3:
            continue
        if len(np.unique(out[out != 0])) < 2:
            continue
        if np.array_equal(out, sprite):
            continue
        key = _sprite_key(out)
        if key in forbidden_keys:
            continue
        return out
    return None


def _place_sprite_candidates(grid: np.ndarray, sprite: np.ndarray, forbidden: np.ndarray) -> list[tuple[int, int]]:
    H, W = grid.shape
    h, w = sprite.shape
    coords = np.argwhere(sprite != 0)
    options = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            ok = True
            for dr, dc in coords:
                rr = r0 + int(dr)
                cc = c0 + int(dc)
                if forbidden[rr, cc]:
                    ok = False
                    break
            if ok:
                options.append((r0, c0))
    return options


def _update_forbidden(forbidden: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    H, W = forbidden.shape
    for dr, dc in np.argwhere(sprite != 0):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        forbidden[rr, cc] = True
        for ddr, ddc in _FOUR_NEIGHBORS:
            nr = rr + ddr
            nc = cc + ddc
            if 0 <= nr < H and 0 <= nc < W:
                forbidden[nr, nc] = True


def _place_sprites(sprites: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(sprite.shape[0] for sprite in sprites)
    max_w = max(sprite.shape[1] for sprite in sprites)
    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        H = int(_RNG.integers(max(16, max_h + 6), _GRID_H_RANGE[1] + 1))
        W = int(_RNG.integers(max(16, max_w + 8), _GRID_W_RANGE[1] + 1))
        grid = np.zeros((H, W), dtype=int)
        forbidden = np.zeros((H, W), dtype=bool)
        order = list(range(len(sprites)))
        _RNG.shuffle(order)
        order.sort(key=lambda idx: sprites[idx].shape[0] * sprites[idx].shape[1], reverse=True)
        placements: dict[int, tuple[int, int]] = {}

        success = True
        for idx in order:
            sprite = sprites[idx]
            options = _place_sprite_candidates(grid, sprite, forbidden)
            if not options:
                success = False
                break
            r0, c0 = options[int(_RNG.integers(0, len(options)))]
            h, w = sprite.shape
            patch = grid[r0 : r0 + h, c0 : c0 + w]
            mask = sprite != 0
            patch[mask] = sprite[mask]
            _update_forbidden(forbidden, sprite, r0, c0)
            placements[idx] = (r0, c0)

        if success:
            return grid
    return None


def _build_sprite_set() -> tuple[np.ndarray, list[np.ndarray], int] | None:
    n_unique = int(_RNG.choice(_UNIQUE_COUNT_CHOICES))
    duplicate = _random_sprite(require_non_symmetric=True)
    if duplicate is None:
        return None

    used = {_sprite_key(duplicate)}
    unique_sprites: list[np.ndarray] = []

    mirror = np.fliplr(duplicate)
    mirror_key = _sprite_key(mirror)
    if mirror_key not in used:
        unique_sprites.append(mirror.copy())
        used.add(mirror_key)

    challenger = _mutate_sprite(duplicate, used)
    if challenger is not None and len(unique_sprites) < n_unique:
        unique_sprites.append(challenger)
        used.add(_sprite_key(challenger))

    while len(unique_sprites) < n_unique:
        force_shape = duplicate.shape if _RNG.random() < 0.45 else None
        sprite = _random_sprite(force_shape=force_shape, forbidden_keys=used)
        if sprite is None:
            return None
        unique_sprites.append(sprite)
        used.add(_sprite_key(sprite))

    sprites = [duplicate.copy(), duplicate.copy(), *unique_sprites]
    _RNG.shuffle(sprites)
    return duplicate, sprites, n_unique


def _expected_output(tile: np.ndarray, n_unique: int) -> np.ndarray:
    mirrored = np.fliplr(tile)
    return np.concatenate(
        [mirrored.copy() if i % 2 == 0 else tile.copy() for i in range(n_unique)],
        axis=1,
    )


def _gen_attempt() -> np.ndarray | None:
    built = _build_sprite_set()
    if built is None:
        return None
    duplicate, sprites, n_unique = built
    expected = _expected_output(duplicate, n_unique)
    if max(expected.shape) > 30:
        return None

    grid = _place_sprites(sprites)
    if grid is None:
        return None

    try:
        analyzed_tile, analyzed_n = _analyze_grid(grid)
        actual = main(grid)
    except AssertionError:
        return None

    if analyzed_n != n_unique:
        return None
    if not np.array_equal(analyzed_tile, duplicate):
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(duplicate, np.fliplr(duplicate)):
        return None

    output_key = _sprite_key(actual)
    if output_key in _SEEN_OUTPUT_KEYS:
        return None
    _SEEN_OUTPUT_KEYS.add(output_key)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
