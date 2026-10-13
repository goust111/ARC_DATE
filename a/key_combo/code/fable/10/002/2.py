from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
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
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    coords = list(coords)
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = int(grid[r, c])
    return out


def _dihedral_variants(arr: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    seen_keys: set[tuple[tuple[int, int], ...]] = set()
    for rot in range(4):
        rotated = np.rot90(arr, rot)
        for mirrored in (False, True):
            variant = np.fliplr(rotated) if mirrored else rotated
            cells = [(int(r), int(c), int(variant[r, c])) for r, c in np.argwhere(variant != BG)]
            footprint = tuple(sorted((r, c) for r, c, _ in cells))
            if footprint in seen_keys:
                continue
            seen_keys.add(footprint)
            variants.append(
                {
                    "rot": rot,
                    "mirrored": mirrored,
                    "cells": cells,
                    "footprint": footprint,
                    "shape": set(footprint),
                    "shape_key": footprint,
                    "height": int(variant.shape[0]),
                    "width": int(variant.shape[1]),
                }
            )
    return variants


def _extract_scene(grid: np.ndarray) -> tuple[list[np.ndarray], list[set[tuple[int, int]]]]:
    comps = _components_from_mask(grid != BG)
    sprites: list[np.ndarray] = []
    shadows: list[set[tuple[int, int]]] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if colors == {GRAY}:
            shadows.append(set(comp))
        else:
            assert GRAY not in colors, "gray is reserved for silhouettes"
            assert len(colors) >= 2, "source sprites must be multicolor"
            sprites.append(_crop_component(grid, comp))
    sprites.sort(key=lambda arr: (arr.shape[0], arr.shape[1], arr.tobytes()))
    shadows.sort(key=lambda comp: min(comp))
    return sprites, shadows


def _shadow_candidates(
    grid: np.ndarray,
    shadow: set[tuple[int, int]],
    sprite_variants: list[list[dict]],
) -> list[dict]:
    h, w = grid.shape
    shadow_list = sorted(shadow)
    candidates: dict[tuple[int, tuple[tuple[int, int], ...]], dict] = {}
    for sprite_idx, variants in enumerate(sprite_variants):
        for variant in variants:
            footprint = variant["footprint"]
            shape = variant["shape"]
            shifts = {
                (sr - fr, sc - fc)
                for sr, sc in shadow_list
                for fr, fc in footprint
            }
            for dr, dc in shifts:
                placed_shape = {(r + dr, c + dc) for r, c in shape}
                if not shadow.issubset(placed_shape):
                    continue
                placed_cells = []
                ok = True
                for r, c, color in variant["cells"]:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        ok = False
                        break
                    existing = int(grid[rr, cc])
                    if existing != BG and not (existing == GRAY and (rr, cc) in shadow):
                        ok = False
                        break
                    placed_cells.append((rr, cc, color))
                if not ok:
                    continue
                key = (sprite_idx, tuple(sorted(placed_shape)))
                candidates[key] = {
                    "sprite_idx": sprite_idx,
                    "variant": variant,
                    "cells": placed_cells,
                    "footprint": key[1],
                    "top_left": (dr, dc),
                }
    return [candidates[key] for key in sorted(candidates)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each gray silhouette by stamping the uniquely matching sprite pose over it.

    Input structure:
    - The canvas has a black background with one or more 8-connected multicolor source sprites.
    - Separate gray 8-connected components are partial shape traces of some rotated or mirrored sprite.
    - Gray carries only footprint information; source sprites keep the original colors to be copied later.

    Rule:
    - For each gray component, test every source sprite under all eight dihedral poses and all translations.
    - Keep the unique candidate whose full footprint completely covers that gray component, stays in bounds,
      and does not overlap any other foreground component.
    - Stamp that full colored sprite at the matched pose and position, leaving the original source sprites in place.

    Key insight:
    - The gray clues are incomplete shape fragments, so the task is solved by geometric template matching first;
      only after the unique pose is identified can the missing colors be restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30
    sprites, shadows = _extract_scene(g)
    assert sprites and shadows

    sprite_variants = [_dihedral_variants(sprite) for sprite in sprites]
    out = g.copy()
    for shadow in shadows:
        matches = _shadow_candidates(g, shadow, sprite_variants)
        assert len(matches) == 1, "each gray silhouette must have exactly one full-cover match"
        for r, c, color in matches[0]["cells"]:
            out[r, c] = color
    return out


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _touches_bbox_edges(coords: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return (
        any(r == r0 for r, _ in coords)
        and any(r == r1 for r, _ in coords)
        and any(c == c0 for _, c in coords)
        and any(c == c1 for _, c in coords)
    )


def _removable_cells(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    options = []
    for cell in coords:
        remaining = set(coords)
        remaining.remove(cell)
        if _is_connected(remaining) and _touches_bbox_edges(remaining):
            options.append(cell)
    return options


def _max_removals(coords: set[tuple[int, int]], trials: int = 40) -> int:
    best = 0
    for _ in range(trials):
        current = set(coords)
        removed = 0
        while True:
            options = _removable_cells(current)
            if not options:
                break
            current.remove(random.choice(options))
            removed += 1
        best = max(best, removed)
    return best


def _generate_shape(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(300):
        h = rng.randint(4, 5)
        w = rng.randint(4, 5)
        area = rng.randint(7, min(9, h * w - 1))
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        if not _touches_bbox_edges(cells):
            continue
        mask = np.zeros((h, w), dtype=int)
        for r, c in cells:
            mask[r, c] = 1
        if len(_dihedral_variants(mask)) != 8:
            continue
        if _max_removals(cells) < 2:
            continue
        return cells
    raise RuntimeError("failed to generate a suitable sprite shape")


def _color_sprite(coords: set[tuple[int, int]], rng: random.Random) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    area = len(coords)
    n_colors = 2 if area < 8 else rng.choice((2, 3))
    palette = rng.sample(SPRITE_COLORS, n_colors)
    dominant = palette[0]
    ordered = sorted(coords)
    color_list = [dominant] * area
    accent_slots = list(range(area))
    rng.shuffle(accent_slots)
    for idx, color in enumerate(palette[1:]):
        color_list[accent_slots[idx]] = color
    for slot in accent_slots[len(palette) - 1 :]:
        if rng.random() < 0.35:
            color_list[slot] = rng.choice(palette[1:])
    arr = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(ordered, color_list):
        arr[r, c] = int(color)
    return arr


def _blocked_with_padding(coords: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _place_variant(
    variant: dict,
    grid_h: int,
    grid_w: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], list[tuple[int, int, int]]]:
    max_r = grid_h - variant["height"] - 1
    max_c = grid_w - variant["width"] - 1
    if max_r < 1 or max_c < 1:
        raise RuntimeError("grid too small for placement")
    for _ in range(400):
        top = rng.randint(1, max_r)
        left = rng.randint(1, max_c)
        abs_cells = [(r + top, c + left, color) for r, c, color in variant["cells"]]
        if all((r, c) not in blocked for r, c, _ in abs_cells):
            return (top, left), abs_cells
    raise RuntimeError("failed to place component without overlap")


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int, int]]) -> None:
    for r, c, color in cells:
        grid[r, c] = int(color)


def _make_partial_shape(
    footprint: set[tuple[int, int]],
    style: str,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], int]:
    if style == "heavy":
        target_choices = [3, 2]
    elif style == "light":
        target_choices = [1]
    else:
        target_choices = [rng.choice((1, 2, 3)), 2, 1]

    best_partial = None
    best_removed = -1
    for target in target_choices:
        for _ in range(80):
            current = set(footprint)
            removed = 0
            while removed < target:
                options = _removable_cells(current)
                if not options:
                    break
                current.remove(rng.choice(options))
                removed += 1
            if removed > best_removed:
                best_partial = set(current)
                best_removed = removed
            if removed == target and removed > 0:
                return current, removed
    assert best_partial is not None and best_removed > 0
    return best_partial, best_removed


def _intended_full_footprint(cells: list[tuple[int, int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r, c) for r, c, _ in cells))


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    shadow_targets: list[tuple[tuple[int, int], ...]],
) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    sprites, shadows = _extract_scene(inp)
    if len(shadows) != len(shadow_targets) or len(sprites) < 2:
        return False
    sprite_variants = [_dihedral_variants(sprite) for sprite in sprites]
    matched_targets = []
    for shadow in shadows:
        matches = _shadow_candidates(inp, shadow, sprite_variants)
        if len(matches) != 1:
            return False
        matched_targets.append(matches[0]["footprint"])
    return sorted(matched_targets) == sorted(shadow_targets)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        try:
            n_sprites = rng.choice((2, 3))
            sprite_arrays: list[np.ndarray] = []
            occupied_shape_keys: set[tuple[tuple[int, int], ...]] = set()
            for _sprite_idx in range(n_sprites):
                for _inner in range(200):
                    shape = _generate_shape(rng)
                    probe = np.zeros(
                        (max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1),
                        dtype=int,
                    )
                    for r, c in shape:
                        probe[r, c] = 1
                    shape_keys = {variant["shape_key"] for variant in _dihedral_variants(probe)}
                    if shape_keys & occupied_shape_keys:
                        continue
                    occupied_shape_keys |= shape_keys
                    sprite_arrays.append(_color_sprite(shape, rng))
                    break
                else:
                    raise RuntimeError("failed to find a distinct sprite")

            sprite_variants = [_dihedral_variants(sprite) for sprite in sprite_arrays]
            shadow_sprite_ids = list(range(n_sprites))
            if n_sprites == 2:
                extra = rng.choice((0, 1))
                if extra:
                    shadow_sprite_ids.append(rng.randrange(n_sprites))
            else:
                if rng.random() < 0.7:
                    shadow_sprite_ids.append(rng.randrange(n_sprites))
            rng.shuffle(shadow_sprite_ids)

            grid_h = rng.randint(20, 28)
            grid_w = rng.randint(20, 28)
            output = np.zeros((grid_h, grid_w), dtype=int)
            input_grid = np.zeros((grid_h, grid_w), dtype=int)
            blocked: set[tuple[int, int]] = set()

            source_cells_per_sprite: list[list[tuple[int, int, int]]] = []
            for sprite_idx, variants in enumerate(sprite_variants):
                source_variant = next(
                    variant
                    for variant in variants
                    if variant["rot"] == 0 and not variant["mirrored"]
                )
                _top_left, abs_cells = _place_variant(source_variant, grid_h, grid_w, blocked, rng)
                _paint_cells(output, abs_cells)
                _paint_cells(input_grid, abs_cells)
                source_cells_per_sprite.append(abs_cells)
                blocked |= _blocked_with_padding([(r, c) for r, c, _ in abs_cells], grid_h, grid_w)

            shadow_specs = []
            for shadow_idx, sprite_idx in enumerate(shadow_sprite_ids):
                variants = sprite_variants[sprite_idx]
                if shadow_idx == 0:
                    choices = [v for v in variants if (not v["mirrored"] and v["rot"] != 0)]
                    style = "heavy"
                elif shadow_idx == 1:
                    choices = [v for v in variants if v["mirrored"]]
                    style = "light"
                else:
                    if rng.random() < 0.5:
                        choices = [v for v in variants if v["mirrored"]]
                    else:
                        choices = [v for v in variants if not (v["rot"] == 0 and not v["mirrored"])]
                    style = "random"
                if not choices:
                    choices = [v for v in variants if not (v["rot"] == 0 and not v["mirrored"])] or variants
                variant = rng.choice(choices)
                _top_left, abs_cells = _place_variant(variant, grid_h, grid_w, blocked, rng)
                _paint_cells(output, abs_cells)
                blocked |= _blocked_with_padding([(r, c) for r, c, _ in abs_cells], grid_h, grid_w)
                partial_rel, removed = _make_partial_shape(set(variant["footprint"]), style, rng)
                shadow_cells = [(r + _top_left[0], c + _top_left[1], GRAY) for r, c in sorted(partial_rel)]
                _paint_cells(input_grid, shadow_cells)
                shadow_specs.append(
                    {
                        "full": _intended_full_footprint(abs_cells),
                        "removed": removed,
                        "mirrored": variant["mirrored"],
                        "rot": variant["rot"],
                    }
                )

            if sum(spec["mirrored"] for spec in shadow_specs) == 0:
                continue
            if sum((not spec["mirrored"]) and spec["rot"] != 0 for spec in shadow_specs) == 0:
                continue
            if max(spec["removed"] for spec in shadow_specs) < 2:
                continue
            if not _validate_case(input_grid, output, [spec["full"] for spec in shadow_specs]):
                continue
            return input_grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid gray-silhouette puzzle")
