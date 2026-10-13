from __future__ import annotations

import numpy as np


BG = 0
PALETTE = np.arange(1, 10, dtype=int)
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _zone_of_bbox(box: tuple[int, int, int, int], zone_h: int, zone_w: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    zr0, zc0 = r0 // zone_h, c0 // zone_w
    zr1, zc1 = r1 // zone_h, c1 // zone_w
    assert zr0 == zr1 and zc0 == zc1, "object must stay inside one hidden zone"
    return int(zr0), int(zc0)


def _inspect_grid(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, tuple[int, int], tuple[int, int, int, int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 3 == 0 and g.shape[1] % 3 == 0
    bg = _get_bg(g)
    zone_h, zone_w = g.shape[0] // 3, g.shape[1] // 3

    key_patch = None
    key_zone = None
    key_bbox = None
    sprites: list[dict[str, object]] = []
    for coords in _components(g != bg):
        box = _bbox(coords)
        patch = _crop(g, box)
        mask = patch != bg
        colors = np.unique(patch[mask])
        if patch.shape == (3, 3) and bool(mask.all()) and len(colors) == 9:
            assert key_patch is None, "expected exactly one 3x3 nine-color key block"
            key_patch = patch
            key_zone = _zone_of_bbox(box, zone_h, zone_w)
            key_bbox = box
            continue
        assert len(colors) == 1, "all non-key objects must be monochrome sprites"
        sprites.append(
            {
                "coords": coords,
                "bbox": box,
                "zone": _zone_of_bbox(box, zone_h, zone_w),
                "mask": mask,
                "color": int(colors[0]),
            }
        )

    assert key_patch is not None and key_zone is not None and key_bbox is not None
    assert len(sprites) >= 1
    ref_mask = sprites[0]["mask"]
    for sprite in sprites[1:]:
        assert sprite["mask"].shape == ref_mask.shape
        assert np.array_equal(sprite["mask"], ref_mask), "all sprites must share one normalized mask"
    return bg, key_patch, key_zone, key_bbox, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each repeated sprite from the 3x3 key block using wrapped zone offsets relative to the key zone.

    Input structure:
    - The canvas size is divisible into an implicit 3x3 partition by trisecting rows and columns.
    - One object is a solid 3x3 key block whose nine cells all have different colors.
    - The other foreground objects are monochrome sprites with identical normalized masks, each fully contained in one hidden zone.

    Rule:
    - Identify the hidden 3x3 zone containing the key block and treat that zone as the origin.
    - For each sprite, compute its zone offset from the key zone with row and column differences taken modulo 3.
    - Read the color from the key block at that wrapped relative address, repaint the whole sprite with it, and erase the key block.

    Key insight:
    - The lookup address is not the sprite's absolute zone; the whole canvas is an unmarked 3x3 table indexed relative to the key block's zone, with wraparound.
    """
    g = np.asarray(grid, dtype=int)
    bg, key_patch, key_zone, _key_bbox, sprites = _inspect_grid(g)

    out = np.full_like(g, bg)
    kr, kc = key_zone
    for sprite in sprites:
        sr, sc = sprite["zone"]
        target = int(key_patch[(sr - kr) % 3, (sc - kc) % 3])
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = target
    return out


def _generate_mask(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(200):
        h = int(np.random.randint(2, min(5, max_h) + 1))
        w = int(np.random.randint(2, min(5, max_w) + 1))
        min_area = max(3, int(np.ceil(0.45 * h * w)))
        max_area = min(h * w - 1, int(np.floor(0.75 * h * w)))
        if min_area > max_area:
            continue
        target_area = int(np.random.randint(min_area, max_area + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(np.random.randint(h)), int(np.random.randint(w)))
        mask[start] = True
        while int(mask.sum()) < target_area:
            choices = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                        choices.append((r, c))
            if not choices:
                break
            r, c = choices[int(np.random.randint(len(choices)))]
            mask[r, c] = True
        if int(mask.sum()) != target_area:
            continue
        coords = np.argwhere(mask)
        if _bbox(coords) != (0, 0, h - 1, w - 1):
            continue
        if h == 3 and w == 3 and bool(mask.all()):
            continue
        return mask
    return None


def _pick_key_zone() -> tuple[int, int]:
    choices = ((1, 1), (1, 2), (2, 1), (2, 2))
    return choices[int(np.random.randint(len(choices)))]


def _pick_sprite_zones(key_zone: tuple[int, int]) -> list[tuple[int, int]] | None:
    kr, kc = key_zone
    all_zones = [(r, c) for r in range(3) for c in range(3) if (r, c) != key_zone]
    for _ in range(200):
        count = int(np.random.randint(5, 7))
        pick = [all_zones[i] for i in np.random.choice(len(all_zones), size=count, replace=False)]
        has_above = any(r < kr for r, _ in pick)
        has_left = any(c < kc for _, c in pick)
        has_same_row = any(r == kr for r, _ in pick)
        has_same_col = any(c == kc for _, c in pick)
        has_diagonal = any(r != kr and c != kc for r, c in pick)
        if has_above and has_left and has_same_row and has_same_col and has_diagonal:
            return pick
    return None


def _deranged_colors(targets: list[int]) -> list[int] | None:
    k = len(targets)
    for _ in range(200):
        candidate = [int(v) for v in np.random.permutation(PALETTE)[:k]]
        if len(set(candidate)) == k and all(a != b for a, b in zip(candidate, targets)):
            return candidate
    return None


def _render_input(
    zone_h: int,
    zone_w: int,
    key_zone: tuple[int, int],
    sprite_zones: list[tuple[int, int]],
    mask: np.ndarray,
    key_patch: np.ndarray,
    sprite_colors: list[int],
) -> np.ndarray | None:
    h, w = 3 * zone_h, 3 * zone_w
    out = np.zeros((h, w), dtype=int)
    key_r, key_c = key_zone
    local_r = int(np.random.randint(1, zone_h - 3))
    local_c = int(np.random.randint(1, zone_w - 3))
    gr = key_r * zone_h + local_r
    gc = key_c * zone_w + local_c
    out[gr : gr + 3, gc : gc + 3] = key_patch

    sh, sw = mask.shape
    for (zr, zc), color in zip(sprite_zones, sprite_colors):
        top = int(np.random.randint(1, zone_h - sh))
        left = int(np.random.randint(1, zone_w - sw))
        rr = zr * zone_h + top
        cc = zc * zone_w + left
        patch = out[rr : rr + sh, cc : cc + sw]
        if np.any(patch[mask] != BG):
            return None
        patch[mask] = int(color)
    return out


def _paint_output(
    grid: np.ndarray,
    key_patch: np.ndarray,
    key_zone: tuple[int, int],
    sprites: list[dict[str, object]],
    *,
    relative: bool,
) -> np.ndarray:
    out = np.zeros_like(grid)
    kr, kc = key_zone
    for sprite in sprites:
        sr, sc = sprite["zone"]
        if relative:
            color = int(key_patch[(sr - kr) % 3, (sc - kc) % 3])
        else:
            color = int(key_patch[sr, sc])
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = color
    return out


def _validate_generated(
    grid: np.ndarray,
    key_zone: tuple[int, int],
    sprite_zones: list[tuple[int, int]],
    key_patch: np.ndarray,
) -> bool:
    try:
        bg, found_key, found_zone, key_bbox, sprites = _inspect_grid(grid)
    except AssertionError:
        return False
    if bg != BG or found_zone != key_zone or not np.array_equal(found_key, key_patch):
        return False
    if sorted(tuple(sprite["zone"]) for sprite in sprites) != sorted(sprite_zones):
        return False
    if len({int(sprite["color"]) for sprite in sprites}) != len(sprites):
        return False
    if not any(zr < key_zone[0] for zr, _ in sprite_zones):
        return False
    if not any(zc < key_zone[1] for _, zc in sprite_zones):
        return False
    if not any(zr == key_zone[0] for zr, _ in sprite_zones):
        return False
    if not any(zc == key_zone[1] for _, zc in sprite_zones):
        return False
    if not any(zr != key_zone[0] and zc != key_zone[1] for zr, zc in sprite_zones):
        return False
    targets = [int(key_patch[(zr - key_zone[0]) % 3, (zc - key_zone[1]) % 3]) for zr, zc in sprite_zones]
    if len(set(targets)) != len(targets):
        return False
    for sprite in sprites:
        sr, sc = sprite["zone"]
        target = int(key_patch[(sr - key_zone[0]) % 3, (sc - key_zone[1]) % 3])
        if int(sprite["color"]) == target:
            return False
    if len({tuple(sprite["zone"]) for sprite in sprites}) != len(sprites):
        return False
    expected = _paint_output(grid, key_patch, key_zone, sprites, relative=True)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    r0, c0, r1, c1 = key_bbox
    if np.any(expected[r0 : r1 + 1, c0 : c1 + 1] != BG):
        return False
    absolute = _paint_output(grid, key_patch, key_zone, sprites, relative=False)
    if np.array_equal(expected, absolute):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        zone_h = int(np.random.randint(6, 11))
        zone_w = int(np.random.randint(6, 11))
        mask = _generate_mask(zone_h - 2, zone_w - 2)
        if mask is None:
            continue
        key_zone = _pick_key_zone()
        sprite_zones = _pick_sprite_zones(key_zone)
        if sprite_zones is None:
            continue
        key_patch = np.random.permutation(PALETTE).reshape(3, 3)
        targets = [int(key_patch[(zr - key_zone[0]) % 3, (zc - key_zone[1]) % 3]) for zr, zc in sprite_zones]
        sprite_colors = _deranged_colors(targets)
        if sprite_colors is None:
            continue
        grid = _render_input(zone_h, zone_w, key_zone, sprite_zones, mask, key_patch, sprite_colors)
        if grid is None:
            continue
        _bg, _found_key, _found_zone, _key_bbox, sprites = _inspect_grid(grid)
        expected = _paint_output(grid, key_patch, key_zone, sprites, relative=True)
        if not np.array_equal(main(grid), expected):
            continue
        if not _validate_generated(grid, key_zone, sprite_zones, key_patch):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
