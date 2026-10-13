from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
FOREGROUND_COLORS = (BLUE, RED)
BOX_OPTIONS = ((2, 2), (2, 3), (3, 3), (3, 4), (4, 4))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != bg)


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = _neighbors(connectivity)
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
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_rel(shape: tuple[int, int], rel_coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rel_coords[:, 0], rel_coords[:, 1]] = True
    return mask


def _patch_from_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = _colors_present(g, bg)
    assert len(colors) == 2, "expected exactly two foreground colors"

    sprites: list[dict] = []
    templates: dict[int, np.ndarray] = {}

    for color in colors:
        comps = _components(g == color, connectivity=8)
        assert 2 <= len(comps) <= 3, "expected 2-3 copies per foreground color"

        masks = []
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            rel = coords - np.array([r0, c0], dtype=int)
            mask = _mask_from_rel((r1 - r0 + 1, c1 - c0 + 1), rel)
            masks.append(mask)
            sprites.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "coords": coords.copy(),
                    "rel_coords": rel.copy(),
                    "shape": mask.shape,
                }
            )
        first = masks[0]
        assert all(mask.shape == first.shape and np.array_equal(mask, first) for mask in masks[1:])
        templates[int(color)] = first

    shapes = {mask.shape for mask in templates.values()}
    assert len(shapes) == 1, "templates must share one bbox size"
    return bg, sprites, templates


def _settle_right(sprites: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    settled: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    order = sorted(
        range(len(sprites)),
        key=lambda idx: (sprites[idx]["bbox"][3], sprites[idx]["bbox"][1]),
        reverse=True,
    )
    for idx in order:
        sprite = sprites[idx]
        coords = sprite["coords"].copy()
        while True:
            nxt = coords + np.array([0, 1], dtype=int)
            if np.any(nxt[:, 1] >= w):
                break
            if any((int(r), int(c)) in occupied for r, c in nxt):
                break
            coords = nxt
        for r, c in coords:
            occupied.add((int(r), int(c)))
        settled.append(
            {
                "color": sprite["color"],
                "coords": coords,
                "bbox": _bbox(coords),
                "shape": sprite["shape"],
                "rel_coords": sprite["rel_coords"],
                "orig_bbox": sprite["bbox"],
            }
        )
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored sprite right, then refill every vacated original box with the other sprite template.

    Input structure:
    - The background is black and exactly two foreground colors are present.
    - Each foreground color appears as 2-3 identical 8-connected sprite copies.
    - The two sprite templates have the same bounding-box size, but different masks.

    Rule:
    - Detect every sprite copy and record its original bounding box and color.
    - Slide all sprites rigidly to the right; settle the rightmost sprite first, then each remaining sprite against the border or already settled sprites.
    - After sliding, draw the other color's template back at each sprite's original box position.
    - Finally overlay the slid sprites on top, so moved sprite pixels overwrite any swapped template pixels they cover.

    Key insight:
    - The transformation combines rightward rigid-body settling with simultaneous template swapping at the sprites' original locations, and the moved sprites have the top visual layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, sprites, templates = _extract_sprites(g)
    settled = _settle_right(sprites, g.shape)

    colors = sorted(templates)
    other = {colors[0]: colors[1], colors[1]: colors[0]}
    swapped_patches = {color: _patch_from_mask(mask, color, bg) for color, mask in templates.items()}

    out = np.full_like(g, bg)
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        patch = swapped_patches[other[sprite["color"]]]
        rr = slice(r0, r0 + patch.shape[0])
        cc = slice(c0, c0 + patch.shape[1])
        draw_mask = patch != bg
        view = out[rr, cc]
        view[draw_mask] = patch[draw_mask]

    for sprite in settled:
        color = int(sprite["color"])
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _split_nonnegative(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    vals = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return vals


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask.astype(bool), connectivity=8)) == 1


def _random_template_mask(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        density = random.uniform(0.45, 0.75)
        mask = np.random.random((h, w)) < density
        area = int(mask.sum())
        if area < 2 or area >= h * w:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not _is_connected(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a connected sprite mask")


def _random_template_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        a = _random_template_mask(h, w)
        b = _random_template_mask(h, w)
        if np.array_equal(a, b):
            continue
        if int(np.sum(a != b)) < 2:
            continue
        return a, b
    raise RuntimeError("failed to sample distinct template masks")


def _row_starts(total: int, sprite_h: int, grid_h: int) -> list[int]:
    min_needed = total * sprite_h + (total - 1)
    extra = grid_h - min_needed
    gaps = _split_nonnegative(extra, total + 1)
    starts = []
    row = gaps[0]
    for idx in range(total):
        starts.append(row)
        row += sprite_h
        if idx < total - 1:
            row += 1 + gaps[idx + 1]
    return starts


def _build_input_attempt() -> tuple[np.ndarray, list[dict]] | None:
    box_h, box_w = random.choice(BOX_OPTIONS)
    total_blue = random.randint(2, 3)
    total_red = random.randint(2, 3)
    total = total_blue + total_red

    min_h = total * box_h + (total - 1) + 2
    low_h = max(10, min_h)
    high_h = min(30, min_h + 4)
    if low_h > high_h:
        return None
    grid_h = random.randint(low_h, high_h)
    min_w = max(11, 2 * box_w + 3)
    grid_w = random.randint(min_w, min(18, min_w + 5))
    final_c0 = grid_w - box_w
    if final_c0 < box_w:
        return None

    blue_mask, red_mask = _random_template_pair(box_h, box_w)
    masks = {BLUE: blue_mask, RED: red_mask}

    colors = [BLUE] * total_blue + [RED] * total_red
    random.shuffle(colors)
    rows = _row_starts(total, box_h, grid_h)

    force_overlap = random.random() < 0.6
    overlap_slots: set[int] = set()
    if force_overlap:
        overlap_slots.add(random.randrange(total))
        if total >= 5 and random.random() < 0.25:
            overlap_slots.add(random.randrange(total))

    sprites: list[dict] = []
    for idx, color in enumerate(colors):
        if idx in overlap_slots:
            shift = random.randint(1, box_w - 1)
        else:
            shift = random.randint(box_w, final_c0)
        c0 = final_c0 - shift
        sprites.append({"color": color, "r0": rows[idx], "c0": c0})

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    for sprite in sprites:
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        color = int(sprite["color"])
        mask = masks[color]
        patch = _patch_from_mask(mask, color, BACKGROUND)
        rr = slice(r0, r0 + box_h)
        cc = slice(c0, c0 + box_w)
        view = grid[rr, cc]
        view[mask] = patch[mask]

    return grid, sprites


def _visible_swaps_ok(output: np.ndarray, sprites: list[dict], masks: dict[int, np.ndarray]) -> bool:
    other = {BLUE: RED, RED: BLUE}
    for sprite in sprites:
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        color = int(sprite["color"])
        swapped_color = other[color]
        swapped_mask = masks[swapped_color]
        view = output[r0 : r0 + swapped_mask.shape[0], c0 : c0 + swapped_mask.shape[1]]
        if not np.any(view[swapped_mask] == swapped_color):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_input_attempt()
        if built is None:
            continue
        grid, sprite_meta = built
        output = main(grid)

        if np.array_equal(grid, output):
            continue
        if max(grid.shape) > 30 or max(output.shape) > 30:
            continue

        bg, sprites, templates = _extract_sprites(grid)
        if bg != BACKGROUND:
            continue
        row_ranges = sorted((s["bbox"][0], s["bbox"][2]) for s in sprites)
        if any(row_ranges[i][1] >= row_ranges[i + 1][0] for i in range(len(row_ranges) - 1)):
            continue
        if not _visible_swaps_ok(output, sprite_meta, templates):
            continue

        moved = _settle_right(sprites, grid.shape)
        if not all(s["bbox"][1] > s["orig_bbox"][1] for s in moved):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
