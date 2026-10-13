from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
TARGET_RED = 2
SEPARATOR = 5
GEN_ATTEMPTS = 500


def _most_common(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(pts, dtype=int))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    pts = [(int(r) - r0, int(c) - c0) for r, c in coords]
    return tuple(sorted(pts))


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _shape_key_from_coords(np.argwhere(mask))


def _coords_to_patch(coords: np.ndarray, color: int, bg: int = BACKGROUND) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch


def _parse_panels(area: np.ndarray, bg: int):
    h, _w = area.shape
    if h < 2 or h % 2:
        return None
    ph = h // 2
    top = area[:ph, :]
    bottom = area[ph:, :]
    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    if bg not in top_colors or bg not in bottom_colors:
        return None
    top_fg = top_colors - {bg}
    bottom_fg = bottom_colors - {bg}
    if len(top_fg) != 1 or len(bottom_fg) != 1:
        return None
    if next(iter(top_fg)) == next(iter(bottom_fg)):
        return None
    target_full = (top == bg) & (bottom == bg)
    if not target_full.any() or target_full.all():
        return None
    box = _bbox_mask(target_full)
    assert box is not None
    r0, c0, r1, c1 = box
    target = target_full[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "target": target,
        "fg": (next(iter(top_fg)), next(iter(bottom_fg))),
    }


def _find_layout(g: np.ndarray):
    bg = _most_common(g)
    h, w = g.shape
    candidates = []

    for c in range(1, w - 1):
        if len(set(map(int, g[:, c]))) == 1 and int(g[0, c]) != bg:
            for panel_area, sprite_area in ((g[:, :c], g[:, c + 1 :]), (g[:, c + 1 :], g[:, :c])):
                parsed = _parse_panels(panel_area, bg)
                if parsed is not None and np.any(sprite_area != bg):
                    candidates.append((parsed, sprite_area, bg))

    for r in range(1, h - 1):
        if len(set(map(int, g[r, :]))) == 1 and int(g[r, 0]) != bg:
            for panel_area, sprite_area in ((g[:r, :], g[r + 1 :, :]), (g[r + 1 :, :], g[:r, :])):
                parsed = _parse_panels(panel_area, bg)
                if parsed is not None and np.any(sprite_area != bg):
                    candidates.append((parsed, sprite_area, bg))

    assert len(candidates) == 1, "expected one separator and one stacked-panel side"
    return candidates[0]


def _sprites(area: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(set(map(int, np.unique(area))) - {bg}):
        for coords in _components(area == color):
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "key": _shape_key_from_coords(coords),
                }
            )
    return sprites


def _complement_matches(
    target_coords: np.ndarray, sprite_coords: np.ndarray
) -> list[dict[str, object]]:
    target = {(int(r), int(c)) for r, c in target_coords}
    sh = int(sprite_coords[:, 0].max() - sprite_coords[:, 0].min() + 1)
    sw = int(sprite_coords[:, 1].max() - sprite_coords[:, 1].min() + 1)
    th = int(target_coords[:, 0].max() + 1)
    tw = int(target_coords[:, 1].max() + 1)
    sprite_norm = [(int(r - sprite_coords[:, 0].min()), int(c - sprite_coords[:, 1].min())) for r, c in sprite_coords]

    matches = {}
    for dr in range(-sh, th + 1):
        for dc in range(-sw, tw + 1):
            moved = {(r + dr, c + dc) for r, c in sprite_norm}
            if target & moved:
                continue
            union = target | moved
            r0 = min(r for r, _c in union)
            c0 = min(c for _r, c in union)
            r1 = max(r for r, _c in union)
            c1 = max(c for _r, c in union)
            rect_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(union) != rect_area:
                continue
            target_out = tuple(sorted((r - r0, c - c0) for r, c in target))
            sprite_out = tuple(sorted((r - r0, c - c0) for r, c in moved))
            key = (r1 - r0 + 1, c1 - c0 + 1, target_out, sprite_out)
            matches[key] = {
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "target": target_out,
                "sprite": sprite_out,
            }
    return list(matches.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the sprite that complements the two-panel common background into a solid rectangle.

    Input structure:
    - A full separator line divides the grid into a stacked-panel side and a sprite side.
    - The stacked side contains two equal-size panels, one above the other; both use the
      same background color and each panel has exactly one foreground mask color.
    - The other side contains several separated monochrome sprites, including one decoy
      congruent to the common-background shape and one true complementary sprite.

    Rule:
    - Compare the two panels cell by cell and keep exactly the positions where both cells
      are background; crop this common-background mask to its bounding box.
    - Among the sprites, use no rotation or reflection; only translate each sprite and keep
      the unique one whose cells are disjoint from the cropped mask and whose union with it
      is a completely filled rectangle.
    - Return that rectangle, coloring the common-background cells red and preserving the
      complementary sprite's original color on the remaining cells.

    Key insight:
    - The needed piece is not drawn in the input: it is the intersection of the two panel
      backgrounds, and the matching sprite is the complementary shape rather than the
      same-shaped decoy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    parsed, sprite_area, bg = _find_layout(g)
    target_mask = parsed["target"]
    target_coords = np.argwhere(target_mask)
    target_key = _shape_key_from_coords(target_coords)

    sprites = _sprites(sprite_area, bg)
    assert len(sprites) >= 2, "expected at least the decoy and the complement"
    same_shape = [s for s in sprites if s["key"] == target_key]
    assert len(same_shape) == 1, "expected exactly one same-shaped decoy sprite"

    candidates = []
    for sprite in sprites:
        matches = _complement_matches(target_coords, sprite["coords"])
        for match in matches:
            candidates.append((sprite, match))
    assert len(candidates) == 1, "expected one complementary sprite placement"

    sprite, match = candidates[0]
    oh, ow = match["shape"]
    out = np.full((int(oh), int(ow)), TARGET_RED, dtype=int)
    for r, c in match["sprite"]:
        out[int(r), int(c)] = int(sprite["color"])
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(*start))
    frontier = {(r, c) for r, c in frontier if 0 <= r < h and 0 <= c < w}
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_target_mask() -> np.ndarray | None:
    for _ in range(2000):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        total = h * w
        lo = max(4, min(total - 4, total // 3))
        hi = min(total - 4, max(lo, (2 * total) // 3))
        if lo > hi:
            continue
        area = random.randint(lo, hi)
        mask = _grow_connected_mask(h, w, area)
        if int(mask.sum()) != area:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        comp = ~mask
        if not _is_connected(comp):
            continue
        if _shape_key_from_mask(mask) == _shape_key_from_mask(comp):
            continue
        coords = np.argwhere(mask)
        if _complement_matches(coords, coords):
            continue
        return mask
    return None


def _build_panels(target: np.ndarray, bg: int, top_color: int, bottom_color: int) -> tuple[np.ndarray, np.ndarray]:
    th, tw = target.shape
    pads = [random.randint(0, 2) for _ in range(4)]
    if random.random() < 0.45 and not any(pads):
        pads[random.randrange(4)] = 1
    pt, pb, pl, pr = pads
    ph = th + pt + pb
    pw = tw + pl + pr
    target_full = np.zeros((ph, pw), dtype=bool)
    target_full[pt : pt + th, pl : pl + tw] = target

    top = np.full((ph, pw), bg, dtype=int)
    bottom = np.full((ph, pw), bg, dtype=int)
    outside = [(int(r), int(c)) for r, c in np.argwhere(~target_full)]
    random.shuffle(outside)
    forced = ["top", "bottom", "both"]
    for i, (r, c) in enumerate(outside):
        mode = forced[i] if i < len(forced) else random.choice(forced)
        if mode in ("top", "both"):
            top[r, c] = top_color
        if mode in ("bottom", "both"):
            bottom[r, c] = bottom_color
    return top, bottom


def _random_sprite_coords(forbidden: set[tuple[tuple[int, int], ...]], target_coords: np.ndarray) -> np.ndarray | None:
    for _ in range(500):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        area = random.randint(2, min(8, h * w - 1))
        mask = _grow_connected_mask(h, w, area)
        mask = _crop_mask(mask)
        key = _shape_key_from_mask(mask)
        if key in forbidden:
            continue
        coords = np.argwhere(mask)
        if _complement_matches(target_coords, coords):
            continue
        forbidden.add(key)
        return coords
    return None


def _mutated_complement(
    target: np.ndarray,
    forbidden: set[tuple[tuple[int, int], ...]],
    target_coords: np.ndarray,
) -> np.ndarray | None:
    comp = ~target
    comp_cells = [(int(r), int(c)) for r, c in np.argwhere(comp)]
    target_cells = [(int(r), int(c)) for r, c in np.argwhere(target)]
    for _ in range(300):
        cells = set(comp_cells)
        remove = random.choice(comp_cells)
        cells.remove(remove)
        add_options = [
            (r, c)
            for r, c in target_cells
            if any((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        ]
        if not add_options:
            continue
        cells.add(random.choice(add_options))
        mask = np.zeros(target.shape, dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        mask = _crop_mask(mask)
        key = _shape_key_from_mask(mask)
        if key in forbidden:
            continue
        coords = np.argwhere(mask)
        if _complement_matches(target_coords, coords):
            continue
        forbidden.add(key)
        return coords
    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r = r0 + int(rr)
        c = c0 + int(cc)
        if occupied[max(0, r - 1) : min(occupied.shape[0], r + 2), max(0, c - 1) : min(occupied.shape[1], c + 2)].any():
            return False
    return 0 <= r0 and 0 <= c0 and r0 + h <= occupied.shape[0] and c0 + w <= occupied.shape[1]


def _scatter_sprites(sprites: list[dict], height: int, width: int, bg: int) -> np.ndarray | None:
    canvas = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(sprites)), key=lambda i: -len(sprites[i]["coords"]))
    for idx in order:
        sprite = sprites[idx]
        patch = _coords_to_patch(sprite["coords"], sprite["color"], bg)
        mask = patch != bg
        ph, pw = patch.shape
        positions = []
        r_values = range(1, max(1, height - ph))
        c_values = range(1, max(1, width - pw))
        for r in r_values:
            for c in c_values:
                if r + ph < height and c + pw < width and _can_place(occupied, mask, r, c):
                    positions.append((r, c))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        canvas[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw][mask] = True
    return canvas


def _gen_attempt() -> np.ndarray | None:
    target = _sample_target_mask()
    if target is None:
        return None
    comp = ~target
    target_coords = np.argwhere(target)
    comp_coords = np.argwhere(comp)

    palette = [1, 3, 4, 6, 7, 8, 9]
    random.shuffle(palette)
    top_color, bottom_color = palette[:2]
    top, bottom = _build_panels(target, BACKGROUND, top_color, bottom_color)
    panel_h, panel_w = top.shape
    grid_h = panel_h * 2
    if grid_h > 30 or panel_w + 1 >= 30:
        return None

    available_sprite_colors = palette[2:]
    distractor_count = random.randint(1, min(3, len(available_sprite_colors) - 2))
    sprite_count = 2 + distractor_count
    sprite_colors = available_sprite_colors[:sprite_count]

    forbidden = {_shape_key_from_mask(target), _shape_key_from_mask(comp)}
    sprites = [
        {"role": "answer", "coords": comp_coords, "color": sprite_colors[0]},
        {"role": "decoy", "coords": target_coords, "color": sprite_colors[1]},
    ]
    near = _mutated_complement(target, forbidden, target_coords)
    if near is not None and distractor_count:
        sprites.append({"role": "near", "coords": near, "color": sprite_colors[len(sprites)]})
    while len(sprites) < sprite_count:
        coords = _random_sprite_coords(forbidden, target_coords)
        if coords is None:
            return None
        sprites.append({"role": "distractor", "coords": coords, "color": sprite_colors[len(sprites)]})
    random.shuffle(sprites)

    max_sprite_width = 30 - panel_w - 1
    min_sprite_width = max(10, max(int(s["coords"][:, 1].max() - s["coords"][:, 1].min() + 1) for s in sprites) + 4)
    if min_sprite_width > max_sprite_width:
        return None
    widths = list(range(min_sprite_width, max_sprite_width + 1))
    random.shuffle(widths)
    sprite_area = None
    for sprite_w in widths[:8]:
        sprite_area = _scatter_sprites(sprites, grid_h, sprite_w, BACKGROUND)
        if sprite_area is not None:
            break
    if sprite_area is None:
        return None

    stacked = np.vstack([top, bottom])
    if random.choice((True, False)):
        grid = np.full((grid_h, panel_w + 1 + sprite_area.shape[1]), BACKGROUND, dtype=int)
        grid[:, :panel_w] = stacked
        grid[:, panel_w] = SEPARATOR
        grid[:, panel_w + 1 :] = sprite_area
    else:
        grid = np.full((grid_h, panel_w + 1 + sprite_area.shape[1]), BACKGROUND, dtype=int)
        grid[:, : sprite_area.shape[1]] = sprite_area
        grid[:, sprite_area.shape[1]] = SEPARATOR
        grid[:, sprite_area.shape[1] + 1 :] = stacked

    answer_color = next(s["color"] for s in sprites if s["role"] == "answer")
    expected = np.full(target.shape, TARGET_RED, dtype=int)
    expected[comp] = int(answer_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(_sprites(sprite_area, BACKGROUND)) != len(sprites):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
