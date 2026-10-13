import random
from collections import Counter

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = DIR4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def get_bg(grid: np.ndarray) -> int:
    """Shared ARC convention: background is the most common color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    parts: list[np.ndarray] = []

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
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)

    if monochromatic:
        colors = sorted(set(map(int, g.ravel())) - {bg})
        parts: list[dict] = []
        for color in colors:
            for coords in components_mask(g == color, connectivity=connectivity):
                parts.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return parts

    parts = []
    for coords in components_mask(g != bg, connectivity=connectivity):
        parts.append({"color": None, "coords": coords, "bbox": bbox_coords(coords), "area": int(coords.shape[0])})
    return parts


def _mask_to_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords.astype(int)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _reflect_coords(coords: np.ndarray, axis_c: int) -> np.ndarray:
    reflected = coords.copy()
    reflected[:, 1] = 2 * int(axis_c) - reflected[:, 1]
    return reflected


def _drop_distance(coords: np.ndarray, occupied: np.ndarray, height: int) -> int:
    drop = 0
    while True:
        trial = coords.copy()
        trial[:, 0] += drop + 1
        if int(trial[:, 0].max()) >= int(height):
            return drop
        if np.any(occupied[trial[:, 0], trial[:, 1]]):
            return drop
        drop += 1


def _settle_reason(coords: np.ndarray, occupied: np.ndarray, height: int, drop: int) -> str:
    trial = coords.copy()
    trial[:, 0] += int(drop) + 1
    if int(trial[:, 0].max()) >= int(height):
        return "floor"
    return "collision"


def _pair_sprites_and_markers(grid: np.ndarray) -> tuple[int, list[tuple[dict, dict]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    sprites = [comp for comp in components if comp["area"] > 1]
    markers = [comp for comp in components if comp["area"] == 1]
    assert 1 <= len(sprites) <= 2, "expected one or two sprites"
    assert len(markers) == len(sprites), "expected one marker per sprite"

    sprite_colors = {int(comp["color"]) for comp in sprites}
    marker_colors = [int(comp["color"]) for comp in markers]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be unique"
    assert sprite_colors.isdisjoint(marker_colors), "markers must use distinct colors"

    paired: list[tuple[dict, dict]] = []
    used_markers: set[int] = set()
    for sprite in sorted(sprites, key=lambda comp: (comp["bbox"][0], comp["bbox"][1])):
        r0, c0, r1, c1 = sprite["bbox"]
        mask = _component_mask(sprite)
        assert not np.array_equal(mask, mask[:, ::-1]), "sprite must be left-right asymmetric"

        hits: list[tuple[int, dict]] = []
        for idx, marker in enumerate(markers):
            mr, mc = map(int, marker["coords"][0])
            if r0 <= mr <= r1:
                hits.append((idx, marker))
                assert mc < c0 or mc > c1, "marker must lie to the left or right of its sprite"
        assert len(hits) == 1, "each sprite row band must contain exactly one marker"
        idx, marker = hits[0]
        assert idx not in used_markers, "markers must pair uniquely"
        used_markers.add(idx)
        paired.append((sprite, marker))

    assert len(used_markers) == len(markers), "every marker must be paired"
    return bg, paired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each sprite across its marker column, recolor the mirrored copy, and let that copy fall.

    Input structure:
    - A dominant background contains one or two 4-connected monochrome sprites.
    - Each sprite's row band contains exactly one singleton marker pixel of a unique color, positioned strictly to that sprite's left or right.
    - Sprites are left-right asymmetric, and the mirrored starting copies stay inside the canvas without overlapping existing foreground.

    Rule:
    - Keep every original sprite unchanged, but delete all marker pixels.
    - For each sprite, use the marker's whole column as a vertical mirror axis and reflect the sprite across that column.
    - Recolor the reflected copy to the marker color, then drop that rigid reflected copy straight downward until the next step would hit the bottom edge or occupied foreground.

    Key insight:
    - The visible marker identifies the mirror axis column, but the reflected sprite never appears at its mirrored starting position; only the final gravity-settled colored copy is shown.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, pairs = _pair_sprites_and_markers(g)
    out = g.copy()
    for _sprite, marker in pairs:
        mr, mc = map(int, marker["coords"][0])
        out[mr, mc] = bg

    for sprite, marker in pairs:
        axis_c = int(marker["coords"][0, 1])
        marker_color = int(marker["color"])
        reflected = _reflect_coords(sprite["coords"], axis_c)
        assert reflected[:, 1].min() >= 0 and reflected[:, 1].max() < g.shape[1], "mirrored copy must stay in bounds"
        assert np.all(g[reflected[:, 0], reflected[:, 1]] == bg), "mirrored start must be empty"

        occupied = out != bg
        drop = _drop_distance(reflected, occupied, g.shape[0])
        landed = reflected.copy()
        landed[:, 0] += drop
        assert np.all(out[landed[:, 0], landed[:, 1]] == bg), "settled mirrored copy must not overlap"
        out[landed[:, 0], landed[:, 1]] = marker_color

    return out


GEN_MAX_ATTEMPTS = 500
GEN_BG = 0
GEN_COLORS = tuple(range(1, 10))
GEN_SPRITE_H_RANGE = (2, 4)
GEN_SPRITE_W_RANGE = (3, 5)
GEN_GAP_RANGE = (1, 3)
GEN_ROW_SEP_RANGE = (4, 8)
GEN_PAD_LIMIT = 8


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return np.zeros((h, w), dtype=bool)
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_sprite_mask() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(*GEN_SPRITE_H_RANGE)
        w = random.randint(*GEN_SPRITE_W_RANGE)
        lo = max(h + w - 2, 4)
        hi = h * w - 1
        if lo > hi:
            continue
        mask = _grow_connected_mask(h, w, random.randint(lo, hi))
        if mask.sum() <= 1:
            continue
        if not _touches_all_sides(mask):
            continue
        if np.array_equal(mask, mask[:, ::-1]):
            continue
        if np.all(mask):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        return mask
    return None


def _sample_margins(base: int, limit: int = 30) -> tuple[int, int] | None:
    cap = int(limit) - int(base)
    if cap < 2:
        return None
    total = random.randint(2, min(GEN_PAD_LIMIT, cap))
    a = random.randint(1, total - 1)
    return a, total - a


def _render_scene(objects: list[tuple[np.ndarray, int]], base_h: int, base_w: int) -> np.ndarray | None:
    row_pad = _sample_margins(base_h, 30)
    col_pad = _sample_margins(base_w, 30)
    if row_pad is None or col_pad is None:
        return None
    top, bottom = row_pad
    left, right = col_pad
    h = base_h + top + bottom
    w = base_w + left + right
    grid = np.full((h, w), GEN_BG, dtype=int)
    for patch, color in objects:
        coords = patch.copy()
        coords[:, 0] += top
        coords[:, 1] += left
        grid[coords[:, 0], coords[:, 1]] = int(color)
    if random.random() < 0.5:
        grid = grid[:, ::-1].copy()
    return grid


def _single_scene() -> np.ndarray | None:
    mask = _gen_sprite_mask()
    if mask is None:
        return None

    sprite_color, marker_color = random.sample(GEN_COLORS, 2)
    h1, w1 = mask.shape
    gap = random.randint(*GEN_GAP_RANGE)
    sprite_r0 = 0
    sprite_c0 = w1 - 1 + 2 * gap
    axis_c = sprite_c0 - gap
    marker_r = sprite_r0 + random.randrange(h1)

    sprite = _mask_to_coords(mask, sprite_r0, sprite_c0)
    marker = np.array([[marker_r, axis_c]], dtype=int)
    reflected = _reflect_coords(sprite, axis_c)

    base_h = max(int(sprite[:, 0].max()), int(reflected[:, 0].max()) + 1) + 1
    base_w = max(int(sprite[:, 1].max()), int(marker[:, 1].max()), int(reflected[:, 1].max())) + 1

    return _render_scene([(sprite, sprite_color), (marker, marker_color)], base_h, base_w)


def _double_scene() -> np.ndarray | None:
    top_mask = _gen_sprite_mask()
    bottom_mask = _gen_sprite_mask()
    if top_mask is None or bottom_mask is None:
        return None

    top_color, bottom_color, top_marker_color, bottom_marker_color = random.sample(GEN_COLORS, 4)

    h1, w1 = top_mask.shape
    h2, w2 = bottom_mask.shape
    top_gap = random.randint(*GEN_GAP_RANGE)
    bottom_gap = random.randint(*GEN_GAP_RANGE)

    top_r0 = 0
    top_c0 = w1 - 1 + 2 * top_gap
    top_axis_c = top_c0 - top_gap
    top_sprite = _mask_to_coords(top_mask, top_r0, top_c0)
    top_marker = np.array([[top_r0 + random.randrange(h1), top_axis_c]], dtype=int)
    top_reflected = _reflect_coords(top_sprite, top_axis_c)

    min_bottom_c0 = max(0, w1 - w2 - 2 * bottom_gap + 1)
    max_bottom_c0 = w1 - 2
    if min_bottom_c0 > max_bottom_c0:
        return None
    bottom_c0 = random.randint(min_bottom_c0, max_bottom_c0)
    bottom_r0 = h1 + random.randint(*GEN_ROW_SEP_RANGE)
    bottom_axis_c = bottom_c0 + w2 - 1 + bottom_gap
    bottom_sprite = _mask_to_coords(bottom_mask, bottom_r0, bottom_c0)
    bottom_marker = np.array([[bottom_r0 + random.randrange(h2), bottom_axis_c]], dtype=int)
    bottom_reflected = _reflect_coords(bottom_sprite, bottom_axis_c)

    base_w = max(
        int(top_sprite[:, 1].max()),
        int(top_marker[:, 1].max()),
        int(top_reflected[:, 1].max()),
        int(bottom_sprite[:, 1].max()),
        int(bottom_marker[:, 1].max()),
        int(bottom_reflected[:, 1].max()),
    ) + 1

    probe_h = bottom_r0 + h2 + h1 + 12
    occ_all = np.zeros((probe_h, base_w), dtype=bool)
    occ_bottom = np.zeros((probe_h, base_w), dtype=bool)
    occ_all[top_sprite[:, 0], top_sprite[:, 1]] = True
    occ_all[bottom_sprite[:, 0], bottom_sprite[:, 1]] = True
    occ_bottom[bottom_sprite[:, 0], bottom_sprite[:, 1]] = True
    top_drop = _drop_distance(top_reflected, occ_all, probe_h)
    if top_drop <= 0:
        return None
    if _settle_reason(top_reflected, occ_all, probe_h, top_drop) != "collision":
        return None
    trial = top_reflected.copy()
    trial[:, 0] += top_drop + 1
    if not np.any(occ_bottom[trial[:, 0], trial[:, 1]]):
        return None
    bbox_drop = int(bottom_sprite[:, 0].min()) - int(top_reflected[:, 0].max()) - 1
    if top_drop <= bbox_drop:
        return None
    if set(map(int, top_reflected[:, 1])) & set(map(int, bottom_reflected[:, 1])):
        return None

    base_h = max(
        int(bottom_sprite[:, 0].max()),
        int(top_reflected[:, 0].max()) + top_drop,
        int(bottom_reflected[:, 0].max()) + 1,
    ) + 1

    return _render_scene(
        [
            (top_sprite, top_color),
            (bottom_sprite, bottom_color),
            (top_marker, top_marker_color),
            (bottom_marker, bottom_marker_color),
        ],
        base_h,
        base_w,
    )


def _copy_semantics(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg, pairs = _pair_sprites_and_markers(g)
    occ_originals = g != bg
    for _sprite, marker in pairs:
        mr, mc = map(int, marker["coords"][0])
        occ_originals[mr, mc] = False

    copies: list[dict] = []
    for sprite, marker in pairs:
        axis_c = int(marker["coords"][0, 1])
        reflected = _reflect_coords(sprite["coords"], axis_c)
        assert reflected[:, 1].min() >= 0 and reflected[:, 1].max() < g.shape[1]
        assert np.all(g[reflected[:, 0], reflected[:, 1]] == bg)
        drop = _drop_distance(reflected, occ_originals, g.shape[0])
        reason = _settle_reason(reflected, occ_originals, g.shape[0], drop)
        copies.append(
            {
                "sprite": sprite,
                "marker": marker,
                "reflected": reflected,
                "drop": int(drop),
                "reason": reason,
            }
        )
    return copies


def _semantic_checks(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg, pairs = _pair_sprites_and_markers(g)
    copies = _copy_semantics(g)

    copy_cols = []
    for item in copies:
        sprite = item["sprite"]
        marker = item["marker"]
        reflected = item["reflected"]
        drop = item["drop"]

        if drop <= 0:
            return False
        if int(marker["color"]) in {int(sprite["color"]), bg}:
            return False
        copy_cols.append(set(map(int, reflected[:, 1])))

    for i in range(len(copy_cols)):
        for j in range(i + 1, len(copy_cols)):
            if copy_cols[i] & copy_cols[j]:
                return False

    if len(pairs) == 2:
        copies = sorted(copies, key=lambda item: item["sprite"]["bbox"][0])
        top_copy, bottom_copy = copies
        if top_copy["reason"] != "collision":
            return False
        if bottom_copy["reason"] != "floor":
            return False

        bottom_coords = pairs[1][0]["coords"]
        bottom_occ = np.zeros_like(g, dtype=bool)
        bottom_occ[bottom_coords[:, 0], bottom_coords[:, 1]] = True
        next_step = top_copy["reflected"].copy()
        next_step[:, 0] += top_copy["drop"] + 1
        if not np.any(bottom_occ[next_step[:, 0], next_step[:, 1]]):
            return False

        bbox_drop = int(bottom_coords[:, 0].min()) - int(top_copy["reflected"][:, 0].max()) - 1
        if top_copy["drop"] <= bbox_drop:
            return False

    out = main(g)
    if out.shape != g.shape:
        return False
    return True


def _attempt() -> np.ndarray | None:
    builders: list = []
    if random.random() < 0.8:
        builders.extend(
            [
                _double_scene,
                _double_scene,
                _double_scene,
                _double_scene,
                _double_scene,
                _double_scene,
                _single_scene,
                _single_scene,
            ]
        )
    else:
        builders.extend([_single_scene, _double_scene, _double_scene, _single_scene])

    for builder in builders:
        grid = builder()
        if grid is None:
            continue
        try:
            if _semantic_checks(grid):
                return grid
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-then-fall ARC input")
