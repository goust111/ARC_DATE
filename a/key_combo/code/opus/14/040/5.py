from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TERRAIN = 5
ANCHOR = 8
NON_ANCHOR_COLORS = (1, 2, 3, 4, 6, 7, 9)

COUNT_CHOICES = (4, 5)
MIN_GAP = 2
GEN_ATTEMPTS = 600


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= {int(v) for v in exclude}
    return sorted(colors)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    dst_slice = dst[top : top + h, left : left + w]
    keep = patch != int(transparent)
    dst_slice[keep] = patch[keep]


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _mirror_mask(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1].copy()


def _terrain_info(grid: np.ndarray) -> tuple[int, np.ndarray]:
    h, w = grid.shape
    starts = np.empty(w, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == TERRAIN)
        assert rows.size > 0, "each column must contain a grey terrain suffix"
        start = int(rows[0])
        assert np.all(grid[start:, c] == TERRAIN), "terrain must be a contiguous suffix"
        starts[c] = start
    terrain_top = int(starts.min())
    assert terrain_top < h - 1, "terrain cannot be empty"
    return terrain_top, starts


def _extract_sprites(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    colors = _colors_present(grid, exclude={BG, TERRAIN})
    assert ANCHOR in colors, "expected one cyan anchor sprite"

    sprites: list[dict] = []
    ref_mask: np.ndarray | None = None
    for color in colors:
        comps = _component_coords(grid == color)
        assert len(comps) == 1, "each sprite color must form one 4-connected component"
        coords = comps[0]
        bbox = _bbox_from_coords(coords)
        mask = _crop_bbox(grid == color, bbox)
        if ref_mask is None:
            ref_mask = mask
        else:
            assert np.array_equal(mask, ref_mask), "all sprites must share the same normalized mask"
        sprites.append({"color": int(color), "bbox": bbox, "mask": mask})

    assert ref_mask is not None and not np.array_equal(ref_mask, _mirror_mask(ref_mask)), (
        "sprite mask must be horizontally asymmetric"
    )
    return sprites, ref_mask


def _render_transformed(
    shape: tuple[int, int],
    mask: np.ndarray,
    colors: list[int],
    lefts: list[int],
    tops: list[int],
    terrain_top: int,
    starts: np.ndarray,
    *,
    align: bool,
    mirror_mode: str,
    mirror_anchor: bool = False,
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for c, start in enumerate(starts):
        out[int(start) :, c] = TERRAIN

    mid_offset = (mask.shape[1] - 1) // 2
    for i, color in enumerate(colors):
        top = tops[i]
        if align:
            top = tops[colors.index(ANCHOR)]
        if mirror_mode == "center":
            ref_col = lefts[i] + mid_offset
            should_mirror = int(starts[ref_col] - terrain_top) % 2 == 1
        elif mirror_mode == "left":
            ref_col = lefts[i]
            should_mirror = int(starts[ref_col] - terrain_top) % 2 == 1
        elif mirror_mode == "right":
            ref_col = lefts[i] + mask.shape[1] - 1
            should_mirror = int(starts[ref_col] - terrain_top) % 2 == 1
        elif mirror_mode == "none":
            should_mirror = False
        elif mirror_mode == "all":
            should_mirror = color != ANCHOR or mirror_anchor
        else:
            raise ValueError(f"unknown mirror mode: {mirror_mode}")

        if color == ANCHOR and not mirror_anchor:
            should_mirror = False
        use_mask = _mirror_mask(mask) if should_mirror else mask
        _paste(out, _patch_from_mask(use_mask, color), top, lefts[i], transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vertically align repeated sprites to the cyan anchor, then mirror selected sprites by terrain-depth parity.

    Input structure:
    - A black background contains 4-5 monochrome 4-connected sprites in the upper region.
    - All sprites share the same normalized mask, and the cyan sprite is the vertical alignment anchor.
    - The lower region is a grey terrain profile: in every column, black cells are followed by a contiguous grey suffix.

    Rule:
    - Find the cyan sprite's top row and move every sprite so its top row matches that anchor row while keeping its original left column.
    - For each aligned sprite, compute the bbox center column as floor((left + right) / 2).
    - Read the terrain depth at that column as the number of black rows above the grey suffix within the terrain profile.
    - Mirror a non-cyan sprite horizontally inside its own bbox iff that depth is odd; keep it unchanged iff the depth is even.
    - The cyan sprite never mirrors, and the grey terrain stays unchanged.

    Key insight:
    - The terrain encodes one parity bit per sprite at its aligned center column, and that hidden bit decides whether the aligned sprite flips.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    terrain_top, starts = _terrain_info(g)
    sprites, ref_mask = _extract_sprites(g)

    anchor_sprite = next(sprite for sprite in sprites if sprite["color"] == ANCHOR)
    anchor_top = int(anchor_sprite["bbox"][0])

    out = np.where(g == TERRAIN, TERRAIN, BG)
    mid_offset = (ref_mask.shape[1] - 1) // 2
    for sprite in sprites:
        color = int(sprite["color"])
        _r0, c0, _r1, _c1 = sprite["bbox"]
        center_col = int(c0) + mid_offset
        depth = int(starts[center_col] - terrain_top)
        use_mask = ref_mask
        if color != ANCHOR and depth % 2 == 1:
            use_mask = _mirror_mask(ref_mask)
        _paste(out, _patch_from_mask(use_mask, color), anchor_top, int(c0), transparent=BG)
    return out


def _random_connected_sprite_mask() -> np.ndarray:
    for _ in range(300):
        area = random.randint(4, 8)
        raw = np.zeros((5, 5), dtype=bool)
        start = (random.randrange(5), random.randrange(5))
        cells = {start}
        raw[start] = True
        while len(cells) < area:
            rr, cc = random.choice(tuple(cells))
            nbrs = []
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                continue
            pick = random.choice(nbrs)
            cells.add(pick)
            raw[pick] = True

        coords = np.argwhere(raw)
        bbox = _bbox_from_coords(coords)
        mask = _crop_bbox(raw, bbox)
        h, w = mask.shape
        if not (3 <= h <= 5 and 3 <= w <= 5):
            continue
        if np.array_equal(mask, _mirror_mask(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric connected sprite mask")


def _sample_lefts(width: int, count: int, sprite_w: int) -> list[int]:
    base = count * sprite_w + (count - 1) * MIN_GAP
    slack = width - base
    extras = np.random.multinomial(slack, np.full(count + 1, 1.0 / (count + 1)))
    lefts: list[int] = []
    cur = int(extras[0])
    for i in range(count):
        lefts.append(cur)
        cur += sprite_w + MIN_GAP + int(extras[i + 1])
    return lefts


def _random_depths(width: int, terrain_h: int) -> np.ndarray:
    depths = np.empty(width, dtype=int)
    depths[0] = random.randrange(terrain_h)
    for c in range(1, width):
        step = random.choices((-2, -1, 0, 1, 2), weights=(1, 3, 4, 3, 1), k=1)[0]
        depths[c] = min(max(int(depths[c - 1]) + step, 0), terrain_h - 1)
    depths -= int(depths.min())
    if random.random() < 0.5:
        depths = depths[::-1].copy()
    return depths


def _choose_anchor_index(lefts: list[int], depths: np.ndarray, sprite_w: int) -> int | None:
    mid_offset = (sprite_w - 1) // 2
    parities = [int(depths[left + mid_offset] % 2) for left in lefts]
    want_odd = random.random() < 0.5
    for desired in (1 if want_odd else 0, 0 if want_odd else 1):
        candidates = []
        for idx, parity in enumerate(parities):
            if parity != desired:
                continue
            remain = [parities[j] for j in range(len(parities)) if j != idx]
            if 0 in remain and 1 in remain:
                candidates.append(idx)
        if candidates:
            return random.choice(candidates)
    return None


def _sample_tops(count: int, max_top: int, anchor_top: int, anchor_idx: int) -> list[int] | None:
    choices = list(range(max_top + 1))
    if anchor_top not in choices:
        return None
    choices.remove(anchor_top)
    if len(choices) < count - 1:
        return None

    picked: list[int] = []
    above = [v for v in choices if v < anchor_top]
    below = [v for v in choices if v > anchor_top]
    if above:
        picked.append(random.choice(above))
    if below:
        cand = [v for v in below if v not in picked]
        if cand:
            picked.append(random.choice(cand))

    remaining = [v for v in choices if v not in picked]
    need = count - 1 - len(picked)
    if need > len(remaining):
        return None
    picked.extend(random.sample(remaining, need))
    random.shuffle(picked)

    tops = [anchor_top] * count
    k = 0
    for i in range(count):
        if i == anchor_idx:
            continue
        tops[i] = int(picked[k])
        k += 1
    return tops


def _assign_colors(count: int, anchor_idx: int) -> list[int]:
    chosen = random.sample(NON_ANCHOR_COLORS, count - 1)
    random.shuffle(chosen)
    colors = [ANCHOR] * count
    k = 0
    for i in range(count):
        if i == anchor_idx:
            continue
        colors[i] = int(chosen[k])
        k += 1
    return colors


def _build_grid(
    mask: np.ndarray,
    lefts: list[int],
    tops: list[int],
    colors: list[int],
    depths: np.ndarray,
    terrain_top: int,
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for c, depth in enumerate(depths):
        start = terrain_top + int(depth)
        grid[start:, c] = TERRAIN
    for left, top, color in zip(lefts, tops, colors):
        _paste(grid, _patch_from_mask(mask, color), int(top), int(left), transparent=BG)
    return grid


def _expected_output(mask: np.ndarray, lefts: list[int], tops: list[int], colors: list[int], depths: np.ndarray, terrain_top: int, shape: tuple[int, int]) -> np.ndarray:
    starts = terrain_top + depths
    return _render_transformed(
        shape,
        mask,
        colors,
        lefts,
        tops,
        terrain_top,
        starts,
        align=True,
        mirror_mode="center",
        mirror_anchor=False,
    )


def _validate_semantics(grid: np.ndarray, meta: dict) -> bool:
    mask = meta["mask"]
    lefts = meta["lefts"]
    tops = meta["tops"]
    colors = meta["colors"]
    depths = meta["depths"]
    terrain_top = meta["terrain_top"]
    shape = meta["shape"]
    anchor_idx = meta["anchor_idx"]

    if not (4 <= int(mask.sum()) <= 8):
        return False
    if np.array_equal(mask, _mirror_mask(mask)):
        return False
    if grid.shape != shape or max(shape) > 30:
        return False
    if not np.all(depths >= 0) or int(depths.min()) != 0:
        return False

    centers = [left + (mask.shape[1] - 1) // 2 for left in lefts]
    parities = [int(depths[c] % 2) for c in centers]
    non_anchor_parities = [parities[i] for i in range(len(parities)) if i != anchor_idx]
    if 0 not in non_anchor_parities or 1 not in non_anchor_parities:
        return False
    if not any(tops[i] != tops[anchor_idx] for i in range(len(tops)) if i != anchor_idx):
        return False

    ref_sensitive = False
    for i in range(len(lefts)):
        if i == anchor_idx:
            continue
        left_parity = int(depths[lefts[i]] % 2)
        center_parity = int(depths[centers[i]] % 2)
        right_parity = int(depths[lefts[i] + mask.shape[1] - 1] % 2)
        if center_parity != left_parity or center_parity != right_parity:
            ref_sensitive = True
            break
    if not ref_sensitive:
        return False

    expected = _expected_output(mask, lefts, tops, colors, depths, terrain_top, shape)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    starts = terrain_top + depths
    no_align = _render_transformed(
        shape, mask, colors, lefts, tops, terrain_top, starts, align=False, mirror_mode="center"
    )
    if np.array_equal(no_align, expected):
        return False

    no_mirror = _render_transformed(
        shape, mask, colors, lefts, tops, terrain_top, starts, align=True, mirror_mode="none"
    )
    if np.array_equal(no_mirror, expected):
        return False

    mirror_all_non_anchor = _render_transformed(
        shape, mask, colors, lefts, tops, terrain_top, starts, align=True, mirror_mode="all"
    )
    if np.array_equal(mirror_all_non_anchor, expected):
        return False

    left_rule = _render_transformed(
        shape, mask, colors, lefts, tops, terrain_top, starts, align=True, mirror_mode="left"
    )
    right_rule = _render_transformed(
        shape, mask, colors, lefts, tops, terrain_top, starts, align=True, mirror_mode="right"
    )
    if np.array_equal(left_rule, expected) and np.array_equal(right_rule, expected):
        return False

    anchor_center = centers[anchor_idx]
    if int(depths[anchor_center] % 2) == 1:
        mirror_anchor = _render_transformed(
            shape,
            mask,
            colors,
            lefts,
            tops,
            terrain_top,
            starts,
            align=True,
            mirror_mode="center",
            mirror_anchor=True,
        )
        if np.array_equal(mirror_anchor, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    mask = _random_connected_sprite_mask()
    sprite_h, sprite_w = mask.shape

    feasible_counts = [count for count in COUNT_CHOICES if count * sprite_w + (count - 1) * MIN_GAP <= 30]
    if not feasible_counts:
        return None
    count = random.choice(feasible_counts)

    min_width = count * sprite_w + (count - 1) * MIN_GAP
    width = random.randint(max(18, min_width), min(30, min_width + 7))
    lefts = _sample_lefts(width, count, sprite_w)

    terrain_h = random.randint(6, 8)
    min_height = max(18, terrain_h + sprite_h + count + 1)
    if min_height > 26:
        return None
    height = random.randint(min_height, 26)
    terrain_top = height - terrain_h

    max_top = terrain_top - sprite_h - 1
    if max_top < count - 1:
        return None

    depths = _random_depths(width, terrain_h)
    if int(depths.max()) == 0:
        return None

    anchor_idx = _choose_anchor_index(lefts, depths, sprite_w)
    if anchor_idx is None:
        return None

    if max_top >= 2:
        anchor_top = random.randint(1, max_top - 1)
    else:
        anchor_top = random.randint(0, max_top)

    tops = _sample_tops(count, max_top, anchor_top, anchor_idx)
    if tops is None:
        return None

    colors = _assign_colors(count, anchor_idx)
    shape = (height, width)
    grid = _build_grid(mask, lefts, tops, colors, depths, terrain_top, shape)
    meta = {
        "mask": mask,
        "lefts": lefts,
        "tops": tops,
        "colors": colors,
        "depths": depths,
        "terrain_top": terrain_top,
        "shape": shape,
        "anchor_idx": anchor_idx,
    }
    return grid if _validate_semantics(grid, meta) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
