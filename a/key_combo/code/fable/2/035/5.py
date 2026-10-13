from __future__ import annotations

from collections import Counter

import numpy as np


GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 500
SPRITE_RETRIES = 300
ALL_COLORS = tuple(range(10))
_GEN_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    sprites = []
    for coords in components_mask(grid != bg):
        bbox = bbox_coords(coords)
        crop = crop_bbox(grid, bbox)
        key_coords = np.argwhere(crop == GREY)
        assert key_coords.shape[0] == 1, "each sprite must contain exactly one grey key pixel"

        body_colors = sorted(set(map(int, crop[(crop != bg) & (crop != GREY)])))
        assert len(body_colors) == 1, "each sprite body must be monochrome"
        color = int(body_colors[0])
        allowed = {int(bg), GREY, color}
        assert set(map(int, np.unique(crop))).issubset(allowed)
        assert int(np.count_nonzero(crop != bg)) >= 2

        key_r, key_c = map(int, key_coords[0])
        solid = np.array(crop, copy=True)
        solid[key_r, key_c] = color
        r0, c0, _r1, _c1 = bbox
        sprites.append(
            {
                "bbox": bbox,
                "crop": crop,
                "solid": solid,
                "color": color,
                "key_r": key_r,
                "key_c": key_c,
                "top": int(r0),
                "left": int(c0),
            }
        )
    return sprites


def _render_output(
    sprites: list[dict[str, object]],
    bg: int,
    *,
    order_mode: str,
    flip_mode: str,
    grid_width: int | None = None,
) -> np.ndarray:
    assert sprites
    h, w = map(int, sprites[0]["solid"].shape)
    pieces = []
    for sprite in sprites:
        patch = np.array(sprite["solid"], copy=True)
        key_c = int(sprite["key_c"])
        if flip_mode == "local-half":
            flip = key_c >= w // 2
        elif flip_mode == "none":
            flip = False
        elif flip_mode == "global-half":
            assert grid_width is not None
            flip = int(sprite["left"]) + key_c >= int(grid_width) / 2.0
        else:
            raise ValueError(f"unknown flip mode: {flip_mode}")
        if flip:
            patch = np.fliplr(patch)
        pieces.append((sprite, patch))

    if order_mode == "key-row":
        pieces.sort(key=lambda item: int(item[0]["key_r"]))
    elif order_mode == "x":
        pieces.sort(key=lambda item: int(item[0]["left"]))
    elif order_mode == "global-row":
        pieces.sort(key=lambda item: int(item[0]["top"]) + int(item[0]["key_r"]))
    else:
        raise ValueError(f"unknown order mode: {order_mode}")

    out = np.full((h, w * len(pieces)), int(bg), dtype=int)
    for i, (_sprite, patch) in enumerate(pieces):
        paste(out, patch, 0, i * w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder cropped sprites by the grey key row inside each bbox, mirroring only the sprites whose key sits in the right half.

    Input structure:
    - A background grid contains several separated foreground sprites laid out horizontally.
    - Every sprite fits in the same even-width bounding box and is monochrome except for one grey key pixel inside that bbox.
    - Across the sprites, the key rows are all different and form exactly 0..n-1 in bbox-local coordinates.

    Rule:
    - Crop every sprite to its own bounding box.
    - If a sprite's grey key lies in the left half of that bbox, keep the crop as-is; if the key lies in the right half, mirror the whole crop left-right first.
    - Replace the grey key with the sprite's own body color.
    - Sort the crops by the key row from 0 upward and concatenate them from left to right.

    Key insight:
    - The input left-to-right arrangement is a decoy: the key row encodes output order, while the key half inside the sprite's own bbox encodes whether to mirror.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg != GREY, "grey must be reserved for the key pixels"

    sprites = _extract_sprites(g, bg)
    assert len(sprites) >= 2, "expected at least two sprites"
    shapes = {tuple(map(int, sprite["crop"].shape)) for sprite in sprites}
    assert len(shapes) == 1, "all sprite bounding boxes must have the same size"

    h, w = next(iter(shapes))
    assert w % 2 == 0, "sprite bbox width must be even"

    n = len(sprites)
    key_rows = sorted(int(sprite["key_r"]) for sprite in sprites)
    assert key_rows == list(range(n)), "key rows must be exactly 0..n-1"

    return _render_output(sprites, bg, order_mode="key-row", flip_mode="local-half")


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
    raise ValueError("line endpoints must share a row or column")


def _orth_path(
    start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator
) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        mid = (int(start[0]), int(end[1]))
    else:
        mid = (int(end[0]), int(start[1]))
    first = _line(start, mid)
    second = _line(mid, end)
    return first[:-1] + second


def _sample_sprite_mask(
    h: int, w: int, key_r: int, key_c: int, rng: np.random.Generator
) -> np.ndarray | None:
    key = (int(key_r), int(key_c))
    max_area = min(h * w - 2, int(np.ceil(h * w * 0.72)))
    if max_area < h + w - 1:
        return None

    for _ in range(SPRITE_RETRIES):
        anchors = [
            (0, int(rng.integers(0, w))),
            (h - 1, int(rng.integers(0, w))),
            (int(rng.integers(0, h)), 0),
            (int(rng.integers(0, h)), w - 1),
        ]
        cells = {key}
        for anchor in anchors:
            cells.update(_orth_path(key, anchor, rng))

        min_area = max(len(cells) + 1, h + w - 1)
        if min_area > max_area:
            continue
        target = int(rng.integers(min_area, max_area + 1))
        while len(cells) < target:
            frontier = []
            frontier_set = set()
            for rr, cc in cells:
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    nxt = (int(nr), int(nc))
                    if 0 <= nr < h and 0 <= nc < w and nxt not in cells and nxt not in frontier_set:
                        frontier.append(nxt)
                        frontier_set.add(nxt)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True

        if not bool(mask[key]):
            continue
        if int(mask.sum()) < min_area or int(mask.sum()) > max_area:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if int(mask[0].sum()) == 0 or int(mask[-1].sum()) == 0:
            continue
        if int(mask[:, 0].sum()) == 0 or int(mask[:, -1].sum()) == 0:
            continue
        return mask
    return None


def _build_sprite_crop(
    h: int,
    w: int,
    *,
    color: int,
    bg: int,
    key_r: int,
    flip_flag: bool,
    rng: np.random.Generator,
) -> np.ndarray | None:
    cols = range(w // 2, w) if bool(flip_flag) else range(0, w // 2)
    col_choices = [int(c) for c in cols]
    rng.shuffle(col_choices)
    for key_c in col_choices:
        mask = _sample_sprite_mask(h, w, key_r, key_c, rng)
        if mask is None:
            continue
        crop = np.full((h, w), int(bg), dtype=int)
        crop[mask] = int(color)
        crop[key_r, key_c] = GREY
        solid = np.array(crop, copy=True)
        solid[key_r, key_c] = int(color)
        if bool(flip_flag) and np.array_equal(solid, np.fliplr(solid)):
            continue
        return crop
    return None


def _sample_geometry(rng: np.random.Generator) -> tuple[int, int, int]:
    n = int(rng.integers(3, 6))
    width_choices = {3: (4, 6, 8), 4: (4, 6), 5: (4,)}
    w = int(rng.choice(width_choices[n]))
    h = int(rng.integers(n, min(8, n + 2) + 1))
    return n, h, w


def _sample_top_rows(n: int, h: int, rng: np.random.Generator) -> list[int] | None:
    max_top = min(8, 29 - h)
    if max_top < 1:
        return None
    for _ in range(200):
        tops = [int(rng.integers(1, max_top + 1)) for _ in range(n)]
        global_rows = [tops[i] + i for i in range(n)]
        if len(set(tops)) < 2:
            continue
        if len(set(global_rows)) < n:
            continue
        if sorted(range(n), key=lambda i: global_rows[i]) == list(range(n)):
            continue
        return tops
    return None


def _sample_left_cols(n: int, w: int, rng: np.random.Generator) -> list[int]:
    min_total = n * w + (n - 1) + 2
    assert min_total <= 30
    slack = 30 - min_total
    used_extra = int(rng.integers(0, slack + 1))
    parts = [1] + [1] * (n - 1) + [1]
    for _ in range(used_extra):
        parts[int(rng.integers(0, len(parts)))] += 1

    left_margin = parts[0]
    gaps = parts[1:-1]
    lefts = []
    col = int(left_margin)
    for i in range(n):
        lefts.append(col)
        col += w
        if i < n - 1:
            col += int(gaps[i])
    return lefts


def _manual_expected(specs: list[dict[str, object]], bg: int, h: int, w: int) -> np.ndarray:
    ordered = []
    for spec in specs:
        patch = np.array(spec["solid"], copy=True)
        if bool(spec["flip_flag"]):
            patch = np.fliplr(patch)
        ordered.append((int(spec["order"]), patch))
    ordered.sort(key=lambda item: item[0])
    out = np.full((h, w * len(ordered)), int(bg), dtype=int)
    for i, (_order, patch) in enumerate(ordered):
        paste(out, patch, 0, i * w)
    return out


def _validate_example(grid: np.ndarray, specs: list[dict[str, object]], bg: int, h: int, w: int) -> bool:
    try:
        parsed = _extract_sprites(grid, bg)
        if len(parsed) != len(specs):
            return False

        shapes = {tuple(map(int, sprite["crop"].shape)) for sprite in parsed}
        if shapes != {(h, w)}:
            return False

        key_rows = sorted(int(sprite["key_r"]) for sprite in parsed)
        if key_rows != list(range(len(parsed))):
            return False

        local_flips = [int(sprite["key_c"]) >= w // 2 for sprite in parsed]
        if not any(local_flips) or all(local_flips):
            return False

        x_order = [int(sprite["key_r"]) for sprite in sorted(parsed, key=lambda sprite: int(sprite["left"]))]
        if x_order == list(range(len(parsed))):
            return False

        transformed = [
            np.fliplr(np.array(sprite["solid"], copy=True)) if int(sprite["key_c"]) >= w // 2 else np.array(sprite["solid"], copy=True)
            for sprite in parsed
        ]
        if len({patch.tobytes() for patch in transformed}) != len(parsed):
            return False

        expected = _manual_expected(specs, bg, h, w)
        if not np.array_equal(main(grid), expected):
            return False

        parsed_expected = _render_output(parsed, bg, order_mode="key-row", flip_mode="local-half", grid_width=grid.shape[1])
        if not np.array_equal(parsed_expected, expected):
            return False

        if np.array_equal(expected, _render_output(parsed, bg, order_mode="x", flip_mode="local-half", grid_width=grid.shape[1])):
            return False
        if np.array_equal(expected, _render_output(parsed, bg, order_mode="key-row", flip_mode="none", grid_width=grid.shape[1])):
            return False
        if np.array_equal(expected, _render_output(parsed, bg, order_mode="global-row", flip_mode="local-half", grid_width=grid.shape[1])):
            return False
        if np.array_equal(expected, _render_output(parsed, bg, order_mode="key-row", flip_mode="global-half", grid_width=grid.shape[1])):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = _GEN_RNG
    n, h, w = _sample_geometry(rng)

    tops = _sample_top_rows(n, h, rng)
    if tops is None:
        return None

    base_lefts = _sample_left_cols(n, w, rng)
    x_perm = list(map(int, rng.permutation(n)))
    if x_perm == list(range(n)):
        return None

    bg_choices = [c for c in ALL_COLORS if c != GREY]
    bg = int(rng.choice(bg_choices))
    palette = [c for c in ALL_COLORS if c not in {bg, GREY}]
    rng.shuffle(palette)
    colors = [int(c) for c in palette[:n]]

    flip_flags = rng.integers(0, 2, size=n).astype(bool).tolist()
    if not any(flip_flags) or all(flip_flags):
        return None

    lefts_by_order = [0] * n
    for x_rank, order in enumerate(x_perm):
        lefts_by_order[int(order)] = int(base_lefts[x_rank])

    specs: list[dict[str, object]] = []
    for order in range(n):
        crop = _build_sprite_crop(
            h,
            w,
            color=colors[order],
            bg=bg,
            key_r=order,
            flip_flag=bool(flip_flags[order]),
            rng=rng,
        )
        if crop is None:
            return None
        solid = np.array(crop, copy=True)
        key_pos = np.argwhere(crop == GREY)
        key_r, key_c = map(int, key_pos[0])
        solid[key_r, key_c] = int(colors[order])
        specs.append(
            {
                "order": int(order),
                "color": int(colors[order]),
                "crop": crop,
                "solid": solid,
                "flip_flag": bool(flip_flags[order]),
                "top": int(tops[order]),
                "left": int(lefts_by_order[order]),
                "key_c": int(key_c),
            }
        )

    bottom_margin = int(rng.integers(1, 4))
    right_edge = max(int(spec["left"]) + w for spec in specs)
    width = right_edge + 1
    height = max(int(spec["top"]) + h for spec in specs) + bottom_margin
    if max(height, width) > 30:
        return None

    global_half = [int(spec["left"]) + int(spec["key_c"]) >= width / 2.0 for spec in specs]
    if global_half == [bool(spec["flip_flag"]) for spec in specs]:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    for spec in specs:
        paste(grid, np.asarray(spec["crop"], dtype=int), int(spec["top"]), int(spec["left"]))

    if not _validate_example(grid, specs, bg, h, w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-key sprite ordering input")
