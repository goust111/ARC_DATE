from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _marker_info(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int, int]:
    values = [int(grid[r, c]) for r, c in coords]
    color_counts: dict[int, int] = {}
    for value in values:
        color_counts[value] = color_counts.get(value, 0) + 1

    marker_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(marker_colors) == 1, "each sprite must contain exactly one singleton marker color"
    marker_color = marker_colors[0]
    for r, c in coords:
        if int(grid[r, c]) == marker_color:
            return marker_color, int(r), int(c)
    raise AssertionError("marker pixel missing")


def _quadrant(local_r: int, local_c: int, h: int, w: int) -> str:
    if h % 2 == 1:
        assert local_r != h // 2, "marker must not lie on the bbox middle row"
    if w % 2 == 1:
        assert local_c != w // 2, "marker must not lie on the bbox middle column"
    top = local_r < (h / 2.0)
    left = local_c < (w / 2.0)
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and not left:
        return "br"
    return "bl"


def _centered_origin(r0: int, c0: int, h: int, w: int, h2: int, w2: int) -> tuple[int, int]:
    center2_r = 2 * r0 + h - 1
    center2_c = 2 * c0 + w - 1
    return (
        int((center2_r - (h2 - 1)) // 2),
        int((center2_c - (w2 - 1)) // 2),
    )


def _paste_clipped(dst: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = sprite.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(dst.shape[0], top + h)
    c1 = min(dst.shape[1], left + w)
    if r0 >= r1 or c0 >= c1:
        return

    sr0 = r0 - top
    sc0 = c0 - left
    sr1 = sr0 + (r1 - r0)
    sc1 = sc0 + (c1 - c0)
    block = sprite[sr0:sr1, sc0:sc1]
    mask = block != bg
    target = dst[r0:r1, c0:c1]
    target[mask] = block[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore all rotated copies of one template to a shared standard orientation.

    Each foreground sprite contains a unique marker pixel. The marker's quadrant inside the
    sprite bbox encodes how far that sprite was rotated away from the standard orientation:
    top-left = 0°, top-right = 90° clockwise, bottom-right = 180°, bottom-left = 270° clockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    comps = _components(g != bg)
    assert comps, "expected at least one foreground sprite"

    quadrant_to_ccw = {"tl": 0, "tr": 1, "br": 2, "bl": 3}
    out = np.full_like(g, bg)

    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        _marker_color, mr, mc = _marker_info(g, coords)
        local_r = int(mr - r0)
        local_c = int(mc - c0)
        quadrant = _quadrant(local_r, local_c, sub.shape[0], sub.shape[1])
        restored = np.rot90(sub, quadrant_to_ccw[quadrant])
        nr0, nc0 = _centered_origin(r0, c0, sub.shape[0], sub.shape[1], restored.shape[0], restored.shape[1])
        _paste_clipped(out, restored, nr0, nc0, bg)

    return out


# --------------- generator ---------------

_BG = 0
_PALETTE = list(range(1, 10))
_GRID_RANGE = (16, 30)
_TEMPLATE_DIMS = [
    (4, 6),
    (6, 4),
    (4, 8),
    (8, 4),
    (5, 7),
    (7, 5),
    (5, 9),
    (9, 5),
    (6, 8),
    (8, 6),
]
_SPRITE_COUNT_RANGE = (3, 6)
_GEN_ATTEMPTS = 400


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(_components(mask)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _unique_rotations(mask: np.ndarray) -> bool:
    variants = [np.rot90(mask, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(variants[i], variants[j]):
                return False
    return True


def _top_left_marker_choices(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    row_stop = h // 2
    col_stop = w // 2
    return [(r, c) for r in range(row_stop) for c in range(col_stop) if mask[r, c]]


def _random_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(300):
        mask = np.ones((h, w), dtype=bool)

        corner = random.choice(("tl", "tr", "bl", "br"))
        dh = random.randint(1, max(1, h // 2))
        dw = random.randint(1, max(1, w // 2))
        if corner == "tl":
            mask[:dh, :dw] = False
        elif corner == "tr":
            mask[:dh, w - dw :] = False
        elif corner == "bl":
            mask[h - dh :, :dw] = False
        else:
            mask[h - dh :, w - dw :] = False

        for _ in range(random.randint(0, 2)):
            mode = random.choice(("edge", "block", "cell"))
            trial = mask.copy()
            if mode == "edge":
                side = random.choice(("top", "bottom", "left", "right"))
                if side in ("top", "bottom"):
                    depth = random.randint(1, max(1, h // 3))
                    width = random.randint(1, max(1, w - 2))
                    start = random.randint(0, w - width)
                    rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                    trial[rows, start : start + width] = False
                else:
                    depth = random.randint(1, max(1, w // 3))
                    height = random.randint(1, max(1, h - 2))
                    start = random.randint(0, h - height)
                    cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                    trial[start : start + height, cols] = False
            elif mode == "block":
                bh = random.randint(1, max(1, h // 3))
                bw = random.randint(1, max(1, w // 3))
                rr = random.randint(0, h - bh)
                cc = random.randint(0, w - bw)
                trial[rr : rr + bh, cc : cc + bw] = False
            else:
                removals = random.randint(1, max(1, (h * w) // 8))
                cells = [(r, c) for r in range(h) for c in range(w) if trial[r, c]]
                if len(cells) <= removals:
                    continue
                for r, c in random.sample(cells, removals):
                    trial[r, c] = False

            mask = trial

        if int(mask.sum()) < max(8, (h * w) // 2):
            continue
        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if not _unique_rotations(mask):
            continue
        if not _top_left_marker_choices(mask):
            continue
        return mask
    return None


def _make_standard_sprite(mask: np.ndarray, body: int, marker: int) -> np.ndarray | None:
    choices = _top_left_marker_choices(mask)
    if not choices:
        return None
    r, c = random.choice(choices)
    sprite = np.full(mask.shape, _BG, dtype=int)
    sprite[mask] = int(body)
    sprite[r, c] = int(marker)
    return sprite


def _render_cells(sprite: np.ndarray, top: int, left: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(sprite != _BG)
    rows = coords[:, 0] + int(top)
    cols = coords[:, 1] + int(left)
    vals = sprite[coords[:, 0], coords[:, 1]]
    return rows.astype(int), cols.astype(int), vals.astype(int)


def _render_cells_clipped(
    sprite: np.ndarray,
    top: int,
    left: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, vals = _render_cells(sprite, top, left)
    keep = (
        (rows >= 0)
        & (rows < shape[0])
        & (cols >= 0)
        & (cols < shape[1])
    )
    return rows[keep], cols[keep], vals[keep]


def _positions_for_sprite(
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    prefer_clip: bool,
    standard_shape: tuple[int, int],
    orientation: int,
) -> list[tuple[int, int]]:
    h, w = sprite_shape
    H, W = grid_shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    if not prefer_clip or orientation not in (1, 3):
        return positions

    sh, sw = standard_shape
    preferred: list[tuple[int, int]] = []
    if sw > sh:
        edge_cols = [0, W - w]
        for c in edge_cols:
            for r in range(H - h + 1):
                preferred.append((r, c))
    else:
        edge_rows = [0, H - h]
        for r in edge_rows:
            for c in range(W - w + 1):
                preferred.append((r, c))
    random.shuffle(preferred)

    preferred_set = set(preferred)
    return preferred + [pos for pos in positions if pos not in preferred_set]


def _choose_orientations(count: int) -> list[int]:
    distinct = random.randint(2, min(4, count))
    base = [0] + random.sample([1, 2, 3], distinct - 1)
    out = base[:]
    while len(out) < count:
        out.append(random.choice(base))
    random.shuffle(out)
    return out


def _place_sprites(
    standard_sprite: np.ndarray,
    orientations: list[int],
    shape: tuple[int, int],
    prefer_clip: bool,
) -> list[dict] | None:
    H, W = shape
    input_forbidden = np.zeros((H, W), dtype=bool)
    output_forbidden = np.zeros((H, W), dtype=bool)

    clip_index = None
    if prefer_clip:
        candidates = [i for i, ori in enumerate(orientations) if ori in (1, 3)]
        if candidates:
            clip_index = random.choice(candidates)

    order = list(range(len(orientations)))
    if clip_index is not None:
        order.remove(clip_index)
        random.shuffle(order)
        order = [clip_index] + order
    else:
        random.shuffle(order)

    placed_by_index: dict[int, dict] = {}
    for idx in order:
        orientation = int(orientations[idx])
        sprite = np.rot90(standard_sprite, -orientation)
        prefer_this_clip = idx == clip_index

        placed = None
        for top, left in _positions_for_sprite(
            (H, W),
            sprite.shape,
            prefer_this_clip,
            standard_sprite.shape,
            orientation,
        ):
            in_rows, in_cols, _ = _render_cells(sprite, top, left)
            if bool(input_forbidden[in_rows, in_cols].any()):
                continue

            out_top, out_left = _centered_origin(
                int(top),
                int(left),
                sprite.shape[0],
                sprite.shape[1],
                standard_sprite.shape[0],
                standard_sprite.shape[1],
            )
            out_rows, out_cols, _ = _render_cells_clipped(standard_sprite, out_top, out_left, (H, W))
            if out_rows.size == 0:
                continue
            if bool(output_forbidden[out_rows, out_cols].any()):
                continue

            in_mask = np.zeros((H, W), dtype=bool)
            in_mask[in_rows, in_cols] = True
            out_mask = np.zeros((H, W), dtype=bool)
            out_mask[out_rows, out_cols] = True

            input_forbidden |= _dilate8(in_mask)
            output_forbidden |= _dilate8(out_mask)
            placed = {
                "orientation": orientation,
                "input_sprite": sprite,
                "input_top": int(top),
                "input_left": int(left),
                "output_top": int(out_top),
                "output_left": int(out_left),
            }
            break

        if placed is None:
            return None
        placed_by_index[idx] = placed

    return [placed_by_index[i] for i in range(len(orientations))]


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    body, marker = random.sample(_PALETTE, 2)
    template_h, template_w = random.choice(_TEMPLATE_DIMS)
    mask = _random_mask(template_h, template_w)
    if mask is None:
        return None

    standard_sprite = _make_standard_sprite(mask, body, marker)
    if standard_sprite is None:
        return None

    count = random.randint(*_SPRITE_COUNT_RANGE)
    orientations = _choose_orientations(count)
    if 0 not in orientations or all(ori == 0 for ori in orientations):
        return None

    max_side = max(template_h, template_w)
    min_h = max(_GRID_RANGE[0], max_side + 6)
    min_w = max(_GRID_RANGE[0], max_side + 6)
    H = random.randint(min_h, _GRID_RANGE[1])
    W = random.randint(min_w, _GRID_RANGE[1])

    prefer_clip = bool(random.getrandbits(1))
    placements = _place_sprites(standard_sprite, orientations, (H, W), prefer_clip)
    if placements is None:
        return None

    inp = np.full((H, W), _BG, dtype=int)
    expected = np.full((H, W), _BG, dtype=int)
    for info in placements:
        in_rows, in_cols, in_vals = _render_cells(info["input_sprite"], info["input_top"], info["input_left"])
        inp[in_rows, in_cols] = in_vals
        out_rows, out_cols, out_vals = _render_cells_clipped(
            standard_sprite,
            info["output_top"],
            info["output_left"],
            (H, W),
        )
        expected[out_rows, out_cols] = out_vals

    if len(_components(inp != _BG)) != count:
        return None
    if len(_components(expected != _BG)) != count:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid orientation-normalization puzzle")
