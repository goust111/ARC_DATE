import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 400


def _separator_columns(grid: np.ndarray) -> tuple[int, int]:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 2, f"expected exactly two grey separator columns, got {cols}"
    left_w = cols[0]
    mid_w = cols[1] - cols[0] - 1
    right_w = grid.shape[1] - cols[1] - 1
    assert left_w > 0 and left_w == mid_w == right_w, "separator columns must split the grid into three equal panels"
    return int(cols[0]), int(cols[1])


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep0, sep1 = _separator_columns(grid)
    left = grid[:, :sep0]
    middle = grid[:, sep0 + 1 : sep1]
    right = grid[:, sep1 + 1 :]
    return left, middle, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge three panels by layer masking.

    The left panel is the bottom layer, the middle panel is the visible layer,
    and Maroon pixels in the right panel punch through the middle layer so the
    corresponding left-panel colors become visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    left, middle, right = _extract_panels(g)

    left_colors = {int(v) for v in np.unique(left)}
    assert BLACK in left_colors, "left panel must use black as background"
    left_payload = left_colors - {BLACK}
    assert len(left_payload) == 1, f"left panel must contain exactly one non-black color, got {sorted(left_payload)}"
    assert not (left_payload & {GREY, MAROON}), "left panel must not contain separator or mask colors"

    middle_colors = {int(v) for v in np.unique(middle)}
    assert len(middle_colors) >= 2, "middle panel must contain a multi-color pattern"
    assert GREY not in middle_colors and MAROON not in middle_colors, "middle panel must not contain separator or mask colors"

    right_colors = {int(v) for v in np.unique(right)}
    assert right_colors <= {BLACK, MAROON}, f"right panel must contain only black and maroon, got {sorted(right_colors)}"
    assert MAROON in right_colors, "right panel must contain a maroon mask"

    out = middle.copy()
    mask = right == MAROON
    out[mask] = left[mask]
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _rect_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    rh = rng.randint(2, h)
    rw = rng.randint(2, w)
    r0 = rng.randint(0, h - rh)
    c0 = rng.randint(0, w - rw)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _plus_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    center_r = rng.randrange(h)
    center_c = rng.randrange(w)

    vert_h = rng.randint(max(2, h // 2), h)
    vert_w = rng.randint(1, max(1, w // 3))
    horiz_h = rng.randint(1, max(1, h // 3))
    horiz_w = rng.randint(max(2, w // 2), w)

    vr0_min = max(0, center_r - vert_h + 1)
    vr0_max = min(center_r, h - vert_h)
    vc0_min = max(0, center_c - vert_w + 1)
    vc0_max = min(center_c, w - vert_w)
    hr0_min = max(0, center_r - horiz_h + 1)
    hr0_max = min(center_r, h - horiz_h)
    hc0_min = max(0, center_c - horiz_w + 1)
    hc0_max = min(center_c, w - horiz_w)

    vr0 = rng.randint(vr0_min, vr0_max)
    vc0 = rng.randint(vc0_min, vc0_max)
    hr0 = rng.randint(hr0_min, hr0_max)
    hc0 = rng.randint(hc0_min, hc0_max)

    mask = np.zeros((h, w), dtype=bool)
    mask[vr0 : vr0 + vert_h, vc0 : vc0 + vert_w] = True
    mask[hr0 : hr0 + horiz_h, hc0 : hc0 + horiz_w] = True
    return mask


def _diamond_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    rr, cc = np.indices((h, w))
    cr = rng.randrange(h)
    cw = rng.randrange(w)
    radius = rng.randint(2, max(2, min(h, w) - 1))
    return (np.abs(rr - cr) + np.abs(cc - cw)) <= radius


def _stairs_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    span_h = rng.randint(max(3, h // 2), h)
    r0 = rng.randint(0, h - span_h)
    min_len = rng.randint(1, max(1, w // 3))
    max_len = rng.randint(max(min_len + 1, w // 2), w)
    orient = rng.choice(("tl", "tr", "bl", "br"))

    mask = np.zeros((h, w), dtype=bool)
    for idx in range(span_h):
        frac = idx / max(1, span_h - 1)
        length = int(round(min_len + frac * (max_len - min_len)))
        rr = r0 + idx
        if orient in ("bl", "br"):
            rr = r0 + span_h - 1 - idx
        if orient in ("tl", "bl"):
            mask[rr, :length] = True
        else:
            mask[rr, w - length :] = True
    return mask


def _l_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    orient = rng.choice(("tl", "tr", "bl", "br"))
    vert_h = rng.randint(max(2, h // 2), h)
    horiz_w = rng.randint(max(2, w // 2), w)
    thick_v = rng.randint(1, max(1, w // 3))
    thick_h = rng.randint(1, max(1, h // 3))

    mask = np.zeros((h, w), dtype=bool)
    if orient == "tl":
        mask[:vert_h, :thick_v] = True
        mask[:thick_h, :horiz_w] = True
    elif orient == "tr":
        mask[:vert_h, w - thick_v :] = True
        mask[:thick_h, w - horiz_w :] = True
    elif orient == "bl":
        mask[h - vert_h :, :thick_v] = True
        mask[h - thick_h :, :horiz_w] = True
    else:
        mask[h - vert_h :, w - thick_v :] = True
        mask[h - thick_h :, w - horiz_w :] = True
    return mask


def _blob_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    target = rng.randint(max(6, (h * w) // 5), max(max(6, (h * w) // 5), (3 * h * w) // 5))
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    frontier = set(_neighbors4(seed[0], seed[1], h, w))
    while len(cells) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb not in cells:
                frontier.add(nb)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_connected_mask(
    h: int,
    w: int,
    rng: random.Random,
    kinds: tuple[str, ...],
    area_ratio_bounds: tuple[float, float],
) -> tuple[np.ndarray, str]:
    builders = {
        "rect": _rect_mask,
        "plus": _plus_mask,
        "diamond": _diamond_mask,
        "stairs": _stairs_mask,
        "l": _l_mask,
        "blob": _blob_mask,
    }
    min_ratio, max_ratio = area_ratio_bounds
    min_area = max(3, int(np.ceil(h * w * min_ratio)))
    max_area = min(h * w - 1, int(np.floor(h * w * max_ratio)))
    for _ in range(200):
        kind = rng.choice(kinds)
        mask = builders[kind](h, w, rng)
        area = int(mask.sum())
        if not (min_area <= area <= max_area):
            continue
        if not _is_connected(mask):
            continue
        return mask, kind
    raise RuntimeError("failed to build a connected mask")


def _checker_pattern(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    rr, cc = np.indices((h, w))
    phase = rng.randrange(len(palette))
    idx = (rr + cc + phase) % len(palette)
    return np.array(palette, dtype=int)[idx]


def _stripe_pattern(h: int, w: int, palette: list[int], rng: random.Random, horizontal: bool) -> np.ndarray:
    thickness = rng.randint(1, max(1, (h if horizontal else w) // 3))
    phase = rng.randrange(len(palette))
    base = np.arange(h)[:, None] if horizontal else np.arange(w)[None, :]
    idx = ((base // thickness) + phase) % len(palette)
    if horizontal:
        idx = np.broadcast_to(idx, (h, w))
    else:
        idx = np.broadcast_to(idx, (h, w))
    return np.array(palette, dtype=int)[idx]


def _diagonal_pattern(h: int, w: int, palette: list[int], rng: random.Random, up: bool) -> np.ndarray:
    rr, cc = np.indices((h, w))
    thickness = rng.randint(1, max(1, min(h, w) // 3))
    phase = rng.randrange(len(palette))
    diag = rr - cc if up else rr + cc
    idx = ((diag // thickness) + phase) % len(palette)
    return np.array(palette, dtype=int)[idx]


def _block_pattern(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    rr, cc = np.indices((h, w))
    block_h = rng.randint(1, max(1, h // 3))
    block_w = rng.randint(1, max(1, w // 3))
    phase_r = rng.randrange(len(palette))
    phase_c = rng.randrange(len(palette))
    idx = ((rr // block_h) + phase_r + (cc // block_w) + phase_c) % len(palette)
    return np.array(palette, dtype=int)[idx]


def _make_middle_pattern(h: int, w: int, palette: list[int], rng: random.Random) -> tuple[np.ndarray, str]:
    builders = [
        ("checker", lambda: _checker_pattern(h, w, palette, rng)),
        ("stripes_h", lambda: _stripe_pattern(h, w, palette, rng, horizontal=True)),
        ("stripes_v", lambda: _stripe_pattern(h, w, palette, rng, horizontal=False)),
        ("diag_down", lambda: _diagonal_pattern(h, w, palette, rng, up=False)),
        ("diag_up", lambda: _diagonal_pattern(h, w, palette, rng, up=True)),
        ("blocks", lambda: _block_pattern(h, w, palette, rng)),
    ]
    rng.shuffle(builders)
    for name, build in builders:
        pattern = build().astype(int)
        present = {int(v) for v in np.unique(pattern)}
        if len(present) >= 2 and present <= set(palette):
            return pattern, name
    raise RuntimeError("failed to build a multi-color middle pattern")


def _render_input(
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    h, w = left.shape
    grid = np.full((h, 3 * w + 2), BLACK, dtype=int)
    grid[:, :w] = left
    grid[:, w] = GREY
    grid[:, w + 1 : 2 * w + 1] = middle
    grid[:, 2 * w + 1] = GREY
    grid[:, 2 * w + 2 :] = right
    return grid


def _compose_output(left: np.ndarray, middle: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    out = middle.copy()
    out[right_mask] = left[right_mask]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_RETRIES):
        h = rng.randint(6, 12)
        w = rng.randint(6, 12)

        left_color = rng.choice(COLOR_POOL)
        middle_palette = list(rng.sample([c for c in COLOR_POOL if c != left_color], rng.randint(2, 4)))

        left_mask, _ = _build_connected_mask(h, w, rng, ("rect", "plus", "diamond", "stairs", "l"), (0.22, 0.72))
        right_mask, _ = _build_connected_mask(h, w, rng, ("rect", "plus", "diamond", "stairs", "l", "blob"), (0.18, 0.62))
        if not (right_mask & left_mask).any():
            continue
        if not (right_mask & ~left_mask).any():
            continue
        if not (~right_mask).any():
            continue

        left = np.full((h, w), BLACK, dtype=int)
        left[left_mask] = left_color

        middle, _ = _make_middle_pattern(h, w, middle_palette, rng)

        right = np.full((h, w), BLACK, dtype=int)
        right[right_mask] = MAROON

        grid = _render_input(left, middle, right)
        expected = _compose_output(left, middle, right_mask)

        if not np.array_equal(main(grid), expected):
            continue
        if np.array_equal(expected, middle):
            continue
        return grid

    raise RuntimeError("generate_input: exceeded retry budget")
