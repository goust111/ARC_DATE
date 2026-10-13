from __future__ import annotations

import random
from collections import deque

import numpy as np


GEN_ATTEMPTS = 500
ALL_COLORS = list(range(10))


def _border_bg(g: np.ndarray) -> int:
    border = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    colors, counts = np.unique(border, return_counts=True)
    winners = colors[counts == counts.max()]
    assert winners.size == 1, "expected a unique border background color"
    return int(winners[0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        q = deque([(r0, c0)])
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_sprite(g: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for comp in _components4(g != bg):
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        if h != w or h < 2:
            continue
        if len(comp) != h * w or not np.all(patch != bg):
            continue
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        candidates.append(((r0, c0, r1, c1), patch.copy()))

    assert len(candidates) == 1, "expected exactly one filled multicolor square control sprite"
    return candidates[0]


def _scan_kind(mask: np.ndarray) -> str:
    h, w = mask.shape
    return "row" if h >= w else "col"


def _ordered_mask_cells(mask: np.ndarray, kind: str | None = None) -> list[tuple[int, int]]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert cells, "expected at least one filled mask cell"
    if kind is None:
        kind = _scan_kind(mask)
    if kind == "row":
        return sorted(cells)
    assert kind == "col"
    return sorted(cells, key=lambda rc: (rc[1], rc[0]))


def _render(mask: np.ndarray, sprite: np.ndarray, bg: int, kind: str | None = None) -> np.ndarray:
    m = int(sprite.shape[0])
    mh, mw = mask.shape
    out = np.full((mh * m, mw * m), int(bg), dtype=int)
    for rank, (mr, mc) in enumerate(_ordered_mask_cells(mask, kind)):
        shifted = np.roll(sprite, rank, axis=0)
        out[mr * m : (mr + 1) * m, mc * m : (mc + 1) * m] = shifted
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each filled coarse mask cell by a phase-shifted copy of the control sprite.

    Input structure:
    - A background surrounds two separated parts: a one-color coarse mask and one filled multicolor square sprite.
    - Empty coarse cells are background pixels inside the mask foreground bbox; filled coarse cells share one marker color.
    - The square sprite has side length m and contains no background pixels.

    Rule:
    - Crop the coarse mask to the bbox of its filled cells, and make an output with one m by m block per coarse cell.
    - If the mask bbox is at least as tall as it is wide, order filled cells row-major; otherwise order them column-major.
    - Put the control sprite in each filled block after cyclically shifting its rows downward by that cell's zero-based scan rank.

    Key insight:
    - The varied output blocks are not independent patterns: they are phases of the same sprite, and the phase is determined by scan rank under the mask bbox's longer-axis scan direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _border_bg(g)
    (sr0, sc0, sr1, sc1), sprite = _find_sprite(g, bg)
    m = int(sprite.shape[0])
    assert sprite.shape == (m, m), "control sprite must be square"

    sprite_area = np.zeros(g.shape, dtype=bool)
    sprite_area[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    coarse_fg = (g != bg) & ~sprite_area
    coarse_coords = np.argwhere(coarse_fg)
    assert coarse_coords.size > 0, "expected a coarse mask outside the sprite"

    marker_colors = set(map(int, g[coarse_fg]))
    assert len(marker_colors) == 1, "coarse mask filled cells must share one marker color"

    r0, c0, r1, c1 = _bbox(coarse_coords)
    mask = coarse_fg[r0 : r1 + 1, c0 : c1 + 1]
    assert np.any(mask), "expected a non-empty cropped coarse mask"
    return _render(mask, sprite, bg)


# --------------- generator ---------------


def _shape_choices(m: int, kind: str) -> list[tuple[int, int]]:
    max_dim = min(9, 30 // int(m))
    choices: list[tuple[int, int]] = []
    for h in range(2, max_dim + 1):
        for w in range(2, max_dim + 1):
            if h * w < m + 4:
                continue
            if kind == "row" and h >= w:
                choices.append((h, w))
            if kind == "col" and w > h:
                choices.append((h, w))
    return choices


def _orders_differ(mask: np.ndarray) -> bool:
    return _ordered_mask_cells(mask, "row") != _ordered_mask_cells(mask, "col")


def _full_row_period(sprite: np.ndarray) -> bool:
    m = int(sprite.shape[0])
    return all(not np.array_equal(np.roll(sprite, shift, axis=0), sprite) for shift in range(1, m))


def _gen_sprite(m: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(200):
        values = [random.choice(palette) for _ in range(m * m)]
        sprite = np.array(values, dtype=int).reshape((m, m))
        if len(set(map(int, sprite.ravel()))) < 2:
            continue
        if not set(palette).issubset(set(map(int, sprite.ravel()))):
            continue
        if not _full_row_period(sprite):
            continue
        return sprite
    return None


def _gen_mask(h: int, w: int, m: int, sprite: np.ndarray) -> np.ndarray | None:
    total = h * w
    low = max(4, m + 1, int(np.ceil(total * 0.35)))
    high = min(total - 1, int(np.floor(total * 0.72)))
    if low > high:
        return None

    all_cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(300):
        target = random.randint(low, high)
        cells = {
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        }
        pool = [cell for cell in all_cells if cell not in cells]
        random.shuffle(pool)
        cells.update(pool[: max(0, target - len(cells))])
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if np.all(mask):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _orders_differ(mask):
            continue

        actual = _scan_kind(mask)
        wrong = "col" if actual == "row" else "row"
        if np.array_equal(_render(mask, sprite, 0, actual), _render(mask, sprite, 0, wrong)):
            continue
        phases = {
            _render(mask, sprite, 0)[r * m : (r + 1) * m, c * m : (c + 1) * m].tobytes()
            for r, c in np.argwhere(mask)
        }
        if len(phases) < m:
            continue
        return mask
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    view = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    view[mask] = int(color)


def _assemble_input(mask: np.ndarray, sprite: np.ndarray, bg: int, mask_color: int) -> np.ndarray | None:
    mh, mw = mask.shape
    m = int(sprite.shape[0])
    gap = random.randint(2, 4)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)

    layout = random.choice(("horizontal", "vertical"))
    mask_first = random.choice((True, False))

    if layout == "horizontal":
        used_h = max(mh, m) + random.randint(0, 3)
        used_w = mw + gap + m
        h = top_pad + used_h + bottom_pad
        w = left_pad + used_w + right_pad
        if h > 30 or w > 30:
            return None
        grid = np.full((h, w), int(bg), dtype=int)
        r_mask = top_pad + random.randint(0, used_h - mh)
        r_sprite = top_pad + random.randint(0, used_h - m)
        if mask_first:
            c_mask = left_pad
            c_sprite = left_pad + mw + gap
        else:
            c_sprite = left_pad
            c_mask = left_pad + m + gap
    else:
        used_h = mh + gap + m
        used_w = max(mw, m) + random.randint(0, 3)
        h = top_pad + used_h + bottom_pad
        w = left_pad + used_w + right_pad
        if h > 30 or w > 30:
            return None
        grid = np.full((h, w), int(bg), dtype=int)
        c_mask = left_pad + random.randint(0, used_w - mw)
        c_sprite = left_pad + random.randint(0, used_w - m)
        if mask_first:
            r_mask = top_pad
            r_sprite = top_pad + mh + gap
        else:
            r_sprite = top_pad
            r_mask = top_pad + m + gap

    _paint_mask(grid, mask, r_mask, c_mask, mask_color)
    grid[r_sprite : r_sprite + m, c_sprite : c_sprite + m] = sprite
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(ALL_COLORS)
    available = [c for c in ALL_COLORS if c != bg]
    random.shuffle(available)
    mask_color = available[0]
    sprite_palette_size = random.randint(2, min(4, len(available) - 1))
    sprite_palette = available[1 : 1 + sprite_palette_size]

    m = random.choice((2, 3, 4, 5))
    sprite = _gen_sprite(m, sprite_palette)
    if sprite is None:
        return None

    kind = random.choice(("row", "col"))
    choices = _shape_choices(m, kind)
    if not choices:
        return None
    h, w = random.choice(choices)
    mask = _gen_mask(h, w, m, sprite)
    if mask is None:
        return None

    inp = _assemble_input(mask, sprite, bg, mask_color)
    if inp is None:
        return None

    expected = _render(mask, sprite, bg)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic guards: the branch direction is necessary, cyclic phases are visible,
    # and empty coarse cells remain background blocks in the output.
    wrong_kind = "col" if _scan_kind(mask) == "row" else "row"
    if np.array_equal(expected, _render(mask, sprite, bg, wrong_kind)):
        return None
    if int(np.count_nonzero(mask)) < m + 1:
        return None
    block_sigs = set()
    for r, c in np.argwhere(mask):
        block = actual[int(r) * m : (int(r) + 1) * m, int(c) * m : (int(c) + 1) * m]
        block_sigs.add(block.tobytes())
    if len(block_sigs) < m:
        return None
    for r, c in np.argwhere(~mask):
        block = actual[int(r) * m : (int(r) + 1) * m, int(c) * m : (int(c) + 1) * m]
        if not np.all(block == bg):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phase-stamp input")
