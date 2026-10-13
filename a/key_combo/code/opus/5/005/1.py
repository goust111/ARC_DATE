import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
BAND_COLORS = [1, 2, 3, 4, 6]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    shape = _crop_mask_to_bbox(mask)
    if shape.size == 0:
        return 0
    h, w = shape.shape
    holes = 0
    for comp in _components(~shape):
        touches_border = False
        for r, c in comp:
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                touches_border = True
                break
        if not touches_border:
            holes += 1
    return holes


def _active_band_ranges(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    active = np.any(grid != bg, axis=0)
    runs: list[tuple[int, int]] = []
    c = 0
    while c < active.size:
        if not active[c]:
            c += 1
            continue
        c0 = c
        while c + 1 < active.size and active[c + 1]:
            c += 1
        runs.append((c0, c))
        c += 1
    return runs


def _band_info(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bands: list[dict] = []
    for c0, c1 in _active_band_ranges(g, bg):
        band = g[:, c0 : c1 + 1]
        colors = sorted(int(v) for v in np.unique(band) if v not in (bg, GREY))
        assert len(colors) == 1, "each vertical band must have exactly one non-grey foreground color"
        grey_components = _components(band == GREY)
        assert len(grey_components) == 1, "each band must contain exactly one grey sprite"
        grey_mask = np.zeros_like(band, dtype=bool)
        for r, c in grey_components[0]:
            grey_mask[r, c] = True
        bands.append(
            {
                "c0": int(c0),
                "c1": int(c1),
                "color": int(colors[0]),
                "holes": int(_hole_count(grey_mask)),
            }
        )
    assert len(bands) >= 4, "expected at least four bands"
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each vertical band's black gaps with a cyclically borrowed band color determined by the grey sprite's hole count.

    Input structure:
    - Black background surrounds 4-5 narrow vertical bands, separated by fully black columns.
    - Each band has one unique foreground color and contains irregular black gaps throughout the band columns.
    - Near the top of each band sits exactly one grey 4-connected sprite whose enclosed black regions encode a hole count.

    Rule:
    - Order the bands from left to right as B1, B2, ..., BN.
    - For each band Bi, count the number hi of 4-connected black background components fully enclosed by its grey sprite.
    - Replace every black cell inside band Bi's column span with the foreground color of band B((i-1+hi) mod N)+1.
    - Keep existing band-color cells, grey sprite cells, and all background outside the band columns unchanged.

    Key insight:
    The grey sprite does not map directly to a color; its topology supplies the cyclic offset that selects which neighboring band color should flood that band's gaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bands = _band_info(g)
    n = len(bands)
    out = g.copy()
    for i, band in enumerate(bands):
        target = bands[(i + band["holes"]) % n]["color"]
        c0, c1 = band["c0"], band["c1"]
        band_slice = out[:, c0 : c1 + 1]
        fill_mask = band_slice == bg
        band_slice[fill_mask] = int(target)
    return out


def _make_hole_sprite(hole_count: int, width: int) -> np.ndarray:
    height = 2 * hole_count + 1
    sprite = np.zeros((height, width), dtype=int)
    for r in range(height):
        if r % 2 == 0:
            sprite[r, :] = GREY
        else:
            sprite[r, 0] = GREY
            sprite[r, -1] = GREY
            if random.random() < 0.55:
                sprite[r, random.choice([1, width - 2])] = GREY
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    if top_pad or bottom_pad:
        padded = np.full((top_pad + height + bottom_pad, width), GREY, dtype=int)
        padded[top_pad : top_pad + height] = sprite
        sprite = padded
    return sprite


def _sample_band_body(height: int, width: int) -> np.ndarray:
    for _ in range(60):
        density = random.uniform(0.42, 0.58)
        body = np.random.random((height, width)) < density
        for c in range(width):
            if not body[:, c].any():
                body[random.randrange(height), c] = True
        if body.all():
            body[random.randrange(height), random.randrange(width)] = False
        if 0.38 <= body.mean() <= 0.62:
            return body
    body = np.zeros((height, width), dtype=bool)
    for c in range(width):
        rows = random.sample(range(height), k=max(1, height // 2))
        for r in rows:
            body[r, c] = True
    return body


def _best_hole_permutation(n: int) -> list[int]:
    values = list(range(1, n + 1))
    best = values[:]
    best_score = -1
    for _ in range(300):
        trial = values[:]
        random.shuffle(trial)
        targets = [(i + trial[i]) % n for i in range(n)]
        score = len(set(targets))
        if score > best_score:
            best = trial[:]
            best_score = score
        if score == n:
            break
    return best


def _oracle_from_spec(
    grid: np.ndarray,
    band_ranges: list[tuple[int, int]],
    colors: list[int],
    hole_counts: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    n = len(band_ranges)
    for i, (c0, c1) in enumerate(band_ranges):
        target = colors[(i + hole_counts[i]) % n]
        band_slice = out[:, c0 : c1 + 1]
        fill_mask = band_slice == BG
        band_slice[fill_mask] = int(target)
    return out


def _validate_generated_case(
    grid: np.ndarray,
    band_ranges: list[tuple[int, int]],
    colors: list[int],
    hole_counts: list[int],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    expected = _oracle_from_spec(grid, band_ranges, colors, hole_counts)
    if np.array_equal(grid, expected):
        return False
    detected = _band_info(grid)
    if len(detected) != len(band_ranges):
        return False
    if [d["color"] for d in detected] != list(colors):
        return False
    if [d["holes"] for d in detected] != list(hole_counts):
        return False
    if len(set(hole_counts)) != len(hole_counts):
        return False
    target_indices = [(i + hole_counts[i]) % len(hole_counts) for i in range(len(hole_counts))]
    if len(set(target_indices)) < min(3, len(hole_counts)):
        return False
    if not all(np.any(grid[:, c0 : c1 + 1] == BG) for c0, c1 in band_ranges):
        return False
    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_case() -> np.ndarray | None:
    n = random.choice([4, 5])
    band_width = 4
    gap_width = 1
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    width = left_margin + n * band_width + (n - 1) * gap_width + right_margin
    if width > 30:
        return None

    hole_counts = _best_hole_permutation(n)
    colors = random.sample(BAND_COLORS, n)
    sprites = [_make_hole_sprite(hc, band_width) for hc in hole_counts]
    max_sprite_h = max(sprite.shape[0] for sprite in sprites)

    height_min = max(18, max_sprite_h + 7)
    height_max = min(28, max_sprite_h + 15)
    if height_min > height_max:
        return None
    height = random.randint(height_min, height_max)
    grid = np.full((height, width), BG, dtype=int)

    band_ranges: list[tuple[int, int]] = []
    for i in range(n):
        c0 = left_margin + i * (band_width + gap_width)
        c1 = c0 + band_width - 1
        band_ranges.append((c0, c1))

    for (c0, c1), color, sprite in zip(band_ranges, colors, sprites, strict=False):
        sprite_h = sprite.shape[0]
        top_gap = random.randint(0, min(2, height - sprite_h - 5))
        body_gap = random.randint(0, 2)
        body_top = top_gap + sprite_h + body_gap
        if body_top >= height - 3:
            return None

        for r in range(sprite_h):
            for c in range(band_width):
                if sprite[r, c] == GREY:
                    grid[top_gap + r, c0 + c] = GREY

        accent_rows = list(range(top_gap)) + list(range(top_gap + sprite_h, body_top))
        for r in accent_rows:
            for c in range(c0, c1 + 1):
                if random.random() < 0.22:
                    grid[r, c] = color

        body = _sample_band_body(height - body_top, band_width)
        for rr in range(body.shape[0]):
            for cc in range(body.shape[1]):
                if body[rr, cc]:
                    grid[body_top + rr, c0 + cc] = color

    return grid if _validate_generated_case(grid, band_ranges, colors, hole_counts) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-count cyclic-band input")
