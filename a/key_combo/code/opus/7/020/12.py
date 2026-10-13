import random
from collections import deque

import numpy as np


BG = 0
INACTIVE = 5
ACTIVE = 8
SPRITE_PALETTE = (2, 3, 4, 6, 7, 9)
SPRITE_SHAPES = (
    ((1, 0, 0), (1, 1, 0), (0, 1, 1)),
    ((0, 1, 0), (1, 1, 0), (1, 0, 1)),
    ((1, 1, 0), (0, 1, 0), (1, 0, 0)),
    ((0, 1, 1), (1, 1, 0), (0, 0, 1)),
    ((1, 0, 1), (1, 1, 0), (0, 0, 1)),
    ((1, 0, 0), (0, 1, 1), (1, 1, 0)),
    ((0, 0, 1), (1, 1, 0), (1, 0, 1)),
    ((1, 1, 0), (1, 0, 1), (0, 1, 0)),
)
GEN_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _extract_sprites(grid: np.ndarray) -> list[np.ndarray]:
    sprites: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for color in colors_present(grid, exclude={BG, INACTIVE, ACTIVE}):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "sprite color must be present"
        box = bbox_coords(coords)
        sprite = crop_bbox(grid, box)
        assert sprite.shape == (3, 3), f"sprite bbox must be 3x3, got {sprite.shape}"
        assert np.any(sprite == color), "sprite bbox must contain its color"
        sprites.append((box[0], box, sprite))
    assert 2 <= len(sprites) <= 3, f"expected 2 or 3 sprites, got {len(sprites)}"
    sprites.sort(key=lambda item: item[0])
    return [sprite for _top, _box, sprite in sprites]


def _extract_selector(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = (grid == INACTIVE) | (grid == ACTIVE)
    coords = np.argwhere(mask)
    assert coords.size > 0, "macro pattern not found"
    box = bbox_coords(coords)
    patch = crop_bbox(grid, box)
    h, w = patch.shape
    assert h % 3 == 0 and w % 3 == 0, "macro pattern must be tiled by 3x3 cells"
    macro_h, macro_w = h // 3, w // 3
    selector = np.zeros((macro_h, macro_w), dtype=bool)
    for r in range(macro_h):
        for c in range(macro_w):
            cell = patch[r * 3 : (r + 1) * 3, c * 3 : (c + 1) * 3]
            vals = set(map(int, np.unique(cell)))
            assert len(vals) == 1, f"macro-cell {(r, c)} is not uniform"
            color = next(iter(vals))
            assert color in {INACTIVE, ACTIVE}, f"unexpected macro color {color}"
            selector[r, c] = color == ACTIVE
    return box, selector


def _macro_depths(selector: np.ndarray) -> np.ndarray:
    h, w = selector.shape
    depths = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if not selector[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                depths[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and selector[nr, nc] and depths[nr, nc] == -1:
                depths[nr, nc] = depths[r, c] + 1
                q.append((nr, nc))
    leftovers = np.argwhere(selector & (depths < 0))
    for r, c in leftovers:
        depths[r, c] = 0
    return depths


def _render_output(grid: np.ndarray, sprites: list[np.ndarray], selector: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    (r0, c0, r1, c1), _ = _extract_selector(grid)
    depths = _macro_depths(selector)
    out[r0 : r1 + 1, c0 : c1 + 1] = INACTIVE
    for r in range(selector.shape[0]):
        for c in range(selector.shape[1]):
            if not selector[r, c]:
                continue
            depth = int(depths[r, c])
            sprite = sprites[depth % len(sprites)]
            stamp = np.fliplr(sprite) if depth % 2 == 0 else sprite
            paste(out, stamp, r0 + r * 3, c0 + c * 3)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each active 3x3 macro-cell by a depth-selected control sprite, mirroring even macro-depth layers.

    Input structure:
    - Black is the background.
    - The left side contains 2 or 3 control sprites, each occupying a 3x3 bbox and using one unique paint color plus black.
    - The right side contains one rectangular macro-grid made of uniform 3x3 cells colored grey or teal.

    Rule:
    - Read the left control sprites from top to bottom as S0, S1, S2, ...
    - Interpret each teal 3x3 macro-cell on the right as active and each grey 3x3 macro-cell as inactive.
    - On the macro-grid, compute the active-cell depth as the 4-connected distance through active cells to the nearest active macro-cell on the outer macro-grid border.
    - Replace each active macro-cell by sprite S[d mod len(S)] where d is its depth.
    - Mirror the chosen sprite horizontally before stamping when d is even; keep it unchanged when d is odd.
    - Keep inactive macro-cells grey and leave the left control area unchanged.

    Key insight:
    - The right pattern is solved at macro scale first: border-reaching active cells define nested depth layers, and only then are the 3x3 sprites chosen and mirrored for the fine-scale replacement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sprites = _extract_sprites(g)
    _box, selector = _extract_selector(g)
    return _render_output(g, sprites, selector)


def _random_sprite(color: int, used: set[bytes]) -> np.ndarray | None:
    shapes = list(SPRITE_SHAPES)
    random.shuffle(shapes)
    for shape in shapes:
        mask = np.array(shape, dtype=bool)
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        sprite = np.where(mask, color, BG).astype(int)
        key = sprite.tobytes()
        mirror_key = np.fliplr(sprite).tobytes()
        if key in used or mirror_key in used:
            continue
        used.add(key)
        used.add(mirror_key)
        return sprite
    return None


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        count += 1
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _touches_border(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _selector_ok(selector: np.ndarray, sprite_count: int) -> bool:
    if not selector.any():
        return False
    if _component_count(selector) != 1:
        return False
    if not _touches_border(selector):
        return False
    depths = _macro_depths(selector)
    active_depths = depths[selector]
    if active_depths.size == 0 or np.any(active_depths < 0):
        return False
    if len(set(map(int, active_depths))) < 3:
        return False
    if int(active_depths.max()) < sprite_count:
        return False
    if not np.any(active_depths % 2 == 0) or not np.any(active_depths % 2 == 1):
        return False
    if len({int(d % sprite_count) for d in active_depths}) != sprite_count:
        return False
    naive = np.fromfunction(
        lambda r, c: np.minimum.reduce([r, c, selector.shape[0] - 1 - r, selector.shape[1] - 1 - c]),
        selector.shape,
        dtype=int,
    ).astype(int)
    if not np.any(selector & (depths != naive)):
        return False
    return True


def _random_selector(sprite_count: int) -> np.ndarray | None:
    size = 5 if sprite_count == 2 else 7
    target_carves = random.randint(size, size + 4)
    center = size // 2
    selector = np.ones((size, size), dtype=bool)
    protected = {(center, center)}
    if size == 7:
        protected |= {(center - 1, center), (center, center - 1), (center + 1, center), (center, center + 1)}
    for _ in range(target_carves):
        candidates = [(r, c) for r, c in np.argwhere(selector) if (int(r), int(c)) not in protected]
        random.shuffle(candidates)
        carved = False
        for r, c in candidates:
            trial = selector.copy()
            trial[int(r), int(c)] = False
            if trial.sum() < size * size // 2:
                continue
            if not _selector_ok(trial, sprite_count):
                continue
            selector = trial
            carved = True
            break
        if not carved:
            break
    if _selector_ok(selector, sprite_count):
        return selector
    return None


def _build_input(
    sprites: list[np.ndarray],
    selector: np.ndarray,
    *,
    control_top: int,
    control_left: int,
    pattern_top: int,
    pattern_left: int,
    height: int,
    width: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for idx, sprite in enumerate(sprites):
        paste(grid, sprite, control_top + idx * 4, control_left)
    for r in range(selector.shape[0]):
        for c in range(selector.shape[1]):
            color = ACTIVE if selector[r, c] else INACTIVE
            patch = np.full((3, 3), color, dtype=int)
            paste(grid, patch, pattern_top + r * 3, pattern_left + c * 3)
    return grid


def _independent_expected(grid: np.ndarray, sprites: list[np.ndarray], selector: np.ndarray) -> np.ndarray:
    pattern_mask = (grid == INACTIVE) | (grid == ACTIVE)
    box = bbox_coords(np.argwhere(pattern_mask))
    r0, c0, r1, c1 = box
    out = grid.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = INACTIVE
    depths = _macro_depths(selector)
    for r in range(selector.shape[0]):
        for c in range(selector.shape[1]):
            if not selector[r, c]:
                continue
            depth = int(depths[r, c])
            sprite = sprites[depth % len(sprites)]
            stamp = np.fliplr(sprite) if depth % 2 == 0 else sprite
            rr = r0 + 3 * r
            cc = c0 + 3 * c
            out[rr : rr + 3, cc : cc + 3] = stamp
    return out


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.choice((2, 3))
    colors = random.sample(SPRITE_PALETTE, sprite_count)
    used: set[bytes] = set()
    sprites: list[np.ndarray] = []
    for color in colors:
        sprite = _random_sprite(color, used)
        if sprite is None:
            return None
        sprites.append(sprite)

    selector = _random_selector(sprite_count)
    if selector is None:
        return None

    pattern_h, pattern_w = selector.shape[0] * 3, selector.shape[1] * 3
    control_h = sprite_count * 3 + (sprite_count - 1)
    height = max(pattern_h, control_h) + random.randint(2, 5)
    width = 3 + pattern_w + random.randint(4, 8)
    if height > 30 or width > 30:
        return None

    control_top = random.randint(1, height - control_h - 1)
    control_left = random.randint(1, 2)
    pattern_top = random.randint(1, height - pattern_h - 1)
    min_pattern_left = control_left + 3 + random.randint(2, 4)
    max_pattern_left = width - pattern_w - 1
    if min_pattern_left > max_pattern_left:
        return None
    pattern_left = random.randint(min_pattern_left, max_pattern_left)

    grid = _build_input(
        sprites,
        selector,
        control_top=control_top,
        control_left=control_left,
        pattern_top=pattern_top,
        pattern_left=pattern_left,
        height=height,
        width=width,
    )

    if len(colors_present(grid, exclude={BG})) != sprite_count + 2:
        return None

    depths = _macro_depths(selector)
    if not np.any(selector & (depths % 2 == 0)) or not np.any(selector & (depths % 2 == 1)):
        return None
    if not all(not np.array_equal(sprite, np.fliplr(sprite)) for sprite in sprites):
        return None

    expected = _independent_expected(grid, sprites, selector)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 12")
