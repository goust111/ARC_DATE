from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 7
    TEAL = 8


SPRITE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
BAR_COLOR = Color.ORANGE
DIAMOND_COLOR = Color.TEAL
GENERATION_ATTEMPTS = 400


@dataclass
class Component:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    patch: np.ndarray


@dataclass
class Bar:
    row: int
    c0: int
    c1: int
    radius: int
    outline: set[tuple[int, int]]
    upper_outline: set[tuple[int, int]]
    bbox: tuple[int, int, int, int]


@dataclass
class SpritePlacement:
    patch: np.ndarray
    final_row: int
    col: int
    support_kind: str
    support_bar: int
    support_from: int | None = None
    input_row: int | None = None
    color: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(map(int, self.patch.shape))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        h, w = self.shape
        return self.final_row, self.col, self.final_row + h - 1, self.col + w - 1

    @property
    def coords(self) -> set[tuple[int, int]]:
        return _absolute_coords(self.patch, self.final_row, self.col)

    @property
    def top_surface(self) -> set[tuple[int, int]]:
        return _top_surface(self.patch, self.final_row, self.col)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_components(grid: np.ndarray, colors: list[int]) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[Component] = []
    allowed = set(map(int, colors))
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in allowed or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            patch = (grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            out.append(Component(color=color, coords=coords, bbox=bbox, patch=patch))
    return out


def _absolute_coords(patch: np.ndarray, row: int, col: int) -> set[tuple[int, int]]:
    coords = np.argwhere(patch)
    return {(int(row + rr), int(col + cc)) for rr, cc in coords}


def _top_surface(patch: np.ndarray, row: int, col: int) -> set[tuple[int, int]]:
    h, w = patch.shape
    surf: set[tuple[int, int]] = set()
    for cc in range(w):
        rows = np.flatnonzero(patch[:, cc])
        if len(rows):
            surf.add((int(row + rows[0]), int(col + cc)))
    return surf


def _bottom_rel_cells(patch: np.ndarray) -> list[tuple[int, int]]:
    rows = np.argwhere(patch)
    max_r = int(rows[:, 0].max())
    return [(int(rr), int(cc)) for rr, cc in rows if int(rr) == max_r]


def _make_bar(component: Component) -> Bar:
    r0, c0, r1, c1 = component.bbox
    assert r0 == r1, "orange components must be horizontal one-cell bars"
    assert int(component.patch.sum()) == c1 - c0 + 1, "orange bars must be solid"
    radius = (c1 - c0) // 2
    outline = {(r0, c) for c in range(c0, c1 + 1)}
    upper_outline: set[tuple[int, int]] = set()
    for offset in range(1, radius + 1):
        left = c0 + offset
        right = c1 - offset
        for c in {left, right}:
            outline.add((r0 - offset, c))
            outline.add((r0 + offset, c))
            upper_outline.add((r0 - offset, c))
    return Bar(
        row=int(r0),
        c0=int(c0),
        c1=int(c1),
        radius=int(radius),
        outline=outline,
        upper_outline=upper_outline,
        bbox=(int(r0 - radius), int(c0), int(r0 + radius), int(c1)),
    )


def _occupied_overlap(occupied: np.ndarray, patch: np.ndarray, row: int, col: int) -> bool:
    h, w = patch.shape
    return bool((occupied[row : row + h, col : col + w] & patch).any())


def _collision_coords(occupied: np.ndarray, patch: np.ndarray, row: int, col: int) -> set[tuple[int, int]]:
    h, w = patch.shape
    hit = np.argwhere(occupied[row : row + h, col : col + w] & patch)
    return {(int(row + rr), int(col + cc)) for rr, cc in hit}


def _drop_until_stop(occupied: np.ndarray, patch: np.ndarray, col: int) -> tuple[int | None, set[tuple[int, int]], bool]:
    h, w = patch.shape
    grid_h, grid_w = occupied.shape
    if col < 0 or col + w > grid_w:
        return None, set(), False
    row = 0
    if _occupied_overlap(occupied, patch, row, col):
        return None, set(), False
    while True:
        if row + h == grid_h:
            return int(row), set(), True
        hits = _collision_coords(occupied, patch, row + 1, col)
        if hits:
            return int(row), hits, False
        row += 1


def _paint_patch(grid: np.ndarray, patch: np.ndarray, row: int, col: int, color: int) -> None:
    h, w = patch.shape
    view = grid[row : row + h, col : col + w]
    view[patch] = int(color)


def _build_output(shape: tuple[int, int], bars: list[Bar], sprites: list[SpritePlacement], bg: int = Color.BLACK) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for bar in bars:
        for r, c in bar.outline:
            out[r, c] = DIAMOND_COLOR
    for sprite in sprites:
        _paint_patch(out, sprite.patch, sprite.final_row, sprite.col, sprite.color)
    return out


def _support_grid(shape: tuple[int, int], bars: list[Bar], sprites: list[SpritePlacement] | None = None) -> np.ndarray:
    occ = np.zeros(shape, dtype=bool)
    for bar in bars:
        for r, c in bar.outline:
            occ[r, c] = True
    for sprite in sprites or []:
        h, w = sprite.patch.shape
        occ[sprite.final_row : sprite.final_row + h, sprite.col : sprite.col + w] |= sprite.patch
    return occ


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each orange bar into a teal diamond outline, erase the bars, then let the small sprites fall until they hit the outline, an earlier settled sprite, or the floor.

    Input structure:
    - The grid has a black background, 2-3 orange horizontal solid bars in the middle-lower area, and 3-5 small monochrome sprites above them.
    - Each sprite is a 4-connected shape of 3-6 pixels with bounding box at most 3x3 and keeps its original columns.
    - Teal is absent from the input and is reserved for the generated diamond-outline platforms.

    Rule:
    - Replace every orange horizontal bar by a teal diamond outline whose center line is the bar itself and whose upper/lower endpoints shrink inward by one cell per row.
    - Remove the original orange color entirely; only the teal outline remains as the collision surface.
    - Process sprites from lower initial position to higher initial position and slide each one straight downward without changing shape or columns.
    - A sprite stops just before it would overlap the teal outline, a previously settled sprite, or the bottom boundary; erase the original sprite positions.

    Key insight:
    - The visible bars are not the final support objects: their length defines an invisible diamond envelope, and the sprites must be dropped onto that derived outline rather than onto the original bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    bar_components = _extract_components(g, [BAR_COLOR])
    assert 2 <= len(bar_components) <= 3, "expected 2-3 orange bars"
    bars = [_make_bar(component) for component in sorted(bar_components, key=lambda comp: comp.bbox)]

    sprite_colors = [color for color in sorted(set(map(int, np.unique(g)))) if color not in {bg, BAR_COLOR}]
    sprite_components = _extract_components(g, sprite_colors)
    assert sprite_components, "expected at least one sprite"

    out = np.full_like(g, bg)
    for bar in bars:
        for r, c in bar.outline:
            out[r, c] = DIAMOND_COLOR

    occupied = out != bg
    ordered_sprites = sorted(sprite_components, key=lambda comp: (-comp.bbox[2], comp.bbox[0], comp.bbox[1]))
    for component in ordered_sprites:
        row, _hits, _bottom = _drop_until_stop(occupied, component.patch, component.bbox[1])
        assert row is not None, "sprite must have a valid falling path"
        _paint_patch(out, component.patch, row, component.bbox[1], component.color)
        h, w = component.patch.shape
        occupied[row : row + h, component.bbox[1] : component.bbox[1] + w] |= component.patch
    return out


BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
]


def _all_shape_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in BASE_SHAPES:
        for rot in range(4):
            arr = np.rot90(base, rot)
            for flip in (False, True):
                cur = np.fliplr(arr) if flip else arr
                key = tuple(tuple(int(v) for v in row) for row in cur.astype(int))
                if key in seen:
                    continue
                seen.add(key)
                variants.append(cur.copy())
    return [variant for variant in variants if 3 <= int(variant.sum()) <= 6 and max(variant.shape) <= 3]


SHAPES = _all_shape_variants()


def _bar_bbox_clear(new_bbox: tuple[int, int, int, int], bars: list[Bar]) -> bool:
    nr0, nc0, nr1, nc1 = new_bbox
    for bar in bars:
        br0, bc0, br1, bc1 = bar.bbox
        if nr1 + 1 < br0 or br1 + 1 < nr0 or nc1 + 1 < bc0 or bc1 + 1 < nc0:
            continue
        return False
    return True


def _sample_bars(rng: random.Random, height: int, width: int) -> list[Bar] | None:
    bar_count = rng.randint(2, 3)
    lengths = [rng.randint(3, 7) for _ in range(bar_count)]
    if max(lengths) < 5:
        lengths[rng.randrange(bar_count)] = rng.randint(5, 7)
    if len({length % 2 for length in lengths}) == 1 and rng.random() < 0.6:
        idx = rng.randrange(bar_count)
        options = [cand for cand in range(3, 8) if cand % 2 != lengths[idx] % 2]
        lengths[idx] = rng.choice(options)
    bars: list[Bar] = []
    row_floor = max(9, height // 2)
    for length in sorted(lengths, reverse=True):
        radius = (length - 1) // 2
        placed = False
        row_choices = list(range(row_floor + radius, height - radius - 1))
        rng.shuffle(row_choices)
        for row in row_choices:
            col_choices = list(range(1, width - length))
            rng.shuffle(col_choices)
            for c0 in col_choices:
                c1 = c0 + length - 1
                new_bbox = (row - radius, c0, row + radius, c1)
                if not _bar_bbox_clear(new_bbox, bars):
                    continue
                if any(existing.c0 == c0 or existing.c1 == c1 for existing in bars):
                    continue
                component = Component(
                    color=BAR_COLOR,
                    coords=np.array([(row, c) for c in range(c0, c1 + 1)], dtype=int),
                    bbox=(row, c0, row, c1),
                    patch=np.ones((1, length), dtype=bool),
                )
                bars.append(_make_bar(component))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return sorted(bars, key=lambda bar: (bar.row, bar.c0))


def _find_direct_landing(
    rng: random.Random,
    occupied: np.ndarray,
    patch: np.ndarray,
    bar: Bar,
) -> tuple[int, int, set[tuple[int, int]]] | None:
    _h, w = patch.shape
    cols = list(range(max(0, bar.c0 - w + 1), min(occupied.shape[1] - w, bar.c1) + 1))
    rng.shuffle(cols)
    for col in cols:
        row, hits, hit_bottom = _drop_until_stop(occupied, patch, col)
        if row is None or hit_bottom or not hits:
            continue
        if not hits <= bar.outline:
            continue
        if not (hits & bar.upper_outline):
            continue
        return int(row), int(col), hits
    return None


def _find_stack_landing(
    rng: random.Random,
    occupied: np.ndarray,
    patch: np.ndarray,
    lower: SpritePlacement,
    bar: Bar,
) -> tuple[int, int] | None:
    _h, w = patch.shape
    cols = list(range(max(0, lower.col - w + 1), min(occupied.shape[1] - w, lower.col + lower.shape[1] - 1) + 1))
    rng.shuffle(cols)
    for col in cols:
        row, hits, hit_bottom = _drop_until_stop(occupied, patch, col)
        if row is None or hit_bottom or not hits:
            continue
        if not hits <= lower.top_surface:
            continue
        if max(col, bar.c0) > min(col + w - 1, bar.c1):
            continue
        return int(row), int(col)
    return None


def _assign_colors(rng: random.Random, sprites: list[SpritePlacement]) -> None:
    distinct_count = min(len(SPRITE_COLORS), max(2, min(len(sprites), rng.randint(2, 4))))
    chosen = list(rng.sample(SPRITE_COLORS, distinct_count))
    colors = chosen.copy()
    while len(colors) < len(sprites):
        colors.append(rng.choice(chosen))
    rng.shuffle(colors)
    for sprite, color in zip(sprites, colors):
        sprite.color = int(color)


def _assign_input_rows(rng: random.Random, sprites: list[SpritePlacement], min_bar_row: int, shape: tuple[int, int]) -> bool:
    occupied = np.zeros(shape, dtype=bool)
    order = sorted(range(len(sprites)), key=lambda idx: sprites[idx].bbox[2], reverse=True)
    for idx in order:
        sprite = sprites[idx]
        h, _w = sprite.shape
        row_high = min(min_bar_row - h, sprite.final_row - 1)
        if sprite.support_kind == "stack":
            assert sprite.support_from is not None
            lower = sprites[sprite.support_from]
            assert lower.input_row is not None
            row_high = min(row_high, lower.input_row - h - 1)
        if row_high < 0:
            return False
        viable = [row for row in range(row_high, -1, -1) if not _occupied_overlap(occupied, sprite.patch, row, sprite.col)]
        if not viable:
            return False
        pick_pool = viable[: min(7, len(viable))]
        sprite.input_row = int(rng.choice(pick_pool))
        hh, ww = sprite.shape
        occupied[sprite.input_row : sprite.input_row + hh, sprite.col : sprite.col + ww] |= sprite.patch
    return True


def _build_input(shape: tuple[int, int], bars: list[Bar], sprites: list[SpritePlacement]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for bar in bars:
        grid[bar.row, bar.c0 : bar.c1 + 1] = BAR_COLOR
    for sprite in sprites:
        assert sprite.input_row is not None
        _paint_patch(grid, sprite.patch, sprite.input_row, sprite.col, sprite.color)
    return grid


def _same_output(expected: np.ndarray, actual: np.ndarray) -> bool:
    return expected.shape == actual.shape and np.array_equal(expected, actual)


def _generate_sprite_scene(rng: random.Random, height: int, width: int, bars: list[Bar]) -> list[SpritePlacement] | None:
    sprite_count = rng.randint(3, 5)
    want_stack = sprite_count >= 4 and rng.random() < 0.6
    direct_count = sprite_count - (1 if want_stack else 0)

    occupied = _support_grid((height, width), bars)
    sprites: list[SpritePlacement] = []

    bar_ids = list(range(len(bars)))
    rng.shuffle(bar_ids)
    target_sequence: list[int] = []
    while len(target_sequence) < direct_count:
        if len(target_sequence) < len(bar_ids):
            target_sequence.append(bar_ids[len(target_sequence)])
        else:
            target_sequence.append(rng.randrange(len(bars)))

    for bar_id in target_sequence:
        placed = False
        for _ in range(80):
            patch = rng.choice(SHAPES)
            landing = _find_direct_landing(rng, occupied, patch, bars[bar_id])
            if landing is None:
                continue
            row, col, _hits = landing
            sprite = SpritePlacement(
                patch=patch.copy(),
                final_row=row,
                col=col,
                support_kind="diamond",
                support_bar=bar_id,
            )
            sprites.append(sprite)
            h, w = sprite.shape
            occupied[row : row + h, col : col + w] |= sprite.patch
            placed = True
            break
        if not placed:
            return None

    if want_stack:
        lower_ids = [idx for idx, sprite in enumerate(sprites) if sprite.support_kind == "diamond"]
        rng.shuffle(lower_ids)
        placed = False
        for lower_id in lower_ids:
            lower = sprites[lower_id]
            for _ in range(80):
                patch = rng.choice(SHAPES)
                landing = _find_stack_landing(rng, occupied, patch, lower, bars[lower.support_bar])
                if landing is None:
                    continue
                row, col = landing
                sprite = SpritePlacement(
                    patch=patch.copy(),
                    final_row=row,
                    col=col,
                    support_kind="stack",
                    support_bar=lower.support_bar,
                    support_from=lower_id,
                )
                sprites.append(sprite)
                h, w = sprite.shape
                occupied[row : row + h, col : col + w] |= sprite.patch
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    _assign_colors(rng, sprites)
    return sprites


def _validate_scene(inp: np.ndarray, out: np.ndarray, bars: list[Bar], sprites: list[SpritePlacement]) -> bool:
    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if np.any(inp == DIAMOND_COLOR):
        return False
    if not _same_output(out, main(inp)):
        return False
    if np.array_equal(inp, out):
        return False
    if len({sprite.color for sprite in sprites}) < 2:
        return False
    if not any(sprite.support_kind == "diamond" for sprite in sprites):
        return False
    if not any(sprite.support_kind == "stack" for sprite in sprites) and len(sprites) >= 4 and max(bar.radius for bar in bars) >= 2:
        return False
    if not any(bars[sprite.support_bar].radius >= 2 for sprite in sprites):
        return False
    min_bar_row = min(bar.row for bar in bars)
    for sprite in sprites:
        if sprite.input_row is None or sprite.input_row >= sprite.final_row:
            return False
        if sprite.input_row + sprite.shape[0] - 1 >= min_bar_row:
            return False
        if max(sprite.col, bars[sprite.support_bar].c0) > min(sprite.col + sprite.shape[1] - 1, bars[sprite.support_bar].c1):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(19, 24)
    width = rng.randint(18, 24)
    bars = _sample_bars(rng, height, width)
    if bars is None:
        return None
    sprites = _generate_sprite_scene(rng, height, width, bars)
    if sprites is None:
        return None
    if not _assign_input_rows(rng, sprites, min(bar.row for bar in bars), (height, width)):
        return None
    inp = _build_input((height, width), bars, sprites)
    out = _build_output((height, width), bars, sprites)
    if not _validate_scene(inp, out, bars, sprites):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
