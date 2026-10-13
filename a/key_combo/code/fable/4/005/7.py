from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
ARC_COLORS = list(range(10))
GEN_BACKGROUND = BLACK
GEN_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = DIR4
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_sprite_crops(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    groups: dict[tuple[int, int], list[tuple[tuple[int, int, int, int], np.ndarray]]] = {}
    for coords in _components(grid != bg, connectivity=4):
        box = _bbox(coords)
        patch = _crop(grid, box)
        if coords.shape[0] != patch.size:
            continue
        if not bool(np.all(patch != bg)):
            continue
        groups.setdefault(tuple(map(int, patch.shape)), []).append((box, patch))

    assert groups, "expected filled same-size sprite copies"
    max_count = max(len(entries) for entries in groups.values())
    best_shapes = [shape for shape, entries in groups.items() if len(entries) == max_count]
    assert len(best_shapes) == 1 and max_count >= 3, "expected a unique repeated sprite size"

    entries = groups[best_shapes[0]]
    entries.sort(key=lambda item: item[0])
    return [patch for _box, patch in entries]


def _majority_template(stack: np.ndarray) -> np.ndarray:
    n, h, w = map(int, stack.shape)
    template = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            counts = Counter(int(stack[i, r, c]) for i in range(n))
            color, support = counts.most_common(1)[0]
            assert support > n // 2, "each sprite cell must have a strict majority color"
            template[r, c] = int(color)
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the connected shape formed by corruption locations after reconstructing the clean sprite by majority vote.

    Input structure:
    - The grid has a dominant background and several separated, filled, same-size multicolor sprite copies.
    - All copies share one hidden template with no rotation or reflection.
    - Each copy has two to four corrupted cells, no copy is clean, and any sprite-local cell is corrupted in at most one copy.

    Rule:
    - Align the same-size sprite crops and take the strict per-cell majority color to reconstruct the hidden template.
    - Compare every sprite copy against that reconstructed template in sprite-local coordinates.
    - Return a black grid of one sprite's size, coloring the union of all differing local cells grey.

    Key insight:
    - The template is only an intermediate consensus: once it is reconstructed, the deviations from that consensus spell the output shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    sprites = _find_sprite_crops(g, bg)
    stack = np.stack(sprites, axis=0)
    template = _majority_template(stack)

    diffs = stack != template[None, :, :]
    errors_per_copy = np.sum(diffs, axis=(1, 2))
    assert bool(np.all((2 <= errors_per_copy) & (errors_per_copy <= 4))), "each copy must have two to four corruptions"
    assert bool(np.all(np.sum(diffs, axis=0) <= 1)), "a local cell cannot be corrupted in two copies"

    error_mask = np.any(diffs, axis=0)
    assert bool(np.any(error_mask)), "expected at least one corruption"
    assert len(_components(error_mask, connectivity=4)) == 1, "corruption positions must form one connected shape"

    out = np.zeros(template.shape, dtype=int)
    out[error_mask] = GREY
    return out


# ---------------- generator ----------------

GEN_COPY_COUNT = 5
GEN_SPRITE_H_RANGE = (5, 8)
GEN_SPRITE_W_RANGE = (5, 8)
GEN_MAX_ATTEMPTS = 600
_GEN_CALL_COUNT = 0


def _random_split(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _make_template(h: int, w: int) -> tuple[np.ndarray, list[int]]:
    color_count = random.randint(4, min(7, len(GEN_PALETTE)))
    palette = random.sample(GEN_PALETTE, color_count)
    values: list[int] = []
    for color in palette:
        values.extend([int(color), int(color)])
    while len(values) < h * w:
        values.append(int(random.choice(palette)))
    random.shuffle(values)
    template = np.array(values[: h * w], dtype=int).reshape(h, w)
    assert set(map(int, np.unique(template))) == set(palette)
    return template, palette


def _neighbor_count(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in DIR4)


def _degree_stats(mask: np.ndarray) -> tuple[int, bool]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    max_degree = 0
    has_block = False
    for r, c in cells:
        max_degree = max(max_degree, _neighbor_count((r, c), cells))
        if (r + 1, c) in cells and (r, c + 1) in cells and (r + 1, c + 1) in cells:
            has_block = True
    return max_degree, has_block


def _mask_quality(mask: np.ndarray, style: str) -> bool:
    if len(_components(mask, connectivity=4)) != 1:
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    box_h, box_w = r1 - r0 + 1, c1 - c0 + 1
    if box_h < 3 or box_w < 3:
        return False
    if box_h * box_w - int(mask.sum()) < 2:
        return False
    max_degree, has_block = _degree_stats(mask)
    if style == "branch" and max_degree < 3:
        return False
    if style == "compact" and not has_block:
        return False
    return True


def _connected_error_mask(h: int, w: int, area: int, style: str) -> np.ndarray | None:
    for _ in range(400):
        start = (random.randrange(h), random.randrange(w))
        cells: set[tuple[int, int]] = {start}
        tip = start
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break

            choices = list(frontier)
            if style == "path":
                r, c = tip
                tip_choices = [
                    (r + dr, c + dc)
                    for dr, dc in DIR4
                    if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
                ]
                if tip_choices and random.random() < 0.75:
                    choices = tip_choices
            elif style == "branch":
                branched = [cell for cell in choices if _neighbor_count(cell, cells) >= 2]
                if branched and random.random() < 0.65:
                    choices = branched

            new_cell = random.choice(choices)
            cells.add(new_cell)
            if style == "path" or random.random() < 0.35:
                tip = new_cell

        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _mask_quality(mask, style):
            return mask
    return None


def _corruption_counts() -> list[int]:
    counts = [2, 4] + [random.randint(2, 4) for _ in range(GEN_COPY_COUNT - 2)]
    random.shuffle(counts)
    return counts


def _partition_errors(mask: np.ndarray, counts: list[int]) -> list[list[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    groups: list[list[tuple[int, int]]] = []
    start = 0
    for count in counts:
        groups.append(coords[start : start + count])
        start += count
    assert start == len(coords)
    return groups


def _corrupt_sprites(template: np.ndarray, palette: list[int], groups: list[list[tuple[int, int]]]) -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    for group in groups:
        sprite = template.copy()
        used_in_copy: set[int] = set()
        for r, c in group:
            true_color = int(template[r, c])
            choices = [color for color in palette if color != true_color and color not in used_in_copy]
            if not choices:
                choices = [color for color in palette if color != true_color]
            new_color = int(random.choice(choices))
            sprite[r, c] = new_color
            used_in_copy.add(new_color)
        sprites.append(sprite)
    return sprites


def _sprite_positions(sprite_h: int, sprite_w: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    min_h = 2 * sprite_h + 4
    min_w = 3 * sprite_w + 6
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    top_extra, row_gap_extra, _bottom_extra = _random_split(height - min_h, 3)
    left_extra, gap1_extra, gap2_extra, _right_extra = _random_split(width - min_w, 4)

    row0 = 1 + top_extra
    row1 = row0 + sprite_h + 2 + row_gap_extra
    col0 = 1 + left_extra
    col1 = col0 + sprite_w + 2 + gap1_extra
    col2 = col1 + sprite_w + 2 + gap2_extra

    slots = [(r, c) for r in (row0, row1) for c in (col0, col1, col2)]
    random.shuffle(slots)
    return (height, width), slots[:GEN_COPY_COUNT]


def _expected_output(error_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(error_mask.shape, dtype=int)
    out[error_mask] = GREY
    return out


def _semantic_check(grid: np.ndarray, template: np.ndarray, expected: np.ndarray) -> bool:
    try:
        sprites = _find_sprite_crops(grid, _get_bg(grid))
        if len(sprites) != GEN_COPY_COUNT:
            return False
        stack = np.stack(sprites, axis=0)
        rebuilt = _majority_template(stack)
        if not np.array_equal(rebuilt, template):
            return False
        diffs = stack != rebuilt[None, :, :]
        if not bool(np.all(np.sum(diffs, axis=0) <= 1)):
            return False
        counts = np.sum(diffs, axis=(1, 2))
        if not bool(np.all((2 <= counts) & (counts <= 4))):
            return False
        union = np.any(diffs, axis=0)
        if len(_components(union, connectivity=4)) != 1:
            return False
        corrupt_colors = set(map(int, stack[diffs]))
        template_colors = set(map(int, template.ravel()))
        if not corrupt_colors <= template_colors:
            return False
        if not np.array_equal(_expected_output(union), expected):
            return False
        if np.array_equal(main(grid), expected):
            return True
    except Exception:
        return False
    return False


def _gen_attempt(style: str) -> np.ndarray | None:
    sprite_h = random.randint(*GEN_SPRITE_H_RANGE)
    sprite_w = random.randint(*GEN_SPRITE_W_RANGE)
    template, palette = _make_template(sprite_h, sprite_w)

    counts = _corruption_counts()
    error_mask = _connected_error_mask(sprite_h, sprite_w, sum(counts), style)
    if error_mask is None:
        return None

    groups = _partition_errors(error_mask, counts)
    sprites = _corrupt_sprites(template, palette, groups)
    random.shuffle(sprites)

    (height, width), positions = _sprite_positions(sprite_h, sprite_w)
    grid = np.full((height, width), GEN_BACKGROUND, dtype=int)
    for sprite, (r, c) in zip(sprites, positions):
        _paste(grid, sprite, r, c)

    expected = _expected_output(error_mask)
    if not _semantic_check(grid, template, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    styles = ["branch", "path", "compact"]
    preferred = styles[_GEN_CALL_COUNT % len(styles)]
    _GEN_CALL_COUNT += 1

    for i in range(GEN_MAX_ATTEMPTS):
        style = preferred if i < GEN_MAX_ATTEMPTS // 2 else random.choice(styles)
        grid = _gen_attempt(style)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corrupted-sprite consensus input")
