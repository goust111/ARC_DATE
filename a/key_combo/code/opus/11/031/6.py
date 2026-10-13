from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
TOP_COLORS = (1, 2, 3, 4)
MID_FG = 6
BOT_FG = 7
SEP = 9
MASK_H = 5
TARGET_AREAS = (2, 3, 4)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_shape(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in coords}
    assert pts
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return (
        max(r for r, _ in shape) + 1,
        max(c for _, c in shape) + 1,
    )


def _shape_mask(shape: tuple[tuple[int, int], ...], color: int = 1, bg: int = BG) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _scaled_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(patch, np.ones((int(factor), int(factor)), dtype=int))


def _shape_pool() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    pools: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(2, 5):
        cur: set[tuple[tuple[int, int], ...]] = set()
        for shape in pools[area - 1]:
            pts = set(shape)
            for r, c in shape:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in pts:
                        continue
                    grown = _normalize_shape(pts | {(nr, nc)})
                    h, w = _shape_dims(grown)
                    if h <= 3 and w <= 3:
                        cur.add(grown)
        pools[area] = cur
    return {area: sorted(pools[area]) for area in TARGET_AREAS}


SHAPES_BY_AREA = _shape_pool()


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEP)]
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return int(rows[0])


def _extract_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep_r = _separator_row(grid)
    assert sep_r >= MASK_H and grid.shape[0] >= sep_r + 1 + MASK_H, "grid does not contain two 5-row masks"
    top = grid[: sep_r - MASK_H, :]
    middle = grid[sep_r - MASK_H : sep_r, :]
    bottom = grid[sep_r + 1 : sep_r + 1 + MASK_H, :]
    assert middle.shape == bottom.shape == (MASK_H, grid.shape[1])
    return top, middle, bottom


def _parse_templates(top: np.ndarray) -> list[dict]:
    templates: list[dict] = []
    for color in sorted(set(map(int, np.unique(top))) - {BG}):
        coords = np.argwhere(top == int(color))
        assert coords.size > 0
        shape = _normalize_shape(map(tuple, coords.tolist()))
        h, w = _shape_dims(shape)
        patch = np.full((h, w), BG, dtype=int)
        for r, c in shape:
            patch[r, c] = int(color)
        templates.append({"color": int(color), "shape": shape, "patch": patch})
    assert 2 <= len(templates) <= 4, "expected 2-4 top templates"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the both-background intersection of two stacked masks to choose a top template and a scale.

    Input structure:
    - The grid has an upper template strip and, below it, two 5-row masks separated by one full-width maroon row.
    - Upper templates are monochrome 4-connected shapes on black background, with distinct colors and distinct shapes.
    - The middle mask uses black and pink; the lower mask uses black and orange.

    Rule:
    - Compare the two 5-row masks cell by cell and mark the cells where both are black.
    - Normalize that both-background coordinate set by translating its bbox to the origin.
    - Find the unique upper template with exactly the same normalized shape.
    - Let n be the number of both-background cells, and output the matched template scaled by n in both axes on black background.

    Key insight:
    - The same intersection pattern carries two pieces of information at once: its shape selects which template survives, and its pixel count sets the integer enlargement factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    top, middle, bottom = _extract_regions(g)
    templates = _parse_templates(top)

    both_bg = np.argwhere((middle == BG) & (bottom == BG))
    assert both_bg.size > 0, "expected a non-empty both-background intersection"
    shape = _normalize_shape(map(tuple, both_bg.tolist()))
    scale = int(len(shape))
    assert scale in TARGET_AREAS, "intersection size must encode a 2x/3x/4x scale"

    matches = [template for template in templates if template["shape"] == shape]
    assert len(matches) == 1, "intersection shape must match exactly one template"
    template = matches[0]

    return _scaled_patch(template["patch"], scale)


def _pick_distinct_shapes(rng: random.Random, count: int, target_area: int) -> tuple[list[tuple[tuple[int, int], ...]], int]:
    target = rng.choice(SHAPES_BY_AREA[target_area])
    shapes = [target]

    same_area = [shape for shape in SHAPES_BY_AREA[target_area] if shape != target]
    if count >= 2 and same_area:
        shapes.append(rng.choice(same_area))

    target_dims = _shape_dims(target)
    same_bbox = [
        shape
        for area, pool in SHAPES_BY_AREA.items()
        for shape in pool
        if shape not in shapes and _shape_dims(shape) == target_dims
    ]
    if count >= 3 and same_bbox:
        shapes.append(rng.choice(same_bbox))

    all_shapes = [
        shape
        for area, pool in SHAPES_BY_AREA.items()
        for shape in pool
        if shape not in shapes
    ]
    rng.shuffle(all_shapes)
    while len(shapes) < count:
        shapes.append(all_shapes.pop())

    rng.shuffle(shapes)
    return shapes, shapes.index(target)


def _assign_columns(rng: random.Random, widths: list[int], total_w: int, min_gap: int = 2) -> list[int] | None:
    need = sum(widths) + min_gap * (len(widths) - 1)
    if need > total_w:
        return None
    slack = total_w - need
    extras = rng.choices(range(slack + 1), k=len(widths) + 1)
    extras.sort()
    pieces = [extras[0]]
    for i in range(1, len(extras)):
        pieces.append(extras[i] - extras[i - 1])
    pieces.append(slack - extras[-1])
    cols: list[int] = []
    c = pieces[0]
    for i, w in enumerate(widths):
        cols.append(c)
        c += w
        if i < len(widths) - 1:
            c += min_gap + pieces[i + 1]
    return cols


def _target_offset(rng: random.Random, max_r: int, max_c: int) -> tuple[int, int]:
    row_choices = [0, max_r] if max_r > 0 else [0]
    col_choices = [0, max_c] if max_c > 0 else [0]
    if max_r > 1:
        row_choices.append(rng.randrange(max_r + 1))
    if max_c > 1:
        col_choices.append(rng.randrange(max_c + 1))
    return rng.choice(row_choices), rng.choice(col_choices)


def _fill_masks(
    rng: random.Random,
    shape: tuple[tuple[int, int], ...],
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    middle = np.zeros((MASK_H, width), dtype=int)
    bottom = np.zeros((MASK_H, width), dtype=int)
    h, w = _shape_dims(shape)
    r0, c0 = _target_offset(rng, MASK_H - h, width - w)
    target = {(r0 + r, c0 + c) for r, c in shape}

    state_options = ((MID_FG, BG), (BG, BOT_FG), (MID_FG, BOT_FG))
    state_weights = rng.choice(
        (
            (4, 4, 2),
            (4, 2, 4),
            (2, 4, 4),
            (3, 3, 4),
        )
    )
    used_states: set[int] = set()

    for r in range(MASK_H):
        for c in range(width):
            if (r, c) in target:
                continue
            state = rng.choices(range(3), weights=state_weights, k=1)[0]
            used_states.add(state)
            middle[r, c], bottom[r, c] = state_options[state]

    assert len(used_states) >= 2, "need at least two non-target comparison outcomes"
    assert np.count_nonzero(middle == MID_FG) > 0 and np.count_nonzero(bottom == BOT_FG) > 0
    assert np.array_equal(np.argwhere((middle == BG) & (bottom == BG)), np.array(sorted(target)))
    return middle, bottom


def _validate_instance(inp: np.ndarray) -> np.ndarray | None:
    try:
        out = main(inp)
    except Exception:
        return None

    if inp.ndim != 2 or out.ndim != 2:
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None

    top, middle, bottom = _extract_regions(inp)
    templates = _parse_templates(top)
    both_bg = np.argwhere((middle == BG) & (bottom == BG))
    shape = _normalize_shape(map(tuple, both_bg.tolist()))
    matches = [template for template in templates if template["shape"] == shape]
    if len(matches) != 1:
        return None

    target = matches[0]
    distractors = [template for template in templates if template is not target]
    if not any(len(template["shape"]) == len(shape) for template in distractors):
        return None

    expected = _scaled_patch(target["patch"], len(shape))
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, target["patch"]):
        return None
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    template_count = rng.choice((2, 3, 4))
    target_area = rng.choice(TARGET_AREAS)
    shapes, target_index = _pick_distinct_shapes(rng, template_count, target_area)

    width_min = max(14, sum(_shape_dims(shape)[1] for shape in shapes) + 2 * (template_count - 1))
    width = rng.randint(width_min, 22)
    top_h = rng.randint(4, 6)
    total_h = top_h + MASK_H + 1 + MASK_H
    assert total_h <= 30

    top = np.zeros((top_h, width), dtype=int)
    colors = list(TOP_COLORS)
    rng.shuffle(colors)
    colors = colors[:template_count]
    patches = [_shape_mask(shape, color=color, bg=BG) for shape, color in zip(shapes, colors)]
    cols = _assign_columns(rng, [patch.shape[1] for patch in patches], width, min_gap=2)
    if cols is None:
        return None

    template_meta: list[dict] = []
    for i, (shape, color, patch, c0) in enumerate(zip(shapes, colors, patches, cols)):
        h, w = patch.shape
        r0_candidates = [0, top_h - h]
        if top_h - h >= 1:
            r0_candidates.append(rng.randrange(top_h - h + 1))
        r0 = rng.choice(r0_candidates)
        top[r0 : r0 + h, c0 : c0 + w] = patch
        template_meta.append({"shape": shape, "color": int(color), "patch": patch, "is_target": i == target_index})

    target_template = next(item for item in template_meta if item["is_target"])
    middle, bottom = _fill_masks(rng, target_template["shape"], width)

    inp = np.zeros((total_h, width), dtype=int)
    inp[:top_h, :] = top
    inp[top_h : top_h + MASK_H, :] = middle
    inp[top_h + MASK_H, :] = SEP
    inp[top_h + MASK_H + 1 :, :] = bottom

    if len(_parse_templates(top)) != template_count:
        return None
    out = _validate_instance(inp)
    if out is None:
        return None
    if np.count_nonzero(out) <= len(target_template["shape"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
