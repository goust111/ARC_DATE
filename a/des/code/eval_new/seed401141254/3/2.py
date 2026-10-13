from __future__ import annotations

import random
from collections import Counter

import numpy as np


AXIS_COLOR = 5
COLOR_POOL = tuple(range(10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
)

GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
SEED_COUNT_CHOICES = (2, 3, 4)
TEMPLATE_SIZES = ((3, 3), (3, 5), (5, 3), (5, 5))
GEN_RETRIES = 400
TEMPLATE_RETRIES = 200
PLACEMENT_RETRIES = 250


def _mode(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_axis(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    row_candidates: list[tuple[int, int]] = []
    col_candidates: list[tuple[int, int]] = []

    for r in range(h):
        if np.all(grid[r, :] != bg):
            row_candidates.append((int(np.max(np.unique(grid[r, :], return_counts=True)[1])), r))
    for c in range(w):
        if np.all(grid[:, c] != bg):
            col_candidates.append((int(np.max(np.unique(grid[:, c], return_counts=True)[1])), c))

    if row_candidates and not col_candidates:
        _, axis_idx = max(row_candidates)
        axis_color = _mode(grid[axis_idx, :])
        return "row", axis_idx, axis_color
    if col_candidates and not row_candidates:
        _, axis_idx = max(col_candidates)
        axis_color = _mode(grid[:, axis_idx])
        return "col", axis_idx, axis_color
    raise AssertionError("expected exactly one full non-background symmetry axis")


def _template_data(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(comp)
    assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1, "template bbox must be odd-sized"
    anchor = ((r0 + r1) // 2, (c0 + c1) // 2)
    assert anchor in set(comp), "template center cell must be colored"
    rel = [(r - anchor[0], c - anchor[1]) for r, c in comp]
    return anchor, rel


def _mirror_rel(rel: list[tuple[int, int]], axis_kind: str) -> list[tuple[int, int]]:
    if axis_kind == "row":
        return [(-dr, dc) for dr, dc in rel]
    return [(dr, -dc) for dr, dc in rel]


def _paint_shape(
    out: np.ndarray,
    color: int,
    seed: tuple[int, int],
    rel_coords: list[tuple[int, int]],
    axis_kind: str,
    axis_idx: int,
) -> None:
    h, w = out.shape
    sr, sc = seed
    for dr, dc in rel_coords:
        rr, cc = sr + dr, sc + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if axis_kind == "row" and rr == axis_idx and (rr, cc) != seed:
            continue
        if axis_kind == "col" and cc == axis_idx and (rr, cc) != seed:
            continue
        out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relocate each color-matched template to its seed on the grey axis and add its mirror across that axis.

    Input structure:
    - bg is the dominant background color.
    - One full grey row or one full grey column spans the grid and acts as the symmetry axis.
    - Several non-grey single-pixel seeds of distinct colors lie on that axis.
    - Away from the axis, each seed color appears exactly once more as a small connected monochrome template shape.

    Rule:
    - Find the unique grey axis and the seed pixels placed on it.
    - For each seed color, locate the matching off-axis template and treat the center cell of its odd tight bbox as the anchor.
    - Remove every off-axis template from its original location.
    - Repaint the template with its anchor aligned to the seed, and also repaint its reflection across the grey axis, clipping any out-of-bounds cells.
    - Keep the grey axis and the seed pixels themselves unchanged.

    Key insight:
    - Color identifies which template belongs to which seed, while the grey full-length line identifies the reflection axis used to stamp the template twice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _mode(g)
    axis_kind, axis_idx, axis_color = _detect_axis(g, bg)

    if axis_kind == "row":
        seeds = [(axis_idx, c, int(g[axis_idx, c])) for c in range(g.shape[1]) if int(g[axis_idx, c]) != axis_color]
    else:
        seeds = [(r, axis_idx, int(g[r, axis_idx])) for r in range(g.shape[0]) if int(g[r, axis_idx]) != axis_color]
    assert seeds, "expected at least one seed on the axis"

    out = g.copy()
    seed_colors = {color for _, _, color in seeds}
    for color in seed_colors:
        comps = _components_of_color(g, color)
        template_comp = max(comps, key=len)
        assert len(template_comp) > 1, "matching template must be larger than the 1-pixel seed"
        for r, c in template_comp:
            out[r, c] = bg

    for sr, sc, color in seeds:
        comps = _components_of_color(g, color)
        template_comp = max(comps, key=len)
        _, rel = _template_data(template_comp)
        mirrored = _mirror_rel(rel, axis_kind)
        _paint_shape(out, color, (sr, sc), rel, axis_kind, axis_idx)
        _paint_shape(out, color, (sr, sc), mirrored, axis_kind, axis_idx)

    return out


def _make_vertical_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    cr, cc = h // 2, w // 2
    mask = np.zeros((h, w), dtype=bool)

    left = rng.randint(0, cc - 1)
    right = rng.randint(cc + 1, w - 1)
    mask[cr, left : right + 1] = True

    top_col = rng.choice([c for c in range(left, right + 1) if c != cc])
    bottom_col = rng.choice([c for c in range(left, right + 1) if c != cc])
    mask[: cr + 1, top_col] = True
    mask[cr:, bottom_col] = True

    allowed = np.ones((h, w), dtype=bool)
    allowed[:, cc] = False
    allowed[cr, cc] = True
    target_area = rng.randint(int(mask.sum()), min(int(allowed.sum()), 10))

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = rng.choice(list(frontier))
        mask[nr, nc] = True

    return mask


def _make_template(axis_kind: str, color: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        h, w = rng.choice(TEMPLATE_SIZES)
        if axis_kind == "col":
            mask = _make_vertical_mask(h, w, rng)
            if rng.random() < 0.5:
                mask = np.flipud(mask)
            if rng.random() < 0.5:
                mask = np.fliplr(mask)
            if np.array_equal(mask, np.fliplr(mask)):
                continue
            if int(mask[:, w // 2].sum()) != 1:
                continue
        else:
            mask = _make_vertical_mask(w, h, rng).T
            if rng.random() < 0.5:
                mask = np.flipud(mask)
            if rng.random() < 0.5:
                mask = np.fliplr(mask)
            if np.array_equal(mask, np.flipud(mask)):
                continue
            if int(mask[h // 2, :].sum()) != 1:
                continue
        if not (np.any(mask[0, :]) and np.any(mask[-1, :]) and np.any(mask[:, 0]) and np.any(mask[:, -1])):
            continue
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = color
        return patch
    return None


def _template_rel_from_patch(patch: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != 0)]
    anchor, rel = _template_data(coords)
    assert patch[anchor] != 0
    return rel


def _visible_pattern_cells(
    shape_rel: list[tuple[int, int]],
    seed: tuple[int, int],
    axis_kind: str,
    axis_idx: int,
    shape_color: int,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    tmp = np.zeros((h, w), dtype=int)
    _paint_shape(tmp, shape_color, seed, shape_rel, axis_kind, axis_idx)
    _paint_shape(tmp, shape_color, seed, _mirror_rel(shape_rel, axis_kind), axis_kind, axis_idx)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == shape_color)}


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _choose_seed_positions(
    h: int,
    w: int,
    axis_kind: str,
    axis_idx: int,
    templates: list[np.ndarray],
    colors: list[int],
    rng: random.Random,
) -> list[tuple[int, int, int]] | None:
    line_len = w if axis_kind == "row" else h
    candidate_indices = list(range(line_len))
    rng.shuffle(candidate_indices)

    order = list(range(len(templates)))
    order.sort(key=lambda i: int(np.count_nonzero(templates[i])), reverse=True)
    placements: list[tuple[int, int, int] | None] = [None] * len(templates)
    reserved: set[tuple[int, int]] = set()

    for idx in order:
        color = colors[idx]
        rel = _template_rel_from_patch(templates[idx])
        for pos in candidate_indices:
            seed = (axis_idx, pos) if axis_kind == "row" else (pos, axis_idx)
            visible = _visible_pattern_cells(rel, seed, axis_kind, axis_idx, color, h, w)
            if seed not in visible or len(visible) < 5:
                continue
            if any(cell in reserved for cell in visible):
                continue

            if axis_kind == "row":
                sides = {np.sign(r - axis_idx) for r, _ in visible if r != axis_idx}
            else:
                sides = {np.sign(c - axis_idx) for _, c in visible if c != axis_idx}
            if sides != {-1, 1}:
                continue

            placements[idx] = (seed[0], seed[1], color)
            reserved |= _dilate(visible, h, w)
            break
        else:
            return None

    return [item for item in placements if item is not None]


def _place_templates_in_input(
    grid: np.ndarray,
    axis_kind: str,
    axis_idx: int,
    templates: list[np.ndarray],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    occupied = grid != _mode(grid)
    placements: list[tuple[int, int] | None] = [None] * len(templates)
    order = list(range(len(templates)))
    order.sort(key=lambda i: templates[i].shape[0] * templates[i].shape[1], reverse=True)

    for idx in order:
        patch = templates[idx]
        ph, pw = patch.shape
        for _ in range(PLACEMENT_RETRIES):
            if axis_kind == "row":
                sides = []
                if axis_idx - ph - 1 >= 0:
                    sides.append("top")
                if axis_idx + 2 <= h - ph:
                    sides.append("bottom")
                if not sides:
                    return None
                side = rng.choice(sides)
                if side == "top":
                    r0 = rng.randint(0, axis_idx - ph - 1)
                else:
                    r0 = rng.randint(axis_idx + 2, h - ph)
                c0 = rng.randint(0, w - pw)
            else:
                sides = []
                if axis_idx - pw - 1 >= 0:
                    sides.append("left")
                if axis_idx + 2 <= w - pw:
                    sides.append("right")
                if not sides:
                    return None
                side = rng.choice(sides)
                if side == "left":
                    c0 = rng.randint(0, axis_idx - pw - 1)
                else:
                    c0 = rng.randint(axis_idx + 2, w - pw)
                r0 = rng.randint(0, h - ph)

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            region = grid[r0 : r0 + ph, c0 : c0 + pw]
            region[patch != 0] = patch[patch != 0]
            occupied[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            break
        else:
            return None

    return [item for item in placements if item is not None]


def _render_expected(
    shape: tuple[int, int],
    bg: int,
    axis_kind: str,
    axis_idx: int,
    seeds: list[tuple[int, int, int]],
    templates: list[np.ndarray],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    if axis_kind == "row":
        out[axis_idx, :] = AXIS_COLOR
    else:
        out[:, axis_idx] = AXIS_COLOR

    for r, c, color in seeds:
        out[r, c] = color
    for (r, c, color), patch in zip(seeds, templates):
        rel = _template_rel_from_patch(patch)
        _paint_shape(out, color, (r, c), rel, axis_kind, axis_idx)
        _paint_shape(out, color, (r, c), _mirror_rel(rel, axis_kind), axis_kind, axis_idx)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    axis_kind = rng.choice(["row", "col"])
    axis_idx = rng.randint(2, h - 3) if axis_kind == "row" else rng.randint(2, w - 3)

    bg_choices = [c for c in COLOR_POOL if c != AXIS_COLOR]
    bg = rng.choice(bg_choices)
    seed_count = rng.choice(SEED_COUNT_CHOICES)
    seed_colors = rng.sample([c for c in COLOR_POOL if c not in {0, bg, AXIS_COLOR}], seed_count)

    templates: list[np.ndarray] = []
    for color in seed_colors:
        patch = _make_template(axis_kind, color, rng)
        if patch is None:
            return None
        templates.append(patch)

    seeds = _choose_seed_positions(h, w, axis_kind, axis_idx, templates, seed_colors, rng)
    if seeds is None:
        return None

    expected = _render_expected((h, w), bg, axis_kind, axis_idx, seeds, templates)
    inp = np.full((h, w), bg, dtype=int)
    if axis_kind == "row":
        inp[axis_idx, :] = AXIS_COLOR
    else:
        inp[:, axis_idx] = AXIS_COLOR
    for r, c, color in seeds:
        inp[r, c] = color

    placements = _place_templates_in_input(inp, axis_kind, axis_idx, templates, rng)
    if placements is None:
        return None

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
