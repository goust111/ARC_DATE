from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREEN = 3
GREY = 5
COLOR_POOL = (1, 2, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_ROT_K = {
    (-1, 0): 0,  # green above seed -> keep orientation
    (0, 1): 3,   # green right of seed -> rotate clockwise 90
    (1, 0): 2,   # green below seed -> rotate 180
    (0, -1): 1,  # green left of seed -> rotate counter-clockwise 90
}
GEN_RETRIES = 300


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_to_non_bg(arr: np.ndarray) -> np.ndarray:
    mask = arr != BG
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_anchor(shape: tuple[int, int], anchor: tuple[int, int], k: int) -> tuple[int, int]:
    h, w = shape
    r, c = anchor
    for _ in range(k % 4):
        r, c = w - 1 - c, r
        h, w = w, h
    return int(r), int(c)


def _rotate_pattern(pattern: np.ndarray, direction: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    k = DIR_TO_ROT_K[direction]
    anchor = (pattern.shape[0] - 1, pattern.shape[1] // 2)
    assert int(pattern[anchor]) != BG, "template anchor must be colored"
    rotated = np.rot90(pattern, k=k)
    return rotated, _rotate_anchor(pattern.shape, anchor, k)


def _inside_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _parse_dictionary(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, np.ndarray]]:
    grey_mask = grid == GREY
    assert np.any(grey_mask), "dictionary frame not found"
    dict_bbox = _bbox_from_mask(grey_mask)
    r0, c0, r1, c1 = dict_bbox
    assert r1 == grid.shape[0] - 1, "dictionary should sit at the bottom"

    separators = [
        c
        for c in range(c0, c1 + 1)
        if np.all(grid[r0 : r1 + 1, c] == GREY)
    ]
    assert len(separators) >= 3, "expected at least two dictionary cells"

    templates: dict[int, np.ndarray] = {}
    for left, right in zip(separators, separators[1:]):
        if right - left <= 1:
            continue
        cell = grid[r0 + 1 : r1, left + 1 : right]
        mask = (cell != BG) & (cell != GREY)
        assert np.any(mask), "dictionary cell must contain a template"
        colors = sorted(set(map(int, cell[mask])))
        assert len(colors) == 1, "template cell must use a single color"
        color = colors[0]
        assert color != GREEN, "template color cannot be green"
        pattern = _crop_to_non_bg(np.where(mask, cell, BG))
        assert pattern.shape[1] % 2 == 1, "template width must be odd"
        assert int(pattern[-1, pattern.shape[1] // 2]) == color, "template anchor must be bottom-center"
        templates[color] = pattern

    assert templates, "no templates extracted from dictionary"
    return dict_bbox, templates


def _find_seed_records(
    grid: np.ndarray, dict_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int, tuple[int, int]]]:
    h, w = grid.shape
    records: dict[tuple[int, int], tuple[int, int, int, tuple[int, int]]] = {}
    for gr, gc in map(tuple, np.argwhere(grid == GREEN)):
        gr = int(gr)
        gc = int(gc)
        if _inside_bbox(gr, gc, dict_bbox):
            continue
        colored_neighbors: list[tuple[int, int]] = []
        for dr, dc in DIRS4:
            nr = gr + dr
            nc = gc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if _inside_bbox(nr, nc, dict_bbox):
                continue
            val = int(grid[nr, nc])
            if val in (BG, GREEN, GREY):
                continue
            colored_neighbors.append((nr, nc))
        if len(colored_neighbors) != 1:
            continue
        sr, sc = colored_neighbors[0]
        direction = (gr - sr, gc - sc)
        records[(sr, sc)] = (sr, sc, int(grid[sr, sc]), direction)

    seeds = sorted(records.values(), key=lambda item: (item[0], item[1]))
    assert seeds, "no seed groups found"
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow dictionary-matched templates from colored seeds using adjacent green direction markers.

    Input structure:
    - Black background fills most of the grid.
    - A grey-framed dictionary rectangle sits at the bottom and is split into cells by grey vertical lines.
    - Each dictionary cell contains one monochrome template whose color keys the corresponding seed color.
    - Above the dictionary, each seed group has one colored seed pixel and one orthogonally adjacent green marker.

    Rule:
    - Match each seed color to the template with the same color in the dictionary.
    - Decode the green marker position relative to the seed as the template rotation.
    - Stamp the rotated template so the seed lands on the corresponding edge midpoint anchor.
    - Remove the green markers and keep the dictionary unchanged.
    """
    g = np.asarray(grid, dtype=int)
    dict_bbox, templates = _parse_dictionary(g)
    seeds = _find_seed_records(g, dict_bbox)

    out = g.copy()
    for gr, gc in map(tuple, np.argwhere(g == GREEN)):
        gr = int(gr)
        gc = int(gc)
        if not _inside_bbox(gr, gc, dict_bbox):
            out[gr, gc] = BG

    for sr, sc, color, direction in seeds:
        pattern = templates[color]
        rotated, anchor = _rotate_pattern(pattern, direction)
        out[sr, sc] = BG
        top = sr - anchor[0]
        left = sc - anchor[1]
        for r, c in map(tuple, np.argwhere(rotated != BG)):
            rr = top + int(r)
            cc = left + int(c)
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                continue
            if _inside_bbox(rr, cc, dict_bbox):
                continue
            out[rr, cc] = int(rotated[r, c])

    r0, c0, r1, c1 = dict_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _random_template_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        canvas_h = rng.randint(3, 5)
        canvas_w = rng.choice((3, 5))
        anchor = (canvas_h - 1, canvas_w // 2)
        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        mask[anchor] = True
        target = rng.randint(3, min(canvas_h * canvas_w - 1, 7))

        frontier = [anchor]
        while int(mask.sum()) < target:
            base_r, base_c = rng.choice(frontier)
            choices = [
                (nr, nc)
                for nr, nc in _neighbors4(base_r, base_c, canvas_h, canvas_w)
                if not mask[nr, nc]
            ]
            if not choices:
                frontier = [tuple(rc) for rc in np.argwhere(mask)]
                if not frontier:
                    break
                continue
            nr, nc = rng.choice(choices)
            mask[nr, nc] = True
            frontier.append((nr, nc))

        rows, cols = np.where(mask)
        cropped = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        if cropped.shape[0] < 2 or cropped.shape[1] % 2 == 0:
            continue
        if not cropped[-1, cropped.shape[1] // 2]:
            continue
        if int(cropped.sum()) < 3 or int(cropped.sum()) == cropped.size:
            continue
        if not np.any(cropped[:-1, :] != 0):
            continue

        unique = {
            (np.rot90(cropped, k=k).shape, np.rot90(cropped, k=k).tobytes())
            for k in range(4)
        }
        if len(unique) < 3:
            continue
        return cropped
    raise RuntimeError("failed to generate a non-degenerate template")


def _paint_dictionary(
    grid: np.ndarray,
    templates: dict[int, np.ndarray],
    dict_r0: int,
    dict_c0: int,
    cell_h: int,
    cell_w: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    colors = list(templates)
    dict_h = cell_h + 2
    dict_w = len(colors) * cell_w + len(colors) + 1
    dict_r1 = dict_r0 + dict_h - 1
    dict_c1 = dict_c0 + dict_w - 1

    grid[dict_r0 : dict_r1 + 1, dict_c0] = GREY
    grid[dict_r0 : dict_r1 + 1, dict_c1] = GREY
    grid[dict_r0, dict_c0 : dict_c1 + 1] = GREY
    grid[dict_r1, dict_c0 : dict_c1 + 1] = GREY

    for idx, color in enumerate(colors):
        left = dict_c0 + idx * (cell_w + 1)
        sep = left + cell_w + 1
        grid[dict_r0 : dict_r1 + 1, sep] = GREY

        cell_r0 = dict_r0 + 1
        cell_c0 = left + 1
        cell = grid[cell_r0 : cell_r0 + cell_h, cell_c0 : cell_c0 + cell_w]
        template = templates[color]
        pad_top = rng.randint(0, cell_h - template.shape[0])
        pad_left = rng.randint(0, cell_w - template.shape[1])
        mask = template != BG
        cell[pad_top : pad_top + template.shape[0], pad_left : pad_left + template.shape[1]][mask] = color

    return dict_r0, dict_c0, dict_r1, dict_c1


def _mark_buffer(mask: np.ndarray, cells: list[tuple[int, int]], radius: int) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                mask[rr, cc] = True


def _place_seed_groups(
    grid: np.ndarray,
    templates: dict[int, np.ndarray],
    top_h: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    input_blocked = np.zeros((top_h, w), dtype=bool)
    output_blocked = np.zeros((top_h, w), dtype=bool)

    colors = list(templates)
    seed_count = rng.randint(max(4, len(colors)), min(8, len(colors) + 4))
    seed_colors = colors[:]
    while len(seed_colors) < seed_count:
        seed_colors.append(rng.choice(colors))
    rng.shuffle(seed_colors)

    directions = [DIRS4[i % 4] for i in range(seed_count)]
    rng.shuffle(directions)
    specs = list(zip(seed_colors, directions))

    for color, direction in specs:
        rotated, anchor = _rotate_pattern(templates[color], direction)
        paint_cells = [tuple(map(int, rc)) for rc in np.argwhere(rotated != BG)]
        placed = False
        for _ in range(400):
            sr = rng.randint(1, top_h - 2)
            sc = rng.randint(1, w - 2)
            gr = sr + direction[0]
            gc = sc + direction[1]
            if not (0 <= gr < top_h and 0 <= gc < w):
                continue
            top = sr - anchor[0]
            left = sc - anchor[1]

            absolute = []
            ok = True
            for pr, pc in paint_cells:
                rr = top + pr
                cc = left + pc
                if not (0 <= rr < top_h and 0 <= cc < w):
                    ok = False
                    break
                absolute.append((rr, cc))
            if not ok:
                continue
            if input_blocked[sr, sc] or input_blocked[gr, gc]:
                continue
            if any(output_blocked[rr, cc] for rr, cc in absolute):
                continue

            grid[sr, sc] = color
            grid[gr, gc] = GREEN
            _mark_buffer(input_blocked, [(sr, sc), (gr, gc)], radius=1)
            _mark_buffer(output_blocked, absolute, radius=1)
            placed = True
            break
        if not placed:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 4)
    chosen_colors = rng.sample(COLOR_POOL, n_templates)
    templates: dict[int, np.ndarray] = {}
    for color in chosen_colors:
        mask = _random_template_mask(rng)
        arr = np.zeros(mask.shape, dtype=int)
        arr[mask] = color
        templates[color] = arr

    cell_h = 6
    cell_w = 7
    dict_h = cell_h + 2
    dict_w = n_templates * cell_w + n_templates + 1

    top_h = rng.randint(9, 15)
    grid_w = rng.randint(max(dict_w, 24), max(dict_w, 24) + 10)
    grid_h = top_h + dict_h
    dict_c0 = rng.randint(0, grid_w - dict_w)
    dict_r0 = top_h

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _paint_dictionary(grid, templates, dict_r0, dict_c0, cell_h, cell_w, rng)
    if not _place_seed_groups(grid, templates, top_h, rng):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
