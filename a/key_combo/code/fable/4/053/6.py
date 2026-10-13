from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


Coord = tuple[int, int]
Placement = tuple[int, int, int]  # template index, top row, left col


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, np.asarray(grid).ravel())))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _color_info(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == int(color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "color": int(color),
        "coords": [(int(r), int(c)) for r, c in coords],
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "area": int(mask.sum()),
    }


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_from_cells(cells: set[Coord]) -> list[set[Coord]]:
    remaining = set(cells)
    comps: list[set[Coord]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                nxt = (int(nr), int(nc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _mask_components(mask: np.ndarray) -> list[set[Coord]]:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    return _components_from_cells(cells)


def _is_connected(cells: set[Coord]) -> bool:
    return len(_components_from_cells(cells)) == 1


def _touches(a: set[Coord], b: set[Coord]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (int(nr), int(nc)) in b:
                return True
    return False


def _transform_mask(mask: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, rot_k)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _placements_covering_cells(
    cells: list[Coord],
    template_masks: list[np.ndarray],
    grid_shape: tuple[int, int],
) -> list[Placement]:
    h_grid, w_grid = map(int, grid_shape)
    placements: list[Placement] = []
    for template_idx, mask in enumerate(template_masks):
        h_mask, w_mask = map(int, mask.shape)
        for r0 in range(h_grid - h_mask + 1):
            for c0 in range(w_grid - w_mask + 1):
                ok = True
                for r, c in cells:
                    rr = int(r - r0)
                    cc = int(c - c0)
                    if rr < 0 or rr >= h_mask or cc < 0 or cc >= w_mask or not bool(mask[rr, cc]):
                        ok = False
                        break
                if ok:
                    placements.append((int(template_idx), int(r0), int(c0)))
    return placements


def _placement_cells(mask: np.ndarray, r0: int, c0: int) -> list[Coord]:
    coords = np.argwhere(mask)
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _try_solve_with_pair(grid: np.ndarray, info_a: dict, info_b: dict) -> np.ndarray | None:
    template_masks = [info_a["mask"], info_b["mask"]]
    template_colors = {int(info_a["color"]), int(info_b["color"])}
    out = np.asarray(grid, dtype=int).copy()
    bg = get_bg(out)

    infos = [_color_info(out, color) for color in colors_present(out, exclude={bg})]
    for info in infos:
        color = int(info["color"])
        if color in template_colors:
            continue
        placements = _placements_covering_cells(info["coords"], template_masks, out.shape)
        if len(placements) != 1:
            return None
        template_idx, r0, c0 = placements[0]
        for rr, cc in _placement_cells(template_masks[template_idx], r0, c0):
            existing = int(out[rr, cc])
            if existing not in (bg, color):
                return None
            out[rr, cc] = color
    return out


def _select_template_pair(grid: np.ndarray) -> tuple[dict, dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    infos = [_color_info(g, color) for color in colors_present(g, exclude={bg})]
    candidates: list[tuple[tuple[int, int, int], dict, dict]] = []
    for info_a, info_b in combinations(infos, 2):
        if info_a["mask"].shape != info_b["mask"].shape:
            continue
        if np.array_equal(info_a["mask"], info_b["mask"]):
            continue
        score = (
            int(info_a["area"] + info_b["area"]),
            int(min(info_a["area"], info_b["area"])),
            -abs(int(info_a["area"]) - int(info_b["area"])),
        )
        candidates.append((score, info_a, info_b))

    candidates.sort(key=lambda item: item[0], reverse=True)
    valid: list[tuple[dict, dict]] = []
    for _score, info_a, info_b in candidates:
        if _try_solve_with_pair(g, info_a, info_b) is not None:
            valid.append((info_a, info_b))

    assert len(valid) == 1, "expected a unique valid template pair"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each fragment color into the unique translated copy of one of two full templates.

    Input structure:
    - A uniform background contains two complete single-color template sprites with the same bounding-box size but different shapes.
    - Every other foreground color is one translated fragment set from exactly one of those templates, possibly split into multiple 4-connected pieces.
    - At least one fragment color may have an individual piece that fits both templates, while the same-color union fits only one placement.

    Rule:
    - Identify the two complete template colors as the unique same-size, different-shape pair that lets every other color be covered by exactly one translated placement from one of the templates.
    - For each non-template color, search all placements of both templates and keep the single placement whose template cells cover every pixel of that color.
    - Paint the whole template shape at that placement in the fragment color, leaving the two original templates and the background unchanged.

    Key insight:
    - Color groups must be solved jointly: a single fragment can be template-ambiguous, but the full same-color set anchors one exact template and translation.
    """
    g = np.asarray(grid, dtype=int)
    info_a, info_b = _select_template_pair(g)
    solved = _try_solve_with_pair(g, info_a, info_b)
    assert solved is not None
    return solved


BASE_TEMPLATE_PAIRS = [
    (
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    (
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
]

GRID_MIN = 18
GRID_MAX = 30
BOX_GAP = 2
FRAGMENT_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _sample_template_pair() -> tuple[np.ndarray, np.ndarray]:
    base_a, base_b = random.choice(BASE_TEMPLATE_PAIRS)
    rot_k = random.randrange(4)
    flip_lr = random.choice([False, True])
    return _transform_mask(base_a, rot_k, flip_lr), _transform_mask(base_b, rot_k, flip_lr)


def _can_place_box(placed: list[tuple[int, int]], size: int, r0: int, c0: int) -> bool:
    for pr, pc in placed:
        if not (
            r0 + size + BOX_GAP <= pr
            or pr + size + BOX_GAP <= r0
            or c0 + size + BOX_GAP <= pc
            or pc + size + BOX_GAP <= c0
        ):
            return False
    return True


def _sample_box_positions(grid_shape: tuple[int, int], size: int, count: int) -> list[tuple[int, int]] | None:
    h, w = map(int, grid_shape)
    positions: list[tuple[int, int]] = []
    for _ in range(3000):
        if len(positions) == count:
            return positions
        r0 = random.randint(0, h - size)
        c0 = random.randint(0, w - size)
        if _can_place_box(positions, size, r0, c0):
            positions.append((int(r0), int(c0)))
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[int(r0 + rr), int(c0 + cc)] = int(color)


def _placed_cells(local_cells: set[Coord], top_left: tuple[int, int]) -> list[Coord]:
    r0, c0 = top_left
    return [(int(r0 + r), int(c0 + c)) for r, c in sorted(local_cells)]


def _random_unique_fragment(
    template_masks: list[np.ndarray],
    template_idx: int,
    top_left: tuple[int, int],
    grid_shape: tuple[int, int],
) -> set[Coord] | None:
    template_cells = {(int(r), int(c)) for r, c in np.argwhere(template_masks[template_idx])}
    cells_list = list(template_cells)
    for _ in range(300):
        keep_size = random.randint(2, len(cells_list) - 1)
        kept = set(random.sample(cells_list, keep_size))
        comps = _components_from_cells(kept)
        if not (1 <= len(comps) <= 3):
            continue
        placements = _placements_covering_cells(
            _placed_cells(kept, top_left),
            template_masks,
            grid_shape,
        )
        if placements == [(int(template_idx), int(top_left[0]), int(top_left[1]))]:
            return kept
    return None


def _special_ambiguous_fragment(
    template_masks: list[np.ndarray],
    template_idx: int,
    top_left: tuple[int, int],
    grid_shape: tuple[int, int],
) -> set[Coord] | None:
    chosen = template_masks[template_idx]
    other = template_masks[1 - template_idx]
    common_components = _mask_components(chosen & other)
    random.shuffle(common_components)

    for shared in common_components:
        if len(shared) < 2:
            continue
        shared_places = _placements_covering_cells(
            _placed_cells(shared, top_left),
            template_masks,
            grid_shape,
        )
        if {idx for idx, _r, _c in shared_places} != {0, 1}:
            continue

        exclusive_cells = {
            (int(r), int(c))
            for r, c in np.argwhere(chosen & ~other)
            if (int(r), int(c)) not in shared
        }
        candidate_subsets: list[set[Coord]] = []
        for size in (2, 3, 1):
            if len(exclusive_cells) < size:
                continue
            for picked in combinations(sorted(exclusive_cells), size):
                subset = set(picked)
                if not _is_connected(subset):
                    continue
                if _touches(subset, shared):
                    continue
                candidate_subsets.append(subset)
            if candidate_subsets:
                break
        random.shuffle(candidate_subsets)

        for anchor in candidate_subsets:
            fragment = shared | anchor
            placements = _placements_covering_cells(
                _placed_cells(fragment, top_left),
                template_masks,
                grid_shape,
            )
            if placements == [(int(template_idx), int(top_left[0]), int(top_left[1]))]:
                return fragment
    return None


def _expected_output(
    grid: np.ndarray,
    template_masks: list[np.ndarray],
    assignments: list[tuple[int, tuple[int, int], int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for template_idx, (r0, c0), color in assignments:
        _paint_mask(out, template_masks[template_idx], r0, c0, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    template_a, template_b = _sample_template_pair()
    size = int(template_a.shape[0])
    fragment_count = random.randint(*FRAGMENT_COUNT_RANGE)

    min_h = max(GRID_MIN, size * 2 + BOX_GAP + 1)
    min_w = max(GRID_MIN, size * 2 + BOX_GAP + 1)
    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)

    positions = _sample_box_positions((h, w), size, fragment_count + 2)
    if positions is None:
        return None
    random.shuffle(positions)

    bg = random.randint(0, 9)
    needed_colors = fragment_count + 2
    palette = [color for color in range(10) if color != bg]
    if len(palette) < needed_colors:
        return None
    chosen_colors = random.sample(palette, needed_colors)
    template_colors = chosen_colors[:2]
    fragment_colors = chosen_colors[2:]

    grid = np.full((h, w), bg, dtype=int)
    template_masks = [template_a, template_b]

    template_positions = positions[:2]
    fragment_positions = positions[2:]
    for template_idx, (r0, c0) in enumerate(template_positions):
        _paint_mask(grid, template_masks[template_idx], r0, c0, template_colors[template_idx])

    special_idx = random.randrange(fragment_count)
    chosen_templates = [random.randrange(2) for _ in range(fragment_count)]
    if fragment_count >= 2:
        chosen_templates[0] = random.randrange(2)
        chosen_templates[1] = 1 - chosen_templates[0]
    chosen_templates[special_idx] = random.randrange(2)

    expected_assignments: list[tuple[int, tuple[int, int], int]] = []
    for frag_idx in range(fragment_count):
        template_idx = chosen_templates[frag_idx]
        top_left = fragment_positions[frag_idx]
        color = fragment_colors[frag_idx]
        if frag_idx == special_idx:
            local_cells = _special_ambiguous_fragment(template_masks, template_idx, top_left, grid.shape)
        else:
            local_cells = _random_unique_fragment(template_masks, template_idx, top_left, grid.shape)
        if local_cells is None:
            return None

        for rr, cc in _placed_cells(local_cells, top_left):
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
        expected_assignments.append((template_idx, top_left, color))

    expected = _expected_output(grid, template_masks, expected_assignments)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-template anchoring grid")
