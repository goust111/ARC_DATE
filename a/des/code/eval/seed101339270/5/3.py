from __future__ import annotations

from collections import deque
import random

import numpy as np

_BG = 0
_WALL = 5
_CARDINAL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIAGONAL = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
_DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    h, _w = grid.shape
    candidates = []
    for r in range(h):
        if h - r - 1 < 3:
            continue
        row = grid[r]
        if int(row[0]) != _BG and bool(np.all(row == row[0])):
            candidates.append(int(r))
    assert candidates, "expected a full-width wall row above the legend"
    sep_row = max(candidates)
    return sep_row, int(grid[sep_row, 0])


def _parse_templates(legend: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    legend = np.asarray(legend, dtype=int)
    assert legend.shape[0] == 3, "legend region must be 3 rows tall"
    assert legend.shape[1] % 3 == 0, "legend width must be a multiple of 3"

    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for c0 in range(0, legend.shape[1], 3):
        patch = legend[:, c0 : c0 + 3]
        if not bool(np.any(patch != _BG)):
            continue

        center = int(patch[1, 1])
        assert center != _BG, "template center must be colored"
        assert not bool(np.any((patch != _BG) & (patch != center))), "template must be monochrome"
        assert center not in templates, "template center colors must be unique"

        dirs = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                if int(patch[dr + 1, dc + 1]) == center:
                    dirs.append((int(dr), int(dc)))
        templates[center] = tuple(sorted(dirs))

    assert templates, "expected at least one legend template"
    return templates


def _grow_mask(
    shape: tuple[int, int],
    allowed: np.ndarray,
    seed: tuple[int, int],
    dirs: tuple[tuple[int, int], ...],
    rounds: int,
) -> np.ndarray:
    h, w = shape
    filled = np.zeros((h, w), dtype=bool)
    frontier = [seed]
    filled[seed] = True

    for _ in range(int(rounds)):
        new_frontier: list[tuple[int, int]] = []
        for r, c in frontier:
            for dr, dc in dirs:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and bool(allowed[nr, nc]) and not bool(filled[nr, nc]):
                    filled[nr, nc] = True
                    new_frontier.append((nr, nc))
        if not new_frontier:
            break
        frontier = new_frontier

    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed inside its grey rectangular room according to the matching 3x3 legend template.
    The room's shorter interior side length is the growth round count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row, wall = _find_separator_row(g)
    main_grid = g[:sep_row, :].copy()
    legend = g[sep_row + 1 :, :]
    templates = _parse_templates(legend)

    h, w = main_grid.shape
    passable = main_grid != wall
    out = main_grid.copy()

    for comp in _components(passable):
        if _touches_boundary(comp, h, w):
            continue

        seed_cells = [(r, c) for r, c in comp if int(main_grid[r, c]) != _BG]
        seed_colors = {int(main_grid[r, c]) for r, c in seed_cells}
        assert len(seed_cells) == 1, "each room must contain exactly one seed"
        assert len(seed_colors) == 1, "room seeds must be monochrome"

        color = next(iter(seed_colors))
        assert color in templates, "every seed color must appear in the legend"

        r0, c0, r1, c1 = _bbox(comp)
        rounds = min(r1 - r0 + 1, c1 - c0 + 1)
        allowed = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            allowed[r, c] = True

        grown = _grow_mask((h, w), allowed, seed_cells[0], templates[color], rounds)
        out[grown] = color

    return out


def _sorted_dirs(dirs: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(dr), int(dc)) for dr, dc in dirs))


def _sample_direction_set(kind: str) -> tuple[tuple[int, int], ...]:
    if kind == "cardinal":
        return _sorted_dirs(random.sample(_CARDINAL, random.randint(1, len(_CARDINAL))))
    if kind == "diagonal":
        return _sorted_dirs(random.sample(_DIAGONAL, random.randint(1, len(_DIAGONAL))))
    if kind == "mixed":
        dirs = set(random.sample(_CARDINAL, random.randint(1, 2)))
        dirs.update(random.sample(_DIAGONAL, random.randint(1, 2)))
        extras = [d for d in _DIR8 if d not in dirs]
        if extras and random.random() < 0.5:
            dirs.update(random.sample(extras, random.randint(0, min(2, len(extras)))))
        return _sorted_dirs(list(dirs))
    return _sorted_dirs(random.sample(_DIR8, random.randint(1, min(6, len(_DIR8)))))


def _sample_templates() -> dict[int, tuple[tuple[int, int], ...]] | None:
    template_count = random.randint(3, 4)
    colors = random.sample(_COLOR_POOL, template_count)
    kinds = ["cardinal", "diagonal", "mixed"]
    if template_count == 4:
        kinds.append(random.choice(["cardinal", "diagonal", "mixed", "any"]))
    random.shuffle(kinds)

    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    used_patterns: set[tuple[tuple[int, int], ...]] = set()
    for color, kind in zip(colors, kinds, strict=False):
        for _ in range(100):
            dirs = _sample_direction_set(kind)
            if dirs not in used_patterns:
                templates[int(color)] = dirs
                used_patterns.add(dirs)
                break
        else:
            return None
    return templates


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _pick_seed_position(inner_h: int, inner_w: int) -> tuple[int, int]:
    if inner_h >= 4 and inner_w >= 4 and random.random() < 0.65:
        return random.randint(1, inner_h - 2), random.randint(1, inner_w - 2)
    return random.randint(0, inner_h - 1), random.randint(0, inner_w - 1)


def _render_main_grid(
    main_h: int,
    main_w: int,
    rooms: list[dict[str, int]],
) -> np.ndarray:
    main_grid = np.full((main_h, main_w), _BG, dtype=int)
    for room in rooms:
        r0 = int(room["top"])
        c0 = int(room["left"])
        inner_h = int(room["inner_h"])
        inner_w = int(room["inner_w"])
        color = int(room["color"])
        seed_r = int(room["seed_r"])
        seed_c = int(room["seed_c"])
        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1

        main_grid[r0, c0 : c1 + 1] = _WALL
        main_grid[r1, c0 : c1 + 1] = _WALL
        main_grid[r0 : r1 + 1, c0] = _WALL
        main_grid[r0 : r1 + 1, c1] = _WALL
        main_grid[r0 + 1 + seed_r, c0 + 1 + seed_c] = color

    return main_grid


def _expected_output(
    main_grid: np.ndarray,
    rooms: list[dict[str, int]],
    templates: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    out = main_grid.copy()
    h, w = main_grid.shape
    for room in rooms:
        r0 = int(room["top"])
        c0 = int(room["left"])
        inner_h = int(room["inner_h"])
        inner_w = int(room["inner_w"])
        color = int(room["color"])
        seed = (r0 + 1 + int(room["seed_r"]), c0 + 1 + int(room["seed_c"]))

        allowed = np.zeros((h, w), dtype=bool)
        allowed[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = True
        grown = _grow_mask((h, w), allowed, seed, templates[color], min(inner_h, inner_w))
        out[grown] = color
    return out


def _render_legend(
    main_w: int,
    templates: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray | None:
    slot_count = main_w // 3
    if slot_count < len(templates):
        return None

    legend = np.full((3, main_w), _BG, dtype=int)
    slots = random.sample(range(slot_count), len(templates))
    items = list(templates.items())
    random.shuffle(items)

    for slot, (color, dirs) in zip(slots, items, strict=False):
        patch = np.full((3, 3), _BG, dtype=int)
        patch[1, 1] = int(color)
        for dr, dc in dirs:
            patch[1 + dr, 1 + dc] = int(color)
        legend[:, 3 * slot : 3 * slot + 3] = patch

    return legend


def _generate_attempt() -> np.ndarray | None:
    templates = _sample_templates()
    if templates is None:
        return None

    room_count = random.randint(len(templates), min(len(templates) + 2, 5))
    slot_count = random.randint(max(len(templates) + 1, 6), 8)
    main_w = 3 * slot_count
    main_h = random.randint(12, 18)

    room_colors = list(templates)
    while len(room_colors) < room_count:
        room_colors.append(random.choice(list(templates)))
    random.shuffle(room_colors)

    rooms: list[dict[str, int]] = []
    for color in room_colors:
        inner_h = random.randint(2, 5 if room_count >= 5 else 6)
        inner_w = random.randint(2, 5 if room_count >= 5 else 6)
        rooms.append(
            {
                "color": int(color),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "seed_r": 0,
                "seed_c": 0,
                "top": 0,
                "left": 0,
            }
        )
    rooms.sort(key=lambda room: (room["inner_h"] + 2) * (room["inner_w"] + 2), reverse=True)

    placed_boxes: list[tuple[int, int, int, int]] = []
    for room in rooms:
        outer_h = int(room["inner_h"]) + 2
        outer_w = int(room["inner_w"]) + 2
        max_top = main_h - outer_h - 1
        max_left = main_w - outer_w - 1
        if max_top < 1 or max_left < 1:
            return None

        for _ in range(200):
            top = random.randint(1, max_top)
            left = random.randint(1, max_left)
            box = (top, left, top + outer_h - 1, left + outer_w - 1)
            if any(_boxes_too_close(box, other, gap=1) for other in placed_boxes):
                continue

            room["top"] = int(top)
            room["left"] = int(left)
            room["seed_r"], room["seed_c"] = _pick_seed_position(
                int(room["inner_h"]), int(room["inner_w"])
            )
            placed_boxes.append(box)
            break
        else:
            return None

    main_grid = _render_main_grid(main_h, main_w, rooms)
    legend = _render_legend(main_w, templates)
    if legend is None:
        return None

    inp = np.full((main_h + 4, main_w), _BG, dtype=int)
    inp[:main_h, :] = main_grid
    inp[main_h, :] = _WALL
    inp[main_h + 1 :, :] = legend

    expected = _expected_output(main_grid, rooms, templates)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
