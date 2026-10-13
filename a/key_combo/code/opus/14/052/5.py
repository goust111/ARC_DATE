from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
RIGHT_WALL = 1
LEFT_WALL = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _coords_have_hole(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0])
    local[shifted[:, 0], shifted[:, 1]] = True
    for hole in _components(~local):
        if not np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == local.shape[0] - 1)
            | (hole[:, 1] == local.shape[1] - 1)
        ):
            return True
    return False


def _component_order(coords: np.ndarray, direction: int) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == -1:
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    return (-int(cols.max()), int(rows.min()), -int(cols.min()), int(rows.max()))


def _extract_objects(grid: np.ndarray, color: int, wall_col: int) -> list[np.ndarray]:
    movable = grid == int(color)
    movable[:, wall_col] = False
    return _components(movable)


def _settle_components(
    grid: np.ndarray, color: int, wall_col: int, direction: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    comps = _extract_objects(grid, color, wall_col)
    comps.sort(key=lambda coords: _component_order(coords, direction))
    settled = np.zeros(grid.shape, dtype=bool)
    settled[:, wall_col] = True
    placements: list[tuple[np.ndarray, np.ndarray]] = []

    for coords in comps:
        shift = 0
        while True:
            candidate = coords + np.array([0, shift + direction])
            if int(candidate[:, 1].min()) < 0 or int(candidate[:, 1].max()) >= grid.shape[1]:
                break
            if np.any(settled[candidate[:, 0], candidate[:, 1]]):
                break
            shift += direction
        placed = coords + np.array([0, shift])
        assert not np.any(settled[placed[:, 0], placed[:, 1]])
        settled[placed[:, 0], placed[:, 1]] = True
        placements.append((coords, placed))

    return placements


def _solve(grid: np.ndarray, recolor_holes: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    left_color = int(g[0, 0])
    right_color = int(g[0, -1])
    assert g.ndim == 2 and g.size > 0
    assert left_color != right_color
    assert bg not in {left_color, right_color}
    assert np.all(g[:, 0] == left_color)
    assert np.all(g[:, -1] == right_color)

    out = np.full_like(g, bg)
    out[:, 0] = left_color
    out[:, -1] = right_color
    occupied = out != bg

    for color, wall_col, direction, opposite in (
        (left_color, 0, -1, right_color),
        (right_color, g.shape[1] - 1, 1, left_color),
    ):
        for original, placed in _settle_components(g, color, wall_col, direction):
            final_color = int(color)
            if recolor_holes and _coords_have_hole(placed):
                final_color = int(opposite)
            assert not np.any(occupied[placed[:, 0], placed[:, 1]])
            out[placed[:, 0], placed[:, 1]] = final_color
            occupied[placed[:, 0], placed[:, 1]] = True
            assert placed.shape == original.shape

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each inner object toward its same-colored wall, then recolor only the hollow ones to the opposite wall color.

    Input structure:
    - The background is black, the full left column is a red wall, and the full right column is a blue wall.
    - Inside the walls are several 4-connected monochrome red or blue objects, each treated as a rigid shape.
    - Some objects enclose one or more black 4-connected holes inside their own shape, while others are solid.

    Rule:
    - Red objects slide horizontally left toward the red wall, and blue objects slide horizontally right toward the blue wall.
    - Within one color, process objects from nearest to farthest from their target wall; each object stops just before the wall or an already settled object of that same color.
    - After all objects have settled, inspect each moved object's topology: if it encloses at least one black hole, recolor that whole object to the opposite wall color while keeping the hole black.
    - Solid objects keep their original color, and both wall columns remain unchanged.

    Key insight:
    - Motion depends on the object's current color, but final color depends only on whether the moved shape is hollow, so an object can park beside its own wall and then visually betray it.
    """
    return _solve(grid, recolor_holes=True)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return _trim_mask(out.copy())


def _ring_mask(missing_corner: tuple[int, int] | None) -> np.ndarray:
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    if missing_corner is not None:
        mask[missing_corner] = False
    return mask


HOLE_MASKS = [
    _ring_mask(None),
    _ring_mask((0, 0)),
    _ring_mask((0, 2)),
    _ring_mask((2, 0)),
    _ring_mask((2, 2)),
]


def _random_hole_mask() -> np.ndarray:
    return _random_transform(random.choice(HOLE_MASKS))


def _random_solid_mask() -> np.ndarray:
    for _ in range(200):
        area = random.randint(4, 8)
        canvas = np.zeros((4, 4), dtype=bool)
        start = (random.randint(0, 3), random.randint(0, 3))
        canvas[start] = True
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(canvas.shape, r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            chosen = random.choice(tuple(frontier))
            cells.add(chosen)
            canvas[chosen] = True
        if len(cells) != area:
            continue
        mask = _trim_mask(canvas)
        if _coords_have_hole(np.argwhere(mask)):
            continue
        if max(mask.shape) > 4:
            continue
        return _random_transform(mask)
    raise RuntimeError("failed to generate a solid mask")


def _make_specs(count: int, force_chain_pair: bool) -> list[dict]:
    assert count in (2, 3)
    chain_pair = [
        {"mask": _random_solid_mask(), "has_hole": False},
        {"mask": _random_hole_mask(), "has_hole": True},
    ]
    random.shuffle(chain_pair)
    specs = chain_pair
    if count == 3:
        if random.random() < 0.5:
            specs.append({"mask": _random_solid_mask(), "has_hole": False})
        else:
            specs.append({"mask": _random_hole_mask(), "has_hole": True})
    if not force_chain_pair:
        random.shuffle(specs)
    return specs


def _row_span_overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _first_settle_col(side_occ: np.ndarray, mask: np.ndarray, r0: int, direction: int):
    h, w = side_occ.shape
    mh, mw = mask.shape
    base = np.argwhere(mask)
    if direction == -1:
        cols = range(1, w - mw)
    else:
        cols = range(w - mw - 1, 0, -1)
    for c0 in cols:
        coords = base + np.array([r0, c0])
        if not np.any(side_occ[coords[:, 0], coords[:, 1]]):
            return c0, coords
    return None, None


def _place_side(
    shape: tuple[int, int],
    color: int,
    specs: list[dict],
    direction: int,
    global_occ: np.ndarray,
    require_chain: bool,
) -> list[dict] | None:
    h, w = shape
    wall_col = 0 if direction == -1 else w - 1
    side_occ = np.zeros(shape, dtype=bool)
    side_occ[:, wall_col] = True
    placements: list[dict] = []

    for idx, spec in enumerate(specs):
        mask = spec["mask"]
        mh, mw = mask.shape
        if h - mh < 2:
            row_choices = list(range(0, h - mh + 1))
        else:
            row_choices = list(range(1, h - mh))
        random.shuffle(row_choices)
        chosen = None
        for r0 in row_choices:
            row_span = (r0, r0 + mh - 1)
            if require_chain and idx == 1:
                first_span = placements[0]["row_span"]
                if not _row_span_overlaps(row_span, first_span):
                    continue
            elif idx >= 1:
                if any(_row_span_overlaps(row_span, prev["row_span"]) for prev in placements):
                    continue
            c0, coords = _first_settle_col(side_occ, mask, r0, direction)
            if coords is None:
                continue
            touches_wall = c0 == 1 if direction == -1 else c0 + mw - 1 == w - 2
            if require_chain and idx == 1 and touches_wall:
                continue
            if np.any(global_occ[coords[:, 0], coords[:, 1]]):
                continue
            chosen = (r0, c0, coords)
            break
        if chosen is None:
            return None
        r0, c0, coords = chosen
        side_occ[coords[:, 0], coords[:, 1]] = True
        global_occ[coords[:, 0], coords[:, 1]] = True
        placements.append(
            {
                "color": int(color),
                "has_hole": bool(spec["has_hole"]),
                "mask": mask.copy(),
                "coords": coords.copy(),
                "bbox": _bbox(coords),
                "row_span": (r0, r0 + mh - 1),
                "c0": int(c0),
            }
        )

    if require_chain:
        if len(placements) < 2:
            return None
        second = placements[1]
        touches_wall = second["c0"] == 1 if direction == -1 else second["bbox"][3] == w - 2
        if touches_wall:
            return None
    return placements


def _assign_input_side(
    shape: tuple[int, int],
    placements: list[dict],
    direction: int,
    occupied: np.ndarray,
    rng: random.Random | None = None,
) -> list[np.ndarray] | None:
    rng = rng or random
    h, w = shape
    ordered = sorted(
        placements,
        key=lambda p: p["bbox"][1] if direction == -1 else -p["bbox"][3],
    )
    assigned: list[np.ndarray] = []
    base_masks = [np.argwhere(p["mask"]) for p in ordered]

    def backtrack(idx: int, last_metric: int | None) -> bool:
        if idx == len(ordered):
            return True
        placement = ordered[idx]
        mask_coords = base_masks[idx]
        mh, mw = placement["mask"].shape
        final_c0 = int(placement["c0"])
        if direction == -1:
            candidates = list(range(final_c0 + 1, w - mw))
        else:
            candidates = list(range(1, final_c0))
        rng.shuffle(candidates)

        for c0 in candidates:
            metric = c0 if direction == -1 else c0 + mw - 1
            if last_metric is not None:
                if direction == -1 and metric <= last_metric:
                    continue
                if direction == 1 and metric >= last_metric:
                    continue
            r0 = placement["row_span"][0]
            moved = mask_coords + np.array([r0, c0])
            if (
                int(moved[:, 0].min()) < 0
                or int(moved[:, 0].max()) >= h
                or int(moved[:, 1].min()) <= 0
                or int(moved[:, 1].max()) >= w - 1
                or np.any(occupied[moved[:, 0], moved[:, 1]])
            ):
                continue
            occupied[moved[:, 0], moved[:, 1]] = True
            assigned.append(moved.copy())
            if backtrack(idx + 1, metric):
                return True
            assigned.pop()
            occupied[moved[:, 0], moved[:, 1]] = False
        return False

    if not backtrack(0, None):
        return None
    return assigned


def _render_input(shape: tuple[int, int], placements: list[dict]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    grid[:, 0] = LEFT_WALL
    grid[:, -1] = RIGHT_WALL
    occupied = grid != BG

    by_color = {
        LEFT_WALL: [p for p in placements if p["color"] == LEFT_WALL],
        RIGHT_WALL: [p for p in placements if p["color"] == RIGHT_WALL],
    }
    for order in ((LEFT_WALL, RIGHT_WALL), (RIGHT_WALL, LEFT_WALL)):
        occupied_try = occupied.copy()
        assigned_by_color: dict[int, list[np.ndarray]] = {}
        ok = True
        for color in order:
            direction = -1 if color == LEFT_WALL else 1
            assigned = _assign_input_side(shape, by_color[color], direction, occupied_try)
            if assigned is None:
                ok = False
                break
            assigned_by_color[color] = assigned
        if not ok:
            continue

        built = grid.copy()
        for color in (LEFT_WALL, RIGHT_WALL):
            for coords in assigned_by_color[color]:
                built[coords[:, 0], coords[:, 1]] = color
        return built
    return None


def _direct_wall_slide(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    out[:, 0] = LEFT_WALL
    out[:, -1] = RIGHT_WALL
    for color, wall_col in ((LEFT_WALL, 1), (RIGHT_WALL, g.shape[1] - 2)):
        for coords in _extract_objects(g, color, 0 if color == LEFT_WALL else g.shape[1] - 1):
            shifted = coords.copy()
            if color == LEFT_WALL:
                shifted[:, 1] += wall_col - int(coords[:, 1].min())
            else:
                shifted[:, 1] += wall_col - int(coords[:, 1].max())
            out[shifted[:, 0], shifted[:, 1]] = color
    return out


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not (np.all(g[:, 0] == LEFT_WALL) and np.all(g[:, -1] == RIGHT_WALL)):
        return False

    objects = []
    for color, wall_col in ((LEFT_WALL, 0), (RIGHT_WALL, g.shape[1] - 1)):
        comps = _extract_objects(g, color, wall_col)
        if len(comps) not in (2, 3):
            return False
        for coords in comps:
            objects.append((color, coords, _coords_have_hole(coords)))
    if len(objects) < 4 or len(objects) > 6:
        return False

    holes = {color for color, _coords, has_hole in objects if has_hole}
    solids = {color for color, _coords, has_hole in objects if not has_hole}
    if holes != {LEFT_WALL, RIGHT_WALL}:
        return False
    if solids != {LEFT_WALL, RIGHT_WALL}:
        return False

    out = main(g)
    if np.array_equal(g, out):
        return False
    if np.array_equal(out, _solve(g, recolor_holes=False)):
        return False
    left_blocked = any(
        int(placed[:, 1].min()) > 1 for _original, placed in _settle_components(g, LEFT_WALL, 0, -1)
    )
    right_blocked = any(
        int(placed[:, 1].max()) < g.shape[1] - 2
        for _original, placed in _settle_components(g, RIGHT_WALL, g.shape[1] - 1, 1)
    )
    if not (left_blocked or right_blocked):
        return False

    left_side_blue = np.any(out[:, 1 : g.shape[1] // 2] == RIGHT_WALL)
    right_side_red = np.any(out[:, g.shape[1] // 2 : -1] == LEFT_WALL)
    if not (left_side_blue and right_side_red):
        return False

    return True


def _generate_once() -> np.ndarray | None:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    left_count = random.randint(2, 3)
    right_count = random.randint(2, 3)
    chain_colors = {random.choice([LEFT_WALL, RIGHT_WALL])}
    if random.random() < 0.45:
        chain_colors.add(LEFT_WALL if RIGHT_WALL in chain_colors else RIGHT_WALL)

    left_specs = _make_specs(left_count, force_chain_pair=LEFT_WALL in chain_colors)
    right_specs = _make_specs(right_count, force_chain_pair=RIGHT_WALL in chain_colors)

    order = [(LEFT_WALL, -1, left_specs), (RIGHT_WALL, 1, right_specs)]
    random.shuffle(order)
    global_occ = np.zeros((h, w), dtype=bool)
    global_occ[:, 0] = True
    global_occ[:, -1] = True
    all_placements: list[dict] = []

    for color, direction, specs in order:
        placed = _place_side(
            (h, w),
            color,
            specs,
            direction,
            global_occ,
            require_chain=color in chain_colors,
        )
        if placed is None:
            return None
        all_placements.extend(placed)

    for _ in range(20):
        grid = _render_input((h, w), all_placements)
        if grid is None:
            continue
        if not _validate_semantics(grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
