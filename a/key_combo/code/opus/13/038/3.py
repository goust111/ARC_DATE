from __future__ import annotations

from collections import Counter, deque
import hashlib
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
BG = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 600


SHRINK_DISAPPEAR_TEMPLATES = [
    {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)},
    {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 1), (1, 1), (1, 0), (2, 0), (2, 1)},
]

SHRINK_SURVIVE_TEMPLATES = [
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (0, 0)},
]

GENERAL_OBJECT_TEMPLATES = SHRINK_DISAPPEAR_TEMPLATES + SHRINK_SURVIVE_TEMPLATES + [
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)},
]

_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    return int(np.argmax(counts))


def _component_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    arr = np.array(coords, dtype=int)
    values = [int(grid[r, c]) for r, c in coords]
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return {
        "coords": coords,
        "coord_set": set(coords),
        "array": arr,
        "area": len(coords),
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
        "colors": set(values),
    }


def _find_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(_component_info(grid, coords))
    return comps


def _is_simple_path(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    degs = []
    for r, c in coords:
        deg = sum((r + dr, c + dc) in pts for dr, dc in DIR4)
        degs.append(deg)
    return degs.count(1) == 2 and all(deg in (1, 2) for deg in degs)


def _trace_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    ends = []
    for r, c in coords:
        deg = sum((r + dr, c + dc) in pts for dr, dc in DIR4)
        if deg == 1:
            ends.append((r, c))
    assert len(ends) == 2, "snake must have exactly two endpoints"
    cur = min(ends)
    prev = None
    order: list[tuple[int, int]] = []
    for _ in range(len(coords)):
        order.append(cur)
        nxts = []
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in pts and nxt != prev:
                nxts.append(nxt)
        if not nxts:
            break
        assert len(nxts) == 1, "snake path cannot branch"
        prev, cur = cur, nxts[0]
    assert len(order) == len(coords), "snake traversal must cover every path cell"
    return order


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _find_foreground_components(g, bg)
    snake_candidates = [comp for comp in comps if len(comp["colors"]) > 1]
    assert len(snake_candidates) == 1, "expected one multicolor snake"
    snake = snake_candidates[0]
    assert 8 <= snake["area"] <= 15, "snake length must be 8..15"
    assert _is_simple_path(snake["coords"]), "snake must be a simple 4-connected chain"
    order = _trace_path(snake["coords"])
    objects = [comp for comp in comps if comp is not snake]
    assert 3 <= len(objects) <= 5, "expected 3..5 independent objects"
    assert all(len(comp["colors"]) == 1 for comp in objects), "objects must be monochromatic"
    object_colors = [next(iter(comp["colors"])) for comp in objects]
    assert len(set(object_colors)) == len(object_colors), "object colors must be distinct"
    snake_colors = [int(g[r, c]) for r, c in order]
    snake_palette = set(snake_colors)
    assert snake_palette == set(object_colors), "snake colors must match object colors exactly"
    first_seen: dict[int, int] = {}
    for idx, color in enumerate(snake_colors):
        first_seen.setdefault(int(color), idx)
    assert len(first_seen) == len(object_colors), "every object color must appear on the snake"
    ordered_colors = sorted(first_seen, key=lambda color: first_seen[color])
    color_to_object = {next(iter(comp["colors"])): comp for comp in objects}
    return {
        "grid": g,
        "bg": bg,
        "snake": snake,
        "snake_order": order,
        "snake_colors": snake_colors,
        "ordered_colors": ordered_colors,
        "objects": objects,
        "color_to_object": color_to_object,
    }


def _growth_cells(grid: np.ndarray, bg: int, obj_coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    H, W = grid.shape
    growth: set[tuple[int, int]] = set()
    for r, c in obj_coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == bg:
                growth.add((nr, nc))
    return growth


def _shrink_cells(grid: np.ndarray, bg: int, obj_coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    H, W = grid.shape
    pts = set(obj_coords)
    removed: set[tuple[int, int]] = set()
    for r, c in obj_coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in pts and int(grid[nr, nc]) == bg:
                removed.add((r, c))
                break
    return removed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a snake's color-first-order and use it to choose which object grows, which shrinks, then erase the snake.

    Input structure:
    - The background is black.
    - One foreground component is a multicolor 4-connected snake with exactly two endpoints.
    - The remaining 3-5 foreground components are separated monochromatic 4-connected objects with distinct colors.
    - The set of object colors matches the set of colors appearing on the snake.

    Rule:
    - Find the snake and traverse it from the top-left-most endpoint.
    - Record the first position where each color appears along that traversal.
    - The earliest first-seen color's object grows by one 4-neighbor layer into black cells.
    - The latest first-seen color's object shrinks by removing every pixel that touches black in the initial grid.
    - Apply grow and shrink from the initial state, keep other objects unchanged, and erase the whole snake to black.

    Key insight:
    - The decisive priority signal is not spatial object order but the hidden first-occurrence order of colors along the snake path.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    out = g.copy()

    grow_color = scene["ordered_colors"][0]
    shrink_color = scene["ordered_colors"][-1]
    grow_obj = scene["color_to_object"][grow_color]
    shrink_obj = scene["color_to_object"][shrink_color]

    growth = _growth_cells(g, bg, grow_obj["coords"])
    shrink = _shrink_cells(g, bg, shrink_obj["coords"])

    for r, c in scene["snake"]["coords"]:
        out[r, c] = bg
    for r, c in shrink:
        out[r, c] = bg
    for r, c in growth:
        out[r, c] = grow_color
    return out


def _normalize(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return sorted((r - r0, c - c0) for r, c in coords)


def _shift_path_to_origin(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _c in path)
    c0 = min(c for _r, c in path)
    return [(r - r0, c - c0) for r, c in path]


def _transform_shape(base: set[tuple[int, int]]) -> list[tuple[int, int]]:
    coords = list(base)
    if random.random() < 0.5:
        coords = [(r, -c) for r, c in coords]
    turns = random.randrange(4)
    for _ in range(turns):
        coords = [(c, -r) for r, c in coords]
    return _normalize(set(coords))


def _render_patch(coords: list[tuple[int, int]], color: int) -> np.ndarray:
    h = max(r for r, _c in coords) + 1
    w = max(c for _r, c in coords) + 1
    patch = np.full((h, w), BG, dtype=int)
    for r, c in coords:
        patch[r, c] = color
    return patch


def _patch_fg_coords(patch: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch != BG)]


def _patch_with_colors(path_coords: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    h = max(r for r, _c in path_coords) + 1
    w = max(c for _r, c in path_coords) + 1
    patch = np.full((h, w), BG, dtype=int)
    for (r, c), color in zip(path_coords, colors):
        patch[r, c] = color
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r, c in _patch_fg_coords(patch):
        grid[top + r, left + c] = int(patch[r, c])


def _can_place(occupied: set[tuple[int, int]], grid_shape: tuple[int, int], patch_coords: list[tuple[int, int]], top: int, left: int) -> bool:
    H, W = grid_shape
    translated = [(top + r, left + c) for r, c in patch_coords]
    if any(not (0 <= r < H and 0 <= c < W) for r, c in translated):
        return False
    for r, c in translated:
        for dr, dc in DIR8:
            if (r + dr, c + dc) in occupied:
                return False
        if (r, c) in occupied:
            return False
    return True


def _random_place(grid: np.ndarray, occupied: set[tuple[int, int]], patch: np.ndarray) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = patch.shape
    coords = _patch_fg_coords(patch)
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        if _can_place(occupied, grid.shape, coords, top, left):
            _place_patch(grid, patch, top, left)
            for r, c in coords:
                occupied.add((top + r, left + c))
            return top, left
    return None


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if a != b:
            turns += 1
    return turns


def _gen_snake_path(length: int) -> list[tuple[int, int]] | None:
    board_h = random.randint(4, 7)
    board_w = random.randint(4, 8)
    cells = [(r, c) for r in range(board_h) for c in range(board_w)]
    random.shuffle(cells)
    for start in cells:
        path = [start]
        visited = {start}

        def dfs(cur: tuple[int, int]) -> bool:
            if len(path) == length:
                return True
            nbrs = []
            for dr, dc in DIR4:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nxt[0] < board_h and 0 <= nxt[1] < board_w):
                    continue
                if nxt in visited:
                    continue
                touched = sum((nxt[0] + rr, nxt[1] + cc) in visited for rr, cc in DIR4)
                if touched != 1:
                    continue
                nbrs.append(nxt)
            random.shuffle(nbrs)
            nbrs.sort(key=lambda cell: abs(cell[0] - board_h / 2) + abs(cell[1] - board_w / 2))
            for nxt in nbrs:
                visited.add(nxt)
                path.append(nxt)
                if dfs(nxt):
                    return True
                path.pop()
                visited.remove(nxt)
            return False

        if dfs(start):
            norm_path = _shift_path_to_origin(path)
            if _count_turns(norm_path) >= 3 and len(set(r for r, _c in norm_path)) >= 3 and len(set(c for _r, c in norm_path)) >= 3:
                return norm_path
    return None


def _first_order(colors: list[int]) -> list[int]:
    seen: dict[int, int] = {}
    for idx, color in enumerate(colors):
        seen.setdefault(int(color), idx)
    return sorted(seen, key=lambda color: seen[color])


def _build_snake_colors(priority: list[int], length: int) -> list[int] | None:
    if length < len(priority) + 3:
        return None
    middle = priority[1:-1]
    if not middle:
        return None
    for _ in range(400):
        intro_positions = [0] + sorted(random.sample(range(1, length - 1), len(priority) - 1))
        seq = [None] * length
        introduced: list[int] = []
        for i, color in enumerate(priority):
            pos = intro_positions[i]
            seq[pos] = color
            introduced.append(color)
            end = intro_positions[i + 1] if i + 1 < len(priority) else length
            for j in range(pos + 1, end):
                choices = introduced[:]
                if j == length - 1:
                    choices = middle[:]
                seq[j] = random.choice(choices)
        colors = [int(v) for v in seq]
        if _first_order(colors) != priority:
            continue
        if colors[-1] not in middle:
            continue
        counts = Counter(colors)
        max_count = max(counts.values())
        if counts[priority[0]] == max_count:
            continue
        if counts[middle[0]] <= counts[priority[0]]:
            continue
        return colors
    return None


def _object_templates_for_colors(colors: list[int], shrink_color: int, shrink_vanishes: bool) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        if color == shrink_color:
            base = random.choice(SHRINK_DISAPPEAR_TEMPLATES if shrink_vanishes else SHRINK_SURVIVE_TEMPLATES)
        else:
            base = random.choice(GENERAL_OBJECT_TEMPLATES)
        out[color] = _transform_shape(base)
    return out


def _snake_and_objects_heuristics(scene: dict) -> bool:
    objects = scene["objects"]
    color_to_object = scene["color_to_object"]
    grow_color = scene["ordered_colors"][0]
    shrink_color = scene["ordered_colors"][-1]

    areas = {next(iter(obj["colors"])): int(obj["area"]) for obj in objects}
    anchors = {next(iter(obj["colors"])): obj["bbox"][:2] for obj in objects}
    max_area_color = max(sorted(areas), key=lambda color: (areas[color], -color))
    min_area_color = min(sorted(areas), key=lambda color: (areas[color], color))
    first_anchor_color = min(sorted(anchors), key=lambda color: anchors[color])
    last_anchor_color = max(sorted(anchors), key=lambda color: anchors[color])

    if grow_color in {max_area_color, min_area_color, first_anchor_color}:
        return False
    if shrink_color in {max_area_color, min_area_color, last_anchor_color}:
        return False

    growth = _growth_cells(scene["grid"], scene["bg"], color_to_object[grow_color]["coords"])
    shrink = _shrink_cells(scene["grid"], scene["bg"], color_to_object[shrink_color]["coords"])
    if not growth or not shrink:
        return False

    snake_colors = scene["snake_colors"]
    counts = Counter(snake_colors)
    latest_last_color = snake_colors[-1]
    if latest_last_color in {grow_color, shrink_color}:
        return False
    if counts[grow_color] >= max(counts.values()):
        return False
    return True


def _try_generate() -> np.ndarray | None:
    H = random.randint(14, 18)
    W = random.randint(14, 18)
    n_objects = random.choice([3, 3, 4, 4, 5])
    colors = random.sample(PALETTE, n_objects)
    shrink_color = random.choice(colors)
    shrink_vanishes = random.random() < 0.5
    obj_shapes = _object_templates_for_colors(colors, shrink_color, shrink_vanishes)

    grid = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    object_places: dict[int, tuple[int, int]] = {}
    for color in sorted(colors, key=lambda c: len(obj_shapes[c]), reverse=True):
        patch = _render_patch(obj_shapes[color], color)
        loc = _random_place(grid, occupied, patch)
        if loc is None:
            return None
        object_places[color] = loc

    areas = {color: len(obj_shapes[color]) for color in colors}
    anchors = {color: object_places[color] for color in colors}
    grow_choices = [
        color
        for color in colors
        if color != shrink_color
        and color != max(sorted(areas), key=lambda c: (areas[c], -c))
        and color != min(sorted(areas), key=lambda c: (areas[c], c))
        and color != min(sorted(anchors), key=lambda c: anchors[c])
    ]
    if not grow_choices:
        return None
    grow_color = random.choice(grow_choices)
    middle = [color for color in colors if color not in {grow_color, shrink_color}]
    random.shuffle(middle)
    priority = [grow_color] + middle + [shrink_color]

    length = random.randint(max(8, n_objects + 4), 15)
    snake_path = _gen_snake_path(length)
    if snake_path is None:
        return None
    snake_colors = _build_snake_colors(priority, length)
    if snake_colors is None:
        return None
    snake_patch = _patch_with_colors(snake_path, snake_colors)
    if _random_place(grid, occupied, snake_patch) is None:
        return None

    try:
        scene = _parse_scene(grid)
        out = main(grid)
    except Exception:
        return None

    if not _snake_and_objects_heuristics(scene):
        return None

    if shrink_vanishes:
        shrink_obj = scene["color_to_object"][shrink_color]
        remaining = set(shrink_obj["coords"]) - _shrink_cells(scene["grid"], scene["bg"], shrink_obj["coords"])
        if remaining:
            return None
    else:
        shrink_obj = scene["color_to_object"][shrink_color]
        remaining = set(shrink_obj["coords"]) - _shrink_cells(scene["grid"], scene["bg"], shrink_obj["coords"])
        if not remaining:
            return None

    if np.array_equal(grid, out):
        return None

    output_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _try_generate()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid snake-priority ARC input")
