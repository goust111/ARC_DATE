import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
LINE_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _monochrome_components(grid: np.ndarray, coords: list[tuple[int, int]]) -> list[dict]:
    coord_set = set(coords)
    seen = set()
    components = []
    for start in coords:
        if start in seen:
            continue
        color = int(grid[start])
        queue = deque([start])
        seen.add(start)
        comp = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if nxt in coord_set and nxt not in seen and int(grid[nxt]) == color:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append({"color": color, "coords": comp})
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_patch(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0)


def _simple_path_info(coords: list[tuple[int, int]]) -> tuple[bool, list[tuple[int, int]]]:
    if len(coords) < 2:
        return False, []
    coord_set = set(coords)
    degrees = {}
    for r, c in coords:
        deg = sum((r + dr, c + dc) in coord_set for dr, dc in NEIGHBORS4)
        degrees[(r, c)] = deg
        if deg not in (1, 2):
            return False, []
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return False, []

    ordered = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        neighbors = [
            (cur[0] + dr, cur[1] + dc)
            for dr, dc in NEIGHBORS4
            if (cur[0] + dr, cur[1] + dc) in coord_set and (cur[0] + dr, cur[1] + dc) != prev
        ]
        if len(neighbors) != 1:
            return False, []
        nxt = neighbors[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    return True, ordered


def _compressed_dirs(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        step = (r1 - r0, c1 - c0)
        if not dirs or dirs[-1] != step:
            dirs.append(step)
    return dirs


def _transform_patch(patch: np.ndarray, turns: int, vertical_dir: int | None) -> np.ndarray:
    if turns == 0:
        return np.fliplr(patch)
    if turns == 1:
        assert vertical_dir in (-1, 1)
        return np.rot90(patch, 1 if vertical_dir < 0 else 3)
    if turns == 2:
        return np.rot90(patch, 2)
    raise AssertionError(f"unsupported turn count: {turns}")


def _paste(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    dest = out[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    dest[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each paired endpoint object according to the bend pattern of its connecting line.

    Input structure:
    - The background is black.
    - Each foreground structure is a 4-connected cluster made of two small same-color endpoint objects
      and one single-pixel-wide orthogonal connecting line in another color.
    - The connecting line is a simple path with 0, 1, or 2 right-angle turns.
    - Each endpoint object is a small non-symmetric monochrome shape with a bounding box no larger than 4x4.

    Rule:
    - Split every foreground structure into its two endpoint objects and its connecting line.
    - Count the connecting line's turns.
    - If the line is straight, horizontally mirror both endpoint objects in place.
    - If the line has one turn, rotate both endpoint objects 90 degrees in the turn direction:
      when the path runs horizontally and then upward, rotate counterclockwise; when it runs
      horizontally and then downward, rotate clockwise.
    - If the line has two turns, rotate both endpoint objects 180 degrees.
    - Draw transformed objects back at their original top-left anchors and keep the connecting line unchanged.

    Key insight:
    - The gray connector encodes both the pairing and the geometric operation: its topology determines
      which in-place transformation the endpoint sprites receive.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    structures = _components_from_mask(g != bg)
    line_cells_to_restore = []

    for structure in structures:
        mono_components = _monochrome_components(g, structure)
        path_candidates = []
        for comp in mono_components:
            is_path, ordered = _simple_path_info(comp["coords"])
            if is_path:
                path_candidates.append((comp, ordered))

        assert len(path_candidates) == 1, "expected exactly one line component per structure"
        line_comp, path = path_candidates[0]
        objects = [comp for comp in mono_components if comp is not line_comp]
        assert len(objects) == 2, "expected exactly two endpoint objects"

        dirs = _compressed_dirs(path)
        turns = len(dirs) - 1
        assert turns in (0, 1, 2), "line must have 0, 1, or 2 turns"

        vertical_dir = None
        if turns == 1:
            if dirs[0][0] != 0:
                path = list(reversed(path))
                dirs = _compressed_dirs(path)
            assert dirs[0][0] == 0 and dirs[1][1] == 0, "L path must be horizontal then vertical after canonicalization"
            vertical_dir = dirs[1][0]
            assert vertical_dir in (-1, 1)

        for obj in objects:
            patch, (r0, c0) = _component_patch(g, obj["coords"], bg)
            transformed = _transform_patch(patch, turns, vertical_dir)
            for r, c in obj["coords"]:
                out[r, c] = bg
            _paste(out, transformed, r0, c0, bg)

        line_cells_to_restore.extend(line_comp["coords"])

    for r, c in line_cells_to_restore:
        out[r, c] = g[r, c]
    return out


def _mask_has_branch(mask: np.ndarray) -> bool:
    coords = [(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if mask[r, c]]
    coord_set = set(coords)
    return any(sum((r + dr, c + dc) in coord_set for dr, dc in NEIGHBORS4) >= 3 for r, c in coords)


def _mask_has_side(mask: np.ndarray, side: str) -> bool:
    if side == "left":
        return bool(mask[:, 0].any())
    if side == "right":
        return bool(mask[:, -1].any())
    if side == "top":
        return bool(mask[0, :].any())
    if side == "bottom":
        return bool(mask[-1, :].any())
    raise ValueError(side)


def _mask_changes_under_transforms(mask: np.ndarray) -> bool:
    return (
        not np.array_equal(mask, np.fliplr(mask))
        and not np.array_equal(mask, np.rot90(mask, 1))
        and not np.array_equal(mask, np.rot90(mask, 2))
    )


def _random_connected_mask(size: int, cells: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < size and 0 <= nc < size:
            frontier.add((nr, nc))
    while int(mask.sum()) < cells and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _generate_object_mask(required_side: str) -> np.ndarray:
    for _ in range(200):
        size = random.randint(3, 4)
        cells = random.randint(size + 1, min(size * size - 1, size + 4))
        mask = _random_connected_mask(size, cells)
        if not _mask_has_side(mask, required_side):
            continue
        if not all(_mask_has_side(mask, side) for side in ("left", "right", "top", "bottom")):
            continue
        if not _mask_has_branch(mask):
            continue
        if not _mask_changes_under_transforms(mask):
            continue
        return mask
    raise RuntimeError(f"failed to create object mask for side {required_side}")


def _attachment(mask: np.ndarray, side: str) -> tuple[int, int]:
    size = mask.shape[0]
    if side == "left":
        choices = [(r, 0) for r in range(size) if mask[r, 0]]
    elif side == "right":
        choices = [(r, size - 1) for r in range(size) if mask[r, size - 1]]
    elif side == "top":
        choices = [(0, c) for c in range(size) if mask[0, c]]
    elif side == "bottom":
        choices = [(size - 1, c) for c in range(size) if mask[size - 1, c]]
    else:
        raise ValueError(side)
    return random.choice(choices)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment endpoints must be orthogonally aligned")


def _normalize_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    return [(r - min_r, c - min_c, color) for r, c, color in cells]


def _build_structure(path_type: int, color: int) -> list[tuple[int, int, int]]:
    if path_type == 0:
        if random.random() < 0.5:
            side_a, side_b = "right", "left"
            mask_a = _generate_object_mask(side_a)
            mask_b = _generate_object_mask(side_b)
            ar, _ = _attachment(mask_a, side_a)
            br, _ = _attachment(mask_b, side_b)
            gap = random.randint(3, 8)
            pos_a = (0, 0)
            pos_b = (ar - br, mask_a.shape[1] + gap)
            line_start = (pos_a[0] + ar, pos_a[1] + mask_a.shape[1])
            line_end = (pos_b[0] + br, pos_b[1] - 1)
            line = _segment(line_start, line_end)
        else:
            side_a, side_b = "bottom", "top"
            mask_a = _generate_object_mask(side_a)
            mask_b = _generate_object_mask(side_b)
            _, ac = _attachment(mask_a, side_a)
            _, bc = _attachment(mask_b, side_b)
            gap = random.randint(3, 8)
            pos_a = (0, 0)
            pos_b = (mask_a.shape[0] + gap, ac - bc)
            line_start = (pos_a[0] + mask_a.shape[0], pos_a[1] + ac)
            line_end = (pos_b[0] - 1, pos_b[1] + bc)
            line = _segment(line_start, line_end)
    elif path_type == 1:
        turn_up = random.random() < 0.5
        mask_a = _generate_object_mask("right")
        mask_b = _generate_object_mask("bottom" if turn_up else "top")
        ar, _ = _attachment(mask_a, "right")
        _, bc = _attachment(mask_b, "bottom" if turn_up else "top")
        min_horiz = bc + 1
        horiz = random.randint(min_horiz, min_horiz + 5)
        vert = random.randint(2, 7)
        pos_a = (0, 0)
        start = (pos_a[0] + ar, pos_a[1] + mask_a.shape[1])
        elbow = (start[0], start[1] + horiz)
        if turn_up:
            end = (start[0] - vert, elbow[1])
            pos_b = (end[0] - mask_b.shape[0], end[1] - bc)
        else:
            end = (start[0] + vert, elbow[1])
            pos_b = (end[0] + 1, end[1] - bc)
        line = _segment(start, elbow) + _segment(elbow, end)[1:]
    elif path_type == 2:
        if random.random() < 0.5:
            mask_a = _generate_object_mask("right")
            mask_b = _generate_object_mask("left")
            ar, _ = _attachment(mask_a, "right")
            br, _ = _attachment(mask_b, "left")
            first = random.randint(2, 5)
            last = random.randint(2, 5)
            vdir = random.choice((-1, 1))
            vert = random.randint(2, 7)
            pos_a = (0, 0)
            start = (pos_a[0] + ar, pos_a[1] + mask_a.shape[1])
            elbow1 = (start[0], start[1] + first)
            end_row = start[0] + vdir * vert
            elbow2 = (end_row, elbow1[1])
            end = (end_row, elbow2[1] + last)
            pos_b = (end[0] - br, end[1] + 1)
            line = _segment(start, elbow1) + _segment(elbow1, elbow2)[1:] + _segment(elbow2, end)[1:]
        else:
            mask_a = _generate_object_mask("bottom")
            mask_b = _generate_object_mask("top")
            _, ac = _attachment(mask_a, "bottom")
            _, bc = _attachment(mask_b, "top")
            first = random.randint(2, 5)
            last = random.randint(2, 5)
            hdir = random.choice((-1, 1))
            horiz = random.randint(2, 7)
            pos_a = (0, 0)
            start = (pos_a[0] + mask_a.shape[0], pos_a[1] + ac)
            elbow1 = (start[0] + first, start[1])
            end_col = start[1] + hdir * horiz
            elbow2 = (elbow1[0], end_col)
            end = (elbow2[0] + last, end_col)
            pos_b = (end[0] + 1, end[1] - bc)
            line = _segment(start, elbow1) + _segment(elbow1, elbow2)[1:] + _segment(elbow2, end)[1:]
    else:
        raise ValueError(path_type)

    cells = []
    for r in range(mask_a.shape[0]):
        for c in range(mask_a.shape[1]):
            if mask_a[r, c]:
                cells.append((pos_a[0] + r, pos_a[1] + c, color))
    for r in range(mask_b.shape[0]):
        for c in range(mask_b.shape[1]):
            if mask_b[r, c]:
                cells.append((pos_b[0] + r, pos_b[1] + c, color))
    for r, c in line:
        cells.append((r, c, LINE_COLOR))

    normalized = _normalize_cells(cells)
    occupied = set()
    for r, c, _ in normalized:
        if (r, c) in occupied:
            raise RuntimeError("structure cells overlap")
        occupied.add((r, c))
    return normalized


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_structure(
    grid: np.ndarray,
    blocked: np.ndarray,
    cells: list[tuple[int, int, int]],
) -> bool:
    h, w = grid.shape
    max_r = max(r for r, _, _ in cells)
    max_c = max(c for _, c, _ in cells)
    if max_r >= h or max_c >= w:
        return False

    for _ in range(80):
        shift_r = random.randint(0, h - max_r - 1)
        shift_c = random.randint(0, w - max_c - 1)
        placed = [(r + shift_r, c + shift_c, color) for r, c, color in cells]
        coords = [(r, c) for r, c, _ in placed]
        if any(blocked[r, c] for r, c in coords):
            continue
        for r, c, color in placed:
            grid[r, c] = color
        _mark_blocked(blocked, coords)
        return True
    return False


def _path_type_sequence(pair_count: int) -> list[int]:
    if pair_count == 1:
        return [random.choice([0, 1, 2])]
    if pair_count == 2:
        seq = random.sample([0, 1, 2], 2)
        random.shuffle(seq)
        return seq
    seq = [0, 1, 2]
    random.shuffle(seq)
    return seq


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    pair_count = random.randint(1, 3)
    colors = random.sample(OBJECT_COLORS, pair_count)
    path_types = _path_type_sequence(pair_count)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for color, path_type in zip(colors, path_types):
        structure = _build_structure(path_type, color)
        if not _place_structure(grid, blocked, structure):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
