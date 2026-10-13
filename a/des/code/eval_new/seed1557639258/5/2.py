import numpy as np


_RNG = np.random.default_rng()

_ARC_LIMIT = 30
_BG = 0
_COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_MAX_GENERATION_ATTEMPTS = 500
_PARTITION_TREE_ATTEMPTS = 100
_PARTITION_CUT_ATTEMPTS = 150
_PLACEMENT_ATTEMPTS = 400
_MIN_PIECE_AREA = 3
_RECT_CANDIDATES = [
    (h, w)
    for h in range(3, 7)
    for w in range(3, 8)
    if 12 <= h * w <= 24
]


def _most_common_color(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(values))


def _foreground_colors(grid: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    seen = {int(bg)}
    for value in np.asarray(grid, dtype=int).ravel():
        color = int(value)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _connected_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _piece_from_coords(coords: list[tuple[int, int]] | np.ndarray) -> dict:
    arr = np.asarray(coords, dtype=int)
    rows = arr[:, 0]
    cols = arr[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    cells = tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))
    height = int(rows.max() - r0 + 1)
    width = int(cols.max() - c0 + 1)
    return {
        "cells": cells,
        "height": height,
        "width": width,
        "area": len(cells),
        "origin": (r0, c0),
    }


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area ** 0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    pairs.sort()
    return pairs


def _placement_bitmasks(piece: dict, rect_h: int, rect_w: int) -> list[int]:
    masks: list[int] = []
    for r0 in range(rect_h - piece["height"] + 1):
        for c0 in range(rect_w - piece["width"] + 1):
            mask = 0
            for dr, dc in piece["cells"]:
                bit = (r0 + dr) * rect_w + (c0 + dc)
                mask |= 1 << int(bit)
            masks.append(mask)
    return masks


def _rectangle_solutions(pieces: list[dict], limit: int = 2) -> list[dict]:
    total_area = sum(int(piece["area"]) for piece in pieces)
    solutions: list[dict] = []
    seen_keys: set[tuple[int, int, tuple[int, ...]]] = set()

    for rect_h, rect_w in _factor_pairs(total_area):
        if any(piece["height"] > rect_h or piece["width"] > rect_w for piece in pieces):
            continue

        placements = [_placement_bitmasks(piece, rect_h, rect_w) for piece in pieces]
        if any(not options for options in placements):
            continue

        full_mask = (1 << (rect_h * rect_w)) - 1

        def backtrack(covered: int, remaining: tuple[int, ...], chosen: list[int]) -> None:
            if len(solutions) >= limit:
                return
            if not remaining:
                if covered == full_mask:
                    key = (rect_h, rect_w, tuple(sorted(chosen)))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        solutions.append({"shape": (rect_h, rect_w), "placements": key[2]})
                return

            uncovered = full_mask & ~covered
            lowbit = uncovered & -uncovered
            options: list[tuple[int, int, int, list[int]]] = []
            for idx in remaining:
                valid = [mask for mask in placements[idx] if (mask & lowbit) and not (mask & covered)]
                if valid:
                    options.append((len(valid), -int(pieces[idx]["area"]), idx, valid))
            if not options:
                return

            options.sort()
            for _count, _neg_area, idx, valid in options:
                next_remaining = tuple(piece_idx for piece_idx in remaining if piece_idx != idx)
                for mask in valid:
                    backtrack(covered | mask, next_remaining, chosen + [mask])
                    if len(solutions) >= limit:
                        return

        backtrack(0, tuple(range(len(pieces))), [])
        if len(solutions) >= limit:
            break

    return solutions


def _analyze_scene(grid: np.ndarray, require_unique: bool = True) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    groups: list[dict] = []

    for color in _foreground_colors(g, bg):
        components = _connected_components(g, color)
        if not (2 <= len(components) <= 4):
            raise AssertionError("each color must have between 2 and 4 fragments")
        pieces = [_piece_from_coords(component) for component in components]
        solutions = _rectangle_solutions(pieces, limit=2 if require_unique else 1)
        if not solutions:
            raise AssertionError("fragments do not assemble into a rectangle")
        if require_unique and len(solutions) != 1:
            raise AssertionError("fragment assembly must be unique")
        shape = tuple(map(int, solutions[0]["shape"]))
        anchor = min(piece["origin"] for piece in pieces)
        groups.append(
            {
                "color": int(color),
                "shape": shape,
                "area": int(shape[0] * shape[1]),
                "anchor": tuple(map(int, anchor)),
                "pieces": pieces,
            }
        )

    if not groups:
        raise AssertionError("expected at least one foreground color")

    return bg, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored fragments, reassemble each color into its full rectangle, and stack the rectangles by area.

    Input structure:
    - The most common color is the background.
    - Every non-background color appears as 2-4 separate 4-connected fragments scattered across the grid.
    - Fragments of the same color are irregular complementary pieces of one hidden solid rectangle.

    Rule:
    - Group fragments by color.
    - For each color, find the unique translation-only arrangement that makes its pieces tile a solid rectangle with no overlaps or gaps.
    - Replace each group by that filled rectangle, then sort the rectangles by descending area.
    - Output the sorted rectangles top-to-bottom, left-aligned, with one background row between consecutive rectangles.

    Key insight:
    - The task is a translation-only packing puzzle first: each color's scattered fragments must be spatially reassembled before the final area-based ordering is even visible.
    """
    g = np.asarray(grid, dtype=int)
    bg, groups = _analyze_scene(g, require_unique=True)
    groups.sort(key=lambda group: (-int(group["area"]), int(group["anchor"][0]), int(group["anchor"][1])))

    out_h = sum(int(group["shape"][0]) for group in groups) + len(groups) - 1
    out_w = max(int(group["shape"][1]) for group in groups)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    row = 0
    for group in groups:
        rect_h, rect_w = map(int, group["shape"])
        out[row : row + rect_h, :rect_w] = int(group["color"])
        row += rect_h + 1

    return out


def _grid_neighbors(index: int, width: int, height: int) -> list[int]:
    r, c = divmod(index, width)
    neighbors: list[int] = []
    if r > 0:
        neighbors.append((r - 1) * width + c)
    if r + 1 < height:
        neighbors.append((r + 1) * width + c)
    if c > 0:
        neighbors.append(r * width + (c - 1))
    if c + 1 < width:
        neighbors.append(r * width + (c + 1))
    return neighbors


def _random_spanning_tree(height: int, width: int) -> list[tuple[int, int]]:
    total = height * width
    start = int(_RNG.integers(total))
    visited = {start}
    stack = [start]
    edges: list[tuple[int, int]] = []

    while stack:
        current = stack[-1]
        neighbors = [node for node in _grid_neighbors(current, width, height) if node not in visited]
        if not neighbors:
            stack.pop()
            continue
        nxt = int(_RNG.choice(neighbors))
        visited.add(nxt)
        edges.append((current, nxt))
        stack.append(nxt)

    return edges


def _tree_components(total: int, edges: list[tuple[int, int]], cuts: set[int]) -> list[list[int]]:
    adjacency = [[] for _ in range(total)]
    for edge_idx, (a, b) in enumerate(edges):
        if edge_idx in cuts:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)

    seen = [False] * total
    components: list[list[int]] = []
    for start in range(total):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in adjacency[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(comp)
    return components


def _partition_rectangle(height: int, width: int, piece_count: int) -> list[dict] | None:
    total = height * width
    if total < piece_count * _MIN_PIECE_AREA:
        return None

    for _ in range(_PARTITION_TREE_ATTEMPTS):
        edges = _random_spanning_tree(height, width)
        if len(edges) < piece_count - 1:
            continue
        for _ in range(_PARTITION_CUT_ATTEMPTS):
            cut_ids = {
                int(idx)
                for idx in _RNG.choice(len(edges), size=piece_count - 1, replace=False)
            }
            components = _tree_components(total, edges, cut_ids)
            if len(components) != piece_count:
                continue

            pieces: list[dict] = []
            ok = True
            for component in components:
                if len(component) < _MIN_PIECE_AREA:
                    ok = False
                    break
                coords = [(idx // width, idx % width) for idx in component]
                piece = _piece_from_coords(coords)
                if int(piece["area"]) == int(piece["height"] * piece["width"]):
                    ok = False
                    break
                pieces.append(piece)
            if not ok:
                continue

            solutions = _rectangle_solutions(pieces, limit=2)
            if len(solutions) != 1 or tuple(map(int, solutions[0]["shape"])) != (height, width):
                continue
            return pieces

    return None


def _can_place_piece(occupied: np.ndarray, piece: dict, top: int, left: int) -> bool:
    h, w = occupied.shape
    for dr, dc in piece["cells"]:
        rr = top + dr
        cc = left + dc
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _scatter_groups(groups: list[dict]) -> np.ndarray | None:
    canvas = np.full((_ARC_LIMIT, _ARC_LIMIT), _BG, dtype=int)
    occupied = np.zeros((_ARC_LIMIT, _ARC_LIMIT), dtype=bool)

    items = [(piece, int(group["color"])) for group in groups for piece in group["pieces"]]
    _RNG.shuffle(items)
    items.sort(key=lambda item: -int(item[0]["area"]))

    for piece, color in items:
        placed = False
        for _ in range(_PLACEMENT_ATTEMPTS):
            top = int(_RNG.integers(0, _ARC_LIMIT - piece["height"] + 1))
            left = int(_RNG.integers(0, _ARC_LIMIT - piece["width"] + 1))
            if not _can_place_piece(occupied, piece, top, left):
                continue
            for dr, dc in piece["cells"]:
                rr = top + dr
                cc = left + dc
                canvas[rr, cc] = color
                occupied[rr, cc] = True
            placed = True
            break
        if not placed:
            return None

    return canvas


def _crop_canvas(canvas: np.ndarray, max_group_area: int) -> np.ndarray | None:
    coords = np.argwhere(canvas != _BG)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    for _ in range(120):
        top_pad = int(_RNG.integers(0, int(r0) + 1))
        left_pad = int(_RNG.integers(0, int(c0) + 1))
        bottom_pad = int(_RNG.integers(0, int(canvas.shape[0] - r1)))
        right_pad = int(_RNG.integers(0, int(canvas.shape[1] - c1)))
        grid = canvas[int(r0) - top_pad : int(r1) + bottom_pad + 1, int(c0) - left_pad : int(c1) + right_pad + 1].copy()
        bg_count = int(np.sum(grid == _BG))
        other_counts = np.bincount(grid.ravel(), minlength=10)
        if bg_count <= max_group_area:
            continue
        if bg_count <= max(int(other_counts[color]) for color in range(1, 10)):
            continue
        return grid

    if int(np.sum(canvas == _BG)) > max_group_area:
        return canvas.copy()
    return None


def _sample_groups() -> list[dict] | None:
    group_count = int(_RNG.integers(2, 5))
    colors = list(map(int, _RNG.choice(_COLOR_POOL, size=group_count, replace=False)))
    used_areas: set[int] = set()
    groups: list[dict] = []

    for color in colors:
        piece_count = int(_RNG.integers(2, 5))
        choices = [shape for shape in _RECT_CANDIDATES if shape[0] * shape[1] not in used_areas and shape[0] * shape[1] >= piece_count * _MIN_PIECE_AREA]
        if not choices:
            return None
        _RNG.shuffle(choices)

        selected = None
        for height, width in choices:
            pieces = _partition_rectangle(height, width, piece_count)
            if pieces is None:
                continue
            selected = {
                "color": int(color),
                "shape": (int(height), int(width)),
                "pieces": pieces,
                "area": int(height * width),
            }
            break
        if selected is None:
            return None

        used_areas.add(int(selected["area"]))
        groups.append(selected)

    out_height = sum(int(group["shape"][0]) for group in groups) + len(groups) - 1
    out_width = max(int(group["shape"][1]) for group in groups)
    if out_height > _ARC_LIMIT or out_width > _ARC_LIMIT:
        return None

    return groups


def _gen_attempt() -> np.ndarray | None:
    groups = _sample_groups()
    if groups is None:
        return None

    canvas = _scatter_groups(groups)
    if canvas is None:
        return None

    grid = _crop_canvas(canvas, max_group_area=max(int(group["area"]) for group in groups))
    if grid is None or max(grid.shape) > _ARC_LIMIT:
        return None

    try:
        bg, analyzed = _analyze_scene(grid, require_unique=True)
    except AssertionError:
        return None

    if bg != _BG or len(analyzed) != len(groups):
        return None

    expected_shapes = {int(group["color"]): tuple(map(int, group["shape"])) for group in groups}
    for group in analyzed:
        if expected_shapes.get(int(group["color"])) != tuple(map(int, group["shape"])):
            return None

    if len({int(group["area"]) for group in analyzed}) != len(analyzed):
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > _ARC_LIMIT:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_GENERATION_ATTEMPTS} attempts")
