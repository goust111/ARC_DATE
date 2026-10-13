import random
from collections import Counter, deque

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 160


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _detect_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color) and r == h - 1 - r:
            candidates.append((0, r, color))
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color) and c == w - 1 - c:
            candidates.append((1, c, color))
    assert len(candidates) == 1, "expected exactly one central divider line"
    return candidates[0]


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    mins = arr.min(axis=0)
    shifted = arr - mins
    return tuple(sorted(map(tuple, shifted.tolist())))


def _mirror_signature(coords, axis: int) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    if axis == 1:
        reflected = np.column_stack((arr[:, 0] - r0, c1 - arr[:, 1]))
    else:
        reflected = np.column_stack((r1 - arr[:, 0], arr[:, 1] - c0))
    return tuple(sorted(map(tuple, reflected.tolist())))


def _component_side(coords: np.ndarray, axis: int, div_idx: int) -> int:
    if axis == 1:
        cols = coords[:, 1]
        assert np.all(cols != div_idx)
        if np.all(cols < div_idx):
            return 0
        if np.all(cols > div_idx):
            return 1
    else:
        rows = coords[:, 0]
        assert np.all(rows != div_idx)
        if np.all(rows < div_idx):
            return 0
        if np.all(rows > div_idx):
            return 1
    raise AssertionError("component must lie entirely on one side of the divider")


def _find_components(grid: np.ndarray, bg: int, divider_color: int, axis: int, div_idx: int):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == bg or color == divider_color:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            components.append(
                {
                    "id": next_id,
                    "color": color,
                    "coords": coords,
                    "side": _component_side(coords, axis, div_idx),
                    "sig": _normalize_coords(coords),
                    "mirror_sig": _mirror_signature(coords, axis),
                }
            )
            next_id += 1
    assert components, "expected at least one foreground component"
    return components


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)
    axis, div_idx, divider_color = _detect_divider(g, bg)
    components = _find_components(g, bg, divider_color, axis, div_idx)

    side0 = [comp for comp in components if comp["side"] == 0]
    side1 = [comp for comp in components if comp["side"] == 1]
    assert side0 and side1, "both sides must contain components"

    side0_map = {}
    for comp in side0:
        assert comp["mirror_sig"] not in side0_map, "side-0 pairing must be unambiguous"
        side0_map[comp["mirror_sig"]] = comp
    side1_map = {}
    for comp in side1:
        assert comp["sig"] not in side1_map, "side-1 pairing must be unambiguous"
        side1_map[comp["sig"]] = comp

    pairs = [(side0_map[key], side1_map[key]) for key in sorted(side0_map.keys() & side1_map.keys())]
    paired_ids = {comp["id"] for pair in pairs for comp in pair}
    unmatched = [comp for comp in components if comp["id"] not in paired_ids]
    assert pairs, "expected at least one matched mirrored pair"
    assert unmatched, "expected at least one unmatched component"
    return {
        "grid": g,
        "bg": bg,
        "axis": axis,
        "div_idx": div_idx,
        "divider_color": divider_color,
        "components": components,
        "pairs": pairs,
        "unmatched": unmatched,
        "side_components": (side0, side1),
    }


def _mirror_global(coords: np.ndarray, axis: int, div_idx: int) -> np.ndarray:
    mirrored = coords.copy()
    if axis == 1:
        mirrored[:, 1] = 2 * div_idx - mirrored[:, 1]
    else:
        mirrored[:, 0] = 2 * div_idx - mirrored[:, 0]
    return mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap colors between mirrored-shape pairs across the central grey divider,
    and mirror unmatched components to the opposite side.
    """
    info = _analyze(grid)
    g = info["grid"]
    bg = info["bg"]
    axis = info["axis"]
    div_idx = info["div_idx"]
    divider_color = info["divider_color"]

    out = np.full_like(g, bg)
    if axis == 1:
        out[:, div_idx] = divider_color
    else:
        out[div_idx, :] = divider_color

    for left_comp, right_comp in info["pairs"]:
        lcoords = left_comp["coords"]
        rcoords = right_comp["coords"]
        assert np.all(out[lcoords[:, 0], lcoords[:, 1]] == bg)
        assert np.all(out[rcoords[:, 0], rcoords[:, 1]] == bg)
        out[lcoords[:, 0], lcoords[:, 1]] = int(right_comp["color"])
        out[rcoords[:, 0], rcoords[:, 1]] = int(left_comp["color"])

    for comp in info["unmatched"]:
        mirrored = _mirror_global(comp["coords"], axis, div_idx)
        assert np.all(out[mirrored[:, 0], mirrored[:, 1]] == bg), "mirrored destination must be empty"
        out[mirrored[:, 0], mirrored[:, 1]] = int(comp["color"])

    return out


def _normalize_cells(cells) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(sorted(cells), dtype=int)
    mins = arr.min(axis=0)
    shifted = arr - mins
    return tuple(sorted(map(tuple, shifted.tolist())))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _ = _shape_dims(shape)
    return _normalize_cells({(c, h - 1 - r) for r, c in shape})


def _flip_h(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _shape_dims(shape)
    return _normalize_cells({(r, w - 1 - c) for r, c in shape})


def _random_transform(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    out = shape
    for _ in range(random.randrange(4)):
        out = _rotate_shape(out)
    if random.random() < 0.5:
        out = _flip_h(out)
    return out


def _reflect_shape_for_axis(shape: tuple[tuple[int, int], ...], axis: int) -> tuple[tuple[int, int], ...]:
    if axis == 1:
        return _flip_h(shape)
    h, _ = _shape_dims(shape)
    return _normalize_cells({(h - 1 - r, c) for r, c in shape})


def _shape_mirror_sig(shape: tuple[tuple[int, int], ...], axis: int) -> tuple[tuple[int, int], ...]:
    return _reflect_shape_for_axis(shape, axis)


def _grow_shape(max_h: int, max_w: int) -> tuple[tuple[int, int], ...] | None:
    for _ in range(SHAPE_ATTEMPTS):
        target_area = random.randint(3, min(7, max_h * max_w))
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            candidate = random.choice(tuple(frontier))
            cells.add(candidate)
            shape = _normalize_cells(cells)
            h, w = _shape_dims(shape)
            if h > max_h or w > max_w:
                break
        else:
            shape = _normalize_cells(cells)
            h, w = _shape_dims(shape)
            if h >= 2 and w >= 2:
                return _random_transform(shape)
    return None


def _mirror_local(coords, axis: int, side_h: int, side_w: int):
    if axis == 1:
        return {(r, side_w - 1 - c) for r, c in coords}
    return {(side_h - 1 - r, c) for r, c in coords}


def _expanded(cells, side_h: int, side_w: int, radius: int = 1):
    blocked = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side_h and 0 <= cc < side_w:
                    blocked.add((rr, cc))
    return blocked


def _place_shape(
    shape: tuple[tuple[int, int], ...],
    side_h: int,
    side_w: int,
    blocked,
    axis: int,
    forbidden_dest,
):
    h, w = _shape_dims(shape)
    positions = [(r0, c0) for r0 in range(side_h - h + 1) for c0 in range(side_w - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions[:PLACEMENT_ATTEMPTS]:
        coords = {(r0 + r, c0 + c) for r, c in shape}
        if coords & blocked:
            continue
        mirrored = _mirror_local(coords, axis, side_h, side_w)
        if mirrored & forbidden_dest:
            continue
        return coords
    return None


def _paint(canvas: np.ndarray, coords, color: int) -> None:
    for r, c in coords:
        canvas[r, c] = color


def _build_candidate() -> np.ndarray | None:
    axis = random.choice((0, 1))
    if axis == 1:
        side_h = random.randint(9, 15)
        side_w = random.randint(6, 9)
        full_shape = (side_h, 2 * side_w + 1)
    else:
        side_h = random.randint(6, 9)
        side_w = random.randint(9, 15)
        full_shape = (2 * side_h + 1, side_w)

    max_shape_h = min(5, side_h - 1)
    max_shape_w = min(5, side_w - 1)
    if max_shape_h < 2 or max_shape_w < 2:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        pair_left = _grow_shape(max_shape_h, max_shape_w)
        iso_left = _grow_shape(max_shape_h, max_shape_w)
        iso_right = _grow_shape(max_shape_h, max_shape_w)
        if pair_left is None or iso_left is None or iso_right is None:
            continue
        pair_right = _reflect_shape_for_axis(pair_left, axis)

        pair_key = _shape_mirror_sig(pair_left, axis)
        if pair_key != _normalize_cells(pair_right):
            continue
        if _shape_mirror_sig(iso_left, axis) in {_normalize_cells(pair_right), _normalize_cells(iso_right)}:
            continue
        if _shape_mirror_sig(pair_left, axis) == _normalize_cells(iso_right):
            continue
        break
    else:
        return None

    colors = random.sample(PALETTE, 4)
    left_pair_color, right_pair_color, left_iso_color, right_iso_color = colors

    left_blocked: set[tuple[int, int]] = set()
    right_blocked: set[tuple[int, int]] = set()

    left_pair_coords = _place_shape(pair_left, side_h, side_w, left_blocked, axis, set())
    if left_pair_coords is None:
        return None
    left_blocked |= _expanded(left_pair_coords, side_h, side_w)

    right_pair_coords = _place_shape(pair_right, side_h, side_w, right_blocked, axis, set())
    if right_pair_coords is None:
        return None
    right_blocked |= _expanded(right_pair_coords, side_h, side_w)

    left_iso_coords = _place_shape(iso_left, side_h, side_w, left_blocked, axis, right_pair_coords)
    if left_iso_coords is None:
        return None
    left_blocked |= _expanded(left_iso_coords, side_h, side_w)

    right_iso_coords = _place_shape(iso_right, side_h, side_w, right_blocked, axis, left_pair_coords)
    if right_iso_coords is None:
        return None

    side0 = np.full((side_h, side_w), BG, dtype=int)
    side1 = np.full((side_h, side_w), BG, dtype=int)
    _paint(side0, left_pair_coords, left_pair_color)
    _paint(side1, right_pair_coords, right_pair_color)
    _paint(side0, left_iso_coords, left_iso_color)
    _paint(side1, right_iso_coords, right_iso_color)

    grid = np.full(full_shape, BG, dtype=int)
    if axis == 1:
        grid[:, side_w] = DIVIDER
        grid[:, :side_w] = side0
        grid[:, side_w + 1 :] = side1
    else:
        grid[side_h, :] = DIVIDER
        grid[:side_h, :] = side0
        grid[side_h + 1 :, :] = side1

    try:
        info = _analyze(grid)
        assert len(info["components"]) == 4
        assert len(info["pairs"]) == 1
        assert len(info["unmatched"]) == 2
        assert [len(side) for side in info["side_components"]] == [2, 2]
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-pair puzzle")
