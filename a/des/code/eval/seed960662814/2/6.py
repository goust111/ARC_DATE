from __future__ import annotations

import random

import numpy as np


PATH_COLOR = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
SYMMETRIES = ("id", "rot90", "rot180", "rot270", "flipud", "fliplr", "transpose", "anti")


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


STAMP_LIBRARY = [
    _parse_mask(["111", "110", "010"]),
    _parse_mask(["111", "011", "010"]),
    _parse_mask(["110", "111", "010"]),
    _parse_mask(["011", "111", "010"]),
    _parse_mask(["101", "111", "010"]),
    _parse_mask(["1110", "0110", "0100", "0100"]),
    _parse_mask(["0111", "0110", "0010", "0010"]),
    _parse_mask(["1110", "1100", "0110", "0010"]),
    _parse_mask(["0110", "1110", "0100", "0100"]),
    _parse_mask(["0111", "0011", "0010", "0010"]),
]


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in directions:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_patch(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = True
    bottom = int(rel[:, 0].max())
    bottom_cols = sorted(int(c) for c in rel[rel[:, 0] == bottom, 1])
    anchor = (bottom, bottom_cols[len(bottom_cols) // 2])
    return patch, anchor


def _rotate_mask_anchor(mask: np.ndarray, anchor: tuple[int, int], k: int) -> tuple[np.ndarray, tuple[int, int]]:
    one_hot = np.zeros(mask.shape, dtype=int)
    one_hot[anchor[0], anchor[1]] = 1
    rotated_mask = np.rot90(mask, k)
    rotated_anchor = tuple(int(v) for v in np.argwhere(np.rot90(one_hot, k) == 1)[0])
    return rotated_mask.astype(bool), rotated_anchor


def _absolute_coords(mask: np.ndarray, anchor: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
    rel = np.argwhere(mask) - np.array(anchor, dtype=int)
    return rel + np.array(target, dtype=int)


def _placement_fits(
    grid: np.ndarray,
    coords: np.ndarray,
    anchor_pos: tuple[int, int],
    bg: int,
) -> bool:
    h, w = grid.shape
    ar, ac = anchor_pos
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if rr == ar and cc == ac:
            continue
        if int(grid[rr, cc]) != int(bg):
            return False
    return True


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _path_pressure(path_mask: np.ndarray, marker: tuple[int, int], direction: str, radius: int = 4) -> int:
    h, w = path_mask.shape
    r, c = marker
    if direction == "up":
        rs = slice(max(0, r - radius), r)
        cs = slice(max(0, c - radius), min(w, c + radius + 1))
    elif direction == "down":
        rs = slice(r + 1, min(h, r + radius + 1))
        cs = slice(max(0, c - radius), min(w, c + radius + 1))
    elif direction == "left":
        rs = slice(max(0, r - radius), min(h, r + radius + 1))
        cs = slice(max(0, c - radius), c)
    else:
        rs = slice(max(0, r - radius), min(h, r + radius + 1))
        cs = slice(c + 1, min(w, c + radius + 1))
    return int(path_mask[rs, cs].sum())


def _segment_orientation(path_mask: np.ndarray, marker: tuple[int, int]) -> str:
    h, w = path_mask.shape
    r, c = marker
    neighbors = {(nr - r, nc - c) for nr, nc in _neighbors4(r, c, h, w) if path_mask[nr, nc]}
    if neighbors == {(0, -1), (0, 1)}:
        return "horizontal"
    if neighbors == {(-1, 0), (1, 0)}:
        return "vertical"
    raise AssertionError("marker must lie on a straight path segment")


def _infer_path_and_objects(grid: np.ndarray) -> tuple[int, int, dict[int, tuple[int, int]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    assert colors, "expected foreground colors"

    counts = {color: int(np.sum(g == color)) for color in colors}
    max_count = max(counts.values())
    path_candidates = [color for color, count in counts.items() if count == max_count]
    assert len(path_candidates) == 1, "expected a unique path color"
    path_color = int(path_candidates[0])

    markers: dict[int, tuple[int, int]] = {}
    stamps: dict[int, np.ndarray] = {}
    stamp_coords: dict[int, np.ndarray] = {}
    for color in colors:
        if color == path_color:
            continue
        comps = _components(g == color, connectivity=4)
        assert len(comps) == 2, "each stamp color should appear once as a stamp and once as a marker"
        comps.sort(key=len)
        marker_comp, stamp_comp = comps
        assert len(marker_comp) == 1 and len(stamp_comp) > 1
        markers[color] = tuple(int(v) for v in marker_comp[0])
        patch, _anchor = _extract_patch(stamp_comp)
        stamps[color] = patch
        stamp_coords[color] = stamp_comp
    assert markers, "expected at least one colored marker"
    return bg, path_color, markers, stamps, stamp_coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored marker on the grey polyline with the same-colored monochrome stamp,
    rotate the stamp so its contact edge aligns with the local path direction, and place it
    outward from the path while removing the original loose stamp.
    """
    g = np.asarray(grid, dtype=int)
    bg, path_color, markers, stamps, stamp_coords = _infer_path_and_objects(g)

    path_mask = g == int(path_color)
    for marker in markers.values():
        path_mask[marker[0], marker[1]] = True

    out = g.copy()
    for coords in stamp_coords.values():
        out[coords[:, 0], coords[:, 1]] = int(bg)

    dir_to_rotation = {"up": 0, "down": 2, "left": 1, "right": 3}
    for color in sorted(stamps):
        marker = markers[color]
        patch = stamps[color]
        anchor = _extract_patch(stamp_coords[color])[1]
        orientation = _segment_orientation(path_mask, marker)
        candidate_dirs = ["up", "down"] if orientation == "horizontal" else ["left", "right"]

        choices = []
        for direction in candidate_dirs:
            rotated, rotated_anchor = _rotate_mask_anchor(patch, anchor, dir_to_rotation[direction])
            coords = _absolute_coords(rotated, rotated_anchor, marker)
            if _placement_fits(out, coords, marker, bg):
                pressure = _path_pressure(path_mask, marker, direction)
                choices.append((pressure, direction, coords))

        assert choices, "expected at least one outward placement"
        choices.sort(key=lambda item: (item[0], item[1]))
        _paint_coords(out, choices[0][2], color)

    return out


def _render_polyline(shape: tuple[int, int], bg: int, points: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            a, b = sorted((c0, c1))
            grid[int(r0), a : b + 1] = PATH_COLOR
        else:
            a, b = sorted((r0, r1))
            grid[a : b + 1, int(c0)] = PATH_COLOR
    return grid


def _snake_layout() -> tuple[int, int, list[tuple[int, int]], dict[str, tuple[int, int]]]:
    h = random.randint(20, 27)
    w = random.randint(20, 27)

    top = random.randint(3, 4)
    gaps = [random.randint(3, 4) for _ in range(3)]
    rows = [top]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    if rows[-1] > h - 4:
        raise ValueError("not enough bottom clearance")

    left = random.randint(3, 4)
    right = w - 1 - random.randint(3, 4)
    if right - left < 9:
        raise ValueError("horizontal span too short")

    points = [
        (rows[0], left),
        (rows[0], right),
        (rows[1], right),
        (rows[1], left),
        (rows[2], left),
        (rows[2], right),
        (rows[3], right),
        (rows[3], left),
    ]
    slots = {
        "up": (rows[0], right - 1),
        "left": (rows[1] + 1, left),
        "right": (rows[2] + 1, right),
        "down": (rows[3], right - 1),
    }
    return h, w, points, slots


def _direction_rotation(direction: str) -> int:
    return {"up": 0, "down": 2, "left": 1, "right": 3}[direction]


def _opposite_direction(direction: str) -> str:
    return {"up": "down", "down": "up", "left": "right", "right": "left"}[direction]


def _candidate_slot_valid(
    base_grid: np.ndarray,
    mask: np.ndarray,
    marker: tuple[int, int],
    direction: str,
    bg: int,
) -> tuple[bool, np.ndarray]:
    anchor = _extract_patch(np.argwhere(mask))[1]
    rotated, rotated_anchor = _rotate_mask_anchor(mask, anchor, _direction_rotation(direction))
    coords = _absolute_coords(rotated, rotated_anchor, marker)
    return _placement_fits(base_grid, coords, marker, bg), coords


def _apply_symmetry(grid: np.ndarray, sym: str) -> np.ndarray:
    if sym == "id":
        return grid.copy()
    if sym == "rot90":
        return np.rot90(grid, 1).copy()
    if sym == "rot180":
        return np.rot90(grid, 2).copy()
    if sym == "rot270":
        return np.rot90(grid, 3).copy()
    if sym == "flipud":
        return np.flipud(grid).copy()
    if sym == "fliplr":
        return np.fliplr(grid).copy()
    if sym == "transpose":
        return grid.T.copy()
    if sym == "anti":
        return np.fliplr(np.rot90(grid, 1)).copy()
    raise ValueError(f"unknown symmetry {sym}")


def _can_scatter_stamp(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = mask.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rel = np.argwhere(mask)
    for dr, dc in rel:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if int(canvas[rr, cc]) != int(bg):
            return False
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if np.any(canvas[r1:r2, c1:c2] != int(bg)):
            return False
    return True


def _scatter_original_stamps(
    base_grid: np.ndarray,
    chosen: list[tuple[str, int, np.ndarray]],
    bg: int,
) -> np.ndarray | None:
    out = base_grid.copy()
    order = sorted(chosen, key=lambda item: int(item[2].sum()), reverse=True)
    for _slot, color, mask in order:
        h, w = mask.shape
        positions = [(r0, c0) for r0 in range(out.shape[0] - h + 1) for c0 in range(out.shape[1] - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_scatter_stamp(out, mask, r0, c0, bg):
                continue
            rel = np.argwhere(mask)
            abs_coords = rel + np.array([r0, c0], dtype=int)
            out[abs_coords[:, 0], abs_coords[:, 1]] = int(color)
            placed = True
            break
        if not placed:
            return None
    return out


def _build_pair() -> tuple[np.ndarray, np.ndarray] | None:
    try:
        h, w, points, slots = _snake_layout()
    except ValueError:
        return None

    bg = random.choice(COLOR_POOL)
    available_colors = [color for color in COLOR_POOL if color != int(bg)]
    n_objects = random.choice([3, 4])
    chosen_slots = random.sample(["up", "left", "right", "down"], n_objects)
    chosen_colors = random.sample(available_colors, n_objects)
    chosen_masks = random.sample(STAMP_LIBRARY, n_objects)
    chosen = list(zip(chosen_slots, chosen_colors, chosen_masks))

    base = _render_polyline((h, w), bg, points)
    expected = base.copy()

    placements: list[tuple[int, np.ndarray]] = []
    for slot_name, color, mask in chosen:
        marker = slots[slot_name]
        valid, coords = _candidate_slot_valid(base, mask, marker, slot_name, bg)
        if not valid:
            return None
        opposite_valid, _ = _candidate_slot_valid(base, mask, marker, _opposite_direction(slot_name), bg)
        if opposite_valid:
            return None
        for existing_color, existing_coords in placements:
            if set(map(tuple, coords.tolist())) & set(map(tuple, existing_coords.tolist())):
                return None
        placements.append((color, coords))
        _paint_coords(expected, coords, color)

    inp_base = base.copy()
    for slot_name, color, _mask in chosen:
        r, c = slots[slot_name]
        inp_base[int(r), int(c)] = int(color)

    inp = _scatter_original_stamps(inp_base, chosen, bg)
    if inp is None:
        return None

    sym = random.choice(SYMMETRIES)
    return _apply_symmetry(inp, sym), _apply_symmetry(expected, sym)


def _valid_library() -> bool:
    for mask in STAMP_LIBRARY:
        comps = _components(mask, connectivity=4)
        if len(comps) != 1:
            return False
        bottom = int(np.argwhere(mask)[:, 0].max())
        if int(mask[bottom].sum()) != 1:
            return False
    return True


def generate_input() -> np.ndarray:
    assert _valid_library(), "stamp library must contain connected single-tip shapes"
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_pair()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid sample for task 6")
