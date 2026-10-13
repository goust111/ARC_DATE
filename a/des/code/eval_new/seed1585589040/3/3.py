from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
MAROON = 9
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_REL = {
    "top": (0, 1),
    "right": (1, 2),
    "bottom": (2, 1),
    "left": (1, 0),
}
SIDE_TO_DIR = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
TOWER_COLORS = (1, 2, 3, 5, 6, 7, 8)

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 300


def _antenna_coord(top: int, left: int, side: str) -> tuple[int, int]:
    dr, dc = SIDE_TO_REL[side]
    return top + dr, left + dc


def _box_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + dr, left + dc)
        for dr in range(3)
        for dc in range(3)
    }


def _expanded_box(top: int, left: int, shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(top - pad, top + 3 + pad):
        for c in range(left - pad, left + 3 + pad):
            if 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
    return cells


def _tower_from_window(g: np.ndarray, top: int, left: int) -> dict | None:
    window = g[top : top + 3, left : left + 3]
    body = int(window[1, 1])
    if body in (BG, YELLOW, MAROON):
        return None

    corners = (window[0, 0], window[0, 2], window[2, 0], window[2, 2], window[1, 1])
    if any(int(value) != body for value in corners):
        return None

    antennas: list[str] = []
    for side in SIDES:
        rr, cc = SIDE_TO_REL[side]
        value = int(window[rr, cc])
        if value == YELLOW:
            antennas.append(side)
        elif value != body:
            return None

    if len(antennas) not in (1, 2):
        return None
    return {"top": top, "left": left, "color": body, "antennas": tuple(antennas)}


def _find_towers(g: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = g.shape
    towers: list[dict] = []
    tower_mask = np.zeros((h, w), dtype=bool)

    for top in range(h - 2):
        for left in range(w - 2):
            tower = _tower_from_window(g, top, left)
            if tower is None:
                continue

            cells = _box_cells(top, left)
            assert not any(tower_mask[r, c] for r, c in cells), "3x3 towers must not overlap"
            for r, c in cells:
                tower_mask[r, c] = True
            towers.append(tower)

    assert towers, "expected at least one signal tower"
    assert np.array_equal(tower_mask, g != BG), "input should contain only black background and 3x3 towers"

    colors = [int(tower["color"]) for tower in towers]
    assert len(colors) == len(set(colors)), "signal tower colors should be unique"
    return towers, tower_mask


def _ray_path(top: int, left: int, side: str, tower_mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = tower_mask.shape
    antenna_r, antenna_c = _antenna_coord(top, left, side)
    dr, dc = SIDE_TO_DIR[side]

    rr = antenna_r + dr
    cc = antenna_c + dc
    step = 1
    path: list[tuple[int, int, int]] = []
    while 0 <= rr < h and 0 <= cc < w and not bool(tower_mask[rr, cc]):
        path.append((rr, cc, step))
        rr += dr
        cc += dc
        step += 1
    return path


def _is_true_cross(pre: np.ndarray, ray_mask: np.ndarray, tower_mask: np.ndarray, r: int, c: int) -> bool:
    h, w = pre.shape

    def ray_color(rr: int, cc: int) -> int | None:
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if bool(tower_mask[rr, cc]) or not bool(ray_mask[rr, cc]):
            return None
        value = int(pre[rr, cc])
        if value in (BG, YELLOW, MAROON):
            return None
        return value

    left = ray_color(r, c - 1)
    right = ray_color(r, c + 1)
    up = ray_color(r - 1, c)
    down = ray_color(r + 1, c)
    return (
        left is not None
        and left == right
        and up is not None
        and up == down
        and left != up
    )


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    towers, tower_mask = _find_towers(g)

    ray_mask = np.zeros_like(g, dtype=bool)
    hits: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    order = 0
    for tower in sorted(towers, key=lambda item: (item["top"], item["left"])):
        for side in SIDES:
            if side not in tower["antennas"]:
                continue
            for r, c, step in _ray_path(int(tower["top"]), int(tower["left"]), side, tower_mask):
                hits.setdefault((r, c), []).append((step, order, int(tower["color"])))
                ray_mask[r, c] = True
            order += 1

    pre = g.copy()
    fallback: dict[tuple[int, int], int] = {}
    conflicts: list[tuple[int, int]] = []
    for (r, c), entries in hits.items():
        colors = {color for _step, _order, color in entries}
        if len(colors) == 1:
            pre[r, c] = next(iter(colors))
        else:
            pre[r, c] = MAROON
            conflicts.append((r, c))
        fallback[(r, c)] = max(entries, key=lambda item: (item[0], item[1]))[2]

    out = pre.copy()
    kept = 0
    reverted = 0
    for r, c in conflicts:
        if _is_true_cross(pre, ray_mask, tower_mask, r, c):
            kept += 1
        else:
            out[r, c] = fallback[(r, c)]
            reverted += 1

    info = {
        "tower_count": len(towers),
        "kept": kept,
        "reverted": reverted,
        "conflicts": len(conflicts),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast colored rays from yellow tower antennas, keeping only true perpendicular crossings as maroon.

    Input structure:
    - Background is black.
    - The foreground consists of several non-overlapping 3x3 solid-color towers, each with a unique body color.
    - On each tower, one or two side-center cells are replaced by yellow antenna pixels that point outward from the tower.

    Rule:
    - Keep every 3x3 tower unchanged, including its yellow antenna cells.
    - From each antenna, draw a straight orthogonal ray in the pointed direction using the tower's body color, starting just outside the tower and stopping before another tower or outside the grid.
    - Any cell visited by rays of two or more different colors first becomes maroon.
    - Then inspect each maroon cell: keep it maroon only when its four orthogonal neighbors show a genuine cross pattern with one color continuing horizontally and a different color continuing vertically; otherwise restore the cell to the latest-arriving ray color.

    Key insight:
    - The antenna positions encode the ray directions, and the final maroon pixels are not all conflicts: only neighborhood-confirmed cross intersections survive the cleanup step.
    """
    out, _info = _simulate(np.asarray(grid, dtype=int))
    return out


def _tower_mask_from_specs(shape: tuple[int, int], towers: list[dict]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for tower in towers:
        for r, c in _box_cells(int(tower["top"]), int(tower["left"])):
            mask[r, c] = True
    return mask


def _path_cells(spec: dict, side: str, tower_mask: np.ndarray) -> list[tuple[int, int]]:
    return [(r, c) for r, c, _step in _ray_path(int(spec["top"]), int(spec["left"]), side, tower_mask)]


def _position_clear(top: int, left: int, shape: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
    h, w = shape
    if not (1 <= top <= h - 4 and 1 <= left <= w - 4):
        return False
    return not (_box_cells(top, left) & blocked)


def _commit_tower_reservation(
    tower_reserved: set[tuple[int, int]],
    spec: dict,
    shape: tuple[int, int],
) -> None:
    tower_reserved.update(_expanded_box(int(spec["top"]), int(spec["left"]), shape))


def _place_cross(
    shape: tuple[int, int],
    towers: list[dict],
    tower_reserved: set[tuple[int, int]],
    ray_reserved: set[tuple[int, int]],
    colors: tuple[int, int],
) -> list[dict] | None:
    h, w = shape
    blocked = tower_reserved | ray_reserved

    for _ in range(PLACE_ATTEMPTS):
        cross_r = random.randint(3, h - 4)
        cross_c = random.randint(3, w - 4)
        from_left = random.choice((True, False))
        from_top = random.choice((True, False))

        if from_left:
            max_dh = min(8, cross_c - 3)
            if max_dh < 2:
                continue
            dh = random.randint(2, max_dh)
            h_top = cross_r - 1
            h_left = cross_c - dh - 2
            h_side = "right"
        else:
            max_dh = min(8, w - 4 - cross_c)
            if max_dh < 2:
                continue
            dh = random.randint(2, max_dh)
            h_top = cross_r - 1
            h_left = cross_c + dh
            h_side = "left"

        if from_top:
            max_dv = min(8, cross_r - 3)
            if max_dv < 2:
                continue
            dv = random.randint(2, max_dv)
            v_top = cross_r - dv - 2
            v_left = cross_c - 1
            v_side = "bottom"
        else:
            max_dv = min(8, h - 4 - cross_r)
            if max_dv < 2:
                continue
            dv = random.randint(2, max_dv)
            v_top = cross_r + dv
            v_left = cross_c - 1
            v_side = "top"

        if not _position_clear(h_top, h_left, shape, blocked):
            continue
        if not _position_clear(v_top, v_left, shape, blocked):
            continue
        if _box_cells(h_top, h_left) & _expanded_box(v_top, v_left, shape):
            continue

        horizontal = {"top": h_top, "left": h_left, "color": colors[0], "antennas": [h_side]}
        vertical = {"top": v_top, "left": v_left, "color": colors[1], "antennas": [v_side]}
        temp_towers = towers + [horizontal, vertical]
        tower_mask = _tower_mask_from_specs(shape, temp_towers)

        path_h = _path_cells(horizontal, h_side, tower_mask)
        path_v = _path_cells(vertical, v_side, tower_mask)
        if not path_h or not path_v:
            continue
        if (cross_r, cross_c) not in path_h or (cross_r, cross_c) not in path_v:
            continue
        if set(path_h) & ray_reserved:
            continue
        if set(path_v) & ray_reserved:
            continue
        if set(path_h) & set(path_v) != {(cross_r, cross_c)}:
            continue

        return [horizontal, vertical]
    return None


def _place_overlap(
    shape: tuple[int, int],
    towers: list[dict],
    tower_reserved: set[tuple[int, int]],
    ray_reserved: set[tuple[int, int]],
    colors: tuple[int, int],
) -> list[dict] | None:
    h, w = shape
    blocked = tower_reserved | ray_reserved

    for _ in range(PLACE_ATTEMPTS):
        horizontal = random.choice((True, False))
        if horizontal:
            row = random.randint(2, h - 3)
            deltas = [delta for delta in range(5, w - 4) if delta % 2 == 1]
            if not deltas:
                continue
            delta = random.choice(deltas)
            max_left = w - 4 - delta
            if max_left < 1:
                continue
            left_left = random.randint(1, max_left)
            right_left = left_left + delta

            first = {"top": row - 1, "left": left_left, "color": colors[0], "antennas": ["right"]}
            second = {"top": row - 1, "left": right_left, "color": colors[1], "antennas": ["left"]}
            sides = ("right", "left")
        else:
            col = random.randint(2, w - 3)
            deltas = [delta for delta in range(5, h - 4) if delta % 2 == 1]
            if not deltas:
                continue
            delta = random.choice(deltas)
            max_top = h - 4 - delta
            if max_top < 1:
                continue
            top_top = random.randint(1, max_top)
            bottom_top = top_top + delta

            first = {"top": top_top, "left": col - 1, "color": colors[0], "antennas": ["bottom"]}
            second = {"top": bottom_top, "left": col - 1, "color": colors[1], "antennas": ["top"]}
            sides = ("bottom", "top")

        if not _position_clear(int(first["top"]), int(first["left"]), shape, blocked):
            continue
        if not _position_clear(int(second["top"]), int(second["left"]), shape, blocked):
            continue
        if _box_cells(int(first["top"]), int(first["left"])) & _expanded_box(int(second["top"]), int(second["left"]), shape):
            continue

        temp_towers = towers + [first, second]
        tower_mask = _tower_mask_from_specs(shape, temp_towers)
        path_a = _path_cells(first, sides[0], tower_mask)
        path_b = _path_cells(second, sides[1], tower_mask)
        if not path_a or not path_b:
            continue
        if set(path_a) & ray_reserved:
            continue
        if set(path_b) & ray_reserved:
            continue
        if len(set(path_a) & set(path_b)) < 2:
            continue
        return [first, second]
    return None


def _place_filler(
    shape: tuple[int, int],
    towers: list[dict],
    tower_reserved: set[tuple[int, int]],
    ray_reserved: set[tuple[int, int]],
    color: int,
) -> bool:
    blocked = tower_reserved | ray_reserved

    for _ in range(PLACE_ATTEMPTS):
        top = random.randint(1, shape[0] - 4)
        left = random.randint(1, shape[1] - 4)
        if not _position_clear(top, left, shape, blocked):
            continue

        spec = {"top": top, "left": left, "color": color, "antennas": []}
        tower_mask = _tower_mask_from_specs(shape, towers + [spec])
        safe: list[tuple[str, list[tuple[int, int]]]] = []
        for side in SIDES:
            path = _path_cells(spec, side, tower_mask)
            if not path or set(path) & ray_reserved:
                continue
            safe.append((side, path))
        if not safe:
            continue

        antenna_count = 2 if len(safe) >= 2 and random.random() < 0.5 else 1
        choices = random.sample(safe, antenna_count)
        spec["antennas"] = [side for side, _path in choices]
        towers.append(spec)
        _commit_tower_reservation(tower_reserved, spec, shape)
        for _side, path in choices:
            ray_reserved.update(path)
        return True
    return False


def _augment_antennas(
    shape: tuple[int, int],
    towers: list[dict],
    ray_reserved: set[tuple[int, int]],
) -> None:
    tower_mask = _tower_mask_from_specs(shape, towers)
    indices = list(range(len(towers)))
    random.shuffle(indices)

    for idx in indices:
        spec = towers[idx]
        if len(spec["antennas"]) != 1 or random.random() < 0.4:
            continue

        available: list[tuple[str, list[tuple[int, int]]]] = []
        for side in SIDES:
            if side in spec["antennas"]:
                continue
            path = _path_cells(spec, side, tower_mask)
            if not path or set(path) & ray_reserved:
                continue
            available.append((side, path))
        if not available:
            continue

        side, path = random.choice(available)
        spec["antennas"].append(side)
        ray_reserved.update(path)


def _render_input(shape: tuple[int, int], towers: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for tower in towers:
        top = int(tower["top"])
        left = int(tower["left"])
        color = int(tower["color"])
        grid[top : top + 3, left : left + 3] = color
        for side in tower["antennas"]:
            r, c = _antenna_coord(top, left, side)
            grid[r, c] = YELLOW
    return grid


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("cross", "overlap", "both"))
    if mode == "both":
        h = random.randint(18, 28)
        w = random.randint(18, 28)
    else:
        h = random.randint(14, 24)
        w = random.randint(14, 24)

    special_count = 4 if mode == "both" else 2
    filler_count = random.randint(0, 2)
    total_towers = special_count + filler_count
    if total_towers > len(TOWER_COLORS):
        return None
    colors = list(random.sample(TOWER_COLORS, total_towers))

    towers: list[dict] = []
    tower_reserved: set[tuple[int, int]] = set()
    ray_reserved: set[tuple[int, int]] = set()
    color_idx = 0

    steps = ["cross", "overlap"] if mode == "both" else [mode]
    if mode == "both" and random.random() < 0.5:
        steps.reverse()

    for step in steps:
        pair = (colors[color_idx], colors[color_idx + 1])
        color_idx += 2
        if step == "cross":
            placed = _place_cross((h, w), towers, tower_reserved, ray_reserved, pair)
        else:
            placed = _place_overlap((h, w), towers, tower_reserved, ray_reserved, pair)
        if placed is None:
            return None

        towers.extend(placed)
        for spec in placed:
            _commit_tower_reservation(tower_reserved, spec, (h, w))
        tower_mask = _tower_mask_from_specs((h, w), towers)
        for spec in placed:
            for side in spec["antennas"]:
                ray_reserved.update(_path_cells(spec, side, tower_mask))

    for _ in range(filler_count):
        if not _place_filler((h, w), towers, tower_reserved, ray_reserved, colors[color_idx]):
            return None
        color_idx += 1

    _augment_antennas((h, w), towers, ray_reserved)
    grid = _render_input((h, w), towers)

    try:
        out, info = _simulate(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if mode == "cross" and not (info["kept"] >= 1 and info["reverted"] == 0):
        return None
    if mode == "overlap" and not (info["kept"] == 0 and info["reverted"] >= 1):
        return None
    if mode == "both" and not (info["kept"] >= 1 and info["reverted"] >= 1):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
