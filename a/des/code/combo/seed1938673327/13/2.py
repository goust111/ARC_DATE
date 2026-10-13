import random

import numpy as np


BG_COLOR = 5
GEN_RETRIES = 300
PALETTE = [c for c in range(10) if c != BG_COLOR]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _u_template(height: int, width: int, direction: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if direction in ("up", "down"):
        mask[:, 0] = True
        mask[:, -1] = True
        if direction == "up":
            mask[-1, :] = True
        else:
            mask[0, :] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        if direction == "left":
            mask[:, -1] = True
        else:
            mask[:, 0] = True
    return mask


def _classify_u(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 3 or w < 3:
        return None
    for direction in ("up", "down", "left", "right"):
        if np.array_equal(mask, _u_template(h, w, direction)):
            return direction
    return None


def _edge_name(r: int, c: int, h: int, w: int) -> str:
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    return "right"


def _find_markers(grid: np.ndarray, bg: int) -> tuple[list[dict], str]:
    h, w = grid.shape
    markers = []
    for coords in _components(grid != bg):
        if coords.shape[0] != 1:
            continue
        r, c = map(int, coords[0])
        if r not in (0, h - 1) and c not in (0, w - 1):
            continue
        markers.append(
            {
                "r": r,
                "c": c,
                "color": int(grid[r, c]),
                "edge": _edge_name(r, c, h, w),
            }
        )

    assert len(markers) == 2, "expected exactly two border marker pixels"
    edges = {marker["edge"] for marker in markers}
    if edges == {"top", "bottom"}:
        axis = "vertical"
    else:
        assert edges == {"left", "right"}, "markers must sit on opposite edges"
        axis = "horizontal"
    return markers, axis


def _find_u_objects(grid: np.ndarray, bg: int, marker_positions: set[tuple[int, int]]) -> list[dict]:
    mask = grid != bg
    for r, c in marker_positions:
        mask[r, c] = False

    objects = []
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        submask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        direction = _classify_u(submask)
        if direction is None:
            continue
        objects.append(
            {
                "color": next(iter(colors)),
                "direction": direction,
                "mask": submask,
                "r0": r0,
                "c0": c0,
                "h": int(submask.shape[0]),
                "w": int(submask.shape[1]),
                "coords": coords,
            }
        )
    return objects


def _target_origin(obj: dict, axis: str, height: int, width: int) -> tuple[int, int]:
    direction = obj["direction"]
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    h = int(obj["h"])
    w = int(obj["w"])
    if axis == "vertical":
        if direction == "up":
            return 1, c0
        if direction == "down":
            return height - h - 1, c0
        return r0, c0
    if direction == "left":
        return r0, 1
    if direction == "right":
        return r0, width - w - 1
    return r0, c0


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    markers, axis = _find_markers(g, bg)
    marker_positions = {(int(marker["r"]), int(marker["c"])) for marker in markers}
    objects = _find_u_objects(g, bg, marker_positions)

    movers = []
    for obj in objects:
        if axis == "vertical" and obj["direction"] in ("up", "down"):
            movers.append(obj)
        if axis == "horizontal" and obj["direction"] in ("left", "right"):
            movers.append(obj)

    out = g.copy()
    for obj in movers:
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = bg

    for obj in movers:
        nr0, nc0 = _target_origin(obj, axis, h, w)
        assert 0 <= nr0 <= h - obj["h"]
        assert 0 <= nc0 <= w - obj["w"]
        rr, cc = np.nonzero(obj["mask"])
        out[nr0 + rr, nc0 + cc] = int(obj["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group U-shaped objects by opening direction using the opposite-edge markers,
    then move the matching groups toward the corresponding marked side.
    """
    return _transform(grid)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _occupy(occupied: np.ndarray, cells: np.ndarray, r0: int, c0: int) -> None:
    occupied[r0 + cells[:, 0], c0 + cells[:, 1]] = True


def _stamp(grid: np.ndarray, cells: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 + cells[:, 0], c0 + cells[:, 1]] = int(color)


def _make_spec(direction: str, color: int, rng: random.Random) -> dict:
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    mask = _u_template(h, w, direction)
    return {
        "direction": direction,
        "color": int(color),
        "h": h,
        "w": w,
        "mask": mask,
        "cells": np.argwhere(mask),
    }


def _shuffle_copy(values: list, rng: random.Random) -> list:
    result = list(values)
    rng.shuffle(result)
    return result


def _place_fixed_row(occupied: np.ndarray, spec: dict, r0: int, rng: random.Random) -> int | None:
    candidates = _shuffle_copy(list(range(1, occupied.shape[1] - spec["w"])), rng)
    for c0 in candidates:
        if _can_place(occupied, r0, c0, spec["h"], spec["w"]):
            return int(c0)
    return None


def _place_fixed_col(occupied: np.ndarray, spec: dict, c0: int, rng: random.Random) -> int | None:
    candidates = _shuffle_copy(list(range(1, occupied.shape[0] - spec["h"])), rng)
    for r0 in candidates:
        if _can_place(occupied, r0, c0, spec["h"], spec["w"]):
            return int(r0)
    return None


def _place_free(
    occupied: np.ndarray,
    spec: dict,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    if row_min > row_max or col_min > col_max:
        return None
    candidates = [(r0, c0) for r0 in range(row_min, row_max + 1) for c0 in range(col_min, col_max + 1)]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        if _can_place(occupied, r0, c0, spec["h"], spec["w"]):
            return int(r0), int(c0)
    return None


def _choose_directions(axis: str, rng: random.Random) -> list[str]:
    if axis == "vertical":
        directions = ["up"] * rng.randint(1, 2) + ["down"] * rng.randint(1, 2)
        if rng.random() < 0.5:
            directions.append(rng.choice(["left", "right"]))
        else:
            directions.extend(_shuffle_copy(["left", "right"], rng))
        return directions

    directions = ["left"] * rng.randint(1, 2) + ["right"] * rng.randint(1, 2)
    if rng.random() < 0.5:
        directions.append(rng.choice(["up", "down"]))
    else:
        directions.extend(_shuffle_copy(["up", "down"], rng))
    return directions


def _grid_shape_for(axis: str, specs: list[dict], rng: random.Random) -> tuple[int, int] | None:
    up = [spec for spec in specs if spec["direction"] == "up"]
    down = [spec for spec in specs if spec["direction"] == "down"]
    left = [spec for spec in specs if spec["direction"] == "left"]
    right = [spec for spec in specs if spec["direction"] == "right"]

    if axis == "vertical":
        max_up_h = max(spec["h"] for spec in up)
        max_down_h = max(spec["h"] for spec in down)
        max_static_h = max(spec["h"] for spec in left + right)
        top_span = sum(spec["w"] for spec in up) + max(0, len(up) - 1)
        bottom_span = sum(spec["w"] for spec in down) + max(0, len(down) - 1)
        min_h = max_up_h + max_down_h + max_static_h + 8
        min_w = max(12, top_span + 2, bottom_span + 2)
    else:
        max_left_w = max(spec["w"] for spec in left)
        max_right_w = max(spec["w"] for spec in right)
        max_static_w = max(spec["w"] for spec in up + down)
        left_span = sum(spec["h"] for spec in left) + max(0, len(left) - 1)
        right_span = sum(spec["h"] for spec in right) + max(0, len(right) - 1)
        min_h = max(12, left_span + 2, right_span + 2)
        min_w = max_left_w + max_right_w + max_static_w + 8

    if min_h > 24 or min_w > 24:
        return None
    h = rng.randint(min_h, 24)
    w = rng.randint(min_w, 24)
    return int(h), int(w)


def _build_expected(axis: str, height: int, width: int, specs: list[dict], markers: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for spec in specs:
        _stamp(grid, spec["cells"], spec["final_r0"], spec["final_c0"], spec["color"])
    for r, c, color in markers:
        grid[int(r), int(c)] = int(color)
    return grid


def _place_vertical(specs: list[dict], height: int, width: int, rng: random.Random) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    groups = {
        "up": [spec for spec in specs if spec["direction"] == "up"],
        "down": [spec for spec in specs if spec["direction"] == "down"],
        "static": [spec for spec in specs if spec["direction"] in ("left", "right")],
    }

    for spec in groups["up"]:
        c0 = _place_fixed_row(occupied, spec, 1, rng)
        if c0 is None:
            return False
        spec["final_r0"] = 1
        spec["final_c0"] = c0
        _occupy(occupied, spec["cells"], spec["final_r0"], spec["final_c0"])

    for spec in groups["down"]:
        r0 = height - spec["h"] - 1
        c0 = _place_fixed_row(occupied, spec, r0, rng)
        if c0 is None:
            return False
        spec["final_r0"] = r0
        spec["final_c0"] = c0
        _occupy(occupied, spec["cells"], spec["final_r0"], spec["final_c0"])

    for spec in groups["static"]:
        pos = _place_free(occupied, spec, 3, height - spec["h"] - 3, 2, width - spec["w"] - 2, rng)
        if pos is None:
            return False
        spec["final_r0"], spec["final_c0"] = pos
        _occupy(occupied, spec["cells"], spec["final_r0"], spec["final_c0"])

    return True


def _place_horizontal(specs: list[dict], height: int, width: int, rng: random.Random) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    groups = {
        "left": [spec for spec in specs if spec["direction"] == "left"],
        "right": [spec for spec in specs if spec["direction"] == "right"],
        "static": [spec for spec in specs if spec["direction"] in ("up", "down")],
    }

    for spec in groups["left"]:
        r0 = _place_fixed_col(occupied, spec, 1, rng)
        if r0 is None:
            return False
        spec["final_r0"] = r0
        spec["final_c0"] = 1
        _occupy(occupied, spec["cells"], spec["final_r0"], spec["final_c0"])

    for spec in groups["right"]:
        c0 = width - spec["w"] - 1
        r0 = _place_fixed_col(occupied, spec, c0, rng)
        if r0 is None:
            return False
        spec["final_r0"] = r0
        spec["final_c0"] = c0
        _occupy(occupied, spec["cells"], spec["final_r0"], spec["final_c0"])

    for spec in groups["static"]:
        pos = _place_free(occupied, spec, 2, height - spec["h"] - 2, 3, width - spec["w"] - 3, rng)
        if pos is None:
            return False
        spec["final_r0"], spec["final_c0"] = pos
        _occupy(occupied, spec["cells"], spec["final_r0"], spec["final_c0"])

    return True


def _make_markers(axis: str, height: int, width: int, colors: list[int], rng: random.Random) -> list[tuple[int, int, int]]:
    marker_colors = rng.sample(colors, 2)
    if axis == "vertical":
        top_c, bottom_c = rng.sample(list(range(1, width - 1)), 2)
        return [(0, top_c, marker_colors[0]), (height - 1, bottom_c, marker_colors[1])]
    left_r, right_r = rng.sample(list(range(1, height - 1)), 2)
    return [(left_r, 0, marker_colors[0]), (right_r, width - 1, marker_colors[1])]


def _build_input(axis: str, height: int, width: int, specs: list[dict], markers: list[tuple[int, int, int]], rng: random.Random) -> np.ndarray | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for r, c, color in markers:
        grid[int(r), int(c)] = int(color)
        occupied[int(r), int(c)] = True

    for spec in specs:
        movable = (axis == "vertical" and spec["direction"] in ("up", "down")) or (
            axis == "horizontal" and spec["direction"] in ("left", "right")
        )
        if not movable:
            spec["input_r0"] = spec["final_r0"]
            spec["input_c0"] = spec["final_c0"]
            if not _can_place(occupied, spec["input_r0"], spec["input_c0"], spec["h"], spec["w"]):
                return None
            _occupy(occupied, spec["cells"], spec["input_r0"], spec["input_c0"])
            _stamp(grid, spec["cells"], spec["input_r0"], spec["input_c0"], spec["color"])

    movers = [spec for spec in specs if "input_r0" not in spec]
    rng.shuffle(movers)
    for spec in movers:
        if axis == "vertical":
            row_min = 3
            row_max = height - spec["h"] - 3
            candidates = _shuffle_copy(list(range(row_min, row_max + 1)), rng)
            placed = False
            for r0 in candidates:
                if r0 == spec["final_r0"]:
                    continue
                c0 = spec["final_c0"]
                if _can_place(occupied, r0, c0, spec["h"], spec["w"]):
                    spec["input_r0"] = r0
                    spec["input_c0"] = c0
                    placed = True
                    break
            if not placed:
                return None
        else:
            col_min = 3
            col_max = width - spec["w"] - 3
            candidates = _shuffle_copy(list(range(col_min, col_max + 1)), rng)
            placed = False
            for c0 in candidates:
                if c0 == spec["final_c0"]:
                    continue
                r0 = spec["final_r0"]
                if _can_place(occupied, r0, c0, spec["h"], spec["w"]):
                    spec["input_r0"] = r0
                    spec["input_c0"] = c0
                    placed = True
                    break
            if not placed:
                return None

        _occupy(occupied, spec["cells"], spec["input_r0"], spec["input_c0"])
        _stamp(grid, spec["cells"], spec["input_r0"], spec["input_c0"], spec["color"])

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    axis = rng.choice(["vertical", "horizontal"])
    directions = _choose_directions(axis, rng)
    color_pool = rng.sample(PALETTE, len(directions) + 2)
    specs = [_make_spec(direction, color, rng) for direction, color in zip(directions, color_pool)]
    shape = _grid_shape_for(axis, specs, rng)
    if shape is None:
        return None
    height, width = shape

    if axis == "vertical":
        if not _place_vertical(specs, height, width, rng):
            return None
    else:
        if not _place_horizontal(specs, height, width, rng):
            return None

    markers = _make_markers(axis, height, width, color_pool[-2:], rng)
    expected = _build_expected(axis, height, width, specs, markers)
    grid = _build_input(axis, height, width, specs, markers, rng)
    if grid is None or np.array_equal(grid, expected):
        return None
    if not np.array_equal(_transform(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
