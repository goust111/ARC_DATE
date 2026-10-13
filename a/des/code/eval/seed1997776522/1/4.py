from __future__ import annotations

import numpy as np


BG = 0
GRAY = 5
TOWER_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CATEGORY_ORDER = ("tie", "multi", "single", "unhit")

GRID_H_RANGE = (16, 26)
GRID_W_RANGE = (16, 26)
RECEIVER_COUNT_RANGE = (3, 6)
MAX_GENERATE_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _receiver_cells(r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return ((r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1))


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _mark_reserved(reserved: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in _expand_cells(cells, reserved.shape):
        reserved[r, c] = True


def _make_tower(color: int, r0: int, c0: int, direction: str, attach: int) -> dict:
    assert 0 <= attach <= 2
    tower_cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
    if direction == "up":
        ac = c0 + attach
        antenna = ((r0 - 1, ac), (r0 - 2, ac))
        end = (r0 - 2, ac)
    elif direction == "down":
        ac = c0 + attach
        antenna = ((r0 + 3, ac), (r0 + 4, ac))
        end = (r0 + 4, ac)
    elif direction == "left":
        ar = r0 + attach
        antenna = ((ar, c0 - 1), (ar, c0 - 2))
        end = (ar, c0 - 2)
    else:
        assert direction == "right"
        ar = r0 + attach
        antenna = ((ar, c0 + 3), (ar, c0 + 4))
        end = (ar, c0 + 4)

    return {
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "direction": direction,
        "step": DIR_STEPS[direction],
        "attach": int(attach),
        "tower_cells": tower_cells,
        "antenna": antenna,
        "end": end,
    }


def _tower_all_cells(tower: dict) -> set[tuple[int, int]]:
    return set(tower["tower_cells"]) | set(tower["antenna"])


def _can_place(cells: set[tuple[int, int]], reserved: np.ndarray) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(reserved[r, c]):
            return False
    return True


def _paint_tower(grid: np.ndarray, reserved: np.ndarray, tower: dict) -> None:
    for r, c in tower["tower_cells"]:
        grid[r, c] = int(tower["color"])
    for r, c in tower["antenna"]:
        grid[r, c] = GRAY
    _mark_reserved(reserved, _tower_all_cells(tower))


def _hit_distance(receiver: dict, tower: dict) -> int | None:
    r0 = int(receiver["r0"])
    c0 = int(receiver["c0"])
    r1 = r0 + 1
    c1 = c0 + 1
    end_r, end_c = tower["end"]
    dr, dc = tower["step"]

    if dr == 0:
        if end_r not in (r0, r1):
            return None
        if dc > 0 and c0 > end_c:
            return int(c0 - end_c)
        if dc < 0 and c1 < end_c:
            return int(end_c - c1)
        return None

    if end_c not in (c0, c1):
        return None
    if dr > 0 and r0 > end_r:
        return int(r0 - end_r)
    if dr < 0 and r1 < end_r:
        return int(end_r - r1)
    return None


def _receiver_signal(receiver: dict, towers: list[dict]) -> dict:
    hits: list[tuple[int, int, int, int]] = []
    for tower in towers:
        distance = _hit_distance(receiver, tower)
        if distance is None:
            continue
        hits.append((int(distance), int(tower["r0"]), int(tower["c0"]), int(tower["color"])))

    hits.sort()
    if not hits:
        return {"hits": hits, "best_color": GRAY, "category": "unhit", "tie": False}

    best_distance = int(hits[0][0])
    tie = sum(int(hit[0]) == best_distance for hit in hits) >= 2
    if tie:
        category = "tie"
    elif len(hits) >= 2:
        category = "multi"
    else:
        category = "single"
    return {"hits": hits, "best_color": int(hits[0][3]), "category": category, "tie": tie}


def _decode_antenna(
    antenna_cells: tuple[tuple[int, int], ...],
    bbox: tuple[int, int, int, int],
) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rows = sorted(r for r, _ in antenna_cells)
    cols = sorted(c for _, c in antenna_cells)

    if cols[0] == cols[1] and rows[1] == rows[0] + 1:
        col = cols[0]
        if rows == [r0 - 2, r0 - 1] and c0 <= col <= c1:
            return "up", (rows[0], col)
        if rows == [r1 + 1, r1 + 2] and c0 <= col <= c1:
            return "down", (rows[1], col)

    if rows[0] == rows[1] and cols[1] == cols[0] + 1:
        row = rows[0]
        if cols == [c0 - 2, c0 - 1] and r0 <= row <= r1:
            return "left", (row, cols[0])
        if cols == [c1 + 1, c1 + 2] and r0 <= row <= r1:
            return "right", (row, cols[1])

    raise AssertionError("each tower must have one outward 2-cell grey antenna")


def _detect_grey_objects(grid: np.ndarray) -> tuple[list[tuple[tuple[int, int], ...]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    antennas: list[tuple[tuple[int, int], ...]] = []
    receivers: list[dict] = []
    for coords in _components_from_mask(g == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        pts = tuple(sorted((int(r), int(c)) for r, c in coords))
        if coords.shape[0] == 4:
            assert (r1, c1) == (r0 + 1, c0 + 1), "grey 4-cell objects must be 2x2 receivers"
            assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GRAY))
            receivers.append({"r0": int(r0), "c0": int(c0), "cells": _receiver_cells(int(r0), int(c0))})
        elif coords.shape[0] == 2:
            same_row = int(r0) == int(r1) and int(c1) == int(c0) + 1
            same_col = int(c0) == int(c1) and int(r1) == int(r0) + 1
            assert same_row or same_col, "grey 2-cell objects must be straight antennas"
            antennas.append(pts)
        else:
            raise AssertionError("grey objects must be either 2-cell antennas or 2x2 receivers")

    receivers.sort(key=lambda receiver: (int(receiver["r0"]), int(receiver["c0"])))
    return antennas, receivers


def _detect_towers(grid: np.ndarray, antennas: list[tuple[tuple[int, int], ...]]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    towers: list[dict] = []
    used_antennas: set[int] = set()
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GRAY)):
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 9, "each tower color must occupy a 3x3 block"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1, c1) == (r0 + 2, c0 + 2)
        assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color))

        tower_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        matches: list[tuple[int, tuple[tuple[int, int], ...], str, tuple[int, int]]] = []
        for idx, antenna in enumerate(antennas):
            try:
                direction, end = _decode_antenna(antenna, (r0, c0, r1, c1))
            except AssertionError:
                continue
            matches.append((idx, antenna, direction, end))

        assert len(matches) == 1, "each tower must match exactly one antenna"
        antenna_idx, antenna_cells, direction, end = matches[0]
        assert antenna_idx not in used_antennas, "antenna components must be unique to one tower"
        used_antennas.add(antenna_idx)
        towers.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "direction": direction,
                "step": DIR_STEPS[direction],
                "antenna": antenna_cells,
                "end": end,
                "tower_cells": tower_cells,
            }
        )

    towers.sort(key=lambda tower: (int(tower["r0"]), int(tower["c0"]), int(tower["color"])))
    assert towers, "expected at least one tower"
    assert len(used_antennas) == len(antennas), "every antenna must belong to a tower"
    return towers


def _detect_receivers(receivers: list[dict]) -> list[dict]:
    assert receivers, "expected at least one receiver"
    return receivers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast along each antenna direction; every receiver takes the color of the nearest tower ray that hits it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(g.min()) >= 0 and int(g.max()) <= 9
    assert bool(np.any(g == GRAY))

    antennas, receivers = _detect_grey_objects(g)
    towers = _detect_towers(g, antennas)
    receivers = _detect_receivers(receivers)

    out = g.copy()
    for receiver in receivers:
        signal = _receiver_signal(receiver, towers)
        if signal["category"] == "unhit":
            continue
        for r, c in receiver["cells"]:
            out[r, c] = int(signal["best_color"])
    return out


def _try_horizontal_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    reserved: np.ndarray,
    colors: tuple[int, int],
) -> list[dict] | None:
    h, w = shape
    for _ in range(120):
        attach_left = int(rng.integers(0, 3))
        attach_right = int(rng.integers(0, 3))
        ray_left = int(rng.integers(1, h - 1))
        ray_right = int(ray_left + int(rng.choice(np.array([-1, 0, 1], dtype=int))))
        if not (0 <= ray_right < h):
            continue
        r_left = ray_left - attach_left
        r_right = ray_right - attach_right
        if not (0 <= r_left <= h - 3 and 0 <= r_right <= h - 3):
            continue

        c_left_max = max(0, w // 3)
        c_left = int(rng.integers(0, c_left_max + 1))
        c_right_min = max(3, c_left + 12)
        c_right_max = w - 3
        if c_right_min > c_right_max:
            continue
        parity_choices = [c for c in range(c_right_min, c_right_max + 1) if (c + c_left) % 2 == 1]
        if not parity_choices:
            continue
        c_right = int(rng.choice(np.array(parity_choices, dtype=int)))

        left = _make_tower(int(colors[0]), int(r_left), int(c_left), "right", int(attach_left))
        right = _make_tower(int(colors[1]), int(r_right), int(c_right), "left", int(attach_right))
        group_cells = _tower_all_cells(left) | _tower_all_cells(right)
        if not _can_place(group_cells, reserved):
            continue
        return [left, right]
    return None


def _try_vertical_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    reserved: np.ndarray,
    colors: tuple[int, int],
) -> list[dict] | None:
    h, w = shape
    for _ in range(120):
        attach_top = int(rng.integers(0, 3))
        attach_bottom = int(rng.integers(0, 3))
        ray_top = int(rng.integers(1, w - 1))
        ray_bottom = int(ray_top + int(rng.choice(np.array([-1, 0, 1], dtype=int))))
        if not (0 <= ray_bottom < w):
            continue
        c_top = ray_top - attach_top
        c_bottom = ray_bottom - attach_bottom
        if not (0 <= c_top <= w - 3 and 0 <= c_bottom <= w - 3):
            continue

        r_top_max = max(0, h // 3)
        r_top = int(rng.integers(0, r_top_max + 1))
        r_bottom_min = max(3, r_top + 12)
        r_bottom_max = h - 3
        if r_bottom_min > r_bottom_max:
            continue
        parity_choices = [r for r in range(r_bottom_min, r_bottom_max + 1) if (r + r_top) % 2 == 1]
        if not parity_choices:
            continue
        r_bottom = int(rng.choice(np.array(parity_choices, dtype=int)))

        top = _make_tower(int(colors[0]), int(r_top), int(c_top), "down", int(attach_top))
        bottom = _make_tower(int(colors[1]), int(r_bottom), int(c_bottom), "up", int(attach_bottom))
        group_cells = _tower_all_cells(top) | _tower_all_cells(bottom)
        if not _can_place(group_cells, reserved):
            continue
        return [top, bottom]
    return None


def _try_standalone_tower(
    rng: np.random.Generator,
    shape: tuple[int, int],
    reserved: np.ndarray,
    color: int,
) -> dict | None:
    h, w = shape
    directions = ["up", "down", "left", "right"]
    for _ in range(160):
        direction = directions[int(rng.integers(0, len(directions)))]
        attach = int(rng.integers(0, 3))
        if direction == "up":
            r0 = int(rng.integers(3, h - 2))
            c0 = int(rng.integers(0, w - 2))
        elif direction == "down":
            r0 = int(rng.integers(0, h - 5))
            c0 = int(rng.integers(0, w - 2))
        elif direction == "left":
            r0 = int(rng.integers(0, h - 2))
            c0 = int(rng.integers(3, w - 2))
        else:
            r0 = int(rng.integers(0, h - 2))
            c0 = int(rng.integers(0, w - 5))

        tower = _make_tower(int(color), int(r0), int(c0), direction, int(attach))
        if _can_place(_tower_all_cells(tower), reserved):
            return tower
    return None


def _receiver_candidates(
    shape: tuple[int, int],
    reserved: np.ndarray,
    towers: list[dict],
) -> dict[str, list[tuple[int, int, dict]]]:
    h, w = shape
    by_category = {category: [] for category in CATEGORY_ORDER}
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            cells = _receiver_cells(r0, c0)
            if any(bool(reserved[r, c]) for r, c in cells):
                continue
            receiver = {"r0": int(r0), "c0": int(c0), "cells": cells}
            signal = _receiver_signal(receiver, towers)
            by_category[signal["category"]].append((int(r0), int(c0), signal))
    return by_category


def _weighted_category(
    rng: np.random.Generator,
    candidates: dict[str, list[tuple[int, int, dict]]],
    counts: dict[str, int],
) -> str | None:
    available = [category for category in CATEGORY_ORDER if candidates[category]]
    if not available:
        return None
    weights = np.array([1.0 / (1 + counts[category]) for category in available], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(np.arange(len(available)), p=weights))
    return available[idx]


def _place_receivers(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    towers: list[dict],
) -> list[tuple[int, int, dict]] | None:
    initial = _receiver_candidates(grid.shape, reserved, towers)
    if not any(initial[category] for category in ("tie", "multi", "single")):
        return None

    wants: list[str] = []
    if initial["tie"] and bool(rng.integers(0, 2)):
        wants.append("tie")
    if initial["multi"] and bool(rng.integers(0, 2)):
        wants.append("multi")
    if initial["single"] and bool(rng.integers(0, 2)):
        wants.append("single")
    if initial["unhit"] and bool(rng.integers(0, 2)):
        wants.append("unhit")
    if not any(category in wants for category in ("tie", "multi", "single")):
        hit_choices = [category for category in ("tie", "multi", "single") if initial[category]]
        wants.append(hit_choices[int(rng.integers(0, len(hit_choices)))])
    wants = [category for category in CATEGORY_ORDER if category in wants]

    receiver_target = max(int(rng.integers(RECEIVER_COUNT_RANGE[0], RECEIVER_COUNT_RANGE[1] + 1)), len(wants))
    counts = {category: 0 for category in CATEGORY_ORDER}
    placed: list[tuple[int, int, dict]] = []

    for idx in range(receiver_target):
        candidates = _receiver_candidates(grid.shape, reserved, towers)
        if idx < len(wants):
            category = wants[idx]
            if not candidates[category]:
                return None
        else:
            category = _weighted_category(rng, candidates, counts)
            if category is None:
                break

        options = candidates[category]
        choice = options[int(rng.integers(0, len(options)))]
        r0, c0, signal = choice
        for r, c in _receiver_cells(r0, c0):
            grid[r, c] = GRAY
        _mark_reserved(reserved, set(_receiver_cells(r0, c0)))
        placed.append((int(r0), int(c0), signal))
        counts[category] += 1

    if not placed:
        return None
    if not any(signal["category"] != "unhit" for _r0, _c0, signal in placed):
        return None
    return placed


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    layout_choices = ("h_pair", "v_pair", "both_pairs")
    layout = layout_choices[int(rng.integers(0, len(layout_choices)))]
    extra_tower = bool(rng.integers(0, 2))

    tower_count = 4 if layout == "both_pairs" else (3 if extra_tower else 2)
    colors = rng.choice(TOWER_COLORS, size=tower_count, replace=False).astype(int).tolist()
    color_idx = 0
    towers: list[dict] = []

    if layout in ("h_pair", "both_pairs"):
        pair = _try_horizontal_pair(rng, grid.shape, reserved, (colors[color_idx], colors[color_idx + 1]))
        if pair is None:
            return None
        for tower in pair:
            _paint_tower(grid, reserved, tower)
            towers.append(tower)
        color_idx += 2

    if layout in ("v_pair", "both_pairs"):
        pair = _try_vertical_pair(rng, grid.shape, reserved, (colors[color_idx], colors[color_idx + 1]))
        if pair is None:
            return None
        for tower in pair:
            _paint_tower(grid, reserved, tower)
            towers.append(tower)
        color_idx += 2

    if color_idx < tower_count:
        tower = _try_standalone_tower(rng, grid.shape, reserved, colors[color_idx])
        if tower is None:
            return None
        _paint_tower(grid, reserved, tower)
        towers.append(tower)

    placed_receivers = _place_receivers(rng, grid, reserved, towers)
    if placed_receivers is None:
        return None

    categories = {signal["category"] for _r0, _c0, signal in placed_receivers}
    if "unhit" not in categories and bool(rng.integers(0, 2)):
        return None

    solved = main(grid)
    if solved.shape != grid.shape:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
