from __future__ import annotations

import random
from collections import Counter

import numpy as np


ARROW_TEMPLATES = {
    "up": {
        "shape": (4, 3),
        "cells": {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)},
        "tip": (0, 1),
        "delta": (-1, 0),
    },
    "down": {
        "shape": (4, 3),
        "cells": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)},
        "tip": (3, 1),
        "delta": (1, 0),
    },
    "left": {
        "shape": (3, 4),
        "cells": {(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)},
        "tip": (1, 0),
        "delta": (0, -1),
    },
    "right": {
        "shape": (3, 4),
        "cells": {(0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)},
        "tip": (1, 3),
        "delta": (0, 1),
    },
}

ALL_DIRECTIONS = tuple(ARROW_TEMPLATES)
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    target = int(color)
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != target:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == target:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coord_array = np.array(coords, dtype=int)
            components.append(
                {
                    "color": target,
                    "coords": coord_array,
                    "bbox": _bbox_from_coords(coord_array),
                    "area": int(coord_array.shape[0]),
                }
            )
    return components


def _match_arrow(comp: dict, color: int) -> dict | None:
    bbox = comp["bbox"]
    r0, c0, _r1, _c1 = bbox
    rel = {
        (int(rr - r0), int(cc - c0))
        for rr, cc in np.array(comp["coords"], dtype=int)
    }
    for direction, spec in ARROW_TEMPLATES.items():
        if (int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)) == spec["shape"] and rel == spec["cells"]:
            tip = (
                int(bbox[0] + spec["tip"][0]),
                int(bbox[1] + spec["tip"][1]),
            )
            return {
                "kind": "arrow",
                "color": int(color),
                "direction": direction,
                "tip": tip,
                "bbox": bbox,
                "coords": np.array(comp["coords"], dtype=int),
            }
    return None


def _is_solid_square(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_r = int(r1 - r0 + 1)
    side_c = int(c1 - c0 + 1)
    return side_r == side_c and int(comp["area"]) == side_r * side_c


def _parse_scene(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 8 and w >= 8
    frame = int(g[0, 0])
    assert bool(np.all(g[0, :] == frame))
    assert bool(np.all(g[h - 1, :] == frame))
    assert bool(np.all(g[:, 0] == frame))
    assert bool(np.all(g[:, w - 1] == frame))
    bg = _most_common_color(g[1:-1, 1:-1])

    arrows: list[dict] = []
    cargos: list[dict] = []
    for color in sorted(set(map(int, np.unique(g)))):
        if color in {frame, bg}:
            continue
        for comp in _find_color_components(g, color):
            arrow = _match_arrow(comp, color)
            if arrow is not None:
                arrows.append(arrow)
                continue
            assert _is_solid_square(comp), "every non-arrow object must be a solid square cargo"
            r0, c0, r1, c1 = comp["bbox"]
            cargos.append(
                {
                    "kind": "cargo",
                    "color": int(color),
                    "coords": np.array(comp["coords"], dtype=int),
                    "bbox": comp["bbox"],
                    "size": int(r1 - r0 + 1),
                }
            )

    assert arrows, "expected at least one arrow"
    assert cargos, "expected at least one cargo"
    return frame, bg, arrows, cargos


def _cargo_on_path(cargo: dict, arrow: dict) -> bool:
    r0, c0, r1, c1 = cargo["bbox"]
    tip_r, tip_c = arrow["tip"]
    direction = arrow["direction"]
    if direction == "right":
        return r0 <= tip_r <= r1 and c0 > tip_c
    if direction == "left":
        return r0 <= tip_r <= r1 and c1 < tip_c
    if direction == "down":
        return c0 <= tip_c <= c1 and r0 > tip_r
    if direction == "up":
        return c0 <= tip_c <= c1 and r1 < tip_r
    raise ValueError(f"unknown direction: {direction}")


def _slide_shift(arrow: dict, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    tip_r, tip_c = arrow["tip"]
    direction = arrow["direction"]
    if direction == "right":
        return 0, int((w - 2) - tip_c)
    if direction == "left":
        return 0, int(1 - tip_c)
    if direction == "down":
        return int((h - 2) - tip_r), 0
    if direction == "up":
        return int(1 - tip_r), 0
    raise ValueError(f"unknown direction: {direction}")


def _stack_destinations(cargos: list[dict], arrow: dict, shape: tuple[int, int]) -> list[tuple[dict, tuple[int, int]]]:
    h, w = shape
    direction = arrow["direction"]
    placements: list[tuple[dict, tuple[int, int]]] = []
    if direction == "right":
        ordered = sorted(cargos, key=lambda cargo: cargo["bbox"][1], reverse=True)
        next_col = w - 2
        for cargo in ordered:
            size = int(cargo["size"])
            r0 = int(cargo["bbox"][0])
            c0 = int(next_col - size + 1)
            placements.append((cargo, (r0, c0)))
            next_col = c0 - 1
        return placements
    if direction == "left":
        ordered = sorted(cargos, key=lambda cargo: cargo["bbox"][1])
        next_col = 1
        for cargo in ordered:
            size = int(cargo["size"])
            r0 = int(cargo["bbox"][0])
            c0 = int(next_col)
            placements.append((cargo, (r0, c0)))
            next_col = c0 + size
        return placements
    if direction == "down":
        ordered = sorted(cargos, key=lambda cargo: cargo["bbox"][0], reverse=True)
        next_row = h - 2
        for cargo in ordered:
            size = int(cargo["size"])
            r0 = int(next_row - size + 1)
            c0 = int(cargo["bbox"][1])
            placements.append((cargo, (r0, c0)))
            next_row = r0 - 1
        return placements
    if direction == "up":
        ordered = sorted(cargos, key=lambda cargo: cargo["bbox"][0])
        next_row = 1
        for cargo in ordered:
            size = int(cargo["size"])
            r0 = int(next_row)
            c0 = int(cargo["bbox"][1])
            placements.append((cargo, (r0, c0)))
            next_row = r0 + size
        return placements
    raise ValueError(f"unknown direction: {direction}")


def _apply_rule(grid: np.ndarray, bg: int, arrows: list[dict], cargos: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    hit_lists: list[list[int]] = []
    targeted_by: dict[int, int] = {}
    for arrow_index, arrow in enumerate(arrows):
        hits = [idx for idx, cargo in enumerate(cargos) if _cargo_on_path(cargo, arrow)]
        hit_lists.append(hits)
        for idx in hits:
            assert idx not in targeted_by, "a cargo may not lie on two arrow paths"
            targeted_by[idx] = arrow_index

    for cargo_index in targeted_by:
        r0, c0, r1, c1 = cargos[cargo_index]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    moved_arrows: list[tuple[dict, tuple[int, int]]] = []
    for arrow, hits in zip(arrows, hit_lists):
        if hits:
            continue
        out[arrow["coords"][:, 0], arrow["coords"][:, 1]] = int(bg)
        moved_arrows.append((arrow, _slide_shift(arrow, out.shape)))

    for arrow, hits in zip(arrows, hit_lists):
        if not hits:
            continue
        placements = _stack_destinations([cargos[idx] for idx in hits], arrow, out.shape)
        for cargo, (r0, c0) in placements:
            size = int(cargo["size"])
            out[r0 : r0 + size, c0 : c0 + size] = int(cargo["color"])

    for arrow, (dr, dc) in moved_arrows:
        shifted = np.array(arrow["coords"], dtype=int) + np.array([dr, dc], dtype=int)
        out[shifted[:, 0], shifted[:, 1]] = int(arrow["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push every cargo chain to the inner frame while empty arrows slide there themselves.

    Input structure:
    - A one-pixel rectangular frame encloses a background region.
    - Inside the frame are several six-pixel T-shaped arrows and several solid square cargos.
    - Each arrow uses a length-4 stem plus a length-3 crossbar offset toward the tip side, and arrow paths do not cross.

    Rule:
    - Read each arrow direction from which end of the stem the crossbar is biased toward.
    - If one or more cargos lie on the arrow's forward ray, the arrow stays put and every cargo on that ray is pushed in order all the way to the inner frame, packed tightly against the wall or the previous cargo.
    - If no cargo lies on that ray, erase the arrow from its source and slide the arrow itself along its direction until its tip touches the inner frame.
    - Cargos not struck by any arrow stay where they are.

    Key insight:
    - The T-shape is only a directional pusher: hit cargos move while the arrow remains, and only a missed push turns the arrow into the moving object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _frame, bg, arrows, cargos = _parse_scene(g)
    return _apply_rule(g, bg, arrows, cargos)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _sample_spaced_values(lo: int, hi: int, count: int, padding: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    for _ in range(200):
        random.shuffle(values)
        picked: list[int] = []
        for value in values:
            if all(abs(value - other) >= padding for other in picked):
                picked.append(int(value))
                if len(picked) == count:
                    return sorted(picked)
    return None


def _absolute_arrow(direction: str, tip: tuple[int, int], color: int) -> dict:
    spec = ARROW_TEMPLATES[direction]
    top = int(tip[0] - spec["tip"][0])
    left = int(tip[1] - spec["tip"][1])
    coords = np.array([(top + r, left + c) for r, c in sorted(spec["cells"])], dtype=int)
    bbox = (top, left, top + spec["shape"][0] - 1, left + spec["shape"][1] - 1)
    return {
        "kind": "arrow",
        "color": int(color),
        "direction": direction,
        "tip": (int(tip[0]), int(tip[1])),
        "bbox": bbox,
        "coords": coords,
    }


def _square_object(r0: int, c0: int, size: int, color: int) -> dict:
    coords = np.array(
        [(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)],
        dtype=int,
    )
    return {
        "kind": "cargo",
        "color": int(color),
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)),
        "size": int(size),
    }


def _choose_chain_sizes() -> list[int]:
    count = random.choice([2, 2, 3])
    while True:
        sizes = [random.choice([1, 2]) for _ in range(count)]
        if len(set(sizes)) > 1:
            return sizes


def _forward_positions_right(tip_c: int, sizes_near_to_far: list[int], width: int) -> list[int] | None:
    available = int(width - tip_c - 2)
    required = int(sum(sizes_near_to_far) + max(0, len(sizes_near_to_far) - 1))
    if required > available:
        return None
    extras = _random_composition(available - required, len(sizes_near_to_far) + 1)
    cur = int(tip_c + 1 + extras[0])
    positions: list[int] = []
    for idx, size in enumerate(sizes_near_to_far):
        positions.append(cur)
        cur += size
        if idx < len(sizes_near_to_far) - 1:
            cur += 1 + extras[idx + 1]
    return positions


def _forward_positions_left(tip_c: int, sizes_near_to_far: list[int]) -> list[int] | None:
    available = int(tip_c - 1)
    required = int(sum(sizes_near_to_far) + max(0, len(sizes_near_to_far) - 1))
    if required > available:
        return None
    far_to_near = list(reversed(sizes_near_to_far))
    extras = _random_composition(available - required, len(far_to_near) + 1)
    cur = int(1 + extras[0])
    left_to_right: list[int] = []
    for idx, size in enumerate(far_to_near):
        left_to_right.append(cur)
        cur += size
        if idx < len(far_to_near) - 1:
            cur += 1 + extras[idx + 1]
    return list(reversed(left_to_right))


def _forward_positions_down(tip_r: int, sizes_near_to_far: list[int], height: int) -> list[int] | None:
    available = int(height - tip_r - 2)
    required = int(sum(sizes_near_to_far) + max(0, len(sizes_near_to_far) - 1))
    if required > available:
        return None
    extras = _random_composition(available - required, len(sizes_near_to_far) + 1)
    cur = int(tip_r + 1 + extras[0])
    positions: list[int] = []
    for idx, size in enumerate(sizes_near_to_far):
        positions.append(cur)
        cur += size
        if idx < len(sizes_near_to_far) - 1:
            cur += 1 + extras[idx + 1]
    return positions


def _forward_positions_up(tip_r: int, sizes_near_to_far: list[int]) -> list[int] | None:
    available = int(tip_r - 1)
    required = int(sum(sizes_near_to_far) + max(0, len(sizes_near_to_far) - 1))
    if required > available:
        return None
    far_to_near = list(reversed(sizes_near_to_far))
    extras = _random_composition(available - required, len(far_to_near) + 1)
    cur = int(1 + extras[0])
    top_to_bottom: list[int] = []
    for idx, size in enumerate(far_to_near):
        top_to_bottom.append(cur)
        cur += size
        if idx < len(far_to_near) - 1:
            cur += 1 + extras[idx + 1]
    return list(reversed(top_to_bottom))


def _paint_arrow(grid: np.ndarray, occupancy: np.ndarray, arrow: dict) -> bool:
    coords = arrow["coords"]
    if bool(occupancy[coords[:, 0], coords[:, 1]].any()):
        return False
    grid[coords[:, 0], coords[:, 1]] = int(arrow["color"])
    occupancy[coords[:, 0], coords[:, 1]] = True
    return True


def _can_place_cargo(
    occupancy: np.ndarray,
    cargo_occupancy: np.ndarray,
    r0: int,
    c0: int,
    size: int,
) -> bool:
    h, w = occupancy.shape
    if r0 < 1 or c0 < 1 or r0 + size > h - 1 or c0 + size > w - 1:
        return False
    for rr in range(r0, r0 + size):
        for cc in range(c0, c0 + size):
            if occupancy[rr, cc]:
                return False
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and cargo_occupancy[nr, nc]:
                    if not (r0 <= nr < r0 + size and c0 <= nc < c0 + size):
                        return False
    return True


def _paint_cargo(
    grid: np.ndarray,
    occupancy: np.ndarray,
    cargo_occupancy: np.ndarray,
    cargo: dict,
) -> bool:
    r0, c0, r1, c1 = cargo["bbox"]
    size = int(cargo["size"])
    if not _can_place_cargo(occupancy, cargo_occupancy, r0, c0, size):
        return False
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(cargo["color"])
    occupancy[r0 : r1 + 1, c0 : c1 + 1] = True
    cargo_occupancy[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _choose_tip_horizontal(width: int, direction: str, forward_sizes: list[int], back_size: int | None) -> int | None:
    forward_need = int(sum(forward_sizes) + max(0, len(forward_sizes) - 1))
    back_need = 0 if back_size is None else int(back_size)
    if direction == "right":
        lo = max(4, 4 + back_need)
        hi = width - 2 - forward_need
        if forward_need == 0:
            hi -= 1
    else:
        lo = 1 + forward_need
        if forward_need == 0:
            lo += 1
        hi = min(width - 5, width - 5 - back_need)
    if lo > hi:
        return None
    return random.randint(int(lo), int(hi))


def _choose_tip_vertical(height: int, direction: str, forward_sizes: list[int], back_size: int | None) -> int | None:
    forward_need = int(sum(forward_sizes) + max(0, len(forward_sizes) - 1))
    back_need = 0 if back_size is None else int(back_size)
    if direction == "down":
        lo = max(4, 4 + back_need)
        hi = height - 2 - forward_need
        if forward_need == 0:
            hi -= 1
    else:
        lo = 1 + forward_need
        if forward_need == 0:
            lo += 1
        hi = min(height - 5, height - 5 - back_need)
    if lo > hi:
        return None
    return random.randint(int(lo), int(hi))


def _random_same_lane_spectator(
    family: str,
    direction: str,
    lane: int,
    tip: int,
    size: int,
    color: int,
    shape: tuple[int, int],
) -> dict | None:
    h, w = shape
    if family == "horizontal":
        if direction == "right":
            hi = tip - size - 3
            if hi < 1:
                return None
            c0 = random.randint(1, hi)
        else:
            lo = tip + 4
            hi = w - 1 - size
            if lo > hi:
                return None
            c0 = random.randint(lo, hi)
        return _square_object(lane, c0, size, color)
    if direction == "down":
        hi = tip - size - 3
        if hi < 1:
            return None
        r0 = random.randint(1, hi)
    else:
        lo = tip + 4
        hi = h - 1 - size
        if lo > hi:
            return None
        r0 = random.randint(lo, hi)
    return _square_object(r0, lane, size, color)


def _place_random_offlane_spectator(
    grid: np.ndarray,
    occupancy: np.ndarray,
    cargo_occupancy: np.ndarray,
    family: str,
    lanes: set[int],
    cargo_color: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(200):
        size = random.choice([1, 2])
        if family == "horizontal":
            r0 = random.randint(1, h - 1 - size)
            if any((r0 + delta) in lanes for delta in range(size)):
                continue
            c0 = random.randint(1, w - 1 - size)
        else:
            c0 = random.randint(1, w - 1 - size)
            if any((c0 + delta) in lanes for delta in range(size)):
                continue
            r0 = random.randint(1, h - 1 - size)
        cargo = _square_object(r0, c0, size, cargo_color)
        if _paint_cargo(grid, occupancy, cargo_occupancy, cargo):
            return cargo
    return None


def _generate_scene() -> tuple[np.ndarray, list[dict], list[dict], int]:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    bg, frame_color, arrow_color, cargo_color = random.sample(PALETTE, 4)
    family = random.choice(["horizontal", "vertical"])
    arrow_count = random.choice([2, 3, 4])
    grid = np.full((height, width), int(bg), dtype=int)
    grid[0, :] = int(frame_color)
    grid[height - 1, :] = int(frame_color)
    grid[:, 0] = int(frame_color)
    grid[:, width - 1] = int(frame_color)
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[0, :] = True
    occupancy[height - 1, :] = True
    occupancy[:, 0] = True
    occupancy[:, width - 1] = True
    cargo_occupancy = np.zeros((height, width), dtype=bool)

    roles = ["empty", "chain"] + [random.choice(["empty", "chain"]) for _ in range(arrow_count - 2)]
    if family == "horizontal":
        lanes = _sample_spaced_values(2, height - 3, arrow_count, padding=4)
        if lanes is None:
            raise ValueError("failed to sample horizontal lanes")
        directions = [random.choice(["left", "right"]) for _ in lanes]
        arrows: list[dict] = []
        cargos: list[dict] = []
        forced_same_lane = False
        for idx, (lane, role, direction) in enumerate(zip(lanes, roles, directions)):
            path_sizes = [] if role == "empty" else _choose_chain_sizes()
            back_size = random.choice([1, 2]) if not forced_same_lane else None
            if role == "empty" and not forced_same_lane:
                back_size = random.choice([1, 2])
            tip_c = _choose_tip_horizontal(width, direction, path_sizes, back_size)
            if tip_c is None:
                raise ValueError("no horizontal tip fits")
            arrow = _absolute_arrow(direction, (lane, tip_c), arrow_color)
            if not _paint_arrow(grid, occupancy, arrow):
                raise ValueError("horizontal arrow placement failed")
            arrows.append(arrow)

            if direction == "right":
                positions = _forward_positions_right(tip_c, path_sizes, width)
            else:
                positions = _forward_positions_left(tip_c, path_sizes)
            if positions is None:
                raise ValueError("horizontal path placement failed")

            for size, c0 in zip(path_sizes, positions):
                cargo = _square_object(lane, c0, size, cargo_color)
                if not _paint_cargo(grid, occupancy, cargo_occupancy, cargo):
                    raise ValueError("horizontal cargo placement failed")
                cargos.append(cargo)

            if back_size is not None:
                spectator = _random_same_lane_spectator(
                    family="horizontal",
                    direction=direction,
                    lane=lane,
                    tip=tip_c,
                    size=back_size,
                    color=cargo_color,
                    shape=grid.shape,
                )
                if spectator is not None and _paint_cargo(grid, occupancy, cargo_occupancy, spectator):
                    cargos.append(spectator)
                    forced_same_lane = True

        extra_count = random.choice([0, 1, 2])
        for _ in range(extra_count):
            cargo = _place_random_offlane_spectator(
                grid,
                occupancy,
                cargo_occupancy,
                family="horizontal",
                lanes=set(lanes),
                cargo_color=cargo_color,
            )
            if cargo is not None:
                cargos.append(cargo)
        return grid, arrows, cargos, int(bg)

    lanes = _sample_spaced_values(2, width - 3, arrow_count, padding=4)
    if lanes is None:
        raise ValueError("failed to sample vertical lanes")
    directions = [random.choice(["up", "down"]) for _ in lanes]
    arrows = []
    cargos = []
    forced_same_lane = False
    for lane, role, direction in zip(lanes, roles, directions):
        path_sizes = [] if role == "empty" else _choose_chain_sizes()
        back_size = random.choice([1, 2]) if not forced_same_lane else None
        if role == "empty" and not forced_same_lane:
            back_size = random.choice([1, 2])
        tip_r = _choose_tip_vertical(height, direction, path_sizes, back_size)
        if tip_r is None:
            raise ValueError("no vertical tip fits")
        arrow = _absolute_arrow(direction, (tip_r, lane), arrow_color)
        if not _paint_arrow(grid, occupancy, arrow):
            raise ValueError("vertical arrow placement failed")
        arrows.append(arrow)

        if direction == "down":
            positions = _forward_positions_down(tip_r, path_sizes, height)
        else:
            positions = _forward_positions_up(tip_r, path_sizes)
        if positions is None:
            raise ValueError("vertical path placement failed")

        for size, r0 in zip(path_sizes, positions):
            cargo = _square_object(r0, lane, size, cargo_color)
            if not _paint_cargo(grid, occupancy, cargo_occupancy, cargo):
                raise ValueError("vertical cargo placement failed")
            cargos.append(cargo)

        if back_size is not None:
            spectator = _random_same_lane_spectator(
                family="vertical",
                direction=direction,
                lane=lane,
                tip=tip_r,
                size=back_size,
                color=cargo_color,
                shape=grid.shape,
            )
            if spectator is not None and _paint_cargo(grid, occupancy, cargo_occupancy, spectator):
                cargos.append(spectator)
                forced_same_lane = True

    extra_count = random.choice([0, 1, 2])
    for _ in range(extra_count):
        cargo = _place_random_offlane_spectator(
            grid,
            occupancy,
            cargo_occupancy,
            family="vertical",
            lanes=set(lanes),
            cargo_color=cargo_color,
        )
        if cargo is not None:
            cargos.append(cargo)
    return grid, arrows, cargos, int(bg)


def _validate_scene(grid: np.ndarray, arrows: list[dict], cargos: list[dict], bg: int) -> np.ndarray | None:
    assert len(arrows) >= 2
    path_counts = []
    has_same_line_spectator = False
    for arrow in arrows:
        hit_indices = {idx for idx, cargo in enumerate(cargos) if _cargo_on_path(cargo, arrow)}
        path_counts.append(len(hit_indices))
        if arrow["direction"] in {"left", "right"}:
            same_line = [
                idx
                for idx, cargo in enumerate(cargos)
                if cargo["bbox"][0] <= arrow["tip"][0] <= cargo["bbox"][2]
            ]
        else:
            same_line = [
                idx
                for idx, cargo in enumerate(cargos)
                if cargo["bbox"][1] <= arrow["tip"][1] <= cargo["bbox"][3]
            ]
        behind = [idx for idx in same_line if idx not in hit_indices]
        if behind:
            has_same_line_spectator = True

    if not any(count == 0 for count in path_counts):
        return None
    if not any(count >= 2 for count in path_counts):
        return None
    if not has_same_line_spectator:
        return None

    parsed_frame, parsed_bg, parsed_arrows, parsed_cargos = _parse_scene(grid)
    if parsed_frame == parsed_bg:
        return None
    expected = _apply_rule(grid, bg, arrows, cargos)
    actual = _apply_rule(grid, parsed_bg, parsed_arrows, parsed_cargos)
    if not np.array_equal(expected, actual):
        return None
    if np.array_equal(expected, grid):
        return None
    return expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, arrows, cargos, bg = _generate_scene()
        except ValueError:
            continue
        expected = _validate_scene(grid, arrows, cargos, bg)
        if expected is None:
            continue
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid arrow-pusher input")
