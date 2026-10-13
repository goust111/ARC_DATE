from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
WALL_COLOR = 1
PALETTE_POOL = tuple(range(2, 10))

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
ROOM_COUNT_OPTIONS = (4, 5, 6)
ROOM_H_RANGE = (5, 9)
ROOM_W_RANGE = (5, 10)
ROOM_MARGIN = 1
PALETTE_KEEP_OUT = 5
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _find_palette_and_wall(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    palette_entries: list[tuple[int, int, int]] = []
    wall_colors: list[int] = []

    for color, count in zip(colors, counts):
        color = int(color)
        count = int(count)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if count == 1:
            r, c = map(int, coords[0])
            palette_entries.append((r, c, color))
        else:
            wall_colors.append(color)

    assert len(palette_entries) == 3, "expected a 3-pixel palette strip"
    assert len(wall_colors) == 1, "expected one wall color"

    palette_entries.sort()
    rows = [r for r, _, _ in palette_entries]
    cols = [c for _, c, _ in palette_entries]
    assert len(set(cols)) == 1, "palette must be vertical"
    assert rows == list(range(rows[0], rows[0] + 3)), "palette pixels must be consecutive"
    assert cols[0] in (0, w - 1), "palette must sit on a corner column"
    assert rows[0] in (0, h - 3), "palette must sit on a corner row span"

    return [color for _, _, color in palette_entries], wall_colors[0]


def _room_from_component(
    grid: np.ndarray,
    component: list[tuple[int, int]],
    bg: int,
    wall_color: int,
) -> dict | None:
    r0, c0, r1, c1 = _bbox(component)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    comp_set = set(component)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(comp_set):
        return None
    if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in comp_set):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.all(interior == bg):
        return None

    doors_by_side = {"top": [], "bottom": [], "left": [], "right": []}

    for c in range(c0 + 1, c1):
        top_val = int(grid[r0, c])
        bottom_val = int(grid[r1, c])
        if top_val == bg:
            doors_by_side["top"].append((r0, c))
        elif top_val != wall_color:
            return None
        if bottom_val == bg:
            doors_by_side["bottom"].append((r1, c))
        elif bottom_val != wall_color:
            return None

    for r in range(r0 + 1, r1):
        left_val = int(grid[r, c0])
        right_val = int(grid[r, c1])
        if left_val == bg:
            doors_by_side["left"].append((r, c0))
        elif left_val != wall_color:
            return None
        if right_val == bg:
            doors_by_side["right"].append((r, c1))
        elif right_val != wall_color:
            return None

    door_count = sum(len(v) for v in doors_by_side.values())
    if not 1 <= door_count <= 3:
        return None

    perimeter = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(comp_set) != perimeter - door_count:
        return None

    stripe = "horizontal" if (doors_by_side["left"] or doors_by_side["right"]) else "vertical"
    return {
        "bbox": (r0, c0, r1, c1),
        "door_count": door_count,
        "stripe": stripe,
    }


def _parse_rooms(grid: np.ndarray, bg: int, wall_color: int) -> list[dict]:
    h, w = grid.shape
    wall = grid == wall_color
    bg_mask = grid == bg
    other = ~(wall | bg_mask)
    bg_ps = _prefix_sum(bg_mask)
    other_ps = _prefix_sum(other)

    rooms: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            inner_h = r1 - r0 - 1
            for c0 in range(w - 2):
                if not wall[r0, c0]:
                    continue
                for c1 in range(c0 + 2, w):
                    bbox = (r0, c0, r1, c1)
                    if bbox in seen:
                        continue
                    if not (wall[r0, c1] and wall[r1, c0] and wall[r1, c1]):
                        continue
                    if _rect_sum(other_ps, r0, c0, r1, c1) != 0:
                        continue

                    inner_w = c1 - c0 - 1
                    interior_area = inner_h * inner_w
                    if _rect_sum(bg_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != interior_area:
                        continue

                    perimeter_bg = _rect_sum(bg_ps, r0, c0, r1, c1) - _rect_sum(
                        bg_ps,
                        r0 + 1,
                        c0 + 1,
                        r1 - 1,
                        c1 - 1,
                    )
                    if not 1 <= perimeter_bg <= 3:
                        continue

                    left_bg = _rect_sum(bg_ps, r0 + 1, c0, r1 - 1, c0)
                    right_bg = _rect_sum(bg_ps, r0 + 1, c1, r1 - 1, c1)
                    rooms.append(
                        {
                            "bbox": bbox,
                            "door_count": perimeter_bg,
                            "stripe": "horizontal" if (left_bg + right_bg) else "vertical",
                        }
                    )
                    seen.add(bbox)

    rooms.sort(key=lambda room: room["bbox"])
    assert rooms, "expected at least one room"
    return rooms


def _paint_room(out: np.ndarray, room: dict, fill_color: int, bg: int) -> None:
    r0, c0, r1, c1 = room["bbox"]
    if room["stripe"] == "horizontal":
        for r in range(r0 + 1, r1):
            if (r - (r0 + 1)) % 2 == 0:
                out[r, c0 + 1 : c1] = fill_color
            else:
                out[r, c0 + 1 : c1] = bg
    else:
        for c in range(c0 + 1, c1):
            if (c - (c0 + 1)) % 2 == 0:
                out[r0 + 1 : r1, c] = fill_color
            else:
                out[r0 + 1 : r1, c] = bg


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    palette, wall_color = _find_palette_and_wall(g, bg)
    rooms = _parse_rooms(g, bg, wall_color)

    out = g.copy()
    for room in rooms:
        _paint_room(out, room, palette[room["door_count"] - 1], bg)
    return out


def _palette_layout(h: int, w: int, corner: str) -> tuple[list[tuple[int, int]], tuple[slice, slice]]:
    if corner == "tl":
        return [(0, 0), (1, 0), (2, 0)], (slice(0, PALETTE_KEEP_OUT), slice(0, PALETTE_KEEP_OUT))
    if corner == "tr":
        return [(0, w - 1), (1, w - 1), (2, w - 1)], (
            slice(0, PALETTE_KEEP_OUT),
            slice(w - PALETTE_KEEP_OUT, w),
        )
    if corner == "bl":
        return [(h - 3, 0), (h - 2, 0), (h - 1, 0)], (
            slice(h - PALETTE_KEEP_OUT, h),
            slice(0, PALETTE_KEEP_OUT),
        )
    if corner == "br":
        return [(h - 3, w - 1), (h - 2, w - 1), (h - 1, w - 1)], (
            slice(h - PALETTE_KEEP_OUT, h),
            slice(w - PALETTE_KEEP_OUT, w),
        )
    raise AssertionError("unknown palette corner")


def _orientation_list(room_count: int, rng: random.Random) -> list[str]:
    orientations = ["horizontal"] * (room_count // 2) + ["vertical"] * (room_count // 2)
    if room_count % 2:
        orientations.append(rng.choice(["horizontal", "vertical"]))
    rng.shuffle(orientations)
    return orientations


def _door_counts(room_count: int, rng: random.Random) -> list[int]:
    counts = [1, 2, 3] + [rng.choice((1, 2, 3)) for _ in range(room_count - 3)]
    rng.shuffle(counts)
    return counts


def _sample_room_specs(room_count: int, rng: random.Random) -> list[dict]:
    counts = _door_counts(room_count, rng)
    orientations = _orientation_list(room_count, rng)
    specs = []
    for i in range(room_count):
        specs.append(
            {
                "door_count": counts[i],
                "stripe": orientations[i],
                "height": rng.randint(*ROOM_H_RANGE),
                "width": rng.randint(*ROOM_W_RANGE),
            }
        )
    specs.sort(key=lambda spec: spec["height"] * spec["width"], reverse=True)
    return specs


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rs = max(0, r0 - ROOM_MARGIN)
    re = min(occupied.shape[0], r0 + h + ROOM_MARGIN)
    cs = max(0, c0 - ROOM_MARGIN)
    ce = min(occupied.shape[1], c0 + w + ROOM_MARGIN)
    return not bool(occupied[rs:re, cs:ce].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rs = max(0, r0 - ROOM_MARGIN)
    re = min(occupied.shape[0], r0 + h + ROOM_MARGIN)
    cs = max(0, c0 - ROOM_MARGIN)
    ce = min(occupied.shape[1], c0 + w + ROOM_MARGIN)
    occupied[rs:re, cs:ce] = True


def _place_rooms(h: int, w: int, specs: list[dict], blocked: np.ndarray, rng: random.Random) -> list[dict] | None:
    occupied = blocked.copy()
    placed: list[dict] = []
    for spec in specs:
        room_h = spec["height"]
        room_w = spec["width"]
        if h - room_h - 2 < 1 or w - room_w - 2 < 1:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(1, h - room_h - 1)
            c0 = rng.randint(1, w - room_w - 1)
            if not _can_place(occupied, r0, c0, room_h, room_w):
                continue
            placed_spec = dict(spec)
            placed_spec["top"] = r0
            placed_spec["left"] = c0
            placed.append(placed_spec)
            _mark_occupied(occupied, r0, c0, room_h, room_w)
            break
        else:
            return None
    return placed


def _side_positions(room_h: int, room_w: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(0, c) for c in range(1, room_w - 1)],
        "bottom": [(room_h - 1, c) for c in range(1, room_w - 1)],
        "left": [(r, 0) for r in range(1, room_h - 1)],
        "right": [(r, room_w - 1) for r in range(1, room_h - 1)],
    }


def _sample_doors(room_h: int, room_w: int, door_count: int, stripe: str, rng: random.Random) -> list[tuple[int, int]]:
    sides = _side_positions(room_h, room_w)
    top_bottom = sides["top"] + sides["bottom"]
    left_right = sides["left"] + sides["right"]
    all_positions = top_bottom + left_right

    if stripe == "vertical":
        return rng.sample(top_bottom, door_count)

    if door_count > 1 and len(left_right) >= door_count and rng.random() < 0.5:
        return rng.sample(left_right, door_count)

    chosen = [rng.choice(left_right)]
    remaining = [pos for pos in all_positions if pos not in chosen]
    chosen.extend(rng.sample(remaining, door_count - 1))
    return chosen


def _draw_room(grid: np.ndarray, spec: dict) -> None:
    r0 = spec["top"]
    c0 = spec["left"]
    room_h = spec["height"]
    room_w = spec["width"]
    r1 = r0 + room_h - 1
    c1 = c0 + room_w - 1

    grid[r0, c0 : c1 + 1] = WALL_COLOR
    grid[r1, c0 : c1 + 1] = WALL_COLOR
    grid[r0 : r1 + 1, c0] = WALL_COLOR
    grid[r0 : r1 + 1, c1] = WALL_COLOR

    doors = _sample_doors(room_h, room_w, spec["door_count"], spec["stripe"], spec["rng"])
    spec["doors"] = [(r0 + dr, c0 + dc) for dr, dc in doors]
    for rr, cc in spec["doors"]:
        grid[rr, cc] = BG_COLOR


def _expected_output(grid: np.ndarray, palette: list[int], specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        room = {
            "bbox": (
                spec["top"],
                spec["left"],
                spec["top"] + spec["height"] - 1,
                spec["left"] + spec["width"] - 1,
            ),
            "door_count": spec["door_count"],
            "stripe": spec["stripe"],
        }
        _paint_room(out, room, palette[spec["door_count"] - 1], BG_COLOR)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    room_count = rng.choice(ROOM_COUNT_OPTIONS)
    palette = rng.sample(PALETTE_POOL, 3)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    palette_cells, blocked_region = _palette_layout(h, w, corner)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[blocked_region] = True

    specs = _sample_room_specs(room_count, rng)
    placed = _place_rooms(h, w, specs, blocked, rng)
    if placed is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for (r, c), color in zip(palette_cells, palette):
        grid[r, c] = color

    for spec in placed:
        spec["rng"] = rng
        _draw_room(grid, spec)
        spec.pop("rng")

    colors, counts = np.unique(grid, return_counts=True)
    if int(colors[int(np.argmax(counts))]) != BG_COLOR:
        return None

    expected = _expected_output(grid, palette, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
