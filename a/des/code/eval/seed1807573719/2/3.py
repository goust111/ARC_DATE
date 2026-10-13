from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
WALL_COLOR = 1
PALETTE = tuple(range(10))

# --------------- solver helpers ---------------


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
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


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _side_coords(r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }


def _infer_wall_color(g: np.ndarray, bg: int, boxes: list[tuple[int, int, int, int]]) -> int:
    color_sets: list[set[int]] = []
    for r0, c0, r1, c1 in boxes:
        colors = {int(g[r, c]) for r, c in _perimeter_coords(r0, c0, r1, c1) if int(g[r, c]) != bg}
        assert colors, "room perimeter must contain non-background cells"
        color_sets.append(colors)

    shared = set.intersection(*color_sets)
    shared.discard(bg)
    if len(shared) == 1:
        return int(next(iter(shared)))

    perimeter_colors = [
        int(g[r, c])
        for r0, c0, r1, c1 in boxes
        for r, c in _perimeter_coords(r0, c0, r1, c1)
        if int(g[r, c]) != bg
    ]
    assert perimeter_colors, "expected non-background perimeter colors"
    return _most_common_color(np.array(perimeter_colors, dtype=int))


def _parse_room(
    g: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    bg: int,
    wall_color: int,
) -> dict | None:
    r0, c0, r1, c1 = room_bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if any(int(g[r, c]) != wall_color for r, c in corners):
        return None

    window_color: int | None = None
    side_windows: dict[str, list[tuple[int, int]]] = {}
    total_windows = 0
    for side, cells in _side_coords(r0, c0, r1, c1).items():
        windows: list[tuple[int, int]] = []
        for r, c in cells:
            value = int(g[r, c])
            if value == wall_color:
                continue
            if value == bg:
                return None
            if window_color is None:
                window_color = value
            elif value != window_color:
                return None
            windows.append((r, c))
        side_windows[side] = windows
        total_windows += len(windows)

    if window_color is None or total_windows == 0:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "window_color": int(window_color),
        "side_windows": side_windows,
        "total_windows": int(total_windows),
    }


def _extract_rooms(g: np.ndarray) -> tuple[int, int, list[dict]]:
    h, w = g.shape
    bg = _most_common_color(g)

    candidate_boxes: list[tuple[int, int, int, int]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for comp in _components(g == bg):
        if _touches_edge(comp, h, w):
            continue
        ir0, ic0, ir1, ic1 = _bbox(comp)
        if len(comp) != (ir1 - ir0 + 1) * (ic1 - ic0 + 1):
            continue
        if ir0 == 0 or ic0 == 0 or ir1 == h - 1 or ic1 == w - 1:
            continue
        box = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
        if box not in seen_boxes:
            seen_boxes.add(box)
            candidate_boxes.append(box)

    assert candidate_boxes, "expected enclosed rectangular room interiors"
    wall_color = _infer_wall_color(g, bg, candidate_boxes)

    rooms = []
    for box in candidate_boxes:
        room = _parse_room(g, box, bg, wall_color)
        if room is not None:
            rooms.append(room)

    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    assert rooms, "expected valid rooms"
    return bg, wall_color, rooms


def _cast_ray(
    out: np.ndarray,
    source: np.ndarray,
    *,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    length: int,
    color: int,
    bg: int,
) -> None:
    h, w = source.shape
    rr, cc = start_r, start_c
    for _ in range(length):
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if int(source[rr, cc]) != bg:
            break
        out[rr, cc] = color
        rr += dr
        cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify rectangular rooms, count colored windows on each wall, and emit
    outward rays from the wall(s) with the most windows. Ray length equals the
    total number of windows of that room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, _wall_color, rooms = _extract_rooms(g)
    out = g.copy()

    for room in rooms:
        total = int(room["total_windows"])
        side_windows = room["side_windows"]
        max_count = max(len(side_windows[side]) for side in SIDES)
        color = int(room["window_color"])
        r0, c0, r1, c1 = room["bbox"]

        for side in SIDES:
            windows = side_windows[side]
            if len(windows) != max_count:
                continue
            for r, c in windows:
                if side == "top":
                    _cast_ray(out, g, start_r=r0 - 1, start_c=c, dr=-1, dc=0, length=total, color=color, bg=bg)
                elif side == "bottom":
                    _cast_ray(out, g, start_r=r1 + 1, start_c=c, dr=1, dc=0, length=total, color=color, bg=bg)
                elif side == "left":
                    _cast_ray(out, g, start_r=r, start_c=c0 - 1, dr=0, dc=-1, length=total, color=color, bg=bg)
                else:
                    _cast_ray(out, g, start_r=r, start_c=c1 + 1, dr=0, dc=1, length=total, color=color, bg=bg)

    return out


# --------------- generator helpers ---------------

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
ROOM_COUNT_RANGE = (2, 4)
ROOM_H_RANGE = (5, 8)
ROOM_W_RANGE = (5, 8)
MAX_TOTAL_WINDOWS = 8
MAX_SELECTED_WINDOWS = 3
GEN_RETRY_LIMIT = 400
PLACE_RETRIES = 250


def _sample_branches(n_rooms: int) -> list[str]:
    n_tie = n_rooms // 2
    if n_rooms % 2 == 1 and random.random() < 0.5:
        n_tie += 1
    branches = ["tie"] * n_tie + ["unique"] * (n_rooms - n_tie)
    random.shuffle(branches)
    return branches


def _non_adjacent_capacity(length: int) -> int:
    return (length + 1) // 2


def _sample_spaced_positions(length: int, count: int) -> list[int]:
    if count == 0:
        return []
    base = sorted(random.sample(range(1, length - count + 2), count))
    return [int(base[i] + i) for i in range(count)]


def _sample_room_spec(window_color: int, branch: str) -> dict:
    for _ in range(200):
        room_h = random.randint(*ROOM_H_RANGE)
        room_w = random.randint(*ROOM_W_RANGE)
        side_lengths = {
            "top": room_w - 2,
            "bottom": room_w - 2,
            "left": room_h - 2,
            "right": room_h - 2,
        }
        capacities = {side: _non_adjacent_capacity(length) for side, length in side_lengths.items()}

        if branch == "unique":
            chosen_sides = random.sample(SIDES, 1)
        else:
            chosen_sides = random.sample(SIDES, random.randint(2, 4))

        max_hi = min(
            MAX_SELECTED_WINDOWS,
            MAX_TOTAL_WINDOWS // len(chosen_sides),
            *(capacities[side] for side in chosen_sides),
        )
        if max_hi < 1:
            continue
        max_count = random.randint(1, max_hi)

        counts: dict[str, int] = {}
        for side in SIDES:
            cap = capacities[side]
            if side in chosen_sides:
                counts[side] = max_count
            else:
                upper = min(cap, max_count - 1)
                counts[side] = random.randint(0, upper) if upper > 0 else 0

        total = sum(counts.values())
        if not (1 <= total <= MAX_TOTAL_WINDOWS):
            continue

        windows = {
            "top": _sample_spaced_positions(side_lengths["top"], counts["top"]),
            "bottom": _sample_spaced_positions(side_lengths["bottom"], counts["bottom"]),
            "left": _sample_spaced_positions(side_lengths["left"], counts["left"]),
            "right": _sample_spaced_positions(side_lengths["right"], counts["right"]),
        }
        return {
            "h": room_h,
            "w": room_w,
            "window_color": int(window_color),
            "counts": counts,
            "windows": windows,
            "total_windows": int(total),
        }

    raise AssertionError("failed to sample a valid room spec")


def _place_rooms(specs: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    occupied = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(specs)), key=lambda i: specs[i]["h"] * specs[i]["w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        room_h = int(spec["h"])
        room_w = int(spec["w"])
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, h - room_h)
            c0 = random.randint(0, w - room_w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + room_h + 1)
            cc1 = min(w, c0 + room_w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r0), int(c0)) for r0, c0 in positions if r0 is not None and c0 is not None]


def _render_input(h: int, w: int, bg: int, specs: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec, (r0, c0) in zip(specs, positions, strict=True):
        room_h = int(spec["h"])
        room_w = int(spec["w"])
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1
        color = int(spec["window_color"])

        grid[r0, c0 : c1 + 1] = WALL_COLOR
        grid[r1, c0 : c1 + 1] = WALL_COLOR
        grid[r0 : r1 + 1, c0] = WALL_COLOR
        grid[r0 : r1 + 1, c1] = WALL_COLOR

        for rel_c in spec["windows"]["top"]:
            grid[r0, c0 + rel_c] = color
        for rel_c in spec["windows"]["bottom"]:
            grid[r1, c0 + rel_c] = color
        for rel_r in spec["windows"]["left"]:
            grid[r0 + rel_r, c0] = color
        for rel_r in spec["windows"]["right"]:
            grid[r0 + rel_r, c1] = color

    return grid


def _expected_from_specs(
    grid: np.ndarray,
    bg: int,
    specs: list[dict],
    positions: list[tuple[int, int]],
) -> tuple[np.ndarray, bool, bool, bool]:
    out = grid.copy()
    h, w = grid.shape
    has_room_block = False
    has_edge_block = False
    any_painted = False

    for spec, (r0, c0) in zip(specs, positions, strict=True):
        room_h = int(spec["h"])
        room_w = int(spec["w"])
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1
        total = int(spec["total_windows"])
        max_count = max(int(spec["counts"][side]) for side in SIDES)
        color = int(spec["window_color"])

        for side in SIDES:
            if int(spec["counts"][side]) != max_count:
                continue

            if side == "top":
                starts = [(r0 - 1, c0 + rel_c, -1, 0) for rel_c in spec["windows"]["top"]]
            elif side == "bottom":
                starts = [(r1 + 1, c0 + rel_c, 1, 0) for rel_c in spec["windows"]["bottom"]]
            elif side == "left":
                starts = [(r0 + rel_r, c0 - 1, 0, -1) for rel_r in spec["windows"]["left"]]
            else:
                starts = [(r0 + rel_r, c1 + 1, 0, 1) for rel_r in spec["windows"]["right"]]

            for start_r, start_c, dr, dc in starts:
                rr, cc = start_r, start_c
                for _ in range(total):
                    if not (0 <= rr < h and 0 <= cc < w):
                        has_edge_block = True
                        break
                    if int(grid[rr, cc]) != bg:
                        has_room_block = True
                        break
                    out[rr, cc] = color
                    any_painted = True
                    rr += dr
                    cc += dc

    return out, has_room_block, has_edge_block, any_painted


def _branch_kinds(specs: list[dict]) -> set[str]:
    kinds: set[str] = set()
    for spec in specs:
        counts = [int(spec["counts"][side]) for side in SIDES]
        max_count = max(counts)
        if counts.count(max_count) == 1:
            kinds.add("unique")
        else:
            kinds.add("tie")
    return kinds


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice([color for color in PALETTE if color != WALL_COLOR])
    n_rooms = random.randint(*ROOM_COUNT_RANGE)
    branches = _sample_branches(n_rooms)
    window_palette = [color for color in PALETTE if color not in (bg, WALL_COLOR)]
    window_colors = random.sample(window_palette, n_rooms)
    specs = [_sample_room_spec(window_colors[i], branches[i]) for i in range(n_rooms)]
    positions = _place_rooms(specs, h, w)
    if positions is None or len(positions) != len(specs):
        return None

    grid = _render_input(h, w, bg, specs, positions)
    if _most_common_color(grid) != bg:
        return None

    non_bg = grid[grid != bg]
    if non_bg.size == 0 or _most_common_color(non_bg) != WALL_COLOR:
        return None

    kinds = _branch_kinds(specs)
    if kinds != {"unique", "tie"}:
        return None

    expected, has_room_block, _has_edge_block, any_painted = _expected_from_specs(grid, bg, specs, positions)
    if not any_painted:
        return None

    need_room_block = bool(random.getrandbits(1))
    if need_room_block and not has_room_block:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
