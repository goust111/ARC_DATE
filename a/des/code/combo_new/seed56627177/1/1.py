from __future__ import annotations

import numpy as np


_BG = 0
_COLORS = tuple(range(1, 10))
_DIRECTIONS = ("up", "down", "left", "right")
_GEN_ATTEMPTS = 400
_ROOM_ATTEMPTS = 160
_PATTERN_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_direction(gap: tuple[int, int], bbox_: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox_
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "up"
    if gr == r1 and c0 < gc < c1:
        return "down"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    return None


def _frontier_aligned(pattern: list[tuple[int, int]], gap: tuple[int, int], direction: str) -> bool:
    gr, gc = gap
    if direction == "up":
        frontier = min(r for r, _ in pattern)
        return any(r == frontier and c == gc for r, c in pattern)
    if direction == "down":
        frontier = max(r for r, _ in pattern)
        return any(r == frontier and c == gc for r, c in pattern)
    if direction == "left":
        frontier = min(c for _, c in pattern)
        return any(c == frontier and r == gr for r, c in pattern)
    frontier = max(c for _, c in pattern)
    return any(c == frontier and r == gr for r, c in pattern)


def _project_coords(pattern: list[tuple[int, int]], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if direction == "up":
        shift = -min(r for r, _ in pattern)
        return [(r + shift, c) for r, c in pattern]
    if direction == "down":
        shift = (h - 1) - max(r for r, _ in pattern)
        return [(r + shift, c) for r, c in pattern]
    if direction == "left":
        shift = -min(c for _, c in pattern)
        return [(r, c + shift) for r, c in pattern]
    shift = (w - 1) - max(c for _, c in pattern)
    return [(r, c + shift) for r, c in pattern]


def _marker_segment(gap: tuple[int, int], direction: str, projected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    gr, gc = gap
    if direction == "up":
        hits = [r for r, c in projected if c == gc and r < gr]
        if not hits:
            return []
        nearest = max(hits)
        return [(r, gc) for r in range(gr - 1, nearest, -1)]
    if direction == "down":
        hits = [r for r, c in projected if c == gc and r > gr]
        if not hits:
            return []
        nearest = min(hits)
        return [(r, gc) for r in range(gr + 1, nearest)]
    if direction == "left":
        hits = [c for r, c in projected if r == gr and c < gc]
        if not hits:
            return []
        nearest = max(hits)
        return [(gr, c) for c in range(gc - 1, nearest, -1)]
    hits = [c for r, c in projected if r == gr and c > gc]
    if not hits:
        return []
    nearest = min(hits)
    return [(gr, c) for c in range(gc + 1, nearest)]


def _first_marker_color(grid: np.ndarray, gap: tuple[int, int], direction: str, projected: list[tuple[int, int]]) -> int:
    for r, c in _marker_segment(gap, direction, projected):
        value = int(grid[r, c])
        if value != _BG:
            return value
    raise AssertionError("expected at least one marker on the projection ray")


def _room_from_component(grid: np.ndarray, color: int, component: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(component)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    border = _rect_border_cells(r0, c0, r1, c1)
    comp_set = set(component)
    if len(comp_set) != len(border) - 1 or not comp_set.issubset(border):
        return None

    missing = list(border - comp_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    direction = _gap_direction(gap, (r0, c0, r1, c1))
    if direction is None:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    non_bg = np.argwhere(interior != _BG)
    if non_bg.size == 0:
        return None

    pattern_colors = np.unique(interior[interior != _BG])
    if len(pattern_colors) != 1:
        return None
    pattern_color = int(pattern_colors[0])
    if pattern_color == color:
        return None

    pattern = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in non_bg]
    if not _is_connected(pattern):
        return None
    if not _frontier_aligned(pattern, gap, direction):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_color": color,
        "pattern_color": pattern_color,
        "gap": gap,
        "direction": direction,
        "pattern": pattern,
    }


def _detect_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != _BG):
        for component in _components_of_color(grid, color):
            room = _room_from_component(grid, color, component)
            if room is not None:
                rooms.append(room)
    rooms.sort(key=lambda room: tuple(room["bbox"]))  # type: ignore[arg-type]
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each room's interior pattern out through its border gap and repaint the projection with the first marker found on that ray.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular room borders appear as monochrome frames with exactly one missing border pixel on one side.
    - Inside each room there is one connected single-color pattern, and outside the gap there are one or more single-pixel markers on the same row or column.

    Rule:
    - For each room, detect which side contains the one-cell gap to get the outward direction.
    - Translate the whole interior pattern straight along that axis, without sideways shift, until the pattern's leading edge touches the grid boundary.
    - On the gap ray between the room and the projected copy, find the first non-black marker pixel and use that marker's color to repaint the projected copy.
    - Keep the original room, its original interior pattern, and the ray markers unchanged.

    Key insight:
    - The gap controls motion direction, but the projected copy's color is chosen by the first external marker on that same ray rather than by any color already inside the room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _detect_rooms(g)
    assert rooms, "expected at least one room"

    out = g.copy()
    for room in rooms:
        pattern = room["pattern"]
        direction = room["direction"]
        gap = room["gap"]
        projected = _project_coords(pattern, direction, g.shape)  # type: ignore[arg-type]
        paint_color = _first_marker_color(g, gap, direction, projected)  # type: ignore[arg-type]
        for r, c in projected:
            if out[r, c] == _BG:
                out[r, c] = paint_color
    return out


def _grow_pattern(interior_h: int, interior_w: int, direction: str, rng: np.random.Generator) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(_PATTERN_ATTEMPTS):
        if direction == "up":
            anchor = (int(rng.integers(0, interior_h - 1)), int(rng.integers(0, interior_w)))
            allowed = {
                (r, c)
                for r in range(interior_h)
                for c in range(interior_w)
                if r > anchor[0] or (r, c) == anchor
            }
        elif direction == "down":
            anchor = (int(rng.integers(1, interior_h)), int(rng.integers(0, interior_w)))
            allowed = {
                (r, c)
                for r in range(interior_h)
                for c in range(interior_w)
                if r < anchor[0] or (r, c) == anchor
            }
        elif direction == "left":
            anchor = (int(rng.integers(0, interior_h)), int(rng.integers(0, interior_w - 1)))
            allowed = {
                (r, c)
                for r in range(interior_h)
                for c in range(interior_w)
                if c > anchor[1] or (r, c) == anchor
            }
        else:
            anchor = (int(rng.integers(0, interior_h)), int(rng.integers(1, interior_w)))
            allowed = {
                (r, c)
                for r in range(interior_h)
                for c in range(interior_w)
                if c < anchor[1] or (r, c) == anchor
            }

        max_size = min(7, len(allowed))
        if max_size < 1:
            continue
        min_size = 1 if rng.random() < 0.15 else 2
        if max_size < min_size:
            min_size = 1
        target_size = int(rng.integers(min_size, max_size + 1))

        cells = {anchor}
        frontier = [
            cell
            for cell in sorted(allowed)
            if cell != anchor and any(nb in cells for nb in _neighbors4(*cell))
        ]
        while len(cells) < target_size and frontier:
            idx = int(rng.integers(0, len(frontier)))
            cell = frontier.pop(idx)
            if cell in cells:
                continue
            if not any(nb in cells for nb in _neighbors4(*cell)):
                continue
            cells.add(cell)
            for nb in _neighbors4(*cell):
                if nb in allowed and nb not in cells and nb not in frontier:
                    frontier.append(nb)

        if len(cells) != target_size or not _is_connected(cells):
            continue
        return sorted(cells), anchor
    return None


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _build_room(
    shape: tuple[int, int],
    direction: str,
    blocked: set[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(_ROOM_ATTEMPTS):
        interior_h = int(rng.integers(3, 7))
        interior_w = int(rng.integers(3, 7))
        room_h = interior_h + 2
        room_w = interior_w + 2
        if h - room_h <= 1 or w - room_w <= 1:
            continue

        grown = _grow_pattern(interior_h, interior_w, direction, rng)
        if grown is None:
            continue
        local_pattern, anchor = grown

        r0 = int(rng.integers(1, h - room_h))
        c0 = int(rng.integers(1, w - room_w))
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1

        if direction == "up":
            gap = (r0, c0 + 1 + anchor[1])
        elif direction == "down":
            gap = (r1, c0 + 1 + anchor[1])
        elif direction == "left":
            gap = (r0 + 1 + anchor[0], c0)
        else:
            gap = (r0 + 1 + anchor[0], c1)

        border = _rect_border_cells(r0, c0, r1, c1)
        border.remove(gap)
        pattern = [(r0 + 1 + pr, c0 + 1 + pc) for pr, pc in local_pattern]
        projected = _project_coords(pattern, direction, shape)
        marker_cells = _marker_segment(gap, direction, projected)
        if not marker_cells:
            continue

        marker_count = int(rng.integers(1, min(4, len(marker_cells) + 1)))
        picked = sorted(int(i) for i in rng.choice(len(marker_cells), size=marker_count, replace=False).tolist())
        markers = [marker_cells[i] for i in picked]

        unit_in = set(border) | set(pattern) | set(markers)
        unit_out = unit_in | set(projected)
        if len(unit_out) != len(border) + len(pattern) + len(markers) + len(projected):
            continue
        if any(cell in blocked for cell in unit_out):
            continue

        border_color = int(rng.choice(_COLORS))
        pattern_color_pool = [color for color in _COLORS if color != border_color]
        pattern_color = int(rng.choice(pattern_color_pool))
        marker_color_pool = [color for color in _COLORS if color not in {border_color, pattern_color}]
        if len(marker_color_pool) < marker_count:
            continue
        marker_colors = [int(v) for v in rng.choice(marker_color_pool, size=marker_count, replace=False).tolist()]

        return {
            "bbox": (r0, c0, r1, c1),
            "direction": direction,
            "gap": gap,
            "border": sorted(border),
            "pattern": pattern,
            "projected": projected,
            "markers": list(zip(markers, marker_colors, strict=False)),
            "border_color": border_color,
            "pattern_color": pattern_color,
            "paint_color": marker_colors[0],
            "unit_out": unit_out,
        }
    return None


def _render_case(shape: tuple[int, int], rooms: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, _BG, dtype=int)
    out = np.full(shape, _BG, dtype=int)
    for room in rooms:
        border_color = int(room["border_color"])
        pattern_color = int(room["pattern_color"])
        for r, c in room["border"]:
            inp[r, c] = border_color
            out[r, c] = border_color
        for r, c in room["pattern"]:
            inp[r, c] = pattern_color
            out[r, c] = pattern_color
        for (r, c), color in room["markers"]:
            inp[r, c] = int(color)
            out[r, c] = int(color)
        for r, c in room["projected"]:
            out[r, c] = int(room["paint_color"])
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        h = int(rng.integers(18, 31))
        w = int(rng.integers(18, 31))
        max_rooms = 3 if min(h, w) >= 22 else 2
        room_count = int(rng.integers(1, max_rooms + 1))

        blocked: set[tuple[int, int]] = set()
        rooms: list[dict[str, object]] = []
        ok = True
        for _room_idx in range(room_count):
            room = None
            for _ in range(_ROOM_ATTEMPTS):
                direction = _DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))]
                room = _build_room((h, w), direction, blocked, rng)
                if room is not None:
                    rooms.append(room)
                    blocked |= _dilate(room["unit_out"], (h, w), pad=1)  # type: ignore[arg-type]
                    break
            if room is None:
                ok = False
                break
        if not ok or not rooms:
            continue

        inp, expected = _render_case((h, w), rooms)
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("generate_input failed to build a valid puzzle instance")
