import numpy as np


BLACK = 0
AZURE = 8
MARKER_COLORS = np.array([1, 2, 3, 4, 6, 7, 9], dtype=int)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    rooms: list[dict[str, object]] = []
    marker_total = int(np.count_nonzero((grid != BLACK) & (grid != AZURE)))

    for comp in _components(grid == AZURE):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 3 and c1 - c0 >= 3, "room walls must enclose at least a 2x2 interior"

        boundary = _boundary_cells(r0, c0, r1, c1)
        boundary_set = set(boundary)
        comp_set = set(comp)
        assert comp_set <= boundary_set, "azure room component must lie on its rectangle boundary"
        missing = [cell for cell in boundary if cell not in comp_set]
        assert len(missing) == 1, "each room boundary must have exactly one door gap"
        door_r, door_c = missing[0]
        assert grid[door_r, door_c] == BLACK, "door gap must be black"
        assert not ((door_r in (r0, r1)) and (door_c in (c0, c1))), "door may not be on a corner"
        assert len(comp) == len(boundary) - 1, "room wall should be a full rectangle minus one door"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert not np.any(interior == AZURE), "room interior may not contain azure"
        markers = np.argwhere((interior != BLACK) & (interior != AZURE))
        assert markers.shape[0] == 1, "each room must contain exactly one marker"
        mr, mc = map(int, markers[0])
        mr += r0 + 1
        mc += c0 + 1
        color = int(grid[mr, mc])

        if door_r == r0:
            side = "top"
            outside = (r0 - 1, door_c)
            axis = "h"
            pos = door_c
        elif door_r == r1:
            side = "bottom"
            outside = (r1 + 1, door_c)
            axis = "h"
            pos = door_c
        elif door_c == c0:
            side = "left"
            outside = (door_r, c0 - 1)
            axis = "v"
            pos = door_r
        else:
            side = "right"
            outside = (door_r, c1 + 1)
            axis = "v"
            pos = door_r

        orow, ocol = outside
        assert 0 <= orow < h and 0 <= ocol < w, "door must open onto an in-bounds corridor cell"
        assert grid[orow, ocol] == BLACK, "door must open onto a black corridor cell"

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "door": (door_r, door_c),
                "marker": (mr, mc),
                "color": color,
                "side": side,
                "axis": axis,
                "outside": outside,
                "pos": pos,
            }
        )

    assert len(rooms) == marker_total, "every marker must belong to exactly one room"
    return rooms


def _corridor_id(grid: np.ndarray, axis: str, outside: tuple[int, int]) -> tuple[str, int, int, int]:
    h, w = grid.shape
    r, c = outside
    if axis == "h":
        left = c
        right = c
        while left - 1 >= 0 and grid[r, left - 1] != AZURE:
            left -= 1
        while right + 1 < w and grid[r, right + 1] != AZURE:
            right += 1
        return ("h", r, left, right)

    top = r
    bottom = r
    while top - 1 >= 0 and grid[top - 1, c] != AZURE:
        top -= 1
    while bottom + 1 < h and grid[bottom + 1, c] != AZURE:
        bottom += 1
    return ("v", c, top, bottom)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spill each room's marker color into its adjacent corridor segment while keeping the azure walls unchanged.

    Input structure:
    - The background is black, and azure pixels form one-pixel-thick rectangular room walls.
    - Each room wall is a rectangle with exactly one missing boundary pixel that acts as a door.
    - Each room interior contains exactly one non-black, non-azure marker pixel.
    - The doors open onto one-pixel-wide straight black corridors that continue until another door position or the grid edge.

    Rule:
    - Keep every azure wall pixel unchanged.
    - For each room, find its door and identify the straight corridor line immediately outside that door.
    - Remove the room's marker pixel, then color black corridor cells by the nearest door on that corridor line using that room's marker color.
    - If two doors are equally close to a corridor cell, leave that cell black; otherwise the corridor colors extend until another door takes over or the corridor reaches the boundary.

    Key insight:
    - The missing wall pixel identifies both the room and the corridor it controls; the output is produced by partitioning each straight corridor by proximity to the doors attached to it, not by filling room interiors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == AZURE), "expected azure room walls"

    rooms = _find_rooms(g)
    out = g.copy()
    for room in rooms:
        mr, mc = room["marker"]
        out[mr, mc] = BLACK

    corridors: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    for room in rooms:
        cid = _corridor_id(g, room["axis"], room["outside"])
        corridors.setdefault(cid, []).append(room)

    for cid, entries in corridors.items():
        positions = [int(entry["pos"]) for entry in entries]
        assert len(positions) == len(set(positions)), "corridor door positions must be unique"

        if cid[0] == "h":
            _, row, left, right = cid
            for col in range(left, right + 1):
                if out[row, col] != BLACK:
                    continue
                distances = [abs(col - pos) for pos in positions]
                best = min(distances)
                if distances.count(best) != 1:
                    continue
                winner = entries[distances.index(best)]
                out[row, col] = int(winner["color"])
        else:
            _, col, top, bottom = cid
            for row in range(top, bottom + 1):
                if out[row, col] != BLACK:
                    continue
                distances = [abs(row - pos) for pos in positions]
                best = min(distances)
                if distances.count(best) != 1:
                    continue
                winner = entries[distances.index(best)]
                out[row, col] = int(winner["color"])

    return out


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    door: tuple[int, int],
    marker: tuple[int, int],
    color: int,
) -> None:
    grid[r0, c0 : c1 + 1] = AZURE
    grid[r1, c0 : c1 + 1] = AZURE
    grid[r0 : r1 + 1, c0] = AZURE
    grid[r0 : r1 + 1, c1] = AZURE
    dr, dc = door
    grid[dr, dc] = BLACK
    mr, mc = marker
    grid[mr, mc] = color


def _make_horizontal(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 31))
    w = int(rng.integers(18, 31))
    corridor_row = int(rng.integers(7, h - 7))
    max_rooms = min(5, (w + 1) // 5)
    if max_rooms < 3:
        return None
    room_count = int(rng.integers(3, max_rooms + 1))

    for _ in range(80):
        widths = rng.integers(4, 9, size=room_count)
        base_between = rng.integers(1, 4, size=room_count - 1)
        min_total = int(widths.sum() + base_between.sum())
        if min_total > w:
            continue

        extras = np.zeros(room_count + 1, dtype=int)
        for _ in range(w - min_total):
            extras[int(rng.integers(0, room_count + 1))] += 1
        gaps = extras.copy()
        gaps[1:-1] += base_between

        colors = rng.choice(MARKER_COLORS, size=room_count, replace=False)
        sides = ["above", "below"] + [rng.choice(["above", "below"]) for _ in range(room_count - 2)]
        rng.shuffle(sides)
        start_parity = int(rng.integers(0, 2))

        grid = np.zeros((h, w), dtype=int)
        x = int(gaps[0])
        rooms: list[dict[str, int | str]] = []
        ok = True

        for idx in range(room_count):
            outer_w = int(widths[idx])
            side = str(sides[idx])
            outer_h = int(rng.integers(4, 8))
            if side == "above":
                r0, r1 = corridor_row - outer_h, corridor_row - 1
            else:
                r0, r1 = corridor_row + 1, corridor_row + outer_h
            if r0 < 0 or r1 >= h:
                ok = False
                break

            c0 = x
            c1 = x + outer_w - 1
            desired_parity = start_parity ^ (idx % 2)
            door_candidates = [c for c in range(c0 + 1, c1) if c % 2 == desired_parity]
            if not door_candidates:
                ok = False
                break
            door_c = int(rng.choice(door_candidates))
            if side == "above":
                door = (r1, door_c)
            else:
                door = (r0, door_c)
            mr = int(rng.integers(r0 + 1, r1))
            mc = int(rng.integers(c0 + 1, c1))
            rooms.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "door_r": int(door[0]),
                    "door_c": int(door[1]),
                    "mr": mr,
                    "mc": mc,
                    "color": int(colors[idx]),
                }
            )
            x = c1 + 1 + int(gaps[idx + 1])

        if not ok:
            continue

        for room in rooms:
            _draw_room(
                grid,
                room["r0"],
                room["c0"],
                room["r1"],
                room["c1"],
                (room["door_r"], room["door_c"]),
                (room["mr"], room["mc"]),
                room["color"],
            )

        return grid

    return None


def _make_vertical(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(16, 31))
    corridor_col = int(rng.integers(7, w - 7))
    max_rooms = min(5, (h + 1) // 5)
    if max_rooms < 3:
        return None
    room_count = int(rng.integers(3, max_rooms + 1))

    for _ in range(80):
        heights = rng.integers(4, 9, size=room_count)
        base_between = rng.integers(1, 4, size=room_count - 1)
        min_total = int(heights.sum() + base_between.sum())
        if min_total > h:
            continue

        extras = np.zeros(room_count + 1, dtype=int)
        for _ in range(h - min_total):
            extras[int(rng.integers(0, room_count + 1))] += 1
        gaps = extras.copy()
        gaps[1:-1] += base_between

        colors = rng.choice(MARKER_COLORS, size=room_count, replace=False)
        sides = ["left", "right"] + [rng.choice(["left", "right"]) for _ in range(room_count - 2)]
        rng.shuffle(sides)
        start_parity = int(rng.integers(0, 2))

        grid = np.zeros((h, w), dtype=int)
        y = int(gaps[0])
        rooms: list[dict[str, int | str]] = []
        ok = True

        for idx in range(room_count):
            outer_h = int(heights[idx])
            side = str(sides[idx])
            outer_w = int(rng.integers(4, 8))
            if side == "left":
                c0, c1 = corridor_col - outer_w, corridor_col - 1
            else:
                c0, c1 = corridor_col + 1, corridor_col + outer_w
            if c0 < 0 or c1 >= w:
                ok = False
                break

            r0 = y
            r1 = y + outer_h - 1
            desired_parity = start_parity ^ (idx % 2)
            door_candidates = [r for r in range(r0 + 1, r1) if r % 2 == desired_parity]
            if not door_candidates:
                ok = False
                break
            door_r = int(rng.choice(door_candidates))
            if side == "left":
                door = (door_r, c1)
            else:
                door = (door_r, c0)
            mr = int(rng.integers(r0 + 1, r1))
            mc = int(rng.integers(c0 + 1, c1))
            rooms.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "door_r": int(door[0]),
                    "door_c": int(door[1]),
                    "mr": mr,
                    "mc": mc,
                    "color": int(colors[idx]),
                }
            )
            y = r1 + 1 + int(gaps[idx + 1])

        if not ok:
            continue

        for room in rooms:
            _draw_room(
                grid,
                room["r0"],
                room["c0"],
                room["r1"],
                room["c1"],
                (room["door_r"], room["door_c"]),
                (room["mr"], room["mc"]),
                room["color"],
            )

        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        if bool(rng.integers(0, 2)):
            grid = _make_horizontal(rng)
        else:
            grid = _make_vertical(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape == grid.shape and np.any(out != grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
