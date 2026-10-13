import random
from collections import deque

import numpy as np


GEN_ATTEMPTS = 400
SIDES = ("top", "right", "bottom", "left")


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in coords]
            cs = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )
    return comps


def _frame_info(grid: np.ndarray, bg: int, comp: dict):
    color = comp["color"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[0, :] == color):
        return None
    if not np.all(patch[-1, :] == color):
        return None
    if not np.all(patch[:, 0] == color):
        return None
    if not np.all(patch[:, -1] == color):
        return None
    if np.any(patch[1:-1, 1:-1] != bg):
        return None
    if comp["area"] != 2 * (h + w) - 4:
        return None
    for r, c in comp["coords"]:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    return {
        "color": color,
        "bbox": comp["bbox"],
        "area": comp["area"],
    }


def _clockwise_index(contact: tuple[int, int], frame: dict) -> int:
    r, c = contact
    r0, c0, r1, c1 = frame["bbox"]
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    if r == r0:
        return c - c0
    if c == c1:
        return (width - 1) + (r - r0)
    if r == r1:
        return (width - 1) + (height - 1) + (c1 - c)
    if c == c0:
        return (width - 1) + (height - 1) + (width - 1) + (r1 - r)
    raise AssertionError("contact must lie on the frame border")


def _contact_info(grid: np.ndarray, comp: dict, frame: dict) -> dict:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    contacts = set()
    for r, c in comp["coords"]:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(grid[nr, nc]) != frame["color"]:
                continue
            if (nr in (r0, r1) and c0 <= nc <= c1) or (nc in (c0, c1) and r0 <= nr <= r1):
                contacts.add((nr, nc))
    if len(contacts) != 1:
        raise AssertionError("each wire component must touch the display border exactly once")
    contact = next(iter(contacts))
    cr, cc = contact
    if cr == r0:
        side = "top"
    elif cc == c1:
        side = "right"
    elif cr == r1:
        side = "bottom"
    elif cc == c0:
        side = "left"
    else:
        raise AssertionError("invalid contact side")
    return {
        "color": comp["color"],
        "contact": contact,
        "side": side,
        "order": _clockwise_index(contact, frame),
    }


def _extract_task(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _find_components(g, bg)
    frames = []
    for idx, comp in enumerate(comps):
        info = _frame_info(g, bg, comp)
        if info is not None:
            info["id"] = idx
            frames.append(info)
    assert frames, "expected a rectangular display frame"
    frame = max(frames, key=lambda item: (item["area"], item["bbox"][2] - item["bbox"][0], item["bbox"][3] - item["bbox"][1]))
    wires = []
    for idx, comp in enumerate(comps):
        if idx == frame["id"]:
            continue
        wires.append(_contact_info(g, comp, frame))
    assert wires, "expected at least one signal wire"
    wires.sort(key=lambda item: item["order"])
    return bg, frame, wires


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each external signal wire onto the display interior.

    A large rectangular frame encloses an empty display. Each colored 2x2 signal
    block is connected to the frame by a same-colored wire. The contacted border
    side determines whether the signal lights a full interior row or a full
    interior column, and clashes keep the earlier clockwise signal color.
    """
    g = np.asarray(grid, dtype=int)
    bg, frame, wires = _extract_task(g)
    r0, c0, r1, c1 = frame["bbox"]
    out = g.copy()
    for wire in wires:
        color = wire["color"]
        if wire["side"] in ("top", "bottom"):
            col = wire["contact"][1]
            segment = out[r0 + 1 : r1, col]
            segment[segment == bg] = color
        else:
            row = wire["contact"][0]
            segment = out[row, c0 + 1 : c1]
            segment[segment == bg] = color
    return out


def _frame_moat(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    moat = np.zeros((h, w), dtype=bool)
    if r0 - 1 >= 0:
        moat[r0 - 1, c0 : c1 + 1] = True
    if r1 + 1 < h:
        moat[r1 + 1, c0 : c1 + 1] = True
    if c0 - 1 >= 0:
        moat[r0 : r1 + 1, c0 - 1] = True
    if c1 + 1 < w:
        moat[r0 : r1 + 1, c1 + 1] = True
    return moat


def _clockwise_order_key(wire: dict, frame_bbox: tuple[int, int, int, int]) -> int:
    return _clockwise_index(wire["contact"], {"bbox": frame_bbox})


def _expected_output(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int], wires: list[dict]) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    out = grid.copy()
    for wire in sorted(wires, key=lambda item: _clockwise_order_key(item, frame_bbox)):
        color = wire["color"]
        if wire["side"] in ("top", "bottom"):
            col = wire["contact"][1]
            segment = out[r0 + 1 : r1, col]
            segment[segment == bg] = color
        else:
            row = wire["contact"][0]
            segment = out[row, c0 + 1 : c1]
            segment[segment == bg] = color
    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    gh, gw = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(gh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(gw, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _bfs_path(
    occupied: np.ndarray,
    moat: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: random.Random,
):
    h, w = occupied.shape
    if occupied[start] or occupied[goal]:
        return None
    seen = np.zeros((h, w), dtype=bool)
    q = deque([start])
    seen[start] = True
    parent = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        nbrs = list(_neighbors4(r, c))
        rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if seen[nr, nc] or occupied[nr, nc]:
                continue
            if moat[nr, nc] and (nr, nc) != goal:
                continue
            seen[nr, nc] = True
            parent[(nr, nc)] = (r, c)
            q.append((nr, nc))
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _side_counts(rng: random.Random) -> dict[str, int]:
    total = rng.randint(4, 6)
    sides = list(SIDES)
    rng.shuffle(sides)
    chosen = sides[:]
    while len(chosen) < total:
        chosen.append(rng.choice(SIDES))
    counts = {side: chosen.count(side) for side in SIDES}
    assert all(counts[side] >= 1 for side in SIDES)
    return counts


def _wire_specs(
    rng: random.Random,
    colors: list[int],
    bbox: tuple[int, int, int, int],
    counts: dict[str, int],
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    total_cols = counts["top"] + counts["bottom"]
    total_rows = counts["left"] + counts["right"]
    col_slots = list(range(c0 + 1, c1))
    row_slots = list(range(r0 + 1, r1))
    rng.shuffle(col_slots)
    rng.shuffle(row_slots)
    top_cols = col_slots[: counts["top"]]
    bottom_cols = col_slots[counts["top"] : total_cols]
    left_rows = row_slots[: counts["left"]]
    right_rows = row_slots[counts["left"] : total_rows]

    wires = []
    color_iter = iter(colors)
    for col in top_cols:
        wires.append({"color": next(color_iter), "side": "top", "contact": (r0, col)})
    for row in right_rows:
        wires.append({"color": next(color_iter), "side": "right", "contact": (row, c1)})
    for col in bottom_cols:
        wires.append({"color": next(color_iter), "side": "bottom", "contact": (r1, col)})
    for row in left_rows:
        wires.append({"color": next(color_iter), "side": "left", "contact": (row, c0)})
    rng.shuffle(wires)
    return wires


def _start_cell(side: str, contact: tuple[int, int]) -> tuple[int, int]:
    r, c = contact
    if side == "top":
        return r - 1, c
    if side == "right":
        return r, c + 1
    if side == "bottom":
        return r + 1, c
    if side == "left":
        return r, c - 1
    raise AssertionError("bad side")


def _candidate_blocks(side: str, shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[int, int]] = []
    if side == "top":
        for sr in range(0, r0 - 3):
            for sc in range(0, w - 1):
                candidates.append((sr, sc))
    elif side == "bottom":
        for sr in range(r1 + 3, h - 1):
            for sc in range(0, w - 1):
                candidates.append((sr, sc))
    elif side == "left":
        for sr in range(0, h - 1):
            for sc in range(0, c0 - 3):
                candidates.append((sr, sc))
    else:
        for sr in range(0, h - 1):
            for sc in range(c1 + 3, w - 1):
                candidates.append((sr, sc))
    return candidates


def _port_cell(side: str, block_rc: tuple[int, int], offset: int) -> tuple[int, int]:
    sr, sc = block_rc
    if side == "top":
        return sr + 2, sc + offset
    if side == "bottom":
        return sr - 1, sc + offset
    if side == "left":
        return sr + offset, sc + 2
    if side == "right":
        return sr + offset, sc - 1
    raise AssertionError("bad side")


def _place_wire(
    grid: np.ndarray,
    occupied: np.ndarray,
    moat: np.ndarray,
    bbox: tuple[int, int, int, int],
    wire: dict,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    side = wire["side"]
    color = wire["color"]
    start = _start_cell(side, wire["contact"])
    if not (0 <= start[0] < h and 0 <= start[1] < w) or occupied[start]:
        return False
    candidates = _candidate_blocks((side), (h, w), bbox)
    rng.shuffle(candidates)
    for sr, sc in candidates:
        if not _can_place(occupied, sr, sc, 2, 2, pad=1):
            continue
        offsets = [0, 1]
        rng.shuffle(offsets)
        for offset in offsets:
            port = _port_cell(side, (sr, sc), offset)
            pr, pc = port
            if not (0 <= pr < h and 0 <= pc < w):
                continue
            if occupied[pr, pc] or moat[pr, pc]:
                continue
            path = _bfs_path(occupied, moat, port, start, rng)
            if path is None:
                continue
            grid[sr : sr + 2, sc : sc + 2] = color
            occupied[sr : sr + 2, sc : sc + 2] = True
            for r, c in path:
                grid[r, c] = color
                occupied[r, c] = True
            return True
    return False


def _gen_attempt(rng: random.Random):
    counts = _side_counts(rng)
    inner_h = rng.randint(max(5, counts["left"] + counts["right"]), 9)
    inner_w = rng.randint(max(6, counts["top"] + counts["bottom"]), 11)
    top_margin = rng.randint(6, 9)
    bottom_margin = rng.randint(6, 9)
    left_margin = rng.randint(6, 9)
    right_margin = rng.randint(6, 9)

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    h = r1 + bottom_margin + 1
    w = c1 + right_margin + 1

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    frame_color = rng.choice(palette)
    palette.remove(frame_color)
    total_wires = sum(counts.values())
    wire_colors = rng.sample(palette, total_wires)

    grid = np.full((h, w), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    bbox = (r0, c0, r1, c1)
    wires = _wire_specs(rng, wire_colors, bbox, counts)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    moat = _frame_moat((h, w), bbox)

    route_order = wires[:]
    rng.shuffle(route_order)
    for wire in route_order:
        if not _place_wire(grid, occupied, moat, bbox, wire, rng):
            return None

    expected = _expected_output(grid, bg, bbox, wires)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    _, frame, extracted = _extract_task(grid)
    if frame["bbox"] != bbox or len(extracted) != len(wires):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
