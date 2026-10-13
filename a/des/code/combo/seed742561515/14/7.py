from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
ROOM_COUNT_RANGE = (4, 7)
ROOM_SIZE_RANGE = (4, 11)
PLACEMENT_TRIES = 120
GENERATE_RETRIES = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rooms: list[dict] = []
    for coords in _components4(g != GREY):
        if _touches_border(coords, h, w):
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        ir0, ir1 = min(rows), max(rows)
        ic0, ic1 = min(cols), max(cols)
        if len(coords) != (ir1 - ir0 + 1) * (ic1 - ic0 + 1):
            continue
        r0, c0, r1, c1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if np.any(g[ir0 : ir1 + 1, ic0 : ic1 + 1] == GREY):
            continue
        if not np.all(g[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == GREY):
            continue
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (ir0, ic0, ir1, ic1),
                "coords": coords,
            }
        )
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _paint_thick_border(block: np.ndarray, color: int, thickness: int) -> None:
    h, w = block.shape
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    dist = np.minimum(np.minimum(rr, h - 1 - rr), np.minimum(cc, w - 1 - cc))
    block[dist < thickness] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use same-color indicator pixels inside each grey rectangle to control the
    border recolor (by indicator color) and inward border thickness (by count).
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _find_rooms(g):
        r0, c0, r1, c1 = room["bbox"]
        ir0, ic0, ir1, ic1 = room["inner_bbox"]
        markers = [(r, c) for r, c in room["coords"] if int(g[r, c]) not in (BLACK, GREY)]
        if not markers:
            continue

        marker_values = [int(g[r, c]) for r, c in sorted(markers)]
        color = Counter(marker_values).most_common(1)[0][0]
        thickness = len(markers)

        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BLACK
        room_block = out[r0 : r1 + 1, c0 : c1 + 1]
        _paint_thick_border(room_block, color, thickness)

    return out


def _expected_room_block(height: int, width: int, color: int | None, count: int) -> np.ndarray:
    block = np.full((height, width), BLACK, dtype=int)
    block[0, :] = GREY
    block[-1, :] = GREY
    block[:, 0] = GREY
    block[:, -1] = GREY
    if count > 0 and color is not None:
        _paint_thick_border(block, color, count)
    return block


def _place_room(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: np.random.Generator,
    height: int,
    width: int,
    color: int | None,
    count: int,
) -> dict | None:
    h, w = grid.shape
    if height > h - 2 or width > w - 2:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = int(rng.integers(1, h - height))
        c0 = int(rng.integers(1, w - width))
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        pad_r0 = max(0, r0 - 1)
        pad_c0 = max(0, c0 - 1)
        pad_r1 = min(h, r1 + 2)
        pad_c1 = min(w, c1 + 2)
        if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
            continue

        block = _expected_room_block(height, width, color, 0)
        grid[r0 : r1 + 1, c0 : c1 + 1] = block

        if count > 0 and color is not None:
            inner_cells = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
            ]
            if count > len(inner_cells):
                return None
            picks = rng.choice(len(inner_cells), size=count, replace=False)
            for idx in np.atleast_1d(picks):
                rr, cc = inner_cells[int(idx)]
                grid[rr, cc] = int(color)

        occupied[pad_r0:pad_r1, pad_c0:pad_c1] = True
        return {
            "bbox": (r0, c0, r1, c1),
            "height": height,
            "width": width,
            "color": color,
            "count": count,
        }
    return None


def _random_partial_count(rng: np.random.Generator, height: int, width: int) -> int | None:
    threshold = (min(height, width) + 1) // 2
    if threshold <= 1:
        return None
    upper = min(2, threshold - 1)
    if upper < 1:
        return None
    return int(rng.integers(1, upper + 1))


def _random_fill_count(rng: np.random.Generator, height: int, width: int) -> int | None:
    threshold = (min(height, width) + 1) // 2
    inner_area = (height - 2) * (width - 2)
    if inner_area < threshold:
        return None
    low = threshold + 1 if threshold + 1 <= inner_area else threshold
    high = min(inner_area, threshold + 3, 8)
    return int(rng.integers(low, high + 1))


def _sample_room_specs(rng: np.random.Generator, room_total: int) -> list[dict]:
    colors = list(rng.permutation(PALETTE))
    color_a, color_b, color_c = colors[:3]
    shared_partial_count = int(rng.choice([1, 2]))

    specs = [
        {"kind": "inactive", "color": None},
        {"kind": "partial", "color": color_a, "fixed_count": shared_partial_count},
        {"kind": "fill", "color": color_a},
        {"kind": "partial", "color": color_b, "fixed_count": shared_partial_count},
    ]

    extras = room_total - len(specs)
    for _ in range(extras):
        kind = rng.choice(["inactive", "partial", "fill"], p=[0.35, 0.4, 0.25])
        specs.append(
            {
                "kind": str(kind),
                "color": int(rng.choice([color_a, color_b, color_c])) if kind != "inactive" else None,
            }
        )
    rng.shuffle(specs)
    return specs


def _choose_size_for_kind(rng: np.random.Generator, kind: str) -> tuple[int, int]:
    if kind == "inactive":
        low_h = ROOM_SIZE_RANGE[0]
        low_w = ROOM_SIZE_RANGE[0]
    else:
        low_h = max(5, ROOM_SIZE_RANGE[0])
        low_w = max(5, ROOM_SIZE_RANGE[0])
    height = int(rng.integers(low_h, ROOM_SIZE_RANGE[1] + 1))
    width = int(rng.integers(low_w, ROOM_SIZE_RANGE[1] + 1))
    return height, width


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    room_total = int(rng.integers(*ROOM_COUNT_RANGE))

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    specs = _sample_room_specs(rng, room_total)

    records: list[dict] = []
    for spec in specs:
        placed = None
        for _ in range(PLACEMENT_TRIES):
            height, width = _choose_size_for_kind(rng, spec["kind"])
            if spec["kind"] == "inactive":
                count = 0
            elif spec["kind"] == "partial":
                count = int(spec.get("fixed_count", -1))
                if count < 1:
                    count = _random_partial_count(rng, height, width)
                elif count >= (min(height, width) + 1) // 2:
                    continue
            else:
                count = _random_fill_count(rng, height, width)
            if count is None:
                continue
            placed = _place_room(grid, occupied, rng, height, width, spec["color"], count)
            if placed is not None:
                records.append(placed)
                break
        if placed is None:
            return None

    active = [rec for rec in records if rec["count"] > 0]
    if len(active) < 3:
        return None
    if not any(rec["count"] == 0 for rec in records):
        return None
    if len({rec["color"] for rec in active}) < 2:
        return None
    if not any(
        rec["count"] < (min(rec["height"], rec["width"]) + 1) // 2
        for rec in active
    ):
        return None
    if not any(
        rec["count"] >= (min(rec["height"], rec["width"]) + 1) // 2
        for rec in active
    ):
        return None
    if not any(
        a["color"] == b["color"] and a["count"] != b["count"]
        for i, a in enumerate(active)
        for b in active[i + 1 :]
    ):
        return None
    if not any(
        a["count"] == b["count"] and a["color"] != b["color"]
        for i, a in enumerate(active)
        for b in active[i + 1 :]
    ):
        return None

    rooms = _find_rooms(grid)
    if len(rooms) != len(records):
        return None

    out = main(grid)
    for rec in records:
        r0, c0, r1, c1 = rec["bbox"]
        expected = _expected_room_block(rec["height"], rec["width"], rec["color"], rec["count"])
        if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], expected):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
