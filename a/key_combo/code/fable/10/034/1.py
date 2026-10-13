from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PAYLOAD_COLORS = [RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]
FRAME_PAD = 1
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_corner(rc: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = rc
    r0, c0, r1, c1 = bbox
    return (r, c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return -1, 0
    if r == r1:
        return 1, 0
    if c == c0:
        return 0, -1
    assert c == c1
    return 0, 1


def _frame_border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                out.append((r, c))
    return out


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK

    frames = []
    for coords in _find_components_of_color(g, BLUE):
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4
        coord_set = {tuple(map(int, rc)) for rc in coords}
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in coord_set)

        missing = []
        for rc in _frame_border_cells(bbox):
            if rc not in coord_set:
                missing.append(rc)
        assert len(missing) in (0, 1)
        if missing:
            assert not _is_corner(missing[0], bbox)

        payload = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                val = int(g[r, c])
                if val == BLACK:
                    continue
                assert val != BLUE
                payload.append((r, c, val))
                for nr, nc in neighbors4(r, c):
                    if r0 < nr < r1 and c0 < nc < c1 and int(g[nr, nc]) != BLACK:
                        raise AssertionError("payload pixels inside a gapped frame must stay isolated")

        gap = missing[0] if missing else None
        if gap is not None:
            assert payload
            dists = [abs(r - gap[0]) + abs(c - gap[1]) for r, c, _ in payload]
            assert len(set(dists)) == len(dists)
            direction = _gap_direction(gap, bbox)
        else:
            dists = []
            direction = None

        frames.append(
            {
                "bbox": bbox,
                "gap": gap,
                "direction": direction,
                "payload": payload,
                "distances": dists,
            }
        )

    frames.sort(key=lambda item: item["bbox"])
    assert frames
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spill each gapped blue container outward by distance order and stack the emitted pixels against the first blocker.

    Input structure:
    - The background is black.
    - Several blue one-cell-thick rectangular frames appear; each frame is either complete or has exactly one missing non-corner border cell.
    - Every gapped frame contains isolated non-blue payload pixels in its interior, and their Manhattan distances to that frame's gap are all distinct.
    - Along the outward line perpendicular to a gap, empty cells continue until the grid edge or another non-black obstacle.

    Rule:
    - Keep every complete frame, including anything inside it, unchanged.
    - For each gapped frame, sort its interior payload pixels by increasing Manhattan distance to the gap.
    - Launch them one by one straight outward from the cell just outside the gap; each pixel flies until the next step would leave the grid or hit any non-black cell, then it stops in the previous cell.
    - Earlier launches therefore land farther away, later launches stack closer to the gap; after all launches, clear the frame interior and close the gap with blue.

    Key insight:
    - The missing border cell is an exit, not damage: distance-to-gap determines launch order, and collision stacking reverses that order along the outside line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK

    frames = _extract_frames(g)
    out = g.copy()

    for frame in frames:
        gap = frame["gap"]
        if gap is None:
            continue

        for r, c, _color in frame["payload"]:
            out[r, c] = BLACK

        dr, dc = frame["direction"]
        ordered = sorted(
            frame["payload"],
            key=lambda item: abs(item[0] - gap[0]) + abs(item[1] - gap[1]),
        )
        for _r, _c, color in ordered:
            rr, cc = gap[0] + dr, gap[1] + dc
            last = None
            while in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BLACK:
                last = (rr, cc)
                rr += dr
                cc += dc
            assert last is not None, "the outward corridor must have enough capacity"
            out[last] = int(color)

        out[gap] = BLUE

    return out


def _reserve_bbox(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = FRAME_PAD) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0] - 1, r1 + pad)
    cc1 = min(reserved.shape[1] - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _bbox_is_free(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = FRAME_PAD) -> bool:
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= reserved.shape[0] or c1 >= reserved.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0] - 1, r1 + pad)
    cc1 = min(reserved.shape[1] - 1, c1 + pad)
    return not reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _cells_are_free(reserved: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(in_bounds(reserved.shape, r, c) and not reserved[r, c] for r, c in cells)


def _reserve_cells(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved[r, c] = True


def _corridor_until_edge(shape: tuple[int, int], gap: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    rr, cc = gap[0] + dr, gap[1] + dc
    cells = []
    while in_bounds(shape, rr, cc):
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _corridor_until_block(
    gap: tuple[int, int], direction: tuple[int, int], block_cell: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = direction
    rr, cc = gap[0] + dr, gap[1] + dc
    cells = []
    while (rr, cc) != block_cell:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _sample_isolated_points(
    bbox: tuple[int, int, int, int],
    count: int,
    *,
    gap: tuple[int, int] | None = None,
    require_nontrivial: bool = False,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if len(interior) < count:
        return None

    axis_orders = [
        lambda rc: (rc[0], rc[1]),
        lambda rc: (rc[1], rc[0]),
    ]
    if gap is not None:
        axis_orders.extend(
            [
                lambda rc: (abs(rc[0] - gap[0]), abs(rc[1] - gap[1]), rc[0], rc[1]),
                lambda rc: (abs(rc[1] - gap[1]), abs(rc[0] - gap[0]), rc[0], rc[1]),
            ]
        )

    for _ in range(200):
        choice_idx = np.random.choice(len(interior), size=count, replace=False)
        pts = [interior[int(i)] for i in choice_idx]
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1 for i, a in enumerate(pts) for b in pts[i + 1 :]):
            continue
        if gap is not None:
            dists = [abs(r - gap[0]) + abs(c - gap[1]) for r, c in pts]
            if len(set(dists)) != count:
                continue
            if require_nontrivial:
                by_dist = [pt for _, pt in sorted(zip(dists, pts), key=lambda item: (item[0], item[1][0], item[1][1]))]
                if any(by_dist == sorted(pts, key=order) for order in axis_orders):
                    continue
        return pts
    return None


def _sample_colors(count: int) -> list[int]:
    palette = list(np.random.permutation(PAYLOAD_COLORS))
    if count <= len(palette):
        return [int(c) for c in palette[:count]]
    out = []
    for i in range(count):
        out.append(int(palette[i % len(palette)]))
    return out


def _make_complete_frame_record(
    bbox: tuple[int, int, int, int],
    payload_count: int,
) -> dict:
    points = []
    for want in range(payload_count, -1, -1):
        points = _sample_isolated_points(bbox, want) if want else []
        if points is not None:
            payload_count = want
            break
    if points is None:
        raise AssertionError("failed to place complete-frame payloads")
    colors = _sample_colors(payload_count)
    payload = [(r, c, colors[i]) for i, (r, c) in enumerate(points)]
    return {"bbox": bbox, "gap": None, "direction": None, "payload": payload, "blocking": None}


def _pick_payload_count(corridor_len: int, bbox: tuple[int, int, int, int], force_large: bool) -> int:
    r0, c0, r1, c1 = bbox
    interior_area = max(0, (r1 - r0 - 1) * (c1 - c0 - 1))
    upper = min(4, corridor_len, interior_area)
    lower = 3 if force_large and upper >= 3 else 2
    if upper < lower:
        lower = min(upper, 2)
    return int(np.random.randint(lower, upper + 1))


def _place_edge_gadget(
    reserved: np.ndarray,
    direction_name: str,
    require_nontrivial: bool,
) -> list[dict] | None:
    h_grid, w_grid = reserved.shape
    direction_map = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = direction_map[direction_name]

    for _ in range(160):
        h = int(np.random.randint(5, 8))
        w = int(np.random.randint(5, 8))
        if direction_name == "up":
            r0 = int(np.random.randint(2, h_grid - h))
            c0 = int(np.random.randint(1, w_grid - w - 1))
        elif direction_name == "down":
            r0 = int(np.random.randint(1, h_grid - h - 2))
            c0 = int(np.random.randint(1, w_grid - w - 1))
        elif direction_name == "left":
            r0 = int(np.random.randint(1, h_grid - h - 1))
            c0 = int(np.random.randint(2, w_grid - w))
        else:
            r0 = int(np.random.randint(1, h_grid - h - 1))
            c0 = int(np.random.randint(1, w_grid - w - 2))
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _bbox_is_free(reserved, bbox):
            continue

        if direction_name in ("up", "down"):
            gap_col = c0 + int(np.random.randint(1, w - 1))
            gap = (r0, gap_col) if direction_name == "up" else (r0 + h - 1, gap_col)
        else:
            gap_row = r0 + int(np.random.randint(1, h - 1))
            gap = (gap_row, c0) if direction_name == "left" else (gap_row, c0 + w - 1)

        corridor = _corridor_until_edge(reserved.shape, gap, (dr, dc))
        if len(corridor) < 2 or not _cells_are_free(reserved, corridor):
            continue

        payload_count = _pick_payload_count(len(corridor), bbox, force_large=require_nontrivial)
        points = _sample_isolated_points(bbox, payload_count, gap=gap, require_nontrivial=require_nontrivial)
        if points is None:
            continue

        colors = _sample_colors(payload_count)
        payload = [(r, c, colors[i]) for i, (r, c) in enumerate(points)]
        _reserve_bbox(reserved, bbox)
        _reserve_cells(reserved, corridor)
        return [
            {
                "bbox": bbox,
                "gap": gap,
                "direction": (dr, dc),
                "payload": payload,
                "corridor": corridor,
                "blocking": "edge",
            }
        ]
    return None


def _place_blocker_gadget(
    reserved: np.ndarray,
    direction_name: str,
    require_nontrivial: bool,
) -> list[dict] | None:
    h_grid, w_grid = reserved.shape
    direction_map = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = direction_map[direction_name]

    for _ in range(220):
        hs = int(np.random.randint(5, 8))
        ws = int(np.random.randint(5, 8))
        hb = int(np.random.randint(4, 8))
        wb = int(np.random.randint(4, 8))

        if direction_name in ("up", "down"):
            gap_offset = int(np.random.randint(1, ws - 1))
            align_size = hb
        else:
            gap_offset = int(np.random.randint(1, hs - 1))
            align_size = wb

        lane_len = int(np.random.randint(2, 7))

        if direction_name == "right":
            max_c0 = w_grid - (ws + lane_len + wb)
            if max_c0 <= 1 or h_grid - hs <= 2:
                continue
            r0s = int(np.random.randint(1, h_grid - hs))
            c0s = int(np.random.randint(1, max_c0))
            gap = (r0s + gap_offset, c0s + ws - 1)
            min_r0b = max(1, gap[0] - hb + 1)
            max_r0b = min(gap[0], h_grid - hb - 1)
            if min_r0b > max_r0b:
                continue
            r0b = int(np.random.randint(min_r0b, max_r0b + 1))
            c0b = c0s + ws + lane_len
            block_cell = (gap[0], c0b)
        elif direction_name == "left":
            min_c0s = lane_len + wb + 1
            if min_c0s >= w_grid - ws or h_grid - hs <= 2:
                continue
            r0s = int(np.random.randint(1, h_grid - hs))
            c0s = int(np.random.randint(min_c0s, w_grid - ws))
            gap = (r0s + gap_offset, c0s)
            min_r0b = max(1, gap[0] - hb + 1)
            max_r0b = min(gap[0], h_grid - hb - 1)
            if min_r0b > max_r0b:
                continue
            r0b = int(np.random.randint(min_r0b, max_r0b + 1))
            c0b = c0s - lane_len - wb
            block_cell = (gap[0], c0b + wb - 1)
        elif direction_name == "down":
            max_r0 = h_grid - (hs + lane_len + hb)
            if max_r0 <= 1 or w_grid - ws <= 2:
                continue
            r0s = int(np.random.randint(1, max_r0))
            c0s = int(np.random.randint(1, w_grid - ws))
            gap = (r0s + hs - 1, c0s + gap_offset)
            min_c0b = max(1, gap[1] - wb + 1)
            max_c0b = min(gap[1], w_grid - wb - 1)
            if min_c0b > max_c0b:
                continue
            c0b = int(np.random.randint(min_c0b, max_c0b + 1))
            r0b = r0s + hs + lane_len
            block_cell = (r0b, gap[1])
        else:
            min_r0s = lane_len + hb + 1
            if min_r0s >= h_grid - hs or w_grid - ws <= 2:
                continue
            r0s = int(np.random.randint(min_r0s, h_grid - hs))
            c0s = int(np.random.randint(1, w_grid - ws))
            gap = (r0s, c0s + gap_offset)
            min_c0b = max(1, gap[1] - wb + 1)
            max_c0b = min(gap[1], w_grid - wb - 1)
            if min_c0b > max_c0b:
                continue
            c0b = int(np.random.randint(min_c0b, max_c0b + 1))
            r0b = r0s - lane_len - hb
            block_cell = (r0b + hb - 1, gap[1])

        source_bbox = (r0s, c0s, r0s + hs - 1, c0s + ws - 1)
        blocker_bbox = (r0b, c0b, r0b + hb - 1, c0b + wb - 1)
        corridor = _corridor_until_block(gap, (dr, dc), block_cell)
        if len(corridor) < 2:
            continue
        if not _bbox_is_free(reserved, source_bbox):
            continue
        if not _bbox_is_free(reserved, blocker_bbox):
            continue
        if not _cells_are_free(reserved, corridor):
            continue

        payload_count = _pick_payload_count(len(corridor), source_bbox, force_large=require_nontrivial)
        points = _sample_isolated_points(source_bbox, payload_count, gap=gap, require_nontrivial=require_nontrivial)
        if points is None:
            continue

        blocker_payload = None
        blocker_payload_count = 0
        for want in range(3, 0, -1):
            blocker_payload = _sample_isolated_points(blocker_bbox, want)
            if blocker_payload is not None:
                blocker_payload_count = want
                break
        if blocker_payload is None or blocker_payload_count == 0:
            continue

        source_colors = _sample_colors(payload_count)
        blocker_colors = _sample_colors(blocker_payload_count)
        source_payload = [(r, c, source_colors[i]) for i, (r, c) in enumerate(points)]
        blocker_payload_full = [(r, c, blocker_colors[i]) for i, (r, c) in enumerate(blocker_payload)]

        _reserve_bbox(reserved, source_bbox)
        _reserve_bbox(reserved, blocker_bbox)
        _reserve_cells(reserved, corridor)
        return [
            {
                "bbox": source_bbox,
                "gap": gap,
                "direction": (dr, dc),
                "payload": source_payload,
                "corridor": corridor,
                "blocking": "frame",
            },
            {
                "bbox": blocker_bbox,
                "gap": None,
                "direction": None,
                "payload": blocker_payload_full,
                "blocking": None,
            },
        ]
    return None


def _place_standalone_complete(reserved: np.ndarray) -> dict | None:
    h_grid, w_grid = reserved.shape
    for _ in range(140):
        h = int(np.random.randint(4, 8))
        w = int(np.random.randint(4, 8))
        r0 = int(np.random.randint(1, h_grid - h))
        c0 = int(np.random.randint(1, w_grid - w))
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _bbox_is_free(reserved, bbox):
            continue
        payload_count = int(np.random.randint(1, min(4, (h - 2) * (w - 2)) + 1))
        record = _make_complete_frame_record(bbox, payload_count)
        _reserve_bbox(reserved, bbox)
        return record
    return None


def _render_records(shape: tuple[int, int], records: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        gap = record["gap"]
        for r, c in _frame_border_cells(record["bbox"]):
            if gap is not None and (r, c) == gap:
                continue
            grid[r, c] = BLUE
        for r, c, color in record["payload"]:
            grid[r, c] = int(color)
    return grid


def _expected_from_records(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in sorted(records, key=lambda item: item["bbox"]):
        if record["gap"] is None:
            continue
        for r, c, _color in record["payload"]:
            out[r, c] = BLACK
        gap = record["gap"]
        ordered = sorted(
            record["payload"],
            key=lambda item: abs(item[0] - gap[0]) + abs(item[1] - gap[1]),
        )
        cells = list(reversed(record["corridor"]))
        assert len(cells) >= len(ordered)
        for (_r, _c, color), (tr, tc) in zip(ordered, cells):
            out[tr, tc] = int(color)
        out[gap] = BLUE
    return out


def _validate_records(inp: np.ndarray, records: list[dict]) -> bool:
    parsed = _extract_frames(inp)
    if len(parsed) != len(records):
        return False

    parsed_by_bbox = {frame["bbox"]: frame for frame in parsed}
    if set(parsed_by_bbox) != {record["bbox"] for record in records}:
        return False

    seen_edge = False
    seen_blocker = False
    seen_complete_with_payload = False
    seen_nontrivial = False

    for record in records:
        frame = parsed_by_bbox[record["bbox"]]
        if record["gap"] != frame["gap"]:
            return False

        if record["gap"] is None:
            if len(record["payload"]) > 0:
                seen_complete_with_payload = True
            if sorted(record["payload"]) != sorted(frame["payload"]):
                return False
            continue

        if record["blocking"] == "edge":
            seen_edge = True
        if record["blocking"] == "frame":
            seen_blocker = True

        if len(record["corridor"]) < len(record["payload"]):
            return False
        if sorted(record["payload"]) != sorted(frame["payload"]):
            return False
        dists = [abs(r - record["gap"][0]) + abs(c - record["gap"][1]) for r, c, _ in record["payload"]]
        if len(set(dists)) != len(dists):
            return False

        order_by_dist = [pt[:2] for pt in sorted(record["payload"], key=lambda item: abs(item[0] - record["gap"][0]) + abs(item[1] - record["gap"][1]))]
        if order_by_dist != sorted(order_by_dist) and order_by_dist != sorted(order_by_dist, key=lambda rc: (rc[1], rc[0])):
            seen_nontrivial = True

        expected_stack = [color for _r, _c, color in sorted(record["payload"], key=lambda item: abs(item[0] - record["gap"][0]) + abs(item[1] - record["gap"][1]))]
        solved = _expected_from_records(inp, [record])
        farthest_cells = list(reversed(record["corridor"]))[: len(record["payload"])]
        stack_colors = [int(solved[r, c]) for r, c in farthest_cells]
        if stack_colors != expected_stack:
            return False

    return seen_edge and seen_blocker and seen_complete_with_payload and seen_nontrivial


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(22, 29))
    w = int(np.random.randint(22, 29))
    reserved = np.zeros((h, w), dtype=bool)
    records: list[dict] = []

    gapped_count = int(np.random.randint(2, 5))
    directions = list(np.random.permutation(["up", "down", "left", "right"])[:gapped_count])
    blocker_count = 1 if gapped_count == 2 else int(np.random.randint(1, 3))
    blocker_count = min(blocker_count, gapped_count - 1)
    modes = ["blocker"] * blocker_count + ["edge"] * (gapped_count - blocker_count)
    np.random.shuffle(modes)

    nontrivial_used = False
    for idx, (direction_name, mode) in enumerate(zip(directions, modes)):
        require_nontrivial = (idx == 0) or (not nontrivial_used and idx == gapped_count - 1)
        if mode == "edge":
            gadget = _place_edge_gadget(reserved, direction_name, require_nontrivial=require_nontrivial)
        else:
            gadget = _place_blocker_gadget(reserved, direction_name, require_nontrivial=require_nontrivial)
        if gadget is None:
            return None
        if any(record["gap"] is not None for record in gadget):
            nontrivial_used = True
        records.extend(gadget)

    extra_complete_count = int(np.random.randint(1, 3))
    for _ in range(extra_complete_count):
        record = _place_standalone_complete(reserved)
        if record is None:
            return None
        records.append(record)

    inp = _render_records((h, w), records)
    if not _validate_records(inp, records):
        return None

    expected = _expected_from_records(inp, records)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spill-container ARC input")
