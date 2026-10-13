from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
BLOCK_COLORS = [1, 2, 3, 6, 7, 8, 9]

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

SIZE_OPTIONS = {
    "up": [(5, 3), (5, 5), (7, 3), (7, 5)],
    "down": [(5, 3), (5, 5), (7, 3), (7, 5)],
    "left": [(3, 5), (5, 5), (3, 7), (5, 7)],
    "right": [(3, 5), (5, 5), (3, 7), (5, 7)],
}

MAX_GENERATION_ATTEMPTS = 400
MIN_GRID = 18
MAX_GRID = 30
MIN_BG_GAP = 2
MAX_BG_GAP = 5
MIN_FINAL_TRACE = 2
PATH_LENGTH_CHOICES = [1, 2, 3, 4]
PATH_LENGTH_WEIGHTS = [1, 3, 3, 2]
DECOY_COUNT_CHOICES = [0, 1, 2, 3]
DECOY_COUNT_WEIGHTS = [1, 3, 3, 2]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _component_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for coords in _components4(grid != bg):
        bbox = _bbox_from_coords(coords)
        colors = [int(grid[r, c]) for r, c in coords]
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "area": int(coords.shape[0]),
                "colors": set(colors),
            }
        )
    return objects


def _inside_bbox(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _marker_direction(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    dr = marker[0] - cr
    dc = marker[1] - cc
    eps = 1e-9
    if abs(dr) < eps and abs(dc) > eps:
        return "right" if dc > 0 else "left"
    if abs(dc) < eps and abs(dr) > eps:
        return "down" if dr > 0 else "up"
    raise AssertionError("marker offset must be purely axial and off-center")


def _parse_launcher_and_blocks(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], list[dict]]:
    bg = _get_bg(grid)
    comps = _component_objects(grid, bg)
    launcher = []
    blocks = []

    for comp in comps:
        if comp["area"] == 2 and comp["colors"] == {YELLOW, GREY}:
            coords = comp["coords"]
            colors = [int(grid[r, c]) for r, c in coords]
            if colors.count(YELLOW) == 1 and colors.count(GREY) == 1:
                yellow_idx = colors.index(YELLOW)
                grey_idx = colors.index(GREY)
                head = tuple(map(int, coords[yellow_idx]))
                tail = tuple(map(int, coords[grey_idx]))
                if abs(head[0] - tail[0]) + abs(head[1] - tail[1]) == 1:
                    launcher.append((head, tail))
                    continue

        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert comp["area"] == patch.size, "each non-launcher component must be a solid rectangle"
        vals, counts = np.unique(patch, return_counts=True)
        assert len(vals) == 2, "each rectangle must have one body color and one marker color"
        body = int(vals[np.argmax(counts)])
        marker_color = int(vals[np.argmin(counts)])
        assert int(counts.max()) == patch.size - 1 and int(counts.min()) == 1
        marker_local = np.argwhere(patch == marker_color)
        assert marker_local.shape == (1, 2)
        marker = (r0 + int(marker_local[0, 0]), c0 + int(marker_local[0, 1]))
        blocks.append(
            {
                "bbox": comp["bbox"],
                "body": body,
                "marker_color": marker_color,
                "marker": marker,
                "dir": _marker_direction(marker, comp["bbox"]),
            }
        )

    assert len(launcher) == 1, "expected one yellow-head grey-tail launcher"
    head, tail = launcher[0]
    return head, tail, blocks


def _recolor_block(out: np.ndarray, block: dict) -> None:
    r0, c0, r1, c1 = block["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
    mr, mc = block["marker"]
    out[mr, mc] = block["marker_color"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Forward the launcher ray from block to block, repainting each visited rectangle yellow and leaving grey tracks between relays.

    Input structure:
    - A dominant background contains one two-cell launcher made of a yellow head and an adjacent grey tail.
    - The other foreground objects are solid rectangles, each with exactly one off-center marker pixel of a different color.
    - The marker's axial offset from its rectangle center encodes one of the four cardinal forwarding directions.

    Rule:
    - Read the launch direction from tail to head, then march one cell at a time from the head.
    - Every background cell traversed by the ray becomes grey; when the ray first enters a rectangle, repaint that whole rectangle yellow but keep its marker color.
    - Resume from that marker cell in the marker-indicated direction, relay through at most one visit per rectangle, and stop only when the ray leaves the grid.

    Key insight:
    - The marker is a direction relay, not a preserved drawing: the hidden path must be reconstructed segment by segment by chaining the launcher direction with each rectangle's marker offset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    head, tail, blocks = _parse_launcher_and_blocks(g)
    out = g.copy()
    owner = np.full(g.shape, -1, dtype=int)
    for idx, block in enumerate(blocks):
        r0, c0, r1, c1 = block["bbox"]
        owner[r0 : r1 + 1, c0 : c1 + 1] = idx
    owner[tail] = -2
    owner[head] = -2

    direction = (head[0] - tail[0], head[1] - tail[1])
    current = head
    current_block = -1
    visited: set[int] = set()

    while True:
        rr = current[0] + direction[0]
        cc = current[1] + direction[1]
        hit = None

        while _in_bounds(g.shape, rr, cc):
            block_id = int(owner[rr, cc])
            if block_id == -1:
                out[rr, cc] = GREY
            elif block_id == current_block:
                pass
            elif block_id >= 0:
                assert block_id not in visited, "relay path must not revisit rectangles"
                hit = block_id
                break
            else:
                raise AssertionError("relay path must not hit the launcher again")
            rr += direction[0]
            cc += direction[1]

        if hit is None:
            return out

        block = blocks[hit]
        _recolor_block(out, block)
        visited.add(hit)
        current = block["marker"]
        direction = DIRS[block["dir"]]
        current_block = hit


def _step(coord: tuple[int, int], direction: str, amount: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return coord[0] + dr * amount, coord[1] + dc * amount


def _first_outside(start: tuple[int, int], direction: str, bbox: tuple[int, int, int, int] | None) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    rr, cc = start
    if bbox is None:
        return rr + dr, cc + dc
    while True:
        rr += dr
        cc += dc
        if not _inside_bbox(bbox, rr, cc):
            return rr, cc


def _offset_options(direction: str, h: int, w: int) -> list[int]:
    axis_half = h // 2 if direction in ("up", "down") else w // 2
    return list(range(1, axis_half))


def _marker_from_center(center: tuple[int, int], direction: str, offset: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return center[0] + dr * offset, center[1] + dc * offset


def _halo_ok(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    return not bool(np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _path_cells_before_block(
    start: tuple[int, int],
    direction: str,
    current_bbox: tuple[int, int, int, int] | None,
    gap: int,
) -> list[tuple[int, int]]:
    front = _first_outside(start, direction, current_bbox)
    return [_step(front, direction, k) for k in range(gap)]


def _candidate_blocks(
    shape: tuple[int, int],
    occupied: np.ndarray,
    path_mask: np.ndarray,
    start: tuple[int, int],
    incoming_dir: str,
    current_bbox: tuple[int, int, int, int] | None,
    outgoing_dir: str,
) -> list[dict]:
    h_grid, w_grid = shape
    front = _first_outside(start, incoming_dir, current_bbox)
    if not _in_bounds(shape, *front):
        return []

    candidates = []
    size_options = SIZE_OPTIONS[outgoing_dir][:]
    random.shuffle(size_options)
    gaps = list(range(MIN_BG_GAP, MAX_BG_GAP + 1))
    random.shuffle(gaps)

    for h, w in size_options:
        offsets = _offset_options(outgoing_dir, h, w)
        random.shuffle(offsets)

        if incoming_dir in ("left", "right"):
            row = start[0]
            center_rows = list(
                range(
                    max(h // 2, row - h // 2),
                    min(h_grid - 1 - h // 2, row + h // 2) + 1,
                )
            )
            random.shuffle(center_rows)
            for gap in gaps:
                cells = _path_cells_before_block(start, incoming_dir, current_bbox, gap)
                if not cells or not all(_in_bounds(shape, r, c) for r, c in cells):
                    continue
                if any(occupied[r, c] or path_mask[r, c] for r, c in cells):
                    continue

                if incoming_dir == "right":
                    c0 = front[1] + gap
                    c1 = c0 + w - 1
                else:
                    c1 = front[1] - gap
                    c0 = c1 - w + 1
                if c0 < 0 or c1 >= w_grid:
                    continue
                center_c = c0 + w // 2

                for center_r in center_rows:
                    bbox = (center_r - h // 2, c0, center_r + h // 2, c1)
                    if not _halo_ok(occupied, bbox):
                        continue
                    if np.any(path_mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]):
                        continue
                    for offset in offsets:
                        marker = _marker_from_center((center_r, center_c), outgoing_dir, offset)
                        candidates.append(
                            {
                                "bbox": bbox,
                                "marker": marker,
                                "dir": outgoing_dir,
                                "segment": cells,
                            }
                        )
        else:
            col = start[1]
            center_cols = list(
                range(
                    max(w // 2, col - w // 2),
                    min(w_grid - 1 - w // 2, col + w // 2) + 1,
                )
            )
            random.shuffle(center_cols)
            for gap in gaps:
                cells = _path_cells_before_block(start, incoming_dir, current_bbox, gap)
                if not cells or not all(_in_bounds(shape, r, c) for r, c in cells):
                    continue
                if any(occupied[r, c] or path_mask[r, c] for r, c in cells):
                    continue

                if incoming_dir == "down":
                    r0 = front[0] + gap
                    r1 = r0 + h - 1
                else:
                    r1 = front[0] - gap
                    r0 = r1 - h + 1
                if r0 < 0 or r1 >= h_grid:
                    continue
                center_r = r0 + h // 2

                for center_c in center_cols:
                    bbox = (r0, center_c - w // 2, r1, center_c + w // 2)
                    if not _halo_ok(occupied, bbox):
                        continue
                    if np.any(path_mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]):
                        continue
                    for offset in offsets:
                        marker = _marker_from_center((center_r, center_c), outgoing_dir, offset)
                        candidates.append(
                            {
                                "bbox": bbox,
                                "marker": marker,
                                "dir": outgoing_dir,
                                "segment": cells,
                            }
                        )

    random.shuffle(candidates)
    return candidates


def _exit_trace(
    shape: tuple[int, int],
    occupied: np.ndarray,
    path_mask: np.ndarray,
    start: tuple[int, int],
    direction: str,
    current_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    rr, cc = _first_outside(start, direction, current_bbox)
    cells = []
    while _in_bounds(shape, rr, cc):
        if occupied[rr, cc] or path_mask[rr, cc]:
            return None
        cells.append((rr, cc))
        rr, cc = _step((rr, cc), direction)
    return cells if len(cells) >= MIN_FINAL_TRACE else None


def _paint_input_rect(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    body: int,
    marker: tuple[int, int],
    marker_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = body
    grid[marker] = marker_color


def _render_instance(shape: tuple[int, int], head: tuple[int, int], tail: tuple[int, int], blocks: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[tail] = GREY
    grid[head] = YELLOW
    for block in blocks:
        _paint_input_rect(grid, block["bbox"], block["body"], block["marker"], block["marker_color"])
    return grid


def _oracle_output(shape: tuple[int, int], head: tuple[int, int], tail: tuple[int, int], blocks: list[dict], path_cells: list[tuple[int, int]], visited_ids: list[int]) -> np.ndarray:
    out = _render_instance(shape, head, tail, blocks).copy()
    for r, c in path_cells:
        out[r, c] = GREY
    for idx in visited_ids:
        _recolor_block(out, blocks[idx])
    return out


def _place_decoys(
    shape: tuple[int, int],
    occupied: np.ndarray,
    path_mask: np.ndarray,
    count: int,
) -> list[dict] | None:
    placed = []
    attempts = 0
    while len(placed) < count and attempts < 200:
        attempts += 1
        direction = random.choice(list(DIRS))
        h, w = random.choice(SIZE_OPTIONS[direction])
        offset = random.choice(_offset_options(direction, h, w))
        center_r = random.randint(h // 2, shape[0] - 1 - h // 2)
        center_c = random.randint(w // 2, shape[1] - 1 - w // 2)
        bbox = (center_r - h // 2, center_c - w // 2, center_r + h // 2, center_c + w // 2)
        if not _halo_ok(occupied, bbox):
            continue
        if np.any(path_mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]):
            continue
        marker = _marker_from_center((center_r, center_c), direction, offset)
        placed.append({"bbox": bbox, "marker": marker, "dir": direction})
        occupied[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = True
    return placed if len(placed) == count else None


def _build_path_instance(shape: tuple[int, int], path_len: int) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    path_mask = np.zeros(shape, dtype=bool)

    launch_dir = random.choice(list(DIRS))
    launch_dr, launch_dc = DIRS[launch_dir]

    valid_heads = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            tail = (r - launch_dr, c - launch_dc)
            if _in_bounds(shape, *tail):
                valid_heads.append((r, c))
    if not valid_heads:
        return None
    head = random.choice(valid_heads)
    tail = (head[0] - launch_dr, head[1] - launch_dc)
    occupied[head] = True
    occupied[tail] = True

    blocks: list[dict] = []
    visited_ids: list[int] = []
    all_path_cells: list[tuple[int, int]] = []

    def backtrack(step_idx: int, start: tuple[int, int], incoming_dir: str, current_bbox: tuple[int, int, int, int] | None) -> bool:
        outgoing_choices = [d for d in DIRS if d != OPPOSITE[incoming_dir]]
        random.shuffle(outgoing_choices)

        for outgoing_dir in outgoing_choices:
            for cand in _candidate_blocks(shape, occupied, path_mask, start, incoming_dir, current_bbox, outgoing_dir):
                bbox = cand["bbox"]
                seg = cand["segment"]
                path_len_before = len(all_path_cells)
                occupied[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = True
                for r, c in seg:
                    path_mask[r, c] = True
                block = {"bbox": bbox, "marker": cand["marker"], "dir": outgoing_dir}
                blocks.append(block)
                visited_ids.append(len(blocks) - 1)
                all_path_cells.extend(seg)

                if step_idx == path_len - 1:
                    exit_cells = _exit_trace(shape, occupied, path_mask, block["marker"], outgoing_dir, bbox)
                    if exit_cells is not None:
                        for r, c in exit_cells:
                            path_mask[r, c] = True
                        all_path_cells.extend(exit_cells)
                        return True
                else:
                    if backtrack(step_idx + 1, block["marker"], outgoing_dir, bbox):
                        return True

                while len(all_path_cells) > path_len_before:
                    r, c = all_path_cells.pop()
                    path_mask[r, c] = False
                visited_ids.pop()
                blocks.pop()
                occupied[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = False
        return False

    if not backtrack(0, head, launch_dir, None):
        return None

    decoy_count = random.choices(DECOY_COUNT_CHOICES, weights=DECOY_COUNT_WEIGHTS, k=1)[0]
    decoys = _place_decoys(shape, occupied.copy(), path_mask.copy(), decoy_count)
    if decoys is None:
        return None

    all_blocks = blocks + decoys
    body_pool = BLOCK_COLORS[:]
    random.shuffle(body_pool)
    if len(all_blocks) > len(body_pool):
        return None
    marker_choices = BLOCK_COLORS[:]

    for idx, block in enumerate(all_blocks):
        body = body_pool[idx]
        marker_color = random.choice([c for c in marker_choices if c != body])
        block["body"] = body
        block["marker_color"] = marker_color

    inp = _render_instance(shape, head, tail, all_blocks)
    expected = _oracle_output(shape, head, tail, all_blocks, all_path_cells, visited_ids)
    return inp, expected


def _try_generate() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    path_len = random.choices(PATH_LENGTH_CHOICES, weights=PATH_LENGTH_WEIGHTS, k=1)[0]
    built = _build_path_instance((h, w), path_len)
    if built is None:
        return None
    inp, expected = built

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    _, _, blocks = _parse_launcher_and_blocks(inp)
    if len(blocks) < path_len:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-forwarding input")
