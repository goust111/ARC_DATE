from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}
SIDE_MID = {
    (-1, 0): (0, 1),
    (0, 1): (1, 2),
    (1, 0): (2, 1),
    (0, -1): (1, 0),
}
TOKEN_TOP_LEFT_OFFSET = {
    (-1, 0): (1, -1),
    (0, 1): (-1, -3),
    (1, 0): (-3, -1),
    (0, -1): (-1, 1),
}
GEN_CALLS = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "cell_set": set(coords),
                    "bbox": bbox,
                    "mask": _component_mask(coords, bbox),
                }
            )
    return comps


def _is_rot_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not cells:
        return False
    target = set(cells)
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen == target


def _graph_neighbors(cells: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    return {cell: [nbr for nbr in _neighbors4(*cell) if nbr in cells] for cell in cells}


def _token_top_left(endpoint: tuple[int, int], heading: tuple[int, int]) -> tuple[int, int]:
    dr, dc = TOKEN_TOP_LEFT_OFFSET[heading]
    return endpoint[0] + dr, endpoint[1] + dc


def _mask_cells(mask: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((int(r0 + r), int(c0 + c)))
    return cells


def _adjacent_pairs(
    cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = set()
    for r, c in cells_a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells_b:
                pairs.add(((r, c), (nr, nc)))
    return pairs


def _rotation_from_dirs(dirs: list[tuple[int, int]]) -> int:
    if not dirs:
        return 0
    turn_sum = 0
    for prev, cur in zip(dirs, dirs[1:]):
        delta = (DIR_TO_INDEX[cur] - DIR_TO_INDEX[prev]) % 4
        if delta == 0:
            continue
        assert delta in (1, 3), "wire must bend by right angles without U-turns"
        turn_sum += 1 if delta == 1 else -1
    return turn_sum % 4


def _ordered_path(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]], start: tuple[int, int]
) -> list[tuple[int, int]]:
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [nbr for nbr in neighbors[cur] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "wire should be a simple path"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _mask_cell_set(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.argwhere(mask)}


def _parse_instances(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    token_ids = set()
    tokens = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if not _is_rot_asymmetric(comp["mask"]):
            continue
        if not any(comp["mask"][cell] for cell in SIDE_MID.values()):
            continue
        if _is_simple_path_cells(comp["cell_set"]):
            continue
        token_ids.add(idx)
        tokens.append(
            {
                "id": idx,
                "color": comp["color"],
                "bbox": comp["bbox"],
                "mask": comp["mask"],
                "cell_set": comp["cell_set"],
            }
        )

    wires = []
    endpoint_to_wire: dict[tuple[int, int], dict] = {}
    for idx, comp in enumerate(comps):
        if idx in token_ids:
            continue
        neighbors = _graph_neighbors(comp["cell_set"])
        degrees = {cell: len(nbrs) for cell, nbrs in neighbors.items()}
        if len(comp["coords"]) < 2 or any(deg > 2 for deg in degrees.values()):
            continue
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        if len(endpoints) != 2:
            continue
        info = {
            "id": idx,
            "color": comp["color"],
            "cell_set": comp["cell_set"],
            "neighbors": neighbors,
            "endpoints": endpoints,
            "endpoint_heading": {
                ep: (neighbors[ep][0][0] - ep[0], neighbors[ep][0][1] - ep[1]) for ep in endpoints
            },
        }
        wires.append(info)
        for ep in endpoints:
            endpoint_to_wire[ep] = info

    instances = []
    used_wires = set()
    for token in tokens:
        r0, c0, _r1, _c1 = token["bbox"]
        matches = []
        for heading, (lr, lc) in SIDE_MID.items():
            if not token["mask"][lr, lc]:
                continue
            endpoint = (r0 + lr + heading[0], c0 + lc + heading[1])
            wire = endpoint_to_wire.get(endpoint)
            if wire is None:
                continue
            if wire["endpoint_heading"][endpoint] != heading:
                continue
            matches.append((wire, endpoint, heading))
        assert len(matches) == 1, "expected each token to attach to exactly one wire endpoint"
        wire, start_endpoint, start_heading = matches[0]
        assert wire["id"] not in used_wires, "each wire should have exactly one token"
        used_wires.add(wire["id"])
        path = _ordered_path(wire["neighbors"], start_endpoint)
        assert len(path) == len(wire["cell_set"]), "ordered path should cover the full wire"
        dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        assert dirs and dirs[0] == start_heading
        instances.append(
            {
                "wire_color": wire["color"],
                "token_color": token["color"],
                "token_bbox": token["bbox"],
                "token_mask": token["mask"],
                "wire_path": path,
                "dirs": dirs,
            }
        )

    assert len(instances) == len(wires) == len(tokens), "expected a one-to-one token/wire pairing"
    return instances


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each 3x3 token to the far end of its wire after rotating it by the wire's net turn.

    Input structure:
    - Black is the background.
    - Several non-touching single-cell-wide monochrome wires run orthogonally with right-angle bends.
    - Each wire has one attached 3x3 monochrome asymmetric token at one endpoint; the token connects through the middle cell of one side.

    Rule:
    - For each wire, walk from the token endpoint to the free endpoint and accumulate bends: every right turn adds a clockwise quarter-turn and every left turn subtracts one.
    - Keep the original wire and original token unchanged.
    - Place a copy of the token at the other endpoint, rotated by the accumulated net turn and recolored to the wire's color, with the endpoint touching the copy's attached side-middle cell.

    Key insight:
    - The wire is both a transport path and a rotation program: its ordered bend sequence determines the orientation of the transmitted token.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for inst in _parse_instances(g):
        rot = _rotation_from_dirs(inst["dirs"])
        end_heading = inst["dirs"][-1]
        output_attach = (-end_heading[0], -end_heading[1])
        rotated_mask = np.rot90(inst["token_mask"], -rot)
        assert rotated_mask[SIDE_MID[output_attach]], "rotated token should still attach through a side middle"
        top_left = _token_top_left(inst["wire_path"][-1], output_attach)
        copy_cells = _mask_cells(rotated_mask, top_left)
        for r, c in copy_cells:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "output token must stay in bounds"
            assert int(out[r, c]) == bg, "copied token should land on empty background"
            out[r, c] = int(inst["wire_color"])

    return out


def _build_token_library() -> list[np.ndarray]:
    masks = []
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if not mask[1, 0]:
            continue
        count = int(mask.sum())
        if count < 4 or count > 6:
            continue
        if not mask[0, :].any() or not mask[2, :].any() or not mask[:, 2].any():
            continue
        if not _is_connected(mask) or not _is_rot_asymmetric(mask):
            continue
        if _is_simple_path_cells(_mask_cell_set(mask)):
            continue
        masks.append(mask)
    assert masks, "token library should not be empty"
    return masks

def _choose_turns(target_rot: int, *, prefer_bent_zero: bool) -> list[int]:
    if target_rot == 0:
        base_choices = [[1, -1], [-1, 1]] if prefer_bent_zero else [[], [1, -1], [-1, 1]]
    elif target_rot == 1:
        base_choices = [[1], [-1, -1, -1]]
    elif target_rot == 2:
        base_choices = [[1, 1], [-1, -1]]
    else:
        base_choices = [[-1], [1, 1, 1]]

    turns = list(random.choice(base_choices))
    if len(turns) <= 2 and random.random() < 0.5:
        pair = list(random.choice(([1, -1], [-1, 1])))
        pos = random.randint(0, len(turns))
        turns[pos:pos] = pair
    if len(turns) > 4:
        turns = turns[:4]
    assert sum(turns) % 4 == target_rot
    return turns


def _build_path(start_idx: int, turns: list[int], lengths: list[int]) -> list[tuple[int, int]]:
    path = [(0, 0)]
    dir_idx = int(start_idx)
    for seg_idx, steps in enumerate(lengths):
        dr, dc = DIRS[dir_idx]
        for _ in range(int(steps)):
            r, c = path[-1]
            path.append((r + dr, c + dc))
        if seg_idx < len(turns):
            dir_idx = (dir_idx + int(turns[seg_idx])) % 4
    return path


def _is_simple_path_cells(cells: set[tuple[int, int]]) -> bool:
    neighbors = _graph_neighbors(cells)
    degrees = [len(nbrs) for nbrs in neighbors.values()]
    return len(cells) >= 2 and all(deg <= 2 for deg in degrees) and degrees.count(1) == 2


def _only_attachment(
    token_cells: set[tuple[int, int]],
    wire_cells: set[tuple[int, int]],
    attach_cell: tuple[int, int],
    endpoint: tuple[int, int],
) -> bool:
    pairs = _adjacent_pairs(token_cells, wire_cells)
    return pairs == {(attach_cell, endpoint)}


TOKEN_LIBRARY = _build_token_library()


def _make_local_object(target_rot: int, *, prefer_bent_zero: bool) -> dict | None:
    for _ in range(300):
        start_idx = random.randrange(4)
        turns = _choose_turns(target_rot, prefer_bent_zero=prefer_bent_zero)
        lengths = [random.randint(2, 5) for _ in range(len(turns) + 1)]
        path = _build_path(start_idx, turns, lengths)
        path_set = set(path)
        if len(path) != len(path_set):
            continue
        if not _is_simple_path_cells(path_set):
            continue

        start_heading = DIRS[start_idx]
        end_heading = DIRS[(start_idx + sum(turns)) % 4]

        token_mask = np.rot90(
            random.choice(TOKEN_LIBRARY),
            -(start_idx - DIR_TO_INDEX[(0, -1)]) % 4,
        )
        token_top = _token_top_left(path[0], start_heading)
        token_cells = _mask_cells(token_mask, token_top)
        attach_start = (
            token_top[0] + SIDE_MID[start_heading][0],
            token_top[1] + SIDE_MID[start_heading][1],
        )
        if token_cells & path_set:
            continue
        if not _only_attachment(token_cells, path_set, attach_start, path[0]):
            continue

        rot = _rotation_from_dirs([(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])])
        if rot != target_rot:
            continue
        rotated_mask = np.rot90(token_mask, -rot)
        output_attach = (-end_heading[0], -end_heading[1])
        if not rotated_mask[SIDE_MID[output_attach]]:
            continue
        copy_top = _token_top_left(path[-1], output_attach)
        copy_cells = _mask_cells(rotated_mask, copy_top)
        attach_end = (
            copy_top[0] + SIDE_MID[output_attach][0],
            copy_top[1] + SIDE_MID[output_attach][1],
        )
        if path_set & copy_cells or token_cells & copy_cells:
            continue
        if not _only_attachment(copy_cells, path_set, attach_end, path[-1]):
            continue

        reserve = path_set | token_cells | copy_cells
        return {
            "wire_path_local": path,
            "token_mask": token_mask,
            "token_top_local": token_top,
            "copy_mask": rotated_mask,
            "copy_top_local": copy_top,
            "reserve_local": reserve,
        }
    return None


def _translate_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _translate_path(path: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in path]


def _cells_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _can_place(reserve_cells: set[tuple[int, int]], blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    for r, c in reserve_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(reserve_cells: set[tuple[int, int]], blocked: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in reserve_cells:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        blocked[r0:r1, c0:c1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    for r, c in np.argwhere(mask):
        grid[r0 + int(r), c0 + int(c)] = int(color)


def _render_input(shape: tuple[int, int], specs: list[dict], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        for r, c in spec["wire_path"]:
            grid[r, c] = int(spec["wire_color"])
        _paint_mask(grid, spec["token_mask"], spec["token_top_left"], spec["token_color"])
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict], bg: int) -> np.ndarray:
    out = _render_input(shape, specs, bg)
    for spec in specs:
        _paint_mask(out, spec["copy_mask"], spec["copy_top_left"], spec["wire_color"])
    return out


def _validate_specs(shape: tuple[int, int], specs: list[dict], bg: int) -> None:
    h, w = shape
    seen_input: set[tuple[int, int]] = set()
    seen_reserve: set[tuple[int, int]] = set()

    for spec in specs:
        path = spec["wire_path"]
        path_set = set(path)
        assert _is_simple_path_cells(path_set)
        assert len(path_set) == len(path)

        dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        assert dirs == spec["dirs"]
        start_heading = dirs[0]
        end_heading = dirs[-1]
        output_attach = (-end_heading[0], -end_heading[1])
        token_mask = spec["token_mask"]
        copy_mask = spec["copy_mask"]
        assert token_mask.shape == (3, 3) and copy_mask.shape == (3, 3)
        assert _is_rot_asymmetric(token_mask)
        assert token_mask[SIDE_MID[start_heading]]
        assert copy_mask[SIDE_MID[output_attach]]

        token_cells = _mask_cells(token_mask, spec["token_top_left"])
        copy_cells = _mask_cells(copy_mask, spec["copy_top_left"])
        attach_start = (
            spec["token_top_left"][0] + SIDE_MID[start_heading][0],
            spec["token_top_left"][1] + SIDE_MID[start_heading][1],
        )
        attach_end = (
            spec["copy_top_left"][0] + SIDE_MID[output_attach][0],
            spec["copy_top_left"][1] + SIDE_MID[output_attach][1],
        )
        assert _only_attachment(token_cells, path_set, attach_start, path[0])
        assert _only_attachment(copy_cells, path_set, attach_end, path[-1])

        input_cells = path_set | token_cells
        reserve = input_cells | copy_cells
        for r, c in reserve:
            assert 0 <= r < h and 0 <= c < w
        assert not (seen_input & input_cells)
        assert not (seen_reserve & reserve)
        seen_input |= input_cells
        seen_reserve |= reserve

    grid = _render_input(shape, specs, bg)
    assert max(grid.shape) <= 30
    for spec in specs:
        for r, c in _mask_cells(spec["copy_mask"], spec["copy_top_left"]):
            assert int(grid[r, c]) == bg, "copy destination should be empty in the input"


def _place_objects_on_canvas(local_objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    wire_colors = random.sample(range(1, 10), len(local_objects))
    remaining = [c for c in range(1, 10) if c not in wire_colors]
    token_colors = random.sample(remaining, len(local_objects))

    for local, wire_color, token_color in zip(local_objects, wire_colors, token_colors):
        r0, c0, r1, c1 = _cells_bbox(local["reserve_local"])
        row_lo, row_hi = -r0, h - 1 - r1
        col_lo, col_hi = -c0, w - 1 - c1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placements = [
            (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
            for _ in range(120)
        ]
        placements.extend(
            [
                (row_lo, col_lo),
                (row_lo, col_hi),
                (row_hi, col_lo),
                (row_hi, col_hi),
            ]
        )

        placed = False
        for dr, dc in placements:
            reserve = _translate_cells(local["reserve_local"], dr, dc)
            if not _can_place(reserve, blocked):
                continue
            wire_path = _translate_path(local["wire_path_local"], dr, dc)
            token_top = (local["token_top_local"][0] + dr, local["token_top_local"][1] + dc)
            copy_top = (local["copy_top_local"][0] + dr, local["copy_top_local"][1] + dc)
            dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(wire_path, wire_path[1:])]
            specs.append(
                {
                    "wire_color": int(wire_color),
                    "token_color": int(token_color),
                    "wire_path": wire_path,
                    "dirs": dirs,
                    "token_mask": local["token_mask"],
                    "token_top_left": token_top,
                    "copy_mask": local["copy_mask"],
                    "copy_top_left": copy_top,
                }
            )
            _mark_blocked(reserve, blocked)
            placed = True
            break
        if not placed:
            return None

    return specs


def _attempt_generate(call_id: int) -> np.ndarray | None:
    count_cycle = [2, 3, 4, 1]
    count = count_cycle[call_id % len(count_cycle)] if random.random() < 0.65 else random.choice([1, 2, 3, 4])
    forced_rot = call_id % 4
    prefer_bent_zero = (call_id // 4) % 2 == 1

    local_objects = []
    for obj_idx in range(count):
        target_rot = forced_rot if obj_idx == 0 else random.randrange(4)
        local = _make_local_object(target_rot, prefer_bent_zero=prefer_bent_zero and obj_idx == 0)
        if local is None:
            return None
        local_objects.append(local)

    min_side = {1: 14, 2: 18, 3: 22, 4: 24}[count]
    shape = (random.randint(min_side, 30), random.randint(min_side, 30))
    specs = _place_objects_on_canvas(local_objects, shape)
    if specs is None:
        return None

    bg = 0
    _validate_specs(shape, specs, bg)
    inp = _render_input(shape, specs, bg)
    expected = _render_expected(shape, specs, bg)
    assert np.array_equal(main(inp), expected), "solver should match the generator spec"
    return inp


def generate_input() -> np.ndarray:
    global GEN_CALLS
    call_id = GEN_CALLS
    GEN_CALLS += 1

    for _ in range(500):
        inp = _attempt_generate(call_id)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid wire-rotation transport instance")
