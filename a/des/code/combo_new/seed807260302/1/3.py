from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7)

UP = "U"
DOWN = "D"
LEFT = "L"
RIGHT = "R"
DIRS = (UP, DOWN, LEFT, RIGHT)
STEP = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

SHAPES = {
    RIGHT: frozenset({(0, 0), (1, 0), (0, 1)}),
    DOWN: frozenset({(0, 0), (0, 1), (1, 1)}),
    LEFT: frozenset({(0, 1), (1, 0), (1, 1)}),
    UP: frozenset({(0, 0), (1, 0), (1, 1)}),
}
DIR_BY_SHAPE = {shape: direction for direction, shape in SHAPES.items()}
HEAD_REL = {RIGHT: (0, 1), DOWN: (1, 1), LEFT: (1, 0), UP: (0, 0)}


def _cells_at(r0: int, c0: int, direction: str) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr, dc in SHAPES[direction]}


def _bbox_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}


def _head_at(r0: int, c0: int, direction: str) -> tuple[int, int]:
    dr, dc = HEAD_REL[direction]
    return r0 + dr, c0 + dc


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == BG:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _decode_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(grid):
        coords = comp["coords"]
        assert len(coords) == 3, "each arrow must contain exactly three pixels"
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 == 1 and c1 - c0 == 1, "each arrow must fit a 2x2 L shape"
        rel = frozenset((r - r0, c - c0) for r, c in coords)
        assert rel in DIR_BY_SHAPE, "unknown arrow orientation"
        arrows.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "color": int(comp["color"]),
                "dir": DIR_BY_SHAPE[rel],
            }
        )
    return arrows


def _occupancy(arrows: list[dict]) -> dict[tuple[int, int], int]:
    occ: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in _cells_at(arrow["r0"], arrow["c0"], arrow["dir"]):
            occ[cell] = idx
    return occ


def _flip_arrow(out: np.ndarray, arrows: list[dict], idx: int, occ: dict[tuple[int, int], int]) -> None:
    arrow = arrows[idx]
    old_dir = arrow["dir"]
    old_cells = _cells_at(arrow["r0"], arrow["c0"], old_dir)
    for r, c in old_cells:
        out[r, c] = BG
        occ.pop((r, c), None)

    new_dir = OPPOSITE[old_dir]
    arrow["dir"] = new_dir
    for r, c in _cells_at(arrow["r0"], arrow["c0"], new_dir):
        out[r, c] = arrow["color"]
        occ[(r, c)] = idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw chained arrow rays, flipping each newly hit arrow before it fires.

    Input structure:
    - The background is black.
    - Each foreground object is a unique-color three-pixel L-shaped arrow in a
      2x2 box.
    - The missing corner of the L shape encodes one of the four cardinal arrow
      directions.

    Rule:
    - Choose the arrow with the smallest top row, then smallest left column, as
      the first active arrow.
    - From the active arrow head, draw its color one cell at a time in its
      direction until the ray leaves the grid or reaches another arrow.
    - A hit stops before the hit cell; if the hit arrow has not flipped before,
      rotate it to the opposite direction and continue the chain from its new
      head.
    - Stop when a ray leaves the grid or hits an already flipped arrow; arrows
      never hit by the chain keep their original shape.

    Key insight:
    - The L triomino shape is both the object and the direction code; the chain
      order comes from ray hits after the top-left arrow starts the reaction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    arrows = _decode_arrows(g)
    if not arrows:
        return out

    active = min(range(len(arrows)), key=lambda i: (arrows[i]["r0"], arrows[i]["c0"]))
    flipped: set[int] = set()
    occ = _occupancy(arrows)
    h, w = g.shape

    while True:
        arrow = arrows[active]
        dr, dc = STEP[arrow["dir"]]
        r, c = _head_at(arrow["r0"], arrow["c0"], arrow["dir"])
        r += dr
        c += dc

        hit: int | None = None
        while 0 <= r < h and 0 <= c < w:
            owner = occ.get((r, c))
            if owner is not None:
                hit = owner
                break
            out[r, c] = arrow["color"]
            r += dr
            c += dc

        if hit is None or hit in flipped:
            break

        flipped.add(hit)
        _flip_arrow(out, arrows, hit, occ)
        active = hit

    return out


def _cast(
    head: tuple[int, int],
    direction: str,
    occ: dict[tuple[int, int], int],
    h: int,
    w: int,
) -> tuple[int | None, list[tuple[int, int]], tuple[int, int] | None]:
    dr, dc = STEP[direction]
    r, c = head[0] + dr, head[1] + dc
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        owner = occ.get((r, c))
        if owner is not None:
            return owner, path, (r, c)
        path.append((r, c))
        r += dr
        c += dc
    return None, path, None


def _ray_cells_before(
    head: tuple[int, int], direction: str, target: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = STEP[direction]
    r, c = head[0] + dr, head[1] + dc
    cells: list[tuple[int, int]] = []
    while (r, c) != target:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _near_side_rels(direction: str, incoming: str) -> list[tuple[int, int]]:
    if incoming == RIGHT:
        rels = [rel for rel in SHAPES[direction] if rel[1] == 0]
    elif incoming == LEFT:
        rels = [rel for rel in SHAPES[direction] if rel[1] == 1]
    elif incoming == DOWN:
        rels = [rel for rel in SHAPES[direction] if rel[0] == 0]
    else:
        rels = [rel for rel in SHAPES[direction] if rel[0] == 1]
    random.shuffle(rels)
    return rels


def _lex_after_initial(r0: int, c0: int, initial: tuple[int, int]) -> bool:
    return (r0, c0) > initial


def _in_bounds_bbox(r0: int, c0: int, h: int, w: int) -> bool:
    return 0 <= r0 and 0 <= c0 and r0 + 1 < h and c0 + 1 < w


def _touches_existing(cells: set[tuple[int, int]], occ: dict[tuple[int, int], int]) -> bool:
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occ:
                    return True
    return False


def _can_place(
    r0: int,
    c0: int,
    original_dir: str,
    flipped_dir: str,
    h: int,
    w: int,
    initial: tuple[int, int],
    input_occ: dict[tuple[int, int], int],
    dyn_occ: dict[tuple[int, int], int],
    ray_cells: set[tuple[int, int]],
) -> bool:
    if not _in_bounds_bbox(r0, c0, h, w):
        return False
    if not _lex_after_initial(r0, c0, initial):
        return False
    bbox = _bbox_cells(r0, c0)
    if bbox & ray_cells:
        return False

    original_cells = _cells_at(r0, c0, original_dir)
    flipped_cells = _cells_at(r0, c0, flipped_dir)
    if _touches_existing(original_cells, input_occ):
        return False
    if original_cells & dyn_occ.keys() or flipped_cells & dyn_occ.keys():
        return False
    return True


def _add_to_occ(occ: dict[tuple[int, int], int], cells: set[tuple[int, int]], idx: int) -> None:
    for cell in cells:
        occ[cell] = idx


def _candidate_targets(
    head: tuple[int, int],
    incoming: str,
    existing_dyn: dict[tuple[int, int], int],
    h: int,
    w: int,
) -> list[tuple[tuple[int, int], int]]:
    dr, dc = STEP[incoming]
    r, c = head[0] + dr, head[1] + dc
    dist = 1
    points: list[tuple[tuple[int, int], int]] = []
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in existing_dyn:
            break
        if dist >= 4:
            points.append(((r, c), dist))
        r += dr
        c += dc
        dist += 1
    random.shuffle(points)
    return points


def _place_chain_arrow(
    records: list[dict],
    current_head: tuple[int, int],
    incoming: str,
    h: int,
    w: int,
    initial: tuple[int, int],
    input_occ: dict[tuple[int, int], int],
    dyn_occ: dict[tuple[int, int], int],
    ray_cells: set[tuple[int, int]],
    flipped_ids: set[int],
    final_mode: str | None,
) -> tuple[tuple[int, int], str] | None:
    next_dirs = list(DIRS)
    random.shuffle(next_dirs)
    targets = _candidate_targets(current_head, incoming, dyn_occ, h, w)

    for target, _dist in targets:
        for next_dir in next_dirs:
            original_dir = OPPOSITE[next_dir]
            for rr, cc in _near_side_rels(original_dir, incoming):
                r0, c0 = target[0] - rr, target[1] - cc
                if not _can_place(
                    r0,
                    c0,
                    original_dir,
                    next_dir,
                    h,
                    w,
                    initial,
                    input_occ,
                    dyn_occ,
                    ray_cells,
                ):
                    continue

                idx = len(records)
                next_head = _head_at(r0, c0, next_dir)
                original_cells = _cells_at(r0, c0, original_dir)
                flipped_cells = _cells_at(r0, c0, next_dir)

                test_dyn = dict(dyn_occ)
                _add_to_occ(test_dyn, flipped_cells, idx)
                if final_mode is not None:
                    owner, _path, _hit_cell = _cast(next_head, next_dir, test_dyn, h, w)
                    if final_mode == "boundary" and owner is not None:
                        continue
                    if final_mode == "repeat" and owner not in flipped_ids:
                        continue

                records.append({"r0": r0, "c0": c0, "dir": original_dir})
                _add_to_occ(input_occ, original_cells, idx)
                _add_to_occ(dyn_occ, flipped_cells, idx)
                ray_cells.update(_ray_cells_before(current_head, incoming, target))
                flipped_ids.add(idx)
                return next_head, next_dir
    return None


def _place_distractor(
    records: list[dict],
    h: int,
    w: int,
    initial: tuple[int, int],
    input_occ: dict[tuple[int, int], int],
    dyn_occ: dict[tuple[int, int], int],
    ray_cells: set[tuple[int, int]],
) -> bool:
    for _ in range(200):
        direction = random.choice(DIRS)
        r0 = random.randint(0, h - 2)
        c0 = random.randint(0, w - 2)
        if not _can_place(
            r0,
            c0,
            direction,
            direction,
            h,
            w,
            initial,
            input_occ,
            dyn_occ,
            ray_cells,
        ):
            continue
        idx = len(records)
        records.append({"r0": r0, "c0": c0, "dir": direction})
        cells = _cells_at(r0, c0, direction)
        _add_to_occ(input_occ, cells, idx)
        _add_to_occ(dyn_occ, cells, idx)
        return True
    return False


def _render(records: list[dict], h: int, w: int) -> np.ndarray:
    colors = list(PALETTE)
    random.shuffle(colors)
    grid = np.zeros((h, w), dtype=int)
    for idx, rec in enumerate(records):
        color = colors[idx]
        for r, c in _cells_at(rec["r0"], rec["c0"], rec["dir"]):
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 30)
    w = random.randint(12, 30)
    total_arrows = random.randint(3, len(PALETTE))
    final_mode = random.choice(("boundary", "repeat"))
    if final_mode == "repeat":
        chain_len = random.randint(3, min(total_arrows, 5))
    else:
        chain_len = random.randint(2, min(total_arrows, 5))

    start_dir = random.choice((RIGHT, DOWN))
    start_r0 = 0
    start_c0 = random.randint(0, min(2, w - 2))
    initial = (start_r0, start_c0)
    records = [{"r0": start_r0, "c0": start_c0, "dir": start_dir}]
    input_occ: dict[tuple[int, int], int] = {}
    dyn_occ: dict[tuple[int, int], int] = {}
    _add_to_occ(input_occ, _cells_at(start_r0, start_c0, start_dir), 0)
    _add_to_occ(dyn_occ, _cells_at(start_r0, start_c0, start_dir), 0)

    current_head = _head_at(start_r0, start_c0, start_dir)
    current_dir = start_dir
    ray_cells: set[tuple[int, int]] = set()
    flipped_ids: set[int] = set()

    for chain_idx in range(1, chain_len):
        mode = final_mode if chain_idx == chain_len - 1 else None
        placed = _place_chain_arrow(
            records,
            current_head,
            current_dir,
            h,
            w,
            initial,
            input_occ,
            dyn_occ,
            ray_cells,
            flipped_ids,
            mode,
        )
        if placed is None:
            return None
        current_head, current_dir = placed

    owner, final_path, _hit = _cast(current_head, current_dir, dyn_occ, h, w)
    if final_mode == "boundary":
        if owner is not None:
            return None
    elif owner not in flipped_ids:
        return None
    ray_cells.update(final_path)

    for _ in range(total_arrows - chain_len):
        if not _place_distractor(records, h, w, initial, input_occ, dyn_occ, ray_cells):
            return None

    grid = _render(records, h, w)
    try:
        solved = main(grid)
    except Exception:
        return None
    if solved.shape != grid.shape or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-chain input")
