from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
ARROW_TAIL = 5
RAY = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, valid_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comp_grid = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not valid_mask[r, c] or visited[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            comp_id = len(comps)
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                comp_grid[rr, cc] = comp_id
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and valid_mask[nr, nc]
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps, comp_grid


def _border_side(shape: tuple[int, int], cell: tuple[int, int]) -> str:
    h, w = shape
    r, c = cell
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("portal must lie on the border")


def _inward_normal(shape: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    side = _border_side(shape, cell)
    if side == "top":
        return 1, 0
    if side == "bottom":
        return -1, 0
    if side == "left":
        return 0, 1
    return 0, -1


def _parse_portals(grid: np.ndarray) -> dict[tuple[int, int], dict]:
    h, w = grid.shape
    colors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            value = int(grid[r, c])
            if value == BG:
                continue
            assert (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
            colors[value].append((r, c))

    portals: dict[tuple[int, int], dict] = {}
    for color, coords in colors.items():
        assert len(coords) == 2, "portal colors must appear exactly twice"
        a, b = coords
        portals[a] = {"color": int(color), "pair": b}
        portals[b] = {"color": int(color), "pair": a}
    return portals


def _find_arrow(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for r, c in np.argwhere(grid == ARROW_TAIL):
        rr, cc = int(r), int(c)
        heads = []
        for nr, nc in _neighbors4(rr, cc):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == RAY:
                grey_neighbors = sum(
                    _in_bounds(grid.shape, ar, ac) and int(grid[ar, ac]) == ARROW_TAIL
                    for ar, ac in _neighbors4(nr, nc)
                )
                if grey_neighbors == 1:
                    heads.append((nr, nc))
        if len(heads) == 1:
            head = heads[0]
            candidates.append(((rr, cc), head, (head[0] - rr, head[1] - cc)))

    assert len(candidates) == 1, "expected exactly one two-cell arrow"
    return candidates[0]


def _find_blocks(
    grid: np.ndarray,
    portal_cells: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
) -> tuple[list[dict], np.ndarray]:
    valid = grid != BG
    for r, c in portal_cells | arrow_cells:
        valid[r, c] = False
    return _components(grid, valid)


def _trace_beam(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    portals = _parse_portals(g)
    tail, head, direction = _find_arrow(g)
    blocks, comp_grid = _find_blocks(g, set(portals), {tail, head})

    current = head
    start_direction = direction
    path: list[tuple[int, int]] = []
    path_set: set[tuple[int, int]] = set()
    teleports: list[dict] = []
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for _ in range(g.shape[0] * g.shape[1] * 8):
        state = (current, direction)
        assert state not in seen_states, "beam entered a cycle"
        seen_states.add(state)

        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            return {
                "path": path,
                "teleports": teleports,
                "termination": "exit",
                "component": None,
                "blocks": blocks,
                "start_direction": start_direction,
            }

        nxt = (nr, nc)
        if nxt in portals:
            exit_cell = portals[nxt]["pair"]
            new_direction = _inward_normal(g.shape, exit_cell)
            teleports.append(
                {
                    "entry": nxt,
                    "exit": exit_cell,
                    "color": portals[nxt]["color"],
                    "direction": new_direction,
                }
            )
            current = exit_cell
            direction = new_direction
            continue

        comp_id = int(comp_grid[nr, nc])
        if comp_id >= 0:
            return {
                "path": path,
                "teleports": teleports,
                "termination": "block",
                "component": comp_id,
                "blocks": blocks,
                "start_direction": start_direction,
            }

        assert nxt not in {tail, head}, "beam should not hit the arrow"
        assert int(g[nr, nc]) == BG, "beam can only traverse background before terminating"
        assert nxt not in path_set, "beam path must not self-intersect"
        path_set.add(nxt)
        path.append(nxt)
        current = nxt

    raise AssertionError("beam did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the arrow beam through paired border portals and repaint the final hit block.

    Input structure:
    - The grid has black background, one two-cell arrow (gray tail plus teal head),
      several solid interior color blocks, and same-color portal pairs on border cells.
    - Each portal color appears exactly twice on the border, never on a corner, and
      every portal pair defines a teleport connection.

    Rule:
    - Start from the cell in front of the arrow head and move straight in the arrow
      direction, painting traversed black cells teal.
    - When the next cell is a portal, jump to the matching border portal and continue
      along that exit portal's inward normal, keeping both portal cells unchanged.
    - When the next cell is a block, stop and repaint that whole 4-connected
      monochromatic block teal; when the next step would leave the grid, stop.

    Key insight:
    - Portal pairs do not define straight connections between themselves; each exit
      portal rewrites the beam direction according to the side it sits on, so the
      answer requires exact segment-by-segment simulation.
    """
    g = np.asarray(grid, dtype=int)
    trace = _trace_beam(g)
    out = g.copy()
    for r, c in trace["path"]:
        out[r, c] = RAY
    if trace["termination"] == "block":
        coords = trace["blocks"][trace["component"]]["coords"]
        out[coords[:, 0], coords[:, 1]] = RAY
    return out


def _sample_spaced_values(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picked = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _empty_grid(h: int, w: int) -> np.ndarray:
    return np.full((h, w), BG, dtype=int)


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, rh, cw = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r0 + rh, c0 : c0 + cw] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, rh, cw = rect
    grid[r0 : r0 + rh, c0 : c0 + cw] = int(color)


def _choose_horizontal_target(
    shape: tuple[int, int],
    blocked: np.ndarray,
    beam_row: int,
    c_lo: int,
    c_hi: int,
    r_lo: int = 1,
    r_hi: int | None = None,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    if r_hi is None:
        r_hi = h - 2
    candidates = []
    for rh in range(2, 5):
        row_low = max(r_lo, beam_row - rh + 1)
        row_high = min(r_hi, beam_row, h - rh - 1)
        if row_low > row_high:
            continue
        for cw in range(2, 5):
            col_high = min(c_hi, w - cw - 1)
            if c_lo > col_high:
                continue
            for r0 in range(row_low, row_high + 1):
                for c0 in range(c_lo, col_high + 1):
                    if blocked[r0 : r0 + rh, c0 : c0 + cw].any():
                        continue
                    candidates.append((r0, c0, rh, cw))
    return random.choice(candidates) if candidates else None


def _choose_vertical_target(
    shape: tuple[int, int],
    blocked: np.ndarray,
    beam_col: int,
    r_lo: int,
    r_hi: int,
    c_lo: int = 1,
    c_hi: int | None = None,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    if c_hi is None:
        c_hi = w - 2
    candidates = []
    for rh in range(2, 5):
        row_high = min(r_hi, h - rh - 1)
        if r_lo > row_high:
            continue
        for cw in range(2, 5):
            col_low = max(c_lo, beam_col - cw + 1)
            col_high = min(c_hi, beam_col, w - cw - 1)
            if col_low > col_high:
                continue
            for r0 in range(r_lo, row_high + 1):
                for c0 in range(col_low, col_high + 1):
                    if blocked[r0 : r0 + rh, c0 : c0 + cw].any():
                        continue
                    candidates.append((r0, c0, rh, cw))
    return random.choice(candidates) if candidates else None


def _place_extra_blocks(
    grid: np.ndarray,
    base_forbidden: np.ndarray,
    colors: list[int],
) -> bool:
    forbidden = base_forbidden.copy()
    h, w = grid.shape
    for color in colors:
        candidates = []
        for rh in range(2, 5):
            for cw in range(2, 5):
                for r0 in range(1, h - rh):
                    for c0 in range(1, w - cw):
                        if forbidden[r0 : r0 + rh, c0 : c0 + cw].any():
                            continue
                        candidates.append((r0, c0, rh, cw))
        if not candidates:
            return False
        rect = random.choice(candidates)
        _paint_rect(grid, rect, color)
        forbidden |= _dilate8(_rect_mask(grid.shape, rect))
    return True


def _choose_color_split(portal_count: int, total_blocks: int) -> tuple[list[int], list[int]]:
    colors = random.sample(PALETTE, portal_count + total_blocks)
    return colors[:portal_count], colors[portal_count:]


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _validate_generated(
    grid: np.ndarray,
    expected_teleports: int,
    expected_end: str,
    expected_turns: int,
) -> bool:
    try:
        trace = _trace_beam(grid)
    except AssertionError:
        return False

    if len(trace["teleports"]) != expected_teleports:
        return False
    if trace["termination"] != expected_end:
        return False

    dirs = [trace["start_direction"], *[tp["direction"] for tp in trace["teleports"]]]
    turn_count = sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))
    if turn_count != expected_turns:
        return False

    portals = _parse_portals(grid)
    if len(portals) != 2 * expected_teleports:
        return False
    for cell, spec in portals.items():
        if spec["pair"] == cell:
            return False
    for tp in trace["teleports"]:
        if _inward_normal(grid.shape, tp["exit"]) != tp["direction"]:
            return False

    for block in trace["blocks"]:
        r0, c0, r1, c1 = block["bbox"]
        if block["area"] != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch == block["color"]):
            return False

    out = main(grid)
    allowed = np.zeros(grid.shape, dtype=bool)
    for r, c in trace["path"]:
        allowed[r, c] = True
    if trace["termination"] == "block":
        coords = trace["blocks"][trace["component"]]["coords"]
        allowed[coords[:, 0], coords[:, 1]] = True

    changed = out != grid
    if np.any(changed & ~allowed):
        return False
    if allowed.any() and not np.all(out[allowed] == RAY):
        return False
    return True


def _build_parallel(end_kind: str) -> tuple[np.ndarray, dict] | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    teleports = random.choice([1, 2, 3])
    rows = _sample_spaced_values(2, h - 3, teleports + 1, 3)
    if rows is None:
        return None

    head_c_hi = min(w - 8, max(3, w // 3))
    if head_c_hi < 2:
        return None
    head_c = random.randint(2, head_c_hi)
    total_blocks = random.randint(2, min(3, len(PALETTE) - teleports))
    portal_colors, block_colors = _choose_color_split(teleports, total_blocks)

    grid = _empty_grid(h, w)
    grid[rows[0], head_c - 1] = ARROW_TAIL
    grid[rows[0], head_c] = RAY

    for i, color in enumerate(portal_colors):
        grid[rows[i], w - 1] = color
        grid[rows[i + 1], 0] = color

    path_mask = np.zeros((h, w), dtype=bool)
    path_mask[rows[0], head_c + 1 : w - 1] = True
    for row in rows[1:-1]:
        path_mask[row, 1 : w - 1] = True

    occupied = grid != BG
    if end_kind == "block":
        blocked = occupied | path_mask
        target = _choose_horizontal_target(
            (h, w),
            blocked,
            rows[-1],
            c_lo=4,
            c_hi=w - 5,
        )
        if target is None:
            return None
        _paint_rect(grid, target, block_colors[0])
        path_mask[rows[-1], 1 : target[1]] = True
        extra_colors = block_colors[1:]
    else:
        path_mask[rows[-1], 1:w] = True
        extra_colors = block_colors

    base_forbidden = _dilate8(path_mask | (grid != BG))
    if not _place_extra_blocks(grid, base_forbidden, extra_colors):
        return None
    return grid, {"teleports": teleports, "end": end_kind, "turns": 0}


def _build_single_turn(end_kind: str) -> tuple[np.ndarray, dict] | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    r0 = random.randint(5, h - 7)
    head_c = random.randint(5, max(5, w // 2))
    if head_c > w - 6:
        head_c = w - 6
    if head_c < 5:
        return None
    exit_col = random.randint(1, head_c - 2)

    total_blocks = random.randint(2, min(3, len(PALETTE) - 1))
    portal_colors, block_colors = _choose_color_split(1, total_blocks)
    color = portal_colors[0]

    grid = _empty_grid(h, w)
    grid[r0, head_c - 1] = ARROW_TAIL
    grid[r0, head_c] = RAY
    grid[r0, w - 1] = color
    grid[0, exit_col] = color

    path_mask = np.zeros((h, w), dtype=bool)
    path_mask[r0, head_c + 1 : w - 1] = True
    occupied = grid != BG

    if end_kind == "block":
        blocked = occupied | path_mask
        target = _choose_vertical_target(
            (h, w),
            blocked,
            exit_col,
            r_lo=r0 + 2,
            r_hi=h - 5,
        )
        if target is None:
            return None
        _paint_rect(grid, target, block_colors[0])
        path_mask[1 : target[0], exit_col] = True
        extra_colors = block_colors[1:]
    else:
        path_mask[1:h, exit_col] = True
        extra_colors = block_colors

    base_forbidden = _dilate8(path_mask | (grid != BG))
    if not _place_extra_blocks(grid, base_forbidden, extra_colors):
        return None
    return grid, {"teleports": 1, "end": end_kind, "turns": 1}


def _build_double_turn() -> tuple[np.ndarray, dict] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    r0 = random.randint(5, h - 8)
    head_c = random.randint(6, w - 7)
    exit_col = random.randint(4, head_c - 2)
    final_row = random.randint(r0 + 3, h - 3)
    total_blocks = random.randint(2, min(3, len(PALETTE) - 2))
    portal_colors, block_colors = _choose_color_split(2, total_blocks)

    grid = _empty_grid(h, w)
    grid[r0, head_c - 1] = ARROW_TAIL
    grid[r0, head_c] = RAY
    grid[r0, w - 1] = portal_colors[0]
    grid[0, exit_col] = portal_colors[0]
    grid[h - 1, exit_col] = portal_colors[1]
    grid[final_row, 0] = portal_colors[1]

    path_mask = np.zeros((h, w), dtype=bool)
    path_mask[r0, head_c + 1 : w - 1] = True
    path_mask[1 : h - 1, exit_col] = True
    occupied = grid != BG
    blocked = occupied | path_mask
    target = _choose_horizontal_target(
        (h, w),
        blocked,
        final_row,
        c_lo=2,
        c_hi=exit_col - 3,
        r_lo=1,
        r_hi=h - 2,
    )
    if target is None or target[1] + target[3] - 1 >= exit_col:
        return None
    _paint_rect(grid, target, block_colors[0])
    path_mask[final_row, 1 : target[1]] = True

    base_forbidden = _dilate8(path_mask | (grid != BG))
    if not _place_extra_blocks(grid, base_forbidden, block_colors[1:]):
        return None
    return grid, {"teleports": 2, "end": "block", "turns": 2}


def _generate_attempt() -> np.ndarray | None:
    template = random.choices(
        ["parallel", "single_turn", "double_turn"],
        weights=[0.4, 0.4, 0.2],
        k=1,
    )[0]
    if template == "parallel":
        end_kind = random.choices(["block", "exit"], weights=[0.65, 0.35], k=1)[0]
        built = _build_parallel(end_kind)
    elif template == "single_turn":
        end_kind = random.choice(["block", "exit"])
        built = _build_single_turn(end_kind)
    else:
        built = _build_double_turn()

    if built is None:
        return None
    grid, meta = built
    grid = _apply_random_symmetry(grid)
    if not _validate_generated(
        grid,
        expected_teleports=meta["teleports"],
        expected_end=meta["end"],
        expected_turns=meta["turns"],
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-ballistics input")
