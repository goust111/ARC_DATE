from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 8, 9)
QUADRANT_NAMES = ("lu", "ru", "lb", "rb")
LAYER_ORDER = ("lb", "ru", "lu", "rb")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PATH_STEPS = ((-2, 0), (2, 0), (0, -2), (0, 2))
SIDE_TO_OFFSET = {
    "U": (-1, 1),
    "D": (3, 1),
    "L": (1, -1),
    "R": (1, 3),
}

CHECKER_PATTERNS = (
    np.array(((ORANGE, BLACK, ORANGE), (BLACK, ORANGE, BLACK), (ORANGE, BLACK, ORANGE)), dtype=int),
    np.array(((BLACK, ORANGE, BLACK), (ORANGE, BLACK, ORANGE), (BLACK, ORANGE, BLACK)), dtype=int),
)

BASE_LAYER_MASKS = {
    "lb": np.array(((1, 0, 0), (1, 0, 1), (1, 0, 1)), dtype=bool),
    "ru": np.array(((1, 1, 1), (0, 0, 1), (0, 1, 0)), dtype=bool),
    "lu": np.array(((1, 0, 1), (1, 0, 1), (1, 0, 0)), dtype=bool),
    "rb": np.array(((1, 1, 1), (1, 0, 1), (0, 0, 0)), dtype=bool),
}

TARGET_MASKS = (
    np.array(((1, 1, 1), (1, 0, 1), (1, 1, 1)), dtype=bool),
    np.array(((1, 0, 1), (1, 0, 1), (1, 1, 1)), dtype=bool),
    np.array(((1, 1, 1), (1, 0, 1), (1, 0, 1)), dtype=bool),
    np.array(((1, 1, 1), (0, 0, 1), (1, 1, 1)), dtype=bool),
    np.array(((1, 1, 1), (1, 0, 0), (1, 1, 1)), dtype=bool),
    np.array(((1, 0, 1), (0, 0, 1), (1, 1, 1)), dtype=bool),
    np.array(((1, 1, 1), (1, 0, 0), (1, 0, 1)), dtype=bool),
    np.array(((0, 1, 0), (1, 0, 1), (1, 1, 1)), dtype=bool),
    np.array(((1, 1, 1), (1, 0, 1), (0, 1, 0)), dtype=bool),
    np.array(((0, 1, 1), (1, 0, 1), (1, 1, 0)), dtype=bool),
    np.array(((1, 1, 0), (1, 0, 1), (0, 1, 1)), dtype=bool),
    BASE_LAYER_MASKS["lb"],
    BASE_LAYER_MASKS["ru"],
    BASE_LAYER_MASKS["lu"],
    BASE_LAYER_MASKS["rb"],
)

MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _crop(grid: np.ndarray, top: int, left: int, height: int = 3, width: int = 3) -> np.ndarray:
    return grid[top : top + height, left : left + width].copy()


def _adjacent_sites(top: int, left: int) -> dict[str, tuple[int, int]]:
    return {
        side: (top + dr, left + dc)
        for side, (dr, dc) in SIDE_TO_OFFSET.items()
    }


def _render_target(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros((3, 3), dtype=int)
    out[mask] = int(color)
    return out


def _transform_mask(mask: np.ndarray, rotation: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, k=rotation)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _overlay_named_crops(crops: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    out = np.zeros((3, 3), dtype=int)
    for name in LAYER_ORDER:
        patch = crops[name]
        mask = patch != BLACK
        out[mask] = patch[mask]

    covered = np.zeros((3, 3), dtype=bool)
    visible: dict[str, int] = {}
    for name in reversed(LAYER_ORDER):
        mask = crops[name] != BLACK
        vis = mask & ~covered
        visible[name] = int(vis.sum())
        covered |= mask
    return out, visible


def _is_checker(window: np.ndarray) -> bool:
    return window.shape == (3, 3) and any(np.array_equal(window, pat) for pat in CHECKER_PATTERNS)


def _is_target_window(window: np.ndarray) -> bool:
    if window.shape != (3, 3) or int(window[1, 1]) != BLACK:
        return False
    border = [int(window[r, c]) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    nz = [value for value in border if value != BLACK]
    if len(nz) < 5:
        return False
    colors = set(nz)
    if len(colors) != 1:
        return False
    color = next(iter(colors))
    if color in {GREY, ORANGE}:
        return False
    return all(value in {BLACK, color} for value in border)


def _find_checker_top_left(local: np.ndarray) -> tuple[int, int]:
    matches = [
        (r, c)
        for r in range(local.shape[0] - 2)
        for c in range(local.shape[1] - 2)
        if _is_checker(local[r : r + 3, c : c + 3])
    ]
    assert len(matches) == 1, "each quadrant must contain exactly one checkerboard source"
    return matches[0]


def _find_adjacent_guide(local: np.ndarray, checker_top: int, checker_left: int) -> tuple[int, int, int]:
    connectors = []
    for rr, cc in _adjacent_sites(checker_top, checker_left).values():
        if not _in_bounds(local.shape, rr, cc):
            continue
        value = int(local[rr, cc])
        if value not in {BLACK, GREY, ORANGE}:
            connectors.append((rr, cc, value))
    assert len(connectors) == 1, "checkerboard must touch exactly one guide pixel"
    return connectors[0]


def _guide_neighbors(local: np.ndarray, r: int, c: int, color: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in PATH_STEPS:
        rr = r + dr
        cc = c + dc
        if _in_bounds(local.shape, rr, cc) and int(local[rr, cc]) == color:
            out.append((rr, cc))
    return out


def _follow_path(local: np.ndarray, start: tuple[int, int], color: int) -> tuple[int, int]:
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        nbrs = _guide_neighbors(local, cur[0], cur[1], color)
        assert len(nbrs) <= 2, "guide path may not branch"
        nexts = nbrs if prev is None else [node for node in nbrs if node != prev]
        if not nexts:
            break
        assert len(nexts) == 1, "guide path must be a single chain"
        nxt = nexts[0]
        assert nxt not in seen, "guide path may not form a cycle"
        seen.add(nxt)
        prev, cur = cur, nxt

    guide_nodes = {
        (int(r), int(c))
        for r, c in np.argwhere(local == color)
    }
    assert guide_nodes == seen, "all same-color guide pixels in the quadrant must lie on the traced path"
    return cur


def _find_target_windows(local: np.ndarray) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(local.shape[0] - 2)
        for c in range(local.shape[1] - 2)
        if _is_target_window(local[r : r + 3, c : c + 3])
    ]


def _find_terminal_target(local: np.ndarray, terminal: tuple[int, int]) -> tuple[int, int]:
    matches = [
        (r, c)
        for r, c in _find_target_windows(local)
        if terminal in _adjacent_sites(r, c).values()
    ]
    assert len(matches) == 1, "path terminal must point at exactly one target object"
    return matches[0]


def _extract_quadrant_target(local: np.ndarray) -> np.ndarray:
    checker_top, checker_left = _find_checker_top_left(local)
    start_r, start_c, color = _find_adjacent_guide(local, checker_top, checker_left)
    terminal = _follow_path(local, (start_r, start_c), color)
    target_top, target_left = _find_terminal_target(local, terminal)
    return _crop(local, target_top, target_left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the guide chain in each cross-separated quadrant, extract the reached 3x3 target, and stack the four crops by quadrant order.

    Input structure:
    - A grey full row and a grey full column cross at the grid center and split the canvas into four equal quadrants.
    - Each quadrant contains one orange-black 3x3 checkerboard, one same-color guide chain on a two-cell lattice, one selected 3x3 target object, and extra distractor 3x3 target objects.
    - Each target object is a monochrome 3x3 ring-like mask with a black center and black transparency holes on its border.

    Rule:
    - In each quadrant, find the unique checkerboard and its unique adjacent guide pixel.
    - Follow that guide color through horizontal or vertical jumps of exactly two cells until the path ends.
    - Crop the unique 3x3 target object adjacent to that terminal pixel.
    - Transparently overlay the four 3x3 crops on a black 3x3 canvas in this bottom-to-top order: left-bottom, right-top, left-top, right-bottom.

    Key insight:
    - The path selects which local 3x3 target matters in each quadrant, while the final picture depends on the fixed global layer order because black cells stay transparent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    grey_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "expected exactly one grey cross row and one grey cross column"
    mid_r = grey_rows[0]
    mid_c = grey_cols[0]
    assert mid_r == g.shape[0] // 2 and mid_c == g.shape[1] // 2, "grey cross must be centered"

    quadrants = {
        "lu": g[:mid_r, :mid_c],
        "ru": g[:mid_r, mid_c + 1 :],
        "lb": g[mid_r + 1 :, :mid_c],
        "rb": g[mid_r + 1 :, mid_c + 1 :],
    }
    q_shapes = {quad.shape for quad in quadrants.values()}
    assert len(q_shapes) == 1, "quadrants must all have the same shape"

    crops = {name: _extract_quadrant_target(local) for name, local in quadrants.items()}
    out, _visible = _overlay_named_crops(crops)
    return out


def _expanded_boxes_separate(a: tuple[int, int], b: tuple[int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a[0] - pad, a[1] - pad, a[0] + 2 + pad, a[1] + 2 + pad
    br0, bc0, br1, bc1 = b[0] - pad, b[1] - pad, b[0] + 2 + pad, b[1] + 2 + pad
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _candidate_object_positions(q: int, row_parity: int, col_parity: int) -> list[tuple[int, int]]:
    rows = [r for r in range(1, q - 2) if r % 2 == row_parity]
    cols = [c for c in range(1, q - 2) if c % 2 == col_parity]
    return [(r, c) for r in rows for c in cols]


def _object_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr in range(3) for dc in range(3)}


def _checker_sprite() -> np.ndarray:
    return CHECKER_PATTERNS[random.randrange(len(CHECKER_PATTERNS))].copy()


def _node_neighbors(q: int, node: tuple[int, int], row_parity: int, col_parity: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in PATH_STEPS:
        rr = node[0] + dr
        cc = node[1] + dc
        if 0 <= rr < q and 0 <= cc < q and rr % 2 == row_parity and cc % 2 == col_parity:
            out.append((rr, cc))
    random.shuffle(out)
    return out


def _shortest_path(
    q: int,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked_cells: set[tuple[int, int]],
    forbidden_sites: set[tuple[int, int]],
    row_parity: int,
    col_parity: int,
) -> list[tuple[int, int]] | None:
    def allowed(node: tuple[int, int]) -> bool:
        if not _in_bounds((q, q), node[0], node[1]):
            return False
        if node[0] % 2 != row_parity or node[1] % 2 != col_parity:
            return False
        if node in blocked_cells:
            return False
        if node in forbidden_sites and node not in {start, end}:
            return False
        return True

    if not allowed(start) or not allowed(end):
        return None

    parents = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        for nxt in _node_neighbors(q, cur, row_parity, col_parity):
            if nxt not in parents and allowed(nxt):
                parents[nxt] = cur
                queue.append(nxt)
    if end not in parents:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _random_path(
    q: int,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked_cells: set[tuple[int, int]],
    forbidden_sites: set[tuple[int, int]],
    row_parity: int,
    col_parity: int,
) -> list[tuple[int, int]] | None:
    def is_allowed_waypoint(node: tuple[int, int]) -> bool:
        return (
            _in_bounds((q, q), node[0], node[1])
            and node[0] % 2 == row_parity
            and node[1] % 2 == col_parity
            and node not in blocked_cells
            and node not in forbidden_sites
            and node not in {start, end}
        )

    direct = _shortest_path(q, start, end, blocked_cells, forbidden_sites, row_parity, col_parity)
    if direct is None:
        return None

    if random.random() < 0.65:
        waypoints = [
            (r, c)
            for r in range(q)
            for c in range(q)
            if is_allowed_waypoint((r, c))
        ]
        random.shuffle(waypoints)
        for waypoint in waypoints[:16]:
            left = _shortest_path(q, start, waypoint, blocked_cells, forbidden_sites, row_parity, col_parity)
            right = _shortest_path(q, waypoint, end, blocked_cells, forbidden_sites, row_parity, col_parity)
            if left is None or right is None:
                continue
            joined = left[:-1] + right
            if len(set(joined)) != len(joined):
                continue
            if len(joined) >= len(direct):
                return joined
    return direct


def _is_chain_path(path: list[tuple[int, int]]) -> bool:
    path_set = set(path)
    if len(path_set) != len(path):
        return False
    degrees = []
    for r, c in path:
        deg = sum((r + dr, c + dc) in path_set for dr, dc in PATH_STEPS)
        if deg > 2:
            return False
        degrees.append(deg)
    return degrees.count(1) == 2 and all(deg in {1, 2} for deg in degrees)


def _pick_selected_crops() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    colors = random.sample(list(NON_SPECIAL_COLORS), 4)
    rotation = random.randrange(4)
    flip_lr = random.random() < 0.5
    masks = {
        name: _transform_mask(BASE_LAYER_MASKS[name], rotation, flip_lr)
        for name in QUADRANT_NAMES
    }
    crops = {
        name: _render_target(masks[name], color)
        for name, color in zip(QUADRANT_NAMES, colors)
    }
    out, visible = _overlay_named_crops(crops)
    assert all(visible[name] > 0 for name in QUADRANT_NAMES)
    return crops, out, visible


def _choose_distinct_positions(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for pos in candidates:
        if all(_expanded_boxes_separate(pos, other, pad=1) for other in picked):
            picked.append(pos)
            if len(picked) == count:
                return picked
    return None


def _plan_quadrant(selected_crop: np.ndarray, q: int) -> dict | None:
    selected_color = int(selected_crop[selected_crop != BLACK][0])
    row_parity = random.randrange(2)
    col_parity = random.randrange(2)
    positions = _candidate_object_positions(q, row_parity, col_parity)
    random.shuffle(positions)

    for _ in range(250):
        checker_pos = random.choice(positions)
        target_choices = [
            pos
            for pos in positions
            if pos != checker_pos and _expanded_boxes_separate(pos, checker_pos, pad=1)
        ]
        if not target_choices:
            continue
        target_pos = random.choice(target_choices)

        side_pairs = [(a, b) for a in SIDE_TO_OFFSET for b in SIDE_TO_OFFSET]
        random.shuffle(side_pairs)

        for checker_side, target_side in side_pairs:
            start = _adjacent_sites(*checker_pos)[checker_side]
            end = _adjacent_sites(*target_pos)[target_side]
            if abs(start[0] - end[0]) + abs(start[1] - end[1]) < 4:
                continue

            blocked_cells = _object_cells(*checker_pos) | _object_cells(*target_pos)
            forbidden_sites = set(_adjacent_sites(*checker_pos).values()) | set(_adjacent_sites(*target_pos).values())
            forbidden_sites -= {start, end}
            path = _random_path(q, start, end, blocked_cells, forbidden_sites, start[0] % 2, start[1] % 2)
            if path is None or len(path) < 3:
                continue
            if not _is_chain_path(path):
                continue

            path_set = set(path)
            distractor_count = random.randint(1, 2)
            distractor_candidates = []
            for pos in positions:
                if pos in {checker_pos, target_pos}:
                    continue
                if not _expanded_boxes_separate(pos, checker_pos, pad=1):
                    continue
                if not _expanded_boxes_separate(pos, target_pos, pad=1):
                    continue
                if _object_cells(*pos) & path_set:
                    continue
                if set(_adjacent_sites(*pos).values()) & path_set:
                    continue
                distractor_candidates.append(pos)
            distractor_positions = _choose_distinct_positions(distractor_candidates, distractor_count)
            if distractor_positions is None:
                continue

            object_colors = {selected_color}
            distractor_colors = random.sample(
                [color for color in NON_SPECIAL_COLORS if color != selected_color],
                distractor_count,
            )
            object_colors.update(distractor_colors)
            path_color_choices = [color for color in NON_SPECIAL_COLORS if color not in object_colors]
            if not path_color_choices:
                continue
            path_color = random.choice(path_color_choices)

            distractor_masks = random.sample(list(TARGET_MASKS), distractor_count)
            local = np.zeros((q, q), dtype=int)
            local[tuple(zip(*path))] = path_color
            local[checker_pos[0] : checker_pos[0] + 3, checker_pos[1] : checker_pos[1] + 3] = _checker_sprite()
            local[target_pos[0] : target_pos[0] + 3, target_pos[1] : target_pos[1] + 3] = selected_crop
            for pos, color, mask in zip(distractor_positions, distractor_colors, distractor_masks):
                local[pos[0] : pos[0] + 3, pos[1] : pos[1] + 3] = _render_target(mask, color)

            if not np.array_equal(_extract_quadrant_target(local), selected_crop):
                continue
            if len(_find_target_windows(local)) != 1 + distractor_count:
                continue

            return {
                "grid": local,
                "checker_pos": checker_pos,
                "target_pos": target_pos,
                "distractor_positions": distractor_positions,
                "path": path,
                "path_color": path_color,
                "selected_crop": selected_crop,
            }
    return None


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    q = random.randint(10, 13)
    size = 2 * q + 1
    crops, expected, visible = _pick_selected_crops()

    plans: dict[str, dict] = {}
    for name in QUADRANT_NAMES:
        plan = _plan_quadrant(crops[name], q)
        if plan is None:
            return None
        plans[name] = plan

    grid = np.zeros((size, size), dtype=int)
    grid[q, :] = GREY
    grid[:, q] = GREY
    offsets = {
        "lu": (0, 0),
        "ru": (0, q + 1),
        "lb": (q + 1, 0),
        "rb": (q + 1, q + 1),
    }
    for name, (r0, c0) in offsets.items():
        quad = plans[name]["grid"]
        grid[r0 : r0 + q, c0 : c0 + q] = quad

    if not all(visible[name] > 0 for name in QUADRANT_NAMES):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if any(np.array_equal(expected, crops[name]) for name in QUADRANT_NAMES):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
