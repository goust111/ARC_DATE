import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8)

ARROW_SHAPES = {
    "tl": {
        "direction": (1, 1),
        "occupied": ((0, 2), (1, 2), (2, 0), (2, 1), (2, 2)),
        "stamp": ((0, 0), (0, 1), (1, 0), (1, 1)),
    },
    "tr": {
        "direction": (1, -1),
        "occupied": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        "stamp": ((0, 1), (0, 2), (1, 1), (1, 2)),
    },
    "bl": {
        "direction": (-1, 1),
        "occupied": ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)),
        "stamp": ((1, 0), (1, 1), (2, 0), (2, 1)),
    },
    "br": {
        "direction": (-1, -1),
        "occupied": ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        "stamp": ((1, 1), (1, 2), (2, 1), (2, 2)),
    },
}

ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
ARROW_COUNT_BY_AREA = (
    (125, 2),
    (185, 3),
    (10**9, 4),
)
EXTRA_SEGMENT_RANGE = (0, 2)
SEGMENT_LEN_RANGE = (2, 6)
PLACEMENT_ATTEMPTS = 240
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ORTHOGONAL:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _identify_arrow(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 5:
        return None
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    local = frozenset((r - r0, c - c0) for r, c in coords)
    for missing, shape in ARROW_SHAPES.items():
        if local == frozenset(shape["occupied"]):
            return {
                "missing": missing,
                "top_left": (int(r0), int(c0)),
                "direction": shape["direction"],
                "occupied": shape["occupied"],
                "stamp": shape["stamp"],
            }
    return None


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)]
    for color in colors:
        for comp in _components(grid == int(color)):
            arrow = _identify_arrow(comp)
            if arrow is None:
                continue
            arrow["color"] = int(color)
            arrows.append(arrow)
    arrows.sort(key=lambda a: (int(a["top_left"][0]), int(a["top_left"][1]), int(a["color"])))
    return arrows


def _translate_cells(top_left: tuple[int, int], local_cells: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(int(r0 + dr), int(c0 + dc)) for dr, dc in local_cells]


def _bbox_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(int(r0 + dr), int(c0 + dc)) for dr in range(3) for dc in range(3)}


def _bbox_in_bounds(top_left: tuple[int, int], h: int, w: int) -> bool:
    r0, c0 = top_left
    return 0 <= int(r0) <= h - 3 and 0 <= int(c0) <= w - 3


def _one_step_stop_reason(
    top_left: tuple[int, int],
    arrow: dict[str, object],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> str | None:
    dr, dc = arrow["direction"]
    nxt = (int(top_left[0] + dr), int(top_left[1] + dc))
    for rr, cc in _translate_cells(nxt, arrow["occupied"]):
        if not (0 <= rr < h and 0 <= cc < w):
            return "boundary"
    for cell in _translate_cells(nxt, arrow["occupied"]):
        if cell in obstacles:
            return "obstacle"
    return None


def _simulate_projection(
    top_left: tuple[int, int],
    arrow: dict[str, object],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[tuple[int, int], str]:
    cur = (int(top_left[0]), int(top_left[1]))
    while True:
        reason = _one_step_stop_reason(cur, arrow, obstacles, h, w)
        if reason is not None:
            return cur, reason
        dr, dc = arrow["direction"]
        cur = (int(cur[0] + dr), int(cur[1] + dc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each 3x3 L-shaped arrow diagonally until its next step would hit the boundary
    or a grey obstacle, then stamp the complementary 2x2 missing corner in maroon.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    obstacle_set = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}

    for arrow in _find_arrows(g):
        final_top_left, _reason = _simulate_projection(
            arrow["top_left"], arrow, obstacle_set, h, w
        )
        for rr, cc in _translate_cells(final_top_left, arrow["stamp"]):
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == BG:
                out[rr, cc] = MAROON

    return out


def _area_arrow_cap(area: int) -> int:
    for limit, cap in ARROW_COUNT_BY_AREA:
        if area <= limit:
            return int(cap)
    return 4


def _path_positions(final_top_left: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    dr, dc = direction
    start = (
        int(final_top_left[0] - dr * steps),
        int(final_top_left[1] - dc * steps),
    )
    return [
        (int(start[0] + dr * t), int(start[1] + dc * t))
        for t in range(steps + 1)
    ]


def _path_bbox_union(positions: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for top_left in positions:
        cells |= _bbox_cells(top_left)
    return cells


def _candidate_paths(
    final_top_left: tuple[int, int],
    arrow: dict[str, object],
    reserved: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, list[tuple[int, int]], set[tuple[int, int]]]]:
    out: list[tuple[int, list[tuple[int, int]], set[tuple[int, int]]]] = []
    step = 1
    while True:
        start = (
            int(final_top_left[0] - arrow["direction"][0] * step),
            int(final_top_left[1] - arrow["direction"][1] * step),
        )
        if not _bbox_in_bounds(start, h, w):
            break
        positions = _path_positions(final_top_left, arrow["direction"], step)
        bbox_union = _path_bbox_union(positions)
        if not (bbox_union & reserved) and not (bbox_union & obstacles):
            out.append((int(step), positions, bbox_union))
        step += 1
    return out


def _segment_cells_ok(
    cells: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    cell_set = set(cells)
    for rr, cc in cells:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if (rr, cc) in blocked or (rr, cc) in obstacles:
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if (nr, nc) in obstacles and (nr, nc) not in cell_set:
                    return False
    return True


def _choose_blocker_segment(
    final_top_left: tuple[int, int],
    arrow: dict[str, object],
    blocked: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    dr, dc = arrow["direction"]
    next_top_left = (int(final_top_left[0] + dr), int(final_top_left[1] + dc))
    next_cells = _translate_cells(next_top_left, arrow["occupied"])
    random.shuffle(next_cells)

    candidates: list[list[tuple[int, int]]] = []
    axes = [(0, 1), (1, 0)]
    for block_cell in next_cells:
        random.shuffle(axes)
        for axis_r, axis_c in axes:
            signs = [-1, 1]
            random.shuffle(signs)
            for sign in signs:
                ray: list[tuple[int, int]] = []
                for dist in range(SEGMENT_LEN_RANGE[1]):
                    rr = int(block_cell[0] + axis_r * sign * dist)
                    cc = int(block_cell[1] + axis_c * sign * dist)
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    ray.append((rr, cc))
                preferred_lengths = list(
                    range(
                        min(len(ray), SEGMENT_LEN_RANGE[1]),
                        SEGMENT_LEN_RANGE[0] - 1,
                        -1,
                    )
                )
                random.shuffle(preferred_lengths)
                for length in preferred_lengths:
                    segment = ray[:length]
                    if _segment_cells_ok(segment, blocked, obstacles, h, w):
                        candidates.append(segment)
    if not candidates:
        return None
    return random.choice(candidates)


def _place_free_segment(
    reserved: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        length = random.randint(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1])
        if random.random() < 0.5:
            rr = random.randint(0, h - 1)
            cc0 = random.randint(0, w - length)
            segment = [(int(rr), int(cc0 + k)) for k in range(length)]
        else:
            rr0 = random.randint(0, h - length)
            cc = random.randint(0, w - 1)
            segment = [(int(rr0 + k), int(cc)) for k in range(length)]
        if _segment_cells_ok(segment, reserved, obstacles, h, w):
            return segment
    return None


def _balanced_stop_kinds(count: int) -> list[str]:
    if count == 1:
        return [random.choice(["boundary", "obstacle"])]
    obstacle_count = random.choice([count // 2, (count + 1) // 2])
    obstacle_count = max(1, min(count - 1, obstacle_count))
    kinds = ["obstacle"] * obstacle_count + ["boundary"] * (count - obstacle_count)
    random.shuffle(kinds)
    return kinds


def _try_place_arrow(
    stop_kind: str,
    color: int,
    reserved: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> dict[str, object] | None:
    missing_keys = list(ARROW_SHAPES.keys())
    for _ in range(PLACEMENT_ATTEMPTS):
        missing = random.choice(missing_keys)
        shape = ARROW_SHAPES[missing]
        arrow = {
            "missing": missing,
            "direction": shape["direction"],
            "occupied": shape["occupied"],
            "stamp": shape["stamp"],
            "color": int(color),
        }
        final_top_left = (random.randint(0, h - 3), random.randint(0, w - 3))
        final_bbox = _bbox_cells(final_top_left)
        if final_bbox & reserved or final_bbox & obstacles:
            continue
        reason_now = _one_step_stop_reason(final_top_left, arrow, obstacles, h, w)
        if stop_kind == "boundary":
            if reason_now != "boundary":
                continue
        else:
            if reason_now is not None:
                continue
        path_options = _candidate_paths(final_top_left, arrow, reserved, obstacles, h, w)
        if not path_options:
            continue
        longer = [item for item in path_options if item[0] >= 2]
        steps, positions, bbox_union = random.choice(longer or path_options)
        blocker_segment = None
        if stop_kind == "obstacle":
            blocker_segment = _choose_blocker_segment(
                final_top_left,
                arrow,
                reserved | bbox_union,
                obstacles,
                h,
                w,
            )
            if blocker_segment is None:
                continue
        start_top_left = positions[0]
        arrow.update(
            {
                "top_left": start_top_left,
                "final_top_left": final_top_left,
                "stop_kind": stop_kind,
                "steps": int(steps),
                "path_bboxes": bbox_union,
                "blocker_segment": blocker_segment or [],
            }
        )
        return arrow
    return None


def _render_grid(
    arrows: list[dict[str, object]],
    obstacles: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for rr, cc in obstacles:
        grid[int(rr), int(cc)] = GREY
    for arrow in arrows:
        for rr, cc in _translate_cells(arrow["top_left"], arrow["occupied"]):
            grid[int(rr), int(cc)] = int(arrow["color"])
    return grid


def _validate_generated(grid: np.ndarray, arrows: list[dict[str, object]]) -> bool:
    found = _find_arrows(grid)
    if len(found) != len(arrows):
        return False
    obstacle_set = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    expected = {
        (
            int(arrow["top_left"][0]),
            int(arrow["top_left"][1]),
            int(arrow["color"]),
        ): arrow
        for arrow in arrows
    }
    for found_arrow in found:
        key = (
            int(found_arrow["top_left"][0]),
            int(found_arrow["top_left"][1]),
            int(found_arrow["color"]),
        )
        if key not in expected:
            return False
        spec = expected[key]
        final_top_left, reason = _simulate_projection(
            found_arrow["top_left"],
            found_arrow,
            obstacle_set,
            grid.shape[0],
            grid.shape[1],
        )
        if reason != spec["stop_kind"]:
            return False
        if final_top_left != spec["final_top_left"]:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[grid == GREY], grid[grid == GREY]):
        return False
    if int(np.count_nonzero(out == MAROON)) != 4 * len(arrows):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    arrow_cap = _area_arrow_cap(h * w)
    arrow_count = random.randint(1, arrow_cap)
    stop_kinds = _balanced_stop_kinds(arrow_count)
    colors = random.sample(ARROW_COLORS, arrow_count)

    arrows: list[dict[str, object]] = []
    reserved: set[tuple[int, int]] = set()
    obstacles: set[tuple[int, int]] = set()

    for stop_kind, color in zip(stop_kinds, colors, strict=False):
        arrow = _try_place_arrow(stop_kind, color, reserved, obstacles, h, w)
        if arrow is None:
            return None
        arrows.append(arrow)
        reserved |= arrow["path_bboxes"]
        obstacles |= set(arrow["blocker_segment"])

    extra_segments = random.randint(*EXTRA_SEGMENT_RANGE)
    if not obstacles:
        extra_segments = max(extra_segments, 1)
    for _ in range(extra_segments):
        segment = _place_free_segment(reserved, obstacles, h, w)
        if segment is None:
            if not obstacles:
                return None
            break
        obstacles |= set(segment)

    grid = _render_grid(arrows, obstacles, h, w)
    if not _validate_generated(grid, arrows):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
