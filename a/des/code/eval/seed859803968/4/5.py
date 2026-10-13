from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

EMITTER_COLORS = (RED, BLUE, YELLOW)
COLOR_TO_BIT = {
    RED: 1,
    BLUE: 2,
    YELLOW: 4,
}
BIT_TO_COLOR = {
    1: RED,
    2: BLUE,
    3: MAGENTA,
    4: YELLOW,
    5: ORANGE,
    6: GREEN,
    7: MAROON,
}

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

TOP = "top"
BOTTOM = "bottom"
LEFT_SIDE = "left"
RIGHT_SIDE = "right"
SIDES = (TOP, BOTTOM, LEFT_SIDE, RIGHT_SIDE)

MAX_GEN_ATTEMPTS = 600


@dataclass(frozen=True)
class RayPlan:
    color: int
    anchor: tuple[int, int]
    dir_in: tuple[int, int]
    dir_out: tuple[int, int]
    start_side: str
    end_side: str


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _side_of_dir(d: tuple[int, int]) -> str:
    if d == UP:
        return TOP
    if d == DOWN:
        return BOTTOM
    if d == LEFT:
        return LEFT_SIDE
    if d == RIGHT:
        return RIGHT_SIDE
    raise ValueError(f"unknown direction {d}")


def _opposite_dir(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (-dr, -dc)


def _opposite_side(side: str) -> str:
    return {
        TOP: BOTTOM,
        BOTTOM: TOP,
        LEFT_SIDE: RIGHT_SIDE,
        RIGHT_SIDE: LEFT_SIDE,
    }[side]


def _perpendicular_sides(d: tuple[int, int]) -> tuple[str, str]:
    if d in (UP, DOWN):
        return (LEFT_SIDE, RIGHT_SIDE)
    return (TOP, BOTTOM)


def _reflect_slash(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (-dc, -dr)


def _reflect_backslash(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (dc, dr)


def _orientation_for_turn(
    dir_in: tuple[int, int], dir_out: tuple[int, int]
) -> str | None:
    if dir_out == _reflect_slash(dir_in):
        return "/"
    if dir_out == _reflect_backslash(dir_in):
        return "\\"
    return None


def _reflect(orientation: str, d: tuple[int, int]) -> tuple[int, int]:
    if orientation == "/":
        return _reflect_slash(d)
    if orientation == "\\":
        return _reflect_backslash(d)
    raise ValueError(f"unknown mirror orientation {orientation}")


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 != r1 and c0 != c1:
        raise ValueError("segment must be axis-aligned")
    if (r0, c0) == (r1, c1):
        raise ValueError("segment endpoint must move")

    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
    else:
        step = 1 if r1 > r0 else -1
        for r in range(r0, r1 + step, step):
            cells.append((r, c0))
    return cells


def _point_on_segment(
    p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]
) -> bool:
    pr, pc = p
    (r0, c0), (r1, c1) = a, b
    if r0 == r1 == pr:
        lo, hi = sorted((c0, c1))
        return lo <= pc <= hi
    if c0 == c1 == pc:
        lo, hi = sorted((r0, r1))
        return lo <= pr <= hi
    return False


def _direction_between(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[int, int]:
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    return (dr, dc)


def _find_emitters(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    emitters: list[tuple[int, int, int, int, int]] = []

    for c in range(1, w - 1):
        color = int(g[0, c])
        if color in EMITTER_COLORS:
            emitters.append((0, c, DOWN[0], DOWN[1], color))
    for c in range(1, w - 1):
        color = int(g[h - 1, c])
        if color in EMITTER_COLORS:
            emitters.append((h - 1, c, UP[0], UP[1], color))
    for r in range(1, h - 1):
        color = int(g[r, 0])
        if color in EMITTER_COLORS:
            emitters.append((r, 0, RIGHT[0], RIGHT[1], color))
    for r in range(1, h - 1):
        color = int(g[r, w - 1])
        if color in EMITTER_COLORS:
            emitters.append((r, w - 1, LEFT[0], LEFT[1], color))
    return emitters


def _find_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    mirrors: dict[tuple[int, int], str] = {}

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != GREY:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != GREY:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            assert len(comp) == 2, "each mirror must be a 2-pixel diagonal component"
            rows = sorted(r0 for r0, _c0 in comp)
            cols = sorted(c0 for _r0, c0 in comp)
            cell_set = set(comp)
            diag_main = {(rows[0], cols[0]), (rows[1], cols[1])}
            diag_other = {(rows[0], cols[1]), (rows[1], cols[0])}
            if cell_set == diag_main:
                orientation = "\\"
            else:
                assert cell_set == diag_other, "mirror must be diagonal"
                orientation = "/"
            for cell in comp:
                mirrors[cell] = orientation
    return mirrors


def _simulate(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mirrors = _find_mirrors(g)
    emitters = _find_emitters(g)
    visit_mask = np.zeros((h, w), dtype=np.uint8)
    ray_infos: list[dict[str, object]] = []

    for r0, c0, dr0, dc0, color in emitters:
        r, c = int(r0), int(c0)
        dr, dc = int(dr0), int(dc0)
        seen_states: set[tuple[int, int, int, int]] = set()
        path: list[tuple[int, int]] = []
        reflections = 0
        looped = False

        while True:
            state = (r, c, dr, dc)
            if state in seen_states:
                looped = True
                break
            seen_states.add(state)

            nr = r + dr
            nc = c + dc
            if not _in_bounds(g.shape, nr, nc):
                break

            orientation = mirrors.get((nr, nc))
            if orientation is not None:
                dr, dc = _reflect(orientation, (dr, dc))
                reflections += 1
                continue

            r, c = nr, nc
            path.append((r, c))
            if int(g[r, c]) == BLACK:
                visit_mask[r, c] |= COLOR_TO_BIT[color]

        ray_infos.append(
            {
                "emitter": (int(r0), int(c0)),
                "color": int(color),
                "path": path,
                "reflections": int(reflections),
                "looped": bool(looped),
            }
        )
    return visit_mask, ray_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge emitters through diagonal grey mirrors, mixing colors where
    different rays visit the same background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    visit_mask, ray_infos = _simulate(g)
    assert all(not info["looped"] for info in ray_infos), "ray paths must not loop"

    rows, cols = np.where(visit_mask > 0)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        out[r, c] = BIT_TO_COLOR[int(visit_mask[r, c])]
    return out


def _mirror_component_options(
    trigger: tuple[int, int], orientation: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r, c = trigger
    if orientation == "/":
        return [
            (trigger, (r - 1, c + 1)),
            (trigger, (r + 1, c - 1)),
        ]
    assert orientation == "\\"
    return [
        (trigger, (r - 1, c - 1)),
        (trigger, (r + 1, c + 1)),
    ]


def _boundary_point(
    shape: tuple[int, int], side: str, fixed: int
) -> tuple[int, int]:
    h, w = shape
    if side == TOP:
        return (0, fixed)
    if side == BOTTOM:
        return (h - 1, fixed)
    if side == LEFT_SIDE:
        return (fixed, 0)
    if side == RIGHT_SIDE:
        return (fixed, w - 1)
    raise ValueError(f"unknown side {side}")


def _sample_before_anchor(
    shape: tuple[int, int], anchor: tuple[int, int], d: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = shape
    ar, ac = anchor
    if d == DOWN:
        choices = list(range(1, ar))
        return None if not choices else (random.choice(choices), ac)
    if d == UP:
        choices = list(range(ar + 1, h - 1))
        return None if not choices else (random.choice(choices), ac)
    if d == RIGHT:
        choices = list(range(1, ac))
        return None if not choices else (ar, random.choice(choices))
    if d == LEFT:
        choices = list(range(ac + 1, w - 1))
        return None if not choices else (ar, random.choice(choices))
    return None


def _sample_after_anchor(
    shape: tuple[int, int], anchor: tuple[int, int], d: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = shape
    ar, ac = anchor
    if d == DOWN:
        choices = list(range(ar + 1, h - 1))
        return None if not choices else (random.choice(choices), ac)
    if d == UP:
        choices = list(range(1, ar))
        return None if not choices else (random.choice(choices), ac)
    if d == RIGHT:
        choices = list(range(ac + 1, w - 1))
        return None if not choices else (ar, random.choice(choices))
    if d == LEFT:
        choices = list(range(1, ac))
        return None if not choices else (ar, random.choice(choices))
    return None


def _points_for_plan(
    shape: tuple[int, int], plan: RayPlan
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    anchor = plan.anchor
    dir_in = plan.dir_in
    dir_out = plan.dir_out
    points: list[tuple[int, int]] = []

    if plan.start_side == _opposite_side(_side_of_dir(dir_in)):
        if dir_in in (UP, DOWN):
            emitter = _boundary_point(shape, plan.start_side, anchor[1])
        else:
            emitter = _boundary_point(shape, plan.start_side, anchor[0])
    else:
        if plan.start_side not in _perpendicular_sides(dir_in):
            return None
        pre = _sample_before_anchor(shape, anchor, dir_in)
        if pre is None:
            return None
        fixed = pre[0] if plan.start_side in (LEFT_SIDE, RIGHT_SIDE) else pre[1]
        emitter = _boundary_point(shape, plan.start_side, fixed)
        points.append(pre)

    if dir_in != dir_out:
        points.append(anchor)

    if plan.end_side == _side_of_dir(dir_out):
        if dir_out in (UP, DOWN):
            endpoint = _boundary_point(shape, plan.end_side, anchor[1])
        else:
            endpoint = _boundary_point(shape, plan.end_side, anchor[0])
    else:
        if plan.end_side not in _perpendicular_sides(dir_out):
            return None
        post = _sample_after_anchor(shape, anchor, dir_out)
        if post is None:
            return None
        points.append(post)
        fixed = post[0] if plan.end_side in (LEFT_SIDE, RIGHT_SIDE) else post[1]
        endpoint = _boundary_point(shape, plan.end_side, fixed)

    full = [emitter] + points + [endpoint]
    for a, b in zip(full, full[1:], strict=False):
        if a == b:
            return None
        if a[0] != b[0] and a[1] != b[1]:
            return None
    cells: list[tuple[int, int]] = []
    for idx, (a, b) in enumerate(zip(full, full[1:], strict=False)):
        seg = _segment_cells(a, b)
        if idx > 0:
            seg = seg[1:]
        cells.extend(seg)

    if anchor not in cells:
        return None
    if any(not _in_bounds(shape, r, c) for r, c in cells):
        return None
    return full, cells


def _collect_turns(
    points: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    turns: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for idx in range(1, len(points) - 1):
        p = points[idx]
        dir_in = _direction_between(points[idx - 1], p)
        dir_out = _direction_between(p, points[idx + 1])
        if dir_in == dir_out:
            continue
        turns.append((p, dir_in, dir_out))
    return turns


def _can_place_component(
    shape: tuple[int, int],
    component: tuple[tuple[int, int], tuple[int, int]],
    blocked: set[tuple[int, int]],
    placed: set[tuple[int, int]],
) -> bool:
    comp_set = set(component)
    for r, c in component:
        if not _in_bounds(shape, r, c):
            return False
        if _is_edge(shape, r, c):
            return False
        if (r, c) in blocked or (r, c) in placed:
            return False
    for r, c in component:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in placed and (nr, nc) not in comp_set:
                return False
    return True


def _place_mirrors_for_turns(
    shape: tuple[int, int],
    turns: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
    blocked: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    choices: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
    for corner, dir_in, dir_out in turns:
        orientation = _orientation_for_turn(dir_in, dir_out)
        if orientation is None:
            return None
        trigger = (corner[0] + dir_in[0], corner[1] + dir_in[1])
        opts = _mirror_component_options(trigger, orientation)
        choices.append(opts)

    placed: list[tuple[tuple[int, int], tuple[int, int]]] = []
    used_cells: set[tuple[int, int]] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(choices):
            return True
        for comp in choices[idx]:
            if not _can_place_component(shape, comp, blocked, used_cells):
                continue
            placed.append(comp)
            used_cells.update(comp)
            if backtrack(idx + 1):
                return True
            placed.pop()
            for cell in comp:
                used_cells.remove(cell)
        return False

    if not backtrack(0):
        return None
    return placed


def _render_plans(
    shape: tuple[int, int], plans: list[RayPlan]
) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    emitter_positions: dict[tuple[int, int], int] = {}
    all_cells: list[set[tuple[int, int]]] = []
    all_points: list[list[tuple[int, int]]] = []

    for plan in plans:
        built = _points_for_plan(shape, plan)
        if built is None:
            return None
        points, cells = built
        emitter = points[0]
        if emitter in emitter_positions:
            return None
        if emitter[0] in (0, h - 1) and emitter[1] in (0, w - 1):
            return None
        emitter_positions[emitter] = plan.color
        all_points.append(points)
        all_cells.append(set(cells))

    emitters_set = set(emitter_positions)
    for idx, cell_set in enumerate(all_cells):
        for emitter in emitters_set - {all_points[idx][0]}:
            if emitter in cell_set:
                return None

    blocked: set[tuple[int, int]] = set().union(*all_cells, emitters_set)
    all_turns: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for points in all_points:
        all_turns.extend(_collect_turns(points))
    mirrors = _place_mirrors_for_turns(shape, all_turns, blocked)
    if mirrors is None:
        return None

    for pos, color in emitter_positions.items():
        grid[pos] = color
    for comp in mirrors:
        for r, c in comp:
            if int(grid[r, c]) != BLACK:
                return None
            grid[r, c] = GREY
    return grid


def _sample_anchor(shape: tuple[int, int], margin: int) -> tuple[int, int] | None:
    h, w = shape
    if h <= 2 * margin or w <= 2 * margin:
        return None
    return (
        random.randint(margin, h - margin - 1),
        random.randint(margin, w - margin - 1),
    )


def _build_pair_plans(shape: tuple[int, int]) -> list[RayPlan] | None:
    anchor = _sample_anchor(shape, margin=4)
    if anchor is None:
        return None
    colors = random.sample(list(EMITTER_COLORS), 2)
    v_dir = random.choice([UP, DOWN])
    h_dir = random.choice([LEFT, RIGHT])
    v_end = (
        random.choice(_perpendicular_sides(v_dir))
        if random.random() < 0.5
        else _side_of_dir(v_dir)
    )
    h_end = (
        random.choice(_perpendicular_sides(h_dir))
        if random.random() < 0.5
        else _side_of_dir(h_dir)
    )
    return [
        RayPlan(colors[0], anchor, v_dir, v_dir, random.choice(_perpendicular_sides(v_dir)), v_end),
        RayPlan(colors[1], anchor, h_dir, h_dir, random.choice(_perpendicular_sides(h_dir)), h_end),
    ]


def _build_triple_plans(shape: tuple[int, int]) -> list[RayPlan] | None:
    anchor = _sample_anchor(shape, margin=4)
    if anchor is None:
        return None
    colors = random.sample(list(EMITTER_COLORS), 3)
    top_turn = random.choice([LEFT, RIGHT])
    bottom_turn = LEFT if top_turn == RIGHT else RIGHT
    top_start = TOP if random.random() < 0.7 else random.choice((LEFT_SIDE, RIGHT_SIDE))
    bottom_start = BOTTOM if random.random() < 0.7 else random.choice((LEFT_SIDE, RIGHT_SIDE))
    h_dir = random.choice([LEFT, RIGHT])
    h_end = (
        random.choice(_perpendicular_sides(h_dir))
        if random.random() < 0.5
        else _side_of_dir(h_dir)
    )
    return [
        RayPlan(colors[0], anchor, DOWN, top_turn, top_start, _side_of_dir(top_turn)),
        RayPlan(colors[1], anchor, UP, bottom_turn, bottom_start, _side_of_dir(bottom_turn)),
        RayPlan(colors[2], anchor, h_dir, h_dir, random.choice(_perpendicular_sides(h_dir)), h_end),
    ]


def _build_extra_plan(
    shape: tuple[int, int], avoid: tuple[int, int]
) -> RayPlan | None:
    for _ in range(60):
        anchor = _sample_anchor(shape, margin=4)
        if anchor is None:
            return None
        if max(abs(anchor[0] - avoid[0]), abs(anchor[1] - avoid[1])) < 5:
            continue
        color = random.choice(EMITTER_COLORS)
        d = random.choice(list(DIRS))
        return RayPlan(
            color=color,
            anchor=anchor,
            dir_in=d,
            dir_out=d,
            start_side=random.choice(_perpendicular_sides(d)),
            end_side=_side_of_dir(d),
        )
    return None


def _build_quad_plans(shape: tuple[int, int]) -> list[RayPlan] | None:
    triple = _build_triple_plans(shape)
    if triple is None:
        return None
    extra = _build_extra_plan(shape, triple[0].anchor)
    if extra is None:
        return None
    return triple + [extra]


def _count_mirrors(grid: np.ndarray) -> int:
    return len(_find_mirrors(grid)) // 2


def _mix_counts(mask: np.ndarray) -> Counter:
    cnt = Counter()
    for value in mask.flat:
        cnt[int(value)] += 1
    cnt.pop(0, None)
    return cnt


def _is_valid_generated(
    grid: np.ndarray,
    expected_emitters: int,
    want_triple: bool,
) -> bool:
    if grid is None:
        return False

    emitters = _find_emitters(grid)
    if len(emitters) != expected_emitters:
        return False
    if len({(r, c) for r, c, _dr, _dc, _color in emitters}) != expected_emitters:
        return False

    mirror_count = _count_mirrors(grid)
    if not (2 <= mirror_count <= 5):
        return False

    try:
        mask, infos = _simulate(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    if any(info["looped"] for info in infos):
        return False
    if any(len(info["path"]) < 6 for info in infos):
        return False

    total_reflections = sum(int(info["reflections"]) for info in infos)
    if total_reflections < expected_emitters:
        return False

    counts = _mix_counts(mask)
    pair_mixes = counts[3] + counts[5] + counts[6]
    triple_mixes = counts[7]
    colored = sum(counts.values())
    mixed = pair_mixes + triple_mixes
    if colored < 12 or colored > 2 * (grid.shape[0] + grid.shape[1]):
        return False
    if mixed == 0 or mixed > (grid.shape[0] + grid.shape[1]):
        return False
    if want_triple and triple_mixes == 0:
        return False
    if not want_triple and triple_mixes != 0:
        return False
    return True


def _apply_transform(grid: np.ndarray) -> np.ndarray:
    op = random.randint(0, 7)
    g = grid
    if op == 0:
        return g.copy()
    if op == 1:
        return np.rot90(g, 1).copy()
    if op == 2:
        return np.rot90(g, 2).copy()
    if op == 3:
        return np.rot90(g, 3).copy()
    if op == 4:
        return np.fliplr(g).copy()
    if op == 5:
        return np.flipud(g).copy()
    if op == 6:
        return np.transpose(g).copy()
    return np.fliplr(np.rot90(g, 1)).copy()


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(15, 25), random.randint(15, 25))
    family = random.choice(["pair", "triple", "quad"])

    if family == "pair":
        plans = _build_pair_plans(shape)
        want_triple = False
        expected_emitters = 2
    elif family == "triple":
        plans = _build_triple_plans(shape)
        want_triple = True
        expected_emitters = 3
    else:
        plans = _build_quad_plans(shape)
        want_triple = True
        expected_emitters = 4

    if plans is None:
        return None
    grid = _render_plans(shape, plans)
    if grid is None:
        return None

    grid = _apply_transform(grid)
    if not _is_valid_generated(grid, expected_emitters=expected_emitters, want_triple=want_triple):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
