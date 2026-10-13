from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np


BG = 0
ANCHOR = 8
STAIR = 5
RECT_DIMS = ((2, 3), (3, 2))
RECT_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_TRACE_STEPS = 512
GEN_ATTEMPTS = 500
TRACE_OVERLAP_LIMIT = 1


class Rect(NamedTuple):
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == ANCHOR)
    if len(coords) != 1:
        raise AssertionError("expected exactly one teal anchor")
    return int(coords[0, 0]), int(coords[0, 1])


def _extract_rectangles(grid: np.ndarray) -> list[Rect]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    rects: list[Rect] = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color in (BG, ANCHOR) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(cells) != area:
                raise AssertionError("foreground objects must be solid rectangles")
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                raise AssertionError("rectangle bbox must be solid")
            rects.append(Rect(color=color, r0=r0, c0=c0, r1=r1, c1=c1))
    if len(rects) not in (2, 3):
        raise AssertionError("expected two or three solid rectangles")
    colors = [rect.color for rect in rects]
    if len(set(colors)) != len(colors):
        raise AssertionError("rectangle colors must be distinct")
    return rects


def _cell_to_rect(rects: list[Rect]) -> dict[tuple[int, int], Rect]:
    out: dict[tuple[int, int], Rect] = {}
    for rect in rects:
        for r in range(rect.r0, rect.r1 + 1):
            for c in range(rect.c0, rect.c1 + 1):
                out[(r, c)] = rect
    return out


def _trace_arm(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    rect_map: dict[tuple[int, int], Rect],
    *,
    hdir: int,
    vdir: int,
    start_color: int = STAIR,
    max_steps: int = MAX_TRACE_STEPS,
) -> dict:
    H, W = shape
    pos = (int(anchor[0]), int(anchor[1]))
    axis = "H"
    remain = 2
    color = int(start_color)
    drawn: list[tuple[int, int, int]] = []
    successful_states: list[dict] = []
    collisions: list[dict] = []

    for _ in range(max_steps):
        dr, dc = (0, hdir) if axis == "H" else (vdir, 0)
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break

        state = {
            "pos": pos,
            "next": (nr, nc),
            "axis": axis,
            "remain": remain,
            "hdir": hdir,
            "vdir": vdir,
            "color": color,
        }

        rect = rect_map.get((nr, nc))
        if rect is not None:
            if axis == "H":
                hdir *= -1
                edge = "vertical"
            else:
                vdir *= -1
                edge = "horizontal"
            color = int(rect.color)
            collisions.append(
                {
                    "at": (nr, nc),
                    "boundary": pos,
                    "axis": axis,
                    "edge": edge,
                    "rect": rect,
                    "new_hdir": hdir,
                    "new_vdir": vdir,
                    "new_color": color,
                    "drawn_so_far": len(drawn),
                }
            )
            continue

        pos = (nr, nc)
        drawn.append((nr, nc, color))
        successful_states.append(state)
        remain -= 1
        if remain == 0:
            axis = "V" if axis == "H" else "H"
            remain = 2
    else:
        raise AssertionError("trace exceeded safety limit")

    return {
        "drawn": drawn,
        "collisions": collisions,
        "end": pos,
        "successful_states": successful_states,
    }


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    anchor = _find_anchor(g)
    rects = _extract_rectangles(g)
    rect_map = _cell_to_rect(rects)

    out = g.copy()
    traces = {}
    for name, (hdir, vdir) in {"ne": (1, -1), "sw": (-1, 1)}.items():
        trace = _trace_arm(g.shape, anchor, rect_map, hdir=hdir, vdir=vdir)
        traces[name] = trace
        for r, c, color in trace["drawn"]:
            if (r, c) not in rect_map:
                out[r, c] = color
    out[anchor] = ANCHOR
    return out, {"anchor": anchor, "rects": rects, "traces": traces}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw two step-2 staircase rays from the teal anchor, bouncing them off solid rectangles and recoloring post-bounce segments with the rectangle color.

    Input structure:
    - The grid is mostly black and contains exactly one teal anchor pixel.
    - It also contains two or three non-touching solid colored rectangles, each a single distinct color.
    - The anchor is separated from the rectangles and the grid size stays unchanged.

    Rule:
    - From the anchor, trace two opposite staircase rays: one goes right then up in repeated step lengths of two, and the other goes left then down with the same 2-horizontal then 2-vertical rhythm.
    - Whenever the next staircase pixel would land inside a rectangle, do not draw that pixel; instead reflect the current movement component at that rectangle edge, keep the staircase step pattern unchanged, and switch the ray color to that rectangle's color.
    - Continue applying the same bounce-and-recolor rule for later rectangle hits, stop when a ray would leave the grid, and keep the anchor plus all rectangles unchanged.

    Key insight:
    - The output depends on simulating the staircase one pixel at a time: its 2-by-2 stepping order determines which rectangle edge is hit, which component flips, and where the recolored reflected path goes next.
    """
    return _solve_with_trace(np.asarray(grid, dtype=int))[0]


def _expanded_overlap(a: Rect, b: Rect, pad: int = 1) -> bool:
    return not (
        a.r1 + pad < b.r0
        or b.r1 + pad < a.r0
        or a.c1 + pad < b.c0
        or b.c1 + pad < a.c0
    )


def _rect_near_anchor(rect: Rect, anchor: tuple[int, int], pad: int = 1) -> bool:
    ar, ac = anchor
    return rect.r0 - pad <= ar <= rect.r1 + pad and rect.c0 - pad <= ac <= rect.c1 + pad


def _place_rect_for_state(
    shape: tuple[int, int],
    state: dict,
    rects: list[Rect],
    anchor: tuple[int, int],
    color: int,
    rng: random.Random,
) -> list[Rect]:
    H, W = shape
    nr, nc = state["next"]
    axis = state["axis"]
    hdir = state["hdir"]
    vdir = state["vdir"]
    candidates: list[Rect] = []

    dims = list(RECT_DIMS)
    rng.shuffle(dims)
    for height, width in dims:
        if axis == "H":
            c0 = nc if hdir == 1 else nc - width + 1
            for offset in range(height):
                r0 = nr - offset
                rect = Rect(color=color, r0=r0, c0=c0, r1=r0 + height - 1, c1=c0 + width - 1)
                candidates.append(rect)
        else:
            r0 = nr if vdir == 1 else nr - height + 1
            for offset in range(width):
                c0 = nc - offset
                rect = Rect(color=color, r0=r0, c0=c0, r1=r0 + height - 1, c1=c0 + width - 1)
                candidates.append(rect)

    good = []
    for rect in candidates:
        if rect.r0 < 0 or rect.c0 < 0 or rect.r1 >= H or rect.c1 >= W:
            continue
        if _rect_near_anchor(rect, anchor):
            continue
        if any(_expanded_overlap(rect, other, pad=1) for other in rects):
            continue
        good.append(rect)
    rng.shuffle(good)
    return good


def _paint_input(shape: tuple[int, int], anchor: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for rect in rects:
        g[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color
    g[anchor] = ANCHOR
    return g


def _quadrant_ok(anchor: tuple[int, int], state: dict, quadrant: str) -> bool:
    nr, nc = state["next"]
    ar, ac = anchor
    if quadrant == "ur":
        return nr < ar and nc > ac
    if quadrant == "ll":
        return nr > ar and nc < ac
    if quadrant == "ul":
        return nr < ar and nc < ac
    if quadrant == "lr":
        return nr > ar and nc > ac
    raise ValueError(quadrant)


def _find_candidate_state(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    rects: list[Rect],
    *,
    arm: str,
    target_axis: str,
    quadrant: str,
    min_success_before: int,
    min_row_margin: int,
    min_col_margin: int,
) -> list[dict]:
    dirs = {"ne": (1, -1), "sw": (-1, 1)}[arm]
    rect_map = _cell_to_rect(rects)
    trace = _trace_arm(shape, anchor, rect_map, hdir=dirs[0], vdir=dirs[1])
    states = []
    for idx, state in enumerate(trace["successful_states"]):
        nr, nc = state["next"]
        if idx + 1 < min_success_before:
            continue
        if state["axis"] != target_axis:
            continue
        if not _quadrant_ok(anchor, state, quadrant):
            continue
        if not (min_row_margin <= nr < shape[0] - min_row_margin):
            continue
        if not (min_col_margin <= nc < shape[1] - min_col_margin):
            continue
        states.append(state)
    return states


def _path_overlap_ok(info: dict) -> bool:
    anchor = info["anchor"]
    visits: dict[tuple[int, int], int] = {}
    for trace in info["traces"].values():
        for r, c, _ in trace["drawn"]:
            key = (r, c)
            if key == anchor:
                continue
            visits[key] = visits.get(key, 0) + 1
            if visits[key] > TRACE_OVERLAP_LIMIT:
                return False
    return True


def _semantic_ok(inp: np.ndarray, info: dict, plan: dict) -> bool:
    rects = info["rects"]
    traces = info["traces"]
    if len(rects) != plan["rect_count"]:
        return False
    if not _path_overlap_ok(info):
        return False

    for name in ("ne", "sw"):
        if len(traces[name]["collisions"]) < 1:
            return False
        if traces[name]["collisions"][0]["drawn_so_far"] < 3:
            return False

    if plan["extra_arm"] is not None and len(traces[plan["extra_arm"]]["collisions"]) < 2:
        return False

    first_colors = {
        "ne": traces["ne"]["collisions"][0]["rect"].color,
        "sw": traces["sw"]["collisions"][0]["rect"].color,
    }
    if first_colors["ne"] != plan["first_colors"]["ne"]:
        return False
    if first_colors["sw"] != plan["first_colors"]["sw"]:
        return False

    if plan["extra_arm"] is not None:
        second = traces[plan["extra_arm"]]["collisions"][1]["rect"].color
        if second != plan["extra_color"]:
            return False

    for trace in traces.values():
        for event in trace["collisions"]:
            rect = event["rect"]
            cell = event["at"]
            if inp[cell] != rect.color:
                return False

    out, _ = _solve_with_trace(inp)
    if np.array_equal(out, inp):
        return False

    return True


def _construct_case(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(14, 20)
    W = rng.randint(14, 20)
    anchor = (rng.randint(5, H - 6), rng.randint(5, W - 6))

    rect_count = rng.choice((2, 3))
    colors = list(rng.sample(RECT_COLORS, rect_count))
    rng.shuffle(colors)
    first_axes = {"ne": rng.choice(("H", "V")), "sw": rng.choice(("H", "V"))}
    extra_arm = rng.choice(("ne", "sw")) if rect_count == 3 else None
    extra_axis = rng.choice(("H", "V")) if extra_arm is not None else None
    rects: list[Rect] = []

    ne_states = _find_candidate_state(
        (H, W),
        anchor,
        rects,
        arm="ne",
        target_axis=first_axes["ne"],
        quadrant="ur",
        min_success_before=4,
        min_row_margin=2,
        min_col_margin=2,
    )
    if not ne_states:
        return None
    state = rng.choice(ne_states)
    ne_candidates = _place_rect_for_state((H, W), state, rects, anchor, colors[0], rng)
    if not ne_candidates:
        return None
    rects.append(ne_candidates[0])

    sw_states = _find_candidate_state(
        (H, W),
        anchor,
        rects,
        arm="sw",
        target_axis=first_axes["sw"],
        quadrant="ll",
        min_success_before=4,
        min_row_margin=2,
        min_col_margin=2,
    )
    if not sw_states:
        return None
    state = rng.choice(sw_states)
    sw_candidates = _place_rect_for_state((H, W), state, rects, anchor, colors[1], rng)
    if not sw_candidates:
        return None
    rects.append(sw_candidates[0])

    if extra_arm is not None:
        extra_quadrant = {
            ("ne", "H"): "ul",
            ("ne", "V"): "lr",
            ("sw", "H"): "lr",
            ("sw", "V"): "ul",
        }[(extra_arm, first_axes[extra_arm])]
        states = _find_candidate_state(
            (H, W),
            anchor,
            rects,
            arm=extra_arm,
            target_axis=extra_axis,
            quadrant=extra_quadrant,
            min_success_before=8,
            min_row_margin=2,
            min_col_margin=2,
        )
        if not states:
            return None
        state = rng.choice(states)
        extra_candidates = _place_rect_for_state((H, W), state, rects, anchor, colors[2], rng)
        if not extra_candidates:
            return None
        rects.append(extra_candidates[0])

    inp = _paint_input((H, W), anchor, rects)
    out, info = _solve_with_trace(inp)

    plan = {
        "rect_count": rect_count,
        "first_colors": {"ne": colors[0], "sw": colors[1]},
        "extra_arm": extra_arm,
        "extra_color": colors[2] if rect_count == 3 else None,
    }
    if not _semantic_ok(inp, info, plan):
        return None

    used_colors = set(int(v) for v in out.flatten() if v != BG)
    if STAIR not in used_colors:
        return None
    if len(used_colors) < 4:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        grid = _construct_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid staircase-reflection input")
