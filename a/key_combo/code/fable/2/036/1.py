from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
MOVABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = set()
    stack = [tuple(map(int, pts[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def find_components(
    grid: np.ndarray,
    *,
    bg: int = BG,
    colors: tuple[int, ...] | list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if colors is None:
        palette = sorted(int(v) for v in np.unique(g) if int(v) != int(bg))
    else:
        palette = [int(v) for v in colors if int(v) != int(bg)]

    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for color in palette:
        locs = np.argwhere(g == color)
        for start_r, start_c in locs:
            sr = int(start_r)
            sc = int(start_c)
            if seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            coords: list[tuple[int, int]] = []
            seen[sr, sc] = True
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _right_slide_order(component: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return (-int(c1), -int(c0), int(r0), int(r1), -int(component["area"]))


def _find_walls(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    walls = []
    for c in range(w):
        col = g[:, c]
        if not np.all((col == BG) | (col == GREY)):
            continue
        if int(col[0]) != GREY or int(col[-1]) != GREY:
            continue
        if not np.any(col == BG):
            continue
        openings = []
        row_to_opening = np.full(h, -1, dtype=int)
        r = 0
        while r < h:
            if int(col[r]) == BG:
                start = r
                while r + 1 < h and int(col[r + 1]) == BG:
                    r += 1
                end = r
                idx = len(openings)
                openings.append((int(start), int(end)))
                row_to_opening[start : end + 1] = idx
            r += 1
        walls.append({"col": int(c), "openings": openings, "row_to_opening": row_to_opening})
    walls.sort(key=lambda wall: int(wall["col"]))
    assert 1 <= len(walls) <= 2, "expected one or two gate walls"
    return walls


def _opening_index_for_span(wall: dict, row0: int, row1: int) -> int | None:
    for idx, (start, end) in enumerate(wall["openings"]):
        if int(start) <= int(row0) and int(row1) <= int(end):
            return int(idx)
    return None


def _clear_movable(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    mask = (out != BG) & (out != GREY)
    out[mask] = BG
    return out


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == BG, "expected black background"

    walls = _find_walls(g)
    wall_by_col = {int(wall["col"]): idx for idx, wall in enumerate(walls)}
    leftmost_wall = int(walls[0]["col"])

    palette = [int(v) for v in np.unique(g) if int(v) not in {BG, GREY}]
    components = find_components(g, bg=bg, colors=palette)
    assert components, "expected movable objects"
    for component in components:
        assert int(component["bbox"][3]) < leftmost_wall, "movable objects must start left of the first wall"

    static_grid = _clear_movable(g)
    static_grey = static_grid == GREY
    out = static_grid.copy()
    occupied = np.zeros_like(g, dtype=bool)
    sealed: set[tuple[int, int]] = set()
    trace_objects = []
    h, w = g.shape

    for component in sorted(components, key=_right_slide_order):
        color = int(component["color"])
        current = np.asarray(component["coords"], dtype=int).copy()
        used_openings: set[tuple[int, int]] = set()
        steps = 0
        row0 = int(component["bbox"][0])
        row1 = int(component["bbox"][2])
        fit_openings = tuple(_opening_index_for_span(wall, row0, row1) for wall in walls)

        stop_reason = "edge"
        blocker = None
        while True:
            nxt = current.copy()
            nxt[:, 1] += 1
            if int(nxt[:, 1].max()) >= w:
                stop_reason = "edge"
                blocker = None
                break

            blocked = None
            for rr, cc in nxt:
                r = int(rr)
                c = int(cc)
                if occupied[r, c]:
                    blocked = ("settled", (r, c))
                    break
                if bool(static_grey[r, c]):
                    wall_idx = wall_by_col.get(c)
                    if wall_idx is None:
                        blocked = ("grey_block", (r, c))
                    else:
                        blocked = ("wall", (int(wall_idx), r))
                    break
                wall_idx = wall_by_col.get(c)
                if wall_idx is not None:
                    opening_idx = int(walls[wall_idx]["row_to_opening"][r])
                    if opening_idx >= 0 and (int(wall_idx), opening_idx) in sealed:
                        blocked = ("sealed", (int(wall_idx), opening_idx))
                        break

            if blocked is not None:
                stop_reason, blocker = blocked
                break

            current = nxt
            steps += 1
            for rr, cc in current:
                wall_idx = wall_by_col.get(int(cc))
                if wall_idx is None:
                    continue
                opening_idx = int(walls[wall_idx]["row_to_opening"][int(rr)])
                if opening_idx >= 0:
                    used_openings.add((int(wall_idx), opening_idx))

        out[current[:, 0], current[:, 1]] = color
        occupied[current[:, 0], current[:, 1]] = True
        for key in used_openings:
            sealed.add(key)

        trace_objects.append(
            {
                "color": color,
                "initial_bbox": tuple(map(int, component["bbox"])),
                "final_bbox": bbox_coords(current),
                "row_span": (row0, row1),
                "steps": int(steps),
                "fit_openings": fit_openings,
                "used_openings": tuple(sorted(used_openings)),
                "stop_reason": stop_reason,
                "blocker": blocker,
            }
        )

    assert int(np.sum(out != bg)) == int(np.sum(g != bg)), "movement must preserve pixel count"
    return out, {"walls": walls, "objects": trace_objects}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored object right so it can pass only through matching gate-wall openings, while used openings seal behind it.

    Input structure:
    - The grid has a black background, one or two vertical grey gate walls, and optional extra grey fixed blocks.
    - Each gate wall occupies a full column, with one or more contiguous background openings cut through that column.
    - Several disconnected monochromatic movable objects sit entirely to the left of the first wall.

    Rule:
    - Process movable objects from right to left by their initial bounding boxes.
    - Each object keeps its shape and moves one column right at a time while every next occupied cell is background.
    - Grey wall cells block movement, but an object may enter a wall column through an opening only when its full row span is contained inside that opening.
    - After an object has used an opening and settles, that opening becomes closed for all later objects; movement otherwise stops at the wall, a grey fixed block, the right edge, or an already settled object.

    Key insight:
    - The puzzle is not just ordinary right-sliding: whether an object can pass depends on opening containment in the wall's own row coordinates, and every successful passage changes later reachability by sealing that specific gate.
    """
    return _solve_with_trace(grid)[0]


def _distribute(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(int(total)):
        parts[random.randrange(slots)] += 1
    return parts


def _sample_openings(height: int, count: int, *, require_large: bool) -> list[tuple[int, int]]:
    for _ in range(200):
        lengths = [random.randint(2, 4) for _ in range(count)]
        if require_large and max(lengths) < 3:
            lengths[random.randrange(count)] = random.randint(3, 4)
        minimum = 2 + sum(lengths) + (count - 1)
        if minimum > height:
            continue
        slack = height - minimum
        gaps = [1 + extra for extra in _distribute(slack, count + 1)]
        row = gaps[0]
        openings = []
        for idx, length in enumerate(lengths):
            openings.append((int(row), int(row + length - 1)))
            row += length + gaps[idx + 1]
        return openings
    raise RuntimeError("failed to sample gate openings")


def _intervals_inside(span: tuple[int, int], *, max_height: int = 3) -> list[tuple[int, int]]:
    start, end = map(int, span)
    out = []
    limit = min(int(max_height), end - start + 1)
    for height in range(1, limit + 1):
        for row0 in range(start, end - height + 2):
            out.append((int(row0), int(row0 + height - 1)))
    return out


def _intersection(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    start = max(int(a[0]), int(b[0]))
    end = min(int(a[1]), int(b[1]))
    if start > end:
        return None
    return int(start), int(end)


def _disjoint_spans(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return int(a[1]) < int(b[0]) or int(b[1]) < int(a[0])


def _span_in_openings(span: tuple[int, int], openings: list[tuple[int, int]]) -> tuple[int, int] | None:
    row0, row1 = map(int, span)
    for idx, opening in enumerate(openings):
        if int(opening[0]) <= row0 and row1 <= int(opening[1]):
            return int(idx), row1 - row0 + 1
    return None


def _choose_same_gate_pair(opening: tuple[int, int], *, require_disjoint: bool = True) -> tuple[tuple[int, int], tuple[int, int]] | None:
    spans = _intervals_inside(opening, max_height=2)
    pairs = []
    for first in spans:
        for second in spans:
            if first == second:
                continue
            overlap = not (int(first[1]) < int(second[0]) or int(second[1]) < int(first[0]))
            if require_disjoint and overlap:
                continue
            pairs.append((first, second))
    if not pairs and require_disjoint:
        return _choose_same_gate_pair(opening, require_disjoint=False)
    if not pairs:
        return None
    return random.choice(pairs)


def _boundary_cross_span(height: int, opening: tuple[int, int]) -> tuple[int, int] | None:
    start, end = map(int, opening)
    options = []
    if start - 1 >= 0:
        options.append((int(start - 1), int(start)))
    if end + 1 < height:
        options.append((int(end), int(end + 1)))
    if not options:
        if start - 1 >= 0:
            options.append((int(start - 1), int(start - 1)))
        if end + 1 < height:
            options.append((int(end + 1), int(end + 1)))
    if not options:
        return None
    return random.choice(options)


def _make_sprite(height: int, width: int, color: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    candidates = [mask.copy()]
    if height >= 2 and width >= 2:
        for rr, cc in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
            candidate = mask.copy()
            candidate[rr, cc] = False
            if _connected_mask(candidate) and np.all(candidate.sum(axis=0) >= 1) and np.all(candidate.sum(axis=1) >= 1):
                candidates.append(candidate)
    chosen = random.choice(candidates)
    sprite = np.full((height, width), BG, dtype=int)
    sprite[chosen] = int(color)
    return sprite


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any(target[mask] != BG), "patch placement must not overlap"
    target[mask] = patch[mask]


def _choose_one_wall_plan(height: int, openings: list[tuple[int, int]]) -> list[dict] | None:
    primary_openings = [idx for idx, opening in enumerate(openings) if int(opening[1] - opening[0] + 1) >= 3]
    others = list(range(len(openings)))
    random.shuffle(primary_openings)
    for primary_idx in primary_openings:
        pair = _choose_same_gate_pair(openings[primary_idx], require_disjoint=True)
        if pair is None:
            continue
        primary_span, blocked_span = pair
        secondary_choices = [idx for idx in others if idx != primary_idx]
        random.shuffle(secondary_choices)
        for secondary_idx in secondary_choices:
            secondary_spans = _intervals_inside(openings[secondary_idx], max_height=3)
            if not secondary_spans:
                continue
            alt_span = random.choice(secondary_spans)
            fail_span = _boundary_cross_span(height, openings[primary_idx])
            if fail_span is None:
                continue
            plan = [
                {
                    "role": "primary",
                    "span": primary_span,
                    "expect_used": {(0, int(primary_idx))},
                },
                {
                    "role": "alt_pass",
                    "span": alt_span,
                    "expect_used": {(0, int(secondary_idx))},
                },
                {
                    "role": "blocked_same_gate",
                    "span": blocked_span,
                    "expect_stop": ("sealed", (0, int(primary_idx))),
                    "expect_used": set(),
                },
                {
                    "role": "wall_fail",
                    "span": fail_span,
                    "expect_stop": ("wall", 0),
                    "expect_used": set(),
                },
            ]

            extra_pair = _choose_same_gate_pair(openings[secondary_idx], require_disjoint=True)
            if extra_pair is not None and random.random() < 0.55:
                _, extra_blocked_span = extra_pair
                if extra_blocked_span != alt_span and _disjoint_spans(extra_blocked_span, alt_span):
                    plan.append(
                        {
                            "role": "extra_blocked",
                            "span": extra_blocked_span,
                            "expect_stop": ("sealed", (0, int(secondary_idx))),
                            "expect_used": set(),
                        }
                    )
            return plan
    return None


def _choose_two_wall_plan(
    height: int,
    openings1: list[tuple[int, int]],
    openings2: list[tuple[int, int]],
) -> list[dict] | None:
    shared_candidates = []
    for idx1, opening1 in enumerate(openings1):
        if int(opening1[1] - opening1[0] + 1) < 3:
            continue
        for idx2, opening2 in enumerate(openings2):
            overlap = _intersection(opening1, opening2)
            if overlap is None:
                continue
            for span in _intervals_inside(overlap, max_height=2):
                shared_candidates.append((int(idx1), int(idx2), span))
    random.shuffle(shared_candidates)

    for primary_idx1, primary_idx2, primary_span in shared_candidates:
        blocked_options = []
        for span in _intervals_inside(openings1[primary_idx1], max_height=2):
            if span == primary_span:
                continue
            if span[1] < primary_span[0] or primary_span[1] < span[0]:
                blocked_options.append(span)
        if not blocked_options:
            blocked_options = [span for span in _intervals_inside(openings1[primary_idx1], max_height=2) if span != primary_span]
        if not blocked_options:
            continue

        first_other = [idx for idx in range(len(openings1)) if idx != primary_idx1]
        random.shuffle(first_other)
        for between_idx in first_other:
            between_spans = [
                span for span in _intervals_inside(openings1[between_idx], max_height=3) if _span_in_openings(span, openings2) is None
            ]
            if not between_spans:
                continue

            third_choices = [idx for idx in first_other if idx != between_idx]
            random.shuffle(third_choices)
            for alt_idx1 in third_choices:
                alt_options = []
                for alt_idx2, opening2 in enumerate(openings2):
                    if alt_idx2 == primary_idx2:
                        continue
                    overlap = _intersection(openings1[alt_idx1], opening2)
                    if overlap is None:
                        continue
                    for span in _intervals_inside(overlap, max_height=3):
                        alt_options.append((int(alt_idx2), span))
                if not alt_options:
                    continue

                fail_span = _boundary_cross_span(height, openings1[primary_idx1])
                if fail_span is None:
                    continue
                alt_idx2, alt_span = random.choice(alt_options)
                return [
                    {
                        "role": "primary",
                        "span": primary_span,
                        "expect_used": {(0, int(primary_idx1)), (1, int(primary_idx2))},
                    },
                    {
                        "role": "between_fail",
                        "span": random.choice(between_spans),
                        "expect_used": {(0, int(between_idx))},
                        "expect_stop": ("wall", 1),
                    },
                    {
                        "role": "alt_pass",
                        "span": alt_span,
                        "expect_used": {(0, int(alt_idx1)), (1, int(alt_idx2))},
                    },
                    {
                        "role": "blocked_same_gate",
                        "span": random.choice(blocked_options),
                        "expect_used": set(),
                        "expect_stop": ("sealed", (0, int(primary_idx1))),
                    },
                    {
                        "role": "wall_fail",
                        "span": fail_span,
                        "expect_used": set(),
                        "expect_stop": ("wall", 0),
                    },
                ]
    return None


def _assign_geometry(plan: list[dict]) -> tuple[list[int], list[int], int]:
    widths = [random.randint(1, 2) for _ in plan]
    gaps = [random.randint(1, 2) for _ in range(len(plan) - 1)]
    need = int(sum(widths) + sum(gaps) + random.randint(2, 4))
    return widths, gaps, need


def _build_grid(
    height: int,
    n_walls: int,
    openings1: list[tuple[int, int]],
    openings2: list[tuple[int, int]] | None,
    plan: list[dict],
) -> np.ndarray | None:
    widths, gaps, wall1_col = _assign_geometry(plan)
    if n_walls == 1:
        last_wall = wall1_col
        width = last_wall + random.randint(7, 11)
        wall_cols = [wall1_col]
    else:
        corridor = random.randint(4, 6)
        wall2_col = wall1_col + corridor
        last_wall = wall2_col
        width = last_wall + random.randint(7, 10)
        wall_cols = [wall1_col, wall2_col]
    if width > 30 or height > 30:
        return None

    colors = random.sample(MOVABLE_COLORS, len(plan))
    grid = np.zeros((height, width), dtype=int)
    for wall_col, openings in zip(wall_cols, [openings1, openings2]):
        if openings is None:
            continue
        grid[:, wall_col] = GREY
        for row0, row1 in openings:
            grid[row0 : row1 + 1, wall_col] = BG

    right = wall1_col - 2
    for idx, spec in enumerate(plan):
        span = spec["span"]
        obj_height = int(span[1] - span[0] + 1)
        obj_width = int(widths[idx])
        left = right - obj_width + 1
        if left < 0:
            return None
        spec["color"] = int(colors[idx])
        spec["left"] = int(left)
        patch = _make_sprite(obj_height, obj_width, spec["color"])
        _paint_patch(grid, patch, int(span[0]), int(left))
        if idx < len(plan) - 1:
            right = left - gaps[idx] - 1

    pass_roles = [spec for spec in plan if spec["role"] in {"primary", "alt_pass"}]
    if pass_roles and random.random() < 0.35 and last_wall + 3 < width - 1:
        target = random.choice(pass_roles)
        row0, row1 = map(int, target["span"])
        col = random.randint(last_wall + 2, width - 2)
        grid[row0 : row1 + 1, col] = GREY

    return grid


def _validate_plan(grid: np.ndarray, plan: list[dict], n_walls: int) -> bool:
    out, trace = _solve_with_trace(grid)
    if np.array_equal(out, grid):
        return False

    by_color = {int(obj["color"]): obj for obj in trace["objects"]}
    if len(by_color) != len(plan):
        return False
    if sum(int(obj["steps"]) > 0 for obj in trace["objects"]) < 2:
        return False

    for spec in plan:
        obj = by_color.get(int(spec["color"]))
        if obj is None:
            return False
        used = set(tuple(item) for item in obj["used_openings"])
        if used != set(spec["expect_used"]):
            return False
        expect_stop = spec.get("expect_stop")
        if expect_stop is None:
            if obj["stop_reason"] in {"wall", "sealed"}:
                return False
        else:
            reason, detail = expect_stop
            if obj["stop_reason"] != reason:
                return False
            if reason == "sealed":
                if tuple(obj["blocker"]) != tuple(detail):
                    return False
            elif reason == "wall":
                blocker = obj["blocker"]
                if blocker is None or int(blocker[0]) != int(detail):
                    return False

    if n_walls == 2:
        second_wall_interactions = 0
        for obj in trace["objects"]:
            if any(int(wall_idx) == 1 for wall_idx, _ in obj["used_openings"]):
                second_wall_interactions += 1
            elif obj["stop_reason"] in {"wall", "sealed"} and obj["blocker"] is not None and int(obj["blocker"][0]) == 1:
                second_wall_interactions += 1
        if second_wall_interactions == 0:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_walls = 1 if random.random() < 0.45 else 2
    if n_walls == 1:
        height = random.randint(10, 16)
        opening_count = 2 if height < 11 else random.randint(2, 3)
        openings1 = _sample_openings(height, opening_count, require_large=True)
        plan = _choose_one_wall_plan(height, openings1)
        if plan is None:
            return None
        grid = _build_grid(height, n_walls, openings1, None, plan)
    else:
        height = random.randint(12, 18)
        openings1 = _sample_openings(height, 3, require_large=True)
        openings2 = _sample_openings(height, random.randint(2, 3), require_large=False)
        plan = _choose_two_wall_plan(height, openings1, openings2)
        if plan is None:
            return None
        grid = _build_grid(height, n_walls, openings1, openings2, plan)
    if grid is None:
        return None
    if not _validate_plan(grid, plan, n_walls):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-wall input")
