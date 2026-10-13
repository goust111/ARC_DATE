from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int = GREY) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _fill_interior(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 + 1 : r1, c0 + 1 : c1] = color


def _frame_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return 2 * h + 2 * w - 4


def _frame_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for coords in _components_mask(grid == GREY):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "grey components must be nontrivial hollow rectangles"
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(sub[border] == GREY), "rectangle border must be solid grey"
        assert np.all(sub[~border] == bg), "rectangle interior must be background in the input"
        assert int(coords.shape[0]) == int(border.sum()), "grey component must be one-cell thick"
        frames.append({"bbox": (r0, c0, r1, c1)})
    assert frames, "expected at least one grey hollow rectangle"
    return frames


def _render_rect_id_grid(shape: tuple[int, int], frames: list[dict | tuple[int, int, int, int]]) -> np.ndarray:
    ids = np.full(shape, -1, dtype=int)
    for idx, frame in enumerate(frames):
        bbox = frame["bbox"] if isinstance(frame, dict) else frame
        r0, c0, r1, c1 = bbox
        ids[r0, c0 : c1 + 1] = idx
        ids[r1, c0 : c1 + 1] = idx
        ids[r0 : r1 + 1, c0] = idx
        ids[r0 : r1 + 1, c1] = idx
    return ids


def _trace_ray(
    rect_ids: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict:
    h, w = rect_ids.shape
    r, c = map(int, start)
    dr, dc = map(int, direction)
    assert abs(dr) == 1 and abs(dc) == 1
    assert 0 <= r < h and 0 <= c < w

    first_hits: list[dict] = []
    all_hits: list[dict] = []
    trail = [(r, c)]
    hit_seen: set[int] = set()
    state_seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in state_seen, "beam must not loop forever"
        state_seen.add(state)

        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            break

        hit_h_id = int(rect_ids[r + dr, c]) if rect_ids[r + dr, c] >= 0 else -1
        hit_v_id = int(rect_ids[r, c + dc]) if rect_ids[r, c + dc] >= 0 else -1

        if hit_h_id < 0 and hit_v_id < 0:
            assert rect_ids[r + dr, c + dc] < 0, "valid instances avoid corner-only wall entry"
            r += dr
            c += dc
            trail.append((r, c))
            continue

        hit_ids = {rid for rid in (hit_h_id, hit_v_id) if rid >= 0}
        assert len(hit_ids) == 1, "a bounce may involve only one rectangle at a time"
        rect_id = next(iter(hit_ids))
        dir_before = (dr, dc)

        if hit_h_id >= 0:
            dr = -dr
        if hit_v_id >= 0:
            dc = -dc

        kind = "corner" if hit_h_id >= 0 and hit_v_id >= 0 else "horizontal" if hit_h_id >= 0 else "vertical"
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            assert rect_ids[nr, nc] < 0, "beam must leave a bounce cell back into empty space"

        event = {
            "rect_id": rect_id,
            "kind": kind,
            "at": (r, c),
            "dir_before": dir_before,
            "dir_after": (dr, dc),
            "next_pos": (nr, nc),
        }
        all_hits.append(event)
        if rect_id not in hit_seen:
            hit_seen.add(rect_id)
            first_hits.append(event)

        if not (0 <= nr < h and 0 <= nc < w):
            break
        r, c = nr, nc
        trail.append((r, c))

    return {"first_hits": first_hits, "all_hits": all_hits, "trail": trail}


def _legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    colors = [int(grid[r, 0]) for r in range(grid.shape[0]) if int(grid[r, 0]) not in (bg, GREY)]
    assert colors, "expected a nonempty left-edge legend"
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return colors


def _launcher_pair(grid: np.ndarray, bg: int, legend: list[int]) -> tuple[int, tuple[int, int], tuple[int, int]]:
    remaining = set(map(int, np.unique(grid))) - {bg, GREY, *legend}
    assert len(remaining) == 1, "expected one launcher color distinct from the legend"
    color = next(iter(remaining))
    coords = np.argwhere(grid == color)
    assert coords.shape == (2, 2), "launcher must contain exactly two same-color pixels"
    a = tuple(map(int, coords[0]))
    b = tuple(map(int, coords[1]))
    assert abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1, "launcher pixels must be diagonal neighbors"
    return int(color), a, b


def _choose_orientation(shape: tuple[int, int], frames: list[dict], a: tuple[int, int], b: tuple[int, int]) -> dict:
    rect_ids = _render_rect_id_grid(shape, frames)
    candidates = []
    for base, cap in ((a, b), (b, a)):
        direction = (cap[0] - base[0], cap[1] - base[1])
        try:
            trace = _trace_ray(rect_ids, cap, direction)
        except AssertionError:
            continue
        candidates.append({"base": base, "cap": cap, "dir": direction, "trace": trace})

    positive = [cand for cand in candidates if cand["trace"]["first_hits"]]
    assert len(positive) == 1, "the launcher orientation must be uniquely determined by the scene"
    return positive[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill grey hollow rectangles by the invisible diagonal ray's first-hit order, using the left legend colors.

    Input structure:
    - The background is a mostly black canvas containing several one-cell-thick grey hollow rectangles.
    - The left edge holds a vertical legend of distinct colors, ordered from top to bottom.
    - One non-grey color appears exactly twice as a diagonal two-pixel launcher made of a base pixel and a same-color cap pixel.

    Rule:
    - Launch an invisible diagonal ray in the cap's direction and let grey rectangles act as mirrors.
    - Hitting a horizontal side flips the vertical component; hitting a vertical side flips the horizontal component; the ray stops on reaching the grid edge.
    - On the first hit of the k-th distinct rectangle, fill that rectangle's interior with the k-th legend color; later hits of the same rectangle do not advance the legend.
    - Leave the ray itself, the launcher, the legend, the grey borders, and any never-hit rectangles unchanged.

    Key insight:
    - The grey frames serve both as reflectors and as the objects to recolor, so the output order is determined by an entirely invisible bounce path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0
    bg = _get_bg(g)
    frames = _extract_frames(g, bg)
    legend = _legend_colors(g, bg)
    assert len(legend) >= len(frames), "legend must list at least as many colors as rectangles"
    _launcher_color, a, b = _launcher_pair(g, bg, legend)

    chosen = _choose_orientation(g.shape, frames, a, b)
    hits = chosen["trace"]["first_hits"]
    assert hits, "the ray must hit at least one rectangle"
    assert len(hits) <= len(legend)

    out = g.copy()
    for idx, event in enumerate(hits):
        _fill_interior(out, frames[event["rect_id"]]["bbox"], legend[idx])
    return out


GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (22, 28)
VISITED_COUNT_RANGE = (3, 3)
EXTRA_COUNT_RANGE = (1, 2)
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
TRAVEL_STEPS_RANGE = (2, 5)
STEP_RETRIES = 500
EXTRA_RETRIES = 400
GEN_RETRIES = 300


def _launcher_spec(h: int, w: int, direction: tuple[int, int]) -> dict:
    if direction == (1, 1):
        return {
            "base": (2, 2),
            "cap": (3, 3),
            "reserve": (0, 0, 5, 5),
            "blocked_legend_rows": set(range(0, 6)),
        }
    if direction == (-1, 1):
        return {
            "base": (h - 3, 2),
            "cap": (h - 4, 3),
            "reserve": (h - 6, 0, h - 1, 5),
            "blocked_legend_rows": set(range(h - 6, h)),
        }
    if direction == (1, -1):
        return {
            "base": (2, w - 3),
            "cap": (3, w - 4),
            "reserve": (0, w - 6, 5, w - 1),
            "blocked_legend_rows": set(range(0, 6)),
        }
    assert direction == (-1, -1)
    return {
        "base": (h - 3, w - 3),
        "cap": (h - 4, w - 4),
        "reserve": (h - 6, w - 6, h - 1, w - 1),
        "blocked_legend_rows": set(range(h - 6, h)),
    }


def _bbox_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _bbox_clear(
    bbox: tuple[int, int, int, int],
    h: int,
    w: int,
    frames: list[tuple[int, int, int, int]],
    reserve: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    if not (1 <= r0 < r1 <= h - 2 and 2 <= c0 < c1 <= w - 2):
        return False
    if _bbox_too_close(bbox, reserve, gap=0):
        return False
    return all(not _bbox_too_close(bbox, other, gap=1) for other in frames)


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + k * dr, c + k * dc) for k in range(1, steps + 1)]


def _steps_to_edge(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    dr, dc = direction
    steps_r = (h - 1 - r) if dr > 0 else r
    steps_c = (w - 1 - c) if dc > 0 else c
    return int(min(steps_r, steps_c))


def _random_bbox_for_hit(
    impact: tuple[int, int],
    direction: tuple[int, int],
    kind: str,
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    r, c = impact
    dr, dc = direction
    if kind == "vertical":
        border_c = c + dc
        if dc == 1:
            c0, c1 = border_c, border_c + w - 1
        else:
            c0, c1 = border_c - w + 1, border_c
        r0_lo = r - (h - 2)
        r0_hi = r - 1
        if r0_lo > r0_hi:
            return None
        r0 = random.randint(r0_lo, r0_hi)
        return r0, c0, r0 + h - 1, c1

    border_r = r + dr
    if dr == 1:
        r0, r1 = border_r, border_r + h - 1
    else:
        r0, r1 = border_r - h + 1, border_r
    c0_lo = c - (w - 2)
    c0_hi = c - 1
    if c0_lo > c0_hi:
        return None
    c0 = random.randint(c0_lo, c0_hi)
    return r0, c0, r1, c0 + w - 1


def _candidate_bboxes_for_hit(
    shape: tuple[int, int],
    impact: tuple[int, int],
    direction: tuple[int, int],
    kind: str,
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    out = []
    for hh in range(FRAME_H_RANGE[0], FRAME_H_RANGE[1] + 1):
        for ww in range(FRAME_W_RANGE[0], FRAME_W_RANGE[1] + 1):
            if kind == "vertical":
                border_c = impact[1] + direction[1]
                if direction[1] == 1:
                    c0, c1 = border_c, border_c + ww - 1
                else:
                    c0, c1 = border_c - ww + 1, border_c
                r0_lo = impact[0] - (hh - 2)
                r0_hi = impact[0] - 1
                for r0 in range(r0_lo, r0_hi + 1):
                    out.append((r0, c0, r0 + hh - 1, c1))
            else:
                border_r = impact[0] + direction[0]
                if direction[0] == 1:
                    r0, r1 = border_r, border_r + hh - 1
                else:
                    r0, r1 = border_r - hh + 1, border_r
                c0_lo = impact[1] - (ww - 2)
                c0_hi = impact[1] - 1
                for c0 in range(c0_lo, c0_hi + 1):
                    out.append((r0, c0, r1, c0 + ww - 1))
    return out


def _scene_input(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    legend: list[int],
    legend_row0: int,
    launcher_color: int,
    base: tuple[int, int],
    cap: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for bbox in frames:
        _draw_frame(grid, bbox, GREY)
    for idx, color in enumerate(legend):
        grid[legend_row0 + idx, 0] = color
    grid[base] = launcher_color
    grid[cap] = launcher_color
    return grid


def _expected_output(
    inp: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    first_hits: list[dict],
    legend: list[int],
) -> np.ndarray:
    out = inp.copy()
    for idx, event in enumerate(first_hits):
        _fill_interior(out, frames[event["rect_id"]], legend[idx])
    return out


def _simple_orders_ok(frames: list[tuple[int, int, int, int]], hit_ids: list[int], start: tuple[int, int]) -> bool:
    def bbox(i: int) -> tuple[int, int, int, int]:
        return frames[i]

    def center_dist(i: int) -> float:
        cr, cc = _frame_center(bbox(i))
        return abs(cr - start[0]) + abs(cc - start[1])

    orders = [
        sorted(hit_ids, key=lambda i: (bbox(i)[0], bbox(i)[1])),
        sorted(hit_ids, key=lambda i: (bbox(i)[1], bbox(i)[0])),
        sorted(hit_ids, key=lambda i: (center_dist(i), bbox(i)[0], bbox(i)[1])),
        sorted(hit_ids, key=lambda i: (_frame_area(bbox(i)), bbox(i)[0], bbox(i)[1])),
    ]
    return all(order != hit_ids for order in orders)


def _pick_legend_rows(
    h: int,
    legend_len: int,
    blocked_rows: set[int],
    trail: list[tuple[int, int]],
) -> int | None:
    edge_rows_used = {r for r, c in trail if c == 0}
    options = []
    for row0 in range(0, h - legend_len + 1):
        rows = set(range(row0, row0 + legend_len))
        if rows & blocked_rows:
            continue
        if rows & edge_rows_used:
            continue
        options.append(row0)
    return random.choice(options) if options else None


def _forward_and_reverse_traces(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    base: tuple[int, int],
    cap: tuple[int, int],
) -> tuple[dict, dict] | None:
    rect_ids = _render_rect_id_grid(shape, frames)
    try:
        forward = _trace_ray(rect_ids, cap, (cap[0] - base[0], cap[1] - base[1]))
        reverse = _trace_ray(rect_ids, base, (base[0] - cap[0], base[1] - cap[1]))
    except AssertionError:
        return None
    return forward, reverse


def _search_next_hit(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    current: tuple[int, int],
    direction: tuple[int, int],
    desired_kind: str,
    remaining_after: int,
    reserve: tuple[int, int, int, int],
    base: tuple[int, int],
    cap: tuple[int, int],
    avoid_points: set[tuple[int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], str] | None:
    h, w = shape

    step_options = list(range(TRAVEL_STEPS_RANGE[0], TRAVEL_STEPS_RANGE[1] + 1))
    random.shuffle(step_options)
    for steps in step_options:
        impact = (current[0] + steps * direction[0], current[1] + steps * direction[1])
        if not (1 <= impact[0] < h - 1 and 1 <= impact[1] < w - 1):
            continue
        segment = _segment_cells(current, direction, steps)
        if any(cell in avoid_points for cell in segment):
            continue

        bboxes = _candidate_bboxes_for_hit(shape, impact, direction, desired_kind)
        random.shuffle(bboxes)
        for bbox in bboxes:
            if not _bbox_clear(bbox, h, w, frames, reserve):
                continue

            trial_frames = frames + [bbox]
            traces = _forward_and_reverse_traces(shape, trial_frames, base, cap)
            if traces is None:
                continue
            forward, _reverse = traces
            first_hits = forward["first_hits"]
            if [event["rect_id"] for event in first_hits] != list(range(len(trial_frames))):
                continue
            event = first_hits[-1]
            if event["kind"] != desired_kind:
                continue
            if event["next_pos"][0] in (-1, h) or event["next_pos"][1] in (-1, w):
                continue
            if _steps_to_edge(shape, event["next_pos"], event["dir_after"]) < max(1, 2 * remaining_after):
                continue
            return bbox, event["next_pos"], event["dir_after"], desired_kind
    return None


def _add_distractors(
    shape: tuple[int, int],
    frames: list[tuple[int, int, int, int]],
    extra_count: int,
    reserve: tuple[int, int, int, int],
    base: tuple[int, int],
    cap: tuple[int, int],
    expected_hit_count: int,
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    out = list(frames)
    for _ in range(extra_count):
        placed = False
        for _ in range(EXTRA_RETRIES):
            hh = random.randint(*FRAME_H_RANGE)
            ww = random.randint(*FRAME_W_RANGE)
            bbox = (
                random.randint(1, h - hh - 1),
                random.randint(2, w - ww - 1),
                0,
                0,
            )
            bbox = (bbox[0], bbox[1], bbox[0] + hh - 1, bbox[1] + ww - 1)
            if not _bbox_clear(bbox, h, w, out, reserve):
                continue
            trial_frames = out + [bbox]
            traces = _forward_and_reverse_traces(shape, trial_frames, base, cap)
            if traces is None:
                continue
            forward, reverse = traces
            if len(forward["first_hits"]) != expected_hit_count:
                continue
            if [event["rect_id"] for event in forward["first_hits"]] != list(range(expected_hit_count)):
                continue
            if reverse["first_hits"]:
                continue
            out.append(bbox)
            placed = True
            break
        if not placed:
            return None
    return out


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    direction = random.choice(DIAG_DIRS)
    launch = _launcher_spec(h, w, direction)
    base = launch["base"]
    cap = launch["cap"]
    reserve = launch["reserve"]
    visited_target = random.randint(*VISITED_COUNT_RANGE)
    extra_count = random.randint(*EXTRA_COUNT_RANGE)

    frames: list[tuple[int, int, int, int]] = []
    current = cap
    cur_dir = direction
    avoid_points = {base, cap}
    required_kinds = {"vertical", "horizontal"}

    for step_idx in range(visited_target):
        kind_choices = list(required_kinds) if visited_target - step_idx == len(required_kinds) else []
        kind_choices += ["vertical", "horizontal"]
        random.shuffle(kind_choices)
        found = None
        for desired_kind in kind_choices:
            found = _search_next_hit(
                (h, w),
                frames,
                current,
                cur_dir,
                desired_kind,
                visited_target - step_idx - 1,
                reserve,
                base,
                cap,
                avoid_points,
            )
            if found is not None:
                break
        if found is None:
            return None
        bbox, current, cur_dir, kind = found
        frames.append(bbox)
        required_kinds.discard(kind)

    if required_kinds:
        return None

    frames = _add_distractors((h, w), frames, extra_count, reserve, base, cap, visited_target)
    if frames is None:
        return None

    traces = _forward_and_reverse_traces((h, w), frames, base, cap)
    if traces is None:
        return None
    forward, reverse = traces
    first_hits = forward["first_hits"]
    hit_ids = [event["rect_id"] for event in first_hits]

    if len(first_hits) != visited_target:
        return None
    if reverse["first_hits"]:
        return None
    if {event["kind"] for event in first_hits} != {"vertical", "horizontal"}:
        return None
    if not _simple_orders_ok(frames, hit_ids, cap):
        return None

    total_frames = len(frames)
    if total_frames > len(PALETTE) - 1:
        return None

    palette = random.sample(PALETTE, total_frames + 1)
    launcher_color = palette[0]
    legend = palette[1:]
    if len(legend) < total_frames:
        return None

    legend_row0 = _pick_legend_rows(h, len(legend), launch["blocked_legend_rows"], forward["trail"])
    if legend_row0 is None:
        return None

    inp = _scene_input((h, w), frames, legend, legend_row0, launcher_color, base, cap)
    expected = _expected_output(inp, frames, first_hits, legend)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid launcher-bounce coloring input")
