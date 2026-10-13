from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sort_frames(frames: list[dict]) -> list[dict]:
    return sorted(frames, key=lambda fr: fr["bbox"])


def _gap_info(
    gap: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top", (-1, 0), (r0 + 1, c)
    if r == r1:
        return "bottom", (1, 0), (r1 - 1, c)
    if c == c0:
        return "left", (0, -1), (r, c0 + 1)
    return "right", (0, 1), (r, c1 - 1)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GREY):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        comp_set = set(comp)
        missing: list[tuple[int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if not on_border:
                    continue
                if (r, c) not in comp_set:
                    missing.append((r, c))
        assert len(missing) == 1, "each frame must have exactly one border gap"
        gap = missing[0]
        assert gap[0] not in (r0, r1) or gap[1] not in (c0, c1), "gap cannot be a corner"
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in comp)
        frames.append({"bbox": bbox, "gap": gap})
    assert 2 <= len(frames) <= 3
    return _sort_frames(frames)


def _frame_red_row(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    red_rows = [
        r
        for r in range(r0 + 1, r1)
        if np.all(grid[r, c0 + 1 : c1] == RED)
    ]
    assert len(red_rows) == 1, "expected one full-width red baseline row per frame"
    return int(red_rows[0])


def _column_profile(
    grid: np.ndarray, red_row: int, bottom_inner: int, col: int
) -> tuple[int, int]:
    segment = grid[red_row + 1 : bottom_inner + 1, col]
    non_black = np.where(segment != BLACK)[0]
    assert non_black.size > 0, "each column must contain a two-color bar below the baseline"
    first = int(non_black[0])
    assert np.all(segment[:first] == BLACK)
    active = segment[first:]
    assert active[0] == TEAL
    teal_height = 0
    while teal_height < len(active) and int(active[teal_height]) == TEAL:
        teal_height += 1
    assert 0 < teal_height < len(active), "each bar needs both teal and blue segments"
    assert np.all(active[:teal_height] == TEAL)
    assert np.all(active[teal_height:] == BLUE)
    return first, teal_height


def _normalize_frame(
    out: np.ndarray, grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    red_row = _frame_red_row(grid, bbox)
    bottom_inner = r1 - 1
    teal_heights = [
        _column_profile(grid, red_row, bottom_inner, c)[1]
        for c in range(c0 + 1, c1)
    ]
    h_max = max(teal_heights)
    assert red_row + h_max <= bottom_inner, "normalized teal band must leave space for blue"
    for c in range(c0 + 1, c1):
        out[red_row + 1 : bottom_inner + 1, c] = BLUE
        out[red_row + 1 : red_row + 1 + h_max, c] = TEAL
    out[red_row, c0 + 1 : c1] = RED
    return red_row, h_max


def _solve(grid: np.ndarray, *, normalize: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    frames = _find_frames(g)

    if normalize:
        for frame in frames:
            _normalize_frame(out, g, frame["bbox"])
    base = out.copy() if normalize else g

    for frame in frames:
        side, step, inner = _gap_info(frame["gap"], frame["bbox"])
        _ = side
        ray_color = int(base[inner])
        r, c = frame["gap"][0] + step[0], frame["gap"][1] + step[1]
        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            if int(base[r, c]) != BLACK:
                break
            out[r, c] = ray_color
            r += step[0]
            c += step[1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize the teal bar heights below each red baseline, then cast a ray whose color is the post-normalization gap-adjacent color.

    Input structure:
    - The grid has a black background and 2-3 grey one-pixel hollow rectangular frames, each with exactly one non-corner border gap.
    - Inside each frame, every interior column contains a bottom-aligned two-color vertical bar: teal above blue, with optional black space above the bar.
    - Each frame also contains one full-width red baseline row strictly above the bars.

    Rule:
    - For each frame, find the red baseline row and compute the maximum teal height across its interior columns.
    - Replace every interior column below the red row by a normalized bar: teal for exactly that maximum height immediately below the red row, then blue down to the inner bottom edge; keep the red row unchanged.
    - After normalization, inspect the pixel just inside the frame at the gap, then cast a straight ray outward from the cell beyond the gap; recolor only black background cells on that ray with the inspected color until a non-black cell or the grid edge stops the ray.

    Key insight:
    - The ray color is not attached to the frame itself; it is determined by whichever bar color becomes adjacent to the gap only after the baseline-driven teal normalization.
    """
    return _solve(grid, normalize=True)


def _sample_int(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _sample_frame(kind: str) -> dict | None:
    inner_h = _sample_int(6, 10)
    inner_w = _sample_int(3, 5)
    if kind == "change":
        h_max = _sample_int(2, 4)
    else:
        h_max = _sample_int(1, 4)
    pre_gap = _sample_int(0, 1)

    forced_top_red = kind == "stable_red" and _RNG.random() < 0.35
    if forced_top_red:
        side = "top"
        red_rel = 0
    elif kind == "stable_red":
        side = "left" if _RNG.random() < 0.5 else "right"
        red_cap = inner_h - h_max - pre_gap - 2
        if red_cap < 0:
            return None
        red_rel = _sample_int(0, min(3, red_cap))
    elif kind == "change":
        side = "left" if _RNG.random() < 0.5 else "right"
        red_cap = inner_h - h_max - pre_gap - 2
        if red_cap < 0:
            return None
        red_rel = _sample_int(0, min(3, red_cap))
    else:
        side = "bottom"
        red_cap = inner_h - h_max - pre_gap - 2
        if red_cap < 0:
            return None
        red_rel = _sample_int(0, min(3, red_cap))

    starts = [red_rel + 1 + pre_gap + _sample_int(0, 1) for _ in range(inner_w)]
    max_allowed = [inner_h - start - 1 for start in starts]
    if any(limit < 1 for limit in max_allowed):
        return None
    candidates = [i for i in range(inner_w) if starts[i] == red_rel + 1 + pre_gap and max_allowed[i] >= h_max]
    if not candidates:
        return None

    if side in ("left", "right"):
        edge_col = 0 if side == "left" else inner_w - 1
    else:
        edge_col = _sample_int(0, inner_w - 1)

    if kind == "change":
        non_edge = [i for i in candidates if i != edge_col]
        if not non_edge:
            return None
        anchor_col = int(non_edge[_sample_int(0, len(non_edge) - 1)])
    else:
        anchor_col = int(candidates[_sample_int(0, len(candidates) - 1)])

    teal_heights = [_sample_int(1, min(h_max, max_allowed[i])) for i in range(inner_w)]
    teal_heights[anchor_col] = h_max

    if kind == "change":
        if max_allowed[edge_col] < 1:
            return None
        starts[edge_col] = red_rel + 1 + pre_gap
        max_allowed[edge_col] = inner_h - starts[edge_col] - 1
        if max_allowed[edge_col] < 1 or h_max <= 1:
            return None
        teal_heights[edge_col] = _sample_int(1, min(h_max - 1, max_allowed[edge_col]))
        gap_row_rel = starts[edge_col] + teal_heights[edge_col]
        if gap_row_rel >= inner_h:
            return None
        gap_index = gap_row_rel
    elif kind == "stable_red":
        gap_index = 0 if side == "top" else red_rel
    else:
        gap_index = edge_col

    return {
        "kind": kind,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "side": side,
        "red_rel": red_rel,
        "h_max": h_max,
        "starts": starts,
        "teal_heights": teal_heights,
        "gap_index": gap_index,
    }


def _frame_bbox_at(spec: dict, top: int, left: int) -> tuple[int, int, int, int]:
    return top, left, top + spec["inner_h"] + 1, left + spec["inner_w"] + 1


def _gap_at(spec: dict, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    side = spec["side"]
    if side == "top":
        return r0, c0 + 1 + spec["gap_index"]
    if side == "bottom":
        return r1, c0 + 1 + spec["gap_index"]
    if side == "left":
        return r0 + 1 + spec["gap_index"], c0
    return r0 + 1 + spec["gap_index"], c1


def _ray_cells(shape: tuple[int, int], gap: tuple[int, int], side: str) -> list[tuple[int, int]]:
    steps = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = steps[side]
    h, w = shape
    r, c = gap[0] + dr, gap[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    order = sorted(range(len(specs)), key=lambda i: (specs[i]["inner_h"] + 2) * (specs[i]["inner_w"] + 2), reverse=True)
    placements: list[dict | None] = [None] * len(specs)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_ray_sets: list[set[tuple[int, int]]] = []

    def search(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        spec = specs[idx]
        bbox_h = spec["inner_h"] + 2
        bbox_w = spec["inner_w"] + 2
        candidates = [
            (r0, c0)
            for r0 in range(1, h - bbox_h)
            for c0 in range(1, w - bbox_w)
        ]
        _RNG.shuffle(candidates)
        for top, left in candidates:
            bbox = _frame_bbox_at(spec, top, left)
            if any(not _boxes_separated(bbox, other, 1) for other in placed_boxes):
                continue
            gap = _gap_at(spec, bbox)
            ray = _ray_cells(shape, gap, spec["side"])
            if not ray:
                continue
            ray_set = set(ray)
            if any(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in ray for r0, c0, r1, c1 in placed_boxes):
                continue
            if any(rr >= bbox[0] and rr <= bbox[2] and cc >= bbox[1] and cc <= bbox[3] for rr, cc in ray):
                continue
            if any(ray_set & other for other in placed_ray_sets):
                continue
            placed_boxes.append(bbox)
            placed_ray_sets.append(ray_set)
            placements[idx] = {"bbox": bbox, "gap": gap, "ray": ray}
            if search(k + 1):
                return True
            placements[idx] = None
            placed_boxes.pop()
            placed_ray_sets.pop()
        return False

    if not search(0):
        return None
    return [pl for pl in placements if pl is not None]


def _render_input(shape: tuple[int, int], specs: list[dict], placements: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec, placed in zip(specs, placements):
        r0, c0, r1, c1 = placed["bbox"]
        gap = placed["gap"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[gap] = BLACK

        red_row = r0 + 1 + spec["red_rel"]
        grid[red_row, c0 + 1 : c1] = RED

        for i, c in enumerate(range(c0 + 1, c1)):
            start = r0 + 1 + spec["starts"][i]
            teal_h = spec["teal_heights"][i]
            grid[start : start + teal_h, c] = TEAL
            grid[start + teal_h : r1, c] = BLUE
    return grid


def _frame_summary(
    inp: np.ndarray, out: np.ndarray, frame: dict
) -> tuple[int, int, str, int]:
    bbox = frame["bbox"]
    gap = frame["gap"]
    side, _step, inner = _gap_info(gap, bbox)
    before = int(inp[inner])
    after = int(out[inner])
    return before, after, side, after


def _validate_sample(inp: np.ndarray) -> bool:
    out = main(inp)
    naive = _solve(inp, normalize=False)
    if np.array_equal(out, naive):
        return False

    frames = _find_frames(inp)
    changed = 0
    unchanged = 0
    saw_non_teal_stable = False
    ray_cells_seen: set[tuple[int, int]] = set()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        red_row = _frame_red_row(inp, frame["bbox"])
        teal_heights = [_column_profile(inp, red_row, r1 - 1, c)[1] for c in range(c0 + 1, c1)]
        h_max = max(teal_heights)
        if red_row + h_max > r1 - 1:
            return False

        before, after, side, ray_color = _frame_summary(inp, out, frame)
        if ray_color == BLACK:
            return False
        gap = frame["gap"]
        ray = _ray_cells(inp.shape, gap, side)
        if not ray:
            return False
        if ray_cells_seen & set(ray):
            return False
        ray_cells_seen |= set(ray)

        steps = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
        dr, dc = steps[side]
        rr, cc = gap[0] + dr, gap[1] + dc
        while 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
            if inp[rr, cc] != BLACK:
                break
            if int(out[rr, cc]) != ray_color:
                return False
            rr += dr
            cc += dc

        if before == BLUE and after == TEAL:
            changed += 1
        if before == after:
            unchanged += 1
            if after in (BLUE, RED):
                saw_non_teal_stable = True

    return changed >= 1 and unchanged >= 1 and saw_non_teal_stable


def _generate_once() -> np.ndarray | None:
    shape = (_sample_int(18, 24), _sample_int(18, 24))
    frame_count = 2 if _RNG.random() < 0.45 else 3

    stable_kind = "stable_red" if _RNG.random() < 0.5 else "stable_blue"
    kinds = ["change", stable_kind]
    extra_pool = ["change", "stable_blue", "stable_red"]
    while len(kinds) < frame_count:
        kinds.append(str(extra_pool[_sample_int(0, len(extra_pool) - 1)]))
    _RNG.shuffle(kinds)

    specs: list[dict] = []
    for kind in kinds:
        spec = _sample_frame(kind)
        if spec is None:
            return None
        specs.append(spec)

    placements = _place_specs(shape, specs)
    if placements is None:
        return None

    inp = _render_input(shape, specs, placements)
    if not _validate_sample(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
