from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


_SIDE_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_GRID_SIZE_RANGE = (18, 30)
_FRAME_COUNT_RANGE = (4, 6)
_FRAME_HEIGHT_RANGE = (4, 8)
_FRAME_WIDTH_RANGE = (4, 8)
_EDGE_MARGIN = 1
_PLACEMENT_ATTEMPTS = 120
_SCENE_ATTEMPTS = 300
_HIT_SAMPLES_PER_TARGET = 2
_MISS_SAMPLES_PER_SIDE = 1


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(int(lo), int(hi) + 1))


def _bbox_sort_key(item: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = map(int, item["bbox"])
    return r0, c0, r1, c1, int(item["color"])


def _side_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown side {side}")


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = map(int, gap)
    r0, c0, r1, c1 = map(int, bbox)
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the bbox border")


def _paint_border(arr: np.ndarray, bbox: tuple[int, int, int, int], value: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    arr[r0, c0 : c1 + 1] = int(value)
    arr[r1, c0 : c1 + 1] = int(value)
    arr[r0 : r1 + 1, c0] = int(value)
    arr[r0 : r1 + 1, c1] = int(value)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    row_gap = _interval_gap(ar0, ar1, br0, br1)
    col_gap = _interval_gap(ac0, ac1, bc0, bc1)
    return row_gap == 0 and col_gap == 0


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == Color.BLACK

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one colored frame"
    comps = sorted(comps, key=_bbox_sort_key)

    colors = [int(comp["color"]) for comp in comps]
    assert len(set(colors)) == len(colors), "frame colors must be unique"

    owner = np.full(g.shape, -1, dtype=int)
    frames: list[dict] = []
    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame bbox must be at least 3x3"

        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        expected_border: set[tuple[int, int]] = set()
        for r in range(r0, r1 + 1):
            expected_border.add((r, c0))
            expected_border.add((r, c1))
        for c in range(c0, c1 + 1):
            expected_border.add((r0, c))
            expected_border.add((r1, c))

        missing = [cell for cell in sorted(expected_border) if int(g[cell]) == bg]
        assert len(missing) == 1, "each frame must have exactly one border gap"
        gap = missing[0]
        side = _gap_side(gap, comp["bbox"])
        if side in ("top", "bottom"):
            assert c0 < gap[1] < c1, "gap must be non-corner"
        else:
            assert r0 < gap[0] < r1, "gap must be non-corner"

        assert coords == expected_border - {gap}, "frame component must be a rectangle border with one gap"
        assert int(comp["area"]) == len(expected_border) - 1
        if r1 - r0 > 1 and c1 - c0 > 1:
            assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "frame interior must be black"

        for r, c in coords:
            owner[r, c] = idx

        frames.append(
            {
                "index": idx,
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "dir": _SIDE_DIRS[side],
            }
        )

    return bg, frames, owner


def _trace_from_gap(
    grid: np.ndarray,
    owner: np.ndarray,
    gap: tuple[int, int],
    direction: tuple[int, int],
    bg: int,
) -> tuple[int | None, tuple[int, int] | None]:
    h, w = map(int, grid.shape)
    dr, dc = map(int, direction)
    r = int(gap[0] + dr)
    c = int(gap[1] + dc)
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != bg:
            target = int(owner[r, c])
            assert target >= 0
            return target, (r, c)
        r += dr
        c += dc
    return None, None


def _scene_relations(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg, frames, owner = _extract_frames(g)
    incoming: dict[int, int] = {}
    for frame in frames:
        target, hit_cell = _trace_from_gap(g, owner, frame["gap"], frame["dir"], bg)
        frame["target"] = target
        frame["hit_cell"] = hit_cell
        if target is not None:
            assert target != int(frame["index"])
            assert target not in incoming, "each target frame must have at most one incoming ray"
            incoming[target] = int(frame["index"])
    return g, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame gap as a sightline cannon that fills the struck frame and reports the target color back to the source gap.

    Input structure:
    - The grid has a black background with several non-touching monochrome hollow rectangular frames of distinct colors.
    - Every frame border has exactly one missing non-corner cell, and every frame interior is black.
    - Looking straight outward from a gap, the sightline either crosses only black cells and hits another frame border, or leaves the grid.

    Rule:
    - For every frame, follow the ray that starts at its gap and goes outward perpendicular to that border side.
    - If the ray first hits another frame, fill the struck frame interior with the source frame color and recolor the source gap cell with the struck frame color.
    - If the ray leaves the grid without hitting a frame, keep that gap black; never draw the ray itself.

    Key insight:
    - A gap is simultaneously an emitter and an identifier port: it sends its own color into the first frame on its sightline, while the color that seals the gap tells you which frame was hit.
    """
    g, frames = _scene_relations(grid)
    out = g.copy()

    for frame in frames:
        target = frame["target"]
        if target is None:
            continue
        source_color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frames[target]["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1] = source_color

    for frame in frames:
        target = frame["target"]
        if target is None:
            continue
        r, c = map(int, frame["gap"])
        out[r, c] = int(frames[target]["color"])

    return out


def _random_frames(shape: tuple[int, int]) -> list[dict] | None:
    h, w = map(int, shape)
    count = _randint(*_FRAME_COUNT_RANGE)
    colors = [int(c) for c in np.random.choice(np.array(Color.NOT_BLACK), size=count, replace=False)]

    frames: list[dict] = []
    for color in colors:
        placed = False
        for _ in range(_PLACEMENT_ATTEMPTS):
            fh = _randint(*_FRAME_HEIGHT_RANGE)
            fw = _randint(*_FRAME_WIDTH_RANGE)
            if fh + 2 * _EDGE_MARGIN > h or fw + 2 * _EDGE_MARGIN > w:
                return None
            r0 = _randint(_EDGE_MARGIN, h - fh - _EDGE_MARGIN)
            c0 = _randint(_EDGE_MARGIN, w - fw - _EDGE_MARGIN)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_bboxes_too_close(bbox, other["bbox"]) for other in frames):
                continue
            frames.append({"color": int(color), "bbox": bbox})
            placed = True
            break
        if not placed:
            return None

    frames.sort(key=_bbox_sort_key)
    for idx, frame in enumerate(frames):
        frame["index"] = idx
    return frames


def _build_closed_scene(shape: tuple[int, int], frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full(shape, Color.BLACK, dtype=int)
    owner = np.full(shape, -1, dtype=int)
    for frame in frames:
        idx = int(frame["index"])
        _paint_border(grid, frame["bbox"], int(frame["color"]))
        _paint_border(owner, frame["bbox"], idx)
    return grid, owner


def _raw_candidates(shape: tuple[int, int], frames: list[dict]) -> list[list[dict]]:
    grid, owner = _build_closed_scene(shape, frames)
    per_frame: list[list[dict]] = []
    for frame in frames:
        frame_idx = int(frame["index"])
        frame_candidates: list[dict] = []
        for side, direction in _SIDE_DIRS.items():
            for gap in _side_cells(frame["bbox"], side):
                target, hit_cell = _trace_from_gap(grid, owner, gap, direction, Color.BLACK)
                frame_candidates.append(
                    {
                        "source": frame_idx,
                        "gap": gap,
                        "side": side,
                        "dir": direction,
                        "target": target,
                        "hit_cell": hit_cell,
                    }
                )
        assert frame_candidates
        per_frame.append(frame_candidates)
    return per_frame


def _sample_candidates(group: list[dict], limit: int) -> list[dict]:
    if len(group) <= limit:
        return list(group)
    picks = np.random.choice(len(group), size=limit, replace=False)
    return [group[int(i)] for i in picks]


def _reduced_candidates(raw: list[list[dict]]) -> list[list[dict]]:
    reduced: list[list[dict]] = []
    for frame_candidates in raw:
        hits_by_target: dict[int, list[dict]] = {}
        misses_by_side: dict[str, list[dict]] = {}
        for cand in frame_candidates:
            if cand["target"] is None:
                misses_by_side.setdefault(str(cand["side"]), []).append(cand)
            else:
                hits_by_target.setdefault(int(cand["target"]), []).append(cand)

        kept: list[dict] = []
        for target in sorted(hits_by_target):
            kept.extend(_sample_candidates(hits_by_target[target], _HIT_SAMPLES_PER_TARGET))
        for side in sorted(misses_by_side):
            kept.extend(_sample_candidates(misses_by_side[side], _MISS_SAMPLES_PER_SIDE))

        if not kept:
            kept = list(frame_candidates)
        reduced.append(kept)
    return reduced


def _assignment_score(assignment: list[dict], prefer_mutual: bool) -> tuple[int, ...] | None:
    targets = [cand["target"] for cand in assignment]
    hit_pairs = [(i, int(t)) for i, t in enumerate(targets) if t is not None]
    hit_count = len(hit_pairs)
    miss_count = len(assignment) - hit_count
    if hit_count == 0 or miss_count == 0:
        return None

    has_mutual = any(targets[target] == src for src, target in hit_pairs)
    has_one_way = any(targets[target] != src for src, target in hit_pairs)
    receiver_only = any(
        any(target == idx for _src, target in hit_pairs) and targets[idx] is None for idx in range(len(assignment))
    )
    distinct_hit_sides = len({assignment[src]["side"] for src, _target in hit_pairs})
    distinct_all_sides = len({cand["side"] for cand in assignment})
    balance = -abs(hit_count * 2 - len(assignment))

    if prefer_mutual:
        primary = int(has_mutual)
        secondary = int(has_one_way)
    else:
        primary = int(has_one_way)
        secondary = int(has_mutual)

    return (
        primary,
        secondary,
        int(receiver_only),
        distinct_hit_sides,
        distinct_all_sides,
        balance,
        hit_count,
    )


def _choose_assignment(candidates: list[list[dict]]) -> list[dict] | None:
    n = len(candidates)
    order = sorted(range(n), key=lambda idx: (len(candidates[idx]), idx))
    prefer_mutual = bool(_randint(0, 1))

    chosen: list[dict | None] = [None] * n
    incoming_from: list[int | None] = [None] * n
    gap_choice: list[tuple[int, int] | None] = [None] * n
    best_score: tuple[int, ...] | None = None
    best_assignment: list[dict] | None = None

    def can_still_make_hit(pos: int) -> bool:
        return any(any(cand["target"] is not None for cand in candidates[idx]) for idx in order[pos:])

    def can_still_make_miss(pos: int) -> bool:
        return any(any(cand["target"] is None for cand in candidates[idx]) for idx in order[pos:])

    def backtrack(pos: int, hits_so_far: int, misses_so_far: int) -> None:
        nonlocal best_score, best_assignment

        if hits_so_far == 0 and not can_still_make_hit(pos):
            return
        if misses_so_far == 0 and not can_still_make_miss(pos):
            return

        if pos == n:
            assignment = [cand for cand in chosen if cand is not None]
            score = _assignment_score(assignment, prefer_mutual)
            if score is None:
                return
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = assignment
            return

        frame_idx = order[pos]
        permutation = np.random.permutation(len(candidates[frame_idx]))
        for pick in permutation:
            cand = candidates[frame_idx][int(pick)]
            target = cand["target"]
            if target is not None:
                if incoming_from[int(target)] is not None:
                    continue
                if gap_choice[int(target)] == cand["hit_cell"]:
                    continue

            incoming_src = incoming_from[frame_idx]
            if incoming_src is not None:
                assert chosen[incoming_src] is not None
                if cand["gap"] == chosen[incoming_src]["hit_cell"]:
                    continue

            chosen[frame_idx] = cand
            gap_choice[frame_idx] = cand["gap"]
            previous_incoming = None
            if target is not None:
                previous_incoming = incoming_from[int(target)]
                incoming_from[int(target)] = frame_idx

            backtrack(
                pos + 1,
                hits_so_far + int(target is not None),
                misses_so_far + int(target is None),
            )

            if target is not None:
                incoming_from[int(target)] = previous_incoming
            gap_choice[frame_idx] = None
            chosen[frame_idx] = None

    backtrack(0, 0, 0)
    return best_assignment


def _build_input(shape: tuple[int, int], frames: list[dict], assignment: list[dict]) -> np.ndarray:
    grid, _owner = _build_closed_scene(shape, frames)
    for cand in assignment:
        r, c = map(int, cand["gap"])
        grid[r, c] = Color.BLACK
    return grid


def _expected_output(inp: np.ndarray, frames: list[dict], assignment: list[dict]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for cand in assignment:
        target = cand["target"]
        if target is None:
            continue
        source = int(cand["source"])
        r0, c0, r1, c1 = map(int, frames[int(target)]["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(frames[source]["color"])
    for cand in assignment:
        target = cand["target"]
        if target is None:
            continue
        r, c = map(int, cand["gap"])
        out[r, c] = int(frames[int(target)]["color"])
    return out


def _validate_generated_case(inp: np.ndarray, frames: list[dict], assignment: list[dict]) -> None:
    scene_grid, scene_frames = _scene_relations(inp)
    assert np.array_equal(scene_grid, inp)
    assert len(scene_frames) == len(frames)

    expected_targets = [cand["target"] for cand in assignment]
    expected_gaps = [cand["gap"] for cand in assignment]
    actual_targets = [frame["target"] for frame in scene_frames]
    actual_gaps = [frame["gap"] for frame in scene_frames]
    assert expected_targets == actual_targets, "selected ray graph must survive after opening all gaps"
    assert expected_gaps == actual_gaps, "generated gaps must match extracted gaps"

    assert any(target is not None for target in actual_targets), "need at least one emitting hit case"
    assert any(target is None for target in actual_targets), "need at least one out-of-grid miss case"

    expected = _expected_output(inp, frames, assignment)
    actual = main(inp)
    assert np.array_equal(actual, expected), "solver output must match the planned semantic transform"


def _generate_case() -> np.ndarray | None:
    height = _randint(*_GRID_SIZE_RANGE)
    width = _randint(*_GRID_SIZE_RANGE)
    frames = _random_frames((height, width))
    if frames is None:
        return None

    raw = _raw_candidates((height, width), frames)
    reduced = _reduced_candidates(raw)
    assignment = _choose_assignment(reduced)
    if assignment is None:
        return None

    inp = _build_input((height, width), frames, assignment)
    _validate_generated_case(inp, frames, assignment)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_SCENE_ATTEMPTS):
        try:
            grid = _generate_case()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-directed color-transfer scene")
