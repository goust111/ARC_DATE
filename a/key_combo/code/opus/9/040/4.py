from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
MARKER = 1
OBSTACLE = 2
FRAME = 4
PATH = 8

MAX_GENERATE_ATTEMPTS = 400


@dataclass(frozen=True)
class ProtectionBox:
    frame_bbox: tuple[int, int, int, int]
    marker_bbox: tuple[int, int, int, int]
    marker_count: int

    @property
    def top(self) -> int:
        return self.marker_bbox[0]

    @property
    def left(self) -> int:
        return self.marker_bbox[1]

    @property
    def bottom(self) -> int:
        return self.marker_bbox[2]

    @property
    def right(self) -> int:
        return self.marker_bbox[3]

    @property
    def has_interior(self) -> bool:
        return self.bottom - self.top >= 2 and self.right - self.left >= 2

    def interior_contains_col(self, col: int) -> bool:
        return self.left < int(col) < self.right


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _contiguous_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    if cols.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for raw in cols[1:]:
        col = int(raw)
        if col == prev + 1:
            prev = col
        else:
            runs.append((start, prev))
            start = prev = col
    runs.append((start, prev))
    return runs


def _nearest_side(col: int, left: int, right: int, width: int) -> tuple[int, str]:
    candidates: list[tuple[int, int, int, str]] = []
    if left - 1 >= 0:
        candidates.append((col - (left - 1), 0, left - 1, "left"))
    if right + 1 < width:
        candidates.append(((right + 1) - col, 1, right + 1, "right"))
    assert candidates
    _, _, side_col, side_name = min(candidates, key=lambda item: (item[0], -item[1]))
    return int(side_col), side_name


def _paint_if_bg(out: np.ndarray, r: int, c: int, color: int) -> None:
    if out[r, c] == BG:
        out[r, c] = color


def _paint_hsegment_if_bg(out: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    lo, hi = sorted((int(c0), int(c1)))
    for c in range(lo, hi + 1):
        if out[row, c] == BG:
            out[row, c] = color


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    coord_set = {tuple(map(int, pt)) for pt in coords}
    if len(coord_set) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _parse_protection_boxes(grid: np.ndarray) -> list[ProtectionBox]:
    yellow_comps = _components(grid == FRAME)
    boxes: list[ProtectionBox] = []
    for comp in yellow_comps:
        assert _is_rectangular_frame(comp), "yellow objects must be hollow rectangles"
        fr0, fc0, fr1, fc1 = _bbox(comp)
        inside = []
        for r, c in np.argwhere(grid == MARKER):
            if fr0 < int(r) < fr1 and fc0 < int(c) < fc1:
                inside.append((int(r), int(c)))
        assert 2 <= len(inside) <= 3, "each frame must contain 2-3 blue markers"
        marker_arr = np.array(inside, dtype=int)
        mr0, mc0, mr1, mc1 = _bbox(marker_arr)
        assert fr0 < mr0 <= mr1 < fr1
        assert fc0 < mc0 <= mc1 < fc1
        boxes.append(
            ProtectionBox(
                frame_bbox=(fr0, fc0, fr1, fc1),
                marker_bbox=(mr0, mc0, mr1, mc1),
                marker_count=len(inside),
            )
        )
    boxes.sort(key=lambda box: (box.top, box.left))
    return boxes


def _parse_obstacles(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    rows: dict[int, list[tuple[int, int]]] = {}
    for r in range(grid.shape[0]):
        runs = _contiguous_runs(np.flatnonzero(grid[r] == OBSTACLE))
        if runs:
            rows[int(r)] = runs
    return rows


def _run_covering_col(runs: list[tuple[int, int]], col: int) -> tuple[int, int] | None:
    for left, right in runs:
        if left <= col <= right:
            return left, right
    return None


def _protection_hit(boxes: list[ProtectionBox], row: int, col: int) -> ProtectionBox | None:
    hits = [box for box in boxes if box.top == row and box.has_interior and box.interior_contains_col(col)]
    assert len(hits) <= 1, "protection regions must be non-overlapping at a trigger row"
    return hits[0] if hits else None


def _route_paths(
    grid: np.ndarray,
    *,
    use_obstacles: bool = True,
    use_protection: bool = True,
) -> tuple[np.ndarray, dict[int, list[dict[str, object]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    sources = [int(c) for c in np.flatnonzero(g[0] == PATH)]
    obstacle_rows = _parse_obstacles(g) if use_obstacles else {}
    protect_boxes = _parse_protection_boxes(g) if use_protection else []
    logs: dict[int, list[dict[str, object]]] = {}

    for source_col in sources:
        col = source_col
        events: list[dict[str, object]] = []
        for r in range(h):
            _paint_if_bg(out, r, col, PATH)
            if r == h - 1:
                break
            seen_cols_this_row = {col}
            steps = 0
            while True:
                steps += 1
                assert steps <= w + 2, "same-row detours should terminate"
                obstacle_run = None
                if use_obstacles and int(g[r + 1, col]) == OBSTACLE:
                    obstacle_run = _run_covering_col(obstacle_rows.get(r + 1, []), col)
                    assert obstacle_run is not None
                protection_box = None
                if obstacle_run is None and use_protection:
                    protection_box = _protection_hit(protect_boxes, r, col)

                if obstacle_run is not None:
                    left, right = obstacle_run
                    side_col, side_name = _nearest_side(col, left, right, w)
                    _paint_hsegment_if_bg(out, r, col, side_col, PATH)
                    old_col = col
                    col = side_col
                    events.append(
                        {
                            "type": "red",
                            "row": r + 1,
                            "from_col": old_col,
                            "to_col": col,
                            "side": side_name,
                            "left": left,
                            "right": right,
                            "tie": (old_col - (left - 1)) == ((right + 1) - old_col),
                        }
                    )
                elif protection_box is not None:
                    left, right = protection_box.left, protection_box.right
                    side_col, side_name = _nearest_side(col, left, right, w)
                    _paint_hsegment_if_bg(out, r, col, side_col, PATH)
                    old_col = col
                    col = side_col
                    events.append(
                        {
                            "type": "protect",
                            "row": r + 1,
                            "from_col": old_col,
                            "to_col": col,
                            "side": side_name,
                            "left": left,
                            "right": right,
                            "tie": (old_col - (left - 1)) == ((right + 1) - old_col),
                        }
                    )
                else:
                    break

                assert col not in seen_cols_this_row, "same-row detours cannot cycle"
                seen_cols_this_row.add(col)
        logs[source_col] = events

    return out, logs


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each top teal source downward while detouring around red segments and blue-marker protection regions.

    Input structure:
    - The grid is mostly black background.
    - Row 0 contains 3-4 teal source pixels.
    - Lower rows contain 2-3 red horizontal obstacle segments and 1-2 yellow hollow rectangles with 2-3 blue markers inside each one.

    Rule:
    - Copy the input, then extend a teal path downward from every teal source to the bottom row.
    - If the next cell is red, detour horizontally on the row above that segment to the nearest outside side column, using the right side on ties, then continue downward there.
    - If descending would enter the strict interior of a blue-marker bounding box, detour horizontally on that bounding box's top row to the nearest outside side column, again breaking ties to the right, then continue downward there.
    - Paint new teal cells only onto black background cells; keep red, yellow, and blue cells unchanged.

    Key insight:
    - Red segments are solid collision obstacles, while blue-marker boxes define protected no-entry rectangles; a detour caused by one constraint can immediately move the path into the other constraint's trigger zone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert _get_bg(g) == BG
    assert set(map(int, np.unique(g))).issubset({BG, MARKER, OBSTACLE, FRAME, PATH})

    sources = np.flatnonzero(g[0] == PATH)
    assert 3 <= len(sources) <= 4
    assert np.count_nonzero(g[1:] == PATH) == 0

    obstacle_rows = _parse_obstacles(g)
    obstacle_runs = [(r, left, right) for r, runs in obstacle_rows.items() for left, right in runs]
    assert 2 <= len(obstacle_runs) <= 3

    boxes = _parse_protection_boxes(g)
    assert 1 <= len(boxes) <= 2

    out, _ = _route_paths(g, use_obstacles=True, use_protection=True)
    return out


def _choose_source_cols(width: int, count: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    random.shuffle(cols)
    chosen: list[int] = []
    for col in cols:
        if all(abs(col - prev) >= 4 for prev in chosen):
            chosen.append(col)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _obstacle_candidates(width: int, col: int, mode: str) -> list[dict[str, int | str | bool]]:
    candidates = []
    for length in range(3, 6):
        for left in range(1, width - length):
            right = left + length - 1
            if not (left <= col <= right):
                continue
            side_col, side_name = _nearest_side(col, left, right, width)
            tie = (col - (left - 1)) == ((right + 1) - col)
            actual_mode = "tie" if tie else side_name
            if mode not in {"any", actual_mode}:
                continue
            candidates.append(
                {
                    "left": left,
                    "right": right,
                    "side_col": side_col,
                    "side_name": side_name,
                    "tie": tie,
                }
            )
    return candidates


def _protection_candidates(height: int, width: int, col: int, top: int, mode: str) -> list[dict[str, int | str | bool]]:
    candidates = []
    for box_h in range(3, 6):
        bottom = top + box_h - 1
        if bottom >= height - 1:
            continue
        for box_w in range(3, 6):
            for left in range(1, width - box_w):
                right = left + box_w - 1
                if not (left < col < right):
                    continue
                side_col, side_name = _nearest_side(col, left, right, width)
                tie = (col - (left - 1)) == ((right + 1) - col)
                actual_mode = "tie" if tie else side_name
                if mode not in {"any", actual_mode}:
                    continue
                for gap_top in (1, 2):
                    for gap_bottom in (1, 2):
                        for gap_left in (1, 2):
                            for gap_right in (1, 2):
                                fr0 = top - gap_top
                                fr1 = bottom + gap_bottom
                                fc0 = left - gap_left
                                fc1 = right + gap_right
                                if min(fr0, fc0) < 1:
                                    continue
                                if fr1 >= height or fc1 >= width:
                                    continue
                                frame_h = fr1 - fr0 + 1
                                frame_w = fc1 - fc0 + 1
                                if not (5 <= frame_h <= 7 and 5 <= frame_w <= 7):
                                    continue
                                candidates.append(
                                    {
                                        "mr0": top,
                                        "mc0": left,
                                        "mr1": bottom,
                                        "mc1": right,
                                        "fr0": fr0,
                                        "fc0": fc0,
                                        "fr1": fr1,
                                        "fc1": fc1,
                                        "side_col": side_col,
                                        "side_name": side_name,
                                        "tie": tie,
                                    }
                                )
    return candidates


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _boxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _segment_hits_box(row: int, left: int, right: int, box: tuple[int, int, int, int]) -> bool:
    br0, bc0, br1, bc1 = box
    return br0 <= row <= br1 and not (right < bc0 or left > bc1)


def _place_frame(grid: np.ndarray, info: dict[str, int | str | bool], count: int) -> None:
    fr0 = int(info["fr0"])
    fc0 = int(info["fc0"])
    fr1 = int(info["fr1"])
    fc1 = int(info["fc1"])
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME

    mr0 = int(info["mr0"])
    mc0 = int(info["mc0"])
    mr1 = int(info["mr1"])
    mc1 = int(info["mc1"])
    markers = [(mr0, mc0), (mr1, mc1)]
    extra_options = [(mr0, mc1), (mr1, mc0), ((mr0 + mr1) // 2, (mc0 + mc1) // 2)]
    while len(markers) < count:
        cell = random.choice(extra_options)
        if cell not in markers:
            markers.append(cell)
    for r, c in markers:
        grid[r, c] = MARKER


def _layout_ok(
    source_cols: list[int],
    obstacle_specs: list[tuple[int, int, int]],
    frame_specs: list[dict[str, int | str | bool]],
) -> bool:
    source_set = set(source_cols)
    for row, left, right in obstacle_specs:
        if row <= 1:
            return False
        if any(left <= src <= right for src in source_set) and row == 0:
            return False
        for frame in frame_specs:
            if _segment_hits_box(row, left, right, (int(frame["fr0"]), int(frame["fc0"]), int(frame["fr1"]), int(frame["fc1"]))):
                return False
    for i in range(len(frame_specs)):
        a = frame_specs[i]
        box_a = (int(a["fr0"]), int(a["fc0"]), int(a["fr1"]), int(a["fc1"]))
        for j in range(i + 1, len(frame_specs)):
            b = frame_specs[j]
            box_b = (int(b["fr0"]), int(b["fc0"]), int(b["fr1"]), int(b["fc1"]))
            if _boxes_touch_or_overlap(box_a, box_b):
                return False
    return True


def _analyze_events(logs: dict[int, list[dict[str, object]]]) -> dict[str, object]:
    chain_sources = []
    event_types = set()
    tie_count = 0
    for source, events in logs.items():
        seq = [str(event["type"]) for event in events]
        event_types.update(seq)
        tie_count += sum(bool(event["tie"]) for event in events)
        if "red" in seq and "protect" in seq:
            chain_sources.append((source, tuple(seq)))
    return {
        "has_red": "red" in event_types,
        "has_protect": "protect" in event_types,
        "has_chain": bool(chain_sources),
        "chain_sources": chain_sources,
        "tie_count": tie_count,
    }


def _validate_generated_grid(grid: np.ndarray, require_tie: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _get_bg(grid) != BG:
        return False
    if np.count_nonzero(grid[1:] == PATH) != 0:
        return False

    sources = [int(c) for c in np.flatnonzero(grid[0] == PATH)]
    if not (3 <= len(sources) <= 4):
        return False
    if any(b - a < 4 for a, b in zip(sources, sources[1:])):
        return False

    obstacle_rows = _parse_obstacles(grid)
    obstacle_specs = [(r, left, right) for r, runs in obstacle_rows.items() for left, right in runs]
    if not (2 <= len(obstacle_specs) <= 3):
        return False
    if len({r for r, _, _ in obstacle_specs}) != len(obstacle_specs):
        return False
    if not all(3 <= right - left + 1 <= 5 for _, left, right in obstacle_specs):
        return False

    boxes = _parse_protection_boxes(grid)
    if not (1 <= len(boxes) <= 2):
        return False
    if not all(box.has_interior for box in boxes):
        return False

    full_out, full_logs = _route_paths(grid, use_obstacles=True, use_protection=True)
    red_only, _ = _route_paths(grid, use_obstacles=True, use_protection=False)
    protect_only, _ = _route_paths(grid, use_obstacles=False, use_protection=True)
    if np.array_equal(full_out, red_only):
        return False
    if np.array_equal(full_out, protect_only):
        return False

    summary = _analyze_events(full_logs)
    if not (summary["has_red"] and summary["has_protect"] and summary["has_chain"]):
        return False
    if require_tie and int(summary["tie_count"]) == 0:
        return False
    if np.array_equal(full_out, grid):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(17, 21)
    width = random.randint(17, 22)
    source_count = random.randint(3, 4)
    source_cols = _choose_source_cols(width, source_count)
    if source_cols is None:
        return None

    primary_idx = random.randrange(source_count)
    primary_source = source_cols[primary_idx]
    order = random.choice(["red-first", "protect-first"])
    require_tie = random.random() < 0.45

    obstacle_specs: list[tuple[int, int, int]] = []
    frame_specs: list[dict[str, int | str | bool]] = []

    if order == "red-first":
        red_row = random.randint(4, min(7, height - 9))
        red_mode = "tie" if require_tie and random.random() < 0.5 else random.choice(["left", "right", "any"])
        red_choices = _obstacle_candidates(width, primary_source, red_mode)
        if not red_choices:
            return None
        red1 = random.choice(red_choices)
        obstacle_specs.append((red_row, int(red1["left"]), int(red1["right"])))

        next_col = int(red1["side_col"])
        prot_top = random.randint(red_row + 2, height - 7)
        prot_mode = "tie" if require_tie and not bool(red1["tie"]) else random.choice(["left", "right", "any"])
        prot_choices = _protection_candidates(height, width, next_col, prot_top, prot_mode)
        if not prot_choices:
            return None
        prot1 = random.choice(prot_choices)
        frame_specs.append(prot1)
        current_col = int(prot1["side_col"])
    else:
        prot_top = random.randint(3, min(7, height - 8))
        prot_mode = "tie" if require_tie and random.random() < 0.5 else random.choice(["left", "right", "any"])
        prot_choices = _protection_candidates(height, width, primary_source, prot_top, prot_mode)
        if not prot_choices:
            return None
        prot1 = random.choice(prot_choices)
        frame_specs.append(prot1)

        next_col = int(prot1["side_col"])
        red_row = random.randint(prot_top + 3, height - 4)
        red_mode = "tie" if require_tie and not bool(prot1["tie"]) else random.choice(["left", "right", "any"])
        red_choices = _obstacle_candidates(width, next_col, red_mode)
        if not red_choices:
            return None
        red1 = random.choice(red_choices)
        obstacle_specs.append((red_row, int(red1["left"]), int(red1["right"])))
        current_col = int(red1["side_col"])

    extra_sources = [col for col in source_cols if col != primary_source]
    random.shuffle(extra_sources)

    if extra_sources:
        src = extra_sources.pop()
        row = random.randint(8, height - 4)
        choices = _obstacle_candidates(width, src, "any")
        choices = [item for item in choices if item["side_col"] != current_col]
        if choices:
            red2 = random.choice(choices)
            obstacle_specs.append((row, int(red2["left"]), int(red2["right"])))

    if extra_sources and random.random() < 0.7:
        src = extra_sources.pop()
        top = random.randint(4, height - 8)
        choices = _protection_candidates(height, width, src, top, "any")
        choices = [item for item in choices if int(item["side_col"]) != current_col]
        if choices:
            frame_specs.append(random.choice(choices))

    target_obstacles = random.randint(2, 3)
    target_frames = random.randint(1, 2)

    free_rows = [r for r in range(3, height - 2) if r not in {row for row, _, _ in obstacle_specs}]
    random.shuffle(free_rows)
    while len(obstacle_specs) < target_obstacles and free_rows:
        row = free_rows.pop()
        length = random.randint(3, 5)
        left = random.randint(1, width - length - 1)
        obstacle_specs.append((row, left, left + length - 1))

    tries = 0
    while len(frame_specs) < target_frames and tries < 40:
        tries += 1
        src = random.choice(source_cols)
        top = random.randint(3, height - 8)
        choices = _protection_candidates(height, width, src, top, "any")
        if not choices:
            continue
        candidate = random.choice(choices)
        frame_specs.append(candidate)
        if not _layout_ok(source_cols, obstacle_specs, frame_specs):
            frame_specs.pop()

    if not _layout_ok(source_cols, obstacle_specs, frame_specs):
        return None

    grid = np.zeros((height, width), dtype=int)
    for col in source_cols:
        grid[0, col] = PATH
    for row, left, right in obstacle_specs:
        grid[row, left : right + 1] = OBSTACLE
    for frame in frame_specs:
        _place_frame(grid, frame, count=random.randint(2, 3))

    if not _validate_generated_grid(grid, require_tie=require_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for dual-constraint routing task")
