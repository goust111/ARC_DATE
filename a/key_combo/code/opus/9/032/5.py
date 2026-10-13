from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
TIE = 9
MOTIF_COLORS = (1, 2, 3, 4, 7)

GEN_FRAME_COUNTS = (3, 4)
GEN_FRAME_COUNT_WEIGHTS = (0.68, 0.32)
GEN_GRID_H_RANGE = (15, 22)
GEN_GAPS_3 = (2, 3, 4, 5)
GEN_GAPS_4 = (2, 3, 4)
GEN_INNER_DIMS = {
    "wide": ((3, 5), (3, 6), (4, 5), (4, 6)),
    "tall": ((5, 3), (6, 3), (5, 4), (6, 4)),
    "square": ((3, 3), (4, 4), (5, 5)),
}
GEN_TYPE_WEIGHTS = {"wide": 0.44, "tall": 0.36, "square": 0.20}
GEN_RETRIES = 400
GEN_MOTIF_RETRIES = 140


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _axis_for_dims(inner_h: int, inner_w: int) -> str:
    return "vertical" if int(inner_w) >= int(inner_h) else "horizontal"


def _side_half(axis: str, side: str, inner_h: int, inner_w: int) -> np.ndarray:
    if axis == "vertical":
        if side == "left":
            cols = range(0, inner_w // 2)
        else:
            cols = range((inner_w + 1) // 2, inner_w)
        cells = [(r, c) for r in range(inner_h) for c in cols]
    else:
        if side == "top":
            rows = range(0, inner_h // 2)
        else:
            rows = range((inner_h + 1) // 2, inner_h)
        cells = [(r, c) for r in rows for c in range(inner_w)]
    return np.array(cells, dtype=int)


def _infer_side(axis: str, coords: np.ndarray, inner_h: int, inner_w: int) -> str:
    if axis == "vertical":
        cols = coords[:, 1]
        if bool(np.all(cols < inner_w // 2)):
            return "left"
        if bool(np.all(cols > (inner_w - 1) // 2)):
            return "right"
    else:
        rows = coords[:, 0]
        if bool(np.all(rows < inner_h // 2)):
            return "top"
        if bool(np.all(rows > (inner_h - 1) // 2)):
            return "bottom"
    raise AssertionError("motif must stay strictly in one half of its frame interior")


def _reflect_coords(coords: np.ndarray, inner_h: int, inner_w: int, axis: str) -> np.ndarray:
    if axis == "vertical":
        return np.array([(int(r), int(inner_w - 1 - c)) for r, c in coords], dtype=int)
    return np.array([(int(inner_h - 1 - r), int(c)) for r, c in coords], dtype=int)


def _build_frame_record(spec: dict, motif_local: np.ndarray) -> dict:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    axis = _axis_for_dims(inner_h, inner_w)
    side = _infer_side(axis, motif_local, inner_h, inner_w)
    reflected = _reflect_coords(motif_local, inner_h, inner_w, axis)
    completed = np.unique(np.vstack([motif_local, reflected]), axis=0)
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    motif_abs = motif_local + np.array([r0 + 1, c0 + 1])
    completed_abs = completed + np.array([r0 + 1, c0 + 1])
    return {
        "bbox": (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1),
        "r0": r0,
        "c0": c0,
        "r1": r0 + inner_h + 1,
        "c1": c0 + inner_w + 1,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "axis": axis,
        "color": int(spec["color"]),
        "motif": motif_local.astype(int),
        "motif_abs": motif_abs.astype(int),
        "completed": completed.astype(int),
        "completed_abs": completed_abs.astype(int),
        "side": side,
        "type": spec.get("type", "unknown"),
    }


def _parse_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for coords in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 5 and w >= 5, "frame too small"

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert bool(np.array_equal(patch == FRAME, border)), "grey component must be a 1-pixel rectangular frame"

        inner = patch[1:-1, 1:-1]
        colors = sorted(int(c) for c in np.unique(inner) if int(c) not in (BG, FRAME))
        assert len(colors) == 1, "each frame must contain exactly one motif color"
        color = int(colors[0])

        motif_mask = inner == color
        motif_components = _components(motif_mask)
        assert len(motif_components) == 1, "motif must be 4-connected"
        motif_local = motif_components[0]
        assert 3 <= len(motif_local) <= 6, "motif must contain 3-6 pixels"

        spec = {"r0": r0, "c0": c0, "inner_h": h - 2, "inner_w": w - 2, "color": color}
        frames.append(_build_frame_record(spec, motif_local))

    frames.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    assert len(frames) >= 2, "expected at least two frames"
    return frames


def _count_rows(coords: np.ndarray, row0: int, row1: int) -> int:
    rows = coords[:, 0]
    return int(np.count_nonzero((rows >= int(row0)) & (rows <= int(row1))))


def _winner_color(frame_a: dict, frame_b: dict, row0: int, row1: int, *, completed: bool) -> tuple[int, int, int]:
    coords_a = frame_a["completed_abs"] if completed else frame_a["motif_abs"]
    coords_b = frame_b["completed_abs"] if completed else frame_b["motif_abs"]
    count_a = _count_rows(coords_a, row0, row1)
    count_b = _count_rows(coords_b, row0, row1)
    if count_a > count_b:
        return count_a, count_b, int(frame_a["color"])
    if count_b > count_a:
        return count_a, count_b, int(frame_b["color"])
    return count_a, count_b, TIE


def _pair_records(frames: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            left = frames[i]
            right = frames[j]
            if int(left["c1"]) >= int(right["c0"]):
                continue

            row0 = max(int(left["r0"]), int(right["r0"]))
            row1 = min(int(left["r1"]), int(right["r1"]))
            if row0 > row1:
                continue

            gap0 = int(left["c1"]) + 1
            gap1 = int(right["c0"]) - 1
            blocked = False
            for k, blocker in enumerate(frames):
                if k in (i, j):
                    continue
                if _ranges_overlap(gap0, gap1, blocker["c0"], blocker["c1"]) and _ranges_overlap(
                    row0, row1, blocker["r0"], blocker["r1"]
                ):
                    blocked = True
                    break

            raw_a, raw_b, raw_color = _winner_color(left, right, row0, row1, completed=False)
            count_a, count_b, fill_color = _winner_color(left, right, row0, row1, completed=True)
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "row0": row0,
                    "row1": row1,
                    "gap0": gap0,
                    "gap1": gap1,
                    "blocked": blocked,
                    "raw_a": raw_a,
                    "raw_b": raw_b,
                    "raw_color": raw_color,
                    "count_a": count_a,
                    "count_b": count_b,
                    "fill_color": fill_color,
                }
            )
    return pairs


def _analyze(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    frames = _parse_frames(grid)
    return frames, _pair_records(frames)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each framed motif by aspect-ratio-guided mirroring, then fill unobstructed gaps with the color of the stronger completed motif.

    Input structure:
    - The grid has a black background and several disjoint grey rectangular frames of border width 1.
    - Each frame contains one 4-connected monochrome motif of a distinct color, placed strictly on one side of the frame interior.
    - Frames are horizontally separated by black gaps, and some pairs overlap in their row spans.

    Rule:
    - For each frame, inspect the interior width and height: if width >= height, mirror the motif across the vertical middle axis; otherwise mirror it across the horizontal middle axis.
    - Keep the original motif and add the reflected copy in the same motif color.
    - For every horizontally separated frame pair with overlapping row spans, take their shared row interval and skip the pair if any third frame bbox crosses both that row interval and the gap columns between them.
    - Otherwise fill the entire gap rectangle across the shared rows with the color of the frame whose completed motif has more pixels in those shared rows; use maroon when the two counts are equal.

    Key insight:
    - Mirroring changes per-row motif mass differently in wide versus tall frames, so the gap color is determined only after the frame-specific completion step.
    """
    g = np.asarray(grid, dtype=int)
    frames, pairs = _analyze(g)

    out = g.copy()
    for frame in frames:
        rr = frame["completed_abs"][:, 0]
        cc = frame["completed_abs"][:, 1]
        out[rr, cc] = int(frame["color"])

    for pair in pairs:
        if pair["blocked"]:
            continue
        out[pair["row0"] : pair["row1"] + 1, pair["gap0"] : pair["gap1"] + 1] = int(pair["fill_color"])

    return out


def _weighted_choice(cells: list[tuple[int, int]], preferred_rows: set[int] | None) -> tuple[int, int]:
    weights = []
    for r, _c in cells:
        weights.append(5 if preferred_rows and int(r) in preferred_rows else 1)
    return random.choices(cells, weights=weights, k=1)[0]


def _sample_motif(
    inner_h: int,
    inner_w: int,
    axis: str,
    *,
    side: str | None = None,
    size: int | None = None,
    preferred_rows: set[int] | None = None,
    row_pool: set[int] | None = None,
) -> np.ndarray | None:
    side = side or random.choice(("left", "right") if axis == "vertical" else ("top", "bottom"))
    size = int(size if size is not None else random.randint(3, 6))
    half_cells = _side_half(axis, side, inner_h, inner_w)
    if row_pool is not None:
        half_cells = np.array([cell for cell in half_cells if int(cell[0]) in row_pool], dtype=int)
    if half_cells.size == 0 or len(half_cells) < size:
        return None

    allowed = {tuple(map(int, cell)) for cell in half_cells}
    preferred_rows = set() if preferred_rows is None else {int(r) for r in preferred_rows}

    for _ in range(40):
        start = _weighted_choice(sorted(allowed), preferred_rows if preferred_rows else None)
        chosen = {start}
        while len(chosen) < size:
            frontier = []
            for r, c in chosen:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in allowed and (nr, nc) not in chosen:
                        frontier.append((nr, nc))
            frontier = sorted(set(frontier))
            if not frontier:
                break
            chosen.add(_weighted_choice(frontier, preferred_rows if preferred_rows else None))
        if len(chosen) != size:
            continue
        coords = np.array(sorted(chosen), dtype=int)
        return coords
    return None


def _shared_rows_within_interior(frame: dict, shared_row0: int, shared_row1: int) -> tuple[int, int] | None:
    top = int(frame["r0"]) + 1
    bottom = int(frame["r1"]) - 1
    row0 = max(int(shared_row0), top)
    row1 = min(int(shared_row1), bottom)
    if row0 > row1:
        return None
    return int(row0 - top), int(row1 - top)


def _sample_specs() -> list[dict] | None:
    n = random.choices(GEN_FRAME_COUNTS, weights=GEN_FRAME_COUNT_WEIGHTS, k=1)[0]
    types = []
    while True:
        types = random.choices(
            tuple(GEN_TYPE_WEIGHTS),
            weights=tuple(GEN_TYPE_WEIGHTS.values()),
            k=n,
        )
        if "tall" not in types or all(t == "tall" for t in types):
            continue
        if any((types[i] == "tall") != (types[i + 1] == "tall") for i in range(n - 1)):
            break

    specs = []
    for idx, typ in enumerate(types):
        inner_h, inner_w = random.choice(GEN_INNER_DIMS[typ])
        specs.append({"idx": idx, "type": typ, "inner_h": inner_h, "inner_w": inner_w})

    widths = [int(spec["inner_w"]) + 2 for spec in specs]
    gaps = [random.choice(GEN_GAPS_3 if n == 3 else GEN_GAPS_4) for _ in range(n - 1)]
    total_w = sum(widths) + sum(gaps)
    if total_w > 28:
        return None
    left_margin = random.randint(1, max(1, 30 - total_w - 1))
    right_margin = random.randint(1, 30 - total_w - left_margin)
    grid_w = int(total_w + left_margin + right_margin)
    c = left_margin
    for idx, (spec, outer_w) in enumerate(zip(specs, widths)):
        spec["c0"] = int(c)
        c += int(outer_w)
        spec["c1"] = int(c - 1)
        if idx < n - 1:
            c += int(gaps[idx])

    overlaps = []
    for i in range(n - 1):
        h0 = int(specs[i]["inner_h"]) + 2
        h1 = int(specs[i + 1]["inner_h"]) + 2
        overlaps.append(random.randint(2, min(3, h0, h1)))

    direction = random.choice(("down", "up"))
    r_positions = [0]
    for i in range(1, n):
        prev_h = int(specs[i - 1]["inner_h"]) + 2
        cur_h = int(specs[i]["inner_h"]) + 2
        ov = overlaps[i - 1]
        if direction == "down":
            r_positions.append(r_positions[-1] + prev_h - ov)
        else:
            r_positions.append(r_positions[-1] - (cur_h - ov))

    top_margin = random.randint(1, 2)
    shift = top_margin - min(r_positions)
    shifted = [int(r0 + shift) for r0 in r_positions]
    bottom_margin = random.randint(1, 2)
    H = max(shifted[i] + int(specs[i]["inner_h"]) + 2 for i in range(n)) + bottom_margin
    if not (GEN_GRID_H_RANGE[0] <= H <= GEN_GRID_H_RANGE[1]):
        return None

    for spec, r0 in zip(specs, shifted):
        spec["r0"] = int(r0)
        spec["r1"] = int(r0 + spec["inner_h"] + 1)

    for i in range(n):
        for j in range(i + 2, n):
            if _ranges_overlap(specs[i]["r0"], specs[i]["r1"], specs[j]["r0"], specs[j]["r1"]):
                return None

    for spec, color in zip(specs, random.sample(MOTIF_COLORS, n)):
        spec["color"] = int(color)
        spec["grid_h"] = int(H)
        spec["grid_w"] = int(grid_w)
    return specs


def _focus_candidates(specs: list[dict]) -> list[dict]:
    frames = []
    for spec in specs:
        dummy = {"bbox": (spec["r0"], spec["c0"], spec["r1"], spec["c1"]), **spec}
        frames.append(dummy)

    candidates = []
    for i in range(len(frames) - 1):
        left = frames[i]
        right = frames[i + 1]
        if (left["type"] == "tall") == (right["type"] == "tall"):
            continue
        shared0 = max(int(left["r0"]), int(right["r0"]))
        shared1 = min(int(left["r1"]), int(right["r1"]))
        tall = left if left["type"] == "tall" else right
        other = right if tall is left else left
        tall_rows = _shared_rows_within_interior(tall, shared0, shared1)
        other_rows = _shared_rows_within_interior(other, shared0, shared1)
        if tall_rows is None or other_rows is None:
            continue
        ov0, ov1 = tall_rows
        source_rows = set(range(int(tall["inner_h"]) - 1 - ov1, int(tall["inner_h"]) - ov0))
        if not source_rows:
            continue
        if max(source_rows) < tall["inner_h"] // 2:
            tall_side = "top"
        elif min(source_rows) > (tall["inner_h"] - 1) // 2:
            tall_side = "bottom"
        else:
            continue
        if len(source_rows) * int(tall["inner_w"]) < 3:
            continue

        other_local0, other_local1 = other_rows
        other_outside = {r for r in range(int(other["inner_h"])) if not (other_local0 <= r <= other_local1)}
        if not other_outside:
            continue
        other_axis = _axis_for_dims(int(other["inner_h"]), int(other["inner_w"]))
        if other_axis == "vertical":
            half_width = len(range(0, int(other["inner_w"]) // 2))
        else:
            half_width = int(other["inner_w"])
        if len(other_outside) * half_width < 3:
            continue

        candidates.append(
            {
                "pair": (i, i + 1),
                "tall_index": int(tall["idx"]),
                "other_index": int(other["idx"]),
                "tall_side": tall_side,
                "tall_rows": source_rows,
                "other_pref_rows": other_outside,
            }
        )
    return candidates


def _paint_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0 = int(frame["r0"]), int(frame["c0"])
    r1, c1 = int(frame["r1"]), int(frame["c1"])
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME


def _attempt_generate(require_tie: bool) -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None
    focus = _focus_candidates(specs)
    if not focus:
        return None
    focus_choice = random.choice(focus)

    axis_by_idx = {spec["idx"]: _axis_for_dims(int(spec["inner_h"]), int(spec["inner_w"])) for spec in specs}
    side_plan: dict[int, str | None] = {}
    row_pool_plan: dict[int, set[int] | None] = {}
    preferred_rows_plan: dict[int, set[int] | None] = {}
    size_plan: dict[int, tuple[int, int]] = {}

    for spec in specs:
        idx = int(spec["idx"])
        axis = axis_by_idx[idx]
        side_plan[idx] = None
        row_pool_plan[idx] = None
        preferred_rows_plan[idx] = None
        size_plan[idx] = (3, 6)
        if idx == focus_choice["tall_index"]:
            side_plan[idx] = str(focus_choice["tall_side"])
            row_pool_plan[idx] = set(int(r) for r in focus_choice["tall_rows"])
            preferred_rows_plan[idx] = set(int(r) for r in focus_choice["tall_rows"])
            size_plan[idx] = (4, 6)
        elif idx == focus_choice["other_index"]:
            if axis == "vertical":
                side_plan[idx] = random.choice(("left", "right"))
            preferred_rows_plan[idx] = set(int(r) for r in focus_choice["other_pref_rows"])
            size_plan[idx] = (3, 4)

    for _ in range(GEN_MOTIF_RETRIES):
        frames: list[dict] = []
        grid = np.zeros((int(specs[0]["grid_h"]), int(specs[0]["grid_w"])), dtype=int)
        for spec in specs:
            _paint_frame(grid, spec)

        ok = True
        for spec in specs:
            idx = int(spec["idx"])
            axis = axis_by_idx[idx]
            lo, hi = size_plan[idx]
            motif = _sample_motif(
                int(spec["inner_h"]),
                int(spec["inner_w"]),
                axis,
                side=side_plan[idx],
                size=random.randint(lo, hi),
                preferred_rows=preferred_rows_plan[idx],
                row_pool=row_pool_plan[idx],
            )
            if motif is None:
                ok = False
                break
            frame = _build_frame_record(spec, motif)
            rr = frame["motif_abs"][:, 0]
            cc = frame["motif_abs"][:, 1]
            grid[rr, cc] = int(frame["color"])
            frames.append(frame)
        if not ok:
            continue

        frames.sort(key=lambda d: int(d["c0"]))
        pairs = _pair_records(frames)
        paintable = [p for p in pairs if not p["blocked"]]
        if len(paintable) < 2:
            continue
        if any(p["blocked"] for p in pairs):
            continue

        changed_pairs = [p for p in paintable if int(p["fill_color"]) != int(p["raw_color"])]
        if not changed_pairs:
            continue

        focus_pair = None
        for p in paintable:
            if (p["i"], p["j"]) == tuple(focus_choice["pair"]):
                focus_pair = p
                break
        if focus_pair is None:
            continue
        if int(focus_pair["fill_color"]) == int(focus_pair["raw_color"]):
            continue

        if require_tie and not any(int(p["fill_color"]) == TIE for p in paintable):
            continue
        if not require_tie and len({int(p["fill_color"]) for p in paintable}) < 2 and random.random() < 0.65:
            continue

        out = main(grid)
        if bool(np.array_equal(out, grid)):
            continue
        if sum(len(frame["completed"]) - len(frame["motif"]) for frame in frames) <= 0:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    require_tie = random.random() < 0.45
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(require_tie=require_tie)
        if grid is None:
            continue
        frames, pairs = _analyze(grid)
        paintable = [p for p in pairs if not p["blocked"]]
        if len(frames) not in (3, 4):
            continue
        if len(paintable) < 2:
            continue
        if not any(int(p["fill_color"]) != int(p["raw_color"]) for p in paintable):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
