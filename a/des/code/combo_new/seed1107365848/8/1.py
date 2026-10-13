from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = map(int, arr.min(axis=0))
            r1, c1 = map(int, arr.max(axis=0))
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _frame_bbox(component: dict) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected = {
        (r, c0) for r in range(r0, r1 + 1)
    } | {
        (r, c1) for r in range(r0, r1 + 1)
    } | {
        (r0, c) for c in range(c0, c1 + 1)
    } | {
        (r1, c) for c in range(c0, c1 + 1)
    }
    actual = {tuple(map(int, p)) for p in component["coords"]}
    if actual != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _contains_with_gap(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return ir0 >= or0 + 2 and ic0 >= oc0 + 2 and ir1 <= or1 - 2 and ic1 <= oc1 - 2


def _pair_record(
    g: np.ndarray,
    bg: int,
    outer: dict,
    inner: dict,
) -> dict | None:
    outer_bbox = outer["bbox"]
    inner_bbox = inner["bbox"]
    if not _contains_with_gap(outer_bbox, inner_bbox):
        return None

    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox

    outer_inner = g[or0 + 1 : or1, oc0 + 1 : oc1]
    mask = np.zeros_like(outer_inner, dtype=bool)
    mask[ir0 - or0 - 1 : ir1 - or0, ic0 - oc0 - 1 : ic1 - oc0] = True
    if not np.all(outer_inner[~mask] == bg):
        return None

    fill = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
    vals = np.unique(fill)
    if vals.size != 1:
        return None
    fill_color = int(vals[0])
    if fill_color in {bg, int(outer["color"]), int(inner["color"])}:
        return None

    return {
        "outer_id": id(outer),
        "inner_id": id(inner),
        "outer_color": int(outer["color"]),
        "inner_color": int(inner["color"]),
        "fill_color": fill_color,
        "fill_bbox": (ir0 + 1, ic0 + 1, ir1 - 1, ic1 - 1),
    }


def _find_nested_pairs(g: np.ndarray) -> list[dict]:
    bg = _get_bg(g)
    frames: list[dict] = []
    for comp in _components(g, bg):
        bbox = _frame_bbox(comp)
        if bbox is None:
            continue
        entry = dict(comp)
        entry["bbox"] = bbox
        frames.append(entry)

    pairs: list[dict] = []
    used_inner_ids: set[int] = set()
    outer_ids: set[int] = set()
    frame_by_identity = {id(frame): frame for frame in frames}

    for outer in frames:
        candidates: list[dict] = []
        for inner in frames:
            if inner is outer:
                continue
            record = _pair_record(g, bg, outer, inner)
            if record is not None:
                candidates.append(record)
        if candidates:
            assert len(candidates) == 1, "expected each outer frame to contain a unique valid inner frame"
            record = candidates[0]
            pairs.append(record)
            outer_ids.add(record["outer_id"])
            used_inner_ids.add(record["inner_id"])

    assert pairs, "expected at least one nested pair"
    assert len(pairs) * 2 == len(frames), "expected every frame to serve exactly once as outer or inner"
    assert len(used_inner_ids) == len(pairs), "expected inner frames to be unique"
    assert not (outer_ids & used_inner_ids), "no frame should be both an outer and an inner"

    # Keep a stable order independent of component scan side effects.
    ordered_pairs = []
    for record in pairs:
        outer = frame_by_identity[record["outer_id"]]
        ordered_pairs.append((outer["bbox"], record))
    ordered_pairs.sort(key=lambda item: item[0])
    return [record for _bbox, record in ordered_pairs]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the solid fill colors between paired nested rectangle structures whose border colors cross-reference each other.

    Input structure:
    - The background is the dominant color.
    - Several disjoint objects each contain a one-pixel-thick outer rectangular frame
      and, fully inside it, a one-pixel-thick inner rectangular frame separated by at
      least one background cell.
    - The interior enclosed by each inner frame is a solid non-background color.
    - The number of nested objects is even, outer-frame colors are all distinct, and
      each object's outer color matches exactly one other object's inner color.

    Rule:
    - Recover every nested rectangle object from its outer frame, inner frame, and
      solid fill rectangle.
    - Pair two objects when each object's outer-frame color equals the other's
      inner-frame color.
    - For every such pair, swap the two solid fill colors while leaving all frame
      pixels, shapes, positions, and background cells unchanged.

    Key insight:
    - The required pairing is encoded only by the bidirectional color references
      between outer and inner frames, so the solver must infer the object graph
      before changing any fill.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _find_nested_pairs(g)

    by_outer: dict[int, dict] = {}
    for record in pairs:
        assert record["outer_color"] not in by_outer, "outer-frame colors must be unique"
        by_outer[record["outer_color"]] = record

    out = g.copy()
    visited: set[int] = set()
    for record in pairs:
        if record["outer_id"] in visited:
            continue
        partner = by_outer.get(record["inner_color"])
        assert partner is not None, "every inner color must match one outer color"
        assert partner["inner_color"] == record["outer_color"], "pairing must be bidirectional"
        assert partner["outer_id"] != record["outer_id"], "a pair must contain two distinct objects"

        visited.add(record["outer_id"])
        visited.add(partner["outer_id"])

        ar0, ac0, ar1, ac1 = record["fill_bbox"]
        br0, bc0, br1, bc1 = partner["fill_bbox"]
        out[ar0 : ar1 + 1, ac0 : ac1 + 1] = partner["fill_color"]
        out[br0 : br1 + 1, bc0 : bc1 + 1] = record["fill_color"]

    assert len(visited) == len(pairs), "all objects must belong to a unique two-cycle"
    return out


_COLORS = tuple(range(10))
_MAX_GENERATE_ATTEMPTS = 400
_GRID_MAX = 30


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _sample_geometry(object_count: int) -> dict:
    if object_count == 2:
        outer_h = random.randint(7, 13)
        outer_w = random.randint(7, 13)
    elif object_count == 4:
        outer_h = random.randint(7, 11)
        outer_w = random.randint(7, 11)
    else:
        outer_h = random.randint(7, 9)
        outer_w = random.randint(7, 9)

    valid_gaps: list[tuple[int, int, int, int]] = []
    for top_gap in range(1, 4):
        for bottom_gap in range(1, 4):
            for left_gap in range(1, 4):
                for right_gap in range(1, 4):
                    inner_h = outer_h - 2 - top_gap - bottom_gap
                    inner_w = outer_w - 2 - left_gap - right_gap
                    if inner_h >= 3 and inner_w >= 3:
                        valid_gaps.append((top_gap, bottom_gap, left_gap, right_gap))
    assert valid_gaps
    top_gap, bottom_gap, left_gap, right_gap = random.choice(valid_gaps)
    return {
        "outer_h": outer_h,
        "outer_w": outer_w,
        "top_gap": top_gap,
        "bottom_gap": bottom_gap,
        "left_gap": left_gap,
        "right_gap": right_gap,
    }


def _sample_specs() -> tuple[int, list[dict]]:
    group_count = random.choice((1, 2, 3))
    object_count = 2 * group_count
    bg = random.choice(_COLORS)

    remaining = [c for c in _COLORS if c != bg]
    random.shuffle(remaining)
    border_colors = remaining[:object_count]
    fill_pool = [c for c in remaining[object_count:] if c != bg]
    assert len(fill_pool) >= 2

    specs: list[dict] = []
    for i in range(group_count):
        a = border_colors[2 * i]
        b = border_colors[2 * i + 1]
        fill_a, fill_b = random.sample(fill_pool, 2)
        geom_a = _sample_geometry(object_count)
        geom_b = _sample_geometry(object_count)
        specs.append(
            {
                "group": i,
                "outer_color": a,
                "inner_color": b,
                "fill_color": fill_a,
                **geom_a,
            }
        )
        specs.append(
            {
                "group": i,
                "outer_color": b,
                "inner_color": a,
                "fill_color": fill_b,
                **geom_b,
            }
        )
    random.shuffle(specs)
    return bg, specs


def _candidate_grid_shape(specs: list[dict]) -> tuple[int, int]:
    object_count = len(specs)
    total_h = sum(spec["outer_h"] for spec in specs)
    total_w = sum(spec["outer_w"] for spec in specs)
    min_rows = max(max(spec["outer_h"] for spec in specs) + 2, min(_GRID_MAX, total_h // 2 + object_count + 2))
    min_cols = max(max(spec["outer_w"] for spec in specs) + 2, min(_GRID_MAX, total_w // 2 + object_count + 2))
    rows = random.randint(min_rows, _GRID_MAX)
    cols = random.randint(min_cols, _GRID_MAX)
    return rows, cols


def _place_specs(rows: int, cols: int, specs: list[dict]) -> bool:
    occupied = np.zeros((rows, cols), dtype=bool)
    for spec in sorted(specs, key=lambda s: s["outer_h"] * s["outer_w"], reverse=True):
        h = spec["outer_h"]
        w = spec["outer_w"]
        positions = [
            (r, c)
            for r in range(1, rows - h)
            for c in range(1, cols - w)
            if not occupied[max(0, r - 1) : min(rows, r + h + 1), max(0, c - 1) : min(cols, c + w + 1)].any()
        ]
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        spec["outer_bbox"] = (r0, c0, r1, c1)

        ir0 = r0 + 1 + spec["top_gap"]
        ir1 = r1 - 1 - spec["bottom_gap"]
        ic0 = c0 + 1 + spec["left_gap"]
        ic1 = c1 - 1 - spec["right_gap"]
        spec["inner_bbox"] = (ir0, ic0, ir1, ic1)
        spec["fill_bbox"] = (ir0 + 1, ic0 + 1, ir1 - 1, ic1 - 1)

        occupied[max(0, r0 - 1) : min(rows, r1 + 2), max(0, c0 - 1) : min(cols, c1 + 2)] = True
    return True


def _render_input(bg: int, rows: int, cols: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((rows, cols), bg, dtype=int)
    for spec in specs:
        _draw_frame(grid, spec["outer_bbox"], spec["outer_color"])
        _draw_frame(grid, spec["inner_bbox"], spec["inner_color"])
        fr0, fc0, fr1, fc1 = spec["fill_bbox"]
        grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = spec["fill_color"]
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    grouped: dict[int, list[dict]] = {}
    for spec in specs:
        grouped.setdefault(spec["group"], []).append(spec)
    for members in grouped.values():
        assert len(members) == 2
        a, b = members
        ar0, ac0, ar1, ac1 = a["fill_bbox"]
        br0, bc0, br1, bc1 = b["fill_bbox"]
        out[ar0 : ar1 + 1, ac0 : ac1 + 1] = b["fill_color"]
        out[br0 : br1 + 1, bc0 : bc1 + 1] = a["fill_color"]
    return out


def _valid_background_dominance(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flat))
    bg_count = counts.pop(bg)
    return all(bg_count > count for count in counts.values())


def _generate_attempt() -> np.ndarray | None:
    bg, specs = _sample_specs()
    rows, cols = _candidate_grid_shape(specs)
    if not _place_specs(rows, cols, specs):
        return None

    inp = _render_input(bg, rows, cols, specs)
    if not _valid_background_dominance(inp, bg):
        return None

    expected = _expected_output(inp, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATE_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input grid")
