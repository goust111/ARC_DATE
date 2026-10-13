from __future__ import annotations

import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
EDGE_ORDER = {"top": 0, "bottom": 1, "left": 2, "right": 3}
EDGES = tuple(EDGE_ORDER)

GRID_MIN = 12
GRID_MAX = 18
KEEP_COUNT_MIN = 2
KEEP_COUNT_MAX = 5
REMOVED_COUNT_MIN = 1
REMOVED_COUNT_MAX = 3
CLUSTER_PROB = 0.7
CLUSTER_MAX = 3
CASE_ATTEMPTS = 400
RECORD_ATTEMPTS = 160
INNER_MARGIN = 1
ODD_SIZES = (1, 3, 5)
EVEN_SIZES = (2, 4, 6)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _intervals_overlap(a[0], a[2], b[0], b[2]) and _intervals_overlap(a[1], a[3], b[1], b[3])


def _nearest_edge(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> tuple[str, int, bool]:
    dists = [
        ("top", r0 + r1),
        ("bottom", 2 * (h - 1) - (r0 + r1)),
        ("left", c0 + c1),
        ("right", 2 * (w - 1) - (c0 + c1)),
    ]
    best = min(dist for _, dist in dists)
    best_edges = [edge for edge, dist in dists if dist == best]
    return best_edges[0], best, len(best_edges) == 1


def _expanded_clear(
    bbox: tuple[int, int, int, int],
    occupied: np.ndarray,
    margin: int = INNER_MARGIN,
) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(occupied.shape[0] - 1, r1 + margin)
    cc1 = min(occupied.shape[1] - 1, c1 + margin)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    rects: list[dict[str, int]] = []
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    for color in colors:
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            rects.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                }
            )
    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["color"]))
    return rects


def _settle_rect(
    edge: str,
    rect_h: int,
    rect_w: int,
    fixed_start: int,
    placed: list[dict[str, int]],
    grid_h: int,
    grid_w: int,
) -> dict[str, int] | None:
    if edge in ("top", "bottom"):
        c0 = fixed_start
        c1 = c0 + rect_w - 1
        if c0 < 0 or c1 >= grid_w:
            return None
        conflicts = [p for p in placed if _intervals_overlap(c0, c1, p["c0"], p["c1"])]
        if edge == "top":
            r0 = 0
            for other in sorted(conflicts, key=lambda p: (p["r0"], p["c0"], p["r1"], p["c1"])):
                if _intervals_overlap(r0, r0 + rect_h - 1, other["r0"], other["r1"]):
                    r0 = other["r1"] + 1
        else:
            r0 = grid_h - rect_h
            for other in sorted(conflicts, key=lambda p: (-p["r1"], p["c0"], p["r0"], p["c1"])):
                if _intervals_overlap(r0, r0 + rect_h - 1, other["r0"], other["r1"]):
                    r0 = other["r0"] - rect_h
        r1 = r0 + rect_h - 1
        if r0 < 0 or r1 >= grid_h:
            return None
        return {"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)}

    r0 = fixed_start
    r1 = r0 + rect_h - 1
    if r0 < 0 or r1 >= grid_h:
        return None
    conflicts = [p for p in placed if _intervals_overlap(r0, r1, p["r0"], p["r1"])]
    if edge == "left":
        c0 = 0
        for other in sorted(conflicts, key=lambda p: (p["c0"], p["r0"], p["c1"], p["r1"])):
            if _intervals_overlap(c0, c0 + rect_w - 1, other["c0"], other["c1"]):
                c0 = other["c1"] + 1
    else:
        c0 = grid_w - rect_w
        for other in sorted(conflicts, key=lambda p: (-p["c1"], p["r0"], p["c0"], p["r1"])):
            if _intervals_overlap(c0, c0 + rect_w - 1, other["c0"], other["c1"]):
                c0 = other["c0"] - rect_w
    c1 = c0 + rect_w - 1
    if c0 < 0 or c1 >= grid_w:
        return None
    return {"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)}


def _transform_rectangles(rects: list[dict[str, int]], grid_h: int, grid_w: int) -> list[dict[str, int]]:
    movers: list[dict[str, int]] = []
    for rect in rects:
        if rect["h"] % 2 != rect["w"] % 2:
            continue
        edge, dist2, _unique = _nearest_edge(rect["r0"], rect["c0"], rect["r1"], rect["c1"], grid_h, grid_w)
        item = dict(rect)
        item["edge"] = edge
        item["dist2"] = int(dist2)
        movers.append(item)

    movers.sort(
        key=lambda rect: (
            rect["dist2"],
            EDGE_ORDER[rect["edge"]],
            rect["r0"],
            rect["c0"],
            rect["color"],
        )
    )

    placed: list[dict[str, int]] = []
    for rect in movers:
        fixed_start = rect["c0"] if rect["edge"] in ("top", "bottom") else rect["r0"]
        settled = _settle_rect(rect["edge"], rect["h"], rect["w"], fixed_start, placed, grid_h, grid_w)
        if settled is None:
            continue
        settled["color"] = rect["color"]
        settled["h"] = rect["h"]
        settled["w"] = rect["w"]
        settled["edge"] = rect["edge"]
        settled["dist2"] = rect["dist2"]
        placed.append(settled)
    return placed


def _paint_rectangles(shape: tuple[int, int], rects: list[dict[str, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    rects = _extract_rectangles(g)
    moved = _transform_rectangles(rects, g.shape[0], g.shape[1])
    return _paint_rectangles(g.shape, moved)


def _sample_dims(rng: np.random.Generator, keep: bool) -> tuple[int, int]:
    first_parity = int(rng.integers(0, 2))
    second_parity = first_parity if keep else 1 - first_parity
    heights = EVEN_SIZES if first_parity == 0 else ODD_SIZES
    widths = EVEN_SIZES if second_parity == 0 else ODD_SIZES
    return int(rng.choice(heights)), int(rng.choice(widths))


def _choose_fixed_start(
    rng: np.random.Generator,
    edge: str,
    rect_h: int,
    rect_w: int,
    grid_h: int,
    grid_w: int,
    overlap_range: tuple[int, int] | None,
) -> int | None:
    if edge in ("top", "bottom"):
        lo = 1
        hi = grid_w - rect_w - 1
    else:
        lo = 1
        hi = grid_h - rect_h - 1
    if hi < lo:
        return None
    if overlap_range is None:
        return int(rng.integers(lo, hi + 1))

    if edge in ("top", "bottom"):
        ov0, ov1 = overlap_range
        lo = max(lo, ov0 - rect_w + 1)
        hi = min(hi, ov1)
    else:
        ov0, ov1 = overlap_range
        lo = max(lo, ov0 - rect_h + 1)
        hi = min(hi, ov1)
    if hi < lo:
        return None
    return int(rng.integers(lo, hi + 1))


def _build_output_records(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    keep_count: int,
) -> list[dict[str, int]] | None:
    placed: list[dict[str, int]] = []
    records: list[dict[str, int]] = []

    cluster_count = 0
    if keep_count >= 2 and rng.random() < CLUSTER_PROB:
        cluster_count = min(keep_count, int(rng.integers(2, min(CLUSTER_MAX, keep_count) + 1)))
    cluster_edge = str(rng.choice(EDGES)) if cluster_count else ""
    overlap_range: tuple[int, int] | None = None

    for idx in range(keep_count):
        wants_cluster = idx < cluster_count
        edge = cluster_edge if wants_cluster else str(rng.choice(EDGES))
        overlap = overlap_range if wants_cluster else None
        built = None
        for _ in range(RECORD_ATTEMPTS):
            rect_h, rect_w = _sample_dims(rng, keep=True)
            fixed_start = _choose_fixed_start(rng, edge, rect_h, rect_w, grid_h, grid_w, overlap)
            if fixed_start is None:
                continue
            settled = _settle_rect(edge, rect_h, rect_w, fixed_start, placed, grid_h, grid_w)
            if settled is None:
                continue
            built = dict(settled)
            built["h"] = rect_h
            built["w"] = rect_w
            built["edge"] = edge
            if edge in ("top", "bottom"):
                span0 = fixed_start
                span1 = fixed_start + rect_w - 1
            else:
                span0 = fixed_start
                span1 = fixed_start + rect_h - 1
            built["fixed_start"] = fixed_start
            built["span0"] = int(span0)
            built["span1"] = int(span1)
            break
        if built is None:
            return None
        placed.append(built)
        records.append(built)
        if wants_cluster:
            if overlap_range is None:
                overlap_range = (built["span0"], built["span1"])
            else:
                overlap_range = (
                    max(overlap_range[0], built["span0"]),
                    min(overlap_range[1], built["span1"]),
                )
                if overlap_range[0] > overlap_range[1]:
                    return None
    return records


def _shift_from_output(record: dict[str, int], offset: int) -> tuple[int, int, int, int]:
    edge = record["edge"]
    if edge == "top":
        return (
            record["r0"] + offset,
            record["c0"],
            record["r1"] + offset,
            record["c1"],
        )
    if edge == "bottom":
        return (
            record["r0"] - offset,
            record["c0"],
            record["r1"] - offset,
            record["c1"],
        )
    if edge == "left":
        return (
            record["r0"],
            record["c0"] + offset,
            record["r1"],
            record["c1"] + offset,
        )
    return (
        record["r0"],
        record["c0"] - offset,
        record["r1"],
        record["c1"] - offset,
    )


def _max_inward_offset(record: dict[str, int], grid_h: int, grid_w: int) -> int:
    edge = record["edge"]
    if edge == "top":
        return grid_h - 2 - record["r1"]
    if edge == "bottom":
        return record["r0"] - 1
    if edge == "left":
        return grid_w - 2 - record["c1"]
    return record["c0"] - 1


def _build_input_from_output(
    records: list[dict[str, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[list[dict[str, int]], np.ndarray] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    input_records: list[dict[str, int]] = []
    prev_dist2 = -1

    for record in records:
        max_offset = _max_inward_offset(record, grid_h, grid_w)
        chosen = None
        for offset in range(1, max_offset + 1):
            bbox = _shift_from_output(record, offset)
            if bbox[0] < 1 or bbox[1] < 1 or bbox[2] >= grid_h - 1 or bbox[3] >= grid_w - 1:
                continue
            if not _expanded_clear(bbox, occupied):
                continue
            edge, dist2, unique = _nearest_edge(bbox[0], bbox[1], bbox[2], bbox[3], grid_h, grid_w)
            if edge != record["edge"] or not unique or dist2 <= prev_dist2:
                continue
            chosen = {
                "r0": int(bbox[0]),
                "c0": int(bbox[1]),
                "r1": int(bbox[2]),
                "c1": int(bbox[3]),
                "h": record["h"],
                "w": record["w"],
                "edge": edge,
                "dist2": int(dist2),
            }
            break
        if chosen is None:
            return None
        input_records.append(chosen)
        _mark_bbox(occupied, (chosen["r0"], chosen["c0"], chosen["r1"], chosen["c1"]))
        prev_dist2 = chosen["dist2"]

    return input_records, occupied


def _build_removed_records(
    rng: np.random.Generator,
    occupied: np.ndarray,
    grid_h: int,
    grid_w: int,
    removed_count: int,
) -> list[dict[str, int]] | None:
    removed: list[dict[str, int]] = []
    for _ in range(removed_count):
        built = None
        for _ in range(RECORD_ATTEMPTS):
            rect_h, rect_w = _sample_dims(rng, keep=False)
            if rect_h >= grid_h - 1 or rect_w >= grid_w - 1:
                continue
            r0 = int(rng.integers(1, grid_h - rect_h))
            c0 = int(rng.integers(1, grid_w - rect_w))
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _expanded_clear(bbox, occupied):
                continue
            _edge, _dist2, unique = _nearest_edge(bbox[0], bbox[1], bbox[2], bbox[3], grid_h, grid_w)
            if not unique:
                continue
            built = {
                "r0": int(bbox[0]),
                "c0": int(bbox[1]),
                "r1": int(bbox[2]),
                "c1": int(bbox[3]),
                "h": rect_h,
                "w": rect_w,
            }
            break
        if built is None:
            return None
        removed.append(built)
        _mark_bbox(occupied, (built["r0"], built["c0"], built["r1"], built["c1"]))
    return removed


def _assign_colors(
    rng: np.random.Generator,
    kept_input: list[dict[str, int]],
    kept_output: list[dict[str, int]],
    removed: list[dict[str, int]],
) -> None:
    total = len(kept_input) + len(removed)
    colors = [int(v) for v in rng.choice(PALETTE, size=total, replace=False)]
    for record, color in zip(kept_input, colors[: len(kept_input)]):
        record["color"] = color
    for record, color in zip(kept_output, colors[: len(kept_output)]):
        record["color"] = color
    for record, color in zip(removed, colors[len(kept_input) :]):
        record["color"] = color


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid_w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    keep_count = int(rng.integers(KEEP_COUNT_MIN, KEEP_COUNT_MAX + 1))
    removed_count = int(rng.integers(REMOVED_COUNT_MIN, REMOVED_COUNT_MAX + 1))
    total = keep_count + removed_count
    if total > len(PALETTE):
        return None

    kept_output = _build_output_records(rng, grid_h, grid_w, keep_count)
    if kept_output is None:
        return None

    built_input = _build_input_from_output(kept_output, grid_h, grid_w)
    if built_input is None:
        return None
    kept_input, occupied = built_input

    removed = _build_removed_records(rng, occupied, grid_h, grid_w, removed_count)
    if removed is None:
        return None

    _assign_colors(rng, kept_input, kept_output, removed)
    grid = _paint_rectangles((grid_h, grid_w), kept_input + removed)
    expected = _paint_rectangles((grid_h, grid_w), kept_output)

    if len(_extract_rectangles(grid)) != total:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(CASE_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
