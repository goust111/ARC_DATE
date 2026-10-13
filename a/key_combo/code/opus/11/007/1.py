import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 5
TRAIL = 7
RECT_COLORS = (1, 2, 3)
EDGES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _boxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _cells_in_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _paint_bbox(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _edge_from_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = bbox
    hits = []
    if r0 == 0:
        hits.append("top")
    if r1 == h - 1:
        hits.append("bottom")
    if c0 == 0:
        hits.append("left")
    if c1 == w - 1:
        hits.append("right")
    assert len(hits) == 1, f"rectangle must touch exactly one edge, got {hits}"
    return hits[0]


def _dest_bbox(bbox: tuple[int, int, int, int], edge: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    bh = _bbox_h(bbox)
    bw = _bbox_w(bbox)
    if edge == "top":
        return (h - bh, c0, h - 1, c1)
    if edge == "bottom":
        return (0, c0, bh - 1, c1)
    if edge == "left":
        return (r0, w - bw, r1, w - 1)
    assert edge == "right"
    return (r0, 0, r1, bw - 1)


def _stripe_mask(bbox: tuple[int, int, int, int], edge: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    if edge in ("top", "bottom"):
        mask[:, c0 : c1 + 1] = True
    else:
        mask[r0 : r1 + 1, :] = True
    return mask


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in {BG, MARKER})
    rects = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, f"missing rectangle color {color}"
        bbox = _bbox_coords(coords)
        area = int(coords.shape[0])
        assert area == _bbox_h(bbox) * _bbox_w(bbox), f"color {color} is not a solid rectangle"
        r0, c0, r1, c1 = bbox
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), f"color {color} must fill its bbox"
        edge = _edge_from_bbox(bbox, g.shape)
        dest = _dest_bbox(bbox, edge, g.shape)
        stripe = _stripe_mask(bbox, edge, g.shape)
        rects.append(
            {
                "color": color,
                "bbox": bbox,
                "edge": edge,
                "dest": dest,
                "stripe": stripe,
            }
        )
    assert len(rects) == 3, f"expected exactly 3 rectangles, got {len(rects)}"
    rects.sort(key=lambda item: item["color"])
    return rects


def _destination_marker_count(grid: np.ndarray, dest: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = dest
    return int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == MARKER))


def _solve_metadata(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    rects = _detect_rectangles(g)
    dest_counts = [_destination_marker_count(g, rect["dest"]) for rect in rects]
    assert len(set(dest_counts)) == len(dest_counts), f"destination counts must be distinct, got {dest_counts}"
    loser = int(min(range(len(rects)), key=lambda idx: dest_counts[idx]))
    start_counts = [_destination_marker_count(g, rect["bbox"]) for rect in rects]
    stripe_counts = [int(np.count_nonzero((g == MARKER) & rect["stripe"])) for rect in rects]
    return {
        "rects": rects,
        "dest_counts": dest_counts,
        "start_counts": start_counts,
        "stripe_counts": stripe_counts,
        "loser": loser,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the edge-touching rectangles to the opposite side, then erase the one whose destination covers the fewest static markers.

    Input structure:
    - Black background contains exactly three solid rectangles in three different colors.
    - Each rectangle touches exactly one outer edge and therefore determines its slide direction.
    - Grey single-pixel markers are static clues: they are not inside any source rectangle and do not touch the grid border.

    Rule:
    - Slide all three rectangles simultaneously to the opposite edge along their row band or column band.
    - Each moving rectangle paints its entire swept stripe orange and lands at the opposite edge in its original color.
    - Count grey markers only inside each rectangle's destination footprint; markers elsewhere in the orange stripe do not count.
    - Remove the unique rectangle with the smallest destination-marker count together with its whole stripe; preserve the other two stripes, destination rectangles, and any untouched grey markers.

    Key insight:
    - The elimination score comes from markers covered after the slide, so the slide must be simulated before counting and stripe-only markers must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    meta = _solve_metadata(g)

    out = g.copy()
    moving_mask = ~np.isin(out, [BG, MARKER])
    out[moving_mask] = BG

    for idx, rect in enumerate(meta["rects"]):
        if idx == meta["loser"]:
            continue
        out[rect["stripe"]] = TRAIL

    for idx, rect in enumerate(meta["rects"]):
        if idx == meta["loser"]:
            continue
        _paint_bbox(out, rect["dest"], rect["color"])

    return out


def _rand_rect_dims(edge: str, h: int, w: int, rng: random.Random) -> tuple[int, int]:
    for _ in range(80):
        bh = rng.randint(2, min(4, h - 3))
        bw = rng.randint(2, min(4, w - 3))
        if edge in ("top", "bottom"):
            if 2 * bh >= h or (bh - 1) * bw < 3:
                continue
        else:
            if 2 * bw >= w or bh * (bw - 1) < 3:
                continue
        return bh, bw
    raise RuntimeError("failed to sample rectangle dimensions")


def _sample_layout(h: int, w: int, rng: random.Random) -> list[dict] | None:
    omitted = rng.choice(EDGES)
    edges = [edge for edge in EDGES if edge != omitted]
    colors = list(RECT_COLORS)
    rng.shuffle(colors)

    rects = []
    for edge, color in zip(edges, colors):
        bh, bw = _rand_rect_dims(edge, h, w, rng)
        if edge == "top":
            c0 = rng.randint(1, w - bw - 1)
            bbox = (0, c0, bh - 1, c0 + bw - 1)
        elif edge == "bottom":
            c0 = rng.randint(1, w - bw - 1)
            bbox = (h - bh, c0, h - 1, c0 + bw - 1)
        elif edge == "left":
            r0 = rng.randint(1, h - bh - 1)
            bbox = (r0, 0, r0 + bh - 1, bw - 1)
        else:
            r0 = rng.randint(1, h - bh - 1)
            bbox = (r0, w - bw, r0 + bh - 1, w - 1)
        edge_name = _edge_from_bbox(bbox, (h, w))
        dest = _dest_bbox(bbox, edge_name, (h, w))
        rects.append(
            {
                "color": color,
                "bbox": bbox,
                "edge": edge_name,
                "dest": dest,
                "stripe": _stripe_mask(bbox, edge_name, (h, w)),
            }
        )

    boxes = [rect["bbox"] for rect in rects]
    dests = [rect["dest"] for rect in rects]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _boxes_touch_or_overlap(boxes[i], boxes[j]):
                return None
            if _boxes_touch_or_overlap(dests[i], dests[j]):
                return None
    all_boxes = [(i, "src", rect["bbox"]) for i, rect in enumerate(rects)] + [
        (i, "dst", rect["dest"]) for i, rect in enumerate(rects)
    ]
    for i, kind_i, box_i in all_boxes:
        for j, kind_j, box_j in all_boxes:
            if (i, kind_i) >= (j, kind_j):
                continue
            if i == j and kind_i != kind_j:
                continue
            if _boxes_overlap(box_i, box_j):
                return None
    return rects


def _interior_cells(shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    return {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}


def _pick_cells(
    candidates: Iterable[tuple[int, int]],
    count: int,
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = [cell for cell in candidates if cell not in occupied]
    if len(pool) < count:
        return None
    chosen = rng.sample(pool, count)
    occupied.update(chosen)
    return chosen


def _build_marker_plan(rects: list[dict], shape: tuple[int, int], rng: random.Random) -> dict | None:
    pattern = rng.choice(((0, 2, 3), (1, 2, 3)))
    shuffled = list(pattern)
    rng.shuffle(shuffled)
    dest_counts = shuffled
    loser = int(min(range(3), key=lambda idx: dest_counts[idx]))
    contenders = [idx for idx in range(3) if idx != loser]
    stripe_extras = [0, 0, 0]
    stripe_extras[loser] = dest_counts[min(contenders, key=lambda idx: dest_counts[idx])] - dest_counts[loser] + 1
    if rng.random() < 0.65:
        stripe_extras[max(contenders, key=lambda idx: dest_counts[idx])] = 1
    survivors = rng.randint(1, 2)
    total = sum(dest_counts) + sum(stripe_extras) + survivors
    if total > 12:
        return None
    if total < 8:
        survivors += 8 - total
    if sum(dest_counts) + sum(stripe_extras) + survivors > 12:
        return None

    occupied: set[tuple[int, int]] = set()
    stripe_union = np.zeros(shape, dtype=bool)
    for rect in rects:
        stripe_union |= rect["stripe"]
    interior = _interior_cells(shape)
    all_src_cells = {cell for rect in rects for cell in _cells_in_bbox(rect["bbox"])}
    placement = {
        "dest": [[] for _ in range(3)],
        "extra": [[] for _ in range(3)],
        "survivors": [],
        "dest_counts": dest_counts,
        "stripe_extras": stripe_extras,
    }

    for idx, rect in enumerate(rects):
        dest_cells = [cell for cell in _cells_in_bbox(rect["dest"]) if cell in interior]
        chosen = _pick_cells(dest_cells, dest_counts[idx], occupied, rng)
        if chosen is None:
            return None
        placement["dest"][idx] = chosen

    for idx, rect in enumerate(rects):
        extra_cells = []
        for r, c in np.argwhere(rect["stripe"]):
            cell = (int(r), int(c))
            if cell not in interior:
                continue
            if cell in all_src_cells:
                continue
            if cell in _cells_in_bbox(rect["dest"]):
                continue
            extra_cells.append(cell)
        chosen = _pick_cells(extra_cells, stripe_extras[idx], occupied, rng)
        if chosen is None:
            return None
        placement["extra"][idx] = chosen

    survivor_cells = []
    for r, c in np.argwhere(~stripe_union):
        cell = (int(r), int(c))
        if cell not in interior:
            continue
        if cell in all_src_cells:
            continue
        survivor_cells.append(cell)
    chosen = _pick_cells(survivor_cells, survivors, occupied, rng)
    if chosen is None:
        return None
    placement["survivors"] = chosen
    return placement


def _render_input(shape: tuple[int, int], rects: list[dict], marker_plan: dict) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        _paint_bbox(grid, rect["bbox"], rect["color"])
    for cells in marker_plan["dest"]:
        for r, c in cells:
            grid[r, c] = MARKER
    for cells in marker_plan["extra"]:
        for r, c in cells:
            grid[r, c] = MARKER
    for r, c in marker_plan["survivors"]:
        grid[r, c] = MARKER
    return grid


def _wrong_stripe_solver(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    meta = _solve_metadata(g)
    stripe_loser = int(min(range(3), key=lambda idx: meta["stripe_counts"][idx]))
    out = g.copy()
    moving_mask = ~np.isin(out, [BG, MARKER])
    out[moving_mask] = BG
    for idx, rect in enumerate(meta["rects"]):
        if idx == stripe_loser:
            continue
        out[rect["stripe"]] = TRAIL
    for idx, rect in enumerate(meta["rects"]):
        if idx == stripe_loser:
            continue
        _paint_bbox(out, rect["dest"], rect["color"])
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    marker_cells = np.argwhere(g == MARKER)
    if not (8 <= len(marker_cells) <= 12):
        return False
    if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in marker_cells):
        return False

    try:
        meta = _solve_metadata(g)
    except AssertionError:
        return False

    if len(set(meta["dest_counts"])) != 3:
        return False
    if len(set(meta["start_counts"])) != 1:
        return False
    if min(meta["stripe_counts"]) == meta["stripe_counts"][meta["loser"]]:
        return False
    if not any(count > 0 for idx, count in enumerate(meta["stripe_counts"]) if idx != meta["loser"]):
        return False

    stripe_union = np.zeros_like(g, dtype=bool)
    for rect in meta["rects"]:
        stripe_union |= rect["stripe"]
    survivor_count = int(np.count_nonzero((g == MARKER) & ~stripe_union))
    if survivor_count < 1:
        return False

    correct = main(g)
    wrong = _wrong_stripe_solver(g)
    if np.array_equal(correct, wrong):
        return False
    if np.array_equal(correct, g):
        return False

    kept_colors = sorted(int(c) for c in np.unique(correct) if int(c) not in {BG, MARKER, TRAIL})
    if len(kept_colors) != 2:
        return False
    if int(np.count_nonzero(correct == TRAIL)) == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 17)
    w = rng.randint(13, 17)
    rects = _sample_layout(h, w, rng)
    if rects is None:
        return None
    marker_plan = _build_marker_plan(rects, (h, w), rng)
    if marker_plan is None:
        return None
    grid = _render_input((h, w), rects, marker_plan)
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
