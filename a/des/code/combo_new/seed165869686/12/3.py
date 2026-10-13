from __future__ import annotations

import random

import numpy as np


ARC_MAX = 30
COLORS = tuple(range(10))
CORNERS = ("tl", "tr", "bl", "br")
OPPOSITE_CORNER = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}

# --------------- solver helpers ---------------


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox_intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _marker_bbox_from_rect(rect: dict[str, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    corner = rect["corner"]
    if corner == "tl":
        return (r0, c0, r0 + 1, c0 + 1)
    if corner == "tr":
        return (r0, c1 - 1, r0 + 1, c1)
    if corner == "bl":
        return (r1 - 1, c0, r1, c0 + 1)
    return (r1 - 1, c1 - 1, r1, c1)


def _marker_cells_from_bbox(
    bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _body_probe_cells(
    marker_bbox: tuple[int, int, int, int], corner: str
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = marker_bbox
    if corner == "tl":
        return [(r0, c1 + 1), (r1, c1 + 1), (r1 + 1, c0), (r1 + 1, c1)]
    if corner == "tr":
        return [(r0, c0 - 1), (r1, c0 - 1), (r1 + 1, c0), (r1 + 1, c1)]
    if corner == "bl":
        return [(r0 - 1, c0), (r0 - 1, c1), (r0, c1 + 1), (r1, c1 + 1)]
    return [(r0 - 1, c0), (r0 - 1, c1), (r0, c0 - 1), (r1, c0 - 1)]


def _infer_corner_and_body(
    grid: np.ndarray,
    marker_bbox: tuple[int, int, int, int],
    bg: int,
    marker_colors: set[int],
) -> tuple[str, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int]] = []
    for corner in CORNERS:
        probe = _body_probe_cells(marker_bbox, corner)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in probe):
            continue
        values = [int(grid[r, c]) for r, c in probe]
        if len(set(values)) != 1:
            continue
        body = values[0]
        if body == bg or body in marker_colors:
            continue
        candidates.append((corner, body))
    assert len(candidates) == 1, "each marker must identify exactly one rectangle corner"
    return candidates[0]


def _reconstruct_bbox(
    body_coords: np.ndarray,
    marker_bbox: tuple[int, int, int, int],
    corner: str,
) -> tuple[int, int, int, int]:
    rows = body_coords[:, 0]
    cols = body_coords[:, 1]
    mr0, mc0, mr1, mc1 = marker_bbox
    if corner == "tl":
        return (mr0, mc0, int(rows.max()), int(cols.max()))
    if corner == "tr":
        return (mr0, int(cols.min()), int(rows.max()), mc1)
    if corner == "bl":
        return (int(rows.min()), mc0, mr1, int(cols.max()))
    return (int(rows.min()), int(cols.min()), mr1, mc1)


def _topological_order(edges: list[set[int]], tie_keys: list[tuple[int, int, int]]) -> list[int]:
    indegree = [0] * len(edges)
    for src, nbrs in enumerate(edges):
        for dst in nbrs:
            indegree[dst] += 1

    available = [i for i, deg in enumerate(indegree) if deg == 0]
    available.sort(key=lambda idx: tie_keys[idx])
    order: list[int] = []

    while available:
        cur = available.pop(0)
        order.append(cur)
        for nxt in sorted(edges[cur], key=lambda idx: tie_keys[idx]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                available.append(nxt)
        available.sort(key=lambda idx: tie_keys[idx])

    assert len(order) == len(edges), "rectangle overlap graph must be acyclic"
    return order


def _render_scene(
    shape: tuple[int, int], bg: int, rects: list[dict[str, int]], order: list[int]
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rects[idx]["body"]
    for rect in rects:
        mr0, mc0, mr1, mc1 = _marker_bbox_from_rect(rect)
        out[mr0 : mr1 + 1, mc0 : mc1 + 1] = rect["marker"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the stacking order of overlapping rectangles by reconstructing them from their visible corner markers.

    Input structure
    - The dominant color is the background.
    - Several solid rectangles of distinct colors partially overlap like stacked windows.
    - Each rectangle has one always-visible 2x2 marker block of a different color placed on one true corner of the rectangle.

    Rule
    - Find the 2x2 marker blocks, then use the visible same-color L-shaped body cells next to each marker to determine which rectangle corner it marks and which body color belongs to that rectangle.
    - Recover each full rectangle bbox from the marked corner plus the furthest remaining visible body cells, then compare every overlapping pair: inside their shared area, the visible body color belongs to the rectangle that was originally on top.
    - Repaint the same full rectangles in the reversed bottom-to-top order and finally restore every marker block at its original corner.

    Key insight
    - The 2x2 blocks are corner anchors rather than standalone objects: they reveal hidden rectangle extents, while overlap visibility reveals the original front/back layering that must be inverted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    counts = np.bincount(g.ravel(), minlength=10)

    marker_infos: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in set(map(int, np.unique(g))) - {bg}:
        if int(counts[color]) != 4:
            continue
        coords = np.argwhere(g == int(color))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0) != 1 or int(c1 - c0) != 1:
            continue
        if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
            marker_infos.append({"marker": int(color), "bbox": (int(r0), int(c0), int(r1), int(c1))})

    assert marker_infos, "expected at least one 2x2 marker block"
    marker_colors = {int(info["marker"]) for info in marker_infos}

    rects: list[dict[str, int]] = []
    used_bodies: set[int] = set()
    for info in marker_infos:
        marker = int(info["marker"])
        marker_bbox = info["bbox"]
        assert isinstance(marker_bbox, tuple)
        corner, body = _infer_corner_and_body(g, marker_bbox, bg, marker_colors)
        assert body not in used_bodies, "rectangle body colors must be unique"
        used_bodies.add(body)
        body_coords = np.argwhere(g == body)
        assert body_coords.size > 0, "each rectangle must have visible body cells"
        bbox = _reconstruct_bbox(body_coords, marker_bbox, corner)
        r0, c0, r1, c1 = bbox
        assert r0 < r1 and c0 < c1, "reconstructed rectangle must be at least 2x2"
        rects.append(
            {
                "bbox": bbox,
                "body": body,
                "marker": marker,
                "corner": corner,
            }
        )

    marker_mask = np.zeros(g.shape, dtype=bool)
    for rect in rects:
        mr0, mc0, mr1, mc1 = _marker_bbox_from_rect(rect)
        marker_mask[mr0 : mr1 + 1, mc0 : mc1 + 1] = True

    edges: list[set[int]] = [set() for _ in rects]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            inter = _bbox_intersection(rects[i]["bbox"], rects[j]["bbox"])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            submask = ~marker_mask[r0 : r1 + 1, c0 : c1 + 1]
            if not bool(submask.any()):
                continue
            visible = set(map(int, np.unique(g[r0 : r1 + 1, c0 : c1 + 1][submask])))
            visible.discard(bg)
            visible -= marker_colors
            if not visible:
                continue

            body_i = rects[i]["body"]
            body_j = rects[j]["body"]
            if visible == {body_i}:
                edges[j].add(i)
            elif visible == {body_j}:
                edges[i].add(j)
            else:
                raise AssertionError("overlap must reveal exactly one top rectangle")

    tie_keys = [
        (
            int(np.count_nonzero(g == rect["body"])),
            rect["bbox"][0],
            rect["bbox"][1],
        )
        for rect in rects
    ]
    original_order = _topological_order(edges, tie_keys)
    reversed_order = list(reversed(original_order))
    return _render_scene(g.shape, bg, rects, reversed_order)


# --------------- generator helpers ---------------


GEN_RETRIES = 500
RECT_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 6
RECT_SIDE_MAX = 11
OVERLAP_MIN = 3
MARGIN_MIN = 1
MARGIN_MAX = 3


def _rect_dict(
    r0: int, c0: int, h: int, w: int, body: int | None = None, marker: int | None = None
) -> dict[str, int]:
    return {
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "h": int(h),
        "w": int(w),
        "body": -1 if body is None else int(body),
        "marker": -1 if marker is None else int(marker),
        "corner": "tl",
    }


def _place_chain(rect_count: int) -> list[dict[str, int]] | None:
    first = _rect_dict(0, 0, random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX), random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX))
    rects = [first]

    for _ in range(1, rect_count):
        prev = rects[-1]
        for _attempt in range(120):
            h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
            w = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
            dr = random.choice((-1, 1))
            dc = random.choice((-1, 1))
            max_ov_h = min(prev["h"] - 3, h - 3)
            max_ov_w = min(prev["w"] - 3, w - 3)
            if max_ov_h < OVERLAP_MIN or max_ov_w < OVERLAP_MIN:
                continue
            ov_h = random.randint(OVERLAP_MIN, max_ov_h)
            ov_w = random.randint(OVERLAP_MIN, max_ov_w)

            pr0, pc0, pr1, pc1 = prev["bbox"]
            if dr > 0:
                r0 = pr1 - ov_h + 1
            else:
                r0 = pr0 - h + ov_h
            if dc > 0:
                c0 = pc1 - ov_w + 1
            else:
                c0 = pc0 - w + ov_w

            cur = _rect_dict(r0, c0, h, w)
            if any(_bbox_intersection(cur["bbox"], older["bbox"]) is not None for older in rects[:-1]):
                continue
            rects.append(cur)
            break
        else:
            return None

    return rects


def _overlap_corner(lower: dict[str, int], upper: dict[str, int]) -> str:
    inter = _bbox_intersection(lower["bbox"], upper["bbox"])
    assert inter is not None
    lr0, lc0, lr1, lc1 = lower["bbox"]
    ir0, ic0, _ir1, _ic1 = inter
    vertical = "t" if ir0 == lr0 else "b"
    horizontal = "l" if ic0 == lc0 else "r"
    corner = vertical + horizontal
    assert corner in CORNERS
    return corner


def _shift_rects(rects: list[dict[str, int]]) -> tuple[list[dict[str, int]], tuple[int, int]] | None:
    min_r = min(rect["bbox"][0] for rect in rects)
    min_c = min(rect["bbox"][1] for rect in rects)
    max_r = max(rect["bbox"][2] for rect in rects)
    max_c = max(rect["bbox"][3] for rect in rects)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    top = random.randint(MARGIN_MIN, MARGIN_MAX)
    left = random.randint(MARGIN_MIN, MARGIN_MAX)
    extra_h = random.randint(0, max(0, ARC_MAX - span_h - top - MARGIN_MIN))
    extra_w = random.randint(0, max(0, ARC_MAX - span_w - left - MARGIN_MIN))
    bottom = MARGIN_MIN + extra_h
    right = MARGIN_MIN + extra_w
    h = span_h + top + bottom
    w = span_w + left + right
    if h > ARC_MAX or w > ARC_MAX:
        return None

    dr = top - min_r
    dc = left - min_c
    shifted: list[dict[str, int]] = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        moved = dict(rect)
        moved["bbox"] = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
        shifted.append(moved)
    return shifted, (h, w)


def _marker_footprints_disjoint(rects: list[dict[str, int]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for rect in rects:
        for cell in _marker_cells_from_bbox(_marker_bbox_from_rect(rect)):
            if cell in seen:
                return False
            seen.add(cell)
    return True


def _visible_side_ok(grid: np.ndarray, rect: dict[str, int]) -> bool:
    body = rect["body"]
    r0, c0, r1, c1 = rect["bbox"]
    corner = rect["corner"]
    mr0, mc0, mr1, mc1 = _marker_bbox_from_rect(rect)

    if any(int(grid[r, c]) != body for r, c in _body_probe_cells((mr0, mc0, mr1, mc1), corner)):
        return False

    if corner == "tl":
        return bool(np.any(grid[r1, c0 : c1 + 1] == body) and np.any(grid[r0 : r1 + 1, c1] == body))
    if corner == "tr":
        return bool(np.any(grid[r1, c0 : c1 + 1] == body) and np.any(grid[r0 : r1 + 1, c0] == body))
    if corner == "bl":
        return bool(np.any(grid[r0, c0 : c1 + 1] == body) and np.any(grid[r0 : r1 + 1, c1] == body))
    return bool(np.any(grid[r0, c0 : c1 + 1] == body) and np.any(grid[r0 : r1 + 1, c0] == body))


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[bg]) > max(int(counts[c]) for c in range(10) if c != bg)


def _scene_is_clear(
    inp: np.ndarray, rects: list[dict[str, int]], original_order: list[int], bg: int
) -> bool:
    if not _background_is_dominant(inp, bg):
        return False
    if not _marker_footprints_disjoint(rects):
        return False

    marker_colors = {rect["marker"] for rect in rects}
    for rect in rects:
        if int(np.count_nonzero(inp == rect["body"])) < 8:
            return False
        if not _visible_side_ok(inp, rect):
            return False
        if rect["body"] == bg or rect["body"] in marker_colors:
            return False

    marker_mask = np.zeros(inp.shape, dtype=bool)
    for rect in rects:
        mr0, mc0, mr1, mc1 = _marker_bbox_from_rect(rect)
        marker_mask[mr0 : mr1 + 1, mc0 : mc1 + 1] = True

    for lower_idx, upper_idx in zip(original_order[:-1], original_order[1:]):
        inter = _bbox_intersection(rects[lower_idx]["bbox"], rects[upper_idx]["bbox"])
        if inter is None:
            return False
        r0, c0, r1, c1 = inter
        submask = ~marker_mask[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(submask.any()):
            return False
        visible = set(map(int, np.unique(inp[r0 : r1 + 1, c0 : c1 + 1][submask])))
        visible.discard(bg)
        visible -= marker_colors
        if visible != {rects[upper_idx]["body"]}:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    needed_colors = 1 + 2 * rect_count
    if needed_colors > len(COLORS):
        return None

    palette = random.sample(COLORS, needed_colors)
    bg = palette[0]
    body_colors = palette[1 : 1 + rect_count]
    marker_colors = palette[1 + rect_count :]

    rects = _place_chain(rect_count)
    if rects is None:
        return None

    for idx, rect in enumerate(rects):
        rect["body"] = body_colors[idx]
        rect["marker"] = marker_colors[idx]

    for idx in range(rect_count - 1):
        overlap = _overlap_corner(rects[idx], rects[idx + 1])
        rects[idx]["corner"] = OPPOSITE_CORNER[overlap]
    rects[-1]["corner"] = random.choice(CORNERS)

    shifted = _shift_rects(rects)
    if shifted is None:
        return None
    rects, shape = shifted

    original_order = list(range(rect_count))
    inp = _render_scene(shape, bg, rects, original_order)
    out = _render_scene(shape, bg, rects, list(reversed(original_order)))

    if not _scene_is_clear(inp, rects, original_order, bg):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid scene after {GEN_RETRIES} attempts")
