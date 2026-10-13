from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 800


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _touching_side(box: tuple[int, int, int, int], h: int, w: int) -> str:
    r0, c0, r1, c1 = box
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each rectangle must touch exactly one outer edge"
    return sides[0]


def _slide_distance(box: tuple[int, int, int, int], side: str, h: int, w: int) -> int:
    r0, c0, r1, c1 = box
    if side == "top":
        return int(h - 1 - r1)
    if side == "bottom":
        return int(r0)
    if side == "left":
        return int(w - 1 - c1)
    return int(c0)


def _extract_edge_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"

    rectangles: list[dict] = []
    total_area = 0
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components_mask(g == color)
        assert len(comps) == 1, "each foreground color must form one rectangle"
        coords = comps[0]
        box = _bbox_coords(coords)
        area = int(coords.shape[0])
        bh = _bbox_h(box)
        bw = _bbox_w(box)
        assert area == bh * bw, "foreground objects must be solid rectangles"
        side = _touching_side(box, *g.shape)
        depth = bh if side in {"top", "bottom"} else bw
        assert 1 <= depth <= 3, "depth must be in 1..3"
        rectangles.append(
            {
                "color": color,
                "bbox": box,
                "side": side,
                "depth": int(depth),
                "distance": _slide_distance(box, side, *g.shape),
            }
        )
        total_area += area

    assert 3 <= len(rectangles) <= 5, "expected 3-5 edge rectangles"
    assert len({rect["color"] for rect in rectangles}) == len(rectangles), "colors must be unique"
    assert len({rect["distance"] for rect in rectangles}) == len(rectangles), "distances must be unique"
    assert total_area == int((g != bg).sum()), "foreground decomposition mismatch"
    return rectangles


def _build_nested(rectangles: list[dict], *, override_depth: int | None = None) -> np.ndarray:
    assert rectangles
    first = rectangles[0]
    center_side = int(override_depth if override_depth is not None else first["depth"])
    cur = np.full((center_side, center_side), int(first["color"]), dtype=int)
    for rect in rectangles[1:]:
        thickness = int(override_depth if override_depth is not None else rect["depth"])
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(rect["color"]), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _orientation(rect: dict) -> str:
    return "horizontal" if rect["side"] in {"top", "bottom"} else "vertical"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build concentric square layers whose order comes from edge-slide distance and whose thickness comes from rectangle depth.

    Input structure:
    - A black background contains 3-5 solid monochrome rectangles of distinct colors.
    - Every rectangle touches exactly one outer edge: top, bottom, left, or right.
    - The rectangle depth is its size perpendicular to the touched edge, and each depth is 1-3 cells.

    Rule:
    - For each edge rectangle, detect which edge it touches.
    - Compute its depth and its slide distance, defined as the distance from the rectangle's inner edge to the opposite grid boundary.
    - Sort rectangles by slide distance from small to large.
    - The nearest rectangle becomes the central solid square with side length equal to its depth; each later rectangle wraps the current square with a same-color frame whose thickness equals that rectangle's depth.

    Key insight:
    - Each source rectangle contributes two independent parameters from the same geometry: edge-relative depth sets layer thickness, while distance to the opposite boundary sets the nesting order.
    """
    rectangles = _extract_edge_rectangles(np.asarray(grid, dtype=int))
    ordered = sorted(rectangles, key=lambda rect: int(rect["distance"]))
    return _build_nested(ordered)


def _choose_canvas() -> tuple[int, int]:
    for _ in range(100):
        h = random.randint(13, 18)
        w = h + random.choice((-2, -1, 1, 2))
        if 11 <= w <= 19:
            return int(h), int(w)
    raise RuntimeError("failed to choose canvas")


def _choose_side_lists(n_h: int, n_v: int, total: int) -> tuple[list[str], list[str]] | None:
    min_distinct = 3 if total >= 4 else 2
    for _ in range(80):
        h_sides = [random.choice(("top", "bottom")) for _ in range(n_h)]
        v_sides = [random.choice(("left", "right")) for _ in range(n_v)]
        if len(set(h_sides + v_sides)) >= min_distinct:
            return h_sides, v_sides
    return None


def _choose_params() -> tuple[int, int, list[dict]] | None:
    for _ in range(300):
        h, w = _choose_canvas()
        total = random.choices([3, 4, 5], weights=[2, 4, 4])[0]
        n_h = random.randint(1, min(3, total - 1))
        n_v = total - n_h
        if not (1 <= n_v <= 3):
            continue

        depths_h = random.sample([1, 2, 3], k=n_h)
        depths_v = random.sample([1, 2, 3], k=n_v)
        if not (set(depths_h) & set(depths_v)):
            continue

        side_lists = _choose_side_lists(n_h, n_v, total)
        if side_lists is None:
            continue
        h_sides, v_sides = side_lists
        colors = random.sample(PALETTE, k=total)

        params: list[dict] = []
        for depth, side in zip(depths_h, h_sides, strict=True):
            params.append(
                {
                    "side": side,
                    "depth": int(depth),
                    "distance": int(h - depth),
                }
            )
        for depth, side in zip(depths_v, v_sides, strict=True):
            params.append(
                {
                    "side": side,
                    "depth": int(depth),
                    "distance": int(w - depth),
                }
            )

        if len({param["distance"] for param in params}) != total:
            continue

        ordered = sorted(params, key=lambda rect: int(rect["distance"]))
        depth_seq = [int(rect["depth"]) for rect in ordered]
        orient_seq = [_orientation(rect) for rect in ordered]
        if depth_seq == sorted(depth_seq) or depth_seq == sorted(depth_seq, reverse=True):
            continue
        if not any(orient_seq[i] != orient_seq[i + 1] for i in range(total - 1)):
            continue

        for idx, param in enumerate(params):
            param["color"] = int(colors[idx])
        random.shuffle(params)
        return h, w, params
    return None


def _span_limit(side_counts: Counter, side: str, long_dim: int) -> int:
    slots = int(side_counts[side])
    available = int(long_dim - 2)
    return max(2, min(available, available // slots + 2))


def _try_place_rectangles(h: int, w: int, params: list[dict]) -> np.ndarray | None:
    side_counts = Counter(param["side"] for param in params)
    for _ in range(250):
        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        order = params[:]
        random.shuffle(order)
        ok = True

        for param in order:
            side = str(param["side"])
            depth = int(param["depth"])
            color = int(param["color"])

            if side in {"top", "bottom"}:
                span_hi = _span_limit(side_counts, side, w)
                span = random.randint(2, span_hi)
                placed = False
                for _ in range(120):
                    r0 = 0 if side == "top" else h - depth
                    c0 = random.randint(1, w - span - 1)
                    if blocked[r0 : r0 + depth, c0 : c0 + span].any():
                        continue
                    grid[r0 : r0 + depth, c0 : c0 + span] = color
                    rr0 = max(0, r0 - 1)
                    rr1 = min(h, r0 + depth + 1)
                    cc0 = max(0, c0 - 1)
                    cc1 = min(w, c0 + span + 1)
                    blocked[rr0:rr1, cc0:cc1] = True
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            else:
                span_hi = _span_limit(side_counts, side, h)
                span = random.randint(2, span_hi)
                placed = False
                for _ in range(120):
                    r0 = random.randint(1, h - span - 1)
                    c0 = 0 if side == "left" else w - depth
                    if blocked[r0 : r0 + span, c0 : c0 + depth].any():
                        continue
                    grid[r0 : r0 + span, c0 : c0 + depth] = color
                    rr0 = max(0, r0 - 1)
                    rr1 = min(h, r0 + span + 1)
                    cc0 = max(0, c0 - 1)
                    cc1 = min(w, c0 + depth + 1)
                    blocked[rr0:rr1, cc0:cc1] = True
                    placed = True
                    break
                if not placed:
                    ok = False
                    break

        if ok:
            return grid
    return None


def _semantic_checks(grid: np.ndarray) -> bool:
    rectangles = _extract_edge_rectangles(grid)
    ordered = sorted(rectangles, key=lambda rect: int(rect["distance"]))
    correct = _build_nested(ordered)

    wrong_depth_asc = _build_nested(sorted(rectangles, key=lambda rect: int(rect["depth"])))
    wrong_depth_desc = _build_nested(sorted(rectangles, key=lambda rect: -int(rect["depth"])))
    wrong_uniform = _build_nested(ordered, override_depth=1)

    if np.array_equal(correct, wrong_depth_asc):
        return False
    if np.array_equal(correct, wrong_depth_desc):
        return False
    if np.array_equal(correct, wrong_uniform):
        return False

    depth_seq = [int(rect["depth"]) for rect in ordered]
    if len(set(depth_seq)) < 2:
        return False
    if max(depth_seq.count(depth) for depth in set(depth_seq)) < 2:
        return False

    orient_seq = [_orientation(rect) for rect in ordered]
    if not any(orient_seq[i] != orient_seq[i + 1] for i in range(len(orient_seq) - 1)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    chosen = _choose_params()
    if chosen is None:
        return None
    h, w, params = chosen
    grid = _try_place_rectangles(h, w, params)
    if grid is None:
        return None

    rectangles = _extract_edge_rectangles(grid)
    intended = sorted((param["color"], param["side"], param["depth"], param["distance"]) for param in params)
    observed = sorted((rect["color"], rect["side"], rect["depth"], rect["distance"]) for rect in rectangles)
    if intended != observed:
        return None
    if not _semantic_checks(grid):
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-distance/frame-depth puzzle input")
