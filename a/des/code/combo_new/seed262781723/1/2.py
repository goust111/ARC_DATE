from __future__ import annotations

from collections import Counter, deque
import numpy as np


BG = 0
ARROW = 5
OVERLAP = 7
OBJECT_COLORS = (1, 2, 3, 4, 6, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

# The 3-pixel arrow is a 2x2 L triomino. The missing corner determines direction.
PATTERN_TO_DIR = {
    frozenset(((0, 1), (1, 0), (1, 1))): UP,
    frozenset(((0, 0), (0, 1), (1, 0))): DOWN,
    frozenset(((0, 0), (0, 1), (1, 1))): LEFT,
    frozenset(((0, 0), (1, 0), (1, 1))): RIGHT,
}
DIR_TO_PATTERN = {direction: tuple(sorted(pattern)) for pattern, direction in PATTERN_TO_DIR.items()}

DIR_NAMES = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_arrow(gray_points: np.ndarray) -> tuple[int, int] | None:
    if gray_points.shape != (3, 2):
        return None
    r0 = int(gray_points[:, 0].min())
    c0 = int(gray_points[:, 1].min())
    r1 = int(gray_points[:, 0].max())
    c1 = int(gray_points[:, 1].max())
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    normalized = frozenset((int(r - r0), int(c - c0)) for r, c in gray_points)
    return PATTERN_TO_DIR.get(normalized)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BG):
            continue
        colors = {int(v) for v in np.unique(patch) if int(v) not in (BG, ARROW)}
        if len(colors) != 1:
            continue
        gray_points = np.argwhere(patch == ARROW)
        direction = _detect_arrow(gray_points)
        if direction is None:
            continue
        objects.append(
            {
                "color": next(iter(colors)),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "direction": direction,
            }
        )
    return objects


def _render_translated(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    cover_count = np.zeros((h, w), dtype=int)
    first_color = np.full((h, w), BG, dtype=int)
    counts = Counter(int(obj["color"]) for obj in objects)

    for obj in objects:
        step = int(counts[int(obj["color"])])
        dy, dx = obj["direction"]
        nr0 = int(obj["r0"] + dy * step)
        nc0 = int(obj["c0"] + dx * step)
        for rr in range(nr0, nr0 + int(obj["h"])):
            if not (0 <= rr < h):
                continue
            for cc in range(nc0, nc0 + int(obj["w"])):
                if not (0 <= cc < w):
                    continue
                cover_count[rr, cc] += 1
                if cover_count[rr, cc] == 1:
                    first_color[rr, cc] = int(obj["color"])

    out = np.full((h, w), BG, dtype=int)
    out[cover_count == 1] = first_color[cover_count == 1]
    out[cover_count >= 2] = OVERLAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored rectangle by the number of same-color rectangles, following its internal arrow.

    Input structure:
    - Black background with several separated filled rectangles in different colors.
    - Each rectangle contains exactly one gray 3-pixel L-shaped arrow marker inside it.
    - Rectangles may share colors, but input rectangles do not overlap.

    Rule:
    - Group rectangles by their non-gray body color and count how many rectangles each color has.
    - For every rectangle, move its full bounding rectangle by N cells in the arrow direction, where N is that color-count.
    - Clear all original rectangles; draw only the translated solid rectangles, with the gray arrow markers removed.
    - If two or more translated rectangles cover the same output cells, replace the overlapped cells with orange.

    Key insight:
    - The arrow gives direction, but the travel distance is a global property: the count of all rectangles with the same color.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    if not objects:
        return g.copy()
    return _render_translated(objects, g.shape)


def _rects_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    rr0 = max(ar0, br0)
    cc0 = max(ac0, bc0)
    rr1 = min(ar0 + ah, br0 + bh)
    cc1 = min(ac0 + aw, bc0 + bw)
    return max(0, rr1 - rr0) * max(0, cc1 - cc0)


def _shifted_box(obj: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    dy = int(obj["direction"][0] * obj["step"])
    dx = int(obj["direction"][1] * obj["step"])
    return r0 + dy, c0 + dx, int(obj["h"]), int(obj["w"])


def _sample_color_groups(rng: np.random.Generator) -> tuple[list[int], list[int]] | None:
    for _ in range(60):
        n_colors = int(rng.integers(2, 5))
        colors = [int(v) for v in rng.choice(np.array(OBJECT_COLORS), size=n_colors, replace=False)]
        counts = [int(rng.integers(1, 4)) for _ in range(n_colors)]
        if max(counts) == 1:
            counts[int(rng.integers(0, n_colors))] = 2
        total = sum(counts)
        if 3 <= total <= 7:
            return colors, counts
    return None


def _sample_objects(rng: np.random.Generator) -> list[dict] | None:
    groups = _sample_color_groups(rng)
    if groups is None:
        return None
    colors, counts = groups

    objects: list[dict] = []
    for color, count in zip(colors, counts):
        for _ in range(count):
            h = int(rng.integers(4, 7))
            w = int(rng.integers(4, 7))
            direction = DIRS[int(rng.integers(0, len(DIRS)))]
            arrow_r = int(rng.integers(1, h - 2))
            arrow_c = int(rng.integers(1, w - 2))
            objects.append(
                {
                    "color": int(color),
                    "step": int(count),
                    "h": h,
                    "w": w,
                    "direction": direction,
                    "arrow_r": arrow_r,
                    "arrow_c": arrow_c,
                }
            )

    if len({obj["direction"] for obj in objects}) == 1 and len(objects) >= 2:
        idx = int(rng.integers(0, len(objects)))
        choices = [d for d in DIRS if d != objects[idx]["direction"]]
        objects[idx]["direction"] = choices[int(rng.integers(0, len(choices)))]

    rng.shuffle(objects)
    return objects


def _grid_shape_for(objects: list[dict], rng: np.random.Generator) -> tuple[int, int] | None:
    total_area = sum((int(obj["h"]) + 2) * (int(obj["w"]) + 2) for obj in objects)
    max_span = max(max(int(obj["h"]), int(obj["w"])) + int(obj["step"]) for obj in objects)
    base = max(14, max_span + 4, int(np.ceil(np.sqrt(total_area * 1.15))))
    if base > 30:
        return None
    low = max(14, base - 2)
    high = min(30, base + 4)
    h = int(rng.integers(low, high + 1))
    w = int(rng.integers(low, high + 1))
    return h, w


def _input_ranges(obj: dict, h: int, w: int) -> tuple[int, int, int, int] | None:
    shift_r = int(obj["direction"][0] * obj["step"])
    shift_c = int(obj["direction"][1] * obj["step"])
    r_lo = max(0, -shift_r)
    c_lo = max(0, -shift_c)
    r_hi = min(h - int(obj["h"]), h - int(obj["h"]) - shift_r)
    c_hi = min(w - int(obj["w"]), w - int(obj["w"]) - shift_c)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return r_lo, r_hi, c_lo, c_hi


def _place_nonoverlap(
    obj: dict,
    h: int,
    w: int,
    input_boxes: list[tuple[int, int, int, int]],
    output_boxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    ranges = _input_ranges(obj, h, w)
    if ranges is None:
        return None
    r_lo, r_hi, c_lo, c_hi = ranges
    for _ in range(200):
        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        in_box = (r0, c0, int(obj["h"]), int(obj["w"]))
        out_box = _shifted_box(obj, r0, c0)
        if any(_rects_conflict(in_box, other, pad=1) for other in input_boxes):
            continue
        if any(_rects_conflict(out_box, other, pad=1) for other in output_boxes):
            continue
        return r0, c0
    return None


def _place_overlap(
    obj: dict,
    target_out_box: tuple[int, int, int, int],
    h: int,
    w: int,
    input_boxes: list[tuple[int, int, int, int]],
    output_boxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    tr0, tc0, th, tw = target_out_box
    out_r_lo = max(0, tr0 - int(obj["h"]) + 1)
    out_r_hi = min(h - int(obj["h"]), tr0 + th - 1)
    out_c_lo = max(0, tc0 - int(obj["w"]) + 1)
    out_c_hi = min(w - int(obj["w"]), tc0 + tw - 1)
    if out_r_lo > out_r_hi or out_c_lo > out_c_hi:
        return None

    shift_r = int(obj["direction"][0] * obj["step"])
    shift_c = int(obj["direction"][1] * obj["step"])
    target_area = th * tw
    this_area = int(obj["h"]) * int(obj["w"])

    for _ in range(300):
        out_r = int(rng.integers(out_r_lo, out_r_hi + 1))
        out_c = int(rng.integers(out_c_lo, out_c_hi + 1))
        out_box = (out_r, out_c, int(obj["h"]), int(obj["w"]))
        area = _intersection_area(out_box, target_out_box)
        if area < 2 or area >= min(target_area, this_area):
            continue
        if any(other != target_out_box and _rects_conflict(out_box, other, pad=1) for other in output_boxes):
            continue
        r0 = out_r - shift_r
        c0 = out_c - shift_c
        in_box = (r0, c0, int(obj["h"]), int(obj["w"]))
        if r0 < 0 or c0 < 0 or r0 + int(obj["h"]) > h or c0 + int(obj["w"]) > w:
            continue
        if any(_rects_conflict(in_box, other, pad=1) for other in input_boxes):
            continue
        return r0, c0
    return None


def _choose_overlap_pair(objects: list[dict], rng: np.random.Generator) -> tuple[int, int] | None:
    pairs = [(i, j) for i in range(len(objects)) for j in range(i + 1, len(objects))]
    if not pairs:
        return None
    preferred = [(i, j) for i, j in pairs if int(objects[i]["color"]) != int(objects[j]["color"])]
    pool = preferred if preferred else pairs
    return pool[int(rng.integers(0, len(pool)))]


def _place_objects(objects: list[dict], h: int, w: int, force_overlap: bool, rng: np.random.Generator) -> list[dict] | None:
    placed = [dict(obj) for obj in objects]
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    overlap_pair = _choose_overlap_pair(placed, rng) if force_overlap else None
    if overlap_pair is None:
        order = sorted(range(len(placed)), key=lambda i: -(int(placed[i]["h"]) * int(placed[i]["w"])))
    else:
        first, second = overlap_pair
        rest = [i for i in range(len(placed)) if i not in overlap_pair]
        rest.sort(key=lambda i: -(int(placed[i]["h"]) * int(placed[i]["w"])))
        order = [first, second] + rest

    for idx in order:
        obj = placed[idx]
        if overlap_pair is not None and idx == overlap_pair[1]:
            target_box = placed[overlap_pair[0]]["out_box"]
            pos = _place_overlap(obj, target_box, h, w, input_boxes, output_boxes, rng)
        else:
            pos = _place_nonoverlap(obj, h, w, input_boxes, output_boxes, rng)
        if pos is None:
            return None
        r0, c0 = pos
        in_box = (r0, c0, int(obj["h"]), int(obj["w"]))
        out_box = _shifted_box(obj, r0, c0)
        obj["r0"] = r0
        obj["c0"] = c0
        obj["out_box"] = out_box
        input_boxes.append(in_box)
        output_boxes.append(out_box)

    if overlap_pair is None:
        if any(_intersection_area(a["out_box"], b["out_box"]) > 0 for i, a in enumerate(placed) for b in placed[i + 1 :]):
            return None
    else:
        overlap_cells = sum(
            1
            for i, a in enumerate(placed)
            for b in placed[i + 1 :]
            if _intersection_area(a["out_box"], b["out_box"]) > 0
        )
        if overlap_cells == 0:
            return None

    return placed


def _build_input_grid(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h0 = int(obj["h"])
        w0 = int(obj["w"])
        color = int(obj["color"])
        grid[r0 : r0 + h0, c0 : c0 + w0] = color
        arrow_origin_r = r0 + int(obj["arrow_r"])
        arrow_origin_c = c0 + int(obj["arrow_c"])
        for dr, dc in DIR_TO_PATTERN[obj["direction"]]:
            grid[arrow_origin_r + dr, arrow_origin_c + dc] = ARROW
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    objects = _sample_objects(rng)
    if objects is None:
        return None
    shape = _grid_shape_for(objects, rng)
    if shape is None:
        return None
    force_overlap = bool(rng.integers(0, 2))
    placed = _place_objects(objects, shape[0], shape[1], force_overlap, rng)
    if placed is None:
        return None

    grid = _build_input_grid(placed, shape)
    expected = _render_translated(placed, shape)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if force_overlap and not np.any(expected == OVERLAP):
        return None
    if (not force_overlap) and np.any(expected == OVERLAP):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
