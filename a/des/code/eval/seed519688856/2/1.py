from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 5
INNER = 3
OUTER = INNER + 2
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR_ORDER = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


@dataclass
class Box:
    r0: int
    c0: int
    r1: int
    c1: int
    border: int
    gap_r: int
    gap_c: int
    direction: str
    template: np.ndarray

    def tile_origin(self, depth: int) -> tuple[int, int]:
        th, tw = self.template.shape
        if self.direction == "up":
            return self.r0 - th * (depth + 1), self.c0 + 1
        if self.direction == "down":
            return self.r1 + 1 + th * depth, self.c0 + 1
        if self.direction == "left":
            return self.r0 + 1, self.c0 - tw * (depth + 1)
        return self.r0 + 1, self.c1 + 1 + tw * depth


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common(arr: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _build_box(r0: int, c0: int, direction: str, template: np.ndarray, border: int = FRAME) -> Box:
    th, tw = template.shape
    r1 = r0 + th + 1
    c1 = c0 + tw + 1
    if direction == "up":
        gap_r, gap_c = r0, c0 + tw // 2 + 1
    elif direction == "down":
        gap_r, gap_c = r1, c0 + tw // 2 + 1
    elif direction == "left":
        gap_r, gap_c = r0 + th // 2 + 1, c0
    else:
        gap_r, gap_c = r0 + th // 2 + 1, c1
    return Box(r0, c0, r1, c1, int(border), int(gap_r), int(gap_c), direction, np.asarray(template, dtype=int))


def _draw_box(grid: np.ndarray, box: Box) -> None:
    grid[box.r0 : box.r1 + 1, box.c0] = box.border
    grid[box.r0 : box.r1 + 1, box.c1] = box.border
    grid[box.r0, box.c0 : box.c1 + 1] = box.border
    grid[box.r1, box.c0 : box.c1 + 1] = box.border
    grid[box.gap_r, box.gap_c] = BG
    grid[box.r0 + 1 : box.r1, box.c0 + 1 : box.c1] = box.template


def _expanded_overlap(a: Box, b: Box, margin: int = 1) -> bool:
    return not (
        a.r1 + margin < b.r0
        or b.r1 + margin < a.r0
        or a.c1 + margin < b.c0
        or b.c1 + margin < a.c0
    )


def _detect_boxes(grid: np.ndarray) -> list[Box]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g)
    boxes: list[Box] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        for comp in _components(g == color):
            if comp.shape[0] < 8:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue

            perimeter: list[tuple[int, int]] = []
            for c in range(c0, c1 + 1):
                perimeter.append((r0, c))
                perimeter.append((r1, c))
            for r in range(r0 + 1, r1):
                perimeter.append((r, c0))
                perimeter.append((r, c1))
            perimeter_set = set(perimeter)
            comp_set = {tuple(pt) for pt in comp.tolist()}
            if len(comp_set) != len(perimeter_set) - 1:
                continue
            if not comp_set <= perimeter_set:
                continue

            missing = list(perimeter_set - comp_set)
            if len(missing) != 1:
                continue
            gap_r, gap_c = missing[0]
            if g[gap_r, gap_c] != bg:
                continue
            if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
                continue

            if gap_r == r0:
                direction = "up"
            elif gap_r == r1:
                direction = "down"
            elif gap_c == c0:
                direction = "left"
            else:
                direction = "right"

            template = g[r0 + 1 : r1, c0 + 1 : c1]
            if template.shape[0] < 2 or template.shape[1] < 2:
                continue
            if np.any(template == bg) or np.any(template == color):
                continue
            if len(set(map(int, np.unique(template)))) != 2:
                continue

            boxes.append(Box(r0, c0, r1, c1, color, gap_r, gap_c, direction, template.copy()))

    boxes.sort(key=lambda box: (box.r0, box.c0, DIR_ORDER.index(box.direction)))
    return boxes


def _paint_spreads(base: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, bool, bool] | None:
    g = np.asarray(base, dtype=int)
    h, w = g.shape
    out = g.copy()
    best_depth = np.full((h, w), 10**9, dtype=int)
    best_order = np.full((h, w), 10**9, dtype=int)
    overlap = False
    blocked = False

    for order, box in enumerate(sorted(boxes, key=lambda item: (item.r0, item.c0, DIR_ORDER.index(item.direction)))):
        th, tw = box.template.shape
        depth = 0
        painted_any = False
        while True:
            tr, tc = box.tile_origin(depth)
            if tr < 0 or tc < 0 or tr + th > h or tc + tw > w:
                break

            patch = g[tr : tr + th, tc : tc + tw]
            if np.any(patch != BG):
                blocked = True
                break

            for lr in range(th):
                for lc in range(tw):
                    rr = tr + lr
                    cc = tc + lc
                    if best_depth[rr, cc] < 10**9:
                        overlap = True
                    if depth < best_depth[rr, cc] or (depth == best_depth[rr, cc] and order < best_order[rr, cc]):
                        best_depth[rr, cc] = depth
                        best_order[rr, cc] = order
                        out[rr, cc] = int(box.template[lr, lc])
            painted_any = True
            depth += 1

        if not painted_any:
            return None

    return out, overlap, blocked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each box's inner two-color template and tile it outward from the single border gap.
    """
    g = np.asarray(grid, dtype=int)
    boxes = _detect_boxes(g)
    if not boxes:
        return g.copy()
    rendered = _paint_spreads(g, boxes)
    if rendered is None:
        return g.copy()
    out, _, _ = rendered
    return out


PATTERN_MASKS = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int),
)


def _random_template(rng: np.random.Generator) -> np.ndarray:
    mask = PATTERN_MASKS[int(rng.integers(len(PATTERN_MASKS)))].copy()
    mask = np.rot90(mask, int(rng.integers(4)))
    if bool(rng.integers(2)):
        mask = np.fliplr(mask)
    if bool(rng.integers(2)):
        mask = np.flipud(mask)
    color_a, color_b = rng.choice(PALETTE, size=2, replace=False).tolist()
    return np.where(mask == 1, int(color_a), int(color_b)).astype(int)


def _random_box(rng: np.random.Generator, height: int, width: int) -> Box | None:
    direction = str(rng.choice(np.array(DIR_ORDER, dtype=object)))
    template = _random_template(rng)

    if direction == "up":
        r_min, r_max = INNER, height - OUTER
        c_min, c_max = 0, width - OUTER
    elif direction == "down":
        r_min, r_max = 0, height - OUTER - INNER
        c_min, c_max = 0, width - OUTER
    elif direction == "left":
        r_min, r_max = 0, height - OUTER
        c_min, c_max = INNER, width - OUTER
    else:
        r_min, r_max = 0, height - OUTER
        c_min, c_max = 0, width - OUTER - INNER

    if r_min > r_max or c_min > c_max:
        return None
    r0 = int(rng.integers(r_min, r_max + 1))
    c0 = int(rng.integers(c_min, c_max + 1))
    return _build_box(r0, c0, direction, template)


def _build_input(shape: tuple[int, int], boxes: list[Box]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for box in boxes:
        _draw_box(grid, box)
    return grid


def _fit_extra_box(rng: np.random.Generator, shape: tuple[int, int], boxes: list[Box]) -> list[Box]:
    height, width = shape
    for _ in range(80):
        candidate = _random_box(rng, height, width)
        if candidate is None:
            continue
        if any(_expanded_overlap(candidate, other) for other in boxes):
            continue
        return boxes + [candidate]
    return boxes


def _sample_plain(rng: np.random.Generator, shape: tuple[int, int]) -> list[Box] | None:
    height, width = shape
    count = int(rng.integers(2, 4))
    boxes: list[Box] = []
    for _ in range(count):
        placed = False
        for _ in range(120):
            candidate = _random_box(rng, height, width)
            if candidate is None:
                continue
            if any(_expanded_overlap(candidate, other) for other in boxes):
                continue
            boxes.append(candidate)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _sample_blocked(rng: np.random.Generator, shape: tuple[int, int]) -> list[Box] | None:
    height, width = shape
    src_dir = str(rng.choice(np.array(DIR_ORDER, dtype=object)))
    steps = int(rng.integers(1, 3))
    src_template = _random_template(rng)
    blk_template = _random_template(rng)

    if src_dir == "right":
        c_max = width - (OUTER + INNER * steps + OUTER)
        if c_max < 0:
            return None
        r0 = int(rng.integers(0, height - OUTER + 1))
        c0 = int(rng.integers(0, c_max + 1))
        src = _build_box(r0, c0, "right", src_template)
        blk_dir = str(rng.choice(np.array(["up", "down", "right"], dtype=object)))
        blocker = _build_box(r0, c0 + OUTER + INNER * steps, blk_dir, blk_template)
    elif src_dir == "left":
        c_min = OUTER + INNER * steps
        if c_min > width - OUTER:
            return None
        r0 = int(rng.integers(0, height - OUTER + 1))
        c0 = int(rng.integers(c_min, width - OUTER + 1))
        src = _build_box(r0, c0, "left", src_template)
        blk_dir = str(rng.choice(np.array(["up", "down", "left"], dtype=object)))
        blocker = _build_box(r0, c0 - OUTER - INNER * steps, blk_dir, blk_template)
    elif src_dir == "down":
        r_max = height - (OUTER + INNER * steps + OUTER)
        if r_max < 0:
            return None
        r0 = int(rng.integers(0, r_max + 1))
        c0 = int(rng.integers(0, width - OUTER + 1))
        src = _build_box(r0, c0, "down", src_template)
        blk_dir = str(rng.choice(np.array(["down", "left", "right"], dtype=object)))
        blocker = _build_box(r0 + OUTER + INNER * steps, c0, blk_dir, blk_template)
    else:
        r_min = OUTER + INNER * steps
        if r_min > height - OUTER:
            return None
        r0 = int(rng.integers(r_min, height - OUTER + 1))
        c0 = int(rng.integers(0, width - OUTER + 1))
        src = _build_box(r0, c0, "up", src_template)
        blk_dir = str(rng.choice(np.array(["up", "left", "right"], dtype=object)))
        blocker = _build_box(r0 - OUTER - INNER * steps, c0, blk_dir, blk_template)

    if _expanded_overlap(src, blocker):
        return None
    boxes = [src, blocker]
    if bool(rng.integers(2)):
        boxes = _fit_extra_box(rng, shape, boxes)
    return boxes


def _sample_overlap(rng: np.random.Generator, shape: tuple[int, int]) -> list[Box] | None:
    height, width = shape
    h_dir = str(rng.choice(np.array(["right", "left"], dtype=object)))
    v_dir = str(rng.choice(np.array(["up", "down"], dtype=object)))
    h_depth = int(rng.integers(0, 3))
    v_depth = int(rng.integers(0, 3))
    if h_depth == v_depth:
        v_depth = (v_depth + 1) % 3

    tr_low, tr_high = 0, height - INNER
    tc_low, tc_high = 0, width - INNER

    if h_dir == "right":
        tr_low = max(tr_low, 1)
        tr_high = min(tr_high, height - 4)
        tc_low = max(tc_low, OUTER + INNER * h_depth)
    else:
        tr_low = max(tr_low, 1)
        tr_high = min(tr_high, height - 4)
        tc_high = min(tc_high, width - (OUTER + INNER * (h_depth + 1)))

    if v_dir == "down":
        tr_low = max(tr_low, OUTER + INNER * v_depth)
        tc_low = max(tc_low, 1)
        tc_high = min(tc_high, width - 4)
    else:
        tr_high = min(tr_high, height - (OUTER + INNER * (v_depth + 1)))
        tc_low = max(tc_low, 1)
        tc_high = min(tc_high, width - 4)

    if tr_low > tr_high or tc_low > tc_high:
        return None
    tr = int(rng.integers(tr_low, tr_high + 1))
    tc = int(rng.integers(tc_low, tc_high + 1))

    if h_dir == "right":
        h_box = _build_box(tr - 1, tc - OUTER - INNER * h_depth, "right", _random_template(rng))
    else:
        h_box = _build_box(tr - 1, tc + INNER * (h_depth + 1), "left", _random_template(rng))

    if v_dir == "down":
        v_box = _build_box(tr - OUTER - INNER * v_depth, tc - 1, "down", _random_template(rng))
    else:
        v_box = _build_box(tr + INNER * (v_depth + 1), tc - 1, "up", _random_template(rng))

    if _expanded_overlap(h_box, v_box):
        return None
    boxes = [h_box, v_box]
    if bool(rng.integers(2)):
        boxes = _fit_extra_box(rng, shape, boxes)
    return boxes


def _scenario_ok(name: str, overlap: bool, blocked: bool) -> bool:
    if name == "plain":
        return (not overlap) and (not blocked)
    if name == "blocked":
        return blocked and (not overlap)
    return overlap


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    shape = (int(rng.integers(18, 25)), int(rng.integers(18, 25)))
    scenario = str(rng.choice(np.array(["plain", "blocked", "overlap"], dtype=object)))

    for _ in range(120):
        if scenario == "plain":
            boxes = _sample_plain(rng, shape)
        elif scenario == "blocked":
            boxes = _sample_blocked(rng, shape)
        else:
            boxes = _sample_overlap(rng, shape)
        if boxes is None:
            continue

        boxes = sorted(boxes, key=lambda box: (box.r0, box.c0, DIR_ORDER.index(box.direction)))
        if any(_expanded_overlap(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            continue

        inp = _build_input(shape, boxes)
        rendered = _paint_spreads(inp, boxes)
        if rendered is None:
            continue
        expected, overlap, blocked = rendered
        if not _scenario_ok(scenario, overlap, blocked):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_case(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
