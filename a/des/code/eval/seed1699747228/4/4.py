import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRY_LIMIT = 400
MAIN_H_RANGE = (12, 18)
MAIN_W_RANGE = (14, 22)
LEGEND_EXTRA_WIDTH = (0, 2)
OBJECT_COUNT_CHOICES = (4, 5)
OBJECT_SIZE_CHOICES = (1, 3, 5)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(pts, dtype=int))
    return parts


def _monochrome_components(grid: np.ndarray, allowed: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros_like(allowed, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not allowed[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not allowed[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            parts.append(np.array(pts, dtype=int))
    return parts


def _read_legend_rows(box: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for rr in range(1, box.shape[0] - 1):
        row = box[rr, 1:-1]
        nz = np.flatnonzero(row != BLACK)
        if nz.size == 0:
            continue
        key_idx = int(nz[0])
        key_color = int(row[key_idx])
        if key_color in (BLACK, GREY, MAROON):
            continue
        count = int(np.sum(row[key_idx + 1 :] == MAROON))
        if count > 0:
            mapping[key_color] = count
    return mapping


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    best_key = None
    best_bbox = None
    for coords in _components_mask(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(box[0, :] == GREY)
            and np.all(box[-1, :] == GREY)
            and np.all(box[:, 0] == GREY)
            and np.all(box[:, -1] == GREY)
        ):
            continue
        interior = box[1:-1, 1:-1]
        if interior.size == 0 or not (interior != GREY).any():
            continue
        mapping = _read_legend_rows(box)
        if not mapping:
            continue
        key = (r1 + c1, coords.shape[0], r1 - r0, c1 - c0)
        if best_key is None or key > best_key:
            best_key = key
            best_bbox = (r0, c0, r1, c1)
    if best_bbox is None:
        raise AssertionError("legend bbox not found")
    return best_bbox


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    lr0, lc0, lr1, lc1 = _find_legend_bbox(g)
    legend = _read_legend_rows(g[lr0 : lr1 + 1, lc0 : lc1 + 1])
    if not legend:
        raise AssertionError("legend mapping must be non-empty")

    allowed = g != BLACK
    allowed[0, :] = False
    allowed[:, 0] = False
    allowed[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    objects = []
    for coords in _monochrome_components(g, allowed):
        r0, c0, r1, c1 = _bbox(coords)
        color = int(g[r0, c0])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if coords.shape[0] != height * width:
            raise AssertionError("objects must be solid rectangles")
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        row_color = int(g[center_r, 0])
        col_color = int(g[0, center_c])
        if row_color not in legend or col_color not in legend:
            raise AssertionError("ruler colors must exist in legend")
        score = max(int(legend[row_color]), int(legend[col_color]))
        objects.append((score, center_r, center_c, color, height, width))

    if not objects:
        raise AssertionError("at least one object is required")

    objects.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    out_h = max(height * score for score, _r, _c, _col, height, _width in objects)
    out_w = sum(width * score for score, _r, _c, _col, _height, width in objects) + len(objects) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    cursor = 0
    for score, _r, _c, color, height, width in objects:
        scaled_h = height * score
        scaled_w = width * score
        out[:scaled_h, cursor : cursor + scaled_w] = color
        cursor += scaled_w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the bottom-right color legend, score each rectangle by the larger of its
    row/column ruler values, then sort and scale the rectangles by that score.
    """
    return _solve(grid)


def _choose_branch_assignments(rng: random.Random, n_objects: int) -> list[tuple[int, str]]:
    scores = rng.sample((1, 2, 3, 4, 5), n_objects)

    row_score = rng.choice([s for s in scores if s > 1])
    remaining = [s for s in scores if s != row_score]
    col_score = rng.choice([s for s in remaining if s > 1])
    remaining.remove(col_score)
    equal_score = rng.choice(remaining)
    remaining.remove(equal_score)

    assignments = [(row_score, "row"), (col_score, "col"), (equal_score, "equal")]
    for score in remaining:
        modes = ["equal"] if score == 1 else ["row", "col", "equal"]
        assignments.append((score, rng.choice(modes)))
    rng.shuffle(assignments)
    return assignments


def _score_pair(rng: random.Random, score: int, mode: str) -> tuple[int, int]:
    if mode == "equal":
        return score, score
    if mode == "row":
        return score, rng.randint(1, score - 1)
    if mode == "col":
        return rng.randint(1, score - 1), score
    raise ValueError(mode)


def _legend_block(rng: random.Random, color_to_value: dict[int, int]) -> np.ndarray:
    inner_h = len(color_to_value)
    inner_w = 1 + 5 + rng.randint(*LEGEND_EXTRA_WIDTH)
    block = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    block[0, :] = GREY
    block[-1, :] = GREY
    block[:, 0] = GREY
    block[:, -1] = GREY

    rows = list(color_to_value.items())
    rng.shuffle(rows)
    for rr, (color, value) in enumerate(rows, start=1):
        block[rr, 1] = color
        block[rr, 2 : 2 + value] = MAROON
    return block


def _place_objects(
    rng: random.Random,
    main_h: int,
    main_w: int,
    assignments: list[tuple[int, str]],
) -> list[dict[str, int]]:
    occupied = np.zeros((main_h + 1, main_w + 1), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    objects = []

    for score, mode in assignments:
        row_value, col_value = _score_pair(rng, score, mode)
        size_candidates = [(h, w) for h in OBJECT_SIZE_CHOICES for w in OBJECT_SIZE_CHOICES]
        rng.shuffle(size_candidates)

        placed = None
        for height, width in size_candidates:
            dh = height // 2
            dw = width // 2
            row_candidates = list(range(2 + dh, main_h - dh + 1))
            col_candidates = list(range(2 + dw, main_w - dw + 1))
            rng.shuffle(row_candidates)
            rng.shuffle(col_candidates)
            for center_r in row_candidates:
                if center_r in used_rows:
                    continue
                for center_c in col_candidates:
                    if center_c in used_cols:
                        continue
                    r0 = center_r - dh
                    r1 = center_r + dh
                    c0 = center_c - dw
                    c1 = center_c + dw
                    gr0 = max(1, r0 - 1)
                    gr1 = min(main_h, r1 + 1)
                    gc0 = max(1, c0 - 1)
                    gc1 = min(main_w, c1 + 1)
                    if occupied[gr0 : gr1 + 1, gc0 : gc1 + 1].any():
                        continue
                    placed = {
                        "score": score,
                        "mode": 0 if mode == "row" else 1 if mode == "col" else 2,
                        "row_value": row_value,
                        "col_value": col_value,
                        "center_r": center_r,
                        "center_c": center_c,
                        "height": height,
                        "width": width,
                        "r0": r0,
                        "r1": r1,
                        "c0": c0,
                        "c1": c1,
                    }
                    occupied[gr0 : gr1 + 1, gc0 : gc1 + 1] = True
                    used_rows.add(center_r)
                    used_cols.add(center_c)
                    break
                if placed is not None:
                    break
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError("failed to place objects")
        objects.append(placed)
    return objects


def _gen_attempt(rng: random.Random) -> np.ndarray:
    value_colors = rng.sample(COLOR_POOL, 5)
    color_to_value = {color: value for value, color in enumerate(value_colors, start=1)}
    value_to_color = {value: color for color, value in color_to_value.items()}

    legend = _legend_block(rng, color_to_value)
    legend_h, legend_w = map(int, legend.shape)

    main_h = rng.randint(*MAIN_H_RANGE)
    main_w = rng.randint(*MAIN_W_RANGE)
    total_h = main_h + legend_h + 2
    total_w = main_w + legend_w + 2
    grid = np.zeros((total_h, total_w), dtype=int)

    legend_r0 = total_h - legend_h
    legend_c0 = total_w - legend_w
    grid[legend_r0:, legend_c0:] = legend

    row_colors = [0] * total_h
    col_colors = [0] * total_w
    for rr in range(1, main_h + 1):
        row_colors[rr] = rng.choice(value_colors)
        grid[rr, 0] = row_colors[rr]
    for cc in range(1, main_w + 1):
        col_colors[cc] = rng.choice(value_colors)
        grid[0, cc] = col_colors[cc]

    assignments = _choose_branch_assignments(rng, rng.choice(OBJECT_COUNT_CHOICES))
    objects = _place_objects(rng, main_h, main_w, assignments)

    obj_colors = rng.sample(COLOR_POOL, len(objects))
    rng.shuffle(obj_colors)
    for spec, color in zip(objects, obj_colors):
        spec["color"] = color
        row_colors[spec["center_r"]] = value_to_color[spec["row_value"]]
        col_colors[spec["center_c"]] = value_to_color[spec["col_value"]]
        grid[spec["center_r"], 0] = row_colors[spec["center_r"]]
        grid[0, spec["center_c"]] = col_colors[spec["center_c"]]
        grid[spec["r0"] : spec["r1"] + 1, spec["c0"] : spec["c1"] + 1] = color

    scores = [spec["score"] for spec in objects]
    if len(scores) != len(set(scores)):
        raise AssertionError("scores must stay unique")

    out = _solve(grid)
    if out.size == 0 or out.shape[0] < 1 or out.shape[1] < len(objects):
        raise AssertionError("unexpected output shape")
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
