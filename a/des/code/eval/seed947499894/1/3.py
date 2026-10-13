from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator bounds.
REGION_COUNT_RANGE = (2, 4)
INTERIOR_H_RANGE = (3, 7)
INTERIOR_W_RANGE = (3, 7)
STRAIGHT_REGION_PROB = 0.35
MAX_PATH_LEN = 14
REGION_ATTEMPTS = 200
GRID_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _infer_bg_wall(g: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, g.reshape(-1)))
    bg = BG if BG in counts else int(counts.most_common(1)[0][0])
    non_bg = {color: count for color, count in counts.items() if int(color) != int(bg)}
    assert non_bg, "expected at least one non-background color"
    dominant_non_bg = max(non_bg, key=non_bg.get)
    wall = WALL if WALL in non_bg and non_bg[WALL] >= non_bg[dominant_non_bg] else int(dominant_non_bg)
    return int(bg), int(wall)


def _shortest_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in _neighbors4(cur[0], cur[1], h, w):
            if not bool(mask[nxt]) or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    assert end in prev, "expected a path between the two markers"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _turn_cells(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    turns: list[tuple[int, int]] = []
    for i in range(1, len(path) - 1):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        if (r1 - r0, c1 - c0) != (r2 - r1, c2 - c1):
            turns.append((int(r1), int(c1)))
    return turns


def _check_simple_path_component(mask: np.ndarray) -> None:
    coords = np.argwhere(mask)
    if coords.shape[0] <= 1:
        return
    deg1 = 0
    for r, c in coords:
        degree = 0
        for nr, nc in _neighbors4(int(r), int(c), mask.shape[0], mask.shape[1]):
            degree += int(mask[nr, nc])
        assert degree <= 2, "maze component must not branch"
        if degree == 1:
            deg1 += 1
    assert deg1 == 2, "maze component must have exactly two endpoints"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each isolated one-cell-wide maze path, fill the path with the start
    color, and mark every turn with the end color.

    The puzzle description names a start and end point but gives no extra
    endpoint glyph, so role assignment is resolved geometrically: within each
    maze component, the row-major earlier colored endpoint is the start.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, wall = _infer_bg_wall(g)
    passable = g != wall
    out = g.copy()

    for comp in _components(passable):
        markers = [(r, c) for r, c in comp if int(g[r, c]) not in (int(bg), int(wall))]
        if not markers:
            continue
        assert len(markers) == 2, "each maze region must contain exactly two markers"

        comp_mask = np.zeros_like(passable, dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True
        _check_simple_path_component(comp_mask)

        markers.sort()
        start, end = markers
        start_color = int(g[start])
        end_color = int(g[end])
        assert start_color != end_color, "start and end colors must differ"

        path = _shortest_path(comp_mask, start, end)
        for r, c in path[1:-1]:
            if int(g[r, c]) == int(bg):
                out[r, c] = int(start_color)
        for r, c in _turn_cells(path):
            out[r, c] = int(end_color)

    return out


def _valid_extension(
    nxt: tuple[int, int],
    prev: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    r, c = nxt
    if not (0 <= r < h and 0 <= c < w) or nxt in occupied:
        return False
    for nb in _neighbors4(r, c, h, w):
        if nb in occupied and nb != prev:
            return False
    return True


def _future_option_count(
    cur: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> int:
    count = 0
    for nb in _neighbors4(cur[0], cur[1], h, w):
        if _valid_extension(nb, cur, occupied, h, w):
            count += 1
    return count


def _dfs_path(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target_len: int,
    h: int,
    w: int,
    need_turn: bool,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        if not need_turn or _turn_cells(path):
            return list(path)
        return None

    cur = path[-1]
    candidates: list[tuple[int, int]] = []
    for nb in _neighbors4(cur[0], cur[1], h, w):
        if not _valid_extension(nb, cur, occupied, h, w):
            continue
        occupied.add(nb)
        if len(path) + 1 == target_len or _future_option_count(nb, occupied, h, w) > 0:
            candidates.append(nb)
        occupied.remove(nb)

    random.shuffle(candidates)
    candidates.sort(
        key=lambda cell: _future_option_count(cell, occupied | {cell}, h, w),
        reverse=True,
    )
    for nb in candidates:
        path.append(nb)
        occupied.add(nb)
        result = _dfs_path(path, occupied, target_len, h, w, need_turn)
        if result is not None:
            return result
        occupied.remove(nb)
        path.pop()
    return None


def _random_induced_path(h: int, w: int, target_len: int, need_turn: bool) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(starts)
    for start in starts:
        result = _dfs_path([start], {start}, target_len, h, w, need_turn)
        if result is not None:
            return result
    return None


def _straight_path(h: int, w: int) -> list[tuple[int, int]] | None:
    orientations: list[str] = []
    if w >= 3:
        orientations.append("h")
    if h >= 3:
        orientations.append("v")
    if not orientations:
        return None
    orient = random.choice(orientations)
    if orient == "h":
        row = random.randrange(h)
        length = random.randint(3, w)
        c0 = random.randint(0, w - length)
        return [(row, c0 + dc) for dc in range(length)]
    col = random.randrange(w)
    length = random.randint(3, h)
    r0 = random.randint(0, h - length)
    return [(r0 + dr, col) for dr in range(length)]


def _build_region(force_turn: bool) -> dict | None:
    for _ in range(REGION_ATTEMPTS):
        ih = random.randint(*INTERIOR_H_RANGE)
        iw = random.randint(*INTERIOR_W_RANGE)

        use_straight = (not force_turn) and random.random() < STRAIGHT_REGION_PROB
        if use_straight:
            path = _straight_path(ih, iw)
        else:
            min_len = 4 if not force_turn else 5
            max_len = min(MAX_PATH_LEN, ih * iw)
            if max_len < min_len:
                continue
            target_len = random.randint(min_len, max_len)
            path = _random_induced_path(ih, iw, target_len, need_turn=force_turn)

        if path is None:
            continue
        if force_turn and not _turn_cells(path):
            continue
        if path[0] > path[-1]:
            path = list(reversed(path))
        return {"ih": ih, "iw": iw, "path": path}
    return None


def _render_region(region: dict, start_color: int, end_color: int) -> tuple[np.ndarray, np.ndarray]:
    ih = int(region["ih"])
    iw = int(region["iw"])
    path = [(int(r), int(c)) for r, c in region["path"]]
    inp = np.full((ih + 2, iw + 2), WALL, dtype=int)
    for r, c in path:
        inp[r + 1, c + 1] = BG

    start = path[0]
    end = path[-1]
    inp[start[0] + 1, start[1] + 1] = int(start_color)
    inp[end[0] + 1, end[1] + 1] = int(end_color)

    out = inp.copy()
    for r, c in path[1:-1]:
        if int(out[r + 1, c + 1]) == BG:
            out[r + 1, c + 1] = int(start_color)
    for r, c in _turn_cells(path):
        out[r + 1, c + 1] = int(end_color)
    return inp, out


def _split_nonnegative(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    if total <= 0:
        return [0] * parts
    return [int(x) for x in np.random.multinomial(int(total), [1.0 / parts] * parts)]


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    base = [1] * parts
    extra = _split_nonnegative(total - parts, parts)
    return [int(b + e) for b, e in zip(base, extra)]


def _layout_regions(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    items = [{"inp": inp, "out": out, "h": inp.shape[0], "w": inp.shape[1]} for inp, out in patches]
    random.shuffle(items)

    row_count = random.randint(1, min(3, len(items)))
    row_sizes = _positive_composition(len(items), row_count)
    rows: list[list[dict]] = []
    idx = 0
    for size in row_sizes:
        rows.append(items[idx : idx + size])
        idx += size

    row_heights = [max(item["h"] for item in row) for row in rows]
    base_vgaps = [random.randint(1, 3) for _ in range(max(0, row_count - 1))]
    content_h = sum(row_heights) + sum(base_vgaps)
    extra_h = random.randint(0, 6)
    v_slots = _split_nonnegative(extra_h, row_count + 1)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    row_specs = []
    usable_w = 0
    for row in rows:
        base_gaps = [random.randint(1, 4) for _ in range(max(0, len(row) - 1))]
        content_w = sum(item["w"] for item in row) + sum(base_gaps)
        usable_w = max(usable_w, content_w)
        row_specs.append({"row": row, "gaps": base_gaps, "content_w": content_w})

    usable_w += random.randint(0, 6)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)

    canvas_h = top_margin + content_h + extra_h + bottom_margin
    canvas_w = left_margin + usable_w + right_margin
    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    out = inp.copy()

    y = top_margin + v_slots[0]
    for row_index, spec in enumerate(row_specs):
        row = spec["row"]
        row_height = row_heights[row_index]
        content_w = int(spec["content_w"])
        extras = _split_nonnegative(usable_w - content_w, len(row) + 1)
        x = left_margin + extras[0]
        for item_index, item in enumerate(row):
            inset_y = random.randint(0, row_height - item["h"])
            rr = y + inset_y
            cc = x
            inp[rr : rr + item["h"], cc : cc + item["w"]] = item["inp"]
            out[rr : rr + item["h"], cc : cc + item["w"]] = item["out"]
            x += item["w"]
            if item_index < len(row) - 1:
                x += spec["gaps"][item_index] + extras[item_index + 1]
        y += row_height
        if row_index < len(row_specs) - 1:
            y += base_vgaps[row_index] + v_slots[row_index + 1]
    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    region_count = random.randint(*REGION_COUNT_RANGE)
    regions: list[dict] = []
    for i in range(region_count):
        region = _build_region(force_turn=(i == 0))
        if region is None:
            return None
        regions.append(region)

    colors = random.sample(COLOR_POOL, 2 * region_count)
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i, region in enumerate(regions):
        inp_patch, out_patch = _render_region(region, colors[2 * i], colors[2 * i + 1])
        patches.append((inp_patch, out_patch))

    return _layout_regions(patches)


def generate_input() -> np.ndarray:
    for _ in range(GRID_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, out = case
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid maze-coloring task")
