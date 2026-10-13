from __future__ import annotations

import random
from collections import deque

import numpy as np


RED = 2
YELLOW = 4
ALL_COLORS = tuple(range(10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _current_fold_lines(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    lines: list[dict[str, int | str]] = []
    for r in range(h):
        if np.all(g[r, :] == RED):
            lines.append({"orientation": "H", "index": int(r)})
    for c in range(w):
        if np.all(g[:, c] == RED):
            lines.append({"orientation": "V", "index": int(c)})
    return lines


def _pattern_count(region: np.ndarray, bg: int) -> int:
    arr = np.asarray(region, dtype=int)
    return int(np.count_nonzero((arr != bg) & (arr != RED)))


def _small_side_area(shape: tuple[int, int], orientation: str, index: int) -> int:
    h, w = shape
    if orientation == "H":
        return int(min(index, h - index - 1) * w)
    return int(h * min(index, w - index - 1))


def _folded_side_for_line(grid: np.ndarray, bg: int, orientation: str, index: int) -> str:
    g = np.asarray(grid, dtype=int)
    if orientation == "H":
        top = g[:index, :]
        bottom = g[index + 1 :, :]
        top_count = _pattern_count(top, bg)
        bottom_count = _pattern_count(bottom, bg)
        if top_count < bottom_count:
            return "top"
        if bottom_count < top_count:
            return "bottom"
        if top.shape[0] <= bottom.shape[0]:
            return "top"
        return "bottom"

    left = g[:, :index]
    right = g[:, index + 1 :]
    left_count = _pattern_count(left, bg)
    right_count = _pattern_count(right, bg)
    if left_count < right_count:
        return "left"
    if right_count < left_count:
        return "right"
    if left.shape[1] <= right.shape[1]:
        return "left"
    return "right"


def _merge_pixel(keep_value: int, folded_value: int, bg: int) -> int:
    if folded_value in {bg, RED}:
        return int(keep_value)
    if keep_value == bg:
        return int(folded_value)
    if keep_value == RED:
        return RED
    return YELLOW


def _fold_once(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    lines = _current_fold_lines(g)
    assert lines, "expected at least one fold line"

    lines.sort(
        key=lambda item: (
            _small_side_area(g.shape, str(item["orientation"]), int(item["index"])),
            0 if item["orientation"] == "H" else 1,
            int(item["index"]),
        )
    )
    line = lines[0]
    orientation = str(line["orientation"])
    index = int(line["index"])
    folded_side = _folded_side_for_line(g, bg, orientation, index)

    if orientation == "H":
        if folded_side == "top":
            keep = g[index + 1 :, :].copy()
            small = g[:index, :]
            keep_h = keep.shape[0]
            small_h = small.shape[0]
            for sr in range(small_h):
                dist = small_h - sr
                tr = dist - 1
                if not (0 <= tr < keep_h):
                    continue
                for c in range(g.shape[1]):
                    keep[tr, c] = _merge_pixel(int(keep[tr, c]), int(small[sr, c]), bg)
            return keep

        keep = g[:index, :].copy()
        small = g[index + 1 :, :]
        keep_h = keep.shape[0]
        for sr in range(small.shape[0]):
            dist = sr + 1
            tr = keep_h - dist
            if not (0 <= tr < keep_h):
                continue
            for c in range(g.shape[1]):
                keep[tr, c] = _merge_pixel(int(keep[tr, c]), int(small[sr, c]), bg)
        return keep

    if folded_side == "left":
        keep = g[:, index + 1 :].copy()
        small = g[:, :index]
        keep_w = keep.shape[1]
        small_w = small.shape[1]
        for sc in range(small_w):
            dist = small_w - sc
            tc = dist - 1
            if not (0 <= tc < keep_w):
                continue
            for r in range(g.shape[0]):
                keep[r, tc] = _merge_pixel(int(keep[r, tc]), int(small[r, sc]), bg)
        return keep

    keep = g[:, :index].copy()
    small = g[:, index + 1 :]
    keep_w = keep.shape[1]
    for sc in range(small.shape[1]):
        dist = sc + 1
        tc = keep_w - dist
        if not (0 <= tc < keep_w):
            continue
        for r in range(g.shape[0]):
            keep[r, tc] = _merge_pixel(int(keep[r, tc]), int(small[r, sc]), bg)
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat full red rows/columns as fold lines, fold the side with fewer pattern
    pixels onto the other side, paint overlaps yellow, and crop away the folded
    side plus the line itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    assert bg not in {RED, YELLOW}, "background must differ from the instruction colors"

    out = g.copy()
    steps = 0
    while True:
        lines = _current_fold_lines(out)
        if not lines:
            break
        assert steps < 2, "puzzle contains at most two fold lines"
        out = _fold_once(out, bg)
        steps += 1
    return out


# --------------- generator ---------------

GEN_RETRIES = 500
BASE_RETRIES = 120
COMPONENT_RETRIES = 80

BASE_SIDE_RANGE = (4, 9)
BASE_OTHER_RANGE = (5, 11)
BASE_COMPONENT_SIZE_RANGE = (2, 5)
BASE_COMPONENT_COUNT_RANGE = range(2, 6)
WRAP_SMALL_RANGE = (1, 4)
INNER_SMALL_RANGE = (2, 4)
EMPTY_SIDE_PROB = 0.15
CONFLICT_PROB = 0.6
TWO_LINE_PROB = 0.5


def _palette(bg: int) -> list[int]:
    return [color for color in ALL_COLORS if color not in {bg, RED, YELLOW}]


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _mark_with_buffer(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in ((0, 0),) + DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _grow_component(mask: np.ndarray, blocked: np.ndarray, size: int) -> np.ndarray | None:
    available = np.asarray(mask, dtype=bool) & ~np.asarray(blocked, dtype=bool)
    regions = [part for part in _mask_components(available) if len(part) >= size]
    if not regions:
        return None

    region = random.choice(regions)
    seed = tuple(int(x) for x in region[random.randrange(len(region))])
    selected = {seed}
    frontier: set[tuple[int, int]] = set()
    h, w = available.shape

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and available[nr, nc] and (nr, nc) not in selected:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(selected) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        selected.add(cell)
        add_frontier(*cell)

    return np.array(sorted(selected), dtype=int)


def _partition(total: int, count: int, lo: int, hi: int) -> list[int]:
    sizes: list[int] = []
    remaining = total
    for idx in range(count):
        left = count - idx - 1
        cur_lo = max(lo, remaining - left * hi)
        cur_hi = min(hi, remaining - left * lo)
        sizes.append(random.randint(cur_lo, cur_hi))
        remaining -= sizes[-1]
    random.shuffle(sizes)
    return sizes


def _split_budget(total: int, max_parts: int = 3, lo: int = 1, hi: int = 4) -> list[int] | None:
    choices = [parts for parts in range(1, max_parts + 1) if parts * lo <= total <= parts * hi]
    if not choices:
        return None
    return _partition(total, random.choice(choices), lo, hi)


def _make_base_scene(height: int, width: int, bg: int) -> np.ndarray | None:
    palette = _palette(bg)
    for _ in range(BASE_RETRIES):
        grid = np.full((height, width), bg, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)

        area = height * width
        max_total = min(16, max(6, area // 2))
        min_total = min(max_total, max(5, area // 4 + 2))
        total = random.randint(min_total, max_total)
        feasible = [
            count
            for count in BASE_COMPONENT_COUNT_RANGE
            if count * BASE_COMPONENT_SIZE_RANGE[0] <= total <= count * BASE_COMPONENT_SIZE_RANGE[1]
        ]
        if not feasible:
            continue
        sizes = _partition(total, random.choice(feasible), *BASE_COMPONENT_SIZE_RANGE)

        ok = True
        for size in sizes:
            coords = _grow_component(np.ones((height, width), dtype=bool), blocked, size)
            if coords is None:
                ok = False
                break
            color = random.choice(palette)
            grid[coords[:, 0], coords[:, 1]] = color
            _mark_with_buffer(blocked, coords)

        if not ok:
            continue
        if _pattern_count(grid, bg) < 5:
            continue
        return grid
    return None


def _embed_scene(
    scene: np.ndarray,
    bg: int,
    orientation: str,
    keep_side: str,
    small_size: int,
) -> tuple[np.ndarray, int, int]:
    cur = np.asarray(scene, dtype=int)
    h, w = cur.shape
    if orientation == "H":
        out = np.full((h + 1 + small_size, w), bg, dtype=int)
        if keep_side == "top":
            keep_r0 = 0
            line_r = h
        else:
            keep_r0 = small_size + 1
            line_r = small_size
        out[keep_r0 : keep_r0 + h, :] = cur
        out[line_r, :] = RED
        return out, keep_r0, 0

    out = np.full((h, w + 1 + small_size), bg, dtype=int)
    if keep_side == "left":
        keep_c0 = 0
        line_c = w
    else:
        keep_c0 = small_size + 1
        line_c = small_size
    out[:, keep_c0 : keep_c0 + w] = cur
    out[:, line_c] = RED
    return out, 0, keep_c0


def _extend_inner_lines(
    out: np.ndarray,
    scene: np.ndarray,
    keep_r0: int,
    keep_c0: int,
    orientation: str,
) -> None:
    inner_lines = _current_fold_lines(scene)
    assert len(inner_lines) <= 1, "generator only nests one inner fold"
    if not inner_lines:
        return
    inner = inner_lines[0]
    if inner["orientation"] == orientation:
        return
    if inner["orientation"] == "H":
        out[keep_r0 + int(inner["index"]), :] = RED
    else:
        out[:, keep_c0 + int(inner["index"])] = RED


def _target_strip(scene: np.ndarray, bg: int, orientation: str, keep_side: str, small_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    cur = np.asarray(scene, dtype=int)
    h, w = cur.shape
    if orientation == "H":
        if keep_side == "top":
            return cur[h - small_size :, :], (h - small_size, 0)
        return cur[:small_size, :], (0, 0)
    if keep_side == "left":
        return cur[:, w - small_size :], (0, w - small_size)
    return cur[:, :small_size], (0, 0)


def _mirror_target(
    target_r: int,
    target_c: int,
    keep_r0: int,
    keep_c0: int,
    scene_shape: tuple[int, int],
    orientation: str,
    keep_side: str,
    small_size: int,
) -> tuple[int, int]:
    h, w = scene_shape
    abs_r = keep_r0 + target_r
    abs_c = keep_c0 + target_c
    if orientation == "H":
        line_r = h if keep_side == "top" else small_size
        return 2 * line_r - abs_r, abs_c
    line_c = w if keep_side == "left" else small_size
    return abs_r, 2 * line_c - abs_c


def _pick_component_set(mask: np.ndarray, blocked: np.ndarray, budget: int) -> list[np.ndarray] | None:
    sizes = _split_budget(budget)
    if sizes is None:
        return None
    parts: list[np.ndarray] = []
    local_blocked = blocked.copy()
    for size in sizes:
        coords = _grow_component(mask, local_blocked, size)
        if coords is None:
            return None
        parts.append(coords)
        _mark_with_buffer(local_blocked, coords)
    blocked[:, :] = local_blocked
    return parts


def _paint_folded_side(
    out: np.ndarray,
    scene: np.ndarray,
    bg: int,
    keep_r0: int,
    keep_c0: int,
    orientation: str,
    keep_side: str,
    small_size: int,
    allow_empty: bool,
) -> bool:
    large_count = _pattern_count(scene, bg)
    if large_count < 3:
        return False

    strip, (strip_r0, strip_c0) = _target_strip(scene, bg, orientation, keep_side, small_size)
    empty_mask = strip == bg
    occupied_mask = (strip != bg) & (strip != RED)
    if not np.any(empty_mask | occupied_mask):
        return False

    max_small = min(6, large_count - 1, int(np.count_nonzero(empty_mask | occupied_mask)))
    if max_small < 0:
        return False
    if allow_empty and random.random() < EMPTY_SIDE_PROB:
        total_small = 0
    else:
        if max_small < 1:
            return False
        total_small = random.randint(1, max_small)

    if total_small == 0:
        return True

    occupied_count = int(np.count_nonzero(occupied_mask))
    empty_count = int(np.count_nonzero(empty_mask))
    conflict_pixels = 0
    if occupied_count > 0 and random.random() < CONFLICT_PROB:
        conflict_pixels = random.randint(1, min(total_small, occupied_count))
    fill_pixels = total_small - conflict_pixels
    if fill_pixels > empty_count:
        spill = fill_pixels - empty_count
        fill_pixels -= spill
        conflict_pixels += spill
    if conflict_pixels > occupied_count:
        spill = conflict_pixels - occupied_count
        conflict_pixels -= spill
        fill_pixels += spill
    if fill_pixels < 0 or conflict_pixels < 0 or fill_pixels + conflict_pixels != total_small:
        return False

    blocked = np.zeros(strip.shape, dtype=bool)
    palette = _palette(bg)

    for mode, budget, mask in (
        ("fill", fill_pixels, empty_mask),
        ("conflict", conflict_pixels, occupied_mask),
    ):
        if budget == 0:
            continue
        parts = _pick_component_set(mask, blocked, budget)
        if parts is None:
            return False
        for part in parts:
            target_rows = part[:, 0] + strip_r0
            target_cols = part[:, 1] + strip_c0
            if mode == "conflict":
                under = {int(scene[r, c]) for r, c in zip(target_rows.tolist(), target_cols.tolist(), strict=False)}
                color_choices = [color for color in palette if color not in under]
                color = random.choice(color_choices or palette)
            else:
                color = random.choice(palette)
            for tr, tc in zip(target_rows.tolist(), target_cols.tolist(), strict=False):
                mr, mc = _mirror_target(
                    int(tr),
                    int(tc),
                    keep_r0,
                    keep_c0,
                    scene.shape,
                    orientation,
                    keep_side,
                    small_size,
                )
                if not (0 <= mr < out.shape[0] and 0 <= mc < out.shape[1]):
                    return False
                if out[mr, mc] != bg:
                    return False
                out[mr, mc] = color
    return True


def _wrap_with_fold(
    scene: np.ndarray,
    bg: int,
    orientation: str,
    keep_side: str,
    small_size: int,
    allow_empty: bool,
) -> np.ndarray | None:
    for _ in range(COMPONENT_RETRIES):
        out, keep_r0, keep_c0 = _embed_scene(scene, bg, orientation, keep_side, small_size)
        _extend_inner_lines(out, scene, keep_r0, keep_c0, orientation)
        if not _paint_folded_side(out, scene, bg, keep_r0, keep_c0, orientation, keep_side, small_size, allow_empty):
            continue
        if _get_bg(out) != bg:
            continue
        if np.count_nonzero(out == YELLOW) != 0:
            continue
        return out
    return None


def _generate_single_fold(bg: int, small_range: tuple[int, int], allow_empty: bool) -> np.ndarray | None:
    orientation = random.choice(("H", "V"))
    if orientation == "H":
        keep_h = random.randint(*BASE_SIDE_RANGE)
        keep_w = random.randint(*BASE_OTHER_RANGE)
        max_small = min(small_range[1], keep_h - 1)
        if max_small < small_range[0]:
            return None
        small_size = random.randint(small_range[0], max_small)
        keep_side = random.choice(("top", "bottom"))
    else:
        keep_h = random.randint(*BASE_OTHER_RANGE)
        keep_w = random.randint(*BASE_SIDE_RANGE)
        max_small = min(small_range[1], keep_w - 1)
        if max_small < small_range[0]:
            return None
        small_size = random.randint(small_range[0], max_small)
        keep_side = random.choice(("left", "right"))

    scene = _make_base_scene(keep_h, keep_w, bg)
    if scene is None:
        return None
    wrapped = _wrap_with_fold(scene, bg, orientation, keep_side, small_size, allow_empty)
    if wrapped is None:
        return None
    if len(_current_fold_lines(wrapped)) != 1:
        return None
    return wrapped


def _generate_two_folds(bg: int) -> np.ndarray | None:
    inner = _generate_single_fold(bg, INNER_SMALL_RANGE, allow_empty=False)
    if inner is None:
        return None

    inner_line = _current_fold_lines(inner)
    if len(inner_line) != 1:
        return None
    inner_orientation = str(inner_line[0]["orientation"])
    outer_orientation = "V" if inner_orientation == "H" else "H"

    if outer_orientation == "H":
        max_small = min(2, inner.shape[0] - 2)
        if max_small < 1:
            return None
        small_choices = list(range(1, max_small + 1))
        keep_side_choices = ("top", "bottom")
    else:
        max_small = min(2, inner.shape[1] - 2)
        if max_small < 1:
            return None
        small_choices = list(range(1, max_small + 1))
        keep_side_choices = ("left", "right")

    random.shuffle(small_choices)
    keep_sides = list(keep_side_choices)
    random.shuffle(keep_sides)
    for small_size in small_choices:
        for keep_side in keep_sides:
            wrapped = _wrap_with_fold(inner, bg, outer_orientation, keep_side, small_size, allow_empty=False)
            if wrapped is None:
                continue
            lines = _current_fold_lines(wrapped)
            if len(lines) != 2:
                continue
            sorted_lines = sorted(
                lines,
                key=lambda item: (
                    _small_side_area(wrapped.shape, str(item["orientation"]), int(item["index"])),
                    0 if item["orientation"] == "H" else 1,
                    int(item["index"]),
                ),
            )
            if str(sorted_lines[0]["orientation"]) != outer_orientation:
                continue
            intermediate = _fold_once(wrapped, bg)
            if len(_current_fold_lines(intermediate)) != 1:
                continue
            return wrapped
    return None


def generate_input() -> np.ndarray:
    bg_choices = [color for color in ALL_COLORS if color not in {RED, YELLOW}]
    for _ in range(GEN_RETRIES):
        bg = random.choice(bg_choices)
        if random.random() < TWO_LINE_PROB:
            grid = _generate_two_folds(bg)
        else:
            grid = _generate_single_fold(bg, WRAP_SMALL_RANGE, allow_empty=True)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if not isinstance(out, np.ndarray):
            continue
        if np.count_nonzero(grid == YELLOW) != 0:
            continue
        if _get_bg(grid) != bg:
            continue
        return grid
    raise RuntimeError("failed to generate a valid fold-overlay input")
