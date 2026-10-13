from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
CONTROL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey object row-wise from column-0 controls, then locally erase a strip sized by its color diversity.

    Input structure:
    - The background is black, grey pixels form several disjoint 4-connected placeholder objects, and column 0 holds row control pixels.
    - Every row that contains grey pixels has exactly one colored control pixel at column 0; grey objects never overlap column 0.
    - Each grey object spans multiple rows and columns, and different rows of the same object may have the same or different control colors.

    Rule:
    - For each row, replace every grey pixel in that row by the color of the row's column-0 control pixel.
    - For each original grey 4-connected object, count how many distinct colors it contains after that recoloring; call this value D.
    - Within that object's own bounding box, erase only the object's pixels that lie in its leftmost D local columns; preserve the control pixels and all other cells.

    Key insight:
    - The left-edge deletion amount is not fixed: it is encoded indirectly by the row controls through the object's post-recolor color diversity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    grey_mask = g == GREY
    objects = _components(grey_mask)
    assert objects, "expected at least one grey object"

    out = g.copy()
    for r in range(g.shape[0]):
        row_mask = grey_mask[r]
        if not np.any(row_mask):
            continue
        control = int(g[r, 0])
        assert control not in (BG, GREY), "grey rows must have a colored control pixel in column 0"
        out[r, row_mask] = control

    for coords in objects:
        rr = coords[:, 0]
        cc = coords[:, 1]
        _r0, c0, _r1, _c1 = _bbox(coords)
        diversity = len({int(out[r, c]) for r, c in coords})
        assert diversity >= 1
        erase_mask = cc < c0 + diversity
        out[rr[erase_mask], cc[erase_mask]] = BG

    return out


def _make_blob(height: int, width: int) -> np.ndarray:
    min_row_width = max(2, width - 2)
    for _ in range(80):
        pivot = random.randrange(height)
        intervals: list[tuple[int, int] | None] = [None] * height
        intervals[pivot] = (0, width - 1)

        for r in range(pivot - 1, -1, -1):
            row_width = random.randint(min_row_width, width)
            prev_l, prev_r = intervals[r + 1]
            lo = max(0, prev_l - row_width + 1)
            hi = min(width - row_width, prev_r)
            if lo > hi:
                row_width = width
                lo = hi = 0
            left = random.randint(lo, hi)
            intervals[r] = (left, left + row_width - 1)

        for r in range(pivot + 1, height):
            row_width = random.randint(min_row_width, width)
            prev_l, prev_r = intervals[r - 1]
            lo = max(0, prev_l - row_width + 1)
            hi = min(width - row_width, prev_r)
            if lo > hi:
                row_width = width
                lo = hi = 0
            left = random.randint(lo, hi)
            intervals[r] = (left, left + row_width - 1)

        if height > 1 and not any(
            left > 0 or right < width - 1 for left, right in intervals if left is not None
        ):
            row = random.choice([r for r in range(height) if r != pivot])
            row_width = width - 1
            lo = 0
            hi = width - row_width
            for nr in (row - 1, row + 1):
                if 0 <= nr < height:
                    n_left, n_right = intervals[nr]
                    lo = max(lo, n_left - row_width + 1)
                    hi = min(hi, n_right)
            if lo <= hi:
                left = random.randint(lo, hi)
                intervals[row] = (left, left + row_width - 1)

        mask = np.zeros((height, width), dtype=bool)
        for r, interval in enumerate(intervals):
            left, right = interval
            mask[r, left : right + 1] = True

        if mask[:, 0].any() and mask[:, -1].any() and any(mask[r].sum() < width for r in range(height)):
            return mask

    raise RuntimeError("failed to generate compact blob")


def _row_colors(diversity: int, height: int) -> list[int]:
    palette = random.sample(CONTROL_COLORS, diversity)
    colors = palette[:]
    while len(colors) < height:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _choose_diversities() -> list[int]:
    if random.random() < 0.7:
        values = [1, 2, 3, 4]
    else:
        values = list(random.choice([[1, 2, 4], [1, 3, 4], [1, 2, 3], [2, 3, 4]]))
    random.shuffle(values)
    return values


def _choose_specs() -> list[dict[str, object]]:
    for _ in range(200):
        diversities = _choose_diversities()
        n = len(diversities)
        heights = [random.randint(max(2, d), 5) for d in diversities]
        if not any(h > d for h, d in zip(heights, diversities)):
            idx = random.randrange(n)
            if heights[idx] < 5:
                heights[idx] += 1
            else:
                continue

        shared = random.sample(range(n), 2)
        min_shared_width = max(4, diversities[shared[0]] + 1, diversities[shared[1]] + 1)
        if min_shared_width > 6:
            continue
        shared_width = random.randint(min_shared_width, 6)

        widths: list[int] = []
        for i, d in enumerate(diversities):
            if i in shared:
                widths.append(shared_width)
            else:
                widths.append(random.randint(max(4, d + 1), 6))

        edge_idx = random.randrange(n)
        widths[edge_idx] = max(4, diversities[edge_idx] + 1)

        specs = []
        for d, h, w in zip(diversities, heights, widths):
            specs.append(
                {
                    "D": int(d),
                    "height": int(h),
                    "width": int(w),
                    "mask": _make_blob(int(h), int(w)),
                    "row_colors": _row_colors(int(d), int(h)),
                }
            )

        if _informative_specs(specs):
            return specs
    raise RuntimeError("failed to choose informative object specs")


def _informative_specs(specs: list[dict[str, object]]) -> bool:
    diversities = [int(spec["D"]) for spec in specs]
    heights = [int(spec["height"]) for spec in specs]
    widths = [int(spec["width"]) for spec in specs]

    if len(set(diversities)) < 2:
        return False
    if not any(h > d for h, d in zip(heights, diversities)):
        return False
    if not any(w == max(4, d + 1) for w, d in zip(widths, diversities)):
        return False
    if not any(widths[i] == widths[j] and diversities[i] != diversities[j] for i in range(len(specs)) for j in range(i + 1, len(specs))):
        return False

    all_row_colors = {color for spec in specs for color in spec["row_colors"]}
    if len(all_row_colors) < 3:
        return False

    return True


def _layout_rows(specs: list[dict[str, object]]) -> tuple[int, list[int]]:
    top = random.randint(0, 2)
    gaps = [random.randint(1, 2) for _ in range(len(specs) - 1)]
    starts: list[int] = []
    cursor = top
    for i, spec in enumerate(specs):
        starts.append(cursor)
        cursor += int(spec["height"])
        if i < len(gaps):
            cursor += gaps[i]
    bottom = random.randint(0, 2)
    return cursor + bottom, starts


def _build_input(specs: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    height, row_starts = _layout_rows(specs)
    max_width = max(int(spec["width"]) for spec in specs)
    width_min = max(max_width + 2, 9)
    width_max = min(14, max_width + 7)
    canvas_width = random.randint(width_min, width_max)

    grid = np.full((height, canvas_width), BG, dtype=int)
    placed: list[dict[str, object]] = []

    for spec, row0 in zip(specs, row_starts):
        width = int(spec["width"])
        col0 = random.randint(1, canvas_width - width)
        mask = np.asarray(spec["mask"], dtype=bool)
        for local_r, control in enumerate(spec["row_colors"]):
            grid[row0 + local_r, 0] = int(control)
            cols = np.flatnonzero(mask[local_r])
            grid[row0 + local_r, col0 + cols] = GREY

        placed.append(
            {
                **spec,
                "row0": int(row0),
                "col0": int(col0),
            }
        )

    return grid, placed


def _expected_output(inp: np.ndarray, placed: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in placed:
        row0 = int(spec["row0"])
        col0 = int(spec["col0"])
        mask = np.asarray(spec["mask"], dtype=bool)
        for local_r, control in enumerate(spec["row_colors"]):
            cols = np.flatnonzero(mask[local_r])
            out[row0 + local_r, col0 + cols] = int(control)

        diversity = len(set(int(c) for c in spec["row_colors"]))
        for local_r in range(mask.shape[0]):
            cols = np.flatnonzero(mask[local_r] & (np.arange(mask.shape[1]) < diversity))
            out[row0 + local_r, col0 + cols] = BG
    return out


def _scene_is_valid(inp: np.ndarray, placed: list[dict[str, object]]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not np.any(inp == GREY):
        return False

    grey_rows = {r for r in range(inp.shape[0]) if np.any(inp[r] == GREY)}
    if any(int(inp[r, 0]) in (BG, GREY) for r in grey_rows):
        return False
    if any(int(inp[r, 0]) != BG for r in range(inp.shape[0]) if r not in grey_rows):
        return False

    comps = sorted(_components(inp == GREY), key=lambda coords: tuple(_bbox(coords)))
    if len(comps) != len(placed):
        return False

    specs_sorted = sorted(placed, key=lambda spec: (int(spec["row0"]), int(spec["col0"])))
    saw_repeat_color_object = False
    for coords, spec in zip(comps, specs_sorted):
        r0, c0, r1, c1 = _bbox(coords)
        if (r0, c0, r1, c1) != (
            int(spec["row0"]),
            int(spec["col0"]),
            int(spec["row0"]) + int(spec["height"]) - 1,
            int(spec["col0"]) + int(spec["width"]) - 1,
        ):
            return False
        if int(spec["width"]) < int(spec["D"]) + 1:
            return False
        if len(set(int(c) for c in spec["row_colors"])) != int(spec["D"]):
            return False
        if len(spec["row_colors"]) > len(set(int(c) for c in spec["row_colors"])):
            saw_repeat_color_object = True

    if not saw_repeat_color_object:
        return False

    expected = _expected_output(inp, placed)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(actual, inp):
        return False

    diversity_values = {int(spec["D"]) for spec in placed}
    if len(diversity_values) < 2:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    specs = _choose_specs()
    inp, placed = _build_input(specs)
    if not _scene_is_valid(inp, placed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
