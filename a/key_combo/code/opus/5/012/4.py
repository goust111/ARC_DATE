from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, paste


SEPARATOR_COLORS = (2, 1, 3)
CYCLE_MAP = {2: 1, 1: 3, 3: 2}
WIDE_PATCHES = {
    2: [
        np.array([[5, 5]], dtype=int),
        np.array([[5, 0], [5, 5]], dtype=int),
        np.array([[0, 5], [5, 5]], dtype=int),
        np.array([[5, 5], [5, 5]], dtype=int),
        np.array([[5, 5], [5, 0]], dtype=int),
        np.array([[5, 5], [0, 5]], dtype=int),
        np.array([[5, 0], [5, 0], [5, 5]], dtype=int),
        np.array([[0, 5], [0, 5], [5, 5]], dtype=int),
    ],
    3: [
        np.array([[5, 5, 5]], dtype=int),
        np.array([[5, 5, 5], [0, 5, 0]], dtype=int),
        np.array([[5, 0, 0], [5, 5, 5]], dtype=int),
        np.array([[0, 0, 5], [5, 5, 5]], dtype=int),
        np.array([[5, 5, 0], [0, 5, 5]], dtype=int),
        np.array([[0, 5, 5], [5, 5, 0]], dtype=int),
    ],
}


def _separator_row(grid: np.ndarray, bg: int) -> int:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row != bg) and len(set(map(int, row))) > 1:
            rows.append(r)
    assert len(rows) == 1, "expected exactly one full non-background multicolor separator row"
    sep = rows[0]
    assert set(map(int, np.unique(grid[sep]))) == set(SEPARATOR_COLORS)
    return int(sep)


def _sprite_patch(comp: dict, color: int = 5) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.asarray(comp["coords"], dtype=int)
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch, box


def _sorted_gray_components(grid: np.ndarray, sep: int) -> list[dict]:
    comps = find_components(grid, bg=get_bg(grid), colors=[5], connectivity=4, monochromatic=True)
    assert 3 <= len(comps) <= 5, "expected 3-5 gray sprites"
    for comp in comps:
        assert np.all(comp["coords"][:, 0] < sep), "sprites must start above separator"
    return comps


def _drop_solution(
    grid: np.ndarray,
    *,
    treat_all_as_narrow: bool = False,
    treat_all_as_wide: bool = False,
    apply_cycle: bool = True,
    recolor_narrow: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep = _separator_row(g, bg)
    sep_before = g[sep].copy()
    out = g.copy()
    out[g == 5] = bg

    comps = _sorted_gray_components(g, sep)
    narrow = []
    wide = []
    for comp in comps:
        patch, box = _sprite_patch(comp)
        r0, c0, r1, c1 = box
        width = int(len(np.unique(comp["coords"][:, 1])))
        if treat_all_as_narrow:
            width = 1
        if treat_all_as_wide:
            width = 2
        entry = {"comp": comp, "patch": patch, "box": box, "width": width}
        if width == 1:
            narrow.append(entry)
        else:
            wide.append(entry)

    narrow.sort(key=lambda item: item["box"][1])
    wide.sort(key=lambda item: (item["box"][0], item["box"][1]))

    for item in narrow:
        patch = item["patch"].copy()
        _r0, c0, _r1, _c1 = item["box"]
        if recolor_narrow:
            patch[patch == 5] = int(sep_before[c0])
        paste(out, patch, sep + 1, c0, transparent=0)

    for item in wide:
        patch = item["patch"]
        _r0, c0, _r1, c1 = item["box"]
        target_r = sep - patch.shape[0]
        paste(out, patch, target_r, c0, transparent=0)
        if apply_cycle:
            for c in range(c0, c1 + 1):
                out[sep, c] = CYCLE_MAP[int(out[sep, c])]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop gray sprites around a tricolor separator row, recoloring narrow drops and cycling separator colors under wide landings.

    Input structure:
    - The background is black.
    - Exactly one row is fully filled with the three separator colors red, blue, and green, with no black cells in that row.
    - Above that row sit 3-5 disjoint gray 4-connected sprites; each sprite is either one column wide or at least two columns wide.
    - Below the separator row is empty background.

    Rule:
    - Identify the unique full multicolor separator row.
    - Classify each gray sprite by the number of distinct columns it occupies.
    - Narrow sprites (one occupied column) are removed, dropped so their top row starts immediately below the separator, and recolored to the separator color in that column using the separator's original pre-migration state.
    - Wide sprites (two or more occupied columns) are removed, dropped so their bottom row touches the row just above the separator, and remain gray.
    - After each wide sprite settles, cycle the separator colors only in the columns covered by that sprite's horizontal span using red->blue, blue->green, green->red.
    - Process all narrow sprites from left to right first, then all wide sprites from top to bottom.

    Key insight:
    - Width decides whether a sprite passes through the separator or docks on top of it, and only the wide dockings mutate the separator row itself.
    """
    return _drop_solution(grid)


def _rand_comp(n: int, total: int) -> list[int]:
    cuts = sorted(random.sample(range(total + n - 1), n - 1))
    parts = []
    prev = -1
    for cut in cuts + [total + n - 1]:
        parts.append(cut - prev - 1)
        prev = cut
    return parts


def _choose_types(n_sprites: int) -> list[str]:
    n_narrow = random.randint(1, n_sprites - 1)
    if n_sprites >= 4 and random.random() < 0.6:
        n_narrow = random.choice([2, n_sprites - 2])
    types = ["narrow"] * n_narrow + ["wide"] * (n_sprites - n_narrow)
    random.shuffle(types)
    return types


def _choose_widths(types: list[str], width: int) -> list[int] | None:
    widths = [1 if kind == "narrow" else 2 for kind in types]
    wide_indices = [i for i, kind in enumerate(types) if kind == "wide"]
    extra_capacity = width - (sum(widths) + len(types) - 1)
    if extra_capacity < 0:
        return None
    candidates = wide_indices[:]
    random.shuffle(candidates)
    for idx in candidates:
        if extra_capacity <= 0:
            break
        if random.random() < 0.45:
            widths[idx] += 1
            extra_capacity -= 1
    return widths


def _choose_intervals(types: list[str], width: int) -> list[tuple[int, int]] | None:
    widths = _choose_widths(types, width)
    if widths is None:
        return None
    extra = width - (sum(widths) + len(types) - 1)
    if extra < 0:
        return None
    slack = _rand_comp(len(types) + 1, extra)
    intervals = []
    c = slack[0]
    for i, sprite_w in enumerate(widths):
        intervals.append((c, c + sprite_w - 1))
        c += sprite_w
        if i < len(types) - 1:
            c += 1 + slack[i + 1]
    return intervals


def _pick_separator(width: int, intervals: list[tuple[int, int]], types: list[str]) -> np.ndarray | None:
    wide_intervals = [interval for interval, kind in zip(intervals, types) if kind == "wide"]
    narrow_cols = [interval[0] for interval, kind in zip(intervals, types) if kind == "narrow"]
    candidates = []
    for b1 in range(2, width - 3):
        for b2 in range(b1 + 2, width - 1):
            segments = [(0, b1 - 1), (b1, b2 - 1), (b2, width - 1)]
            for order in random.sample(
                [
                    (2, 1, 3),
                    (2, 3, 1),
                    (1, 2, 3),
                    (1, 3, 2),
                    (3, 2, 1),
                    (3, 1, 2),
                ],
                6,
            ):
                row = np.zeros(width, dtype=int)
                for color, (c0, c1) in zip(order, segments):
                    row[c0 : c1 + 1] = color
                narrow_colors = {int(row[c]) for c in narrow_cols}
                wide_color_sets = []
                for c0, c1 in wide_intervals:
                    wide_color_sets.append(set(map(int, row[c0 : c1 + 1])))
                if len(narrow_colors) < min(2, len(narrow_cols)):
                    continue
                if not wide_color_sets:
                    continue
                if not any(len(color_set) >= 2 for color_set in wide_color_sets):
                    continue
                if len(set().union(*wide_color_sets)) < 2:
                    continue
                candidates.append(row)
    if not candidates:
        return None
    return random.choice(candidates)


def _narrow_patch() -> np.ndarray:
    return np.full((random.randint(2, 6), 1), 5, dtype=int)


def _wide_patch(width: int) -> np.ndarray:
    return random.choice(WIDE_PATCHES[width]).copy()


def _place_rows(sep: int, heights: list[int], types: list[str]) -> list[int] | None:
    tops = [None] * len(types)
    wide_indices = [i for i, kind in enumerate(types) if kind == "wide"]
    if wide_indices:
        limits = {idx: sep - heights[idx] - 1 for idx in wide_indices}
        if any(limit < 0 for limit in limits.values()):
            return None
        for _ in range(80):
            used = set()
            for idx in sorted(wide_indices, key=lambda i: limits[i]):
                choices = [row for row in range(limits[idx] + 1) if row not in used]
                if not choices:
                    break
                row = random.choice(choices)
                used.add(row)
                tops[idx] = row
            else:
                break
        else:
            return None
    for idx, kind in enumerate(types):
        if kind == "narrow":
            max_top = sep - heights[idx] - 1
            if max_top < 0:
                return None
            tops[idx] = random.randint(0, max_top)
    return [int(r) for r in tops]


def _build_input(
    height: int,
    width: int,
    sep: int,
    separator: np.ndarray,
    patches: list[np.ndarray],
    intervals: list[tuple[int, int]],
    tops: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[sep] = separator
    for patch, (c0, _c1), r0 in zip(patches, intervals, tops):
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _render_expected(
    height: int,
    width: int,
    sep: int,
    separator: np.ndarray,
    patches: list[np.ndarray],
    intervals: list[tuple[int, int]],
    types: list[str],
    tops: list[int],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    out[sep] = separator.copy()

    narrow_ids = [i for i, kind in enumerate(types) if kind == "narrow"]
    wide_ids = [i for i, kind in enumerate(types) if kind == "wide"]
    narrow_ids.sort(key=lambda i: intervals[i][0])
    wide_ids.sort(key=lambda i: (tops[i], intervals[i][0]))

    sep_before = separator.copy()
    for idx in narrow_ids:
        patch = patches[idx].copy()
        c0, _c1 = intervals[idx]
        patch[patch == 5] = int(sep_before[c0])
        paste(out, patch, sep + 1, c0, transparent=0)

    for idx in wide_ids:
        patch = patches[idx]
        c0, c1 = intervals[idx]
        paste(out, patch, sep - patch.shape[0], c0, transparent=0)
        for c in range(c0, c1 + 1):
            out[sep, c] = CYCLE_MAP[int(out[sep, c])]
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    types: list[str],
    intervals: list[tuple[int, int]],
) -> bool:
    bg = get_bg(inp)
    sep = _separator_row(inp, bg)
    if not np.all(inp[sep + 1 :] == 0):
        return False
    row = inp[sep]
    runs = []
    start = 0
    for c in range(1, row.size):
        if row[c] != row[start]:
            runs.append((int(row[start]), c - start))
            start = c
    runs.append((int(row[start]), row.size - start))
    if len(runs) != 3 or any(length < 2 for _color, length in runs):
        return False
    comps = _sorted_gray_components(inp, sep)
    if len(comps) != len(types):
        return False
    boxes = []
    widths = []
    for comp in comps:
        r0, c0, r1, c1 = bbox_coords(comp["coords"])
        boxes.append((r0, c0, r1, c1))
        widths.append(len(np.unique(comp["coords"][:, 1])))
        if comp["area"] < 2 or comp["area"] > 6:
            return False
    boxes.sort(key=lambda box: box[1])
    for left, right in zip(boxes, boxes[1:]):
        if left[3] + 1 >= right[1]:
            return False
    if 1 not in widths or not any(w >= 2 for w in widths):
        return False
    if np.array_equal(expected, _drop_solution(inp, treat_all_as_narrow=True)):
        return False
    if np.array_equal(expected, _drop_solution(inp, treat_all_as_wide=True)):
        return False
    if np.array_equal(expected, _drop_solution(inp, apply_cycle=False)):
        return False
    if np.array_equal(expected, _drop_solution(inp, recolor_narrow=False)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(10, 15)
    n_sprites = random.randint(3, 5)
    types = _choose_types(n_sprites)
    intervals = _choose_intervals(types, width)
    if intervals is None:
        return None

    separator = _pick_separator(width, intervals, types)
    if separator is None:
        return None

    patches = []
    for (c0, c1), kind in zip(intervals, types):
        sprite_w = c1 - c0 + 1
        patches.append(_narrow_patch() if kind == "narrow" else _wide_patch(sprite_w))

    heights = [patch.shape[0] for patch in patches]
    max_h = max(heights)
    max_narrow_h = max((patch.shape[0] for patch, kind in zip(patches, types) if kind == "narrow"), default=2)
    sep_min = max(5, max_h + 1)
    sep_max = min(8, 30 - max_narrow_h - 2)
    if sep_min > sep_max:
        return None
    sep = random.randint(sep_min, sep_max)
    height = sep + max_narrow_h + random.randint(2, 4)
    if height > 18:
        return None

    tops = _place_rows(sep, heights, types)
    if tops is None:
        return None

    inp = _build_input(height, width, sep, separator, patches, intervals, tops)
    expected = _render_expected(height, width, sep, separator, patches, intervals, types, tops)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(inp, expected, types, intervals):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid width-split gravity puzzle")
