from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np


def _install_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            path_str = str(candidate)
            if path_str not in sys.path:
                sys.path.append(path_str)
            return
    raise ImportError("could not locate new/solvers for arc_utils import")


_install_arc_utils_path()
from arc_utils import bbox, crop_bbox  # noqa: E402


def _analyze_crop(crop: np.ndarray) -> dict[str, object]:
    first_pos: dict[int, tuple[int, int]] = {}
    raw_counts: dict[int, int] = {}
    even_counts: dict[int, int] = {}
    first_order: list[int] = []

    h, w = crop.shape
    for r in range(h):
        for c in range(w):
            color = int(crop[r, c])
            if color not in first_pos:
                first_pos[color] = (r, c)
                first_order.append(color)
                raw_counts[color] = 0
                even_counts[color] = 0
            raw_counts[color] += 1
            if (r + c) % 2 == 0:
                even_counts[color] += 1

    raw_order = sorted(first_order, key=lambda color: (-raw_counts[color], first_pos[color]))
    masked_order = sorted(first_order, key=lambda color: (-even_counts[color], first_pos[color]))
    return {
        "first_pos": first_pos,
        "first_order": first_order,
        "raw_counts": raw_counts,
        "even_counts": even_counts,
        "raw_order": raw_order,
        "masked_order": masked_order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the foreground rectangle by a checkerboard mask and emit its colors in surviving-count order.

    Input structure:
    - The background is black (0).
    - All non-black cells form one tight filled rectangular foreground region.
    - The rectangle contains 3 or 4 colors distributed unevenly across even- and odd-parity cells.

    Rule:
    - Find the tight bounding box of all non-black cells and crop to that rectangle.
    - Using the crop's top-left cell as (0,0), keep only cells where (r+c) mod 2 == 0 and treat odd-parity cells as removed.
    - Count the surviving cells of each original foreground color.
    - Sort colors by surviving count descending; if counts tie, break the tie by the color's first appearance in the cropped rectangle under row-major scan.
    - Return a 1 x k row containing the sorted colors once each.

    Key insight:
    - The checkerboard is not decoration: it resamples space, so color order is determined by parity-sensitive survival counts rather than the original totals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg_bbox = bbox(g != 0)
    assert fg_bbox is not None, "expected a non-black foreground rectangle"
    crop = crop_bbox(g, fg_bbox)
    assert np.all(crop != 0), "foreground bounding box must be fully filled"

    analysis = _analyze_crop(crop)
    masked_order = analysis["masked_order"]
    return np.array([masked_order], dtype=int)


RECT_H_RANGE = (5, 8)
RECT_W_RANGE = (6, 9)
MARGIN_RANGE = (1, 3)
COLOR_COUNT_OPTIONS = (3, 4)
TIE_BRANCH_PROB = 0.55
MAX_ATTEMPTS = 500


def _parity_cell_counts(h: int, w: int) -> tuple[int, int]:
    area = h * w
    return (area + 1) // 2, area // 2


def _sample_first_order(color_count: int, want_tie: bool) -> list[int]:
    indices = list(range(color_count))
    valid_orders = []
    for perm in itertools.permutations(indices):
        if perm.index(0) >= perm.index(1):
            continue
        if want_tie and perm.index(2) >= perm.index(3):
            continue
        valid_orders.append(list(perm))
    return random.choice(valid_orders)


def _sample_profiles(n_even: int, n_odd: int, color_count: int, want_tie: bool) -> dict[int, tuple[int, int]] | None:
    other_count = color_count - 2

    for _ in range(400):
        max_ea = n_even - other_count - 1
        if max_ea < 1:
            return None
        e_a = random.randint(1, max_ea)

        max_eb = n_even - other_count - e_a
        if max_eb <= e_a:
            continue
        e_b = random.randint(e_a + 1, max_eb)

        max_pair_odd = n_odd - other_count
        min_total = e_b + 1
        max_total = (max_pair_odd + e_a + e_b) // 2
        if min_total > max_total:
            continue
        total_ab = random.randint(min_total, max_total)
        o_a = total_ab - e_a
        o_b = total_ab - e_b

        rem_even = n_even - e_a - e_b
        rem_odd = n_odd - o_a - o_b
        if rem_even < other_count or rem_odd < other_count:
            continue

        profiles: dict[int, tuple[int, int]] = {0: (e_a, o_a), 1: (e_b, o_b)}

        if color_count == 3:
            profiles[2] = (rem_even, rem_odd)
        elif want_tie:
            if rem_even % 2 != 0 or rem_even < 2 or rem_odd < 2:
                continue
            tie_even = rem_even // 2
            if tie_even in (e_a, e_b):
                continue
            o_c = random.randint(1, rem_odd - 1)
            o_d = rem_odd - o_c
            profiles[2] = (tie_even, o_c)
            profiles[3] = (tie_even, o_d)
        else:
            e_c = random.randint(1, rem_even - 1)
            e_d = rem_even - e_c
            if len({e_a, e_b, e_c, e_d}) != 4:
                continue
            o_c = random.randint(1, rem_odd - 1)
            o_d = rem_odd - o_c
            profiles[2] = (e_c, o_c)
            profiles[3] = (e_d, o_d)

        even_values = [profiles[idx][0] for idx in range(color_count)]
        if want_tie:
            if color_count != 4 or not (even_values[2] == even_values[3]):
                continue
            if even_values[2] in even_values[:2]:
                continue
        else:
            if len(set(even_values)) != color_count:
                continue

        if any(e <= 0 or o <= 0 for e, o in profiles.values()):
            continue
        return profiles

    return None


def _render_rectangle(
    rect_h: int,
    rect_w: int,
    colors: list[int],
    profiles: dict[int, tuple[int, int]],
    first_order: list[int],
) -> np.ndarray | None:
    rect = np.zeros((rect_h, rect_w), dtype=int)
    positions = [(r, c) for r in range(rect_h) for c in range(rect_w)]
    anchor_positions = positions[: len(colors)]
    even_left = {colors[idx]: profiles[idx][0] for idx in range(len(colors))}
    odd_left = {colors[idx]: profiles[idx][1] for idx in range(len(colors))}
    used = set()

    for pos, color_idx in zip(anchor_positions, first_order):
        r, c = pos
        color = colors[color_idx]
        rect[r, c] = color
        used.add(pos)
        if (r + c) % 2 == 0:
            even_left[color] -= 1
        else:
            odd_left[color] -= 1

    if any(count < 0 for count in even_left.values()) or any(count < 0 for count in odd_left.values()):
        return None

    even_cells = [pos for pos in positions[len(colors) :] if (pos[0] + pos[1]) % 2 == 0]
    odd_cells = [pos for pos in positions[len(colors) :] if (pos[0] + pos[1]) % 2 == 1]

    even_bag = [color for color, count in even_left.items() for _ in range(count)]
    odd_bag = [color for color, count in odd_left.items() for _ in range(count)]
    if len(even_bag) != len(even_cells) or len(odd_bag) != len(odd_cells):
        return None

    random.shuffle(even_bag)
    random.shuffle(odd_bag)
    for (r, c), color in zip(even_cells, even_bag):
        rect[r, c] = color
    for (r, c), color in zip(odd_cells, odd_bag):
        rect[r, c] = color

    if np.any(rect == 0):
        return None
    return rect


def _embed_rectangle(rect: np.ndarray) -> np.ndarray:
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    grid = np.zeros((top + rect.shape[0] + bottom, left + rect.shape[1] + right), dtype=int)
    grid[top : top + rect.shape[0], left : left + rect.shape[1]] = rect
    return grid


def _semantic_checks(
    grid: np.ndarray,
    critical_pair: tuple[int, int],
    want_tie: bool,
) -> bool:
    fg_bbox = bbox(grid != 0)
    if fg_bbox is None:
        return False
    crop = crop_bbox(grid, fg_bbox)
    if not np.all(crop != 0):
        return False

    analysis = _analyze_crop(crop)
    first_pos = analysis["first_pos"]
    first_order = analysis["first_order"]
    raw_counts = analysis["raw_counts"]
    even_counts = analysis["even_counts"]
    raw_order = analysis["raw_order"]
    masked_order = analysis["masked_order"]
    colors = list(first_order)

    if not (3 <= len(colors) <= 4):
        return False
    if any(raw_counts[color] <= 1 for color in colors):
        return False
    if any(even_counts[color] <= 0 for color in colors):
        return False

    color_a, color_b = critical_pair
    if raw_counts[color_a] != raw_counts[color_b]:
        return False
    if first_pos[color_a] >= first_pos[color_b]:
        return False
    if even_counts[color_a] >= even_counts[color_b]:
        return False
    if raw_order.index(color_a) >= raw_order.index(color_b):
        return False
    if masked_order.index(color_b) >= masked_order.index(color_a):
        return False

    if raw_order == masked_order:
        return False
    if masked_order == first_order:
        return False

    survivor_values = [even_counts[color] for color in colors]
    if want_tie:
        tied_pairs = [
            (colors[i], colors[j])
            for i in range(len(colors))
            for j in range(i + 1, len(colors))
            if even_counts[colors[i]] == even_counts[colors[j]]
        ]
        if not tied_pairs:
            return False
    else:
        if len(set(survivor_values)) != len(survivor_values):
            return False

    out = main(grid)
    if out.shape != (1, len(colors)):
        return False
    if list(map(int, out[0])) != list(masked_order):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)
    color_count = random.choice(COLOR_COUNT_OPTIONS)
    want_tie = color_count == 4 and random.random() < TIE_BRANCH_PROB

    n_even, n_odd = _parity_cell_counts(rect_h, rect_w)
    profiles = _sample_profiles(n_even, n_odd, color_count, want_tie)
    if profiles is None:
        return None

    first_order = _sample_first_order(color_count, want_tie)
    colors = random.sample(range(1, 10), color_count)
    rect = _render_rectangle(rect_h, rect_w, colors, profiles, first_order)
    if rect is None:
        return None

    critical_pair = (colors[0], colors[1])
    grid = _embed_rectangle(rect)
    if not _semantic_checks(grid, critical_pair, want_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-count ordering input")
