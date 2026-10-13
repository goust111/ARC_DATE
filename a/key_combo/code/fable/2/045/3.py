from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import paste
except Exception:
    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        h, w = patch.shape
        if not clip and (r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]):
            raise ValueError("paste: patch out of bounds")
        rr0 = max(0, int(r0))
        cc0 = max(0, int(c0))
        rr1 = min(dst.shape[0], int(r0) + h)
        cc1 = min(dst.shape[1], int(c0) + w)
        pr0 = rr0 - int(r0)
        pc0 = cc0 - int(c0)
        patch_view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
        if transparent is None:
            dst[rr0:rr1, cc0:cc1] = patch_view
        else:
            mask = patch_view != int(transparent)
            dst_view = dst[rr0:rr1, cc0:cc1]
            dst_view[mask] = patch_view[mask]
        return dst


PALETTE = tuple(range(1, 10))
STRIPE_COUNT_RANGE = (4, 6)
GRID_HEIGHT_RANGE = (8, 16)
STRIPE_WIDTH_RANGE = (3, 5)
FOREIGN_COUNT_MIN = 2
FOREIGN_COUNT_MAX = 6
GENERATION_RETRIES = 600
PLACEMENT_RETRIES = 120


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty sequence"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _parse_stripes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    column_colors = [_unique_mode(g[:, c]) for c in range(width)]
    bounds: list[tuple[int, int, int]] = []
    start = 0
    current = column_colors[0]
    for col, color in enumerate(column_colors[1:], start=1):
        if color != current:
            bounds.append((start, col, current))
            start = col
            current = color
    bounds.append((start, width, current))

    stripe_widths = {c1 - c0 for c0, c1, _ in bounds}
    assert len(bounds) >= 2, "expected multiple stripes"
    assert len(stripe_widths) == 1, "expected equal-width stripes"

    main_colors = [color for _, _, color in bounds]
    main_set = set(main_colors)
    assert len(main_set) == len(main_colors), "stripe dominant colors must be unique"

    stripes = []
    for index, (c0, c1, main_color) in enumerate(bounds):
        block = g[:, c0:c1].copy()
        others = sorted(set(map(int, block.reshape(-1))) - {int(main_color)})
        assert len(others) <= 1, "each stripe may contain at most one foreign color"

        foreign_color = None if not others else int(others[0])
        coords = np.argwhere(block == foreign_color) if foreign_color is not None else np.empty((0, 2), dtype=int)
        if foreign_color is not None:
            assert foreign_color in main_set and foreign_color != int(main_color), (
                "foreign color must be another stripe's dominant color"
            )

        stripes.append(
            {
                "index": int(index),
                "main": int(main_color),
                "foreign": foreign_color,
                "block": block,
                "count": int(coords.shape[0]),
                "top_row": height if coords.size == 0 else int(coords[:, 0].min()),
            }
        )
    return stripes


def _chain_order(stripes: list[dict]) -> tuple[list[int], int, int]:
    mains = [int(stripe["main"]) for stripe in stripes]
    indegree = {main: 0 for main in mains}
    successor: dict[int, int] = {}
    pure = []

    for stripe in stripes:
        src = int(stripe["main"])
        dst = stripe["foreign"]
        if dst is None:
            pure.append(src)
            continue
        assert src not in successor, "a stripe must have at most one outgoing pointer"
        successor[src] = int(dst)
        indegree[int(dst)] += 1
        assert indegree[int(dst)] == 1, "a dominant color may appear as foreign in at most one stripe"

    assert len(pure) == 1, "expected exactly one pure stripe"
    assert len(successor) == len(stripes) - 1, "all non-terminal stripes must point to a successor"

    heads = [main for main, deg in indegree.items() if deg == 0]
    assert len(heads) == 1, "expected exactly one chain head"
    head = int(heads[0])
    tail = int(pure[0])

    order = []
    seen = set()
    current = head
    while True:
        assert current not in seen, "successor pointers must not form a cycle"
        seen.add(current)
        order.append(current)
        if current == tail:
            break
        assert current in successor, "chain must continue until the pure tail stripe"
        current = successor[current]

    assert len(order) == len(stripes), "successor pointers must cover every stripe exactly once"
    return order, head, tail


def _ordered_output(stripes: list[dict], order: list[int]) -> np.ndarray:
    by_color = {int(stripe["main"]): stripe["block"] for stripe in stripes}
    return np.concatenate([by_color[color] for color in order], axis=1).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the vertical color stripes by reading each stripe's foreign pixels as a successor pointer.

    Input structure:
    - The grid is partitioned into equal-width vertical stripes with unique dominant colors.
    - Every non-terminal stripe contains some pixels of exactly one other stripe's dominant color.
    - Exactly one stripe is pure and contains no foreign pixels at all.

    Rule:
    - If stripe A contains foreign color B, then stripe B must appear immediately to the right of stripe A.
    - These successor pointers form one chain covering all stripes, with the pure stripe as the tail.
    - Reorder the whole stripes into chain order, preserving every stripe's internal pixels exactly.

    Key insight:
    - The scattered foreign pixels are successor labels rather than noise or sortable counts, so the answer is the unique chain induced by color-to-stripe references.
    """
    stripes = _parse_stripes(grid)
    order, _head, _tail = _chain_order(stripes)
    return _ordered_output(stripes, order)


def _sample_foreign_coords(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    max_per_col = max(1, (height - 1) // 3)
    cells = [(r, c) for r in range(height) for c in range(width)]

    for _ in range(PLACEMENT_RETRIES):
        random.shuffle(cells)
        coords = []
        col_counts = [0] * width
        for r, c in cells:
            if col_counts[c] >= max_per_col:
                continue
            coords.append((int(r), int(c)))
            col_counts[c] += 1
            if len(coords) == count:
                break
        if len(coords) != count:
            continue

        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        if len(rows) < min(2, count):
            continue
        if len(cols) < min(2, width):
            continue
        return coords
    return None


def _make_stripe(height: int, width: int, main_color: int, foreign_color: int | None, count: int) -> np.ndarray | None:
    stripe = np.full((height, width), int(main_color), dtype=int)
    if foreign_color is None:
        assert count == 0
        return stripe

    coords = _sample_foreign_coords(height, width, count)
    if coords is None:
        return None
    for r, c in coords:
        stripe[r, c] = int(foreign_color)
    return stripe


def _input_is_easy_shortcut(stripes: list[dict], order: list[int]) -> bool:
    order_t = tuple(int(x) for x in order)
    input_colors = [int(stripe["main"]) for stripe in stripes]
    input_t = tuple(input_colors)
    index_of = {color: i for i, color in enumerate(input_colors)}
    counts = {int(stripe["main"]): int(stripe["count"]) for stripe in stripes}
    top_rows = {int(stripe["main"]): int(stripe["top_row"]) for stripe in stripes}
    tail = next(int(stripe["main"]) for stripe in stripes if stripe["foreign"] is None)

    def stable_sorted(values: list[int], key_fn, reverse: bool = False) -> tuple[int, ...]:
        triples = [(key_fn(color), index_of[color], color) for color in values]
        triples.sort(reverse=reverse)
        return tuple(color for _, _, color in triples)

    candidates = {
        input_t,
        tuple(reversed(input_colors)),
        tuple(input_colors[1:] + input_colors[:1]),
        tuple(input_colors[-1:] + input_colors[:-1]),
        tuple(sorted(input_colors)),
        tuple(sorted(input_colors, reverse=True)),
        stable_sorted(input_colors, lambda color: counts[color]),
        stable_sorted(input_colors, lambda color: counts[color], reverse=True),
        stable_sorted(input_colors, lambda color: top_rows[color]),
        stable_sorted(input_colors, lambda color: top_rows[color], reverse=True),
        tuple([color for color in input_colors if color != tail] + [tail]),
        tuple([tail] + [color for color in input_colors if color != tail]),
    }
    return order_t in candidates


def _semantic_checks(grid: np.ndarray, expected_order: list[int]) -> bool:
    try:
        stripes = _parse_stripes(grid)
        actual_order, _head, tail = _chain_order(stripes)
        if actual_order != list(expected_order):
            return False

        counts = [int(next(stripe["count"] for stripe in stripes if stripe["main"] == color)) for color in actual_order]
        if counts == sorted(counts) or counts == sorted(counts, reverse=True):
            return False

        top_rows = [int(next(stripe["top_row"] for stripe in stripes if stripe["main"] == color)) for color in actual_order]
        if top_rows == sorted(top_rows) or top_rows == sorted(top_rows, reverse=True):
            return False

        if actual_order == sorted(actual_order) or actual_order == sorted(actual_order, reverse=True):
            return False

        if _input_is_easy_shortcut(stripes, actual_order):
            return False

        out = main(grid)
        if not np.array_equal(out, _ordered_output(stripes, actual_order)):
            return False

        out_stripes = _parse_stripes(out)
        out_order = [int(stripe["main"]) for stripe in out_stripes]
        if out_order != actual_order:
            return False
        if int(out_order[-1]) != int(tail):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    stripe_count = random.randint(*STRIPE_COUNT_RANGE)
    width_choices = [width for width in range(STRIPE_WIDTH_RANGE[0], STRIPE_WIDTH_RANGE[1] + 1) if width * stripe_count <= 30]
    stripe_width = random.choice(width_choices)
    height = random.randint(*GRID_HEIGHT_RANGE)

    chain_order = random.sample(PALETTE, stripe_count)
    input_order = random.sample(chain_order, stripe_count)
    if input_order == chain_order:
        return None

    max_foreign = min(FOREIGN_COUNT_MAX, stripe_width * max(1, (height - 1) // 3))
    if max_foreign < FOREIGN_COUNT_MIN:
        return None

    count_by_color: dict[int, int] = {}
    for color in chain_order[:-1]:
        count_by_color[color] = random.randint(FOREIGN_COUNT_MIN, max_foreign)
    count_by_color[chain_order[-1]] = 0

    color_to_stripe: dict[int, np.ndarray] = {}
    for idx, color in enumerate(chain_order):
        nxt = None if idx == stripe_count - 1 else chain_order[idx + 1]
        stripe = _make_stripe(height, stripe_width, color, nxt, count_by_color[color])
        if stripe is None:
            return None
        color_to_stripe[color] = stripe

    grid = np.zeros((height, stripe_count * stripe_width), dtype=int)
    for idx, color in enumerate(input_order):
        paste(grid, color_to_stripe[color], 0, idx * stripe_width)

    return grid if _semantic_checks(grid, chain_order) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid successor-pointer stripe puzzle")
