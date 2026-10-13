from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, find_components, get_bg


GEN_TRIES = 500
LAYOUT_OPTIONS = ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2))
PAIR_COLORS = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.GREY), int(Color.PINK), int(Color.ORANGE), int(Color.TEAL), int(Color.MAROON)]


def _is_solid_two_by_two(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return (
        (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
        and int(comp["area"]) == 4
        and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"]))
    )


def _rect_distance(bbox: tuple[int, int, int, int], r: int, c: int) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return max(r0 - int(r), 0, int(r) - r1) + max(c0 - int(c), 0, int(c) - c1)


def _detect_blocks(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    blocks = []
    for comp in comps:
        if not _is_solid_two_by_two(grid, comp):
            raise AssertionError("expected only solid 2x2 blocks")
        blocks.append(
            {
                "color": int(comp["color"]),
                "bbox": tuple(map(int, comp["bbox"])),
            }
        )
    assert blocks, "expected at least one block pair"
    return bg, blocks


def _find_pairs(grid: np.ndarray, blocks: list[dict], bg: int) -> list[dict]:
    pairs: list[dict] = []
    for idx, a in enumerate(blocks):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for b in blocks[idx + 1 :]:
            if int(a["color"]) == int(b["color"]):
                continue

            br0, bc0, br1, bc1 = b["bbox"]
            if (ar0, ar1) == (br0, br1):
                left, right = sorted((a, b), key=lambda block: block["bbox"][1])
                _lr0, _lc0, lr1, lc1 = left["bbox"]
                rr0, rc0, _rr1, _rc1 = right["bbox"]
                if lc1 + 1 <= rc0 - 1 and np.all(grid[rr0 : lr1 + 1, lc1 + 1 : rc0] == bg):
                    pairs.append(
                        {
                            "blocks": (left, right),
                            "corridor": (rr0, lc1 + 1, lr1, rc0 - 1),
                        }
                    )

            if (ac0, ac1) == (bc0, bc1):
                top, bottom = sorted((a, b), key=lambda block: block["bbox"][0])
                _tr0, tc0, tr1, tc1 = top["bbox"]
                br0, _bc0, _br1, _bc1 = bottom["bbox"]
                if tr1 + 1 <= br0 - 1 and np.all(grid[tr1 + 1 : br0, tc0 : tc1 + 1] == bg):
                    pairs.append(
                        {
                            "blocks": (top, bottom),
                            "corridor": (tr1 + 1, tc0, br0 - 1, tc1),
                        }
                    )

    assert pairs, "expected at least one aligned block pair"
    return pairs


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill each corridor cell with the color of its uniquely nearest endpoint block.
    Cells tied for the smallest Manhattan distance stay background.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg, blocks = _detect_blocks(grid)
    pairs = _find_pairs(grid, blocks, bg)

    contests: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pair in pairs:
        r0, c0, r1, c1 = pair["corridor"]
        for r in range(int(r0), int(r1) + 1):
            for c in range(int(c0), int(c1) + 1):
                bucket = contests.setdefault((int(r), int(c)), [])
                for block in pair["blocks"]:
                    bucket.append((_rect_distance(block["bbox"], r, c), int(block["color"])))

    out = grid.copy()
    for (r, c), candidates in contests.items():
        best = min(distance for distance, _color in candidates)
        winners = [color for distance, color in candidates if distance == best]
        out[int(r), int(c)] = winners[0] if len(winners) == 1 else bg
    return out


def _sample_separated_starts(lo: int, hi: int, count: int) -> list[int] | None:
    starts = list(range(int(lo), int(hi) + 1))
    random.shuffle(starts)
    chosen: list[int] = []
    for start in starts:
        if all(abs(int(start) - existing) >= 2 for existing in chosen):
            chosen.append(int(start))
        if len(chosen) == int(count):
            return sorted(chosen)
    return None


def _gen_attempt() -> np.ndarray | None:
    h_pairs, v_pairs = random.choice(LAYOUT_OPTIONS)
    total_pairs = int(h_pairs + v_pairs)

    height = random.randint(max(12, 8 + 2 * h_pairs), 24)
    width = random.randint(max(12, 8 + 2 * v_pairs), 24)

    row_starts = _sample_separated_starts(3, height - 5, h_pairs)
    col_starts = _sample_separated_starts(3, width - 5, v_pairs)
    if row_starts is None or col_starts is None:
        return None

    colors = random.sample(PAIR_COLORS, 2 * total_pairs)
    color_idx = 0
    grid = np.full((height, width), int(Color.BLACK), dtype=int)

    for row0 in row_starts:
        left_c0 = random.choice((0, 1))
        right_c0 = random.choice((width - 3, width - 2))
        color_a, color_b = colors[color_idx], colors[color_idx + 1]
        color_idx += 2
        grid[row0 : row0 + 2, left_c0 : left_c0 + 2] = int(color_a)
        grid[row0 : row0 + 2, right_c0 : right_c0 + 2] = int(color_b)

    for col0 in col_starts:
        top_r0 = random.choice((0, 1))
        bottom_r0 = random.choice((height - 3, height - 2))
        color_a, color_b = colors[color_idx], colors[color_idx + 1]
        color_idx += 2
        grid[top_r0 : top_r0 + 2, col0 : col0 + 2] = int(color_a)
        grid[bottom_r0 : bottom_r0 + 2, col0 : col0 + 2] = int(color_b)

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if out.ndim != 2 or out.shape[0] > 30 or out.shape[1] > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
