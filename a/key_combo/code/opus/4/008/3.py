from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


for _extra_path in (
    Path(__file__).resolve().parent,
    Path("/home/lixinyu/snap/new/solvers"),
):
    _extra_str = str(_extra_path)
    if _extra_str not in sys.path:
        sys.path.append(_extra_str)

from arc_utils import find_components, get_bg, paste, random_free_location


BG = 0
LEGEND_CENTER = 9
LEGEND_INDICATOR = 7
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 8)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIDE_RANGE = (19, 24)
PAIR_COUNT_CHOICES = (2, 3)
SHARED_SPAN_RANGE = (2, 5)
AXIAL_SPAN_RANGE = (2, 4)
ODD_GAPS = (3, 5, 7)
EVEN_GAPS = (4, 6)
PAIR_BORDER = 4
PAIR_PADDING = 2
LEGEND_PAD = 1
MAX_GENERATION_ATTEMPTS = 400


def _is_solid_rect(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return comp["area"] == h * w and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == comp["color"])


def _parse_legends(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    legends: dict[int, tuple[int, int]] = {}

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if r0 != 0 and c0 != 0:
                continue

            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            if int(patch[1, 1]) != LEGEND_CENTER:
                continue

            indicators = [(dr, dc) for dr, dc in CARDINALS if int(patch[1 + dr, 1 + dc]) == LEGEND_INDICATOR]
            if len(indicators) != 1:
                continue
            dr, dc = indicators[0]

            fill_values = []
            valid = True
            for rr in range(3):
                for cc in range(3):
                    if (rr, cc) == (1, 1) or (rr, cc) == (1 + dr, 1 + dc):
                        continue
                    value = int(patch[rr, cc])
                    if value in (BG, LEGEND_CENTER, LEGEND_INDICATOR):
                        valid = False
                        break
                    fill_values.append(value)
                if not valid:
                    break
            if not valid or len(set(fill_values)) != 1:
                continue

            fill = fill_values[0]
            if fill in legends:
                assert legends[fill] == (dr, dc)
            legends[fill] = (dr, dc)

    return legends


def _collect_pairs(grid: np.ndarray, bg: int, colors: list[int]) -> list[dict]:
    pairs: list[dict] = []

    for color in sorted(colors):
        comps = find_components(
            grid,
            bg=bg,
            colors=[color],
            connectivity=4,
            monochromatic=True,
        )
        rects = [comp for comp in comps if _is_solid_rect(grid, comp)]
        assert len(rects) == 2

        a, b = rects
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        same_rows = (ar0, ar1) == (br0, br1)
        same_cols = (ac0, ac1) == (bc0, bc1)
        assert same_rows ^ same_cols

        if same_rows:
            left, right = sorted(rects, key=lambda comp: comp["bbox"][1])
            lr0, lc0, lr1, lc1 = left["bbox"]
            rr0, rc0, rr1, rc1 = right["bbox"]
            assert (lr0, lr1) == (rr0, rr1)
            gap = rc0 - lc1 - 1
            assert gap >= 1
            pairs.append(
                {
                    "color": color,
                    "orientation": "H",
                    "near": tuple(map(int, left["bbox"])),
                    "far": tuple(map(int, right["bbox"])),
                    "gap": int(gap),
                }
            )
        else:
            top, bottom = sorted(rects, key=lambda comp: comp["bbox"][0])
            tr0, tc0, tr1, tc1 = top["bbox"]
            br0, bc0, br1, bc1 = bottom["bbox"]
            assert (tc0, tc1) == (bc0, bc1)
            gap = br0 - tr1 - 1
            assert gap >= 1
            pairs.append(
                {
                    "color": color,
                    "orientation": "V",
                    "near": tuple(map(int, top["bbox"])),
                    "far": tuple(map(int, bottom["bbox"])),
                    "gap": int(gap),
                }
            )

    return pairs


def _paint_pair_half(out: np.ndarray, pair: dict, side: str) -> None:
    span = int(pair["gap"]) // 2
    if span <= 0:
        return

    color = int(pair["color"])
    r0, c0, r1, c1 = pair["near"]
    rr0, rc0, rr1, rc1 = pair["far"]

    if pair["orientation"] == "H":
        if side in ("near", "both"):
            out[r0 : r1 + 1, c1 + 1 : c1 + 1 + span] = color
        if side in ("far", "both"):
            out[rr0 : rr1 + 1, rc0 - span : rc0] = color
    else:
        if side in ("near", "both"):
            out[r1 + 1 : r1 + 1 + span, c0 : c1 + 1] = color
        if side in ("far", "both"):
            out[rr0 - span : rr0, rc0 : rc1 + 1] = color


def _solve_with_mode(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legends = _parse_legends(g)
    assert legends

    pairs = _collect_pairs(g, bg, list(legends))
    out = g.copy()

    for pair in pairs:
        if mode == "legend":
            dr, dc = legends[pair["color"]]
            if pair["orientation"] == "H":
                assert dr == 0 and dc in (-1, 1)
                side = "near" if dc < 0 else "far"
            else:
                assert dc == 0 and dr in (-1, 1)
                side = "near" if dr < 0 else "far"
        elif mode == "negative":
            side = "near"
        elif mode == "positive":
            side = "far"
        elif mode == "symmetric":
            side = "both"
        else:
            raise ValueError(f"unknown mode: {mode}")

        _paint_pair_half(out, pair, side)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use same-color border legends to choose which side of each aligned rectangle pair extends into the gap.

    Input structure:
    - The background is black, the interior contains 2-3 pairs of same-color solid rectangles, and each pair shares either one row span or one column span with a black gap between the facing edges.
    - The top edge or left edge contains one 3x3 legend per pair; the legend's fill color matches the pair color, its center cell is 9, and one adjacent 7 cell indicates a direction.

    Rule:
    - Match each legend to the rectangle pair of the same color, determine whether that pair is horizontal or vertical, and interpret the legend direction as the side that owns the extension right.
    - The owning rectangle fills only its own half of the gap across the pair's shared span; the opposite half stays unchanged, and odd-width gaps keep their middle row or column black.
    - Legends themselves stay unchanged and different color pairs are processed independently.

    Key insight:
    - The legend does not say where pixels move globally; it breaks the left/right or up/down symmetry of a same-color rectangle pair by selecting exactly one side's half-gap extension.
    """
    return _solve_with_mode(grid, mode="legend")


def _legend_patch(color: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((3, 3), int(color), dtype=int)
    patch[1, 1] = LEGEND_CENTER
    patch[1 + dr, 1 + dc] = LEGEND_INDICATOR
    return patch


def _build_pair_patch(color: int, orientation: str, gap: int, sign: int) -> dict:
    shared = random.randint(*SHARED_SPAN_RANGE)
    side_a = random.randint(*AXIAL_SPAN_RANGE)
    side_b = random.randint(*AXIAL_SPAN_RANGE)

    if orientation == "H":
        h = shared
        w = side_a + gap + side_b
        inp = np.full((h, w), BG, dtype=int)
        inp[:, :side_a] = color
        inp[:, side_a + gap : side_a + gap + side_b] = color
        out = inp.copy()
        span = gap // 2
        if sign < 0:
            out[:, side_a : side_a + span] = color
            direction = (0, -1)
        else:
            out[:, side_a + gap - span : side_a + gap] = color
            direction = (0, 1)
    else:
        h = side_a + gap + side_b
        w = shared
        inp = np.full((h, w), BG, dtype=int)
        inp[:side_a, :] = color
        inp[side_a + gap : side_a + gap + side_b, :] = color
        out = inp.copy()
        span = gap // 2
        if sign < 0:
            out[side_a : side_a + span, :] = color
            direction = (-1, 0)
        else:
            out[side_a + gap - span : side_a + gap, :] = color
            direction = (1, 0)

    return {
        "color": int(color),
        "orientation": orientation,
        "gap": int(gap),
        "direction": direction,
        "input_patch": inp,
        "output_patch": out,
        "block": np.ones(inp.shape, dtype=int),
    }


def _choose_orientations(n_pairs: int) -> list[str]:
    if n_pairs == 2:
        values = ["H", "V"] if random.random() < 0.7 else [random.choice(["H", "V"])] * 2
    else:
        values = ["H", "V", random.choice(["H", "V"])]
    random.shuffle(values)
    return values


def _choose_gaps(n_pairs: int) -> list[int]:
    parity_flags = [0, 1]
    while len(parity_flags) < n_pairs:
        parity_flags.append(random.randint(0, 1))
    random.shuffle(parity_flags)
    return [random.choice(EVEN_GAPS if parity == 0 else ODD_GAPS) for parity in parity_flags]


def _choose_signs(n_pairs: int) -> list[int]:
    signs = [-1, 1]
    while len(signs) < n_pairs:
        signs.append(random.choice([-1, 1]))
    random.shuffle(signs)
    return signs


def _choose_legend_sides(n_pairs: int) -> list[str]:
    if n_pairs == 2:
        sides = ["top", "left"] if random.random() < 0.7 else [random.choice(["top", "left"])] * 2
    else:
        sides = ["top", "left", random.choice(["top", "left"])]
    random.shuffle(sides)
    return sides


def _place_pairs(rows: int, cols: int, pair_specs: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)
    occ = np.zeros((rows, cols), dtype=int)
    placed: list[dict] = []

    ordered = sorted(pair_specs, key=lambda spec: spec["block"].size, reverse=True)
    for spec in ordered:
        try:
            r0, c0 = random_free_location(
                occ,
                spec["block"],
                transparent=BG,
                border=PAIR_BORDER,
                padding=PAIR_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(occ, spec["block"], r0, c0)
        paste(inp, spec["input_patch"], r0, c0, transparent=BG)
        paste(out, spec["output_patch"], r0, c0, transparent=BG)

        record = dict(spec)
        record["pair_pos"] = (int(r0), int(c0))
        placed.append(record)

    return inp, out, placed


def _legend_slot_free(occ: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occ.shape
    rr0 = max(0, r0 - LEGEND_PAD)
    rr1 = min(h, r0 + 3 + LEGEND_PAD)
    cc0 = max(0, c0 - LEGEND_PAD)
    cc1 = min(w, c0 + 3 + LEGEND_PAD)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_legends(inp: np.ndarray, out: np.ndarray, placed_pairs: list[dict]) -> bool:
    rows, cols = inp.shape
    occ = np.zeros((rows, cols), dtype=bool)
    sides = _choose_legend_sides(len(placed_pairs))

    for spec, preferred in zip(placed_pairs, sides):
        direction = spec["direction"]
        patch = _legend_patch(spec["color"], direction[0], direction[1])
        options = [preferred, "left" if preferred == "top" else "top"]

        placed = False
        for side in options:
            candidates = []
            if side == "top":
                for c0 in range(4, cols - 2):
                    if _legend_slot_free(occ, 0, c0):
                        candidates.append((0, c0))
            else:
                for r0 in range(4, rows - 2):
                    if _legend_slot_free(occ, r0, 0):
                        candidates.append((r0, 0))

            if not candidates:
                continue

            r0, c0 = random.choice(candidates)
            paste(inp, patch, r0, c0)
            paste(out, patch, r0, c0)
            occ[r0 : r0 + 3, c0 : c0 + 3] = True
            spec["legend_pos"] = (int(r0), int(c0))
            placed = True
            break

        if not placed:
            return False

    return True


def _flip_direction(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -int(dr), -int(dc)


def _flip_legend(inp: np.ndarray, spec: dict) -> np.ndarray:
    alt = inp.copy()
    r0, c0 = spec["legend_pos"]
    patch = _legend_patch(spec["color"], *_flip_direction(spec["direction"]))
    paste(alt, patch, r0, c0)
    return alt


def _validate_scene(inp: np.ndarray, expected: np.ndarray, placed_pairs: list[dict]) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    legends = _parse_legends(inp)
    if len(legends) != len(placed_pairs):
        return False
    if set(legends) != {spec["color"] for spec in placed_pairs}:
        return False

    symmetric = _solve_with_mode(inp, mode="symmetric")
    negative = _solve_with_mode(inp, mode="negative")
    positive = _solve_with_mode(inp, mode="positive")
    if np.array_equal(symmetric, expected):
        return False
    if np.array_equal(negative, expected):
        return False
    if np.array_equal(positive, expected):
        return False

    for spec in placed_pairs:
        flipped = _flip_legend(inp, spec)
        try:
            flipped_out = main(flipped)
        except Exception:
            return False
        if np.array_equal(flipped_out, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_SIDE_RANGE)
    cols = random.randint(*GRID_SIDE_RANGE)
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PAIR_COLORS, n_pairs)
    orientations = _choose_orientations(n_pairs)
    gaps = _choose_gaps(n_pairs)
    signs = _choose_signs(n_pairs)

    pair_specs = [
        _build_pair_patch(color, orientation, gap, sign)
        for color, orientation, gap, sign in zip(colors, orientations, gaps, signs)
    ]

    placed = _place_pairs(rows, cols, pair_specs)
    if placed is None:
        return None
    inp, expected, placed_pairs = placed

    if not _place_legends(inp, expected, placed_pairs):
        return None
    if not _validate_scene(inp, expected, placed_pairs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
