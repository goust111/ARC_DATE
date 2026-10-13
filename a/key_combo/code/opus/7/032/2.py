from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

CUP_COLORS = [1, GREEN, PINK, ORANGE, MAROON]
OPENINGS = ["up", "down", "left", "right"]

TEMP_H = 22
TEMP_W = 30
LEFT_MAX_COL = 12
BAR_BASELINE_RANGE = (13, 18)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _cup_mask(opening: str, depth: int) -> np.ndarray:
    if opening == "up":
        patch = np.zeros((depth + 1, 3), dtype=bool)
        patch[:, 0] = True
        patch[:, 2] = True
        patch[-1, :] = True
        return patch
    if opening == "down":
        patch = np.zeros((depth + 1, 3), dtype=bool)
        patch[:, 0] = True
        patch[:, 2] = True
        patch[0, :] = True
        return patch
    if opening == "left":
        patch = np.zeros((3, depth + 1), dtype=bool)
        patch[0, :] = True
        patch[2, :] = True
        patch[:, -1] = True
        return patch
    if opening == "right":
        patch = np.zeros((3, depth + 1), dtype=bool)
        patch[0, :] = True
        patch[2, :] = True
        patch[:, 0] = True
        return patch
    raise ValueError(opening)


def _classify_cup(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True

    for opening in OPENINGS:
        depth = h - 1 if opening in ("up", "down") else w - 1
        if depth < 3 or depth > 6:
            continue
        expected = _cup_mask(opening, depth)
        if expected.shape == patch.shape and np.array_equal(expected, patch):
            return {
                "bbox": (r0, c0, r1, c1),
                "opening": opening,
                "depth": depth,
            }
    return None


def _cavity_fill_order(cup: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cup["bbox"]
    opening = cup["opening"]
    if opening == "up":
        return [(r, c0 + 1) for r in range(r1 - 1, r0 - 1, -1)]
    if opening == "down":
        return [(r, c0 + 1) for r in range(r0 + 1, r1 + 1)]
    if opening == "left":
        return [(r0 + 1, c) for c in range(c1 - 1, c0 - 1, -1)]
    return [(r0 + 1, c) for c in range(c0 + 1, c1 + 1)]


def _find_cups(grid: np.ndarray, bg: int) -> dict[int, dict]:
    cups: dict[int, dict] = {}
    for color in sorted(set(map(int, grid.ravel())) - {int(bg), YELLOW}):
        if color in (RED, TEAL):
            continue
        for coords in _components(grid == color):
            if len(coords) <= 1:
                continue
            cup = _classify_cup(coords)
            if cup is not None:
                cups[color] = cup
                break
    return cups


def _bar_amounts(grid: np.ndarray, cups: dict[int, dict]) -> dict[int, int]:
    amounts = {color: 0 for color in cups}
    for bar_color, sign in ((TEAL, 1), (RED, -1)):
        for coords in _components(grid == bar_color):
            r0, c0, r1, c1 = _bbox(coords)
            if c0 != c1:
                continue
            label_r = r1 + 1
            if label_r >= grid.shape[0]:
                continue
            label_color = int(grid[label_r, c0])
            if label_color not in cups:
                continue
            amounts[label_color] += sign * len(coords)
    return amounts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each cup's one-cell-wide cavity from the closed end using the arithmetic result of its tagged bars.

    Input structure:
    - The grid has a black background, several monochrome cup outlines on the left, and thin red/teal vertical bars with same-color labels on the right.
    - Each cup is a one-cell-thick three-sided container whose cavity is a straight one-cell-wide line of black cells.
    - Every bar has a label directly below it; the label color matches exactly one cup color.

    Rule:
    - For each cup, collect every teal bar and red bar whose label matches that cup color.
    - Compute fill amount = sum(teal heights) - sum(red heights).
    - Detect the cup opening direction from the missing side of its bounding box, then fill that many cavity cells in yellow from the closed end toward the opening.

    Key insight:
    - Bar colors determine how much to fill, while the open side of the cup determines which end of the cavity is filled first.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    cups = _find_cups(g, bg)
    assert cups, "expected at least one cup"
    amounts = _bar_amounts(g, cups)

    out = g.copy()
    for color, cup in cups.items():
        amount = amounts[color]
        order = _cavity_fill_order(cup)
        assert 1 <= amount <= len(order), "invalid fill amount for cup depth"
        for r, c in order[:amount]:
            assert int(out[r, c]) == bg
            out[r, c] = YELLOW
    return out


def _split_total(total: int, parts: int) -> list[int] | None:
    if parts <= 0 or total < parts or total > 6 * parts:
        return None
    remaining = total
    out: list[int] = []
    for i in range(parts):
        left = parts - i - 1
        lo = max(1, remaining - 6 * left)
        hi = min(6, remaining - left)
        if lo > hi:
            return None
        value = random.randint(lo, hi)
        out.append(value)
        remaining -= value
    random.shuffle(out)
    return out


def _random_bar_bundle(depth: int, patterns: list[str] | None = None) -> tuple[list[tuple[int, int]], int]:
    choices = patterns or ["single_teal", "double_teal", "mix", "mix3"]
    for _ in range(200):
        pattern = random.choice(choices)
        if pattern == "single_teal":
            fill = random.randint(1, depth)
            return [(TEAL, fill)], fill
        if pattern == "double_teal" and depth >= 2:
            total = random.randint(2, depth)
            parts = _split_total(total, 2)
            if parts is not None:
                return [(TEAL, parts[0]), (TEAL, parts[1])], total
        if pattern == "mix":
            teal_h = random.randint(2, 6)
            red_h = random.randint(1, teal_h - 1)
            fill = teal_h - red_h
            if fill <= depth:
                return [(TEAL, teal_h), (RED, red_h)], fill
        if pattern == "mix3":
            teal_total = random.randint(3, 10)
            red_h = random.randint(1, min(6, teal_total - 1))
            fill = teal_total - red_h
            teal_parts = _split_total(teal_total, 2)
            if teal_parts is not None and 1 <= fill <= depth:
                return [(TEAL, teal_parts[0]), (TEAL, teal_parts[1]), (RED, red_h)], fill
    raise RuntimeError("failed to sample bar bundle")


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(grid.shape[0], top + h + 1)
    cc1 = min(grid.shape[1], left + w + 1)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    grid[top : top + h, left : left + w][patch != 0] = patch[patch != 0]
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _add_random_padding(grid: np.ndarray) -> np.ndarray:
    used = np.argwhere(grid != 0)
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)
    cropped = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    max_extra_h = 30 - cropped.shape[0]
    max_extra_w = 30 - cropped.shape[1]
    extra_h = random.randint(0, min(5, max_extra_h))
    extra_w = random.randint(0, min(6, max_extra_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    bottom = extra_h - top
    right = extra_w - left

    out = np.zeros((cropped.shape[0] + top + bottom, cropped.shape[1] + left + right), dtype=int)
    out[top : top + cropped.shape[0], left : left + cropped.shape[1]] = cropped
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = np.zeros((TEMP_H, TEMP_W), dtype=int)
        occupied = np.zeros((TEMP_H, TEMP_W), dtype=bool)

        cup_count = random.choice([2, 3])
        cup_colors = random.sample(CUP_COLORS, cup_count)
        openings = random.sample(OPENINGS, cup_count)

        cup_specs = []
        for color, opening in zip(cup_colors, openings):
            depth = random.randint(3, 6)
            patch = np.where(_cup_mask(opening, depth), color, 0).astype(int)
            cup_specs.append({"color": color, "opening": opening, "depth": depth, "patch": patch})

        placed = []
        ok = True
        for spec in cup_specs:
            h, w = spec["patch"].shape
            choices = []
            for top in range(1, TEMP_H - h - 2):
                for left in range(1, LEFT_MAX_COL - w + 2):
                    choices.append((top, left))
            random.shuffle(choices)
            spot = None
            for top, left in choices:
                if _place_patch(grid, occupied, spec["patch"], top, left):
                    spot = (top, left)
                    break
            if spot is None:
                ok = False
                break
            spec["top"], spec["left"] = spot
            placed.append(spec)
        if not ok:
            continue

        bundles = []
        need_red_idx = random.randrange(cup_count)
        need_multi_idx = random.randrange(cup_count)
        for idx, spec in enumerate(placed):
            if idx == need_red_idx:
                patterns = ["mix", "mix3"]
            elif idx == need_multi_idx:
                patterns = ["double_teal", "mix", "mix3"]
            else:
                patterns = None
            bars, fill = _random_bar_bundle(spec["depth"], patterns)
            bundles.append({"color": spec["color"], "bars": bars, "fill": fill})

        total_bars = sum(len(bundle["bars"]) for bundle in bundles)
        if total_bars > 7:
            continue

        flat_bars: list[tuple[int, int, int]] = []
        for bundle in bundles:
            for bar_color, height in bundle["bars"]:
                flat_bars.append((bundle["color"], bar_color, height))
        random.shuffle(flat_bars)

        cup_right = max(spec["left"] + spec["patch"].shape[1] - 1 for spec in placed)
        width_needed = 2 * len(flat_bars) - 1
        min_start = max(15, cup_right + 4)
        max_start = TEMP_W - width_needed - 1
        if min_start > max_start:
            continue
        start_col = random.randint(min_start, max_start)
        baseline = random.randint(*BAR_BASELINE_RANGE)

        for idx, (label_color, bar_color, height) in enumerate(flat_bars):
            col = start_col + 2 * idx
            top = baseline - height + 1
            if top < 1 or baseline + 1 >= TEMP_H:
                ok = False
                break
            if np.any(grid[top : baseline + 2, col] != 0):
                ok = False
                break
            grid[top : baseline + 1, col] = bar_color
            grid[baseline + 1, col] = label_color
        if not ok:
            continue

        final = _add_random_padding(grid)
        out = main(final)
        expected_fill = sum(bundle["fill"] for bundle in bundles)
        if int(np.count_nonzero(out == YELLOW)) != expected_fill:
            continue
        if np.array_equal(final, out):
            continue
        return final

    raise RuntimeError("failed to generate a valid input")
