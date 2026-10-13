from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import Color, colors_present, find_components, get_bg
except ModuleNotFoundError:
    SOLVER_DIR = Path(__file__).resolve().parents[6] / "new" / "solvers"
    if str(SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(SOLVER_DIR))
    from arc_utils import Color, colors_present, find_components, get_bg


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    r1: int
    c1: int


def _rect_border_mask(shape: tuple[int, int], r: Rect) -> np.ndarray:
    H, W = map(int, shape)
    mask = np.zeros((H, W), dtype=bool)
    mask[r.r0, r.c0 : r.c1 + 1] = True
    mask[r.r1, r.c0 : r.c1 + 1] = True
    mask[r.r0 : r.r1 + 1, r.c0] = True
    mask[r.r0 : r.r1 + 1, r.c1] = True
    return mask


def _strictly_inside(inner: Rect, outer: Rect) -> bool:
    return (
        outer.r0 < inner.r0 < inner.r1 < outer.r1
        and outer.c0 < inner.c0 < inner.c1 < outer.c1
    )


def _rect_area(r: Rect) -> int:
    return int(r.r1 - r.r0 + 1) * int(r.c1 - r.c0 + 1)


def _complete_rect_from_three_points(points: np.ndarray) -> Rect:
    rows = Counter(map(int, points[:, 0]))
    cols = Counter(map(int, points[:, 1]))

    elbows = [
        (int(r), int(c))
        for r, c in points
        if rows[int(r)] == 2 and cols[int(c)] == 2
    ]
    assert len(elbows) == 1, "expected one elbow pixel for each color"

    elbow_row, elbow_col = elbows[0]
    missing_rows = [r for r, n in rows.items() if n == 1]
    missing_cols = [c for c, n in cols.items() if n == 1]
    assert len(missing_rows) == 1, "expected exactly one missing row"
    assert len(missing_cols) == 1, "expected exactly one missing col"

    missing_row = missing_rows[0]
    missing_col = missing_cols[0]
    rect_r0 = min(elbow_row, missing_row)
    rect_r1 = max(elbow_row, missing_row)
    rect_c0 = min(elbow_col, missing_col)
    rect_c1 = max(elbow_col, missing_col)

    all_points = {
        (elbow_row, elbow_col),
        (int(missing_row), elbow_col),
        (elbow_row, int(missing_col)),
        (int(missing_row), int(missing_col)),
    }
    present = set(map(tuple, points.tolist()))
    expected = all_points - {(missing_row, missing_col)}
    assert present == expected, "input corners must match the inferred rectangle corners"

    return Rect(rect_r0, rect_c0, rect_r1, rect_c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every color's three missing-rectangle corners, then apply crossing and nesting effects.

    Input structure:
    - Background is dominant; every foreground color appears in exactly three isolated pixels.
    - Those three pixels are the three corners of one axis-aligned rectangle, with one pixel sharing its row and column with the other two.
    - Yellow is absent in input and is reserved for crossing marks.

    Rule:
    - Infer each color's fourth corner and draw the full hollow rectangle border using that color.
    - If two different rectangle borders occupy the same cell, paint that overlap cell yellow.
    - If one rectangle is strictly inside another, fill the inner rectangle's interior (excluding its border) with the outer rectangle's color.

    Key insight:
    - The border completion must be done first; only then do crossing and strict-nesting relations become visible.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    colors = colors_present(g, exclude={bg})

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    rects: list[tuple[int, Rect, np.ndarray]] = []
    out = g.copy()

    for color in colors:
        group = by_color.get(color, [])
        assert group, "each foreground color should appear"
        total = int(sum(int(c["area"]) for c in group))
        assert total == 3, "each color must have exactly three pixels"

        pts = np.concatenate([c["coords"] for c in group], axis=0)
        rect = _complete_rect_from_three_points(pts)
        mask = _rect_border_mask(g.shape, rect)
        out[mask] = color
        rects.append((color, rect, mask))

    # Crossing borders -> yellow.
    for i in range(len(rects)):
        ci, _, mi = rects[i]
        for j in range(i + 1, len(rects)):
            cj, _, mj = rects[j]
            assert ci != cj, "foreground colors must be unique"
            overlap = mi & mj
            if np.any(overlap):
                out[overlap] = int(Color.YELLOW)

    # Strict nesting fill.
    for _, inner_rect, _ in rects:
        if inner_rect.r1 - inner_rect.r0 <= 1 or inner_rect.c1 - inner_rect.c0 <= 1:
            continue

        enclosing = [
            ( _rect_area(outer_rect), outer_color, outer_rect)
            for outer_color, outer_rect, _ in rects
            if outer_color != color and _strictly_inside(inner_rect, outer_rect)
        ]

        if not enclosing:
            continue

        # Use the nearest outer rectangle (smallest enclosing area).
        _, outer_color, outer_rect = min(enclosing, key=lambda t: t[0])
        ri0, ci0, ri1, ci1 = inner_rect.r0, inner_rect.c0, inner_rect.r1, inner_rect.c1
        out[ri0 + 1 : ri1, ci0 + 1 : ci1] = int(outer_color)

    return out


GEN_BACKGROUND = (0, 1, 2, 3, 5, 6, 7, 9)
GEN_H_RANGE = (14, 30)
GEN_W_RANGE = (14, 30)
GEN_RECT_MIN = 3
GEN_RECT_MAX = 6
GEN_ATTEMPTS = 500
GEN_MODES = ["base", "base", "crossing", "nested", "both"]


def _palette(bg: int, n: int) -> list[int]:
    candidates = [c for c in range(1, 10) if c != bg and c != Color.YELLOW]
    if n > len(candidates):
        raise RuntimeError("not enough colors")
    return random.sample(candidates, n)


def _candidate_modes() -> str:
    return random.choice(GEN_MODES)


def _place_masked(shape: tuple[int, int], border_mask: np.ndarray, occupied: np.ndarray) -> bool:
    return not np.any(border_mask & occupied)


def _place_independent(shape: tuple[int, int], occupied: np.ndarray) -> Rect | None:
    H, W = map(int, shape)

    for _ in range(400):
        h = random.randint(3, min(9, H - 2))
        w = random.randint(3, min(9, W - 2))
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        rect = Rect(r0, c0, r0 + h - 1, c0 + w - 1)
        mask = _rect_border_mask(shape, rect)
        if _place_masked(shape, mask, occupied):
            return rect
    return None


def _place_nested(shape: tuple[int, int], occupied: np.ndarray, rects: list[Rect]) -> Rect | None:
    candidates = [r for r in rects if (r.r1 - r.r0) >= 5 and (r.c1 - r.c0) >= 5]
    random.shuffle(candidates)
    if not candidates:
        return None

    H, W = map(int, shape)
    for outer in candidates:
        max_h = outer.r1 - outer.r0 - 1
        max_w = outer.c1 - outer.c0 - 1
        if max_h < 3 or max_w < 3:
            continue

        for _ in range(300):
            ih = random.randint(3, min(8, max_h))
            iw = random.randint(3, min(8, max_w))
            r0 = random.randint(outer.r0 + 1, outer.r1 - ih)
            c0 = random.randint(outer.c0 + 1, outer.c1 - iw)
            inner = Rect(r0, c0, r0 + ih - 1, c0 + iw - 1)

            # keep inside outer interior.
            if not _strictly_inside(inner, outer):
                continue

            mask = _rect_border_mask(shape, inner)
            if np.any(mask & occupied):
                continue
            return inner
    return None


def _place_crossing(shape: tuple[int, int], occupied: np.ndarray, rects: list[Rect], border_masks: list[np.ndarray]) -> Rect | None:
    if not rects:
        return None

    H, W = map(int, shape)
    candidates = [r for r in rects if (r.r1 - r.r0) >= 3 and (r.c1 - r.c0) >= 2]
    random.shuffle(candidates)

    for target in candidates:
        target_idx = rects.index(target)
        target_mask = border_masks[target_idx]

        anchor_rows = []
        if target.r0 > 0:
            anchor_rows.append(target.r0)
        if target.r1 + 1 < H:
            anchor_rows.append(target.r1)

        for anchor in anchor_rows:
            if anchor == target.r0:
                r0, r1 = target.r0 - 1, target.r0
            else:
                r0, r1 = target.r1, target.r1 + 1

            column_pairs = []
            if target.c0 > 0:
                column_pairs.append((target.c0 - 1, target.c0))
            if target.c1 + 1 < W:
                column_pairs.append((target.c1, target.c1 + 1))

            for c0, c1 in column_pairs:
                cand = Rect(r0, c0, r1, c1)
                cand_mask = _rect_border_mask(shape, cand)

                overlap_counts = [int(np.sum(cand_mask & m)) for m in border_masks]
                if overlap_counts[target_idx] != 1:
                    continue
                if any(i != target_idx and count > 0 for i, count in enumerate(overlap_counts)):
                    continue
                if np.any(cand_mask & occupied & ~target_mask):
                    continue

                cell = np.argwhere(cand_mask & target_mask)
                if len(cell) != 1:
                    continue
                rr, cc = map(int, cell[0])
                if not (rr == anchor and (cc == target.c0 or cc == target.c1)):
                    continue
                return cand

    return None

def _build_input(shape: tuple[int, int], rects: list[Rect], colors: list[int], bg: int) -> np.ndarray:
    H, W = map(int, shape)
    grid = np.full((H, W), int(bg), dtype=int)
    for rect, color in zip(rects, colors):
        corners = [
            (rect.r0, rect.c0),
            (rect.r0, rect.c1),
            (rect.r1, rect.c0),
            (rect.r1, rect.c1),
        ]
        elbow_idx = random.randint(0, 3)
        elbow = corners[elbow_idx]
        diag = corners[(elbow_idx + 2) % 4]
        for r, c in corners:
            if (r, c) != diag:
                grid[r, c] = color

    return grid


def _relation_counts(shape: tuple[int, int], rects: list[Rect]) -> tuple[int, int]:
    masks = [_rect_border_mask(shape, rect) for rect in rects]
    crossings = 0
    nested = 0

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if np.any(masks[i] & masks[j]):
                crossings += 1
            if _strictly_inside(rects[i], rects[j]):
                nested += 1
            if _strictly_inside(rects[j], rects[i]):
                nested += 1

    return crossings, nested


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        H = random.randint(*GEN_H_RANGE)
        W = random.randint(*GEN_W_RANGE)
        mode = _candidate_modes()

        target_n = random.randint(GEN_RECT_MIN, GEN_RECT_MAX)
        if mode == "both":
            target_n = max(4, target_n)

        rects: list[Rect] = []
        occupied = np.zeros((H, W), dtype=bool)

        # Seed with at least two separated independent rectangles.
        first = _place_independent((H, W), occupied)
        if first is None:
            continue
        rects.append(first)
        occupied |= _rect_border_mask((H, W), first)

        second = _place_independent((H, W), occupied)
        if second is None:
            continue
        rects.append(second)
        occupied |= _rect_border_mask((H, W), second)

        if mode in ("nested", "both"):
            nested = _place_nested((H, W), occupied, rects)
            if nested is None:
                continue
            rects.append(nested)
            occupied |= _rect_border_mask((H, W), nested)

        masks = [_rect_border_mask((H, W), r) for r in rects]

        if mode in ("crossing", "both"):
            crossing = _place_crossing((H, W), occupied, rects, masks)
            if crossing is None:
                continue
            rects.append(crossing)
            occupied |= _rect_border_mask((H, W), crossing)
            masks.append(_rect_border_mask((H, W), crossing))

        while len(rects) < target_n:
            cand = _place_independent((H, W), occupied)
            if cand is None:
                break
            rects.append(cand)
            m = _rect_border_mask((H, W), cand)
            occupied |= m
            masks.append(m)

        if len(rects) != target_n:
            continue

        crossings, nested_count = _relation_counts((H, W), rects)
        if mode == "nested" and nested_count == 0:
            continue
        if mode == "crossing" and crossings == 0:
            continue
        if mode == "both" and (crossings == 0 or nested_count == 0):
            continue

        bg = random.choice(GEN_BACKGROUND)
        colors = _palette(bg, len(rects))

        grid = _build_input((H, W), rects, colors, bg)

        # Self-validate for main() assumptions.
        try:
            _ = main(grid)
            colors_in_input = colors_present(grid, exclude={bg})
            for c in colors_in_input:
                assert int((grid == c).sum()) == 3
            assert Color.YELLOW not in set(map(int, np.unique(grid)))
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 6")
