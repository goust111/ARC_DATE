from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GRAY = 5
ORANGE = 7

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
HORIZONTAL_EDGES = ("top", "bottom")
EDGES = ("top", "bottom", "left", "right")


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    seen = np.zeros(grid.shape, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < grid.shape[0]
                        and 0 <= nc < grid.shape[1]
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _orientation(
    coords: np.ndarray,
    contacts: list[tuple[int, int, int, int]],
) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == 1 or w == 1, "gray source must be one cell thick"
    if h == 1 and w > 1:
        return "horizontal"
    if w == 1 and h > 1:
        return "vertical"

    implied = set()
    for tr, tc, sr, sc in contacts:
        if tr == sr:
            implied.add("horizontal")
        elif tc == sc:
            implied.add("vertical")
        else:
            raise AssertionError("gray source must touch orange by 4-neighbor side")
    assert len(implied) == 1, "singleton gray bar has ambiguous orientation"
    return implied.pop()


def _fill_rows(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    gray_count: int,
    contact_rows: set[int],
) -> None:
    r0, c0, r1, c1 = rect
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    assert gray_count % width == 0
    stripe_count = gray_count // width
    assert 0 < stripe_count <= height
    assert contact_rows in ({r0}, {r1})
    if contact_rows == {r0}:
        out[r0 : r0 + stripe_count, c0 : c1 + 1] = GRAY
    else:
        out[r1 - stripe_count + 1 : r1 + 1, c0 : c1 + 1] = GRAY


def _fill_cols(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    gray_count: int,
    contact_cols: set[int],
) -> None:
    r0, c0, r1, c1 = rect
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    assert gray_count % height == 0
    stripe_count = gray_count // height
    assert 0 < stripe_count <= width
    assert contact_cols in ({c0}, {c1})
    if contact_cols == {c0}:
        out[r0 : r1 + 1, c0 : c0 + stripe_count] = GRAY
    else:
        out[r0 : r1 + 1, c1 - stripe_count + 1 : c1 + 1] = GRAY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gray bar into its adjacent orange rectangle as full stripes from the contacted edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, YELLOW, GRAY, ORANGE}

    orange_parts = _components_of_color(g, ORANGE)
    gray_parts = _components_of_color(g, GRAY)
    assert orange_parts and gray_parts

    orange_rects: list[tuple[int, int, int, int]] = []
    orange_index: dict[tuple[int, int], int] = {}
    for i, coords in enumerate(orange_parts):
        r0, c0, r1, c1 = _bbox(coords)
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "orange object must be solid rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == ORANGE)
        orange_rects.append((r0, c0, r1, c1))
        for r, c in coords:
            orange_index[(int(r), int(c))] = i

    grouped: dict[int, list[tuple[np.ndarray, list[tuple[int, int, int, int]]]]] = {}
    for coords in gray_parts:
        touched: dict[int, list[tuple[int, int, int, int]]] = {}
        for sr, sc in coords:
            for dr, dc in DIR4:
                tr, tc = int(sr + dr), int(sc + dc)
                target_idx = orange_index.get((tr, tc))
                if target_idx is not None:
                    touched.setdefault(target_idx, []).append((tr, tc, int(sr), int(sc)))
        assert len(touched) == 1, "each gray bar must touch exactly one orange rectangle"
        target_idx = next(iter(touched))
        grouped.setdefault(target_idx, []).append((coords, touched[target_idx]))

    out = g.copy()
    out[g == GRAY] = YELLOW

    for idx, sources in grouped.items():
        rect = orange_rects[idx]
        orientations = {_orientation(coords, contacts) for coords, contacts in sources}
        assert len(orientations) == 1, "gray bars for one rectangle must share orientation"
        total_gray = sum(int(coords.shape[0]) for coords, _ in sources)
        contacts = [contact for _coords, source_contacts in sources for contact in source_contacts]
        if orientations == {"horizontal"}:
            contact_rows = {tr for tr, _tc, _sr, _sc in contacts}
            _fill_rows(out, rect, total_gray, contact_rows)
        else:
            contact_cols = {tc for _tr, tc, _sr, _sc in contacts}
            _fill_cols(out, rect, total_gray, contact_cols)

    return out


GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (12, 18)
CONTENT_H_RANGE = (7, 14)
CONTENT_W_RANGE = (8, 15)
BORDER_RANGE = (0, 3)

RECT_COUNT_CHOICES = (1, 2, 3, 4)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 6)
RECT_AREA_RANGE = (4, 18)
PLACE_RETRIES = 80

UNTOUCHED_PROB = 0.35
BARS_PER_RECT_CHOICES = (1, 2)
GEN_RETRIES = 400


def _make_canvas() -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    content_h = random.randint(*CONTENT_H_RANGE)
    content_w = random.randint(*CONTENT_W_RANGE)
    top = random.randint(*BORDER_RANGE)
    bottom = random.randint(*BORDER_RANGE)
    left = random.randint(*BORDER_RANGE)
    right = random.randint(*BORDER_RANGE)
    grid_h = content_h + top + bottom
    grid_w = content_w + left + right
    if not (GRID_H_RANGE[0] <= grid_h <= GRID_H_RANGE[1]):
        return None
    if not (GRID_W_RANGE[0] <= grid_w <= GRID_W_RANGE[1]):
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    cr0, cc0 = top, left
    cr1, cc1 = top + content_h - 1, left + content_w - 1
    grid[cr0 : cr1 + 1, cc0 : cc1 + 1] = YELLOW
    return grid, (cr0, cc0, cr1, cc1)


def _place_rectangle(
    edge: str,
    box: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> tuple[int, int, int, int] | None:
    cr0, cc0, cr1, cc1 = box
    for _ in range(PLACE_RETRIES):
        h = random.randint(*RECT_H_RANGE)
        w = random.randint(*RECT_W_RANGE)
        if not (RECT_AREA_RANGE[0] <= h * w <= RECT_AREA_RANGE[1]):
            continue
        if edge in HORIZONTAL_EDGES:
            r0 = cr0 if edge == "top" else cr1 - h + 1
            r1 = r0 + h - 1
            lo = cc0 + 1
            hi = cc1 - w
            if hi < lo:
                continue
            c0 = random.randint(lo, hi)
            c1 = c0 + w - 1
        else:
            c0 = cc0 if edge == "left" else cc1 - w + 1
            c1 = c0 + w - 1
            lo = cr0 + 1
            hi = cr1 - h
            if hi < lo:
                continue
            r0 = random.randint(lo, hi)
            r1 = r0 + h - 1
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return (r0, c0, r1, c1)
    return None


def _place_rectangles(
    box: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> list[tuple[str, int, int, int, int]] | None:
    count = random.choice(RECT_COUNT_CHOICES)
    edges = random.sample(EDGES, count)
    rects: list[tuple[str, int, int, int, int]] = []
    for edge in edges:
        rect = _place_rectangle(edge, box, occupied)
        if rect is None:
            return None
        r0, c0, r1, c1 = rect
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        rects.append((edge, r0, c0, r1, c1))
    return rects


def _choose_touched(rect_count: int) -> list[int]:
    idxs = list(range(rect_count))
    touched = [i for i in idxs if random.random() > UNTOUCHED_PROB]
    if not touched:
        touched = [random.choice(idxs)]
    if len(touched) == rect_count and rect_count > 1 and random.random() < UNTOUCHED_PROB:
        touched.remove(random.choice(touched))
    return touched


def _split_total(total: int, n_parts: int) -> list[int] | None:
    if total < n_parts:
        return None
    if n_parts == 1:
        return [total]
    cut = random.randint(1, total - 1)
    return [cut, total - cut]


def _place_bars(
    axis: str,
    fixed: int,
    span0: int,
    span1: int,
    lo_bound: int,
    hi_bound: int,
    lengths: list[int],
    occupied: np.ndarray,
) -> list[tuple[str, int, int, int]] | None:
    placed: list[tuple[str, int, int, int]] = []
    for length in lengths:
        choices = []
        if span0 - lo_bound >= length:
            choices.append("low")
        if hi_bound - span1 >= length:
            choices.append("high")
        random.shuffle(choices)
        done = False
        for side in choices:
            start = span0 - length if side == "low" else span1 + 1
            end = start + length - 1
            if axis == "h":
                cells = occupied[fixed, start : end + 1]
            else:
                cells = occupied[start : end + 1, fixed]
            if cells.any():
                continue
            if axis == "h":
                occupied[fixed, start : end + 1] = True
                placed.append(("h", fixed, start, end))
            else:
                occupied[start : end + 1, fixed] = True
                placed.append(("v", start, end, fixed))
            done = True
            break
        if not done:
            return None
    return placed


def _plan_bars(
    rects: list[tuple[str, int, int, int, int]],
    touched: list[int],
    box: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> list[tuple[int, list[tuple[str, int, int, int]], str, int]] | None:
    cr0, cc0, cr1, cc1 = box
    info: list[tuple[int, list[tuple[str, int, int, int]], str, int]] = []
    for idx in touched:
        edge, r0, c0, r1, c1 = rects[idx]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if edge in HORIZONTAL_EDGES:
            stripe_cap = height
            stripe_width = width
            available = (c0 - cc0) + (cc1 - c1)
            if available < stripe_width:
                return None
            stripe_count = random.randint(1, stripe_cap)
            total_gray = stripe_count * stripe_width
            n_bars = 2 if total_gray >= 2 and available >= total_gray and random.random() < 0.5 else 1
            lengths = _split_total(total_gray, n_bars)
            if lengths is None:
                return None
            bars = _place_bars("h", r0 if edge == "top" else r1, c0, c1, cc0, cc1, lengths, occupied)
        else:
            stripe_cap = width
            stripe_height = height
            available = (r0 - cr0) + (cr1 - r1)
            if available < stripe_height:
                return None
            stripe_count = random.randint(1, stripe_cap)
            total_gray = stripe_count * stripe_height
            n_bars = 2 if total_gray >= 2 and available >= total_gray and random.random() < 0.5 else 1
            lengths = _split_total(total_gray, n_bars)
            if lengths is None:
                return None
            bars = _place_bars("v", c0 if edge == "left" else c1, r0, r1, cr0, cr1, lengths, occupied)
        if bars is None:
            return None
        info.append((idx, bars, edge, stripe_count))
    return info


def _render_input(
    grid: np.ndarray,
    rects: list[tuple[str, int, int, int, int]],
    bars_info: list[tuple[int, list[tuple[str, int, int, int]], str, int]],
) -> None:
    for _edge, r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
    for _idx, bars, _edge, _count in bars_info:
        for bar in bars:
            if bar[0] == "h":
                grid[bar[1], bar[2] : bar[3] + 1] = GRAY
            else:
                grid[bar[1] : bar[2] + 1, bar[3]] = GRAY


def _expected_output(
    grid: np.ndarray,
    rects: list[tuple[str, int, int, int, int]],
    bars_info: list[tuple[int, list[tuple[str, int, int, int]], str, int]],
) -> np.ndarray:
    out = grid.copy()
    out[grid == GRAY] = YELLOW
    for idx, _bars, edge, stripe_count in bars_info:
        _edge, r0, c0, r1, c1 = rects[idx]
        if edge == "top":
            out[r0 : r0 + stripe_count, c0 : c1 + 1] = GRAY
        elif edge == "bottom":
            out[r1 - stripe_count + 1 : r1 + 1, c0 : c1 + 1] = GRAY
        elif edge == "left":
            out[r0 : r1 + 1, c0 : c0 + stripe_count] = GRAY
        else:
            out[r0 : r1 + 1, c1 - stripe_count + 1 : c1 + 1] = GRAY
    return out


def _gen_attempt() -> np.ndarray | None:
    made = _make_canvas()
    if made is None:
        return None
    grid, box = made
    occupied = np.zeros(grid.shape, dtype=bool)

    rects = _place_rectangles(box, occupied)
    if rects is None:
        return None
    touched = _choose_touched(len(rects))
    bars_info = _plan_bars(rects, touched, box, occupied)
    if bars_info is None:
        return None

    _render_input(grid, rects, bars_info)
    expected = _expected_output(grid, rects, bars_info)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
