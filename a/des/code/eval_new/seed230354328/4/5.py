from __future__ import annotations

import random

import numpy as np


FOLD_COLOR = 9
ALL_COLORS = tuple(range(10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 80
WRAP_OVERLAY_PROB = 0.45
ACCENT_PROB = 0.35
MAX_FOLDS = 3
MAX_STRIP = 4
SIDE_PRIORITY = {"top": 0, "left": 1, "bottom": 2, "right": 3}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _full_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        value = int(g[r, 0])
        if bool(np.all(g[r, :] == value)):
            rows.append((int(r), int(value)))
    return rows


def _full_cols(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    cols: list[tuple[int, int]] = []
    for c in range(g.shape[1]):
        value = int(g[0, c])
        if bool(np.all(g[:, c] == value)):
            cols.append((int(c), int(value)))
    return cols


def _detect_fold_color(grid: np.ndarray, bg: int) -> int | None:
    counts: dict[int, int] = {}
    for _r, value in _full_rows(grid):
        if value != int(bg):
            counts[int(value)] = counts.get(int(value), 0) + 1
    for _c, value in _full_cols(grid):
        if value != int(bg):
            counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return None
    best_count = max(counts.values())
    best_colors = [color for color, count in counts.items() if count == best_count]
    return int(min(best_colors))


def _current_fold_lines(grid: np.ndarray, bg: int, fold_color: int) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    lines: list[dict[str, int | str]] = []

    for r, value in _full_rows(g):
        if int(value) != int(fold_color):
            continue
        top_dist = int(r)
        bottom_dist = int(h - 1 - r)
        if top_dist <= bottom_dist:
            side = "top"
            distance = top_dist
        else:
            side = "bottom"
            distance = bottom_dist
        lines.append(
            {
                "orientation": "H",
                "index": int(r),
                "side": side,
                "distance": int(distance),
            }
        )

    for c, value in _full_cols(g):
        if int(value) != int(fold_color):
            continue
        left_dist = int(c)
        right_dist = int(w - 1 - c)
        if left_dist <= right_dist:
            side = "left"
            distance = left_dist
        else:
            side = "right"
            distance = right_dist
        lines.append(
            {
                "orientation": "V",
                "index": int(c),
                "side": side,
                "distance": int(distance),
            }
        )

    lines.sort(
        key=lambda item: (
            int(item["distance"]),
            SIDE_PRIORITY[str(item["side"])],
            0 if str(item["orientation"]) == "H" else 1,
            int(item["index"]),
        )
    )
    return lines


def _overlay_with_priority(base: np.ndarray, overlay: np.ndarray, bg: int) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    over = np.asarray(overlay, dtype=int)
    mask = over != int(bg)
    out[mask] = over[mask]
    return out


def _fold_once(grid: np.ndarray, bg: int, fold_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    lines = _current_fold_lines(g, bg, fold_color)
    assert lines, "expected at least one full fold line"
    line = lines[0]
    side = str(line["side"])
    index = int(line["index"])

    if str(line["orientation"]) == "H":
        if side == "top":
            outer = g[:index, :]
            keep = g[index + 1 :, :].copy()
            assert keep.size > 0
            assert outer.shape[0] <= keep.shape[0]
            if outer.size:
                folded = np.flipud(outer)
                keep[: outer.shape[0], :] = _overlay_with_priority(
                    keep[: outer.shape[0], :], folded, bg
                )
            return keep

        outer = g[index + 1 :, :]
        keep = g[:index, :].copy()
        assert keep.size > 0
        assert outer.shape[0] <= keep.shape[0]
        if outer.size:
            folded = np.flipud(outer)
            keep[-outer.shape[0] :, :] = _overlay_with_priority(
                keep[-outer.shape[0] :, :], folded, bg
            )
        return keep

    if side == "left":
        outer = g[:, :index]
        keep = g[:, index + 1 :].copy()
        assert keep.size > 0
        assert outer.shape[1] <= keep.shape[1]
        if outer.size:
            folded = np.fliplr(outer)
            keep[:, : outer.shape[1]] = _overlay_with_priority(
                keep[:, : outer.shape[1]], folded, bg
            )
        return keep

    outer = g[:, index + 1 :]
    keep = g[:, :index].copy()
    assert keep.size > 0
    assert outer.shape[1] <= keep.shape[1]
    if outer.size:
        folded = np.fliplr(outer)
        keep[:, -outer.shape[1] :] = _overlay_with_priority(
            keep[:, -outer.shape[1] :], folded, bg
        )
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the grid inward along full maroon guide rows/columns and keep the folded overlay.

    Input structure:
    - bg = most common color.
    - Several colored patterns sit inside the grid on the background.
    - One or more full maroon rows or columns appear near the outer boundary; each such line acts as a fold marker.

    Rule:
    - Detect the current full maroon row/column that is closest to a grid edge; that edge is the side that gets folded inward.
    - Remove the marker line itself, mirror the cells on the outer side across that line, and lay them onto the remaining inner region.
    - Folded non-background pixels override whatever was already underneath; background pixels do nothing.
    - Repeat from outermost to innermost until no full maroon fold line remains.

    Key insight:
    - The maroon bands are instructions rather than picture content: each one tells which side of the grid to flip inward, and nested lines must be processed in order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2

    bg = _get_bg(g)
    out = g.copy()
    steps = 0
    while True:
        fold_color = _detect_fold_color(out, bg)
        if fold_color is None:
            break
        out = _fold_once(out, bg, fold_color)
        steps += 1
        assert steps <= 6, "expected only a small number of nested folds"
    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return sorted((int(r - r0), int(c - c0)) for r, c in cells)


def _transform_shape(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = np.array(cells, dtype=int)
    for _ in range(random.randint(0, 3)):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    if random.random() < 0.5:
        pts[:, 0] *= -1
    if random.random() < 0.5:
        pts[:, 1] *= -1
    cell_set = {(int(r), int(c)) for r, c in pts.tolist()}
    return _normalize(cell_set)


def _random_shape() -> list[tuple[int, int]] | None:
    area = random.randint(2, 6)
    for _ in range(40):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in NEIGHBORS4:
                nxt = (int(cell[0] + dr), int(cell[1] + dc))
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != area:
            continue
        shaped = _transform_shape(_normalize(cells))
        h = max(r for r, _c in shaped) + 1
        w = max(c for _r, c in shaped) + 1
        if h <= 4 and w <= 4:
            return shaped
    return None


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _has_unexpected_full_lines(grid: np.ndarray, bg: int) -> bool:
    for _r, value in _full_rows(grid):
        if int(value) != int(bg) and int(value) != int(FOLD_COLOR):
            return True
    for _c, value in _full_cols(grid):
        if int(value) != int(bg) and int(value) != int(FOLD_COLOR):
            return True
    return False


def _make_core_scene(bg: int, palette: list[int]) -> np.ndarray | None:
    h = random.randint(7, 12)
    w = random.randint(7, 12)
    grid = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    object_count = random.randint(4, 7)
    placed = 0
    available_colors = palette.copy()
    random.shuffle(available_colors)

    for _ in range(object_count):
        shape = _random_shape()
        if shape is None:
            continue
        sh = max(r for r, _c in shape) + 1
        sw = max(c for _r, c in shape) + 1
        if sh > h or sw > w:
            continue
        color = available_colors[placed % len(available_colors)]

        success = False
        for _ in range(OBJECT_ATTEMPTS):
            r0 = random.randint(0, h - sh)
            c0 = random.randint(0, w - sw)
            cells = [(int(r0 + r), int(c0 + c)) for r, c in shape]
            if any(bool(blocked[r, c]) for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = int(color)

            if random.random() < ACCENT_PROB and len(cells) >= 3:
                accent_choices = [v for v in palette if v != int(color)]
                accent = random.choice(accent_choices)
                accent_count = 1 if len(cells) == 3 else random.randint(1, 2)
                for r, c in random.sample(cells, accent_count):
                    grid[r, c] = int(accent)

            _mark_blocked(blocked, cells)
            placed += 1
            success = True
            break
        if not success:
            continue

    if placed < 4:
        return None
    if _has_unexpected_full_lines(grid, bg):
        return None
    if int(np.count_nonzero(grid != int(bg))) < 10:
        return None
    return grid


def _eligible_positions(
    target: np.ndarray, side: str, strip: int, bg: int, fold_color: int
) -> list[tuple[int, int]]:
    h, w = target.shape
    cells: list[tuple[int, int]] = []
    if side == "top":
        for r in range(strip):
            for c in range(w):
                value = int(target[r, c])
                if value not in {int(bg), int(fold_color)}:
                    cells.append((int(r), int(c)))
    elif side == "bottom":
        for r in range(h - strip, h):
            for c in range(w):
                value = int(target[r, c])
                if value not in {int(bg), int(fold_color)}:
                    cells.append((int(r), int(c)))
    elif side == "left":
        for r in range(h):
            for c in range(strip):
                value = int(target[r, c])
                if value not in {int(bg), int(fold_color)}:
                    cells.append((int(r), int(c)))
    else:
        for r in range(h):
            for c in range(w - strip, w):
                value = int(target[r, c])
                if value not in {int(bg), int(fold_color)}:
                    cells.append((int(r), int(c)))
    return cells


def _choose_overlay_subset(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    selected = [cell for cell in cells if random.random() < WRAP_OVERLAY_PROB]
    if not selected:
        selected = [random.choice(cells)]
    random.shuffle(selected)
    cap = max(1, min(len(cells), 8))
    return selected[:cap]


def _conflict_value(original: int, bg: int, palette: list[int]) -> int:
    options = [value for value in palette if value != int(original)]
    if random.random() < 0.5:
        return int(bg)
    return int(random.choice(options))


def _wrap_once(
    target: np.ndarray,
    side: str,
    strip: int,
    bg: int,
    fold_color: int,
    palette: list[int],
    eligible: list[tuple[int, int]],
) -> np.ndarray | None:
    h, w = target.shape
    keep = target.copy()
    selected = _choose_overlay_subset(eligible)

    if side == "top":
        out = np.full((h + strip + 1, w), int(bg), dtype=int)
        for r, c in selected:
            out[int(strip - 1 - r), int(c)] = int(target[r, c])
            keep[r, c] = _conflict_value(int(target[r, c]), bg, palette)
        out[strip + 1 :, :] = keep
        out[strip, :] = int(fold_color)
    elif side == "bottom":
        out = np.full((h + strip + 1, w), int(bg), dtype=int)
        for r, c in selected:
            outer_r = int(h - 1 - r)
            out[h + 1 + outer_r, c] = int(target[r, c])
            keep[r, c] = _conflict_value(int(target[r, c]), bg, palette)
        out[:h, :] = keep
        out[h, :] = int(fold_color)
    elif side == "left":
        out = np.full((h, w + strip + 1), int(bg), dtype=int)
        for r, c in selected:
            out[r, int(strip - 1 - c)] = int(target[r, c])
            keep[r, c] = _conflict_value(int(target[r, c]), bg, palette)
        out[:, strip + 1 :] = keep
        out[:, strip] = int(fold_color)
    else:
        out = np.full((h, w + strip + 1), int(bg), dtype=int)
        for r, c in selected:
            outer_c = int(w - 1 - c)
            out[r, w + 1 + outer_c] = int(target[r, c])
            keep[r, c] = _conflict_value(int(target[r, c]), bg, palette)
        out[:, :w] = keep
        out[:, w] = int(fold_color)

    if _has_unexpected_full_lines(out, bg):
        return None

    try:
        folded = _fold_once(out, bg, fold_color)
    except AssertionError:
        return None
    if not np.array_equal(folded, target):
        return None
    return out


def _build_input() -> np.ndarray | None:
    bg_choices = [color for color in ALL_COLORS if color != FOLD_COLOR]
    bg = random.choice(bg_choices)
    palette = [color for color in ALL_COLORS if color not in {bg, FOLD_COLOR}]

    core = _make_core_scene(bg, palette)
    if core is None:
        return None

    grid = core
    folds = random.randint(1, MAX_FOLDS)
    orientation = random.choice(("H", "V"))

    for _ in range(folds):
        side_options = ("top", "bottom") if orientation == "H" else ("left", "right")
        options: list[tuple[str, int, list[tuple[int, int]]]] = []
        for side in side_options:
            dim = grid.shape[0] if side in {"top", "bottom"} else grid.shape[1]
            size_limit = 30 - dim - 1
            if size_limit < 1:
                continue
            max_strip = min(MAX_STRIP, dim - 1, size_limit)
            for strip in range(1, max_strip + 1):
                eligible = _eligible_positions(grid, side, strip, bg, FOLD_COLOR)
                if eligible:
                    options.append((side, int(strip), eligible))
        if not options:
            return None

        side, strip, eligible = random.choice(options)
        wrapped = _wrap_once(grid, side, strip, bg, FOLD_COLOR, palette, eligible)
        if wrapped is None:
            return None
        grid = wrapped
        orientation = "V" if orientation == "H" else "H"

    if _get_bg(grid) != int(bg):
        return None
    if _has_unexpected_full_lines(grid, bg):
        return None
    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-overlay input")
