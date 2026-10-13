from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for r in range(1, grid.shape[0] - 1):
        vals = np.unique(grid[r, :])
        if vals.size == 1 and int(vals[0]) != int(bg):
            candidates.append((int(r), int(vals[0])))
    assert len(candidates) == 1, "expected one solid non-background separator row"
    return candidates[0]


def _parse_arrow(coords: list[tuple[int, int]]) -> tuple[str, int]:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}

    if h == 3 and w >= 3 and len(coords) == w + 2:
        row_counts = [sum((rr, cc) in rel for cc in range(w)) for rr in range(h)]
        if row_counts == [1, w, 1]:
            top_cols = [cc for cc in range(w) if (0, cc) in rel]
            bottom_cols = [cc for cc in range(w) if (2, cc) in rel]
            if len(top_cols) == 1 and top_cols == bottom_cols:
                head_col = int(top_cols[0])
                if all((1, cc) in rel for cc in range(w)):
                    if head_col == 0:
                        return "left", int(w - 1)
                    if head_col == w - 1:
                        return "right", int(w - 1)

    if w == 3 and h >= 3 and len(coords) == h + 2:
        col_counts = [sum((rr, cc) in rel for rr in range(h)) for cc in range(w)]
        if col_counts == [1, h, 1]:
            left_rows = [rr for rr in range(h) if (rr, 0) in rel]
            right_rows = [rr for rr in range(h) if (rr, 2) in rel]
            if len(left_rows) == 1 and left_rows == right_rows:
                head_row = int(left_rows[0])
                if all((rr, 1) in rel for rr in range(h)):
                    if head_row == 0:
                        return "up", int(h - 1)
                    if head_row == h - 1:
                        return "down", int(h - 1)

    raise AssertionError("invalid arrow geometry")


def _extract_arrows(instr: np.ndarray, bg: int) -> list[dict[str, int]]:
    arrows: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(instr) if int(v) != int(bg)):
        for comp in _components(instr == int(color)):
            direction, steps = _parse_arrow(comp)
            r0, c0, _r1, _c1 = _bbox(comp)
            arrows.append(
                {
                    "color": int(color),
                    "direction": direction,
                    "steps": int(steps),
                    "row": int(r0),
                    "col": int(c0),
                }
            )
    arrows.sort(key=lambda item: (int(item["col"]), int(item["row"])))
    return arrows


def _translated_cells(
    cells: set[tuple[int, int]],
    dr: int,
    dc: int,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.add((int(rr), int(cc)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify colored arrow commands below a separator, then shift the matching
    color layers from the upper canvas by the arrow direction and line length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    sep_r, sep_color = _find_separator(g, bg)

    top = g[:sep_r, :]
    instr = g[sep_r + 1 :, :]
    assert top.size > 0 and instr.size > 0
    assert np.all(g[sep_r, :] == sep_color)

    arrows = _extract_arrows(instr, bg)
    arrow_colors = {int(arrow["color"]) for arrow in arrows}

    layer_colors = [
        int(v)
        for v in np.unique(top)
        if int(v) not in {int(bg), int(sep_color)}
    ]
    layers = {
        int(color): {
            (int(r), int(c))
            for r, c in map(tuple, np.argwhere(top == int(color)))
        }
        for color in layer_colors
    }

    out = np.full_like(top, int(bg))

    for color, cells in layers.items():
        if int(color) in arrow_colors:
            continue
        for r, c in cells:
            out[r, c] = int(color)

    for arrow in arrows:
        color = int(arrow["color"])
        if color not in layers:
            continue
        dr0, dc0 = DIRS[str(arrow["direction"])]
        dr = int(dr0 * int(arrow["steps"]))
        dc = int(dc0 * int(arrow["steps"]))
        for r, c in _translated_cells(layers[color], dr, dc, *top.shape):
            out[r, c] = int(color)

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _random_shape() -> set[tuple[int, int]]:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    area = random.randint(4, min(8, h * w))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return _random_shape()
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return (
        max(r for r, _c in cells) + 1,
        max(c for _r, c in cells) + 1,
    )


def _expand(cells: set[tuple[int, int]], h: int, w: int, margin: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - margin), min(h, r + margin + 1)):
            for cc in range(max(0, c - margin), min(w, c + margin + 1)):
                out.add((int(rr), int(cc)))
    return out


def _absolute_cells(shape: set[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r + r0), int(c + c0)) for r, c in shape}


def _candidate_placements(
    shape: set[tuple[int, int]],
    canvas_h: int,
    canvas_w: int,
    blocked: set[tuple[int, int]],
    dr: int,
    dc: int,
    want_clip: bool | None,
) -> list[set[tuple[int, int]]]:
    sh, sw = _shape_dims(shape)
    candidates: list[set[tuple[int, int]]] = []
    for r0 in range(canvas_h - sh + 1):
        for c0 in range(canvas_w - sw + 1):
            cells = _absolute_cells(shape, int(r0), int(c0))
            if any((r, c) in blocked for r, c in cells):
                continue
            shifted = {(int(r + dr), int(c + dc)) for r, c in cells}
            in_bounds = {
                (rr, cc)
                for rr, cc in shifted
                if 0 <= rr < canvas_h and 0 <= cc < canvas_w
            }
            clipped = len(in_bounds) != len(cells)
            if not in_bounds:
                continue
            if want_clip is True and not clipped:
                continue
            if want_clip is False and clipped:
                continue
            candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _make_arrow_patch(direction: str, steps: int, color: int) -> np.ndarray:
    if direction == "right":
        patch = np.full((3, steps + 1), BG, dtype=int)
        patch[1, :steps] = int(color)
        patch[:, steps] = int(color)
        return patch
    if direction == "left":
        patch = np.full((3, steps + 1), BG, dtype=int)
        patch[1, 1:] = int(color)
        patch[:, 0] = int(color)
        return patch
    if direction == "down":
        patch = np.full((steps + 1, 3), BG, dtype=int)
        patch[:steps, 1] = int(color)
        patch[steps, :] = int(color)
        return patch
    if direction == "up":
        patch = np.full((steps + 1, 3), BG, dtype=int)
        patch[1:, 1] = int(color)
        patch[0, :] = int(color)
        return patch
    raise AssertionError("unknown direction")


def _place_arrows(
    width: int,
    ins_h: int,
    arrows: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []
    for arrow in arrows:
        patch = np.asarray(arrow["patch"], dtype=int)
        ph, pw = patch.shape
        choices = []
        for r0 in range(ins_h - ph + 1):
            for c0 in range(width - pw + 1):
                cells = {
                    (int(r0 + r), int(c0 + c))
                    for r, c in map(tuple, np.argwhere(patch != BG))
                }
                if any(cell in blocked for cell in cells):
                    continue
                choices.append((int(r0), int(c0), cells))
        if not choices:
            return None
        r0, c0, cells = random.choice(choices)
        placed.append(
            {
                "color": int(arrow["color"]),
                "direction": str(arrow["direction"]),
                "steps": int(arrow["steps"]),
                "patch": patch,
                "row": int(r0),
                "col": int(c0),
                "cells": cells,
            }
        )
        blocked |= _expand(cells, ins_h, width, margin=1)

    keys = [(int(item["col"]), int(item["row"])) for item in placed]
    if len(keys) != len(set(keys)):
        return None
    return placed


def _output_overlap(layers: list[dict[str, object]]) -> bool:
    counts: Counter[tuple[int, int]] = Counter()
    for layer in layers:
        for cell in layer["result_cells"]:
            counts[cell] += 1
            if counts[cell] >= 2:
                return True
    return False


def _oracle_output(
    layers: list[dict[str, object]],
    placed_arrows: list[dict[str, object]],
    top_h: int,
    width: int,
) -> np.ndarray:
    out = np.full((top_h, width), BG, dtype=int)
    layer_map = {int(layer["color"]): set(layer["cells"]) for layer in layers}
    moved_colors = {int(arrow["color"]) for arrow in placed_arrows}

    for layer in layers:
        color = int(layer["color"])
        if color in moved_colors:
            continue
        for r, c in layer["cells"]:
            out[r, c] = int(color)

    for arrow in sorted(placed_arrows, key=lambda item: (int(item["col"]), int(item["row"]))):
        color = int(arrow["color"])
        dr0, dc0 = DIRS[str(arrow["direction"])]
        dr = int(dr0 * int(arrow["steps"]))
        dc = int(dc0 * int(arrow["steps"]))
        for r, c in _translated_cells(layer_map[color], dr, dc, top_h, width):
            out[r, c] = int(color)
    return out


def _generate_attempt() -> np.ndarray | None:
    top_h = random.randint(9, 14)
    width = random.randint(18, 24)
    n_layers = random.randint(3, 5)
    colors = random.sample(PALETTE, n_layers)

    max_moved = min(4, n_layers)
    moved_count = random.randint(1, max_moved)
    moved_colors = set(random.sample(colors, moved_count))
    need_clip = bool(moved_count and random.random() < 0.35)
    need_overlap = bool(moved_count >= 2 and random.random() < 0.35)
    forced_clip_color = random.choice(list(moved_colors)) if need_clip else None

    blocked: set[tuple[int, int]] = set()
    layers: list[dict[str, object]] = []
    arrow_specs: list[dict[str, object]] = []

    for color in colors:
        shape = _random_shape()
        direction = random.choice(list(DIRS)) if color in moved_colors else None
        steps = random.randint(2, 5) if color in moved_colors else 0
        dr0, dc0 = DIRS[direction] if direction is not None else (0, 0)
        dr = int(dr0 * steps)
        dc = int(dc0 * steps)

        want_clip: bool | None = None
        if color in moved_colors:
            if color == forced_clip_color:
                want_clip = True
            else:
                roll = random.random()
                if roll < 0.2:
                    want_clip = True
                elif roll < 0.7:
                    want_clip = False

        choices = _candidate_placements(shape, top_h, width, blocked, dr, dc, want_clip)
        if not choices and want_clip is not None:
            choices = _candidate_placements(shape, top_h, width, blocked, dr, dc, None)
        if not choices:
            return None

        cells = random.choice(choices)
        blocked |= _expand(cells, top_h, width, margin=1)
        result_cells = _translated_cells(cells, dr, dc, top_h, width) if color in moved_colors else set(cells)
        layers.append(
            {
                "color": int(color),
                "cells": cells,
                "moved": bool(color in moved_colors),
                "direction": direction,
                "steps": int(steps),
                "result_cells": result_cells,
                "clipped": bool(len(result_cells) != len(cells)) if color in moved_colors else False,
            }
        )
        if color in moved_colors:
            arrow_specs.append(
                {
                    "color": int(color),
                    "direction": str(direction),
                    "steps": int(steps),
                    "patch": _make_arrow_patch(str(direction), int(steps), int(color)),
                }
            )

    if need_clip and not any(bool(layer["clipped"]) for layer in layers if bool(layer["moved"])):
        return None
    if need_overlap and not _output_overlap(layers):
        return None

    random.shuffle(arrow_specs)
    max_patch_h = max(int(np.asarray(spec["patch"]).shape[0]) for spec in arrow_specs)
    ins_h_lo = max(8, max_patch_h + 1)
    ins_h_hi = max(ins_h_lo, max_patch_h + 6)
    ins_h = random.randint(ins_h_lo, ins_h_hi)
    placed_arrows = _place_arrows(width, ins_h, arrow_specs)
    if placed_arrows is None:
        return None

    grid = np.full((top_h + 1 + ins_h, width), BG, dtype=int)
    grid[top_h, :] = SEP

    for layer in layers:
        color = int(layer["color"])
        for r, c in layer["cells"]:
            grid[r, c] = int(color)

    for arrow in placed_arrows:
        patch = np.asarray(arrow["patch"], dtype=int)
        r0 = int(arrow["row"])
        c0 = int(arrow["col"])
        ph, pw = patch.shape
        target = grid[top_h + 1 + r0 : top_h + 1 + r0 + ph, c0 : c0 + pw]
        target[patch != BG] = patch[patch != BG]

    expected = _oracle_output(layers, placed_arrows, top_h, width)
    out = main(grid)
    if out.shape != (top_h, width):
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")
