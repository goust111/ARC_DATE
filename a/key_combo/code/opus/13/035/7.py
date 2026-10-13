from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
WALL = 5
ORANGE = 7
MARKER_COLORS = (BLUE, RED, GREEN)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

MIN_CHAMBER_SIDE = 5
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            seen[sr, sc] = True
            queue = deque([(sr, sc)])
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _line_info(line_cells: list[tuple[int, int]]) -> dict:
    assert 1 <= len(line_cells) <= 3
    rows = sorted({r for r, _c in line_cells})
    cols = sorted({c for _r, c in line_cells})
    ordered = sorted(line_cells)
    if len(line_cells) == 1:
        return {
            "orientation": "point",
            "length": 1,
            "mid": ordered[0],
            "cells": tuple(ordered),
        }
    if len(rows) == 1:
        row = rows[0]
        ordered = sorted(line_cells, key=lambda rc: rc[1])
        assert [c for _r, c in ordered] == list(range(ordered[0][1], ordered[0][1] + len(ordered)))
        return {
            "orientation": "h",
            "length": len(ordered),
            "mid": ordered[(len(ordered) - 1) // 2],
            "cells": tuple(ordered),
            "row": row,
        }
    assert len(cols) == 1
    col = cols[0]
    ordered = sorted(line_cells, key=lambda rc: rc[0])
    assert [r for r, _c in ordered] == list(range(ordered[0][0], ordered[0][0] + len(ordered)))
    return {
        "orientation": "v",
        "length": len(ordered),
        "mid": ordered[(len(ordered) - 1) // 2],
        "cells": tuple(ordered),
        "col": col,
    }


def _parse_chambers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    chambers = []
    for coords in _components(g != WALL):
        colors = Counter(int(g[r, c]) for r, c in coords)
        marker_cells = [(r, c) for r, c in coords if int(g[r, c]) in MARKER_COLORS]
        line_cells = [(r, c) for r, c in coords if int(g[r, c]) == ORANGE]
        assert len(marker_cells) == 1, "each chamber must contain exactly one marker pixel"
        assert 1 <= len(line_cells) <= 3, "each chamber must contain one 1-3 pixel orange line"
        assert len(colors.keys() - {BLACK, ORANGE, *MARKER_COLORS}) == 0
        cellset = set(coords)
        chambers.append(
            {
                "coords": coords,
                "cellset": cellset,
                "bbox": _bbox(coords),
                "marker": marker_cells[0],
                "marker_color": int(g[marker_cells[0]]),
                "line": _line_info(line_cells),
            }
        )
    return chambers


def _draw_pyramid(
    out: np.ndarray,
    cellset: set[tuple[int, int]],
    fill_color: int,
    line: dict,
) -> None:
    for r, c in cellset:
        out[r, c] = int(fill_color)

    if line["length"] == 1:
        return

    mid_r, mid_c = line["mid"]
    for depth in range(1, line["length"]):
        color = BLACK if depth % 2 == 1 else int(fill_color)
        span = range(-depth, depth + 1)
        if line["orientation"] == "h":
            for rr in (mid_r - depth, mid_r + depth):
                for dc in span:
                    pos = (rr, mid_c + dc)
                    if pos in cellset:
                        out[pos] = color
        else:
            for cc in (mid_c - depth, mid_c + depth):
                for dr in span:
                    pos = (mid_r + dr, cc)
                    if pos in cellset:
                        out[pos] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Flood-fill each wall-bounded chamber, then use its orange line to parameterize an alternating pyramid.

    Input structure:
    - The grid is black except for grey walls that partition it into several rectangular chambers.
    - Each chamber contains exactly one colored marker pixel (blue, red, or green) and one orange 1-3 pixel line segment.
    - Marker and line are both strictly inside the chamber, and the line is horizontal or vertical when its length exceeds 1.

    Rule:
    - Treat grey cells as barriers and process each non-wall chamber independently.
    - Flood-fill the whole chamber with the marker color, but use the orange line to determine a centered pyramid: its length is the number of layers, and its orientation is the pyramid axis.
    - Keep layer 1 as the chamber color, then add symmetric perpendicular layers of widths 3, 5, ... with colors alternating black, chamber color, black, ...; clip any layer cells that would cross chamber walls.

    Key insight:
    - Two different clues inside the same chamber split the job: the marker chooses the fill color, while the orange line chooses the pyramid geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full_like(g, BLACK)
    out[g == WALL] = WALL
    for chamber in _parse_chambers(g):
        _draw_pyramid(out, chamber["cellset"], chamber["marker_color"], chamber["line"])
    return out


def _split_options(rect: tuple[int, int, int, int]) -> list[tuple[str, int]]:
    r0, r1, c0, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    options: list[tuple[str, int]] = []
    if h >= MIN_CHAMBER_SIDE * 2 + 1:
        for wall_r in range(r0 + MIN_CHAMBER_SIDE, r1 - MIN_CHAMBER_SIDE + 1):
            options.append(("h", wall_r))
    if w >= MIN_CHAMBER_SIDE * 2 + 1:
        for wall_c in range(c0 + MIN_CHAMBER_SIDE, c1 - MIN_CHAMBER_SIDE + 1):
            options.append(("v", wall_c))
    return options


def _make_layout(chamber_count: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    for _ in range(80):
        h = random.randint(17, 26)
        w = random.randint(17, 26)
        grid = np.full((h, w), BLACK, dtype=int)
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL
        rects = [(1, h - 2, 1, w - 2)]

        while len(rects) < chamber_count:
            candidates = [
                (idx, rect, _split_options(rect))
                for idx, rect in enumerate(rects)
                if _split_options(rect)
            ]
            if not candidates:
                break

            weights = [((rect[1] - rect[0] + 1) * (rect[3] - rect[2] + 1)) for _idx, rect, _opts in candidates]
            idx, rect, options = random.choices(candidates, weights=weights, k=1)[0]
            orient, wall_pos = random.choice(options)
            r0, r1, c0, c1 = rect
            rects.pop(idx)
            if orient == "h":
                grid[wall_pos, c0 - 1 : c1 + 2] = WALL
                rects.extend([(r0, wall_pos - 1, c0, c1), (wall_pos + 1, r1, c0, c1)])
            else:
                grid[r0 - 1 : r1 + 2, wall_pos] = WALL
                rects.extend([(r0, r1, c0, wall_pos - 1), (r0, r1, wall_pos + 1, c1)])

        if len(rects) == chamber_count:
            return grid, rects
    return None


def _signature_constraints(signatures: list[tuple[str, int]]) -> bool:
    visible_orients = {orient for orient, length in signatures if length > 1}
    lengths = {length for _orient, length in signatures}
    return visible_orients == {"h", "v"} and len(lengths) >= 2 and max(lengths) >= 2


def _sample_signatures(chamber_count: int) -> list[tuple[str, int]]:
    pool = [("h", 1), ("h", 2), ("h", 3), ("v", 1), ("v", 2), ("v", 3)]
    robust_pool = [("h", 2), ("h", 3), ("v", 2), ("v", 3)]
    for _ in range(200):
        if chamber_count == 3:
            signatures = [
                random.choice([("h", 2), ("h", 3)]),
                random.choice([("v", 2), ("v", 3)]),
                random.choice(pool),
            ]
        else:
            dup = random.choice(robust_pool)
            other = random.choice([sig for sig in robust_pool if sig[0] != dup[0]])
            signatures = [dup, dup, other]
            while len(signatures) < chamber_count:
                signatures.append(random.choice(pool))
        random.shuffle(signatures)
        if chamber_count >= 4 and max(Counter(signatures).values()) < 2:
            continue
        if _signature_constraints(signatures):
            return signatures
    raise RuntimeError("failed to sample non-degenerate chamber signatures")


def _sample_colors(signatures: list[tuple[str, int]]) -> list[int]:
    chamber_count = len(signatures)
    for _ in range(200):
        colors = [None] * chamber_count
        if chamber_count >= 4:
            counts = Counter(signatures)
            dup_sig = next(sig for sig, count in counts.items() if count >= 2)
            dup_indices = [i for i, sig in enumerate(signatures) if sig == dup_sig][:2]
            a, b, c = random.sample(MARKER_COLORS, 3)
            colors[dup_indices[0]] = a
            colors[dup_indices[1]] = b
            other_index = next(i for i, sig in enumerate(signatures) if sig != dup_sig)
            colors[other_index] = a
            for i in range(chamber_count):
                if colors[i] is None:
                    colors[i] = random.choice(MARKER_COLORS)
        else:
            i, j = random.sample(range(chamber_count), 2)
            if signatures[i] == signatures[j]:
                continue
            a, b = random.sample(MARKER_COLORS, 2)
            colors[i] = a
            colors[j] = a
            for k in range(chamber_count):
                if colors[k] is None:
                    colors[k] = b if k != i and k != j else colors[k]

        if len(set(colors)) < 2:
            continue
        if not any(colors[i] == colors[j] and signatures[i] != signatures[j] for i in range(chamber_count) for j in range(i + 1, chamber_count)):
            continue
        if chamber_count >= 4 and not any(colors[i] != colors[j] and signatures[i] == signatures[j] for i in range(chamber_count) for j in range(i + 1, chamber_count)):
            continue
        return [int(color) for color in colors]
    raise RuntimeError("failed to sample diagnostic marker colors")


def _line_cells(rect: tuple[int, int, int, int], orient: str, length: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = rect
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    if orient == "h":
        row_choices = [r for r in range(max(r0 + 1, center_r - 1), min(r1 - 1, center_r + 1) + 1)]
        mid_choices = []
        for mid_c in range(c0 + 1, c1):
            start = mid_c - (length - 1) // 2
            end = start + length - 1
            if c0 + 1 <= start and end <= c1 - 1:
                mid_choices.append(mid_c)
        if not row_choices or not mid_choices:
            raise RuntimeError("horizontal line does not fit inside chamber")
        row = random.choice(row_choices)
        mid_c = min(mid_choices, key=lambda value: abs(value - center_c) + random.random() * 0.01)
        start = mid_c - (length - 1) // 2
        return [(row, c) for c in range(start, start + length)]

    if orient == "v":
        col_choices = [c for c in range(max(c0 + 1, center_c - 1), min(c1 - 1, center_c + 1) + 1)]
        mid_choices = []
        for mid_r in range(r0 + 1, r1):
            start = mid_r - (length - 1) // 2
            end = start + length - 1
            if r0 + 1 <= start and end <= r1 - 1:
                mid_choices.append(mid_r)
        if not col_choices or not mid_choices:
            raise RuntimeError("vertical line does not fit inside chamber")
        col = random.choice(col_choices)
        mid_r = min(mid_choices, key=lambda value: abs(value - center_r) + random.random() * 0.01)
        start = mid_r - (length - 1) // 2
        return [(r, col) for r in range(start, start + length)]

    raise AssertionError(f"unknown orientation: {orient}")


def _marker_cell(rect: tuple[int, int, int, int], line_cells: list[tuple[int, int]]) -> tuple[int, int]:
    r0, r1, c0, c1 = rect
    candidates = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in line_cells:
                continue
            cheb = min(max(abs(r - lr), abs(c - lc)) for lr, lc in line_cells)
            if cheb <= 1:
                continue
            manhattan = min(abs(r - lr) + abs(c - lc) for lr, lc in line_cells)
            candidates.append(((r, c), manhattan))
    if not candidates:
        raise RuntimeError("marker cannot be separated from the line")
    best = max(score for _cell, score in candidates)
    good = [cell for cell, score in candidates if score >= best - 1]
    return random.choice(good)


def _render_input(
    wall_grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    signatures: list[tuple[str, int]],
    colors: list[int],
) -> np.ndarray:
    grid = wall_grid.copy()
    for rect, signature, color in zip(rects, signatures, colors):
        orient, length = signature
        line_cells = _line_cells(rect, orient, length)
        marker = _marker_cell(rect, line_cells)
        grid[marker] = int(color)
        for cell in line_cells:
            grid[cell] = ORANGE
    return grid


def _is_rectangular_chamber(chamber: dict) -> bool:
    r0, c0, r1, c1 = chamber["bbox"]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return area == len(chamber["coords"])


def _semantic_check(inp: np.ndarray, out: np.ndarray, chambers: list[dict]) -> bool:
    if np.any(out == ORANGE):
        return False
    if not np.any(out == BLACK):
        return False

    output_patches = set()
    for chamber in chambers:
        if not _is_rectangular_chamber(chamber):
            return False
        r0, c0, r1, c1 = chamber["bbox"]
        if r1 - r0 + 1 < MIN_CHAMBER_SIDE or c1 - c0 + 1 < MIN_CHAMBER_SIDE:
            return False
        mr, mc = chamber["marker"]
        if mr in (r0, r1) or mc in (c0, c1):
            return False
        if any(max(abs(mr - lr), abs(mc - lc)) <= 1 for lr, lc in chamber["line"]["cells"]):
            return False

        expected = np.full(inp.shape, -1, dtype=int)
        _draw_pyramid(expected, chamber["cellset"], chamber["marker_color"], chamber["line"])
        for r, c in chamber["coords"]:
            if out[r, c] != expected[r, c]:
                return False

        crop = out[r0 : r1 + 1, c0 : c1 + 1]
        output_patches.add((crop.shape, crop.tobytes()))

        if chamber["line"]["length"] > 1:
            black_count = sum(int(out[r, c] == BLACK) for r, c in chamber["coords"])
            if black_count == 0:
                return False

    return len(output_patches) >= 2


def _input_is_informative(inp: np.ndarray) -> bool:
    chambers = _parse_chambers(inp)
    if not (3 <= len(chambers) <= 5):
        return False

    signatures = []
    colors = []
    for chamber in chambers:
        line = chamber["line"]
        colors.append(chamber["marker_color"])
        signatures.append((line["orientation"], line["length"]))

    visible_orients = {orient for orient, length in signatures if length > 1 and orient in {"h", "v"}}
    lengths = {length for _orient, length in signatures}
    if visible_orients != {"h", "v"}:
        return False
    if len(lengths) < 2 or max(lengths) < 2:
        return False
    if len(set(colors)) < 2:
        return False
    if not any(colors[i] == colors[j] and signatures[i] != signatures[j] for i in range(len(chambers)) for j in range(i + 1, len(chambers))):
        return False
    if len(chambers) >= 4 and not any(colors[i] != colors[j] and signatures[i] == signatures[j] for i in range(len(chambers)) for j in range(i + 1, len(chambers))):
        return False

    out = main(inp)
    return _semantic_check(inp, out, chambers)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        chamber_count = random.randint(3, 5)
        layout = _make_layout(chamber_count)
        if layout is None:
            continue
        wall_grid, rects = layout
        try:
            signatures = _sample_signatures(chamber_count)
            colors = _sample_colors(signatures)
            inp = _render_input(wall_grid, rects, signatures, colors)
        except RuntimeError:
            continue
        if _input_is_informative(inp):
            return inp
    raise RuntimeError("failed to generate a valid chamber-line ARC input")
