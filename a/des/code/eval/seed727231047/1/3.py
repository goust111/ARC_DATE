from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS = ("right", "left", "down", "up")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _parse_arrow(comp: dict, grid: np.ndarray) -> dict | None:
    coords = comp["coords"]
    values = grid[coords[:, 0], coords[:, 1]]
    if not np.all(values == values[0]):
        return None

    color = int(values[0])
    r0, c0, r1, c1 = comp["bbox"]
    cells = {(int(r), int(c)) for r, c in coords}

    if r1 - r0 == 2:
        mid_r = r0 + 1
        mid_cols = sorted(c for r, c in cells if r == mid_r)
        if mid_cols == list(range(c0, c1 + 1)) and len(mid_cols) >= 2:
            extras = {(r, c) for r, c in cells if r != mid_r}
            if extras == {(r0, c1 - 1), (r1, c1 - 1)}:
                return {
                    "direction": "right",
                    "color": color,
                    "tail": (mid_r, c0),
                    "tip": (mid_r, c1),
                    "length": int(c1 - c0),
                    "cells": cells,
                }
            if extras == {(r0, c0 + 1), (r1, c0 + 1)}:
                return {
                    "direction": "left",
                    "color": color,
                    "tail": (mid_r, c1),
                    "tip": (mid_r, c0),
                    "length": int(c1 - c0),
                    "cells": cells,
                }

    if c1 - c0 == 2:
        mid_c = c0 + 1
        mid_rows = sorted(r for r, c in cells if c == mid_c)
        if mid_rows == list(range(r0, r1 + 1)) and len(mid_rows) >= 2:
            extras = {(r, c) for r, c in cells if c != mid_c}
            if extras == {(r1 - 1, c0), (r1 - 1, c1)}:
                return {
                    "direction": "down",
                    "color": color,
                    "tail": (r0, mid_c),
                    "tip": (r1, mid_c),
                    "length": int(r1 - r0),
                    "cells": cells,
                }
            if extras == {(r0 + 1, c0), (r0 + 1, c1)}:
                return {
                    "direction": "up",
                    "color": color,
                    "tail": (r1, mid_c),
                    "tip": (r0, mid_c),
                    "length": int(r1 - r0),
                    "cells": cells,
                }

    return None


def _parse_pattern(comp: dict, grid: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h > 4 or w > 4:
        return None
    if comp["area"] != h * w:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
    }


def _match_pattern(arrow: dict, patterns: list[dict], used: set[int]) -> int:
    tail_r, tail_c = arrow["tail"]
    matches: list[int] = []
    for idx, pattern in enumerate(patterns):
        if idx in used:
            continue
        patch = pattern["patch"]
        if np.any(patch == arrow["color"]):
            continue
        r0, c0, r1, c1 = pattern["bbox"]
        if arrow["direction"] == "right" and c1 == tail_c - 2 and r0 <= tail_r <= r1:
            matches.append(idx)
        elif arrow["direction"] == "left" and c0 == tail_c + 2 and r0 <= tail_r <= r1:
            matches.append(idx)
        elif arrow["direction"] == "down" and r1 == tail_r - 2 and c0 <= tail_c <= c1:
            matches.append(idx)
        elif arrow["direction"] == "up" and r0 == tail_r + 2 and c0 <= tail_c <= c1:
            matches.append(idx)
    assert len(matches) == 1, "each arrow must have exactly one aligned pattern block"
    return matches[0]


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    assert 0 <= top <= out.shape[0] - ph
    assert 0 <= left <= out.shape[1] - pw
    out[top : top + ph, left : left + pw] = patch


def _stamp_copies(out: np.ndarray, arrow: dict, pattern: dict) -> None:
    patch = pattern["patch"]
    ph, pw = patch.shape
    r0, c0, _r1, _c1 = pattern["bbox"]
    tip_r, tip_c = arrow["tip"]
    length = int(arrow["length"])

    if arrow["direction"] == "right":
        for step in range(length):
            _paste(out, patch, r0, tip_c + 1 + step * pw)
    elif arrow["direction"] == "left":
        for step in range(length):
            _paste(out, patch, r0, tip_c - (step + 1) * pw)
    elif arrow["direction"] == "down":
        for step in range(length):
            _paste(out, patch, tip_r + 1 + step * ph, c0)
    else:
        for step in range(length):
            _paste(out, patch, tip_r - (step + 1) * ph, c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each monochromatic arrow and copy the aligned rectangular pattern block
    forward in the arrow's direction exactly as many times as the shaft length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows: list[dict] = []
    patterns: list[dict] = []
    for comp in _components(g != BG):
        arrow = _parse_arrow(comp, g)
        if arrow is not None:
            arrows.append(arrow)
            continue
        pattern = _parse_pattern(comp, g)
        assert pattern is not None, "unexpected non-arrow component"
        patterns.append(pattern)

    assert arrows, "expected at least one arrow"

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BG

    used_patterns: set[int] = set()
    for arrow in arrows:
        idx = _match_pattern(arrow, patterns, used_patterns)
        used_patterns.add(idx)
        _stamp_copies(out, arrow, patterns[idx])

    return out


def _make_pattern(height: int, width: int, allowed_colors: list[int]) -> np.ndarray:
    area = height * width
    max_colors = min(3, area, len(allowed_colors))
    if area == 1 or max_colors == 1:
        palette = [random.choice(allowed_colors)]
    else:
        low = 2 if random.random() < 0.8 else 1
        palette = random.sample(allowed_colors, random.randint(low, max_colors))

    mode = random.choice(("noise", "rows", "cols", "checker"))
    patch = np.zeros((height, width), dtype=int)

    if len(palette) == 1:
        patch[:] = palette[0]
    elif mode == "noise":
        patch[:] = np.random.choice(palette, size=(height, width))
    elif mode == "rows":
        for r in range(height):
            patch[r, :] = random.choice(palette)
    elif mode == "cols":
        for c in range(width):
            patch[:, c] = random.choice(palette)
    else:
        phase = random.randrange(len(palette))
        for r in range(height):
            for c in range(width):
                patch[r, c] = palette[(r + c + phase) % len(palette)]

    positions = random.sample(range(area), len(palette))
    for pos, color in zip(positions, palette):
        rr, cc = divmod(pos, width)
        patch[rr, cc] = color
    return patch


def _footprint_area(spec: dict) -> int:
    ph, pw = spec["pattern"].shape
    anchor = int(spec["anchor"])
    length = int(spec["length"])
    if spec["direction"] in ("right", "left"):
        height = ph + int(anchor == 0) + int(anchor == ph - 1)
        width = pw + 2 + length + length * pw
    else:
        width = pw + int(anchor == 0) + int(anchor == pw - 1)
        height = ph + 2 + length + length * ph
    return int(height * width)


def _placement_geometry(spec: dict, top: int, left: int, shape: tuple[int, int]) -> dict | None:
    h, w = shape
    patch = spec["pattern"]
    ph, pw = patch.shape
    anchor = int(spec["anchor"])
    length = int(spec["length"])
    direction = spec["direction"]

    if direction == "right":
        row = top + anchor
        tail_c = left + pw + 1
        tip_c = tail_c + length
        arrow_cells = [(row, c) for c in range(tail_c, tip_c + 1)]
        arrow_cells.extend([(row - 1, tip_c - 1), (row + 1, tip_c - 1)])
        copy_origins = [(top, tip_c + 1 + step * pw) for step in range(length)]
    elif direction == "left":
        row = top + anchor
        tail_c = left - 2
        tip_c = tail_c - length
        arrow_cells = [(row, c) for c in range(tip_c, tail_c + 1)]
        arrow_cells.extend([(row - 1, tip_c + 1), (row + 1, tip_c + 1)])
        copy_origins = [(top, tip_c - (step + 1) * pw) for step in range(length)]
    elif direction == "down":
        col = left + anchor
        tail_r = top + ph + 1
        tip_r = tail_r + length
        arrow_cells = [(r, col) for r in range(tail_r, tip_r + 1)]
        arrow_cells.extend([(tip_r - 1, col - 1), (tip_r - 1, col + 1)])
        copy_origins = [(tip_r + 1 + step * ph, left) for step in range(length)]
    else:
        col = left + anchor
        tail_r = top - 2
        tip_r = tail_r - length
        arrow_cells = [(r, col) for r in range(tip_r, tail_r + 1)]
        arrow_cells.extend([(tip_r + 1, col - 1), (tip_r + 1, col + 1)])
        copy_origins = [(tip_r - (step + 1) * ph, left) for step in range(length)]

    pattern_bbox = (top, left, top + ph - 1, left + pw - 1)
    r0, c0, r1, c1 = pattern_bbox

    for rr, cc in arrow_cells:
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        r0 = min(r0, rr)
        c0 = min(c0, cc)
        r1 = max(r1, rr)
        c1 = max(c1, cc)

    for rr, cc in copy_origins:
        if not (0 <= rr <= h - ph and 0 <= cc <= w - pw):
            return None
        r0 = min(r0, rr)
        c0 = min(c0, cc)
        r1 = max(r1, rr + ph - 1)
        c1 = max(c1, cc + pw - 1)

    return {
        "bbox": (r0, c0, r1, c1),
        "pattern_bbox": pattern_bbox,
        "arrow_cells": arrow_cells,
        "copy_origins": copy_origins,
    }


def _bbox_clear(blocked: np.ndarray, bbox: tuple[int, int, int, int], padding: int) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0] - 1, r1 + padding)
    cc1 = min(blocked.shape[1] - 1, c1 + padding)
    return not blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int], padding: int) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0] - 1, r1 + padding)
    cc1 = min(blocked.shape[1] - 1, c1 + padding)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_input_system(grid: np.ndarray, spec: dict, geom: dict) -> None:
    patch = spec["pattern"]
    top, left, _r1, _c1 = geom["pattern_bbox"]
    ph, pw = patch.shape
    grid[top : top + ph, left : left + pw] = patch
    for r, c in geom["arrow_cells"]:
        grid[r, c] = int(spec["arrow_color"])


def _render_expected(inp: np.ndarray, placements: list[tuple[dict, dict]]) -> np.ndarray:
    out = inp.copy()
    for spec, geom in placements:
        for r, c in geom["arrow_cells"]:
            out[r, c] = BG
    for spec, geom in placements:
        patch = spec["pattern"]
        ph, pw = patch.shape
        for top, left in geom["copy_origins"]:
            out[top : top + ph, left : left + pw] = patch
    return out


def _sample_spec(direction: str, arrow_color: int, pattern_colors: list[int], max_length: int) -> dict:
    ph = random.randint(1, 4)
    pw = random.randint(1, 4)
    anchor = random.randrange(ph if direction in ("right", "left") else pw)
    length = random.randint(1, max_length)
    return {
        "direction": direction,
        "arrow_color": arrow_color,
        "anchor": anchor,
        "length": length,
        "pattern": _make_pattern(ph, pw, pattern_colors),
    }


def _valid_placements(spec: dict, blocked: np.ndarray, outer_margin: int, padding: int) -> list[tuple[int, int, dict]]:
    h, w = blocked.shape
    ph, pw = spec["pattern"].shape
    placements: list[tuple[int, int, dict]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            geom = _placement_geometry(spec, top, left, (h, w))
            if geom is None:
                continue
            r0, c0, r1, c1 = geom["bbox"]
            if r0 < outer_margin or c0 < outer_margin or r1 >= h - outer_margin or c1 >= w - outer_margin:
                continue
            if _bbox_clear(blocked, geom["bbox"], padding):
                placements.append((top, left, geom))
    return placements


GRID_MIN = 22
GRID_MAX = 30
SYSTEM_COUNTS = (1, 2, 3)
OUTER_MARGIN = 1
SYSTEM_PADDING = 1
GENERATION_ATTEMPTS = 400


def _gen_attempt() -> np.ndarray | None:
    system_count = random.choice(SYSTEM_COUNTS)
    min_side = GRID_MIN + 2 * (system_count - 1)
    height = random.randint(min_side, GRID_MAX)
    width = random.randint(min_side, GRID_MAX)

    directions = random.sample(DIRS, system_count)
    arrow_colors = random.sample(COLORS, system_count)
    pattern_colors = [color for color in COLORS if color not in arrow_colors]
    max_length = 4 if system_count == 1 else 3

    specs = [
        _sample_spec(direction, arrow_color, pattern_colors, max_length)
        for direction, arrow_color in zip(directions, arrow_colors)
    ]
    specs.sort(key=_footprint_area, reverse=True)

    blocked = np.zeros((height, width), dtype=bool)
    grid = np.zeros((height, width), dtype=int)
    placements: list[tuple[dict, dict]] = []

    for spec in specs:
        options = _valid_placements(spec, blocked, OUTER_MARGIN, SYSTEM_PADDING)
        if not options:
            return None
        top, left, geom = random.choice(options)
        _draw_input_system(grid, spec, geom)
        _mark_bbox(blocked, geom["bbox"], SYSTEM_PADDING)
        placements.append((spec, geom))

    if len(_components(grid != BG)) != 2 * system_count:
        return None

    expected = _render_expected(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-copy input")
