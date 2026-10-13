from __future__ import annotations

import random

import numpy as np


def _infer_bg(g: np.ndarray) -> int:
    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    if np.any(border == 0):
        return 0
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(g == color):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(coords)
    return comps


def _frame_from_component(coords: list[tuple[int, int]], color: int) -> dict | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(coords) != 2 * h + 2 * w - 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    if not np.all(mask[0, :]) or not np.all(mask[-1, :]) or not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return None
    if h > 2 and w > 2 and np.any(mask[1:-1, 1:-1]):
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
    }


def _find_panels(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components_of_color(g, color):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                frames.append(frame)

    groups: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        groups.setdefault(frame["shape"], []).append(frame)

    ranked = sorted(
        (group for group in groups.values() if len(group) >= 3),
        key=lambda group: (len(group), group[0]["shape"][0] * group[0]["shape"][1]),
        reverse=True,
    )
    assert ranked, "expected a repeated panel-frame size"
    if len(ranked) >= 2:
        top = ranked[0]
        nxt = ranked[1]
        top_key = (len(top), top[0]["shape"][0] * top[0]["shape"][1])
        next_key = (len(nxt), nxt[0]["shape"][0] * nxt[0]["shape"][1])
        assert top_key > next_key, "panel-frame size must be uniquely identifiable"

    panels = sorted(ranked[0], key=lambda item: (item["bbox"][0], item["bbox"][1]))
    shapes = {panel["shape"] for panel in panels}
    assert len(shapes) == 1, "all panels must share one outer size"
    return panels


def _interior(g: np.ndarray, panel: dict) -> np.ndarray:
    r0, c0, r1, c1 = panel["bbox"]
    return g[r0 + 1 : r1, c0 + 1 : c1]


def _symmetry_flags(arr: np.ndarray) -> tuple[bool, bool, bool]:
    return (
        bool(np.array_equal(arr, arr[::-1, :])),
        bool(np.array_equal(arr, arr[:, ::-1])),
        bool(np.array_equal(arr, arr[::-1, ::-1])),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the unique symmetric panel interior into every panel that references its border color.

    Input structure:
    - The background is black. Several same-sized rectangular panels are separated by background.
    - Each panel is surrounded by a one-cell-thick colored border, and all border colors are distinct.
    - Each panel interior contains a sparse pattern made from one or two non-background colors.
    - Exactly one panel interior has exactly one symmetry among horizontal reflection, vertical reflection, and 180-degree rotation.

    Rule:
    - Detect all same-sized rectangular panel borders and inspect each interior.
    - Identify the unique "stamp" panel whose interior has exactly one of the allowed symmetries, and record its border color C_stamp.
    - Any other panel whose interior contains at least one C_stamp pixel is a target panel.
    - Replace every target panel interior with an exact copy of the stamp panel interior, leaving the stamp panel and all untouched panels unchanged.

    Key insight:
    - Symmetry selects which panel acts as the stamp, while the stamp border color reappears inside target interiors as the visible reference that says where the copied motif must propagate.
    """
    g = np.asarray(grid, dtype=int)
    bg = _infer_bg(g)
    panels = _find_panels(g, bg)

    stamp_indices: list[int] = []
    for i, panel in enumerate(panels):
        sym_count = sum(_symmetry_flags(_interior(g, panel)))
        if sym_count == 1:
            stamp_indices.append(i)
        else:
            assert sym_count == 0, "non-stamp panels must have no allowed symmetry"
    assert len(stamp_indices) == 1, "expected exactly one symmetric stamp panel"

    stamp_idx = stamp_indices[0]
    stamp_panel = panels[stamp_idx]
    stamp_color = int(stamp_panel["color"])
    stamp_pattern = _interior(g, stamp_panel).copy()

    out = g.copy()
    for i, panel in enumerate(panels):
        if i == stamp_idx:
            continue
        current = _interior(g, panel)
        if np.any(current == stamp_color):
            r0, c0, r1, c1 = panel["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = stamp_pattern
    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
LAYOUTS = ((1, 3), (1, 4), (2, 2), (2, 3), (3, 2))
INNER_H_RANGE = (4, 6)
INNER_W_RANGE = (4, 6)
MAX_EXTRA_GAP = 8
SYMMETRY_MODES = ("h", "v", "rot")


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _orbits(h: int, w: int, mode: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            if mode == "h":
                orbit = {(r, c), (h - 1 - r, c)}
            elif mode == "v":
                orbit = {(r, c), (r, w - 1 - c)}
            elif mode == "rot":
                orbit = {(r, c), (h - 1 - r, w - 1 - c)}
            else:
                raise ValueError(mode)
            ordered = sorted(orbit)
            seen.update(ordered)
            orbits.append(ordered)
    return orbits


def _make_symmetric_pattern(h: int, w: int, bg: int, colors: list[int], mode: str) -> np.ndarray:
    orbit_groups = _orbits(h, w, mode)
    min_orbits = max(2, len(orbit_groups) // 4)
    max_orbits = max(min_orbits, len(orbit_groups) - 1)
    for _ in range(300):
        arr = np.full((h, w), bg, dtype=int)
        active_count = random.randint(min_orbits, max_orbits)
        active_groups = random.sample(orbit_groups, active_count)
        for group in active_groups:
            color = random.choice(colors)
            for r, c in group:
                arr[r, c] = color
        flags = _symmetry_flags(arr)
        if np.count_nonzero(arr != bg) >= 2 and flags[SYMMETRY_MODES.index(mode)] and sum(flags) == 1:
            return arr
    raise RuntimeError("failed to construct symmetric stamp pattern")


def _make_asymmetric_pattern(
    h: int,
    w: int,
    bg: int,
    palette: list[int],
    *,
    must_include: int | None = None,
) -> np.ndarray:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(500):
        if must_include is None:
            color_count = random.randint(1, min(2, len(palette)))
            colors = random.sample(palette, color_count)
        else:
            extras = [color for color in palette if color != must_include]
            if extras and random.random() < 0.5:
                colors = [must_include, random.choice(extras)]
            else:
                colors = [must_include]

        arr = np.full((h, w), bg, dtype=int)
        min_cells = max(2, len(colors))
        max_cells = max(min_cells, min(h * w - 1, (h * w) // 2 + 2))
        paint_count = random.randint(min_cells, max_cells)
        picks = random.sample(cells, paint_count)

        idx = 0
        for color in colors:
            r, c = picks[idx]
            arr[r, c] = color
            idx += 1
        for r, c in picks[idx:]:
            arr[r, c] = random.choice(colors)

        if must_include is not None and not np.any(arr == must_include):
            continue
        if sum(_symmetry_flags(arr)) == 0:
            return arr
    raise RuntimeError("failed to construct asymmetric panel pattern")


def _panel_grid_shape(layout: tuple[int, int], ph: int, pw: int) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    pr, pc = layout
    base_h = pr * ph + (pr + 1)
    base_w = pc * pw + (pc + 1)
    if base_h > 30 or base_w > 30:
        return None

    extra_h = random.randint(0, min(MAX_EXTRA_GAP, 30 - base_h))
    extra_w = random.randint(0, min(MAX_EXTRA_GAP, 30 - base_w))
    gaps_r = np.asarray([1 + part for part in _split_total(extra_h, pr + 1)], dtype=int)
    gaps_c = np.asarray([1 + part for part in _split_total(extra_w, pc + 1)], dtype=int)

    H = int(pr * ph + gaps_r.sum())
    W = int(pc * pw + gaps_c.sum())
    assert H <= 30 and W <= 30
    return gaps_r, gaps_c, H, W


def _draw_panel(grid: np.ndarray, r0: int, c0: int, border_color: int, pattern: np.ndarray) -> None:
    h, w = pattern.shape
    r1 = r0 + h + 1
    c1 = c0 + w + 1
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = pattern


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)
    panel_h = inner_h + 2
    panel_w = inner_w + 2

    layout = random.choice(LAYOUTS)
    gap_info = _panel_grid_shape(layout, panel_h, panel_w)
    if gap_info is None:
        return None
    gaps_r, gaps_c, H, W = gap_info
    pr, pc = layout
    panel_count = pr * pc

    colors = list(range(1, 10))
    border_colors = random.sample(colors, panel_count)
    spare_colors = [color for color in colors if color not in border_colors]
    if not spare_colors:
        return None

    stamp_idx = random.randrange(panel_count)
    other_indices = [idx for idx in range(panel_count) if idx != stamp_idx]
    target_count = random.randint(1, len(other_indices) - 1)
    target_set = set(random.sample(other_indices, target_count))

    stamp_color = border_colors[stamp_idx]
    stamp_palette = random.sample(spare_colors, random.randint(1, min(2, len(spare_colors))))
    stamp_mode = random.choice(SYMMETRY_MODES)
    stamp_pattern = _make_symmetric_pattern(inner_h, inner_w, bg, stamp_palette, stamp_mode)

    patterns: list[np.ndarray] = []
    for idx in range(panel_count):
        if idx == stamp_idx:
            patterns.append(stamp_pattern)
        elif idx in target_set:
            palette = [stamp_color] + spare_colors
            patterns.append(_make_asymmetric_pattern(inner_h, inner_w, bg, palette, must_include=stamp_color))
        else:
            patterns.append(_make_asymmetric_pattern(inner_h, inner_w, bg, spare_colors))

    inp = np.full((H, W), bg, dtype=int)
    starts: list[tuple[int, int]] = []
    rr = int(gaps_r[0])
    for row in range(pr):
        cc = int(gaps_c[0])
        for col in range(pc):
            idx = row * pc + col
            starts.append((rr, cc))
            _draw_panel(inp, rr, cc, border_colors[idx], patterns[idx])
            cc += panel_w + int(gaps_c[col + 1])
        rr += panel_h + int(gaps_r[row + 1])

    expected = inp.copy()
    for idx in target_set:
        r0, c0 = starts[idx]
        expected[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = stamp_pattern

    try:
        panels = _find_panels(inp, bg)
        if len(panels) != panel_count:
            return None
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid panel-stamp input")
