from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

DIR_VECTORS = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0),
}

REFLECT = {
    "\\": {"R": "D", "D": "R", "L": "U", "U": "L"},
    "/": {"R": "U", "U": "R", "L": "D", "D": "L"},
}

STRAIGHT_TEMPLATES = [("R",), ("U",)]
TURN_TEMPLATES = [("R", "U"), ("R", "D"), ("U", "L"), ("U", "R")]
DOUBLE_TEMPLATES = [
    ("R", "U", "R"),
    ("R", "D", "R"),
    ("U", "L", "U"),
    ("U", "R", "U"),
]

GRID_H_RANGE = (10, 17)
GRID_W_RANGE = (12, 20)
TABLE_ROWS_RANGE = (1, 4)
TOWER_COUNT_CHOICES = (1, 2, 3)
GEN_RETRIES = 400
PLACE_RETRIES = 120
SEG_LEN_RANGE = (1, 4)
MIN_TOTAL_LINE_LEN = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _background_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in DIR_VECTORS.values():
                    nr = r + dr
                    nc = c + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _corner_rows(h: int, corner: str) -> list[int]:
    if corner in {"tl", "tr"}:
        return list(range(h))
    return list(range(h - 1, -1, -1))


def _table_columns(w: int, corner: str) -> tuple[int, int, int | None]:
    if corner in {"tl", "bl"}:
        c0, c1 = 0, 1
        inner = 2 if w > 2 else None
    else:
        c0, c1 = w - 2, w - 1
        inner = w - 3 if w > 2 else None
    return c0, c1, inner


def _corner_table_candidate(
    g: np.ndarray, bg: int, corner: str
) -> tuple[dict[int, int], set[tuple[int, int]]] | None:
    h, w = g.shape
    if w < 2:
        return None
    c0, c1, inner = _table_columns(w, corner)
    rows = _corner_rows(h, corner)

    table_rows: list[int] = []
    for r in rows:
        if int(g[r, c0]) == bg or int(g[r, c1]) == bg:
            break
        if inner is not None and int(g[r, inner]) != bg:
            break
        table_rows.append(int(r))

    if not table_rows:
        return None

    if len(table_rows) < len(rows):
        stop_row = rows[len(table_rows)]
        if int(g[stop_row, c0]) != bg or int(g[stop_row, c1]) != bg:
            return None

    mapping: dict[int, int] = {}
    cells: set[tuple[int, int]] = set()
    for r in table_rows:
        a = int(g[r, c0])
        b = int(g[r, c1])
        if a == bg or b == bg:
            return None
        if a in mapping and mapping[a] != b:
            return None
        mapping[a] = b
        cells.add((r, c0))
        cells.add((r, c1))
    return mapping, cells


def _find_mapping_table(g: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    candidates = []
    for corner in ("tl", "tr", "bl", "br"):
        candidate = _corner_table_candidate(g, bg, corner)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected exactly one corner color map"
    return candidates[0]


def _find_prisms(
    g: np.ndarray, table_cells: set[tuple[int, int]]
) -> tuple[list[dict[str, int | str]], dict[tuple[int, int], dict[str, int | str]]]:
    h, w = g.shape
    raw: list[dict[str, int | str]] = []
    ownership: dict[tuple[int, int], list[int]] = {}

    for r in range(h - 1):
        for c in range(w - 1):
            cells = [(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1)]
            if any(cell in table_cells for cell in cells):
                continue
            patch = g[r : r + 2, c : c + 2]
            if int(np.count_nonzero(patch == GREY)) != 3 or int(np.count_nonzero(patch == MAROON)) != 1:
                continue
            maroon_r, maroon_c = map(int, np.argwhere(patch == MAROON)[0])
            kind = "\\" if (maroon_r, maroon_c) in {(0, 0), (1, 1)} else "/"
            idx = len(raw)
            prism = {"top": int(r), "left": int(c), "kind": kind}
            raw.append(prism)
            for cell in cells:
                ownership.setdefault(cell, []).append(idx)

    prisms: list[dict[str, int | str]] = []
    cell_to_prism: dict[tuple[int, int], dict[str, int | str]] = {}
    used_cells: set[tuple[int, int]] = set()

    for idx, prism in enumerate(raw):
        pr = int(prism["top"])
        pc = int(prism["left"])
        cells = [(pr + dr, pc + dc) for dr in (0, 1) for dc in (0, 1)]
        if any(len(ownership[cell]) != 1 for cell in cells):
            continue
        if any(cell in used_cells for cell in cells):
            continue
        prisms.append(prism)
        for cell in cells:
            used_cells.add(cell)
            cell_to_prism[cell] = prism

    grey_maroon_cells = {
        (int(r), int(c))
        for r, c in np.argwhere((g == GREY) | (g == MAROON))
        if (int(r), int(c)) not in table_cells
    }
    assert grey_maroon_cells == used_cells, "grey/maroon cells must belong to prisms"
    return prisms, cell_to_prism


def _find_towers(
    g: np.ndarray,
    bg: int,
    ignored: set[tuple[int, int]],
) -> list[dict[str, int | str]]:
    h, w = g.shape
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != bg and (r, c) not in ignored:
                mask[r, c] = True

    towers: list[dict[str, int | str]] = []
    seen_colors: set[int] = set()
    for comp in _component_cells(mask):
        assert len(comp) == 2, "every tower must be a two-pixel domino"
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "tower pixels must share one color"
        color = next(iter(colors))
        assert color not in seen_colors, "tower colors must be distinct"
        seen_colors.add(color)

        rows = sorted(r for r, _ in comp)
        cols = sorted(c for _, c in comp)
        if rows[0] == rows[1] and cols[1] == cols[0] + 1:
            orient = "h"
            top = rows[0]
            left = cols[0]
        elif cols[0] == cols[1] and rows[1] == rows[0] + 1:
            orient = "v"
            top = rows[0]
            left = cols[0]
        else:
            raise AssertionError("tower must be 1x2 or 2x1")
        towers.append({"top": top, "left": left, "orient": orient, "color": color})

    assert towers, "expected at least one tower"
    towers.sort(key=lambda tower: (int(tower["top"]), int(tower["left"]), int(tower["color"])))
    return towers


def _tower_start(tower: dict[str, int | str]) -> tuple[tuple[int, int], str]:
    r = int(tower["top"])
    c = int(tower["left"])
    if tower["orient"] == "h":
        return (r, c + 2), "R"
    return (r - 1, c), "U"


def _prism_exit(
    prism: dict[str, int | str], direction: str, hit: tuple[int, int]
) -> tuple[tuple[int, int], str]:
    pr = int(prism["top"])
    pc = int(prism["left"])
    kind = str(prism["kind"])
    hr, hc = hit
    if direction == "R":
        assert hc == pc
        idx = hr - pr
    elif direction == "L":
        assert hc == pc + 1
        idx = hr - pr
    elif direction == "U":
        assert hr == pr + 1
        idx = hc - pc
    else:
        assert direction == "D"
        assert hr == pr
        idx = hc - pc

    assert idx in (0, 1)
    out_dir = REFLECT[kind][direction]
    exit_idx = idx if kind == "\\" else 1 - idx
    if out_dir == "U":
        return (pr - 1, pc + exit_idx), out_dir
    if out_dir == "D":
        return (pr + 2, pc + exit_idx), out_dir
    if out_dir == "L":
        return (pr + exit_idx, pc - 1), out_dir
    return (pr + exit_idx, pc + 2), out_dir


def _cast_signal(
    out: np.ndarray,
    static: np.ndarray,
    table_map: dict[int, int],
    prism_lookup: dict[tuple[int, int], dict[str, int | str]],
    line_mask: np.ndarray,
    tower: dict[str, int | str],
) -> None:
    color = int(tower["color"])
    start, direction = _tower_start(tower)
    visited: set[tuple[int, int, str, int]] = set()

    while True:
        state = (start[0], start[1], direction, color)
        if state in visited:
            break
        visited.add(state)

        r, c = start
        dr, dc = DIR_VECTORS[direction]
        while _in_bounds(out.shape, r, c) and not line_mask[r, c] and int(static[r, c]) == BG:
            out[r, c] = color
            line_mask[r, c] = True
            r += dr
            c += dc

        if not _in_bounds(out.shape, r, c):
            break
        if line_mask[r, c]:
            break
        if (r, c) not in prism_lookup:
            break

        color = table_map.get(color, color)
        start, direction = _prism_exit(prism_lookup[(r, c)], direction, (r, c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit each tower's signal, reflect it at 2x2 prisms, and recolor it using the corner map.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    assert bg == BG, "expected black background"
    table_map, table_cells = _find_mapping_table(g, bg)
    prisms, prism_lookup = _find_prisms(g, table_cells)
    ignored = set(table_cells)
    for prism in prisms:
        pr = int(prism["top"])
        pc = int(prism["left"])
        for dr in (0, 1):
            for dc in (0, 1):
                ignored.add((pr + dr, pc + dc))
    towers = _find_towers(g, bg, ignored)

    out = g.copy()
    line_mask = np.zeros_like(g, dtype=bool)
    for tower in towers:
        _cast_signal(out, g, table_map, prism_lookup, line_mask, tower)
    return out


def _turn_kind(d1: str, d2: str) -> str:
    for kind in ("\\", "/"):
        if REFLECT[kind][d1] == d2:
            return kind
    raise AssertionError("invalid turn")


def _segment_cells(start: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    return [(start[0] + dr * i, start[1] + dc * i) for i in range(length)]


def _prism_after(
    last: tuple[int, int], direction: str, kind: str, offset: int
) -> tuple[int, int, tuple[int, int]]:
    r, c = last
    if direction == "R":
        pr, pc = r - offset, c + 1
    elif direction == "L":
        pr, pc = r - offset, c - 2
    elif direction == "U":
        pr, pc = r - 2, c - offset
    else:
        assert direction == "D"
        pr, pc = r + 1, c - offset
    out_dir = REFLECT[kind][direction]
    exit_idx = offset if kind == "\\" else 1 - offset
    if out_dir == "U":
        start = (pr - 1, pc + exit_idx)
    elif out_dir == "D":
        start = (pr + 2, pc + exit_idx)
    elif out_dir == "L":
        start = (pr + exit_idx, pc - 1)
    else:
        start = (pr + exit_idx, pc + 2)
    return pr, pc, start


def _prism_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr in (0, 1) for dc in (0, 1)}


def _build_template_spec(
    directions: tuple[str, ...], lengths: list[int], offsets: list[int]
) -> dict[str, object] | None:
    if directions[0] == "R":
        tower_cells = {(0, 0), (0, 1)}
        current = (0, 2)
    else:
        assert directions[0] == "U"
        tower_cells = {(0, 0), (1, 0)}
        current = (-1, 0)

    pre_line: list[tuple[int, int]] = []
    prisms: list[dict[str, int | str]] = []
    used = set(tower_cells)

    for idx in range(len(directions) - 1):
        seg = _segment_cells(current, directions[idx], lengths[idx])
        if not seg or any(cell in used for cell in seg):
            return None
        pre_line.extend(seg)
        used.update(seg)

        kind = _turn_kind(directions[idx], directions[idx + 1])
        pr, pc, current = _prism_after(seg[-1], directions[idx], kind, offsets[idx])
        prism_cells = _prism_cells(pr, pc)
        if any(cell in used for cell in prism_cells):
            return None
        prisms.append({"top": pr, "left": pc, "kind": kind})
        used.update(prism_cells)

    if current in used:
        return None
    return {
        "tower_cells": tower_cells,
        "pre_line": pre_line,
        "prisms": prisms,
        "final_start": current,
        "final_dir": directions[-1],
    }


def _shift_cells(cells: set[tuple[int, int]] | list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _placeable_translations(
    spec: dict[str, object],
    h: int,
    w: int,
    static_mask: np.ndarray,
    line_mask: np.ndarray,
    reserved_mask: np.ndarray,
) -> list[dict[str, object]]:
    tower_cells = list(spec["tower_cells"])
    pre_line = list(spec["pre_line"])
    prisms = list(spec["prisms"])
    prism_cells_rel = [cell for prism in prisms for cell in _prism_cells(int(prism["top"]), int(prism["left"]))]
    final_start = tuple(spec["final_start"])
    final_dir = str(spec["final_dir"])
    cells_for_bounds = tower_cells + pre_line + prism_cells_rel + [final_start]
    min_r = min(r for r, _ in cells_for_bounds)
    max_r = max(r for r, _ in cells_for_bounds)
    min_c = min(c for _, c in cells_for_bounds)
    max_c = max(c for _, c in cells_for_bounds)

    candidates: list[dict[str, object]] = []
    for dr in range(-min_r, h - max_r):
        for dc in range(-min_c, w - max_c):
            shifted_tower = _shift_cells(tower_cells, dr, dc)
            shifted_pre = _shift_cells(pre_line, dr, dc)
            shifted_prisms = [
                {"top": int(prism["top"]) + dr, "left": int(prism["left"]) + dc, "kind": prism["kind"]}
                for prism in prisms
            ]

            used: set[tuple[int, int]] = set()
            ok = True
            for r, c in shifted_tower:
                if static_mask[r, c] or line_mask[r, c] or reserved_mask[r, c]:
                    ok = False
                    break
                used.add((r, c))
            if not ok:
                continue

            for prism in shifted_prisms:
                for cell in _prism_cells(int(prism["top"]), int(prism["left"])):
                    r, c = cell
                    if static_mask[r, c] or line_mask[r, c] or reserved_mask[r, c] or cell in used:
                        ok = False
                        break
                    used.add(cell)
                if not ok:
                    break
            if not ok:
                continue

            for r, c in shifted_pre:
                if static_mask[r, c] or line_mask[r, c] or reserved_mask[r, c] or (r, c) in used:
                    ok = False
                    break
                used.add((r, c))
            if not ok:
                continue

            start = (final_start[0] + dr, final_start[1] + dc)
            if not _in_bounds((h, w), *start) or start in used or reserved_mask[start]:
                continue

            dray, dcay = DIR_VECTORS[final_dir]
            final_cells: list[tuple[int, int]] = []
            r, c = start
            while _in_bounds((h, w), r, c):
                if static_mask[r, c] or line_mask[r, c] or reserved_mask[r, c] or (r, c) in used:
                    ok = False
                    break
                final_cells.append((r, c))
                r += dray
                c += dcay
            if not ok or not final_cells:
                continue
            if len(shifted_pre) + len(final_cells) < MIN_TOTAL_LINE_LEN:
                continue

            candidates.append(
                {
                    "tower_cells": shifted_tower,
                    "pre_line": shifted_pre,
                    "prisms": shifted_prisms,
                    "final_line": final_cells,
                }
            )
    return candidates


def _table_layout(h: int, w: int, corner: str, rows: int) -> tuple[list[int], tuple[int, int], tuple[slice, slice]]:
    table_rows = _corner_rows(h, corner)[:rows]
    c0, c1, inner = _table_columns(w, corner)

    if corner == "tl":
        rs = slice(0, min(h, rows + 1))
        cs = slice(0, min(w, 3))
    elif corner == "tr":
        rs = slice(0, min(h, rows + 1))
        cs = slice(max(0, w - 3), w)
    elif corner == "bl":
        rs = slice(max(0, h - rows - 1), h)
        cs = slice(0, min(w, 3))
    else:
        assert corner == "br"
        rs = slice(max(0, h - rows - 1), h)
        cs = slice(max(0, w - 3), w)

    assert inner is None or 0 <= inner < w
    return table_rows, (c0, c1), (rs, cs)


def _choose_mapping() -> tuple[dict[int, int], list[tuple[int, int]]]:
    rows = random.randint(*TABLE_ROWS_RANGE)
    seq = random.sample(PALETTE, rows + 1)
    mapping = {seq[i]: seq[i + 1] for i in range(rows)}
    pairs = [(seq[i], seq[i + 1]) for i in range(rows)]
    return mapping, pairs


def _choose_templates(count: int) -> list[tuple[str, ...]]:
    templates: list[tuple[str, ...]] = []
    templates.append(random.choice(TURN_TEMPLATES + DOUBLE_TEMPLATES))
    for _ in range(count - 1):
        pool = STRAIGHT_TEMPLATES + TURN_TEMPLATES + DOUBLE_TEMPLATES
        templates.append(random.choice(pool))
    if count > 1 and not any(len(template) == 1 for template in templates):
        replace_idx = random.randrange(1, count)
        templates[replace_idx] = random.choice(STRAIGHT_TEMPLATES)
    random.shuffle(templates)
    return templates


def _choose_tower_colors(
    templates: list[tuple[str, ...]], mapping: dict[int, int]
) -> list[int]:
    mapped = [c for c in PALETTE if c in mapping]
    unmapped = [c for c in PALETTE if c not in mapping]
    remaining = set(PALETTE)
    colors: list[int] = []

    turned_indices = [i for i, template in enumerate(templates) if len(template) > 1]
    random.shuffle(turned_indices)
    if turned_indices:
        idx = turned_indices[0]
        if mapped:
            chosen = random.choice(mapped)
            colors = [None] * len(templates)  # type: ignore[list-item]
            colors[idx] = chosen
            remaining.remove(chosen)
        else:
            colors = [None] * len(templates)  # type: ignore[list-item]
    else:
        colors = [None] * len(templates)  # type: ignore[list-item]

    for i, template in enumerate(templates):
        if colors[i] is not None:
            continue
        wants_mapped = len(template) > 1 and random.random() < 0.5
        options = []
        if wants_mapped:
            options = [c for c in mapped if c in remaining]
        if not options:
            options = [c for c in unmapped if c in remaining]
        if not options:
            options = [c for c in PALETTE if c in remaining]
        chosen = random.choice(options)
        colors[i] = chosen
        remaining.remove(chosen)

    return [int(color) for color in colors]


def _draw_prism(grid: np.ndarray, prism: dict[str, int | str]) -> None:
    pr = int(prism["top"])
    pc = int(prism["left"])
    kind = str(prism["kind"])
    grid[pr : pr + 2, pc : pc + 2] = GREY
    if kind == "\\":
        maroon_cell = random.choice([(pr, pc), (pr + 1, pc + 1)])
    else:
        maroon_cell = random.choice([(pr, pc + 1), (pr + 1, pc)])
    grid[maroon_cell] = MAROON


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    mapping, pairs = _choose_mapping()
    corner = random.choice(("tl", "tr", "bl", "br"))
    table_rows, (c0, c1), reserved_box = _table_layout(h, w, corner, len(pairs))

    for row, (src, dst) in zip(table_rows, pairs):
        grid[row, c0] = src
        grid[row, c1] = dst

    reserved_mask = np.zeros((h, w), dtype=bool)
    reserved_mask[reserved_box] = True
    static_mask = np.zeros((h, w), dtype=bool)
    static_mask[grid != BG] = True
    line_mask = np.zeros((h, w), dtype=bool)

    tower_count = random.choice(TOWER_COUNT_CHOICES)
    templates = _choose_templates(tower_count)
    colors = _choose_tower_colors(templates, mapping)

    for template, color in zip(templates, colors):
        placed = False
        for _ in range(PLACE_RETRIES):
            lengths = [random.randint(*SEG_LEN_RANGE) for _ in range(max(0, len(template) - 1))]
            offsets = [random.randint(0, 1) for _ in range(max(0, len(template) - 1))]
            spec = _build_template_spec(template, lengths, offsets)
            if spec is None:
                continue
            candidates = _placeable_translations(spec, h, w, static_mask, line_mask, reserved_mask)
            if not candidates:
                continue

            placement = random.choice(candidates)
            for r, c in placement["tower_cells"]:
                grid[r, c] = color
                static_mask[r, c] = True
            for prism in placement["prisms"]:
                _draw_prism(grid, prism)
                for cell in _prism_cells(int(prism["top"]), int(prism["left"])):
                    static_mask[cell] = True
            for r, c in placement["pre_line"] + placement["final_line"]:
                line_mask[r, c] = True
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero((grid == GREY) | (grid == MAROON))) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
