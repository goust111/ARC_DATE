from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
MARKER = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if int(connectivity) == 8 else DIR4
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _step_two_runs(vals: list[int]) -> list[list[int]]:
    if not vals:
        return []
    runs: list[list[int]] = []
    cur = [int(vals[0])]
    for v in vals[1:]:
        v = int(v)
        if v - cur[-1] == 2:
            cur.append(v)
        else:
            runs.append(cur)
            cur = [v]
    runs.append(cur)
    return runs


def _make_line(kind: str, orientation: str, cells: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(cells)
    line = {
        "kind": kind,
        "orientation": orientation,
        "cells": tuple(sorted((int(r), int(c)) for r, c in cells)),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
    }
    if kind in {"solid", "dashed"}:
        line["axis"] = int(r0 if orientation == "h" else c0)
    else:
        line["twor"] = int(r0 + r1)
        line["twoc"] = int(c0 + c1)
    return line


def _extract_lines(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    azure = g == MARKER
    comps = _components(azure, connectivity=4)
    lines: list[dict] = []
    leftover: list[tuple[int, int]] = []

    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = len(comp)
        if area == 1:
            leftover.append(comp[0])
            continue
        full = bool(np.all(azure[r0 : r1 + 1, c0 : c1 + 1]))
        if h == 2 and w >= 3 and area == 2 * w and full:
            lines.append(_make_line("double", "h", comp))
        elif w == 2 and h >= 3 and area == 2 * h and full:
            lines.append(_make_line("double", "v", comp))
        elif h == 1 and w >= 3 and area == w:
            lines.append(_make_line("solid", "h", comp))
        elif w == 1 and h >= 3 and area == h:
            lines.append(_make_line("solid", "v", comp))

    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in leftover:
        by_row[int(r)].append(int(c))
        by_col[int(c)].append(int(r))

    dashed: list[dict] = []
    for r, cols in by_row.items():
        for run in _step_two_runs(sorted(cols)):
            if len(run) >= 3:
                dashed.append(_make_line("dashed", "h", [(r, c) for c in run]))
    for c, rows in by_col.items():
        for run in _step_two_runs(sorted(rows)):
            if len(run) >= 3:
                dashed.append(_make_line("dashed", "v", [(r, c) for r in run]))

    membership: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, line in enumerate(dashed):
        for cell in line["cells"]:
            membership[cell].append(idx)

    chosen = []
    for idx, line in enumerate(dashed):
        if all(len(membership[cell]) == 1 for cell in line["cells"]):
            chosen.append(line)

    lines.extend(chosen)
    lines.sort(key=lambda line: (line["r0"], line["c0"], line["kind"], line["orientation"]))
    return lines


def _point_line_distance(point: tuple[int, int], line: dict) -> int:
    r, c = point
    if line["kind"] == "double":
        dr = 0
        dc = 0
        if r < line["r0"]:
            dr = line["r0"] - r
        elif r > line["r1"]:
            dr = r - line["r1"]
        if c < line["c0"]:
            dc = line["c0"] - c
        elif c > line["c1"]:
            dc = c - line["c1"]
        return int(dr + dc)

    if line["orientation"] == "h":
        dr = abs(r - line["axis"])
        dc = 0
        if c < line["c0"]:
            dc = line["c0"] - c
        elif c > line["c1"]:
            dc = c - line["c1"]
        return int(dr + dc)

    dc = abs(c - line["axis"])
    dr = 0
    if r < line["r0"]:
        dr = line["r0"] - r
    elif r > line["r1"]:
        dr = r - line["r1"]
    return int(dr + dc)


def _nearest_line(comp: list[tuple[int, int]], lines: list[dict]) -> dict:
    scored = []
    for idx, line in enumerate(lines):
        dist = min(_point_line_distance(cell, line) for cell in comp)
        scored.append((int(dist), int(idx)))
    scored.sort()
    return lines[scored[0][1]]


def _transform_cell(cell: tuple[int, int], line: dict) -> tuple[int, int]:
    r, c = cell
    if line["kind"] == "solid":
        if line["orientation"] == "h":
            return (2 * line["axis"] - r, c)
        return (r, 2 * line["axis"] - c)
    if line["kind"] == "dashed":
        if line["orientation"] == "h":
            return (2 * line["axis"] - r, c + 1)
        return (r + 1, 2 * line["axis"] - c)
    return (line["twor"] - r, line["twoc"] - c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each nearby object using the symmetry encoded by its azure marker line.

    Solid line: mirror across the line.
    Dashed line: mirror across the line, then shift +1 along the line direction
    (right for horizontal, down for vertical).
    Double line: 180-degree rotation around the center of the double-line bbox.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    lines = _extract_lines(g)
    objects = _components((g != BG) & (g != MARKER), connectivity=8)

    for comp in objects:
        line = _nearest_line(comp, lines)
        for r, c in comp:
            nr, nc = _transform_cell((r, c), line)
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BG:
                out[nr, nc] = int(g[r, c])
    return out


GEN_SHAPE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_LOCAL_SIZE = 23
GEN_GROUP_RETRIES = 200
GEN_PACK_RETRIES = 200
GEN_TOTAL_RETRIES = 300
GEN_PACK_PAD = 4


def _normalize_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _random_connected_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    area = random.randint(3, min(h * w, 8))
    coords = {(random.randrange(h), random.randrange(w))}
    while len(coords) < area:
        frontier = set()
        for r, c in coords:
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                    frontier.add((nr, nc))
        coords.add(random.choice(list(frontier)))
    mask = _normalize_mask(coords)
    if random.choice([False, True]):
        mask = np.flipud(mask)
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    return mask.copy()


def _random_colored_shape() -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    mask = _random_connected_mask()
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    random.shuffle(cells)
    n_colors = random.randint(1, min(3, len(cells)))
    colors = random.sample(GEN_SHAPE_COLORS, n_colors)
    mapping: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(cells):
        color = colors[idx] if idx < n_colors else random.choice(colors)
        mapping[cell] = int(color)
    return mask, mapping


def _line_from_generation(kind: str, orientation: str, length: int, anchor_r: int, anchor_c: int) -> dict:
    if kind == "solid":
        if orientation == "h":
            cells = [(anchor_r, anchor_c + i) for i in range(length)]
        else:
            cells = [(anchor_r + i, anchor_c) for i in range(length)]
    elif kind == "dashed":
        if orientation == "h":
            cells = [(anchor_r, anchor_c + 2 * i) for i in range(length)]
        else:
            cells = [(anchor_r + 2 * i, anchor_c) for i in range(length)]
    else:
        if orientation == "h":
            cells = [(anchor_r + dr, anchor_c + i) for dr in (0, 1) for i in range(length)]
        else:
            cells = [(anchor_r + i, anchor_c + dc) for dc in (0, 1) for i in range(length)]
    return _make_line(kind, orientation, cells)


def _paint_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nz = np.argwhere(out != BG)
    r0 = max(0, int(nz[:, 0].min()) - 1)
    c0 = max(0, int(nz[:, 1].min()) - 1)
    r1 = min(out.shape[0] - 1, int(nz[:, 0].max()) + 1)
    c1 = min(out.shape[1] - 1, int(nz[:, 1].max()) + 1)
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _generate_group(kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(GEN_GROUP_RETRIES):
        orientation = random.choice(["h", "v"])
        length = random.randint(3, 7)
        span = length if kind != "dashed" else 2 * length - 1

        if orientation == "h":
            row_lo = 7
            row_hi = GEN_LOCAL_SIZE - 8 - (1 if kind == "double" else 0)
            col_lo = 3
            col_hi = GEN_LOCAL_SIZE - span - 4
            if row_lo > row_hi or col_lo > col_hi:
                continue
            anchor_r = random.randint(row_lo, row_hi)
            anchor_c = random.randint(col_lo, col_hi)
        else:
            row_lo = 3
            row_hi = GEN_LOCAL_SIZE - span - 4
            col_lo = 7
            col_hi = GEN_LOCAL_SIZE - 8 - (1 if kind == "double" else 0)
            if row_lo > row_hi or col_lo > col_hi:
                continue
            anchor_r = random.randint(row_lo, row_hi)
            anchor_c = random.randint(col_lo, col_hi)

        line = _line_from_generation(kind, orientation, length, anchor_r, anchor_c)
        line_cells = set(line["cells"])
        mask, color_map = _random_colored_shape()
        sh, sw = mask.shape
        side = random.choice([-1, 1])
        gap = random.randint(1, 3)

        if orientation == "h":
            src_r0 = line["r0"] - gap - sh if side < 0 else line["r1"] + gap + 1
            c_lo = max(1, line["c0"] - sw - 1)
            c_hi = min(GEN_LOCAL_SIZE - sw - 2, line["c1"] + 1)
            if src_r0 < 1 or src_r0 + sh > GEN_LOCAL_SIZE - 1 or c_lo > c_hi:
                continue
            src_c0 = random.randint(c_lo, c_hi)
        else:
            src_c0 = line["c0"] - gap - sw if side < 0 else line["c1"] + gap + 1
            r_lo = max(1, line["r0"] - sh - 1)
            r_hi = min(GEN_LOCAL_SIZE - sh - 2, line["r1"] + 1)
            if src_c0 < 1 or src_c0 + sw > GEN_LOCAL_SIZE - 1 or r_lo > r_hi:
                continue
            src_r0 = random.randint(r_lo, r_hi)

        source_cells = {(src_r0 + r, src_c0 + c) for r, c in color_map}
        if source_cells & line_cells:
            continue

        target_cells = {}
        ok = True
        for (rr, cc), color in color_map.items():
            sr, sc = src_r0 + rr, src_c0 + cc
            tr, tc = _transform_cell((sr, sc), line)
            if not (0 <= tr < GEN_LOCAL_SIZE and 0 <= tc < GEN_LOCAL_SIZE):
                ok = False
                break
            if (tr, tc) in line_cells or (tr, tc) in source_cells:
                ok = False
                break
            target_cells[(tr, tc)] = int(color)
        if not ok:
            continue

        inp = np.full((GEN_LOCAL_SIZE, GEN_LOCAL_SIZE), BG, dtype=int)
        out = np.full_like(inp, BG)
        for r, c in line["cells"]:
            inp[r, c] = MARKER
            out[r, c] = MARKER
        for (rr, cc), color in color_map.items():
            sr, sc = src_r0 + rr, src_c0 + cc
            inp[sr, sc] = int(color)
            out[sr, sc] = int(color)
        for (tr, tc), color in target_cells.items():
            out[tr, tc] = int(color)

        return _crop_pair(inp, out)
    return None


def _pack_groups(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = max(pair[0].shape[0] for pair in groups)
    max_w = max(pair[0].shape[1] for pair in groups)
    total_h = sum(pair[0].shape[0] for pair in groups) + GEN_PACK_PAD * (len(groups) + 1)
    total_w = sum(pair[0].shape[1] for pair in groups) + GEN_PACK_PAD * (len(groups) + 1)

    min_h = max(24, max_h + 2 * GEN_PACK_PAD, min(38, total_h // 2))
    min_w = max(24, max_w + 2 * GEN_PACK_PAD, min(38, total_w // 2))
    max_h_bound = max(min_h, min(42, total_h + GEN_PACK_PAD))
    max_w_bound = max(min_w, min(42, total_w + GEN_PACK_PAD))

    for _ in range(GEN_PACK_RETRIES):
        h = random.randint(min_h, max_h_bound)
        w = random.randint(min_w, max_w_bound)
        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        order = list(range(len(groups)))
        random.shuffle(order)
        order.sort(key=lambda idx: groups[idx][0].shape[0] * groups[idx][0].shape[1], reverse=True)
        ok = True

        for idx in order:
            in_patch, out_patch = groups[idx]
            ph, pw = in_patch.shape
            positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            random.shuffle(positions)
            placed = False
            for r0, c0 in positions:
                if reserved[r0 : r0 + ph, c0 : c0 + pw].any():
                    continue
                _paint_nonzero(inp, in_patch, r0, c0)
                _paint_nonzero(out, out_patch, r0, c0)
                rr0 = max(0, r0 - GEN_PACK_PAD)
                cc0 = max(0, c0 - GEN_PACK_PAD)
                rr1 = min(h, r0 + ph + GEN_PACK_PAD)
                cc1 = min(w, c0 + pw + GEN_PACK_PAD)
                reserved[rr0:rr1, cc0:cc1] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_TOTAL_RETRIES):
        n_groups = 3 + random.randint(0, 1)
        kinds = ["solid", "dashed", "double"]
        if n_groups == 4:
            kinds.append(random.choice(kinds))
        random.shuffle(kinds)

        groups = []
        valid = True
        for kind in kinds:
            pair = _generate_group(kind)
            if pair is None:
                valid = False
                break
            groups.append(pair)
        if not valid:
            continue

        packed = _pack_groups(groups)
        if packed is None:
            continue

        inp, expected = packed
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
