from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_geometry(h: int, w: int, corner: str, rows: int) -> tuple[list[int], list[int]]:
    if corner in ("tl", "bl"):
        cols = [0, 1, 2]
    else:
        cols = [w - 3, w - 2, w - 1]
    if corner in ("tl", "tr"):
        row_ids = list(range(rows))
    else:
        row_ids = list(range(h - rows, h))
    return row_ids, cols


def _legend_candidate(
    grid: np.ndarray, bg: int, corner: str
) -> tuple[dict[tuple[int, int], int], np.ndarray] | None:
    h, w = grid.shape
    if h < 2 or w < 4:
        return None

    rows, cols = _legend_geometry(h, w, corner, h)
    scan_rows = rows if corner in ("tl", "tr") else rows[::-1]
    used_rows: list[int] = []
    for r in scan_rows:
        if np.all(grid[r, cols] != bg):
            used_rows.append(int(r))
        else:
            break
    if len(used_rows) < 2:
        return None

    used_rows = sorted(used_rows)
    inner_c = 3 if corner in ("tl", "bl") else w - 4
    if 0 <= inner_c < w and np.any(grid[np.array(used_rows), inner_c] != bg):
        return None

    edge_row = used_rows[-1] + 1 if corner in ("tl", "tr") else used_rows[0] - 1
    if 0 <= edge_row < h and np.any(grid[edge_row, cols] != bg):
        return None

    mask = np.zeros_like(grid, dtype=bool)
    mapping: dict[tuple[int, int], int] = {}
    for r in used_rows:
        a = int(grid[r, cols[0]])
        m = int(grid[r, cols[1]])
        b = int(grid[r, cols[2]])
        if a == bg or b == bg or m == bg:
            return None
        key = tuple(sorted((a, b)))
        if key in mapping and mapping[key] != m:
            return None
        mapping[key] = m
        mask[r, cols[0] : cols[-1] + 1] = True
    return mapping, mask


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    candidates: list[tuple[dict[tuple[int, int], int], np.ndarray]] = []
    for corner in ("tl", "tr", "bl", "br"):
        cand = _legend_candidate(grid, bg, corner)
        if cand is not None:
            candidates.append(cand)
    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _find_rectangles(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    rects: list[dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or legend_mask[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if not legend_mask[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = len(coords)
            assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "components must be solid rectangles"
            rects.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )

    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["color"]))
    assert rects, "expected at least one rectangle"
    return rects


def _enumerate_bridges(
    grid: np.ndarray,
    bg: int,
    mapping: dict[tuple[int, int], int],
    rects: list[dict[str, int]],
) -> list[dict[str, int]]:
    bridges: list[dict[str, int]] = []

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a = rects[i]
            b = rects[j]
            mix = mapping.get(tuple(sorted((a["color"], b["color"]))))
            if mix is None:
                continue

            row0 = max(a["r0"], b["r0"])
            row1 = min(a["r1"], b["r1"])
            if row0 <= row1:
                if a["c1"] < b["c0"]:
                    c0 = a["c1"] + 1
                    c1 = b["c0"] - 1
                elif b["c1"] < a["c0"]:
                    c0 = b["c1"] + 1
                    c1 = a["c0"] - 1
                else:
                    c0, c1 = 1, 0
                if c0 <= c1 and np.all(grid[row0 : row1 + 1, c0 : c1 + 1] == bg):
                    bridges.append(
                        {
                            "orientation": "horizontal",
                            "r0": int(row0),
                            "r1": int(row1),
                            "c0": int(c0),
                            "c1": int(c1),
                            "color": int(mix),
                        }
                    )
                    continue

            col0 = max(a["c0"], b["c0"])
            col1 = min(a["c1"], b["c1"])
            if col0 <= col1:
                if a["r1"] < b["r0"]:
                    r0 = a["r1"] + 1
                    r1 = b["r0"] - 1
                elif b["r1"] < a["r0"]:
                    r0 = b["r1"] + 1
                    r1 = a["r0"] - 1
                else:
                    r0, r1 = 1, 0
                if r0 <= r1 and np.all(grid[r0 : r1 + 1, col0 : col1 + 1] == bg):
                    bridges.append(
                        {
                            "orientation": "vertical",
                            "r0": int(r0),
                            "r1": int(r1),
                            "c0": int(col0),
                            "c1": int(col1),
                            "color": int(mix),
                        }
                    )

    return bridges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build color-mixed bridges between solid rectangles using a corner legend.

    Legend rows are 1x3 triplets laid out horizontally in one grid corner:
    left color + right color -> middle color. For each pair of rectangles whose
    row spans or column spans overlap, fill the background corridor between them
    with the mapped bridge color. The bridge width equals the overlap length, and
    any non-background blocker cancels that bridge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, legend_mask = _find_legend(g, bg)
    rects = _find_rectangles(g, bg, legend_mask)
    bridges = _enumerate_bridges(g, bg, mapping, rects)

    out = g.copy()
    for bridge in bridges:
        r0, r1 = bridge["r0"], bridge["r1"]
        c0, c1 = bridge["c0"], bridge["c1"]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((target == bg) | (target == bridge["color"])), "bridge conflicts are not allowed"
        out[r0 : r1 + 1, c0 : c1 + 1] = bridge["color"]
    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 26)
GRID_W_RANGE = (20, 26)
RECT_SIDE_RANGE = (2, 4)
PAIR_GAP_RANGE = (1, 4)
BLOCKED_GAP_RANGE = (2, 5)
LEGEND_ROWS_RANGE = (2, 4)
CASE_TYPES = (
    "h",
    "v",
    "hv",
    "h_blocked",
    "v_blocked",
    "h_unmapped",
    "v_unmapped",
)
CASE_MAPPING_COUNT = {
    "h": 1,
    "v": 1,
    "hv": 2,
    "h_blocked": 2,
    "v_blocked": 2,
    "h_unmapped": 1,
    "v_unmapped": 1,
}
PAIR_ATTEMPTS = 250
GEN_ATTEMPTS = 500


def _bbox_of_rects(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    r0 = min(r for r, _, _, _ in rects)
    c0 = min(c for _, c, _, _ in rects)
    r1 = max(r + h - 1 for r, _, h, _ in rects)
    c1 = max(c + w - 1 for _, c, _, w in rects)
    return int(r0), int(c0), int(r1), int(c1)


def _fits_group(blocked: np.ndarray, rects: list[tuple[int, int, int, int]]) -> bool:
    h, w = blocked.shape
    occ = np.zeros_like(blocked, dtype=bool)
    for r0, c0, rh, cw in rects:
        if r0 < 0 or c0 < 0 or r0 + rh > h or c0 + cw > w:
            return False
        if occ[r0 : r0 + rh, c0 : c0 + cw].any():
            return False
        occ[r0 : r0 + rh, c0 : c0 + cw] = True
    br0, bc0, br1, bc1 = _bbox_of_rects(rects)
    rr0 = max(0, br0 - 1)
    rr1 = min(h, br1 + 2)
    cc0 = max(0, bc0 - 1)
    cc1 = min(w, bc1 + 2)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _commit_group(blocked: np.ndarray, rects: list[tuple[int, int, int, int]]) -> None:
    h, w = blocked.shape
    br0, bc0, br1, bc1 = _bbox_of_rects(rects)
    rr0 = max(0, br0 - 1)
    rr1 = min(h, br1 + 2)
    cc0 = max(0, bc0 - 1)
    cc1 = min(w, bc1 + 2)
    blocked[rr0:rr1, cc0:cc1] = True


def _reserve_legend(blocked: np.ndarray, corner: str, rows: int) -> tuple[list[int], list[int]]:
    h, w = blocked.shape
    row_ids, cols = _legend_geometry(h, w, corner, rows)
    _commit_group(blocked, [(row_ids[0], cols[0], len(row_ids), len(cols))])
    return row_ids, cols


def _sample_horizontal_pair(
    blocked: np.ndarray, region: tuple[int, int, int, int], gap_range: tuple[int, int]
) -> list[tuple[int, int, int, int]] | None:
    r_lo, r_hi, c_lo, c_hi = region
    for _ in range(PAIR_ATTEMPTS):
        h1 = random.randint(*RECT_SIDE_RANGE)
        h2 = random.randint(*RECT_SIDE_RANGE)
        w1 = random.randint(*RECT_SIDE_RANGE)
        w2 = random.randint(*RECT_SIDE_RANGE)
        overlap = random.randint(1, min(h1, h2))
        top1 = random.randint(0, h1 - overlap)
        top2 = random.randint(0, h2 - overlap)
        ov_min = max(r_lo + top1, r_lo + top2)
        ov_max = min(r_hi - (h1 - top1 - 1), r_hi - (h2 - top2 - 1))
        if ov_min > ov_max:
            continue
        gap = random.randint(*gap_range)
        c1_max = c_hi - (w1 + gap + w2) + 1
        if c1_max < c_lo:
            continue
        ov0 = random.randint(ov_min, ov_max)
        c1 = random.randint(c_lo, c1_max)
        rects = [
            (ov0 - top1, c1, h1, w1),
            (ov0 - top2, c1 + w1 + gap, h2, w2),
        ]
        if _fits_group(blocked, rects):
            return rects
    return None


def _sample_vertical_pair(
    blocked: np.ndarray, region: tuple[int, int, int, int], gap_range: tuple[int, int]
) -> list[tuple[int, int, int, int]] | None:
    r_lo, r_hi, c_lo, c_hi = region
    for _ in range(PAIR_ATTEMPTS):
        h1 = random.randint(*RECT_SIDE_RANGE)
        h2 = random.randint(*RECT_SIDE_RANGE)
        w1 = random.randint(*RECT_SIDE_RANGE)
        w2 = random.randint(*RECT_SIDE_RANGE)
        overlap = random.randint(1, min(w1, w2))
        left1 = random.randint(0, w1 - overlap)
        left2 = random.randint(0, w2 - overlap)
        ov_min = max(c_lo + left1, c_lo + left2)
        ov_max = min(c_hi - (w1 - left1 - 1), c_hi - (w2 - left2 - 1))
        if ov_min > ov_max:
            continue
        gap = random.randint(*gap_range)
        r1_max = r_hi - (h1 + gap + h2) + 1
        if r1_max < r_lo:
            continue
        ov0 = random.randint(ov_min, ov_max)
        r1 = random.randint(r_lo, r1_max)
        rects = [
            (r1, ov0 - left1, h1, w1),
            (r1 + h1 + gap, ov0 - left2, h2, w2),
        ]
        if _fits_group(blocked, rects):
            return rects
    return None


def _sample_horizontal_blocked(
    blocked: np.ndarray, region: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(PAIR_ATTEMPTS):
        outer = _sample_horizontal_pair(blocked, region, BLOCKED_GAP_RANGE)
        if outer is None:
            continue
        left, right = outer
        row0 = max(left[0], right[0])
        row1 = min(left[0] + left[2] - 1, right[0] + right[2] - 1)
        c0 = left[1] + left[3]
        c1 = right[1] - 1
        if row0 > row1 or c0 > c1:
            continue
        bh = random.randint(1, min(3, row1 - row0 + 1))
        bw = random.randint(1, min(3, c1 - c0 + 1))
        br = random.randint(row0, row1 - bh + 1)
        bc = random.randint(c0, c1 - bw + 1)
        rects = [left, (br, bc, bh, bw), right]
        if _fits_group(blocked, rects):
            return rects
    return None


def _sample_vertical_blocked(
    blocked: np.ndarray, region: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(PAIR_ATTEMPTS):
        outer = _sample_vertical_pair(blocked, region, BLOCKED_GAP_RANGE)
        if outer is None:
            continue
        top, bottom = outer
        col0 = max(top[1], bottom[1])
        col1 = min(top[1] + top[3] - 1, bottom[1] + bottom[3] - 1)
        r0 = top[0] + top[2]
        r1 = bottom[0] - 1
        if col0 > col1 or r0 > r1:
            continue
        bh = random.randint(1, min(3, r1 - r0 + 1))
        bw = random.randint(1, min(3, col1 - col0 + 1))
        br = random.randint(r0, r1 - bh + 1)
        bc = random.randint(col0, col1 - bw + 1)
        rects = [top, (br, bc, bh, bw), bottom]
        if _fits_group(blocked, rects):
            return rects
    return None


def _sample_single_rect(
    blocked: np.ndarray, region: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]] | None:
    r_lo, r_hi, c_lo, c_hi = region
    for _ in range(PAIR_ATTEMPTS):
        rh = random.randint(*RECT_SIDE_RANGE)
        cw = random.randint(*RECT_SIDE_RANGE)
        r_max = r_hi - rh + 1
        c_max = c_hi - cw + 1
        if r_max < r_lo or c_max < c_lo:
            continue
        rect = [(random.randint(r_lo, r_max), random.randint(c_lo, c_max), rh, cw)]
        if _fits_group(blocked, rect):
            return rect
    return None


def _case_layout(
    h: int, w: int, case: str, legend_rows: int, corner: str
) -> tuple[list[dict[str, object]], list[int], list[int]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    legend_row_ids, legend_cols = _reserve_legend(blocked, corner, legend_rows)

    full = (1, h - 2, 1, w - 2)
    upper = (1, h // 2 - 2, 1, w - 2)
    lower = (h // 2 + 1, h - 2, 1, w - 2)
    left = (1, h - 2, 1, w // 2 - 2)
    right = (1, h - 2, w // 2 + 1, w - 2)

    groups: list[dict[str, object]] = []

    def add_group(rects: list[tuple[int, int, int, int]], mapped_pair: tuple[int, int] | None) -> None:
        _commit_group(blocked, rects)
        groups.append({"rects": rects, "mapped_pair": mapped_pair})

    if case == "h":
        rects = _sample_horizontal_pair(blocked, full, PAIR_GAP_RANGE)
        if rects is None:
            return None
        add_group(rects, (0, 1))
        for _ in range(random.randint(0, 2)):
            extra = _sample_single_rect(blocked, full)
            if extra is not None:
                add_group(extra, None)
    elif case == "v":
        rects = _sample_vertical_pair(blocked, full, PAIR_GAP_RANGE)
        if rects is None:
            return None
        add_group(rects, (0, 1))
        for _ in range(random.randint(0, 2)):
            extra = _sample_single_rect(blocked, full)
            if extra is not None:
                add_group(extra, None)
    elif case == "hv":
        rects_h = _sample_horizontal_pair(blocked, upper, PAIR_GAP_RANGE)
        if rects_h is None:
            return None
        add_group(rects_h, (0, 1))
        rects_v = _sample_vertical_pair(blocked, lower, PAIR_GAP_RANGE)
        if rects_v is None:
            return None
        add_group(rects_v, (0, 1))
        if random.random() < 0.5:
            extra = _sample_single_rect(blocked, full)
            if extra is not None:
                add_group(extra, None)
    elif case == "h_blocked":
        rects_h = _sample_horizontal_pair(blocked, upper, PAIR_GAP_RANGE)
        if rects_h is None:
            return None
        add_group(rects_h, (0, 1))
        rects_b = _sample_horizontal_blocked(blocked, lower)
        if rects_b is None:
            return None
        add_group(rects_b, (0, 2))
    elif case == "v_blocked":
        rects_v = _sample_vertical_pair(blocked, left, PAIR_GAP_RANGE)
        if rects_v is None:
            return None
        add_group(rects_v, (0, 1))
        rects_b = _sample_vertical_blocked(blocked, right)
        if rects_b is None:
            return None
        add_group(rects_b, (0, 2))
    elif case == "h_unmapped":
        rects_h = _sample_horizontal_pair(blocked, upper, PAIR_GAP_RANGE)
        if rects_h is None:
            return None
        add_group(rects_h, (0, 1))
        rects_u = _sample_horizontal_pair(blocked, lower, PAIR_GAP_RANGE)
        if rects_u is None:
            return None
        add_group(rects_u, None)
    elif case == "v_unmapped":
        rects_v = _sample_vertical_pair(blocked, left, PAIR_GAP_RANGE)
        if rects_v is None:
            return None
        add_group(rects_v, (0, 1))
        rects_u = _sample_vertical_pair(blocked, right, PAIR_GAP_RANGE)
        if rects_u is None:
            return None
        add_group(rects_u, None)
    else:
        raise AssertionError("unknown case")

    return groups, legend_row_ids, legend_cols


def _draw_case(
    h: int, w: int, case: str, groups: list[dict[str, object]], legend_rows: list[int], legend_cols: list[int]
) -> np.ndarray | None:
    bg = random.randint(0, 9)
    rect_geoms: list[tuple[int, int, int, int]] = []
    pair_specs: list[tuple[int, int]] = []
    for group in groups:
        rects = group["rects"]
        base = len(rect_geoms)
        rect_geoms.extend(rects)
        if group["mapped_pair"] is not None:
            i, j = group["mapped_pair"]
            pair_specs.append((base + i, base + j))

    dummy_rows = len(legend_rows) - len(pair_specs)
    assert dummy_rows >= 0
    needed = len(rect_geoms) + len(pair_specs) + dummy_rows
    if needed > 9:
        return None

    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    rect_colors = palette[: len(rect_geoms)]
    bridge_colors = palette[len(rect_geoms) : len(rect_geoms) + len(pair_specs)]
    dummy_colors = palette[len(rect_geoms) + len(pair_specs) : len(rect_geoms) + len(pair_specs) + dummy_rows]

    grid = np.full((h, w), bg, dtype=int)
    for idx, (r0, c0, rh, cw) in enumerate(rect_geoms):
        grid[r0 : r0 + rh, c0 : c0 + cw] = rect_colors[idx]

    legend_entries: list[tuple[int, int, int]] = []
    for bridge_color, (i, j) in zip(bridge_colors, pair_specs):
        a = rect_colors[i]
        b = rect_colors[j]
        if random.random() < 0.5:
            legend_entries.append((a, bridge_color, b))
        else:
            legend_entries.append((b, bridge_color, a))

    for k, dummy_color in enumerate(dummy_colors):
        rect_color = rect_colors[k % len(rect_colors)]
        mix_color = bridge_colors[k % len(bridge_colors)]
        if random.random() < 0.5:
            legend_entries.append((dummy_color, mix_color, rect_color))
        else:
            legend_entries.append((rect_color, mix_color, dummy_color))

    random.shuffle(legend_entries)
    for r, entry in zip(legend_rows, legend_entries):
        grid[r, legend_cols[0]] = entry[0]
        grid[r, legend_cols[1]] = entry[1]
        grid[r, legend_cols[2]] = entry[2]

    try:
        bg2 = _get_bg(grid)
        mapping, legend_mask = _find_legend(grid, bg2)
        rects = _find_rectangles(grid, bg2, legend_mask)
        bridges = _enumerate_bridges(grid, bg2, mapping, rects)
        out = main(grid)
    except AssertionError:
        return None

    counts = {"horizontal": 0, "vertical": 0}
    for bridge in bridges:
        counts[bridge["orientation"]] += 1

    if case in ("h", "h_blocked", "h_unmapped"):
        if counts["horizontal"] < 1 or counts["vertical"] != 0:
            return None
    elif case in ("v", "v_blocked", "v_unmapped"):
        if counts["vertical"] < 1 or counts["horizontal"] != 0:
            return None
    elif case == "hv":
        if counts["horizontal"] < 1 or counts["vertical"] < 1:
            return None
    else:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate diverse bridge-building scenes with corner legends, including
    horizontal, vertical, blocked, and unmapped pair variants.
    """
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        case = random.choice(CASE_TYPES)
        legend_rows = random.randint(max(LEGEND_ROWS_RANGE[0], CASE_MAPPING_COUNT[case]), LEGEND_ROWS_RANGE[1])
        corner = random.choice(("tl", "tr", "bl", "br"))
        layout = _case_layout(h, w, case, legend_rows, corner)
        if layout is None:
            continue
        groups, row_ids, cols = layout
        grid = _draw_case(h, w, case, groups, row_ids, cols)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bridge puzzle input")
