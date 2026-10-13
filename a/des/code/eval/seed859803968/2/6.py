from __future__ import annotations

import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 500
GRID_SIDE_RANGE = (20, 31)
SIDE_LENGTH_CHOICES = (3, 5, 7, 9)
GROUP_COUNT_CHOICES = (1, 2)


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "r0": min(rows),
                    "r1": max(rows),
                    "c0": min(cols),
                    "c1": max(cols),
                }
            )
    return comps


def _is_line_segment(comp: dict) -> bool:
    if int(comp["area"]) < 2:
        return False
    same_row = int(comp["r0"]) == int(comp["r1"])
    same_col = int(comp["c0"]) == int(comp["c1"])
    if same_row == same_col:
        return False
    expected_len = int(comp["c1"] - comp["c0"] + 1) if same_row else int(comp["r1"] - comp["r0"] + 1)
    return expected_len == int(comp["area"])


def _segment_info(comp: dict) -> dict:
    same_row = int(comp["r0"]) == int(comp["r1"])
    length = int(comp["c1"] - comp["c0"] + 1) if same_row else int(comp["r1"] - comp["r0"] + 1)
    center_r = (int(comp["r0"]) + int(comp["r1"])) / 2.0
    center_c = (int(comp["c0"]) + int(comp["c1"])) / 2.0
    return {
        "color": int(comp["color"]),
        "orientation": "h" if same_row else "v",
        "length": length,
        "center_r": center_r,
        "center_c": center_c,
    }


def _bbox_from_center(center_r: int, center_c: int, side: int) -> tuple[int, int, int, int]:
    top = _round_half_up(center_r - (side - 1) / 2.0)
    left = _round_half_up(center_c - (side - 1) / 2.0)
    return int(top), int(left), int(top + side - 1), int(left + side - 1)


def _fill_seed_for_group(singletons: list[dict], used: set[int], bbox: tuple[int, int, int, int]) -> int:
    top, left, bottom, right = bbox
    matches = []
    for idx, comp in enumerate(singletons):
        if idx in used:
            continue
        r, c = comp["coords"][0]
        if top < r < bottom and left < c < right:
            matches.append((idx, int(comp["color"])))
    assert len(matches) == 1
    used.add(matches[0][0])
    return matches[0][1]


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int], horiz: list[dict], vert: list[dict], fill_color: int) -> None:
    top, left, bottom, right = bbox
    top_seg, bottom_seg = horiz
    left_seg, right_seg = vert

    out[top, left : right + 1] = int(top_seg["color"])
    out[bottom, left : right + 1] = int(bottom_seg["color"])
    out[top : bottom + 1, left] = int(left_seg["color"])
    out[top : bottom + 1, right] = int(right_seg["color"])
    if bottom - top >= 2 and right - left >= 2:
        out[top + 1 : bottom, left + 1 : right] = int(fill_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered line segments by shared length, rebuild each group into a centered square frame,
    and fill its interior with the color of the unique singleton seed inside that rebuilt frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g)
    line_segments = [_segment_info(comp) for comp in comps if _is_line_segment(comp)]
    singletons = [comp for comp in comps if int(comp["area"]) == 1]

    by_length: dict[int, list[dict]] = {}
    for seg in line_segments:
        by_length.setdefault(int(seg["length"]), []).append(seg)

    out = np.full_like(g, BG)
    used_singletons: set[int] = set()
    for length in sorted(by_length):
        group = by_length[length]
        assert len(group) == 4
        horiz = sorted(
            [seg for seg in group if seg["orientation"] == "h"],
            key=lambda seg: (seg["center_r"], seg["center_c"]),
        )
        vert = sorted(
            [seg for seg in group if seg["orientation"] == "v"],
            key=lambda seg: (seg["center_c"], seg["center_r"]),
        )
        assert len(horiz) == 2 and len(vert) == 2

        center_r = _round_half_up(sum(seg["center_r"] for seg in group) / 4.0)
        center_c = _round_half_up(sum(seg["center_c"] for seg in group) / 4.0)
        bbox = _bbox_from_center(center_r, center_c, int(length))
        fill_color = _fill_seed_for_group(singletons, used_singletons, bbox)
        _draw_frame(out, bbox, horiz, vert, fill_color)

    return out


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int, margin: int = 0) -> bool:
    return not (a1 + margin < b0 or b1 + margin < a0)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return _intervals_overlap(a[0], a[2], b[0], b[2], margin) and _intervals_overlap(a[1], a[3], b[1], b[3], margin)


def _place_group_centers(
    height: int, width: int, lengths: list[int], geom_by_len: dict[int, dict]
) -> dict[int, tuple[int, int]] | None:
    centers: dict[int, tuple[int, int]] = {}
    zones: list[tuple[int, int, int, int]] = []
    for side in sorted(lengths, reverse=True):
        geom = geom_by_len[side]
        radius_r = int(geom["radius_r"])
        radius_c = int(geom["radius_c"])
        if 2 * radius_r + 1 > height or 2 * radius_c + 1 > width:
            return None
        placed = False
        for _ in range(80):
            center_r = random.randint(radius_r, height - radius_r - 1)
            center_c = random.randint(radius_c, width - radius_c - 1)
            zone = (
                center_r - radius_r,
                center_c - radius_c,
                center_r + radius_r,
                center_c + radius_c,
            )
            if any(_boxes_overlap(zone, other, margin=2) for other in zones):
                continue
            centers[side] = (center_r, center_c)
            zones.append(zone)
            placed = True
            break
        if not placed:
            return None
    return centers


def _group_geometry(side: int) -> dict:
    half = side // 2
    row_offset = half + random.randint(3, 5)
    col_offset = half + random.randint(3, 5)
    max_hshift = min(2, col_offset - half - 2)
    max_vshift = min(2, row_offset - half - 2)
    hshift = random.randint(0, max_hshift)
    vshift = random.randint(0, max_vshift)
    radius_r = max(row_offset, vshift + half)
    radius_c = max(col_offset, hshift + half)
    return {
        "half": half,
        "row_offset": row_offset,
        "col_offset": col_offset,
        "hshift": hshift,
        "vshift": vshift,
        "radius_r": radius_r,
        "radius_c": radius_c,
    }


def _segment_cells(center_r: int, center_c: int, side: int, orientation: str) -> list[tuple[int, int]]:
    half = side // 2
    if orientation == "h":
        return [(center_r, c) for c in range(center_c - half, center_c + half + 1)]
    return [(r, center_c) for r in range(center_r - half, center_r + half + 1)]


def _render_group(
    inp: np.ndarray,
    out: np.ndarray,
    center_r: int,
    center_c: int,
    side: int,
    geom: dict,
    edge_colors: list[int],
    seed_color: int,
) -> None:
    half = int(geom["half"])
    row_offset = int(geom["row_offset"])
    col_offset = int(geom["col_offset"])
    hshift = int(geom["hshift"])
    vshift = int(geom["vshift"])

    top_center = (center_r - row_offset, center_c - hshift)
    bottom_center = (center_r + row_offset, center_c + hshift)
    left_center = (center_r - vshift, center_c - col_offset)
    right_center = (center_r + vshift, center_c + col_offset)

    segments = [
        ("h", top_center, edge_colors[0]),
        ("h", bottom_center, edge_colors[1]),
        ("v", left_center, edge_colors[2]),
        ("v", right_center, edge_colors[3]),
    ]
    for orientation, (seg_r, seg_c), color in segments:
        for r, c in _segment_cells(seg_r, seg_c, side, orientation):
            inp[r, c] = int(color)

    bbox = (center_r - half, center_c - half, center_r + half, center_c + half)
    interior_rows = list(range(bbox[0] + 1, bbox[2]))
    interior_cols = list(range(bbox[1] + 1, bbox[3]))
    seed_r = random.choice(interior_rows)
    seed_c = random.choice(interior_cols)
    inp[seed_r, seed_c] = int(seed_color)

    horiz = [{"color": edge_colors[0]}, {"color": edge_colors[1]}]
    vert = [{"color": edge_colors[2]}, {"color": edge_colors[3]}]
    _draw_frame(out, bbox, horiz, vert, int(seed_color))


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)

    max_groups = min(len(GROUP_COUNT_CHOICES), len(SIDE_LENGTH_CHOICES))
    group_count = GROUP_COUNT_CHOICES[random.randrange(max_groups)]
    lengths = random.sample(SIDE_LENGTH_CHOICES, group_count)
    geom_by_len = {side: _group_geometry(side) for side in lengths}
    centers = _place_group_centers(height, width, lengths, geom_by_len)
    if centers is None:
        return None

    non_bg_colors = list(range(1, 10))
    if 4 * group_count > len(non_bg_colors):
        return None
    random.shuffle(non_bg_colors)
    edge_palette = non_bg_colors[: 4 * group_count]
    seed_palette = [color for color in range(1, 10) if color not in edge_palette]
    if not seed_palette:
        return None

    inp = np.full((height, width), BG, dtype=int)
    out = np.full_like(inp, BG)
    for idx, side in enumerate(lengths):
        center_r, center_c = centers[side]
        edge_colors = edge_palette[4 * idx : 4 * idx + 4]
        seed_color = random.choice(seed_palette)
        _render_group(inp, out, center_r, center_c, side, geom_by_len[side], edge_colors, seed_color)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_case()
        if candidate is None:
            continue
        inp, expected = candidate
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
