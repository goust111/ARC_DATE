from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROW_TEMPLATES = {
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}): "up",
    frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}): "down",
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}): "left",
    frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}): "right",
}
TEMPLATE_BY_DIR = {direction: template for template, direction in ARROW_TEMPLATES.items()}
STEP_BY_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_RELATIVE = {
    direction: frozenset((r - 1, c - 1) for r, c in template)
    for template, direction in ARROW_TEMPLATES.items()
}

# --------------- solver ---------------


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(cells)),
                }
            )
    return comps


def _arrow_direction(coords: np.ndarray) -> str | None:
    if int(coords.shape[0]) != 5:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    if (r1 - r0, c1 - c0) != (2, 2):
        return None
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return ARROW_TEMPLATES.get(norm)


def _is_solid_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _box_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _scene_parts(grid: np.ndarray) -> tuple[int, np.ndarray, dict[int, int], dict[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps = _find_components(g, bg)
    assert comps, "expected arrows, bands, and obstacles"

    obstacle_colors = sorted(
        {
            int(comp["color"])
            for comp in comps
            if _is_solid_rectangle(comp)
            and min(int(comp["bbox"][2] - comp["bbox"][0] + 1), int(comp["bbox"][3] - comp["bbox"][1] + 1)) >= 2
        }
    )
    assert len(obstacle_colors) == 1, "expected exactly one obstacle color"
    obstacle_color = obstacle_colors[0]

    obstacle_mask = np.zeros((h, w), dtype=bool)
    for comp in comps:
        if int(comp["color"]) != obstacle_color:
            continue
        rh = int(comp["bbox"][2] - comp["bbox"][0] + 1)
        cw = int(comp["bbox"][3] - comp["bbox"][1] + 1)
        assert _is_solid_rectangle(comp) and min(rh, cw) >= 2, "obstacles must be solid rectangles"
        obstacle_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    horizontal_bands: dict[int, int] = {}
    vertical_bands: dict[int, int] = {}
    for r in range(h):
        row_vals = np.unique(g[r, :])
        if row_vals.size == 1:
            color = int(row_vals[0])
            if color not in (bg, obstacle_color):
                horizontal_bands[r] = color
    for c in range(w):
        col_vals = np.unique(g[:, c])
        if col_vals.size == 1:
            color = int(col_vals[0])
            if color not in (bg, obstacle_color):
                vertical_bands[c] = color

    arrows: list[dict] = []
    for comp in comps:
        if int(comp["color"]) == obstacle_color:
            continue
        direction = _arrow_direction(comp["coords"])
        if direction is None:
            continue
        r0, c0, _r1, _c1 = comp["bbox"]
        arrows.append(
            {
                "color": int(comp["color"]),
                "direction": direction,
                "center": (int(r0) + 1, int(c0) + 1),
                "coords": comp["coords"],
                "bbox": (int(r0), int(c0), int(r0) + 2, int(c0) + 2),
            }
        )

    assert arrows, "expected at least one 3x3 arrow"
    return bg, obstacle_mask, horizontal_bands, vertical_bands, arrows


def _crossed_band_color(
    direction: str,
    center_r: int,
    center_c: int,
    horizontal_bands: dict[int, int],
    vertical_bands: dict[int, int],
) -> int | None:
    dr, dc = STEP_BY_DIR[direction]
    if dc != 0 and center_c in vertical_bands:
        return int(vertical_bands[center_c])
    if dr != 0 and center_r in horizontal_bands:
        return int(horizontal_bands[center_r])
    if center_r in horizontal_bands:
        return int(horizontal_bands[center_r])
    if center_c in vertical_bands:
        return int(vertical_bands[center_c])
    return None


def _run_arrow(
    shape: str,
    start_center: tuple[int, int],
    start_color: int,
    obstacle_mask: np.ndarray,
    horizontal_bands: dict[int, int],
    vertical_bands: dict[int, int],
) -> dict:
    h, w = obstacle_mask.shape
    rel = ARROW_RELATIVE[shape]
    dr, dc = STEP_BY_DIR[shape]
    center_r, center_c = start_center
    color = int(start_color)
    crossed_colors: list[int] = []
    travel = 0

    while True:
        next_r = center_r + dr
        next_c = center_c + dc
        stop_kind = "obstacle"
        blocked = False
        for rr, cc in rel:
            ar = next_r + int(rr)
            ac = next_c + int(cc)
            if not (0 <= ar < h and 0 <= ac < w):
                stop_kind = "boundary"
                blocked = True
                break
            if obstacle_mask[ar, ac]:
                stop_kind = "obstacle"
                blocked = True
                break
        if blocked:
            break

        center_r = next_r
        center_c = next_c
        travel += 1
        band_color = _crossed_band_color(shape, center_r, center_c, horizontal_bands, vertical_bands)
        if band_color is not None:
            color = int(band_color)
            crossed_colors.append(int(band_color))

    cells = {(int(center_r + rr), int(center_c + cc)) for rr, cc in rel}
    final_box = (int(center_r) - 1, int(center_c) - 1, int(center_r) + 1, int(center_c) + 1)
    return {
        "final_center": (int(center_r), int(center_c)),
        "final_color": int(color),
        "travel": int(travel),
        "stop_kind": stop_kind,
        "crossed_colors": crossed_colors,
        "final_cells": cells,
        "final_box": final_box,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each 3x3 arrow in the pointed direction, recoloring it by the last full-row/full-column band crossed by its center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, obstacle_mask, horizontal_bands, vertical_bands, arrows = _scene_parts(g)

    out = g.copy()
    for arrow in arrows:
        out[arrow["coords"][:, 0], arrow["coords"][:, 1]] = bg

    for arrow in sorted(arrows, key=lambda item: item["bbox"]):
        result = _run_arrow(
            arrow["direction"],
            arrow["center"],
            arrow["color"],
            obstacle_mask,
            horizontal_bands,
            vertical_bands,
        )
        for r, c in result["final_cells"]:
            out[int(r), int(c)] = int(result["final_color"])

    return out


# --------------- generator ---------------

BG_COLOR = 0
OBSTACLE_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_H_RANGE = (15, 21)
GRID_W_RANGE = (15, 21)
BAND_COUNT_CHOICES = (1, 2, 3)
OBSTACLE_COUNT_CHOICES = (1, 2)
ARROW_COUNT_RANGE = (3, 4)
OBSTACLE_H_RANGE = (2, 5)
OBSTACLE_W_RANGE = (2, 6)
MIN_TRAVEL = 2
GEN_ATTEMPTS = 400


def _line_positions(length: int, count: int) -> list[int] | None:
    candidates = list(range(2, length - 2))
    random.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - old) >= 3 for old in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _arrow_cells_from_top_left(top: int, left: int, direction: str) -> set[tuple[int, int]]:
    template = TEMPLATE_BY_DIR[direction]
    return {(top + int(r), left + int(c)) for r, c in template}


def _draw_bands(
    grid: np.ndarray,
    band_kind: str,
    positions: list[int],
    colors: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    horizontal_bands: dict[int, int] = {}
    vertical_bands: dict[int, int] = {}
    if band_kind == "horizontal":
        for row, color in zip(positions, colors):
            grid[int(row), :] = int(color)
            horizontal_bands[int(row)] = int(color)
    else:
        for col, color in zip(positions, colors):
            grid[:, int(col)] = int(color)
            vertical_bands[int(col)] = int(color)
    return horizontal_bands, vertical_bands


def _place_obstacles(grid: np.ndarray, band_kind: str, positions: list[int], count: int) -> list[tuple[int, int, int, int]] | None:
    h, w = grid.shape
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(200):
            oh = random.randint(*OBSTACLE_H_RANGE)
            ow = random.randint(*OBSTACLE_W_RANGE)
            if oh >= h - 2 or ow >= w - 2:
                continue
            r0 = random.randint(1, h - oh - 1)
            c0 = random.randint(1, w - ow - 1)
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            if band_kind == "horizontal" and any(r in positions for r in range(r0, r1 + 1)):
                continue
            if band_kind == "vertical" and any(c in positions for c in range(c0, c1 + 1)):
                continue

            padded = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
            if any(_box_overlaps(padded, (br0, bc0, br1, bc1)) for br0, bc0, br1, bc1 in boxes):
                continue
            if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BG_COLOR):
                continue

            grid[r0 : r1 + 1, c0 : c1 + 1] = OBSTACLE_COLOR
            boxes.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return boxes


def _enumerate_candidates(
    base: np.ndarray,
    band_kind: str,
    horizontal_bands: dict[int, int],
    vertical_bands: dict[int, int],
) -> list[dict]:
    h, w = base.shape
    obstacle_mask = base == OBSTACLE_COLOR
    candidates: list[dict] = []
    parallel_dirs = {"left", "right"} if band_kind == "horizontal" else {"up", "down"}

    for direction in STEP_BY_DIR:
        for top in range(h - 2):
            for left in range(w - 2):
                if not np.all(base[top : top + 3, left : left + 3] == BG_COLOR):
                    continue
                start_center = (top + 1, left + 1)
                result = _run_arrow(
                    direction,
                    start_center,
                    1,
                    obstacle_mask,
                    horizontal_bands,
                    vertical_bands,
                )
                if int(result["travel"]) < MIN_TRAVEL:
                    continue
                cross_count = int(len(result["crossed_colors"]))
                start_box = (top, left, top + 2, left + 2)
                candidates.append(
                    {
                        "direction": direction,
                        "start_top_left": (top, left),
                        "start_box": start_box,
                        "start_center": start_center,
                        "start_cells": _arrow_cells_from_top_left(top, left, direction),
                        "final_box": result["final_box"],
                        "final_cells": result["final_cells"],
                        "final_center": result["final_center"],
                        "crossed_colors": result["crossed_colors"],
                        "cross_count": cross_count,
                        "stop_kind": result["stop_kind"],
                        "travel": result["travel"],
                        "final_color_if_changed": result["final_color"],
                        "features": {
                            "parallel" if direction in parallel_dirs else "perp",
                            "cross0" if cross_count == 0 else "cross_pos",
                            f"stop_{result['stop_kind']}",
                            *({"cross1"} if cross_count == 1 else set()),
                            *({"cross2"} if cross_count >= 2 else set()),
                        },
                    }
                )
    return candidates


def _candidates_conflict(a: dict, b: dict) -> bool:
    return _box_overlaps(a["start_box"], b["start_box"]) or _box_overlaps(a["final_box"], b["final_box"])


def _select_arrows(candidates: list[dict]) -> list[dict] | None:
    if not candidates:
        return None

    required = {"parallel", "perp", "cross0", "cross_pos", "stop_boundary", "stop_obstacle"}
    if any("cross1" in cand["features"] for cand in candidates):
        required.add("cross1")
    if any("cross2" in cand["features"] for cand in candidates):
        required.add("cross2")

    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    for _ in range(250):
        chosen: list[dict] = []
        covered: set[str] = set()
        feasible = candidates[:]
        random.shuffle(feasible)

        while feasible and len(chosen) < arrow_count:
            valid = [cand for cand in feasible if all(not _candidates_conflict(cand, old) for old in chosen)]
            if not valid:
                break
            gains = [len(cand["features"] - covered) for cand in valid]
            best_gain = max(gains)
            if best_gain == 0 and len(chosen) >= 3 and required <= covered:
                break
            pool = [cand for cand, gain in zip(valid, gains) if gain == best_gain]
            chosen.append(random.choice(pool))
            covered |= chosen[-1]["features"]
            feasible = [cand for cand in feasible if cand is not chosen[-1]]

        if required <= covered and len(chosen) >= 3:
            extra_pool = [
                cand
                for cand in candidates
                if cand not in chosen and all(not _candidates_conflict(cand, old) for old in chosen)
            ]
            if extra_pool and len(chosen) < ARROW_COUNT_RANGE[1] and random.random() < 0.5:
                chosen.append(random.choice(extra_pool))
            return chosen

    return None


def _build_expected(
    base: np.ndarray,
    selected: list[dict],
    arrow_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()
    for spec, color in zip(selected, arrow_colors):
        for r, c in spec["start_cells"]:
            inp[int(r), int(c)] = int(color)
        final_color = int(spec["crossed_colors"][-1]) if spec["crossed_colors"] else int(color)
        for r, c in spec["final_cells"]:
            out[int(r), int(c)] = int(final_color)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    band_kind = random.choice(("horizontal", "vertical"))
    band_count = random.choice(BAND_COUNT_CHOICES)
    positions = _line_positions(h if band_kind == "horizontal" else w, band_count)
    if positions is None:
        return None

    palette = list(PALETTE)
    random.shuffle(palette)
    band_colors = palette[:band_count]
    grid = np.full((h, w), BG_COLOR, dtype=int)
    horizontal_bands, vertical_bands = _draw_bands(grid, band_kind, positions, band_colors)

    obstacle_count = random.choice(OBSTACLE_COUNT_CHOICES)
    if _place_obstacles(grid, band_kind, positions, obstacle_count) is None:
        return None

    candidates = _enumerate_candidates(grid, band_kind, horizontal_bands, vertical_bands)
    selected = _select_arrows(candidates)
    if selected is None:
        return None

    needed_colors = band_count + len(selected)
    if needed_colors > len(PALETTE):
        return None
    used_band_colors = set(band_colors)
    arrow_palette = [color for color in palette if color not in used_band_colors]
    arrow_colors = arrow_palette[: len(selected)]
    if len(arrow_colors) != len(selected):
        return None

    inp, expected = _build_expected(grid, selected, arrow_colors)
    if _bg_color(inp) != BG_COLOR:
        return None
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-and-band scene")
