import random
from collections import deque

import numpy as np


BG = 0
GREEN = 3
GREY = 5
BAND_COLORS = (1, 2, 4, 6, 7, 8, 9)
MAX_RETRIES = 300
_RNG = random.Random()


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _frame_infos(grid: np.ndarray) -> list[dict[str, int]]:
    infos: list[dict[str, int]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        border_size = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        if len(comp) != border_size:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        infos.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
    return infos


def _boundary_band_map(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    band_cells: set[tuple[int, int]] = set()
    boundary_colors: set[int] = set()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            val = int(grid[r, c])
            if val in (bg, GREEN, GREY):
                continue
            band_cells.add((r, c))
            boundary_colors.add(val)

    length_to_color: dict[int, int] = {}
    for color in boundary_colors:
        pts = [(r, c) for (r, c) in band_cells if int(grid[r, c]) == color]
        if not pts:
            continue
        rows = {r for r, _ in pts}
        cols = {c for _, c in pts}
        if len(rows) == 1 or len(cols) == 1:
            length_to_color[len(pts)] = color
    return length_to_color, band_cells


def _anchor_spec(green_pts: list[tuple[int, int]], frame: dict[str, int]) -> tuple[str, int, int]:
    rows = sorted({r for r, _ in green_pts})
    cols = sorted({c for _, c in green_pts})
    count = len(green_pts)
    inner_r0 = frame["r0"] + 1
    inner_c0 = frame["c0"] + 1

    if len(rows) == 1 and len(cols) > 1:
        return "h", rows[0], count
    if len(cols) == 1 and len(rows) > 1:
        return "v", cols[0], count

    r, c = green_pts[0]
    if c == inner_c0 and r != inner_r0:
        return "h", r, count
    if r == inner_r0 and c != inner_c0:
        return "v", c, count

    inner_h = frame["r1"] - frame["r0"] - 1
    inner_w = frame["c1"] - frame["c0"] - 1
    if inner_w >= inner_h:
        return "h", r, count
    return "v", c, count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each anchor line inside a grey frame with a matching edge-band color segment.

    Input structure:
    - bg is black, with several hollow grey rectangular frames fully inside the grid.
    - Each frame contains 1-5 green anchor pixels arranged as one horizontal or vertical line segment.
    - The outer border of the grid contains colored 1-cell-thick bands; each band is a solid row or column segment on an edge.

    Rule:
    - For each frame, count the green anchors to get N and find the edge band whose length is exactly N.
    - If the anchors are horizontal, draw N pixels of that band's color on the anchors' row, starting from the frame's left inner wall.
    - If the anchors are vertical, draw N pixels of that band's color on the anchors' column, starting from the frame's top inner wall.
    - The new colored segment replaces the green anchors, and all edge bands are erased to the black background.

    Key insight:
    - The anchor count selects which border band supplies the color, while the anchor orientation selects whether the fill grows rightward or downward from the frame's inner wall.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    band_map, band_cells = _boundary_band_map(g, bg)
    for r, c in band_cells:
        out[r, c] = bg

    for frame in _frame_infos(g):
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        green_pts = [
            (int(r), int(c))
            for r, c in np.argwhere(g[r0 + 1 : r1, c0 + 1 : c1] == GREEN)
        ]
        green_pts = [(r + r0 + 1, c + c0 + 1) for r, c in green_pts]
        if not green_pts:
            continue

        orient, line_idx, count = _anchor_spec(green_pts, frame)
        color = band_map.get(count)
        if color is None:
            continue

        for r, c in green_pts:
            out[r, c] = bg

        if orient == "h":
            start_c = c0 + 1
            out[line_idx, start_c : start_c + count] = color
        else:
            start_r = r0 + 1
            out[start_r : start_r + count, line_idx] = color

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _choose_specs(rng: random.Random) -> list[dict[str, int | str]]:
    num_frames = rng.randint(2, 4)
    num_h = rng.randint(1, num_frames - 1)
    orientations = ["h"] * num_h + ["v"] * (num_frames - num_h)
    rng.shuffle(orientations)

    max_counts: list[int] = []
    specs: list[dict[str, int | str]] = []
    for orient in orientations:
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 6)
        max_count = min(5, inner_w if orient == "h" else inner_h)
        max_counts.append(max_count)
        specs.append(
            {
                "orient": orient,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "count": rng.randint(1, max_count),
            }
        )

    if len(set(int(spec["count"]) for spec in specs)) == 1:
        for idx in range(len(specs)):
            options = [v for v in range(1, max_counts[idx] + 1) if v != int(specs[idx]["count"])]
            if options:
                specs[idx]["count"] = rng.choice(options)
                break

    if len(specs) >= 2 and rng.random() < 0.35:
        i, j = rng.sample(range(len(specs)), 2)
        common_max = min(max_counts[i], max_counts[j])
        specs[j]["count"] = rng.randint(1, common_max)
        specs[i]["count"] = int(specs[j]["count"])

    return specs


def _place_frames(
    grid: np.ndarray, specs: list[dict[str, int | str]], rng: random.Random
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    padding_mask = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)

    order = sorted(
        range(len(specs)),
        key=lambda idx: -(
            (int(specs[idx]["inner_h"]) + 2) * (int(specs[idx]["inner_w"]) + 2)
        ),
    )

    for idx in order:
        outer_h = int(specs[idx]["inner_h"]) + 2
        outer_w = int(specs[idx]["inner_w"]) + 2
        max_r0 = h - outer_h - 2
        max_c0 = w - outer_w - 2
        if max_r0 < 2 or max_c0 < 2:
            return None

        placed = False
        for _ in range(200):
            r0 = rng.randint(2, max_r0)
            c0 = rng.randint(2, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r1 + 2)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c1 + 2)
            if padding_mask[rr0:rr1, cc0:cc1].any():
                continue
            padding_mask[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [p for p in placements if p is not None]


def _place_anchors(
    grid: np.ndarray, r0: int, c0: int, spec: dict[str, int | str], rng: random.Random
) -> None:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    count = int(spec["count"])
    orient = str(spec["orient"])
    top = r0 + 1
    left = c0 + 1

    if orient == "h":
        row_choices = list(range(top, top + inner_h))
        if count == 1:
            row_choices = list(range(top + 1, top + inner_h))
        row = rng.choice(row_choices)
        grid[row, left : left + count] = GREEN
    else:
        col_choices = list(range(left, left + inner_w))
        if count == 1:
            col_choices = list(range(left + 1, left + inner_w))
        col = rng.choice(col_choices)
        grid[top : top + count, col] = GREEN


def _place_bands(grid: np.ndarray, length_to_color: dict[int, int], rng: random.Random) -> bool:
    h, w = grid.shape
    occupied = {
        "top": np.zeros(w, dtype=bool),
        "bottom": np.zeros(w, dtype=bool),
        "left": np.zeros(h, dtype=bool),
        "right": np.zeros(h, dtype=bool),
    }
    items = list(length_to_color.items())
    rng.shuffle(items)

    for length, color in items:
        placed = False
        for _ in range(200):
            side = rng.choice(["top", "bottom", "left", "right"])
            line_len = w if side in ("top", "bottom") else h
            lo = 1
            hi = line_len - length - 1
            if hi < lo:
                continue
            start = rng.randint(lo, hi)
            pad_lo = max(0, start - 1)
            pad_hi = min(line_len, start + length + 1)
            if occupied[side][pad_lo:pad_hi].any():
                continue
            occupied[side][start : start + length] = True
            if side == "top":
                grid[0, start : start + length] = color
            elif side == "bottom":
                grid[h - 1, start : start + length] = color
            elif side == "left":
                grid[start : start + length, 0] = color
            else:
                grid[start : start + length, w - 1] = color
            placed = True
            break
        if not placed:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _choose_specs(rng)
    num_frames = len(specs)
    min_dim = {2: 18, 3: 22, 4: 25}[num_frames]
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)
    grid = np.full((h, w), BG, dtype=int)

    placements = _place_frames(grid, specs, rng)
    if placements is None:
        return None

    for spec, (r0, c0) in zip(specs, placements):
        _draw_frame(grid, r0, c0, int(spec["inner_h"]), int(spec["inner_w"]))
        _place_anchors(grid, r0, c0, spec, rng)

    used_lengths = sorted({int(spec["count"]) for spec in specs})
    band_colors = rng.sample(BAND_COLORS, len(used_lengths))
    length_to_color = dict(zip(used_lengths, band_colors))
    if not _place_bands(grid, length_to_color, rng):
        return None

    transformed = main(grid)
    if transformed.shape != grid.shape:
        return None
    if np.any(transformed == GREEN):
        return None
    if np.any(transformed[0, :] != BG) or np.any(transformed[-1, :] != BG):
        return None
    if np.any(transformed[:, 0] != BG) or np.any(transformed[:, -1] != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid sample")
