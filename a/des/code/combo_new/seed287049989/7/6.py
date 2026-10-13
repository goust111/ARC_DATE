from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0

# Generator configuration.
MAX_GENERATION_ATTEMPTS = 400
GROUP_COUNT_RANGE = (2, 4)
GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
RECT_HEIGHT_RANGE = (2, 7)
RECT_WIDTH_RANGE = (2, 7)
THIN_RECT_PROB = 0.35
THIN_RECT_LENGTH_RANGE = (3, 8)
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _legend_candidates(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int, int], np.ndarray]] = []
    for coords in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        if c0 != c1:
            continue
        if len(coords) < 2:
            continue
        if r1 - r0 + 1 != len(coords):
            continue
        if r1 < h // 2 or c1 < w // 2:
            continue
        rows = sorted(int(r) for r in coords[:, 0])
        if rows != list(range(r0, r1 + 1)):
            continue
        colors = [int(grid[r, c0]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        score = (r1 + c1, r1, c1, -r0, len(coords))
        candidates.append((score, coords))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [coords for _score, coords in candidates]


def _parse_scene(grid: np.ndarray) -> tuple[list[int], list[dict[str, object]]]:
    h, w = grid.shape
    for legend_coords in _legend_candidates(grid):
        legend_c = int(legend_coords[0, 1])
        legend_rows = sorted(int(r) for r in legend_coords[:, 0])
        legend_cells = {(r, legend_c) for r in legend_rows}
        legend_colors = [int(grid[r, legend_c]) for r in legend_rows]

        by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r in range(h):
            for c in range(w):
                val = int(grid[r, c])
                if val == BG or (r, c) in legend_cells:
                    continue
                by_color[val].append((r, c))

        if not by_color or len(by_color) != len(legend_colors):
            continue

        rects: list[dict[str, object]] = []
        ok = True
        areas: list[int] = []
        for color, pts in by_color.items():
            coords = np.array(pts, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            for r, c in pts:
                if r not in (r0, r1) and c not in (c0, c1):
                    ok = False
                    break
            if not ok:
                break
            area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            areas.append(area)
            rects.append({"color": int(color), "bbox": (r0, c0, r1, c1), "area": area})

        if not ok or len(set(areas)) != len(areas):
            continue

        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if _boxes_too_close(
                    rects[i]["bbox"],  # type: ignore[arg-type]
                    rects[j]["bbox"],  # type: ignore[arg-type]
                    gap=0,
                ):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        return legend_colors, rects

    raise AssertionError("could not identify the lower-right legend column")


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild hidden edge-fragment rectangles and recolor them by area rank.

    Input structure:
    - Black background with several monochrome fragment groups, one color per group.
    - Each group's pixels lie only on the border rows or border columns of that group's minimal axis-aligned bounding rectangle.
    - A small legend sits in the lower-right area as one vertical column of single-pixel color blocks ordered from top to bottom.

    Rule:
    - For each fragment color, take the minimal bounding rectangle covering all pixels of that color.
    - Compute each rectangle's area and rank the rectangles from smallest to largest.
    - Read the legend top-to-bottom as rank 1, rank 2, rank 3, ... and fill each whole rectangle, including its border, with the legend color for that rank.
    - Clear the legend area back to black in the output.

    Key insight:
    - The fragment colors only define which pixels belong together; the output color comes from the independent lower-right legend after sorting rectangles by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_colors, rects = _parse_scene(g)
    out = np.zeros_like(g)

    ranked_rects = sorted(rects, key=lambda item: int(item["area"]))
    for rank, rect in enumerate(ranked_rects):
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(legend_colors[rank])

    return out


def _sample_rect_size(rng: random.Random) -> tuple[int, int]:
    if rng.random() < THIN_RECT_PROB:
        length = rng.randint(*THIN_RECT_LENGTH_RANGE)
        if rng.random() < 0.5:
            return 1, length
        return length, 1
    return rng.randint(*RECT_HEIGHT_RANGE), rng.randint(*RECT_WIDTH_RANGE)


def _border_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]


def _sample_fragment_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    border = _border_cells(h, w)
    mandatory = {
        (0, rng.randrange(w)),
        (h - 1, rng.randrange(w)),
        (rng.randrange(h), 0),
        (rng.randrange(h), w - 1),
    }

    remaining = [cell for cell in border if cell not in mandatory]
    keep_max = len(border)
    if len(remaining) >= 2:
        keep_max -= 1
    keep = rng.randint(len(mandatory), keep_max)
    chosen = set(mandatory)
    if keep > len(mandatory):
        chosen.update(rng.sample(remaining, keep - len(mandatory)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _render_input(
    h: int,
    w: int,
    specs: list[dict[str, object]],
    legend_top: int,
    legend_col: int,
    legend_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        rh = int(spec["h"])
        rw = int(spec["w"])
        color = int(spec["color"])
        mask = spec["mask"]  # type: ignore[assignment]
        patch = grid[r0 : r0 + rh, c0 : c0 + rw]
        patch[np.asarray(mask, dtype=bool)] = color
    for offset, color in enumerate(legend_colors):
        grid[legend_top + offset, legend_col] = int(color)
    return grid


def _expected_output(h: int, w: int, specs: list[dict[str, object]], legend_colors: list[int]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    ranked = sorted(specs, key=lambda spec: int(spec["area"]))
    for rank, spec in enumerate(ranked):
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        rh = int(spec["h"])
        rw = int(spec["w"])
        out[r0 : r0 + rh, c0 : c0 + rw] = int(legend_colors[rank])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_groups = rng.randint(*GROUP_COUNT_RANGE)
    colors = rng.sample(list(range(1, 10)), 2 * n_groups)
    object_colors = colors[:n_groups]
    legend_colors = colors[n_groups:]

    h = rng.randint(max(GRID_HEIGHT_RANGE[0], n_groups + 5), GRID_HEIGHT_RANGE[1])
    w = rng.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])

    bottom_margin = rng.choice((0, 1))
    right_margin = rng.choice((0, 1))
    legend_top = h - n_groups - bottom_margin
    legend_col = w - 1 - right_margin
    if legend_top < h // 2:
        return None

    occupied = [(legend_top, legend_col, legend_top + n_groups - 1, legend_col)]
    used_areas: set[int] = set()
    specs: list[dict[str, object]] = []

    for color in object_colors:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            rh, rw = _sample_rect_size(rng)
            area = rh * rw
            if area in used_areas or rh > h or rw > w:
                continue

            r0 = rng.randint(0, h - rh)
            c0 = rng.randint(0, w - rw)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_boxes_too_close(box, other, PLACEMENT_GAP) for other in occupied):
                continue

            mask = _sample_fragment_mask(rh, rw, rng)
            specs.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(rh),
                    "w": int(rw),
                    "area": int(area),
                    "mask": mask,
                }
            )
            occupied.append(box)
            used_areas.add(area)
            placed = True
            break
        if not placed:
            return None

    grid = _render_input(h, w, specs, legend_top, legend_col, legend_colors)
    expected = _expected_output(h, w, specs, legend_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
