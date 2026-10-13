from __future__ import annotations

from collections import deque
import random

import numpy as np


_RNG = random.Random()

_COLOR_POOL = tuple(range(10))
_RETRY_LIMIT = 300

_REGION_COUNT_RANGE = (3, 5)
_GRID_HEIGHT_RANGE = (20, 28)
_X_GAP_RANGE = (2, 4)
_SIDE_MARGIN_RANGE = (1, 3)
_TOP_MARGIN = 1
_RECT_H_RANGE = (4, 7)
_RECT_W_RANGE = (4, 6)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _anchor(coords: np.ndarray) -> tuple[int, int]:
    rows = coords[:, 0]
    min_row = int(rows.min())
    min_col = int(coords[rows == min_row, 1].min())
    return min_row, min_col


def _extract_regions(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    bg = _background_color(grid)
    regions: list[dict[str, int | tuple[int, int]]] = []
    for coords in _components(grid != bg):
        values = grid[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        order = np.argsort(-counts)
        dominant_count = int(counts[order[0]])
        if len(order) > 1:
            assert dominant_count > int(counts[order[1]]), "each region must have a unique dominant color"
        dominant_color = int(colors[order[0]])
        regions.append(
            {
                "color": dominant_color,
                "area": dominant_count,
                "intrusions": int(len(values) - dominant_count),
                "anchor": _anchor(coords),
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort foreground regions by how many foreign pixels invade them, then redraw them
    as bottom-aligned vertical bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height = int(g.shape[0])
    bg = _background_color(g)

    regions = _extract_regions(g)
    assert regions, "expected at least one foreground region"

    regions.sort(key=lambda region: (int(region["intrusions"]), *region["anchor"]))

    out = np.full((height, len(regions)), bg, dtype=int)
    for col, region in enumerate(regions):
        area = int(region["area"])
        assert 0 < area <= height, "bar height must fit inside the output grid"
        out[height - area :, col] = int(region["color"])
    return out


def _pick_dims(rng: random.Random, height: int, intrusions: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for rect_h in range(_RECT_H_RANGE[0], _RECT_H_RANGE[1] + 1):
        for rect_w in range(_RECT_W_RANGE[0], _RECT_W_RANGE[1] + 1):
            interior = (rect_h - 2) * (rect_w - 2)
            dominant_area = rect_h * rect_w - intrusions
            if interior >= intrusions and 8 <= dominant_area <= height - 1:
                candidates.append((rect_h, rect_w))
    assert candidates, "expected at least one valid rectangle shape"

    style = rng.choice(("tall", "wide", "balanced"))
    weighted: list[tuple[int, int]] = []
    for rect_h, rect_w in candidates:
        repeats = 1
        if style == "tall" and rect_h > rect_w:
            repeats = 3
        elif style == "wide" and rect_w > rect_h:
            repeats = 3
        elif style == "balanced" and abs(rect_h - rect_w) <= 1:
            repeats = 3
        weighted.extend([(rect_h, rect_w)] * repeats)
    return rng.choice(weighted)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(_REGION_COUNT_RANGE[0], _REGION_COUNT_RANGE[1])
    intrusion_counts = rng.sample([1, 2, 3, 4, 5], n_regions)

    height = rng.randint(_GRID_HEIGHT_RANGE[0], _GRID_HEIGHT_RANGE[1])
    palette = rng.sample(_COLOR_POOL, n_regions + 1)
    bg = palette[0]
    colors = palette[1:]

    specs: list[dict[str, int]] = []
    for color, intrusions in zip(colors, intrusion_counts):
        rect_h, rect_w = _pick_dims(rng, height, intrusions)
        specs.append({"color": color, "intrusions": intrusions, "h": rect_h, "w": rect_w})

    order = list(range(n_regions))
    rng.shuffle(order)
    placed = [specs[i] for i in order]

    left_margin = rng.randint(_SIDE_MARGIN_RANGE[0], _SIDE_MARGIN_RANGE[1])
    right_margin = rng.randint(_SIDE_MARGIN_RANGE[0], _SIDE_MARGIN_RANGE[1])
    gaps = [rng.randint(_X_GAP_RANGE[0], _X_GAP_RANGE[1]) for _ in range(max(0, n_regions - 1))]
    width = left_margin + right_margin + sum(spec["w"] for spec in placed) + sum(gaps)

    grid = np.full((height, width), bg, dtype=int)

    x = left_margin
    placements: list[dict[str, int]] = []
    for idx, spec in enumerate(placed):
        max_top = height - spec["h"] - _TOP_MARGIN
        if max_top < _TOP_MARGIN:
            return None
        top = rng.randint(_TOP_MARGIN, max_top)
        grid[top : top + spec["h"], x : x + spec["w"]] = int(spec["color"])
        placement = dict(spec)
        placement["top"] = top
        placement["left"] = x
        placements.append(placement)
        x += spec["w"]
        if idx < len(gaps):
            x += gaps[idx]

    for idx, spec in enumerate(placements):
        top = spec["top"]
        left = spec["left"]
        rect_h = spec["h"]
        rect_w = spec["w"]
        interior = [(top + dr, left + dc) for dr in range(1, rect_h - 1) for dc in range(1, rect_w - 1)]
        if len(interior) < spec["intrusions"]:
            return None

        if len(placements) == 1:
            neighbor_colors = [p["color"] for p in placements if p["color"] != spec["color"]]
        elif idx == 0:
            neighbor_colors = [placements[1]["color"]]
        elif idx == len(placements) - 1:
            neighbor_colors = [placements[-2]["color"]]
        else:
            neighbor_colors = [placements[idx - 1]["color"], placements[idx + 1]["color"]]
        if not neighbor_colors:
            return None

        intrusion_cells = rng.sample(interior, spec["intrusions"])
        for cell in intrusion_cells:
            grid[cell] = int(rng.choice(neighbor_colors))

    regions = _extract_regions(grid)
    if len(regions) != n_regions:
        return None
    intrusion_set = {int(region["intrusions"]) for region in regions}
    if intrusion_set != set(intrusion_counts):
        return None
    if len(intrusion_set) != n_regions:
        return None
    if any(int(region["area"]) > height for region in regions):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
