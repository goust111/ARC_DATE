from __future__ import annotations

from collections import deque

import numpy as np


BG = 8
COLOR_POOL = tuple(color for color in range(10) if color != BG)

GRID_MIN = 18
GRID_MAX = 30
INNER_MIN = 3
INNER_MAX = 8
MAX_STRUCTURES = 4
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "background color must be uniquely most common"
    return int(winners[0])


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

    return components


def _frame_from_component(component: dict) -> dict | None:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    expected_area = 2 * (h + w) - 4
    if int(component["area"]) != expected_area:
        return None

    for r, c in component["coords"]:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    return {"color": int(component["color"]), "bbox": (r0, c0, r1, c1)}


def _pair_matches(grid: np.ndarray, bg: int, outer: dict, inner: dict) -> bool:
    if int(outer["color"]) == int(inner["color"]):
        return False

    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    if (ir0, ic0, ir1, ic1) != (or0 + 2, oc0 + 2, or1 - 2, oc1 - 2):
        return False

    sub = grid[or0 : or1 + 1, oc0 : oc1 + 1]
    expected = np.full(sub.shape, bg, dtype=int)
    _paint_frame(expected, (0, 0, sub.shape[0] - 1, sub.shape[1] - 1), int(outer["color"]))
    _paint_frame(expected, (2, 2, sub.shape[0] - 3, sub.shape[1] - 3), int(inner["color"]))
    return np.array_equal(sub, expected)


def _parse_structures(grid: np.ndarray) -> tuple[int, list[dict], list[tuple[int, int]]]:
    bg = _background_color(grid)
    components = _monochrome_components(grid, bg)
    assert components, "expected at least one nested-frame structure"

    frames: list[dict] = []
    for component in components:
        frame = _frame_from_component(component)
        assert frame is not None, "every foreground component must be a one-pixel rectangular frame"
        frames.append(frame)

    bbox_to_idx = {frame["bbox"]: idx for idx, frame in enumerate(frames)}
    pairs: list[tuple[int, int]] = []
    outer_ids: set[int] = set()
    inner_ids: set[int] = set()

    for idx, outer in enumerate(frames):
        r0, c0, r1, c1 = outer["bbox"]
        inner_idx = bbox_to_idx.get((r0 + 2, c0 + 2, r1 - 2, c1 - 2))
        if inner_idx is None:
            continue
        if _pair_matches(grid, bg, outer, frames[inner_idx]):
            pairs.append((idx, inner_idx))
            outer_ids.add(idx)
            inner_ids.add(inner_idx)

    assert pairs, "expected at least one outer/inner frame pair"
    assert len(pairs) * 2 == len(frames), "every frame must belong to exactly one double-frame structure"
    assert len(outer_ids) == len(pairs) and len(inner_ids) == len(pairs)
    assert outer_ids.isdisjoint(inner_ids), "a frame cannot be both outer and inner"
    return bg, frames, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the colors of each double nested frame while keeping both frame positions fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, frames, pairs = _parse_structures(g)
    out = g.copy()
    for outer_idx, inner_idx in pairs:
        outer = frames[outer_idx]
        inner = frames[inner_idx]
        _paint_frame(out, outer["bbox"], int(inner["color"]))
        _paint_frame(out, inner["bbox"], int(outer["color"]))
    return out


def _choose_canvas(rng: np.random.Generator) -> tuple[int, int]:
    rows = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    if rng.random() < 0.65:
        delta = int(rng.integers(-4, 5))
        cols = int(np.clip(rows + delta, GRID_MIN, GRID_MAX))
    else:
        cols = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    return rows, cols


def _sample_color_pairs(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    if rng.random() < 0.55:
        sampled = [int(x) for x in rng.choice(np.array(COLOR_POOL), size=2 * count, replace=False)]
        return [(sampled[2 * i], sampled[2 * i + 1]) for i in range(count)]

    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        outer = int(rng.choice(COLOR_POOL))
        inner_choices = [color for color in COLOR_POOL if color != outer]
        inner = int(rng.choice(inner_choices))
        pairs.append((outer, inner))
    return pairs


def _sample_specs(rng: np.random.Generator, count: int) -> list[dict]:
    color_pairs = _sample_color_pairs(rng, count)
    specs: list[dict] = []

    for outer_color, inner_color in color_pairs:
        inner_h = int(rng.integers(INNER_MIN, INNER_MAX + 1))
        inner_w = int(rng.integers(INNER_MIN, INNER_MAX + 1))
        specs.append(
            {
                "outer_h": inner_h + 4,
                "outer_w": inner_w + 4,
                "outer_color": outer_color,
                "inner_color": inner_color,
            }
        )

    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    return specs


def _expanded_bounds(
    rows: int,
    cols: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    padding: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - padding),
        max(0, c0 - padding),
        min(rows, r0 + h + padding),
        min(cols, c0 + w + padding),
    )


def _placement_candidates(
    reserved: np.ndarray,
    h: int,
    w: int,
    padding: int,
) -> list[tuple[int, int]]:
    rows, cols = reserved.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(rows - h + 1):
        for c0 in range(cols - w + 1):
            rr0, cc0, rr1, cc1 = _expanded_bounds(rows, cols, r0, c0, h, w, padding)
            if not reserved[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    return candidates


def _draw_structure(
    grid: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    outer_color: int,
    inner_color: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = outer_bbox
    inner_bbox = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    _paint_frame(grid, outer_bbox, outer_color)
    _paint_frame(grid, inner_bbox, inner_color)
    return inner_bbox


def _place_structures(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    specs: list[dict],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((rows, cols), BG, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)
    placements: list[dict] = []

    for spec in specs:
        h = int(spec["outer_h"])
        w = int(spec["outer_w"])
        candidates = _placement_candidates(reserved, h, w, PLACEMENT_PADDING)
        if not candidates:
            return None

        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        rr0, cc0, rr1, cc1 = _expanded_bounds(rows, cols, r0, c0, h, w, PLACEMENT_PADDING)
        reserved[rr0:rr1, cc0:cc1] = True

        outer_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        inner_bbox = _draw_structure(
            grid,
            outer_bbox,
            int(spec["outer_color"]),
            int(spec["inner_color"]),
        )
        placements.append(
            {
                "outer_bbox": outer_bbox,
                "inner_bbox": inner_bbox,
                "outer_color": int(spec["outer_color"]),
                "inner_color": int(spec["inner_color"]),
            }
        )

    return grid, placements


def _expected_output(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        _paint_frame(out, placement["outer_bbox"], int(placement["inner_color"]))
        _paint_frame(out, placement["inner_bbox"], int(placement["outer_color"]))
    return out


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    rows, cols = _choose_canvas(rng)
    max_structures = min(MAX_STRUCTURES, max(1, (rows * cols) // 150))
    structure_count = int(rng.integers(1, max_structures + 1))
    specs = _sample_specs(rng, structure_count)
    placed = _place_structures(rng, rows, cols, specs)
    if placed is None:
        return None

    grid, placements = placed
    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid != BG):
        return None

    expected = _expected_output(grid, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate(_RNG)
        if candidate is not None:
            return candidate
    raise RuntimeError(
        f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts"
    )
