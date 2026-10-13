from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GRID_LIMIT = 30
GEN_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((0, 0), dtype=bool)
    r_idx = np.flatnonzero(rows)
    c_idx = np.flatnonzero(cols)
    return mask[r_idx[0] : r_idx[-1] + 1, c_idx[0] : c_idx[-1] + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    queue: deque[tuple[int, int]] = deque([tuple(map(int, coords[0]))])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or visited[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3]))
    return components


def _pair_representations(components: list[dict]) -> dict[tuple[int, int], list[dict]]:
    pair_reps: dict[tuple[int, int], list[dict]] = {}
    for i, comp_a in enumerate(components):
        mask_a = comp_a["mask"]
        hole = ~mask_a
        hole_area = int(hole.sum())
        if hole_area == 0:
            continue
        hole_mask = _trim_mask(hole)
        if hole_mask.size == 0:
            continue
        for j, comp_b in enumerate(components):
            if i == j or comp_b["area"] != hole_area:
                continue
            if comp_b["mask"].shape != hole_mask.shape or not np.array_equal(comp_b["mask"], hole_mask):
                continue
            assembled = np.full(mask_a.shape, int(comp_a["color"]), dtype=int)
            assembled[hole] = int(comp_b["color"])
            key = (i, j) if i < j else (j, i)
            pair_reps.setdefault(key, []).append(
                {
                    "pair": key,
                    "container": i,
                    "filler": j,
                    "bbox": tuple(map(int, comp_a["bbox"])),
                    "area": int(mask_a.shape[0] * mask_a.shape[1]),
                    "assembled": assembled,
                }
            )
    for key, reps in pair_reps.items():
        reps.sort(key=lambda rep: (rep["area"], rep["bbox"][0], rep["bbox"][1], rep["container"], rep["filler"]))
    return pair_reps


def _count_matchings(pair_reps: dict[tuple[int, int], list[dict]], n_components: int, limit: int = 2) -> int:
    def rec(remaining: tuple[int, ...]) -> int:
        if not remaining:
            return 1
        i = remaining[0]
        rest = set(remaining[1:])
        total = 0
        for j in remaining[1:]:
            key = (i, j)
            if key not in pair_reps:
                continue
            next_remaining = tuple(sorted(rest - {j}))
            total += rec(next_remaining)
            if total >= limit:
                return total
        return total

    return rec(tuple(range(n_components)))


def _match_pairs(pair_reps: dict[tuple[int, int], list[dict]], n_components: int) -> list[dict]:
    def rec(remaining: tuple[int, ...]) -> list[dict] | None:
        if not remaining:
            return []
        i = remaining[0]
        rest = set(remaining[1:])
        for j in remaining[1:]:
            key = (i, j)
            if key not in pair_reps:
                continue
            matched = rec(tuple(sorted(rest - {j})))
            if matched is not None:
                return [pair_reps[key][0], *matched]
        return None

    result = rec(tuple(range(n_components)))
    if result is None:
        raise ValueError("could not find a perfect complementary pairing")
    return result


def _render_output(pairs: list[dict]) -> np.ndarray:
    ordered = sorted(pairs, key=lambda rep: (rep["area"], rep["bbox"][0], rep["bbox"][1], rep["bbox"][2], rep["bbox"][3]))
    height = max(rep["assembled"].shape[0] for rep in ordered)
    width = sum(rep["assembled"].shape[1] for rep in ordered) + len(ordered) - 1
    out = np.zeros((height, width), dtype=int)
    c = 0
    for rep in ordered:
        h, w = rep["assembled"].shape
        out[:h, c : c + w] = rep["assembled"]
        c += w + 1
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    components = _extract_components(g)
    if len(components) == 0 or len(components) % 2 != 0:
        raise ValueError("expected an even number of non-background components")
    pair_reps = _pair_representations(components)
    matched = _match_pairs(pair_reps, len(components))
    return _render_output(matched)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered colored fragments by complementary shape and pack the rebuilt rectangles by area.

    Input structure:
    - Black background.
    - An even number of scattered monochrome connected fragments, each in a different color.
    - Every true pair can be translated into a single rectangle: one fragment occupies part of a rectangle, and the other matches the missing cells inside that fragment's bounding box.

    Rule:
    - For each fragment, inspect the empty cells inside its minimal bounding rectangle.
    - If that missing-cell shape exactly matches another fragment up to translation, those two fragments form a complementary pair.
    - Rebuild each pair as the full rectangle while preserving the two original colors, then sort the completed rectangles by total area and place them left-to-right with one black separator column.

    Key insight:
    - Colors never identify partners; the pairing signal is purely geometric because one fragment's bounding-box complement is exactly the other fragment's shape.
    """
    return _solve(grid)


def _random_transform(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    out = np.rot90(out, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    return np.array(out, dtype=bool, copy=True)


def _random_connected_mask(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(120):
        h = int(rng.integers(1, max_h + 1))
        w = int(rng.integers(1, max_w + 1))
        total = h * w
        if total == 1:
            return np.ones((1, 1), dtype=bool)
        area = total if bool(rng.integers(0, 4) == 0) else int(rng.integers(1, total))
        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {seed}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return _random_transform(rng, _trim_mask(mask))
    return np.ones((1, 1), dtype=bool)


def _make_pair_spec(
    rng: np.random.Generator,
    area_seen: set[int],
    color_a: int,
    color_b: int,
    pair_count: int,
) -> dict | None:
    if pair_count >= 4:
        h_bounds = (4, 7)
        w_bounds = (4, 6)
    elif pair_count == 3:
        h_bounds = (4, 8)
        w_bounds = (4, 7)
    else:
        h_bounds = (4, 8)
        w_bounds = (4, 8)
    for _ in range(160):
        h = int(rng.integers(h_bounds[0], h_bounds[1] + 1))
        w = int(rng.integers(w_bounds[0], w_bounds[1] + 1))
        if bool(rng.integers(0, 2)):
            h, w = w, h
        area = h * w
        if area in area_seen:
            continue
        hole_mask = _random_connected_mask(rng, h - 2, w - 2)
        hole_h, hole_w = hole_mask.shape
        if hole_h > h - 2 or hole_w > w - 2:
            continue
        top = int(rng.integers(1, h - hole_h))
        left = int(rng.integers(1, w - hole_w))
        container_mask = np.ones((h, w), dtype=bool)
        hole_coords = np.argwhere(hole_mask)
        container_mask[top + hole_coords[:, 0], left + hole_coords[:, 1]] = False
        if not _is_connected(container_mask):
            continue
        assembled = np.full((h, w), color_a, dtype=int)
        assembled[top + hole_coords[:, 0], left + hole_coords[:, 1]] = color_b
        return {
            "area": area,
            "assembled": assembled,
            "components": [
                {"mask": container_mask, "color": int(color_a)},
                {"mask": np.array(hole_mask, dtype=bool, copy=True), "color": int(color_b)},
            ],
        }
    return None


def _build_expected_output(pair_specs: list[dict]) -> np.ndarray:
    ordered = sorted(pair_specs, key=lambda spec: spec["area"])
    height = max(spec["assembled"].shape[0] for spec in ordered)
    width = sum(spec["assembled"].shape[1] for spec in ordered) + len(ordered) - 1
    out = np.zeros((height, width), dtype=int)
    c = 0
    for spec in ordered:
        h, w = spec["assembled"].shape
        out[:h, c : c + w] = spec["assembled"]
        c += w + 1
    return out


def _place_components(rng: np.random.Generator, components: list[dict], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(components)), key=lambda idx: components[idx]["mask"].shape[0] * components[idx]["mask"].shape[1], reverse=True)
    for idx in order:
        mask = components[idx]["mask"]
        color = int(components[idx]["color"])
        ph, pw = mask.shape
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(0, h - ph + 1))
            c0 = int(rng.integers(0, w - pw + 1))
            if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                continue
            coords = np.argwhere(mask)
            grid[r0 + coords[:, 0], c0 + coords[:, 1]] = color
            blocked[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 5))
    colors = list(map(int, rng.choice(np.arange(1, 10), size=2 * pair_count, replace=False)))
    pair_specs: list[dict] = []
    area_seen: set[int] = set()
    for i in range(pair_count):
        spec = _make_pair_spec(rng, area_seen, colors[2 * i], colors[2 * i + 1], pair_count)
        if spec is None:
            return None
        pair_specs.append(spec)
        area_seen.add(int(spec["area"]))
    expected = _build_expected_output(pair_specs)
    if max(expected.shape) > GRID_LIMIT:
        return None
    components = [comp for spec in pair_specs for comp in spec["components"]]
    padded_area = sum((comp["mask"].shape[0] + 2) * (comp["mask"].shape[1] + 2) for comp in components)
    side = int(np.ceil(np.sqrt(padded_area)))
    h_min = max(12, max(comp["mask"].shape[0] for comp in components) + 2, side - 2)
    w_min = max(12, max(comp["mask"].shape[1] for comp in components) + 2, side)
    h_max = min(GRID_LIMIT, max(h_min, side + 5))
    w_max = min(GRID_LIMIT, max(w_min, side + 7))
    if h_min > h_max or w_min > w_max:
        return None
    grid = None
    for _ in range(80):
        grid_h = int(rng.integers(h_min, h_max + 1))
        grid_w = int(rng.integers(w_min, w_max + 1))
        grid = _place_components(rng, components, grid_h, grid_w)
        if grid is not None:
            break
    if grid is None:
        return None
    if len(_extract_components(grid)) != 2 * pair_count:
        return None
    pair_reps = _pair_representations(_extract_components(grid))
    if _count_matchings(pair_reps, 2 * pair_count, limit=2) != 1:
        return None
    try:
        solved = _solve(grid)
    except ValueError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment input")
