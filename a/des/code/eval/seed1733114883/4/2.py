import numpy as np
from collections import deque


_RNG = np.random.default_rng()

# Generator configuration.
PAIR_COUNT_RANGE = (2, 4)
RECT_DIM_MIN = 3
RECT_DIM_MAX = 6
GRID_DIM_MIN = 18
GRID_DIM_MAX = 30
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = list(cells)
    assert pts
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, c0 = min(rows), min(cols)
    norm = tuple(sorted((r - r0, c - c0) for r, c in pts))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return h, w, norm


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, c0 = min(rows), min(cols)
            r1, c1 = max(rows), max(cols)
            comp_h = r1 - r0 + 1
            comp_w = c1 - c0 + 1
            norm = tuple(sorted((rr - r0, cc - c0) for rr, cc in cells))
            comps.append(
                {
                    "id": len(comps),
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "h": comp_h,
                    "w": comp_w,
                    "area": len(cells),
                    "norm": norm,
                    "sig": (comp_h, comp_w, norm),
                }
            )
    return comps


def _build_pair_options(components: list[dict]) -> list[dict]:
    by_sig: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[int]] = {}
    for comp in components:
        by_sig.setdefault(comp["sig"], []).append(comp["id"])

    options: list[dict] = []
    for big in components:
        rect_h = int(big["h"])
        rect_w = int(big["w"])
        rect_area = rect_h * rect_w
        missing_area = rect_area - int(big["area"])
        if missing_area <= 0 or missing_area >= int(big["area"]):
            continue

        big_cells = set(big["norm"])
        missing_cells = [(r, c) for r in range(rect_h) for c in range(rect_w) if (r, c) not in big_cells]
        if not missing_cells:
            continue
        missing_sig = _normalize_cells(missing_cells)
        for small_id in by_sig.get(missing_sig, []):
            if small_id == int(big["id"]):
                continue
            small = components[small_id]
            if int(small["area"]) != missing_area:
                continue
            if int(small["color"]) == int(big["color"]):
                continue
            r0a, c0a, _r1a, _c1a = big["bbox"]
            r0b, c0b, _r1b, _c1b = small["bbox"]
            options.append(
                {
                    "big": int(big["id"]),
                    "small": int(small_id),
                    "h": rect_h,
                    "w": rect_w,
                    "area": rect_area,
                    "big_color": int(big["color"]),
                    "small_color": int(small["color"]),
                    "missing_cells": tuple(sorted(missing_cells)),
                    "anchor": (min(r0a, r0b), min(c0a, c0b)),
                }
            )
    options.sort(key=lambda opt: (opt["area"], opt["anchor"][1], opt["anchor"][0], opt["big"], opt["small"]))
    return options


def _solve_pairing(components: list[dict]) -> list[dict]:
    assert len(components) % 2 == 0 and components
    options = _build_pair_options(components)
    by_component = {int(comp["id"]): [] for comp in components}
    for idx, opt in enumerate(options):
        by_component[opt["big"]].append(idx)
        by_component[opt["small"]].append(idx)

    memo: dict[tuple[int, ...], list[dict] | None] = {}

    def dfs(unused: tuple[int, ...]) -> list[dict] | None:
        if not unused:
            return []
        if unused in memo:
            return memo[unused]
        remaining = set(unused)
        pivot = unused[0]
        for opt_idx in by_component[pivot]:
            opt = options[opt_idx]
            if opt["big"] not in remaining or opt["small"] not in remaining:
                continue
            next_unused = tuple(x for x in unused if x not in (opt["big"], opt["small"]))
            tail = dfs(next_unused)
            if tail is not None:
                memo[unused] = [opt] + tail
                return memo[unused]
        memo[unused] = None
        return None

    comp_ids = tuple(sorted(int(comp["id"]) for comp in components))
    solution = dfs(comp_ids)
    assert solution is not None and 2 * len(solution) == len(components)
    return solution


def _render_output_from_pairs(pairs: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(pairs, key=lambda opt: (opt["area"], opt["anchor"][1], opt["anchor"][0]))
    out_h = max(int(opt["h"]) for opt in ordered)
    out_w = sum(int(opt["w"]) for opt in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c0 = 0
    for opt in ordered:
        rect = np.full((int(opt["h"]), int(opt["w"])), int(opt["big_color"]), dtype=int)
        for r, c in opt["missing_cells"]:
            rect[r, c] = int(opt["small_color"])
        out[: int(opt["h"]), c0 : c0 + int(opt["w"])] = rect
        c0 += int(opt["w"]) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments into rectangles, then sort the reconstructed rectangles by area.

    Input structure
    - The most common color is background.
    - The foreground consists of an even number of single-color 4-connected fragments.
    - The fragments can be paired so that each pair exactly tiles a solid rectangle.
    - In every pair, the larger fragment already has the full target rectangle as its bbox.

    Rule
    - Detect the complementary pairing by comparing each large fragment with the missing cells
      inside its bbox.
    - Reconstruct each pair as a solid two-color rectangle: use the larger fragment's color
      everywhere except on the cells contributed by the smaller fragment.
    - Sort the reconstructed rectangles by area ascending and place them left-to-right with
      one background column between consecutive rectangles, all top-aligned.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _find_components(g, bg)
    pairs = _solve_pairing(components)
    return _render_output_from_pairs(pairs, bg)


def _transform_corner_cells(
    local_cells: set[tuple[int, int]],
    rect_h: int,
    rect_w: int,
    corner: str,
) -> set[tuple[int, int]]:
    if corner == "tl":
        return set(local_cells)
    if corner == "tr":
        return {(r, rect_w - 1 - c) for r, c in local_cells}
    if corner == "bl":
        return {(rect_h - 1 - r, c) for r, c in local_cells}
    if corner == "br":
        return {(rect_h - 1 - r, rect_w - 1 - c) for r, c in local_cells}
    raise ValueError(corner)


def _random_growth(window_h: int, window_w: int, target_area: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < window_h and 0 <= nc < window_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        pick = list(frontier)[int(_RNG.integers(len(frontier)))]
        cells.add(pick)
    return cells


def _generate_local_cutout(rect_h: int, rect_w: int) -> set[tuple[int, int]] | None:
    rect_area = rect_h * rect_w
    max_small_area = rect_area // 2 - 1
    if max_small_area < 2:
        return None

    families = ("block", "stair", "growth")
    family = families[int(_RNG.integers(len(families)))]
    for _ in range(80):
        if family == "block":
            cut_h = int(_RNG.integers(1, rect_h))
            cut_w = int(_RNG.integers(1, rect_w))
            local_cells = {(r, c) for r in range(cut_h) for c in range(cut_w)}
        elif family == "stair":
            cut_h = int(_RNG.integers(2, rect_h))
            cut_w = int(_RNG.integers(2, rect_w))
            lengths = [cut_w]
            dropped = False
            for _row in range(1, cut_h):
                hi = lengths[-1]
                lo = 1
                nxt = int(_RNG.integers(lo, hi + 1))
                if nxt < hi:
                    dropped = True
                lengths.append(nxt)
            if not dropped:
                continue
            local_cells = {(r, c) for r, width in enumerate(lengths) for c in range(width)}
        else:
            cut_h = int(_RNG.integers(2, rect_h))
            cut_w = int(_RNG.integers(2, rect_w))
            upper = min(cut_h * cut_w, max_small_area)
            if upper < 2:
                continue
            target_area = int(_RNG.integers(2, upper + 1))
            local_cells = _random_growth(cut_h, cut_w, target_area)
            if local_cells is None:
                continue

        if len(local_cells) < 2 or len(local_cells) > max_small_area:
            continue
        return local_cells
    return None


def _generate_pair_spec(big_color: int, small_color: int) -> dict | None:
    rect_h = int(_RNG.integers(RECT_DIM_MIN, RECT_DIM_MAX + 1))
    rect_w = int(_RNG.integers(RECT_DIM_MIN, RECT_DIM_MAX + 1))
    local_cutout = _generate_local_cutout(rect_h, rect_w)
    if local_cutout is None:
        return None

    corner = ("tl", "tr", "bl", "br")[int(_RNG.integers(4))]
    small_cells = _transform_corner_cells(local_cutout, rect_h, rect_w, corner)
    rect_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
    big_cells = rect_cells - small_cells
    if len(big_cells) <= len(small_cells):
        return None
    if not _connected(small_cells) or not _connected(big_cells):
        return None

    big_sig = _normalize_cells(big_cells)
    if big_sig[0] != rect_h or big_sig[1] != rect_w:
        return None

    return {
        "rect_h": rect_h,
        "rect_w": rect_w,
        "area": rect_h * rect_w,
        "big_color": int(big_color),
        "small_color": int(small_color),
        "big_cells": tuple(sorted(big_cells)),
        "small_cells": tuple(sorted(small_cells)),
    }


def _mask_from_cells(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w, norm = _normalize_cells(list(cells))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _place_components(
    grid_h: int,
    grid_w: int,
    bg: int,
    pair_specs: list[dict],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[dict] = []

    pieces: list[dict] = []
    for pair_idx, spec in enumerate(pair_specs):
        pieces.append(
            {
                "pair_idx": pair_idx,
                "role": "big",
                "color": int(spec["big_color"]),
                "mask": _mask_from_cells(spec["big_cells"]),
            }
        )
        pieces.append(
            {
                "pair_idx": pair_idx,
                "role": "small",
                "color": int(spec["small_color"]),
                "mask": _mask_from_cells(spec["small_cells"]),
            }
        )
    _RNG.shuffle(pieces)

    for piece in sorted(pieces, key=lambda item: item["mask"].size, reverse=True):
        mask = piece["mask"]
        h, w = mask.shape
        placed = False
        for _ in range(180):
            r0 = int(_RNG.integers(0, grid_h - h + 1))
            c0 = int(_RNG.integers(0, grid_w - w + 1))
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if forbidden[rr0:rr1, cc0:cc1].any():
                continue
            rows, cols = np.nonzero(mask)
            grid[r0 + rows, c0 + cols] = int(piece["color"])
            forbidden[rr0:rr1, cc0:cc1] = True
            placements.append(
                {
                    "pair_idx": int(piece["pair_idx"]),
                    "role": piece["role"],
                    "top": r0,
                    "left": c0,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return grid, placements


def _expected_output(pair_specs: list[dict], placements: list[dict], bg: int) -> np.ndarray:
    anchors: dict[int, tuple[int, int]] = {}
    for pair_idx in range(len(pair_specs)):
        pair_places = [p for p in placements if int(p["pair_idx"]) == pair_idx]
        top = min(int(p["top"]) for p in pair_places)
        left = min(int(p["left"]) for p in pair_places)
        anchors[pair_idx] = (top, left)

    pairs = []
    for pair_idx, spec in enumerate(pair_specs):
        pairs.append(
            {
                "h": int(spec["rect_h"]),
                "w": int(spec["rect_w"]),
                "area": int(spec["area"]),
                "big_color": int(spec["big_color"]),
                "small_color": int(spec["small_color"]),
                "missing_cells": tuple(spec["small_cells"]),
                "anchor": anchors[pair_idx],
            }
        )
    return _render_output_from_pairs(pairs, bg)


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.integers(10))
    palette = [c for c in range(10) if c != bg]
    pair_count = int(_RNG.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    if 2 * pair_count > len(palette):
        return None

    _RNG.shuffle(palette)
    pair_specs: list[dict] = []
    used_areas: set[int] = set()
    for pair_idx in range(pair_count):
        big_color = palette[2 * pair_idx]
        small_color = palette[2 * pair_idx + 1]
        spec = None
        for _ in range(80):
            candidate = _generate_pair_spec(big_color, small_color)
            if candidate is None:
                continue
            if int(candidate["area"]) in used_areas:
                continue
            spec = candidate
            break
        if spec is None:
            return None
        used_areas.add(int(spec["area"]))
        pair_specs.append(spec)

    total_colored = sum(int(spec["area"]) for spec in pair_specs)
    max_piece_area = max(max(len(spec["big_cells"]), len(spec["small_cells"])) for spec in pair_specs)

    for _ in range(60):
        min_side = max(GRID_DIM_MIN, max(spec["rect_h"] for spec in pair_specs) + 4, max(spec["rect_w"] for spec in pair_specs) + 4)
        grid_h = int(_RNG.integers(min_side, GRID_DIM_MAX + 1))
        grid_w = int(_RNG.integers(min_side, GRID_DIM_MAX + 1))
        if grid_h * grid_w - total_colored <= max_piece_area:
            continue
        placed = _place_components(grid_h, grid_w, bg, pair_specs)
        if placed is None:
            continue
        grid, placements = placed
        expected = _expected_output(pair_specs, placements, bg)
        if expected.shape[1] > GRID_DIM_MAX:
            continue
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
