from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _extract_templates(grid: np.ndarray, bg: int) -> dict[int, tuple[tuple[int, int], int, list[tuple[int, int]]]]:
    templates: dict[int, tuple[tuple[int, int], int, list[tuple[int, int]]]] = {}
    for comp in _components(grid != bg, connectivity=8):
        if len(comp) < 2:
            continue
        vals = [int(grid[r, c]) for r, c in comp]
        colors = list(set(vals))
        if len(colors) != 2:
            continue
        counts = Counter(vals)
        marker_colors = [color for color, count in counts.items() if count == 1]
        if len(marker_colors) != 1:
            continue
        marker_color = int(marker_colors[0])
        sprite_color = int(next(color for color in colors if color != marker_color))
        marker = next((r, c) for r, c in comp if int(grid[r, c]) == marker_color)
        offsets = [(r - marker[0], c - marker[1]) for r, c in comp if int(grid[r, c]) == sprite_color]
        if not offsets:
            continue
        if marker_color in templates:
            raise AssertionError("each source marker color must define exactly one sprite")
        templates[marker_color] = ((int(marker[0]), int(marker[1])), sprite_color, offsets)
    if not templates:
        raise AssertionError("expected at least one source sprite")
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each marked sprite to every standalone marker of the same color and erase the original sources.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    bg = _get_bg(g)
    out = np.full_like(g, bg)
    templates = _extract_templates(g, bg)

    for marker_color, (source_marker, sprite_color, offsets) in templates.items():
        markers = [tuple(map(int, coord)) for coord in np.argwhere(g == marker_color)]
        targets = [coord for coord in markers if coord != source_marker]
        if not targets:
            raise AssertionError("each source marker color needs at least one target marker")
        for tr, tc in targets:
            out[tr, tc] = marker_color
            for dr, dc in offsets:
                rr = tr + dr
                cc = tc + dc
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    raise AssertionError("stamped sprite must stay inside the grid")
                if int(out[rr, cc]) not in (bg, sprite_color):
                    raise AssertionError("stamped sprites must not conflict")
                out[rr, cc] = sprite_color
    return out


def _normalize_offsets(offsets: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in offsets] + [0]
    cols = [c for _, c in offsets] + [0]
    shift_r = -min(rows)
    shift_c = -min(cols)
    normalized = {(r + shift_r, c + shift_c) for r, c in offsets}
    anchor = (shift_r, shift_c)
    return sorted((r - anchor[0], c - anchor[1]) for r, c in normalized if (r, c) != anchor)


def _random_sprite_offsets() -> list[tuple[int, int]]:
    for _ in range(80):
        target_size = random.randint(1, 6)
        cells = {random.choice([d for d in DIR8 if d != (0, 0)])}
        frontier = set()
        while len(cells) < target_size:
            frontier.clear()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) != (0, 0) and -3 <= nr <= 3 and -3 <= nc <= 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        if len(cells) != target_size:
            continue
        if not any(max(abs(r), abs(c)) == 1 for r, c in cells):
            continue
        norm = _normalize_offsets(cells)
        rows = [r for r, _ in norm] + [0]
        cols = [c for _, c in norm] + [0]
        if max(rows) - min(rows) > 4 or max(cols) - min(cols) > 4:
            continue
        return norm
    raise RuntimeError("failed to sample sprite")


def _render_source_patch(bg: int, marker_color: int, sprite_color: int, offsets: list[tuple[int, int]]):
    rows = [r for r, _ in offsets] + [0]
    cols = [c for _, c in offsets] + [0]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    anchor = (-min_r, -min_c)
    patch[anchor] = marker_color
    for dr, dc in offsets:
        patch[anchor[0] + dr, anchor[1] + dc] = sprite_color
    return patch, anchor


def _reserve_with_margin(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _free_cells_for_patch(blocked: np.ndarray, patch: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = patch.shape
    filled = np.argwhere(patch != bg)
    locs: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if all(not blocked[r0 + int(dr), c0 + int(dc)] for dr, dc in filled):
                locs.append((r0, c0))
    return locs


def _free_singletons(blocked: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(~blocked)]


def _plan_output(shape: tuple[int, int], specs: list[dict], bg: int) -> np.ndarray | None:
    out = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    marker_blocked = np.zeros(shape, dtype=bool)
    for spec in specs:
        marker_color = int(spec["marker_color"])
        sprite_color = int(spec["sprite_color"])
        offsets = spec["offsets"]
        count = random.randint(1, 3)
        placed: list[tuple[int, int]] = []
        for _ in range(count):
            candidates: list[tuple[int, int]] = []
            for r in range(shape[0]):
                for c in range(shape[1]):
                    cells = [(r, c)]
                    ok = not occupied[r, c] and not marker_blocked[r, c]
                    for dr, dc in offsets:
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < shape[0] and 0 <= cc < shape[1]) or occupied[rr, cc]:
                            ok = False
                            break
                        cells.append((rr, cc))
                    if ok:
                        candidates.append((r, c))
            if not candidates:
                return None
            tr, tc = random.choice(candidates)
            placed.append((tr, tc))
            occupied[tr, tc] = True
            out[tr, tc] = marker_color
            _reserve_with_margin(marker_blocked, [(tr, tc)])
            for dr, dc in offsets:
                rr, cc = tr + dr, tc + dc
                occupied[rr, cc] = True
                out[rr, cc] = sprite_color
        spec["targets"] = placed
    return out


def _place_input(shape: tuple[int, int], specs: list[dict], distractor_colors: list[int], bg: int) -> np.ndarray | None:
    grid = np.full(shape, bg, dtype=int)
    blocked = np.zeros(shape, dtype=bool)

    for spec in specs:
        for r, c in spec["targets"]:
            if blocked[r, c]:
                return None
            grid[r, c] = spec["marker_color"]
            _reserve_with_margin(blocked, [(r, c)])

    sources = specs[:]
    random.shuffle(sources)
    for spec in sources:
        patch, anchor = _render_source_patch(bg, spec["marker_color"], spec["sprite_color"], spec["offsets"])
        locs = _free_cells_for_patch(blocked, patch, bg)
        if not locs:
            return None
        r0, c0 = random.choice(locs)
        filled_cells: list[tuple[int, int]] = []
        for dr, dc in np.argwhere(patch != bg):
            rr, cc = r0 + int(dr), c0 + int(dc)
            grid[rr, cc] = int(patch[dr, dc])
            filled_cells.append((rr, cc))
        _reserve_with_margin(blocked, filled_cells)
        spec["source_marker"] = (r0 + anchor[0], c0 + anchor[1])

    singletons: list[tuple[int, int | None, int]] = []
    for color in distractor_colors:
        for _ in range(random.randint(1, 2)):
            singletons.append((None, None, color))
    random.shuffle(singletons)

    for _, _, color in singletons:
        cells = _free_singletons(blocked)
        if not cells:
            return None
        r, c = random.choice(cells)
        grid[r, c] = color
        _reserve_with_margin(blocked, [(r, c)])

    return grid


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    bg = random.choice(PALETTE)

    n_sources = random.randint(1, 3)
    source_marker_colors = random.sample([c for c in PALETTE if c != bg], n_sources)
    remaining = [c for c in PALETTE if c not in source_marker_colors and c != bg]
    n_distractor_colors = min(len(remaining), random.choice([0, 1, 2]))
    distractor_colors = random.sample(remaining, n_distractor_colors)

    specs: list[dict] = []
    for marker_color in source_marker_colors:
        sprite_color = random.choice([c for c in PALETTE if c != bg and c != marker_color])
        specs.append(
            {
                "marker_color": marker_color,
                "sprite_color": sprite_color,
                "offsets": _random_sprite_offsets(),
            }
        )

    expected = _plan_output((h, w), specs, bg)
    if expected is None:
        return None
    inp = _place_input((h, w), specs, distractor_colors, bg)
    if inp is None:
        return None
    if _get_bg(inp) != bg:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
