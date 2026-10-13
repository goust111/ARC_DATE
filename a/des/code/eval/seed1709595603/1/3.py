from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SYMBOL = 5
SEED_COLOR = 9
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ROT = "rot"
HFLIP = "hflip"
VFLIP = "vflip"


def _most_common_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _arrow_offsets(direction: str) -> set[tuple[int, int]]:
    if direction == "U":
        return {(-1, 0), (1, 0), (-1, -1), (-1, 1), (0, 0)}
    if direction == "D":
        return {(-1, 0), (1, 0), (1, -1), (1, 1), (0, 0)}
    if direction == "L":
        return {(0, -1), (0, 1), (-1, -1), (1, -1), (0, 0)}
    if direction == "R":
        return {(0, -1), (0, 1), (-1, 1), (1, 1), (0, 0)}
    raise ValueError(direction)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _match_arrow(coords: set[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    if len(coords) != 5:
        return None
    for center in coords:
        cr, cc = center
        rel = {(r - cr, c - cc) for r, c in coords}
        for direction in DIRS:
            if rel == _arrow_offsets(direction):
                return direction, center
    return None


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(grid, bg):
        matched = _match_arrow(comp["coords"])
        if matched is None:
            continue
        direction, center = matched
        arrows.append(
            {
                "color": int(comp["color"]),
                "coords": set(comp["coords"]),
                "direction": direction,
                "center": center,
            }
        )
    arrows.sort(key=lambda a: a["center"])
    return arrows


def _symbol_kind(mask: set[tuple[int, int]]) -> str | None:
    if mask == {(0, 0), (1, 0), (1, 1)}:
        return ROT
    if mask == {(0, 0), (1, 0), (0, 2), (1, 2)}:
        return HFLIP
    if mask == {(0, 0), (0, 1), (2, 0), (2, 1)}:
        return VFLIP
    return None


def _parse_legend(grid: np.ndarray, bg: int, arrow_colors: set[int]) -> tuple[dict[int, str], set[tuple[int, int]]]:
    h, w = grid.shape
    mapping: dict[int, str] = {}
    legend_cells: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(max(0, w - 10), w - 4):
            color = int(grid[r, c])
            if color == bg or color not in arrow_colors:
                continue
            top = r - 1
            left = c + 2
            if top < 0 or left + 2 >= w or top + 2 >= h:
                continue
            cells: set[tuple[int, int]] = set()
            for rr in range(3):
                for cc in range(3):
                    if int(grid[top + rr, left + cc]) != bg:
                        cells.add((rr, cc))
            kind = _symbol_kind(cells)
            if kind is None:
                continue
            mapping[color] = kind
            legend_cells.add((r, c))
            for rr, cc in cells:
                legend_cells.add((top + rr, left + cc))
    return mapping, legend_cells


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in b:
                    return True
    return False


def _seed_pattern(grid: np.ndarray, bg: int, forbidden: set[tuple[int, int]], arrows: list[dict]) -> tuple[np.ndarray, int, set[tuple[int, int]]]:
    arrow_cells = set().union(*(a["coords"] for a in arrows)) if arrows else set()
    candidates: list[dict] = []
    for comp in _components(grid, bg):
        coords = set(comp["coords"])
        if coords & forbidden:
            continue
        if _touches(coords, arrow_cells):
            candidates.append(comp)
    if not candidates:
        for comp in _components(grid, bg):
            coords = set(comp["coords"])
            if not (coords & forbidden) and len(coords) > 1:
                candidates.append(comp)
    assert candidates, "seed pattern not found"
    comp = max(candidates, key=lambda item: len(item["coords"]))
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = int(comp["color"])
    return patch, int(comp["color"]), set(comp["coords"])


def _apply_transform(patch: np.ndarray, op: str) -> np.ndarray:
    if op == ROT:
        return np.rot90(patch, k=3)
    if op == HFLIP:
        return np.fliplr(patch)
    if op == VFLIP:
        return np.flipud(patch)
    return patch


def _next_arrow(current_idx: int, arrows: list[dict], coord_to_arrow: dict[tuple[int, int], int], shape: tuple[int, int]) -> int | None:
    h, w = shape
    arrow = arrows[current_idx]
    dr, dc = DIRS[arrow["direction"]]
    r, c = arrow["center"]
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        idx = coord_to_arrow.get((r, c))
        if idx is not None and idx != current_idx:
            return idx
        r += dr
        c += dc
    return None


def _paint_at_edge(out: np.ndarray, patch: np.ndarray, center: tuple[int, int], direction: str, bg: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    cr, cc = center
    if direction == "R":
        r0 = max(0, min(h - ph, cr - ph // 2))
        c0 = w - pw
    elif direction == "L":
        r0 = max(0, min(h - ph, cr - ph // 2))
        c0 = 0
    elif direction == "D":
        r0 = h - ph
        c0 = max(0, min(w - pw, cc - pw // 2))
    else:
        r0 = 0
        c0 = max(0, min(w - pw, cc - pw // 2))
    for rr in range(ph):
        for cc2 in range(pw):
            if int(patch[rr, cc2]) != bg:
                out[r0 + rr, c0 + cc2] = int(patch[rr, cc2])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _most_common_color(g)
    arrows = _find_arrows(g, bg)
    if not arrows:
        return out

    arrow_colors = {int(a["color"]) for a in arrows}
    legend, legend_cells = _parse_legend(g, bg, arrow_colors)
    arrow_cells = set().union(*(a["coords"] for a in arrows))
    patch, _, seed_cells = _seed_pattern(g, bg, arrow_cells | legend_cells, arrows)

    first = None
    for idx, arrow in enumerate(arrows):
        if _touches(seed_cells, arrow["coords"]):
            first = idx
            break
    if first is None:
        return out

    coord_to_arrow: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for coord in arrow["coords"]:
            coord_to_arrow[coord] = idx

    current = first
    seen = {current}
    while True:
        nxt = _next_arrow(current, arrows, coord_to_arrow, g.shape)
        if nxt is None or nxt in seen:
            break
        color = int(arrows[nxt]["color"])
        if color in legend:
            patch = _apply_transform(patch, legend[color])
        current = nxt
        seen.add(current)

    _paint_at_edge(out, patch, arrows[current]["center"], arrows[current]["direction"], bg)
    return out


def _draw_arrow(grid: np.ndarray, center: tuple[int, int], direction: str, color: int) -> None:
    cr, cc = center
    for dr, dc in _arrow_offsets(direction):
        grid[cr + dr, cc + dc] = int(color)


def _draw_symbol(grid: np.ndarray, top: int, left: int, op: str) -> None:
    if op == ROT:
        cells = {(0, 0), (1, 0), (1, 1)}
    elif op == HFLIP:
        cells = {(0, 0), (1, 0), (0, 2), (1, 2)}
    else:
        cells = {(0, 0), (0, 1), (2, 0), (2, 1)}
    for dr, dc in cells:
        grid[top + dr, left + dc] = SYMBOL


def _draw_legend(grid: np.ndarray, mapping: dict[int, str]) -> None:
    c0 = grid.shape[1] - 8
    for idx, (color, op) in enumerate(mapping.items()):
        top = 1 + idx * 4
        grid[top + 1, c0] = int(color)
        _draw_symbol(grid, top, c0 + 2, op)


SEED_MASKS = [
    np.array([[1, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool),
]


def _place_seed(grid: np.ndarray) -> np.ndarray:
    mask = random.choice(SEED_MASKS)
    top, left = 3, 2
    grid[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = SEED_COLOR
    return np.where(mask, SEED_COLOR, BG)


def _in_bounds_center(center: tuple[int, int], h: int, w: int) -> bool:
    r, c = center
    return 2 <= r < h - 2 and 2 <= c < w - 10


def _arrow_cells(center: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in _arrow_offsets(direction)}


def _ray_clear(start: tuple[int, int], direction: str, target: tuple[int, int] | None, occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    dr, dc = DIRS[direction]
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if target is not None and (r, c) == target:
            return True
        if (r, c) in occupied:
            return False
        r += dr
        c += dc
    return target is None


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    if a[0] == b[0]:
        return "R" if b[1] > a[1] else "L"
    if a[1] == b[1]:
        return "D" if b[0] > a[0] else "U"
    return None


def _edge_origin(shape: tuple[int, int], patch_shape: tuple[int, int], center: tuple[int, int], direction: str) -> tuple[int, int]:
    h, w = shape
    ph, pw = patch_shape
    cr, cc = center
    if direction == "R":
        return max(0, min(h - ph, cr - ph // 2)), w - pw
    if direction == "L":
        return max(0, min(h - ph, cr - ph // 2)), 0
    if direction == "D":
        return h - ph, max(0, min(w - pw, cc - pw // 2))
    return 0, max(0, min(w - pw, cc - pw // 2))


def _edge_area_free(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int], direction: str) -> bool:
    r0, c0 = _edge_origin(grid.shape, patch.shape, center, direction)
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            if patch[r, c] != BG and grid[r0 + r, c0 + c] != BG:
                return False
    return True


def _make_centers(h: int, w: int, chain_len: int) -> list[tuple[int, int]] | None:
    centers = [(4, 6)]
    for _ in range(chain_len - 1):
        cur = centers[-1]
        options: list[tuple[int, int]] = []
        for direction, (dr, dc) in DIRS.items():
            distances = list(range(6, 13))
            random.shuffle(distances)
            for dist in distances:
                cand = (cur[0] + dr * dist, cur[1] + dc * dist)
                if not _in_bounds_center(cand, h, w):
                    continue
                if any(abs(cand[0] - r) <= 3 and abs(cand[1] - c) <= 3 for r, c in centers):
                    continue
                options.append(cand)
        random.shuffle(options)
        if not options:
            return None
        centers.append(options[0])
    return centers


def _build_attempt() -> np.ndarray | None:
    h = random.randint(24, 34)
    w = random.randint(28, 36)
    grid = np.full((h, w), BG, dtype=int)

    colors = random.sample([1, 2, 3, 4, 6, 8], 3)
    ops = [ROT, HFLIP, VFLIP]
    random.shuffle(ops)
    mapping = dict(zip(colors, ops))
    _draw_legend(grid, mapping)
    seed_patch = _place_seed(grid)

    chain_len = random.randint(3, 6)
    centers = _make_centers(h, w, chain_len)
    if centers is None:
        return None

    directions: list[str] = []
    for a, b in zip(centers, centers[1:]):
        direction = _direction_between(a, b)
        if direction is None:
            return None
        directions.append(direction)

    patch = seed_patch.copy()
    arrow_colors = [random.choice(colors) for _ in centers]
    for color in arrow_colors[1:]:
        patch = _apply_transform(patch, mapping[color])

    occupied_prior: set[tuple[int, int]] = set()
    for center, direction in zip(centers[:-1], directions):
        occupied_prior |= _arrow_cells(center, direction)

    last_dir_options = list(DIRS)
    random.shuffle(last_dir_options)
    last_dir = None
    for direction in last_dir_options:
        if not _ray_clear(centers[-1], direction, None, occupied_prior, grid.shape):
            continue
        if _edge_area_free(grid, patch, centers[-1], direction):
            last_dir = direction
            break
    if last_dir is None:
        return None
    directions.append(last_dir)

    occupied: set[tuple[int, int]] = set()
    for center, direction, color in zip(centers, directions, arrow_colors):
        cells = _arrow_cells(center, direction)
        if occupied & cells:
            return None
        occupied |= cells
        _draw_arrow(grid, center, direction, color)

    for idx in range(len(centers) - 1):
        blockers = occupied - _arrow_cells(centers[idx], directions[idx]) - _arrow_cells(centers[idx + 1], directions[idx + 1])
        if not _ray_clear(centers[idx], directions[idx], centers[idx + 1], blockers, grid.shape):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray) or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate arrow-chain puzzle")


if __name__ == "__main__":
    for _ in range(20):
        sample = generate_input()
        solved = main(sample)
        assert isinstance(sample, np.ndarray)
        assert isinstance(solved, np.ndarray)
        assert sample.shape == solved.shape
