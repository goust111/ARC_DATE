from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_SIDE_RANGE = (20, 31)
ARROW_COUNT_RANGE = (2, 5)
SHAFT_LEN_RANGE = (2, 6)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _arrow_geometry(direction: str, shaft_len: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    shaft: set[tuple[int, int]] = set()
    if direction == "up":
        cells.add((0, 2))
        for c in range(5):
            cells.add((1, c))
        for r in range(2, shaft_len + 2):
            for c in range(1, 4):
                cell = (r, c)
                cells.add(cell)
                shaft.add(cell)
        return cells, shaft, (shaft_len + 2, 5)
    if direction == "down":
        for r in range(shaft_len):
            for c in range(1, 4):
                cell = (r, c)
                cells.add(cell)
                shaft.add(cell)
        for c in range(5):
            cells.add((shaft_len, c))
        cells.add((shaft_len + 1, 2))
        return cells, shaft, (shaft_len + 2, 5)
    if direction == "left":
        cells.add((2, 0))
        for r in range(5):
            cells.add((r, 1))
        for c in range(2, shaft_len + 2):
            for r in range(1, 4):
                cell = (r, c)
                cells.add(cell)
                shaft.add(cell)
        return cells, shaft, (5, shaft_len + 2)
    if direction == "right":
        for c in range(shaft_len):
            for r in range(1, 4):
                cell = (r, c)
                cells.add(cell)
                shaft.add(cell)
        for r in range(5):
            cells.add((r, shaft_len))
        cells.add((2, shaft_len + 1))
        return cells, shaft, (5, shaft_len + 2)
    raise ValueError(f"unknown direction: {direction}")


def _component_cells(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _match_arrow(coords: np.ndarray) -> dict[str, object] | None:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}

    matches: list[tuple[str, int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    if w == 5 and h >= 4:
        shaft_len = h - 2
        if shaft_len >= 2:
            for direction in ("up", "down"):
                shape, shaft, _ = _arrow_geometry(direction, shaft_len)
                if rel == shape:
                    matches.append((direction, shaft_len, shape, shaft))
    if h == 5 and w >= 4:
        shaft_len = w - 2
        if shaft_len >= 2:
            for direction in ("left", "right"):
                shape, shaft, _ = _arrow_geometry(direction, shaft_len)
                if rel == shape:
                    matches.append((direction, shaft_len, shape, shaft))

    if len(matches) != 1:
        return None

    direction, shaft_len, shape, shaft = matches[0]
    return {
        "direction": direction,
        "shaft_len": shaft_len,
        "bbox": (r0, c0, r1, c1),
        "shape": shape,
        "shaft": shaft,
    }


def _extract_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict[str, object]] = []
    for coords in _component_cells(g != BG):
        values = [int(g[r, c]) for r, c in coords]
        counts = Counter(values)
        projectile_colors = [color for color, count in counts.items() if count == 1]
        assert len(counts) == 2 and len(projectile_colors) == 1, "each arrow must contain one projectile color"

        projectile_color = int(projectile_colors[0])
        arrow_color = int(next(color for color in counts if color != projectile_color))
        projectile_coord = next((int(r), int(c)) for r, c in coords if int(g[r, c]) == projectile_color)

        meta = _match_arrow(coords)
        assert meta is not None, "component does not match an arrow silhouette"
        r0, c0, _r1, _c1 = meta["bbox"]
        local_projectile = (projectile_coord[0] - r0, projectile_coord[1] - c0)
        assert local_projectile in meta["shaft"], "projectile must sit inside the shaft"

        arrows.append(
            {
                "direction": meta["direction"],
                "shaft_len": meta["shaft_len"],
                "projectile": projectile_coord,
                "projectile_color": projectile_color,
                "arrow_color": arrow_color,
                "distance": int(counts[arrow_color]),
            }
        )
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each embedded projectile along its arrow direction by the arrow's own pixel count.

    Input structure:
    - Background is black.
    - Each foreground component is an axis-aligned arrow silhouette made from one main color plus one distinct 1x1 projectile color.
    - An arrow consists of a 3-pixel-wide shaft and a 5-pixel-wide, 2-pixel-deep triangular head pointing up, down, left, or right.

    Rule:
    - For each connected arrow component, identify the arrow direction from the head and identify the projectile as the unique singleton color inside the shaft.
    - Let N be the number of pixels of the arrow's main color, excluding the projectile.
    - Erase the whole input component, then place the projectile N cells farther along the arrow direction if that destination stays inside the grid.

    Key insight:
    - The silhouette tells you the travel direction, while the visible arrow-body pixel count encodes the travel distance.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    h, w = g.shape

    for arrow in _extract_arrows(g):
        pr, pc = arrow["projectile"]
        dr, dc = DIR_VECTORS[arrow["direction"]]
        distance = int(arrow["distance"])
        tr = pr + dr * distance
        tc = pc + dc * distance
        if 0 <= tr < h and 0 <= tc < w:
            out[tr, tc] = int(arrow["projectile_color"])
    return out


def _tail_projectile_cells(direction: str, shaft_len: int) -> list[tuple[int, int]]:
    _shape, shaft, _dims = _arrow_geometry(direction, shaft_len)
    if direction == "up":
        min_row = 2 + shaft_len // 2
        return [cell for cell in sorted(shaft) if cell[0] >= min_row]
    if direction == "down":
        max_row = (shaft_len - 1) - shaft_len // 2
        return [cell for cell in sorted(shaft) if cell[0] <= max_row]
    if direction == "left":
        min_col = 2 + shaft_len // 2
        return [cell for cell in sorted(shaft) if cell[1] >= min_col]
    max_col = (shaft_len - 1) - shaft_len // 2
    return [cell for cell in sorted(shaft) if cell[1] <= max_col]


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _sample_arrow_count(rng: np.random.Generator) -> int:
    return int(rng.integers(*ARROW_COUNT_RANGE))


def _inside_flags(rng: np.random.Generator, count: int) -> list[bool]:
    inside = [False] * count
    inside_count_options = [count // 2]
    if count % 2 == 1:
        inside_count_options.append(count // 2 + 1)
    inside_count = int(rng.choice(np.array(inside_count_options, dtype=int)))
    indices = rng.choice(count, size=inside_count, replace=False)
    for idx in np.atleast_1d(indices):
        inside[int(idx)] = True
    return inside


def _choose_colors(rng: np.random.Generator, count: int) -> tuple[list[int], list[int]]:
    picked = list(map(int, rng.choice(np.array(PALETTE, dtype=int), size=2 * count, replace=False)))
    rng.shuffle(picked)
    return picked[:count], picked[count:]


def _place_arrow(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    target_cells: set[tuple[int, int]],
    direction: str,
    keep_inside: bool,
    arrow_color: int,
    projectile_color: int,
) -> dict[str, object] | None:
    h, w = grid.shape

    for _ in range(PLACE_ATTEMPTS):
        shaft_len = int(rng.integers(*SHAFT_LEN_RANGE))
        shape, _shaft, (bh, bw) = _arrow_geometry(direction, shaft_len)
        projectile_local = tuple(_tail_projectile_cells(direction, shaft_len)[int(rng.integers(len(_tail_projectile_cells(direction, shaft_len))))])
        distance = len(shape) - 1
        dr, dc = DIR_VECTORS[direction]
        target_local = (
            projectile_local[0] + dr * distance,
            projectile_local[1] + dc * distance,
        )

        local_cells = sorted(shape)
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(h - bh + 1):
            for c0 in range(w - bw + 1):
                abs_cells = [(r0 + rr, c0 + cc) for rr, cc in local_cells]
                if any(blocked[rr, cc] for rr, cc in abs_cells):
                    continue

                tr = r0 + target_local[0]
                tc = c0 + target_local[1]
                in_bounds = 0 <= tr < h and 0 <= tc < w
                if in_bounds != keep_inside:
                    continue
                if in_bounds and (blocked[tr, tc] or (tr, tc) in target_cells):
                    continue
                candidates.append((r0, c0, tr, tc))

        if not candidates:
            continue

        r0, c0, tr, tc = candidates[int(rng.integers(len(candidates)))]
        abs_cells = [(r0 + rr, c0 + cc) for rr, cc in local_cells]
        for rr, cc in abs_cells:
            grid[rr, cc] = arrow_color
        pr = r0 + projectile_local[0]
        pc = c0 + projectile_local[1]
        grid[pr, pc] = projectile_color

        _mark_blocked(blocked, abs_cells)
        if keep_inside:
            target_cells.add((tr, tc))
            blocked[tr, tc] = True

        return {
            "direction": direction,
            "shaft_len": shaft_len,
            "projectile": (pr, pc),
            "projectile_color": projectile_color,
            "arrow_color": arrow_color,
            "distance": distance,
            "target": (tr, tc) if keep_inside else None,
        }

    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_SIDE_RANGE))
    w = int(rng.integers(*GRID_SIDE_RANGE))
    n_arrows = _sample_arrow_count(rng)
    directions = list(rng.permutation(np.array(DIRECTIONS, dtype=object))[:n_arrows])
    inside_flags = _inside_flags(rng, n_arrows)
    arrow_colors, projectile_colors = _choose_colors(rng, n_arrows)

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    target_cells: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    order = list(range(n_arrows))
    rng.shuffle(order)
    for idx in order:
        spec = _place_arrow(
            rng,
            grid,
            blocked,
            target_cells,
            str(directions[idx]),
            bool(inside_flags[idx]),
            int(arrow_colors[idx]),
            int(projectile_colors[idx]),
        )
        if spec is None:
            return None
        placed.append(spec)

    detected = _extract_arrows(grid)
    if len(detected) != n_arrows:
        return None

    expected = np.zeros_like(grid)
    for spec in placed:
        target = spec["target"]
        if target is not None:
            tr, tc = target
            expected[tr, tc] = int(spec["projectile_color"])
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid arrow-launch input after {GEN_ATTEMPTS} attempts")
