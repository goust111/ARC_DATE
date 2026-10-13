from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BLOCKER_SHAPES = [(1, 2), (2, 1), (2, 2)]

HOLE_SHAPES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
]

SECTOR_SIDES = {
    "tl": ("left", "up"),
    "tr": ("right", "up"),
    "bl": ("left", "down"),
    "br": ("right", "down"),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    next_label = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_label += 1
            q = deque([(r, c)])
            labels[r, c] = next_label
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
    return labels, next_label


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    labels, n = _label_mask(mask.astype(bool))
    return [np.argwhere(labels == k) for k in range(1, n + 1)]


def _hole_infos(grid: np.ndarray) -> list[dict]:
    green_mask = grid == GREEN
    green_labels, green_count = _label_mask(green_mask)

    holes: list[dict] = []
    for comp in _components_mask(~green_mask):
        coords = [(int(r), int(c)) for r, c in comp]
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in coords):
            continue
        adj_labels = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and green_labels[nr, nc] > 0:
                    adj_labels.add(int(green_labels[nr, nc]))
        if len(adj_labels) != 1:
            continue
        holes.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "size": len(coords),
                "green_label": next(iter(adj_labels)),
            }
        )

    counts = {k: 0 for k in range(1, green_count + 1)}
    for hole in holes:
        counts[hole["green_label"]] += 1
    if any(v != 1 for v in counts.values()):
        return []
    return holes


def _ray_mask(grid: np.ndarray, *, block_blue: bool = True, block_green: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros(g.shape, dtype=bool)
    sources = np.argwhere(g == RED)
    for sr, sc in sources:
        sr = int(sr)
        sc = int(sc)
        for dr, dc in DIRS4:
            r, c = sr + dr, sc + dc
            while _in_bounds(g.shape, r, c):
                val = int(g[r, c])
                if block_blue and val == BLUE:
                    break
                if block_green and val == GREEN:
                    break
                if val not in (RED, BLUE, GREEN):
                    out[r, c] = True
                r += dr
                c += dc
    return out


def _touched_holes(grid: np.ndarray, *, block_blue: bool = True, block_green: bool = False) -> list[bool]:
    ray = _ray_mask(grid, block_blue=block_blue, block_green=block_green)
    holes = _hole_infos(grid)
    return [any(ray[r, c] for r, c in hole["coords"]) for hole in holes]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays from red sources through green hollow sprites and fill only the ray-reached holes.

    Input structure:
    - Black is the background, with 2-3 isolated red source pixels, 2-4 small solid blue blockers,
      and 3-4 green hollow sprites whose single enclosed holes are black.
    - Blue blockers are tiny 1x2, 2x1, or 2x2 rectangles that do not overlap the sprites.
    - Yellow and orange are absent from the input.

    Rule:
    - From each red source, extend four axis-aligned rays until the grid edge, stopping one cell
      before any blue blocker; green sprite borders are transparent to the rays.
    - Recolor every ray-visited non-red, non-blue, non-green cell yellow.
    - For each enclosed hole inside a green sprite, if any hole cell was reached by a yellow ray,
      recolor the whole hole orange; otherwise keep the hole black.
    - Remove the red sources to black and keep the blue blockers plus green borders unchanged.

    Key insight:
    - The output depends on tracing the true ray paths with asymmetric blocking: blue stops a ray,
      but green borders do not, so blocker placement determines which enclosed holes become fillable.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    ray = _ray_mask(g, block_blue=True, block_green=False)
    paintable = ray & (g != RED) & (g != BLUE) & (g != GREEN)
    out[paintable] = YELLOW

    for hole in _hole_infos(g):
        if any(ray[r, c] for r, c in hole["coords"]):
            for r, c in hole["coords"]:
                out[r, c] = ORANGE

    out[g == RED] = BLACK
    return out


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    transformed = mask.copy()
    k = rng.randrange(4)
    transformed = np.rot90(transformed, k)
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    return transformed


def _make_sprite_patch(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(40):
        base = _random_transform(rng.choice(HOLE_SHAPES), rng).astype(bool)
        hole = np.pad(base, 1, constant_values=False)
        border = np.zeros_like(hole, dtype=bool)
        for r, c in np.argwhere(hole):
            r = int(r)
            c = int(c)
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(hole.shape, nr, nc) and not hole[nr, nc]:
                    border[nr, nc] = True
        patch = np.zeros(hole.shape, dtype=int)
        patch[border] = GREEN

        hole_infos = _hole_infos(patch)
        green_comps = _components_mask(patch == GREEN)
        if len(green_comps) == 1 and len(hole_infos) == 1 and hole_infos[0]["size"] == int(hole.sum()):
            return patch, hole
    raise RuntimeError("failed to build a valid sprite patch")


def _sector_boxes(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return {
        "tl": (2, 2, mid_r - 3, mid_c - 3),
        "tr": (2, mid_c + 3, mid_r - 3, w - 3),
        "bl": (mid_r + 3, 2, h - 3, mid_c - 3),
        "br": (mid_r + 3, mid_c + 3, h - 3, w - 3),
    }


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BLACK
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _is_source_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != BLACK:
        return False
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BLACK:
            return False
    return True


def _blocker_options(grid: np.ndarray, anchor: tuple[int, int], source: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    ar, ac = anchor
    opts: list[list[tuple[int, int]]] = []
    for bh, bw in BLOCKER_SHAPES:
        for dr in range(bh):
            for dc in range(bw):
                r0 = ar - dr
                c0 = ac - dc
                if r0 < 0 or c0 < 0 or r0 + bh > h or c0 + bw > w:
                    continue
                cells = [(r0 + rr, c0 + cc) for rr in range(bh) for cc in range(bw)]
                if any(grid[r, c] != BLACK for r, c in cells):
                    continue
                if any(abs(r - source[0]) + abs(c - source[1]) <= 1 for r, c in cells):
                    continue
                opts.append(cells)
    return opts


def _source_plan_candidates(grid: np.ndarray, sprite: dict, role: str, rng: random.Random) -> list[dict]:
    h, w = grid.shape
    border_set = sprite["border_set"]
    candidates = []

    holes = sprite["hole_cells"][:]
    rng.shuffle(holes)
    sides = list(SECTOR_SIDES[sprite["sector"]])
    rng.shuffle(sides)

    for side in sides:
        for hr, hc in holes:
            if side == "left":
                source = (hr, 0)
                seg = [(hr, c) for c in range(1, hc)]
            elif side == "right":
                source = (hr, w - 1)
                seg = [(hr, c) for c in range(hc + 1, w - 1)]
            elif side == "up":
                source = (0, hc)
                seg = [(r, hc) for r in range(1, hr)]
            else:
                source = (h - 1, hc)
                seg = [(r, hc) for r in range(hr + 1, h - 1)]

            if not _is_source_isolated(grid, source[0], source[1]):
                continue
            if not seg:
                continue
            if any(grid[r, c] != BLACK and (r, c) not in border_set for r, c in seg):
                continue
            if not any((r, c) in border_set for r, c in seg):
                continue

            if role == "accessible":
                candidates.append({"source": source, "blocker": None})
                continue

            background_cells = [cell for cell in seg if grid[cell] == BLACK]
            rng.shuffle(background_cells)
            for anchor in background_cells:
                blockers = _blocker_options(grid, anchor, source)
                if blockers:
                    candidates.append({"source": source, "blocker": rng.choice(blockers)})
                    break
    return candidates


def _place_extra_blockers(grid: np.ndarray, count: int, rng: random.Random) -> bool:
    h, w = grid.shape
    row_choices = list(range(max(1, h // 2 - 2), min(h - 2, h // 2 + 1)))
    col_choices = list(range(max(1, w // 2 - 2), min(w - 2, w // 2 + 1)))
    if not row_choices or not col_choices:
        return count == 0

    for _ in range(count):
        placed = False
        for _ in range(80):
            bh, bw = rng.choice(BLOCKER_SHAPES)
            r0 = rng.choice(row_choices)
            c0 = rng.choice(col_choices)
            if r0 + bh > h or c0 + bw > w:
                continue
            cells = [(r0 + rr, c0 + cc) for rr in range(bh) for cc in range(bw)]
            if any(grid[r, c] != BLACK for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = BLUE
            placed = True
            break
        if not placed:
            return False
    return True


def _analyze_blue_rectangles(grid: np.ndarray) -> bool:
    comps = _components_mask(grid == BLUE)
    if not 2 <= len(comps) <= 4:
        return False
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        h = int(rows.max() - rows.min() + 1)
        w = int(cols.max() - cols.min() + 1)
        if (h, w) not in {(1, 2), (2, 1), (2, 2)}:
            return False
        if int(comp.shape[0]) != h * w:
            return False
    return True


def _validate_candidate(grid: np.ndarray) -> bool:
    if np.count_nonzero(grid == YELLOW) or np.count_nonzero(grid == ORANGE):
        return False

    red_comps = _components_mask(grid == RED)
    if not 2 <= len(red_comps) <= 3 or any(len(comp) != 1 for comp in red_comps):
        return False
    if not _analyze_blue_rectangles(grid):
        return False

    green_comps = _components_mask(grid == GREEN)
    if not 3 <= len(green_comps) <= 4:
        return False

    holes = _hole_infos(grid)
    if len(holes) != len(green_comps):
        return False
    if any(not 3 <= hole["size"] <= 6 for hole in holes):
        return False

    touched_actual = _touched_holes(grid, block_blue=True, block_green=False)
    touched_no_blue = _touched_holes(grid, block_blue=False, block_green=False)
    touched_green_block = _touched_holes(grid, block_blue=True, block_green=True)

    touched_count = sum(touched_actual)
    if touched_count == 0 or touched_count == len(holes):
        return False
    if not any((not a) and b for a, b in zip(touched_actual, touched_no_blue, strict=False)):
        return False
    if not any(a and (not b) for a, b in zip(touched_actual, touched_green_block, strict=False)):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.count_nonzero(out == RED) != 0:
        return False

    hole_cells = {cell for hole in holes for cell in hole["coords"]}
    outside_yellow = [
        (r, c)
        for r, c in zip(*np.where(out == YELLOW), strict=False)
        if (int(r), int(c)) not in hole_cells
    ]
    if len(outside_yellow) < 3:
        return False

    for touched, hole in zip(touched_actual, holes, strict=False):
        vals = {int(out[r, c]) for r, c in hole["coords"]}
        if touched and vals != {ORANGE}:
            return False
        if (not touched) and vals != {BLACK}:
            return False
    return True


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(19, 24)
    w = rng.randint(19, 24)
    grid = np.zeros((h, w), dtype=int)

    sector_boxes = _sector_boxes(h, w)
    sprite_count = rng.randint(3, 4)
    chosen_sectors = rng.sample(list(sector_boxes.keys()), sprite_count)
    sprites = []

    for sector in chosen_sectors:
        patch, hole_mask = _make_sprite_patch(rng)
        sr0, sc0, sr1, sc1 = sector_boxes[sector]
        patch_h, patch_w = patch.shape
        if sr1 - sr0 + 1 < patch_h or sc1 - sc0 + 1 < patch_w:
            return None
        r0 = rng.randint(sr0, sr1 - patch_h + 1)
        c0 = rng.randint(sc0, sc1 - patch_w + 1)
        _place_patch(grid, patch, r0, c0)

        hole_cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(hole_mask)]
        border_set = {
            (r0 + int(r), c0 + int(c))
            for r, c in np.argwhere(patch == GREEN)
        }
        sprites.append(
            {
                "sector": sector,
                "hole_cells": hole_cells,
                "border_set": border_set,
            }
        )

    source_count = rng.randint(2, min(3, sprite_count))
    roles = ["accessible", "blocked"]
    if source_count == 3:
        roles.append(rng.choice(["accessible", "blocked"]))
    target_sprites = rng.sample(sprites, len(roles))

    for sprite, role in zip(target_sprites, roles, strict=False):
        plans = _source_plan_candidates(grid, sprite, role, rng)
        if not plans:
            return None
        plan = rng.choice(plans)
        sr, sc = plan["source"]
        grid[sr, sc] = RED
        if plan["blocker"] is not None:
            for br, bc in plan["blocker"]:
                grid[br, bc] = BLUE

    blocker_count = rng.randint(2, 4)
    current_blockers = len(_components_mask(grid == BLUE))
    if current_blockers > blocker_count:
        blocker_count = current_blockers
    if blocker_count > 4:
        return None
    if not _place_extra_blockers(grid, blocker_count - current_blockers, rng):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate(random.Random())
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
