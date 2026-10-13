from __future__ import annotations

from typing import Iterable

import numpy as np


BLACK = 0
ALL_DIRECTIONS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bbox_if_valid(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if len(coords) != 2 * h + 2 * w - 4:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return None
    if np.any(patch[~border] == color):
        return None
    return r0, c0, r1, c1


def _is_2x2_block(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 == 1 and c1 - c0 == 1


def _edge_distances(inner_r: int, inner_c: int, inner_h: int, inner_w: int) -> dict[str, int]:
    return {
        "top": inner_r,
        "bottom": inner_h - 1 - inner_r,
        "left": inner_c,
        "right": inner_w - 1 - inner_c,
    }


def _direction_from_seed(inner_r: int, inner_c: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    d = _edge_distances(inner_r, inner_c, inner_h, inner_w)
    mind = min(d.values())
    if d["left"] == mind or d["right"] == mind:
        if d["left"] == mind:
            return 0, 1
        return 0, -1
    if d["top"] == mind:
        return 1, 0
    return -1, 0


def _direction_name_from_delta(delta: tuple[int, int]) -> str:
    mapping = {
        (-1, 0): "up",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right",
    }
    return mapping[delta]


def _is_corner_tie(inner_r: int, inner_c: int, inner_h: int, inner_w: int) -> bool:
    d = _edge_distances(inner_r, inner_c, inner_h, inner_w)
    mind = min(d.values())
    winners = {edge for edge, dist in d.items() if dist == mind}
    return bool(winners & {"left", "right"}) and bool(winners & {"top", "bottom"})


def _frame_cells(r0: int, c0: int, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _path_cells(seed: tuple[int, int], delta: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = seed
    dr, dc = delta
    return [(sr + dr * step, sc + dc * step) for step in range(1, length + 1)]


def _in_bounds(cells: Iterable[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _clear_with_margin(mask: np.ndarray, cells: Iterable[tuple[int, int]], pad: int) -> bool:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if np.any(mask[r0:r1, c0:c1]):
            return False
    return True


def _block_candidates(r0: int, c0: int, outer_h: int, outer_w: int, direction: str, h: int, w: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    sides = {
        "up": ("left", "right", "down"),
        "down": ("left", "right", "up"),
        "left": ("up", "down", "right"),
        "right": ("up", "down", "left"),
    }[direction]

    for side in sides:
        if side == "up":
            br = r0 - 3
            for bc in range(c0 - 1, c0 + outer_w):
                candidates.append((br, bc))
        elif side == "down":
            br = r0 + outer_h + 1
            for bc in range(c0 - 1, c0 + outer_w):
                candidates.append((br, bc))
        elif side == "left":
            bc = c0 - 3
            for br in range(r0 - 1, r0 + outer_h):
                candidates.append((br, bc))
        else:
            bc = c0 + outer_w + 1
            for br in range(r0 - 1, r0 + outer_h):
                candidates.append((br, bc))

    uniq: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for br, bc in candidates:
        if (br, bc) in seen:
            continue
        seen.add((br, bc))
        block = [(br + dr, bc + dc) for dr in range(2) for dc in range(2)]
        if _in_bounds(block, h, w):
            uniq.append((br, bc))
    return uniq


def _candidate_seed_positions(inner_h: int, inner_w: int, direction: str, prefer_tie: bool) -> list[tuple[int, int]]:
    tie_positions: list[tuple[int, int]] = []
    plain_positions: list[tuple[int, int]] = []
    for inner_r in range(inner_h):
        for inner_c in range(inner_w):
            delta = _direction_from_seed(inner_r, inner_c, inner_h, inner_w)
            if _direction_name_from_delta(delta) != direction:
                continue
            if _is_corner_tie(inner_r, inner_c, inner_h, inner_w):
                tie_positions.append((inner_r, inner_c))
            else:
                plain_positions.append((inner_r, inner_c))
    if prefer_tie and tie_positions:
        return tie_positions
    if plain_positions:
        return plain_positions
    return tie_positions


def _scan_frame_positions(
    h: int,
    w: int,
    inner_h: int,
    inner_w: int,
    seed_inner: tuple[int, int],
    direction: str,
) -> tuple[range, range]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    inner_r, inner_c = seed_inner
    length = inner_h * inner_w

    row_lo = 0
    row_hi = h - outer_h
    col_lo = 0
    col_hi = w - outer_w

    if direction == "right":
        col_hi = min(col_hi, w - length - 2 - inner_c)
    elif direction == "left":
        col_lo = max(col_lo, length - 1 - inner_c)
    elif direction == "down":
        row_hi = min(row_hi, h - length - 2 - inner_r)
    else:
        row_lo = max(row_lo, length - 1 - inner_r)

    if row_lo > row_hi or col_lo > col_hi:
        return range(0), range(0)
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == BLACK:
            continue
        comps = _components(grid == color)
        frame_bbox = None
        has_block = False
        for comp in comps:
            bbox = _frame_bbox_if_valid(grid, color, comp)
            if bbox is not None:
                frame_bbox = bbox
            elif _is_2x2_block(comp):
                has_block = True
        if frame_bbox is None or not has_block:
            continue
        r0, c0, r1, c1 = frame_bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((interior != BLACK) & (interior != color))
        if seed_coords.shape[0] != 1:
            continue
        sr = int(r0 + 1 + seed_coords[0, 0])
        sc = int(c0 + 1 + seed_coords[0, 1])
        objects.append({"color": color, "bbox": frame_bbox, "seed": (sr, sc)})
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))  # type: ignore[index]
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-colored branches from seeds inside colored rectangular frames.

    Length = interior area of the frame.
    Direction = away from the nearest interior edge, with horizontal preference on ties.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for obj in _extract_objects(g):
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[misc]
        sr, sc = obj["seed"]  # type: ignore[misc]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        inner_r = sr - (r0 + 1)
        inner_c = sc - (c0 + 1)
        delta = _direction_from_seed(inner_r, inner_c, inner_h, inner_w)
        length = inner_h * inner_w
        for rr, cc in _path_cells((sr, sc), delta, length):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color

    return out


def _sample_directions(rng: np.random.Generator, count: int) -> list[str]:
    if count == 2:
        dirs = [rng.choice(["left", "right"]).item(), rng.choice(["up", "down"]).item()]
    elif count == 3:
        dirs = [rng.choice(["left", "right"]).item(), rng.choice(["up", "down"]).item(), rng.choice(ALL_DIRECTIONS).item()]
    else:
        dirs = list(rng.permutation(np.array(ALL_DIRECTIONS)))
    rng.shuffle(dirs)
    return dirs


def _place_object(
    grid: np.ndarray,
    occupied_input: np.ndarray,
    reserved_output: np.ndarray,
    frame_color: int,
    seed_color: int,
    direction: str,
    prefer_tie: bool,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    for _ in range(90):
        inner_h = int(rng.integers(2, 6))
        inner_w = int(rng.integers(2, 6))
        length = inner_h * inner_w
        if length > max(h, w) - 4:
            continue

        seed_options = _candidate_seed_positions(inner_h, inner_w, direction, prefer_tie)
        if not seed_options:
            continue
        rng.shuffle(seed_options)
        seed_inner = seed_options[0]
        row_range, col_range = _scan_frame_positions(h, w, inner_h, inner_w, seed_inner, direction)
        if len(row_range) == 0 or len(col_range) == 0:
            continue

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        rows = list(row_range)
        cols = list(col_range)
        rng.shuffle(rows)
        rng.shuffle(cols)

        for r0 in rows:
            for c0 in cols:
                frame_cells = _frame_cells(r0, c0, outer_h, outer_w)
                sr = r0 + 1 + seed_inner[0]
                sc = c0 + 1 + seed_inner[1]
                seed = (sr, sc)
                seed_and_frame = frame_cells + [seed]
                if not _clear_with_margin(occupied_input, seed_and_frame, pad=1):
                    continue
                if any(reserved_output[r, c] for r, c in seed_and_frame):
                    continue

                delta = _direction_from_seed(seed_inner[0], seed_inner[1], inner_h, inner_w)
                path = _path_cells(seed, delta, length)
                if not _in_bounds(path, h, w):
                    continue
                if any(occupied_input[r, c] or reserved_output[r, c] for r, c in path):
                    continue

                block_tops = _block_candidates(r0, c0, outer_h, outer_w, direction, h, w)
                rng.shuffle(block_tops)
                for br, bc in block_tops:
                    block_cells = [(br + dr, bc + dc) for dr in range(2) for dc in range(2)]
                    if not _clear_with_margin(occupied_input, block_cells, pad=1):
                        continue
                    if any(reserved_output[r, c] for r, c in block_cells):
                        continue
                    if any((r, c) in set(block_cells) for r, c in path):
                        continue

                    for r, c in frame_cells:
                        grid[r, c] = frame_color
                    grid[sr, sc] = seed_color
                    for r, c in block_cells:
                        grid[r, c] = frame_color

                    for r, c in seed_and_frame + block_cells:
                        occupied_input[r, c] = True
                        reserved_output[r, c] = True
                    for r, c in path:
                        reserved_output[r, c] = True
                    return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    objects = _extract_objects(grid)
    if len(objects) < 2:
        return False
    seen_paths = np.zeros_like(grid, dtype=bool)
    occupied = grid != BLACK
    for obj in objects:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[misc]
        sr, sc = obj["seed"]  # type: ignore[misc]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        delta = _direction_from_seed(sr - (r0 + 1), sc - (c0 + 1), inner_h, inner_w)
        frame_set = set(_frame_cells(r0, c0, inner_h + 2, inner_w + 2))
        for rr, cc in _path_cells((sr, sc), delta, inner_h * inner_w):
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                return False
            if seen_paths[rr, cc]:
                return False
            if occupied[rr, cc] and (rr, cc) not in frame_set and int(grid[rr, cc]) != color:
                return False
            seen_paths[rr, cc] = True
    return not np.array_equal(main(grid), grid)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_objects = int(rng.integers(2, 5))
    h = int(rng.integers(18 + 2 * (num_objects - 2), 31))
    w = int(rng.integers(18 + 2 * (num_objects - 2), 31))
    grid = np.zeros((h, w), dtype=int)

    palette = list(rng.permutation(np.arange(1, 10)))
    frame_colors = [int(color) for color in palette[:num_objects]]
    seed_colors = [int(color) for color in palette[num_objects : 2 * num_objects]]
    directions = _sample_directions(rng, num_objects)

    horizontal_indices = [idx for idx, direction in enumerate(directions) if direction in {"left", "right"}]
    tie_index = None
    if horizontal_indices and rng.random() < 0.6:
        tie_index = int(rng.choice(horizontal_indices))

    occupied_input = np.zeros_like(grid, dtype=bool)
    reserved_output = np.zeros_like(grid, dtype=bool)
    order = list(range(num_objects))
    rng.shuffle(order)

    for idx in order:
        prefer_tie = idx == tie_index
        ok = _place_object(
            grid,
            occupied_input,
            reserved_output,
            frame_colors[idx],
            seed_colors[idx],
            directions[idx],
            prefer_tie,
            rng,
        )
        if not ok:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
