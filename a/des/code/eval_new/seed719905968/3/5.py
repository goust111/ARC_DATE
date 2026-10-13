from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
PAIR_COUNT_OPTIONS = (2, 3, 4)
SCALE_OPTIONS = (2, 3)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(arr)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            primitive, scale = _primitive_shape(mask)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "primitive": primitive,
                    "primitive_sig": _mask_signature(primitive),
                    "scale": scale,
                    "area": int(arr.shape[0]),
                }
            )

    return objects


def _compress_mask(mask: np.ndarray, factor: int) -> np.ndarray | None:
    h, w = mask.shape
    if h % factor != 0 or w % factor != 0:
        return None
    view = mask.reshape(h // factor, factor, w // factor, factor)
    base = view[:, 0, :, 0]
    if np.all(view == base[:, None, :, None]):
        return base.copy()
    return None


def _primitive_shape(mask: np.ndarray) -> tuple[np.ndarray, int]:
    current = np.asarray(mask, dtype=bool).copy()
    total_scale = 1
    while True:
        reduced = None
        for factor in range(2, min(current.shape) + 1):
            candidate = _compress_mask(current, factor)
            if candidate is None:
                continue
            reduced = candidate
            total_scale *= factor
            current = candidate
            break
        if reduced is None:
            return current, total_scale


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, mask.shape)), tuple(int(v) for v in mask.ravel())


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask.copy()
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def _valid_pair(small: dict[str, object], big: dict[str, object]) -> bool:
    if int(small["color"]) == int(big["color"]):
        return False
    small_scale = int(small["scale"])
    big_scale = int(big["scale"])
    if big_scale <= small_scale or big_scale % small_scale != 0:
        return False
    factor = big_scale // small_scale
    return np.array_equal(_scale_mask(np.asarray(small["mask"], dtype=bool), factor), np.asarray(big["mask"], dtype=bool))


def _forced_pairs(group_indices: list[int], objects: list[dict[str, object]]) -> list[tuple[int, int]]:
    local = [objects[idx] for idx in group_indices]
    n = len(local)
    edge_by_nodes: dict[frozenset[int], tuple[int, int]] = {}

    for a in range(n):
        for b in range(a + 1, n):
            obj_a = local[a]
            obj_b = local[b]
            if int(obj_a["area"]) == int(obj_b["area"]):
                continue
            if int(obj_a["area"]) < int(obj_b["area"]):
                small_idx, big_idx = a, b
            else:
                small_idx, big_idx = b, a
            if _valid_pair(local[small_idx], local[big_idx]):
                edge_by_nodes[frozenset((a, b))] = (small_idx, big_idx)

    if not edge_by_nodes:
        return []

    used = [False] * n
    current: list[tuple[int, int]] = []
    best_size = -1
    best_matchings: list[tuple[tuple[int, int], ...]] = []

    def dfs() -> None:
        nonlocal best_size, best_matchings
        remaining = sum(not flag for flag in used)
        if len(current) + remaining // 2 < best_size:
            return

        try:
            first = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            frozen = tuple(sorted(current))
            if len(current) > best_size:
                best_size = len(current)
                best_matchings = [frozen]
            elif len(current) == best_size:
                best_matchings.append(frozen)
            return

        used[first] = True
        dfs()
        used[first] = False

        for second in range(first + 1, n):
            if used[second]:
                continue
            edge = edge_by_nodes.get(frozenset((first, second)))
            if edge is None:
                continue
            used[first] = True
            used[second] = True
            current.append(edge)
            dfs()
            current.pop()
            used[first] = False
            used[second] = False

    dfs()
    if best_size <= 0 or not best_matchings:
        return []

    common = set(best_matchings[0])
    for matching in best_matchings[1:]:
        common &= set(matching)

    return sorted((group_indices[small], group_indices[big]) for small, big in common)


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer each smaller object's color onto its uniquely matched larger scaled copy, then erase the smaller object.

    Input structure:
    - bg is the most common color.
    - The grid contains several monochrome 4-connected objects scattered apart from one another.
    - Objects come in same-shape pairs: within each pair, one object is an integer-scale enlargement of the other without rotation or reflection.

    Rule:
    - Tight-crop every object to its bounding box and compare normalized binary masks up to integer scaling only.
    - For each successfully matched small→large pair, repaint the large object's full footprint with the small object's color.
    - Replace the small object's cells with bg.
    - Leave any object that cannot be paired unmodified.

    Key insight:
    - Pairing is based on geometric equivalence after normalization and integer rescaling, and the color flow is one-way from the smaller copy to the larger one.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    objects = _extract_objects(g, bg)

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[int]] = {}
    for idx, obj in enumerate(objects):
        groups.setdefault(obj["primitive_sig"], []).append(idx)

    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        pairs.extend(_forced_pairs(indices, objects))

    out = g.copy()
    for small_idx, big_idx in pairs:
        small = objects[small_idx]
        big = objects[big_idx]
        small_coords = np.asarray(small["coords"], dtype=int)
        big_coords = np.asarray(big["coords"], dtype=int)
        out[small_coords[:, 0], small_coords[:, 1]] = bg
        out[big_coords[:, 0], big_coords[:, 1]] = int(small["color"])

    return out


def _touches_hole(mask: np.ndarray) -> bool:
    holes = ~mask
    h, w = mask.shape
    seen = np.zeros_like(holes, dtype=bool)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not holes[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and holes[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_edge = True
                        q.append((nr, nc))
            if not touches_edge:
                return True
    return False


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(120):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        max_cells = min(8, h * w - 1)
        if max_cells < 3:
            continue
        target = rng.randint(3, max_cells)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target:
            continue

        mask = _tight_mask(cells)
        mask = np.rot90(mask, rng.randrange(4))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        mask = mask.copy()

        primitive, scale = _primitive_shape(mask)
        if scale != 1:
            continue
        if np.all(mask):
            continue
        if _touches_hole(mask):
            continue
        if max(mask.shape) > 5:
            continue
        return mask

    raise RuntimeError("failed to generate a primitive base shape")


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _paint(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    region = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    region[mask] = color


def _reserve_positions(reserved: np.ndarray, h: int, w: int, padding: int) -> list[tuple[int, int]]:
    gh, gw = reserved.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(gh - h + 1):
        for c0 in range(gw - w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(gh, r0 + h + padding)
            cc1 = min(gw, c0 + w + padding)
            if not reserved[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = rng.randrange(10)
    pair_count = rng.choice(PAIR_COUNT_OPTIONS)
    scales = [rng.choice(SCALE_OPTIONS) for _ in range(pair_count)]

    available_colors = [color for color in range(10) if color != bg]
    if 2 * pair_count > len(available_colors):
        return None
    rng.shuffle(available_colors)

    pairs = []
    used_shapes: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for idx in range(pair_count):
        for _ in range(120):
            base_mask = _random_shape(rng)
            sig = _mask_signature(base_mask)
            if sig in used_shapes:
                continue
            used_shapes.add(sig)
            break
        else:
            return None

        small_color = available_colors[2 * idx]
        big_color = available_colors[2 * idx + 1]
        scale = scales[idx]
        big_mask = _scale_mask(base_mask, scale)
        pairs.append(
            {
                "small_mask": base_mask,
                "big_mask": big_mask,
                "small_color": small_color,
                "big_color": big_color,
                "scale": scale,
            }
        )

    objects = []
    for pair_idx, pair in enumerate(pairs):
        objects.append(
            {
                "pair_idx": pair_idx,
                "role": "big",
                "mask": pair["big_mask"],
                "color": int(pair["big_color"]),
                "area": int(np.count_nonzero(pair["big_mask"])),
            }
        )
        objects.append(
            {
                "pair_idx": pair_idx,
                "role": "small",
                "mask": pair["small_mask"],
                "color": int(pair["small_color"]),
                "area": int(np.count_nonzero(pair["small_mask"])),
            }
        )

    total_fg = sum(int(obj["area"]) for obj in objects)
    max_fg = max(int(obj["area"]) for obj in objects)
    max_dim = max(max(np.asarray(obj["mask"]).shape) for obj in objects)
    side_lo = max(max_dim + 2, math.ceil(math.sqrt(total_fg + max_fg + 1)))
    if side_lo > 30:
        return None

    rows = rng.randint(side_lo, min(30, side_lo + 8))
    cols = rng.randint(side_lo, min(30, side_lo + 8))
    grid = np.full((rows, cols), bg, dtype=int)
    expected = np.full((rows, cols), bg, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)
    placements: dict[tuple[int, str], tuple[int, int]] = {}

    order = list(range(len(objects)))
    rng.shuffle(order)
    order.sort(key=lambda idx: int(objects[idx]["area"]), reverse=True)

    for obj_idx in order:
        obj = objects[obj_idx]
        mask = np.asarray(obj["mask"], dtype=bool)
        h, w = mask.shape
        positions = _reserve_positions(reserved, h, w, padding=1)
        if not positions:
            return None
        top, left = rng.choice(positions)
        _paint(grid, mask, int(obj["color"]), top, left)
        rr0 = max(0, top - 1)
        cc0 = max(0, left - 1)
        rr1 = min(rows, top + h + 1)
        cc1 = min(cols, left + w + 1)
        reserved[rr0:rr1, cc0:cc1] = True
        placements[(int(obj["pair_idx"]), str(obj["role"]))] = (top, left)

    expected[:] = grid
    for pair_idx, pair in enumerate(pairs):
        small_top, small_left = placements[(pair_idx, "small")]
        big_top, big_left = placements[(pair_idx, "big")]
        small_mask = np.asarray(pair["small_mask"], dtype=bool)
        big_mask = np.asarray(pair["big_mask"], dtype=bool)

        expected[small_top : small_top + small_mask.shape[0], small_left : small_left + small_mask.shape[1]][small_mask] = bg
        expected[big_top : big_top + big_mask.shape[0], big_left : big_left + big_mask.shape[1]][big_mask] = int(pair["small_color"])

    if _most_common_color(grid) != bg:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scaled-pair input")
