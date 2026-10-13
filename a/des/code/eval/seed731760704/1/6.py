from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
FRAME = 9
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OBJECT_COLORS = list(range(8))
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_key(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _mask_from_key(key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in key) + 1
    w = max(c for _, c in key) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key:
        mask[r, c] = True
    return mask


def _color_components(
    g: np.ndarray, color: int, forbidden: np.ndarray | None = None
) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color or (forbidden is not None and forbidden[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    if forbidden is not None and forbidden[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_mapping_table(g: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = g.shape
    candidates: list[tuple[tuple[int, int, int, int], dict[int, int]]] = []

    for comp in _color_components(g, FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_w != 4 or box_h < 3:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for rr, cc in comp:
            mask[rr - r0, cc - c0] = True

        border = np.zeros((box_h, box_w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(mask, border):
            continue

        inner = np.asarray(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
        if inner.shape[1] != 2 or inner.shape[0] < 1:
            continue
        if np.any(inner == FRAME) or np.any(inner == BG):
            continue

        mapping: dict[int, int] = {}
        ok = True
        for src, dst in inner.tolist():
            src = int(src)
            dst = int(dst)
            if src in mapping:
                ok = False
                break
            mapping[src] = dst
        if ok:
            candidates.append(((r0, c0, r1, c1), mapping))

    assert candidates, "expected a maroon-framed 2-column mapping table"

    touching_corner = [
        candidate
        for candidate in candidates
        if candidate[0][2] == h - 1 and candidate[0][3] == w - 1
    ]
    bbox, mapping = max(touching_corner or candidates, key=lambda item: item[0])

    legend_mask = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = bbox
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mapping, legend_mask


def _find_objects(g: np.ndarray, legend_mask: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or legend_mask[r, c] or int(g[r, c]) == BG:
                continue

            color = int(g[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or legend_mask[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            objects.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                    "shape": _shape_key(coords),
                    "bbox": (r0, c0, r1, c1),
                    "top": r0,
                    "left": c0,
                    "area": len(coords),
                }
            )

    assert objects, "expected at least one object outside the mapping table"
    return objects


def _group_pairs(objects: list[dict]) -> dict[tuple[tuple[int, int], ...], list[dict]]:
    groups: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for obj in objects:
        groups.setdefault(obj["shape"], []).append(obj)
    assert all(len(group) == 2 for group in groups.values()), "each exact shape must appear exactly twice"
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair identical monochrome objects, keep the upper/left one, erase the other,
    and optionally recolor the survivor according to the maroon-framed lookup table.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _find_mapping_table(g)
    objects = _find_objects(g, legend_mask)
    groups = _group_pairs(objects)

    out = g.copy()
    for pair in groups.values():
        winner, loser = sorted(pair, key=lambda obj: (obj["top"], obj["left"]))
        out[loser["coords"][:, 0], loser["coords"][:, 1]] = BG
        new_color = int(mapping.get(int(winner["color"]), int(winner["color"])))
        out[winner["coords"][:, 0], winner["coords"][:, 1]] = new_color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return np.asarray(mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1], dtype=bool)


def _random_shape_key(
    rng: random.Random, banned: set[tuple[tuple[int, int], ...]]
) -> tuple[tuple[int, int], ...] | None:
    for _ in range(200):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        area = rng.randint(4, min(8, h * w))
        mask = np.zeros((h, w), dtype=bool)
        sr = rng.randrange(h)
        sc = rng.randrange(w)
        mask[sr, sc] = True

        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(sorted(frontier))
            mask[nr, nc] = True

        mask = _crop_mask(mask)
        key = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))
        if key in banned:
            continue
        if not 4 <= len(key) <= 8:
            continue
        return key
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    grid_h, grid_w = occupied.shape
    if top < 0 or left < 0 or top + h > grid_h or left + w > grid_w:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid_h, top + h + 1)
    c1 = min(grid_w, left + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_box(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[top : top + h, left : left + w] = True


def _feasible_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    grid_h, grid_w = occupied.shape
    positions = []
    for top in range(grid_h - h + 1):
        for left in range(grid_w - w + 1):
            if _can_place(occupied, top, left, h, w):
                positions.append((top, left))
    return positions


def _place_pair(
    occupied: np.ndarray, h: int, w: int, relation: str, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    positions = _feasible_positions(occupied, h, w)
    rng.shuffle(positions)
    for top1, left1 in positions:
        temp = occupied.copy()
        _mark_box(temp, top1, left1, h, w)
        positions2 = _feasible_positions(temp, h, w)
        if relation == "row":
            positions2 = [(r, c) for r, c in positions2 if r > top1]
        else:
            positions2 = [(r, c) for r, c in positions2 if r == top1 and c > left1]
        if not positions2:
            continue
        top2, left2 = rng.choice(positions2)
        _mark_box(occupied, top1, left1, h, w)
        _mark_box(occupied, top2, left2, h, w)
        return (top1, left1), (top2, left2)
    return None


def _paint_shape(
    grid: np.ndarray, key: tuple[tuple[int, int], ...], top: int, left: int, color: int
) -> None:
    for dr, dc in key:
        grid[top + dr, left + dc] = color


def _draw_legend(
    grid: np.ndarray, rows: list[tuple[int, int]], top: int, left: int
) -> None:
    height = len(rows) + 2
    grid[top : top + height, left : left + 4] = BG
    grid[top, left : left + 4] = FRAME
    grid[top + height - 1, left : left + 4] = FRAME
    grid[top : top + height, left] = FRAME
    grid[top : top + height, left + 3] = FRAME
    for idx, (src, dst) in enumerate(rows):
        grid[top + 1 + idx, left + 1] = src
        grid[top + 1 + idx, left + 2] = dst


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([3, 4])
    winner_colors = rng.sample(OBJECT_COLORS, pair_count)
    loser_colors = rng.sample([c for c in OBJECT_COLORS if c not in winner_colors], pair_count)

    relation_types = ["row", "col"] + [rng.choice(["row", "col"]) for _ in range(pair_count - 2)]
    rng.shuffle(relation_types)

    mapped_count = 2 if pair_count == 3 else rng.choice([2, 3])
    mapped_indices = set(rng.sample(range(pair_count), mapped_count))
    target_pool = [c for c in OBJECT_COLORS if c not in winner_colors]
    target_colors = rng.sample(target_pool, mapped_count)

    mapping_rows = []
    mapped_lookup: dict[int, int] = {}
    for idx, target_color in zip(sorted(mapped_indices), target_colors):
        src = winner_colors[idx]
        mapped_lookup[src] = target_color
        mapping_rows.append((src, target_color))
    rng.shuffle(mapping_rows)

    shape_keys: list[tuple[tuple[int, int], ...]] = []
    banned: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(pair_count):
        key = _random_shape_key(rng, banned)
        if key is None:
            return None
        shape_keys.append(key)
        banned.add(key)

    legend_h = len(mapping_rows) + 2
    legend_w = 4

    max_shape_h = max(max(r for r, _ in key) + 1 for key in shape_keys)
    max_shape_w = max(max(c for _, c in key) + 1 for key in shape_keys)
    min_h = max(20, legend_h + max_shape_h + 4)
    min_w = max(20, legend_w + max_shape_w + 8)
    h = rng.randint(min_h, 28)
    w = rng.randint(min_w, 28)

    legend_top = h - legend_h
    legend_left = w - legend_w

    occupied = np.zeros((h, w), dtype=bool)
    _mark_box(occupied, legend_top, legend_left, legend_h, legend_w)

    pairs: list[dict] = []
    order = list(range(pair_count))
    order.sort(
        key=lambda idx: (
            -(max(r for r, _ in shape_keys[idx]) + 1) * (max(c for _, c in shape_keys[idx]) + 1),
            -len(shape_keys[idx]),
        )
    )

    placed: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for idx in order:
        key = shape_keys[idx]
        mask = _mask_from_key(key)
        placement = _place_pair(occupied, mask.shape[0], mask.shape[1], relation_types[idx], rng)
        if placement is None:
            return None
        placed[idx] = placement

    for idx in range(pair_count):
        winner_pos, loser_pos = placed[idx]
        pairs.append(
            {
                "shape": shape_keys[idx],
                "winner_color": winner_colors[idx],
                "loser_color": loser_colors[idx],
                "winner_pos": winner_pos,
                "loser_pos": loser_pos,
                "relation": relation_types[idx],
            }
        )

    grid = np.full((h, w), BG, dtype=int)
    for pair in pairs:
        _paint_shape(grid, pair["shape"], pair["winner_pos"][0], pair["winner_pos"][1], pair["winner_color"])
        _paint_shape(grid, pair["shape"], pair["loser_pos"][0], pair["loser_pos"][1], pair["loser_color"])
    _draw_legend(grid, mapping_rows, legend_top, legend_left)

    parsed_mapping, legend_mask = _find_mapping_table(grid)
    objects = _find_objects(grid, legend_mask)
    groups = _group_pairs(objects)
    if len(groups) != pair_count:
        return None

    row_pairs = 0
    col_pairs = 0
    mapped_winners = 0
    unmapped_winners = 0
    for group in groups.values():
        winner, loser = sorted(group, key=lambda obj: (obj["top"], obj["left"]))
        if winner["top"] == loser["top"]:
            col_pairs += 1
        else:
            row_pairs += 1
        if winner["color"] in parsed_mapping:
            mapped_winners += 1
        else:
            unmapped_winners += 1

    if row_pairs < 1 or col_pairs < 1:
        return None
    if mapped_winners < 1 or unmapped_winners < 1:
        return None
    if parsed_mapping != mapped_lookup:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
