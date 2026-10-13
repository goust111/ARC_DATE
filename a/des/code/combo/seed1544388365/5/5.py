from __future__ import annotations

from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
BROWN = 9

SIGNAL_POOL = (RED, BLUE, YELLOW)
TABLE_ANCHOR = BROWN
BASE_MIX = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = ("tl", "tr", "bl", "br")
MAX_GENERATE_ATTEMPTS = 600


def _local_to_global(
    shape: tuple[int, int], corner: str, size: int, local_r: int, local_c: int
) -> tuple[int, int]:
    height, width = shape
    if corner == "tl":
        return local_r, local_c
    if corner == "tr":
        return local_r, width - 1 - local_c
    if corner == "bl":
        return height - 1 - local_r, local_c
    return height - 1 - local_r, width - 1 - local_c


def _extract_corner_block(grid: np.ndarray, corner: str, size: int) -> np.ndarray:
    if corner == "tl":
        return grid[:size, :size]
    if corner == "tr":
        return np.fliplr(grid[:size, -size:])
    if corner == "bl":
        return np.flipud(grid[-size:, :size])
    return np.flipud(np.fliplr(grid[-size:, -size:]))


def _corner_square_mask(
    shape: tuple[int, int], corner: str, size: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for local_r in range(size):
        for local_c in range(size):
            r, c = _local_to_global(shape, corner, size, local_r, local_c)
            mask[r, c] = True
    return mask


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - radius)
        r1 = min(height, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(width, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _parse_table_block(
    block: np.ndarray,
) -> tuple[tuple[int, ...], dict[frozenset[int], int]] | None:
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        return None
    size = int(block.shape[0])
    if size not in (3, 4):
        return None
    if np.any((block == BLACK) | (block == GRAY)):
        return None

    anchor = int(block[0, 0])
    if int(np.count_nonzero(block == anchor)) != 1:
        return None

    row_labels = tuple(int(x) for x in block[0, 1:])
    col_labels = tuple(int(x) for x in block[1:, 0])
    if row_labels != col_labels:
        return None
    if len(set(row_labels)) != size - 1:
        return None
    if anchor in row_labels:
        return None

    mix_map: dict[frozenset[int], int] = {}
    has_true_mix = False
    for i, color_i in enumerate(row_labels, start=1):
        if int(block[i, i]) != color_i:
            return None
        for j, color_j in enumerate(row_labels, start=1):
            value = int(block[i, j])
            if value != int(block[j, i]):
                return None
            mix_map[frozenset((color_i, color_j))] = value
            if i != j and value not in (color_i, color_j):
                has_true_mix = True

    if not has_true_mix:
        return None
    return row_labels, mix_map


def _find_table(
    grid: np.ndarray,
) -> tuple[int, str, tuple[int, ...], dict[frozenset[int], int], np.ndarray]:
    candidates: list[
        tuple[int, str, tuple[int, ...], dict[frozenset[int], int], np.ndarray]
    ] = []
    for size in (4, 3):
        if min(grid.shape) < size:
            continue
        for corner in CORNERS:
            parsed = _parse_table_block(_extract_corner_block(grid, corner, size))
            if parsed is None:
                continue
            labels, mix_map = parsed
            table_mask = _corner_square_mask(grid.shape, corner, size)
            candidates.append((size, corner, labels, mix_map, table_mask))
        if candidates:
            break
    assert candidates, "expected a color-mixing table in a corner"
    return candidates[0]


def _solve_scene(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    table_size, corner, labels, mix_map, table_mask = _find_table(g)
    label_set = set(labels)
    wall_mask = g == GRAY

    scene_colors = set(map(int, np.unique(g[~table_mask])))
    assert scene_colors <= ({BLACK, GRAY} | label_set), (
        "outside the table only black, gray, and tower colors are allowed"
    )

    tower_coords = np.argwhere((~table_mask) & np.isin(g, list(label_set)))
    assert tower_coords.size > 0, "expected at least one signal tower"
    towers = sorted((int(r), int(c), int(g[r, c])) for r, c in tower_coords)
    tower_positions = {(r, c) for r, c, _color in towers}

    hits: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    effective_walls: set[tuple[int, int]] = set()
    for tower_idx, (r, c, color) in enumerate(towers):
        for dr, dc in DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            dist = 1
            while 0 <= rr < height and 0 <= cc < width:
                if wall_mask[rr, cc]:
                    effective_walls.add((rr, cc))
                    break
                if not table_mask[rr, cc] and (rr, cc) not in tower_positions:
                    hits[(rr, cc)].append((dist, tower_idx, color))
                rr += int(dr)
                cc += int(dc)
                dist += 1

    out = g.copy()
    mixed_cells: list[tuple[int, int]] = []
    same_color_cells: list[tuple[int, int]] = []
    triple_cells: list[tuple[int, int]] = []
    for (r, c), cell_hits in hits.items():
        cell_hits.sort(key=lambda item: (item[0], item[1]))
        if len(cell_hits) >= 3:
            triple_cells.append((r, c))
        if len(cell_hits) == 1:
            out[r, c] = cell_hits[0][2]
            continue
        color_a = cell_hits[0][2]
        color_b = cell_hits[1][2]
        if color_a == color_b:
            out[r, c] = color_a
            same_color_cells.append((r, c))
        else:
            out[r, c] = mix_map[frozenset((color_a, color_b))]
            mixed_cells.append((r, c))

    info = {
        "corner": corner,
        "labels": labels,
        "table_size": table_size,
        "table_mask": table_mask,
        "towers": towers,
        "tower_positions": tower_positions,
        "hits": hits,
        "mixed_cells": mixed_cells,
        "same_color_cells": same_color_cells,
        "triple_cells": triple_cells,
        "effective_walls": effective_walls,
        "painted_count": len(hits),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast orthogonal rays from the signal towers and mix crossings using the corner legend.

    The corner square is a lookup table whose first row and first column label the
    available tower colors. Each tower emits rays in the four cardinal directions
    until a gray wall or the grid boundary stops the ray. Towers, walls, and the
    legend remain unchanged. For every non-fixed cell reached by rays, keep the
    tower color if only one color reaches it; otherwise mix the two nearest rays
    according to the table, with same-color overlaps staying unchanged.
    """
    return _solve_scene(grid)[0]


def _make_table_block(labels: tuple[int, ...]) -> np.ndarray:
    size = len(labels) + 1
    block = np.full((size, size), TABLE_ANCHOR, dtype=int)
    block[0, 0] = TABLE_ANCHOR
    for i, color_i in enumerate(labels, start=1):
        block[0, i] = color_i
        block[i, 0] = color_i
        for j, color_j in enumerate(labels, start=1):
            if color_i == color_j:
                block[i, j] = color_i
            else:
                block[i, j] = BASE_MIX[frozenset((color_i, color_j))]
    return block


def _stamp_corner_block(grid: np.ndarray, block: np.ndarray, corner: str) -> None:
    size = int(block.shape[0])
    for local_r in range(size):
        for local_c in range(size):
            r, c = _local_to_global(grid.shape, corner, size, local_r, local_c)
            grid[r, c] = int(block[local_r, local_c])


def _pick_tower_position(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked_mask: np.ndarray,
    towers: list[tuple[int, int, int]],
    protected_cells: set[tuple[int, int]],
    avoid_rows: set[int],
    avoid_cols: set[int],
) -> tuple[int, int] | None:
    height, width = shape
    used_rows = {r for r, _c, _color in towers}
    used_cols = {c for _r, c, _color in towers}

    fresh: list[tuple[int, int]] = []
    partial: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        if r in avoid_rows:
            continue
        for c in range(1, width - 1):
            if c in avoid_cols or blocked_mask[r, c] or (r, c) in protected_cells:
                continue
            if any(max(abs(r - tr), abs(c - tc)) <= 1 for tr, tc, _color in towers):
                continue
            if r not in used_rows and c not in used_cols:
                fresh.append((r, c))
            elif r not in used_rows or c not in used_cols:
                partial.append((r, c))
            else:
                fallback.append((r, c))

    candidates = fresh or partial or fallback
    if not candidates:
        return None
    idx = int(rng.integers(len(candidates)))
    return tuple(map(int, candidates[idx]))


def _place_base_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked_mask: np.ndarray,
    color_a: int,
    color_b: int,
) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    height, width = shape
    cells = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if not blocked_mask[r, c]
    ]
    if len(cells) < 2:
        return None

    order = list(map(int, rng.permutation(len(cells))))
    for idx_a in order:
        r1, c1 = cells[idx_a]
        for idx_b in order:
            if idx_a == idx_b:
                continue
            r2, c2 = cells[idx_b]
            if r1 == r2 or c1 == c2:
                continue
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                continue
            crossings = [(r1, c2), (r2, c1)]
            if any(blocked_mask[r, c] for r, c in crossings):
                continue
            protected = crossings[int(rng.integers(2))]
            towers = [(r1, c1, color_a), (r2, c2, color_b)]
            return towers, (int(protected[0]), int(protected[1]))
    return None


def _place_focus_towers(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked_mask: np.ndarray,
    signal_colors: tuple[int, ...],
    want_same_priority: bool,
) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    height, width = shape
    focus_candidates = [
        (r, c)
        for r in range(5, height - 5)
        for c in range(5, width - 5)
        if not blocked_mask[r, c]
    ]
    if not focus_candidates:
        return None

    patterns = [
        [(0, -1), (0, 1), (-1, 0)],
        [(0, -1), (0, 1), (1, 0)],
        [(-1, 0), (1, 0), (0, -1)],
        [(-1, 0), (1, 0), (0, 1)],
    ]

    for focus_idx in map(int, rng.permutation(len(focus_candidates))):
        focus_r, focus_c = focus_candidates[focus_idx]
        for pattern_idx in map(int, rng.permutation(len(patterns))):
            pattern = patterns[pattern_idx]
            distances = [int(x) for x in rng.permutation(np.array([2, 3, 4]))]

            placements: list[tuple[int, int, int]] = []
            ok = True
            for (dr, dc), dist in zip(pattern, distances):
                r = int(focus_r + dr * dist)
                c = int(focus_c + dc * dist)
                if not (1 <= r < height - 1 and 1 <= c < width - 1):
                    ok = False
                    break
                if blocked_mask[r, c]:
                    ok = False
                    break
                if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc, _d in placements):
                    ok = False
                    break
                placements.append((r, c, dist))
            if not ok:
                continue

            by_distance = sorted(placements, key=lambda item: item[2])
            if want_same_priority:
                same = int(signal_colors[int(rng.integers(len(signal_colors)))])
                other_choices = [color for color in signal_colors if color != same]
                if not other_choices:
                    continue
                other = int(other_choices[int(rng.integers(len(other_choices)))])
                assigned_colors = [same, same, other]
            else:
                perm = [int(x) for x in rng.permutation(np.array(signal_colors))]
                if len(perm) == 2:
                    assigned_colors = [perm[0], perm[1], perm[int(rng.integers(2))]]
                else:
                    third = perm[2] if rng.random() < 0.7 else perm[int(rng.integers(3))]
                    assigned_colors = [perm[0], perm[1], int(third)]

            towers = [
                (int(r), int(c), int(color))
                for (r, c, _dist), color in zip(by_distance, assigned_colors)
            ]
            return towers, (int(focus_r), int(focus_c))
    return None


def _effective_wall_candidates(
    shape: tuple[int, int],
    blocked_mask: np.ndarray,
    towers: list[tuple[int, int, int]],
    protected_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = shape
    tower_positions = {(r, c) for r, c, _color in towers}
    candidates: set[tuple[int, int]] = set()
    for r, c, _color in towers:
        for dr, dc in DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            dist = 1
            seen_protected = False
            while 0 <= rr < height and 0 <= cc < width:
                cell = (rr, cc)
                if cell in protected_cells:
                    seen_protected = True
                elif (
                    dist >= 2
                    and not seen_protected
                    and not blocked_mask[rr, cc]
                    and cell not in tower_positions
                ):
                    candidates.add(cell)
                rr += int(dr)
                cc += int(dc)
                dist += 1
    return sorted(candidates)


def _place_walls(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked_mask: np.ndarray,
    towers: list[tuple[int, int, int]],
    protected_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    line_candidates = _effective_wall_candidates(shape, blocked_mask, towers, protected_cells)
    if not line_candidates:
        return None

    wall_positions = [line_candidates[int(rng.integers(len(line_candidates)))]]
    occupied = {(r, c) for r, c, _color in towers} | set(wall_positions) | set(protected_cells)
    protected_rows = {r for r, _c in protected_cells}
    protected_cols = {c for _r, c in protected_cells}

    height, width = shape
    extras: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        if r in protected_rows:
            continue
        for c in range(1, width - 1):
            if c in protected_cols:
                continue
            if blocked_mask[r, c] or (r, c) in occupied:
                continue
            extras.append((r, c))

    extra_limit = min(3, len(extras))
    if extra_limit > 0:
        extra_count = int(rng.integers(extra_limit + 1))
        if extra_count:
            picks = np.atleast_1d(rng.choice(len(extras), size=extra_count, replace=False))
            for pick in picks:
                wall_positions.append(extras[int(pick)])

    return wall_positions


def _valid_generated_scene(
    inp: np.ndarray,
    out: np.ndarray,
    info: dict,
    signal_colors: tuple[int, ...],
    want_triple: bool,
    want_same_priority: bool,
) -> bool:
    if inp.shape != out.shape or np.array_equal(inp, out):
        return False
    if not np.array_equal(out[info["table_mask"]], inp[info["table_mask"]]):
        return False

    counts = {int(color): 0 for color in signal_colors}
    for _r, _c, color in info["towers"]:
        if int(color) in counts:
            counts[int(color)] += 1
    if any(not (1 <= count <= 3) for count in counts.values()):
        return False

    if len(info["mixed_cells"]) < 1:
        return False
    if len(info["effective_walls"]) < 1:
        return False
    if info["painted_count"] < 8 or info["painted_count"] > int(inp.size * 0.6):
        return False
    if want_triple and len(info["triple_cells"]) < 1:
        return False
    if want_same_priority and len(info["same_color_cells"]) < 1:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(14, 21))
    width = int(rng.integers(14, 21))
    num_signal_colors = 2 if rng.random() < 0.55 else 3
    signal_colors = tuple(
        int(x) for x in rng.choice(np.array(SIGNAL_POOL), size=num_signal_colors, replace=False)
    )
    labels = tuple(int(x) for x in rng.permutation(np.array(signal_colors)))
    corner = str(CORNERS[int(rng.integers(len(CORNERS)))])

    grid = np.full((height, width), BLACK, dtype=int)
    table = _make_table_block(labels)
    _stamp_corner_block(grid, table, corner)
    table_mask = _corner_square_mask(grid.shape, corner, table.shape[0])
    blocked_mask = _dilate_mask(table_mask, 1)

    want_triple = bool(rng.random() < 0.45)
    want_same_priority = bool(want_triple and rng.random() < 0.4)

    towers: list[tuple[int, int, int]] = []
    protected_cells: set[tuple[int, int]] = set()
    avoid_rows: set[int] = set()
    avoid_cols: set[int] = set()

    if want_triple:
        planned = _place_focus_towers(
            rng, grid.shape, blocked_mask, signal_colors, want_same_priority
        )
        if planned is None:
            return None
        seed_towers, focus = planned
        towers.extend(seed_towers)
        protected_cells.add(focus)
        avoid_rows.add(int(focus[0]))
        avoid_cols.add(int(focus[1]))
    else:
        first_two = [int(x) for x in rng.permutation(np.array(signal_colors))[:2]]
        planned = _place_base_pair(
            rng, grid.shape, blocked_mask, first_two[0], first_two[1]
        )
        if planned is None:
            return None
        seed_towers, protected = planned
        towers.extend(seed_towers)
        protected_cells.add(protected)
        avoid_rows.add(int(protected[0]))
        avoid_cols.add(int(protected[1]))

    color_counts = {int(color): 0 for color in signal_colors}
    for _r, _c, color in towers:
        color_counts[int(color)] += 1

    for color in signal_colors:
        if color_counts[int(color)] >= 1:
            continue
        pos = _pick_tower_position(
            rng, grid.shape, blocked_mask, towers, protected_cells, avoid_rows, avoid_cols
        )
        if pos is None:
            return None
        towers.append((pos[0], pos[1], int(color)))
        color_counts[int(color)] += 1

    low_total = max(len(towers), len(signal_colors) + 1)
    high_total = min(7, len(signal_colors) * 3)
    if low_total > high_total:
        high_total = low_total
    target_total = int(rng.integers(low_total, high_total + 1))
    while len(towers) < target_total:
        available_colors = [color for color, count in color_counts.items() if count < 3]
        if not available_colors:
            break
        weights = np.array([4 - color_counts[int(color)] for color in available_colors], dtype=float)
        weights /= weights.sum()
        color = int(rng.choice(np.array(available_colors), p=weights))
        pos = _pick_tower_position(
            rng, grid.shape, blocked_mask, towers, protected_cells, avoid_rows, avoid_cols
        )
        if pos is None:
            return None
        towers.append((pos[0], pos[1], color))
        color_counts[color] += 1

    walls = _place_walls(rng, grid.shape, blocked_mask, towers, protected_cells)
    if walls is None:
        return None

    for r, c, color in towers:
        grid[r, c] = color
    for r, c in walls:
        grid[r, c] = GRAY

    try:
        out, info = _solve_scene(grid)
    except Exception:
        return None

    if not _valid_generated_scene(
        grid, out, info, signal_colors, want_triple, want_same_priority
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
