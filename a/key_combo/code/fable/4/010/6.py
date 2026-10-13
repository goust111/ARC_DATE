from __future__ import annotations

from collections import Counter
import random

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_CALL_COUNT = 0
_GEN_RETRIES = 300
_GEN_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _unique_mode(values: np.ndarray, message: str) -> int:
    colors, counts = np.unique(values, return_counts=True)
    largest = int(counts.max())
    winners = colors[counts == largest]
    assert winners.size == 1, message
    return int(winners[0])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as (row, column) coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        components.append(np.asarray(cells, dtype=int))
    return components


def _ranked_objects(upper: np.ndarray, background: int) -> list[dict]:
    objects: list[dict] = []
    for coords in _components4(upper != background):
        rows, cols = coords[:, 0], coords[:, 1]
        values = upper[rows, cols]
        body = _unique_mode(values, "each object must have one dominant body color")
        impurities = int(np.sum(values != body))
        objects.append(
            {
                "area": int(len(coords)),
                "body": body,
                "impurities": impurities,
                "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
            }
        )

    assert objects, "expected objects above the pointer row"
    objects.sort(key=lambda obj: obj["area"], reverse=True)
    areas = [obj["area"] for obj in objects]
    assert len(areas) == len(set(areas)), "object areas must be pairwise distinct"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Follow the pointer-count/impurity-count rank chain and crop its clean endpoint.

    Input structure:
    - The bottom row is a solid-color pointer row with a few cells of another color; above it,
      a dominant background separates multicolor 4-connected objects of pairwise distinct area.
    - Each object has one dominant body color. Its differently colored embedded cells are impurities,
      while an object with no such cells is clean.

    Rule:
    - Count cells unlike the dominant color of the bottom row to obtain rank k, then select the
      k-th largest upper object, counting every body and impurity cell in its area.
    - If the selected object has m impurities, jump to the m-th largest object and repeat. When a
      clean object is reached, return its inclusive minimum-bounding-box crop from the input.

    Key insight:
    - A dirty selected object is a pointer rather than the answer: its impurity count supplies the
      next area rank, so selection must continue until a clean endpoint is reached.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3, "expected a 2D ARC grid"

    pointer_row = g[-1]
    pointer_color = _unique_mode(pointer_row, "pointer row must have one dominant color")
    rank = int(np.sum(pointer_row != pointer_color))
    assert rank > 0, "pointer row must contain at least one marker cell"

    upper = g[:-1]
    background = _unique_mode(upper, "upper grid must have one dominant background color")
    objects = _ranked_objects(upper, background)
    assert rank <= len(objects), "pointer count must be a valid area rank"

    visited: set[int] = set()
    while True:
        assert rank not in visited, "pointer chain must not contain a cycle"
        visited.add(rank)
        obj = objects[rank - 1]
        next_rank = int(obj["impurities"])
        if next_rank == 0:
            r0, c0, r1, c1 = obj["bbox"]
            return g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert 1 <= next_rank <= len(objects), "impurity count must name an existing area rank"
        rank = next_rank


# ------------------------------ generator ------------------------------


def _make_shapes(count: int) -> list[np.ndarray] | None:
    """Create distinct-area shapes with an intact interior large enough for any rank count."""
    shapes: list[np.ndarray] = []
    areas: list[int] = []
    for _ in range(count):
        found = None
        for _ in range(100):
            h, w = random.randint(4, 7), random.randint(4, 7)
            if (h - 2) * (w - 2) < count:
                continue
            mask = np.ones((h, w), dtype=bool)

            # Optional boundary notches vary silhouettes without reducing impurity capacity.
            removable = (
                [(0, c) for c in range(2, w - 2)]
                + [(h - 1, c) for c in range(2, w - 2)]
                + [(r, 0) for r in range(2, h - 2)]
                + [(r, w - 1) for r in range(2, h - 2)]
            )
            random.shuffle(removable)
            for r, c in removable[: random.randint(0, min(3, len(removable)))]:
                mask[r, c] = False

            area = int(mask.sum())
            if all(abs(area - old) >= 2 for old in areas):
                found = mask
                break
        if found is None:
            return None
        shapes.append(found)
        areas.append(int(found.sum()))
    return shapes


def _choose_chain(count: int, force_one_hop: bool) -> tuple[list[int], int, dict[int, int]]:
    """Plan two distinguishable terminating branches; return active chain, other terminal, pointers."""
    if force_one_hop:
        chain_length = 1
    else:
        chain_length = random.randint(2, min(4, count - 1))

    ranks = list(range(1, count + 1))
    chain = random.sample(ranks, chain_length)
    other_terminal = random.choice([rank for rank in ranks if rank not in chain])

    impurities = {rank: other_terminal for rank in ranks}
    impurities[chain[-1]] = 0
    impurities[other_terminal] = 0
    for current, following in zip(chain, chain[1:]):
        impurities[current] = following
    return chain, other_terminal, impurities


def _slot_positions(count: int, shapes_by_rank: list[np.ndarray], height: int, width: int):
    columns = 3 if count >= 5 else 2
    slots = [(r, c) for r in range(2) for c in range(columns)][:count]
    rank_order = list(range(1, count + 1))
    for _ in range(30):
        random.shuffle(rank_order)
        if rank_order != sorted(rank_order) and rank_order != sorted(rank_order, reverse=True):
            break

    placements: dict[int, tuple[int, int, np.ndarray]] = {}
    for (slot_r, slot_c), rank in zip(slots, rank_order):
        shape = shapes_by_rank[rank - 1]
        h, w = shape.shape
        base_r, base_c = 1 + 9 * slot_r, 1 + 9 * slot_c
        r = base_r + random.randint(0, max(0, 7 - h))
        c = base_c + random.randint(0, max(0, 7 - w))
        if r + h > height - 2 or c + w > width - 1:
            return None
        placements[rank] = (r, c, shape)
    return placements


def _marker_columns(width: int, count: int) -> list[int]:
    candidates = list(range(1, width - 1, 2))
    if random.random() < 0.5:
        candidates = [c + 1 for c in candidates if c + 1 < width - 1]
    return sorted(random.sample(candidates, count))


def _trace_oracle(start: int, impurities: dict[int, int]) -> list[int]:
    trace: list[int] = []
    rank = start
    while True:
        assert rank not in trace and rank in impurities
        trace.append(rank)
        if impurities[rank] == 0:
            return trace
        rank = impurities[rank]


def _attempt_generate(force_one_hop: bool) -> np.ndarray | None:
    count = random.randint(4, 6)
    columns = 3 if count >= 5 else 2
    height = random.randint(21, 30)
    width = random.randint(28 if columns == 3 else 22, 30)

    shapes = _make_shapes(count)
    if shapes is None:
        return None
    shapes_by_rank = sorted(shapes, key=lambda shape: int(shape.sum()), reverse=True)
    areas = [int(shape.sum()) for shape in shapes_by_rank]
    if len(set(areas)) != count:
        return None

    placements = _slot_positions(count, shapes_by_rank, height, width)
    if placements is None:
        return None

    chain, other_terminal, impurities_by_rank = _choose_chain(count, force_one_hop)
    if _trace_oracle(chain[0], impurities_by_rank) != chain:
        return None
    if _trace_oracle(other_terminal, impurities_by_rank) != [other_terminal]:
        return None

    palette = list(range(10))
    background = random.choice(palette)
    impurity_color = random.choice([c for c in palette if c != background])
    body_colors = random.sample([c for c in palette if c not in (background, impurity_color)], count)

    grid = np.full((height, width), background, dtype=int)
    clean_patches: dict[int, np.ndarray] = {}
    for rank in range(1, count + 1):
        r0, c0, shape = placements[rank]
        body = body_colors[rank - 1]
        patch = np.full(shape.shape, background, dtype=int)
        patch[shape] = body

        impurity_count = impurities_by_rank[rank]
        if impurity_count:
            interior = [(r, c) for r in range(1, shape.shape[0] - 1) for c in range(1, shape.shape[1] - 1)]
            if len(interior) < impurity_count:
                return None
            for r, c in random.sample(interior, impurity_count):
                assert shape[r, c]
                patch[r, c] = impurity_color
        else:
            clean_patches[rank] = patch.copy()

        target = grid[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]]
        target[shape] = patch[shape]

    pointer_color, marker_color = random.sample([c for c in palette if c != background], 2)
    grid[-1, :] = pointer_color
    grid[-1, _marker_columns(width, chain[0])] = marker_color

    # Structural oracle checks use the generation plan, not main(generate_input()) alone.
    if not all(1 <= value <= count for value in impurities_by_rank.values() if value):
        return None
    if len([rank for rank, value in impurities_by_rank.items() if value == 0]) < 2:
        return None
    expected = clean_patches[chain[-1]]
    alternate_expected = clean_patches[other_terminal]
    if np.array_equal(expected, alternate_expected):
        return None
    if len(chain) > 1:
        first_patch = grid[
            placements[chain[0]][0] : placements[chain[0]][0] + placements[chain[0]][2].shape[0],
            placements[chain[0]][1] : placements[chain[0]][1] + placements[chain[0]][2].shape[1],
        ]
        if np.array_equal(first_patch, expected):
            return None

    # Verify rendered areas, embedded-count pointers, rank reference frame, and crop oracle.
    upper_bg = _unique_mode(grid[:-1], "generator background must be dominant")
    if upper_bg != background:
        return None
    detected = _ranked_objects(grid[:-1], background)
    if len(detected) != count or [obj["area"] for obj in detected] != areas:
        return None
    if [obj["impurities"] for obj in detected] != [impurities_by_rank[r] for r in range(1, count + 1)]:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    # Counterexample to ignoring the bottom-row count: another valid count reaches another crop.
    alternate = grid.copy()
    alternate[-1, :] = pointer_color
    alternate[-1, _marker_columns(width, other_terminal)] = marker_color
    if not np.array_equal(main(alternate), alternate_expected):
        return None
    if np.array_equal(main(alternate), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked inputs; every three calls include the one-hop branch."""
    global _GEN_CALL_COUNT
    force_one_hop = (_GEN_CALL_COUNT % 3 == 0)
    for _ in range(_GEN_RETRIES):
        grid = _attempt_generate(force_one_hop)
        if grid is not None:
            output = main(grid)
            output_key = (tuple(map(int, output.shape)), output.tobytes())
            if output_key in _GEN_SEEN_OUTPUTS:
                continue
            _GEN_SEEN_OUTPUTS.add(output_key)
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain crop input")
