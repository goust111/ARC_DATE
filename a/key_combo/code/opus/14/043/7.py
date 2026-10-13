from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 500
STRIPE_PERIODS = (3, 4, 5)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _infer_stripes(grid: np.ndarray) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    visible_rows = [r for r in range(h) if np.any(g[r] == RED)]
    assert len(visible_rows) >= 2, "need multiple visible stripe rows to infer the period"

    candidates: list[tuple[int, list[int]]] = []
    for k in STRIPE_PERIODS:
        residues = {int(r % k) for r in visible_rows}
        if len(residues) != 1:
            continue
        residue = next(iter(residues))
        stripe_rows = list(range(residue, h, k))
        if any(np.any(g[r] == RED) for r in stripe_rows):
            candidates.append((k, stripe_rows))

    assert len(candidates) == 1, "stripe period must be uniquely recoverable"
    return candidates[0]


def _analyze(grid: np.ndarray) -> tuple[int, list[int], list[np.ndarray], dict[int, list[np.ndarray]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    k, stripe_rows = _infer_stripes(g)
    objects = _components(g == GREEN)
    assert 4 <= len(objects) <= 6, "expected 4-6 original green objects"

    owner = np.full(g.shape, -1, dtype=int)
    for obj_id, coords in enumerate(objects):
        owner[coords[:, 0], coords[:, 1]] = obj_id

    cut_mask = (g == GREEN).copy()
    cut_mask[stripe_rows, :] = False
    fragments = _components(cut_mask)

    grouped: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(objects))}
    for frag in fragments:
        ids = {int(owner[int(r), int(c)]) for r, c in frag}
        ids.discard(-1)
        assert len(ids) == 1, "each fragment must belong to exactly one original object"
        grouped[next(iter(ids))].append(frag)

    counts = [len(grouped[idx]) for idx in range(len(objects))]
    assert all(count > 0 for count in counts), "every original object must leave at least one fragment"
    return k, stripe_rows, objects, grouped, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut green objects along periodic horizontal red stripes and recolor the resulting fragments by fragment-count uniqueness.

    Input structure:
    - Black (0) background with full-width horizontal red (2) stripe rows repeating every 3-5 rows.
    - Several irregular green (3) 4-connected objects overlap some stripe cells, so green hides parts of the red rows.
    - Each object is tall enough to cross at least two stripe rows, while uncovered red cells still reveal the stripe period.

    Rule:
    - Infer the full set of stripe rows from the visible red periodic pattern.
    - For each original green object, erase its green pixels that lie on stripe rows, restoring those cells to red.
    - Split the remaining green pixels into 4-connected fragments and group each fragment back to its original object.
    - Count how many fragments each original object produces.
    - Recolor every fragment of an object blue (1) if that fragment count is unique among objects, otherwise yellow (4).
    - Keep all stripe rows red and leave the black background unchanged.

    Key insight:
    - The answer depends on the interaction between object geometry and the periodic cut rows: fragment counts emerge only after cutting, not from the stripes or objects alone.
    """
    g = np.asarray(grid, dtype=int)
    _k, stripe_rows, _objects, grouped, counts = _analyze(g)
    freq = Counter(counts)

    out = np.zeros_like(g)
    out[g == RED] = RED
    out[stripe_rows, :] = RED

    for obj_id, frags in grouped.items():
        color = BLUE if freq[counts[obj_id]] == 1 else YELLOW
        for frag in frags:
            out[frag[:, 0], frag[:, 1]] = color

    return out


def _count_profiles(k: int) -> list[list[int]]:
    if k == 5:
        return [
            [3, 3, 3, 4],
            [3, 3, 3, 4, 3],
            [4, 3, 3, 3, 3],
        ]
    return [
        [3, 3, 4, 5],
        [3, 4, 4, 5],
        [3, 3, 4, 4, 5],
        [3, 3, 4, 5, 5],
        [3, 4, 4, 5, 5],
    ]


def _build_object_patch(k: int, fragments: int) -> set[tuple[int, int]] | None:
    islands_needed = fragments - 3
    max_base_size = 15 - 3 * islands_needed
    max_sum = max_base_size - (k + 1)
    if max_sum < k - 1:
        return None

    for _ in range(200):
        candidates: list[tuple[int, int]] = []
        for top_len in range(1, k):
            for bot_len in range(1, k):
                total = top_len + bot_len
                if k - 1 <= total <= max_sum:
                    candidates.append((top_len, bot_len))
        if not candidates:
            return None

        top_len, bot_len = random.choice(candidates)
        cells: set[tuple[int, int]] = set()
        spine_col = 0
        stripe1 = k - 1
        stripe2 = 2 * k - 1

        for r in range(stripe1 - top_len, stripe1):
            cells.add((r, spine_col))
        cells.add((stripe1, spine_col))
        for r in range(stripe1 + 1, stripe2):
            cells.add((r, spine_col))
        cells.add((stripe2, spine_col))
        for r in range(stripe2 + 1, stripe2 + 1 + bot_len):
            cells.add((r, spine_col))

        island_bands = random.sample(["top", "mid", "bot"], islands_needed)
        forbidden_nubs: set[tuple[int, int]] = set()
        for band in island_bands:
            sign = random.choice((-1, 1))
            end_col = 2 * sign
            if band == "top":
                sr = stripe1
                ir = stripe1 - 1
            elif band == "mid":
                if random.random() < 0.5:
                    sr = stripe1
                    ir = stripe1 + 1
                else:
                    sr = stripe2
                    ir = stripe2 - 1
            else:
                sr = stripe2
                ir = stripe2 + 1
            cells.add((sr, sign))
            cells.add((sr, end_col))
            cells.add((ir, end_col))
            forbidden_nubs.add((ir, sign))

        budget = 15 - len(cells)
        nub_choices: list[tuple[int, int, int]] = []

        top_rows = list(range(stripe1 - top_len, stripe1))
        mid_rows = list(range(stripe1 + 1, stripe2))
        bot_rows = list(range(stripe2 + 1, stripe2 + 1 + bot_len))

        if top_rows:
            nub_choices.append((random.choice(top_rows), random.choice((-1, 1)), 0))
        if mid_rows:
            nub_choices.append((random.choice(mid_rows), random.choice((-1, 1)), 0))
        if bot_rows:
            nub_choices.append((random.choice(bot_rows), random.choice((-1, 1)), 0))
        random.shuffle(nub_choices)

        nubs_to_add = 0
        if fragments == 3:
            nubs_to_add = min(budget, random.randint(1, min(2, len(nub_choices))))
        elif budget > 0 and random.random() < 0.5:
            nubs_to_add = 1

        added = 0
        for rr, sign, _ in nub_choices:
            if added >= nubs_to_add:
                break
            candidate = (rr, sign)
            if candidate in cells or candidate in forbidden_nubs:
                continue
            cells.add(candidate)
            added += 1

        rows = np.array(sorted(r for r, _c in cells), dtype=int)
        cols = np.array(sorted(c for _r, c in cells), dtype=int)
        if rows.max() - rows.min() + 1 < 2 * k:
            continue
        if not (6 <= len(cells) <= 15):
            continue

        norm = {(r, c - int(cols.min())) for r, c in cells}
        mask_h = int(rows.max()) + 1
        mask_w = int(max(c for _r, c in norm)) + 1
        mask = np.zeros((mask_h, mask_w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        comps = _components(mask)
        if len(comps) != 1:
            continue

        cut_mask = mask.copy()
        cut_mask[[stripe1, stripe2], :] = False
        if len(_components(cut_mask)) != fragments:
            continue
        return norm
    return None


def _place_patches(
    patches: list[set[tuple[int, int]]],
    k: int,
    num_stripes: int,
) -> np.ndarray | None:
    widths = [1 + max(c for _r, c in patch) for patch in patches]
    gaps = [random.randint(2, 3) for _ in range(max(0, len(patches) - 1))]
    base_width = 2 + sum(widths) + sum(gaps)
    if base_width > 30:
        return None

    w = random.randint(max(18, base_width), min(30, base_width + 4))
    slack = w - base_width
    extras = [0] * (len(patches) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1

    h = (num_stripes + 1) * k - 1
    grid = np.zeros((h, w), dtype=int)
    stripe_rows = list(range(k - 1, h, k))
    grid[stripe_rows, :] = RED

    x = extras[0] + 1
    for idx, patch in enumerate(patches):
        start_band = random.randint(0, num_stripes - 2)
        row_shift = start_band * k
        for r, c in patch:
            rr = row_shift + r
            cc = x + c
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            grid[rr, cc] = GREEN
        if idx < len(patches) - 1:
            x += widths[idx] + gaps[idx] + extras[idx + 1]
    return grid


def _semantic_checks(grid: np.ndarray, intended_counts: list[int]) -> bool:
    try:
        k, stripe_rows, objects, grouped, counts = _analyze(grid)
    except AssertionError:
        return False

    if sorted(counts) != sorted(intended_counts):
        return False
    freq = Counter(counts)
    if not any(v == 1 for v in freq.values()):
        return False
    if not any(v >= 2 for v in freq.values()):
        return False

    stripe_set = set(stripe_rows)
    object_keys = set()
    for coords in objects:
        if not (6 <= len(coords) <= 15):
            return False
        r0, _c0, r1, _c1 = _bbox(coords)
        if r1 - r0 + 1 < 2 * k:
            return False
        touched = sum(1 for r in stripe_rows if np.any(coords[:, 0] == r))
        if touched < 2:
            return False
        key = _shape_key(coords)
        if key in object_keys:
            return False
        object_keys.add(key)

    if len(set(counts)) < 2:
        return False
    if not any(count > 3 for count in counts):
        return False

    repeated_counts = [count for count, n in freq.items() if n >= 2]
    for repeated in repeated_counts:
        keys = set()
        for obj_id, obj in enumerate(objects):
            if counts[obj_id] != repeated:
                continue
            keys.add(_shape_key(obj))
        if len(keys) < 2:
            return False

    stripe_visibility = [int(np.count_nonzero(grid[r] == RED)) for r in stripe_rows]
    if min(stripe_visibility) <= 0:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == BLUE) == 0 or np.count_nonzero(out == YELLOW) == 0:
        return False

    # All objects cross exactly two stripe rows, so fragment-count differences cannot be reduced to "stripes crossed".
    for coords in objects:
        touched = sum(1 for r in stripe_set if np.any(coords[:, 0] == r))
        if touched != 2:
            return False

    return True


def _generate_once() -> np.ndarray | None:
    k = random.choice(STRIPE_PERIODS)
    max_stripes = min(6, (30 + 1) // k - 1)
    num_stripes = random.randint(4, max_stripes)
    counts = random.choice(_count_profiles(k))[:]
    random.shuffle(counts)

    patches: list[set[tuple[int, int]]] = []
    for count in counts:
        patch = _build_object_patch(k, count)
        if patch is None:
            return None
        patches.append(patch)

    random.shuffle(patches)
    grid = _place_patches(patches, k, num_stripes)
    if grid is None:
        return None
    if not _semantic_checks(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
