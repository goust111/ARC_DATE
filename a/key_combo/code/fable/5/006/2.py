from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
RED = 2
GREY = 5
MASK = 8
OBJECT_COLORS = (1, 3, 4, 6, 7, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEXT_OBJECT_COUNT = random.choice((4, 6))


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "shape must be nonempty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(map(tuple, np.argwhere(mask).tolist()))


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight(mask)
    result: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for turns in range(4):
        rotated = np.rot90(base, turns)
        for candidate in (rotated, np.fliplr(rotated)):
            candidate = _tight(candidate)
            key = (candidate.shape, _mask_key(candidate))
            if key not in seen:
                seen.add(key)
                result.append(candidate.copy())
    return result


def _signature(mask: np.ndarray) -> tuple[object, ...]:
    keys = []
    for variant in _variants(mask):
        keys.append((variant.shape, _mask_key(variant)))
    return min(keys)


def _legend_and_table(g: np.ndarray, bg: int) -> tuple[list[int], int]:
    order: list[int] = []
    for r in range(g.shape[0]):
        value = int(g[r, -1])
        if value == bg:
            break
        order.append(value)

    assert len(order) >= 4 and len(order) % 2 == 0, "expected an even right-edge legend"
    assert len(order) == len(set(order)), "legend colors must be unique"
    assert not ({RED, GREY, MASK, bg} & set(order)), "role colors cannot be object labels"

    n = len(order)
    table_size = 3 * n + 2
    assert table_size < g.shape[0] and table_size <= g.shape[1]
    assert np.all(g[n:table_size, -1] == bg), "legend must be isolated on the right edge"

    for i, color in enumerate(order):
        band = 3 + 3 * i
        assert np.all(g[0:2, band : band + 2] == color)
        assert np.all(g[band : band + 2, 0:2] == color)

    for i in range(n):
        red_partners = []
        for j in range(n):
            r, c = 3 + 3 * i, 3 + 3 * j
            block = g[r : r + 2, c : c + 2]
            if i == j:
                assert np.all(block == bg), "table diagonal must be blank"
            else:
                values = np.unique(block)
                assert values.size == 1 and int(values[0]) in {RED, GREY}
                if int(values[0]) == RED:
                    red_partners.append(j)
                reverse = g[3 + 3 * j : 5 + 3 * j, 3 + 3 * i : 5 + 3 * i]
                assert np.array_equal(block, reverse), "equivalence table must be symmetric"
        assert len(red_partners) == 1, "each color must have exactly one red partner"

    return order, table_size + 1


def _cross_center(coords: set[tuple[int, int]]) -> tuple[int, int]:
    assert len(coords) == 5, "the occluder must have five cells"
    for r, c in coords:
        expected = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        if expected == coords:
            return r, c
    raise AssertionError("color 8 must form one 3x3 cross")


def _scene_shape(g: np.ndarray, scene_start: int, color: int) -> np.ndarray:
    scene = g[scene_start:, :]
    coords = np.argwhere(scene == int(color))
    assert coords.size, f"missing scene object of color {color}"
    mask = np.zeros_like(scene, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return _tight(mask)


def _completions(
    donor_mask: np.ndarray,
    visible: set[tuple[int, int]],
    cover: set[tuple[int, int]],
    shape: tuple[int, int],
    scene_start: int,
) -> list[frozenset[tuple[int, int]]]:
    """Return distinct donor placements whose only hidden/extra cells lie under cover."""
    assert visible, "damaged object must retain visible pixels"
    height, width = shape
    solutions: set[frozenset[tuple[int, int]]] = set()
    for variant in _variants(donor_mask):
        rel = [tuple(map(int, p)) for p in np.argwhere(variant)]
        for vr, vc in visible:
            for pr, pc in rel:
                dr, dc = vr - pr, vc - pc
                absolute = frozenset((dr + r, dc + c) for r, c in rel)
                if any(r < scene_start or r >= height or c < 0 or c >= width for r, c in absolute):
                    continue
                if set(absolute) - cover == visible:
                    solutions.add(absolute)
    return sorted(solutions, key=lambda cells: tuple(sorted(cells)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the object hidden by a grey cross using the donor named by the equivalence table.

    Input structure:
    - A dominant background surrounds a top-left color-labeled equivalence table, a
      top-to-bottom right-edge legend, and one separated monochrome scene object per label.
    - Red 2x2 table cells mark dihedral-equivalent pairs, grey 2x2 cells mark unequal pairs,
      and exactly one scene object is partly covered by a five-cell color-8 cross.

    Rule:
    - Identify the damaged color beside the cross, read its sole red-table partner, and
      rotate or reflect that partner's tight scene shape until its visible cells align uniquely.
    - Replace covered donor cells by the damaged color and covered non-shape cells by the
      background; keep every pixel outside the cross unchanged.

    Key insight:
    - Several unequal shapes fit the visible fragment, so the table must select the donor
      before the unique dihedral alignment can reveal which cross cells belong to the object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    order, scene_start = _legend_and_table(g, bg)
    assert scene_start < g.shape[0]
    assert np.all(g[scene_start - 1, :] == bg), "table and scene require a blank separator"

    cover = {tuple(map(int, p)) for p in np.argwhere(g == MASK) if int(p[0]) >= scene_start}
    _cross_center(cover)
    assert int(np.sum(g[:scene_start] == MASK)) == 0, "mask color is reserved for the scene cross"

    adjacent: set[int] = set()
    for r, c in cover:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and (nr, nc) not in cover:
                value = int(g[nr, nc])
                if value in order:
                    adjacent.add(value)
    assert len(adjacent) == 1, "the cross must visibly touch exactly one labeled object"
    target = next(iter(adjacent))
    target_idx = order.index(target)

    partner_indices = []
    for j in range(len(order)):
        if j == target_idx:
            continue
        r, c = 3 + 3 * target_idx, 3 + 3 * j
        if int(g[r, c]) == RED:
            partner_indices.append(j)
    assert len(partner_indices) == 1
    donor_color = order[partner_indices[0]]

    donor = _scene_shape(g, scene_start, donor_color)
    visible = {
        tuple(map(int, p))
        for p in np.argwhere(g == target)
        if int(p[0]) >= scene_start
    }
    solutions = _completions(donor, visible, cover, g.shape, scene_start)
    assert len(solutions) == 1, "table-selected donor must have one alignment"

    completed = solutions[0]
    assert completed & cover, "the donor must restore at least one covered cell"
    out = g.copy()
    for r, c in cover:
        out[r, c] = target if (r, c) in completed else bg
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if not coords.size:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                if bool(mask[nr, nc]) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _cross4(anchor: tuple[int, int]) -> np.ndarray:
    r, c = anchor
    result = np.zeros((4, 4), dtype=bool)
    result[r, c] = True
    for dr, dc in NEIGHBORS4:
        result[r + dr, c + dc] = True
    return result


def _local_completions(
    shape_mask: np.ndarray, visible_mask: np.ndarray, cover_mask: np.ndarray
) -> list[frozenset[tuple[int, int]]]:
    visible = {tuple(map(int, p)) for p in np.argwhere(visible_mask)}
    cover = {tuple(map(int, p)) for p in np.argwhere(cover_mask)}
    return _completions(shape_mask, visible, cover, (4, 4), 0)


def _ambiguous_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make unequal shapes sharing exactly the same deliberately ambiguous visible fragment."""
    for _ in range(2000):
        anchor = random.choice(((1, 1), (1, 2), (2, 1), (2, 2)))
        cover = _cross4(anchor)
        visible = (~cover) & (np.random.random((4, 4)) < random.uniform(0.48, 0.72))

        # The visible fragment fixes a 4x4 local reference frame and touches the mask.
        if int(visible.sum()) < 5:
            continue
        if not all(bool(visible[edge].any()) for edge in (0, -1)):
            continue
        if not bool(visible[:, 0].any()) or not bool(visible[:, -1].any()):
            continue
        cover_cells = [tuple(map(int, p)) for p in np.argwhere(cover)]
        touching = any(
            0 <= r + dr < 4
            and 0 <= c + dc < 4
            and bool(visible[r + dr, c + dc])
            for r, c in cover_cells
            for dr, dc in NEIGHBORS4
        )
        if not touching:
            continue

        a_hidden = set(random.sample(cover_cells, random.randint(2, 4)))
        b_hidden = set(random.sample(cover_cells, random.randint(2, 4)))
        if a_hidden == b_hidden:
            continue
        a = visible.copy()
        b = visible.copy()
        for cell in a_hidden:
            a[cell] = True
        for cell in b_hidden:
            b[cell] = True
        if not _connected(a) or not _connected(b):
            continue
        if _signature(a) == _signature(b):
            continue
        a_solutions = _local_completions(a, visible, cover)
        b_solutions = _local_completions(b, visible, cover)
        if len(a_solutions) != 1 or not b_solutions:
            continue
        if all(solution == a_solutions[0] for solution in b_solutions):
            continue
        return a, b, cover
    raise RuntimeError("failed to construct ambiguous shape pair")


def _third_shape(excluded: set[tuple[object, ...]]) -> np.ndarray:
    for _ in range(1000):
        mask = np.random.random((4, 4)) < random.uniform(0.42, 0.72)
        if not 6 <= int(mask.sum()) <= 12 or not _connected(mask):
            continue
        if not (bool(mask[0].any()) and bool(mask[-1].any())):
            continue
        if not (bool(mask[:, 0].any()) and bool(mask[:, -1].any())):
            continue
        sig = _signature(mask)
        if sig not in excluded:
            return mask
    raise RuntimeError("failed to construct third shape")


def _orient(mask: np.ndarray) -> np.ndarray:
    result = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        result = np.fliplr(result)
    return result.copy()


def _positions(n: int, height: int, width: int, scene_start: int) -> list[tuple[int, int]]:
    if n == 6:
        row_starts = [scene_start, scene_start + 5]
        choices = list(itertools.combinations(range(1, width - 4), 3))
        choices = [cols for cols in choices if all(b - a >= 5 for a, b in zip(cols, cols[1:]))]
        cols = list(random.choice(choices))
        return [(r, c) for r in row_starts for c in cols]

    if random.random() < 0.5:
        row = random.randint(scene_start, height - 4)
        choices = list(itertools.combinations(range(1, width - 4), 4))
        choices = [cols for cols in choices if all(b - a >= 5 for a, b in zip(cols, cols[1:]))]
        if choices:
            return [(row, c) for c in random.choice(choices)]

    rows = [scene_start, scene_start + 5]
    col_choices = list(itertools.combinations(range(1, width - 4), 2))
    col_choices = [cols for cols in col_choices if cols[1] - cols[0] >= 5]
    cols = list(random.choice(col_choices))
    return [(r, c) for r in rows for c in cols]


def _draw_table(grid: np.ndarray, order: list[int], classes: dict[int, int]) -> None:
    for i, color in enumerate(order):
        band = 3 + 3 * i
        grid[0:2, band : band + 2] = color
        grid[band : band + 2, 0:2] = color
        grid[i, -1] = color
    for i, row_color in enumerate(order):
        for j, col_color in enumerate(order):
            if i == j:
                continue
            r, c = 3 + 3 * i, 3 + 3 * j
            grid[r : r + 2, c : c + 2] = (
                RED if classes[row_color] == classes[col_color] else GREY
            )


def _semantic_accept(
    inp: np.ndarray,
    expected: np.ndarray,
    target: int,
    donor: int,
    distractor: int,
) -> bool:
    try:
        bg = _background(inp)
        order, scene_start = _legend_and_table(inp, bg)
        solved = main(inp)
        if not np.array_equal(solved, expected):
            return False

        changed = {tuple(map(int, p)) for p in np.argwhere(inp != solved)}
        cover = {tuple(map(int, p)) for p in np.argwhere(inp == MASK)}
        if changed != cover or len(cover) != 5:
            return False
        hidden = sum(int(expected[cell]) == target for cell in cover)
        if not 2 <= hidden <= 4:
            return False

        target_idx = order.index(target)
        partner_indices = [
            j
            for j in range(len(order))
            if j != target_idx and int(inp[3 + 3 * target_idx, 3 + 3 * j]) == RED
        ]
        if len(partner_indices) != 1 or order[partner_indices[0]] != donor:
            return False

        # Recompute the complete equivalence matrix independently from the expected scene.
        signatures = {
            color: _signature(_scene_shape(expected, scene_start, color)) for color in order
        }
        for i, row_color in enumerate(order):
            for j, col_color in enumerate(order):
                if i == j:
                    continue
                cell_is_red = int(inp[3 + 3 * i, 3 + 3 * j]) == RED
                if cell_is_red != (signatures[row_color] == signatures[col_color]):
                    return False

        visible = {
            tuple(map(int, p)) for p in np.argwhere(inp == target) if int(p[0]) >= scene_start
        }
        donor_solutions = _completions(
            _scene_shape(inp, scene_start, donor), visible, cover, inp.shape, scene_start
        )
        distractor_solutions = _completions(
            _scene_shape(inp, scene_start, distractor), visible, cover, inp.shape, scene_start
        )
        if len(donor_solutions) != 1 or not distractor_solutions:
            return False
        if donor_solutions[0] != frozenset(
            tuple(map(int, p)) for p in np.argwhere(expected == target) if int(p[0]) >= scene_start
        ):
            return False
        if not any(solution != donor_solutions[0] for solution in distractor_solutions):
            return False
        return True
    except (AssertionError, ValueError):
        return False


def _generate_attempt(n: int) -> np.ndarray | None:
    assert n in (4, 6)
    colors = random.sample(list(OBJECT_COLORS), n)
    a, b, cover_canonical = _ambiguous_pair()
    class_masks = [a, b]
    if n == 6:
        class_masks.append(_third_shape({_signature(a), _signature(b)}))

    classes: dict[int, int] = {}
    for class_id in range(n // 2):
        for color in colors[2 * class_id : 2 * class_id + 2]:
            classes[color] = class_id
    target = random.choice(colors[0:2])
    donor = colors[1] if target == colors[0] else colors[0]
    distractor = random.choice(colors[2:4])

    order = colors.copy()
    random.shuffle(order)
    table_size = 3 * n + 2
    scene_start = table_size + 1
    if n == 6:
        height = 30
        width = random.randint(23, 30)
    else:
        height = random.randint(25, 30)
        width = random.randint(24, 30)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    _draw_table(grid, order, classes)
    places = _positions(n, height, width, scene_start)
    random.shuffle(places)

    full_target: set[tuple[int, int]] | None = None
    target_cover: set[tuple[int, int]] | None = None
    for color, (r0, c0) in zip(colors, places):
        canonical = class_masks[classes[color]]
        turns = random.randrange(4)
        flip = random.random() < 0.5
        oriented = np.rot90(canonical, turns)
        oriented_cover = np.rot90(cover_canonical, turns)
        if flip:
            oriented = np.fliplr(oriented)
            oriented_cover = np.fliplr(oriented_cover)
        patch = grid[r0 : r0 + 4, c0 : c0 + 4]
        assert np.all(patch == BACKGROUND)
        patch[oriented] = color
        if color == target:
            full_target = {
                (r0 + int(r), c0 + int(c)) for r, c in np.argwhere(oriented)
            }
            target_cover = {
                (r0 + int(r), c0 + int(c)) for r, c in np.argwhere(oriented_cover)
            }

    assert full_target is not None and target_cover is not None
    expected = grid.copy()
    inp = grid.copy()
    for cell in target_cover:
        inp[cell] = MASK

    if not _semantic_accept(inp, expected, target, donor, distractor):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _NEXT_OBJECT_COUNT
    n = _NEXT_OBJECT_COUNT
    for _ in range(300):
        result = _generate_attempt(n)
        if result is not None:
            _NEXT_OBJECT_COUNT = 6 if n == 4 else 4
            return result
    raise RuntimeError("failed to generate a valid table-directed occlusion puzzle")
