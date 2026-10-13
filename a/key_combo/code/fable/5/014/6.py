from __future__ import annotations

import random

import numpy as np


_ARC_COLORS = tuple(range(10))
_MAX_SIDE = 30


def _background(grid: np.ndarray) -> int:
    """Return the unique dominant color."""
    colors, counts = np.unique(grid, return_counts=True)
    largest = int(counts.max())
    winners = colors[counts == largest]
    assert winners.size == 1, "expected a unique dominant background color"
    return int(winners[0])


def _straight_bar(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int] | None:
    """Return (length, color) when coords are one filled horizontal/vertical bar."""
    if coords.size == 0:
        return None
    values = grid[coords[:, 0], coords[:, 1]]
    colors = np.unique(values)
    if colors.size != 1:
        return None

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    length = int(coords.shape[0])
    horizontal = rows.size == 1 and int(cols[-1] - cols[0] + 1) == length
    vertical = cols.size == 1 and int(rows[-1] - rows[0] + 1) == length
    if not (horizontal or vertical):
        return None
    return length, int(colors[0])


def _role_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    """Enumerate interpretations using the local square as the reference frame."""
    height, width = map(int, grid.shape)
    foreground = grid != bg
    all_coords = np.argwhere(foreground)
    candidates: list[dict] = []

    # Since h >= 1 and the output is n ** (h + 1) <= 30, n cannot exceed 5.
    for n in range(2, min(height, width, 5) + 1):
        for r0 in range(height - n + 1):
            for c0 in range(width - n + 1):
                seed = grid[r0 : r0 + n, c0 : c0 + n]
                seed_values = np.unique(seed[seed != bg])
                if seed_values.size != 1 or not np.any(seed == bg):
                    continue

                seed_mask = seed != bg
                # The visible foreground must make the otherwise invisible n x n
                # local frame tight on all four sides.
                if not (
                    np.any(seed_mask[0, :])
                    and np.any(seed_mask[-1, :])
                    and np.any(seed_mask[:, 0])
                    and np.any(seed_mask[:, -1])
                ):
                    continue

                outside = all_coords[
                    (all_coords[:, 0] < r0)
                    | (all_coords[:, 0] >= r0 + n)
                    | (all_coords[:, 1] < c0)
                    | (all_coords[:, 1] >= c0 + n)
                ]
                bar_info = _straight_bar(grid, outside)
                if bar_info is None:
                    continue
                depth, bar_color = bar_info
                if n ** (depth + 1) > _MAX_SIDE:
                    continue

                # "Separated" is interpreted in the grid's own 4-neighbor frame:
                # no bar cell shares an edge with seed foreground.
                global_seed = np.argwhere(seed_mask) + np.array([r0, c0])
                if any(
                    abs(int(br) - int(sr)) + abs(int(bc) - int(sc)) <= 1
                    for br, bc in outside
                    for sr, sc in global_seed
                ):
                    continue

                candidates.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "n": n,
                        "depth": depth,
                        "seed_mask": seed_mask.copy(),
                        "seed_color": int(seed_values[0]),
                        "bar_color": bar_color,
                        "bar_coords": outside.copy(),
                    }
                )
    return candidates


def _extract_roles(grid: np.ndarray) -> tuple[int, dict]:
    bg = _background(grid)
    candidates = _role_candidates(grid, bg)
    assert len(candidates) == 1, "seed square and depth bar must have one unique interpretation"
    return bg, candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recursively replace foreground cells by the seed mask, using the separate bar length as the recursion depth.

    Input structure:
    - A dominant background surrounds one tight n x n single-color seed mask containing both foreground and background.
    - One separated, filled horizontal or vertical monochrome bar has length h; no other foreground is present.
    - The square seed region and the straight bar admit exactly one structural interpretation, independent of position or color ID.

    Rule:
    - Start with the seed and perform h replacements. At each replacement, every foreground cell becomes an n x n copy of the original seed mask, while every background cell becomes an all-background n x n block.
    - Preserve the seed foreground color and return only the resulting n ** (h + 1) square; the indicator bar is omitted.

    Key insight:
    - The bar is a counter rather than output content: its visible length selects how many self-similar substitution levels to compose.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert max(g.shape) <= _MAX_SIDE, "input must obey the ARC 30 x 30 limit"

    bg, role = _extract_roles(g)
    seed_mask = role["seed_mask"]
    pattern = seed_mask.copy()
    for _ in range(int(role["depth"])):
        pattern = np.kron(pattern, seed_mask)

    assert pattern.shape[0] == pattern.shape[1] <= _MAX_SIDE
    out = np.full(pattern.shape, bg, dtype=int)
    out[pattern.astype(bool)] = int(role["seed_color"])
    return out


# ------------------------------- generator ---------------------------------


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    todo = [tuple(map(int, coords[0]))]
    seen = {todo[0]}
    height, width = mask.shape
    while todo:
        r, c = todo.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                todo.append((rr, cc))
    return len(seen) == int(mask.sum())


def _available_seed_masks(n: int) -> list[np.ndarray]:
    """Enumerate clear, non-solid, connected masks with a tight local bbox."""
    masks: list[np.ndarray] = []
    area = n * n
    lo, hi = (3, 3) if n == 2 else (4, 6)
    for bits in range(1, (1 << area) - 1):
        count = bits.bit_count()
        if not lo <= count <= hi:
            continue
        mask = np.array([(bits >> i) & 1 for i in range(area)], dtype=bool).reshape(n, n)
        if not (
            np.any(mask[0, :])
            and np.any(mask[-1, :])
            and np.any(mask[:, 0])
            and np.any(mask[:, -1])
        ):
            continue
        if _connected(mask):
            masks.append(mask)
    return masks


_MASKS = {2: _available_seed_masks(2), 3: _available_seed_masks(3)}
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_GENERATION_INDEX = 0


def _oracle_expand(seed_mask: np.ndarray, depth: int, bg: int, color: int) -> np.ndarray:
    """Independent base-n-digit oracle for the recursive Kronecker construction."""
    n = int(seed_mask.shape[0])
    side = n ** (depth + 1)
    result_mask = np.ones((side, side), dtype=bool)
    for r in range(side):
        for c in range(side):
            rr, cc = r, c
            for _ in range(depth + 1):
                if not bool(seed_mask[rr % n, cc % n]):
                    result_mask[r, c] = False
                    break
                rr //= n
                cc //= n
    out = np.full((side, side), bg, dtype=int)
    out[result_mask] = color
    return out


def _rectangles_have_blank_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True when the rectangles have at least one complete background cell between them."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _place_objects(n: int, depth: int, seed_mask: np.ndarray, bg: int, seed_color: int, bar_color: int) -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
    bar_h, bar_w = (1, depth) if orientation == "horizontal" else (depth, 1)
    min_h = max(7, n + 4, bar_h + 4)
    min_w = max(7, n + 4, bar_w + 4)
    height = random.randint(min_h, min(16, min_h + 7))
    width = random.randint(min_w, min(16, min_w + 7))

    placements: list[tuple[int, int, int, int]] = []
    for sr in range(1, height - n):
        for sc in range(1, width - n):
            seed_box = (sr, sc, sr + n - 1, sc + n - 1)
            for br in range(1, height - bar_h):
                for bc in range(1, width - bar_w):
                    bar_box = (br, bc, br + bar_h - 1, bc + bar_w - 1)
                    if _rectangles_have_blank_gap(seed_box, bar_box):
                        placements.append((sr, sc, br, bc))
    if not placements:
        return None

    sr, sc, br, bc = random.choice(placements)
    grid = np.full((height, width), bg, dtype=int)
    seed_view = grid[sr : sr + n, sc : sc + n]
    seed_view[seed_mask] = seed_color
    grid[br : br + bar_h, bc : bc + bar_w] = bar_color
    return grid


def _validate_generated(grid: np.ndarray, expected_mask: np.ndarray, n: int, depth: int, bg: int, color: int) -> np.ndarray | None:
    """Check the stated semantics, unique roles, and common shortcut counterexamples."""
    try:
        parsed_bg, role = _extract_roles(grid)
        if parsed_bg != bg or int(role["n"]) != n or int(role["depth"]) != depth:
            return None
        if int(role["seed_color"]) != color or not np.array_equal(role["seed_mask"], expected_mask):
            return None

        expected = _oracle_expand(expected_mask, depth, bg, color)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return None
        if max(grid.shape) > _MAX_SIDE or max(actual.shape) > _MAX_SIDE:
            return None
        if np.all(actual == bg) or np.all(actual == color):
            return None

        # Reject the tempting but wrong shortcut "scale each seed pixel into one
        # solid block". Recursive substitution must visibly change internal blocks.
        scale = n ** depth
        simple_scale = np.repeat(np.repeat(expected_mask, scale, axis=0), scale, axis=1)
        if np.array_equal(actual == color, simple_scale):
            return None

        # A neighboring legal bar length must select a different-sized result,
        # demonstrating that the counter is a necessary parameter.
        other_depth = depth - 1 if depth > 1 else depth + 1
        if n ** (other_depth + 1) <= _MAX_SIDE:
            other = _oracle_expand(expected_mask, other_depth, bg, color)
            if other.shape == actual.shape or np.array_equal(other, actual):
                return None

        # Exercise main() on a counterfactual shorter bar, rather than merely
        # checking the oracle against itself. This verifies that visible bar
        # length really is consumed by the implementation.
        if depth > 1:
            shorter_grid = grid.copy()
            bar_coords = role["bar_coords"]
            end_r, end_c = map(int, bar_coords[-1])
            shorter_grid[end_r, end_c] = bg
            shorter_expected = _oracle_expand(expected_mask, depth - 1, bg, color)
            if not np.array_equal(main(shorter_grid), shorter_expected):
                return None

        # Also hold area and color constant while changing the seed geometry.
        # This excludes an area-only shortcut and checks the seed's local frame.
        alternatives = [
            mask
            for mask in _MASKS[n]
            if int(mask.sum()) == int(expected_mask.sum()) and not np.array_equal(mask, expected_mask)
        ]
        random.shuffle(alternatives)
        geometry_checked = False
        for alternate_mask in alternatives:
            alternate_grid = grid.copy()
            r0, c0 = int(role["r0"]), int(role["c0"])
            alternate_grid[r0 : r0 + n, c0 : c0 + n] = bg
            alternate_grid[r0 : r0 + n, c0 : c0 + n][alternate_mask] = color
            alternate_expected = _oracle_expand(alternate_mask, depth, bg, color)
            try:
                alternate_actual = main(alternate_grid)
            except AssertionError:
                continue
            if np.array_equal(alternate_actual, alternate_expected) and not np.array_equal(alternate_actual, actual):
                geometry_checked = True
                break
        if not geometry_checked:
            return None
    except (AssertionError, ValueError, IndexError):
        return None
    return actual


def _gen_attempt(n: int, depth: int, same_color: bool) -> np.ndarray | None:
    bg = random.choice(_ARC_COLORS)
    seed_color = random.choice([c for c in _ARC_COLORS if c != bg])
    if same_color:
        bar_color = seed_color
    else:
        bar_color = random.choice([c for c in _ARC_COLORS if c not in (bg, seed_color)])

    seed_mask = random.choice(_MASKS[n]).copy()
    # Rotations/reflections make local geometry diverse without changing its
    # connectivity or tight n x n reference frame.
    seed_mask = np.rot90(seed_mask, random.randrange(4))
    if random.random() < 0.5:
        seed_mask = np.fliplr(seed_mask)

    grid = _place_objects(n, depth, seed_mask, bg, seed_color, bar_color)
    if grid is None:
        return None
    output = _validate_generated(grid, seed_mask, n, depth, bg, seed_color)
    if output is None:
        return None

    output_key = (tuple(map(int, output.shape)), output.tobytes())
    if output_key in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(output_key)
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked examples with scheduled depth counterexamples."""
    global _GENERATION_INDEX

    # The first three generated examples deliberately expose h=1, h=2, and h=3;
    # they also cover both legal seed sizes and a same-color seed/bar case.
    schedule = (
        (2, 1, False),
        (3, 2, False),
        (2, 3, True),
        (3, 1, True),
        (2, 2, False),
        (3, 2, True),
    )
    preferred = schedule[_GENERATION_INDEX % len(schedule)]

    for attempt in range(1200):
        if attempt < 160:
            n, depth, same_color = preferred
        else:
            n = random.choice((2, 3))
            depth = random.choice((1, 2, 3) if n == 2 else (1, 2))
            same_color = random.random() < 0.4
        grid = _gen_attempt(n, depth, same_color)
        if grid is not None:
            _GENERATION_INDEX += 1
            return grid
    raise RuntimeError("failed to generate a unique valid recursive-depth input")
