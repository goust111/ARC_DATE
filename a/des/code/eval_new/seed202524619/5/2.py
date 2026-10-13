import random

import numpy as np


# Fixed colors used by the generator (and understood directly by main's fallback map).
BG_COLOR = 0
RAIL_COLOR = 9
FRAG_RED = 2
FRAG_BLUE = 1
FRAG_YELLOW = 4
OUT_MAGENTA = 6
OUT_ORANGE = 7
OUT_GREEN = 3

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def _rail_neighbors(g: np.ndarray, r: int, c: int, rail: int) -> list[tuple[int, int]]:
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for dr, dc in _DIRECTIONS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(rail):
            out.append((int(rr), int(cc)))
    return out


def _pick_rail_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g, return_counts=True)
    candidates: list[tuple[float, int, int]] = []
    for val, cnt in zip(vals, counts):
        color = int(val)
        if color == bg:
            continue
        mask = g == color
        if not bool(mask.any()):
            continue

        # Connectivity score: number of orthogonal edges inside this color.
        # Tracks usually create many connected neighbors, fragments do not.
        score = 0
        score += np.logical_and(mask[:-1, :], mask[1:, :]).sum()
        score += np.logical_and(mask[:, :-1], mask[:, 1:]).sum()
        candidates.append((int(score), int(cnt), color))

    assert candidates, "no non-background color to infer rail"
    # Prefer the most connected color; ties fall back to count.
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _walk_to_junction(g: np.ndarray, start: tuple[int, int], prev: tuple[int, int], rail: int) -> tuple[int, int]:
    """Follow a straight rail from fragment side until reaching the junction."""
    h, w = g.shape
    pr, pc = prev
    cr, cc = start
    dr, dc = cr - pr, cc - pc

    for _ in range(h * w):
        neighbors: list[tuple[int, int, int, int]] = []
        for ndr, ndc in _DIRECTIONS:
            rr, cc2 = cr + ndr, cc + ndc
            if 0 <= rr < h and 0 <= cc2 < w and int(g[rr, cc2]) == int(rail):
                if (rr, cc2) == (pr, pc):
                    continue
                neighbors.append((rr, cc2, ndr, ndc))

        if not neighbors:
            return (int(cr), int(cc))

        forward_exists = False
        non_forward = 0
        for _, _, ndr, ndc in neighbors:
            if (ndr, ndc) == (dr, dc):
                forward_exists = True
            else:
                non_forward += 1

        # Junction reached if there is any perpendicular/branch continuation.
        if non_forward > 0 or not forward_exists:
            return (int(cr), int(cc))

        # Continue along the straight segment.
        cr, cc = cr + dr, cc + dc

    raise AssertionError("rail walk did not terminate within grid bounds")


def _mix_two_colors(c1: int, c2: int, palette: set[int]) -> int:
    if c1 == c2:
        return int(c1)

    role_mix = {
        frozenset({FRAG_RED, FRAG_BLUE}): OUT_MAGENTA,
        frozenset({FRAG_RED, FRAG_YELLOW}): OUT_ORANGE,
        frozenset({FRAG_BLUE, FRAG_YELLOW}): OUT_GREEN,
    }
    if palette.issuperset({FRAG_RED, FRAG_BLUE, FRAG_YELLOW}):
        return int(role_mix[frozenset({c1, c2})])

    ordered = sorted(palette)
    if len(ordered) >= 3:
        fallback = {
            frozenset({ordered[0], ordered[1]}): OUT_MAGENTA,
            frozenset({ordered[0], ordered[2]}): OUT_ORANGE,
            frozenset({ordered[1], ordered[2]}): OUT_GREEN,
        }
        return int(fallback.get(frozenset({c1, c2}), c1))

    return int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each color fragment along its rail to the junction, then mix the arriving colors.

    Input structure:
    - bg is the most common color.
    - Grey pixels form one-pixel-wide horizontal or vertical rails.
    - Every rail segment has exactly one adjacent colored fragment pixel at its free end.
    - Each rail ends at a junction cell where exactly two segments meet.

    Rule:
    - For each fragment, move along its rail in the segment direction until the first junction point.
    - Clear the fragment's original position to bg.
    - At each junction, combine the two incoming colors:
      - same color + same color stays unchanged
      - red + blue -> magenta
      - red + yellow -> orange
      - blue + yellow -> green
    - Keep all non-junction rail cells unchanged.

    Key insight:
    - The direction is fixed by the unique adjacent rail cell, and the junction is where the segment no longer
      continues straight (branch or turn).
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = _most_common_color(g)
    rail = _pick_rail_color(g, bg)

    fragments: list[tuple[int, int, int]] = []  # (r, c, color)
    for r, c in np.argwhere((g != bg) & (g != rail)):
        r, c = int(r), int(c)
        if len(_rail_neighbors(g, r, c, rail)) == 1:
            fragments.append((r, c, int(g[r, c])))

    assert fragments, "expected at least one valid fragment"
    fragment_colors = {c for _, _, c in fragments}

    dest_colors: dict[tuple[int, int], list[int]] = {}
    for r, c, col in fragments:
        (nr, nc) = _rail_neighbors(g, r, c, rail)[0]
        dst_r, dst_c = _walk_to_junction(g, (nr, nc), (r, c), rail)
        key = (dst_r, dst_c)
        dest_colors.setdefault(key, []).append(int(col))
        out[r, c] = bg

    for (dst_r, dst_c), colors in dest_colors.items():
        if not colors:
            continue
        cur = int(colors[0])
        for nxt in colors[1:]:
            cur = _mix_two_colors(cur, int(nxt), fragment_colors)
        out[int(dst_r), int(dst_c)] = int(cur)

    return out


def _rand_colors_pair() -> tuple[int, int]:
    # Equal chance for same-color and two-color combinations.
    combos = (
        (FRAG_RED, FRAG_RED),
        (FRAG_RED, FRAG_BLUE),
        (FRAG_BLUE, FRAG_RED),
        (FRAG_RED, FRAG_YELLOW),
        (FRAG_BLUE, FRAG_BLUE),
        (FRAG_BLUE, FRAG_YELLOW),
        (FRAG_YELLOW, FRAG_YELLOW),
    )
    return random.choice(combos)


def _is_clear_with_margin(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        for rr in (r - 1, r, r + 1):
            for cc in (c - 1, c, c + 1):
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG_COLOR:
                    return False
    return True


def _place_one_junction(grid: np.ndarray) -> bool:
    h, w = grid.shape
    h0_color, h1_color = _rand_colors_pair()

    for _ in range(200):
        len_h = random.randint(1, 5)
        len_v = random.randint(1, 5)
        h_dir = random.choice((-1, 1))
        v_dir = random.choice((-1, 1))

        if h_dir == 1:
            c_min, c_max = 0, w - len_h - 2
        else:
            c_min, c_max = len_h + 1, w - 1
        if c_min > c_max:
            continue

        if v_dir == 1:
            r_min, r_max = 0, h - len_v - 2
        else:
            r_min, r_max = len_v + 1, h - 1
        if r_min > r_max:
            continue

        c = random.randint(c_min, c_max)
        r = random.randint(r_min, r_max)

        h_cells = [(r, c + h_dir * k) for k in range(1, len_h + 1)]
        v_cells = [(r + v_dir * k, c) for k in range(1, len_v + 1)]
        h_frag = (r, c + h_dir * (len_h + 1))
        v_frag = (r + v_dir * (len_v + 1), c)

        all_cells = [(r, c)] + h_cells + v_cells + [h_frag, v_frag]
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in all_cells):
            continue

        if not _is_clear_with_margin(grid, all_cells):
            continue

        grid[r, c] = RAIL_COLOR
        for rr, cc in h_cells + v_cells:
            grid[rr, cc] = RAIL_COLOR
        grid[h_frag[0], h_frag[1]] = h0_color
        grid[v_frag[0], v_frag[1]] = h1_color
        return True

    return False


def generate_input() -> np.ndarray:
    """Randomly generate valid inputs for the rail-junction color mixing puzzle."""
    for _ in range(400):
        h = random.randint(12, 28)
        w = random.randint(12, 28)
        base = np.full((h, w), BG_COLOR, dtype=int)

        junction_count = random.randint(1, 4)
        ok = True
        for _ in range(junction_count):
            if not _place_one_junction(base):
                ok = False
                break

        if not ok:
            continue

        bg_count = int(np.count_nonzero(base == BG_COLOR))
        if bg_count <= (h * w) // 2:
            continue
        if not np.count_nonzero(base == FRAG_RED):
            continue

        return base

    raise RuntimeError("failed to generate a valid 202524619-5-2 input")
