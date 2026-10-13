import random
from collections import Counter

import numpy as np


BACKGROUND = 0
MARKER = 7
PALETTE = (1, 2, 3, 4, 6, 8, 9)
RING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    """Minimal monochromatic 4-connected components helper, mirroring arc_utils."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _marker_side(grid: np.ndarray) -> str:
    sides = []
    if np.all(grid[0, :] == MARKER):
        sides.append("top")
    if np.all(grid[-1, :] == MARKER):
        sides.append("bottom")
    if np.all(grid[:, 0] == MARKER):
        sides.append("left")
    if np.all(grid[:, -1] == MARKER):
        sides.append("right")
    assert len(sides) == 1, "expected exactly one full-edge marker line"
    return sides[0]


def _is_ring(grid: np.ndarray, comp: dict) -> bool:
    if int(comp["color"]) in (BACKGROUND, MARKER):
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 != 2 or c1 - c0 != 2 or int(comp["area"]) != 8:
        return False
    if int(grid[r0 + 1, c0 + 1]) != BACKGROUND:
        return False
    expected = {
        (r0 + dr, c0 + dc)
        for dr in range(3)
        for dc in range(3)
        if (dr, dc) != (1, 1)
    }
    actual = {tuple(map(int, coord)) for coord in comp["coords"]}
    return actual == expected


def _extract_rings(grid: np.ndarray) -> list[dict]:
    side = _marker_side(grid)
    comps = find_components(grid, bg=BACKGROUND)
    rings = [comp for comp in comps if _is_ring(grid, comp)]
    assert 8 <= len(rings) <= 12, "expected 8..12 hollow 3x3 rings"
    for comp in comps:
        color = int(comp["color"])
        if color == MARKER:
            continue
        assert _is_ring(grid, comp), "all non-marker foreground objects must be hollow 3x3 rings"
    if side == "top":
        assert all(int(ring["bbox"][0]) >= 1 for ring in rings)
    elif side == "bottom":
        assert all(int(ring["bbox"][2]) <= grid.shape[0] - 2 for ring in rings)
    elif side == "left":
        assert all(int(ring["bbox"][1]) >= 1 for ring in rings)
    else:
        assert all(int(ring["bbox"][3]) <= grid.shape[1] - 2 for ring in rings)
    return rings


def _compress(grid: np.ndarray) -> tuple[np.ndarray, list[dict], str]:
    side = _marker_side(grid)
    rings = _extract_rings(grid)
    compact = np.zeros_like(grid)
    for ring in rings:
        r0, c0, _, _ = ring["bbox"]
        compact[r0 + 1, c0 + 1] = int(ring["color"])
    return compact, rings, side


def _sort_by_frequency(colors: list[int], freqs: Counter) -> list[int]:
    return sorted(colors, key=lambda color: -int(freqs[int(color)]))


def _sort_by_color(colors: list[int], reverse: bool) -> list[int]:
    return sorted(colors, reverse=reverse)


def _pack(compact: np.ndarray, side: str, *, mode: str = "frequency_desc") -> np.ndarray:
    h, w = compact.shape
    out = np.zeros_like(compact)
    freqs = Counter(int(v) for v in compact.ravel() if int(v) != BACKGROUND)

    def order(colors: list[int]) -> list[int]:
        if mode == "frequency_desc":
            return _sort_by_frequency(colors, freqs)
        if mode == "stable":
            return list(colors)
        if mode == "frequency_asc":
            return sorted(colors, key=lambda color: int(freqs[int(color)]))
        if mode == "color_desc":
            return _sort_by_color(colors, True)
        if mode == "color_asc":
            return _sort_by_color(colors, False)
        raise ValueError(mode)

    if side in ("top", "bottom"):
        for c in range(w):
            rows = [r for r in range(h) if int(compact[r, c]) != BACKGROUND]
            colors = [int(compact[r, c]) for r in rows]
            colors = order(colors)
            if side == "top":
                targets = range(len(colors))
            else:
                targets = range(h - len(colors), h)
            for tr, color in zip(targets, colors):
                out[int(tr), c] = color
    else:
        for r in range(h):
            cols = [c for c in range(w) if int(compact[r, c]) != BACKGROUND]
            colors = [int(compact[r, c]) for c in cols]
            colors = order(colors)
            if side == "left":
                targets = range(len(colors))
            else:
                targets = range(w - len(colors), w)
            for tc, color in zip(targets, colors):
                out[r, int(tc)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress hollow 3x3 symbols into single pixels, then gravity-pack each stack with global color-frequency ordering.

    Input structure:
    - The grid is black background with one complete orange edge line marking a gravity direction.
    - The remaining foreground objects are 8..12 monochrome hollow 3x3 rings whose centers are black.
    - Rings use 3 or 4 visible colors and sit on a coarse lattice without overlap.

    Rule:
    - Remove the orange marker line and replace every hollow 3x3 ring by one pixel at its center, keeping its color.
    - Count how many compressed pixels each color contributes in the whole grid.
    - Use the marked edge as gravity: top -> up, bottom -> down, left -> left, right -> right.
    - In every affected column or row, slide the compressed pixels tightly toward that edge and order them by descending global color frequency, so the most frequent color is closest to the gravity edge; equal-frequency ties keep their original order.

    Key insight:
    - The ring geometry is only a carrier: once each ring collapses to its center, the final arrangement is determined jointly by the marker-selected gravity axis and a global frequency ranking over colors.
    """
    g = np.asarray(grid, dtype=int)
    compact, _rings, side = _compress(g)
    return _pack(compact, side, mode="frequency_desc")


def _distinct_count_pattern(total: int, num_colors: int) -> list[int]:
    patterns = {
        3: {
            8: ([5, 2, 1], [4, 3, 1]),
            9: ([4, 3, 2], [6, 2, 1]),
            10: ([4, 3, 2], [5, 3, 2]),
            11: ([5, 4, 2], [6, 3, 2]),
            12: ([5, 4, 3], [7, 3, 2]),
        },
        4: {
            10: ([4, 3, 2, 1],),
            11: ([5, 3, 2, 1], [4, 3, 2, 2]),
            12: ([6, 3, 2, 1], [4, 3, 3, 2]),
        },
    }
    choices = [tuple(p) for p in patterns.get(num_colors, {}).get(total, ())]
    choices = [list(p) for p in choices if len(set(p)) >= 3 and len(p) == num_colors and sum(p) == total and min(p) > 0]
    assert choices
    valid = [p for p in choices if len(set(p)) == num_colors]
    if valid:
        choices = valid
    return list(random.choice(choices))


def _candidate_centers(length: int) -> list[int]:
    return list(range(2, length - 1, 4))


def _focus_pair(colors: list[int], counts: dict[int, int]) -> tuple[int, int]:
    pairs = []
    for hi in colors:
        for lo in colors:
            if hi == lo:
                continue
            if counts[hi] > counts[lo]:
                pairs.append((hi, lo))
    random.shuffle(pairs)
    for hi, lo in pairs:
        if hi < lo:
            return hi, lo
    assert pairs
    return pairs[0]


def _place_ring(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = center
    for dr, dc in RING_OFFSETS:
        grid[r + dr, c + dc] = color


def _build_attempt() -> np.ndarray | None:
    h = random.randint(15, 23)
    w = random.randint(15, 23)
    side = random.choice(SIDES)
    rows = _candidate_centers(h)
    cols = _candidate_centers(w)
    if len(rows) < 3 or len(cols) < 3:
        return None

    total = random.randint(8, 12)
    num_colors = random.choice([3, 4]) if total >= 10 else 3
    counts_list = _distinct_count_pattern(total, num_colors)
    colors = random.sample(PALETTE, num_colors)
    random.shuffle(counts_list)
    counts = {color: count for color, count in zip(colors, counts_list)}

    hi_color, lo_color = _focus_pair(colors, counts)
    positions = [(r, c) for r in rows for c in cols]
    used: set[tuple[int, int]] = set()
    assignments: list[tuple[tuple[int, int], int]] = []

    if side in ("top", "bottom"):
        focus_col = random.choice(cols)
        focus_rows = random.sample(rows, 2)
        focus_rows.sort()
        if side == "top":
            ordered = [(focus_rows[0], focus_col, lo_color), (focus_rows[1], focus_col, hi_color)]
        else:
            ordered = [(focus_rows[0], focus_col, hi_color), (focus_rows[1], focus_col, lo_color)]
    else:
        focus_row = random.choice(rows)
        focus_cols = random.sample(cols, 2)
        focus_cols.sort()
        if side == "left":
            ordered = [(focus_row, focus_cols[0], lo_color), (focus_row, focus_cols[1], hi_color)]
        else:
            ordered = [(focus_row, focus_cols[0], hi_color), (focus_row, focus_cols[1], lo_color)]

    for r, c, color in ordered:
        assignments.append(((r, c), color))
        used.add((r, c))
        counts[color] -= 1
        if counts[color] < 0:
            return None

    remaining_positions = [pos for pos in positions if pos not in used]
    random.shuffle(remaining_positions)
    needed = sum(counts.values())
    if len(remaining_positions) < needed:
        return None

    pool = []
    for color, count in counts.items():
        pool.extend([color] * count)
    random.shuffle(pool)

    for pos, color in zip(remaining_positions[:needed], pool):
        assignments.append((pos, color))

    grid = np.zeros((h, w), dtype=int)
    if side == "top":
        grid[0, :] = MARKER
    elif side == "bottom":
        grid[-1, :] = MARKER
    elif side == "left":
        grid[:, 0] = MARKER
    else:
        grid[:, -1] = MARKER

    for center, color in assignments:
        r, c = center
        if r - 1 < 0 or r + 1 >= h or c - 1 < 0 or c + 1 >= w:
            return None
        patch = grid[r - 1 : r + 2, c - 1 : c + 2]
        if np.any(patch != BACKGROUND):
            return None
        _place_ring(grid, center, color)

    return grid


def _has_required_color_frequencies(compact: np.ndarray) -> bool:
    freqs = Counter(int(v) for v in compact.ravel() if int(v) != BACKGROUND)
    counts = sorted(freqs.values(), reverse=True)
    return len(freqs) >= 3 and len(set(counts[:3])) == 3


def _moves_under_gravity(compact: np.ndarray, side: str, out: np.ndarray) -> bool:
    if side == "top":
        return any(r > 0 and int(compact[r, c]) != 0 and int(out[r, c]) == 0 for r in range(compact.shape[0]) for c in range(compact.shape[1]))
    if side == "bottom":
        return any(r < compact.shape[0] - 1 and int(compact[r, c]) != 0 and int(out[r, c]) == 0 for r in range(compact.shape[0]) for c in range(compact.shape[1]))
    if side == "left":
        return any(c > 0 and int(compact[r, c]) != 0 and int(out[r, c]) == 0 for r in range(compact.shape[0]) for c in range(compact.shape[1]))
    return any(c < compact.shape[1] - 1 and int(compact[r, c]) != 0 and int(out[r, c]) == 0 for r in range(compact.shape[0]) for c in range(compact.shape[1]))


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        compact, rings, side = _compress(grid)
    except AssertionError:
        return False

    if len(rings) != len({tuple(map(int, ring["bbox"])) for ring in rings}):
        return False
    if not _has_required_color_frequencies(compact):
        return False

    out = _pack(compact, side, mode="frequency_desc")
    gravity_only = _pack(compact, side, mode="stable")
    freq_asc = _pack(compact, side, mode="frequency_asc")
    color_desc = _pack(compact, side, mode="color_desc")
    color_asc = _pack(compact, side, mode="color_asc")

    if np.array_equal(out, compact):
        return False
    if np.array_equal(out, gravity_only):
        return False
    if np.array_equal(out, freq_asc):
        return False
    if np.array_equal(out, color_desc):
        return False
    if np.array_equal(out, color_asc):
        return False
    if not _moves_under_gravity(compact, side, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None and _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
