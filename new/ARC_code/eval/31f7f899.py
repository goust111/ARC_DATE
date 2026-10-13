from __future__ import annotations
import numpy as np
from arc_utils import bbox_h, find_components, get_bg, randomly_spaced_indices
from arc_utils import paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height and redistribute them across fixed colored slots.

    Input structure:
    - bg background; a horizontal base line of a single color spans the full grid width.
    - Several monochrome vertical bars of distinct colors rise upward from the base line.

    Rule:
    - Rank bar shapes by height (shortest to tallest).
    - Rank original x-positions (slots) left to right.
    - Place the i-th shortest shape into the i-th leftmost slot, recoloring it to that slot's original color.
    - The base line is preserved unchanged.

    Key insight: Each slot keeps its x-position and color identity; only the bar silhouettes are redistributed among slots by height order.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # Base row/color = (row,color) with maximal count on that row (excluding bg).
    best = None  # (count, row, color)
    for r in range(H):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert not (best is None)
    _cnt, base_r, base_col = best

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        b = comp["bbox"]
        height = int(bbox_h(b))

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    # Slots are determined by original x-positions; slot colors must be preserved.
    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    # Shapes are ordered by height (shortest -> tallest).
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out

# --------------- generator (Codex#3) ---------------

def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_scene(n: int, base_r: int, base_color: int, bg: int, lefts: list[int], width: int, colors: list[int], heights: list[int]) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    height_choices = list(range(1, min(n, 11) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, sorted_heights)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, in_heights)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
