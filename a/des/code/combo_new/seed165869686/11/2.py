from collections import Counter

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _track_endpoints(track_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = track_mask.shape
    endpoints: list[tuple[int, int, int, int]] = []
    for r in range(height):
        for c in range(width):
            if not bool(track_mask[r, c]):
                continue
            neighbors = []
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and bool(track_mask[rr, cc]):
                    neighbors.append((dr, dc))
            if len(neighbors) == 1:
                dr_in, dc_in = neighbors[0]
                endpoints.append((int(r), int(c), int(dr_in), int(dc_in)))
    return endpoints


def _extract_fragments(grid: np.ndarray, bg: int, grey: int) -> list[dict]:
    track_mask = grid == grey
    height, width = grid.shape
    fragments: list[dict] = []

    for r, c, dr_in, dc_in in _track_endpoints(track_mask):
        dr_out, dc_out = -dr_in, -dc_in
        sr, sc = r + dr_out, c + dc_out
        if not (0 <= sr < height and 0 <= sc < width):
            continue
        color = int(grid[sr, sc])
        if color in (bg, grey):
            continue

        length = 0
        rr, cc = sr, sc
        while 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) == color:
            length += 1
            rr += dr_out
            cc += dc_out

        targets: list[tuple[int, int]] = []
        for step in range(length):
            tr = r + dr_in * step
            tc = c + dc_in * step
            if not (0 <= tr < height and 0 <= tc < width):
                break
            if not bool(track_mask[tr, tc]):
                break
            targets.append((int(tr), int(tc)))
        if not targets:
            continue

        fragments.append(
            {
                "color": color,
                "length": int(length),
                "horizontal": int(dc_in != 0),
                "targets": targets,
            }
        )

    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored line fragments into the grey track network and recolor the covered track cells.

    Input structure:
    - The background is black, and one grey color forms a set of 1-pixel-wide horizontal and vertical track segments.
    - Colored fragments are straight 1-pixel-thick rectangles placed immediately outside some true track endpoints, aligned with the endpoint's track direction.
    - Horizontal and vertical tracks may cross to form grey intersections.

    Rule:
    - Each colored fragment slides straight inward from its endpoint-adjacent position by a distance equal to its own pixel count.
    - The fragment's original cells turn back to background, and the track cells it lands on take the fragment's color.
    - If multiple fragments claim the same intersection cell, the longer fragment wins; ties go to the horizontal fragment.
    - Any grey track cell not covered by a moved fragment remains grey.

    Key insight:
    - A fragment's area is also its travel distance, so solving the task requires detecting both the endpoint-attached fragment lengths and the overlap priority at crossings.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g)
    non_bg = g[g != bg]
    if non_bg.size == 0:
        return g.copy()
    grey = _most_common(non_bg)

    out = g.copy()
    out[(g != bg) & (g != grey)] = bg

    best_len = np.full(g.shape, -1, dtype=int)
    best_h = np.full(g.shape, -1, dtype=int)
    best_color = np.full(g.shape, bg, dtype=int)

    for frag in _extract_fragments(g, bg, grey):
        length = int(frag["length"])
        horizontal = int(frag["horizontal"])
        color = int(frag["color"])
        for r, c in frag["targets"]:
            if length > best_len[r, c] or (length == best_len[r, c] and horizontal > best_h[r, c]):
                best_len[r, c] = length
                best_h[r, c] = horizontal
                best_color[r, c] = color

    mask = best_len >= 0
    out[mask] = best_color[mask]
    return out


def _sample_spaced_positions(
    rng: np.random.Generator, count: int, low: int, high: int, gap: int
) -> list[int] | None:
    values = np.arange(low, high + 1, dtype=int)
    if values.size < count:
        return None
    for _ in range(200):
        picks = np.sort(rng.choice(values, size=count, replace=False))
        if np.all(np.diff(picks) >= gap):
            return [int(x) for x in picks]
    return None


def _length_pairs(mode: str, d_h: int, cap_h: int, d_v: int, cap_v: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if mode == "tie":
        lo = max(d_h, d_v)
        hi = min(cap_h, cap_v)
        for length in range(lo, hi + 1):
            pairs.append((int(length), int(length)))
        return pairs

    for h_len in range(d_h, cap_h + 1):
        for v_len in range(d_v, cap_v + 1):
            if mode == "horizontal_longer" and h_len > v_len:
                pairs.append((int(h_len), int(v_len)))
            if mode == "vertical_longer" and v_len > h_len:
                pairs.append((int(h_len), int(v_len)))
    return pairs


def _build_candidate(
    rows: list[int],
    cols: list[int],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    height: int,
    width: int,
    mode: str,
) -> list[tuple[int, int, str, str, int, int]]:
    cap_left = min(c0, c1 - c0 + 1)
    cap_right = min(width - 1 - c1, c1 - c0 + 1)
    cap_top = min(r0, r1 - r0 + 1)
    cap_bottom = min(height - 1 - r1, r1 - r0 + 1)
    candidates: list[tuple[int, int, str, str, int, int]] = []

    for hi, row in enumerate(rows):
        for vi, col in enumerate(cols):
            for h_side, cap_h in (("left", cap_left), ("right", cap_right)):
                d_h = col - c0 + 1 if h_side == "left" else c1 - col + 1
                if d_h > cap_h:
                    continue
                for v_side, cap_v in (("top", cap_top), ("bottom", cap_bottom)):
                    d_v = row - r0 + 1 if v_side == "top" else r1 - row + 1
                    if d_v > cap_v:
                        continue
                    for h_len, v_len in _length_pairs(mode, d_h, cap_h, d_v, cap_v):
                        candidates.append((int(hi), int(vi), h_side, v_side, int(h_len), int(v_len)))
    return candidates


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_h = int(rng.integers(1, 4))
    n_v = int(rng.integers(1, 4))

    min_net_h = max(5, 2 * n_h + 1)
    min_net_w = max(5, 2 * n_v + 1)
    net_h = int(rng.integers(min_net_h, 12))
    net_w = int(rng.integers(min_net_w, 12))

    top = int(rng.integers(2, 7))
    bottom = int(rng.integers(2, 7))
    left = int(rng.integers(2, 7))
    right = int(rng.integers(2, 7))

    height = top + net_h + bottom
    width = left + net_w + right
    if height > 30 or width > 30:
        return None

    r0, r1 = top, top + net_h - 1
    c0, c1 = left, left + net_w - 1

    rows = _sample_spaced_positions(rng, n_h, r0 + 1, r1 - 1, gap=2)
    cols = _sample_spaced_positions(rng, n_v, c0 + 1, c1 - 1, gap=2)
    if rows is None or cols is None:
        return None

    mode = str(rng.choice(["horizontal_longer", "vertical_longer", "tie"]))
    candidates = _build_candidate(rows, cols, r0, r1, c0, c1, height, width, mode)
    if not candidates:
        return None

    hi, vi, h_side, v_side, h_len, v_len = candidates[int(rng.integers(len(candidates)))]
    colors = rng.choice(COLOR_POOL, size=n_h + n_v, replace=False)

    grid = np.full((height, width), BG, dtype=int)
    for row in rows:
        grid[row, c0 : c1 + 1] = GREY
    for col in cols:
        grid[r0 : r1 + 1, col] = GREY

    horiz_meta = []
    for idx, row in enumerate(rows):
        color = int(colors[idx])
        if idx == hi:
            side = h_side
            length = h_len
        else:
            side = str(rng.choice(["left", "right"]))
            capacity = min(c0 if side == "left" else width - 1 - c1, net_w)
            if capacity < 1:
                return None
            length = int(rng.integers(1, capacity + 1))
        horiz_meta.append((row, side, length, color))

    vert_meta = []
    for idx, col in enumerate(cols):
        color = int(colors[n_h + idx])
        if idx == vi:
            side = v_side
            length = v_len
        else:
            side = str(rng.choice(["top", "bottom"]))
            capacity = min(r0 if side == "top" else height - 1 - r1, net_h)
            if capacity < 1:
                return None
            length = int(rng.integers(1, capacity + 1))
        vert_meta.append((col, side, length, color))

    for row, side, length, color in horiz_meta:
        if side == "left":
            start = c0 - length
            if start < 0:
                return None
            grid[row, start:c0] = color
        else:
            end = c1 + length
            if end >= width:
                return None
            grid[row, c1 + 1 : end + 1] = color

    for col, side, length, color in vert_meta:
        if side == "top":
            start = r0 - length
            if start < 0:
                return None
            grid[start:r0, col] = color
        else:
            end = r1 + length
            if end >= height:
                return None
            grid[r1 + 1 : end + 1, col] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
