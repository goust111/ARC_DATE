import random
from collections import Counter

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _mode(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    return int(Counter(map(int, flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    candidates_by_color: dict[int, list[dict]] = {}
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        comps = _components(g == color)
        frames: list[dict] = []
        for comp in comps:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            expected = 2 * hh + 2 * ww - 4
            if len(comp) != expected:
                continue
            comp_set = set(comp)
            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            if comp_set != border:
                continue
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            markers = np.argwhere((interior != bg) & (interior != color))
            if markers.shape[0] != 1:
                continue
            if np.count_nonzero(interior != bg) != 1:
                continue
            mr, mc = map(int, markers[0])
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "marker_color": int(interior[mr, mc]),
                    "marker_pos": (r0 + 1 + mr, c0 + 1 + mc),
                    "frame_color": int(color),
                }
            )
        if frames:
            candidates_by_color[int(color)] = frames
    if not candidates_by_color:
        raise ValueError("no valid rectangular frames found")
    best_color = max(
        candidates_by_color,
        key=lambda c: (len(candidates_by_color[c]), sum((f["bbox"][2] - f["bbox"][0] + 1) * (f["bbox"][3] - f["bbox"][1] + 1) for f in candidates_by_color[c]), -c),
    )
    return candidates_by_color[best_color]


def _parse_bottom_strip(strip: np.ndarray, bg: int, marker_colors: set[int]) -> dict[int, np.ndarray] | None:
    if strip.size == 0 or not np.any(strip != bg):
        return None
    occupied_cols = np.any(strip != bg, axis=0)
    segments = _intervals(occupied_cols)
    if not segments:
        return None
    mapping: dict[int, np.ndarray] = {}
    for c0, c1 in segments:
        seg = strip[:, c0 : c1 + 1]
        if seg.shape[1] < 2:
            return None
        first_col_rows = np.flatnonzero(seg[:, 0] != bg)
        if len(first_col_rows) != 1:
            return None
        idx_r = int(first_col_rows[0])
        idx_color = int(seg[idx_r, 0])
        if idx_color not in marker_colors or idx_color in mapping:
            return None
        pattern_zone = seg[:, 1:]
        nz = np.argwhere(pattern_zone != bg)
        if nz.size == 0:
            return None
        pr0, pc0 = nz.min(axis=0)
        pr1, pc1 = nz.max(axis=0)
        pattern = pattern_zone[int(pr0) : int(pr1) + 1, int(pc0) : int(pc1) + 1].copy()
        if not (int(pr0) <= idx_r <= int(pr1)):
            return None
        colors = set(map(int, pattern.ravel())) - {bg}
        if len(colors) != 1:
            return None
        if not np.all(np.any(pattern != bg, axis=0)) or not np.all(np.any(pattern != bg, axis=1)):
            return None
        mapping[idx_color] = pattern
    return mapping if set(mapping) == marker_colors else None


def _parse_right_strip(strip: np.ndarray, bg: int, marker_colors: set[int]) -> dict[int, np.ndarray] | None:
    if strip.size == 0 or not np.any(strip != bg):
        return None
    occupied_rows = np.any(strip != bg, axis=1)
    segments = _intervals(occupied_rows)
    if not segments:
        return None
    mapping: dict[int, np.ndarray] = {}
    for r0, r1 in segments:
        seg = strip[r0 : r1 + 1, :]
        if seg.shape[0] < 2:
            return None
        first_row_cols = np.flatnonzero(seg[0, :] != bg)
        if len(first_row_cols) != 1:
            return None
        idx_c = int(first_row_cols[0])
        idx_color = int(seg[0, idx_c])
        if idx_color not in marker_colors or idx_color in mapping:
            return None
        pattern_zone = seg[1:, :]
        nz = np.argwhere(pattern_zone != bg)
        if nz.size == 0:
            return None
        pr0, pc0 = nz.min(axis=0)
        pr1, pc1 = nz.max(axis=0)
        pattern = pattern_zone[int(pr0) : int(pr1) + 1, int(pc0) : int(pc1) + 1].copy()
        if not (int(pc0) <= idx_c <= int(pc1)):
            return None
        colors = set(map(int, pattern.ravel())) - {bg}
        if len(colors) != 1:
            return None
        if not np.all(np.any(pattern != bg, axis=0)) or not np.all(np.any(pattern != bg, axis=1)):
            return None
        mapping[idx_color] = pattern
    return mapping if set(mapping) == marker_colors else None


def _decode_legend(grid: np.ndarray, bg: int, frames: list[dict]) -> tuple[str, int, dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker_colors = {int(frame["marker_color"]) for frame in frames}
    max_r = max(int(frame["bbox"][2]) for frame in frames)
    max_c = max(int(frame["bbox"][3]) for frame in frames)

    candidates: list[tuple[int, str, int, dict[int, np.ndarray]]] = []

    for cut in range(max_r + 1, h):
        mapping = _parse_bottom_strip(g[cut:, :], bg, marker_colors)
        if mapping is not None:
            candidates.append(((h - cut) * w, "bottom", cut, mapping))

    for cut in range(max_c + 1, w):
        mapping = _parse_right_strip(g[:, cut:], bg, marker_colors)
        if mapping is not None:
            candidates.append((h * (w - cut), "right", cut, mapping))

    if not candidates:
        raise ValueError("legend strip not found")

    candidates.sort(key=lambda item: (item[0], 0 if item[1] == "bottom" else 1, item[2]))
    _, orientation, cut, mapping = candidates[0]
    return orientation, cut, mapping


def _tile(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % pattern.shape[0]
    cc = np.arange(width)[None, :] % pattern.shape[1]
    return pattern[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    frames = _extract_frames(g, bg)
    orientation, cut, mapping = _decode_legend(g, bg, frames)

    out = g[:cut, :].copy() if orientation == "bottom" else g[:, :cut].copy()

    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        pattern = mapping[int(frame["marker_color"])]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile(pattern, inner_h, inner_w)

    return out


PATTERN_MASKS = [
    np.array([[1, 0], [0, 1]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 0], [0, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=int),
]


def _mask_to_pattern(mask: np.ndarray, fill_color: int) -> np.ndarray:
    return np.where(mask.astype(bool), fill_color, BG_COLOR).astype(int)


def _sample_patterns(rng: random.Random, count: int, fill_colors: list[int]) -> list[np.ndarray]:
    order = list(range(len(PATTERN_MASKS)))
    rng.shuffle(order)
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()
    for idx, fill_color in zip(order, fill_colors):
        pattern = _mask_to_pattern(PATTERN_MASKS[idx], fill_color)
        key = pattern.tobytes()
        if key in seen:
            continue
        seen.add(key)
        patterns.append(pattern)
        if len(patterns) == count:
            return patterns
    raise RuntimeError("not enough unique patterns")


def _rect_ok(boxes: list[tuple[int, int, int, int]], r0: int, c0: int, r1: int, c1: int) -> bool:
    for br0, bc0, br1, bc1 in boxes:
        if not (r1 + 1 < br0 or br1 + 1 < r0 or c1 + 1 < bc0 or bc1 + 1 < c0):
            return False
    return True


def _place_frames(
    rng: random.Random,
    main_h: int,
    main_w: int,
    count: int,
    marker_colors: list[int],
    patterns: list[np.ndarray],
    orientation: str,
) -> tuple[np.ndarray, list[dict], np.ndarray] | None:
    canvas = np.full((main_h, main_w), BG_COLOR, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    frames: list[dict] = []

    order = list(range(count))
    if orientation == "bottom":
        order = [0] + [i for i in order if i != 0]
    else:
        order = [0] + [i for i in order if i != 0]

    for idx in order:
        outer_h = rng.randint(4, min(8, main_h))
        outer_w = rng.randint(4, min(8, main_w))
        placed = False
        attempts = 0
        while attempts < 200 and not placed:
            attempts += 1
            if idx == 0 and orientation == "bottom":
                r0 = main_h - outer_h
                c0 = rng.randint(0, main_w - outer_w)
            elif idx == 0 and orientation == "right":
                r0 = rng.randint(0, main_h - outer_h)
                c0 = main_w - outer_w
            else:
                r0 = rng.randint(0, main_h - outer_h)
                c0 = rng.randint(0, main_w - outer_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _rect_ok(boxes, r0, c0, r1, c1):
                continue
            boxes.append((r0, c0, r1, c1))
            canvas[r0 : r1 + 1, c0] = FRAME_COLOR
            canvas[r0 : r1 + 1, c1] = FRAME_COLOR
            canvas[r0, c0 : c1 + 1] = FRAME_COLOR
            canvas[r1, c0 : c1 + 1] = FRAME_COLOR
            inner_h = outer_h - 2
            inner_w = outer_w - 2
            mr = rng.randint(r0 + 1, r1 - 1)
            mc = rng.randint(c0 + 1, c1 - 1)
            canvas[mr, mc] = marker_colors[idx]
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "marker_color": marker_colors[idx],
                    "pattern": patterns[idx],
                }
            )
            placed = True
        if not placed:
            return None

    expected = canvas.copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        pattern = frame["pattern"]
        expected[r0 + 1 : r1, c0 + 1 : c1] = _tile(pattern, r1 - r0 - 1, c1 - c0 - 1)

    return canvas, frames, expected


def _make_bottom_legend(main_w: int, marker_colors: list[int], patterns: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    widths = [1 + p.shape[1] for p in patterns]
    needed = sum(widths) + max(0, len(patterns) - 1)
    if needed > main_w:
        return None
    strip_h = max(p.shape[0] for p in patterns)
    strip = np.full((strip_h, main_w), BG_COLOR, dtype=int)
    start_c = rng.randint(0, main_w - needed)
    c = start_c
    for marker_color, pattern in zip(marker_colors, patterns):
        ph, pw = pattern.shape
        strip[0, c] = marker_color
        strip[0:ph, c + 1 : c + 1 + pw] = pattern
        c += 1 + pw + 1
    return strip


def _make_right_legend(main_h: int, marker_colors: list[int], patterns: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    heights = [1 + p.shape[0] for p in patterns]
    needed = sum(heights) + max(0, len(patterns) - 1)
    if needed > main_h:
        return None
    strip_w = max(p.shape[1] for p in patterns)
    strip = np.full((main_h, strip_w), BG_COLOR, dtype=int)
    start_r = rng.randint(0, main_h - needed)
    r = start_r
    for marker_color, pattern in zip(marker_colors, patterns):
        ph, pw = pattern.shape
        strip[r, 0] = marker_color
        strip[r + 1 : r + 1 + ph, 0:pw] = pattern
        r += 1 + ph + 1
    return strip


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["bottom", "right"])
    count = rng.randint(2, 4)
    colors = PALETTE_COLORS[:]
    rng.shuffle(colors)
    marker_colors = colors[:count]
    fill_colors = colors[count : 2 * count]
    if len(fill_colors) < count:
        return None
    patterns = _sample_patterns(rng, count, fill_colors)

    if orientation == "bottom":
        widths = [1 + p.shape[1] for p in patterns]
        required_w = sum(widths) + max(0, count - 1)
        main_w = rng.randint(max(12, required_w), 22)
        main_h = rng.randint(12, 22)
    else:
        heights = [1 + p.shape[0] for p in patterns]
        required_h = sum(heights) + max(0, count - 1)
        main_h = rng.randint(max(12, required_h), 22)
        main_w = rng.randint(12, 22)

    placed = _place_frames(rng, main_h, main_w, count, marker_colors, patterns, orientation)
    if placed is None:
        return None
    canvas, frames, expected = placed

    if orientation == "bottom":
        legend = _make_bottom_legend(main_w, marker_colors, patterns, rng)
        if legend is None:
            return None
        grid = np.vstack([canvas, legend])
    else:
        legend = _make_right_legend(main_h, marker_colors, patterns, rng)
        if legend is None:
            return None
        grid = np.hstack([canvas, legend])

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
