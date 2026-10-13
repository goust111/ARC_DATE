from __future__ import annotations

from collections import Counter

import numpy as np


GREY = 5
COLORS = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components_mask(grid == color)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != (int(grid[r, c]) == color):
                return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _color_components(grid, color):
            if _is_rect_frame(grid, comp, color):
                frames.append(
                    {
                        "color": color,
                        "coords": comp,
                        "bbox": _bbox(comp),
                    }
                )
    frames.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))  # type: ignore[arg-type]
    return frames


def _inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _index_color_in_frame(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int, frame_color: int
) -> int:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = {
        int(v)
        for v in np.unique(interior)
        if int(v) not in (bg, frame_color)
    }
    if len(colors) != 1:
        raise AssertionError("each frame interior must contain exactly one index color")
    return next(iter(colors))


def _find_palette_bars(
    grid: np.ndarray, bg: int, frames: list[dict[str, object]]
) -> tuple[dict[int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    outside = np.ones((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    candidate = (grid != bg) & outside
    mapping: dict[int, int] = {}
    bar_pixels: list[tuple[int, int]] = []

    for comp in _components_mask(candidate):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 != r1:
            continue
        width = c1 - c0 + 1
        if width < 2 or width > 4 or len(comp) != width:
            continue

        row = r0
        cols = sorted(c for _, c in comp)
        colors = [int(grid[row, c]) for c in cols]
        if len(set(colors)) != 2:
            continue

        index_color = colors[0]
        fill_candidates = {color for color in colors[1:] if color != index_color}
        if len(fill_candidates) != 1:
            continue
        mapping[index_color] = next(iter(fill_candidates))
        bar_pixels.extend((row, c) for c in cols)

    if not mapping:
        raise AssertionError("palette bars not found")
    return mapping, bar_pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use interior fragment colors as keys into external palette bars, then repaint each
    framed region with alternating horizontal fill rows.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    if not frames:
        raise AssertionError("no rectangular frames found")

    mapping, bar_pixels = _find_palette_bars(g, bg, frames)
    out = g.copy()
    for r, c in bar_pixels:
        out[r, c] = bg

    for frame in frames:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        index_color = _index_color_in_frame(g, (r0, c0, r1, c1), bg, frame_color)
        if index_color not in mapping:
            raise AssertionError("missing palette mapping for frame index color")
        fill_color = int(mapping[index_color])
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        out[r0 + 1 : r1 : 2, c0 + 1 : c1] = fill_color

    return out


def _paint_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    frame_color: int,
) -> None:
    grid[r0, c0 : c0 + w] = frame_color
    grid[r0 + h - 1, c0 : c0 + w] = frame_color
    grid[r0 : r0 + h, c0] = frame_color
    grid[r0 : r0 + h, c0 + w - 1] = frame_color


def _fill_expected_region(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    fill_color: int,
) -> None:
    grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = bg
    grid[r0 + 1 : r0 + h - 1 : 2, c0 + 1 : c0 + w - 1] = fill_color


def _sample_frame_sizes(rng: np.random.Generator, n_frames: int) -> list[tuple[int, int]]:
    sizes = []
    for _ in range(n_frames):
        h = int(rng.integers(5, 9))
        w = int(rng.integers(6, 12))
        sizes.append((h, w))
    return sizes


def _place_frames(
    rng: np.random.Generator,
    shape: tuple[int, int],
    side: str,
    band_w: int,
    band_h: int,
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)

    for idx in order:
        fh, fw = sizes[idx]
        if side == "top":
            r_min, r_max = band_h + 1, h - fh - 1
            c_min, c_max = 1, w - fw - 1
        elif side == "bottom":
            r_min, r_max = 1, h - band_h - fh - 1
            c_min, c_max = 1, w - fw - 1
        elif side == "left":
            r_min, r_max = 1, h - fh - 1
            c_min, c_max = band_w + 1, w - fw - 1
        else:
            r_min, r_max = 1, h - fh - 1
            c_min, c_max = 1, w - band_w - fw - 1
        if r_max < r_min or c_max < c_min:
            return None

        found = None
        for _ in range(200):
            r0 = int(rng.integers(r_min, r_max + 1))
            c0 = int(rng.integers(c_min, c_max + 1))
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + fh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + fw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + fh, c0 : c0 + fw] = True
            found = (r0, c0, fh, fw)
            break
        if found is None:
            return None
        placed[idx] = found

    return [item for item in placed if item is not None]


def _bar_rows(rng: np.random.Generator, total: int, span: int) -> list[int] | None:
    min_span = 2 * total - 1
    if span < min_span:
        return None
    offset = int(rng.integers(0, span - min_span + 1))
    return [offset + 2 * i for i in range(total)]


def _paint_bar(
    grid: np.ndarray,
    row: int,
    c0: int,
    length: int,
    index_color: int,
    fill_color: int,
) -> list[tuple[int, int]]:
    coords = []
    for k in range(length):
        color = index_color if k == 0 else fill_color
        grid[row, c0 + k] = color
        coords.append((row, c0 + k))
    return coords


def _build_expected(
    inp: np.ndarray,
    frames: list[dict[str, int]],
    bar_pixels: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for r, c in bar_pixels:
        out[r, c] = bg
    for frame in frames:
        _fill_expected_region(
            out,
            frame["r0"],
            frame["c0"],
            frame["h"],
            frame["w"],
            bg,
            frame["fill_color"],
        )
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(2, 5))
    side = SIDES[int(rng.integers(0, len(SIDES)))]
    bg = int(rng.choice([color for color in COLORS if color != GREY]))

    band_w = int(rng.integers(5, 7))
    band_h = 2 * n_frames - 1 + int(rng.integers(0, 3))
    h = int(max(rng.integers(18, 31), band_h + 10))
    w = int(max(rng.integers(18, 31), band_w + 12))

    if side in ("left", "right"):
        rows = _bar_rows(rng, n_frames, h)
    else:
        rows = _bar_rows(rng, n_frames, band_h)
    if rows is None:
        return None

    palette = [color for color in COLORS if color not in (bg, GREY)]
    if len(palette) < 2 * n_frames:
        return None
    index_colors = list(map(int, rng.choice(palette, size=n_frames, replace=False)))
    fill_pool = [color for color in palette if color not in index_colors]
    fill_colors = list(map(int, rng.choice(fill_pool, size=n_frames, replace=False)))
    rng.shuffle(fill_colors)

    sizes = _sample_frame_sizes(rng, n_frames)
    placements = _place_frames(rng, (h, w), side, band_w, band_h, sizes)
    if placements is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    frame_specs: list[dict[str, int]] = []
    for i, (r0, c0, fh, fw) in enumerate(placements):
        _paint_frame(inp, r0, c0, fh, fw, GREY)
        inner_h = fh - 2
        inner_w = fw - 2
        frag_count = int(rng.integers(1, min(5, inner_h * inner_w + 1)))
        picks = rng.choice(inner_h * inner_w, size=frag_count, replace=False)
        for pick in np.atleast_1d(picks):
            local_r = int(pick) // inner_w
            local_c = int(pick) % inner_w
            inp[r0 + 1 + local_r, c0 + 1 + local_c] = index_colors[i]
        frame_specs.append(
            {
                "r0": r0,
                "c0": c0,
                "h": fh,
                "w": fw,
                "index_color": index_colors[i],
                "fill_color": fill_colors[i],
            }
        )

    order = list(range(n_frames))
    rng.shuffle(order)
    bar_pixels: list[tuple[int, int]] = []
    for rank, idx in enumerate(order):
        length = int(rng.integers(2, 5))
        if side == "top":
            row = rows[rank]
            c0 = int(rng.integers(0, w - length + 1))
        elif side == "bottom":
            row = h - band_h + rows[rank]
            c0 = int(rng.integers(0, w - length + 1))
        elif side == "left":
            row = rows[rank]
            c0 = int(rng.integers(0, band_w - length + 1))
        else:
            row = rows[rank]
            c0 = w - band_w + int(rng.integers(0, band_w - length + 1))
        bar_pixels.extend(
            _paint_bar(inp, row, c0, length, frame_specs[idx]["index_color"], frame_specs[idx]["fill_color"])
        )

    counts = Counter(map(int, inp.ravel()))
    bg_count = counts[bg]
    if any(color != bg and count >= bg_count for color, count in counts.items()):
        return None

    expected = _build_expected(inp, frame_specs, bar_pixels, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
