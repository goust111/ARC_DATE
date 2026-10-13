from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEPARATOR = 5
AVAILABLE_COLORS = tuple(color for color in range(10) if color not in {BG, SEPARATOR})
GENERATION_RETRIES = 400
WINDOW_COUNT_CHOICES = (2, 3, 4)
WIDTH_RANGE_BY_WINDOW_COUNT = {
    2: (16, 24),
    3: (18, 27),
    4: (22, 29),
}
BAND_HEIGHT_RANGE = (6, 9)
SAMPLE_SIDE_MIN = 2
SAMPLE_SIDE_MAX = 5
RECT_PADDING = 1
MOSAIC_ATTEMPTS = 80
_RNG = random.Random()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_bg(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
    counts = np.bincount(border.astype(int), minlength=10)
    return int(np.argmax(counts))


def _find_bands(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    h = int(g.shape[0])
    candidates: list[tuple[int, int, int]] = []
    for r1 in range(1, h - 2):
        color = int(g[r1, 0])
        if color == bg or not np.all(g[r1] == color):
            continue
        for r2 in range(r1 + 2, h - 1):
            if not np.all(g[r2] == color):
                continue
            top_h = r1
            mid_h = r2 - r1 - 1
            bot_h = h - r2 - 1
            if top_h == mid_h == bot_h and top_h > 0:
                candidates.append((int(r1), int(r2), int(color)))
    assert len(candidates) == 1, "expected exactly two equal-spacing separator rows"
    return candidates[0]


def _extract_window_frames(layer: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    frames: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, np.unique(layer))) - {bg}):
        comps = _components(layer == int(color))
        assert len(comps) == 1, "each window color must form one frame"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        sub = layer[r0 : r1 + 1, c0 : c1 + 1]
        sh, sw = map(int, sub.shape)
        assert sh >= 4 and sw >= 4, "window frames need a non-trivial interior"
        border_mask = np.zeros_like(sub, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        assert int(coords.shape[0]) == int(border_mask.sum()), "window frame must be one-pixel thick"
        assert np.all(sub[border_mask] == int(color)), "frame border must be monochrome"
        assert np.all(sub[~border_mask] == bg), "window interior must be transparent"
        frames[int(color)] = (int(r0), int(c0), int(r1), int(c1))
    assert frames, "expected at least one window frame"
    return frames


def _extract_template_rects(layer: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, np.unique(layer))) - {bg}):
        comps = _components(layer == int(color))
        assert len(comps) == 1, "each template color must form one solid rectangle"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        sub = layer[r0 : r1 + 1, c0 : c1 + 1]
        assert int(coords.shape[0]) == int(sub.size), "template component must be a filled rectangle"
        assert np.all(sub == int(color)), "template rectangle must be solid"
        rects[int(color)] = (int(r0), int(c0), int(r1), int(c1))
    assert rects, "expected at least one template rectangle"
    return rects


def _sample_patch(data_layer: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    patch = data_layer[r0 + 1 : r1, c0 + 1 : c1]
    assert patch.size > 0
    return patch.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored window onto the data band and paste the revealed patch into the matching template rectangle.

    Input structure:
    - Black background with two full-width gray separator rows that split the grid into three equal-height horizontal bands.
    - The top band contains one multicolor filled rectangle on the background.
    - The middle band contains several one-pixel-thick colored rectangular window frames with black interiors.
    - The bottom band contains one solid rectangle per window color, each sized to receive that window's interior sample.

    Rule:
    - Detect the two separator rows and slice the grid into the data band, window band, and template band.
    - For each colored window frame, use its interior coordinates as a transparent aperture onto the same rows and columns of the data band.
    - Find the solid template rectangle with the same color as that frame.
    - Replace that template rectangle with the sampled data-band patch.
    - Return only the transformed template band.

    Key insight:
    - The middle band contributes no output pixels directly: it only provides sampling coordinates, while matching colors connect each sampled patch to its destination rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_bg(g)
    sep1, sep2, _sep_color = _find_bands(g, bg)

    data_layer = g[:sep1]
    window_layer = g[sep1 + 1 : sep2]
    template_layer = g[sep2 + 1 :]
    assert data_layer.shape == window_layer.shape == template_layer.shape

    frames = _extract_window_frames(window_layer, bg)
    templates = _extract_template_rects(template_layer, bg)
    assert set(frames) == set(templates), "window colors and template colors must match one-to-one"

    out = template_layer.copy()
    for color, frame_bbox in frames.items():
        patch = _sample_patch(data_layer, frame_bbox)
        assert np.all(patch != bg), "window must sample from inside the data rectangle"
        tr0, tc0, tr1, tc1 = templates[int(color)]
        target_shape = (int(tr1 - tr0 + 1), int(tc1 - tc0 + 1))
        assert patch.shape == target_shape, "template size must match the corresponding window interior"
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = patch
    return out


def _randint(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(int(lo), int(hi)))


def _choice(rng: random.Random, seq):
    return seq[_randint(rng, 0, len(seq) - 1)]


def _place_rectangles(
    rng: random.Random,
    area_h: int,
    area_w: int,
    sizes: list[tuple[int, int]],
    padding: int,
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    all_locations = {
        idx: [(r, c) for r in range(area_h - sizes[idx][0] + 1) for c in range(area_w - sizes[idx][1] + 1)]
        for idx in order
    }

    for _ in range(120):
        occupied = np.zeros((area_h, area_w), dtype=bool)
        placed: list[tuple[int, int] | None] = [None] * len(sizes)
        ok = True
        for idx in order:
            h, w = sizes[idx]
            candidates = all_locations[idx][:]
            rng.shuffle(candidates)
            anchor = None
            for r, c in candidates:
                rr0 = max(0, r - padding)
                cc0 = max(0, c - padding)
                rr1 = min(area_h, r + h + padding)
                cc1 = min(area_w, c + w + padding)
                if bool(occupied[rr0:rr1, cc0:cc1].any()):
                    continue
                anchor = (int(r), int(c))
                occupied[r : r + h, c : c + w] = True
                break
            if anchor is None:
                ok = False
                break
            placed[idx] = anchor
        if not ok:
            continue

        rects: list[tuple[int, int, int, int]] = []
        for idx, anchor in enumerate(placed):
            assert anchor is not None
            r, c = anchor
            h, w = sizes[idx]
            rects.append((int(r), int(c), int(r + h - 1), int(c + w - 1)))
        return rects
    return None


def _make_mosaic(rng: random.Random, h: int, w: int, colors: list[int]) -> np.ndarray | None:
    need = min(3, len(colors))
    for _ in range(MOSAIC_ATTEMPTS):
        block_h = _choice(rng, (1, 1, 2))
        block_w = _choice(rng, (1, 2, 2, 3))
        patch = np.zeros((h, w), dtype=int)
        for r in range(0, h, block_h):
            for c in range(0, w, block_w):
                color = int(_choice(rng, colors))
                patch[r : min(h, r + block_h), c : min(w, c + block_w)] = color

        for _ in range(_randint(rng, 1, 3)):
            color = int(_choice(rng, colors))
            if bool(rng.getrandbits(1)):
                rr = _randint(rng, 0, h - 1)
                patch[rr, :] = color
            else:
                cc = _randint(rng, 0, w - 1)
                patch[:, cc] = color

        for _ in range(_randint(rng, 0, 2)):
            color = int(_choice(rng, colors))
            ph = _randint(rng, 1, min(3, h))
            pw = _randint(rng, 1, min(3, w))
            r0 = _randint(rng, 0, h - ph)
            c0 = _randint(rng, 0, w - pw)
            patch[r0 : r0 + ph, c0 : c0 + pw] = color

        if len(set(map(int, np.unique(patch)))) >= need:
            return patch
    return None


def _build_input_and_output(
    band_h: int,
    width: int,
    data_bbox: tuple[int, int, int, int],
    mosaic: np.ndarray,
    windows: dict[int, tuple[int, int, int, int]],
    templates: dict[int, tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    total_h = band_h * 3 + 2
    grid = np.full((total_h, width), BG, dtype=int)
    grid[band_h, :] = SEPARATOR
    grid[2 * band_h + 1, :] = SEPARATOR

    data_layer = grid[:band_h]
    window_layer = grid[band_h + 1 : 2 * band_h + 1]
    template_layer = grid[2 * band_h + 2 :]

    dr0, dc0, dr1, dc1 = data_bbox
    data_layer[dr0 : dr1 + 1, dc0 : dc1 + 1] = mosaic

    for color, (r0, c0, r1, c1) in windows.items():
        window_layer[r0, c0 : c1 + 1] = int(color)
        window_layer[r1, c0 : c1 + 1] = int(color)
        window_layer[r0 : r1 + 1, c0] = int(color)
        window_layer[r0 : r1 + 1, c1] = int(color)

    expected = template_layer.copy()
    for color, (r0, c0, r1, c1) in templates.items():
        template_layer[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        sample = _sample_patch(data_layer, windows[int(color)])
        expected[r0 : r1 + 1, c0 : c1 + 1] = sample

    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_windows = int(_choice(rng, WINDOW_COUNT_CHOICES))
    band_h = _randint(rng, BAND_HEIGHT_RANGE[0], BAND_HEIGHT_RANGE[1])
    width_lo, width_hi = WIDTH_RANGE_BY_WINDOW_COUNT[n_windows]
    width = _randint(rng, width_lo, width_hi)

    max_sample_h = min(SAMPLE_SIDE_MAX, band_h - 2)
    max_sample_w = min(SAMPLE_SIDE_MAX, max(SAMPLE_SIDE_MIN, (width - (n_windows - 1)) // n_windows))
    if max_sample_h < SAMPLE_SIDE_MIN or max_sample_w < SAMPLE_SIDE_MIN:
        return None

    sample_sizes: list[tuple[int, int]] = []
    for _ in range(n_windows):
        sh = _randint(rng, SAMPLE_SIDE_MIN, max_sample_h)
        sw = _randint(rng, SAMPLE_SIDE_MIN, max_sample_w)
        sample_sizes.append((int(sh), int(sw)))

    template_rects = _place_rectangles(rng, band_h, width, sample_sizes, RECT_PADDING)
    if template_rects is None:
        return None

    window_sizes = [(int(h + 2), int(w + 2)) for h, w in sample_sizes]
    data_h = band_h - _choice(rng, (0, 0, 1))
    min_data_w = max(8, max(size[1] for size in window_sizes))
    data_w = width - _randint(rng, 2, min(5, width - min_data_w))
    if data_h < max(size[0] for size in window_sizes) or data_w < min_data_w:
        return None

    window_rects_local = _place_rectangles(rng, data_h, data_w, window_sizes, RECT_PADDING)
    if window_rects_local is None:
        return None

    dr0 = _randint(rng, 0, band_h - data_h)
    dc0 = _randint(rng, 0, width - data_w)
    data_bbox = (int(dr0), int(dc0), int(dr0 + data_h - 1), int(dc0 + data_w - 1))
    window_rects = [
        (int(r0 + dr0), int(c0 + dc0), int(r1 + dr0), int(c1 + dc0))
        for r0, c0, r1, c1 in window_rects_local
    ]

    data_color_cap = min(4, len(AVAILABLE_COLORS) - n_windows)
    data_color_count = _randint(rng, 3, data_color_cap)
    data_colors = random.sample(list(AVAILABLE_COLORS), k=data_color_count)
    key_pool = [color for color in AVAILABLE_COLORS if color not in data_colors]
    key_colors = random.sample(key_pool, k=n_windows)

    mosaic = None
    for _ in range(MOSAIC_ATTEMPTS):
        candidate = _make_mosaic(rng, data_h, data_w, data_colors)
        if candidate is None:
            continue
        samples = []
        ok = True
        for r0, c0, r1, c1 in window_rects:
            local = candidate[r0 - dr0 + 1 : r1 - dr0, c0 - dc0 + 1 : c1 - dc0]
            if len(set(map(int, np.unique(local)))) < 2:
                ok = False
                break
            samples.append(local.tobytes())
        if ok and len(set(samples)) >= min(2, len(samples)):
            mosaic = candidate
            break
    if mosaic is None:
        return None

    windows = {int(color): rect for color, rect in zip(key_colors, window_rects)}
    templates = {int(color): rect for color, rect in zip(key_colors, template_rects)}
    grid, expected = _build_input_and_output(band_h, width, data_bbox, mosaic, windows, templates)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            result = _gen_attempt(_RNG)
        except AssertionError:
            continue
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle instance")
