import random
from collections import Counter

import numpy as np


GEN_RETRIES = 400
BG = 0
GAP = 1


PATTERNS = {
    2: [
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ],
    3: [
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=int),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask_1d.tolist()):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        rects: list[tuple[int, int, int, int]] = []
        for comp in _components(grid == color):
            coords = np.array(comp, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            hh = int(r1 - r0 + 1)
            ww = int(c1 - c0 + 1)
            if hh < 3 or ww < 3:
                continue
            if len(comp) != 2 * (hh + ww) - 4:
                continue
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.array_equal((grid[r0 : r1 + 1, c0 : c1 + 1] == color), border):
                rects.append((int(r0), int(c0), int(r1), int(c1)))
        if rects:
            candidates[color] = sorted(rects)
    assert candidates, "no rectangular frame components found"
    frame_color = max(candidates, key=lambda color: (len(candidates[color]), sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in candidates[color])))
    return int(frame_color), candidates[frame_color]


def _legend_strip(grid: np.ndarray, bg: int) -> tuple[str, np.ndarray]:
    h, w = grid.shape
    sides: list[tuple[str, np.ndarray]] = []
    if np.any(grid[0, :] != bg):
        for r in range(1, h):
            if np.all(grid[r, :] == bg):
                sides.append(("top", grid[:r, :]))
                break
    if np.any(grid[-1, :] != bg):
        for r in range(h - 2, -1, -1):
            if np.all(grid[r, :] == bg):
                sides.append(("bottom", grid[r + 1 :, :]))
                break
    if np.any(grid[:, 0] != bg):
        for c in range(1, w):
            if np.all(grid[:, c] == bg):
                sides.append(("left", grid[:, :c]))
                break
    if np.any(grid[:, -1] != bg):
        for c in range(w - 2, -1, -1):
            if np.all(grid[:, c] == bg):
                sides.append(("right", grid[:, c + 1 :]))
                break
    assert len(sides) == 1, "legend must occupy exactly one edge strip"
    return sides[0]


def _decode_legend(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    side, strip = _legend_strip(grid, bg)
    mapping: dict[int, np.ndarray] = {}
    if side in ("top", "bottom"):
        span = strip.shape[0]
        assert span in (3, 4), "horizontal legend strip must be 3 or 4 cells tall"
        groups = _runs(np.any(strip != bg, axis=0))
        for c0, c1 in groups:
            assert c1 - c0 + 1 == span, "horizontal legend blocks must be square"
            block = strip[:, c0 : c1 + 1]
            key = int(block[0, 0])
            assert key != bg, "legend key must be visible"
            sample = block[1:, 1:]
            assert sample.shape in ((2, 2), (3, 3)), "legend samples must be 2x2 or 3x3"
            vals = set(map(int, np.unique(sample)))
            assert vals <= {bg, key} and key in vals and bg in vals, "legend sample must use only background and key color"
            assert key not in mapping, "legend keys must be unique"
            mapping[key] = sample.copy()
    else:
        span = strip.shape[1]
        assert span in (3, 4), "vertical legend strip must be 3 or 4 cells wide"
        groups = _runs(np.any(strip != bg, axis=1))
        for r0, r1 in groups:
            assert r1 - r0 + 1 == span, "vertical legend blocks must be square"
            block = strip[r0 : r1 + 1, :]
            key = int(block[0, 0])
            assert key != bg, "legend key must be visible"
            sample = block[1:, 1:]
            assert sample.shape in ((2, 2), (3, 3)), "legend samples must be 2x2 or 3x3"
            vals = set(map(int, np.unique(sample)))
            assert vals <= {bg, key} and key in vals and bg in vals, "legend sample must use only background and key color"
            assert key not in mapping, "legend keys must be unique"
            mapping[key] = sample.copy()
    assert mapping, "legend mapping must be non-empty"
    return mapping


def _tile(texture: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % texture.shape[0]
    cc = np.arange(width)[None, :] % texture.shape[1]
    return texture[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frames = _find_frames(g, bg)
    mapping = _decode_legend(g, bg)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_colors = {int(v) for v in interior.flat if int(v) not in (bg, frame_color)}
        assert len(seed_colors) == 1, "each frame must contain exactly one seed color"
        seed_color = next(iter(seed_colors))
        assert seed_color in mapping, "every seed color must appear in the legend"
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile(mapping[seed_color], interior.shape[0], interior.shape[1])
    return out


def _make_texture(color: int, pattern: np.ndarray) -> np.ndarray:
    return np.where(pattern.astype(bool), color, BG).astype(int)


def _make_legend_block(color: int, texture: np.ndarray) -> np.ndarray:
    n = texture.shape[0]
    block = np.full((n + 1, n + 1), BG, dtype=int)
    block[0, 0] = color
    block[1:, 1:] = texture
    return block


def _draw_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    side: str,
    blocks: list[np.ndarray],
    rng: random.Random,
) -> None:
    h, w = grid.shape
    span = blocks[0].shape[0]
    total_len = len(blocks) * span + (len(blocks) - 1) * GAP

    if side in ("top", "bottom"):
        min_start = 1
        max_start = w - total_len - 1
        assert max_start >= min_start
        start = rng.randint(min_start, max_start)
        row0 = 0 if side == "top" else h - span
        for i, block in enumerate(blocks):
            col0 = start + i * (span + GAP)
            grid[row0 : row0 + span, col0 : col0 + span] = block
        occupied[row0 : row0 + span, :] = True
        sep_row = span if side == "top" else h - span - 1
        occupied[sep_row, :] = True
    else:
        min_start = 1
        max_start = h - total_len - 1
        assert max_start >= min_start
        start = rng.randint(min_start, max_start)
        col0 = 0 if side == "left" else w - span
        for i, block in enumerate(blocks):
            row0 = start + i * (span + GAP)
            grid[row0 : row0 + span, col0 : col0 + span] = block
        occupied[:, col0 : col0 + span] = True
        sep_col = span if side == "left" else w - span - 1
        occupied[:, sep_col] = True


def _valid_positions(occupied: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - height + 1):
        for c0 in range(w - width + 1):
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h, r0 + height + 1)
            pc1 = min(w, c0 + width + 1)
            if not occupied[pr0:pr1, pc0:pc1].any():
                positions.append((r0, c0))
    return positions


def _place_frames(
    grid: np.ndarray,
    occupied: np.ndarray,
    frame_color: int,
    seed_colors: list[int],
    texture_size: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    frames: list[tuple[int, int, int, int, int]] = []
    for seed_color in seed_colors:
        placed = False
        for _ in range(40):
            inner_h_max = min(8, h - 6)
            inner_w_max = min(8, w - 6)
            if inner_h_max < texture_size + 1 or inner_w_max < texture_size + 1:
                return []
            inner_h = rng.randint(texture_size + 1, inner_h_max)
            inner_w = rng.randint(texture_size + 1, inner_w_max)
            hh = inner_h + 2
            ww = inner_w + 2
            positions = _valid_positions(occupied, hh, ww)
            if not positions:
                continue
            r0, c0 = rng.choice(positions)
            r1 = r0 + hh - 1
            c1 = c0 + ww - 1
            grid[r0, c0 : c1 + 1] = frame_color
            grid[r1, c0 : c1 + 1] = frame_color
            grid[r0 : r1 + 1, c0] = frame_color
            grid[r0 : r1 + 1, c1] = frame_color
            sr = rng.randint(r0 + 1, r1 - 1)
            sc = rng.randint(c0 + 1, c1 - 1)
            grid[sr, sc] = seed_color
            occupied[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True
            frames.append((r0, c0, r1, c1, seed_color))
            placed = True
            break
        if not placed:
            return []
    return frames


def _expected_output(
    inp: np.ndarray,
    frames: list[tuple[int, int, int, int, int]],
    textures: dict[int, np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    for r0, c0, r1, c1, seed_color in frames:
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile(textures[seed_color], r1 - r0 - 1, c1 - c0 - 1)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    texture_size = rng.choice([2, 3])
    n_frames = rng.choice([2, 3, 4])
    side = rng.choice(["top", "bottom", "left", "right"])
    span = texture_size + 1

    legend_min = 2 + n_frames * span + (n_frames - 1) * GAP
    if side in ("top", "bottom"):
        h = rng.randint(span + 11, span + 18)
        w = rng.randint(max(legend_min, 16), max(legend_min, 16) + 8)
    else:
        h = rng.randint(max(legend_min, 16), max(legend_min, 16) + 8)
        w = rng.randint(span + 11, span + 18)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    frame_color = colors[0]
    seed_colors = colors[1 : 1 + n_frames]
    patterns = rng.sample(PATTERNS[texture_size], n_frames)
    textures = {color: _make_texture(color, pattern) for color, pattern in zip(seed_colors, patterns)}
    blocks = [_make_legend_block(color, textures[color]) for color in seed_colors]
    rng.shuffle(blocks)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, :] = True
    occupied[-1, :] = True
    occupied[:, 0] = True
    occupied[:, -1] = True

    try:
        _draw_legend(grid, occupied, side, blocks, rng)
    except AssertionError:
        return None

    frames = _place_frames(grid, occupied, frame_color, seed_colors, texture_size, rng)
    if len(frames) != n_frames:
        return None

    if _get_bg(grid) != BG:
        return None

    expected = _expected_output(grid, frames, textures)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
