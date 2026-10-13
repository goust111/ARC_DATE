import random
from collections import deque

import numpy as np


BG = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            seen[sr, sc] = True
            q = deque([(sr, sc)])
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _perimeter_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frames(g: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color0 in np.unique(g):
        color = int(color0)
        if color == BG:
            continue
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            fh, fw = r1 - r0 + 1, c1 - c0 + 1
            if fh < 4 or fw < 4:
                continue

            local = np.zeros((fh, fw), dtype=bool)
            for r, c in pts:
                local[r - r0, c - c0] = True
            if not np.array_equal(local, _perimeter_mask(fh, fw)):
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if np.any(interior == color):
                continue
            seeds = np.argwhere((interior != BG) & (interior != color))
            if len(seeds) != 1:
                continue
            sr, sc = map(int, seeds[0])
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "seed_color": int(interior[sr, sc]),
                    "seed": (r0 + 1 + sr, c0 + 1 + sc),
                }
            )
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _inside_any_box(r: int, c: int, boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes)


def _pattern_candidates(
    g: np.ndarray, label_r: int, label_c: int, frame_boxes: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int, np.ndarray]]:
    h, w = g.shape
    label_color = int(g[label_r, label_c])
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []

    for size in (3, 2):
        for label_row_offset in range(size):
            pr0 = label_r - label_row_offset
            pc0 = label_c + 1
            if pr0 < 0 or pc0 < 0 or pr0 + size > h or pc0 + size > w:
                continue
            if _inside_any_box(label_r, label_c, frame_boxes):
                continue
            blocked = False
            for rr in range(pr0, pr0 + size):
                for cc in range(pc0, pc0 + size):
                    if _inside_any_box(rr, cc, frame_boxes):
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue

            patch = g[pr0 : pr0 + size, pc0 : pc0 + size]
            vals = {int(v) for v in np.unique(patch)}
            nonzero = [v for v in vals if v != BG]
            if len(nonzero) != 1 or BG not in vals:
                continue
            decor = nonzero[0]
            if decor == label_color:
                continue

            visible = patch != BG
            if patch[label_row_offset, 0] == BG:
                continue
            if not (visible[0, :].any() and visible[-1, :].any()):
                continue
            if not (visible[:, 0].any() and visible[:, -1].any()):
                continue

            candidates.append((size, int(np.count_nonzero(visible)), pr0, pc0, patch.copy()))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    return candidates


def _legend_patterns(
    g: np.ndarray, seed_colors: set[int], frame_boxes: list[tuple[int, int, int, int]]
) -> dict[int, np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    for color in seed_colors:
        all_candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        for r0, c0 in np.argwhere(g == color):
            r, c = int(r0), int(c0)
            if _inside_any_box(r, c, frame_boxes):
                continue
            all_candidates.extend(_pattern_candidates(g, r, c, frame_boxes))
        assert all_candidates, f"missing legend pattern for seed color {color}"
        patterns[color] = all_candidates[0][4]
    return patterns


def _tile_patch(out: np.ndarray, bbox: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    ih, iw = ir1 - ir0 + 1, ic1 - ic0 + 1
    ph, pw = pattern.shape
    rr = np.arange(ih) % ph
    cc = np.arange(iw) % pw
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = pattern[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior by tiling the legend pattern keyed by its seed color.

    Input structure:
    - The background is black. Several complete one-pixel-thick rectangular frames are drawn in a common non-black frame color.
    - Each frame interior contains exactly one non-black seed pixel whose color is not the frame color.
    - Away from the frames, every seed color appears as a one-pixel legend label immediately left of a 2x2 or 3x3 black-and-decoration pattern block.

    Rule:
    - Read each frame's seed color and find the legend whose label has the same color.
    - Copy that legend's pattern block into the whole frame interior as a periodic tile, aligned to the interior's upper-left cell and clipped at the frame boundary.
    - Overwrite the original seed pixel with the tiled pattern while leaving the frame borders and the legend area unchanged.

    Key insight:
    - The seed is an index into a visible color legend; the selected block is repeated at its native size instead of being stretched to the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"
    frame_boxes = [frame["bbox"] for frame in frames]  # type: ignore[list-item]
    seed_colors = {int(frame["seed_color"]) for frame in frames}
    patterns = _legend_patterns(g, seed_colors, frame_boxes)

    out = g.copy()
    for frame in frames:
        seed_color = int(frame["seed_color"])
        _tile_patch(out, frame["bbox"], patterns[seed_color])  # type: ignore[arg-type]
    return out


# ----------------------------- generator -----------------------------

_MAX_GEN_ATTEMPTS = 300
_GRID_MIN, _GRID_MAX = 16, 30
_FRAME_COUNT_MAX = 5


def _make_pattern(size: int, color: int) -> np.ndarray:
    if size == 2:
        masks = [
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.array([[1, 0], [1, 1]], dtype=bool),
            np.array([[1, 1], [0, 1]], dtype=bool),
            np.array([[1, 0], [0, 1]], dtype=bool),
        ]
        mask = random.choice(masks).copy()
    else:
        for _ in range(100):
            mask = np.random.random((3, 3)) < random.uniform(0.35, 0.65)
            mask[0, 0] = True
            mask[0, random.randrange(3)] = True
            mask[-1, random.randrange(3)] = True
            mask[random.randrange(3), 0] = True
            mask[random.randrange(3), -1] = True
            if mask.all():
                mask[random.choice([(1, 1), (0, 2), (2, 0), (2, 2)])] = False
            if (not mask.all()) and mask[0, 0] and mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any():
                break
        else:
            mask = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=bool)
    return np.where(mask, int(color), BG).astype(int)


def _draw_legend(
    grid: np.ndarray,
    occ: np.ndarray,
    entries: list[tuple[int, np.ndarray]],
    side: str,
    size: int,
) -> None:
    h, w = grid.shape
    gap = 1
    if side == "left":
        total_h = len(entries) * size + (len(entries) - 1) * gap
        start_r = random.randint(0, h - total_h)
        for i, (seed_color, pattern) in enumerate(entries):
            r0 = start_r + i * (size + gap)
            grid[r0, 0] = seed_color
            grid[r0 : r0 + size, 1 : 1 + size] = pattern
            rr0, rr1 = max(0, r0 - 1), min(h, r0 + size + 1)
            cc0, cc1 = 0, min(w, size + 2)
            occ[rr0:rr1, cc0:cc1] = True
    else:
        total_w = len(entries) * (size + 1) + (len(entries) - 1) * gap
        start_c = random.randint(0, w - total_w)
        for i, (seed_color, pattern) in enumerate(entries):
            c0 = start_c + i * (size + 1 + gap)
            grid[0, c0] = seed_color
            grid[0:size, c0 + 1 : c0 + 1 + size] = pattern
            rr0, rr1 = 0, min(h, size + 1)
            cc0, cc1 = max(0, c0 - 1), min(w, c0 + size + 2)
            occ[rr0:rr1, cc0:cc1] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1, c1 = r0 + height - 1, c0 + width - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _place_frame(occ: np.ndarray, height: int, width: int) -> tuple[int, int] | None:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - height):
        for c0 in range(1, w - width):
            rr0, rr1 = max(0, r0 - 1), min(h, r0 + height + 1)
            cc0, cc1 = max(0, c0 - 1), min(w, c0 + width + 1)
            if not occ[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    if not positions:
        return None
    return random.choice(positions)


def _expected_output(inp: np.ndarray, frame_specs: list[tuple[tuple[int, int, int, int], int]], patterns: dict[int, np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for bbox, seed_color in frame_specs:
        _tile_patch(out, bbox, patterns[seed_color])
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(_GRID_MIN, _GRID_MAX)
    cols = random.randint(_GRID_MIN, _GRID_MAX)
    size = random.choice([2, 3])
    side = random.choice(["left", "top"])

    max_legend = 4
    if side == "left":
        max_fit = (rows + 1) // (size + 1)
    else:
        max_fit = (cols + 1) // (size + 2)
    n_keys = random.randint(2, min(max_legend, max_fit))

    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_color = colors[0]
    seed_colors = colors[1 : 1 + n_keys]
    decor_colors = colors[1 + n_keys : 1 + 2 * n_keys]
    if len(decor_colors) < n_keys:
        return None

    patterns = {
        seed: _make_pattern(size, decor)
        for seed, decor in zip(seed_colors, decor_colors, strict=False)
    }

    inp = np.full((rows, cols), BG, dtype=int)
    occ = np.zeros((rows, cols), dtype=bool)
    entries = [(seed, patterns[seed]) for seed in seed_colors]
    random.shuffle(entries)
    _draw_legend(inp, occ, entries, side, size)

    n_frames = random.randint(n_keys, min(_FRAME_COUNT_MAX, n_keys + 2))
    frame_seeds = seed_colors[:]
    while len(frame_seeds) < n_frames:
        frame_seeds.append(random.choice(seed_colors))
    random.shuffle(frame_seeds)

    frame_specs: list[tuple[tuple[int, int, int, int], int]] = []
    for seed_color in frame_seeds:
        inner_h = random.randint(3, 8)
        inner_w = random.randint(3, 8)
        fh, fw = inner_h + 2, inner_w + 2
        pos = _place_frame(occ, fh, fw)
        if pos is None:
            return None
        r0, c0 = pos
        _draw_frame(inp, r0, c0, fh, fw, frame_color)
        sr = random.randint(r0 + 1, r0 + inner_h)
        sc = random.randint(c0 + 1, c0 + inner_w)
        inp[sr, sc] = int(seed_color)

        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        frame_specs.append((bbox, int(seed_color)))
        rr0, rr1 = max(0, r0 - 1), min(rows, r0 + fh + 1)
        cc0, cc1 = max(0, c0 - 1), min(cols, c0 + fw + 1)
        occ[rr0:rr1, cc0:cc1] = True

    expected = _expected_output(inp, frame_specs, patterns)
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1155420539/5/1 input")
