import random
from typing import Iterable

import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _extract_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for comp in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(comp) != _perimeter_cells(r0, c0, r1, c1):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        non_black = interior[interior != BLACK]
        if non_black.size != 1:
            continue
        seed_color = int(non_black[0])
        if seed_color == GRAY:
            continue
        if np.any((interior != BLACK) & (interior != seed_color)):
            continue
        seed_pos = np.argwhere(interior == seed_color)[0]
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed_color": seed_color,
                "seed_r": int(seed_pos[0] + r0 + 1),
                "seed_c": int(seed_pos[1] + c0 + 1),
            }
        )
    assert frames, "expected at least one framed seed region"
    return frames


def _extract_mapping(
    grid: np.ndarray, frames: list[dict[str, int | tuple[int, int, int, int]]]
) -> dict[int, int]:
    frame_boxes = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_boxes[r0 : r1 + 1, c0 : c1 + 1] = True

    legend_mask = (grid != BLACK) & ~frame_boxes
    comps = _components(legend_mask)
    mapping: dict[int, int] = {}
    for comp in comps:
        assert len(comp) == 2, "legend components must be isolated 2-pixel pairs"
        (r0, c0), (r1, c1) = sorted(comp)
        assert c0 == c1 and r1 == r0 + 1, "legend pairs must be vertical dominos"
        a = int(grid[r0, c0])
        b = int(grid[r1, c1])
        assert a not in (BLACK, GRAY) and b not in (BLACK, GRAY), "legend colors must be visible colors"
        assert a != b, "legend pair colors must differ"
        if a in mapping:
            assert mapping[a] == b
        if b in mapping:
            assert mapping[b] == a
        mapping[a] = b
        mapping[b] = a

    assert mapping, "expected a non-empty complement mapping"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray-framed region with its seed color and recolor the frame
    using that seed color's complement from the separate vertical-pair legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    mapping = _extract_mapping(g, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        seed = int(frame["seed_color"])
        assert seed in mapping, "every seed color must appear in the legend"
        comp = int(mapping[seed])

        out[r0 + 1 : r1, c0 + 1 : c1] = seed
        out[r0, c0 : c1 + 1] = comp
        out[r1, c0 : c1 + 1] = comp
        out[r0 : r1 + 1, c0] = comp
        out[r0 : r1 + 1, c1] = comp

    return out


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    h0, w0 = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _apply_expected(
    grid: np.ndarray, frames: list[dict[str, int]], mapping: dict[int, int]
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        seed = int(frame["seed_color"])
        comp = int(mapping[seed])
        out[r0 + 1 : r1, c0 + 1 : c1] = seed
        out[r0, c0 : c1 + 1] = comp
        out[r1, c0 : c1 + 1] = comp
        out[r0 : r1 + 1, c0] = comp
        out[r0 : r1 + 1, c1] = comp
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(12, 24)
    width = rng.randint(12, 24)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    pair_count = rng.randint(2, 4)
    legend_colors = rng.sample(COLOR_POOL, 2 * pair_count)
    rng.shuffle(legend_colors)
    pairs = [(legend_colors[2 * i], legend_colors[2 * i + 1]) for i in range(pair_count)]

    table_cols = rng.randint(1, min(3, pair_count))
    table_rows = (pair_count + table_cols - 1) // table_cols
    table_h = table_rows * 3 - 1
    table_w = table_cols * 2 - 1

    placed = False
    for _ in range(80):
        tr = rng.randint(0, height - table_h)
        tc = rng.randint(0, width - table_w)
        if not _can_place(reserved, tr, tc, table_h, table_w):
            continue
        for idx, (a, b) in enumerate(pairs):
            rr = idx // table_cols
            cc = idx % table_cols
            base_r = tr + rr * 3
            base_c = tc + cc * 2
            top, bottom = (a, b) if rng.random() < 0.5 else (b, a)
            grid[base_r, base_c] = top
            grid[base_r + 1, base_c] = bottom
        _reserve(reserved, tr, tc, table_h, table_w)
        placed = True
        break
    if not placed:
        return None

    mapping: dict[int, int] = {}
    for a, b in pairs:
        mapping[a] = b
        mapping[b] = a

    frame_count = rng.randint(1, min(4, len(mapping)))
    seed_colors = rng.sample(list(mapping.keys()), frame_count)

    frames: list[dict[str, int]] = []
    for seed_color in seed_colors:
        success = False
        for _ in range(120):
            box_h = rng.randint(4, min(8, height))
            box_w = rng.randint(4, min(8, width))
            r0 = rng.randint(0, height - box_h)
            c0 = rng.randint(0, width - box_w)
            if not _can_place(reserved, r0, c0, box_h, box_w):
                continue

            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            grid[r0, c0 : c1 + 1] = GRAY
            grid[r1, c0 : c1 + 1] = GRAY
            grid[r0 : r1 + 1, c0] = GRAY
            grid[r0 : r1 + 1, c1] = GRAY

            sr = rng.randint(r0 + 1, r1 - 1)
            sc = rng.randint(c0 + 1, c1 - 1)
            grid[sr, sc] = seed_color

            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "seed_color": seed_color,
                    "seed_r": sr,
                    "seed_c": sc,
                }
            )
            _reserve(reserved, r0, c0, box_h, box_w)
            success = True
            break
        if not success:
            return None

    expected = _apply_expected(grid, frames, mapping)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
