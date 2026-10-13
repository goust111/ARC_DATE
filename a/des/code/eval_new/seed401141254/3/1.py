from __future__ import annotations

import random
from collections import deque

import numpy as np


PATTERN_BANK: dict[int, list[np.ndarray]] = {
    2: [
        np.array([[1, 0], [0, 1]], dtype=bool),
        np.array([[0, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 0]], dtype=bool),
        np.array([[1, 0], [1, 0]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
    ],
    3: [
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _outer_border_color(cell: np.ndarray) -> int | None:
    border = np.concatenate([cell[0], cell[-1], cell[1:-1, 0], cell[1:-1, -1]])
    values = np.unique(border)
    if len(values) != 1:
        return None
    return int(values[0])


def _corner_block_candidates(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    if int(grid[0, 0]) != bg:
        hh = 0
        while hh < h and int(grid[hh, 0]) != bg:
            hh += 1
        ww = 0
        while ww < w and int(grid[0, ww]) != bg:
            ww += 1
        block = grid[:hh, :ww]
        if (
            hh >= 3
            and ww >= 3
            and bool((block != bg).all())
            and (hh == h or bool((grid[hh, :ww] == bg).all()))
            and (ww == w or bool((grid[:hh, ww] == bg).all()))
        ):
            candidates.append(((0, 0, hh, ww), block))

    if int(grid[0, w - 1]) != bg:
        hh = 0
        while hh < h and int(grid[hh, w - 1]) != bg:
            hh += 1
        ww = 0
        while ww < w and int(grid[0, w - 1 - ww]) != bg:
            ww += 1
        c0 = w - ww
        block = grid[:hh, c0:]
        if (
            hh >= 3
            and ww >= 3
            and bool((block != bg).all())
            and (hh == h or bool((grid[hh, c0:] == bg).all()))
            and (c0 == 0 or bool((grid[:hh, c0 - 1] == bg).all()))
        ):
            candidates.append(((0, c0, hh, w), block))

    if int(grid[h - 1, 0]) != bg:
        hh = 0
        while hh < h and int(grid[h - 1 - hh, 0]) != bg:
            hh += 1
        ww = 0
        while ww < w and int(grid[h - 1, ww]) != bg:
            ww += 1
        r0 = h - hh
        block = grid[r0:, :ww]
        if (
            hh >= 3
            and ww >= 3
            and bool((block != bg).all())
            and (r0 == 0 or bool((grid[r0 - 1, :ww] == bg).all()))
            and (ww == w or bool((grid[r0:, ww] == bg).all()))
        ):
            candidates.append(((r0, 0, h, ww), block))

    if int(grid[h - 1, w - 1]) != bg:
        hh = 0
        while hh < h and int(grid[h - 1 - hh, w - 1]) != bg:
            hh += 1
        ww = 0
        while ww < w and int(grid[h - 1, w - 1 - ww]) != bg:
            ww += 1
        r0 = h - hh
        c0 = w - ww
        block = grid[r0:, c0:]
        if (
            hh >= 3
            and ww >= 3
            and bool((block != bg).all())
            and (r0 == 0 or bool((grid[r0 - 1, c0:] == bg).all()))
            and (c0 == 0 or bool((grid[r0:, c0 - 1] == bg).all()))
        ):
            candidates.append(((r0, c0, h, w), block))

    return candidates


def _parse_legend_block(block: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], int] | None:
    h, w = block.shape
    valid: list[tuple[dict[int, np.ndarray], int]] = []

    for side in range(3, min(h, w) + 1):
        if h % side != 0 or w % side != 0:
            continue
        n_cells = (h // side) * (w // side)
        if n_cells < 2:
            continue

        mapping: dict[int, np.ndarray] = {}
        ok = True
        for r0 in range(0, h, side):
            for c0 in range(0, w, side):
                cell = block[r0 : r0 + side, c0 : c0 + side]
                border = _outer_border_color(cell)
                if border is None:
                    ok = False
                    break

                colors = set(map(int, np.unique(cell)))
                if len(colors) != 2 or border not in colors:
                    ok = False
                    break
                colors.remove(border)
                fill = colors.pop()

                inner = cell[1:-1, 1:-1]
                if not bool((inner == fill).any()):
                    ok = False
                    break
                if border in mapping:
                    ok = False
                    break
                mapping[int(border)] = np.where(inner == fill, fill, bg).astype(int)
            if not ok:
                break

        if ok and len(mapping) == n_cells:
            valid.append((mapping, side))

    if not valid:
        return None

    valid.sort(key=lambda item: (len(item[0]), item[1]), reverse=True)
    return valid[0]


def _extract_legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int], int]:
    choices: list[tuple[dict[int, np.ndarray], tuple[int, int, int, int], int]] = []
    for bbox, block in _corner_block_candidates(grid, bg):
        parsed = _parse_legend_block(block, bg)
        if parsed is None:
            continue
        mapping, side = parsed
        choices.append((mapping, bbox, side))

    assert choices, "legend table not found"
    choices.sort(key=lambda item: (len(item[0]), item[2]), reverse=True)
    return choices[0]


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    frames: list[dict[str, int]] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or bool(seen[sr, sc]):
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not bool(seen[rr, cc]) and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            rows = [r for r, _ in pts]
            cols = [c for _, c in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            ph = r1 - r0 + 1
            pw = c1 - c0 + 1
            if ph < 3 or pw < 3:
                continue
            if len(pts) != 2 * (ph + pw) - 4:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                bool((patch[0, :] == color).all())
                and bool((patch[-1, :] == color).all())
                and bool((patch[:, 0] == color).all())
                and bool((patch[:, -1] == color).all())
            ):
                continue
            if not bool((patch[1:-1, 1:-1] == bg).all()):
                continue

            frames.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})

    return frames


def _tile_pattern(pattern: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    ph, pw = pattern.shape
    reps_h = (target_h + ph - 1) // ph
    reps_w = (target_w + pw - 1) // pw
    tiled = np.tile(pattern, (reps_h, reps_w))
    return tiled[:target_h, :target_w]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, _bbox, _side = _extract_legend_mapping(g, bg)
    frames = _find_frames(g, bg)
    assert frames, "no rectangular walls found"

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        assert color in mapping, "wall color missing from legend"
        inner_h = int(frame["r1"] - frame["r0"] - 1)
        inner_w = int(frame["c1"] - frame["c0"] - 1)
        fill = _tile_pattern(mapping[color], inner_h, inner_w)
        out[int(frame["r0"]) + 1 : int(frame["r1"]), int(frame["c0"]) + 1 : int(frame["c1"])] = fill

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored rectangular enclosure by tiling the matching corner legend pattern.

    Input structure:
    - bg is the dominant background color.
    - Several hollow rectangular walls of width 1 appear in the main field; each wall has a unique color and encloses only bg.
    - One grid corner contains a tightly packed legend table made of equal square cells. Each cell has a border color matching one wall color and an interior motif drawn with a second color.

    Rule:
    - Locate the corner legend table and split it into equal square legend cells.
    - For each wall color C, find the legend cell whose border is also C, remove that cell's outer border, and treat the remaining interior motif as a repeating tile.
    - Fill the bg interior of the corresponding wall by tiling that motif from the interior's top-left corner, clipping the tile at the wall bounds.
    - Keep the legend table and all wall borders unchanged.

    Key insight:
    - The wall color is a lookup key into the corner legend, while the legend interior encodes both the fill color and the spatial pattern that must be periodically repeated.
    """
    return _solve(grid)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap <= br0 or br1 + gap <= ar0 or ac1 + gap <= bc0 or bc1 + gap <= ac0)


def _make_legend_cell(side: int, border: int, fill: int, mask: np.ndarray) -> np.ndarray:
    cell = np.full((side, side), border, dtype=int)
    inner = cell[1:-1, 1:-1]
    inner[mask] = fill
    return cell


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0:c1] = color
    grid[r1 - 1, c0:c1] = color
    grid[r0:r1, c0] = color
    grid[r0:r1, c1 - 1] = color


def _place_corner_block(grid: np.ndarray, block: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    bh, bw = block.shape
    if corner == "tl":
        grid[:bh, :bw] = block
        return (0, 0, bh, bw)
    if corner == "tr":
        grid[:bh, w - bw :] = block
        return (0, w - bw, bh, w)
    if corner == "bl":
        grid[h - bh :, :bw] = block
        return (h - bh, 0, h, bw)
    grid[h - bh :, w - bw :] = block
    return (h - bh, w - bw, h, w)


def _expected_output(
    grid: np.ndarray,
    frames: list[tuple[int, tuple[int, int, int, int]]],
    truth_patterns: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for color, box in frames:
        r0, c0, r1, c1 = box
        tile = _tile_pattern(truth_patterns[color], r1 - r0 - 2, c1 - c0 - 2)
        out[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = tile
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 4)
    side = rng.choice([4, 5])
    inner_side = side - 2
    shapes = [(r, n_frames // r) for r in range(1, n_frames + 1) if n_frames % r == 0]
    legend_rows, legend_cols = rng.choice(shapes)
    legend_h = legend_rows * side
    legend_w = legend_cols * side

    min_h = max(18, legend_h + 8)
    min_w = max(18, legend_w + 8)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    bg = rng.randrange(10)

    colors = [c for c in range(10) if c != bg]
    chosen = rng.sample(colors, 2 * n_frames)
    border_colors = chosen[:n_frames]
    fill_colors = chosen[n_frames:]

    pattern_bank = PATTERN_BANK[inner_side]
    mask_indices = list(range(len(pattern_bank)))
    if n_frames <= len(mask_indices):
        chosen_masks = rng.sample(mask_indices, n_frames)
    else:
        chosen_masks = [rng.choice(mask_indices) for _ in range(n_frames)]

    legend = np.empty((legend_h, legend_w), dtype=int)
    truth_patterns: dict[int, np.ndarray] = {}
    frame_specs: list[tuple[int, np.ndarray]] = []
    for idx in range(n_frames):
        border = border_colors[idx]
        fill = fill_colors[idx]
        mask = pattern_bank[chosen_masks[idx]]
        truth_patterns[border] = np.where(mask, fill, bg).astype(int)
        frame_specs.append((border, truth_patterns[border]))

        cell = _make_legend_cell(side, border, fill, mask)
        rr = (idx // legend_cols) * side
        cc = (idx % legend_cols) * side
        legend[rr : rr + side, cc : cc + side] = cell

    grid = np.full((h, w), bg, dtype=int)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    legend_box = _place_corner_block(grid, legend, corner)

    occupied = [legend_box]
    frame_boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    frame_order = border_colors[:]
    rng.shuffle(frame_order)

    for color in frame_order:
        placed = False
        for _ in range(120):
            fh = rng.randint(5, min(10, h - 2))
            fw = rng.randint(5, min(10, w - 2))
            if h - fh - 1 < 1 or w - fw - 1 < 1:
                continue
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            box = (r0, c0, r0 + fh, c0 + fw)
            if any(_boxes_too_close(box, other, gap=1) for other in occupied):
                continue
            occupied.append(box)
            frame_boxes.append((color, box))
            _paint_frame(grid, box, color)
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != bg:
        return None

    parsed = _extract_legend_mapping(grid, bg)
    mapping = parsed[0]
    if set(mapping) != set(border_colors):
        return None

    found_frames = _find_frames(grid, bg)
    if len(found_frames) != n_frames or {int(frame["color"]) for frame in found_frames} != set(border_colors):
        return None

    expected = _expected_output(grid, frame_boxes, truth_patterns)
    actual = _solve(grid)
    if not np.array_equal(expected, actual):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
