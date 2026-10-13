import math
from collections import Counter, deque

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "size": len(coords),
                }
            )
    return comps


def _read_pattern(
    inner: np.ndarray, cell_size: int, bg: int, frame_color: int
) -> tuple[np.ndarray, int] | tuple[None, None]:
    pattern = np.full((2, 2), bg, dtype=int)
    filled = 0
    for rr in range(2):
        for cc in range(2):
            cell = inner[
                rr * cell_size : (rr + 1) * cell_size,
                cc * cell_size : (cc + 1) * cell_size,
            ]
            vals = np.unique(cell)
            if len(vals) != 1:
                return None, None
            value = int(vals[0])
            if value == int(frame_color):
                return None, None
            pattern[rr, cc] = value
            if value != int(bg):
                filled += 1
    return pattern, filled


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]

    for color in colors:
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = comp["bbox"]
            side_h = int(r1 - r0 + 1)
            side_w = int(c1 - c0 + 1)
            if side_h != side_w or side_h < 4:
                continue
            if comp["size"] != 4 * side_h - 4:
                continue
            inner_side = side_h - 2
            if inner_side < 2 or inner_side % 2 != 0:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch[0, :] == int(color)):
                continue
            if not np.all(patch[-1, :] == int(color)):
                continue
            if not np.all(patch[:, 0] == int(color)):
                continue
            if not np.all(patch[:, -1] == int(color)):
                continue
            if np.any(patch[1:-1, 1:-1] == int(color)):
                continue

            cell_size = inner_side // 2
            pattern, filled = _read_pattern(
                patch[1:-1, 1:-1], cell_size, bg, int(color)
            )
            if pattern is None or filled is None or not (1 <= filled <= 4):
                continue

            frames.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "frame_color": int(color),
                    "outer_size": int(side_h),
                    "cell_size": int(cell_size),
                    "pattern": pattern,
                    "filled": int(filled),
                }
            )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every hollow frame's 2x2 interior pattern from the single full template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _detect_frames(g, bg)
    assert frames, "no valid frames found"

    by_size: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        key = (int(frame["outer_size"]), int(frame["cell_size"]))
        by_size.setdefault(key, []).append(frame)

    chosen: list[dict] | None = None
    template: dict | None = None
    best_score: tuple[int, int] | None = None
    for key, group in by_size.items():
        templates = [frame for frame in group if int(frame["filled"]) == 4]
        if len(group) < 2 or len(templates) != 1:
            continue
        score = (len(group), key[0])
        if best_score is None or score > best_score:
            best_score = score
            chosen = group
            template = templates[0]

    assert chosen is not None and template is not None, "could not isolate template set"

    template_pattern = np.array(template["pattern"], dtype=int)
    out = g.copy()
    for frame in chosen:
        r0, c0, _r1, _c1 = frame["bbox"]
        cell_size = int(frame["cell_size"])
        for rr in range(2):
            for cc in range(2):
                out[
                    r0 + 1 + rr * cell_size : r0 + 1 + (rr + 1) * cell_size,
                    c0 + 1 + cc * cell_size : c0 + 1 + (cc + 1) * cell_size,
                ] = int(template_pattern[rr, cc])
    return out


def _make_frame(
    frame_color: int,
    pattern: np.ndarray,
    present: np.ndarray,
    cell_size: int,
    bg: int,
) -> np.ndarray:
    outer = 2 * cell_size + 2
    frame = np.full((outer, outer), int(frame_color), dtype=int)
    frame[1:-1, 1:-1] = int(bg)
    for rr in range(2):
        for cc in range(2):
            if present[rr, cc]:
                frame[
                    1 + rr * cell_size : 1 + (rr + 1) * cell_size,
                    1 + cc * cell_size : 1 + (cc + 1) * cell_size,
                ] = int(pattern[rr, cc])
    return frame


def _layout_choices(n_frames: int, outer: int) -> list[tuple[int, int, int, int, int, int]]:
    choices: list[tuple[int, int, int, int, int, int]] = []
    for rows in range(1, n_frames + 1):
        cols = int(math.ceil(n_frames / rows))
        for gap_r in range(1, 4):
            for gap_c in range(1, 4):
                for jitter_r in range(0, 3):
                    for jitter_c in range(0, 3):
                        h = 2 + rows * outer + (rows - 1) * gap_r + rows * jitter_r
                        w = 2 + cols * outer + (cols - 1) * gap_c + cols * jitter_c
                        if 14 <= h <= 30 and 14 <= w <= 30:
                            choices.append((rows, cols, gap_r, gap_c, jitter_r, jitter_c))
    return choices


def _place_positions(
    rng: np.random.Generator,
    n_frames: int,
    outer: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | tuple[None, None]:
    choices = _layout_choices(n_frames, outer)
    if not choices:
        return None, None

    rows, cols, gap_r, gap_c, jitter_r, jitter_c = choices[
        int(rng.integers(len(choices)))
    ]
    h = 2 + rows * outer + (rows - 1) * gap_r + rows * jitter_r
    w = 2 + cols * outer + (cols - 1) * gap_c + cols * jitter_c

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    slots = slots[:n_frames]

    positions: list[tuple[int, int]] = []
    pitch_r = outer + gap_r + jitter_r
    pitch_c = outer + gap_c + jitter_c
    for sr, sc in slots:
        r_jit = int(rng.integers(jitter_r + 1))
        c_jit = int(rng.integers(jitter_c + 1))
        r0 = 1 + sr * pitch_r + r_jit
        c0 = 1 + sc * pitch_c + c_jit
        positions.append((int(r0), int(c0)))

    return positions, (int(h), int(w))


def _missing_mask(rng: np.random.Generator) -> np.ndarray:
    mask = np.ones(4, dtype=bool)
    missing = int(rng.integers(1, 4))
    off = rng.choice(4, size=missing, replace=False)
    mask[off] = False
    return mask.reshape(2, 2)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = 0
    n_frames = int(rng.integers(3, 6))
    max_cell = 4 if n_frames <= 4 else 3
    cell_size = int(rng.integers(1, max_cell + 1))
    outer = 2 * cell_size + 2

    positions, shape = _place_positions(rng, n_frames, outer)
    if positions is None or shape is None:
        return None
    h, w = shape

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    frame_colors = color_pool[:n_frames]
    pattern_colors = color_pool[n_frames : n_frames + 4]
    if len(pattern_colors) < 4:
        return None

    pattern = np.array(rng.permutation(pattern_colors), dtype=int).reshape(2, 2)
    template_idx = int(rng.integers(n_frames))

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    for idx, (r0, c0) in enumerate(positions):
        present = np.ones((2, 2), dtype=bool) if idx == template_idx else _missing_mask(rng)
        frame_in = _make_frame(frame_colors[idx], pattern, present, cell_size, bg)
        frame_out = _make_frame(
            frame_colors[idx], pattern, np.ones((2, 2), dtype=bool), cell_size, bg
        )
        inp[r0 : r0 + outer, c0 : c0 + outer] = frame_in
        out[r0 : r0 + outer, c0 : c0 + outer] = frame_out

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
