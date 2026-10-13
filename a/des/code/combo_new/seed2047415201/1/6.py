import random

import numpy as np


BLACK = 0
RED = 2
MAGENTA = 6

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _rect_perimeter_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _find_magenta_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(g == MAGENTA):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        pts = set(comp)
        perimeter = set(_rect_perimeter_coords(r0, c0, r1, c1))
        if pts == perimeter:
            frames.append((r0, c0, r1, c1))
    return frames


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _draw_expanded_perimeter(out: np.ndarray, box: tuple[int, int, int, int], layer: int) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = box
    top = r0 - layer
    bottom = r1 + layer
    left = c0 - layer
    right = c1 + layer

    col0 = max(0, left)
    col1 = min(w - 1, right)
    if col0 <= col1:
        if 0 <= top < h:
            out[top, col0 : col1 + 1] = MAGENTA
        if 0 <= bottom < h:
            out[bottom, col0 : col1 + 1] = MAGENTA

    row0 = max(0, top)
    row1 = min(h - 1, bottom)
    if row0 <= row1:
        if 0 <= left < w:
            out[row0 : row1 + 1, left] = MAGENTA
        if 0 <= right < w:
            out[row0 : row1 + 1, right] = MAGENTA


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each magenta rectangular frame by the number of red markers inside it.

    Input structure:
    - Black background with one or more one-pixel-thick Magenta rectangular closed frames.
    - Each frame encloses a Black interior containing scattered Red marker pixels.
    - Frames do not overlap; cells outside the original frames are Black.

    Rule:
    - For each Magenta frame, count the Red markers strictly inside that frame.
    - Erase all Red markers to Black.
    - Around each frame, draw that many outward one-pixel-thick concentric rectangular Magenta rings, clipping any out-of-bounds part.
    - Draw frames in descending rectangular area order; later rings overwrite earlier ones where expanded regions overlap.

    Key insight: The Red marker locations are irrelevant; their count encodes the number of geometric expansion layers.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_magenta_frames(g)
    assert frames, "expected at least one magenta rectangular frame"

    tasks = []
    for box in frames:
        r0, c0, r1, c1 = box
        n_layers = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == RED))
        tasks.append((_box_area(box), box, n_layers))

    out[out == RED] = BLACK

    for _area, box, n_layers in sorted(tasks, key=lambda item: (-item[0], item[1])):
        for layer in range(1, n_layers + 1):
            _draw_expanded_perimeter(out, box, layer)

    return out


def _draw_frame(g: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    g[r0, c0 : c1 + 1] = MAGENTA
    g[r1, c0 : c1 + 1] = MAGENTA
    g[r0 : r1 + 1, c0] = MAGENTA
    g[r0 : r1 + 1, c1] = MAGENTA


def _legal_box(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    return not bool(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _all_legal_positions(occupied: np.ndarray, frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(0, h - frame_h + 1):
        r1 = r0 + frame_h - 1
        for c0 in range(0, w - frame_w + 1):
            c1 = c0 + frame_w - 1
            if _legal_box(occupied, r0, c0, r1, c1):
                positions.append((r0, c0))
    return positions


def _expanded_clips(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    layers: int,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    return r0 - layers < 0 or c0 - layers < 0 or r1 + layers >= h or c1 + layers >= w


def _near_overlap_positions(
    frames: list[dict],
    occupied: np.ndarray,
    frame_h: int,
    frame_w: int,
    layers: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []

    for item in frames:
        ar0, ac0, ar1, ac1 = item["box"]
        old_layers = int(item["layers"])
        min_gap = max(old_layers, layers)
        max_gap = old_layers + layers - 1
        if min_gap > max_gap:
            continue

        for gap in range(min_gap, max_gap + 1):
            row_start = max(0, ar0 - frame_h + 1)
            row_stop = min(h - frame_h, ar1)
            for r0 in range(row_start, row_stop + 1):
                for c0 in (ac1 + gap + 1, ac0 - gap - frame_w):
                    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
                    if _legal_box(occupied, r0, c0, r1, c1):
                        positions.append((r0, c0))

            col_start = max(0, ac0 - frame_w + 1)
            col_stop = min(w - frame_w, ac1)
            for c0 in range(col_start, col_stop + 1):
                for r0 in (ar1 + gap + 1, ar0 - gap - frame_h):
                    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
                    if _legal_box(occupied, r0, c0, r1, c1):
                        positions.append((r0, c0))

    return positions


def _sample_frame_size(h: int, w: int, layers: int) -> tuple[int, int] | None:
    max_h = min(9, h)
    max_w = min(9, w)
    choices = []
    for frame_h in range(3, max_h + 1):
        for frame_w in range(3, max_w + 1):
            interior_area = (frame_h - 2) * (frame_w - 2)
            if interior_area >= layers:
                choices.append((frame_h, frame_w))
    if not choices:
        return None
    return random.choice(choices)


def _add_markers(g: np.ndarray, box: tuple[int, int, int, int], count: int) -> None:
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for r, c in random.sample(cells, count):
        g[r, c] = RED


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 30)
    w = random.randint(10, 30)
    g = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []

    target_count = random.choices([1, 2, 3, 4, 5], weights=[1, 3, 3, 2, 1], k=1)[0]
    for i in range(target_count):
        placed = False
        for _ in range(100):
            layers = random.randint(1, 4)
            size = _sample_frame_size(h, w, layers)
            if size is None:
                continue
            frame_h, frame_w = size

            positions: list[tuple[int, int]] = []
            if i > 0 and random.random() < 0.45:
                positions = _near_overlap_positions(frames, occupied, frame_h, frame_w, layers)

            if not positions:
                positions = _all_legal_positions(occupied, frame_h, frame_w)
                if positions and random.random() < 0.35:
                    clipped = []
                    for r0, c0 in positions:
                        r1 = r0 + frame_h - 1
                        c1 = c0 + frame_w - 1
                        if _expanded_clips(r0, c0, r1, c1, layers, (h, w)):
                            clipped.append((r0, c0))
                    if clipped:
                        positions = clipped

            if not positions:
                continue

            r0, c0 = random.choice(positions)
            r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
            if not _legal_box(occupied, r0, c0, r1, c1):
                continue

            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            frames.append({"box": (r0, c0, r1, c1), "layers": layers})
            placed = True
            break

        if not placed:
            break

    if not frames:
        return None

    for item in frames:
        box = item["box"]
        _draw_frame(g, box)
        _add_markers(g, box, int(item["layers"]))

    if len(_find_magenta_frames(g)) != len(frames):
        return None

    try:
        out = main(g)
    except AssertionError:
        return None
    if out.shape != g.shape or max(out.shape) > 30:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to place non-overlapping magenta frames")
