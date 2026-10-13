from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
_RNG = random.Random()
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _row_major_unique(values: np.ndarray, exclude: set[int]) -> list[int]:
    seen = set(exclude)
    ordered: list[int] = []
    for value in values.flat:
        color = int(value)
        if color in seen:
            continue
        seen.add(color)
        ordered.append(color)
    return ordered


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    patch = np.zeros((h, w), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    patch[rr, cc] = True

    expected = np.zeros_like(patch)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(patch, expected)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in _row_major_unique(grid, {bg}):
        for coords in _components(grid == color):
            if not _is_rect_frame(coords):
                continue
            frames.append({"color": color, "bbox": _bbox(coords), "coords": coords})
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each fence's scattered interior colors into filled bounding boxes, with minority colors winning overlaps.

    Input structure:
    - The background is black.
    - Several disjoint one-pixel-wide grey rectangular fences appear on the grid.
    - Each fence encloses a black interior containing scattered pixels from two or three non-grey colors.
    - Outside every fence, the grid stays black.

    Rule:
    - Treat every rectangular fence as an independent region and keep the fence pixels unchanged.
    - Inside one fence, group the scattered pixels by color and compute each color's minimal axis-aligned bounding box.
    - Replace the original scattered pixels by filling every such bounding box solidly with its color.
    - When multiple filled boxes overlap, the color with fewer original pixels inside that fence takes priority on the shared cells.

    Key insight:
    - The colored fragments are not moved individually; each color first expands to its own bounding rectangle, and overlap conflicts are resolved by minority-count priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    out = g.copy()
    frames = _find_frames(g, bg)

    for frame in frames:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = _row_major_unique(inner, {bg, frame_color})

        painted = np.full(inner.shape, bg, dtype=int)
        best_count = np.full(inner.shape, np.iinfo(np.int32).max, dtype=int)
        best_order = np.full(inner.shape, len(colors), dtype=int)

        for order, color in enumerate(colors):
            pts = np.argwhere(inner == color)
            if pts.size == 0:
                continue
            rr0, cc0 = pts.min(axis=0)
            rr1, cc1 = pts.max(axis=0)
            count = int(pts.shape[0])

            count_view = best_count[rr0 : rr1 + 1, cc0 : cc1 + 1]
            order_view = best_order[rr0 : rr1 + 1, cc0 : cc1 + 1]
            paint_view = painted[rr0 : rr1 + 1, cc0 : cc1 + 1]

            update = count < count_view
            update |= (count == count_view) & (order < order_view)

            count_view[update] = count
            order_view[update] = order
            paint_view[update] = color

        out[r0 + 1 : r1, c0 + 1 : c1] = painted

    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _random_rect(
    inner_h: int,
    inner_w: int,
    min_area: int,
    rng: random.Random,
    max_side: int = 6,
) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        h = rng.randint(2, min(inner_h, max_side))
        w = rng.randint(2, min(inner_w, max_side))
        if h * w < min_area:
            continue
        r0 = rng.randint(0, inner_h - h)
        c0 = rng.randint(0, inner_w - w)
        return (r0, c0, r0 + h - 1, c0 + w - 1)
    return None


def _diag_points(rect: tuple[int, int, int, int], rng: random.Random) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if rng.random() < 0.5:
        return [(r0, c0), (r1, c1)]
    return [(r0, c1), (r1, c0)]


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _assign_points(
    rects: list[tuple[int, int, int, int]],
    counts: list[int],
    mandatory: list[list[tuple[int, int]]],
    rng: random.Random,
) -> list[list[tuple[int, int]]] | None:
    rect_sets = [_rect_cells(rect) for rect in rects]
    memberships: dict[tuple[int, int], int] = {}
    for cells in rect_sets:
        for cell in cells:
            memberships[cell] = memberships.get(cell, 0) + 1

    order = sorted(range(len(rects)), key=lambda idx: (-counts[idx], len(rect_sets[idx])))
    occupied: set[tuple[int, int]] = set()
    assigned: list[list[tuple[int, int]]] = [[] for _ in rects]

    for idx in order:
        must = list(dict.fromkeys(mandatory[idx]))
        if len(must) < 2 or any(cell in occupied for cell in must):
            return None

        exclusive = [
            cell
            for cell in rect_sets[idx]
            if cell not in occupied and cell not in must and memberships[cell] == 1
        ]
        shared = [
            cell
            for cell in rect_sets[idx]
            if cell not in occupied and cell not in must and memberships[cell] > 1
        ]
        rng.shuffle(exclusive)
        rng.shuffle(shared)

        need = counts[idx] - len(must)
        if need < 0 or len(exclusive) + len(shared) < need:
            return None

        chosen = must + exclusive[:need]
        if len(chosen) < counts[idx]:
            chosen.extend(shared[: counts[idx] - len(chosen)])
        if len(chosen) != counts[idx]:
            return None

        assigned[idx] = chosen
        occupied.update(chosen)

    return assigned


def _render_region(
    inner_h: int,
    inner_w: int,
    colors: list[int],
    rects: list[tuple[int, int, int, int]],
    counts: list[int],
    mandatory: list[list[tuple[int, int]]],
    rng: random.Random,
) -> np.ndarray | None:
    points = _assign_points(rects, counts, mandatory, rng)
    if points is None:
        return None

    inner = np.full((inner_h, inner_w), BG, dtype=int)
    for color, cells in zip(colors, points):
        for r, c in cells:
            if inner[r, c] != BG:
                return None
            inner[r, c] = color
    return inner


def _generate_disjoint_region(
    inner_h: int,
    inner_w: int,
    num_colors: int,
    rng: random.Random,
) -> tuple[np.ndarray, bool] | None:
    colors = rng.sample(PALETTE, num_colors)
    counts = rng.sample([2, 3, 4, 5, 6], num_colors)

    rects: list[tuple[int, int, int, int]] = []
    mandatory: list[list[tuple[int, int]]] = []
    for count in counts:
        placed = None
        for _ in range(300):
            rect = _random_rect(inner_h, inner_w, count, rng)
            if rect is None:
                continue
            if any(_rects_overlap(rect, other, gap=1) for other in rects):
                continue
            placed = rect
            break
        if placed is None:
            return None
        rects.append(placed)
        mandatory.append(_diag_points(placed, rng))

    inner = _render_region(inner_h, inner_w, colors, rects, counts, mandatory, rng)
    if inner is None:
        return None
    return inner, False


def _generate_overlap_region(
    inner_h: int,
    inner_w: int,
    num_colors: int,
    rng: random.Random,
) -> tuple[np.ndarray, bool] | None:
    if inner_h < 4 or inner_w < 4:
        return None

    colors = rng.sample(PALETTE, num_colors)
    counts = rng.sample([2, 3, 4, 5, 6], num_colors)

    rows = sorted(rng.sample(range(inner_h), 4))
    cols = sorted(rng.sample(range(inner_w), 4))
    rects = [
        (rows[0], cols[0], rows[2], cols[2]),
        (rows[1], cols[1], rows[3], cols[3]),
    ]
    mandatory = [
        [(rows[0], cols[0]), (rows[2], cols[2])],
        [(rows[1], cols[3]), (rows[3], cols[1])],
    ]

    if num_colors == 3:
        extra_rect = None
        for _ in range(300):
            rect = _random_rect(inner_h, inner_w, counts[2], rng)
            if rect is None:
                continue
            if any(_rects_overlap(rect, other, gap=1) for other in rects):
                continue
            extra_rect = rect
            break
        if extra_rect is None:
            return None
        rects.append(extra_rect)
        mandatory.append(_diag_points(extra_rect, rng))

    inner = _render_region(inner_h, inner_w, colors, rects, counts, mandatory, rng)
    if inner is None:
        return None
    return inner, True


def _make_frame_patch(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    patch[1:-1, 1:-1] = inner
    return patch


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph + 2 > h or pw + 2 > w:
            return None
        for _ in range(400):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_rects_overlap(rect, other, gap=1) for other in placed):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            placed.append(rect)
            break
        else:
            return None

    return grid


def _frame_has_overlap(grid: np.ndarray, frame: dict[str, object], bg: int) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = _row_major_unique(inner, {bg, int(frame["color"])})
    bboxes: list[tuple[int, int, int, int]] = []
    counts: list[int] = []
    for color in colors:
        pts = np.argwhere(inner == color)
        if not (2 <= pts.shape[0] <= 6):
            return False
        bboxes.append(_bbox(pts))
        counts.append(int(pts.shape[0]))

    overlap = False
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _rects_overlap(bboxes[i], bboxes[j]):
                if counts[i] == counts[j]:
                    return False
                overlap = True
    return overlap


def _valid_generated_grid(grid: np.ndarray, expected_frames: int, require_overlap: bool) -> bool:
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    bg = _background_color(grid)
    if bg != BG:
        return False

    frames = _find_frames(grid, bg)
    if len(frames) != expected_frames:
        return False

    seen_overlap = False
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = _row_major_unique(inner, {bg, int(frame["color"])})
        if len(colors) not in {2, 3}:
            return False
        for color in colors:
            count = int(np.count_nonzero(inner == color))
            if not (2 <= count <= 6):
                return False
        seen_overlap = seen_overlap or _frame_has_overlap(grid, frame, bg)

    if require_overlap != seen_overlap:
        return False

    out = main(grid)
    return out.shape == grid.shape


def _region_shape(mode: str, num_colors: int, rng: random.Random) -> tuple[int, int]:
    if mode == "overlap" and num_colors == 3:
        low, high = 8, 12
    elif mode == "disjoint" and num_colors == 3:
        low, high = 8, 11
    else:
        low, high = 6, 10
    return rng.randint(low, high), rng.randint(low, high)


def _gen_attempt() -> np.ndarray | None:
    n_frames = _RNG.choices([1, 2, 3, 4], weights=[1, 3, 3, 1], k=1)[0]
    require_overlap = bool(_RNG.getrandbits(1))

    patches: list[np.ndarray] = []
    overlap_count = 0

    for idx in range(n_frames):
        num_colors = 2 if _RNG.random() < 0.5 else 3
        mode = "overlap" if require_overlap and (_RNG.random() < 0.65) else "disjoint"
        if require_overlap and idx == n_frames - 1 and overlap_count == 0:
            mode = "overlap"

        region = None
        for _ in range(300):
            inner_h, inner_w = _region_shape(mode, num_colors, _RNG)
            if mode == "overlap":
                region = _generate_overlap_region(inner_h, inner_w, num_colors, _RNG)
            else:
                region = _generate_disjoint_region(inner_h, inner_w, num_colors, _RNG)
            if region is not None:
                break
        if region is None:
            return None

        inner, has_overlap = region
        overlap_count += int(has_overlap)
        patches.append(_make_frame_patch(inner))

    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    side_hint = int(np.ceil(np.sqrt(total_area * 1.8)))
    min_h = max(14, max_h + 2, side_hint - 2)
    min_w = max(14, max_w + 2, side_hint - 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(300):
        h = _RNG.randint(min_h, 30)
        w = _RNG.randint(min_w, 30)
        grid = _place_patches((h, w), patches, _RNG)
        if grid is None:
            continue
        if _valid_generated_grid(grid, n_frames, require_overlap):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-bbox ARC input")
