import random

import numpy as np


def _row_major_key(cell: tuple[int, int]) -> tuple[int, int]:
    return int(cell[0]), int(cell[1])


def _detect_frame(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if int(coords.shape[0]) < 8:
        return None
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if int(coords.shape[0]) != expected:
        return None

    actual = {(int(r), int(c)) for r, c in coords}
    border = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    if actual != border:
        return None
    return r0, c0, r1, c1


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g) if int(v) != 0]
    frames: list[dict] = []
    pairs: list[dict] = []

    for color in colors:
        coords = np.argwhere(g == color)
        if int(coords.shape[0]) == 2:
            a = (int(coords[0, 0]), int(coords[0, 1]))
            b = (int(coords[1, 0]), int(coords[1, 1]))
            assert a[0] == b[0] or a[1] == b[1], "anchor pair must share a row or a column"
            first = min(a, b, key=_row_major_key)
            second = max(a, b, key=_row_major_key)
            pairs.append({"color": color, "a": first, "b": second, "first": first})
            continue

        frame = _detect_frame(coords)
        assert frame is not None, "every non-black color must be an anchor pair or a hollow rectangle"
        r0, c0, r1, c1 = frame
        frames.append({"color": color, "bbox": (r0, c0, r1, c1)})

    pairs.sort(key=lambda pair: _row_major_key(pair["first"]))
    return frames, pairs


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        lo, hi = sorted((a[1], b[1]))
        return [(row, c) for c in range(lo, hi + 1)]
    col = a[1]
    lo, hi = sorted((a[0], b[0]))
    return [(r, col) for r in range(lo, hi + 1)]


def _pair_hits_frame(pair: dict, frame: dict) -> bool:
    (r0, c0, r1, c1) = frame["bbox"]
    a = pair["a"]
    b = pair["b"]
    if a[0] == b[0]:
        row = a[0]
        lo, hi = sorted((a[1], b[1]))
        return r0 < row < r1 and lo < c0 and hi > c1
    col = a[1]
    lo, hi = sorted((a[0], b[0]))
    return c0 < col < c1 and lo < r0 and hi > r1


def _apply_rule(grid: np.ndarray, frames: list[dict], pairs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    border_mask = np.zeros((h, w), dtype=bool)
    interior_owner = np.full((h, w), -1, dtype=int)

    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        border_mask[r0 : r1 + 1, c0] = True
        border_mask[r0 : r1 + 1, c1] = True
        border_mask[r0, c0 : c1 + 1] = True
        border_mask[r1, c0 : c1 + 1] = True
        interior_owner[r0 + 1 : r1, c0 + 1 : c1] = idx

    filled_frames: set[int] = set()
    for pair in pairs:
        color = int(pair["color"])
        for idx, frame in enumerate(frames):
            if idx in filled_frames:
                continue
            if _pair_hits_frame(pair, frame):
                r0, c0, r1, c1 = frame["bbox"]
                out[r0 + 1 : r1, c0 + 1 : c1] = color
                filled_frames.add(idx)

        for r, c in _segment_cells(pair["a"], pair["b"]):
            if border_mask[r, c]:
                continue
            owner = int(interior_owner[r, c])
            if owner != -1 and owner in filled_frames and int(out[r, c]) != color:
                continue
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored anchor pair and fill every hollow rectangle that the line truly passes through.

    Input structure:
    - The background is black.
    - Several colors form one-pixel-thick hollow axis-aligned rectangular frames; each frame color is unique and its interior is black.
    - Other colors appear exactly twice as anchor pixels, and each anchor pair shares one row or one column.

    Rule:
    - For every anchor color, draw a straight horizontal or vertical segment connecting its two anchor pixels.
    - A frame is considered pierced only when that segment crosses one border side, enters the interior, and exits through the opposite side.
    - Fill the whole interior of every pierced frame with the first line color that reaches it when anchor pairs are scanned left-to-right, top-to-bottom; keep untouched interiors black.
    - Preserve the anchor lines themselves, but never overwrite the colored frame borders.

    Key insight:
    - The anchor pixels imply hidden straight paths, and the real task is to infer which enclosed rectangles those paths genuinely pass through rather than merely touch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames, pairs = _parse_scene(g)
    return _apply_rule(g, frames, pairs)


# --------------- generator ---------------

MIN_SIZE = 14
MAX_SIZE = 30
MIN_FRAMES = 2
MAX_FRAMES = 4
MAX_PAIRS = 4
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300
SELECT_ATTEMPTS = 400
ALL_COLORS = tuple(range(1, 10))


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _sample_frame_size(shape: tuple[int, int], orientation: str, ensure_multi: bool) -> tuple[int, int] | None:
    h, w = shape
    max_h = min(8, h - 5)
    max_w = min(10, w - 5)
    if max_h < 3 or max_w < 3:
        return None

    min_h = 4 if orientation == "horizontal" and ensure_multi else 3
    min_w = 4 if orientation == "vertical" and ensure_multi else 3
    min_h = min(min_h, max_h)
    min_w = min(min_w, max_w)
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _place_frames(shape: tuple[int, int], orientation: str, count: int) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    frames: list[tuple[int, int, int, int]] = []
    for idx in range(count):
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            size = _sample_frame_size(shape, orientation, ensure_multi=(idx == 0))
            if size is None:
                return None
            fh, fw = size
            if h - fh - 3 < 2 or w - fw - 3 < 2:
                continue
            r0 = random.randint(2, h - fh - 3)
            c0 = random.randint(2, w - fw - 3)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_overlap(box, other, pad=1) for other in frames):
                continue
            frames.append(box)
            placed = True
            break
        if not placed:
            return None
    return frames


def _coordinate_sets(
    frames: list[tuple[int, int, int, int]], axis_len: int, orientation: str
) -> tuple[dict[int, set[int]], list[int], list[int]]:
    hits: dict[int, set[int]] = {}
    border_coords: set[int] = set()
    for coord in range(1, axis_len - 1):
        hits[coord] = set()

    for idx, (r0, c0, r1, c1) in enumerate(frames):
        if orientation == "horizontal":
            border_coords.update((r0, r1))
            for row in range(r0 + 1, r1):
                if 1 <= row < axis_len - 1:
                    hits[row].add(idx)
        else:
            border_coords.update((c0, c1))
            for col in range(c0 + 1, c1):
                if 1 <= col < axis_len - 1:
                    hits[col].add(idx)

    clean_hit = [coord for coord, owners in hits.items() if owners and coord not in border_coords]
    clean_free = [coord for coord, owners in hits.items() if not owners and coord not in border_coords]
    all_hit = [coord for coord, owners in hits.items() if owners]
    all_free = [coord for coord, owners in hits.items() if not owners]
    return hits, (clean_hit or all_hit), (clean_free or all_free)


def _select_line_coords(
    frames: list[tuple[int, int, int, int]], shape: tuple[int, int], orientation: str
) -> list[int] | None:
    axis_len = shape[0] if orientation == "horizontal" else shape[1]
    hits, hit_coords, free_coords = _coordinate_sets(frames, axis_len, orientation)
    if not hit_coords:
        return None

    max_pairs = min(MAX_PAIRS, 9 - len(frames), len(hit_coords) + len(free_coords))
    if max_pairs < 1:
        return None

    multi_candidates = {
        idx: [coord for coord in hit_coords if idx in hits[coord]]
        for idx in range(len(frames))
    }
    multi_candidates = {idx: coords for idx, coords in multi_candidates.items() if len(coords) >= 2}

    for _ in range(SELECT_ATTEMPTS):
        want_priority = bool(multi_candidates) and random.random() < 0.6
        want_untouched = len(frames) > 1 and random.random() < 0.7
        want_free = bool(free_coords) and random.random() < 0.5
        min_pairs = 2 if max_pairs >= 2 else 1
        target = random.randint(min_pairs, max_pairs)

        coords: list[int] = []
        used: set[int] = set()

        if want_priority:
            frame_id = random.choice(list(multi_candidates))
            chosen = random.sample(multi_candidates[frame_id], 2)
            coords.extend(chosen)
            used.update(chosen)

        if not coords:
            first = random.choice(hit_coords)
            coords.append(first)
            used.add(first)

        target_hits = max(1, target - (1 if want_free else 0))
        remaining_hits = [coord for coord in hit_coords if coord not in used]
        random.shuffle(remaining_hits)
        while sum(bool(hits[coord]) for coord in coords) < target_hits and remaining_hits:
            coord = remaining_hits.pop()
            coords.append(coord)
            used.add(coord)

        if want_free and len(coords) < target:
            free_pool = [coord for coord in free_coords if coord not in used] or free_coords
            coord = random.choice(free_pool)
            if coord not in used:
                coords.append(coord)
                used.add(coord)

        pool = [coord for coord in hit_coords + free_coords if coord not in used]
        random.shuffle(pool)
        while len(coords) < target and pool:
            coord = pool.pop()
            coords.append(coord)
            used.add(coord)

        coords = sorted(coords)
        hit_frames: set[int] = set()
        hit_counts = {idx: 0 for idx in range(len(frames))}
        for coord in coords:
            for frame_id in hits[coord]:
                hit_frames.add(frame_id)
                hit_counts[frame_id] += 1

        if not hit_frames:
            continue
        if want_priority and max(hit_counts.values()) < 2:
            continue
        if want_untouched and len(hit_frames) == len(frames):
            continue
        return coords

    return None


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _build_input(
    shape: tuple[int, int], frames: list[tuple[int, int, int, int]], coords: list[int], orientation: str
) -> np.ndarray:
    h, w = shape
    total_colors = len(frames) + len(coords)
    palette = random.sample(list(ALL_COLORS), total_colors)
    frame_colors = palette[: len(frames)]
    pair_colors = palette[len(frames) :]

    grid = np.zeros((h, w), dtype=int)
    for box, color in zip(frames, frame_colors, strict=False):
        _draw_frame(grid, box, int(color))

    for coord, color in zip(coords, pair_colors, strict=False):
        if orientation == "horizontal":
            grid[coord, 0] = int(color)
            grid[coord, w - 1] = int(color)
        else:
            grid[0, coord] = int(color)
            grid[h - 1, coord] = int(color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        orientation = random.choice(["horizontal", "vertical"])
        shape = (random.randint(MIN_SIZE, MAX_SIZE), random.randint(MIN_SIZE, MAX_SIZE))
        frame_count = random.randint(MIN_FRAMES, MAX_FRAMES)
        frames = _place_frames(shape, orientation, frame_count)
        if frames is None:
            continue

        coords = _select_line_coords(frames, shape, orientation)
        if coords is None:
            continue

        inp = _build_input(shape, frames, coords, orientation)
        try:
            out = main(inp)
        except Exception:
            continue

        if out.shape != inp.shape:
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(inp, out):
            continue
        return inp

    raise RuntimeError("failed to generate a valid ARC instance")
