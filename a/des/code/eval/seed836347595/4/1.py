import random
from collections import defaultdict

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 300
TRANSFORM_NAMES = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_ud",
    "flip_lr",
    "transpose",
    "anti_transpose",
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "flip_ud":
        return np.flipud(arr)
    if name == "flip_lr":
        return np.fliplr(arr)
    if name == "transpose":
        return arr.T
    if name == "anti_transpose":
        return np.fliplr(np.flipud(arr)).T
    raise ValueError(f"unknown transform: {name}")


def _recolor(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(arr)
    nz = set(int(v) for v in np.unique(arr)) - {BG}
    for color in nz:
        if color not in mapping:
            raise KeyError(color)
        out[arr == color] = mapping[color]
    return out


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in set(int(v) for v in np.unique(g)) - {BG}:
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            perim = 2 * (h + w) - 4
            if len(pts) != perim:
                continue
            pset = set(pts)
            ok = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_edge = rr in (r0, r1) or cc in (c0, c1)
                    if on_edge != ((rr, cc) in pset):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            frames.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "h": h,
                    "w": w,
                    "inner": g[r0 + 1 : r1, c0 + 1 : c1].copy(),
                }
            )
    frames.sort(key=lambda item: (item["r0"], item["c0"]))
    return frames


def _pair_frames(frames: list[dict]) -> list[tuple[dict, dict]]:
    assert frames and len(frames) % 2 == 0, "frame count must be positive and even"
    sizes = {(frame["h"], frame["w"]) for frame in frames}
    assert len(sizes) == 1, "all frames must share one size"

    by_row: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for frame in frames:
        by_row[(frame["r0"], frame["r1"])].append(frame)

    pairs: list[tuple[dict, dict]] = []
    for key in sorted(by_row):
        row_frames = sorted(by_row[key], key=lambda item: item["c0"])
        assert len(row_frames) == 2, "each row span must contain exactly one pair"
        left, right = row_frames
        assert left["c1"] < right["c0"], "paired frames must be horizontally separated"
        pairs.append((left, right))
    return pairs


def _candidate_predictions(
    demo_pairs: list[tuple[np.ndarray, np.ndarray]], test_src: np.ndarray
) -> list[tuple[str, dict[int, int], np.ndarray]]:
    candidates: list[tuple[str, dict[int, int], np.ndarray]] = []
    for name in TRANSFORM_NAMES:
        forward: dict[int, int] = {}
        reverse: dict[int, int] = {}
        valid = True
        for src, dst in demo_pairs:
            transformed = _apply_transform(src, name)
            if transformed.shape != dst.shape:
                valid = False
                break
            for sval, dval in zip(transformed.ravel(), dst.ravel()):
                s = int(sval)
                d = int(dval)
                if s == BG and d == BG:
                    continue
                if s == BG or d == BG:
                    valid = False
                    break
                prev_d = forward.get(s)
                if prev_d is None:
                    forward[s] = d
                elif prev_d != d:
                    valid = False
                    break
                prev_s = reverse.get(d)
                if prev_s is None:
                    reverse[d] = s
                elif prev_s != s:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        try:
            prediction = _recolor(_apply_transform(test_src, name), forward)
        except KeyError:
            continue
        candidates.append((name, forward, prediction))
    return candidates


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    pairs = _pair_frames(frames)

    demo_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    test_pair: tuple[dict, dict] | None = None
    for left, right in pairs:
        left_inner = left["inner"]
        right_inner = right["inner"]
        assert np.any(left_inner != BG), "left frame must contain a source pattern"
        if np.any(right_inner != BG):
            demo_pairs.append((left_inner, right_inner))
        else:
            assert test_pair is None, "there must be exactly one blank answer frame"
            test_pair = (left, right)

    assert demo_pairs, "at least one demonstration pair is required"
    assert test_pair is not None, "a blank answer frame is required"

    candidates = _candidate_predictions(demo_pairs, test_pair[0]["inner"])
    assert candidates, "failed to infer a valid rule"

    unique_predictions: list[np.ndarray] = []
    for _, _, pred in candidates:
        if not any(np.array_equal(pred, seen) for seen in unique_predictions):
            unique_predictions.append(pred)
    assert len(unique_predictions) == 1, "demonstrations must imply one answer"

    out = g.copy()
    answer = unique_predictions[0]
    right = test_pair[1]
    out[right["r0"] + 1 : right["r1"], right["c0"] + 1 : right["c1"]] = answer
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a frame-paired transformation rule from demonstrations, then fill the blank answer frame.

    Input structure:
    - Black background.
    - Several equal-sized hollow square frames. Left frames contain source patterns.
    - Each source frame is paired with a right frame on the same row.
    - Demonstration right frames already contain the transformed pattern.
    - Exactly one right frame is blank; it is the answer slot.

    Rule:
    - Detect rectangular frame pairs by their shared row span and horizontal adjacency.
    - Use the non-blank pairs to infer one consistent transformation: a square dihedral symmetry
      plus a learned color remapping.
    - Apply that inferred rule to the blank pair's left-hand pattern and write the result into
      the blank right-hand frame.
    """
    return _solve_grid(grid)


def _grow_blob(mask: np.ndarray, rng: random.Random, size: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    empties = [(r, c) for r in range(h) for c in range(w) if mask[r, c] == BG]
    if not empties:
        return []
    start = rng.choice(empties)
    cells = {start}
    frontier = {start}
    while len(cells) < size and frontier:
        base = rng.choice(tuple(frontier))
        nbrs = []
        for nr, nc in ((base[0] - 1, base[1]), (base[0] + 1, base[1]), (base[0], base[1] - 1), (base[0], base[1] + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] == BG and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if nbrs:
            cell = rng.choice(nbrs)
            cells.add(cell)
            frontier.add(cell)
        else:
            frontier.remove(base)
    return list(cells)


def _random_pattern(size: int, colors: list[int], rng: random.Random) -> np.ndarray | None:
    arr = np.zeros((size, size), dtype=int)
    for color in colors:
        blob_size = rng.randint(1, min(4, size))
        cells = _grow_blob(arr, rng, blob_size)
        if not cells:
            return None
        for r, c in cells:
            arr[r, c] = color

    extra_blobs = rng.randint(1, max(2, size // 2))
    for _ in range(extra_blobs):
        color = rng.choice(colors)
        blob_size = rng.randint(1, min(3, size))
        cells = _grow_blob(arr, rng, blob_size)
        for r, c in cells:
            arr[r, c] = color

    occ = np.argwhere(arr != BG)
    if len(occ) < len(colors) + 2 or len(occ) > size * size - 2:
        return None
    rows = {int(r) for r, _ in occ}
    cols = {int(c) for _, c in occ}
    if len(rows) < 2 or len(cols) < 2:
        return None
    if set(int(v) for v in np.unique(arr)) - {BG} != set(colors):
        return None
    return arr


def _place_frame(grid: np.ndarray, top: int, left: int, frame_color: int, inner: np.ndarray) -> None:
    size = inner.shape[0] + 2
    grid[top : top + size, left : left + size] = frame_color
    grid[top + 1 : top + size - 1, left + 1 : left + size - 1] = inner


def _compose_rule(
    src: np.ndarray, transform_name: str, color_map: dict[int, int]
) -> np.ndarray:
    return _recolor(_apply_transform(src, transform_name), color_map)


def _sample_rule(
    rng: random.Random,
) -> tuple[str, list[int], dict[int, int]]:
    transform_name = rng.choice(TRANSFORM_NAMES)
    palette_size = rng.choice([2, 3])
    colors = list(rng.sample(COLOR_POOL, palette_size + 2))
    frame_a, frame_b = colors[:2]
    src_colors = colors[2:]

    mode = rng.choice(["same_palette", "disjoint_palette"])
    remaining = [c for c in COLOR_POOL if c not in {frame_a, frame_b} and c not in src_colors]
    if mode == "same_palette":
        dst_colors = src_colors[:]
        rng.shuffle(dst_colors)
    else:
        dst_colors = list(rng.sample(remaining, palette_size))

    color_map = {src: dst for src, dst in zip(src_colors, dst_colors)}
    if transform_name == "id" and all(color_map[c] == c for c in src_colors):
        return _sample_rule(rng)
    return transform_name, [frame_a, frame_b] + src_colors, color_map


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    transform_name, packed_colors, color_map = _sample_rule(rng)
    frame_a, frame_b = packed_colors[:2]
    src_colors = packed_colors[2:]

    inner_size = rng.randint(4, 7)
    frame_size = inner_size + 2
    demo_count = rng.choice([2, 3])
    pair_count = demo_count + 1
    pair_gap = rng.randint(1, 3)

    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 4)
    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    extra_width = rng.randint(0, 6)
    pair_width = frame_size * 2 + pair_gap
    width = left_margin + pair_width + right_margin + extra_width

    row_gaps = [rng.randint(1, 3) for _ in range(pair_count - 1)]
    height = top_margin + bottom_margin + pair_count * frame_size + sum(row_gaps)
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    demo_patterns: list[tuple[np.ndarray, np.ndarray]] = []
    rows: list[int] = []
    cur = top_margin
    for idx in range(pair_count):
        rows.append(cur)
        cur += frame_size
        if idx < pair_count - 1:
            cur += row_gaps[idx]

    max_left = width - right_margin - pair_width
    if max_left < left_margin:
        return None
    x_positions = [rng.randint(left_margin, max_left) for _ in range(pair_count)]
    test_index = rng.randrange(pair_count)

    test_src: np.ndarray | None = None
    test_expected: np.ndarray | None = None

    for idx in range(pair_count):
        src = _random_pattern(inner_size, src_colors, rng)
        if src is None:
            return None
        dst = _compose_rule(src, transform_name, color_map)
        if not np.any(dst != BG):
            return None
        y = rows[idx]
        x = x_positions[idx]
        _place_frame(grid, y, x, frame_a, src)
        _place_frame(expected, y, x, frame_a, src)
        if idx == test_index:
            _place_frame(grid, y, x + frame_size + pair_gap, frame_b, np.zeros_like(dst))
            _place_frame(expected, y, x + frame_size + pair_gap, frame_b, dst)
            test_src = src
            test_expected = dst
        else:
            _place_frame(grid, y, x + frame_size + pair_gap, frame_b, dst)
            _place_frame(expected, y, x + frame_size + pair_gap, frame_b, dst)
            demo_patterns.append((src, dst))

    assert test_src is not None and test_expected is not None

    candidates = _candidate_predictions(demo_patterns, test_src)
    if not candidates:
        return None
    unique_predictions: list[np.ndarray] = []
    for _, _, pred in candidates:
        if not any(np.array_equal(pred, seen) for seen in unique_predictions):
            unique_predictions.append(pred)
    if len(unique_predictions) != 1:
        return None
    if not np.array_equal(unique_predictions[0], test_expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
