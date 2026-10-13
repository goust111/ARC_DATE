import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
STRIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    bg = _most_common_color(grid)
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(coords)
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            if inner_h <= 0 or inner_w <= 0:
                continue
            if r0 == 0 or c0 == 0:
                continue
            if len(coords) != 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
                continue

            coord_set = set(coords)
            valid = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    on_border = rr in (r0, r1) or cc in (c0, c1)
                    if on_border != ((rr, cc) in coord_set):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if np.any(interior != bg):
                continue

            top_strip = grid[r0 - 1, c0 + 1 : c1]
            left_strip = grid[r0 + 1 : r1, c0 - 1]
            if top_strip.size != inner_w or left_strip.size != inner_h:
                continue
            if np.any(top_strip == bg) or np.any(left_strip == bg):
                continue
            if np.any(top_strip == color) or np.any(left_strip == color):
                continue

            frames.append(
                {
                    "frame_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "top_strip": top_strip.astype(int).copy(),
                    "left_strip": left_strip.astype(int).copy(),
                }
            )
    frames.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))
    return frames


def _paint_frame(out: np.ndarray, frame: dict[str, object], bg: int) -> None:
    r0, c0, r1, c1 = [int(v) for v in frame["bbox"]]
    top_strip = np.asarray(frame["top_strip"], dtype=int)
    left_strip = np.asarray(frame["left_strip"], dtype=int)
    for idx, row_color in enumerate(left_strip):
        row = np.where(top_strip == int(row_color), top_strip, int(bg))
        out[r0 + 1 + idx, c0 + 1 : c1] = row


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each frame interior from the equality intersection of its top and left control stripes."""
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g)
    assert frames, "expected at least one framed stripe controller"

    out = g.copy()
    for frame in frames:
        _paint_frame(out, frame, bg)
    return out


def _random_partition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    bounds = [0, *cuts, total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _make_segmented_strip(
    length: int,
    allowed: list[int],
    required: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    needed = list(dict.fromkeys(int(c) for c in required))
    max_parts = min(length, len(needed) + 2)
    part_count = int(rng.integers(len(needed), max_parts + 1))
    part_lengths = _random_partition(length, part_count, rng)

    colors = needed[:]
    while len(colors) < part_count:
        colors.append(int(rng.choice(allowed)))
    rng.shuffle(colors)

    for i in range(1, len(colors)):
        if colors[i] == colors[i - 1]:
            choices = [c for c in allowed if c != colors[i - 1]]
            colors[i] = int(rng.choice(choices))

    strip = np.empty(length, dtype=int)
    start = 0
    for seg_len, color in zip(part_lengths, colors):
        strip[start : start + seg_len] = int(color)
        start += seg_len
    return strip


def _sample_stripes(inner_h: int, inner_w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    palette_size = int(rng.integers(3, 5))
    palette = [int(c) for c in rng.choice(STRIPE_COLORS, size=palette_size, replace=False)]

    if palette_size == 3:
        shared = [palette[0]]
        top_only = [palette[1]]
        left_only = [palette[2]]
    else:
        shared = palette[:2]
        top_only = [palette[2]]
        left_only = [palette[3]]

    top_allowed = shared + top_only
    left_allowed = shared + left_only
    top_required = shared + top_only
    left_required = shared + left_only

    top_strip = _make_segmented_strip(inner_w, top_allowed, top_required, rng)
    left_strip = _make_segmented_strip(inner_h, left_allowed, left_required, rng)
    return top_strip, left_strip


def _expected_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        top_strip = np.asarray(spec["top_strip"], dtype=int)
        left_strip = np.asarray(spec["left_strip"], dtype=int)
        for rr in range(inner_h):
            out[r0 + 1 + rr, c0 + 1 : c0 + 1 + inner_w] = np.where(
                top_strip == int(left_strip[rr]),
                top_strip,
                BG_COLOR,
            )
    return out


def _draw_frame(grid: np.ndarray, spec: dict[str, object]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0 - 1, c0 + 1 : c1] = np.asarray(spec["top_strip"], dtype=int)
    grid[r0 + 1 : r1, c0 - 1] = np.asarray(spec["left_strip"], dtype=int)


def _choose_grid_size(specs: list[dict[str, object]], rng: np.random.Generator) -> tuple[int, int] | None:
    reserved_area = 0
    min_h = 0
    min_w = 0
    for spec in specs:
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        reserved_area += (inner_h + 5) * (inner_w + 5)
        min_h = max(min_h, inner_h + 6)
        min_w = max(min_w, inner_w + 6)

    base = int(np.ceil(np.sqrt(reserved_area * 1.8)))
    low_h = max(16, min_h, base - 2)
    low_w = max(16, min_w, base - 2)
    high_h = min(30, max(low_h, base + 6))
    high_w = min(30, max(low_w, base + 6))
    if low_h > high_h or low_w > high_w:
        return None
    grid_h = int(rng.integers(low_h, high_h + 1))
    grid_w = int(rng.integers(low_w, high_w + 1))
    return grid_h, grid_w


def _place_specs(
    specs: list[dict[str, object]],
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator,
) -> bool:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for spec in sorted(specs, key=lambda item: int(item["inner_h"]) * int(item["inner_w"]), reverse=True):
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        placed = False

        row_low = 2
        row_high = grid_h - frame_h - 1
        col_low = 2
        col_high = grid_w - frame_w - 1
        if row_low > row_high or col_low > col_high:
            return False

        for _ in range(120):
            r0 = int(rng.integers(row_low, row_high + 1))
            c0 = int(rng.integers(col_low, col_high + 1))
            rs0 = r0 - 1
            rs1 = r0 + frame_h - 1
            cs0 = c0 - 1
            cs1 = c0 + frame_w - 1

            pad_r0 = max(0, rs0 - 1)
            pad_r1 = min(grid_h, rs1 + 2)
            pad_c0 = max(0, cs0 - 1)
            pad_c1 = min(grid_w, cs1 + 2)
            if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                continue

            occupied[pad_r0:pad_r1, pad_c0:pad_c1] = True
            spec["r0"] = r0
            spec["c0"] = c0
            placed = True
            break

        if not placed:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(1, 4))
    specs: list[dict[str, object]] = []
    for _ in range(frame_count):
        inner_h = int(rng.integers(3, 9))
        inner_w = int(rng.integers(3, 10))
        top_strip, left_strip = _sample_stripes(inner_h, inner_w, rng)
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "top_strip": top_strip,
                "left_strip": left_strip,
            }
        )

    size = _choose_grid_size(specs, rng)
    if size is None:
        return None
    grid_h, grid_w = size
    if not _place_specs(specs, grid_h, grid_w, rng):
        return None

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for spec in specs:
        _draw_frame(grid, spec)

    expected = _expected_from_specs(grid, specs)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None

    total_inner = sum(int(spec["inner_h"]) * int(spec["inner_w"]) for spec in specs)
    total_matches = int(np.count_nonzero(expected != grid))
    if total_matches <= 0 or total_matches >= total_inner:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
