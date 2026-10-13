import numpy as np
from collections import deque


SEED_COLOR = 7  # orange
NON_SEED_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

GRID_H_RANGE = (24, 37)  # inclusive, exclusive
GRID_W_RANGE = (24, 37)
EDGE_MARGIN = 1
MIN_GAP = 2
EXTRA_FRAMES_RANGE = (0, 3)
MAX_PLACE_TRIES = 250
MAX_GENERATION_ATTEMPTS = 300

RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    assert bg != SEED_COLOR

    frame_pixels = (g != bg) & (g != SEED_COLOR)
    comps = _components(frame_pixels)
    assert comps, "expected at least one rectangular frame"

    frames = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 3 and width >= 3

        local = np.zeros((height, width), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = _frame_mask(height, width)
        assert np.array_equal(local, expected), "foreground component must be a 1px rectangular frame"

        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1
        color = int(colors[0])

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_count = int(np.count_nonzero(interior == SEED_COLOR))
        assert 1 <= seed_count <= 4

        frames.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": int(height),
                "width": int(width),
                "color": int(color),
                "seed_count": int(seed_count),
            }
        )

    frames.sort(key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    return bg, frames


def _centered_clip(src: np.ndarray, dst_shape: tuple[int, int], fill: int) -> np.ndarray:
    dst_h, dst_w = map(int, dst_shape)
    src_h, src_w = map(int, src.shape)
    out = np.full((dst_h, dst_w), fill, dtype=int)

    top = (dst_h - src_h) // 2
    left = (dst_w - src_w) // 2

    dst_r0 = max(0, top)
    dst_c0 = max(0, left)
    src_r0 = max(0, -top)
    src_c0 = max(0, -left)
    copy_h = min(dst_h - dst_r0, src_h - src_r0)
    copy_w = min(dst_w - dst_c0, src_w - src_c0)
    if copy_h > 0 and copy_w > 0:
        out[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w] = src[
            src_r0 : src_r0 + copy_h, src_c0 : src_c0 + copy_w
        ]
    return out


def _transform_frame(color: int, height: int, width: int, seed_count: int, bg: int) -> np.ndarray:
    base = np.full((height, width), bg, dtype=int)
    base[_frame_mask(height, width)] = int(color)

    if seed_count == 1:
        return base
    if seed_count == 2:
        return np.flipud(base)
    if seed_count == 3:
        rotated = np.rot90(base, -1)
        return _centered_clip(rotated, (height, width), bg)
    if seed_count == 4:
        small_h = max(1, height // 2)
        small_w = max(1, width // 2)
        out = np.full((height, width), bg, dtype=int)
        out[:small_h, :small_w][_frame_mask(small_h, small_w)] = int(color)
        return out
    raise AssertionError("seed count must be in 1..4")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count orange seeds inside each closed rectangular frame, remove the seeds,
    and transform the frame chosen by that count:
    1 -> identity
    2 -> vertical mirror
    3 -> rotate 90 degrees clockwise around the frame center, clipped to the
         original bbox
    4 -> shrink to half size and place the smaller frame in the original bbox's
         top-left corner
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _extract_frames(g)

    out = np.where(g == SEED_COLOR, bg, g).copy()
    for frame in frames:
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = frame["r1"]
        c1 = frame["c1"]
        local = _transform_frame(
            color=frame["color"],
            height=frame["height"],
            width=frame["width"],
            seed_count=frame["seed_count"],
            bg=bg,
        )
        out[r0 : r1 + 1, c0 : c1 + 1] = local
    return out


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi))


def _size_options(seed_count: int) -> list[tuple[int, int]]:
    options = []
    for height in range(5, 12):
        for width in range(5, 12):
            if (height - 2) * (width - 2) < seed_count:
                continue
            if seed_count == 3:
                if height == width or abs(height - width) < 2:
                    continue
                if (height % 2) != (width % 2):
                    continue
            if seed_count == 4 and (height < 6 or width < 6):
                continue
            options.append((height, width))
    assert options
    return options


SIZE_OPTIONS = {seed_count: _size_options(seed_count) for seed_count in range(1, 5)}


def _sample_positions(height: int, width: int, count: int) -> list[tuple[int, int]]:
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    picks = RNG.choice(len(interior), size=count, replace=False)
    return [interior[int(idx)] for idx in np.atleast_1d(picks)]


def _build_spec_list(bg: int) -> list[dict[str, int]]:
    extra = _randint(*EXTRA_FRAMES_RANGE)
    seed_counts = [1, 2, 3, 4] + [int(RNG.integers(1, 5)) for _ in range(extra)]
    RNG.shuffle(seed_counts)

    color_pool = [color for color in NON_SEED_COLORS if color != bg]
    palette_size = _randint(2, min(5, len(color_pool) + 1))
    palette = [int(color) for color in RNG.choice(color_pool, size=palette_size, replace=False)]

    specs = []
    for seed_count in seed_counts:
        height, width = SIZE_OPTIONS[int(seed_count)][_randint(0, len(SIZE_OPTIONS[int(seed_count)]))]
        specs.append(
            {
                "seed_count": int(seed_count),
                "height": int(height),
                "width": int(width),
                "color": int(palette[_randint(0, len(palette))]),
            }
        )
    return specs


def _place_specs(shape: tuple[int, int], specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    grid_h, grid_w = map(int, shape)
    order = list(range(len(specs)))
    RNG.shuffle(order)
    placed = np.zeros((grid_h, grid_w), dtype=bool)
    result = [dict(spec) for spec in specs]

    for idx in order:
        spec = result[idx]
        height = int(spec["height"])
        width = int(spec["width"])
        row_lo = EDGE_MARGIN
        row_hi = grid_h - height - EDGE_MARGIN + 1
        col_lo = EDGE_MARGIN
        col_hi = grid_w - width - EDGE_MARGIN + 1
        if row_lo >= row_hi or col_lo >= col_hi:
            return None

        placed_ok = False
        for _ in range(MAX_PLACE_TRIES):
            r0 = _randint(row_lo, row_hi)
            c0 = _randint(col_lo, col_hi)
            rr0 = max(0, r0 - MIN_GAP)
            cc0 = max(0, c0 - MIN_GAP)
            rr1 = min(grid_h, r0 + height + MIN_GAP)
            cc1 = min(grid_w, c0 + width + MIN_GAP)
            if placed[rr0:rr1, cc0:cc1].any():
                continue
            placed[rr0:rr1, cc0:cc1] = True
            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            spec["r1"] = int(r0 + height - 1)
            spec["c1"] = int(c0 + width - 1)
            placed_ok = True
            break
        if not placed_ok:
            return None
    return result


def _render_grid(shape: tuple[int, int], bg: int, specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        height = int(spec["height"])
        width = int(spec["width"])
        color = int(spec["color"])
        local = np.full((height, width), bg, dtype=int)
        local[_frame_mask(height, width)] = color
        for sr, sc in _sample_positions(height, width, int(spec["seed_count"])):
            local[int(sr), int(sc)] = SEED_COLOR
        grid[r0 : r0 + height, c0 : c0 + width] = local
    return grid


def _normalize_specs(specs: list[dict[str, int]]) -> list[tuple[int, ...]]:
    return sorted(
        (
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["r1"]),
            int(spec["c1"]),
            int(spec["color"]),
            int(spec["seed_count"]),
        )
        for spec in specs
    )


def _valid_generated_example(grid: np.ndarray, specs: list[dict[str, int]], bg: int) -> bool:
    if _most_common_color(grid) != bg:
        return False

    try:
        found_bg, frames = _extract_frames(grid)
    except AssertionError:
        return False

    if found_bg != bg:
        return False

    found_specs = _normalize_specs(frames)
    want_specs = _normalize_specs(specs)
    if found_specs != want_specs:
        return False

    counts = sorted(frame["seed_count"] for frame in frames)
    if counts.count(1) < 1 or counts.count(2) < 1 or counts.count(3) < 1 or counts.count(4) < 1:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.any(out == SEED_COLOR):
        return False

    changed = 0
    for frame in frames:
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = frame["r1"]
        c1 = frame["c1"]
        before = np.where(grid[r0 : r1 + 1, c0 : c1 + 1] == SEED_COLOR, bg, grid[r0 : r1 + 1, c0 : c1 + 1])
        after = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(before, after):
            changed += 1
    return changed >= 2


def _gen_attempt() -> np.ndarray | None:
    bg = int(NON_SEED_COLORS[_randint(0, len(NON_SEED_COLORS))])
    specs = _build_spec_list(bg)
    grid_h = _randint(*GRID_H_RANGE)
    grid_w = _randint(*GRID_W_RANGE)
    placed_specs = _place_specs((grid_h, grid_w), specs)
    if placed_specs is None:
        return None

    grid = _render_grid((grid_h, grid_w), bg, placed_specs)
    if not _valid_generated_example(grid, placed_specs, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
