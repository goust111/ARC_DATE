import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

LEGEND_COLORS = (BLUE, RED, GREEN)
GEN_RETRIES = 300

_BASE_TEMPLATES = {
    1: [
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=bool,
        ),
    ],
    2: [
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ],
    3: [
        np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ],
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


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
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    seen = outside.copy()
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            holes += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return holes


def _object_holes(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _hole_count(mask)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    legend_cells = [
        (r, c, int(g[r, c]))
        for r in range(h - 1)
        for c in range(1, w)
        if int(g[r, c]) in LEGEND_COLORS
    ]
    assert len(legend_cells) == 3, "expected a three-cell legend"
    legend_cells.sort()
    legend_cols = {c for _, c, _ in legend_cells}
    assert len(legend_cols) == 1, "legend must be vertical"
    legend_rows = [r for r, _, _ in legend_cells]
    assert legend_rows == list(range(legend_rows[0], legend_rows[0] + 3)), "legend rows must be consecutive"
    color_by_holes = {i + 1: legend_cells[i][2] for i in range(3)}

    out = g.copy()
    out[g == GREY] = BLACK

    for coords in _components(g == GREY):
        holes = _object_holes(coords)
        color = color_by_holes[holes]
        r0, c0, r1, _c1 = _bbox(coords)
        dr = 0
        dc = 0
        if color == BLUE:
            dc = 1 - c0
        elif color == RED:
            dr = (h - 2) - r1

        shifted = coords + np.array([dr, dc], dtype=int)
        assert np.all(shifted[:, 0] >= 0) and np.all(shifted[:, 0] < h)
        assert np.all(shifted[:, 1] >= 0) and np.all(shifted[:, 1] < w)
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == BLACK), "object destinations must be clear"
        out[shifted[:, 0], shifted[:, 1]] = color

    return out


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            rot = np.rot90(base, k)
            key = rot.tobytes() + str(rot.shape).encode()
            if key not in seen:
                seen.add(key)
                variants.append(rot.copy())
    return variants


def _augment_mask(mask: np.ndarray, hole_count: int) -> np.ndarray:
    out = np.pad(mask, 1, constant_values=False)
    extra_steps = random.randint(0, 2 if hole_count < 3 else 1)
    for _ in range(extra_steps):
        candidates = []
        h, w = out.shape
        for r in range(h):
            for c in range(w):
                if out[r, c]:
                    continue
                if not any(out[nr, nc] for nr, nc in _neighbors4(r, c, h, w)):
                    continue
                trial = out.copy()
                trial[r, c] = True
                if _hole_count(trial) == hole_count:
                    candidates.append((r, c))
        if not candidates:
            break
        rr, cc = random.choice(candidates)
        out[rr, cc] = True
    return _trim(out)


def _sample_shape(hole_count: int, color: int) -> np.ndarray:
    base = random.choice(_BASE_TEMPLATES[hole_count])
    transforms = _unique_transforms(base)
    best_height = min(mask.shape[0] for mask in transforms)
    short = [mask for mask in transforms if mask.shape[0] == best_height]
    mask = random.choice(short).copy()
    assert _hole_count(mask) == hole_count
    return mask


def _mask_at(shape: tuple[int, int], coords: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _sweep_mask(coords: np.ndarray, dr: int, dc: int, h: int, w: int) -> np.ndarray:
    sweep = np.zeros((h, w), dtype=bool)
    if dr == 0 and dc == 0:
        sweep[coords[:, 0], coords[:, 1]] = True
        return sweep

    steps = abs(dr) + abs(dc)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    for step in range(steps + 1):
        shifted = coords + np.array([step * step_r, step * step_c], dtype=int)
        sweep[shifted[:, 0], shifted[:, 1]] = True
    return sweep


def _offset_for_color(color: int, r0: int, c0: int, shape: tuple[int, int], h: int) -> tuple[int, int]:
    height, _width = shape
    if color == BLUE:
        return 0, 1 - c0
    if color == RED:
        return (h - 2) - (r0 + height - 1), 0
    return 0, 0


def _candidate_positions(shape: tuple[int, int], color: int, h: int, w: int) -> list[tuple[int, int]]:
    obj_h, obj_w = shape
    row_lo = 1
    row_hi = h - obj_h - 2
    col_lo = 2
    col_hi = w - obj_w - 2
    if color == RED:
        row_hi = min(row_hi, h - obj_h - 4)
    if color == BLUE:
        col_lo = max(col_lo, 3)
    if row_lo > row_hi or col_lo > col_hi:
        return []

    positions = [(r0, c0) for r0 in range(row_lo, row_hi + 1) for c0 in range(col_lo, col_hi + 1)]
    random.shuffle(positions)
    positions.sort(
        key=lambda rc: (
            0 if color != GREEN else 1,
            -(rc[1]) if color == BLUE else 0,
            rc[0] if color == RED else 0,
            random.random(),
        )
    )
    return positions[:80]


def _place_objects(
    objects: list[dict],
    static_mask: np.ndarray,
    h: int,
    w: int,
) -> list[dict] | None:
    ordered = sorted(
        objects,
        key=lambda obj: (
            obj["color"] == GREEN,
            -(obj["mask"].shape[0] + obj["mask"].shape[1]),
            -int(obj["mask"].sum()),
        ),
    )

    def search(
        index: int,
        src_union: np.ndarray,
        dst_union: np.ndarray,
        sweep_union: np.ndarray,
        placed: list[dict],
    ) -> list[dict] | None:
        if index == len(ordered):
            return placed

        obj = ordered[index]
        mask = obj["mask"]
        coords0 = np.argwhere(mask)
        for r0, c0 in _candidate_positions(mask.shape, obj["color"], h, w):
            coords = coords0 + np.array([r0, c0], dtype=int)
            src = _mask_at(mask.shape, coords, h, w)
            if np.any(_dilate8(src) & static_mask):
                continue
            if np.any(_dilate8(src) & src_union):
                continue

            dr, dc = _offset_for_color(obj["color"], r0, c0, mask.shape, h)
            moved = coords + np.array([dr, dc], dtype=int)
            if np.any(moved[:, 0] < 0) or np.any(moved[:, 0] >= h) or np.any(moved[:, 1] < 0) or np.any(moved[:, 1] >= w):
                continue

            dst = _mask_at(mask.shape, moved, h, w)
            if np.any(dst & static_mask):
                continue
            if np.any(_dilate8(dst) & dst_union):
                continue

            sweep = _sweep_mask(coords, dr, dc, h, w)
            if np.any(sweep & static_mask):
                continue
            if np.any(sweep & sweep_union):
                continue

            placed_obj = {
                "mask": mask,
                "color": obj["color"],
                "holes": obj["holes"],
                "src": coords,
                "dst": moved,
            }
            result = search(
                index + 1,
                src_union | src,
                dst_union | dst,
                sweep_union | sweep,
                placed + [placed_obj],
            )
            if result is not None:
                return result
        return None

    empty = np.zeros((h, w), dtype=bool)
    return search(0, empty, empty, empty, [])


def _paint_grid(h: int, w: int, legend: list[int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = BLUE
    grid[h - 1, :] = RED
    grid[h - 1, 0] = RED

    legend_col = w - 2
    for idx, color in enumerate(legend):
        grid[1 + idx, legend_col] = color

    for obj in placed:
        grid[obj["src"][:, 0], obj["src"][:, 1]] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    legend = random.sample(list(LEGEND_COLORS), 3)

    count = random.randint(3, 5)
    extras = random.sample([1, 2, 3], count - 3)
    hole_counts = [1, 2, 3, *extras]
    random.shuffle(hole_counts)

    objects = []
    for holes in hole_counts:
        color = legend[holes - 1]
        mask = _sample_shape(holes, color)
        objects.append({"holes": holes, "color": color, "mask": mask})

    by_color = {BLUE: [], GREEN: [], RED: []}
    for obj in objects:
        by_color[obj["color"]].append(obj)
    for color in LEGEND_COLORS:
        random.shuffle(by_color[color])

    blue_heights = [obj["mask"].shape[0] for obj in by_color[BLUE]]
    blue_total_h = sum(blue_heights) + max(0, len(blue_heights) - 1)
    green_band_h = max(obj["mask"].shape[0] for obj in by_color[GREEN])
    red_band_h = max(obj["mask"].shape[0] for obj in by_color[RED])

    green_total_w = sum(obj["mask"].shape[1] for obj in by_color[GREEN]) + len(by_color[GREEN]) - 1
    red_total_w = sum(obj["mask"].shape[1] for obj in by_color[RED]) + len(by_color[RED]) - 1
    blue_max_w = max(obj["mask"].shape[1] for obj in by_color[BLUE])

    min_h = blue_total_h + green_band_h + red_band_h + 6
    min_w = max(16, green_total_w + 4, red_total_w + 4, blue_max_w + 6)
    if min_h > 20 or min_w > 20:
        return None
    h = random.randint(max(16, min_h), 20)
    w = random.randint(min_w, 20)

    extra_rows = h - min_h
    top_extra = random.randint(0, extra_rows)
    mid_extra = random.randint(0, extra_rows - top_extra)

    blue_top = 1 + top_extra
    blue_end = blue_top + blue_total_h - 1
    green_top = blue_end + 2
    red_top = green_top + green_band_h + 1 + mid_extra
    red_bottom = red_top + red_band_h - 1
    if red_bottom > h - 4:
        return None

    placed = []

    row = blue_top
    for obj in by_color[BLUE]:
        obj_h, obj_w = obj["mask"].shape
        col_hi = w - obj_w - 3
        col_lo = min(col_hi, max(3, w // 2))
        if col_lo > col_hi:
            return None
        col = random.randint(col_lo, col_hi)
        coords = np.argwhere(obj["mask"]) + np.array([row, col], dtype=int)
        placed.append({"mask": obj["mask"], "color": BLUE, "holes": obj["holes"], "src": coords})
        row += obj_h + 1

    def place_band(objs: list[dict], top: int, band_h: int, color: int) -> bool:
        total_w = sum(obj["mask"].shape[1] for obj in objs) + len(objs) - 1
        start_hi = w - total_w - 2
        if start_hi < 2:
            return False
        col = random.randint(2, start_hi)
        for obj in objs:
            obj_h, obj_w = obj["mask"].shape
            row0 = top + random.randint(0, band_h - obj_h)
            coords = np.argwhere(obj["mask"]) + np.array([row0, col], dtype=int)
            placed.append({"mask": obj["mask"], "color": color, "holes": obj["holes"], "src": coords})
            col += obj_w + 1
        return True

    if not place_band(by_color[GREEN], green_top, green_band_h, GREEN):
        return None
    if not place_band(by_color[RED], red_top, red_band_h, RED):
        return None

    grid = _paint_grid(h, w, legend, placed)
    if len(_components(grid == GREY)) != count:
        return None
    if any(_object_holes(coords) not in (1, 2, 3) for coords in _components(grid == GREY)):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
