import numpy as np
from collections import deque


MAX_GENERATION_ATTEMPTS = 500
PALETTE = tuple(range(10))
SHAPE_KINDS = ("rect", "lshape")


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _cc4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _uniform_band_rows(grid: np.ndarray, bg: int) -> dict[int, int]:
    bands: dict[int, int] = {}
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != bg and np.all(row == row[0]):
            bands[int(r)] = int(row[0])
    return bands


def _enclosed_cells(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return np.empty((0, 2), dtype=int)

    er0, ec0 = r0 - 1, c0 - 1
    er1, ec1 = r1 + 1, c1 + 1
    wall = np.zeros((er1 - er0 + 1, ec1 - ec0 + 1), dtype=bool)
    for r, c in coords:
        wall[int(r) - er0, int(c) - ec0] = True

    seen = np.zeros_like(wall, dtype=bool)
    q = deque()
    hh, ww = wall.shape
    border = set()
    for rr in range(hh):
        border.add((rr, 0))
        border.add((rr, ww - 1))
    for cc in range(ww):
        border.add((0, cc))
        border.add((hh - 1, cc))
    for rr, cc in border:
        if not wall[rr, cc] and not seen[rr, cc]:
            seen[rr, cc] = True
            q.append((rr, cc))
    while q:
        rr, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < hh and 0 <= nc < ww and not wall[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    inside = []
    for rr in range(hh):
        for cc in range(ww):
            if not wall[rr, cc] and not seen[rr, cc]:
                inside.append((rr + er0, cc + ec0))
    return np.array(inside, dtype=int)


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    band_rows = _uniform_band_rows(g, bg)
    band_colors = set(band_rows.values())
    candidates: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in band_colors | {bg}):
        for coords in _cc4(g == color):
            inside = _enclosed_cells(g.shape, coords)
            if inside.size == 0:
                continue
            marker_count = sum(int(g[int(r), int(c)] != bg) for r, c in inside)
            if marker_count >= len(inside):
                continue
            candidates.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "inside": inside,
                    "bbox": _bbox(coords),
                    "count": int(marker_count),
                }
            )
    if not candidates:
        return []

    grouped: dict[int, list[dict]] = {}
    for item in candidates:
        grouped.setdefault(int(item["color"]), []).append(item)
    best_color = max(
        grouped,
        key=lambda color: (
            len(grouped[color]),
            sum(len(frame["inside"]) for frame in grouped[color]),
            sum(len(frame["coords"]) for frame in grouped[color]),
        ),
    )
    frames = grouped[best_color]
    frames.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["color"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each closed outline upward by the number of colored pixels inside it.
    The moved outline becomes the color of the highest horizontal band it crosses.
    Horizontal bands remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    band_rows = _uniform_band_rows(g, bg)
    bands_desc = sorted(band_rows, reverse=True)
    frames = _detect_frames(g)

    out = np.full_like(g, bg)
    for row, color in band_rows.items():
        out[int(row), :] = int(color)

    for frame in frames:
        count = int(frame["count"])
        r0 = int(frame["bbox"][0])
        new_top = r0 - count
        new_color = int(frame["color"])
        crossed = [row for row in bands_desc if new_top <= row <= r0]
        if crossed:
            new_color = int(band_rows[crossed[-1]])
        for r, c in frame["coords"]:
            rr = int(r) - count
            cc = int(c)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and rr not in band_rows:
                out[rr, cc] = new_color
    return out


def _shell_from_inner(inner: np.ndarray) -> np.ndarray:
    pad = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2), dtype=bool)
    pad[1:-1, 1:-1] = inner
    shell = np.zeros_like(pad)
    h, w = pad.shape
    for r in range(h):
        for c in range(w):
            if pad[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and pad[nr, nc]:
                        shell[r, c] = True
                        break
                if shell[r, c]:
                    break
    return shell


def _build_inner_shape(rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "rect":
        inner_h = int(rng.integers(3, 7))
        inner_w = int(rng.integers(4, 9))
        return np.ones((inner_h, inner_w), dtype=bool)

    inner_h = int(rng.integers(4, 8))
    inner_w = int(rng.integers(4, 9))
    vert_t = int(rng.integers(2, max(3, inner_w)))
    horiz_t = int(rng.integers(2, max(3, inner_h)))
    base = np.zeros((inner_h, inner_w), dtype=bool)
    base[:, :vert_t] = True
    base[inner_h - horiz_t :, :] = True
    orientation = int(rng.integers(0, 4))
    if orientation == 1:
        base = np.fliplr(base)
    elif orientation == 2:
        base = np.flipud(base)
    elif orientation == 3:
        base = np.flipud(np.fliplr(base))
    return base


def _sample_band_rows(rng: np.random.Generator, height: int, count: int) -> list[int] | None:
    top_limit = max(6, height // 2 - 2)
    for _ in range(100):
        rows = sorted(rng.choice(np.arange(2, top_limit), size=count, replace=False).tolist())
        if rows[-1] >= height // 2:
            continue
        if all(b - a >= 3 for a, b in zip(rows, rows[1:])):
            return [int(v) for v in rows]
    return None


def _branch_bounds(branch: int, bands_desc: list[int], height: int, frame_h: int) -> tuple[int, int] | None:
    if branch == 0:
        lower = int(bands_desc[0] + 1)
        upper = int(min(height // 2, height - frame_h - 2))
    elif branch < len(bands_desc):
        lower = int(bands_desc[branch] + 1)
        upper = int(bands_desc[branch - 1])
    else:
        lower = 1
        upper = int(bands_desc[-1])
    if lower > upper:
        return None
    return lower, upper


def _make_frame_spec(
    rng: np.random.Generator,
    height: int,
    bands_desc: list[int],
    branch: int,
) -> dict | None:
    for _ in range(120):
        kind = SHAPE_KINDS[int(rng.integers(0, len(SHAPE_KINDS)))]
        inner = _build_inner_shape(rng, kind)
        shell = _shell_from_inner(inner)
        if len(_cc4(shell)) != 1:
            continue

        frame_h, frame_w = map(int, shell.shape)
        bounds = _branch_bounds(branch, bands_desc, height, frame_h)
        if bounds is None:
            continue
        final_low, final_high = bounds
        final_top = int(rng.integers(final_low, final_high + 1))
        inner_area = int(inner.sum())

        old_top_min = max(height // 2, bands_desc[0] + 3)
        shift_min = max(1, old_top_min - final_top)
        shift_max = min(inner_area - 1, height - frame_h - 1 - final_top)
        if shift_min > shift_max:
            continue
        shift_high = min(shift_max, shift_min + max(2, inner_area // 4))
        shift = int(rng.integers(shift_min, shift_high + 1))
        old_top = int(final_top + shift)

        shell_coords = np.argwhere(shell)
        inner_coords = np.argwhere(inner) + 1
        marker_idx = rng.choice(len(inner_coords), size=shift, replace=False)
        markers = inner_coords[marker_idx]
        return {
            "branch": int(branch),
            "kind": kind,
            "old_top": old_top,
            "final_top": final_top,
            "shift": shift,
            "height": frame_h,
            "width": frame_w,
            "shell_coords": shell_coords,
            "marker_coords": markers,
        }
    return None


def _choose_colors(rng: np.random.Generator, band_count: int) -> tuple[int, int, int, list[int]]:
    colors = rng.permutation(PALETTE).tolist()
    bg = int(colors[0])
    outline = int(colors[1])
    marker = int(colors[2])
    band_colors = [int(v) for v in colors[3 : 3 + band_count]]
    return bg, outline, marker, band_colors


def _layout_frames(rng: np.random.Generator, frames: list[dict]) -> int | None:
    gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(frames) - 1))]
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    cursor = left_margin
    for i, frame in enumerate(frames):
        frame["left"] = int(cursor)
        cursor += int(frame["width"])
        if i < len(gaps):
            cursor += gaps[i]
    width = int(cursor + right_margin)
    if 18 <= width <= 38:
        return width
    return None


def _render_input(
    height: int,
    width: int,
    bg: int,
    outline: int,
    marker: int,
    band_rows: list[int],
    band_colors: list[int],
    frames: list[dict],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for row, color in zip(band_rows, band_colors):
        grid[int(row), :] = int(color)
    for frame in frames:
        top = int(frame["old_top"])
        left = int(frame["left"])
        for dr, dc in frame["shell_coords"]:
            grid[top + int(dr), left + int(dc)] = outline
        for dr, dc in frame["marker_coords"]:
            grid[top + int(dr), left + int(dc)] = marker
    return grid


def _render_expected_output(
    height: int,
    width: int,
    bg: int,
    outline: int,
    band_rows: list[int],
    band_colors: list[int],
    frames: list[dict],
) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    for row, color in zip(band_rows, band_colors):
        out[int(row), :] = int(color)

    band_map = {int(r): int(c) for r, c in zip(band_rows, band_colors)}
    bands_desc = sorted((int(v) for v in band_rows), reverse=True)
    for frame in frames:
        old_top = int(frame["old_top"])
        shift = int(frame["shift"])
        new_top = int(frame["final_top"])
        new_color = outline
        crossed = [row for row in bands_desc if new_top <= row <= old_top]
        if crossed:
            new_color = int(band_map[crossed[-1]])
        left = int(frame["left"])
        for dr, dc in frame["shell_coords"]:
            rr = new_top + int(dr)
            cc = left + int(dc)
            if rr not in band_map:
                out[rr, cc] = new_color
    return out


def _grid_is_clear(grid: np.ndarray, frames: list[dict], band_rows: list[int], bg: int) -> bool:
    band_set = set(int(v) for v in band_rows)
    occupied = np.zeros_like(grid, dtype=bool)
    for row in band_set:
        occupied[int(row), :] = True
    for frame in frames:
        top = int(frame["old_top"])
        left = int(frame["left"])
        for dr, dc in np.vstack((frame["shell_coords"], frame["marker_coords"])):
            rr = top + int(dr)
            cc = left + int(dc)
            if occupied[rr, cc]:
                return False
            occupied[rr, cc] = True
        for dr, dc in frame["shell_coords"]:
            rr = int(frame["final_top"]) + int(dr)
            cc = left + int(dc)
            if rr < 0 or rr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]:
                return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(20, 31))
    band_count = int(rng.integers(2, 4))
    band_rows = _sample_band_rows(rng, height, band_count)
    if band_rows is None:
        return None
    bands_desc = sorted(band_rows, reverse=True)

    frame_count = int(rng.integers(2, 5))
    if band_count == 2:
        branch_values = np.array([0, 1, 2], dtype=int)
        branch_probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        branch_values = np.array([0, 1, 2, 3], dtype=int)
        branch_probs = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 2], dtype=float)
    branches = [int(v) for v in rng.choice(branch_values, size=frame_count, replace=True, p=branch_probs)]
    if frame_count > 1 and len(set(branches)) == 1:
        branches[0] = (branches[0] + 1) % (band_count + 1)

    frames = []
    for branch in branches:
        frame = _make_frame_spec(rng, height, bands_desc, branch)
        if frame is None:
            return None
        frames.append(frame)

    width = _layout_frames(rng, frames)
    if width is None:
        return None

    bg, outline, marker, band_colors = _choose_colors(rng, band_count)
    grid = _render_input(height, width, bg, outline, marker, band_rows, band_colors, frames)
    if not _grid_is_clear(grid, frames, band_rows, bg):
        return None

    expected = _render_expected_output(height, width, bg, outline, band_rows, band_colors, frames)
    if not np.array_equal(main(grid), expected):
        return None

    detected = _detect_frames(grid)
    if len(detected) != len(frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
