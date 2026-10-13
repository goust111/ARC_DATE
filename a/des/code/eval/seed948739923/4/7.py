from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AXIS_COLOR = 1
LEGEND_WIDTH = 7
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LAYOUTS = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1), (2, 2)],
    4: [(2, 2), (1, 4), (4, 1)],
}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_legend_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    h, w = g.shape
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        if c1 != w - 1:
            continue
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < LEGEND_WIDTH:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        row_bars = [idx for idx in range(sub.shape[0]) if np.all(sub[idx, :] == GREY)]
        if len(row_bars) < 2:
            continue
        candidates.append((len(pts), (r0, c0, r1, c1)))
    assert candidates, "legend frame not found"
    candidates.sort()
    return candidates[-1][1]


def _parse_legend(g: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    r0, c0, r1, c1 = _find_legend_bbox(g)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    row_bars = [idx for idx in range(sub.shape[0]) if np.all(sub[idx, :] == GREY)]
    mapping: dict[int, np.ndarray] = {}
    for top, bottom in zip(row_bars, row_bars[1:], strict=False):
        if bottom - top < 4:
            continue
        body = sub[top + 1 : bottom, 1:-1]
        if body.shape[0] != 3 or body.shape[1] < 4:
            continue

        sample = None
        sample_c0 = None
        for col in range(body.shape[1] - 2):
            window = body[:, col : col + 3]
            if np.all(window != BLACK) and np.all(window != GREY):
                sample = window.copy()
                sample_c0 = col
                break
        assert sample is not None and sample_c0 is not None, "legend sample missing"

        marker_mask = (body != BLACK) & (body != GREY)
        marker_mask[:, sample_c0 : sample_c0 + 3] = False
        marker_pts = np.argwhere(marker_mask)
        assert marker_pts.shape[0] == 1, "legend marker must be unique"
        mr, mc = map(int, marker_pts[0])
        marker_color = int(body[mr, mc])
        mapping[marker_color] = sample

    assert mapping, "empty legend mapping"
    return mapping, c0


def _pointer_direction(
    g: np.ndarray, center: tuple[int, int], pointer_color: int
) -> tuple[int, int, int]:
    h, w = g.shape
    cr, cc = center
    candidates: list[tuple[int, int, int]] = []
    for dr, dc in CARDINALS:
        run = 0
        rr, cc2 = cr + dr, cc + dc
        while 0 <= rr < h and 0 <= cc2 < w and int(g[rr, cc2]) == pointer_color:
            run += 1
            rr += dr
            cc2 += dc
        if run >= 2:
            candidates.append((dr, dc, run))
    assert len(candidates) == 1, "pointer direction must be unique"
    return candidates[0]


def _transform_disk(
    out: np.ndarray,
    coords: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    sample: np.ndarray,
) -> None:
    cr, cc = center
    dr, dc = direction
    r0, c0, _, _ = _bbox(coords)
    for r, c in coords:
        rr = int(r)
        cc2 = int(c)
        dot = (rr - cr) * dr + (cc2 - cc) * dc
        if dot < 0:
            out[rr, cc2] = int(sample[(rr - r0) % 3, (cc2 - c0) % 3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the pointer direction to keep one disk half visible and refill the opposite half
    with the 3x3 legend pattern keyed by the pointer color. Remove the legend area.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_left = _parse_legend(g)
    canvas = g[:, :legend_left].copy()
    out = canvas.copy()

    for pts in _components(canvas != BLACK):
        r0, c0, r1, c1 = _bbox(pts)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        center_color = int(canvas[center])
        colors = {int(canvas[int(r), int(c)]) for r, c in pts}
        pointer_colors = [color for color in colors if color in mapping]
        assert len(pointer_colors) == 1, "each disk must contain exactly one legend-linked pointer color"
        pointer_color = pointer_colors[0]
        dr, dc, pointer_len = _pointer_direction(canvas, center, pointer_color)
        _transform_disk(out, pts, center, (dr, dc), mapping[pointer_color])
        out[center] = center_color
        for step in range(1, pointer_len + 1):
            rr = center[0] + dr * step
            cc = center[1] + dc * step
            out[rr, cc] = pointer_color

    return out


def _circle_offsets(radius: int) -> list[tuple[int, int]]:
    pts = []
    limit = radius * radius + radius // 2
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= limit:
                pts.append((dr, dc))
    return pts


def _make_tile(rng: random.Random, colors: list[int], used: set[bytes]) -> np.ndarray | None:
    if len(colors) < 2:
        return None
    modes = ["checker", "rows", "cols", "diag", "rings", "cross"]
    for _ in range(80):
        k = 3 if len(colors) >= 3 and rng.random() < 0.5 else 2
        palette = rng.sample(colors, k)
        mode = rng.choice(modes)
        tile = np.zeros((3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                if mode == "checker":
                    idx = (r + c) % k
                elif mode == "rows":
                    idx = r % k
                elif mode == "cols":
                    idx = c % k
                elif mode == "diag":
                    idx = (r - c) % k
                elif mode == "rings":
                    idx = max(abs(r - 1), abs(c - 1)) % k
                else:
                    if r == 1 and c == 1:
                        idx = 0
                    elif r == 1 or c == 1:
                        idx = 1 % k
                    else:
                        idx = 2 % k
                tile[r, c] = palette[idx]
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        for _ in range(rng.randint(0, 3)):
            tile = np.rot90(tile)
        if rng.random() < 0.5:
            tile = np.fliplr(tile)
        sig = tile.tobytes()
        if sig in used:
            continue
        used.add(sig)
        return tile
    return None


def _render_disk(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    direction: tuple[int, int],
    pointer_len: int,
    pointer_color: int,
    base_tile: np.ndarray,
) -> np.ndarray:
    cr, cc = center
    r0 = cr - radius
    c0 = cc - radius
    coords = []
    for dr, dc in _circle_offsets(radius):
        rr = cr + dr
        cc2 = cc + dc
        canvas[rr, cc2] = int(base_tile[(rr - r0) % 3, (cc2 - c0) % 3])
        coords.append((rr, cc2))
    canvas[cr, cc] = AXIS_COLOR
    dr, dc = direction
    for step in range(1, pointer_len + 1):
        rr = cr + dr * step
        cc2 = cc + dc * step
        canvas[rr, cc2] = pointer_color
    return np.array(coords, dtype=int)


def _apply_expected(left_canvas: np.ndarray, disks: list[dict]) -> np.ndarray:
    out = left_canvas.copy()
    for disk in disks:
        _transform_disk(out, disk["coords"], disk["center"], disk["direction"], disk["sample"])
        out[disk["center"]] = AXIS_COLOR
        dr, dc = disk["direction"]
        for step in range(1, disk["pointer_len"] + 1):
            rr = disk["center"][0] + dr * step
            cc = disk["center"][1] + dc * step
            out[rr, cc] = disk["pointer_color"]
    return out


def _draw_legend(
    grid: np.ndarray,
    legend_top: int,
    legend_left: int,
    items: list[tuple[int, np.ndarray]],
) -> None:
    legend_h = 4 * len(items) + 1
    patch = np.full((legend_h, LEGEND_WIDTH), BLACK, dtype=int)
    patch[:, 0] = GREY
    patch[:, 2] = GREY
    patch[:, -1] = GREY
    for idx, (pointer_color, sample) in enumerate(items):
        top = idx * 4
        patch[top, :] = GREY
        patch[top + 4, :] = GREY
        patch[top + 2, 1] = pointer_color
        patch[top + 1 : top + 4, 3:6] = sample
    grid[legend_top : legend_top + legend_h, legend_left : legend_left + LEGEND_WIDTH] = patch


def _slot_positions(
    rng: random.Random, count: int, radius: int
) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    layout_rows, layout_cols = rng.choice(LAYOUTS[count])
    pitch_h = 2 * radius + rng.randint(3, 5)
    pitch_w = 2 * radius + rng.randint(3, 6)
    top_margin = rng.randint(1, 2)
    left_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    right_margin = rng.randint(1, 2)

    legend_h = 4 * count + 1
    height = max(legend_h + rng.randint(0, 2), top_margin + layout_rows * pitch_h + bottom_margin)
    canvas_w = left_margin + layout_cols * pitch_w + right_margin

    all_slots = [(rr, cc) for rr in range(layout_rows) for cc in range(layout_cols)]
    chosen = rng.sample(all_slots, count)
    centers = []
    for sr, sc in chosen:
        cell_top = top_margin + sr * pitch_h
        cell_left = left_margin + sc * pitch_w
        cr = rng.randint(cell_top + radius, cell_top + pitch_h - radius - 2)
        cc = rng.randint(cell_left + radius, cell_left + pitch_w - radius - 2)
        centers.append((cr, cc))

    return height, canvas_w, pitch_h, pitch_w, centers


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    disk_count = rng.randint(2, 4)
    radius = rng.randint(3, 5)
    height, canvas_w, _pitch_h, _pitch_w, centers = _slot_positions(rng, disk_count, radius)
    gap = rng.randint(1, 3)
    legend_left = canvas_w + gap
    width = legend_left + LEGEND_WIDTH
    legend_h = 4 * disk_count + 1
    legend_top = rng.randint(0, height - legend_h)

    pointer_palette = [2, 3, 4, 6, 7, 8, 9]
    pointer_colors = rng.sample(pointer_palette, disk_count)
    pattern_colors = [color for color in pointer_palette if color not in pointer_colors]
    if len(pattern_colors) < 2:
        return None

    directions = CARDINALS[:]
    rng.shuffle(directions)

    used_tiles: set[bytes] = set()
    base_tiles = []
    sample_tiles = []
    for _ in range(disk_count):
        base_tile = _make_tile(rng, pattern_colors, used_tiles)
        sample_tile = _make_tile(rng, pattern_colors, used_tiles)
        if base_tile is None or sample_tile is None:
            return None
        base_tiles.append(base_tile)
        sample_tiles.append(sample_tile)

    canvas = np.zeros((height, canvas_w), dtype=int)
    disks: list[dict] = []
    for idx, center in enumerate(centers):
        pointer_len = rng.randint(2, 3)
        direction = directions[idx]
        pointer_color = pointer_colors[idx]
        coords = _render_disk(
            canvas,
            center,
            radius,
            direction,
            pointer_len,
            pointer_color,
            base_tiles[idx],
        )
        disks.append(
            {
                "coords": coords,
                "center": center,
                "direction": direction,
                "pointer_len": pointer_len,
                "pointer_color": pointer_color,
                "sample": sample_tiles[idx],
            }
        )

    if len(_components(canvas != BLACK)) != disk_count:
        return None
    if np.any(canvas == GREY):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, :canvas_w] = canvas
    legend_items = list(zip(pointer_colors, sample_tiles, strict=False))
    rng.shuffle(legend_items)
    _draw_legend(grid, legend_top, legend_left, legend_items)

    left_canvas = np.zeros((height, legend_left), dtype=int)
    left_canvas[:, :canvas_w] = canvas
    expected = _apply_expected(left_canvas, disks)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
