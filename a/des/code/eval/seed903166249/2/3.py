from __future__ import annotations

import random

import numpy as np


BG = 0
DIRECTIONS = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 300
MAX_SIDE = 34


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_gap(
    missing: set[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> tuple[str, int, int] | None:
    r0, c0, r1, c1 = bbox
    if len(missing) not in (1, 2):
        return None

    rows = sorted(r for r, _ in missing)
    cols = sorted(c for _, c in missing)

    if all(r == r0 and c0 < c < c1 for r, c in missing):
        if cols[-1] - cols[0] + 1 == len(cols):
            return "top", cols[0], len(cols)
    if all(r == r1 and c0 < c < c1 for r, c in missing):
        if cols[-1] - cols[0] + 1 == len(cols):
            return "bottom", cols[0], len(cols)
    if all(c == c0 and r0 < r < r1 for r, c in missing):
        if rows[-1] - rows[0] + 1 == len(rows):
            return "left", rows[0], len(rows)
    if all(c == c1 and r0 < r < r1 for r, c in missing):
        if rows[-1] - rows[0] + 1 == len(rows):
            return "right", rows[0], len(rows)
    return None


def _extract_fences(grid: np.ndarray, bg: int) -> list[dict]:
    fences: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue

            comp_set = set(comp)
            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            if comp_set - border:
                continue
            missing = border - comp_set
            gap = _classify_gap(missing, (r0, c0, r1, c1))
            if gap is None:
                continue

            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            pattern_mask = (inner != bg) & (inner != color)
            if not pattern_mask.any():
                continue

            coords = np.argwhere(pattern_mask)
            pr0, pc0 = map(int, coords.min(axis=0))
            pr1, pc1 = map(int, coords.max(axis=0))
            if pr0 == 0 or pc0 == 0 or pr1 == inner.shape[0] - 1 or pc1 == inner.shape[1] - 1:
                continue

            stamp = inner[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
            side, gap_start, gap_len = gap
            fences.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "gap_start": int(gap_start),
                    "gap_len": int(gap_len),
                    "area": int(inner.shape[0] * inner.shape[1]),
                    "stamp": stamp,
                }
            )
    return fences


def _copy_origins(
    bbox: tuple[int, int, int, int],
    side: str,
    gap_start: int,
    gap_len: int,
    stamp_shape: tuple[int, int],
    copies: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    sh, sw = stamp_shape
    origins: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        col0 = gap_start + (gap_len - sw) // 2
        for i in range(copies):
            row0 = r0 - sh - i * (sh + 1) if side == "top" else r1 + 1 + i * (sh + 1)
            origins.append((int(row0), int(col0)))
    else:
        row0 = gap_start + (gap_len - sh) // 2
        for i in range(copies):
            col0 = c0 - sw - i * (sw + 1) if side == "left" else c1 + 1 + i * (sw + 1)
            origins.append((int(row0), int(col0)))
    return origins


def _paint_stamp(out: np.ndarray, stamp: np.ndarray, row0: int, col0: int, bg: int) -> None:
    h, w = out.shape
    sh, sw = stamp.shape
    for dr in range(sh):
        for dc in range(sw):
            value = int(stamp[dr, dc])
            if value == bg:
                continue
            rr = row0 + dr
            cc = col0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reuse each fenced region's interior pattern as a stamp, and repeat it outside the gate-facing side.

    For every rectangular fence with one 1- or 2-cell gap, extract the interior pattern, determine the
    gap side, rank all fences by enclosed interior area, and stamp rank-many copies outward from the gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    fences = _extract_fences(g, bg)
    assert fences, "expected at least one fence"

    ordered = sorted(fences, key=lambda fence: (fence["area"], fence["bbox"][0], fence["bbox"][1]))
    out = g.copy()
    for rank, fence in enumerate(ordered, start=1):
        origins = _copy_origins(
            fence["bbox"],
            fence["side"],
            fence["gap_start"],
            fence["gap_len"],
            tuple(map(int, fence["stamp"].shape)),
            rank,
        )
        for row0, col0 in origins:
            _paint_stamp(out, fence["stamp"], row0, col0, bg)
    return out


def _random_mask(shape: tuple[int, int], rng: random.Random) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    cells: set[tuple[int, int]] = set()
    for r in range(h):
        cells.add((r, rng.randrange(w)))
    for c in range(w):
        cells.add((rng.randrange(h), c))
    all_cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in cells]
    rng.shuffle(all_cells)
    extra = rng.randint(0, len(all_cells))
    cells.update(all_cells[:extra])
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_inner_pattern(
    inner_h: int, inner_w: int, colors: list[int], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    inner = np.full((inner_h, inner_w), BG, dtype=int)
    max_h = min(3, inner_h - 2)
    max_w = min(3, inner_w - 2)
    stamp_h = 1
    stamp_w = 1
    for _ in range(40):
        stamp_h = rng.randint(1, max_h)
        stamp_w = rng.randint(1, max_w)
        if len(colors) == 1 or stamp_h * stamp_w >= 2:
            break
    else:
        colors = colors[:1]
    top = rng.randint(1, inner_h - stamp_h - 1)
    left = rng.randint(1, inner_w - stamp_w - 1)

    mask = _random_mask((stamp_h, stamp_w), rng)
    stamp = np.full((stamp_h, stamp_w), BG, dtype=int)
    color_count = len(colors)
    coords = [(r, c) for r in range(stamp_h) for c in range(stamp_w) if mask[r, c]]
    if color_count == 1:
        for r, c in coords:
            stamp[r, c] = colors[0]
    else:
        rng.shuffle(coords)
        split = rng.randint(1, len(coords) - 1)
        first = set(coords[:split])
        for r, c in coords:
            stamp[r, c] = colors[0] if (r, c) in first else colors[1]
    inner[top : top + stamp_h, left : left + stamp_w] = stamp
    return inner, stamp.copy()


def _build_fence_patch(
    inner: np.ndarray, fence_color: int, side: str, gap_offset: int, gap_len: int
) -> np.ndarray:
    inner_h, inner_w = map(int, inner.shape)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = fence_color
    patch[-1, :] = fence_color
    patch[:, 0] = fence_color
    patch[:, -1] = fence_color
    if side == "top":
        patch[0, gap_offset : gap_offset + gap_len] = BG
    elif side == "bottom":
        patch[-1, gap_offset : gap_offset + gap_len] = BG
    elif side == "left":
        patch[gap_offset : gap_offset + gap_len, 0] = BG
    else:
        patch[gap_offset : gap_offset + gap_len, -1] = BG
    patch[1:-1, 1:-1] = inner
    return patch


def _reserve_geometry(spec: dict) -> tuple[int, int, int, int, int, int]:
    inner_h, inner_w = spec["inner"].shape
    local_bbox = (0, 0, inner_h + 1, inner_w + 1)
    origins = _copy_origins(
        local_bbox,
        spec["side"],
        spec["gap_offset"],
        spec["gap_len"],
        tuple(map(int, spec["stamp"].shape)),
        int(spec["copies"]),
    )
    sh, sw = map(int, spec["stamp"].shape)
    min_r = 0
    min_c = 0
    max_r = inner_h + 1
    max_c = inner_w + 1
    for row0, col0 in origins:
        min_r = min(min_r, row0)
        min_c = min(min_c, col0)
        max_r = max(max_r, row0 + sh - 1)
        max_c = max(max_c, col0 + sw - 1)
    pad = 1
    reserve_h = max_r - min_r + 1 + 2 * pad
    reserve_w = max_c - min_c + 1 + 2 * pad
    fence_row_offset = pad - min_r
    fence_col_offset = pad - min_c
    return reserve_h, reserve_w, fence_row_offset, fence_col_offset, min_r, min_c


def _render_specs(specs: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    for spec in specs:
        patch = _build_fence_patch(spec["inner"], spec["fence_color"], spec["side"], spec["gap_offset"], spec["gap_len"])
        fr, fc = spec["fence_pos"]
        ph, pw = patch.shape
        inp[fr : fr + ph, fc : fc + pw] = patch

    out = inp.copy()
    for spec in specs:
        fr, fc = spec["fence_pos"]
        inner_h, inner_w = spec["inner"].shape
        bbox = (fr, fc, fr + inner_h + 1, fc + inner_w + 1)
        if spec["side"] in ("top", "bottom"):
            gap_start = fc + spec["gap_offset"]
        else:
            gap_start = fr + spec["gap_offset"]
        origins = _copy_origins(
            bbox,
            spec["side"],
            gap_start,
            spec["gap_len"],
            tuple(map(int, spec["stamp"].shape)),
            int(spec["copies"]),
        )
        for row0, col0 in origins:
            _paint_stamp(out, spec["stamp"], row0, col0, BG)
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    fence_count = rng.choice((2, 3))
    directions = rng.sample(list(DIRECTIONS), k=fence_count)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    fence_colors = palette[:fence_count]
    color_cursor = fence_count

    specs: list[dict] = []
    areas: set[int] = set()
    for idx in range(fence_count):
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 6)
        area = inner_h * inner_w
        if area in areas:
            return None
        areas.add(area)

        pattern_color_count = rng.choice((1, 2))
        pattern_colors = palette[color_cursor : color_cursor + pattern_color_count]
        color_cursor += pattern_color_count
        if len(pattern_colors) != pattern_color_count:
            return None

        inner, stamp = _make_inner_pattern(inner_h, inner_w, pattern_colors, rng)
        gap_len = rng.choice((1, 2))
        if directions[idx] in ("top", "bottom"):
            gap_offset = rng.randint(1, inner_w - gap_len + 1)
        else:
            gap_offset = rng.randint(1, inner_h - gap_len + 1)
        specs.append(
            {
                "inner": inner,
                "stamp": stamp,
                "fence_color": fence_colors[idx],
                "side": directions[idx],
                "gap_offset": int(gap_offset),
                "gap_len": int(gap_len),
                "area": int(area),
            }
        )

    ordered = sorted(range(fence_count), key=lambda i: specs[i]["area"])
    for rank, idx in enumerate(ordered, start=1):
        specs[idx]["copies"] = rank

    for spec in specs:
        reserve_h, reserve_w, row_offset, col_offset, _min_r, _min_c = _reserve_geometry(spec)
        spec["reserve_h"] = reserve_h
        spec["reserve_w"] = reserve_w
        spec["fence_row_offset"] = row_offset
        spec["fence_col_offset"] = col_offset

    slot_choices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assigned_slots = rng.sample(slot_choices, k=fence_count)
    order_for_slots = list(range(fence_count))
    rng.shuffle(order_for_slots)
    for spec_idx, slot in zip(order_for_slots, assigned_slots):
        specs[spec_idx]["slot"] = slot

    row_reqs = [0, 0]
    col_reqs = [0, 0]
    for spec in specs:
        sr, sc = spec["slot"]
        row_reqs[sr] = max(row_reqs[sr], spec["reserve_h"])
        col_reqs[sc] = max(col_reqs[sc], spec["reserve_w"])

    row_sizes = [req + rng.randint(1, 4) if req else rng.randint(4, 7) for req in row_reqs]
    col_sizes = [req + rng.randint(1, 4) if req else rng.randint(4, 7) for req in col_reqs]
    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    row_gap = rng.randint(2, 4)
    col_gap = rng.randint(2, 4)

    height = top_margin + row_sizes[0] + row_gap + row_sizes[1] + bottom_margin
    width = left_margin + col_sizes[0] + col_gap + col_sizes[1] + right_margin
    if height > MAX_SIDE or width > MAX_SIDE:
        return None

    row_starts = [top_margin, top_margin + row_sizes[0] + row_gap]
    col_starts = [left_margin, left_margin + col_sizes[0] + col_gap]
    for spec in specs:
        sr, sc = spec["slot"]
        slot_r0 = row_starts[sr]
        slot_c0 = col_starts[sc]
        slack_r = row_sizes[sr] - spec["reserve_h"]
        slack_c = col_sizes[sc] - spec["reserve_w"]
        if slack_r < 0 or slack_c < 0:
            return None
        reserve_r0 = slot_r0 + (rng.randint(0, slack_r) if slack_r else 0)
        reserve_c0 = slot_c0 + (rng.randint(0, slack_c) if slack_c else 0)
        spec["fence_pos"] = (
            int(reserve_r0 + spec["fence_row_offset"]),
            int(reserve_c0 + spec["fence_col_offset"]),
        )

    inp, expected = _render_specs(specs, (height, width))
    if _most_common_color(inp) != BG:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
