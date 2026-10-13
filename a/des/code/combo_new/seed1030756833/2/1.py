import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "empty mask has no bbox"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(comp)
    return out


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    by_color: dict[int, list[dict]] = {}
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    max_side = min(10, h, w)

    for frame_color in colors:
        found: list[dict] = []
        for size in range(4, max_side + 1):
            for r0 in range(h - size + 1):
                for c0 in range(w - size + 1):
                    sub = grid[r0 : r0 + size, c0 : c0 + size]
                    border = np.ones((size, size), dtype=bool)
                    border[1:-1, 1:-1] = False
                    if not np.all(sub[border] == frame_color):
                        continue
                    interior = sub[1:-1, 1:-1]
                    vals = np.unique(interior)
                    if len(vals) != 1:
                        continue
                    fill = int(vals[0])
                    if fill in (bg, frame_color):
                        continue
                    found.append(
                        {
                            "r0": r0,
                            "c0": c0,
                            "r1": r0 + size - 1,
                            "c1": c0 + size - 1,
                            "size": size,
                            "frame_color": frame_color,
                            "fill": fill,
                        }
                    )
        if found:
            by_color[frame_color] = found

    assert by_color, "expected at least one square frame"
    max_count = max(len(v) for v in by_color.values())
    frame_colors = [c for c, v in by_color.items() if len(v) == max_count]
    assert len(frame_colors) == 1, "expected a unique frame border color"
    frames = by_color[frame_colors[0]]
    frames.sort(key=lambda f: (f["r0"], f["c0"], f["size"]))
    return frames


def _frame_area_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        mask[frame["r0"] : frame["r1"] + 1, frame["c0"] : frame["c1"] + 1] = True
    return mask


def _legend_side(legend_mask: np.ndarray) -> str:
    h, w = legend_mask.shape
    edge_counts = {
        "top": int(legend_mask[0, :].sum()),
        "bottom": int(legend_mask[-1, :].sum()),
        "left": int(legend_mask[:, 0].sum()),
        "right": int(legend_mask[:, -1].sum()),
    }
    best_count = max(edge_counts.values())
    if best_count > 0:
        winners = [side for side, count in edge_counts.items() if count == best_count]
        assert len(winners) == 1, "legend must touch one grid edge unambiguously"
        return winners[0]

    r0, c0, r1, c1 = _bbox_from_mask(legend_mask)
    distances = {
        "top": r0,
        "bottom": h - 1 - r1,
        "left": c0,
        "right": w - 1 - c1,
    }
    best_distance = min(distances.values())
    winners = [side for side, dist in distances.items() if dist == best_distance]
    assert len(winners) == 1, "legend must be nearest to one grid edge"
    return winners[0]


def _touches(square: set[tuple[int, int]], rem: np.ndarray) -> bool:
    rem_pts = set(map(tuple, np.argwhere(rem)))
    for r, c in square:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in rem_pts:
                return True
    return False


def _square_is_outer(
    side: str,
    sr0: int,
    sc0: int,
    sr1: int,
    sc1: int,
    rr0: int,
    rc0: int,
    rr1: int,
    rc1: int,
) -> bool:
    if side == "right":
        return sc0 > rc1
    if side == "left":
        return sc1 < rc0
    if side == "bottom":
        return sr0 > rr1
    if side == "top":
        return sr1 < rr0
    return False


def _outer_score(side: str, sr0: int, sc0: int, sr1: int, sc1: int) -> int:
    if side == "right":
        return sc1
    if side == "left":
        return -sc0
    if side == "bottom":
        return sr1
    if side == "top":
        return -sr0
    return 0


def _legend_shape_from_local(local: np.ndarray, side: str) -> np.ndarray:
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    h, w = local.shape
    for strict_side in (True, False):
        candidates.clear()
        for k in (2, 1):
            for sr0 in range(h - k + 1):
                for sc0 in range(w - k + 1):
                    if not np.all(local[sr0 : sr0 + k, sc0 : sc0 + k]):
                        continue
                    rem = local.copy()
                    rem[sr0 : sr0 + k, sc0 : sc0 + k] = False
                    if not rem.any():
                        continue
                    rr0, rc0, rr1, rc1 = _bbox_from_mask(rem)
                    if rr1 - rr0 + 1 > 3 or rc1 - rc0 + 1 > 3:
                        continue
                    crop = rem[rr0 : rr1 + 1, rc0 : rc1 + 1]
                    if bool(np.all(crop)):
                        continue
                    square = {
                        (r, c)
                        for r in range(sr0, sr0 + k)
                        for c in range(sc0, sc0 + k)
                    }
                    if not _touches(square, rem):
                        continue
                    sr1, sc1 = sr0 + k - 1, sc0 + k - 1
                    if strict_side and not _square_is_outer(
                        side, sr0, sc0, sr1, sc1, rr0, rc0, rr1, rc1
                    ):
                        continue
                    score = _outer_score(side, sr0, sc0, sr1, sc1)
                    candidates.append((k, score, int(crop.sum()), crop.copy()))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            return candidates[0][3]
    raise AssertionError("could not split legend swatch from shape")


def _legend_shape_for_color(
    grid: np.ndarray, color: int, frame_mask: np.ndarray, side: str
) -> np.ndarray:
    mask = (grid == color) & (~frame_mask)
    assert mask.any(), f"missing legend for color {color}"
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    local = mask[r0 : r1 + 1, c0 : c1 + 1]
    return _legend_shape_from_local(local, side)


def _paint_centered_shape(
    out: np.ndarray, frame: dict, shape: np.ndarray, fill_color: int, bg: int
) -> None:
    ir0, ic0 = frame["r0"] + 1, frame["c0"] + 1
    ir1, ic1 = frame["r1"] - 1, frame["c1"] - 1
    interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
    interior[:, :] = bg
    ih, iw = interior.shape
    ph, pw = shape.shape
    assert ph <= ih and pw <= iw, "legend shape must fit inside frame"
    roff = (ih - ph) // 2
    coff = (iw - pw) // 2
    target = interior[roff : roff + ph, coff : coff + pw]
    target[shape] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each framed fill block with the matching legend shape.

    Input structure:
    - The background is the dominant color. Several small square frames share one border color.
    - Each frame interior is filled by one solid lookup color.
    - On one grid edge, each legend entry has an outer 1x1 or 2x2 color swatch touching a same-color non-rectangular shape of size at most 3x3.

    Rule:
    - Find the square frames and read each frame's interior fill color.
    - For every fill color, split the edge legend into its outer swatch and its remaining shape.
    - Clear the legend area to background, then replace each solid frame interior by the matching shape centered in that frame, preserving the original fill color and frame border.

    Key insight:
    The solid interior color is only a key into the outside legend; the visible legend shape supplies the final interior mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    frames = _find_frames(g, bg)
    frame_mask = _frame_area_mask(g.shape, frames)
    legend_mask = (g != bg) & (~frame_mask)
    assert legend_mask.any(), "expected an outside legend"
    side = _legend_side(legend_mask)

    fill_colors = sorted({int(frame["fill"]) for frame in frames})
    shapes = {
        color: _legend_shape_for_color(g, color, frame_mask, side)
        for color in fill_colors
    }

    out = g.copy()
    out[(~frame_mask) & (out != bg)] = bg
    for frame in frames:
        fill = int(frame["fill"])
        _paint_centered_shape(out, frame, shapes[fill], fill, bg)
    return out


_PATTERNS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    region = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    region[mask] = patch[mask]


def _build_legend_patch(
    pattern: np.ndarray, color: int, bg: int, swatch_size: int, side: str, rng: random.Random
) -> np.ndarray:
    pat = _trim_mask(pattern)
    ph, pw = pat.shape
    k = swatch_size

    if side in ("right", "left"):
        edge_col = pw - 1 if side == "right" else 0
        touch_rows = [int(r) for r in np.where(pat[:, edge_col])[0]]
        touch_row = rng.choice(touch_rows)
        height = max(ph, k)
        lo = max(0, touch_row - k + 1)
        hi = min(height - k, touch_row)
        swatch_r0 = rng.randint(lo, hi)
        patch = np.full((height, pw + k), bg, dtype=int)
        if side == "right":
            patch[:ph, :pw][pat] = color
            patch[swatch_r0 : swatch_r0 + k, pw : pw + k] = color
        else:
            patch[:ph, k : k + pw][pat] = color
            patch[swatch_r0 : swatch_r0 + k, :k] = color
        return patch

    edge_row = ph - 1 if side == "bottom" else 0
    touch_cols = [int(c) for c in np.where(pat[edge_row, :])[0]]
    touch_col = rng.choice(touch_cols)
    width = max(pw, k)
    lo = max(0, touch_col - k + 1)
    hi = min(width - k, touch_col)
    swatch_c0 = rng.randint(lo, hi)
    patch = np.full((ph + k, width), bg, dtype=int)
    if side == "bottom":
        patch[:ph, :pw][pat] = color
        patch[ph : ph + k, swatch_c0 : swatch_c0 + k] = color
    else:
        patch[k : k + ph, :pw][pat] = color
        patch[:k, swatch_c0 : swatch_c0 + k] = color
    return patch


def _place_palette(
    grid: np.ndarray,
    occupied: np.ndarray,
    patches: list[np.ndarray],
    side: str,
    bg: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    gap = 1
    if side in ("right", "left"):
        total_h = sum(p.shape[0] for p in patches) + gap * (len(patches) - 1)
        if total_h > h:
            return False
        margin = h - total_h
        start = rng.randint(1, margin - 1) if margin >= 2 else rng.randint(0, margin)
        r = start
        for patch in patches:
            c = w - patch.shape[1] if side == "right" else 0
            _paste_patch(grid, patch, r, c, bg)
            occupied[max(0, r - 1) : min(h, r + patch.shape[0] + 1),
                     max(0, c - 1) : min(w, c + patch.shape[1] + 1)] = True
            r += patch.shape[0] + gap
        return True

    total_w = sum(p.shape[1] for p in patches) + gap * (len(patches) - 1)
    if total_w > w:
        return False
    margin = w - total_w
    start = rng.randint(1, margin - 1) if margin >= 2 else rng.randint(0, margin)
    c = start
    for patch in patches:
        r = h - patch.shape[0] if side == "bottom" else 0
        _paste_patch(grid, patch, r, c, bg)
        occupied[max(0, r - 1) : min(h, r + patch.shape[0] + 1),
                 max(0, c - 1) : min(w, c + patch.shape[1] + 1)] = True
        c += patch.shape[1] + gap
    return True


def _can_place_square(occupied: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
        return False
    return not bool(
        occupied[max(0, r0 - 1) : min(h, r0 + size + 1),
                 max(0, c0 - 1) : min(w, c0 + size + 1)].any()
    )


def _draw_input_frame(
    grid: np.ndarray, r0: int, c0: int, size: int, frame_color: int, fill_color: int
) -> None:
    r1, c1 = r0 + size - 1, c0 + size - 1
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill_color


def _place_frames(
    grid: np.ndarray,
    occupied: np.ndarray,
    fills: list[int],
    frame_color: int,
    rng: random.Random,
) -> list[dict] | None:
    h, w = grid.shape
    frames: list[dict] = []
    for fill_color in fills:
        size = rng.randint(5, 7)
        positions = [
            (r, c)
            for r in range(h - size + 1)
            for c in range(w - size + 1)
            if _can_place_square(occupied, r, c, size)
        ]
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _draw_input_frame(grid, r0, c0, size, frame_color, fill_color)
        occupied[max(0, r0 - 1) : min(h, r0 + size + 1),
                 max(0, c0 - 1) : min(w, c0 + size + 1)] = True
        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r0 + size - 1,
                "c1": c0 + size - 1,
                "fill": int(fill_color),
            }
        )
    return frames


def _expected_output(
    inp: np.ndarray, frames: list[dict], shapes: dict[int, np.ndarray], bg: int
) -> np.ndarray:
    frame_mask = _frame_area_mask(inp.shape, frames)
    out = inp.copy()
    out[(~frame_mask) & (out != bg)] = bg
    for frame in frames:
        fill = int(frame["fill"])
        _paint_centered_shape(out, frame, shapes[fill], fill, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    side = rng.choice(["right", "bottom"])

    all_colors = list(range(10))
    bg = rng.choice(all_colors)
    remaining = [c for c in all_colors if c != bg]
    frame_color = rng.choice(remaining)
    usable = [c for c in remaining if c != frame_color]

    num_colors = rng.randint(2, 4)
    fill_colors = rng.sample(usable, num_colors)
    patterns = [_trim_mask(p.copy()) for p in rng.sample(_PATTERNS, num_colors)]
    swatches = [rng.choice([1, 2]) for _ in range(num_colors)]

    patches = [
        _build_legend_patch(pattern, color, bg, swatch, side, rng)
        for pattern, color, swatch in zip(patterns, fill_colors, swatches)
    ]
    order = list(range(num_colors))
    rng.shuffle(order)
    patches = [patches[i] for i in order]

    if side == "right":
        need_h = sum(p.shape[0] for p in patches) + len(patches) - 1 + 2
        need_w = max(p.shape[1] for p in patches) + 10
    else:
        need_h = max(p.shape[0] for p in patches) + 10
        need_w = sum(p.shape[1] for p in patches) + len(patches) - 1 + 2

    if need_h > 30 or need_w > 30:
        return None
    h = rng.randint(max(16, need_h), 30)
    w = rng.randint(max(16, need_w), 30)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    if not _place_palette(grid, occupied, patches, side, bg, rng):
        return None

    num_frames = rng.randint(num_colors, min(num_colors + 2, 6))
    fills = list(fill_colors)
    while len(fills) < num_frames:
        fills.append(rng.choice(fill_colors))
    rng.shuffle(fills)

    frames = _place_frames(grid, occupied, fills, frame_color, rng)
    if frames is None:
        return None
    if _background(grid) != bg:
        return None

    shapes = {color: pattern for color, pattern in zip(fill_colors, patterns)}
    expected = _expected_output(grid, frames, shapes, bg)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
