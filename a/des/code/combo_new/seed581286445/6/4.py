from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 400


def _get_bg(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(g: np.ndarray, ignore: set[int]) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in ignore:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _find_separator(g: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        (r, int(g[r, 0]))
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != int(bg)
    ]
    assert len(candidates) == 1, f"expected one separator row, got {candidates}"
    return int(candidates[0][0]), int(candidates[0][1])


def _frame_gap(panel: np.ndarray, frame: dict) -> dict:
    color = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    sides = {
        "top": panel[r0, c0 : c1 + 1] == color,
        "bottom": panel[r1, c0 : c1 + 1] == color,
        "left": panel[r0 : r1 + 1, c0] == color,
        "right": panel[r0 : r1 + 1, c1] == color,
    }

    gap_infos: list[dict] = []
    for side, arr in sides.items():
        assert bool(arr[0]) and bool(arr[-1]), "frame corners must be present"
        interior = arr[1:-1]
        missing = np.flatnonzero(~interior)
        if missing.size == 0:
            continue
        assert int(missing[-1] - missing[0] + 1) == int(missing.size), "gap must be contiguous"
        start = int(missing[0]) + 1
        end = int(missing[-1]) + 1
        if side in ("top", "bottom"):
            gap_infos.append(
                {
                    "side": side,
                    "span_start": int(c0 + start),
                    "span_end": int(c0 + end),
                    "gap_len": int(end - start + 1),
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                }
            )
        else:
            gap_infos.append(
                {
                    "side": side,
                    "span_start": int(r0 + start),
                    "span_end": int(r0 + end),
                    "gap_len": int(end - start + 1),
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                }
            )

    assert len(gap_infos) == 1, f"expected one frame gap, got {gap_infos}"
    return gap_infos[0]


def _block_shape(block: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, block["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    patch = np.zeros((h, w), dtype=int)
    coords = block["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = 1
    assert bool(patch.all()), "bottom object must be a solid rectangle"
    return h, w


def _paint_insert(out: np.ndarray, frame_gap: dict, block_h: int, block_w: int) -> None:
    r0, c0, r1, c1 = map(int, frame_gap["bbox"])
    color = int(frame_gap["color"])
    side = frame_gap["side"]
    span_start = int(frame_gap["span_start"])
    gap_len = int(frame_gap["gap_len"])

    if side in ("top", "bottom"):
        assert block_w == gap_len, "horizontal gap expects matching block width"
        place_c0 = span_start
        place_r0 = int(r1 - block_h) if side == "top" else int(r0 + 1)
    else:
        assert block_h == gap_len, "vertical gap expects matching block height"
        place_r0 = span_start
        place_c0 = int(c1 - block_w) if side == "left" else int(c0 + 1)

    out[place_r0 : place_r0 + block_h, place_c0 : place_c0 + block_w] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each lower solid block into the same-colored upper frame through that frame's opening.

    Input structure:
    - The grid has a black background and one full-width grey separator row.
    - Above the separator are several one-pixel-thick colored rectangular frames, each with exactly one contiguous gap on one side.
    - Below the separator are solid colored rectangles, one per frame color.

    Rule:
    - Split the grid at the separator and discard the separator plus the lower panel in the output.
    - Match every lower rectangle to the upper frame of the same color.
    - Detect which side of the frame contains the gap, then slide the rectangle straight through that gap so it stays aligned with the opening and ends flush against the opposite inner wall.

    Key insight:
    - The missing border segment is not damage to repair; it is an entrance whose side fixes both the insertion direction and the rectangle's final resting wall.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_row, sep_color = _find_separator(g, bg)

    top = g[:sep_row, :].copy()
    bottom = g[sep_row + 1 :, :]

    frames = _find_components(top, ignore={bg, sep_color})
    blocks = _find_components(bottom, ignore={bg, sep_color})

    frame_by_color = {int(frame["color"]): frame for frame in frames}
    block_by_color = {int(block["color"]): block for block in blocks}
    assert frame_by_color.keys() == block_by_color.keys(), "frame/block colors must match exactly"

    out = top.copy()
    for color, frame in frame_by_color.items():
        gap = _frame_gap(top, frame)
        block_h, block_w = _block_shape(block_by_color[int(color)])
        _paint_insert(out, gap, block_h, block_w)

    return out


def _can_place(
    occ: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    *,
    padding: int,
) -> bool:
    rr0 = max(0, int(r0) - int(padding))
    cc0 = max(0, int(c0) - int(padding))
    rr1 = min(occ.shape[0], int(r0 + h + padding))
    cc1 = min(occ.shape[1], int(c0 + w + padding))
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _place_rectangles(
    rng: np.random.Generator,
    h: int,
    w: int,
    sizes: list[tuple[int, int]],
    *,
    padding: int,
    edge_margin: int,
    tries_per_rect: int = 300,
) -> list[tuple[int, int]] | None:
    occ = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        rh, rw = sizes[idx]
        r_min = edge_margin
        c_min = edge_margin
        r_max = h - rh - edge_margin
        c_max = w - rw - edge_margin
        if r_max < r_min or c_max < c_min:
            return None

        placed = False
        for _ in range(tries_per_rect):
            r0 = int(rng.integers(r_min, r_max + 1))
            c0 = int(rng.integers(c_min, c_max + 1))
            if _can_place(occ, r0, c0, rh, rw, padding=padding):
                _mark_place(occ, r0, c0, rh, rw)
                placements[idx] = (r0, c0)
                placed = True
                break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _center_gap_start(side_len: int, gap_len: int) -> int:
    usable = side_len - 2
    assert 1 <= gap_len <= usable
    return 1 + (usable - gap_len) // 2


def _make_frame_patch(color: int, h: int, w: int, side: str, gap_len: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    if side == "top":
        start = _center_gap_start(w, gap_len)
        patch[0, start : start + gap_len] = BG
    elif side == "bottom":
        start = _center_gap_start(w, gap_len)
        patch[-1, start : start + gap_len] = BG
    elif side == "left":
        start = _center_gap_start(h, gap_len)
        patch[start : start + gap_len, 0] = BG
    else:
        start = _center_gap_start(h, gap_len)
        patch[start : start + gap_len, -1] = BG

    return patch


def _frame_gap_from_spec(r0: int, c0: int, h: int, w: int, color: int, side: str, gap_len: int) -> dict:
    if side in ("top", "bottom"):
        start = _center_gap_start(w, gap_len)
        span_start = int(c0 + start)
        span_end = int(span_start + gap_len - 1)
    else:
        start = _center_gap_start(h, gap_len)
        span_start = int(r0 + start)
        span_end = int(span_start + gap_len - 1)

    return {
        "side": side,
        "span_start": span_start,
        "span_end": span_end,
        "gap_len": int(gap_len),
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "color": int(color),
    }


def _sample_specs(rng: np.random.Generator, n_frames: int, colors: list[int]) -> list[dict]:
    specs: list[dict] = []
    for color in colors:
        side = SIDES[int(rng.integers(0, len(SIDES)))]
        outer_h = int(rng.integers(5, 10))
        outer_w = int(rng.integers(5, 10))
        inner_h = outer_h - 2
        inner_w = outer_w - 2

        if side in ("top", "bottom"):
            gap_len = int(rng.integers(1, inner_w + 1))
            block_h = int(rng.integers(1, inner_h))
            block_w = gap_len
        else:
            gap_len = int(rng.integers(1, inner_h + 1))
            block_h = gap_len
            block_w = int(rng.integers(1, inner_w))

        specs.append(
            {
                "color": int(color),
                "side": side,
                "frame_h": outer_h,
                "frame_w": outer_w,
                "gap_len": gap_len,
                "block_h": block_h,
                "block_w": block_w,
            }
        )

    return specs


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(2, 5))
    colors = [int(x) for x in rng.choice(np.array(COLOR_POOL, dtype=int), size=n_frames, replace=False)]
    specs = _sample_specs(rng, n_frames, colors)

    max_frame_h = max(int(spec["frame_h"]) for spec in specs)
    max_block_h = max(int(spec["block_h"]) for spec in specs)
    width = int(rng.integers(max(18, max(int(spec["frame_w"]) for spec in specs) + 4), 31))

    top_h_min = max(max_frame_h + 2, 10)
    top_h_max = min(18, 29 - (max_block_h + 3))
    if top_h_min > top_h_max:
        return None
    top_h = int(rng.integers(top_h_min, top_h_max + 1))

    bottom_h_min = max(max_block_h + 2, 5)
    bottom_h_max = min(10, 29 - top_h)
    if bottom_h_min > bottom_h_max:
        return None
    bottom_h = int(rng.integers(bottom_h_min, bottom_h_max + 1))

    frame_sizes = [(int(spec["frame_h"]), int(spec["frame_w"])) for spec in specs]
    block_sizes = [(int(spec["block_h"]), int(spec["block_w"])) for spec in specs]

    frame_pos = _place_rectangles(rng, top_h, width, frame_sizes, padding=1, edge_margin=1)
    if frame_pos is None:
        return None
    block_pos = _place_rectangles(rng, bottom_h, width, block_sizes, padding=1, edge_margin=1)
    if block_pos is None:
        return None

    top = np.zeros((top_h, width), dtype=int)
    bottom = np.zeros((bottom_h, width), dtype=int)
    out = top.copy()

    for spec, (fr, fc), (br, bc) in zip(specs, frame_pos, block_pos):
        color = int(spec["color"])
        frame_patch = _make_frame_patch(
            color,
            int(spec["frame_h"]),
            int(spec["frame_w"]),
            str(spec["side"]),
            int(spec["gap_len"]),
        )
        top[fr : fr + frame_patch.shape[0], fc : fc + frame_patch.shape[1]] = frame_patch
        out[fr : fr + frame_patch.shape[0], fc : fc + frame_patch.shape[1]] = frame_patch

        bottom[
            br : br + int(spec["block_h"]),
            bc : bc + int(spec["block_w"]),
        ] = color

        gap = _frame_gap_from_spec(
            fr,
            fc,
            int(spec["frame_h"]),
            int(spec["frame_w"]),
            color,
            str(spec["side"]),
            int(spec["gap_len"]),
        )
        _paint_insert(out, gap, int(spec["block_h"]), int(spec["block_w"]))

    sep_row = np.full((1, width), SEP, dtype=int)
    inp = np.vstack([top, sep_row, bottom])

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
