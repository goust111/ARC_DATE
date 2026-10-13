from __future__ import annotations

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rectangles(grid: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray]:
    rects: list[dict[str, int]] = []
    rect_id = -np.ones(grid.shape, dtype=int)
    for comp in _components(grid == ORANGE):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if len(comp) != height * width:
            continue
        if not bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE)):
            continue
        idx = len(rects)
        rects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(height),
                "w": int(width),
            }
        )
        rect_id[r0 : r1 + 1, c0 : c1 + 1] = idx
    return rects, rect_id


def _bar_orientation(coords: list[tuple[int, int]]) -> str | None:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        return "h"
    if len(cols) == 1:
        return "v"
    return None


def _bar_side(
    coords: list[tuple[int, int]], orientation: str, rect: dict[str, int]
) -> str | None:
    if orientation == "h":
        row = coords[0][0]
        if row == rect["r0"] - 1:
            return "top"
        if row == rect["r1"] + 1:
            return "bottom"
    else:
        col = coords[0][1]
        if col == rect["c0"] - 1:
            return "left"
        if col == rect["c1"] + 1:
            return "right"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn grey bars into yellow, then convert each bar's pixel count into yellow
    edge-aligned stripes inside the adjacent orange rectangle.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == GREY] = YELLOW

    rects, rect_id = _find_rectangles(g)
    info = [
        {
            "orientation": None,
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0,
        }
        for _ in rects
    ]

    for comp in _components(g == GREY):
        orientation = _bar_orientation(comp)
        if orientation is None:
            continue

        touching: set[int] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    rid = int(rect_id[nr, nc])
                    if rid >= 0:
                        touching.add(rid)

        if len(touching) != 1:
            continue

        rid = next(iter(touching))
        side = _bar_side(comp, orientation, rects[rid])
        if side is None:
            continue

        rect_info = info[rid]
        if rect_info["orientation"] is None:
            rect_info["orientation"] = orientation
        if rect_info["orientation"] != orientation:
            continue
        rect_info[side] += len(comp)

    for rect, rect_info in zip(rects, info):
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        if rect_info["orientation"] == "h":
            top = min(int(rect_info["top"]), rect["h"])
            bottom = min(int(rect_info["bottom"]), rect["h"])
            if top:
                out[r0 : r0 + top, c0 : c1 + 1] = YELLOW
            if bottom:
                out[r1 - bottom + 1 : r1 + 1, c0 : c1 + 1] = YELLOW
        elif rect_info["orientation"] == "v":
            left = min(int(rect_info["left"]), rect["w"])
            right = min(int(rect_info["right"]), rect["w"])
            if left:
                out[r0 : r1 + 1, c0 : c0 + left] = YELLOW
            if right:
                out[r0 : r1 + 1, c1 - right + 1 : c1 + 1] = YELLOW

    return out


# --------------- generator ---------------

GRID_H_RANGE = (16, 31)
GRID_W_RANGE = (16, 31)
RECT_COUNT_RANGE = (2, 6)
RECT_H_RANGE = (3, 9)
RECT_W_RANGE = (3, 10)
MAX_BAR_RUNS = 3
PLACEMENT_PAD = 1
GEN_ATTEMPTS = 300


def _positive_split(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    values = np.diff(np.concatenate(([0], cuts, [total])))
    return [int(v) for v in values]


def _run_intervals(
    span: int, total_len: int, runs: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    if runs < 1 or total_len < runs or total_len + runs - 1 > span:
        return None

    seg_lengths = _positive_split(total_len, runs, rng)
    extra = list(
        map(
            int,
            rng.multinomial(
                span - total_len - (runs - 1),
                np.full(runs + 1, 1.0 / (runs + 1)),
            ),
        )
    )

    pos = extra[0]
    intervals: list[tuple[int, int]] = []
    for idx, seg_len in enumerate(seg_lengths):
        intervals.append((int(pos), int(seg_len)))
        pos += seg_len
        if idx < runs - 1:
            pos += 1 + extra[idx + 1]
    return intervals


def _build_cluster(
    rng: np.random.Generator, touched: bool, force_orientation: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    rect_h = int(rng.integers(*RECT_H_RANGE))
    rect_w = int(rng.integers(*RECT_W_RANGE))

    if not touched:
        patch = np.full((rect_h, rect_w), ORANGE, dtype=int)
        return patch.copy(), patch

    orientation = force_orientation or ("h" if rng.random() < 0.5 else "v")
    stripe_count = int(rng.integers(1, min(rect_h, rect_w) + 1))

    if orientation == "h":
        side = "top" if rng.random() < 0.5 else "bottom"
        max_runs = min(MAX_BAR_RUNS, stripe_count, rect_w - stripe_count + 1)
        runs = int(rng.integers(1, max_runs + 1))
        intervals = _run_intervals(rect_w, stripe_count, runs, rng)
        if intervals is None:
            raise RuntimeError("invalid horizontal bar layout")

        inp = np.full((rect_h + 1, rect_w), BLACK, dtype=int)
        out = np.full((rect_h + 1, rect_w), BLACK, dtype=int)
        if side == "top":
            inp[1:, :] = ORANGE
            out[1:, :] = ORANGE
            out[1 : 1 + stripe_count, :] = YELLOW
            bar_row = 0
        else:
            inp[:-1, :] = ORANGE
            out[:-1, :] = ORANGE
            out[rect_h - stripe_count : rect_h, :] = YELLOW
            bar_row = rect_h

        for start, length in intervals:
            inp[bar_row, start : start + length] = GREY
            out[bar_row, start : start + length] = YELLOW
        return inp, out

    side = "left" if rng.random() < 0.5 else "right"
    max_runs = min(MAX_BAR_RUNS, stripe_count, rect_h - stripe_count + 1)
    runs = int(rng.integers(1, max_runs + 1))
    intervals = _run_intervals(rect_h, stripe_count, runs, rng)
    if intervals is None:
        raise RuntimeError("invalid vertical bar layout")

    inp = np.full((rect_h, rect_w + 1), BLACK, dtype=int)
    out = np.full((rect_h, rect_w + 1), BLACK, dtype=int)
    if side == "left":
        inp[:, 1:] = ORANGE
        out[:, 1:] = ORANGE
        out[:, 1 : 1 + stripe_count] = YELLOW
        bar_col = 0
    else:
        inp[:, :-1] = ORANGE
        out[:, :-1] = ORANGE
        out[:, rect_w - stripe_count : rect_w] = YELLOW
        bar_col = rect_w

    for start, length in intervals:
        inp[start : start + length, bar_col] = GREY
        out[start : start + length, bar_col] = YELLOW
    return inp, out


def _place_patch(
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    reserved: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    h, w = canvas_in.shape
    ph, pw = patch_in.shape
    positions: list[tuple[int, int]] = []

    for top in range(1, h - ph):
        for left in range(1, w - pw):
            r0 = max(0, top - PLACEMENT_PAD)
            c0 = max(0, left - PLACEMENT_PAD)
            r1 = min(h, top + ph + PLACEMENT_PAD)
            c1 = min(w, left + pw + PLACEMENT_PAD)
            if bool(reserved[r0:r1, c0:c1].any()):
                continue
            positions.append((int(top), int(left)))

    if not positions:
        return False

    top, left = positions[int(rng.integers(0, len(positions)))]
    canvas_in[top : top + ph, left : left + pw] = patch_in
    canvas_out[top : top + ph, left : left + pw] = patch_out
    reserved[top : top + ph, left : left + pw] = True
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rects = int(rng.integers(*RECT_COUNT_RANGE))
    touched = rng.random(n_rects) < 0.75
    if not bool(touched.any()):
        touched[int(rng.integers(0, n_rects))] = True
    if bool(touched.all()) and n_rects > 1 and rng.random() < 0.6:
        touched[int(rng.integers(0, n_rects))] = False

    clusters: list[tuple[np.ndarray, np.ndarray]] = []
    touched_ids = [idx for idx, flag in enumerate(touched) if flag]
    forced: dict[int, str] = {}
    if len(touched_ids) >= 2 and rng.random() < 0.5:
        forced[touched_ids[0]] = "h"
        forced[touched_ids[1]] = "v"

    for idx in range(n_rects):
        clusters.append(_build_cluster(rng, bool(touched[idx]), forced.get(idx)))

    clusters.sort(key=lambda pair: pair[0].shape[0] * pair[0].shape[1], reverse=True)

    min_h = max(16, max(p[0].shape[0] for p in clusters) + 2, int(sum(p[0].shape[0] for p in clusters) * 0.55))
    min_w = max(16, max(p[0].shape[1] for p in clusters) + 2, int(sum(p[0].shape[1] for p in clusters) * 0.55))
    if min_h >= GRID_H_RANGE[1] or min_w >= GRID_W_RANGE[1]:
        return None

    h = int(rng.integers(min_h, GRID_H_RANGE[1]))
    w = int(rng.integers(min_w, GRID_W_RANGE[1]))
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for patch_in, patch_out in clusters:
        if not _place_patch(inp, out, reserved, patch_in, patch_out, rng):
            return None

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
