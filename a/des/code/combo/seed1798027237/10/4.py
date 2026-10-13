from __future__ import annotations

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")


def _components_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, np.asarray(cells, dtype=int)))
    return comps


def _missing_run(line: np.ndarray) -> tuple[int, int] | None:
    miss = np.flatnonzero(~line)
    if miss.size == 0:
        return (0, 0)
    if miss[-1] - miss[0] + 1 != miss.size:
        return None
    return int(miss[0]), int(miss.size)


def _analyze_frame_component(color: int, coords: np.ndarray) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    if patch[1:-1, 1:-1].any():
        return None
    if not (patch[0, 0] and patch[0, -1] and patch[-1, 0] and patch[-1, -1]):
        return None

    side_runs = {
        "top": _missing_run(patch[0, 1:-1]),
        "bottom": _missing_run(patch[-1, 1:-1]),
        "left": _missing_run(patch[1:-1, 0]),
        "right": _missing_run(patch[1:-1, -1]),
    }
    if any(run is None for run in side_runs.values()):
        return None

    missing = []
    for side, run in side_runs.items():
        assert run is not None
        _start, length = run
        if length > 0:
            missing.append((side, length))

    if len(missing) != 1:
        return None
    gap_side, gap_width = missing[0]
    if not (1 <= gap_width <= 3):
        return None

    expected_area = 2 * (h + w) - 4 - gap_width
    if int(len(coords)) != expected_area:
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "gap_width": int(gap_width),
    }


def _extract_sprite(grid: np.ndarray, frame: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != bg) & (inner != int(frame["color"]))
    if not mask.any():
        return None

    pts = np.argwhere(mask)
    sr0, sc0 = pts.min(axis=0)
    sr1, sc1 = pts.max(axis=0)
    sr0, sc0, sr1, sc1 = map(int, (sr0, sc0, sr1, sc1))
    patch = inner[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    if np.any((patch != bg) & (patch == int(frame["color"]))):
        return None

    return {
        "top": int(r0 + 1 + sr0),
        "left": int(c0 + 1 + sc0),
        "patch": patch,
    }


def _paste_clipped(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    for r in range(ph):
        rr = top + r
        if not (0 <= rr < h):
            continue
        for c in range(pw):
            cc = left + c
            if 0 <= cc < w and int(patch[r, c]) != bg:
                out[rr, cc] = int(patch[r, c])


def _apply_extrusion(out: np.ndarray, frame: dict, sprite: dict, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["gap_side"]
    count = int(frame["gap_width"])
    top = int(sprite["top"])
    left = int(sprite["left"])
    patch = np.asarray(sprite["patch"], dtype=int)
    ph, pw = patch.shape

    for step in range(1, count + 1):
        if side == "top":
            dest_top = r0 - step * ph
            dest_left = left
        elif side == "bottom":
            dest_top = r1 + 1 + (step - 1) * ph
            dest_left = left
        elif side == "left":
            dest_top = top
            dest_left = c0 - step * pw
        else:
            dest_top = top
            dest_left = c1 + 1 + (step - 1) * pw
        _paste_clipped(out, patch, int(dest_top), int(dest_left), bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect rectangular frames with one gap and extrude each inner sprite outward.

    The gap side gives the extrusion direction; the gap width gives the number
    of copies. Each copy keeps the original sprite unchanged and is shifted by
    one sprite-bbox width/height per step, clipped at the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    frames: list[tuple[dict, dict]] = []
    for color, coords in _components_by_color(g, bg):
        frame = _analyze_frame_component(color, coords)
        if frame is None:
            continue
        sprite = _extract_sprite(g, frame, bg)
        if sprite is None:
            continue
        frames.append((frame, sprite))

    frames.sort(key=lambda item: item[0]["bbox"])
    for frame, sprite in frames:
        _apply_extrusion(out, frame, sprite, bg)
    return out


GRID_H_RANGE = (18, 33)
GRID_W_RANGE = (18, 33)
FRAME_COUNT_RANGE = (1, 5)
FRAME_H_RANGE = (6, 12)
FRAME_W_RANGE = (6, 12)
SPRITE_THIN_RANGE = (1, 4)
MAX_PLACE_ATTEMPTS = 120
MAX_GEN_ATTEMPTS = 400
RESERVE_MARGIN = 1
EDGE_CLIP_PROB = 0.35


def _random_sprite_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    min_area = max(h, w)
    max_area = h * w
    for _ in range(80):
        area = int(rng.integers(min_area, max_area + 1))
        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) != h or len(cols) != w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return np.ones((h, w), dtype=bool)


def _make_frame_patch(h: int, w: int, color: int, side: str, start: int, gap: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, start : start + gap] = BG
    elif side == "bottom":
        patch[-1, start : start + gap] = BG
    elif side == "left":
        patch[start : start + gap, 0] = BG
    else:
        patch[start : start + gap, -1] = BG
    return patch


def _reserve_bounds(spec: dict, r0: int, c0: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    extra_top = spec["gap_width"] * spec["sprite_h"] if spec["gap_side"] == "top" else 0
    extra_bottom = spec["gap_width"] * spec["sprite_h"] if spec["gap_side"] == "bottom" else 0
    extra_left = spec["gap_width"] * spec["sprite_w"] if spec["gap_side"] == "left" else 0
    extra_right = spec["gap_width"] * spec["sprite_w"] if spec["gap_side"] == "right" else 0
    rr0 = max(0, r0 - extra_top - RESERVE_MARGIN)
    cc0 = max(0, c0 - extra_left - RESERVE_MARGIN)
    rr1 = min(h - 1, r0 + spec["frame_h"] - 1 + extra_bottom + RESERVE_MARGIN)
    cc1 = min(w - 1, c0 + spec["frame_w"] - 1 + extra_right + RESERVE_MARGIN)
    return int(rr0), int(cc0), int(rr1), int(cc1)


def _sample_origin(
    rng: np.random.Generator,
    spec: dict,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    frame_h = spec["frame_h"]
    frame_w = spec["frame_w"]
    row_lo, row_hi = 0, h - frame_h
    col_lo, col_hi = 0, w - frame_w

    if spec["gap_side"] == "top":
        row_lo = 1
    elif spec["gap_side"] == "bottom":
        row_hi = h - frame_h - 1
    elif spec["gap_side"] == "left":
        col_lo = 1
    else:
        col_hi = w - frame_w - 1

    if row_lo > row_hi or col_lo > col_hi:
        return None

    edge_bias = bool(spec["allow_clip"])
    if spec["gap_side"] == "top" and edge_bias:
        hi = min(row_hi, spec["gap_width"] * spec["sprite_h"])
        row = int(rng.integers(row_lo, hi + 1 if row_lo <= hi else row_hi + 1))
        col = int(rng.integers(col_lo, col_hi + 1))
        return row, col
    if spec["gap_side"] == "bottom" and edge_bias:
        lo = max(row_lo, row_hi - spec["gap_width"] * spec["sprite_h"])
        row = int(rng.integers(lo if lo <= row_hi else row_lo, row_hi + 1))
        col = int(rng.integers(col_lo, col_hi + 1))
        return row, col
    if spec["gap_side"] == "left" and edge_bias:
        hi = min(col_hi, spec["gap_width"] * spec["sprite_w"])
        row = int(rng.integers(row_lo, row_hi + 1))
        col = int(rng.integers(col_lo, hi + 1 if col_lo <= hi else col_hi + 1))
        return row, col
    if spec["gap_side"] == "right" and edge_bias:
        lo = max(col_lo, col_hi - spec["gap_width"] * spec["sprite_w"])
        row = int(rng.integers(row_lo, row_hi + 1))
        col = int(rng.integers(lo if lo <= col_hi else col_lo, col_hi + 1))
        return row, col

    row = int(rng.integers(row_lo, row_hi + 1))
    col = int(rng.integers(col_lo, col_hi + 1))
    return row, col


def _build_spec(rng: np.random.Generator, frame_color: int, sprite_color: int) -> dict:
    side = SIDES[int(rng.integers(0, len(SIDES)))]
    gap = int(rng.integers(1, 4))

    if side in ("top", "bottom"):
        sprite_w = int(rng.integers(1, min(gap, 3) + 1))
        sprite_h = int(rng.integers(1, SPRITE_THIN_RANGE[1] + 1))
        frame_h = int(rng.integers(max(FRAME_H_RANGE[0], sprite_h + 4), FRAME_H_RANGE[1] + 1))
        frame_w = int(rng.integers(max(FRAME_W_RANGE[0], gap + 4), FRAME_W_RANGE[1] + 1))
        gap_start = int(rng.integers(1, frame_w - gap))
        sprite_left = int(rng.integers(gap_start, gap_start + gap - sprite_w + 1))
        sprite_top = int(rng.integers(2, frame_h - sprite_h - 1))
    else:
        sprite_h = int(rng.integers(1, min(gap, 3) + 1))
        sprite_w = int(rng.integers(1, SPRITE_THIN_RANGE[1] + 1))
        frame_h = int(rng.integers(max(FRAME_H_RANGE[0], gap + 4), FRAME_H_RANGE[1] + 1))
        frame_w = int(rng.integers(max(FRAME_W_RANGE[0], sprite_w + 4), FRAME_W_RANGE[1] + 1))
        gap_start = int(rng.integers(1, frame_h - gap))
        sprite_top = int(rng.integers(gap_start, gap_start + gap - sprite_h + 1))
        sprite_left = int(rng.integers(2, frame_w - sprite_w - 1))

    sprite_mask = _random_sprite_mask(rng, sprite_h, sprite_w)
    sprite_patch = np.where(sprite_mask, sprite_color, BG).astype(int)
    reserve_h = frame_h + gap * sprite_h if side in ("top", "bottom") else frame_h
    reserve_w = frame_w + gap * sprite_w if side in ("left", "right") else frame_w

    return {
        "frame_color": int(frame_color),
        "sprite_color": int(sprite_color),
        "gap_side": side,
        "gap_width": int(gap),
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "gap_start": int(gap_start),
        "sprite_top_local": int(sprite_top),
        "sprite_left_local": int(sprite_left),
        "sprite_h": int(sprite_h),
        "sprite_w": int(sprite_w),
        "sprite_patch": sprite_patch,
        "allow_clip": bool(rng.random() < EDGE_CLIP_PROB),
        "reserve_area": int(reserve_h * reserve_w),
    }


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*GRID_H_RANGE))
    width = int(rng.integers(*GRID_W_RANGE))
    count = int(rng.integers(*FRAME_COUNT_RANGE))

    palette = rng.permutation(np.arange(1, 10)).tolist()
    if 2 * count > len(palette):
        return None

    specs = []
    for i in range(count):
        specs.append(_build_spec(rng, int(palette[2 * i]), int(palette[2 * i + 1])))
    specs.sort(key=lambda spec: -spec["reserve_area"])

    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    placed_specs: list[dict] = []

    for spec in specs:
        placed = None
        for _ in range(MAX_PLACE_ATTEMPTS):
            origin = _sample_origin(rng, spec, grid.shape)
            if origin is None:
                break
            r0, c0 = origin
            rr0, cc0, rr1, cc1 = _reserve_bounds(spec, r0, c0, grid.shape)
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            frame_patch = _make_frame_patch(
                spec["frame_h"],
                spec["frame_w"],
                spec["frame_color"],
                spec["gap_side"],
                spec["gap_start"],
                spec["gap_width"],
            )
            sprite_top = r0 + spec["sprite_top_local"]
            sprite_left = c0 + spec["sprite_left_local"]
            sprite_patch = spec["sprite_patch"]

            if np.any(grid[r0 : r0 + spec["frame_h"], c0 : c0 + spec["frame_w"]] != BG):
                continue
            if np.any(
                grid[
                    sprite_top : sprite_top + spec["sprite_h"],
                    sprite_left : sprite_left + spec["sprite_w"],
                ]
                != BG
            ):
                continue

            grid[r0 : r0 + spec["frame_h"], c0 : c0 + spec["frame_w"]] = frame_patch
            _paste_clipped(grid, sprite_patch, sprite_top, sprite_left, BG)
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

            placed = dict(spec)
            placed["frame_bbox"] = (
                int(r0),
                int(c0),
                int(r0 + spec["frame_h"] - 1),
                int(c0 + spec["frame_w"] - 1),
            )
            placed["sprite_top"] = int(sprite_top)
            placed["sprite_left"] = int(sprite_left)
            placed_specs.append(placed)
            break

        if placed is None:
            return None

    expected = grid.copy()
    for spec in placed_specs:
        frame = {
            "bbox": spec["frame_bbox"],
            "gap_side": spec["gap_side"],
            "gap_width": spec["gap_width"],
        }
        sprite = {
            "top": spec["sprite_top"],
            "left": spec["sprite_left"],
            "patch": spec["sprite_patch"],
        }
        _apply_extrusion(expected, frame, sprite, BG)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
