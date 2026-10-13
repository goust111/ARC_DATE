import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(1, 10))

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (2, 4)
SHORT_SIDE_VALUES = (1, 2, 3)
WINDOW_INNER_MIN = 4
WINDOW_INNER_MAX = 11
EDGE_GAP = 1
OBJECT_PAD = 1
MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _component_bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "area": len(coords),
                }
            )
    return comps


def _is_hollow_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(comp["h"])
    w = int(comp["w"])
    color = int(comp["color"])
    if h < 3 or w < 3:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch == color, border) and np.all(patch[~border] == bg))


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == color))


def _detect_objects(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, dict]]:
    bg = _background_color(grid)
    frames: dict[int, dict] = {}
    keys: dict[int, dict] = {}

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        frame_candidates: list[dict] = []
        key_candidates: list[dict] = []
        for comp in _components_of_color(grid, color):
            if _is_hollow_frame(grid, comp, bg):
                frame_candidates.append(comp)
            elif _is_solid_rectangle(grid, comp) and int(comp["h"]) != int(comp["w"]):
                key_candidates.append(comp)

        if frame_candidates:
            frames[color] = max(frame_candidates, key=lambda item: int(item["area"]))
        if key_candidates:
            keys[color] = min(key_candidates, key=lambda item: int(item["area"]))

    return bg, frames, keys


def _paint_stripes(out: np.ndarray, frame: dict, horizontal: bool, gap: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    top = int(r0 + 1)
    left = int(c0 + 1)
    bottom = int(r1 - 1)
    right = int(c1 - 1)
    step = int(gap + 1)

    if horizontal:
        for r in range(top, bottom + 1, step):
            out[r, left : right + 1] = color
    else:
        for c in range(left, right + 1, step):
            out[top : bottom + 1, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each solid key rectangle with its same-color hollow window frame.
    The key's aspect ratio selects stripe orientation, and its shorter side
    selects the number of black rows or columns between stripes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, frames, keys = _detect_objects(g)

    for color in sorted(set(frames) & set(keys)):
        frame = frames[color]
        key = keys[color]
        horizontal = int(key["w"]) > int(key["h"])
        gap = min(int(key["h"]), int(key["w"]))
        _paint_stripes(out, frame, horizontal=horizontal, gap=gap)

    return out


def _make_frame_patch(color: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _make_key_patch(color: int, key_h: int, key_w: int) -> np.ndarray:
    return np.full((key_h, key_w), color, dtype=int)


def _sample_window_spec(rng: random.Random, horizontal: bool, gap: int) -> tuple[int, int]:
    min_inner_h = max(WINDOW_INNER_MIN, gap + 2 if horizontal else WINDOW_INNER_MIN)
    min_inner_w = max(WINDOW_INNER_MIN, gap + 2 if not horizontal else WINDOW_INNER_MIN)
    inner_h = rng.randint(min_inner_h, WINDOW_INNER_MAX)
    inner_w = rng.randint(min_inner_w, WINDOW_INNER_MAX)
    return inner_h, inner_w


def _fits(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _candidate_positions(occ: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    h0, w0 = occ.shape
    positions: list[tuple[int, int]] = []
    for r in range(EDGE_GAP, h0 - h - EDGE_GAP + 1):
        for c in range(EDGE_GAP, w0 - w - EDGE_GAP + 1):
            if _fits(occ, r, c, h, w, OBJECT_PAD):
                positions.append((r, c))
    return positions


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != 0
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        _paint_stripes(out, spec["frame"], horizontal=spec["horizontal"], gap=spec["gap"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    n_objects = rng.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])
    colors = rng.sample(COLOR_POOL, n_objects)

    orientations = [rng.random() < 0.5 for _ in range(n_objects)]
    if n_objects >= 2 and (all(orientations) or not any(orientations)):
        orientations[rng.randrange(n_objects)] = not orientations[0]

    raw_specs: list[dict] = []
    for color, horizontal in zip(colors, orientations):
        short = rng.choice(SHORT_SIDE_VALUES)
        long = rng.randint(short + 1, short + 4)
        key_h = short if horizontal else long
        key_w = long if horizontal else short
        inner_h, inner_w = _sample_window_spec(rng, horizontal, short)
        frame_patch = _make_frame_patch(color, inner_h, inner_w)
        key_patch = _make_key_patch(color, key_h, key_w)
        raw_specs.append(
            {
                "color": color,
                "horizontal": horizontal,
                "gap": short,
                "frame_patch": frame_patch,
                "key_patch": key_patch,
            }
        )

    for spec in sorted(raw_specs, key=lambda item: item["frame_patch"].size, reverse=True):
        fh, fw = spec["frame_patch"].shape
        frame_positions = _candidate_positions(occ, fh, fw)
        if not frame_positions:
            return None
        fr0, fc0 = frame_positions[rng.randrange(len(frame_positions))]
        _paste(grid, spec["frame_patch"], fr0, fc0)
        _mark(occ, fr0, fc0, fh, fw, OBJECT_PAD)
        spec["frame"] = {"color": spec["color"], "bbox": (fr0, fc0, fr0 + fh - 1, fc0 + fw - 1)}

    for spec in sorted(raw_specs, key=lambda item: item["key_patch"].size, reverse=True):
        kh, kw = spec["key_patch"].shape
        key_positions = _candidate_positions(occ, kh, kw)
        if not key_positions:
            return None
        kr0, kc0 = key_positions[rng.randrange(len(key_positions))]
        _paste(grid, spec["key_patch"], kr0, kc0)
        _mark(occ, kr0, kc0, kh, kw, OBJECT_PAD)

    bg, frames, keys = _detect_objects(grid)
    if bg != 0:
        return None
    if set(frames) != set(colors) or set(keys) != set(colors):
        return None

    expected = _build_expected(grid, raw_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid input")
