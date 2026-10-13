from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
FRAME_COUNT_RANGE = (3, 6)
INTERIOR_SIZE_RANGE = (3, 7)
FRAME_PADDING = 2
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_frame_component(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(patch == color, border)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg or color == GREY:
            continue
        for coords in _connected_components(grid == color):
            if _is_frame_component(grid, color, coords):
                r0, c0, r1, c1 = _bbox(coords)
                frames.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                    }
                )
    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one hollow rectangular frame"
    frame_colors = [frame["color"] for frame in frames]
    assert len(frame_colors) == len(set(frame_colors)), "frame colors must be unique"
    return frames


def _extract_frame_contents(grid: np.ndarray, frame: dict, bg: int) -> tuple[tuple[int, int] | None, dict | None]:
    ir0, ic0, ir1, ic1 = frame["interior_bbox"]
    interior = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]

    lock_positions = np.argwhere(interior == GREY)
    assert len(lock_positions) <= 1, "each frame may contain at most one lock marker"
    lock = None
    if len(lock_positions) == 1:
        lr, lc = map(int, lock_positions[0])
        lock = (ir0 + lr, ic0 + lc)

    key_mask = (interior != bg) & (interior != GREY)
    if not np.any(key_mask):
        return lock, None

    key_colors = {int(v) for v in interior[key_mask]}
    assert len(key_colors) == 1, "each frame may contain at most one key color"
    key_color = key_colors.pop()
    key_components = _connected_components(key_mask)
    assert len(key_components) == 1, "the key shape must be 4-connected"

    coords_local = key_components[0]
    anchor = min(coords_local)
    rel = sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in coords_local)
    coords_global = [(ir0 + int(r), ic0 + int(c)) for r, c in coords_local]
    return lock, {
        "target_color": key_color,
        "coords_global": coords_global,
        "relative": rel,
        "anchor_global": (ir0 + int(anchor[0]), ic0 + int(anchor[1])),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each key shape from its source frame into the lock-marked target frame whose border color
    matches the key color. The pasted shape uses the source frame's border color, not the key color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    by_color = {frame["color"]: frame for frame in frames}

    keys: list[dict] = []
    incoming_targets: set[int] = set()
    lock_by_color: dict[int, tuple[int, int]] = {}

    for frame in frames:
        lock, key = _extract_frame_contents(g, frame, bg)
        if lock is not None:
            lock_by_color[frame["color"]] = lock
        if key is not None:
            target_color = int(key["target_color"])
            assert target_color in by_color, "each key color must match a frame color"
            assert target_color not in incoming_targets, "each target frame should receive at most one key"
            incoming_targets.add(target_color)
            keys.append(
                {
                    "source_color": int(frame["color"]),
                    "target_color": target_color,
                    "coords_global": key["coords_global"],
                    "relative": key["relative"],
                }
            )

    out = g.copy()
    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["interior_bbox"]
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        interior[interior == GREY] = bg

    for key in keys:
        for r, c in key["coords_global"]:
            out[int(r), int(c)] = bg

    for key in keys:
        target_color = int(key["target_color"])
        assert target_color in lock_by_color, "each target frame for a key must contain a lock marker"
        lock_r, lock_c = lock_by_color[target_color]
        target_frame = by_color[target_color]
        ir0, ic0, ir1, ic1 = target_frame["interior_bbox"]
        for dr, dc in key["relative"]:
            rr = int(lock_r + dr)
            cc = int(lock_c + dc)
            assert ir0 <= rr <= ir1 and ic0 <= cc <= ic1, "stamped key must fit inside the target frame"
            out[rr, cc] = int(key["source_color"])

    return out


def _rectangles_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _random_frame_roles(rng: np.random.Generator, n_frames: int) -> list[str]:
    if rng.random() < 0.75:
        paired = int(round(n_frames / 3.0))
        paired = min(paired, n_frames // 2)
        roles = ["key_only"] * paired + ["lock_only"] * paired + ["both"] * (n_frames - 2 * paired)
    else:
        roles = ["both"] * n_frames
    rng.shuffle(roles)
    return roles


def _sample_frame_layout(rng: np.random.Generator) -> list[dict] | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))
    colors = [int(c) for c in rng.choice(np.array(FRAME_COLORS), size=n_frames, replace=False)]
    frames: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            ih = int(rng.integers(*INTERIOR_SIZE_RANGE))
            iw = int(rng.integers(*INTERIOR_SIZE_RANGE))
            oh = ih + 2
            ow = iw + 2
            r0 = int(rng.integers(0, h - oh + 1))
            c0 = int(rng.integers(0, w - ow + 1))
            bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            if any(_rectangles_intersect(bbox, other["bbox"], pad=FRAME_PADDING) for other in frames):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "interior_bbox": (r0 + 1, c0 + 1, r0 + ih, c0 + iw),
                    "interior_shape": (ih, iw),
                }
            )
            placed = True
            break
        if not placed:
            return None

    for frame, role in zip(frames, _random_frame_roles(rng, n_frames)):
        frame["role"] = role
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _sample_mapping(
    rng: np.random.Generator, sources: list[int], targets: list[int]
) -> dict[int, int] | None:
    assert len(sources) == len(targets)
    target_arr = np.array(targets, dtype=int)
    for _ in range(200):
        perm = [int(x) for x in rng.permutation(target_arr)]
        if all(src != tgt for src, tgt in zip(sources, perm)):
            return {int(src): int(tgt) for src, tgt in zip(sources, perm)}
    return None


def _grow_shape(
    rng: np.random.Generator, height: int, width: int, area: int
) -> list[tuple[int, int]] | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add((int(pick[0]), int(pick[1])))
    return sorted(cells)


def _sample_shape(
    rng: np.random.Generator, src_shape: tuple[int, int], dst_shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    max_h = min(int(src_shape[0]), int(dst_shape[0]), 4)
    max_w = min(int(src_shape[1]), int(dst_shape[1]), 4)
    candidates = [
        (h, w)
        for h in range(1, max_h + 1)
        for w in range(1, max_w + 1)
        if h * w >= 2
    ]
    rng.shuffle(candidates)
    for h, w in candidates:
        max_area = min(5, h * w)
        min_area = 2
        areas = list(range(min_area, max_area + 1))
        rng.shuffle(areas)
        for area in areas:
            shape = _grow_shape(rng, h, w, area)
            if shape is None:
                continue
            rows = [r for r, _ in shape]
            cols = [c for _, c in shape]
            if min(rows) != 0 or min(cols) != 0:
                continue
            if (0, 0) not in shape:
                continue
            return shape
    return None


def _all_shape_origins(
    interior_shape: tuple[int, int],
    shape: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    ih, iw = interior_shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    origins: list[tuple[int, int]] = []
    for r0 in range(ih - max_r):
        for c0 in range(iw - max_c):
            shifted = {(r0 + r, c0 + c) for r, c in shape}
            if shifted & forbidden:
                continue
            origins.append((int(r0), int(c0)))
    return origins


def _all_lock_positions(interior_shape: tuple[int, int], shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ih, iw = interior_shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    positions: list[tuple[int, int]] = []
    for r in range(ih - max_r):
        for c in range(iw - max_c):
            positions.append((int(r), int(c)))
    return positions


def _build_grid(rng: np.random.Generator) -> np.ndarray | None:
    frames = _sample_frame_layout(rng)
    if frames is None:
        return None

    sources = [i for i, frame in enumerate(frames) if frame["role"] in {"key_only", "both"}]
    targets = [i for i, frame in enumerate(frames) if frame["role"] in {"lock_only", "both"}]
    mapping = _sample_mapping(rng, sources, targets)
    if mapping is None:
        return None

    incoming_from = {tgt: src for src, tgt in mapping.items()}
    shapes: dict[int, list[tuple[int, int]]] = {}
    for src, tgt in mapping.items():
        shape = _sample_shape(rng, frames[src]["interior_shape"], frames[tgt]["interior_shape"])
        if shape is None:
            return None
        shapes[src] = shape

    for tgt, src in incoming_from.items():
        positions = _all_lock_positions(frames[tgt]["interior_shape"], shapes[src])
        if not positions:
            return None
        frames[tgt]["lock_local"] = positions[int(rng.integers(0, len(positions)))]

    for src in sources:
        frame = frames[src]
        forbidden: set[tuple[int, int]] = set()
        if "lock_local" in frame:
            forbidden.add(tuple(frame["lock_local"]))
        origins = _all_shape_origins(frame["interior_shape"], shapes[src], forbidden)
        if not origins:
            return None
        frame["key_origin"] = origins[int(rng.integers(0, len(origins)))]

    grid_h = max(frame["bbox"][2] for frame in frames) + 1
    grid_w = max(frame["bbox"][3] for frame in frames) + 1
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color

    for tgt, src in incoming_from.items():
        ir0, ic0, _ir1, _ic1 = frames[tgt]["interior_bbox"]
        lr, lc = frames[tgt]["lock_local"]
        grid[ir0 + int(lr), ic0 + int(lc)] = GREY

    for src in sources:
        frame = frames[src]
        ir0, ic0, _ir1, _ic1 = frame["interior_bbox"]
        origin_r, origin_c = frame["key_origin"]
        key_color = int(frames[mapping[src]]["color"])
        for dr, dc in shapes[src]:
            grid[ir0 + origin_r + dr, ic0 + origin_c + dc] = key_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid grid after {GENERATION_ATTEMPTS} attempts")
