from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PAYLOAD_COLORS = (3, 4, 5, 6, 7, 8, 9)

DIR_TO_ROT = {
    "right": 0,
    "down": 3,   # clockwise 90
    "left": 2,
    "up": 1,     # counter-clockwise 90
}

ARROW_VARIANTS = {
    "right": [
        ((3, 2), {(0, 0), (1, 1), (2, 0)}),
    ],
    "left": [
        ((3, 2), {(0, 1), (1, 0), (2, 1)}),
    ],
    "up": [
        ((2, 3), {(1, 0), (0, 1), (1, 2)}),
    ],
    "down": [
        ((2, 3), {(0, 0), (1, 1), (0, 2)}),
    ],
}

GEN_RETRIES = 400
PAIR_COUNT_RANGE = (2, 6)  # high exclusive
INNER_SIZE_RANGE = (5, 9)  # high exclusive
GRID_MAX = 35


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components4(grid == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != border:
            continue
        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "outer_shape": (r1 - r0 + 1, c1 - c0 + 1),
                "inner_shape": (r1 - r0 - 1, c1 - c0 - 1),
                "interior": grid[r0 + 1 : r1, c0 + 1 : c1].copy(),
            }
        )
    frames.sort(key=lambda fr: (fr["r0"], fr["c0"]))
    return frames


def _detect_direction(interior: np.ndarray) -> str | None:
    coords = np.argwhere(interior == RED)
    if len(coords) not in (2, 3):
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}

    for direction, variants in ARROW_VARIANTS.items():
        for _, shape in variants:
            if len(shape) == len(coords) and rel == shape:
                if len(coords) == 3:
                    return direction

    if len(coords) == 2:
        if rel == {(0, 0), (1, 1)}:
            score_right = c0
            score_down = r0
            return "right" if score_right <= score_down else "down"
        if rel == {(0, 1), (1, 0)}:
            score_left = interior.shape[1] - 1 - c1
            score_up = interior.shape[0] - 1 - r1
            return "left" if score_left <= score_up else "up"
        if rel == {(0, 0), (0, 1)}:
            return "right" if c0 <= interior.shape[1] - 1 - c1 else "left"
        if rel == {(0, 0), (1, 0)}:
            return "down" if r0 <= interior.shape[0] - 1 - r1 else "up"
    return None


def _extract_payload(interior: np.ndarray) -> np.ndarray | None:
    mask = (interior != BLACK) & (interior != BLUE) & (interior != RED)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return interior[r0 : r1 + 1, c0 : c1 + 1].copy()


def _classify_frames(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    loaded: list[dict] = []
    empty: list[dict] = []
    for frame in _find_frames(grid):
        interior = frame["interior"]
        if np.all(interior == BLACK):
            empty.append(frame)
            continue
        direction = _detect_direction(interior)
        payload = _extract_payload(interior)
        if direction is None or payload is None:
            continue
        item = dict(frame)
        item["direction"] = direction
        item["payload"] = payload
        loaded.append(item)
    return loaded, empty


def _pair_frames(loaded: list[dict], empty: list[dict]) -> list[tuple[dict, dict]]:
    remaining = set(range(len(empty)))
    pairs: list[tuple[dict, dict]] = []
    for frame in sorted(loaded, key=lambda fr: (fr["r0"], fr["c0"])):
        candidates: list[tuple[int, int, int, int]] = []
        for idx, other in enumerate(empty):
            if idx not in remaining or other["outer_shape"] != frame["outer_shape"]:
                continue
            dist = None
            if frame["direction"] == "right" and other["r0"] == frame["r0"] and other["c0"] > frame["c0"]:
                dist = other["c0"] - frame["c0"]
            elif frame["direction"] == "left" and other["r0"] == frame["r0"] and other["c0"] < frame["c0"]:
                dist = frame["c0"] - other["c0"]
            elif frame["direction"] == "down" and other["c0"] == frame["c0"] and other["r0"] > frame["r0"]:
                dist = other["r0"] - frame["r0"]
            elif frame["direction"] == "up" and other["c0"] == frame["c0"] and other["r0"] < frame["r0"]:
                dist = frame["r0"] - other["r0"]
            if dist is not None:
                candidates.append((dist, other["r0"], other["c0"], idx))
        if not candidates:
            continue
        _, _, _, idx = min(candidates)
        remaining.remove(idx)
        pairs.append((frame, empty[idx]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each payload from a blue frame into the nearest same-sized empty frame
    along its red arrow direction, rotating the payload to match the direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    loaded, empty = _classify_frames(g)

    for src, dst in _pair_frames(loaded, empty):
        sr0, sc0, sr1, sc1 = src["r0"], src["c0"], src["r1"], src["c1"]
        out[sr0 + 1 : sr1, sc0 + 1 : sc1] = BLACK

        payload = np.rot90(src["payload"], DIR_TO_ROT[src["direction"]])
        inner_h, inner_w = dst["inner_shape"]
        ph, pw = payload.shape
        if ph > inner_h or pw > inner_w:
            continue
        top = dst["r0"] + 1 + (inner_h - ph) // 2
        left = dst["c0"] + 1 + (inner_w - pw) // 2
        patch = out[top : top + ph, left : left + pw]
        mask = payload != BLACK
        patch[mask] = payload[mask]

    return out


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r0 + h - 1 + pad)
    cc1 = min(occupied.shape[1] - 1, c0 + w - 1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rect_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _balanced_directions(rng: np.random.Generator, n: int) -> list[str]:
    base = ["right", "down", "left", "up"]
    out: list[str] = []
    while len(out) < n:
        block = list(base)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def _payload_patch(rng: np.random.Generator, inner_h: int, inner_w: int) -> np.ndarray | None:
    size_cap = min(4, inner_h, inner_w)
    if size_cap < 2:
        return None
    for _ in range(120):
        ph = int(rng.integers(2, size_cap + 1))
        pw = int(rng.integers(2, size_cap + 1))
        area = int(rng.integers(3, min(ph * pw, 6) + 1))

        cells = {(int(rng.integers(ph)), int(rng.integers(pw)))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            cells.add((nr, nc))

        if len(cells) != area:
            continue

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        crop_h = max(rs) - min(rs) + 1
        crop_w = max(cs) - min(cs) + 1
        patch = np.full((crop_h, crop_w), BLACK, dtype=int)
        color = int(rng.choice(PAYLOAD_COLORS))
        for r, c in cells:
            patch[r - min(rs), c - min(cs)] = color

        symmetric = False
        for k in (1, 2, 3):
            rot = np.rot90(patch, k)
            if patch.shape == rot.shape and np.array_equal(patch, rot):
                symmetric = True
                break
        if symmetric:
            continue
        return patch
    return None


def _arrow_cells(rng: np.random.Generator, inner_h: int, inner_w: int, direction: str) -> set[tuple[int, int]] | None:
    variants = ARROW_VARIANTS[direction]
    (ah, aw), rel = variants[int(rng.integers(len(variants)))]
    if inner_h < ah or inner_w < aw:
        return None

    row_limit = inner_h - ah
    col_limit = inner_w - aw

    def near_low(limit: int) -> int:
        return int(rng.integers(0, min(2, limit) + 1))

    def near_high(limit: int) -> int:
        return int(limit - rng.integers(0, min(2, limit) + 1))

    if direction == "right":
        top = int(rng.integers(0, row_limit + 1))
        left = near_low(col_limit)
    elif direction == "left":
        top = int(rng.integers(0, row_limit + 1))
        left = near_high(col_limit)
    elif direction == "down":
        top = near_low(row_limit)
        left = int(rng.integers(0, col_limit + 1))
    else:
        top = near_high(row_limit)
        left = int(rng.integers(0, col_limit + 1))
    return {(top + r, left + c) for r, c in rel}


def _expanded_cells(cells: set[tuple[int, int]], inner_h: int, inner_w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < inner_h and 0 <= nc < inner_w:
                    blocked.add((nr, nc))
    return blocked


def _place_pair_positions(
    rng: np.random.Generator,
    occupied: np.ndarray,
    outer_h: int,
    outer_w: int,
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = occupied.shape
    for _ in range(200):
        gap = int(rng.integers(1, 7))
        if direction in ("right", "left"):
            total_w = outer_w * 2 + gap
            if outer_h + 2 > h or total_w + 2 > w:
                continue
            max_r0 = h - outer_h - 1
            max_c0 = w - total_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            r0 = int(rng.integers(1, max_r0 + 1))
            left_c = int(rng.integers(1, max_c0 + 1))
            a = (r0, left_c)
            b = (r0, left_c + outer_w + gap)
            src, dst = (a, b) if direction == "right" else (b, a)
        else:
            total_h = outer_h * 2 + gap
            if total_h + 2 > h or outer_w + 2 > w:
                continue
            max_r0 = h - total_h - 1
            max_c0 = w - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            top_r = int(rng.integers(1, max_r0 + 1))
            c0 = int(rng.integers(1, max_c0 + 1))
            a = (top_r, c0)
            b = (top_r + outer_h + gap, c0)
            src, dst = (a, b) if direction == "down" else (b, a)

        if not _rect_free(occupied, src[0], src[1], outer_h, outer_w):
            continue
        if not _rect_free(occupied, dst[0], dst[1], outer_h, outer_w):
            continue
        _mark_rect(occupied, src[0], src[1], outer_h, outer_w)
        _mark_rect(occupied, dst[0], dst[1], outer_h, outer_w)
        return src, dst
    return None


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = BLUE
    grid[r0 + h - 1, c0 : c0 + w] = BLUE
    grid[r0 : r0 + h, c0] = BLUE
    grid[r0 : r0 + h, c0 + w - 1] = BLUE


def _fill_loaded_frame(
    grid: np.ndarray,
    rng: np.random.Generator,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    direction: str,
) -> bool:
    inner_h = outer_h - 2
    inner_w = outer_w - 2
    arrow = _arrow_cells(rng, inner_h, inner_w, direction)
    payload = _payload_patch(rng, inner_h, inner_w)
    if arrow is None or payload is None:
        return False

    ph, pw = payload.shape
    blocked = _expanded_cells(arrow, inner_h, inner_w)
    placements: list[tuple[int, int]] = []
    for pr in range(inner_h - ph + 1):
        for pc in range(inner_w - pw + 1):
            ok = True
            for rr in range(pr, pr + ph):
                for cc in range(pc, pc + pw):
                    if (rr, cc) in blocked:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                placements.append((pr, pc))
    if not placements:
        return False

    pr, pc = placements[int(rng.integers(len(placements)))]
    for rr, cc in arrow:
        grid[r0 + 1 + rr, c0 + 1 + cc] = RED
    for rr in range(ph):
        for cc in range(pw):
            val = int(payload[rr, cc])
            if val != BLACK:
                grid[r0 + 1 + pr + rr, c0 + 1 + pc + cc] = val
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.integers(*PAIR_COUNT_RANGE))
    size_pool = [(ih + 2, iw + 2) for ih in range(*INNER_SIZE_RANGE) for iw in range(*INNER_SIZE_RANGE)]
    rng.shuffle(size_pool)
    pair_sizes = size_pool[:n_pairs]
    directions = _balanced_directions(rng, n_pairs)

    base = 20 + 2 * n_pairs
    h = int(rng.integers(base, GRID_MAX + 1))
    w = int(rng.integers(base, GRID_MAX + 1))
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    specs: list[tuple[tuple[int, int], tuple[int, int], int, int, str]] = []
    for (outer_h, outer_w), direction in zip(pair_sizes, directions):
        placed = _place_pair_positions(rng, occupied, outer_h, outer_w, direction)
        if placed is None:
            return None
        src, dst = placed
        specs.append((src, dst, outer_h, outer_w, direction))

    for src, dst, outer_h, outer_w, _ in specs:
        _draw_frame(grid, src[0], src[1], outer_h, outer_w)
        _draw_frame(grid, dst[0], dst[1], outer_h, outer_w)

    for src, _, outer_h, outer_w, direction in specs:
        if not _fill_loaded_frame(grid, rng, src[0], src[1], outer_h, outer_w, direction):
            return None

    loaded, empty = _classify_frames(grid)
    if len(loaded) != n_pairs or len(empty) != n_pairs:
        return None
    if len(_pair_frames(loaded, empty)) != n_pairs:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
