import random
from collections import defaultdict, deque

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- helpers ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_segment(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    pts = np.array(comp, dtype=int)
    rows = pts[:, 0]
    cols = pts[:, 1]
    colors = {int(grid[r, c]) for r, c in comp}
    assert len(colors) == 1
    color = colors.pop()
    assert color != BG

    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h == 1:
        ordered = sorted(int(c) for c in cols)
        assert len(comp) == w
        assert ordered == list(range(c0, c1 + 1))
        return {
            "color": color,
            "length": w,
            "orient": "h",
            "row": r0,
            "c0": c0,
            "c1": c1,
        }

    if w == 1:
        ordered = sorted(int(r) for r in rows)
        assert len(comp) == h
        assert ordered == list(range(r0, r1 + 1))
        return {
            "color": color,
            "length": h,
            "orient": "v",
            "col": c0,
            "r0": r0,
            "r1": r1,
        }

    raise AssertionError("component is not a straight line segment")


def _segments(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != BG)
    assert comps
    return [_extract_segment(grid, comp) for comp in comps]


def _expanded_bounds(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _area_clear(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    rr0, cc0, rr1, cc1 = _expanded_bounds(r0, c0, r1, c1, h, w, pad=pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _occupy(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0, cc0, rr1, cc1 = _expanded_bounds(r0, c0, r1, c1, h, w, pad=pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_square(blocked: np.ndarray, length: int) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates = []
    for r0 in range(h - length + 1):
        for c0 in range(w - length + 1):
            r1 = r0 + length - 1
            c1 = c0 + length - 1
            if _area_clear(blocked, r0, c0, r1, c1, pad=1):
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    _occupy(blocked, r0, c0, r0 + length - 1, c0 + length - 1, pad=1)
    return r0, c0


def _place_line(blocked: np.ndarray, length: int, orient: str, min_row: int = 0) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates = []
    if orient == "h":
        for r in range(min_row, h):
            for c0 in range(w - length + 1):
                if _area_clear(blocked, r, c0, r, c0 + length - 1, pad=1):
                    candidates.append((r, c0))
        if not candidates:
            return None
        r, c0 = random.choice(candidates)
        _occupy(blocked, r, c0, r, c0 + length - 1, pad=1)
        return r, c0

    assert orient == "v"
    for r0 in range(h - length + 1):
        for c in range(w):
            if _area_clear(blocked, r0, c, r0 + length - 1, c, pad=1):
                candidates.append((r0, c))
    if not candidates:
        return None
    r0, c = random.choice(candidates)
    _occupy(blocked, r0, c, r0 + length - 1, c, pad=1)
    return r0, c


def _draw_horizontal(grid: np.ndarray, row: int, c0: int, length: int, color: int) -> None:
    grid[row, c0 : c0 + length] = color


def _draw_vertical(grid: np.ndarray, r0: int, col: int, length: int, color: int) -> None:
    grid[r0 : r0 + length, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered line segments by equal length, assemble each group into a closed rectangle outline,
    and fill the interior with the group's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments = _segments(g)
    groups: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        groups[int(seg["length"])].append(seg)

    assembled: list[tuple[int, int, int, int]] = []
    for length, segs in groups.items():
        assert len(segs) == 4
        horiz = [seg for seg in segs if seg["orient"] == "h"]
        vert = [seg for seg in segs if seg["orient"] == "v"]
        assert len(horiz) == 2
        assert len(vert) == 2

        colors = {int(seg["color"]) for seg in segs}
        assert len(colors) == 1
        color = colors.pop()

        horiz.sort(key=lambda seg: (int(seg["row"]), int(seg["c0"])))
        top = horiz[0]
        assert int(horiz[1]["row"]) > int(top["row"])

        r0 = int(top["row"])
        c0 = int(top["c0"])
        r1 = r0 + int(length) - 1
        c1 = c0 + int(length) - 1
        assert 0 <= r1 < g.shape[0]
        assert 0 <= c1 < g.shape[1]
        assembled.append((r0, c0, int(length), color))

    out = np.full_like(g, BG)
    for r0, c0, length, color in sorted(assembled):
        r1 = r0 + length - 1
        c1 = c0 + length - 1
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        if length > 2:
            out[r0 + 1 : r1, c0 + 1 : c1] = color

    return out


# --------------- generator ---------------

MIN_GROUPS = 2
MAX_GROUPS = 3
MIN_SIDE = 4
MAX_SIDE_TWO_GROUPS = 9
MAX_SIDE_THREE_GROUPS = 7
MIN_SIZE = 18
MAX_SIZE = 30
GEN_RETRIES = 400
COLOR_POOL = tuple(range(1, 10))


def _sample_groups() -> list[dict]:
    n_groups = random.randint(MIN_GROUPS, MAX_GROUPS)
    max_side = MAX_SIDE_TWO_GROUPS if n_groups == 2 else MAX_SIDE_THREE_GROUPS
    lengths = random.sample(list(range(MIN_SIDE, max_side + 1)), k=n_groups)
    colors = random.sample(COLOR_POOL, k=n_groups)
    groups = [{"length": int(length), "color": int(color)} for length, color in zip(lengths, colors)]
    random.shuffle(groups)
    return groups


def _grid_shape(groups: list[dict]) -> tuple[int, int]:
    max_len = max(group["length"] for group in groups)
    min_side = max(MIN_SIZE, max_len + 8, 10 + 4 * len(groups))
    h = random.randint(min_side, MAX_SIZE)
    w = random.randint(min_side, MAX_SIZE)
    return h, w


def _gen_attempt() -> np.ndarray | None:
    groups = _sample_groups()
    h, w = _grid_shape(groups)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    placed = []
    for group in sorted(groups, key=lambda group: -group["length"]):
        anchor = _place_square(blocked, int(group["length"]))
        if anchor is None:
            return None
        placed.append({**group, "anchor": anchor})

    for group in placed:
        length = int(group["length"])
        color = int(group["color"])
        top_r, left_c = group["anchor"]
        _draw_horizontal(grid, top_r, left_c, length, color)

    for group in placed:
        length = int(group["length"])
        color = int(group["color"])
        top_r, _left_c = group["anchor"]

        extra_h = _place_line(blocked, length, "h", min_row=top_r + 1)
        left_v = _place_line(blocked, length, "v")
        right_v = _place_line(blocked, length, "v")
        if extra_h is None or left_v is None or right_v is None:
            return None

        hr, hc = extra_h
        _draw_horizontal(grid, hr, hc, length, color)

        vr0, vc = left_v
        _draw_vertical(grid, vr0, vc, length, color)

        vr0, vc = right_v
        _draw_vertical(grid, vr0, vc, length, color)

    if len(_components(grid != BG)) != 4 * len(placed):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
