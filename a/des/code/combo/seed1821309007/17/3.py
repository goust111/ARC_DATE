from __future__ import annotations

from collections import Counter, deque
import numpy as np


BG = 7
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400
PALETTE = [c for c in range(10) if c != BG]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append({"color": color, "coords": arr, "bbox": (r0, c0, r1, c1)})
    return comps


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _principal_axis(a: dict, b: dict) -> str:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    ac = (ac0 + ac1) / 2.0
    bc = (bc0 + bc1) / 2.0
    ar = (ar0 + ar1) / 2.0
    br = (br0 + br1) / 2.0
    dx = abs(ac - bc)
    dy = abs(ar - br)
    return "horizontal" if dx >= dy else "vertical"


def _paint_rect(out: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> None:
    if r0 <= r1 and c0 <= c1:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _draw_bridge(out: np.ndarray, a: dict, b: dict) -> None:
    color = int(a["color"])
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    axis = _principal_axis(a, b)

    if axis == "horizontal":
        left, right = (a, b) if ac0 <= bc0 else (b, a)
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        gap0, gap1 = int(lc1 + 1), int(rc0 - 1)
        if gap0 > gap1:
            return

        overlap0 = max(int(lr0), int(rr0))
        overlap1 = min(int(lr1), int(rr1))
        if overlap0 <= overlap1:
            _paint_rect(out, overlap0, overlap1, gap0, gap1, color)
            return

        if int(lr1) < int(rr0):
            src_r, dst_r = int(lr1), int(rr0)
        else:
            src_r, dst_r = int(lr0), int(rr1)
        turn_c = int(rc0)
        _paint_rect(out, src_r, src_r, gap0, gap1, color)
        _paint_rect(out, min(src_r, dst_r), max(src_r, dst_r), turn_c, turn_c, color)
        return

    top, bottom = (a, b) if ar0 <= br0 else (b, a)
    tr0, tc0, tr1, tc1 = top["bbox"]
    br0, bc0, br1, bc1 = bottom["bbox"]
    gap0, gap1 = int(tr1 + 1), int(br0 - 1)
    if gap0 > gap1:
        return

    overlap0 = max(int(tc0), int(bc0))
    overlap1 = min(int(tc1), int(bc1))
    if overlap0 <= overlap1:
        _paint_rect(out, gap0, gap1, overlap0, overlap1, color)
        return

    if int(tc1) < int(bc0):
        src_c, dst_c = int(tc1), int(bc0)
    else:
        src_c, dst_c = int(tc0), int(bc1)
    turn_r = int(br0)
    _paint_rect(out, gap0, gap1, src_c, src_c, color)
    _paint_rect(out, turn_r, turn_r, min(src_c, dst_c), max(src_c, dst_c), color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    comps = _find_components(g, bg)
    assert comps

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)
    assert all(len(group) == 2 for group in by_color.values())

    out = g.copy()
    for color, pair in by_color.items():
        assert color != bg
        _draw_bridge(out, pair[0], pair[1])
    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _paint_object(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _pair_bboxes(
    h: int,
    w: int,
    axis: str,
    overlap_case: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    if axis == "horizontal":
        h1 = int(RNG.integers(1, 5))
        w1 = int(RNG.integers(1, 5))
        h2 = int(RNG.integers(1, 5))
        w2 = int(RNG.integers(1, 5))
        gap = int(RNG.integers(2, 7))
        c0a = int(RNG.integers(1, max(2, w - (w1 + w2 + gap) - 1)))
        c0b = c0a + w1 + gap
        if c0b + w2 > w - 1:
            return None

        if overlap_case:
            ov = int(RNG.integers(1, min(h1, h2) + 1))
            extra_up = int(RNG.integers(0, h1 - ov + 1))
            extra_down = int(RNG.integers(0, h2 - ov + 1))
            top_overlap = int(RNG.integers(1, h - max(h1, h2)))
            r0a = top_overlap - extra_up
            r0b = top_overlap - extra_down
        else:
            sep = int(RNG.integers(1, 4))
            upper_is_a = bool(RNG.integers(0, 2))
            total_h = h1 + h2 + sep
            if total_h > h - 2:
                return None
            top = int(RNG.integers(1, h - total_h))
            if upper_is_a:
                r0a = top
                r0b = top + h1 + sep
            else:
                r0b = top
                r0a = top + h2 + sep

        if min(r0a, r0b) < 1 or max(r0a + h1, r0b + h2) > h - 1:
            return None
        a = (r0a, c0a, r0a + h1 - 1, c0a + w1 - 1)
        b = (r0b, c0b, r0b + h2 - 1, c0b + w2 - 1)
    else:
        h1 = int(RNG.integers(1, 5))
        w1 = int(RNG.integers(1, 5))
        h2 = int(RNG.integers(1, 5))
        w2 = int(RNG.integers(1, 5))
        gap = int(RNG.integers(2, 7))
        r0a = int(RNG.integers(1, max(2, h - (h1 + h2 + gap) - 1)))
        r0b = r0a + h1 + gap
        if r0b + h2 > h - 1:
            return None

        if overlap_case:
            ov = int(RNG.integers(1, min(w1, w2) + 1))
            extra_left = int(RNG.integers(0, w1 - ov + 1))
            extra_right = int(RNG.integers(0, w2 - ov + 1))
            left_overlap = int(RNG.integers(1, w - max(w1, w2)))
            c0a = left_overlap - extra_left
            c0b = left_overlap - extra_right
        else:
            sep = int(RNG.integers(1, 4))
            left_is_a = bool(RNG.integers(0, 2))
            total_w = w1 + w2 + sep
            if total_w > w - 2:
                return None
            left = int(RNG.integers(1, w - total_w))
            if left_is_a:
                c0a = left
                c0b = left + w1 + sep
            else:
                c0b = left
                c0a = left + w2 + sep

        if min(c0a, c0b) < 1 or max(c0a + w1, c0b + w2) > w - 1:
            return None
        a = (r0a, c0a, r0a + h1 - 1, c0a + w1 - 1)
        b = (r0b, c0b, r0b + h2 - 1, c0b + w2 - 1)

    ac = (a[1] + a[3]) / 2.0
    bc = (b[1] + b[3]) / 2.0
    ar = (a[0] + a[2]) / 2.0
    br = (b[0] + b[2]) / 2.0
    if axis == "horizontal" and not (abs(ac - bc) > abs(ar - br)):
        return None
    if axis == "vertical" and not (abs(ar - br) > abs(ac - bc)):
        return None
    return a, b


def _validate_input(grid: np.ndarray) -> bool:
    comps = _find_components(grid, BG)
    if not comps:
        return False
    counts = Counter(int(comp["color"]) for comp in comps)
    return all(v == 2 for v in counts.values())


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(14, 24))
    w = int(RNG.integers(14, 24))
    grid = np.full((h, w), BG, dtype=int)

    pair_count = int(RNG.integers(2, 5))
    colors = list(RNG.choice(PALETTE, size=pair_count, replace=False))
    occupied: list[tuple[int, int, int, int]] = []

    for idx, color in enumerate(colors):
        pair_axis = "horizontal" if idx % 2 == 0 else "vertical"
        if RNG.random() < 0.35:
            pair_axis = "vertical" if pair_axis == "horizontal" else "horizontal"
        overlap_case = bool((idx + int(RNG.integers(0, 2))) % 2 == 0)

        placed = None
        for _ in range(120):
            pair = _pair_bboxes(h, w, pair_axis, overlap_case)
            if pair is None:
                continue
            a, b = pair
            if any(_rects_overlap(a, other, margin=2) or _rects_overlap(b, other, margin=2) for other in occupied):
                continue
            placed = (a, b)
            break
        if placed is None:
            return None

        a, b = placed
        _paint_object(grid, a, color)
        _paint_object(grid, b, color)
        occupied.extend([a, b])

    if not _validate_input(grid):
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
