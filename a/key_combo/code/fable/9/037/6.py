from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StripInfo:
    grid: np.ndarray
    top_sig: tuple[int, ...]
    bottom_sig: tuple[int, ...]
    sign: int
    flipped: bool


GEN_RETRY_LIMIT = 400
GEN_STRIP_COUNT = (3, 6)
GEN_STRIP_HEIGHT = (3, 5)
GEN_WIDTH = (12, 20)
GEN_BAND_WIDTH = (2, 5)
GEN_CENTER_SHIFT = (1, 5)


def _is_contiguous(cols: np.ndarray) -> bool:
    if cols.size == 0:
        return False
    return bool(np.all(np.diff(cols) == 1))


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]


def _split_strips(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0

    sep_rows = _separator_rows(g)
    assert sep_rows, "expected at least one separator row"
    sep_color = int(g[sep_rows[0], 0])
    assert all(int(g[r, 0]) == sep_color for r in sep_rows), "separator rows must share one color"
    assert sep_rows[0] > 0 and sep_rows[-1] < g.shape[0] - 1, "separator rows must be internal"
    assert all(b - a >= 2 for a, b in zip(sep_rows, sep_rows[1:])), "separator rows must be isolated"

    strips: list[np.ndarray] = []
    start = 0
    for sep in sep_rows:
        strips.append(g[start:sep].copy())
        start = sep + 1
    strips.append(g[start:].copy())

    heights = {strip.shape[0] for strip in strips}
    widths = {strip.shape[1] for strip in strips}
    assert len(heights) == 1 and len(widths) == 1, "expected equal-sized horizontal strips"
    assert len(strips) >= 2, "expected at least two strips"
    return strips, sep_color


def _band_color(strips: list[np.ndarray], sep_color: int) -> int:
    rows = np.vstack(strips)
    candidates = sorted(int(x) for x in np.unique(rows) if int(x) != sep_color)
    valid: list[int] = []
    for color in candidates:
        ok = True
        for row in rows:
            cols = np.flatnonzero(row == color)
            if not _is_contiguous(cols):
                ok = False
                break
        if ok:
            valid.append(color)
    assert len(valid) == 1, "expected one color to form the through diagonal band"
    return valid[0]


def _signature(row: np.ndarray, color: int) -> tuple[int, ...]:
    cols = np.flatnonzero(row == color)
    assert _is_contiguous(cols), "band must appear as one contiguous run on each boundary row"
    return tuple(int(c) for c in cols)


def _strip_info(strip: np.ndarray, band_color: int, *, flipped: bool) -> StripInfo:
    top_sig = _signature(strip[0], band_color)
    bottom_sig = _signature(strip[-1], band_color)
    top_center2 = top_sig[0] + top_sig[-1]
    bottom_center2 = bottom_sig[0] + bottom_sig[-1]
    sign = int(np.sign(bottom_center2 - top_center2))
    assert sign != 0, "each strip must contain a slanted band segment"
    return StripInfo(strip.copy(), top_sig, bottom_sig, sign, flipped)


def _path_solutions(strips: list[np.ndarray], band_color: int) -> list[tuple[tuple[int, ...], list[StripInfo]]]:
    original = [_strip_info(strip, band_color, flipped=False) for strip in strips]
    mirrored = [_strip_info(np.fliplr(strip), band_color, flipped=True) for strip in strips]
    solutions: list[tuple[tuple[int, ...], list[StripInfo]]] = []

    for global_sign in (-1, 1):
        oriented: list[StripInfo] = []
        valid = True
        for i in range(len(strips)):
            choices = [info for info in (original[i], mirrored[i]) if info.sign == global_sign]
            if len(choices) != 1:
                valid = False
                break
            oriented.append(choices[0])
        if not valid:
            continue

        next_by_sig: dict[tuple[int, ...], list[int]] = {}
        for idx, info in enumerate(oriented):
            next_by_sig.setdefault(info.top_sig, []).append(idx)

        def dfs(order: list[int], used: set[int]) -> None:
            if len(order) == len(oriented):
                solutions.append((tuple(order), [oriented[i] for i in order]))
                return
            last = oriented[order[-1]]
            for nxt in next_by_sig.get(last.bottom_sig, []):
                if nxt not in used:
                    used.add(nxt)
                    order.append(nxt)
                    dfs(order, used)
                    order.pop()
                    used.remove(nxt)

        for start in range(len(oriented)):
            dfs([start], {start})

    return solutions


def _solve(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], list[StripInfo]]:
    strips, sep_color = _split_strips(grid)
    band_color = _band_color(strips, sep_color)
    candidates = _path_solutions(strips, band_color)
    assert candidates, "failed to find any valid strip reconstruction"

    preferred = [cand for cand in candidates if sum(info.flipped for info in cand[1]) == 1]
    if preferred:
        candidates = preferred

    unique: dict[bytes, tuple[np.ndarray, tuple[int, ...], list[StripInfo]]] = {}
    for order, infos in candidates:
        out = np.vstack([info.grid for info in infos])
        key = out.tobytes() + str(out.shape).encode()
        unique.setdefault(key, (out, order, infos))

    assert len(unique) == 1, "reassembly must be unique"
    return next(iter(unique.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scrambled horizontal strips by restoring the one mirrored strip and chaining the diagonal band across strip boundaries.

    Input structure:
    - The grid is split into equal-height horizontal strips, separated by single full-width gray rows.
    - One non-gray color forms a thick diagonal band segment inside every strip and touches the strip's top and bottom edges.
    - The strip order is scrambled, and exactly one strip has been mirrored left-to-right.

    Rule:
    - Treat the band columns on each strip's top and bottom edges as boundary interfaces.
    - Choose the unique strip order in which each strip's bottom interface matches the next strip's top interface and all strip-local band directions agree globally.
    - If a strip's local band direction is reversed, mirror that strip horizontally first.
    - Concatenate the restored strips and remove the gray separator rows.

    Key insight:
    - The strips are only fragments; the real object is the single through-diagonal band whose cross-boundary continuity determines both the ordering and the lone mirror repair.
    """
    out, _order, _infos = _solve(np.asarray(grid, dtype=int))
    return out


def _band_candidates(width: int, strip_h: int, current: tuple[int, int], sign: int) -> list[tuple[int, int]]:
    cur_l, cur_r = current
    cur_center2 = cur_l + cur_r
    cands: list[tuple[int, int]] = []
    for left in range(1, width - 2):
        for right in range(left + GEN_BAND_WIDTH[0] - 1, min(width - 1, left + GEN_BAND_WIDTH[1])):
            center_shift = sign * ((left + right) - cur_center2)
            if not (GEN_CENTER_SHIFT[0] <= center_shift <= GEN_CENTER_SHIFT[1]):
                continue
            if abs(left - cur_l) > strip_h - 1 or abs(right - cur_r) > strip_h - 1:
                continue
            if left == cur_l and right == cur_r:
                continue
            cands.append((left, right))
    return cands


def _render_strip(
    top_interval: tuple[int, int],
    bottom_interval: tuple[int, int],
    strip_h: int,
    width: int,
    bg: int,
    band_color: int,
) -> np.ndarray:
    strip = np.full((strip_h, width), bg, dtype=int)
    tl, tr = top_interval
    bl, br = bottom_interval
    for r in range(strip_h):
        t = 0.0 if strip_h == 1 else r / (strip_h - 1)
        left = int(round(tl + (bl - tl) * t))
        right = int(round(tr + (br - tr) * t))
        assert 1 <= left <= right <= width - 2, "band must stay away from the side borders"
        strip[r, left : right + 1] = band_color
    return strip


def _simple_edge_sorts_fail(infos: list[StripInfo], order: tuple[int, ...], sign: int) -> bool:
    def edge_seq(which: str) -> tuple[int, ...]:
        reverse = sign < 0
        if which == "top_left":
            return tuple(sorted(range(len(infos)), key=lambda i: infos[i].top_sig[0], reverse=reverse))
        if which == "top_right":
            return tuple(sorted(range(len(infos)), key=lambda i: infos[i].top_sig[-1], reverse=reverse))
        if which == "bottom_left":
            return tuple(sorted(range(len(infos)), key=lambda i: infos[i].bottom_sig[0], reverse=reverse))
        return tuple(sorted(range(len(infos)), key=lambda i: infos[i].bottom_sig[-1], reverse=reverse))

    return all(edge_seq(name) != order for name in ("top_left", "top_right", "bottom_left", "bottom_right"))


def _build_interval_chain(
    rng: np.random.Generator,
    strip_count: int,
    strip_h: int,
    width: int,
    sign: int,
) -> list[tuple[int, int]] | None:
    starts = [
        (left, right)
        for left in range(1, width - 2)
        for right in range(left + GEN_BAND_WIDTH[0] - 1, min(width - 1, left + GEN_BAND_WIDTH[1]))
    ]
    rng.shuffle(starts)

    def dfs(
        intervals: list[tuple[int, int]],
        edge_wobble: bool,
        width_changed: bool,
    ) -> list[tuple[int, int]] | None:
        if len(intervals) == strip_count + 1:
            return intervals if edge_wobble and width_changed else None

        current = intervals[-1]
        cands = _band_candidates(width, strip_h, current, sign)
        if not cands:
            return None
        rng.shuffle(cands)

        def score(cand: tuple[int, int]) -> tuple[int, int, float]:
            wobble_now = sign * (cand[0] - current[0]) < 0 or sign * (cand[1] - current[1]) < 0
            width_now = (cand[1] - cand[0]) != (current[1] - current[0])
            need_score = int(not edge_wobble and wobble_now) + int(not width_changed and width_now)
            center_shift = abs((cand[0] + cand[1]) - (current[0] + current[1]))
            return (-need_score, -int(width_now), float(center_shift))

        cands.sort(key=score)
        for cand in cands:
            wobble_now = sign * (cand[0] - current[0]) < 0 or sign * (cand[1] - current[1]) < 0
            width_now = (cand[1] - cand[0]) != (current[1] - current[0])
            result = dfs(intervals + [cand], edge_wobble or wobble_now, width_changed or width_now)
            if result is not None:
                return result
        return None

    for start in starts:
        chain = dfs([start], False, False)
        if chain is not None:
            return chain
    return None


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    strip_count = int(rng.integers(GEN_STRIP_COUNT[0], GEN_STRIP_COUNT[1] + 1))
    strip_h = int(rng.integers(GEN_STRIP_HEIGHT[0], GEN_STRIP_HEIGHT[1] + 1))
    width = int(rng.integers(GEN_WIDTH[0], GEN_WIDTH[1] + 1))
    if strip_count * strip_h + (strip_count - 1) > 30:
        return None

    colors = rng.choice(np.arange(10, dtype=int), size=3, replace=False)
    bg, band_color, sep_color = (int(x) for x in colors)
    sign = int(rng.choice([-1, 1]))

    intervals = _build_interval_chain(rng, strip_count, strip_h, width, sign)
    if intervals is None:
        return None

    strips = [
        _render_strip(intervals[i], intervals[i + 1], strip_h, width, bg, band_color)
        for i in range(strip_count)
    ]
    expected = np.vstack(strips)

    mirror_idx = int(rng.integers(0, strip_count))
    perm = rng.permutation(strip_count)
    if np.array_equal(perm, np.arange(strip_count)):
        return None

    scrambled = []
    for idx in perm:
        strip = strips[int(idx)]
        if int(idx) == mirror_idx:
            strip = np.fliplr(strip)
        scrambled.append(strip)

    separator = np.full((1, width), sep_color, dtype=int)
    rows = []
    for i, strip in enumerate(scrambled):
        rows.append(strip)
        if i + 1 < len(scrambled):
            rows.append(separator)
    inp = np.vstack(rows)

    if inp.shape[0] > 30 or inp.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    try:
        solved, order, infos = _solve(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if sum(info.flipped for info in infos) != 1:
        return None
    if not _simple_edge_sorts_fail(infos, order, infos[0].sign):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
