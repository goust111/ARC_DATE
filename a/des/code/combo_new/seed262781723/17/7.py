from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
PATTERN_COLORS = (1, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300
CONFIGS = (
    ("single", "h"),
    ("single", "v"),
    ("double", "h"),
    ("double", "v"),
)


def _bbox_non_bg(g: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(g != BG)
    assert coords.size > 0, "grid must contain a non-background rectangle"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _line_candidates(g: np.ndarray, bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    lines: list[dict] = []
    for r in range(r0, r1 + 1):
        seg = g[r, c0 : c1 + 1]
        if np.all((seg == RED) | (seg == GREEN)) and np.count_nonzero(seg == RED) >= max(2, seg.size - 1):
            lines.append({"axis": "h", "index": r})
    for c in range(c0, c1 + 1):
        seg = g[r0 : r1 + 1, c]
        if np.all((seg == RED) | (seg == GREEN)) and np.count_nonzero(seg == RED) >= max(2, seg.size - 1):
            lines.append({"axis": "v", "index": c})
    assert 1 <= len(lines) <= 2, "expected one or two spanning fold lines"
    return lines


def _marker_options(
    line: dict,
    greens: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    out: list[dict] = []
    if line["axis"] == "h":
        r = int(line["index"])
        for gr, gc in greens:
            if gc not in {c0, c1}:
                continue
            if gr == r - 1:
                out.append({"side": "top", "marker": (gr, gc)})
            elif gr == r + 1:
                out.append({"side": "bottom", "marker": (gr, gc)})
    else:
        c = int(line["index"])
        for gr, gc in greens:
            if gr not in {r0, r1}:
                continue
            if gc == c - 1:
                out.append({"side": "left", "marker": (gr, gc)})
            elif gc == c + 1:
                out.append({"side": "right", "marker": (gr, gc)})
    return out


def _assign_markers(
    lines: list[dict],
    greens: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> list[dict]:
    options = [_marker_options(line, greens, bbox) for line in lines]
    assert all(opts for opts in options), "each fold line must have an endpoint-adjacent green marker"

    assigned: list[dict] = []
    used: set[tuple[int, int]] = set()

    def backtrack(i: int) -> bool:
        if i == len(lines):
            return True
        for opt in options[i]:
            marker = tuple(opt["marker"])
            if marker in used:
                continue
            used.add(marker)
            line = dict(lines[i])
            line.update(opt)
            assigned.append(line)
            if backtrack(i + 1):
                return True
            assigned.pop()
            used.remove(marker)
        return False

    assert len(greens) == len(lines), "expected exactly one green marker per fold line"
    assert backtrack(0), "green markers must pair with fold lines unambiguously"
    return assigned


def _fold_width(line: dict, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    if line["axis"] == "h":
        r = int(line["index"])
        return int(r - r0) if line["side"] == "top" else int(r1 - r)
    c = int(line["index"])
    return int(c - c0) if line["side"] == "left" else int(c1 - c)


def _source_range(
    line: dict,
    bbox: tuple[int, int, int, int],
) -> tuple[range, range]:
    r0, c0, r1, c1 = bbox
    if line["axis"] == "h":
        r = int(line["index"])
        if line["side"] == "top":
            return range(r0, r), range(c0, c1 + 1)
        return range(r + 1, r1 + 1), range(c0, c1 + 1)
    c = int(line["index"])
    if line["side"] == "left":
        return range(r0, r1 + 1), range(c0, c)
    return range(r0, r1 + 1), range(c + 1, c1 + 1)


def _mirror_target(line: dict, r: int, c: int) -> tuple[int, int]:
    if line["axis"] == "h":
        axis_r = int(line["index"])
        return 2 * axis_r - r, c
    axis_c = int(line["index"])
    return r, 2 * axis_c - c


def _apply_fold(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    line: dict,
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    base = g.copy()
    out = g.copy()

    rows, cols = _source_range(line, bbox)
    for r in rows:
        for c in cols:
            value = int(base[r, c])
            if value not in {BG, RED, GREEN}:
                out[r, c] = BG

    if line["axis"] == "h":
        out[int(line["index"]), c0 : c1 + 1] = BG
    else:
        out[r0 : r1 + 1, int(line["index"])] = BG
    mr, mc = map(int, line["marker"])
    out[mr, mc] = BG

    for r in rows:
        for c in cols:
            value = int(base[r, c])
            if value in {BG, RED, GREEN}:
                continue
            tr, tc = _mirror_target(line, int(r), int(c))
            if not (r0 <= tr <= r1 and c0 <= tc <= c1):
                continue
            if int(base[tr, tc]) == BG:
                out[tr, tc] = value

    return out


def _find_fold_instructions(g: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    bbox = _bbox_non_bg(g)
    greens = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    lines = _line_candidates(g, bbox)
    instructions = _assign_markers(lines, greens, bbox)
    instructions.sort(key=lambda line: (_fold_width(line, bbox), line["axis"], int(line["index"])))
    return bbox, instructions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the marked flap of a rectangular pattern across each spanning red line.

    Input structure:
    - Black background surrounds one rectangular pattern area made of several visible colors.
    - Inside that rectangle there are one or two full-span red fold lines, each horizontal or vertical.
    - A green pixel sits next to one endpoint of each red line on the side that must be folded.

    Rule:
    - Identify the rectangular pattern bbox, the spanning red lines, and the green side markers.
    - Process the fold lines from outer to inner, i.e. by increasing width of the marked flap between the line and the bbox edge.
    - For each line, mirror every non-black, non-red, non-green pixel from the marked side across the line into the opposite side.
    - A mirrored pixel is written only onto black target cells; existing non-black target pixels keep their original color.
    - After each fold, erase the current red line, its green marker, and the original colored pixels from the folded side; pixels mirrored outside the bbox are discarded.

    Key insight:
    - The green marker does not name a color mapping; it geometrically selects which flap of the rectangle is reflected, and multi-line cases depend on applying those reflections in the correct outside-to-inside order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bbox, instructions = _find_fold_instructions(g)
    out = g.copy()
    for line in instructions:
        out = _apply_fold(out, bbox, line)
    return out


def _sample_dimensions(kind: str, axis: str) -> tuple[int, int, int, int, int, int]:
    min_rect_h = 7 if kind == "double" and axis == "h" else 6
    min_rect_w = 7 if kind == "double" and axis == "v" else 6

    grid_h = random.randint(max(10, min_rect_h + 2), 24)
    grid_w = random.randint(max(10, min_rect_w + 2), 24)
    rect_h = random.randint(min_rect_h, min(18, grid_h - 2))
    rect_w = random.randint(min_rect_w, min(18, grid_w - 2))
    top = random.randint(1, grid_h - rect_h - 1)
    left = random.randint(1, grid_w - rect_w - 1)
    return grid_h, grid_w, top, left, rect_h, rect_w


def _sample_lines(kind: str, axis: str, rect_h: int, rect_w: int) -> list[dict] | None:
    lines: list[dict] = []
    if axis == "h":
        candidates = list(range(1, rect_h - 1))
        if kind == "single":
            positions = [random.choice(candidates)]
        else:
            pairs = [(a, b) for a in candidates for b in candidates if a < b and b - a >= 2]
            if not pairs:
                return None
            positions = list(random.choice(pairs))
        for r in positions:
            side = random.choice(("top", "bottom"))
            marker = (r - 1 if side == "top" else r + 1, random.choice((0, rect_w - 1)))
            lines.append({"axis": "h", "index": r, "side": side, "marker": marker})
    else:
        candidates = list(range(1, rect_w - 1))
        if kind == "single":
            positions = [random.choice(candidates)]
        else:
            pairs = [(a, b) for a in candidates for b in candidates if a < b and b - a >= 2]
            if not pairs:
                return None
            positions = list(random.choice(pairs))
        for c in positions:
            side = random.choice(("left", "right"))
            marker = (random.choice((0, rect_h - 1)), c - 1 if side == "left" else c + 1)
            lines.append({"axis": "v", "index": c, "side": side, "marker": marker})

    flaps = [_fold_width(line, (0, 0, rect_h - 1, rect_w - 1)) for line in lines]
    if len(set(flaps)) != len(flaps):
        return None
    markers = [tuple(line["marker"]) for line in lines]
    if len(set(markers)) != len(markers):
        return None
    return lines


def _make_pattern(rect_h: int, rect_w: int, palette: tuple[int, ...]) -> np.ndarray:
    rr, cc = np.indices((rect_h, rect_w))
    colors = np.asarray(palette, dtype=int)
    mode = random.randrange(3)

    if mode == 0:
        idx = (rr + 2 * cc + random.randrange(9)) % len(colors)
        mod = random.choice((4, 5))
        keep = (2 * rr + cc + random.randrange(mod)) % mod != 0
    elif mode == 1:
        idx = ((rr // random.choice((1, 2))) + (cc // random.choice((1, 2))) + random.randrange(9)) % len(colors)
        mod = random.choice((4, 5, 6))
        keep = (rr + 2 * (cc // 2) + random.randrange(mod)) % mod < mod - 1
    else:
        if random.choice((False, True)):
            idx = ((rr // random.choice((1, 2, 3))) + random.randrange(9)) % len(colors)
            mod = random.choice((3, 4, 5))
            keep = (cc + random.randrange(mod)) % mod != 0
        else:
            idx = ((cc // random.choice((1, 2, 3))) + random.randrange(9)) % len(colors)
            mod = random.choice((3, 4, 5))
            keep = (rr + random.randrange(mod)) % mod != 0

    pattern = np.where(keep, colors[idx], BG).astype(int)

    top = colors[(np.arange(rect_w) + random.randrange(len(colors))) % len(colors)]
    left = colors[(np.arange(rect_h) + random.randrange(len(colors))) % len(colors)]
    pattern[0, :] = top
    pattern[-1, :] = np.roll(top, 1)
    pattern[:, 0] = left
    pattern[:, -1] = np.roll(left, 1)
    return pattern


def _instruction_mask(rect_h: int, rect_w: int, lines: list[dict]) -> np.ndarray:
    mask = np.zeros((rect_h, rect_w), dtype=bool)
    for line in lines:
        if line["axis"] == "h":
            mask[int(line["index"]), :] = True
        else:
            mask[:, int(line["index"])] = True
        mr, mc = map(int, line["marker"])
        mask[mr, mc] = True
    return mask


def _open_target_holes(pattern: np.ndarray, lines: list[dict], palette: tuple[int, ...]) -> None:
    rect_h, rect_w = pattern.shape
    reserved = _instruction_mask(rect_h, rect_w, lines)
    bbox = (0, 0, rect_h - 1, rect_w - 1)

    for line in lines:
        candidates: list[tuple[int, int, int, int]] = []
        rows, cols = _source_range(line, bbox)
        for r in rows:
            for c in cols:
                if reserved[r, c]:
                    continue
                tr, tc = _mirror_target(line, int(r), int(c))
                if not (0 <= tr < rect_h and 0 <= tc < rect_w):
                    continue
                if reserved[tr, tc]:
                    continue
                candidates.append((int(r), int(c), int(tr), int(tc)))
        random.shuffle(candidates)
        if not candidates:
            continue
        target_count = min(len(candidates), random.randint(4, min(10, len(candidates))))
        changed = 0
        for sr, sc, tr, tc in candidates:
            if pattern[sr, sc] == BG:
                pattern[sr, sc] = random.choice(palette)
            pattern[tr, tc] = BG
            changed += 1
            if changed >= target_count:
                break


def _draw_input(
    grid_h: int,
    grid_w: int,
    top: int,
    left: int,
    pattern: np.ndarray,
    lines: list[dict],
) -> np.ndarray:
    rect_h, rect_w = pattern.shape
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[top : top + rect_h, left : left + rect_w] = pattern

    for line in lines:
        if line["axis"] == "h":
            grid[top + int(line["index"]), left : left + rect_w] = RED
        else:
            grid[top : top + rect_h, left + int(line["index"])] = RED
    for line in lines:
        mr, mc = map(int, line["marker"])
        grid[top + mr, left + mc] = GREEN
    return grid


def _gen_attempt() -> np.ndarray | None:
    kind, axis = random.choice(CONFIGS)
    grid_h, grid_w, top, left, rect_h, rect_w = _sample_dimensions(kind, axis)
    lines = _sample_lines(kind, axis, rect_h, rect_w)
    if lines is None:
        return None

    palette = tuple(random.sample(PATTERN_COLORS, random.randint(3, 4)))
    pattern = _make_pattern(rect_h, rect_w, palette)
    _open_target_holes(pattern, lines, palette)
    inp = _draw_input(grid_h, grid_w, top, left, pattern, lines)

    try:
        out = main(inp)
    except AssertionError:
        return None

    bbox, parsed_lines = _find_fold_instructions(inp)
    if len(parsed_lines) != len(lines):
        return None
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(inp != out) < 10:
        return None
    if np.count_nonzero(out != BG) < 8:
        return None

    r0, c0, r1, c1 = bbox
    if not np.all(inp[:r0, :] == BG) or not np.all(inp[r1 + 1 :, :] == BG):
        return None
    if not np.all(inp[:, :c0] == BG) or not np.all(inp[:, c1 + 1 :] == BG):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-instruction grid")
