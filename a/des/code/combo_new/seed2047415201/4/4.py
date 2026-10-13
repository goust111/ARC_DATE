import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
MAX_LINK_GAP = 1
HORIZONTAL_COLORS = (2, 7, 4, 9)
VERTICAL_COLORS = (1, 3, 8)
GENERATION_RETRIES = 500


def _center_dist2(shape: tuple[int, int], cell: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    dr = 2 * r - (h - 1)
    dc = 2 * c - (w - 1)
    return dr * dr + dc * dc


def _segment_from_color(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == color)
    assert coords.size > 0
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})

    if len(rows) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        assert len(cols) >= 2
        return {
            "color": int(color),
            "orient": "h",
            "fixed": row,
            "lo": cols[0],
            "hi": cols[-1],
        }

    if len(cols) == 1:
        col = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1))
        assert len(rows) >= 2
        return {
            "color": int(color),
            "orient": "v",
            "fixed": col,
            "lo": rows[0],
            "hi": rows[-1],
        }

    raise AssertionError("each foreground color must form one straight segment")


def _link_gap(a: dict, b: dict) -> int | None:
    if a["orient"] != b["orient"] or a["fixed"] != b["fixed"]:
        return None
    if a["hi"] < b["lo"]:
        return int(b["lo"] - a["hi"] - 1)
    if b["hi"] < a["lo"]:
        return int(a["lo"] - b["hi"] - 1)
    raise AssertionError("same-line segments must not overlap")


def _chain_components(segments: list[dict]) -> list[list[int]]:
    edges: dict[int, set[int]] = {i: set() for i in range(len(segments))}
    by_orient: dict[str, list[int]] = defaultdict(list)
    for i, seg in enumerate(segments):
        by_orient[seg["orient"]].append(i)

    for indices in by_orient.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                gap = _link_gap(segments[i], segments[j])
                if gap is not None and 0 <= gap <= MAX_LINK_GAP:
                    edges[i].add(j)
                    edges[j].add(i)

    for i, nbrs in edges.items():
        assert len(nbrs) <= 2, f"segment {i} branches into more than one path"

    seen: set[int] = set()
    comps: list[list[int]] = []
    for i in range(len(segments)):
        if i in seen:
            continue
        q = deque([i])
        seen.add(i)
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in edges[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _chain_endpoints(segments: list[dict], comp: list[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    orient = segments[comp[0]]["orient"]
    assert all(segments[i]["orient"] == orient for i in comp)
    assert all(segments[i]["fixed"] == segments[comp[0]]["fixed"] for i in comp)

    first = min(comp, key=lambda i: int(segments[i]["lo"]))
    last = max(comp, key=lambda i: int(segments[i]["hi"]))
    if orient == "h":
        row = int(segments[first]["fixed"])
        return (row, int(segments[first]["lo"])), (row, int(segments[last]["hi"]))

    col = int(segments[first]["fixed"])
    return (int(segments[first]["lo"]), col), (int(segments[last]["hi"]), col)


def main(grid: np.ndarray) -> np.ndarray:
    """Shorten each linked horizontal or vertical colored path by one pixel at its far end.

    Input structure:
    - The background is black.
    - Every non-black color forms exactly one contiguous one-pixel-wide horizontal
      or vertical segment, and the colors are distinct between segments.
    - Segments with the same orientation may sit on the same row or column with
      touching endpoints, or with a single black cell between endpoints, forming a
      path chain; other segments are isolated from those chains.

    Rule:
    - Group segments by their geometric orientation.
    - Within each orientation, link same-row or same-column segments whose facing
      endpoints are adjacent or separated by one black cell.
    - For every linked chain, treat the endpoint nearer the grid center as the
      path start and erase one pixel from the opposite endpoint.
    - An isolated segment follows the same center cue, so it is shortened from the
      endpoint farther from the grid center.

    Key insight:
    The colors separate the visible path pieces, but the transformation is applied
    to whole same-orientation endpoint chains; only the terminal pixel farthest
    from the center-directed start is removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert BLACK in set(map(int, np.unique(g))), "expected black background"

    colors = [int(c) for c in np.unique(g) if int(c) != BLACK]
    segments = [_segment_from_color(g, color) for color in colors]

    out = g.copy()
    for comp in _chain_components(segments):
        end_a, end_b = _chain_endpoints(segments, comp)
        if _center_dist2(g.shape, end_a) >= _center_dist2(g.shape, end_b):
            far = end_a
        else:
            far = end_b
        out[far] = BLACK
    return out


def _line_cells(orient: str, fixed: int, lo: int, hi: int) -> list[tuple[int, int]]:
    if orient == "h":
        return [(fixed, c) for c in range(lo, hi + 1)]
    return [(r, fixed) for r in range(lo, hi + 1)]


def _chain_candidate(
    shape: tuple[int, int],
    orient: str,
    count: int,
) -> list[tuple[int, int, int]]:
    h, w = shape
    lengths = [random.randint(2, 5) for _ in range(count)]
    gaps = [random.randint(0, 1) for _ in range(count - 1)]
    span = sum(lengths) + sum(gaps)

    if orient == "h":
        center = (w - 1) / 2
        side = random.choice(["left", "right"])
        if side == "left":
            max_start = int(center) - span - 1
            if max_start < 1:
                raise ValueError
            lo = random.randint(1, max_start)
        else:
            min_start = int(center) + 2
            max_start = w - span - 2
            if min_start > max_start:
                raise ValueError
            lo = random.randint(min_start, max_start)
        fixed = random.randint(1, h - 2)
    else:
        center = (h - 1) / 2
        side = random.choice(["top", "bottom"])
        if side == "top":
            max_start = int(center) - span - 1
            if max_start < 1:
                raise ValueError
            lo = random.randint(1, max_start)
        else:
            min_start = int(center) + 2
            max_start = h - span - 2
            if min_start > max_start:
                raise ValueError
            lo = random.randint(min_start, max_start)
        fixed = random.randint(1, w - 2)

    result: list[tuple[int, int, int]] = []
    cur = lo
    for i, length in enumerate(lengths):
        result.append((fixed, cur, cur + length - 1))
        cur += length
        if i < len(gaps):
            cur += gaps[i] + 1
    return result


def _protected_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    protected: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    protected.add((rr, cc))
    return protected


def _place_object(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    palette: list[int],
    orient: str,
    specs: list[tuple[int, int, int]],
) -> bool:
    if len(specs) > len(palette):
        return False

    actual: set[tuple[int, int]] = set()
    protected_basis: set[tuple[int, int]] = set()
    for fixed, lo, hi in specs:
        cells = set(_line_cells(orient, fixed, lo, hi))
        actual.update(cells)
        protected_basis.update(cells)
        if orient == "h":
            protected_basis.update((fixed, c) for c in range(lo, hi + 1))
        else:
            protected_basis.update((r, fixed) for r in range(lo, hi + 1))

    if actual & blocked:
        return False
    if _protected_cells(grid.shape, actual) & {cell for cell in blocked if grid[cell] != BLACK}:
        return False

    for color, (fixed, lo, hi) in zip(palette[: len(specs)], specs):
        for r, c in _line_cells(orient, fixed, lo, hi):
            grid[r, c] = int(color)

    blocked.update(_protected_cells(grid.shape, protected_basis))
    del palette[: len(specs)]
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 30)
    w = random.randint(13, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()

    h_colors = list(HORIZONTAL_COLORS)
    v_colors = list(VERTICAL_COLORS)
    random.shuffle(h_colors)
    random.shuffle(v_colors)

    h_chain_len = random.choice([2, 2, 3])
    v_chain_len = random.choice([2, 2, 3])
    if h_chain_len + 1 > len(h_colors):
        h_chain_len = 2
    if v_chain_len > len(v_colors):
        v_chain_len = 2

    objects: list[tuple[str, int, bool]] = [
        ("h", h_chain_len, True),
        ("v", v_chain_len, True),
    ]

    remaining_h = len(h_colors) - h_chain_len
    remaining_v = len(v_colors) - v_chain_len
    isolated_choices = []
    if remaining_h > 0:
        isolated_choices.append("h")
    if remaining_v > 0:
        isolated_choices.append("v")
    if not isolated_choices:
        return None
    objects.append((random.choice(isolated_choices), 1, False))

    if random.random() < 0.45 and len(objects) < 4:
        extra_choices = []
        if remaining_h - sum(1 for orient, count, _ in objects[2:] if orient == "h" for _i in range(count)) > 0:
            extra_choices.append("h")
        if remaining_v - sum(1 for orient, count, _ in objects[2:] if orient == "v" for _i in range(count)) > 0:
            extra_choices.append("v")
        if extra_choices:
            objects.append((random.choice(extra_choices), 1, False))

    random.shuffle(objects)
    for orient, count, _is_chain in objects:
        palette = h_colors if orient == "h" else v_colors
        placed = False
        for _ in range(80):
            try:
                specs = _chain_candidate((h, w), orient, count)
            except ValueError:
                continue
            trial = grid.copy()
            trial_blocked = set(blocked)
            trial_palette = list(palette)
            if _place_object(trial, trial_blocked, trial_palette, orient, specs):
                grid[:, :] = trial
                blocked.clear()
                blocked.update(trial_blocked)
                palette[:] = trial_palette
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    removed = int(np.count_nonzero(grid != BLACK) - np.count_nonzero(out != BLACK))
    if removed < 3:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-shortening input")
