from __future__ import annotations

import random
import numpy as np


BG = 0
MIRROR = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
EDGE_TO_DIR = {
    "top": "down",
    "bottom": "up",
    "left": "right",
    "right": "left",
}
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_mirrors(grid: np.ndarray) -> tuple[list[dict[str, int | str]], dict[tuple[int, int], int]]:
    mirrors: list[dict[str, int | str]] = []
    pixel_to_mirror: dict[tuple[int, int], int] = {}
    for comp in _components(grid == MIRROR):
        rows = sorted({r for r, _c in comp})
        cols = sorted({c for _r, c in comp})
        assert len(comp) >= 2, "each mirror must have length at least 2"
        if len(rows) == 1:
            r0 = r1 = int(rows[0])
            c0 = int(min(cols))
            c1 = int(max(cols))
            expected = [(r0, c) for c in range(c0, c1 + 1)]
            orientation = "h"
        elif len(cols) == 1:
            c0 = c1 = int(cols[0])
            r0 = int(min(rows))
            r1 = int(max(rows))
            expected = [(r, c0) for r in range(r0, r1 + 1)]
            orientation = "v"
        else:
            raise AssertionError("mirror components must be straight horizontal or vertical segments")
        assert set(comp) == set(expected), "mirror segments must be contiguous"
        mid = int(len(mirrors))
        mirrors.append(
            {
                "id": mid,
                "orientation": orientation,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
            }
        )
        for cell in expected:
            pixel_to_mirror[cell] = mid
    assert mirrors, "expected at least one mirror"
    return mirrors, pixel_to_mirror


def _extract_sources(grid: np.ndarray) -> tuple[str, list[tuple[int, int, int]]]:
    h, w = grid.shape
    edge_cells = (
        {(0, c) for c in range(w)}
        | {(h - 1, c) for c in range(w)}
        | {(r, 0) for r in range(h)}
        | {(r, w - 1) for r in range(h)}
    )
    sources = [(r, c, int(grid[r, c])) for (r, c) in sorted(edge_cells) if int(grid[r, c]) not in (BG, MIRROR)]
    assert sources, "expected edge sources"

    fg = np.argwhere((grid != BG) & (grid != MIRROR))
    for r, c in fg:
        rr = int(r)
        cc = int(c)
        assert rr in (0, h - 1) or cc in (0, w - 1), "all non-black non-gray pixels must be edge sources"

    candidate_edges = []
    if all(r == 0 for r, _c, _col in sources):
        candidate_edges.append("top")
    if all(r == h - 1 for r, _c, _col in sources):
        candidate_edges.append("bottom")
    if all(c == 0 for _r, c, _col in sources):
        candidate_edges.append("left")
    if all(c == w - 1 for _r, c, _col in sources):
        candidate_edges.append("right")
    assert len(candidate_edges) == 1, "all sources must lie on exactly one edge"
    edge = candidate_edges[0]

    if edge in ("top", "bottom"):
        sources.sort(key=lambda x: (x[1], x[0]))
    else:
        sources.sort(key=lambda x: (x[0], x[1]))
    return edge, sources


def _reflect(mirror: dict[str, int | str], hit_r: int, hit_c: int, direction: str) -> str | None:
    if direction in ("left", "right") and mirror["orientation"] == "v":
        midpoint = (int(mirror["r0"]) + int(mirror["r1"])) / 2.0
        return "up" if hit_r <= midpoint else "down"
    if direction in ("up", "down") and mirror["orientation"] == "h":
        midpoint = (int(mirror["c0"]) + int(mirror["c1"])) / 2.0
        return "left" if hit_c <= midpoint else "right"
    return None


def _trace_source(
    grid: np.ndarray,
    out: np.ndarray,
    source: tuple[int, int, int],
    start_dir: str,
    mirrors: list[dict[str, int | str]],
    pixel_to_mirror: dict[tuple[int, int], int],
) -> None:
    h, w = grid.shape
    r, c, color = source
    direction = start_dir
    touched: set[int] = set()
    black_mask = grid == BG

    while True:
        dr, dc = DIRS[direction]
        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            return
        if int(grid[nr, nc]) == MIRROR:
            mid = int(pixel_to_mirror[(nr, nc)])
            if mid in touched:
                return
            new_dir = _reflect(mirrors[mid], nr, nc, direction)
            if new_dir is None:
                return
            touched.add(mid)
            direction = new_dir
            continue
        r, c = nr, nc
        if bool(black_mask[r, c]):
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each edge light source through gray mirror segments and color the black pixels on its reflected path.

    Input structure:
    - The background is black.
    - Gray pixels form one-pixel-thick straight mirror segments, each strictly horizontal or strictly vertical and at least length 2.
    - One grid edge contains one or more colored source pixels; all other non-black non-gray pixels are absent.
    - The source edge determines the initial ray direction: top shoots downward, bottom upward, left rightward, and right leftward.

    Rule:
    - Process the sources in edge order: left to right on the top/bottom edge, or top to bottom on the left/right edge.
    - Each source emits a one-pixel-wide ray that advances through the grid until it would leave the grid, or until it hits the same mirror for a second time.
    - A horizontal ray reflects only on a vertical mirror; an upper-half or middle hit turns upward, and a lower-half hit turns downward.
    - A vertical ray reflects only on a horizontal mirror; a left-half or middle hit turns leftward, and a right-half hit turns rightward.
    - Mirror pixels and source pixels stay unchanged. Any traversed black pixel is recolored to the source color, and later rays overwrite earlier ray colors where they cross.

    Key insight:
    - The gray line segments are mirrors rather than blockers, and the reflected direction is determined by where along the struck segment the collision occurs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    mirrors, pixel_to_mirror = _extract_mirrors(g)
    edge, sources = _extract_sources(g)

    out = g.copy()
    start_dir = EDGE_TO_DIR[edge]
    for source in sources:
        _trace_source(g, out, source, start_dir, mirrors, pixel_to_mirror)
    return out


def _paint_h(arr: np.ndarray, row: int, col0: int, col1: int) -> None:
    arr[row, col0 : col1 + 1] = MIRROR


def _paint_v(arr: np.ndarray, col: int, row0: int, row1: int) -> None:
    arr[row0 : row1 + 1, col] = MIRROR


def _make_boundary_right(height: int, width: int, color: int, rng: random.Random) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    src_c = rng.randint(1, width - 3)
    first_mirror_row = rng.randint(3, height - 4)
    h_left = rng.randint(max(0, src_c - 3), src_c - 1)
    _paint_h(patch, first_mirror_row, h_left, src_c)

    second_mirror_col = rng.randint(src_c + 2, width - 1)
    second_len = rng.randint(2, min(5, first_mirror_row - 1))
    second_top = first_mirror_row - second_len
    _paint_v(patch, second_mirror_col, second_top, first_mirror_row - 1)

    patch[0, src_c] = int(color)
    return patch


def _make_repeat_right(height: int, width: int, color: int, rng: random.Random) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    src_c = rng.randint(2, width - 3)
    return_c = rng.randint(1, src_c - 1)
    first_mirror_row = rng.randint(3, height - 7)
    second_mirror_col = rng.randint(src_c + 2, width - 1)
    lower_mirror_row = rng.randint(first_mirror_row + 4, height - 3)
    down_col = second_mirror_col - 1

    _paint_h(patch, first_mirror_row, return_c, src_c)

    second_len = rng.randint(2, min(5, first_mirror_row - 1))
    second_top = first_mirror_row - second_len
    _paint_v(patch, second_mirror_col, second_top, first_mirror_row - 1)

    if rng.random() < 0.5 and down_col + 1 < width:
        _paint_h(patch, lower_mirror_row, down_col - 1, down_col + 1)
    else:
        right_end = rng.randint(down_col + 1, min(width - 1, down_col + 3))
        _paint_h(patch, lower_mirror_row, down_col, right_end)

    hit_row = lower_mirror_row - 1
    if rng.random() < 0.5:
        _paint_v(patch, return_c - 1, hit_row - 1, hit_row + 1)
    else:
        bottom = rng.randint(hit_row + 1, min(height - 2, hit_row + 3))
        _paint_v(patch, return_c - 1, hit_row, bottom)

    patch[0, src_c] = int(color)
    return patch


def _make_gadget(kind: str, height: int, width: int, color: int, rng: random.Random) -> np.ndarray:
    if kind == "boundary":
        patch = _make_boundary_right(height, width, color, rng)
    else:
        patch = _make_repeat_right(height, width, color, rng)
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _sample_kinds(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(("boundary", "repeat"))]
    kinds = [rng.choice(("boundary", "repeat")) for _ in range(count)]
    if len(set(kinds)) == 1:
        kinds[rng.randrange(count)] = "repeat" if kinds[0] == "boundary" else "boundary"
    rng.shuffle(kinds)
    return kinds


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    source_count = rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 1], k=1)[0]
    kinds = _sample_kinds(int(source_count), rng)
    height = rng.randint(12, 24)

    widths = []
    for kind in kinds:
        if kind == "boundary":
            widths.append(rng.randint(5, 8))
        else:
            widths.append(rng.randint(7, 10))
    gap = rng.randint(1, 2)
    core_w = int(sum(widths) + gap * (source_count - 1))
    if core_w > 28:
        return None

    colors = rng.sample(SOURCE_COLORS, int(source_count))
    core = np.full((height, core_w), BG, dtype=int)
    offset = 0
    for kind, width, color in zip(kinds, widths, colors):
        patch = _make_gadget(kind, height, int(width), int(color), rng)
        core[:, offset : offset + width] = patch
        offset += int(width + gap)

    max_extra_w = 30 - core_w
    left_pad = rng.randint(0, min(2, max_extra_w))
    right_pad = rng.randint(0, min(2, max_extra_w - left_pad))
    max_extra_h = 30 - height
    bottom_pad = rng.randint(0, min(4, max_extra_h))
    grid = np.pad(core, ((0, bottom_pad), (left_pad, right_pad)), constant_values=BG)
    grid = np.rot90(grid, rng.randrange(4))

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-reflection ARC input")
