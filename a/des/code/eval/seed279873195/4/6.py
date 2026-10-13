import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

EMITTER_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE)
LEGEND_WIDTH = 7
MAX_GENERATE_ATTEMPTS = 500


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
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == MAROON)
    assert len(pts) >= 12, "expected a maroon legend frame"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 == grid.shape[0] - 1, "legend must reach the bottom edge"

    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert frame.shape[0] >= 3 and frame.shape[1] >= 5, "legend frame too small"
    assert np.all(frame[0] == MAROON) and np.all(frame[-1] == MAROON), "legend top/bottom border missing"
    assert np.all(frame[:, 0] == MAROON) and np.all(frame[:, -1] == MAROON), "legend side border missing"
    assert not np.any(grid[:r0] == MAROON), "maroon should only appear in the legend"
    return r0, c0, r1, c1


def _parse_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, tuple[int, int]]]:
    r0, c0, r1, c1 = _find_legend_bbox(grid)
    mapping: dict[int, tuple[int, int]] = {}
    for r in range(r0 + 1, r1):
        cols = [c for c in range(c0 + 1, c1) if int(grid[r, c]) != BLACK]
        if not cols:
            continue
        vals = [int(grid[r, c]) for c in cols]
        assert len(vals) == 3, "each legend row must contain exactly three colored pixels"
        assert vals[1] != vals[2], "split colors must be distinct"
        mapping[int(vals[0])] = (int(vals[1]), int(vals[2]))
    assert mapping, "legend mapping must be non-empty"
    return (r0, c0, r1, c1), mapping


def _find_prisms(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    prisms: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 == 1 and c1 - c0 == 1, "every grey object must be a 2x2 prism"
        assert len(comp) == 4, "prism must be solid 2x2"
        prisms.append((int(r0), int(c0), int(r1), int(c1)))
    return prisms


def _find_pairs(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int], str]]:
    colors = sorted(
        int(v)
        for v in np.unique(grid)
        if int(v) not in (BLACK, GREY, MAROON)
    )
    pairs: list[tuple[int, tuple[int, int], tuple[int, int], str]] = []
    for color in colors:
        pts = [tuple(map(int, p)) for p in np.argwhere(grid == int(color))]
        assert len(pts) == 2, "each emitter color must appear exactly twice"
        (r0, c0), (r1, c1) = pts
        if r0 == r1:
            a, b = sorted(pts, key=lambda p: p[1])
            pairs.append((int(color), a, b, "h"))
        else:
            assert c0 == c1, "emitters must align in a row or a column"
            a, b = sorted(pts, key=lambda p: p[0])
            pairs.append((int(color), a, b, "v"))
    pairs.sort(key=lambda item: (item[1][0], item[1][1], item[2][0], item[2][1]))
    return pairs


def _prism_on_pair_path(
    prisms: list[tuple[int, int, int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
    orient: str,
) -> tuple[int, int, int, int] | None:
    hits: list[tuple[int, int, int, int]] = []
    if orient == "h":
        row = int(a[0])
        c0, c1 = int(a[1]), int(b[1])
        for prism in prisms:
            pr0, pc0, pr1, pc1 = prism
            if pr0 <= row <= pr1 and c0 < pc0 and pc1 < c1:
                hits.append(prism)
        if not hits:
            return None
        hits.sort(key=lambda prism: prism[1])
        assert len(hits) == 1, "generator guarantees at most one prism on each emitter path"
        return hits[0]

    col = int(a[1])
    r0, r1 = int(a[0]), int(b[0])
    for prism in prisms:
        pr0, pc0, pr1, pc1 = prism
        if pc0 <= col <= pc1 and r0 < pr0 and pr1 < r1:
            hits.append(prism)
    if not hits:
        return None
    hits.sort(key=lambda prism: prism[0])
    assert len(hits) == 1, "generator guarantees at most one prism on each emitter path"
    return hits[0]


def _paint_vertical_band(out: np.ndarray, color: int, col0: int, col1: int, row_iter) -> None:
    for r in row_iter:
        if not np.all(out[int(r), int(col0) : int(col1) + 1] == BLACK):
            break
        out[int(r), int(col0) : int(col1) + 1] = int(color)


def _paint_horizontal_band(out: np.ndarray, color: int, row0: int, row1: int, col_iter) -> None:
    for c in col_iter:
        if not np.all(out[int(row0) : int(row1) + 1, int(c)] == BLACK):
            break
        out[int(row0) : int(row1) + 1, int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    full = np.asarray(grid, dtype=int)
    assert full.ndim == 2, "input must be a 2D grid"

    legend_bbox, mapping = _parse_legend(full)
    main_h = int(legend_bbox[0])
    g = full[:main_h, :].copy()

    prisms = _find_prisms(g)
    pairs = _find_pairs(g)
    out = g.copy()

    for color, a, b, orient in pairs:
        prism = _prism_on_pair_path(prisms, a, b, orient)
        if prism is None:
            if orient == "h":
                r = int(a[0])
                out[r, int(a[1]) : int(b[1]) + 1] = int(color)
            else:
                c = int(a[1])
                out[int(a[0]) : int(b[0]) + 1, c] = int(color)
            continue

        assert color in mapping, "split color must exist in legend"
        side_a, side_b = mapping[int(color)]
        pr0, pc0, pr1, pc1 = prism

        if orient == "h":
            row = int(a[0])
            assert np.all(g[row, int(a[1]) + 1 : pc0] == BLACK), "horizontal beam must be clear before the prism"
            assert np.all(g[row, pc1 + 1 : int(b[1])] == BLACK), "horizontal beam must be clear after the prism"
            out[row, int(a[1]) : pc0] = int(color)
            out[row, pc1 + 1 : int(b[1]) + 1] = int(color)
            _paint_vertical_band(out, side_a, pc0, pc1, range(pr0 - 1, -1, -1))
            _paint_vertical_band(out, side_b, pc0, pc1, range(pr1 + 1, out.shape[0]))
        else:
            col = int(a[1])
            assert np.all(g[int(a[0]) + 1 : pr0, col] == BLACK), "vertical beam must be clear before the prism"
            assert np.all(g[pr1 + 1 : int(b[0]), col] == BLACK), "vertical beam must be clear after the prism"
            out[int(a[0]) : pr0, col] = int(color)
            out[pr1 + 1 : int(b[0]) + 1, col] = int(color)
            _paint_horizontal_band(out, side_a, pr0, pr1, range(pc0 - 1, -1, -1))
            _paint_horizontal_band(out, side_b, pr0, pr1, range(pc1 + 1, out.shape[1]))

    return out


def _mark_reserved(reserved: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, int(r) - pad), min(h, int(r) + pad + 1)):
            for cc in range(max(0, int(c) - pad), min(w, int(c) + pad + 1)):
                reserved[rr, cc] = True


def _can_place(reserved: np.ndarray, cells: set[tuple[int, int]], pad: int = 1) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        for rr in range(max(0, int(r) - pad), min(h, int(r) + pad + 1)):
            for cc in range(max(0, int(c) - pad), min(w, int(c) + pad + 1)):
                if bool(reserved[rr, cc]):
                    return False
    return True


def _prism_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {
        (int(r0), int(c0)),
        (int(r0), int(c0) + 1),
        (int(r0) + 1, int(c0)),
        (int(r0) + 1, int(c0) + 1),
    }


def _build_horizontal_split_spec(main_h: int, main_w: int, color: int) -> dict[str, object] | None:
    if main_h < 10 or main_w < 10:
        return None
    for _ in range(200):
        pr = random.randint(3, main_h - 5)
        pc = random.randint(3, main_w - 5)
        row = int(pr + random.randint(0, 1))
        left = random.randint(0, pc - 2)
        right = random.randint(pc + 3, main_w - 1)
        mode = random.choice(("none", "up", "down", "both"))

        cells = set()
        cells.update(_prism_cells(pr, pc))
        cells.add((row, left))
        cells.add((row, right))
        for c in range(left, pc):
            cells.add((row, c))
        for c in range(pc + 2, right + 1):
            cells.add((row, c))

        blockers: list[tuple[int, int]] = []
        up_start = 0
        down_stop = main_h - 1
        if mode in ("up", "both"):
            bu = random.randint(0, pr - 3)
            blockers.append((int(bu), int(pc)))
            cells.update(_prism_cells(int(bu), int(pc)))
            up_start = int(bu) + 2
        if mode in ("down", "both"):
            bd = random.randint(pr + 3, main_h - 2)
            blockers.append((int(bd), int(pc)))
            cells.update(_prism_cells(int(bd), int(pc)))
            down_stop = int(bd) - 1

        for r in range(up_start, pr):
            cells.add((r, pc))
            cells.add((r, pc + 1))
        for r in range(pr + 2, down_stop + 1):
            cells.add((r, pc))
            cells.add((r, pc + 1))

        return {
            "type": "split",
            "orient": "h",
            "color": int(color),
            "emitters": ((row, left), (row, right)),
            "prisms": [(int(pr), int(pc))] + blockers,
            "footprint": cells,
        }
    return None


def _build_vertical_split_spec(main_h: int, main_w: int, color: int) -> dict[str, object] | None:
    if main_h < 10 or main_w < 10:
        return None
    for _ in range(200):
        pr = random.randint(3, main_h - 5)
        pc = random.randint(3, main_w - 5)
        col = int(pc + random.randint(0, 1))
        top = random.randint(0, pr - 2)
        bottom = random.randint(pr + 3, main_h - 1)
        mode = random.choice(("none", "left", "right", "both"))

        cells = set()
        cells.update(_prism_cells(pr, pc))
        cells.add((top, col))
        cells.add((bottom, col))
        for r in range(top, pr):
            cells.add((r, col))
        for r in range(pr + 2, bottom + 1):
            cells.add((r, col))

        blockers: list[tuple[int, int]] = []
        left_start = 0
        right_stop = main_w - 1
        if mode in ("left", "both"):
            bl = random.randint(0, pc - 3)
            blockers.append((int(pr), int(bl)))
            cells.update(_prism_cells(int(pr), int(bl)))
            left_start = int(bl) + 2
        if mode in ("right", "both"):
            br = random.randint(pc + 3, main_w - 2)
            blockers.append((int(pr), int(br)))
            cells.update(_prism_cells(int(pr), int(br)))
            right_stop = int(br) - 1

        for c in range(left_start, pc):
            cells.add((pr, c))
            cells.add((pr + 1, c))
        for c in range(pc + 2, right_stop + 1):
            cells.add((pr, c))
            cells.add((pr + 1, c))

        return {
            "type": "split",
            "orient": "v",
            "color": int(color),
            "emitters": ((top, col), (bottom, col)),
            "prisms": [(int(pr), int(pc))] + blockers,
            "footprint": cells,
        }
    return None


def _build_plain_spec(main_h: int, main_w: int, color: int, orient: str) -> dict[str, object] | None:
    for _ in range(200):
        if orient == "h":
            row = random.randint(0, main_h - 1)
            left = random.randint(0, main_w - 4)
            right = random.randint(left + 3, main_w - 1)
            cells = {(int(row), int(c)) for c in range(left, right + 1)}
            emitters = ((int(row), int(left)), (int(row), int(right)))
        else:
            col = random.randint(0, main_w - 1)
            top = random.randint(0, main_h - 4)
            bottom = random.randint(top + 3, main_h - 1)
            cells = {(int(r), int(col)) for r in range(top, bottom + 1)}
            emitters = ((int(top), int(col)), (int(bottom), int(col)))
        return {
            "type": "plain",
            "orient": orient,
            "color": int(color),
            "emitters": emitters,
            "prisms": [],
            "footprint": cells,
        }
    return None


def _build_legend(order: list[int], mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    legend = np.full((len(order) + 2, LEGEND_WIDTH), BLACK, dtype=int)
    legend[0, :] = MAROON
    legend[-1, :] = MAROON
    legend[:, 0] = MAROON
    legend[:, -1] = MAROON
    for idx, color in enumerate(order, start=1):
        a, b = mapping[int(color)]
        legend[idx, 1] = int(color)
        legend[idx, 3] = int(a)
        legend[idx, 5] = int(b)
    return legend


def _sample_mapping(colors: list[int]) -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    for color in colors:
        choices = [c for c in EMITTER_COLORS if c != int(color)]
        a, b = random.sample(choices, 2)
        mapping[int(color)] = (int(a), int(b))
    return mapping


def _paint_input(main_h: int, main_w: int, specs: list[dict[str, object]], mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    order = list(mapping)
    random.shuffle(order)
    legend = _build_legend(order, mapping)

    grid = np.full((main_h + legend.shape[0], main_w), BLACK, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["emitters"]:
            grid[int(r), int(c)] = int(color)
        for pr, pc in spec["prisms"]:
            grid[int(pr) : int(pr) + 2, int(pc) : int(pc) + 2] = GREY

    legend_c0 = random.randint(0, main_w - legend.shape[1])
    grid[main_h:, legend_c0 : legend_c0 + legend.shape[1]] = legend
    return grid


def _scenario_orients(split_orient: str) -> list[str]:
    if split_orient == "h":
        return random.choice((["v"], ["h"], ["v", "h"]))
    return random.choice((["h"], ["v"], ["h", "v"]))


def _gen_attempt() -> np.ndarray | None:
    main_h = random.randint(12, 18)
    main_w = random.randint(12, 18)
    split_orient = random.choice(("h", "v"))
    plain_orients = _scenario_orients(split_orient)

    pair_count = 1 + len(plain_orients)
    colors = random.sample(EMITTER_COLORS, pair_count)
    mapping = _sample_mapping(colors)

    reserved = np.zeros((main_h, main_w), dtype=bool)
    specs: list[dict[str, object]] = []

    split_builder = _build_horizontal_split_spec if split_orient == "h" else _build_vertical_split_spec
    split_spec = split_builder(main_h, main_w, colors[0])
    if split_spec is None or not _can_place(reserved, split_spec["footprint"], pad=1):
        return None
    _mark_reserved(reserved, split_spec["footprint"], pad=1)
    specs.append(split_spec)

    for color, orient in zip(colors[1:], plain_orients):
        placed = None
        for _ in range(200):
            spec = _build_plain_spec(main_h, main_w, color, orient)
            if spec is None:
                continue
            if _can_place(reserved, spec["footprint"], pad=1):
                placed = spec
                break
        if placed is None:
            return None
        _mark_reserved(reserved, placed["footprint"], pad=1)
        specs.append(placed)

    grid = _paint_input(main_h, main_w, specs, mapping)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != (main_h, main_w):
        return None
    if np.array_equal(out, grid[:main_h]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-splitting ARC input")
