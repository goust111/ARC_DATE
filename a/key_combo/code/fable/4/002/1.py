from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 4
PATTERN_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _find_frame(g: np.ndarray) -> tuple[int, int, int, int]:
    yellow = np.argwhere(g == FRAME)
    assert yellow.size > 0, "expected one yellow query frame"
    r0, c0, r1, c1 = _bbox(yellow)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "query frame must have an interior"

    expected = np.zeros_like(g, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    assert np.array_equal(g == FRAME, expected), "yellow cells must form exactly one hollow rectangle"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG), "query frame interior must be black"
    return r0, c0, r1, c1


def _fragment_bboxes(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = (g != BG) & (g != FRAME)
    comps = _components(mask)
    assert comps, "expected at least one pattern fragment"
    boxes: list[tuple[int, int, int, int]] = []
    for coords in comps:
        b = _bbox(coords)
        r0, c0, r1, c1 = b
        patch = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch)), "pattern fragments must be solid rectangles"
        assert FRAME not in set(map(int, g[r0 : r1 + 1, c0 : c1 + 1].ravel()))
        boxes.append(b)
    return boxes


def _samples(g: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((g != BG) & (g != FRAME))
    return [(int(r), int(c), int(g[r, c])) for r, c in coords]


def _query_coords(frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _candidate_tile(
    samples: list[tuple[int, int, int]],
    query: list[tuple[int, int]],
    pr: int,
    pc: int,
) -> dict[tuple[int, int], int] | None:
    tile: dict[tuple[int, int], int] = {}
    for r, c, color in samples:
        key = (r % pr, c % pc)
        if key in tile and tile[key] != color:
            return None
        tile[key] = color

    needed = {(r % pr, c % pc) for r, c in query}
    if not needed.issubset(tile):
        return None
    return tile


def _infer_period(
    samples: list[tuple[int, int, int]],
    query: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int, dict[tuple[int, int], int]]:
    h, w = shape
    candidates: list[tuple[tuple[int, int, int, int], int, int, dict[tuple[int, int], int]]] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = _candidate_tile(samples, query, pr, pc)
            if tile is not None:
                candidates.append(((pr * pc, pr + pc, pr, pc), pr, pc, tile))
    assert candidates, "visible fragments do not determine the query window"
    candidates.sort(key=lambda item: item[0])
    best_key, best_pr, best_pc, best_tile = candidates[0]
    assert len(candidates) == 1 or candidates[1][0] != best_key, "period choice is ambiguous"
    return best_pr, best_pc, best_tile


def _render_query(
    frame: tuple[int, int, int, int],
    pr: int,
    pc: int,
    tile: dict[tuple[int, int], int],
) -> np.ndarray:
    r0, c0, r1, c1 = frame
    out = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=int)
    for rr, r in enumerate(range(r0 + 1, r1)):
        for cc, c in enumerate(range(c0 + 1, c1)):
            key = (r % pr, c % pc)
            assert key in tile, "query uses an unobserved phase cell"
            out[rr, cc] = tile[key]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a hidden globally anchored periodic pattern from scattered fragments and return the query-frame contents.

    Input structure:
    - A black background contains several disjoint solid rectangular multicolor fragments.
    - Every non-black, non-yellow fragment cell is a sample of one shared periodic pattern indexed by the cell's full-grid row and column.
    - Exactly one yellow one-cell-thick hollow rectangle has a black interior and does not overlap any fragment.

    Rule:
    - Treat all fragment cells as samples of one global pattern whose row and column periods are measured in full-grid coordinates.
    - Enumerate row/column periods consistent with all samples, keeping only candidates whose sampled phases cover every cell needed by the yellow frame's interior.
    - Use the unique smallest consistent period and fill an output grid the same size as the frame interior from those global-coordinate phases.

    Key insight:
    - The colored rectangles are not independent patches to copy; together they reveal the phase of one hidden periodic sheet, and the yellow frame is only a query window into that sheet.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30

    frame = _find_frame(g)
    _fragment_bboxes(g)
    samples = _samples(g)
    query = _query_coords(frame)
    pr, pc, tile = _infer_period(samples, query, g.shape)
    return _render_query(frame, pr, pc, tile)


def _period_color(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[int(r) % tile.shape[0], int(c) % tile.shape[1]])


def _draw_frame(g: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    g[r0 : r1 + 1, c0] = FRAME
    g[r0 : r1 + 1, c1] = FRAME
    g[r0, c0 : c1 + 1] = FRAME
    g[r1, c0 : c1 + 1] = FRAME


def _clear_with_padding(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(rows, r0 + h + pad)
    cc1 = min(cols, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _random_frame(rows: int, cols: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    h = inner_h + 2
    w = inner_w + 2
    r0 = random.randint(1, rows - h - 1)
    c0 = random.randint(1, cols - w - 1)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_horizontal_strips(
    g: np.ndarray,
    occupied: np.ndarray,
    tile: np.ndarray,
) -> list[tuple[int, int, int, int]] | None:
    rows, cols = g.shape
    pr, pc = tile.shape
    for _ in range(120):
        c0 = random.randint(1, cols - pc - 1)
        boxes: list[tuple[int, int, int, int]] = []
        temp = occupied.copy()
        residues = list(range(pr))
        random.shuffle(residues)
        ok = True
        for residue in residues:
            choices = [
                r
                for r in range(1, rows - 1)
                if r % pr == residue and _clear_with_padding(temp, r, c0, 1, pc)
            ]
            if not choices:
                ok = False
                break
            r = random.choice(choices)
            _reserve(temp, r, c0, 1, pc)
            boxes.append((r, c0, r, c0 + pc - 1))
        if ok:
            for r0, c0, r1, c1 in boxes:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        g[r, c] = _period_color(tile, r, c)
                _reserve(occupied, r0, c0, r1 - r0 + 1, c1 - c0 + 1)
            return boxes
    return None


def _place_vertical_strips(
    g: np.ndarray,
    occupied: np.ndarray,
    tile: np.ndarray,
) -> list[tuple[int, int, int, int]] | None:
    rows, cols = g.shape
    pr, pc = tile.shape
    for _ in range(120):
        r0 = random.randint(1, rows - pr - 1)
        boxes: list[tuple[int, int, int, int]] = []
        temp = occupied.copy()
        residues = list(range(pc))
        random.shuffle(residues)
        ok = True
        for residue in residues:
            choices = [
                c
                for c in range(1, cols - 1)
                if c % pc == residue and _clear_with_padding(temp, r0, c, pr, 1)
            ]
            if not choices:
                ok = False
                break
            c = random.choice(choices)
            _reserve(temp, r0, c, pr, 1)
            boxes.append((r0, c, r0 + pr - 1, c))
        if ok:
            for r0, c0, r1, c1 in boxes:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        g[r, c] = _period_color(tile, r, c)
                _reserve(occupied, r0, c0, r1 - r0 + 1, c1 - c0 + 1)
            return boxes
    return None


def _random_tile(pr: int, pc: int) -> np.ndarray | None:
    palette = random.sample(PATTERN_COLORS, random.randint(3, 5))
    for _ in range(200):
        tile = np.array([[random.choice(palette) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.ravel()))) < 3:
            continue
        samples = [(r, c, int(tile[r, c])) for r in range(pr) for c in range(pc)]
        query = [(r, c) for r in range(pr) for c in range(pc)]
        best_pr, best_pc, _tile = _infer_period(samples, query, (pr, pc))
        if (best_pr, best_pc) == (pr, pc):
            return tile
    return None


def _fragment_has_alternative_period(
    box: tuple[int, int, int, int],
    g: np.ndarray,
    true_period: tuple[int, int],
) -> bool:
    r0, c0, r1, c1 = box
    samples = [(r, c, int(g[r, c])) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    max_pr = min(g.shape[0], true_period[0] + 4)
    max_pc = min(g.shape[1], true_period[1] + 4)
    seen: set[tuple[int, int]] = set()
    for pr in range(1, max_pr + 1):
        for pc in range(1, max_pc + 1):
            tile: dict[tuple[int, int], int] = {}
            ok = True
            for r, c, color in samples:
                key = (r % pr, c % pc)
                if key in tile and tile[key] != color:
                    ok = False
                    break
                tile[key] = color
            if ok:
                seen.add((pr, pc))
                if len(seen) >= 2 and (pr, pc) != true_period:
                    return True
    return False


def _validate_generated(
    g: np.ndarray,
    tile: np.ndarray,
    frame: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    try:
        parsed_frame = _find_frame(g)
        if parsed_frame != frame:
            return False
        parsed_boxes = _fragment_bboxes(g)
        if sorted(parsed_boxes) != sorted(boxes):
            return False

        samples = _samples(g)
        query = _query_coords(frame)
        pr, pc, inferred = _infer_period(samples, query, g.shape)
        if (pr, pc) != tile.shape:
            return False

        expected = np.array(
            [[_period_color(tile, r, c) for c in range(frame[1] + 1, frame[3])] for r in range(frame[0] + 1, frame[2])],
            dtype=int,
        )
        if not np.array_equal(_render_query(frame, pr, pc, inferred), expected):
            return False
        if len(set(map(int, expected.ravel()))) < 3:
            return False

        for box in boxes:
            if not _fragment_has_alternative_period(box, g, tile.shape):
                return False

        for remove_idx, box in enumerate(boxes):
            r0, c0, r1, c1 = box
            reduced = g.copy()
            reduced[r0 : r1 + 1, c0 : c1 + 1] = BG
            try:
                reduced_samples = _samples(reduced)
                rpr, rpc, _ = _infer_period(reduced_samples, query, reduced.shape)
            except AssertionError:
                continue
            if (rpr, rpc) == tile.shape:
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    pr = random.randint(3, 6)
    pc = random.randint(3, 6)
    tile = _random_tile(pr, pc)
    if tile is None:
        return None

    inner_h = random.randint(pr, min(9, pr + 4))
    inner_w = random.randint(pc, min(9, pc + 4))
    rows = random.randint(max(20, inner_h + 8, pr * 3 + 4), 30)
    cols = random.randint(max(20, inner_w + 8, pc * 3 + 4), 30)
    g = np.zeros((rows, cols), dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    frame = _random_frame(rows, cols, inner_h, inner_w)
    fr0, fc0, fr1, fc1 = frame
    _reserve(occupied, fr0, fc0, fr1 - fr0 + 1, fc1 - fc0 + 1)
    _draw_frame(g, frame)

    placers = [_place_horizontal_strips, _place_vertical_strips]
    random.shuffle(placers)
    boxes = placers[0](g, occupied, tile)
    if boxes is None:
        boxes = placers[1](g, occupied, tile)
    if boxes is None:
        return None
    if not _validate_generated(g, tile, frame, boxes):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-period query input")
