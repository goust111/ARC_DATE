from __future__ import annotations

from collections import Counter, deque

import numpy as np


RNG = np.random.default_rng()

BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_ORDER = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_frame_component(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    if not bool((grid[r0, c0 : c1 + 1] == color).all()):
        return False
    if not bool((grid[r1, c0 : c1 + 1] == color).all()):
        return False
    if not bool((grid[r0 : r1 + 1, c0] == color).all()):
        return False
    if not bool((grid[r0 : r1 + 1, c1] == color).all()):
        return False
    if r1 - r0 > 1 and c1 - c0 > 1 and bool((grid[r0 + 1 : r1, c0 + 1 : c1] == color).any()):
        return False
    return True


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[tuple[int, int], int, tuple[int, int, int, int]] | None = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            if not _is_frame_component(grid, color, coords):
                continue
            box = _bbox(coords)
            score = ((box[2] - box[0] + 1) * (box[3] - box[1] + 1), int(coords.shape[0]))
            if best is None or score > best[0]:
                best = (score, int(color), box)
    assert best is not None, "expected a single large frame"
    return int(best[1]), best[2]


def _nearest_direction(anchor: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    ar, ac = anchor
    r0, c0, r1, c1 = frame_bbox
    dists = [
        ("up", int(ar - r0)),
        ("down", int(r1 - ar)),
        ("left", int(ac - c0)),
        ("right", int(c1 - ac)),
    ]
    return min(dists, key=lambda item: (item[1], DIR_ORDER.index(item[0])))[0]


def _ray_start(anchor: tuple[int, int], frame_bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    ar, ac = anchor
    r0, c0, r1, c1 = frame_bbox
    if direction == "up":
        return int(r0 - 1), int(ac)
    if direction == "down":
        return int(r1 + 1), int(ac)
    if direction == "left":
        return int(ar), int(c0 - 1)
    assert direction == "right"
    return int(ar), int(c1 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each two-color fragment onto the matching anchor inside the frame,
    then emit an outward ray whose direction is the anchor's nearest frame edge
    and whose length is the fragment's body-pixel count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox

    anchors: dict[int, tuple[int, int]] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color in {bg, frame_color}:
                continue
            assert color not in anchors, "anchor colors must be unique"
            anchors[color] = (int(r), int(c))
    assert anchors, "expected colored anchors inside the frame"

    outside_mask = g != bg
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    fragments = _components(outside_mask)
    assert fragments, "expected scattered fragments outside the frame"

    out = g.copy()
    for frag in fragments:
        out[frag[:, 0], frag[:, 1]] = int(bg)

    placements: list[tuple[tuple[int, int], int, int]] = []
    for frag in fragments:
        colors = [int(g[r, c]) for r, c in frag]
        counts = Counter(colors)
        assert len(counts) == 2, "each fragment must use exactly two colors"

        anchor_candidates = [color for color, count in counts.items() if count == 1 and color in anchors]
        if len(anchor_candidates) != 1:
            anchor_candidates = [color for color, count in counts.items() if count == 1]
        assert len(anchor_candidates) == 1, "expected a unique single-pixel anchor color"

        anchor_color = int(anchor_candidates[0])
        assert anchor_color in anchors, "fragment anchor color must match an internal anchor"
        body_colors = [color for color in counts if color != anchor_color]
        assert len(body_colors) == 1
        body_color = int(body_colors[0])
        body_count = int(counts[body_color])

        anchor_index = next(i for i, color in enumerate(colors) if color == anchor_color)
        src_anchor = tuple(map(int, frag[anchor_index]))
        dst_anchor = anchors[anchor_color]
        shift_r = int(dst_anchor[0] - src_anchor[0])
        shift_c = int(dst_anchor[1] - src_anchor[1])

        for r, c in frag:
            nr = int(r + shift_r)
            nc = int(c + shift_c)
            assert 0 <= nr < h and 0 <= nc < w
            out[nr, nc] = int(g[r, c])
        placements.append((dst_anchor, body_color, body_count))

    for anchor, body_color, body_count in placements:
        direction = _nearest_direction(anchor, frame_bbox)
        rr, cc = _ray_start(anchor, frame_bbox, direction)
        dr, dc = DIR_VECTORS[direction]
        for _ in range(body_count):
            assert 0 <= rr < h and 0 <= cc < w, "ray must fit in the canvas"
            out[rr, cc] = int(body_color)
            rr += int(dr)
            cc += int(dc)

    return out


def _random_body(area: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = sorted(
            {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in NEIGHBORS4
                if (r + dr, c + dc) not in cells
            }
        )
        cells.add(frontier[int(RNG.integers(len(frontier)))])
    return sorted(cells)


def _make_piece(direction: str, body_color: int, anchor_color: int) -> dict:
    body_area = int(RNG.integers(2, 6))
    body = _random_body(body_area)
    frontier = sorted(
        {
            (r + dr, c + dc)
            for r, c in body
            for dr, dc in NEIGHBORS4
            if (r + dr, c + dc) not in body
        }
    )
    anchor = frontier[int(RNG.integers(len(frontier)))]
    body_rels = sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in body)
    rows = [0] + [int(r) for r, _ in body_rels]
    cols = [0] + [int(c) for _, c in body_rels]
    return {
        "direction": direction,
        "body_color": int(body_color),
        "anchor_color": int(anchor_color),
        "body_rels": body_rels,
        "body_area": body_area,
        "span_h": int(max(rows) - min(rows) + 1),
        "span_w": int(max(cols) - min(cols) + 1),
    }


def _anchor_dists(anchor_local: tuple[int, int], inner_h: int, inner_w: int) -> dict[str, int]:
    r, c = anchor_local
    return {
        "up": int(r),
        "down": int(inner_h + 1 - r),
        "left": int(c),
        "right": int(inner_w + 1 - c),
    }


def _piece_cells_local(piece: dict, anchor_local: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor_local
    cells = [(int(ar), int(ac))]
    cells.extend((int(ar + dr), int(ac + dc)) for dr, dc in piece["body_rels"])
    return cells


def _far_enough(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for orow, ocol in occupied:
            if max(abs(int(r - orow)), abs(int(c - ocol))) <= 1:
                return False
    return True


def _place_inside(pieces: list[dict], inner_h: int, inner_w: int) -> dict[int, tuple[int, int]] | None:
    occupied: set[tuple[int, int]] = set()
    placements: dict[int, tuple[int, int]] = {}
    order = list(range(len(pieces)))
    RNG.shuffle(order)
    order.sort(key=lambda idx: -int(pieces[idx]["body_area"]))

    for idx in order:
        piece = pieces[idx]
        candidates: list[tuple[int, tuple[int, int]]] = []
        for ar in range(1, inner_h + 1):
            for ac in range(1, inner_w + 1):
                dists = _anchor_dists((ar, ac), inner_h, inner_w)
                near = int(dists[piece["direction"]])
                if any(name != piece["direction"] and int(dist) <= near for name, dist in dists.items()):
                    continue

                cells = _piece_cells_local(piece, (ar, ac))
                if any(not (1 <= r <= inner_h and 1 <= c <= inner_w) for r, c in cells):
                    continue
                if any((r, c) in occupied for r, c in cells):
                    continue
                if not _far_enough(cells, occupied):
                    continue
                candidates.append((near, (int(ar), int(ac))))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_dist = int(candidates[0][0])
        shortlist = [anchor for dist, anchor in candidates if dist <= best_dist + 1]
        anchor_local = shortlist[int(RNG.integers(len(shortlist)))]
        placements[idx] = anchor_local
        occupied.update(_piece_cells_local(piece, anchor_local))

    return placements


def _mark_padded(mask: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, int(r - pad))
        r1 = min(h - 1, int(r + pad))
        c0 = max(0, int(c - pad))
        c1 = min(w - 1, int(c + pad))
        mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _scatter_piece_cells(
    grid: np.ndarray,
    pieces: list[dict],
    frame_bbox: tuple[int, int, int, int],
) -> dict[int, list[tuple[int, int, int]]] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    scattered: dict[int, list[tuple[int, int, int]]] = {}
    order = list(range(len(pieces)))
    RNG.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        rel_cells = [(0, 0, int(piece["anchor_color"]))] + [
            (int(dr), int(dc), int(piece["body_color"])) for dr, dc in piece["body_rels"]
        ]
        for _ in range(600):
            ar = int(RNG.integers(0, h))
            ac = int(RNG.integers(0, w))
            abs_cells: list[tuple[int, int, int]] = []
            ok = True
            for dr, dc, color in rel_cells:
                rr = int(ar + dr)
                cc = int(ac + dc)
                if not (0 <= rr < h and 0 <= cc < w) or bool(forbidden[rr, cc]):
                    ok = False
                    break
                abs_cells.append((rr, cc, int(color)))
            if not ok:
                continue

            scattered[idx] = abs_cells
            _mark_padded(forbidden, [(r, c) for r, c, _ in abs_cells], pad=1)
            break
        else:
            return None

    return scattered


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _expected_output(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    pieces: list[dict],
    placements: dict[int, tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _draw_frame(grid, frame_bbox, FRAME)
    frame_r0, frame_c0, _frame_r1, _frame_c1 = frame_bbox

    for idx, piece in enumerate(pieces):
        ar_local, ac_local = placements[idx]
        ar = int(frame_r0 + ar_local)
        ac = int(frame_c0 + ac_local)
        anchor = (ar, ac)
        assert _nearest_direction(anchor, frame_bbox) == piece["direction"]
        grid[ar, ac] = int(piece["anchor_color"])
        for dr, dc in piece["body_rels"]:
            grid[ar + int(dr), ac + int(dc)] = int(piece["body_color"])

    for idx, piece in enumerate(pieces):
        ar_local, ac_local = placements[idx]
        anchor = (int(frame_r0 + ar_local), int(frame_c0 + ac_local))
        rr, cc = _ray_start(anchor, frame_bbox, piece["direction"])
        dr, dc = DIR_VECTORS[piece["direction"]]
        for _ in range(int(piece["body_area"])):
            grid[rr, cc] = int(piece["body_color"])
            rr += int(dr)
            cc += int(dc)

    return grid


def _gen_attempt() -> np.ndarray | None:
    piece_count = int(RNG.integers(2, 5))
    directions = list(RNG.choice(np.array(DIR_ORDER), size=piece_count, replace=False))
    colors = [int(color) for color in RNG.permutation(np.array(PALETTE))]
    body_colors = colors[:piece_count]
    anchor_colors = colors[piece_count : 2 * piece_count]

    pieces = [
        _make_piece(str(direction), int(body_color), int(anchor_color))
        for direction, body_color, anchor_color in zip(directions, body_colors, anchor_colors)
    ]

    min_inner_h = max(8, max(int(piece["span_h"]) for piece in pieces) + 4, piece_count + 5)
    min_inner_w = max(8, max(int(piece["span_w"]) for piece in pieces) + 4, piece_count + 5)
    inner_h = int(RNG.integers(min_inner_h, min_inner_h + 5))
    inner_w = int(RNG.integers(min_inner_w, min_inner_w + 5))
    placements = _place_inside(pieces, inner_h, inner_w)
    if placements is None:
        return None

    max_ray = max(int(piece["body_area"]) for piece in pieces)
    margin_base = max_ray + 3
    top_margin = int(RNG.integers(margin_base, margin_base + 5))
    bottom_margin = int(RNG.integers(margin_base, margin_base + 5))
    left_margin = int(RNG.integers(margin_base, margin_base + 5))
    right_margin = int(RNG.integers(margin_base, margin_base + 5))

    frame_h = int(inner_h + 2)
    frame_w = int(inner_w + 2)
    h = int(top_margin + frame_h + bottom_margin)
    w = int(left_margin + frame_w + right_margin)
    frame_bbox = (
        int(top_margin),
        int(left_margin),
        int(top_margin + frame_h - 1),
        int(left_margin + frame_w - 1),
    )

    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid, frame_bbox, FRAME)

    frame_r0, frame_c0, _frame_r1, _frame_c1 = frame_bbox
    for idx, piece in enumerate(pieces):
        ar_local, ac_local = placements[idx]
        grid[frame_r0 + ar_local, frame_c0 + ac_local] = int(piece["anchor_color"])

    scattered = _scatter_piece_cells(grid, pieces, frame_bbox)
    if scattered is None:
        return None
    for abs_cells in scattered.values():
        for r, c, color in abs_cells:
            grid[r, c] = int(color)

    expected = _expected_output(grid.shape, frame_bbox, pieces, placements)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if int(np.sum(grid == BG)) <= int(grid.size // 2):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
