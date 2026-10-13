from __future__ import annotations

import itertools
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_FLIP = {
    "top": "lr",
    "bottom": "lr",
    "left": "ud",
    "right": "ud",
}
SIDE_DELTAS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_BENDS = {
    "top": [(0, -1), (0, 1)],
    "bottom": [(0, -1), (0, 1)],
    "left": [(-1, 0), (1, 0)],
    "right": [(-1, 0), (1, 0)],
}


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[set[tuple[int, int]]] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = set()
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _l_candidate_info(
    arrow: set[tuple[int, int]],
    body: set[tuple[int, int]],
) -> tuple[str, str] | None:
    if len(arrow) != 3 or len(body) < 5:
        return None
    if not _is_connected(body):
        return None

    degrees: dict[tuple[int, int], int] = {}
    for cell in arrow:
        degrees[cell] = sum((cell[0] + dr, cell[1] + dc) in arrow for dr, dc in NEIGHBORS4)
    if sorted(degrees.values()) != [1, 1, 2]:
        return None

    corner = next(cell for cell, degree in degrees.items() if degree == 2)
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]

    body_contacts: list[tuple[tuple[int, int], tuple[int, int]]] = []
    touching_arrow_cells = set()
    for cell in arrow:
        for dr, dc in NEIGHBORS4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr in body:
                body_contacts.append((cell, nbr))
                touching_arrow_cells.add(cell)
    if len(body_contacts) != 1 or len(touching_arrow_cells) != 1:
        return None

    anchor = next(iter(touching_arrow_cells))
    if anchor == corner:
        return None
    bend = endpoints[0] if endpoints[1] == anchor else endpoints[1]

    dr1 = corner[0] - anchor[0]
    dc1 = corner[1] - anchor[1]
    dr2 = bend[0] - corner[0]
    dc2 = bend[1] - corner[1]
    if abs(dr1) + abs(dc1) != 1 or abs(dr2) + abs(dc2) != 1:
        return None

    if dr1 != 0 and dc1 == 0 and dr2 == 0 and dc2 != 0:
        flip = "lr"
    elif dc1 != 0 and dr1 == 0 and dc2 == 0 and dr2 != 0:
        flip = "ud"
    else:
        return None

    r0, c0, r1, c1 = _bbox(body)
    side_hits = []
    if all(r < r0 for r, _ in arrow) and max(r for r, _ in arrow) == r0 - 1:
        side_hits.append("top")
    if all(r > r1 for r, _ in arrow) and min(r for r, _ in arrow) == r1 + 1:
        side_hits.append("bottom")
    if all(c < c0 for _, c in arrow) and max(c for _, c in arrow) == c0 - 1:
        side_hits.append("left")
    if all(c > c1 for _, c in arrow) and min(c for _, c in arrow) == c1 + 1:
        side_hits.append("right")
    if len(side_hits) != 1:
        return None

    side = side_hits[0]
    if SIDE_TO_FLIP[side] != flip:
        return None
    return side, flip


def _split_sprite_and_arrow(
    coords: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], str]:
    candidates = []
    for subset in itertools.combinations(sorted(coords), 3):
        arrow = set(subset)
        body = coords - arrow
        info = _l_candidate_info(arrow, body)
        if info is None:
            continue
        _side, flip = info
        candidates.append((body, arrow, flip))

    assert len(candidates) == 1, "expected exactly one removable L marker per sprite"
    return candidates[0]


def _flip_body_coords(body: set[tuple[int, int]], flip: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(body)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in body:
        mask[r - r0, c - c0] = True

    if flip == "lr":
        mask = np.fliplr(mask)
    else:
        assert flip == "ud"
        mask = np.flipud(mask)

    out = set()
    rows, cols = np.where(mask)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        out.add((r0 + rr, c0 + cc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each monochrome sprite in place according to its attached same-color L marker.

    Input structure:
    - The dominant color is the background.
    - Every other monochrome 4-connected component is a same-color composite:
      a sprite body of at least 5 cells plus an attached 3-cell L marker.
    - The L marker sits directly outside one side of the body and uses the same
      color as that body, so sprite and marker arrive as one connected component.

    Rule:
    - For each non-background component, identify the unique 3-cell L subset
      whose removal leaves a connected sprite body.
    - If the L bends left/right relative to its 2-cell stem, reflect the body
      left-right inside its own bounding box; if it bends up/down, reflect the
      body up-down inside its own bounding box.
    - Remove the L marker by restoring those cells to the background.
    - Leave every transformed sprite in its original body position and keep the
      overall grid size unchanged.

    Key insight:
    - The small same-color appendage is not part of the sprite itself: it is a
      detachable L code whose bend direction tells you which mirror axis to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in _component_sets(g == color):
            body, _arrow, flip = _split_sprite_and_arrow(coords)
            flipped = _flip_body_coords(body, flip)
            for r, c in coords:
                out[r, c] = bg
            for r, c in flipped:
                out[r, c] = color

    return out


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _grow_body() -> set[tuple[int, int]]:
    target_area = random.randint(5, 11)
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    start = (random.randrange(h), random.randrange(w))
    body = {start}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(body) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        body.add(cell)
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in body:
                frontier.add((nr, nc))

    body = _normalize(body)
    if len(body) < 5:
        return _grow_body()
    return body


def _add_arrow(
    body: set[tuple[int, int]],
    side: str,
    bend_delta: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(body)
    if side == "top":
        anchors = [(r, c) for r, c in body if r == r0]
    elif side == "bottom":
        anchors = [(r, c) for r, c in body if r == r1]
    elif side == "left":
        anchors = [(r, c) for r, c in body if c == c0]
    else:
        anchors = [(r, c) for r, c in body if c == c1]

    anchor = random.choice(anchors)
    dr, dc = SIDE_DELTAS[side]
    stem0 = (anchor[0] + dr, anchor[1] + dc)
    stem1 = (anchor[0] + 2 * dr, anchor[1] + 2 * dc)
    bend = (stem1[0] + bend_delta[0], stem1[1] + bend_delta[1])
    arrow = {stem0, stem1, bend}
    assert not (arrow & body)
    union = body | arrow
    shift_r = -min(r for r, _ in union)
    shift_c = -min(c for _, c in union)
    shifted_body = {(r + shift_r, c + shift_c) for r, c in body}
    shifted_arrow = {(r + shift_r, c + shift_c) for r, c in arrow}
    return shifted_body, shifted_arrow


def _patch_from_coords(
    coords: set[tuple[int, int]],
    color: int,
    bg: int,
) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _make_object(bg: int, color: int) -> dict | None:
    for _ in range(200):
        body = _grow_body()
        side = random.choice(["top", "bottom", "left", "right"])
        bend = random.choice(SIDE_BENDS[side])
        body, arrow = _add_arrow(body, side, bend)
        union = body | arrow
        try:
            det_body, det_arrow, flip = _split_sprite_and_arrow(union)
        except AssertionError:
            continue
        if det_arrow != arrow or det_body != body:
            continue
        return {
            "color": color,
            "coords": union,
            "body": body,
            "arrow": arrow,
            "flip": flip,
            "patch": _patch_from_coords(union, color, bg),
        }
    return None


def _can_place(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int = 1,
) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    block = patch != bg
    canvas[r0 : r0 + h, c0 : c0 + w][block] = patch[block]


def _paint_coords(
    canvas: np.ndarray,
    coords: set[tuple[int, int]],
    color: int,
    offset_r: int,
    offset_c: int,
) -> None:
    for r, c in coords:
        canvas[offset_r + r, offset_c + c] = color


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randrange(10)
    count = random.randint(1, 4)
    colors = random.sample([c for c in range(10) if c != bg], count)
    objects = []
    for color in colors:
        obj = _make_object(bg, color)
        if obj is None:
            return None
        objects.append(obj)

    max_h = max(obj["patch"].shape[0] for obj in objects)
    max_w = max(obj["patch"].shape[1] for obj in objects)
    H = random.randint(max(14, max_h + 4), min(30, max(18, max_h + 14)))
    W = random.randint(max(14, max_w + 4), min(30, max(18, max_w + 14)))

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = objects[:]
    random.shuffle(order)
    order.sort(key=lambda obj: obj["patch"].shape[0] * obj["patch"].shape[1], reverse=True)

    for obj in order:
        patch = obj["patch"]
        h, w = patch.shape
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place(occupied, r0, c0, h, w, pad=1):
                continue
            _paint_patch(inp, patch, r0, c0, bg)
            flipped = _flip_body_coords(obj["body"], obj["flip"])
            _paint_coords(out, flipped, obj["color"], r0, c0)
            occupied[r0 : r0 + h, c0 : c0 + w] |= patch != bg
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
