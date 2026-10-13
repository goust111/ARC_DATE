from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500
C_COLORS = (BLUE, GREEN)
D_COLORS = (RED, YELLOW)
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (21, 29)
BRACKET_LEN_RANGE = (3, 5)

C_SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
]

D_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _apply_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = dst.shape
    mh, mw = mask.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h, r0 + mh)
    cc1 = min(w, c0 + mw)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    submask = mask[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]
    view = dst[rr0:rr1, cc0:cc1]
    view[submask] = color


def _corner_shapes(r0: int, c0: int, r1: int, c1: int) -> list[set[tuple[int, int]]]:
    return [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]


def _is_corner_bracket(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or h > 5 or w < 3 or w > 5:
        return False
    if len(coords) != h + w - 1:
        return False
    coord_set = set(coords)
    return any(coord_set == candidate for candidate in _corner_shapes(r0, c0, r1, c1))


def _parse(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))

    grey_parts = _components(g == GREY)
    brackets = [coords for coords in grey_parts if _is_corner_bracket(coords)]
    assert len(brackets) == 1, "expected exactly one grey corner bracket"
    bracket_coords = brackets[0]
    bracket_set = set(bracket_coords)
    bracket_bbox = _bbox(bracket_coords)
    br0, bc0, br1, bc1 = bracket_bbox

    markers = [
        (r, c)
        for r in range(br0, br1 + 1)
        for c in range(bc0, bc1 + 1)
        if (r, c) not in bracket_set and g[r, c] != BLACK
    ]
    assert len(markers) == 1, "expected exactly one marker inside the bracket"
    marker = markers[0]
    c_color = int(g[marker])
    assert c_color in C_COLORS

    other_colors = sorted(set(map(int, np.unique(g))) - {BLACK, GREY, c_color})
    assert len(other_colors) == 1, "expected exactly one counter color"
    d_color = other_colors[0]
    assert d_color in D_COLORS

    c_mask = g == c_color
    c_mask[marker] = False
    d_mask = g == d_color

    c_parts = _components(c_mask)
    d_parts = _components(d_mask)
    assert 1 <= len(c_parts) <= 3
    assert 2 <= len(d_parts) <= 5

    for coords in c_parts:
        assert 3 <= len(coords) <= 6
    for coords in d_parts:
        assert 2 <= len(coords) <= 4

    c_items = []
    for coords in c_parts:
        bb = _bbox(coords)
        c_items.append({"coords": coords, "bbox": bb, "mask": _crop_component_mask(coords)})

    c_items.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert len(c_items) == 1 or (
        c_items[0]["bbox"][0], c_items[0]["bbox"][1]
    ) < (
        c_items[1]["bbox"][0], c_items[1]["bbox"][1]
    ), "template selector must be unique"

    return {
        "grid": g,
        "marker": marker,
        "c_color": c_color,
        "d_color": d_color,
        "bracket_coords": bracket_coords,
        "bracket_bbox": bracket_bbox,
        "c_items": c_items,
        "d_parts": d_parts,
        "template_mask": c_items[0]["mask"],
        "k": len(d_parts),
    }


def _render_from_parts(
    shape: tuple[int, int],
    bracket_coords: list[tuple[int, int]],
    marker: tuple[int, int],
    c_color: int,
    template_mask: np.ndarray,
    anchor: tuple[int, int],
    k: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    stamp = np.zeros(shape, dtype=int)
    h, w = template_mask.shape
    start_r, start_c = anchor

    slot_positions = []
    idx = 1
    c = start_c
    while c < shape[1]:
        slot_positions.append((idx, start_r, c))
        idx += 1
        c += w + 1

    for idx, r0, c0 in slot_positions:
        color = PINK if idx % k == 0 else c_color
        _apply_mask(stamp, template_mask, r0, c0, color)

    for idx, r0, c0 in slot_positions:
        if idx % k != 0:
            continue
        r = r0
        while r < shape[0]:
            _apply_mask(stamp, template_mask, r, c0, PINK)
            r += h + 1

    out[stamp != BLACK] = stamp[stamp != BLACK]
    for r, c in bracket_coords:
        out[r, c] = GREY
    out[marker] = c_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the marked-color template, count the other color's objects, and stamp a periodic grid whose magenta rhythm uses that count.

    Input structure:
    - Black background contains exactly one grey 4-connected corner bracket and one marker pixel inside the bracket bbox.
    - The marker color is blue or green; separate 4-connected sprites of that same color are the template candidates.
    - A second foreground color, red or yellow, appears in several disconnected sprites that serve only as a counter.

    Rule:
    - Read the marker color C, find the other foreground color D, and count the 4-connected D sprites to get k.
    - Among all C sprites excluding the marker, choose the one whose bbox upper-left corner is lexicographically smallest as template T.
    - Erase all original C and D sprites, then stamp copies of T from the bracket bbox upper-left corner, stepping right by template_width+1.
    - Number those horizontal copies from left to right; every k-th copy becomes magenta, and each magenta copy also repeats downward every template_height+1 rows. Clip out-of-bounds parts and keep the bracket plus marker unchanged.

    Key insight:
    - The marker picks which same-color sprite becomes the stamp, while the disconnected sprites of the other color encode the magenta period that changes the whole stamped layout.
    """
    info = _parse(grid)
    return _render_from_parts(
        info["grid"].shape,
        info["bracket_coords"],
        info["marker"],
        info["c_color"],
        info["template_mask"],
        info["bracket_bbox"][:2],
        info["k"],
    )


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return np.array(out, copy=True)


def _make_bracket_mask(h: int, w: int, orientation: int) -> np.ndarray:
    r1 = h - 1
    c1 = w - 1
    masks = [
        {(0, c) for c in range(w)} | {(r, 0) for r in range(h)},
        {(0, c) for c in range(w)} | {(r, c1) for r in range(h)},
        {(r1, c) for c in range(w)} | {(r, 0) for r in range(h)},
        {(r1, c) for c in range(w)} | {(r, c1) for r in range(h)},
    ]
    out = np.zeros((h, w), dtype=bool)
    for r, c in masks[orientation]:
        out[r, c] = True
    return out


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _free_positions(blocked: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    H, W = blocked.shape
    out = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if not blocked[r : r + h, c : c + w].any():
                out.append((r, c))
    return out


def _place_mask(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    for rr in range(h):
        for cc in range(w):
            if mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = color
    _mark_blocked(blocked, r0, c0, h, w)


def _semantic_input_checks(grid: np.ndarray) -> dict:
    info = _parse(grid)
    H, W = grid.shape
    bracket_set = set(info["bracket_coords"])
    marker = info["marker"]
    sprite_boxes = [item["bbox"] for item in info["c_items"]] + [_bbox(coords) for coords in info["d_parts"]]

    def expanded(bb: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        r0, c0, r1, c1 = bb
        return max(0, r0 - 1), max(0, c0 - 1), min(H - 1, r1 + 1), min(W - 1, c1 + 1)

    for i, bb1 in enumerate(sprite_boxes):
        e1 = expanded(bb1)
        for bb2 in sprite_boxes[i + 1 :]:
            e2 = expanded(bb2)
            disjoint = e1[2] < e2[0] or e2[2] < e1[0] or e1[3] < e2[1] or e2[3] < e1[1]
            assert disjoint, "sprites must remain visually separated"

    br_bb = info["bracket_bbox"]
    ebr = expanded(br_bb)
    for bb in sprite_boxes:
        e2 = expanded(bb)
        disjoint = ebr[2] < e2[0] or e2[2] < ebr[0] or ebr[3] < e2[1] or e2[3] < ebr[1]
        assert disjoint, "sprites must not touch the bracket"

    assert marker not in bracket_set
    return info


def _alt_outputs_differ(grid: np.ndarray, info: dict) -> bool:
    correct = main(grid)
    anchor = info["bracket_bbox"][:2]
    template_mask = info["template_mask"]
    c_color = info["c_color"]
    marker = info["marker"]
    bracket_coords = info["bracket_coords"]
    shape = grid.shape
    k = info["k"]

    if _render_from_parts(shape, bracket_coords, marker, c_color, template_mask, info["c_items"][0]["bbox"][:2], k).tolist() == correct.tolist():
        return False

    candidates = []
    if k > 2:
        candidates.append(k - 1)
    if k < 5:
        candidates.append(k + 1)
    for alt_k in candidates:
        alt = _render_from_parts(shape, bracket_coords, marker, c_color, template_mask, anchor, alt_k)
        if np.array_equal(alt, correct):
            return False

    for item in info["c_items"][1:]:
        alt = _render_from_parts(shape, bracket_coords, marker, c_color, item["mask"], anchor, k)
        if np.array_equal(alt, correct):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    template_mask = _random_transform(random.choice(C_SHAPES))
    th, tw = template_mask.shape
    k = random.randint(2, 5)
    c_count = random.choices([1, 2, 3], weights=[1, 4, 4], k=1)[0]
    min_slots = {2: 4, 3: 6, 4: 5, 5: 6}[k]
    min_v_slots = random.choice([2, 3])

    bh = random.randint(*BRACKET_LEN_RANGE)
    bw = random.randint(*BRACKET_LEN_RANGE)
    min_h = max(GRID_H_RANGE[0], bh + 2, 1 + (min_v_slots - 1) * (th + 1))
    min_w = max(GRID_W_RANGE[0], bw + 2, 1 + (min_slots - 1) * (tw + 1))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    H = random.randint(min_h, GRID_H_RANGE[1])
    W = random.randint(min_w, GRID_W_RANGE[1])
    grid = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    max_br0 = min(H - bh, H - 1 - (min_v_slots - 1) * (th + 1))
    max_bc0 = min(W - bw, W - 1 - (min_slots - 1) * (tw + 1))
    if max_br0 < 0 or max_bc0 < 0:
        return None
    br0 = random.randint(0, max_br0)
    bc0 = random.randint(0, max_bc0)
    bracket_mask = _make_bracket_mask(bh, bw, random.randrange(4))
    bracket_cells = [(br0 + r, bc0 + c) for r, c in zip(*np.where(bracket_mask))]
    for r, c in bracket_cells:
        grid[r, c] = GREY
    _mark_blocked(blocked, br0, bc0, bh, bw)

    marker_candidates = [(r, c) for r in range(bh) for c in range(bw) if not bracket_mask[r, c]]
    if not marker_candidates:
        return None
    mr, mc = random.choice(marker_candidates)
    c_color = random.choice(C_COLORS)
    d_color = random.choice(D_COLORS)
    grid[br0 + mr, bc0 + mc] = c_color

    c_positions = _free_positions(blocked, th, tw)
    if not c_positions:
        return None
    template_pos = random.choice(c_positions)
    _place_mask(grid, blocked, template_mask, template_pos[0], template_pos[1], c_color)
    c_masks = [template_mask]

    for _ in range(c_count - 1):
        other_mask = _random_transform(random.choice(C_SHAPES))
        tries = 0
        while _same_mask(other_mask, template_mask) and tries < 10:
            other_mask = _random_transform(random.choice(C_SHAPES))
            tries += 1
        if _same_mask(other_mask, template_mask):
            return None
        oh, ow = other_mask.shape
        candidates = [
            (r, c)
            for r, c in _free_positions(blocked, oh, ow)
            if (r, c) > template_pos
        ]
        if not candidates:
            return None
        pos = random.choice(candidates)
        _place_mask(grid, blocked, other_mask, pos[0], pos[1], c_color)
        c_masks.append(other_mask)

    for _ in range(k):
        d_mask = _random_transform(random.choice(D_SHAPES))
        dh, dw = d_mask.shape
        candidates = _free_positions(blocked, dh, dw)
        if not candidates:
            return None
        pos = random.choice(candidates)
        _place_mask(grid, blocked, d_mask, pos[0], pos[1], d_color)

    try:
        info = _semantic_input_checks(grid)
    except AssertionError:
        return None

    if not all(not _same_mask(item["mask"], template_mask) for item in info["c_items"][1:]):
        return None

    output = main(grid)
    if np.array_equal(output, grid):
        return None
    if PINK not in output:
        return None
    if c_color not in output:
        return None
    if not _alt_outputs_differ(grid, info):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
