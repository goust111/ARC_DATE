from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
ROTATIONS = {"top": 2, "bottom": 0, "left": 3, "right": 1}
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_frame_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == GREY)
    assert coords.size > 0, "missing grey frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 + 1 >= 10 and c1 - c0 + 1 >= 10

    expected = np.zeros_like(g, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(g == GREY, expected), "grey pixels must form one hollow rectangle"
    return r0, c0, r1, c1


def _anchor_side(
    frame: tuple[int, int, int, int], r: int, c: int
) -> str | None:
    r0, c0, r1, c1 = frame
    sides = []
    if r == r0 + 1 and c0 + 1 <= c <= c1 - 1:
        sides.append("top")
    if r == r1 - 1 and c0 + 1 <= c <= c1 - 1:
        sides.append("bottom")
    if c == c0 + 1 and r0 + 1 <= r <= r1 - 1:
        sides.append("left")
    if c == c1 - 1 and r0 + 1 <= r <= r1 - 1:
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _collect_anchors(g: np.ndarray, frame: tuple[int, int, int, int]) -> dict[int, dict]:
    r0, c0, r1, c1 = frame
    anchors: dict[int, dict] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            val = int(g[r, c])
            if val == BG:
                continue
            side = _anchor_side(frame, r, c)
            assert side is not None, "non-background interior pixels must be anchors on the inner wall"
            assert val not in {GREY, MAROON}
            assert val not in anchors, "anchor colors must be unique"
            anchors[val] = {"side": side, "row": int(r), "col": int(c)}
    assert anchors, "expected at least one anchor"
    return anchors


def _extract_patterns(g: np.ndarray, frame: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0f, c0f, r1f, c1f = frame
    mask = (g != BG) & (g != GREY)
    patterns: dict[int, np.ndarray] = {}
    for comp in _components(mask):
        rs = comp[:, 0]
        cs = comp[:, 1]
        r0, c0, r1, c1 = int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())
        outside = r1 < r0f or r0 > r1f or c1 < c0f or c0 > c1f
        if not outside:
            continue
        assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "external motifs must be 3x3"
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        border_mask = np.ones((3, 3), dtype=bool)
        border_mask[1, 1] = False
        border_vals = patch[border_mask]
        color = int(border_vals[0])
        assert color not in {BG, GREY, MAROON}
        assert np.all(border_vals == color), "motif border must be monochrome"
        assert int(patch[1, 1]) in {BG, MAROON}
        assert color not in patterns, "motif colors must be unique"
        patterns[color] = patch
    assert patterns, "expected at least one external motif"
    return patterns


def _stamp_position(anchor: dict) -> tuple[int, int]:
    side = anchor["side"]
    r = int(anchor["row"])
    c = int(anchor["col"])
    if side == "top":
        return r, c - 1
    if side == "bottom":
        return r - 2, c - 1
    if side == "left":
        return r - 1, c
    if side == "right":
        return r - 1, c - 2
    raise AssertionError(side)


def _apply_motifs(
    g: np.ndarray, anchors: dict[int, dict], patterns: dict[int, np.ndarray]
) -> np.ndarray:
    out = g.copy()
    assert set(anchors) == set(patterns), "anchors and motifs must match by color"
    for color, anchor in anchors.items():
        patch = np.rot90(patterns[color], ROTATIONS[anchor["side"]])
        rr, cc = _stamp_position(anchor)
        assert 0 <= rr and 0 <= cc
        assert rr + 3 <= g.shape[0] and cc + 3 <= g.shape[1]
        mask = patch != BG
        out[rr : rr + 3, cc : cc + 3][mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each external 3x3 color-matched motif into the frame interior at its anchor.

    The grey rectangle defines the unique frame. Colored singleton anchors sit on the
    frame's inner wall, and each anchor color matches exactly one external 3x3 motif.
    The motif is rotated according to which inner side hosts the anchor, then stamped
    so the anchor aligns with the middle pixel of the wall-adjacent motif edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frame = _find_frame_bbox(g)
    anchors = _collect_anchors(g, frame)
    patterns = _extract_patterns(g, frame)
    return _apply_motifs(g, anchors, patterns)


def _build_pattern(color: int, center: int) -> np.ndarray:
    patch = np.full((3, 3), color, dtype=int)
    patch[1, 1] = center
    return patch


def _draw_frame(g: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    g[r0, c0 : c1 + 1] = GREY
    g[r1, c0 : c1 + 1] = GREY
    g[r0 : r1 + 1, c0] = GREY
    g[r0 : r1 + 1, c1] = GREY


def _reserve_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _available_boxes(occupied: np.ndarray, box_h: int, box_w: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    out: list[tuple[int, int]] = []
    for r in range(h - box_h + 1):
        for c in range(w - box_w + 1):
            if not occupied[r : r + box_h, c : c + box_w].any():
                out.append((r, c))
    return out


def _boxes_overlap(a: tuple[int, int], b: tuple[int, int], size: int = 3) -> bool:
    ar, ac = a
    br, bc = b
    return not (ar + size <= br or br + size <= ar or ac + size <= bc or bc + size <= ac)


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(8, 13)
    inner_w = random.randint(8, 13)
    top = random.randint(4, 8)
    bottom = random.randint(4, 8)
    left = random.randint(4, 8)
    right = random.randint(4, 8)

    height = top + inner_h + bottom + 2
    width = left + inner_w + right + 2
    frame = (top, left, top + inner_h + 1, left + inner_w + 1)
    r0, c0, r1, c1 = frame

    g = np.full((height, width), BG, dtype=int)
    _draw_frame(g, frame)

    colors = random.sample(PALETTE, 4)
    anchors = None
    for _ in range(60):
        trial = {
            colors[0]: {"side": "top", "row": r0 + 1, "col": random.randint(c0 + 2, c1 - 2)},
            colors[1]: {"side": "bottom", "row": r1 - 1, "col": random.randint(c0 + 2, c1 - 2)},
            colors[2]: {"side": "left", "row": random.randint(r0 + 2, r1 - 2), "col": c0 + 1},
            colors[3]: {"side": "right", "row": random.randint(r0 + 2, r1 - 2), "col": c1 - 1},
        }
        boxes = [_stamp_position(anchor) for anchor in trial.values()]
        if all(not _boxes_overlap(boxes[i], boxes[j]) for i in range(4) for j in range(i + 1, 4)):
            anchors = trial
            break
    if anchors is None:
        return None
    for color, anchor in anchors.items():
        g[int(anchor["row"]), int(anchor["col"])] = int(color)

    centers = [random.choice((BG, MAROON)) for _ in range(4)]
    patterns = {color: _build_pattern(color, center) for color, center in zip(colors, centers)}

    occupied = np.zeros_like(g, dtype=bool)
    _reserve_box(occupied, r0, c0, r1 - r0 + 1, c1 - c0 + 1, pad=1)
    for anchor in anchors.values():
        rr, cc = _stamp_position(anchor)
        _reserve_box(occupied, rr, cc, 3, 3, pad=0)

    placements: dict[int, tuple[int, int]] = {}
    order = colors[:]
    random.shuffle(order)
    for color in order:
        candidates = _available_boxes(occupied, 3, 3)
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements[color] = pos
        _reserve_box(occupied, pos[0], pos[1], 3, 3, pad=1)

    for color, (rr, cc) in placements.items():
        patch = patterns[color]
        mask = patch != BG
        g[rr : rr + 3, cc : cc + 3][mask] = patch[mask]

    expected = _apply_motifs(g, anchors, patterns)
    if np.array_equal(main(g), expected):
        return g
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
