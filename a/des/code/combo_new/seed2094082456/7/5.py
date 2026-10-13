from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("tl", "tr", "bl", "br")
MATCHING_SIDES = {
    "tl": {"top", "left"},
    "tr": {"top", "right"},
    "bl": {"bottom", "left"},
    "br": {"bottom", "right"},
}
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _missing_bounds(
    corner: str,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    if corner == "tl":
        if not (r0 <= mr < r1 and c0 <= mc < c1):
            return None
        return r0, mr, c0, mc
    if corner == "tr":
        if not (r0 <= mr < r1 and c0 < mc <= c1):
            return None
        return r0, mr, mc, c1
    if corner == "bl":
        if not (r0 < mr <= r1 and c0 <= mc < c1):
            return None
        return mr, r1, c0, mc
    if not (r0 < mr <= r1 and c0 < mc <= c1):
        return None
    return mr, r1, mc, c1


def _midpoints(start: int, end: int) -> list[int]:
    if start > end:
        return []
    length = end - start + 1
    if length % 2 == 1:
        return [(start + end) // 2]
    return [start + length // 2 - 1, start + length // 2]


def _side_anchors(
    corner: str,
    direction: str,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    if corner == "tl":
        if direction == "top":
            start, end = mc + 1, c1
            return [(r0, cc) for cc in _midpoints(start, end)]
        if direction == "bottom":
            return [(r1, cc) for cc in _midpoints(c0, c1)]
        if direction == "left":
            start, end = mr + 1, r1
            return [(rr, c0) for rr in _midpoints(start, end)]
        return [(rr, c1) for rr in _midpoints(r0, r1)]
    if corner == "tr":
        if direction == "top":
            start, end = c0, mc - 1
            return [(r0, cc) for cc in _midpoints(start, end)]
        if direction == "bottom":
            return [(r1, cc) for cc in _midpoints(c0, c1)]
        if direction == "left":
            return [(rr, c0) for rr in _midpoints(r0, r1)]
        start, end = mr + 1, r1
        return [(rr, c1) for rr in _midpoints(start, end)]
    if corner == "bl":
        if direction == "top":
            return [(r0, cc) for cc in _midpoints(c0, c1)]
        if direction == "bottom":
            start, end = mc + 1, c1
            return [(r1, cc) for cc in _midpoints(start, end)]
        if direction == "left":
            start, end = r0, mr - 1
            return [(rr, c0) for rr in _midpoints(start, end)]
        return [(rr, c1) for rr in _midpoints(r0, r1)]
    if direction == "top":
        return [(r0, cc) for cc in _midpoints(c0, c1)]
    if direction == "bottom":
        start, end = c0, mc - 1
        return [(r1, cc) for cc in _midpoints(start, end)]
    if direction == "left":
        return [(rr, c0) for rr in _midpoints(r0, r1)]
    start, end = r0, mr - 1
    return [(rr, c1) for rr in _midpoints(start, end)]


def _side_anchor(
    corner: str,
    direction: str,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> tuple[int, int] | None:
    anchors = _side_anchors(corner, direction, bbox, marker)
    return anchors[0] if anchors else None


def _step_out(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = cell
    if direction == "top":
        return r - 1, c
    if direction == "bottom":
        return r + 1, c
    if direction == "left":
        return r, c - 1
    return r, c + 1


def _matches_shape(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    missing: tuple[int, int, int, int],
    marker: tuple[int, int],
    body: int,
    marker_color: int,
) -> bool:
    r0, c0, r1, c1 = bbox
    mr0, mr1, mc0, mc1 = missing
    mset = set(range(mr0, mr1 + 1))
    nset = set(range(mc0, mc1 + 1))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            value = int(g[r, c])
            if (r, c) == marker:
                if value != marker_color:
                    return False
            elif r in mset and c in nset:
                if value != 0:
                    return False
            elif value != body:
                return False
    return True


def _parse_object(g: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    colors = Counter(int(g[r, c]) for r, c in coords)
    assert 0 not in colors
    assert len(colors) == 2
    marker_colors = [color for color, count in colors.items() if count == 1]
    assert len(marker_colors) == 1
    marker_color = int(marker_colors[0])
    body_color = int(next(color for color in colors if color != marker_color))
    marker = next((r, c) for r, c in coords if int(g[r, c]) == marker_color)

    matches: list[dict] = []
    for ext in coords:
        if int(g[ext]) != body_color:
            continue
        remaining = [coord for coord in coords if coord != ext]
        bbox = _bbox(remaining)
        r0, c0, r1, c1 = bbox
        er, ec = ext
        if r0 <= er <= r1 and c0 <= ec <= c1:
            continue

        direction = None
        if er == r0 - 1 and c0 <= ec <= c1:
            direction = "top"
        elif er == r1 + 1 and c0 <= ec <= c1:
            direction = "bottom"
        elif ec == c0 - 1 and r0 <= er <= r1:
            direction = "left"
        elif ec == c1 + 1 and r0 <= er <= r1:
            direction = "right"
        if direction is None:
            continue

        for corner in CORNERS:
            missing = _missing_bounds(corner, bbox, marker)
            if missing is None:
                continue
            anchors = _side_anchors(corner, direction, bbox, marker)
            if not anchors or ext not in {_step_out(anchor, direction) for anchor in anchors}:
                continue
            if not _matches_shape(g, bbox, missing, marker, body_color, marker_color):
                continue
            matches.append(
                {
                    "bbox": bbox,
                    "corner": corner,
                    "missing": missing,
                    "marker": marker,
                    "marker_color": marker_color,
                    "body_color": body_color,
                    "extension": ext,
                    "direction": direction,
                }
            )

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each L-shaped cutout into its parent rectangle when the short same-color pointer aims toward the missing corner.

    Input structure:
    - The background is black.
    - Each foreground object is a 4-connected multicolor component made of one main color A, one single-pixel inner marker color B, and one 1-cell A extension sticking out from one outer side.
    - Removing the extension leaves an A-colored rectangle with one rectangular corner removed; the B marker sits at the concave inner corner of that missing block.

    Rule:
    - For each component, remove the 1-cell extension and identify the outer rectangle, the missing corner block, and the marker color.
    - The extension side gives a direction: top, bottom, left, or right.
    - If that direction is one of the two sides touched by the missing corner, fill the rest of the missing block with color B and turn the marker cell into color A.
    - Otherwise keep the L shape unfilled; in that branch the marker disappears back to background.

    Key insight:
    - The object already reveals the hidden rectangle geometry, but the extra A pixel decides whether the corner should actually be completed; the marker color then supplies the fill for that completion.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp in _components(g != 0):
        obj = _parse_object(g, comp)
        er, ec = obj["extension"]
        mr, mc = obj["marker"]
        out[er, ec] = 0
        out[mr, mc] = 0
        if obj["direction"] not in MATCHING_SIDES[obj["corner"]]:
            continue

        r0, r1, c0, c1 = obj["missing"]
        out[r0 : r1 + 1, c0 : c1 + 1] = obj["marker_color"]
        out[mr, mc] = obj["body_color"]
    return out


def _render_object(inp: np.ndarray, out: np.ndarray, spec: dict) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    h = int(spec["h"])
    w = int(spec["w"])
    corner = str(spec["corner"])
    direction = str(spec["direction"])
    body = int(spec["body"])
    marker_color = int(spec["marker_color"])
    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    marker = spec["marker"]
    missing = _missing_bounds(corner, bbox, marker)
    assert missing is not None

    mr0, mr1, mc0, mc1 = missing
    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
            in_missing = mr0 <= rr <= mr1 and mc0 <= cc <= mc1
            if (rr, cc) == marker:
                inp[rr, cc] = marker_color
            elif in_missing:
                continue
            else:
                inp[rr, cc] = body

    anchor = _side_anchor(corner, direction, bbox, marker)
    assert anchor is not None
    er, ec = _step_out(anchor, direction)
    inp[er, ec] = body

    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
            in_missing = mr0 <= rr <= mr1 and mc0 <= cc <= mc1
            if not in_missing:
                out[rr, cc] = body

    if spec["expand"]:
        out[mr0 : mr1 + 1, mc0 : mc1 + 1] = marker_color
        out[marker] = body


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _actual_bbox(r0: int, c0: int, h: int, w: int, direction: str) -> tuple[int, int, int, int]:
    top = r0 - 1 if direction == "top" else r0
    bottom = r0 + h if direction == "bottom" else r0 + h - 1
    left = c0 - 1 if direction == "left" else c0
    right = c0 + w if direction == "right" else c0 + w - 1
    return top, left, bottom, right


def _marker_from_spec(r0: int, c0: int, h: int, w: int, corner: str, mh: int, mw: int) -> tuple[int, int]:
    if corner == "tl":
        return r0 + mh - 1, c0 + mw - 1
    if corner == "tr":
        return r0 + mh - 1, c0 + w - mw
    if corner == "bl":
        return r0 + h - mh, c0 + mw - 1
    return r0 + h - mh, c0 + w - mw


def _side_length(corner: str, direction: str, h: int, w: int, mh: int, mw: int) -> int:
    if corner == "tl":
        if direction in ("top", "bottom"):
            return w - mw if direction == "top" else w
        return h - mh if direction == "left" else h
    if corner == "tr":
        if direction in ("top", "bottom"):
            return w - mw if direction == "top" else w
        return h if direction == "left" else h - mh
    if corner == "bl":
        if direction in ("top", "bottom"):
            return w if direction == "top" else w - mw
        return h - mh if direction == "left" else h
    if direction in ("top", "bottom"):
        return w if direction == "top" else w - mw
    return h if direction == "left" else h - mh


def _sample_spec(
    height: int,
    width: int,
    body: int,
    marker_color: int,
    expand: bool,
) -> dict | None:
    corner = random.choice(CORNERS)
    h = random.choice((5, 7, 9))
    w = random.choice((5, 7, 9))
    if h > height - 1 or w > width - 1:
        return None

    mh_choices = [v for v in range(2, h - 1, 2) if v <= h - 3]
    mw_choices = [v for v in range(2, w - 1, 2) if v <= w - 3]
    if not mh_choices or not mw_choices:
        return None
    mh = random.choice(mh_choices)
    mw = random.choice(mw_choices)

    options = sorted(MATCHING_SIDES[corner] if expand else ({"top", "bottom", "left", "right"} - MATCHING_SIDES[corner]))
    random.shuffle(options)
    direction = next((side for side in options if _side_length(corner, side, h, w, mh, mw) >= 3), None)
    if direction is None:
        return None

    r_low = 1 if direction == "top" else 0
    r_high = height - h - (1 if direction == "bottom" else 0)
    c_low = 1 if direction == "left" else 0
    c_high = width - w - (1 if direction == "right" else 0)
    if r_low > r_high or c_low > c_high:
        return None

    return {
        "h": h,
        "w": w,
        "mh": mh,
        "mw": mw,
        "corner": corner,
        "direction": direction,
        "body": body,
        "marker_color": marker_color,
        "expand": expand,
        "r_range": (r_low, r_high),
        "c_range": (c_low, c_high),
    }


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (inp != 0) | (out != 0)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(2, int(r0)))
    left = random.randint(0, min(2, int(c0)))
    bottom = random.randint(0, min(2, int(inp.shape[0] - 1 - r1)))
    right = random.randint(0, min(2, int(inp.shape[1] - 1 - c1)))
    sr0 = int(r0 - top)
    sc0 = int(c0 - left)
    sr1 = int(r1 + bottom)
    sc1 = int(c1 + right)
    return inp[sr0 : sr1 + 1, sc0 : sc1 + 1].copy(), out[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    n_objects = random.randint(2, 4)

    body_colors = random.sample(PALETTE, n_objects)
    marker_colors = [random.choice([c for c in PALETTE if c != body]) for body in body_colors]
    expand_flags = [i % 2 == 0 for i in range(n_objects)]
    if n_objects % 2 == 1:
        expand_flags[-1] = bool(random.getrandbits(1))
    random.shuffle(expand_flags)

    specs: list[dict] = []
    padded_boxes: list[tuple[int, int, int, int]] = []
    order = list(range(n_objects))
    random.shuffle(order)

    for idx in order:
        spec = _sample_spec(height, width, body_colors[idx], marker_colors[idx], expand_flags[idx])
        if spec is None:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(*spec["r_range"])
            c0 = random.randint(*spec["c_range"])
            actual = _actual_bbox(r0, c0, spec["h"], spec["w"], spec["direction"])
            padded = (actual[0] - 1, actual[1] - 1, actual[2] + 1, actual[3] + 1)
            if any(_boxes_overlap(padded, other) for other in padded_boxes):
                continue
            spec = dict(spec)
            spec["r0"] = r0
            spec["c0"] = c0
            spec["marker"] = _marker_from_spec(r0, c0, spec["h"], spec["w"], spec["corner"], spec["mh"], spec["mw"])
            specs.append(spec)
            padded_boxes.append(padded)
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros((height, width), dtype=int)
    out = np.zeros_like(inp)
    for spec in specs:
        _render_object(inp, out, spec)

    inp, out = _crop_pair(inp, out)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, expected = attempt
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for ARC task 5")
