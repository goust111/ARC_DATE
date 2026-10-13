from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BAR = 1
THUMB = 2

ORIENTS = ("H", "V")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LENGTHS = (3, 5)
SIZES = (3, 4)
SCALES = (5, 6)
MAX_GEN_ATTEMPTS = 800


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bar_cells(center: tuple[int, int], length: int, orient: str) -> list[tuple[int, int]]:
    r, c = map(int, center)
    half = int(length) // 2
    if orient == "H":
        return [(r, c + dc) for dc in range(-half, half + 1)]
    return [(r + dr, c) for dr in range(-half, half + 1)]


def _other_orient(orient: str) -> str:
    return "V" if orient == "H" else "H"


def _thumb_mask(grid: np.ndarray) -> np.ndarray:
    red = np.argwhere(grid == THUMB)
    assert red.size > 0, "expected red thumbnail pixels"
    r0, c0, r1, c1 = _bbox(red)
    assert (r0, c0) == (0, 0), "thumbnail must start at top-left"
    size = r1 - r0 + 1
    assert size == c1 - c0 + 1 and size in SIZES, "thumbnail must be 3x3 or 4x4"
    thumb = grid[:size, :size]
    assert not np.any(thumb == BAR), "bars must not overlap the thumbnail"
    return thumb == THUMB


def _bars(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == BAR)
    assert comps, "expected blue bars"
    bars: list[dict] = []
    for idx, coords in enumerate(comps):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = int(coords.shape[0])
        assert height == 1 or width == 1, "every blue component must be a straight bar"
        length = max(height, width)
        assert area == length and length in LENGTHS and length % 2 == 1
        orient = "H" if height == 1 else "V"
        bars.append(
            {
                "id": idx,
                "coords": coords,
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "length": length,
                "orient": orient,
            }
        )
    return sorted(bars, key=lambda b: (b["center"][0], b["center"][1], b["length"]))


def _default_orient(bars: list[dict]) -> str:
    h_count = sum(bar["orient"] == "H" for bar in bars)
    v_count = len(bars) - h_count
    assert h_count != v_count, "default orientation must be the strict majority"
    assert h_count > 0 and v_count > 0, "need both default bars and rotated anchors"
    return "H" if h_count > v_count else "V"


def _mapping_candidates(
    bars: list[dict],
    thumb: np.ndarray,
    default: str,
    shape: tuple[int, int],
) -> list[dict]:
    h, w = map(int, shape)
    size = int(thumb.shape[0])
    anchors = {bar["id"] for bar in bars if bar["orient"] != default}
    cands: list[dict] = []
    for k in range(1, max(h, w) + 1):
        for dr in range(h):
            for dc in range(w):
                mapping: dict[int, tuple[int, int]] = {}
                used: set[tuple[int, int]] = set()
                okay = True
                for bar in bars:
                    r, c = map(int, bar["center"])
                    rr_num = r - dr
                    cc_num = c - dc
                    if rr_num % k or cc_num % k:
                        continue
                    rr = rr_num // k
                    cc = cc_num // k
                    if not (0 <= rr < size and 0 <= cc < size):
                        continue
                    pos = (int(rr), int(cc))
                    if pos in used:
                        okay = False
                        break
                    used.add(pos)
                    mapping[int(bar["id"])] = pos
                if not okay:
                    continue
                covered_red = 0
                anchor_red = 0
                for bar in bars:
                    pos = mapping.get(int(bar["id"]))
                    if pos is None:
                        continue
                    if thumb[pos]:
                        covered_red += 1
                        if int(bar["id"]) in anchors:
                            anchor_red += 1
                cands.append(
                    {
                        "k": int(k),
                        "dr": int(dr),
                        "dc": int(dc),
                        "mapping": mapping,
                        "mapped": len(mapping),
                        "covered_red": int(covered_red),
                        "anchor_red": int(anchor_red),
                    }
                )
    assert cands, "expected at least one candidate mapping"
    return cands


def _best_mapping(
    bars: list[dict],
    thumb: np.ndarray,
    default: str,
    shape: tuple[int, int],
) -> dict:
    cands = _mapping_candidates(bars, thumb, default, shape)
    best_score = max((c["anchor_red"], c["mapped"], c["covered_red"]) for c in cands)
    best = [
        c
        for c in cands
        if (c["anchor_red"], c["mapped"], c["covered_red"]) == best_score
    ]
    assert len(best) == 1, "mapping must be uniquely determined by anchors and lattice fit"
    chosen = best[0]
    assert chosen["mapped"] == len(bars), "all bars must map into the thumbnail lattice"
    red_total = int(thumb.sum())
    assert chosen["covered_red"] == red_total, "every red pixel must have exactly one bar"
    return chosen


def _render(
    base: np.ndarray,
    bars: list[dict],
    thumb: np.ndarray,
    default: str,
    mapping: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    out[out == BAR] = BG
    occupied: set[tuple[int, int]] = set()
    for bar in bars:
        pos = mapping[int(bar["id"])]
        target = _other_orient(default) if thumb[pos] else default
        cells = _bar_cells(bar["center"], int(bar["length"]), target)
        for r, c in cells:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert (r, c) not in occupied, "rotated bars must not overlap"
            occupied.add((r, c))
            out[r, c] = BAR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the top-left red thumbnail to decide which blue bars must end up rotated 90 degrees.

    Input structure:
    - Black background, a 3x3 or 4x4 red-on-black thumbnail anchored at the top-left corner, and several blue 1-cell-thick odd bars in the main area.
    - Every blue component is a horizontal or vertical bar with odd length, and each bar center lies on one lattice point of an unknown thumbnail-to-canvas affine map.
    - A strict majority of bars share one default orientation; the minority bars are already in the rotated target orientation and therefore act as visible anchors.

    Rule:
    - Read the red thumbnail as a boolean instruction map: red cells want the corresponding bar to be in the rotated orientation, black thumbnail cells want the default orientation.
    - Infer the unknown scale and offset by finding the unique lattice map whose mapped bar centers fit the thumbnail grid and whose already-rotated minority bars land on red thumbnail cells.
    - Redraw every blue bar so that bars mapped to red cells are vertical if the default is horizontal (or horizontal if the default is vertical), while bars mapped to black cells keep the default orientation; centers, lengths, colors, and the thumbnail itself stay unchanged.

    Key insight:
    - The thumbnail does not encode bar colors or placements directly; it encodes final orientation targets, and the few bars already showing the minority orientation are the only reliable anchors for recovering the hidden coordinate map.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    thumb = _thumb_mask(g)
    bars = _bars(g)
    default = _default_orient(bars)
    chosen = _best_mapping(bars, thumb, default, g.shape)
    return _render(g, bars, thumb, default, chosen["mapping"])


def _sample_thumbnail(size: int) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    cells = [(r, c) for r in range(size) for c in range(size)]
    corners = {(0, 0), (size - 1, size - 1)}
    red_min = max(4, size + 1)
    red_max = min(size * size - 2, red_min + 1)
    for _ in range(200):
        red_count = random.randint(red_min, red_max)
        red = set(corners)
        remaining = [cell for cell in cells if cell not in red]
        red.update(random.sample(remaining, red_count - len(red)))
        black = set(cells) - red
        if len(black) >= 2:
            thumb = np.zeros((size, size), dtype=int)
            for r, c in red:
                thumb[r, c] = THUMB
            return thumb, red, black
    raise RuntimeError("failed to sample thumbnail")


def _place_instance(spec: dict) -> np.ndarray:
    size = int(spec["size"])
    thumb = np.asarray(spec["thumb"], dtype=int)
    h = int(spec["height"])
    w = int(spec["width"])
    grid = np.zeros((h, w), dtype=int)
    grid[:size, :size] = thumb
    for pos in spec["occupied"]:
        rr, cc = map(int, pos)
        center = (
            int(spec["dr"]) + int(spec["k"]) * rr,
            int(spec["dc"]) + int(spec["k"]) * cc,
        )
        orient = spec["input_orients"][pos]
        for r, c in _bar_cells(center, int(spec["lengths"][pos]), orient):
            assert 0 <= r < h and 0 <= c < w
            assert grid[r, c] == BG
            grid[r, c] = BAR
    return grid


def _expected_output(spec: dict) -> np.ndarray:
    size = int(spec["size"])
    thumb_mask = np.asarray(spec["thumb"], dtype=int) == THUMB
    h = int(spec["height"])
    w = int(spec["width"])
    grid = np.zeros((h, w), dtype=int)
    grid[:size, :size] = spec["thumb"]
    target = _other_orient(spec["default"])
    for pos in spec["occupied"]:
        rr, cc = map(int, pos)
        center = (
            int(spec["dr"]) + int(spec["k"]) * rr,
            int(spec["dc"]) + int(spec["k"]) * cc,
        )
        orient = target if thumb_mask[rr, cc] else spec["default"]
        for r, c in _bar_cells(center, int(spec["lengths"][pos]), orient):
            assert 0 <= r < h and 0 <= c < w
            assert grid[r, c] == BG
            grid[r, c] = BAR
    return grid


def _validate_semantics(inp: np.ndarray, spec: dict) -> bool:
    try:
        bars = _bars(inp)
        thumb = _thumb_mask(inp)
        default = _default_orient(bars)
        if default != spec["default"]:
            return False
        cands = _mapping_candidates(bars, thumb, default, inp.shape)
        actual = [
            c
            for c in cands
            if c["k"] == spec["k"] and c["dr"] == spec["dr"] and c["dc"] == spec["dc"]
        ]
        if len(actual) != 1:
            return False
        actual = actual[0]
        if actual["mapped"] != len(bars):
            return False
        if actual["covered_red"] != len(spec["red"]):
            return False
        if actual["anchor_red"] != len(spec["anchors"]):
            return False

        best_score = max((c["anchor_red"], c["mapped"], c["covered_red"]) for c in cands)
        best = [
            c
            for c in cands
            if (c["anchor_red"], c["mapped"], c["covered_red"]) == best_score
        ]
        if len(best) != 1:
            return False
        if best[0]["k"] != spec["k"] or best[0]["dr"] != spec["dr"] or best[0]["dc"] != spec["dc"]:
            return False

        if len(spec["black_occupied"]) < 1:
            return False
        non_anchor_red = spec["red"] - spec["anchors"]
        if len(non_anchor_red) < 1:
            return False

        out = main(inp)
        if not np.array_equal(out, _expected_output(spec)):
            return False
        if np.array_equal(out, inp):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    size = 4 if random.random() < 0.7 else 3
    thumb, red, black = _sample_thumbnail(size)

    black_extra = random.randint(1, min(2, len(black)))
    black_occupied = set(random.sample(list(black), black_extra))
    occupied = set(red) | black_occupied

    anchor_max = min(4, len(red) - 1, (len(occupied) - 1) // 2)
    if anchor_max < 2:
        return None
    anchor_count = random.randint(2, anchor_max)
    anchors = set(random.sample(list(red), anchor_count))

    default = random.choice(ORIENTS)
    k = random.choice(SCALES)
    margin_min = size + 3
    margin_max = min(8, 30 - 1 - k * (size - 1) - 2)
    if margin_max < margin_min:
        return None
    dr = random.randint(margin_min, margin_max)
    dc = random.randint(margin_min, margin_max)

    max_r = dr + k * (size - 1) + 2
    max_c = dc + k * (size - 1) + 2
    height = random.randint(max(max_r + 1, 17), min(30, max_r + 4))
    width = random.randint(max(max_c + 1, 17), min(30, max_c + 4))

    lengths = {pos: random.choice(LENGTHS) for pos in occupied}
    input_orients = {}
    for pos in occupied:
        input_orients[pos] = _other_orient(default) if pos in anchors else default

    spec = {
        "size": size,
        "thumb": thumb,
        "red": set(red),
        "black_occupied": set(black_occupied),
        "occupied": set(occupied),
        "anchors": set(anchors),
        "default": default,
        "k": k,
        "dr": dr,
        "dc": dc,
        "height": height,
        "width": width,
        "lengths": lengths,
        "input_orients": input_orients,
    }
    inp = _place_instance(spec)
    if not _validate_semantics(inp, spec):
        return None
    return inp, spec


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        made = _gen_attempt()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid thumbnail-bar rotation puzzle")
