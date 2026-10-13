from __future__ import annotations

from collections import deque
import hashlib
import itertools
import random

import numpy as np


BG = 0
ANCHOR = 8
OBJECT_COLORS = (1, 2, 3, 4)
QUADRANTS = ("tl", "tr", "bl", "br")
SIDE_OPTIONS = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
SEEN_OUTPUT_SIGS: set[str] = set()
GEN_RETRIES = 1200


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(comp, dtype=int))
    return comps


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0, r1, c1 = bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = coords.copy()
    out[:, 0] += int(dr)
    out[:, 1] += int(dc)
    return out


def _touches_anchor(coords: np.ndarray, anchor_mask: np.ndarray) -> bool:
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < anchor_mask.shape[0] and 0 <= nc < anchor_mask.shape[1] and anchor_mask[nr, nc]:
                return True
    return False


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    return tuple(tuple(int(v) for v in row) for row in cropped.astype(int))


def _is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask.astype(bool))
    return len(comps) == 1


def _is_asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 1)):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    if np.array_equal(mask, np.rot90(mask, 3)):
        return False
    return True


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    seen = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            arr = np.rot90(base, k)
            coords = np.argwhere(arr)
            r0, c0, r1, c1 = bbox_coords(coords)
            cropped = arr[r0 : r1 + 1, c0 : c1 + 1].astype(int)
            sig = _shape_signature(cropped)
            if sig not in seen:
                seen.add(sig)
                out.append(cropped)
    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    area = rng.randint(3, 5)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                frontier.add((nr, nc))
        frontier -= cells
        cells.add(rng.choice(sorted(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    if h > 4 or w > 4 or h == 1 or w == 1:
        raise ValueError("reject")
    mask = np.zeros((h, w), dtype=int)
    for r, c in norm:
        mask[r, c] = 1
    if not _is_connected(mask) or not _is_asymmetric(mask):
        raise ValueError("reject")
    return mask


def _oriented_shape(rng: random.Random, quadrant: str) -> np.ndarray:
    corner_selectors = {
        "tl": lambda arr: arr[-1, -1] == 1,
        "tr": lambda arr: arr[-1, 0] == 1,
        "bl": lambda arr: arr[0, -1] == 1,
        "br": lambda arr: arr[0, 0] == 1,
    }
    for _ in range(400):
        try:
            base = _random_connected_mask(rng)
        except ValueError:
            continue
        choices = [arr for arr in _all_transforms(base) if corner_selectors[quadrant](arr)]
        choices = [arr for arr in choices if _is_asymmetric(arr)]
        if choices:
            return rng.choice(choices).copy()
    raise RuntimeError(f"failed to orient shape for {quadrant}")


def _final_top_left(anchor_bbox: tuple[int, int, int, int], shape: np.ndarray, quadrant: str, side: str) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    h, w = shape.shape
    if quadrant == "tl" and side == "top":
        return ar0 - h, ac0 - w + 1
    if quadrant == "tl" and side == "left":
        return ar0 - h + 1, ac0 - w
    if quadrant == "tr" and side == "top":
        return ar0 - h, ac1
    if quadrant == "tr" and side == "right":
        return ar0 - h + 1, ac1 + 1
    if quadrant == "bl" and side == "bottom":
        return ar1 + 1, ac0 - w + 1
    if quadrant == "bl" and side == "left":
        return ar1, ac0 - w
    if quadrant == "br" and side == "bottom":
        return ar1 + 1, ac1
    if quadrant == "br" and side == "right":
        return ar1, ac1 + 1
    raise AssertionError("invalid placement")


def _side_extents(shape: np.ndarray, quadrant: str, side: str) -> tuple[int, int, int, int]:
    h, w = map(int, shape.shape)
    top = bottom = left = right = 0
    if quadrant == "tl":
        if side == "top":
            top, left = h, w - 1
        else:
            top, left = h - 1, w
    elif quadrant == "tr":
        if side == "top":
            top, right = h, w - 1
        else:
            top, right = h - 1, w
    elif quadrant == "bl":
        if side == "bottom":
            bottom, left = h, w - 1
        else:
            bottom, left = h - 1, w
    else:
        if side == "bottom":
            bottom, right = h, w - 1
        else:
            bottom, right = h - 1, w
    return top, bottom, left, right


def _slide_from_input_to_anchor(coords: np.ndarray, anchor_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    rows = coords[:, 0]
    cols = coords[:, 1]
    row_hits = coords[np.isin(rows, [ar0, ar1])]
    col_hits = coords[np.isin(cols, [ac0, ac1])]
    candidates: list[tuple[int, int]] = []

    if row_hits.size and cols.max() < ac0:
        shift = ac0 - int(row_hits[:, 1].max()) - 1
        candidates.append((0, shift))
    if row_hits.size and cols.min() > ac1:
        shift = ac1 + 1 - int(row_hits[:, 1].min())
        candidates.append((0, shift))
    if col_hits.size and rows.max() < ar0:
        shift = ar0 - int(col_hits[:, 0].max()) - 1
        candidates.append((shift, 0))
    if col_hits.size and rows.min() > ar1:
        shift = ar1 + 1 - int(col_hits[:, 0].min())
        candidates.append((shift, 0))

    assert len(candidates) == 1, "each object must have exactly one legal axis-aligned shortest slide"
    return candidates[0]


def _validate_slide_path(
    coords: np.ndarray,
    delta: tuple[int, int],
    color: int,
    base_grid: np.ndarray,
    anchor_mask: np.ndarray,
) -> None:
    dr, dc = delta
    steps = max(abs(dr), abs(dc))
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    work = base_grid.copy()
    work[coords[:, 0], coords[:, 1]] = BG

    for k in range(1, steps + 1):
        moved = _translate(coords, step_r * k, step_c * k)
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < work.shape[0]))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < work.shape[1]))
        assert not np.any(anchor_mask[moved[:, 0], moved[:, 1]])
        assert not np.any(work[moved[:, 0], moved[:, 1]] != BG), "slide path must stay unobstructed"
        touching = _touches_anchor(moved, anchor_mask)
        if k < steps:
            assert not touching, "slide must stop at first anchor contact"
        else:
            assert touching, "final slide position must contact the anchor"
            work[moved[:, 0], moved[:, 1]] = int(color)


def _parse_objects(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, np.ndarray]]]:
    teal_components = components_mask(grid == ANCHOR)
    anchor_candidates = []
    for comp in teal_components:
        if comp.shape[0] != 4:
            continue
        r0, c0, r1, c1 = bbox_coords(comp)
        if r1 - r0 == 1 and c1 - c0 == 1:
            anchor_candidates.append((r0, c0, r1, c1))
    assert len(anchor_candidates) == 1, "expected one 2x2 anchor square"
    anchor_bbox = anchor_candidates[0]

    colors = sorted(int(c) for c in np.unique(grid) if c not in {BG, ANCHOR})
    assert colors == list(OBJECT_COLORS), "expected four colored objects 1,2,3,4"
    objects: list[tuple[int, np.ndarray]] = []
    for color in colors:
        comps = components_mask(grid == color)
        assert len(comps) == 1, f"color {color} must form one object"
        coords = comps[0]
        assert 3 <= coords.shape[0] <= 5, "objects must have 3-5 pixels"
        mask = np.zeros((bbox_coords(coords)[2] - bbox_coords(coords)[0] + 1, bbox_coords(coords)[3] - bbox_coords(coords)[1] + 1), dtype=int)
        r0, c0, _, _ = bbox_coords(coords)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = 1
        assert _is_asymmetric(mask), "generator expects asymmetric objects"
        objects.append((color, coords))
    return anchor_bbox, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the four quadrant objects into first contact with the central 2x2 anchor, then pinwheel the assembled square tile through the four quarter-turns.

    Input structure:
    - The background is black.
    - A single teal 2x2 square anchor sits near the middle of a square grid.
    - Four non-black, non-teal monochrome objects of colors 1, 2, 3, and 4 occupy the four quadrants.
    - Each object is 4-connected, has 3-5 pixels, is asymmetric, and overlaps the anchor's rows or columns so a straight slide can reach the anchor.

    Rule:
    - For each colored object, find the unique axis-aligned translation that moves it to its first 4-neighbor contact with the teal anchor.
    - Apply all four shortest slides, erase the original object positions, and keep the anchor unchanged.
    - Crop the assembled anchor-plus-objects figure to its minimal bounding box; this tile must be square.
    - Output a 2x2 mosaic with the tile unchanged in the upper-left, rotated 90 degrees clockwise in the upper-right, rotated 90 degrees counterclockwise in the lower-left, and rotated 180 degrees in the lower-right.

    Key insight:
    - The visible result is a rotation mosaic, but the tile being rotated is not present directly in the input; it must first be reconstructed by shortest-contact sliding into the anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    anchor_bbox, objects = _parse_objects(g)
    anchor_mask = g == ANCHOR

    assembled = g.copy()
    assembled[assembled != ANCHOR] = BG
    occupied = np.zeros(g.shape, dtype=bool)

    for color, coords in objects:
        delta = _slide_from_input_to_anchor(coords, anchor_bbox)
        _validate_slide_path(coords, delta, color, g, anchor_mask)
        moved = _translate(coords, *delta)
        assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "moved objects must not overlap"
        occupied[moved[:, 0], moved[:, 1]] = True
        assembled[moved[:, 0], moved[:, 1]] = int(color)

    tile = _crop_nonzero(assembled)
    assert tile.shape[0] == tile.shape[1], "assembled tile must be square for quarter-turn pinwheel"
    return np.block(
        [
            [tile, np.rot90(tile, -1)],
            [np.rot90(tile, 1), np.rot90(tile, 2)],
        ]
    )


def _choose_layout(rng: random.Random):
    shapes = {quad: _oriented_shape(rng, quad) for quad in QUADRANTS}
    combos = list(itertools.product((0, 1), repeat=4))
    rng.shuffle(combos)
    for bits in combos:
        sides = {quad: SIDE_OPTIONS[quad][bit] for quad, bit in zip(QUADRANTS, bits)}
        extents = {quad: _side_extents(shapes[quad], quad, sides[quad]) for quad in QUADRANTS}
        top = max(extents["tl"][0], extents["tr"][0])
        bottom = max(extents["bl"][1], extents["br"][1])
        left = max(extents["tl"][2], extents["bl"][2])
        right = max(extents["tr"][3], extents["br"][3])
        h = top + 2 + bottom
        w = left + 2 + right
        if h == w and 6 <= h <= 15:
            return shapes, sides, h
    return None


def _place_final_objects(n: int, shapes: dict[str, np.ndarray], sides: dict[str, str]) -> tuple[tuple[int, int, int, int], dict[int, dict]]:
    anchor_top = n // 2 - 1
    anchor_bbox = (anchor_top, anchor_top, anchor_top + 1, anchor_top + 1)
    objects: dict[int, dict] = {}
    color_by_quad = {"tl": 1, "tr": 2, "bl": 3, "br": 4}

    occupied = set((r, c) for r in range(anchor_bbox[0], anchor_bbox[2] + 1) for c in range(anchor_bbox[1], anchor_bbox[3] + 1))
    for quad in QUADRANTS:
        shape = shapes[quad]
        side = sides[quad]
        r0, c0 = _final_top_left(anchor_bbox, shape, quad, side)
        coords = np.argwhere(shape == 1)
        coords[:, 0] += r0
        coords[:, 1] += c0
        for cell in map(tuple, coords):
            assert cell not in occupied
            occupied.add(cell)
        objects[color_by_quad[quad]] = {
            "quadrant": quad,
            "side": side,
            "final": coords,
            "shape": shape,
        }
    return anchor_bbox, objects


def _side_gap_limit(coords: np.ndarray, side: str, n: int) -> int:
    if side == "top":
        return int(coords[:, 0].min())
    if side == "bottom":
        return int(n - 1 - coords[:, 0].max())
    if side == "left":
        return int(coords[:, 1].min())
    if side == "right":
        return int(n - 1 - coords[:, 1].max())
    raise AssertionError("invalid side")


def _shift_outward(coords: np.ndarray, side: str, gap: int) -> np.ndarray:
    if side == "top":
        return _translate(coords, -gap, 0)
    if side == "bottom":
        return _translate(coords, gap, 0)
    if side == "left":
        return _translate(coords, 0, -gap)
    if side == "right":
        return _translate(coords, 0, gap)
    raise AssertionError("invalid side")


def _build_input_and_expected(
    n: int,
    anchor_bbox: tuple[int, int, int, int],
    final_objects: dict[int, dict],
    gaps: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((n, n), dtype=int)
    ar0, ac0, ar1, ac1 = anchor_bbox
    inp[ar0 : ar1 + 1, ac0 : ac1 + 1] = ANCHOR
    for color, info in final_objects.items():
        shifted = _shift_outward(info["final"], info["side"], gaps[color])
        inp[shifted[:, 0], shifted[:, 1]] = int(color)

    assembled = np.zeros_like(inp)
    assembled[ar0 : ar1 + 1, ac0 : ac1 + 1] = ANCHOR
    for color, info in final_objects.items():
        coords = info["final"]
        assembled[coords[:, 0], coords[:, 1]] = int(color)

    tile = _crop_nonzero(assembled)
    expected = np.block(
        [
            [tile, np.rot90(tile, -1)],
            [np.rot90(tile, 1), np.rot90(tile, 2)],
        ]
    )
    return inp, expected


def _validate_generated(inp: np.ndarray, expected: np.ndarray, final_objects: dict[int, dict]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    tile = expected[: expected.shape[0] // 2, : expected.shape[1] // 2]
    if np.array_equal(tile, np.rot90(tile, 1)):
        return False
    if np.array_equal(tile, np.rot90(tile, 2)):
        return False
    if expected.shape[0] > 30 or inp.shape[0] > 30:
        return False

    ar0 = inp.shape[0] // 2 - 1
    anchor_rows = {ar0, ar0 + 1}
    anchor_cols = {ar0, ar0 + 1}
    for color, info in final_objects.items():
        shifted_comp = components_mask(inp == color)
        if len(shifted_comp) != 1:
            return False
        coords = shifted_comp[0]
        side = info["side"]
        if side in {"top", "bottom"}:
            if not any(int(c) in anchor_cols for c in coords[:, 1]):
                return False
            if any(int(r) in anchor_rows for r in coords[:, 0]):
                return False
        else:
            if not any(int(r) in anchor_rows for r in coords[:, 0]):
                return False
            if any(int(c) in anchor_cols for c in coords[:, 1]):
                return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        layout = _choose_layout(rng)
        if layout is None:
            continue
        shapes, sides, tile_size = layout

        sizes = [n for n in (11, 13, 15) if n >= tile_size]
        rng.shuffle(sizes)
        built = None
        for n in sizes:
            try:
                anchor_bbox, final_objects = _place_final_objects(n, shapes, sides)
            except AssertionError:
                continue
            limits = {color: _side_gap_limit(info["final"], info["side"], n) for color, info in final_objects.items()}
            if min(limits.values()) < 1:
                continue
            gaps = {}
            at_least_one_long = False
            for color, limit in limits.items():
                cap = min(limit, 4)
                low = 2 if cap >= 2 and not at_least_one_long else 1
                gap = rng.randint(low, cap)
                if gap >= 2:
                    at_least_one_long = True
                gaps[color] = gap
            if not at_least_one_long:
                color = rng.choice(sorted(gaps))
                if limits[color] < 2:
                    continue
                gaps[color] = 2
            inp, expected = _build_input_and_expected(n, anchor_bbox, final_objects, gaps)
            if not _validate_generated(inp, expected, final_objects):
                continue
            sig = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
            if sig in SEEN_OUTPUT_SIGS:
                continue
            SEEN_OUTPUT_SIGS.add(sig)
            built = inp
            break
        if built is not None:
            return built
    raise RuntimeError("failed to generate a valid puzzle-6 input")
