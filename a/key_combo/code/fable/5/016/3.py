from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)


def _background(grid: np.ndarray) -> int:
    """Return the unique modal color; valid task inputs always have one."""
    counts = Counter(map(int, grid.ravel()))
    largest = max(counts.values())
    modes = [color for color, count in counts.items() if count == largest]
    assert len(modes) == 1, "background must be the unique modal color"
    return int(modes[0])


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    """Find monochromatic 4-connected components using (row, column) coordinates."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == bg or bool(seen[r0, c0]):
                continue

            seen[r0, c0] = True
            queue = deque([(r0, c0)])
            coords: list[Coord] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if bool(seen[rr, cc]) or int(grid[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    queue.append((rr, cc))

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
    return objects


def _sprite_style(grid: np.ndarray, obj: dict, bg: int) -> str:
    """Classify a promised sprite as a filled rectangle or a one-cell frame."""
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if bool(np.all(patch == color)):
        return "solid"

    ph, pw = patch.shape
    assert ph >= 3 and pw >= 3, "a hollow sprite must enclose a visible hole"
    frame = np.zeros((ph, pw), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    assert bool(np.all(patch[frame] == color)), "hollow sprite boundary is incomplete"
    assert bool(np.all(patch[~frame] == bg)), "hollow sprite interior must be background"
    return "hollow"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace area-coded sprites with colored square rings around the isolated marker.

    Input structure:
    - The unique modal color is background; exactly one non-background 4-connected
      component is a single isolated marker pixel.
    - Every other component is a small monochromatic sprite, either a filled rectangle
      or a one-cell hollow rectangular frame, and all sprite pixel areas are distinct.

    Rule:
    - Erase every sprite and retain the marker at its original row and column.
    - A sprite of pixel area d claims the complete Chebyshev-distance-d square ring
      around the marker in its own color when solid; when hollow, it colors only the
      four corners of that same ring. Unclaimed distances remain background.

    Key insight:
    - Sprite position is irrelevant: pixel area determines ring distance, while
      solidity independently determines whether the ring is complete or corner-only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    bg = _background(g)
    objects = _components(g, bg)

    markers = [obj for obj in objects if int(obj["area"]) == 1]
    assert len(markers) == 1, "there must be exactly one single-pixel marker"
    marker = markers[0]
    mr, mc = marker["coords"][0]

    sprites = [obj for obj in objects if obj is not marker]
    assert len(sprites) >= 2, "the task contains several sprites"
    areas = [int(obj["area"]) for obj in sprites]
    assert len(areas) == len(set(areas)), "sprite areas must be pairwise distinct"

    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)
    out[mr, mc] = int(marker["color"])

    rr, cc = np.indices((h, w))
    distance = np.maximum(np.abs(rr - mr), np.abs(cc - mc))
    for obj in sprites:
        radius = int(obj["area"])
        assert (
            mr - radius >= 0
            and mr + radius < h
            and mc - radius >= 0
            and mc + radius < w
        ), "every claimed square ring must fit completely inside the grid"

        color = int(obj["color"])
        style = _sprite_style(g, obj, bg)
        if style == "solid":
            out[distance == radius] = color
        else:
            for r in (mr - radius, mr + radius):
                for c in (mc - radius, mc + radius):
                    out[r, c] = color

    return out


# ------------------------------ generator ------------------------------


def _solid(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _hollow(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _make_spec(mask: np.ndarray, style: str) -> dict:
    return {
        "mask": np.asarray(mask, dtype=bool),
        "style": style,
        "area": int(np.sum(mask)),
    }


def _paired_specs() -> list[dict]:
    """Same-bbox pair prevents bbox area from replacing true pixel counting."""
    if random.random() < 0.5:
        h, w = 3, 3
    else:
        h, w = random.choice([(3, 4), (4, 3)])
    return [_make_spec(_solid(h, w), "solid"), _make_spec(_hollow(h, w), "hollow")]


def _extra_candidates() -> list[dict]:
    """Offer overlapping area/style possibilities across separately generated samples."""
    long_2x3 = _solid(2, 3)
    long_2x4 = _solid(2, 4)
    long_2x5 = _solid(2, 5)
    frame_3x4 = _hollow(3, 4)
    frame_3x5 = _hollow(3, 5)
    if random.random() < 0.5:
        long_2x3 = long_2x3.T
    if random.random() < 0.5:
        long_2x4 = long_2x4.T
    if random.random() < 0.5:
        long_2x5 = long_2x5.T
    if random.random() < 0.5:
        frame_3x4 = frame_3x4.T
    if random.random() < 0.5:
        frame_3x5 = frame_3x5.T

    return [
        _make_spec(_solid(2, 2), "solid"),       # area 4
        _make_spec(long_2x3, "solid"),           # area 6
        _make_spec(long_2x4, "solid"),           # area 8
        _make_spec(_hollow(3, 3), "hollow"),     # area 8
        _make_spec(_solid(3, 3), "solid"),       # area 9
        _make_spec(long_2x5, "solid"),           # area 10
        _make_spec(frame_3x4, "hollow"),         # area 10
        _make_spec(_solid(3, 4), "solid"),       # area 12
        _make_spec(frame_3x5, "hollow"),         # area 12
        _make_spec(_hollow(4, 4), "hollow"),     # area 12, alternate shape
    ]


def _choose_specs(count: int) -> list[dict] | None:
    specs = _paired_specs()
    used = {int(spec["area"]) for spec in specs}
    while len(specs) < count:
        choices = [spec for spec in _extra_candidates() if int(spec["area"]) not in used]
        if not choices:
            return None
        spec = random.choice(choices)
        specs.append(spec)
        used.add(int(spec["area"]))
    random.shuffle(specs)
    return specs


def _candidate_positions(
    occupied: np.ndarray,
    mask: np.ndarray,
    marker: Coord,
    claimed_distances: set[int],
) -> list[Coord]:
    """List placements with an 8-neighbor gap and no overlap with future rings."""
    h, w = occupied.shape
    ph, pw = mask.shape
    mr, mc = marker
    points = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    candidates: list[Coord] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            good = True
            for pr, pc in points:
                r, c = r0 + pr, c0 + pc
                if max(abs(r - mr), abs(c - mc)) in claimed_distances:
                    good = False
                    break
                rlo, rhi = max(0, r - 1), min(h, r + 2)
                clo, chi = max(0, c - 1), min(w, c + 2)
                if bool(occupied[rlo:rhi, clo:chi].any()):
                    good = False
                    break
            if good:
                candidates.append((r0, c0))
    return candidates


def _independent_expected(
    shape: tuple[int, int],
    bg: int,
    marker: Coord,
    marker_color: int,
    specs: list[dict],
) -> np.ndarray:
    """Render the oracle directly from sampled semantic metadata."""
    h, w = shape
    mr, mc = marker
    out = np.full((h, w), bg, dtype=int)
    out[mr, mc] = marker_color
    rr, cc = np.indices((h, w))
    distance = np.maximum(np.abs(rr - mr), np.abs(cc - mc))
    for spec in specs:
        radius = int(spec["area"])
        color = int(spec["color"])
        if spec["style"] == "solid":
            out[distance == radius] = color
        else:
            out[mr - radius, mc - radius] = color
            out[mr - radius, mc + radius] = color
            out[mr + radius, mc - radius] = color
            out[mr + radius, mc + radius] = color
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    marker: Coord,
    marker_color: int,
    specs: list[dict],
) -> bool:
    """Check rule dependencies, uniqueness, separation, and non-degeneracy."""
    if inp.ndim != 2 or max(inp.shape) > 30 or _background(inp) != bg:
        return False

    objects = _components(inp, bg)
    singletons = [obj for obj in objects if int(obj["area"]) == 1]
    if len(objects) != len(specs) + 1 or len(singletons) != 1:
        return False
    if singletons[0]["coords"][0] != marker or int(singletons[0]["color"]) != marker_color:
        return False

    areas = [int(spec["area"]) for spec in specs]
    if len(areas) != len(set(areas)):
        return False
    if {spec["style"] for spec in specs} != {"solid", "hollow"}:
        return False

    # At least one solid/hollow pair has identical bbox dimensions. Therefore bbox
    # size cannot stand in for either true area or solidity.
    same_bbox_counterexample = any(
        a["style"] != b["style"] and a["mask"].shape == b["mask"].shape
        for i, a in enumerate(specs)
        for b in specs[i + 1 :]
    )
    if not same_bbox_counterexample:
        return False

    mr, mc = marker
    for spec in specs:
        matching = [obj for obj in objects if int(obj["color"]) == int(spec["color"])]
        if len(matching) != 1:
            return False
        obj = matching[0]
        if int(obj["area"]) != int(spec["area"]):
            return False
        try:
            if _sprite_style(inp, obj, bg) != spec["style"]:
                return False
        except AssertionError:
            return False

        radius = int(spec["area"])
        coords = np.argwhere(expected == int(spec["color"]))
        wanted_count = 8 * radius if spec["style"] == "solid" else 4
        if int(coords.shape[0]) != wanted_count:
            return False
        if any(max(abs(int(r) - mr), abs(int(c) - mc)) != radius for r, c in coords):
            return False

        # Changing only solidity must change the result, so the style branch is real.
        if radius <= 1 or wanted_count == (4 if spec["style"] == "solid" else 8 * radius):
            return False

        # Placement is visibly erased: no old sprite cell is reused by any ring.
        if any(int(expected[r, c]) != bg for r, c in obj["coords"]):
            return False

    # Every sprite color contributes, radii are disjoint, and unclaimed distances stay bg.
    non_bg_colors = set(map(int, np.unique(expected))) - {bg, marker_color}
    if non_bg_colors != {int(spec["color"]) for spec in specs}:
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        return bool(np.array_equal(main(inp), expected))
    except AssertionError:
        return False


_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    specs = _choose_specs(count)
    if specs is None:
        return None

    max_radius = max(int(spec["area"]) for spec in specs)
    minimum_side = 2 * max_radius + 1
    h = random.randint(minimum_side, 30)
    w = random.randint(minimum_side, 30)
    mr = random.randint(max_radius, h - max_radius - 1)
    mc = random.randint(max_radius, w - max_radius - 1)
    marker = (mr, mc)

    palette = list(range(10))
    bg = random.choice(palette)
    colors = [color for color in palette if color != bg]
    random.shuffle(colors)
    marker_color = colors.pop()
    for spec in specs:
        spec["color"] = colors.pop()

    inp = np.full((h, w), bg, dtype=int)
    inp[marker] = marker_color
    occupied = np.zeros((h, w), dtype=bool)
    occupied[marker] = True
    claimed = {int(spec["area"]) for spec in specs}

    # Place larger patches first; random candidate selection keeps locations unrelated
    # to area, color, and eventual ring distance.
    placement_order = sorted(specs, key=lambda spec: int(spec["mask"].size), reverse=True)
    for spec in placement_order:
        candidates = _candidate_positions(occupied, spec["mask"], marker, claimed)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        mask = spec["mask"]
        ph, pw = mask.shape
        view = inp[r0 : r0 + ph, c0 : c0 + pw]
        view[mask] = int(spec["color"])
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
        spec["position"] = (r0, c0)

    expected = _independent_expected(inp.shape, bg, marker, marker_color, specs)
    if not _semantic_checks(inp, expected, bg, marker, marker_color, specs):
        return None

    # Avoid repeated generated outputs even when two inputs differ only in lure positions.
    signature = (tuple(map(int, expected.shape)), expected.tobytes())
    if signature in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(signature)
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, semantically validated ARC input grid."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid area-to-ring sample after 300 attempts")
