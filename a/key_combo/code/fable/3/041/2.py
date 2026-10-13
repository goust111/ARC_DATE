from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
ARC_COLORS = tuple(range(10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: set[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(list(coords) if isinstance(coords, set) else coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, color: int, connectivity: int = 8) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    nbrs = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    out: list[dict] = []

    for sr, sc in np.argwhere(grid == int(color)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and not seen[rr, cc]
                    and int(grid[rr, cc]) == int(color)
                ):
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        arr = np.asarray(coords, dtype=int)
        out.append(
            {
                "color": int(color),
                "coords": arr,
                "bbox": _bbox(arr),
                "area": int(arr.shape[0]),
            }
        )
    return out


def _is_solid_square(comp: dict, grid: np.ndarray, min_side: int = 5) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < min_side or int(comp["area"]) != h * w:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == int(comp["color"])))


def _marker_from_red_component(comp: dict) -> dict:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    area = int(comp["area"])
    marker = dict(comp)
    if area == 1:
        marker["kind"] = "point"
        marker["point"] = tuple(map(int, comp["coords"][0]))
        return marker

    if r0 == r1 and area == c1 - c0 + 1 and area >= 3:
        marker["kind"] = "hline"
        marker["axis"] = r0
        return marker
    if c0 == c1 and area == r1 - r0 + 1 and area >= 3:
        marker["kind"] = "vline"
        marker["axis"] = c0
        return marker
    raise AssertionError("red markers must be isolated points or straight lines")


def _touches(a: dict, b: dict, connectivity: int = 8) -> bool:
    offsets = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    bcoords = {tuple(map(int, xy)) for xy in b["coords"]}
    for r, c in a["coords"]:
        r, c = int(r), int(c)
        for dr, dc in offsets:
            if (r + dr, c + dc) in bcoords:
                return True
    return False


def _completed_coords(kind: str, source: set[tuple[int, int]], marker: set[tuple[int, int]]) -> set[tuple[int, int]]:
    full = set(source)
    if kind == "hline":
        rows = {r for r, _c in marker}
        assert len(rows) == 1
        axis = next(iter(rows))
        deltas = [r - axis for r, _c in source]
        assert all(d != 0 for d in deltas)
        assert all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        full.update((2 * axis - r, c) for r, c in source)
    elif kind == "vline":
        cols = {c for _r, c in marker}
        assert len(cols) == 1
        axis = next(iter(cols))
        deltas = [c - axis for _r, c in source]
        assert all(d != 0 for d in deltas)
        assert all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        full.update((r, 2 * axis - c) for r, c in source)
    elif kind == "point":
        assert len(marker) == 1
        mr, mc = next(iter(marker))
        assert all((r, c) != (mr, mc) for r, c in source)
        full.update((2 * mr - r, 2 * mc - c) for r, c in source)
    else:
        raise AssertionError(f"unknown marker kind: {kind}")
    assert len(full) > len(source), "the reflection must add missing template cells"
    return full


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _template_mask(marker: dict, residue: dict, shape: tuple[int, int]) -> np.ndarray:
    source = {tuple(map(int, xy)) for xy in residue["coords"]}
    marker_cells = {tuple(map(int, xy)) for xy in marker["coords"]}
    full = _completed_coords(str(marker["kind"]), source, marker_cells)

    h, w = shape
    for r, c in full:
        assert 0 <= r < h and 0 <= c < w
    return _mask_from_coords(full)


def _carve(out: np.ndarray, bg: int, block: dict, mask: np.ndarray) -> None:
    r0, c0, r1, c1 = map(int, block["bbox"])
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1
    mh, mw = map(int, mask.shape)
    assert mh <= side and mw <= side
    assert (side - mh) % 2 == 0 and (side - mw) % 2 == 0

    off_r = r0 + (side - mh) // 2
    off_c = c0 + (side - mw) // 2
    rr, cc = np.argwhere(mask).T
    assert bool(np.all(out[off_r + rr, off_c + cc] == int(block["color"])))
    out[off_r + rr, off_c + cc] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each half-drawn same-color residue from its red reflection marker, then punch it into the matching solid block.

    Input structure:
    - The background is the dominant color, and red pixels form separate markers.
    - Every non-background, non-red color has one solid square block and one small
      same-color residue elsewhere in the grid.
    - Each residue touches exactly one red marker: either a straight red line or
      one isolated red point.

    Rule:
    - A red line reflects the residue across that horizontal or vertical line.
    - A red point reflects the residue through that point by 180 degrees.
    - The residue plus its reflection is centered inside the same-color solid
      block, and those covered block cells are recolored to the background.
    - The drawn residues and all red markers are cleared to the background; the
      carved blocks remain in their original positions.

    Key insight:
    - The visible motif is only half of the stencil. The red marker geometry
      determines the missing half before the stencil can be matched by color and
      used as a hole-punch template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert int(RED) != int(bg)

    colors = sorted(set(map(int, np.unique(g))) - {int(bg), RED})
    assert colors, "expected at least one block/residue color"

    red_components = _components(g, RED, connectivity=4)
    markers = [_marker_from_red_component(comp) for comp in red_components]
    assert markers, "expected red reflection markers"

    blocks: dict[int, dict] = {}
    residues: dict[int, dict] = {}
    for color in colors:
        comps = _components(g, color, connectivity=8)
        block_indices = [i for i, comp in enumerate(comps) if _is_solid_square(comp, g)]
        assert len(block_indices) == 1, "each color needs one solid square block"
        block_index = block_indices[0]
        other = [comp for i, comp in enumerate(comps) if i != block_index]
        assert len(other) == 1, "each block color needs one residue component"
        assert int(other[0]["area"]) < int(comps[block_index]["area"])
        blocks[color] = comps[block_index]
        residues[color] = other[0]

    marker_by_color: dict[int, dict] = {}
    for marker in markers:
        touching = [color for color, residue in residues.items() if _touches(marker, residue, connectivity=8)]
        assert len(touching) == 1, "each red marker must touch exactly one residue"
        color = int(touching[0])
        assert color not in marker_by_color
        marker_by_color[color] = marker

    assert set(marker_by_color) == set(colors)

    out = g.copy()
    for marker in markers:
        coords = marker["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for residue in residues.values():
        coords = residue["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for color in colors:
        mask = _template_mask(marker_by_color[color], residues[color], g.shape)
        _carve(out, bg, blocks[color], mask)
    return out


# ---------------- generator ----------------


def _grow_connected(candidates: set[tuple[int, int]], start: tuple[int, int], target: int) -> set[tuple[int, int]] | None:
    if start not in candidates:
        return None
    cells = {start}
    while len(cells) < int(target):
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in NEIGHBORS8
            if (r + dr, c + dc) in candidates and (r + dr, c + dc) not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _normalize_spec(kind: str, source: set[tuple[int, int]], marker: set[tuple[int, int]]) -> dict | None:
    try:
        full = _completed_coords(kind, source, marker)
    except AssertionError:
        return None

    union = set(full) | set(marker)
    r0, c0, _r1, _c1 = _bbox(union)

    def shift(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r - r0, c - c0) for r, c in cells}

    source = shift(source)
    marker = shift(marker)
    full = shift(full)
    mask = _mask_from_coords(full)
    mh, mw = map(int, mask.shape)

    if mh < 3 or mw < 3 or mh > 9 or mw > 9:
        return None
    if mh % 2 != 1 or mw % 2 != 1:
        return None
    if int(mask.sum()) <= len(source):
        return None

    all_cells = set(full) | set(marker)
    br0, bc0, br1, bc1 = _bbox(all_cells)
    assert br0 == 0 and bc0 == 0
    return {
        "kind": kind,
        "source": source,
        "marker": marker,
        "mask": mask,
        "foot_h": br1 - br0 + 1,
        "foot_w": bc1 - bc0 + 1,
    }


def _make_line_spec() -> dict | None:
    for _ in range(200):
        kind = random.choice(("hline", "vline"))
        depth = random.randint(1, 3)
        span = random.choice((3, 5, 7))
        sign = random.choice((-1, 1))

        if kind == "hline":
            axis = depth
            marker = {(axis, c) for c in range(span)}
            candidates = {
                (axis + sign * d, c)
                for d in range(1, depth + 1)
                for c in range(span)
            }
            start = (axis + sign, span // 2)
        else:
            axis = depth
            marker = {(r, axis) for r in range(span)}
            candidates = {
                (r, axis + sign * d)
                for d in range(1, depth + 1)
                for r in range(span)
            }
            start = (span // 2, axis + sign)

        max_area = min(8, len(candidates))
        target = random.randint(3, max_area)
        source = _grow_connected(candidates, start, target)
        if source is None:
            continue
        spec = _normalize_spec(kind, source, marker)
        if spec is not None:
            return spec
    return None


def _make_point_spec() -> dict | None:
    for _ in range(200):
        dr_max = random.randint(1, 3)
        dc_max = random.randint(1, 3)
        mr, mc = dr_max, dc_max
        sr, sc = random.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
        marker = {(mr, mc)}
        candidates = {
            (mr + sr * dr, mc + sc * dc)
            for dr in range(1, dr_max + 1)
            for dc in range(1, dc_max + 1)
        }
        start = (mr + sr, mc + sc)
        max_area = min(7, len(candidates))
        target = random.randint(1, max_area)
        source = _grow_connected(candidates, start, target)
        if source is None:
            continue
        spec = _normalize_spec("point", source, marker)
        if spec is not None:
            return spec
    return None


def _choose_block_side(mask: np.ndarray) -> int | None:
    mh, mw = map(int, mask.shape)
    options = [
        side
        for side in (5, 7, 9, 11)
        if side >= mh and side >= mw and (side - mh) % 2 == 0 and (side - mw) % 2 == 0
    ]
    if not options:
        return None
    return random.choice(options)


def _box_clear(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = occupied.shape
    if r < 0 or c < 0 or r + h > rows or c + w > cols:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(rows, r + h + pad)
    c1 = min(cols, c + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_box(occupied: np.ndarray, h: int, w: int, pad: int = 1) -> tuple[int, int] | None:
    rows, cols = occupied.shape
    choices = [
        (r, c)
        for r in range(rows - h + 1)
        for c in range(cols - w + 1)
        if _box_clear(occupied, r, c, h, w, pad=pad)
    ]
    if not choices:
        return None
    return random.choice(choices)


def _reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    occupied[r : r + h, c : c + w] = True


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], r0: int, c0: int, color: int) -> set[tuple[int, int]]:
    placed: set[tuple[int, int]] = set()
    for r, c in cells:
        rr, cc = r0 + int(r), c0 + int(c)
        grid[rr, cc] = int(color)
        placed.add((rr, cc))
    return placed


def _expected_from_items(inp: np.ndarray, bg: int, items: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in items:
        for r, c in item["source_abs"] | item["marker_abs"]:
            out[r, c] = int(bg)

    for item in items:
        r0, c0, side = item["block"]
        mask = item["mask"]
        mh, mw = map(int, mask.shape)
        rr, cc = np.argwhere(mask).T
        off_r = r0 + (side - mh) // 2
        off_c = c0 + (side - mw) // 2
        out[off_r + rr, off_c + cc] = int(bg)
    return out


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, bg: int, items: list[dict]) -> bool:
    if np.array_equal(inp, expected):
        return False
    if _get_bg(inp) != int(bg):
        return False
    if not any(item["kind"] == "point" for item in items):
        return False
    if not any(item["kind"] in {"hline", "vline"} for item in items):
        return False
    for item in items:
        if int(item["mask"].sum()) <= len(item["source_abs"]):
            return False
        r0, c0, side = item["block"]
        block_after = expected[r0 : r0 + side, c0 : c0 + side]
        if not np.any(block_after == int(bg)):
            return False
        if not np.any(block_after == int(item["color"])):
            return False
    return True


def _make_specs(count: int) -> list[dict] | None:
    branches = ["point", "line"]
    while len(branches) < count:
        branches.append(random.choice(("point", "line")))
    random.shuffle(branches)

    specs: list[dict] = []
    for branch in branches:
        spec = _make_point_spec() if branch == "point" else _make_line_spec()
        if spec is None:
            return None
        side = _choose_block_side(spec["mask"])
        if side is None:
            return None
        spec["side"] = side
        specs.append(spec)
    return specs


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(22, 30)
    cols = random.randint(22, 30)
    bg = random.choice([c for c in ARC_COLORS if c != RED])
    count = random.choice((2, 2, 3))
    colors = random.sample([c for c in ARC_COLORS if c not in {bg, RED}], count)
    specs = _make_specs(count)
    if specs is None:
        return None

    grid = np.full((rows, cols), int(bg), dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    items: list[dict] = []

    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        color = int(colors[idx])
        spec = specs[idx]
        side = int(spec["side"])

        block_pos = _place_box(occupied, side, side, pad=1)
        if block_pos is None:
            return None
        br, bc = block_pos
        grid[br : br + side, bc : bc + side] = color
        _reserve(occupied, br, bc, side, side)

        foot_h, foot_w = int(spec["foot_h"]), int(spec["foot_w"])
        motif_pos = _place_box(occupied, foot_h, foot_w, pad=1)
        if motif_pos is None:
            return None
        mr, mc = motif_pos
        _reserve(occupied, mr, mc, foot_h, foot_w)
        source_abs = _paint_cells(grid, spec["source"], mr, mc, color)
        marker_abs = _paint_cells(grid, spec["marker"], mr, mc, RED)

        items.append(
            {
                "color": color,
                "kind": spec["kind"],
                "block": (br, bc, side),
                "mask": spec["mask"],
                "source_abs": source_abs,
                "marker_abs": marker_abs,
            }
        )

    expected = _expected_from_items(grid, bg, items)
    if not _semantic_ok(grid, expected, bg, items):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residue-completion punching input")
