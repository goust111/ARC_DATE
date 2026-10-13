from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, get_bg, neighbors4


BG = 0
RING = 5
BODY_COLORS = (1, 3)
MARKER_COLORS = (2, 4)
SEED_COLORS = (7, 8)
SPRITE_MASK_COLORS = BODY_COLORS + MARKER_COLORS
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _find_anchors(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    anchors: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if any(
                0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg
                for nr, nc in neighbors4(r, c)
            ):
                continue
            anchors[color] = (int(r), int(c))
    return anchors


def _sprite_components(grid: np.ndarray) -> list[np.ndarray]:
    mask = np.isin(grid, SPRITE_MASK_COLORS)
    return [comp for comp in components_mask(mask, connectivity=4) if len(comp) > 1]


def _marker_info(grid: np.ndarray, coords: np.ndarray) -> tuple[int, tuple[int, int]]:
    colors = [int(grid[r, c]) for r, c in coords]
    counts = {color: colors.count(color) for color in set(colors)}
    marker_candidates = [color for color, count in counts.items() if count == 1]
    assert len(counts) == 2 and len(marker_candidates) == 1, "expected one marked two-color sprite"
    marker = int(marker_candidates[0])
    marker_rc = tuple(map(int, coords[[int(grid[r, c]) == marker for r, c in coords]][0]))
    return marker, marker_rc


def _slide_sprites(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    anchors = _find_anchors(g, bg)

    moved = g.copy()
    translations: list[tuple[np.ndarray, int, int]] = []
    for coords in _sprite_components(g):
        marker_color, (mr, mc) = _marker_info(g, coords)
        assert marker_color in anchors, "sprite marker must match a border anchor"
        ar, ac = anchors[marker_color]
        dr = 0
        dc = 0
        if ar in (0, h - 1):
            dc = int(ac - mc)
        else:
            dr = int(ar - mr)
        translations.append((coords, dr, dc))
        moved[coords[:, 0], coords[:, 1]] = bg

    for coords, dr, dc in translations:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            assert 0 <= rr < h and 0 <= cc < w, "slid sprite must remain in bounds"
            assert int(moved[rr, cc]) == bg, "slid sprite must land only on background"
            moved[rr, cc] = int(g[r, c])

    return moved


def _fill_closed_seeded_regions(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    bg = int(get_bg(out))
    for comp in components_mask(out == bg, connectivity=4):
        if _touches_border(comp, h, w):
            continue
        seed_colors = set()
        for r, c in comp:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) in SEED_COLORS:
                    seed_colors.add(int(out[nr, nc]))
        assert len(seed_colors) <= 1, "each enclosed cavity must have at most one seed color"
        if len(seed_colors) == 1:
            out[comp[:, 0], comp[:, 1]] = next(iter(seed_colors))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marker-matched sprites to seal line gaps, then flood-fill the newly enclosed seeded cavities.

    Input structure:
    - The background is black.
    - One or two gray 4-connected ring drawings each have exactly one straight gap on one side and one seed pixel inside.
    - Each movable sprite is a small two-color component: a main-color body plus a unique marker pixel whose color matches one isolated border anchor.

    Rule:
    - Find isolated non-background border anchors.
    - For each sprite, identify its minority-color marker pixel and pair it with the border anchor of the same color.
    - If the anchor is on the top or bottom edge, slide the whole sprite horizontally until the marker column matches the anchor column; otherwise slide it vertically until the marker row matches the anchor row.
    - After sliding, the sprite body plugs the ring gap while the marker stays outside the ring.
    - Flood-fill every background region that is now closed and contains exactly one adjacent seed color, keeping the rings, sprites, anchors, and seeds themselves.

    Key insight:
    - The slide is not the final effect: it changes background connectivity so an area that was open before becomes a closed cavity that can then be filled from its internal seed.
    """
    after_slide = _slide_sprites(np.asarray(grid, dtype=int))
    return _fill_closed_seeded_regions(after_slide)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rect_boundary(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c0 + w):
        cells.add((r0, c))
        cells.add((r0 + h - 1, c))
    for r in range(r0, r0 + h):
        cells.add((r, c0))
        cells.add((r, c0 + w - 1))
    return cells


def _sprite_geometry(side: str, gap_len: int, marker_index: int, body_color: int, marker_color: int) -> dict:
    if side == "top":
        pixels = [(1, dc, body_color) for dc in range(gap_len)] + [(0, marker_index, marker_color)]
        return {"shape": (2, gap_len), "pixels": pixels, "marker_local": (0, marker_index)}
    if side == "bottom":
        pixels = [(0, dc, body_color) for dc in range(gap_len)] + [(1, marker_index, marker_color)]
        return {"shape": (2, gap_len), "pixels": pixels, "marker_local": (1, marker_index)}
    if side == "left":
        pixels = [(dr, 1, body_color) for dr in range(gap_len)] + [(marker_index, 0, marker_color)]
        return {"shape": (gap_len, 2), "pixels": pixels, "marker_local": (marker_index, 0)}
    pixels = [(dr, 0, body_color) for dr in range(gap_len)] + [(marker_index, 1, marker_color)]
    return {"shape": (gap_len, 2), "pixels": pixels, "marker_local": (marker_index, 1)}


def _protrusion_cell(r0: int, c0: int, h: int, w: int, side: str) -> tuple[int, int] | None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if side == "top":
        return (r0 - 1, random.randint(c0 + 1, c1 - 1))
    if side == "bottom":
        return (r1 + 1, random.randint(c0 + 1, c1 - 1))
    if side == "left":
        return (random.randint(r0 + 1, r1 - 1), c0 - 1)
    return (random.randint(r0 + 1, r1 - 1), c1 + 1)


def _candidate_pair_spec(
    h: int,
    w: int,
    side: str,
    body_color: int,
    marker_color: int,
    seed_color: int,
    reserved_boxes: list[tuple[int, int, int, int]],
) -> dict | None:
    ring_h = random.randint(5, 7)
    ring_w = random.randint(5, 7)
    gap_len = random.choice((2, 3))
    if (side in ("top", "bottom") and ring_w - 2 < gap_len) or (side in ("left", "right") and ring_h - 2 < gap_len):
        return None

    r0 = random.randint(2, h - ring_h - 2)
    c0 = random.randint(2, w - ring_w - 2)
    r1 = r0 + ring_h - 1
    c1 = c0 + ring_w - 1

    if side in ("top", "bottom"):
        start = random.randint(1, ring_w - gap_len - 1)
        gap_cells = [(r0 if side == "top" else r1, c0 + start + i) for i in range(gap_len)]
        marker_index = random.randrange(gap_len)
        geom = _sprite_geometry(side, gap_len, marker_index, body_color, marker_color)
        final_top_left = (r0 - 1, c0 + start) if side == "top" else (r1, c0 + start)
        axis = "horizontal"
    else:
        start = random.randint(1, ring_h - gap_len - 1)
        gap_cells = [(r0 + start + i, c0 if side == "left" else c1) for i in range(gap_len)]
        marker_index = random.randrange(gap_len)
        geom = _sprite_geometry(side, gap_len, marker_index, body_color, marker_color)
        final_top_left = (r0 + start, c0 - 1) if side == "left" else (r0 + start, c1)
        axis = "vertical"

    fr, fc = final_top_left
    sh, sw = geom["shape"]
    if fr < 0 or fc < 0 or fr + sh > h or fc + sw > w:
        return None

    reserve = (r0 - 2, c0 - 2, r1 + 2, c1 + 2)
    if any(_boxes_overlap(reserve, old) for old in reserved_boxes):
        return None

    ring_cells = _rect_boundary(r0, c0, ring_h, ring_w) - set(gap_cells)
    protrusion = None
    if random.random() < 0.75:
        sides = [candidate for candidate in SIDES if candidate != side]
        random.shuffle(sides)
        for candidate in sides:
            cell = _protrusion_cell(r0, c0, ring_h, ring_w, candidate)
            if 0 <= cell[0] < h and 0 <= cell[1] < w and cell not in ring_cells and cell not in gap_cells:
                protrusion = cell
                break

    seed = (random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1))
    marker_local = geom["marker_local"]
    final_marker = (fr + marker_local[0], fc + marker_local[1])

    return {
        "ring_box": (r0, c0, r1, c1),
        "ring_cells": ring_cells,
        "gap_cells": gap_cells,
        "protrusion": protrusion,
        "seed": seed,
        "seed_color": seed_color,
        "body_color": body_color,
        "marker_color": marker_color,
        "side": side,
        "axis": axis,
        "sprite_shape": geom["shape"],
        "sprite_pixels": geom["pixels"],
        "marker_local": marker_local,
        "final_top_left": final_top_left,
        "final_marker": final_marker,
        "reserve": reserve,
    }


def _sprite_cells(top_left: tuple[int, int], spec: dict) -> list[tuple[int, int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc, color) for dr, dc, color in spec["sprite_pixels"]]


def _isolated_anchor_candidate(anchor: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    ar, ac = anchor
    for br, bc in used:
        if abs(ar - br) + abs(ac - bc) <= 1:
            return False
    return True


def _assign_anchor(spec: dict, grid: np.ndarray, used: list[tuple[int, int]]) -> tuple[int, int] | None:
    h, w = grid.shape
    mr, mc = spec["final_marker"]
    candidates = []
    if spec["axis"] == "horizontal":
        candidates = [(0, mc), (h - 1, mc)]
    else:
        candidates = [(mr, 0), (mr, w - 1)]
    random.shuffle(candidates)
    for anchor in candidates:
        ar, ac = anchor
        if int(grid[ar, ac]) != BG:
            continue
        if not _isolated_anchor_candidate(anchor, used):
            continue
        if any(
            0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG
            for nr, nc in neighbors4(ar, ac)
        ):
            continue
        return anchor
    return None


def _bbox_clear(grid: np.ndarray, r0: int, c0: int, sh: int, sw: int, pad: int = 1) -> bool:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + sh > h or c0 + sw > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + sh + pad)
    cc1 = min(w, c0 + sw + pad)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _place_initial_sprite(grid: np.ndarray, spec: dict) -> tuple[int, int] | None:
    sh, sw = spec["sprite_shape"]
    fr, fc = spec["final_top_left"]
    candidates: list[tuple[int, int]] = []
    if spec["axis"] == "horizontal":
        for c0 in range(0, grid.shape[1] - sw + 1):
            if c0 == fc or abs(c0 - fc) < 3:
                continue
            if _bbox_clear(grid, fr, c0, sh, sw, pad=1):
                candidates.append((fr, c0))
    else:
        for r0 in range(0, grid.shape[0] - sh + 1):
            if r0 == fr or abs(r0 - fr) < 3:
                continue
            if _bbox_clear(grid, r0, fc, sh, sw, pad=1):
                candidates.append((r0, fc))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c, _color in _sprite_cells(spec["initial_top_left"], spec):
            out[r, c] = BG
    for spec in specs:
        for r, c, color in _sprite_cells(spec["final_top_left"], spec):
            out[r, c] = color
    for spec in specs:
        r0, c0, r1, c1 = spec["ring_box"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == BG:
                    out[r, c] = spec["seed_color"]
    return out


def _component_touching_seed(grid: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    seed_r, seed_c = seed
    bg_components = components_mask(grid == BG, connectivity=4)
    for comp in bg_components:
        region = {tuple(map(int, cell)) for cell in comp}
        for nr, nc in neighbors4(seed_r, seed_c):
            if (nr, nc) in region:
                return comp
    raise AssertionError("seed must touch a background region before fill")


def _validate_instance(inp: np.ndarray, specs: list[dict], expected: np.ndarray) -> None:
    h, w = inp.shape
    for spec in specs:
        ar, ac = spec["anchor"]
        assert ar in (0, h - 1) or ac in (0, w - 1)
        for nr, nc in neighbors4(ar, ac):
            if 0 <= nr < h and 0 <= nc < w:
                assert int(inp[nr, nc]) == BG
        before = _component_touching_seed(inp, spec["seed"])
        assert _touches_border(before, h, w), "ring interior must be open before the slide"

    solved = main(inp)
    assert np.array_equal(solved, expected), "generator and solver must agree"

    for spec in specs:
        r0, c0, r1, c1 = spec["ring_box"]
        interior = solved[r0 + 1 : r1, c0 + 1 : c1]
        assert not np.any(interior == BG), "closed ring interior must be fully filled"
        assert set(np.unique(interior)) <= {spec["seed_color"]}


def _side_pattern(pair_count: int) -> list[str]:
    if pair_count == 1:
        return [random.choice(SIDES)]
    sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    random.shuffle(sides)
    return sides


def _attempt_generate() -> np.ndarray | None:
    pair_count = random.choice((1, 2, 2))
    if pair_count == 1:
        h = random.randint(16, 22)
        w = random.randint(16, 22)
    else:
        h = random.randint(20, 26)
        w = random.randint(20, 26)

    body_colors = random.sample(BODY_COLORS, pair_count)
    marker_colors = random.sample(MARKER_COLORS, pair_count)
    seed_colors = random.sample(SEED_COLORS, pair_count)
    sides = _side_pattern(pair_count)

    specs: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []
    for side, body_color, marker_color, seed_color in zip(sides, body_colors, marker_colors, seed_colors):
        spec = None
        for _ in range(120):
            spec = _candidate_pair_spec(h, w, side, body_color, marker_color, seed_color, reserved)
            if spec is not None:
                break
        if spec is None:
            return None
        specs.append(spec)
        reserved.append(spec["reserve"])

    inp = np.zeros((h, w), dtype=int)
    for spec in specs:
        for r, c in spec["ring_cells"]:
            inp[r, c] = RING
        if spec["protrusion"] is not None:
            inp[spec["protrusion"]] = RING
        sr, sc = spec["seed"]
        inp[sr, sc] = spec["seed_color"]

    used_anchors: list[tuple[int, int]] = []
    for spec in specs:
        anchor = _assign_anchor(spec, inp, used_anchors)
        if anchor is None:
            return None
        spec["anchor"] = anchor
        used_anchors.append(anchor)
        ar, ac = spec["anchor"]
        inp[ar, ac] = spec["marker_color"]

    for spec in specs:
        initial_top_left = _place_initial_sprite(inp, spec)
        if initial_top_left is None:
            return None
        spec["initial_top_left"] = initial_top_left
        for r, c, color in _sprite_cells(initial_top_left, spec):
            inp[r, c] = color

    expected = _expected_output(inp, specs)
    _validate_instance(inp, specs, expected)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-seal flood-fill instance")
