from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, in_bounds


DIRS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

ROT_DIR = {
    "U": "R",
    "R": "D",
    "D": "L",
    "L": "U",
}


def _rotate_coord(coord: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(int(turns) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_dir(direction: str, turns: int) -> str:
    out = direction
    for _ in range(int(turns) % 4):
        out = ROT_DIR[out]
    return out


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_cups(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    cups: list[dict] = []
    cup_colors: set[int] = set()

    for color in colors_present(g, exclude={bg}):
        comps = find_components(
            g,
            bg=bg,
            colors=[int(color)],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
        if len(comps) != 1:
            continue
        comp = comps[0]
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        ring = _perimeter_cells(comp["bbox"])
        if not coords.issubset(ring):
            continue
        if len(coords) != len(ring) - 1:
            continue

        missing = list(ring - coords)
        if len(missing) != 1:
            continue
        opening = missing[0]
        side_hits = [
            opening[0] == r0,
            opening[0] == r1,
            opening[1] == c0,
            opening[1] == c1,
        ]
        if sum(bool(x) for x in side_hits) != 1:
            continue

        if opening[0] == r0:
            side = "U"
        elif opening[0] == r1:
            side = "D"
        elif opening[1] == c0:
            side = "L"
        else:
            side = "R"

        cups.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "opening": (int(opening[0]), int(opening[1])),
                "side": side,
            }
        )
        cup_colors.add(int(color))

    other_colors = [c for c in colors_present(g, exclude={bg}) if int(c) not in cup_colors]
    assert len(cups) >= 2, "expected at least two cups"
    assert len(other_colors) == 1, "expected exactly one shared water color"
    water = int(other_colors[0])

    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        cup["water"] = int(np.count_nonzero(interior == water))

    return cups, water


def _downstream_indices(shape: tuple[int, int], cups: list[dict]) -> list[int | None]:
    opening_to_idx = {tuple(cup["opening"]): idx for idx, cup in enumerate(cups)}
    down: list[int | None] = []
    for idx, cup in enumerate(cups):
        dr, dc = DIRS[cup["side"]]
        rr = int(cup["opening"][0]) + dr
        cc = int(cup["opening"][1]) + dc
        target = None
        while in_bounds(shape, rr, cc):
            other = opening_to_idx.get((rr, cc))
            if other is not None and other != idx:
                target = int(other)
                break
            rr += dr
            cc += dc
        down.append(target)
    return down


def _accumulate_water(cups: list[dict], down: list[int | None]) -> list[int]:
    indeg = [0] * len(cups)
    for child in down:
        if child is not None:
            indeg[int(child)] += 1

    totals = [int(cup["water"]) for cup in cups]
    q = deque(idx for idx, deg in enumerate(indeg) if deg == 0)
    seen = 0
    while q:
        idx = q.popleft()
        seen += 1
        child = down[idx]
        if child is None:
            continue
        totals[int(child)] += totals[idx]
        indeg[int(child)] -= 1
        if indeg[int(child)] == 0:
            q.append(int(child))

    assert seen == len(cups), "cup transfer graph must be acyclic"
    return totals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear the water inside each gapped rectangular cup, then send each cup's water total along the opening-alignment chain so only terminal cups emit rays.

    Input structure:
    - The background is black.
    - Each cup is a unique-color rectangular outline with exactly one single-cell gap on its perimeter.
    - A single shared water color appears only inside cup interiors.

    Rule:
    - Detect every cup from its almost-closed rectangular border and recover its unique gap cell and facing side.
    - Treat a cup's downstream target as the first other cup opening reached by extending a straight ray from that gap in the cup's facing direction.
    - Erase all water pixels, conserve water counts along this directed acyclic opening graph, and for every cup with no downstream draw a water-colored ray from its opening outward whose length equals the total water accumulated there.

    Key insight:
    - The visible water inside a terminal cup is not enough; the ray length is determined only after reconstructing the opening-to-opening transfer chain and summing upstream water counts.
    """
    g = np.asarray(grid, dtype=int)
    cups, water = _detect_cups(g)
    down = _downstream_indices(g.shape, cups)
    totals = _accumulate_water(cups, down)

    out = g.copy()
    out[out == water] = Color.BLACK
    for idx, cup in enumerate(cups):
        if down[idx] is not None:
            continue
        dr, dc = DIRS[cup["side"]]
        rr, cc = map(int, cup["opening"])
        for _ in range(int(totals[idx])):
            assert in_bounds(out.shape, rr, cc), "terminal ray exceeded grid bounds"
            out[rr, cc] = water
            rr += dr
            cc += dc
    return out


TEMP_SHAPE = (50, 50)
TEMP_ORIGIN = (24, 24)
GEN_ATTEMPTS = 400


def _sample_template(rng: np.random.Generator) -> tuple[dict[str, tuple[int, int]], dict[str, str], dict[str, str | None]]:
    dx_left = int(rng.integers(5, 8))
    dx_right = int(rng.integers(5, 8))
    dy_up = int(rng.integers(5, 8))
    rel_pos: dict[str, tuple[int, int]] = {
        "B": (0, 0),
        "A": (0, -dx_left),
        "C": (0, dx_right),
        "D": (-dy_up, 0),
    }
    rel_side: dict[str, str] = {
        "B": "U",
        "A": "R",
        "C": "L",
        "D": "R",
    }
    expected: dict[str, str | None] = {
        "A": "B",
        "B": "D",
        "C": "B",
        "D": None,
    }

    if bool(rng.integers(0, 2)):
        extra_left = int(rng.integers(4, 7))
        rel_pos["E"] = (-dy_up, -extra_left)
        rel_side["E"] = "R"
        expected["E"] = "D"

    if bool(rng.integers(0, 2)):
        extra_down = int(rng.integers(4, 7))
        rel_pos["F"] = (extra_down, -dx_left)
        rel_side["F"] = "U"
        expected["F"] = "A"

    turns = int(rng.integers(0, 4))
    rot_pos = {name: _rotate_coord(coord, turns) for name, coord in rel_pos.items()}
    rot_side = {name: _rotate_dir(side, turns) for name, side in rel_side.items()}
    return rot_pos, rot_side, expected


def _sample_bbox(
    opening: tuple[int, int],
    side: str,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    r, c = map(int, opening)
    h = int(rng.integers(4, 6))
    w = int(rng.integers(4, 6))
    if side == "U":
        offset = int(rng.integers(1, w - 1))
        return r, c - offset, r + h - 1, c - offset + w - 1
    if side == "D":
        offset = int(rng.integers(1, w - 1))
        return r - h + 1, c - offset, r, c - offset + w - 1
    if side == "L":
        offset = int(rng.integers(1, h - 1))
        return r - offset, c, r - offset + h - 1, c + w - 1
    offset = int(rng.integers(1, h - 1))
    return r - offset, c - w + 1, r - offset + h - 1, c


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _build_specs(rng: np.random.Generator) -> list[dict] | None:
    rel_pos, rel_side, _expected = _sample_template(rng)
    names = sorted(rel_pos)
    abs_openings = {
        name: (TEMP_ORIGIN[0] + rel_pos[name][0], TEMP_ORIGIN[1] + rel_pos[name][1])
        for name in names
    }

    specs: list[dict] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for name in names:
        bbox = _sample_bbox(abs_openings[name], rel_side[name], rng)
        r0, c0, r1, c1 = map(int, bbox)
        if not (1 <= r0 < r1 < TEMP_SHAPE[0] - 1 and 1 <= c0 < c1 < TEMP_SHAPE[1] - 1):
            return None
        if any(_boxes_overlap(bbox, other, pad=1) for other in bboxes):
            return None
        bboxes.append(bbox)
        specs.append(
            {
                "name": name,
                "opening": abs_openings[name],
                "side": rel_side[name],
                "bbox": bbox,
            }
        )

    palette = list(rng.choice(np.arange(1, 10), size=len(specs) + 1, replace=False))
    water_color = int(palette[0])
    cup_colors = palette[1:]
    for spec, color in zip(specs, cup_colors):
        spec["color"] = int(color)
        r0, c0, r1, c1 = spec["bbox"]
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        max_water = min(3, len(interior))
        water_count = int(rng.integers(1, max_water + 1))
        pick = rng.choice(len(interior), size=water_count, replace=False)
        spec["water_cells"] = [interior[int(idx)] for idx in np.atleast_1d(pick)]
        spec["water"] = water_count

    expected_by_name = {
        "A": "B",
        "B": "D",
        "C": "B",
        "D": None,
    }
    if "E" in names:
        expected_by_name["E"] = "D"
    if "F" in names:
        expected_by_name["F"] = "A"

    name_to_idx = {spec["name"]: idx for idx, spec in enumerate(specs)}
    down = _downstream_indices(TEMP_SHAPE, specs)
    for spec, child_idx in zip(specs, down):
        want = expected_by_name[spec["name"]]
        got = None if child_idx is None else specs[int(child_idx)]["name"]
        if got != want:
            return None

    totals = _accumulate_water(specs, down)
    term_totals = [totals[idx] for idx, child in enumerate(down) if child is None]
    if not term_totals or max(term_totals) > 10 or min(term_totals) < 4:
        return None

    for idx, child in enumerate(down):
        if child is not None:
            continue
        dr, dc = DIRS[specs[idx]["side"]]
        rr, cc = map(int, specs[idx]["opening"])
        for _ in range(int(totals[idx])):
            if not in_bounds(TEMP_SHAPE, rr, cc):
                return None
            for j, other in enumerate(specs):
                if j == idx and (rr, cc) == tuple(other["opening"]):
                    continue
                if (rr, cc) in _perimeter_cells(other["bbox"]):
                    return None
            rr += dr
            cc += dc

    specs.append({"water_color": water_color})
    return specs


def _render_input(specs: list[dict]) -> np.ndarray:
    grid = np.full(TEMP_SHAPE, Color.BLACK, dtype=int)
    water = int(specs[-1]["water_color"])
    for spec in specs[:-1]:
        r0, c0, r1, c1 = spec["bbox"]
        for r, c in _perimeter_cells(spec["bbox"]):
            if (r, c) != tuple(spec["opening"]):
                grid[r, c] = int(spec["color"])
        for r, c in spec["water_cells"]:
            grid[int(r), int(c)] = water
    return grid


def _simulate_output(specs: list[dict], inp: np.ndarray) -> np.ndarray:
    cups = [dict(spec) for spec in specs[:-1]]
    water = int(specs[-1]["water_color"])
    down = _downstream_indices(inp.shape, cups)
    totals = _accumulate_water(cups, down)

    out = inp.copy()
    out[out == water] = Color.BLACK
    for idx, cup in enumerate(cups):
        if down[idx] is not None:
            continue
        dr, dc = DIRS[cup["side"]]
        rr, cc = map(int, cup["opening"])
        for _ in range(int(totals[idx])):
            out[rr, cc] = water
            rr += dr
            cc += dc
    return out


def _required_crop(inp: np.ndarray, out: np.ndarray, rng: np.random.Generator) -> tuple[slice, slice] | None:
    pts = np.argwhere((inp != Color.BLACK) | (out != Color.BLACK))
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)

    top = int(rng.integers(0, 3))
    left = int(rng.integers(0, 3))
    bottom = int(rng.integers(0, 3))
    right = int(rng.integers(0, 3))

    r0 = max(0, int(r0) - top)
    c0 = max(0, int(c0) - left)
    r1 = min(TEMP_SHAPE[0] - 1, int(r1) + bottom)
    c1 = min(TEMP_SHAPE[1] - 1, int(c1) + right)
    if r1 - r0 + 1 > 30 or c1 - c0 + 1 > 30:
        return None
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        specs = _build_specs(rng)
        if specs is None:
            continue
        inp_full = _render_input(specs)
        out_full = _simulate_output(specs, inp_full)
        crop = _required_crop(inp_full, out_full, rng)
        if crop is None:
            continue
        rs, cs = crop
        inp = inp_full[rs, cs].copy()
        out = out_full[rs, cs].copy()
        try:
            got = main(inp)
        except AssertionError:
            continue
        if np.array_equal(got, out):
            return inp
    raise RuntimeError("failed to generate a valid cascading-cups input")
