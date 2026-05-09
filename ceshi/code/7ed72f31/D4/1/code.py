# Variant of 7ed72f31
# Dimension: D4 — 标记类型的判定逻辑 (transform)
# Change: 红像素数=1→点反射；同行多个→水平轴反射；同列多个→竖直轴反射... → 红像素数量编码变换参数

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location

MARK = 2
ROT_BY_RED_COUNT = {1: 1, 2: 2, 3: 3}  # clockwise quarter-turns


"""
Design analysis for the D4 variant

Reference-task note
- The repository does not contain a readable standalone solver file for source task
  `6ffbe589`. The available annotation states its key mechanism as
  `np.rot90(d, -s.size)`, i.e. object size encodes clockwise rotation count.
  This variant adopts that same abstract mechanism: red-marker count encodes the
  clockwise rotation angle.

a) Concretization + trace
- Original 7ed72f31:
  a red line inside a mixed sprite means mirror across that line; one red pixel
  means point-reflect around that pixel.
- Variant:
  every actionable group is a mixed 4-connected component made of one monochrome
  payload sprite plus 1/2/3 red pixels. The number of red pixels no longer says
  "row/col/point". Instead it says how many clockwise quarter-turns to apply to
  the payload around a pivot red cell:
    1 red  -> 90 degrees clockwise
    2 reds -> 180 degrees
    3 reds -> 270 degrees clockwise
- The pivot red cell is the unique red cell that is 4-neighbor-adjacent to the
  payload. Extra red pixels extend away from that pivot and act only as a count
  code.
- Concrete trace:
  input local coords (canonical form) may contain payload cells at
  {(-1,0), (-2,0), (-2,1), (-3,1)} above the pivot (0,0).
  If reds are {(0,0), (0,1), (0,2)}, red-count = 3, so rotate payload 270 deg cw:
    (-1,0) -> (0,-1)
    (-2,0) -> (0,-2)
    (-2,1) -> (-1,-2)
    (-3,1) -> (-1,-3)
  The original payload stays; the rotated copy is painted onto background cells.
  Under the original solver this same component would have been treated as a
  horizontal reflection axis because reds share one row; that is precisely the
  behavior being replaced.

b) Complexity calibration
- The count-to-angle code has only three values, which is small enough for humans
  to infer from 2-4 examples, but abstract enough to differ materially from the
  original geometry-driven rule.
- To make the change visible, payload sprites are forced to be rotationally
  asymmetric and to occupy at least a 2x2 bbox. Purely symmetric blobs would make
  90/180/270 branches visually collapse.
- Training-style support is built into each generated sample by including all three
  red counts somewhere in the same grid, with one or more repeated counts.

c) Implicit-assumption audit against the original main()
- Original `len(reds)==1 / len(rs)==1 / len(cs)==1` assumed red geometry directly
  determines transform type. This is false in the variant and must be removed.
- Original `refl_row / refl_col / refl_point` assumed the transform center/axis is
  recoverable from row/col coincidence. Variant instead needs a general rotation
  around a pivot red cell.
- Original bbox/color eligibility scanned the whole grid for same-color pixels
  near the component. That was relatively harmless for reflections, but for 90/270
  rotations it can drag unrelated nearby same-color pixels into remote rotated
  positions. Variant therefore restricts the transform source to the payload cells
  of the marker's own connected component.
- Original isolated-red branch assumed a single red pixel. Variant can still
  support red-only marker components in principle, but the generator keeps the
  clearer mixed-component form so the core count code is unmistakable.

d) Placement legality
- Generation remains output-first. For each group we first construct the full
  output patch = payload + red code + rotated copy, then place that whole patch on
  empty background using `random_free_location`. The input patch is the same patch
  with the rotated copy removed.
- This guarantees:
  no overlap between transformed pixels and existing foreground,
  no out-of-bounds placements,
  and exact compatibility with the solver's local-component logic.
- The canonical local design puts payload strictly on one side of the pivot and
  keeps only one payload cell adjacent to the red code, so the pivot is unique.

e) Visual observability
- A human viewer can compare otherwise similar groups with red-count 1/2/3 and see
  the added copy appear after a quarter-turn, half-turn, or three-quarter-turn.
- Because base orientation of the whole group is also randomized, the rule cannot
  be reduced to a fixed global direction like "1 red means copy on the right".
  The invariant is the amount of rotation around the contact red pixel.

f) Visual cleanliness
- No free same-color distractors are generated.
- Groups are spaced with padding and each uses a distinct payload color.
- Red code clusters are short and contiguous, so the cue remains easy to isolate.

g) Color-change inferability
- Colors do not encode behavior. Each group keeps a single non-red payload color
  in both input and output. The only semantic signal is the number of red pixels.
"""


def _rot_rel(coord: tuple[int, int], quarters: int) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(int(quarters) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_about(pivot: tuple[int, int], coord: tuple[int, int], quarters: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    r, c = map(int, coord)
    dr, dc = int(r - pr), int(c - pc)
    rr, cc = _rot_rel((dr, dc), quarters)
    return int(pr + rr), int(pc + cc)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    lab4, n4 = label_mask(g != bg, connectivity=4)
    nonred8, _ = label_mask((g != bg) & (g != MARK), connectivity=8)

    comps: list[dict[str, object]] = []
    for lbl in range(1, int(n4) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(lbl))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        if not reds or len(reds) not in ROT_BY_RED_COUNT:
            continue

        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != MARK]
        if not payload:
            continue

        payload_set = {(int(r), int(c)) for r, c, _v in payload}
        pivot_candidates: list[tuple[int, int]] = []
        for rr, cc in reds:
            if any((nr, nc) in payload_set for nr, nc in _neighbors4(rr, cc)):
                pivot_candidates.append((int(rr), int(cc)))
        if len(pivot_candidates) != 1:
            continue

        pivot = pivot_candidates[0]
        quarters = ROT_BY_RED_COUNT[len(reds)]
        for r, c, v in payload:
            rr, cc = _rotate_about(pivot, (int(r), int(c)), quarters)
            if inb(rr, cc):
                out[int(rr), int(cc)] = int(v)

    # Retain a compatible red-only branch, now also using red-count -> rotation.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if any(int(v) != MARK for v in vals):
            continue
        if len(coords) not in ROT_BY_RED_COUNT:
            continue

        reds = [(int(r), int(c)) for r, c in coords]
        touched_labels: set[int] = set()
        pivot_candidates: list[tuple[int, int]] = []
        for rr, cc in reds:
            here = {int(nonred8[nr, nc]) for nr, nc in _neighbors8(rr, cc) if inb(nr, nc) and int(nonred8[nr, nc]) != 0}
            if here:
                pivot_candidates.append((int(rr), int(cc)))
                touched_labels |= here
        if len(pivot_candidates) != 1 or len(touched_labels) != 1:
            continue

        pivot = pivot_candidates[0]
        lbl = next(iter(touched_labels))
        quarters = ROT_BY_RED_COUNT[len(reds)]
        for r, c in np.argwhere(nonred8 == int(lbl)):
            rr, cc = _rotate_about(pivot, (int(r), int(c)), quarters)
            if inb(rr, cc):
                out[int(rr), int(cc)] = int(g[int(r), int(c)])

    return out


def _render_cells(
    cells: dict[tuple[int, int], int],
    bg: int,
    bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    coords = list(cells)
    if bounds is None:
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        r0, c0 = min(rs), min(cs)
        r1, c1 = max(rs), max(cs)
    else:
        r0, c0, r1, c1 = map(int, bounds)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.full((h, w), int(bg), dtype=int)
    for (r, c), v in cells.items():
        out[int(r - r0), int(c - c0)] = int(v)
    return out


def _random_payload(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(400):
        size = rng.randint(4, 7)
        up = rng.randint(3, 5)
        left = rng.randint(0, 2)
        right = rng.randint(1, 3)
        cells: set[tuple[int, int]] = {(-1, 0)}

        while len(cells) < size:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if nr >= 0 or nr < -up:
                        continue
                    if nc < -left or nc > right:
                        continue
                    if nr == -1 and nc != 0:
                        continue
                    if (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != size:
            continue
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if sum(1 for r, _c in cells if r == -1) != 1:
            continue
        if any({_rot_rel(cell, q) for cell in cells} == cells for q in (1, 2, 3)):
            continue
        valid = True
        for count, quarters in ROT_BY_RED_COUNT.items():
            reds = {(0, i) for i in range(count)}
            rotated = {_rot_rel(cell, quarters) for cell in cells}
            if rotated & cells or rotated & reds:
                valid = False
                break
        if valid:
            return cells
    raise RuntimeError("failed to build asymmetric payload")


def _build_group(rng: random.Random, bg: int, color: int, red_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    payload = _random_payload(rng)
    reds = {(0, i) for i in range(int(red_count))}
    quarters = ROT_BY_RED_COUNT[int(red_count)]
    rotated = {_rot_rel(cell, quarters) for cell in payload}
    base_rot = rng.randrange(4)

    payload = {_rot_rel(cell, base_rot) for cell in payload}
    reds = {_rot_rel(cell, base_rot) for cell in reds}
    rotated = {_rot_rel(cell, base_rot) for cell in rotated}

    input_cells: dict[tuple[int, int], int] = {tuple(cell): int(color) for cell in payload}
    for cell in reds:
        input_cells[tuple(cell)] = MARK

    output_cells = dict(input_cells)
    for cell in rotated:
        output_cells[tuple(cell)] = int(color)

    all_coords = list(output_cells)
    rs = [r for r, _c in all_coords]
    cs = [c for _r, c in all_coords]
    bounds = (min(rs), min(cs), max(rs), max(cs))
    input_patch = _render_cells(input_cells, bg, bounds=bounds)
    output_patch = _render_cells(output_cells, bg, bounds=bounds)
    meta = {"red_count": int(red_count), "angle_deg": int(90 * quarters), "base_rot": int(base_rot)}
    return input_patch, output_patch, meta


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    rng = rng or random.Random()
    palette = list(range(10))

    for _attempt in range(200):
        H = rng.randint(18, 28)
        W = rng.randint(18, 28)
        bg = rng.choice([c for c in palette if c != MARK])
        input_grid = np.full((H, W), int(bg), dtype=int)
        output_grid = np.full((H, W), int(bg), dtype=int)

        n_groups = rng.randint(4, 6)
        counts = [1, 2, 3]
        while len(counts) < n_groups:
            counts.append(rng.choice([1, 2, 3]))
        rng.shuffle(counts)

        colors = rng.sample([c for c in palette if c not in (bg, MARK)], n_groups)
        meta: list[dict[str, int]] = []
        ok = True

        for color, red_count in zip(colors, counts):
            try:
                input_patch, output_patch, group_meta = _build_group(rng, bg, color, red_count)
                r0, c0 = random_free_location(
                    output_grid,
                    output_patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break

            oh, ow = output_patch.shape
            ih, iw = input_patch.shape
            out_mask = output_patch != int(bg)
            in_mask = input_patch != int(bg)
            output_grid[r0 : r0 + oh, c0 : c0 + ow][out_mask] = output_patch[out_mask]
            input_grid[r0 : r0 + ih, c0 : c0 + iw][in_mask] = input_patch[in_mask]
            meta.append(group_meta)

        if not ok:
            continue
        solved = main(input_grid)
        if not np.array_equal(solved, output_grid):
            continue
        return input_grid, output_grid, meta

    raise RuntimeError("failed to generate a valid variant instance")


def generate_input() -> np.ndarray:
    grid, expected, _meta = generate_case()
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise AssertionError("generator self-check failed")
    return grid
