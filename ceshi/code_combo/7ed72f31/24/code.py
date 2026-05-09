# Combo variant of 7ed72f31 (index 24)
# combo_spec: D4/1+D10/1
# Name: 双重标记编码

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
PALETTE = list(range(10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RAY_PATTERNS: dict[tuple[int, str], tuple[int, ...]] = {
    (1, "solid"): (0,),
    (2, "solid"): (0, 1),
    (2, "gap"): (0, 2),
    (3, "solid"): (0, 1, 2),
    (3, "gap"): (0, 2, 3),
}
PAYLOAD_TEMPLATES: list[set[tuple[int, int]]] = [
    {(-1, 0), (-2, 0), (-2, 1), (-3, 1)},
    {(-1, 0), (-2, 0), (-2, 1), (-2, 2), (-3, 2)},
    {(-1, 0), (-2, 0), (-3, 0), (-3, 1), (-4, 1)},
    {(-1, 0), (-2, 0), (-2, -1), (-3, -1), (-3, 0)},
    {(-1, 0), (-1, -1), (-2, -1), (-3, -1)},
]


"""
Design analysis for combo variant D4/1 + D10/1

a) Concretization + trace
- Original 7ed72f31 used red geometry directly:
  one red pixel => point reflection, same-row reds => horizontal mirror,
  same-col reds => vertical mirror.
- Fused variant replaces that with a two-layer code carried by the same red marker:
  red-count encodes the rotation angle, while solid-vs-gapped layout encodes the
  rotation direction.
  1 red  => 90 degrees
  2 reds => 180 degrees
  3 reds => 270 degrees
  solid  => clockwise
  gap    => counter-clockwise
- Concrete local example in canonical coordinates:
  pivot red at (0,0), extra reds at (0,2),(0,3), so the marker is "3 reds with a gap".
  Payload cells are {(-1,0), (-2,0), (-2,1), (-3,1)}.
  Original solver would read "multiple reds on one row" and try a horizontal mirror.
  New solver reads:
    count = 3 -> rotate by 270 degrees,
    gap   = yes -> use the opposite direction, so 270 degrees counter-clockwise
                   which is equivalent to 90 degrees clockwise.
  The payload therefore rotates around the pivot rather than reflecting across a line.
- The important difference is that no single geometric cue is sufficient anymore:
  the solver must decode quantity and pattern jointly.

b) Complexity calibration
- The code space is intentionally small: counts {1,2,3} and patterns {solid,gap},
  with the generator using all three counts and both patterns in every sample.
- Direction is only visually effective for 270-degree groups, because 180 degrees is
  direction-invariant and 1-red groups cannot show a gap. To keep the second code
  channel observable, every generated sample includes both `3-solid` and `3-gap`.
- Payloads are asymmetric and non-trivial, so 90/180/270 outcomes look different.

c) Implicit-assumption audit against the original main()
- Original assumption: red pixels and payload are one 4-connected mixed component.
  This breaks once a marker may contain an internal gap. The fused solver therefore
  detects markers from red rays anchored at a unique pivot red adjacent to payload.
- Original assumption: red geometry alone chooses the transform family. False here;
  row/col coincidence is used only to read the code layout, not to decide mirror type.
- Original assumption: scanning same-colored pixels in a bbox is safe. False here;
  rotation must use only the monochrome payload component touching the pivot.
- Original assumption: isolated single red pixels point-reflect nearby sprites.
  This conflicts with the new meaning "1 red = 90-degree rotation", so the isolated
  branch is intentionally removed from generation and solver semantics.

d) Placement legality
- Generation is output-first. Each local patch is built as
  input = payload + red code
  output = input + rotated copy
  and then placed with `random_free_location` on the output canvas.
- Because the full output patch is placed first, transformed copies cannot overlap
  existing foreground or leave the grid. Input is derived by erasing only the copy.
- Groups are padded apart, preventing accidental red/payload associations across groups.

e) Visual observability
- The human-visible clues are:
  red-count differs between groups,
  some 3-red codes are contiguous while others have a single hole,
  output copies appear after quarter-turns around the touching red endpoint.
- The generator always includes a `3-solid` and a `3-gap` group so the pattern code
  visibly flips the landing side while keeping the count constant.

f) Visual cleanliness
- One payload color per group, no distractor sprites, and one-cell padding between
  patches keep the grid legible.
- The pivot is always the only red cell touching payload, so the relevant red endpoint
  is visually unambiguous.

g) Change inferability
- No color mapping is introduced. Color is purely object identity.
- The only semantics come from red-count and red-pattern, which are both directly
  visible in the input and repeatedly supported inside each generated sample.

New solver assumptions introduced here
- Each actionable group has exactly one pivot red with a 4-neighbor payload contact.
- Payload is a single 4-connected monochrome component.
- The red code extends from the pivot in exactly one cardinal direction and matches one
  of the canonical patterns listed in `RAY_PATTERNS`.
- `generate_input()` explicitly constructs patches that satisfy those assumptions.
"""


def _inb(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _rot_rel(coord: tuple[int, int], quarters: int) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(int(quarters) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_about(pivot: tuple[int, int], coord: tuple[int, int], quarters: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    r, c = map(int, coord)
    rr, cc = _rot_rel((int(r - pr), int(c - pc)), quarters)
    return int(pr + rr), int(pc + cc)


def _quarters(count: int, pattern: str, *, ignore_gap: bool = False) -> int:
    if pattern == "gap" and not bool(ignore_gap):
        return int((-int(count)) % 4)
    return int(count % 4)


def _ray_positions(count: int, pattern: str) -> tuple[int, ...]:
    key = (int(count), str(pattern))
    if key not in RAY_PATTERNS:
        raise ValueError(f"unsupported marker code: {key}")
    return RAY_PATTERNS[key]


def _decode_marker_ray(pivot: tuple[int, int], red_set: set[tuple[int, int]]) -> tuple[int, str, tuple[int, int]] | None:
    pr, pc = map(int, pivot)
    matches: list[tuple[int, str, tuple[int, int]]] = []
    for dr, dc in DIRS4:
        occ = tuple((int(pr + dr * k), int(pc + dc * k)) in red_set for k in range(4))
        if occ == (True, False, True, True):
            matches.append((3, "gap", (int(dr), int(dc))))
        elif occ == (True, False, True, False):
            matches.append((2, "gap", (int(dr), int(dc))))
        elif occ == (True, True, True, False):
            matches.append((3, "solid", (int(dr), int(dc))))
        elif occ == (True, True, False, False):
            matches.append((2, "solid", (int(dr), int(dc))))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None
    if all(not (int(pr + dr), int(pc + dc)) in red_set for dr, dc in DIRS4):
        if all(not (int(pr + 2 * dr), int(pc + 2 * dc)) in red_set for dr, dc in DIRS4):
            if all(not (int(pr + 3 * dr), int(pc + 3 * dc)) in red_set for dr, dc in DIRS4):
                return 1, "solid", (0, 0)
    return None


def _extract_groups(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    red_set = {(int(r), int(c)) for r, c in np.argwhere(g == int(MARK))}
    color_cache: dict[int, np.ndarray] = {}
    used_payloads: set[tuple[int, int]] = set()
    groups: list[dict[str, object]] = []

    for pr, pc in sorted(red_set):
        neigh = []
        for nr, nc in _neighbors4(int(pr), int(pc)):
            if not _inb(H, W, nr, nc):
                continue
            v = int(g[int(nr), int(nc)])
            if v in (int(bg), int(MARK)):
                continue
            neigh.append((int(nr), int(nc), int(v)))
        if not neigh:
            continue

        colors = {int(v) for _r, _c, v in neigh}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        decoded = _decode_marker_ray((int(pr), int(pc)), red_set)
        if decoded is None:
            continue
        count, pattern, ray_dir = decoded

        if color not in color_cache:
            lab, _ = label_mask(g == int(color), connectivity=4)
            color_cache[int(color)] = lab
        lab = color_cache[int(color)]
        touched = {int(lab[int(r), int(c)]) for r, c, _v in neigh if int(lab[int(r), int(c)]) != 0}
        if len(touched) != 1:
            continue
        lbl = next(iter(touched))
        key = (int(color), int(lbl))
        if key in used_payloads:
            continue
        payload = [(int(r), int(c), int(color)) for r, c in np.argwhere(lab == int(lbl))]
        if not payload:
            continue
        used_payloads.add(key)
        groups.append(
            {
                "pivot": (int(pr), int(pc)),
                "count": int(count),
                "pattern": str(pattern),
                "ray_dir": (int(ray_dir[0]), int(ray_dir[1])),
                "payload": payload,
                "color": int(color),
            }
        )
    return groups


def _solve(grid: np.ndarray, *, ignore_gap: bool = False) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for group in _extract_groups(g):
        pivot = group["pivot"]
        count = int(group["count"])
        pattern = str(group["pattern"])
        payload = group["payload"]
        quarters = _quarters(int(count), str(pattern), ignore_gap=bool(ignore_gap))
        for r, c, v in payload:
            rr, cc = _rotate_about(pivot, (int(r), int(c)), quarters)
            if _inb(H, W, rr, cc):
                out[int(rr), int(cc)] = int(v)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, ignore_gap=False)


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
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for (r, c), v in cells.items():
        out[int(r - r0), int(c - c0)] = int(v)
    return out


def _touch_count(cell: tuple[int, int], others: set[tuple[int, int]]) -> int:
    r, c = map(int, cell)
    return sum(1 for nr, nc in _neighbors4(int(r), int(c)) if (int(nr), int(nc)) in others)


def _build_group(
    rng: random.Random,
    bg: int,
    color: int,
    count: int,
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    ray = _ray_positions(int(count), str(pattern))
    quarters = _quarters(int(count), str(pattern))

    for _attempt in range(400):
        template = set(rng.choice(PAYLOAD_TEMPLATES))
        base_rot = int(rng.randrange(4))
        payload = {_rot_rel(cell, int(base_rot)) for cell in template}
        reds = {_rot_rel((0, step), int(base_rot)) for step in ray}
        pivot = (0, 0)
        rotated = {_rotate_about(pivot, cell, quarters) for cell in payload}

        if payload & reds:
            continue
        if rotated & payload:
            continue
        if rotated & reds:
            continue
        if _touch_count(pivot, payload) != 1:
            continue
        if any(_touch_count(red, payload) > 0 for red in reds if red != pivot):
            continue

        input_cells: dict[tuple[int, int], int] = {tuple(cell): int(color) for cell in payload}
        for cell in reds:
            input_cells[tuple(cell)] = int(MARK)
        output_cells = dict(input_cells)
        for cell in rotated:
            output_cells[tuple(cell)] = int(color)

        all_coords = list(output_cells)
        rs = [r for r, _c in all_coords]
        cs = [c for _r, c in all_coords]
        bounds = (min(rs), min(cs), max(rs), max(cs))
        input_patch = _render_cells(input_cells, bg, bounds)
        output_patch = _render_cells(output_cells, bg, bounds)
        meta = {
            "count": int(count),
            "pattern": str(pattern),
            "quarters": int(quarters),
            "base_rot": int(base_rot),
        }
        return input_patch, output_patch, meta

    raise RuntimeError(f"failed to build group for count={count}, pattern={pattern}")


def _stamp(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != int(bg)
    dst[int(top) : int(top) + int(h), int(left) : int(left) + int(w)][mask] = patch[mask]


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    groups = _extract_groups(np.array(grid, dtype=int))
    out = {
        "groups": len(groups),
        "count1": 0,
        "count2": 0,
        "count3": 0,
        "solid": 0,
        "gap": 0,
        "count3_solid": 0,
        "count3_gap": 0,
    }
    for group in groups:
        count = int(group["count"])
        pattern = str(group["pattern"])
        out[f"count{count}"] += 1
        out[str(pattern)] += 1
        if count == 3:
            out[f"count3_{pattern}"] += 1
    return out


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = rng or random.Random()

    for _attempt in range(300):
        H = int(rng.randint(20, 28))
        W = int(rng.randint(20, 28))
        bg = int(rng.choice([c for c in PALETTE if c != MARK]))
        input_grid = np.full((H, W), int(bg), dtype=int)
        output_grid = np.full((H, W), int(bg), dtype=int)

        specs = [(1, "solid"), (2, "solid"), (3, "solid"), (3, "gap")]
        if rng.random() < 0.7:
            specs.append(rng.choice([(1, "solid"), (2, "solid"), (3, "solid"), (3, "gap")]))
        rng.shuffle(specs)

        colors = rng.sample([c for c in PALETTE if c not in (bg, MARK)], len(specs))
        meta: list[dict[str, object]] = []
        ok = True

        for color, (count, pattern) in zip(colors, specs):
            try:
                pinp, pout, group_meta = _build_group(rng, bg, int(color), int(count), str(pattern))
                top, left = random_free_location(
                    output_grid,
                    pout,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            except RuntimeError:
                ok = False
                break

            _stamp(input_grid, pinp, int(top), int(left), int(bg))
            _stamp(output_grid, pout, int(top), int(left), int(bg))
            meta.append(group_meta)

        if not ok:
            continue

        metrics = _variant_metrics(input_grid)
        if metrics["groups"] != len(specs):
            continue
        if metrics["count1"] < 1 or metrics["count2"] < 1 or metrics["count3"] < 2:
            continue
        if metrics["solid"] < 3 or metrics["gap"] < 1:
            continue
        if metrics["count3_solid"] < 1 or metrics["count3_gap"] < 1:
            continue

        solved = main(input_grid)
        if not np.array_equal(solved, output_grid):
            continue
        if np.array_equal(_solve(input_grid, ignore_gap=True), output_grid):
            continue
        return input_grid, output_grid, meta

    raise RuntimeError("failed to generate a valid combo variant instance")


def generate_input() -> np.ndarray:
    grid, expected, _meta = generate_case()
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise AssertionError("generator self-check failed")
    if np.array_equal(_solve(grid, ignore_gap=True), expected):
        raise AssertionError("gap code had no observable effect")
    return grid
