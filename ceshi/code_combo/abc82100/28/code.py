# Combo variant of abc82100 (index 28)
# combo_spec: D4/3+D6/2+D14/0
# Name: 模板缩放+渐变着色+几何中心锚点

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace:
# - Base task: a local legend teaches "source color -> target color -> nearby teal
#   stencil", then each scattered source pixel receives one flat-colored copy of
#   that stencil with the legend-implied anchor aligned to the source pixel.
# - Combo task: the legend still teaches the source/target color pairing and which
#   teal component is the prototype, but three coupled changes happen at stamping
#   time:
#   1. Each transformable source instance is the NW head of a SE diagonal run of
#      identical source-colored pixels; run length encodes the scale factor.
#   2. The copied stencil is aligned by the geometric center of its bounding box,
#      not by the legend continuation anchor.
#   3. Coloring is recomputed after scaling from Manhattan distance to that center:
#      distance 0 -> target color, distance 1 -> middle color, distance >= 2 ->
#      outer color.
# - Concrete trace:
#   Suppose the teal prototype occupies a connected 3x3-bbox shape containing the
#   center cell. In the legend, the forward anchor may touch the prototype at its
#   top edge, not at its center. A scattered source run of length 3 at head (9,9)
#   therefore yields a 3x-scaled stamp whose 9x9 bounding box is centered on
#   (9,9); cells nearest that center keep the target hue while farther cells use
#   lighter gradient colors. If one incorrectly used the legend-touch point as the
#   placement anchor, the whole stamp would be shifted and the gradient origin
#   would move with it.
# b) Complexity calibration:
# - Scale choices are restricted to odd factors {1, 3}. This keeps geometric
#   centers on integer grid cells and makes the size contrast obvious without
#   producing excessive placement failures.
# - Every generated rule gets both a scale-1 and a scale-3 instance, so the
#   scaling effect is repeated within the same sample instead of being a singleton.
# - Prototypes are compact 3x3-bbox connected shapes with at least one distance-1
#   and one distance-2 cell from the center, ensuring the gradient never collapses
#   into a single color and that scaling genuinely changes the distance layout.
# c) Hidden-assumption audit against the original solver:
# - Original `_discover_rules(...)` assumed the continued legend direction reveals
#   the stamping anchor. That is no longer true. The continued direction is now
#   used only to identify which teal component belongs to the color pair; the real
#   stamping anchor is recomputed from the component's bounding-box center.
# - Original stamping treated every source-colored pixel as an independent source.
#   That breaks under scale encoding because diagonal marker tails share the same
#   color. The new solver must only act on NW-most diagonal heads.
# - Original recoloring was uniform per stamp. That breaks under center-distance
#   gradients because scaling changes which cells fall into distance buckets 0/1/2.
#   The new solver must therefore recolor each stamped cell after expansion.
# - New implicit assumption introduced here: scaled stamps are center-aligned on an
#   integer cell. The generator guarantees this by using odd bbox dimensions and
#   odd scale factors only.
# d) Placement legality:
# - Scattered runs are placed first. Each run must fit entirely in bounds, its full
#   scaled output footprint must fit in bounds, and footprints must stay disjoint
#   from one another.
# - Input objects are kept away from already-reserved output footprints so legends,
#   bystanders, and unrelated runs never occupy regions that will later be used by
#   another stamp.
# - Legend placement chooses a contact point on the prototype that is explicitly
#   not the geometric center, preventing accidental collapse back to the source
#   task's anchor semantics.
# e) Visual observability:
# - A solver can see the same source color appearing as both a short and a long
#   diagonal chain, and the output shows the same target-colored motif appearing
#   small and large.
# - Because the legend-touch point is offset from the prototype center, the output
#   center alignment is not explainable by simply following the legend arrow. This
#   makes the D14 change visible rather than latent.
# - The gradient uses a fixed visible palette per target color rather than numeric
#   color arithmetic, so the mapping is inferable from examples directly.
# f) Visual cleanliness:
# - Output footprints are kept disjoint, legends are placed outside all future
#   stamped regions, and bystanders use colors that do not appear in gradients.
#   This avoids mixed-color overlaps that would obscure the center-based gradient.
# g) Change inferability:
# - The three dimensions reinforce one another: scale enlarges the same prototype,
#   center alignment decides where the enlarged copy is positioned, and the center
#   also defines the gradient origin. If any one of those pieces were removed, the
#   produced image would visibly change, which is the intended non-degenerate combo.
# main()/generator contract:
# - Every learned prototype has a 3x3 bounding box with a center cell present.
# - Every transformable instance is a NW-most SE-diagonal run of its rule's source
#   color, with odd run length in {1, 3}.
# - Every legend's continued anchor touches the correct teal component but is not
#   the component's geometric center.


TARGET_GRADIENTS: dict[int, tuple[int, int, int]] = {
    1: (1, 8, 4),
    2: (2, 7, 4),
    3: (3, 8, 4),
    9: (9, 2, 7),
}
TARGET_COLORS = list(TARGET_GRADIENTS.keys())
SOURCE_COLORS = [4, 5, 6, 7]
CLEAN_BYSTANDER_COLORS = [5, 6]

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 10, 2: 13, 3: 16, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

PROTOTYPE_MIN_CELLS = 4
PROTOTYPE_MAX_CELLS = 5
SCALE_CHOICES = [1, 3]
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 4
SCATTERED_AREA_DIV = 70

BYSTANDER_MAX = 3

LEGEND_RETRIES = 900
SCATTERED_RETRIES_PER_INSTANCE = 200
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _gradient_color(target_color: int, distance: int) -> int:
    palette = TARGET_GRADIENTS[int(target_color)]
    return int(palette[min(max(int(distance), 0), 2)])


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _touches_region(cells: set[tuple[int, int]], anchor: tuple[int, int], radius: int = 1) -> bool:
    ar, ac = int(anchor[0]), int(anchor[1])
    for r, c in cells:
        if max(abs(int(r) - ar), abs(int(c) - ac)) <= int(radius):
            return True
    return False


def _pattern_bbox(pattern: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(pattern[:, 0].min())
    c0 = int(pattern[:, 1].min())
    r1 = int(pattern[:, 0].max())
    c1 = int(pattern[:, 1].max())
    return r0, c0, r1, c1


def _normalize_pattern(pattern: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _pattern_bbox(pattern)
    local = pattern - np.array([[r0, c0]], dtype=int)
    center = ((r1 - r0) // 2, (c1 - c0) // 2)
    return np.asarray(local, dtype=int), (int(center[0]), int(center[1]))


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), np.asarray(pattern, dtype=int)))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)

                local_pattern, center_local = _normalize_pattern(pattern)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "legend_anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": pattern,
                        "local_pattern": local_pattern,
                        "bbox_shape": (
                            int(local_pattern[:, 0].max()) + 1,
                            int(local_pattern[:, 1].max()) + 1,
                        ),
                        "center_local": center_local,
                    }
                )

    return rules


def _diagonal_scale_head(grid: np.ndarray, r: int, c: int, color: int) -> int:
    """Return the SE-diagonal run length if (r,c) is the NW-most head, else 0."""
    h, w = grid.shape
    if int(grid[r, c]) != int(color):
        return 0
    if r > 0 and c > 0 and int(grid[r - 1, c - 1]) == int(color):
        return 0

    length = 0
    rr, cc = int(r), int(c)
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color):
        length += 1
        rr += 1
        cc += 1
    return int(length)


def _scaled_center_offset(size: int, scale: int) -> int:
    return (int(size) * int(scale) - 1) // 2


def _scaled_stamp_cells(
    local_pattern: np.ndarray,
    bbox_shape: tuple[int, int],
    center: tuple[int, int],
    scale: int,
    target_color: int,
) -> list[tuple[int, int, int]]:
    center_r, center_c = int(center[0]), int(center[1])
    h, w = int(bbox_shape[0]), int(bbox_shape[1])
    top = center_r - _scaled_center_offset(h, scale)
    left = center_c - _scaled_center_offset(w, scale)

    cells: list[tuple[int, int, int]] = []
    for pr, pc in map(tuple, np.asarray(local_pattern, dtype=int)):
        base_r = top + int(pr) * int(scale)
        base_c = left + int(pc) * int(scale)
        for dr in range(int(scale)):
            for dc in range(int(scale)):
                rr = base_r + dr
                cc = base_c + dc
                dist = abs(rr - center_r) + abs(cc - center_c)
                cells.append((int(rr), int(cc), _gradient_color(target_color, min(dist, 2))))
    return cells


def _scaled_footprint(
    local_pattern: np.ndarray,
    bbox_shape: tuple[int, int],
    center: tuple[int, int],
    scale: int,
) -> set[tuple[int, int]]:
    return {(r, c) for r, c, _color in _scaled_stamp_cells(local_pattern, bbox_shape, center, scale, 1)}


def _make_prototype() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    center = (1, 1)
    for _ in range(400):
        n = random.randint(PROTOTYPE_MIN_CELLS, PROTOTYPE_MAX_CELLS)
        coords = {center}
        while len(coords) < n:
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(list(frontier)))

        if len(coords) != n:
            continue
        rows = {r for r, _c in coords}
        cols = {c for _r, c in coords}
        if rows != {0, 1, 2} or cols != {0, 1, 2}:
            continue
        dists = {abs(r - 1) + abs(c - 1) for r, c in coords}
        if 1 not in dists or 2 not in dists:
            continue
        arr = np.asarray(sorted(coords), dtype=int)
        contacts = [tuple(map(int, rc)) for rc in arr.tolist() if tuple(map(int, rc)) != center]
        if not contacts:
            continue
        return [tuple(map(int, rc)) for rc in arr.tolist()], random.choice(contacts)
    raise RuntimeError("failed to build a usable 3x3 prototype")


def _choose_scales(n_scattered: int) -> list[int]:
    scales = [1, 3]
    while len(scales) < n_scattered:
        scales.append(random.choice(SCALE_CHOICES))
    random.shuffle(scales)
    return scales


def _build_rules(num_rules: int, area: int):
    source_colors = SOURCE_COLORS[:]
    target_colors = TARGET_COLORS[:]
    random.shuffle(source_colors)
    random.shuffle(target_colors)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        prototype, contact = _make_prototype()
        local_pattern = np.asarray(prototype, dtype=int)
        bbox_shape = (
            int(local_pattern[:, 0].max()) + 1,
            int(local_pattern[:, 1].max()) + 1,
        )
        rules.append(
            {
                "src": int(source_colors[i]),
                "tgt": int(target_colors[i]),
                "local_pattern": local_pattern,
                "bbox_shape": bbox_shape,
                "center_local": (1, 1),
                "legend_contact": (int(contact[0]), int(contact[1])),
                "scales": _choose_scales(random.randint(SCATTERED_MIN, sc_max)),
            }
        )
        used_colors.add(int(source_colors[i]))
        used_colors.add(int(target_colors[i]))
    return rules, used_colors


def _place_scattered(rules, inp, occupied, reserved, output_occupied):
    h, w = inp.shape
    for rule in rules:
        placements: list[tuple[int, int, int]] = []
        for scale in sorted(rule["scales"], reverse=True):
            placed = False
            row_margin = _scaled_center_offset(rule["bbox_shape"][0], scale)
            col_margin = _scaled_center_offset(rule["bbox_shape"][1], scale)
            if row_margin > h - 1 - row_margin or col_margin > w - 1 - col_margin:
                return False

            for _ in range(SCATTERED_RETRIES_PER_INSTANCE):
                sr = random.randint(row_margin, h - 1 - row_margin)
                sc = random.randint(col_margin, w - 1 - col_margin)
                chain = [(sr + i, sc + i) for i in range(scale)]
                if any(not (0 <= r < h and 0 <= c < w) for r, c in chain):
                    continue
                if any((r, c) in occupied or (r, c) in reserved for r, c in chain):
                    continue

                footprint = _scaled_footprint(rule["local_pattern"], rule["bbox_shape"], (sr, sc), scale)
                if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
                    continue
                if footprint & output_occupied:
                    continue

                for r, c in chain:
                    inp[r, c] = rule["src"]
                occupied.update(chain)
                _reserve_8neighbors(reserved, chain)
                reserved.update(footprint)
                output_occupied.update(footprint)
                placements.append((int(sr), int(sc), int(scale)))
                placed = True
                break

            if not placed:
                return False

        rule["scattered"] = placements
    return True


def _place_bystanders(used_colors, inp, occupied, reserved, output_occupied):
    bystander_colors = [c for c in CLEAN_BYSTANDER_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_occupied:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            occupied.add((r, c))
            output_occupied.add((r, c))
            _reserve_8neighbors(reserved, [(r, c)])
            placed.append((int(r), int(c), int(color)))
            break
    return placed


def _place_legends(rules, inp, occupied, reserved, output_occupied):
    h, w = inp.shape
    teal_occupied: set[tuple[int, int]] = set()
    for rule in rules:
        placed = False
        contact_r, contact_c = rule["legend_contact"]
        center_r, center_c = rule["center_local"]
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            top = ar - int(contact_r)
            left = ac - int(contact_c)
            teal_cells = {
                (top + int(pr), left + int(pc))
                for pr, pc in map(tuple, np.asarray(rule["local_pattern"], dtype=int))
            }
            center_abs = (top + int(center_r), left + int(center_c))
            reverse_anchor = (sr - dr, sc - dc)
            if center_abs == (ar, ac):
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in teal_cells):
                continue
            if teal_cells & {(sr, sc), (tr, tc)}:
                continue
            if _touches_region(teal_cells, reverse_anchor):
                continue
            if _touches_region(teal_occupied, (ar, ac)) or _touches_region(teal_occupied, reverse_anchor):
                continue

            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue
            if all_cells & output_occupied:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for r, c in teal_cells:
                inp[r, c] = 8
            occupied.update(all_cells)
            teal_occupied.update(teal_cells)
            _reserve_8neighbors(reserved, all_cells)
            _reserve_4neighbors(reserved, [reverse_anchor, (ar, ac)])
            placed = True
            break
        if not placed:
            return False
    return True


def _build_output(rules, bystander_pos, h: int, w: int) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[r, c] = color
    for rule in rules:
        for sr, sc, scale in rule["scattered"]:
            for r, c, color in _scaled_stamp_cells(
                rule["local_pattern"], rule["bbox_shape"], (sr, sc), scale, rule["tgt"]
            ):
                if 0 <= r < h and 0 <= c < w:
                    output[r, c] = color
    return output


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color legends and stamp scaled center-anchored gradient templates.

    Input structure:
    - The grid has a dominant background, several local source->target legend
      pairs, and nearby teal(8) prototype components.
    - Away from the legends, each usable source instance is the NW head of a
      SE-diagonal run of its own color; the run length is the scale factor.

    Rule:
    - Discover each rule from an adjacent colored pair whose forward continuation
      identifies a teal prototype component, then ignore that continuation point
      for stamping and instead compute the prototype's geometric center from its
      bounding box.
    - For every non-legend diagonal-run head of the matching source color, place
      a scale-k copy of the prototype centered on that head. Each prototype cell
      expands to a k x k block, and the final color of each stamped pixel is
      chosen from the target's gradient palette by Manhattan distance to the
      centered source cell.
    - Remove all legend pixels, all teal prototype pixels, and all source-color
      diagonal markers from the output.

    Key insight: The local legend still teaches which source color maps to which
    target hue and which teal shape to copy, but the actual placement origin is
    the shape's geometric center while the diagonal run length controls scale and
    therefore changes the distance-based gradient pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            scale = _diagonal_scale_head(g, sr, sc, rule["source_color"])
            if scale not in SCALE_CHOICES:
                continue
            for rr, cc, color in _scaled_stamp_cells(
                rule["local_pattern"], rule["bbox_shape"], (sr, sc), scale, rule["target_color"]
            ):
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color

    return out


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_scattered(rules, inp, occupied, reserved, output_occupied):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_occupied)
    if not _place_legends(rules, inp, occupied, reserved, output_occupied):
        return None

    output = _build_output(rules, bystander_pos, h, w)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
