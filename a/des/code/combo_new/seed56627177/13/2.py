from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BG_COLOR = 7
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = (0, 1, 2, 3, 4, 5, 6, 8, 9)
RECT_DIMS = tuple(
    (h, w)
    for h in range(2, 9)
    for w in range(2, 9)
    if 6 <= h * w <= 16
)
PAIR_COUNT_OPTIONS = (1, 2, 3)
GRID_LIMIT = 30
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


@dataclass(frozen=True)
class Component:
    color: int
    coords_abs: tuple[Coord, ...]
    coords_norm: frozenset[Coord]
    area: int
    bbox: tuple[int, int, int, int]
    h: int
    w: int


@dataclass(frozen=True)
class Placement:
    anchor_idx: int
    moved_idx: int
    rect_top: int
    rect_left: int
    rect_h: int
    rect_w: int
    moved_top: int
    moved_left: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (
            int(self.rect_top - 1),
            int(self.rect_left - 1),
            int(self.rect_top + self.rect_h),
            int(self.rect_left + self.rect_w),
        )


@dataclass(frozen=True)
class PairTemplate:
    anchor_norm: frozenset[Coord]
    moved_norm: frozenset[Coord]
    rect_h: int
    rect_w: int
    moved_offset: tuple[int, int]

    @property
    def anchor_area(self) -> int:
        return len(self.anchor_norm)

    @property
    def moved_area(self) -> int:
        return len(self.moved_norm)

    @property
    def box_shape(self) -> tuple[int, int]:
        return (int(self.rect_h + 2), int(self.rect_w + 2))


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: set[Coord] | frozenset[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _normalize(coords: set[Coord] | frozenset[Coord] | tuple[Coord, ...]) -> frozenset[Coord]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _is_connected(coords: set[Coord] | frozenset[Coord]) -> bool:
    pts = set((int(r), int(c)) for r, c in coords)
    if not pts:
        return False
    seen = set()
    q = deque([next(iter(pts))])
    seen.add(q[0])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nb = (int(r + dr), int(c + dc))
            if nb in pts and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(pts)


def _is_irregular(coords: set[Coord] | frozenset[Coord]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) < (int(r1 - r0 + 1) * int(c1 - c0 + 1))


def _perimeter_clockwise(top: int, left: int, height: int, width: int) -> list[Coord]:
    coords: list[Coord] = []
    bottom = int(top + height - 1)
    right = int(left + width - 1)
    for c in range(int(left), int(right) + 1):
        coords.append((int(top), int(c)))
    for r in range(int(top) + 1, int(bottom) + 1):
        coords.append((int(r), int(right)))
    for c in range(int(right) - 1, int(left) - 1, -1):
        coords.append((int(bottom), int(c)))
    for r in range(int(bottom) - 1, int(top), -1):
        coords.append((int(r), int(left)))
    return coords


def _paint_norm(grid: np.ndarray, coords_norm: frozenset[Coord], top: int, left: int, color: int) -> None:
    for r, c in coords_norm:
        grid[int(top + r), int(left + c)] = int(color)


def _extract_components(grid: np.ndarray, bg: int) -> list[Component]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[Component] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            cells = set(coords)
            r0, c0, r1, c1 = _bbox(cells)
            comps.append(
                Component(
                    color=color,
                    coords_abs=tuple(sorted(cells)),
                    coords_norm=_normalize(cells),
                    area=len(cells),
                    bbox=(int(r0), int(c0), int(r1), int(c1)),
                    h=int(r1 - r0 + 1),
                    w=int(c1 - c0 + 1),
                )
            )
    return comps


def _template_options(anchor_norm: frozenset[Coord], moved_norm: frozenset[Coord]) -> list[tuple[int, int, int, int]]:
    if (0, 0) not in anchor_norm:
        return []
    ah = max(int(r) for r, _c in anchor_norm) + 1
    aw = max(int(c) for _r, c in anchor_norm) + 1
    total = len(anchor_norm) + len(moved_norm)
    anchor_cells = set(anchor_norm)
    options: list[tuple[int, int, int, int]] = []

    for rect_h in range(1, total + 1):
        if total % rect_h != 0:
            continue
        rect_w = total // rect_h
        if rect_h < ah or rect_w < aw:
            continue
        if any(int(r) >= rect_h or int(c) >= rect_w for r, c in anchor_cells):
            continue
        comp = {
            (int(r), int(c))
            for r in range(int(rect_h))
            for c in range(int(rect_w))
            if (int(r), int(c)) not in anchor_cells
        }
        if _normalize(comp) != moved_norm:
            continue
        br = min(int(r) for r, _c in comp)
        bc = min(int(c) for _r, c in comp)
        options.append((int(rect_h), int(rect_w), int(br), int(bc)))

    return options


def _pair_options(anchor: Component, moved: Component, shape: tuple[int, int]) -> list[Placement]:
    h, w = shape
    rect_top = int(anchor.bbox[0])
    rect_left = int(anchor.bbox[1])
    placements: list[Placement] = []

    for rect_h, rect_w, br, bc in _template_options(anchor.coords_norm, moved.coords_norm):
        if rect_top < 1 or rect_left < 1:
            continue
        if rect_top + rect_h >= h or rect_left + rect_w >= w:
            continue
        placements.append(
            Placement(
                anchor_idx=-1,
                moved_idx=-1,
                rect_top=rect_top,
                rect_left=rect_left,
                rect_h=int(rect_h),
                rect_w=int(rect_w),
                moved_top=int(rect_top + br),
                moved_left=int(rect_left + bc),
            )
        )
    return placements


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (
        int(a[2]) < int(b[0])
        or int(b[2]) < int(a[0])
        or int(a[3]) < int(b[1])
        or int(b[3]) < int(a[1])
    )


def _solve_pairings(comps: list[Component], shape: tuple[int, int], max_solutions: int) -> list[list[Placement]]:
    n = len(comps)
    option_map: dict[tuple[int, int], list[Placement]] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cur = []
            for p in _pair_options(comps[i], comps[j], shape):
                cur.append(
                    Placement(
                        anchor_idx=int(i),
                        moved_idx=int(j),
                        rect_top=p.rect_top,
                        rect_left=p.rect_left,
                        rect_h=p.rect_h,
                        rect_w=p.rect_w,
                        moved_top=p.moved_top,
                        moved_left=p.moved_left,
                    )
                )
            if cur:
                option_map[(int(i), int(j))] = cur

    used = [False] * n
    chosen: list[Placement] = []
    solutions: list[list[Placement]] = []

    def rec() -> None:
        if len(solutions) >= int(max_solutions):
            return
        try:
            first = next(i for i in range(n) if not used[i])
        except StopIteration:
            solutions.append(list(chosen))
            return

        candidates: list[Placement] = []
        for j in range(first + 1, n):
            if used[j]:
                continue
            candidates.extend(option_map.get((int(first), int(j)), ()))
            candidates.extend(option_map.get((int(j), int(first)), ()))

        candidates.sort(
            key=lambda p: (
                p.box[0],
                p.box[1],
                p.box[2],
                p.box[3],
                p.anchor_idx,
                p.moved_idx,
            )
        )

        for cand in candidates:
            j = cand.moved_idx if cand.anchor_idx == first else cand.anchor_idx
            if used[j]:
                continue
            if any(_boxes_overlap(cand.box, prev.box) for prev in chosen):
                continue
            used[first] = True
            used[j] = True
            chosen.append(cand)
            rec()
            chosen.pop()
            used[first] = False
            used[j] = False

    rec()
    return solutions


def _render_solution(shape: tuple[int, int], bg: int, comps: list[Component], solution: list[Placement]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    ordered = sorted(solution, key=lambda p: (p.box[0], p.box[1], p.anchor_idx, p.moved_idx))

    for place in ordered:
        anchor = comps[place.anchor_idx]
        moved = comps[place.moved_idx]
        for r, c in anchor.coords_abs:
            out[int(r), int(c)] = int(anchor.color)
        _paint_norm(out, moved.coords_norm, place.moved_top, place.moved_left, moved.color)

        first = int(anchor.color) if int(anchor.area) >= int(moved.area) else int(moved.color)
        second = int(moved.color) if first == int(anchor.color) else int(anchor.color)
        border = _perimeter_clockwise(place.box[0], place.box[1], place.rect_h + 2, place.rect_w + 2)
        for idx, (r, c) in enumerate(border):
            out[int(r), int(c)] = first if idx % 2 == 0 else second

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary scraps into solid rectangles, keep the top-left anchor piece fixed, and wrap each result in an alternating border.

    Input structure:
    - The background is orange.
    - Several distinct-color monochrome regions are scattered on the background; every region is 4-connected.
    - The regions come in complementary pairs: one member of each pair already occupies the top-left corner of the rectangle that the pair can complete.

    Rule:
    - Extract every monochrome connected component.
    - Match pairs whose union becomes a solid axis-aligned rectangle after translating one component while keeping its mate fixed at that rectangle's top-left corner.
    - Keep that anchored component in place, move the complementary component into the missing cells, and erase the moved copy from its old location.
    - Add a one-cell outer border around the completed rectangle and color the border clockwise from its top-left pixel by alternating the two component colors, starting with the larger component's color.

    Key insight:
    - The pairing cue is pure shape complementarity, and the stationary member is the one that already defines the completed rectangle's top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _extract_components(g, bg)
    assert comps and len(comps) % 2 == 0, "expected an even number of components"

    solutions = _solve_pairings(comps, g.shape, max_solutions=2)
    assert solutions, "no valid complementary pairing found"
    return _render_solution(g.shape, bg, comps, solutions[0])


def _grow_connected_subset(rect_h: int, rect_w: int, area: int, rng: np.random.Generator) -> set[Coord]:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier: list[Coord] = []
        for r, c in sorted(cells):
            for dr, dc in DIRS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < rect_h and 0 <= nc < rect_w and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            break
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    return cells


def _random_template(rng: np.random.Generator, anchor_larger: bool) -> PairTemplate | None:
    dims = list(RECT_DIMS)
    rng.shuffle(dims)
    for rect_h, rect_w in dims:
        total = int(rect_h * rect_w)
        area_candidates = [
            area
            for area in range(2, min(8, total - 2) + 1)
            if 2 <= total - area <= 8
            and area != total - area
            and ((area > total - area) == bool(anchor_larger))
        ]
        rng.shuffle(area_candidates)
        for area in area_candidates:
            for _ in range(60):
                anchor = _grow_connected_subset(rect_h, rect_w, area, rng)
                moved = {
                    (int(r), int(c))
                    for r in range(int(rect_h))
                    for c in range(int(rect_w))
                    if (int(r), int(c)) not in anchor
                }
                if len(anchor) != int(area) or not _is_connected(anchor) or not _is_connected(moved):
                    continue
                if not _is_irregular(anchor) or not _is_irregular(moved):
                    continue
                anchor_norm = frozenset(anchor)
                moved_norm = _normalize(moved)
                if (0, 0) not in anchor_norm or (0, 0) in moved_norm:
                    continue
                opts_ab = _template_options(anchor_norm, moved_norm)
                opts_ba = _template_options(moved_norm, anchor_norm)
                if len(opts_ab) != 1 or opts_ba:
                    continue
                use_h, use_w, off_r, off_c = opts_ab[0]
                return PairTemplate(
                    anchor_norm=anchor_norm,
                    moved_norm=moved_norm,
                    rect_h=int(use_h),
                    rect_w=int(use_w),
                    moved_offset=(int(off_r), int(off_c)),
                )
    return None


def _paint_template_pair(
    grid: np.ndarray,
    template: PairTemplate,
    box_top: int,
    box_left: int,
    anchor_color: int,
    moved_color: int,
) -> None:
    rect_top = int(box_top + 1)
    rect_left = int(box_left + 1)
    _paint_norm(grid, template.anchor_norm, rect_top, rect_left, anchor_color)
    _paint_norm(
        grid,
        template.moved_norm,
        rect_top + template.moved_offset[0],
        rect_left + template.moved_offset[1],
        moved_color,
    )

    first = int(anchor_color) if template.anchor_area >= template.moved_area else int(moved_color)
    second = int(moved_color) if first == int(anchor_color) else int(anchor_color)
    border = _perimeter_clockwise(box_top, box_left, template.rect_h + 2, template.rect_w + 2)
    for idx, (r, c) in enumerate(border):
        grid[int(r), int(c)] = first if idx % 2 == 0 else second


def _mark_box(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = mask.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(h, int(top + height))
    c1 = min(w, int(left + width))
    if r0 < r1 and c0 < c1:
        mask[r0:r1, c0:c1] = True


def _can_place_box(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = blocked.shape
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(h, int(top + height + pad))
    c1 = min(w, int(left + width + pad))
    return not bool(blocked[r0:r1, c0:c1].any())


def _place_boxes(
    rng: np.random.Generator,
    box_shapes: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    order = list(range(len(box_shapes)))
    rng.shuffle(order)
    order.sort(key=lambda idx: box_shapes[idx][0] * box_shapes[idx][1], reverse=True)

    blocked = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(box_shapes)

    for idx in order:
        box_h, box_w = box_shapes[idx]
        spots: list[tuple[int, int]] = []
        for top in range(0, int(height - box_h + 1)):
            for left in range(0, int(width - box_w + 1)):
                if _can_place_box(blocked, top, left, box_h, box_w, pad=1):
                    spots.append((int(top), int(left)))
        if not spots:
            return None
        pick = spots[int(rng.integers(0, len(spots)))]
        positions[idx] = pick
        _mark_box(blocked, pick[0] - 1, pick[1] - 1, box_h + 2, box_w + 2)

    return [pos for pos in positions if pos is not None]


def _mark_blocked_cells(blocked: np.ndarray, cells: list[Coord] | tuple[Coord, ...], pad: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in range(-int(pad), int(pad) + 1):
            for dc in range(-int(pad), int(pad) + 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_remote_shape(
    rng: np.random.Generator,
    blocked: np.ndarray,
    coords_norm: frozenset[Coord],
) -> tuple[int, int] | None:
    h, w = blocked.shape
    bh = max(int(r) for r, _c in coords_norm) + 1
    bw = max(int(c) for _r, c in coords_norm) + 1
    cells = tuple(sorted(coords_norm))
    spots: list[tuple[int, int]] = []

    for top in range(0, int(h - bh + 1)):
        for left in range(0, int(w - bw + 1)):
            abs_cells = [(int(top + r), int(left + c)) for r, c in cells]
            if any(blocked[r, c] for r, c in abs_cells):
                continue
            spots.append((int(top), int(left)))

    if not spots:
        return None
    return spots[int(rng.integers(0, len(spots)))]


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = int(rng.choice(PAIR_COUNT_OPTIONS))
    templates: list[PairTemplate] = []
    for _ in range(pair_count):
        template = _random_template(rng, anchor_larger=bool(rng.integers(0, 2)))
        if template is None:
            return None
        templates.append(template)

    min_h = max(t.box_shape[0] for t in templates) + 2
    min_w = max(t.box_shape[1] for t in templates) + 2
    grid_h = int(rng.integers(max(14, min_h), GRID_LIMIT + 1))
    grid_w = int(rng.integers(max(14, min_w), GRID_LIMIT + 1))

    box_positions = _place_boxes(rng, [t.box_shape for t in templates], grid_h, grid_w)
    if box_positions is None:
        return None

    palette = rng.choice(np.array(COLOR_POOL, dtype=int), size=2 * pair_count, replace=False)
    colors = [int(v) for v in palette]

    out = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    output_boxes = np.zeros((grid_h, grid_w), dtype=bool)

    for idx, template in enumerate(templates):
        box_top, box_left = box_positions[idx]
        anchor_color = colors[2 * idx]
        moved_color = colors[2 * idx + 1]
        _paint_template_pair(out, template, box_top, box_left, anchor_color, moved_color)
        rect_top = int(box_top + 1)
        rect_left = int(box_left + 1)
        _paint_norm(inp, template.anchor_norm, rect_top, rect_left, anchor_color)
        _mark_box(output_boxes, box_top, box_left, template.box_shape[0], template.box_shape[1])

    blocked = output_boxes.copy()
    _mark_blocked_cells(blocked, list(zip(*np.where(output_boxes))), pad=1)

    order = list(range(pair_count))
    rng.shuffle(order)
    order.sort(key=lambda idx: len(templates[idx].moved_norm), reverse=True)
    for idx in order:
        template = templates[idx]
        loc = _place_remote_shape(rng, blocked, template.moved_norm)
        if loc is None:
            return None
        top, left = loc
        _paint_norm(inp, template.moved_norm, top, left, colors[2 * idx + 1])
        abs_cells = [(int(top + r), int(left + c)) for r, c in sorted(template.moved_norm)]
        _mark_blocked_cells(blocked, abs_cells, pad=1)

    comps = _extract_components(inp, BG_COLOR)
    if len(comps) != 2 * pair_count:
        return None
    solutions = _solve_pairings(comps, inp.shape, max_solutions=2)
    if len(solutions) != 1:
        return None
    if not np.array_equal(_render_solution(inp.shape, BG_COLOR, comps, solutions[0]), out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_case(rng)
        if candidate is not None:
            return candidate[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
