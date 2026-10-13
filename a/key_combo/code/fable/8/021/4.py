from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import random
from typing import Iterable

import numpy as np


def _load_arc_utils():
    path = "/home/lixinyu/snap/new/solvers/arc_utils.py"
    spec = importlib.util.spec_from_file_location("_shared_arc_utils_4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
get_bg = _ARC.get_bg
find_components = _ARC.find_components
crop_bbox = _ARC.crop_bbox
neighbors4 = _ARC.neighbors4
in_bounds = _ARC.in_bounds


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class PlacedSprite:
    color: int
    peg_color: int
    direction: str
    input_r0: int
    input_c0: int
    shift: int
    anchor: tuple[int, int]
    peg: tuple[int, int]
    target_hole: tuple[int, int]
    body_mask: tuple[tuple[int, ...], ...]

    @property
    def height(self) -> int:
        return len(self.body_mask)

    @property
    def width(self) -> int:
        return len(self.body_mask[0])

    @property
    def final_r0(self) -> int:
        dr, _dc = DIRS[self.direction]
        return self.input_r0 + dr * self.shift

    @property
    def final_c0(self) -> int:
        _dr, dc = DIRS[self.direction]
        return self.input_c0 + dc * self.shift


def _mask_to_array(mask: tuple[tuple[int, ...], ...] | np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def _solid_cells(mask: tuple[tuple[int, ...], ...] | np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    body = _mask_to_array(mask)
    return {
        (int(r0 + r), int(c0 + c))
        for r, c in np.argwhere(body)
    }


def _bbox_cells(h: int, w: int, r0: int, c0: int) -> set[tuple[int, int]]:
    return {
        (int(r0 + r), int(c0 + c))
        for r in range(h)
        for c in range(w)
    }


def _hole_cells(mask: tuple[tuple[int, ...], ...] | np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    body = _mask_to_array(mask)
    return {
        (int(r0 + r), int(c0 + c))
        for r, c in np.argwhere(~body)
    }


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _enclosed_holes(body_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = body_mask.shape
    hole_comps: list[list[tuple[int, int]]] = []
    for comp in _component_cells(~body_mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        hole_comps.append(comp)
    return hole_comps


def _border_side(r: int, c: int, shape: tuple[int, int]) -> str | None:
    h, w = shape
    on = []
    if r == 0:
        on.append("up")
    if r == h - 1:
        on.append("down")
    if c == 0:
        on.append("left")
    if c == w - 1:
        on.append("right")
    if len(on) != 1:
        return None
    return on[0]


def _is_isolated_border_pixel(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    if _border_side(r, c, grid.shape) is None:
        return False
    if int(grid[r, c]) == int(bg):
        return False
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != int(bg):
            return False
    return True


def _anchors_by_color(grid: np.ndarray, bg: int, comps: list[dict]) -> dict[int, tuple[int, int, str]]:
    anchors: dict[int, tuple[int, int, str]] = {}
    for comp in comps:
        if int(comp["area"]) != 1:
            continue
        r, c = map(int, comp["coords"][0])
        side = _border_side(r, c, grid.shape)
        if side is None or not _is_isolated_border_pixel(grid, bg, r, c):
            continue
        color = int(comp["color"])
        assert color not in anchors, "anchor color must be unique"
        anchors[color] = (int(r), int(c), side)
    return anchors


def _sprite_data_from_component(grid: np.ndarray, comp: dict) -> dict | None:
    color = int(comp["color"])
    bbox = tuple(map(int, comp["bbox"]))
    patch = crop_bbox(grid, bbox)
    body_mask = patch == color
    holes = _enclosed_holes(body_mask)
    if not holes:
        return None
    return {
        "color": color,
        "bbox": bbox,
        "body_mask": body_mask,
        "holes": holes,
        "solid_coords": {(int(r), int(c)) for r, c in comp["coords"]},
    }


def _corridor_singletons(
    bbox: tuple[int, int, int, int],
    direction: str,
    peg_cells: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: list[tuple[int, int]] = []
    for r, c in peg_cells:
        if direction == "right" and r0 <= r <= r1 and c > c1:
            out.append((int(r), int(c)))
        elif direction == "left" and r0 <= r <= r1 and c < c0:
            out.append((int(r), int(c)))
        elif direction == "down" and c0 <= c <= c1 and r > r1:
            out.append((int(r), int(c)))
        elif direction == "up" and c0 <= c <= c1 and r < r0:
            out.append((int(r), int(c)))
    return out


def _first_alignment_shift(
    bbox: tuple[int, int, int, int],
    holes: list[list[tuple[int, int]]],
    direction: str,
    peg: tuple[int, int],
) -> int | None:
    r0, c0, _r1, _c1 = bbox
    pr, pc = peg
    candidates: list[int] = []
    for hole in holes:
        for hr, hc in hole:
            if direction == "right" and pr == r0 + hr:
                k = pc - (c0 + hc)
            elif direction == "left" and pr == r0 + hr:
                k = (c0 + hc) - pc
            elif direction == "down" and pc == c0 + hc:
                k = pr - (r0 + hr)
            elif direction == "up" and pc == c0 + hc:
                k = (r0 + hr) - pr
            else:
                continue
            if k >= 0:
                candidates.append(int(k))
    if not candidates:
        return None
    return min(candidates)


def _solve_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    anchors = _anchors_by_color(g, bg, comps)

    sprite_by_color: dict[int, dict] = {}
    singleton_pegs: list[tuple[int, int]] = []
    for comp in comps:
        area = int(comp["area"])
        color = int(comp["color"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            if not _is_isolated_border_pixel(g, bg, r, c):
                singleton_pegs.append((int(r), int(c)))
            continue
        if color not in anchors:
            continue
        sprite = _sprite_data_from_component(g, comp)
        if sprite is None:
            continue
        prev = sprite_by_color.get(color)
        if prev is None or len(sprite["solid_coords"]) > len(prev["solid_coords"]):
            sprite_by_color[color] = sprite

    solved: list[dict] = []
    for color, sprite in sprite_by_color.items():
        _ar, _ac, direction = anchors[color]
        corridor = _corridor_singletons(sprite["bbox"], direction, singleton_pegs)
        valid: list[tuple[tuple[int, int], int]] = []
        for peg in corridor:
            shift = _first_alignment_shift(sprite["bbox"], sprite["holes"], direction, peg)
            if shift is not None:
                valid.append((peg, shift))
        assert len(valid) == 1, "expected exactly one usable peg per anchored sprite"
        peg, shift = valid[0]
        solved.append(
            {
                "color": color,
                "direction": direction,
                "bbox": sprite["bbox"],
                "body_mask": sprite["body_mask"],
                "solid_coords": sprite["solid_coords"],
                "shift": int(shift),
                "peg": peg,
            }
        )
    assert solved, "expected at least one sprite"
    return bg, solved


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide each holey monochrome sprite toward its same-colored border anchor until its peg first falls into a hole.

    Input structure:
    - The background is the grid's most common color.
    - Interior monochrome sprites contain one or more background-colored holes inside their own bbox.
    - Each sprite color also appears once as an isolated border anchor, and each sprite's corridor toward that edge contains one interior single-pixel peg.

    Rule:
    - Use the anchor edge to choose the slide direction: top, bottom, left, or right.
    - Move the whole sprite one cell at a time in that direction; a peg may be covered by solid body cells without stopping the motion.
    - Stop at the first translation where the peg lies inside any hole of that same sprite, then keep anchors and pegs fixed in place.

    Key insight:
    - The stopping phase is determined by the hole layout inside the sprite, not by touching the anchor or by colliding with the peg.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, solved = _solve_objects(g)
    out = g.copy()

    for obj in solved:
        for r, c in obj["solid_coords"]:
            out[int(r), int(c)] = int(bg)

    for obj in solved:
        dr, dc = DIRS[obj["direction"]]
        for r, c in obj["solid_coords"]:
            out[int(r + dr * obj["shift"]), int(c + dc * obj["shift"])] = int(obj["color"])

    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
SPRITE_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
BODY_H_RANGE = (5, 8)
BODY_W_RANGE = (5, 8)
HOLE_COUNT_CHOICES = (1, 1, 2, 2, 3)
HOLE_H_RANGE = (1, 3)
HOLE_W_RANGE = (1, 3)
OUTER_MARGIN = 1
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _body_connected(mask: np.ndarray) -> bool:
    comps = _component_cells(mask)
    return len(comps) == 1


def _random_body_mask() -> np.ndarray:
    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(*BODY_H_RANGE)
        w = random.randint(*BODY_W_RANGE)
        body = np.ones((h, w), dtype=bool)
        hole_owner = np.zeros((h, w), dtype=int)
        n_holes = random.choice(HOLE_COUNT_CHOICES)
        placed = 0
        for hole_id in range(1, n_holes + 1):
            choices = []
            for hh in range(HOLE_H_RANGE[0], min(HOLE_H_RANGE[1], h - 2) + 1):
                for ww in range(HOLE_W_RANGE[0], min(HOLE_W_RANGE[1], w - 2) + 1):
                    for r0 in range(1, h - hh):
                        for c0 in range(1, w - ww):
                            rr0 = max(0, r0 - 1)
                            cc0 = max(0, c0 - 1)
                            rr1 = min(h, r0 + hh + 1)
                            cc1 = min(w, c0 + ww + 1)
                            if hole_owner[rr0:rr1, cc0:cc1].any():
                                continue
                            choices.append((r0, c0, hh, ww))
            if not choices:
                break
            r0, c0, hh, ww = random.choice(choices)
            body[r0 : r0 + hh, c0 : c0 + ww] = False
            hole_owner[r0 : r0 + hh, c0 : c0 + ww] = hole_id
            placed += 1
        holes = _enclosed_holes(body)
        if placed == 0 or not holes or not _body_connected(body):
            continue
        return body
    raise RuntimeError("failed to build a holey sprite")


def _extremal_hole_targets(body: np.ndarray, direction: str) -> list[tuple[int, int]]:
    holes = np.argwhere(~body)
    if direction in {"left", "right"}:
        rows = sorted({int(r) for r, _c in holes})
        out = []
        for r in rows:
            cols = [int(c) for rr, c in holes if int(rr) == int(r)]
            out.append((int(r), max(cols) if direction == "right" else min(cols)))
        return out
    cols = sorted({int(c) for _r, c in holes})
    out = []
    for c in cols:
        rows = [int(r) for r, cc in holes if int(cc) == int(c)]
        out.append((max(rows) if direction == "down" else min(rows), int(c)))
    return out


def _min_shift_outside(body: np.ndarray, direction: str, hole: tuple[int, int]) -> int:
    h, w = body.shape
    hr, hc = hole
    if direction == "right":
        return int(w - hc)
    if direction == "left":
        return int(hc + 1)
    if direction == "down":
        return int(h - hr)
    return int(hr + 1)


def _corridor_clear_cells(
    shape: tuple[int, int],
    body: np.ndarray,
    direction: str,
    input_r0: int,
    input_c0: int,
    peg: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = body.shape
    H, W = shape
    cells: set[tuple[int, int]] = set()
    if direction == "right":
        for r in range(input_r0, input_r0 + h):
            for c in range(input_c0 + w, W - 1):
                cells.add((int(r), int(c)))
    elif direction == "left":
        for r in range(input_r0, input_r0 + h):
            for c in range(1, input_c0):
                cells.add((int(r), int(c)))
    elif direction == "down":
        for r in range(input_r0 + h, H - 1):
            for c in range(input_c0, input_c0 + w):
                cells.add((int(r), int(c)))
    else:
        for r in range(1, input_r0):
            for c in range(input_c0, input_c0 + w):
                cells.add((int(r), int(c)))
    cells.discard((int(peg[0]), int(peg[1])))
    return cells


def _anchor_guard(anchor: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = anchor
    guard = set()
    for nr, nc in neighbors4(r, c):
        if in_bounds(shape, nr, nc):
            guard.add((int(nr), int(nc)))
    return guard


def _candidate_anchors(
    shape: tuple[int, int],
    direction: str,
    blocked: set[tuple[int, int]],
    reserved_clear: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    H, W = shape
    candidates: list[tuple[int, int]] = []
    if direction in {"left", "right"}:
        c = 0 if direction == "left" else W - 1
        for r in range(1, H - 1):
            anchor = (int(r), int(c))
            guard = _anchor_guard(anchor, shape)
            if anchor in blocked or anchor in reserved_clear:
                continue
            if any(cell in blocked for cell in guard):
                continue
            if any(cell in reserved_clear for cell in guard):
                continue
            candidates.append(anchor)
    else:
        r = 0 if direction == "up" else H - 1
        for c in range(1, W - 1):
            anchor = (int(r), int(c))
            guard = _anchor_guard(anchor, shape)
            if anchor in blocked or anchor in reserved_clear:
                continue
            if any(cell in blocked for cell in guard):
                continue
            if any(cell in reserved_clear for cell in guard):
                continue
            candidates.append(anchor)
    return candidates


def _render_case(shape: tuple[int, int], bg: int, sprites: list[PlacedSprite]) -> tuple[np.ndarray, np.ndarray]:
    H, W = shape
    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    for spec in sprites:
        in_body = _solid_cells(spec.body_mask, spec.input_r0, spec.input_c0)
        out_body = _solid_cells(spec.body_mask, spec.final_r0, spec.final_c0)
        ar, ac = spec.anchor
        pr, pc = spec.peg
        for r, c in in_body:
            inp[r, c] = int(spec.color)
        for r, c in out_body:
            out[r, c] = int(spec.color)
        inp[ar, ac] = int(spec.color)
        out[ar, ac] = int(spec.color)
        inp[pr, pc] = int(spec.peg_color)
        out[pr, pc] = int(spec.peg_color)
    return inp, out


def _peg_in_hole_after_shift(spec: PlacedSprite, shift: int) -> bool:
    return spec.peg in _hole_cells(spec.body_mask, spec.input_r0 + DIRS[spec.direction][0] * shift, spec.input_c0 + DIRS[spec.direction][1] * shift)


def _make_sprite_spec(
    shape: tuple[int, int],
    body: np.ndarray,
    color: int,
    peg_color: int,
    direction: str,
    occupied_input: set[tuple[int, int]],
    reserved_clear: set[tuple[int, int]],
    reserved_input_space: set[tuple[int, int]],
    reserved_output_space: set[tuple[int, int]],
    existing_pegs: set[tuple[int, int]],
) -> PlacedSprite | None:
    H, W = shape
    h, w = body.shape
    target_holes = _extremal_hole_targets(body, direction)
    random.shuffle(target_holes)
    for target_hole in target_holes:
        min_shift = _min_shift_outside(body, direction, target_hole)
        for _ in range(PLACEMENT_ATTEMPTS):
            input_r0 = random.randint(OUTER_MARGIN, H - h - OUTER_MARGIN)
            input_c0 = random.randint(OUTER_MARGIN, W - w - OUTER_MARGIN)
            if direction == "right":
                max_shift = (W - w - OUTER_MARGIN) - input_c0
            elif direction == "left":
                max_shift = input_c0 - OUTER_MARGIN
            elif direction == "down":
                max_shift = (H - h - OUTER_MARGIN) - input_r0
            else:
                max_shift = input_r0 - OUTER_MARGIN
            if max_shift < min_shift:
                continue
            shift = random.randint(min_shift, max_shift)
            dr, dc = DIRS[direction]
            final_r0 = input_r0 + dr * shift
            final_c0 = input_c0 + dc * shift
            peg = (int(final_r0 + target_hole[0]), int(final_c0 + target_hole[1]))
            in_body = _solid_cells(body, input_r0, input_c0)
            out_body = _solid_cells(body, final_r0, final_c0)
            in_bbox = _bbox_cells(h, w, input_r0, input_c0)
            out_bbox = _bbox_cells(h, w, final_r0, final_c0)
            if peg in in_body or peg in out_body:
                continue
            corridor_clear = _corridor_clear_cells(shape, body, direction, input_r0, input_c0, peg)
            blocked_now = occupied_input | in_body | {peg}
            if in_bbox & reserved_input_space:
                continue
            if in_body & reserved_clear:
                continue
            if peg in reserved_clear or peg in occupied_input or peg in reserved_input_space:
                continue
            if corridor_clear & blocked_now:
                continue
            if corridor_clear & occupied_input:
                continue
            if corridor_clear & reserved_input_space:
                continue
            if out_bbox & reserved_output_space:
                continue
            if out_bbox & existing_pegs:
                continue
            anchors = _candidate_anchors(
                shape,
                direction,
                blocked_now | reserved_input_space,
                reserved_clear | corridor_clear,
            )
            if not anchors:
                continue
            anchor = random.choice(anchors)
            guard = _anchor_guard(anchor, shape)
            if in_body & guard or peg in guard:
                continue
            if out_body & {anchor}:
                continue
            spec = PlacedSprite(
                color=int(color),
                peg_color=int(peg_color),
                direction=direction,
                input_r0=int(input_r0),
                input_c0=int(input_c0),
                shift=int(shift),
                anchor=(int(anchor[0]), int(anchor[1])),
                peg=(int(peg[0]), int(peg[1])),
                target_hole=(int(target_hole[0]), int(target_hole[1])),
                body_mask=tuple(tuple(int(v) for v in row) for row in body.astype(int)),
            )
            if any(_peg_in_hole_after_shift(spec, k) for k in range(spec.shift)):
                continue
            return spec
    return None


def _choose_sprite_count(H: int, W: int) -> int:
    max_count = 4 if min(H, W) >= 22 else 3 if min(H, W) >= 19 else 2
    return random.choice([v for v in SPRITE_COUNT_CHOICES if v <= max_count])


def _generate_case() -> tuple[np.ndarray, np.ndarray, list[PlacedSprite], int]:
    for _ in range(GEN_ATTEMPTS):
        H = random.randint(*GRID_H_RANGE)
        W = random.randint(*GRID_W_RANGE)
        bg = random.choice([0, 5])
        count = _choose_sprite_count(H, W)
        directions = random.sample(list(DIRS), count)
        sprite_colors = random.sample([c for c in range(10) if c != bg], count)

        occupied_input: set[tuple[int, int]] = set()
        reserved_clear: set[tuple[int, int]] = set()
        reserved_input_space: set[tuple[int, int]] = set()
        reserved_output_space: set[tuple[int, int]] = set()
        existing_pegs: set[tuple[int, int]] = set()
        specs: list[PlacedSprite] = []
        ok = True

        for direction, color in zip(directions, sprite_colors):
            body = _random_body_mask()
            peg_choices = [c for c in range(10) if c != bg and c != color]
            peg_color = random.choice(peg_choices)
            spec = _make_sprite_spec(
                (H, W),
                body,
                color,
                peg_color,
                direction,
                occupied_input,
                reserved_clear,
                reserved_input_space,
                reserved_output_space,
                existing_pegs,
            )
            if spec is None:
                ok = False
                break
            in_body = _solid_cells(spec.body_mask, spec.input_r0, spec.input_c0)
            out_body = _solid_cells(spec.body_mask, spec.final_r0, spec.final_c0)
            in_bbox = _bbox_cells(spec.height, spec.width, spec.input_r0, spec.input_c0)
            out_bbox = _bbox_cells(spec.height, spec.width, spec.final_r0, spec.final_c0)
            corridor_clear = _corridor_clear_cells((H, W), _mask_to_array(spec.body_mask), spec.direction, spec.input_r0, spec.input_c0, spec.peg)
            occupied_input |= in_body | {spec.peg, spec.anchor}
            reserved_clear |= corridor_clear | _anchor_guard(spec.anchor, (H, W))
            reserved_input_space |= in_bbox
            reserved_output_space |= out_bbox
            existing_pegs.add(spec.peg)
            specs.append(spec)

        if not ok or not specs:
            continue

        inp, out = _render_case((H, W), bg, specs)
        if int(get_bg(inp)) != int(bg):
            continue
        if np.array_equal(inp, out):
            continue
        return inp, out, specs, int(bg)
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, out, specs, bg = _generate_case()
        if not np.array_equal(main(inp), out):
            continue
        if not _validate_case(inp, out, specs, bg):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")


def _validate_case(inp: np.ndarray, out: np.ndarray, specs: list[PlacedSprite], bg: int) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if int(get_bg(inp)) != int(bg):
        return False
    if not np.array_equal(main(inp), out):
        return False
    directions_seen = {spec.direction for spec in specs}
    if not directions_seen:
        return False
    for spec in specs:
        if inp[spec.anchor] != spec.color or out[spec.anchor] != spec.color:
            return False
        if inp[spec.peg] != spec.peg_color or out[spec.peg] != spec.peg_color:
            return False
        in_body = _solid_cells(spec.body_mask, spec.input_r0, spec.input_c0)
        final_body = _solid_cells(spec.body_mask, spec.final_r0, spec.final_c0)
        final_holes = _hole_cells(spec.body_mask, spec.final_r0, spec.final_c0)
        if spec.peg in in_body or spec.peg in final_body:
            return False
        if spec.peg not in final_holes:
            return False
        if any(_peg_in_hole_after_shift(spec, k) for k in range(spec.shift)):
            return False
        anchor_side = _border_side(spec.anchor[0], spec.anchor[1], inp.shape)
        if anchor_side != spec.direction:
            return False
        if not _is_isolated_border_pixel(inp, bg, spec.anchor[0], spec.anchor[1]):
            return False
    return True
