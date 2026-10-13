from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BBox = tuple[int, int, int, int]
Direction = tuple[int, int]
MAGENTA = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
DIRS: tuple[Direction, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_RANGE = (18, 24)
OBJECT_COUNTS = (2, 3)
BODY_SPAN_RANGE = (2, 4)
TAIL_RANGE = (2, 4)
MAX_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Slider:
    color: int
    body: BBox
    direction: Direction
    length: int


@dataclass(frozen=True)
class SliderSpec:
    color: int
    body_h: int
    body_w: int
    direction: Direction
    length: int


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_full(mask: np.ndarray, bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _shift_bbox(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _bbox_in_bounds(box: BBox, h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _rect_cells(box: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _tail_cells(body: BBox, direction: Direction, length: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = body
    dr, dc = direction
    if dr != 0:
        col = (c0 + c1) // 2
        start = r0 - 1 if dr < 0 else r1 + 1
        return {(start + dr * k, col) for k in range(length)}
    row = (r0 + r1) // 2
    start = c0 - 1 if dc < 0 else c1 + 1
    return {(row, start + dc * k) for k in range(length)}


def _detect_slider(mask: np.ndarray) -> tuple[BBox, Direction, int] | None:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0, r1, c1 = bbox_coords(coords)
    candidates: list[tuple[BBox, Direction, int]] = []

    def add_if_valid(body: BBox, direction: Direction, length: int, outside: np.ndarray) -> None:
        br0, bc0, br1, bc1 = body
        if length < 2:
            return
        if br1 - br0 + 1 < 2 or bc1 - bc0 + 1 < 2:
            return
        if outside.shape[0] != length or not _is_full(mask, body):
            return

        rows = set(map(int, outside[:, 0]))
        cols = set(map(int, outside[:, 1]))
        dr, dc = direction
        if dr == 0:
            expected_cols = set(range(bc1 + 1, c1 + 1)) if dc > 0 else set(range(c0, bc0))
            if len(rows) == 1 and cols == expected_cols:
                candidates.append((body, direction, length))
        else:
            expected_rows = set(range(br1 + 1, r1 + 1)) if dr > 0 else set(range(r0, br0))
            if len(cols) == 1 and rows == expected_rows:
                candidates.append((body, direction, length))

    for body_c1 in range(c0, c1):
        body = (r0, c0, r1, body_c1)
        add_if_valid(body, (0, 1), c1 - body_c1, coords[coords[:, 1] > body_c1])
    for body_c0 in range(c0 + 1, c1 + 1):
        body = (r0, body_c0, r1, c1)
        add_if_valid(body, (0, -1), body_c0 - c0, coords[coords[:, 1] < body_c0])
    for body_r1 in range(r0, r1):
        body = (r0, c0, body_r1, c1)
        add_if_valid(body, (1, 0), r1 - body_r1, coords[coords[:, 0] > body_r1])
    for body_r0 in range(r0 + 1, r1 + 1):
        body = (body_r0, c0, r1, c1)
        add_if_valid(body, (-1, 0), body_r0 - r0, coords[coords[:, 0] < body_r0])

    assert len(candidates) <= 1, "slider decomposition must be unique"
    return candidates[0] if candidates else None


def _parse_sliders(grid: np.ndarray) -> list[Slider]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == 0, "expected black background"
    colors = colors_present(g, exclude={bg})
    assert MAGENTA not in colors, "magenta is reserved for output overlaps"
    assert 2 <= len(colors) <= 3, "expected 2-3 colored slider objects"

    sliders: list[Slider] = []
    for color in colors:
        slider = _detect_slider(g == color)
        assert slider is not None, f"color {color} is not a valid tailed rectangle"
        body, direction, length = slider
        sliders.append(Slider(color=int(color), body=body, direction=direction, length=int(length)))
    return sliders


def _stepped_claim_cells(body: BBox, direction: Direction, period: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = body
    dr, dc = direction
    cells: set[tuple[int, int]] = set()
    stage = 1

    if dr != 0:
        while True:
            if dr < 0:
                sr0 = r0 - stage * period
                sr1 = r0 - (stage - 1) * period - 1
                if sr1 < 0:
                    break_after = True
                else:
                    break_after = False
            else:
                sr0 = r1 + (stage - 1) * period + 1
                sr1 = r1 + stage * period
                if sr0 >= h:
                    break
                break_after = False

            sc0 = c0 - stage
            sc1 = c1 + stage
            rr0 = max(0, sr0)
            rr1 = min(h - 1, sr1)
            cc0 = max(0, sc0)
            cc1 = min(w - 1, sc1)
            if rr0 <= rr1 and cc0 <= cc1:
                for rr in range(rr0, rr1 + 1):
                    for cc in range(cc0, cc1 + 1):
                        cells.add((rr, cc))
            if break_after:
                break
            stage += 1
    else:
        while True:
            if dc < 0:
                sc0 = c0 - stage * period
                sc1 = c0 - (stage - 1) * period - 1
                if sc1 < 0:
                    break_after = True
                else:
                    break_after = False
            else:
                sc0 = c1 + (stage - 1) * period + 1
                sc1 = c1 + stage * period
                if sc0 >= w:
                    break
                break_after = False

            sr0 = r0 - stage
            sr1 = r1 + stage
            rr0 = max(0, sr0)
            rr1 = min(h - 1, sr1)
            cc0 = max(0, sc0)
            cc1 = min(w - 1, sc1)
            if rr0 <= rr1 and cc0 <= cc1:
                for rr in range(rr0, rr1 + 1):
                    for cc in range(cc0, cc1 + 1):
                        cells.add((rr, cc))
            if break_after:
                break
            stage += 1

    return cells


def _render_scene(
    sliders: list[Slider],
    shape: tuple[int, int],
    *,
    slide_lengths: dict[int, int] | None = None,
    period_lengths: dict[int, int] | None = None,
    include_expansion: bool = True,
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=int)
    body_mask = np.zeros((h, w), dtype=bool)
    moved: list[tuple[Slider, BBox]] = []

    for slider in sliders:
        slide_len = slider.length if slide_lengths is None else int(slide_lengths[slider.color])
        dr, dc = slider.direction
        moved_box = _shift_bbox(slider.body, dr * slide_len, dc * slide_len)
        assert _bbox_in_bounds(moved_box, h, w), "moved rectangle must stay in bounds"
        claim = _rect_cells(moved_box)
        assert not any(body_mask[r, c] for r, c in claim), "moved rectangles must not overlap"
        for r, c in claim:
            body_mask[r, c] = True
            out[r, c] = slider.color
        moved.append((slider, moved_box))

    if not include_expansion:
        return out

    claims: dict[tuple[int, int], set[int]] = {}
    for slider, moved_box in moved:
        period = slider.length if period_lengths is None else int(period_lengths[slider.color])
        assert period >= 1
        for cell in _stepped_claim_cells(moved_box, slider.direction, period, (h, w)):
            r, c = cell
            if body_mask[r, c]:
                continue
            claims.setdefault(cell, set()).add(slider.color)

    for (r, c), colors in claims.items():
        out[r, c] = next(iter(colors)) if len(colors) == 1 else MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each tailed rectangle by its tail length, then paint a same-color stepped halo whose period is that same tail length.

    Input structure:
    - The background is black.
    - There are 2-3 foreground colors, each forming one solid rectangle with a straight one-pixel-wide tail.
    - Every tail extends from one side of its rectangle in a cardinal direction, and magenta does not appear in the input.

    Rule:
    - For each colored object, detect the rectangular body, the tail direction, and the tail length.
    - Erase the tail and translate only the body by exactly the tail length in the tail direction.
    - Starting just beyond the moved body's forward edge, extend same-color stepped bands in blocks of that tail length; each successive block widens by one cell on both side axes.
    - Draw all moved bodies first, then draw all stepped expansions without overwriting any moved body cells.
    - If stepped expansions from different colors claim the same non-body cell, recolor that overlap cell magenta.

    Key insight:
    - The tail length is read twice from the same geometry: once as the slide distance for the body, and again as the step period that controls the halo thickness.
    """
    sliders = _parse_sliders(np.asarray(grid, dtype=int))
    return _render_scene(sliders, np.asarray(grid, dtype=int).shape)


def _make_specs(colors: list[int]) -> list[SliderSpec]:
    count = len(colors)
    shared_dir = random.choice(DIRS)
    lengths = random.sample(range(TAIL_RANGE[0], TAIL_RANGE[1] + 1), k=2)
    specs: list[SliderSpec] = []

    for i in range(count):
        direction = shared_dir if i < 2 else random.choice(DIRS)
        if direction[0] != 0:
            body_h = random.randint(BODY_SPAN_RANGE[0], BODY_SPAN_RANGE[1])
            body_w = 3
        else:
            body_h = 3
            body_w = random.randint(BODY_SPAN_RANGE[0], BODY_SPAN_RANGE[1])
        length = lengths[i] if i < 2 else random.randint(TAIL_RANGE[0], TAIL_RANGE[1])
        specs.append(
            SliderSpec(
                color=colors[i],
                body_h=body_h,
                body_w=body_w,
                direction=direction,
                length=length,
            )
        )

    return specs


def _input_body_from_final(final_body: BBox, spec: SliderSpec) -> BBox:
    dr, dc = spec.direction
    return _shift_bbox(final_body, -dr * spec.length, -dc * spec.length)


def _object_cells_from_final(final_body: BBox, spec: SliderSpec) -> tuple[BBox, set[tuple[int, int]]]:
    input_body = _input_body_from_final(final_body, spec)
    cells = _rect_cells(input_body) | _tail_cells(input_body, spec.direction, spec.length)
    return input_body, cells


def _guard_zone(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _random_final_bbox(spec: SliderSpec, shape: tuple[int, int]) -> BBox:
    h, w = map(int, shape)
    dr, dc = spec.direction
    bh, bw = spec.body_h, spec.body_w

    if dr < 0:
        max_ahead = min(8, h - bh - spec.length)
        r0 = random.randint(1, max_ahead)
        c0 = random.randint(0, w - bw)
    elif dr > 0:
        max_ahead = min(8, h - bh - spec.length)
        ahead = random.randint(1, max_ahead)
        r0 = h - bh - ahead
        c0 = random.randint(0, w - bw)
    elif dc < 0:
        r0 = random.randint(0, h - bh)
        max_ahead = min(8, w - bw - spec.length)
        c0 = random.randint(1, max_ahead)
    else:
        r0 = random.randint(0, h - bh)
        max_ahead = min(8, w - bw - spec.length)
        ahead = random.randint(1, max_ahead)
        c0 = w - bw - ahead

    return int(r0), int(c0), int(r0 + bh - 1), int(c0 + bw - 1)


def _guided_final_bbox(spec: SliderSpec, shape: tuple[int, int], ref: BBox) -> BBox:
    h, w = map(int, shape)
    lo_r, lo_c, hi_r, hi_c = _final_bounds(spec, shape)
    rr0, rc0, rr1, rc1 = ref

    if spec.direction[0] != 0:
        r0 = min(max(rr0 + random.randint(-2, 2), lo_r), hi_r)
        side = random.choice((-1, 1))
        gap = random.randint(1, 3)
        if side < 0:
            c0 = rc0 - spec.body_w - gap
        else:
            c0 = rc1 + 1 + gap
        c0 = min(max(c0, lo_c), hi_c)
    else:
        c0 = min(max(rc0 + random.randint(-2, 2), lo_c), hi_c)
        side = random.choice((-1, 1))
        gap = random.randint(1, 3)
        if side < 0:
            r0 = rr0 - spec.body_h - gap
        else:
            r0 = rr1 + 1 + gap
        r0 = min(max(r0, lo_r), hi_r)

    return int(r0), int(c0), int(r0 + spec.body_h - 1), int(c0 + spec.body_w - 1)


def _final_bounds(spec: SliderSpec, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    bh, bw = spec.body_h, spec.body_w
    dr, dc = spec.direction
    if dr < 0:
        return 1, 0, h - bh - spec.length, w - bw
    if dr > 0:
        return spec.length, 0, h - bh - 1, w - bw
    if dc < 0:
        return 0, 1, h - bh, w - bw - spec.length
    return 0, spec.length, h - bh, w - bw - 1


def _can_place(
    final_body: BBox,
    spec: SliderSpec,
    shape: tuple[int, int],
    input_guard: set[tuple[int, int]],
    final_occupied: set[tuple[int, int]],
) -> tuple[BBox, set[tuple[int, int]]] | None:
    h, w = map(int, shape)
    input_body, cells = _object_cells_from_final(final_body, spec)
    if not _bbox_in_bounds(input_body, h, w):
        return None
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None
    if any((r, c) in input_guard for r, c in cells):
        return None
    if any((r, c) in final_occupied for r, c in _rect_cells(final_body)):
        return None
    return input_body, cells


def _render_input(sliders: list[Slider], shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    for slider in sliders:
        for r, c in _rect_cells(slider.body):
            grid[r, c] = slider.color
        for r, c in _tail_cells(slider.body, slider.direction, slider.length):
            grid[r, c] = slider.color
    return grid


def _body_extent_along_direction(slider: Slider) -> int:
    r0, c0, r1, c1 = slider.body
    dr, dc = slider.direction
    return (r1 - r0 + 1) if dr != 0 else (c1 - c0 + 1)


def _semantic_checks(inp: np.ndarray, expected: list[Slider], want_overlap: bool) -> bool:
    counts = np.bincount(inp.ravel(), minlength=10)
    if counts[0] <= max(counts[1:]):
        return False

    try:
        parsed = _parse_sliders(inp)
    except Exception:
        return False

    if len(parsed) != len(expected):
        return False

    by_color = {slider.color: slider for slider in parsed}
    for slider in expected:
        got = by_color.get(slider.color)
        if got is None or got.body != slider.body or got.direction != slider.direction or got.length != slider.length:
            return False

    lengths = {slider.length for slider in parsed}
    if len(lengths) < 2:
        return False

    same_dir_diff_len = any(
        parsed[i].direction == parsed[j].direction and parsed[i].length != parsed[j].length
        for i in range(len(parsed))
        for j in range(i + 1, len(parsed))
    )
    if not same_dir_diff_len:
        return False

    try:
        out = main(inp)
    except Exception:
        return False

    if np.array_equal(out, inp):
        return False

    body_only = _render_scene(parsed, inp.shape, include_expansion=False)
    if np.array_equal(out, body_only):
        return False

    no_slide = _render_scene(
        parsed,
        inp.shape,
        slide_lengths={slider.color: 0 for slider in parsed},
        include_expansion=True,
    )
    if np.array_equal(out, no_slide):
        return False

    fixed_period = parsed[0].length
    wrong_period = _render_scene(
        parsed,
        inp.shape,
        period_lengths={slider.color: fixed_period for slider in parsed},
        include_expansion=True,
    )
    if np.array_equal(out, wrong_period):
        return False

    body_periods = {slider.color: _body_extent_along_direction(slider) for slider in parsed}
    if any(body_periods[slider.color] != slider.length for slider in parsed):
        body_period_out = _render_scene(parsed, inp.shape, period_lengths=body_periods, include_expansion=True)
        if np.array_equal(out, body_period_out):
            return False

    body_mask = np.zeros(inp.shape, dtype=bool)
    moved_boxes: dict[int, BBox] = {}
    for slider in parsed:
        moved_box = _shift_bbox(
            slider.body,
            slider.direction[0] * slider.length,
            slider.direction[1] * slider.length,
        )
        moved_boxes[slider.color] = moved_box
        for r, c in _rect_cells(moved_box):
            body_mask[r, c] = True

    per_slider_halo = []
    for slider in parsed:
        raw = _stepped_claim_cells(moved_boxes[slider.color], slider.direction, slider.length, inp.shape)
        visible = {(r, c) for (r, c) in raw if not body_mask[r, c]}
        per_slider_halo.append(len(visible))
    if min(per_slider_halo) <= 0:
        return False

    magenta_count = int(np.count_nonzero(out == MAGENTA))
    if want_overlap != (magenta_count > 0):
        return False
    if magenta_count > max(12, inp.size // 12):
        return False

    density = int(np.count_nonzero(out)) / float(out.size)
    if density > 0.58:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    count = random.choice(OBJECT_COUNTS)
    colors = random.sample(PALETTE, k=count)
    specs = _make_specs(colors)
    want_overlap = random.random() < 0.45

    sliders: list[Slider] = []
    input_guard: set[tuple[int, int]] = set()
    final_occupied: set[tuple[int, int]] = set()
    placed_finals: list[BBox] = []

    for i, spec in enumerate(specs):
        placed = False
        for _ in range(160):
            if i == 1 and want_overlap and spec.direction == specs[0].direction and random.random() < 0.8:
                final_body = _guided_final_bbox(spec, (h, w), placed_finals[0])
            else:
                final_body = _random_final_bbox(spec, (h, w))

            maybe = _can_place(final_body, spec, (h, w), input_guard, final_occupied)
            if maybe is None:
                continue

            input_body, cells = maybe
            slider = Slider(color=spec.color, body=input_body, direction=spec.direction, length=spec.length)
            sliders.append(slider)
            input_guard |= _guard_zone(cells, h, w)
            final_occupied |= _rect_cells(final_body)
            placed_finals.append(final_body)
            placed = True
            break

        if not placed:
            return None

    inp = _render_input(sliders, (h, w))
    if not _semantic_checks(inp, sliders, want_overlap):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
