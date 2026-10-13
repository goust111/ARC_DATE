from __future__ import annotations

import hashlib
import random

import numpy as np


BG = 0
LINE = 5
RECT_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 600

_SEEN_INPUT_HASHES: set[str] = set()
_SEEN_OUTPUT_HASHES: set[str] = set()
_GEN_CALL_COUNT = 0


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode(), usedforsecurity=False).hexdigest()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _boxes_have_clear_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _parse_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    comps = _components(g != BG)
    assert 2 <= len(comps) <= 3, "expected 2-3 colored rectangles"

    rects: list[dict[str, int]] = []
    seen_colors: set[int] = set()
    for comp in comps:
        coords = np.asarray(comp, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5, "each rectangle must be large enough for a protected interior"
        assert h != w, "rectangles must be non-square"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(v) for v in np.unique(patch) if int(v) != BG]
        assert len(colors) == 1, "each object must be a single-color rectangle with one black marker"
        color = colors[0]
        assert color not in seen_colors, "rectangle colors must be distinct"
        seen_colors.add(color)

        assert np.all((patch == BG) | (patch == color)), "rectangle bbox must contain only its color and the marker"
        assert np.all(patch[0, :] == color) and np.all(patch[-1, :] == color), "top/bottom border must be intact"
        assert np.all(patch[:, 0] == color) and np.all(patch[:, -1] == color), "left/right border must be intact"

        zeros = np.argwhere(patch == BG)
        assert zeros.shape == (1, 2), "each rectangle must contain exactly one black marker"
        local_r, local_c = map(int, zeros[0])
        assert 2 <= local_r <= h - 3 and 2 <= local_c <= w - 3, "marker must avoid the interior outer ring"
        assert len(comp) == h * w - 1, "rectangle should be solid except for the marker hole"

        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": h,
                "w": w,
                "color": color,
                "marker_r": r0 + local_r,
                "marker_c": c0 + local_c,
            }
        )

    rects.sort(key=lambda rect: (rect["r0"], rect["c0"]))
    for i, rect_a in enumerate(rects):
        box_a = (rect_a["r0"], rect_a["c0"], rect_a["r1"], rect_a["c1"])
        for rect_b in rects[i + 1 :]:
            box_b = (rect_b["r0"], rect_b["c0"], rect_b["r1"], rect_b["c1"])
            assert _boxes_have_clear_gap(box_a, box_b, gap=1), "rectangles must have at least one-cell separation"
    return rects


def _apply_rule(grid: np.ndarray, rects: list[dict[str, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        color = rect["color"]
        marker_r, marker_c = rect["marker_r"], rect["marker_c"]
        h, w = rect["h"], rect["w"]

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = color
        rows, cols = np.indices(interior.shape)
        odd_mask = (rows + cols) % 2 == 1
        line_mask = np.zeros_like(interior, dtype=bool)

        if w > h:
            line_mask[:, marker_c - (c0 + 1)] = True
        else:
            line_mask[marker_r - (r0 + 1), :] = True

        interior[odd_mask & ~line_mask] = BG
        interior[line_mask] = LINE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an aspect-ratio-gated grey divider inside each colored rectangle, then etch the remaining interior by odd parity.

    Input structure:
    - The grid has a black background plus 2-3 separated filled non-square rectangles, each with a distinct color.
    - Every rectangle has a 1-cell border of its own color and exactly one black marker pixel inside.
    - The marker is not on the interior's outer ring, so there is editable space on both sides of the future divider.

    Rule:
    - For each rectangle, keep its border unchanged and treat the area inside the border as the interior.
    - If the rectangle is wider than tall, extend the marker into a full-height grey vertical line through the interior; if it is taller than wide, extend it into a full-width grey horizontal line.
    - After drawing that line, etch the interior by the interior top-left anchored checkerboard: every odd-parity interior cell becomes black unless it lies on the grey line, while even-parity interior cells stay the rectangle color.

    Key insight:
    - The marker decides the divider position, but the rectangle's own aspect ratio decides the divider direction; the later parity etching interacts with that divider because grey line cells are protected from being carved away.
    """
    rects = _parse_rectangles(np.asarray(grid, dtype=int))
    return _apply_rule(grid, rects)


def _choose_orientations() -> list[str]:
    global _GEN_CALL_COUNT
    rect_count = 2 if _GEN_CALL_COUNT % 4 in (0, 1) else 3
    orientations = ["wide", "tall"]
    if rect_count == 3:
        orientations.append("wide" if _GEN_CALL_COUNT % 2 == 0 else "tall")
    random.shuffle(orientations)
    return orientations


def _sample_rect_spec(orientation: str) -> dict[str, int | str]:
    if orientation == "wide":
        h = random.randint(5, 7)
        w = random.randint(max(6, h + 1), 10)
    else:
        w = random.randint(5, 7)
        h = random.randint(max(6, w + 1), 10)
    return {"orientation": orientation, "h": h, "w": w}


def _sample_canvas(rect_specs: list[dict[str, int | str]]) -> tuple[int, int]:
    total_area = sum(int(spec["h"]) * int(spec["w"]) for spec in rect_specs)
    max_h = max(int(spec["h"]) for spec in rect_specs)
    max_w = max(int(spec["w"]) for spec in rect_specs)
    base = max(16, max_h + 4, max_w + 4, int(np.ceil(np.sqrt(total_area * 2.4))))
    h = min(28, base + random.randint(0, 3))
    w = min(28, base + random.randint(0, 3))
    return h, w


def _place_rectangles(
    canvas_shape: tuple[int, int], rect_specs: list[dict[str, int | str]]
) -> list[dict[str, int | str]] | None:
    h, w = canvas_shape
    placed: list[dict[str, int | str]] = []
    for spec in sorted(rect_specs, key=lambda item: int(item["h"]) * int(item["w"]), reverse=True):
        rh, rw = int(spec["h"]), int(spec["w"])
        row_max = h - rh - 1
        col_max = w - rw - 1
        if row_max < 1 or col_max < 1:
            return None
        candidates = [
            (r0, c0)
            for r0 in range(1, row_max + 1)
            for c0 in range(1, col_max + 1)
        ]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if all(
                _boxes_have_clear_gap(
                    box,
                    (int(prev["r0"]), int(prev["c0"]), int(prev["r1"]), int(prev["c1"])),
                    gap=1,
                )
                for prev in placed
            ):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        r0, c0 = chosen
        placed.append(
            {
                **spec,
                "r0": r0,
                "c0": c0,
                "r1": r0 + rh - 1,
                "c1": c0 + rw - 1,
            }
        )
    random.shuffle(placed)
    return placed


def _pick_marker(rect: dict[str, int | str], preferred_parity: int) -> tuple[int, int]:
    r0, c0, r1, c1 = (int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"]))
    local_rows = list(range(2, (r1 - r0 + 1) - 2))
    local_cols = list(range(2, (c1 - c0 + 1) - 2))
    orientation = str(rect["orientation"])

    if orientation == "wide":
        good_cols = [c for c in local_cols if (c - 1) % 2 == preferred_parity]
        marker_c = random.choice(good_cols or local_cols)
        marker_r = random.choice(local_rows)
    else:
        good_rows = [r for r in local_rows if (r - 1) % 2 == preferred_parity]
        marker_r = random.choice(good_rows or local_rows)
        marker_c = random.choice(local_cols)
    return r0 + marker_r, c0 + marker_c


def _render_input(placed: list[dict[str, int | str]], canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = canvas_shape
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(RECT_COLOR_POOL, len(placed))
    parity_seed = _GEN_CALL_COUNT % 2

    for idx, (rect, color) in enumerate(zip(placed, colors)):
        r0, c0, r1, c1 = (int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"]))
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        marker_r, marker_c = _pick_marker(rect, preferred_parity=(parity_seed + idx) % 2)
        grid[marker_r, marker_c] = BG
    return grid


def _validate_generated_example(inp: np.ndarray, out: np.ndarray) -> None:
    rects = _parse_rectangles(inp)
    orientations = {"wide" if rect["w"] > rect["h"] else "tall" for rect in rects}
    assert orientations == {"wide", "tall"}, "generator must cover both orientation branches in each example"

    covered = np.zeros(inp.shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        color = rect["color"]
        marker_r, marker_c = rect["marker_r"], rect["marker_c"]
        h, w = rect["h"], rect["w"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

        in_patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(out_patch[0, :], np.full(w, color)), "top border must stay unchanged"
        assert np.array_equal(out_patch[-1, :], np.full(w, color)), "bottom border must stay unchanged"
        assert np.array_equal(out_patch[:, 0], np.full(h, color)), "left border must stay unchanged"
        assert np.array_equal(out_patch[:, -1], np.full(h, color)), "right border must stay unchanged"

        interior = out_patch[1:-1, 1:-1]
        rows, cols = np.indices(interior.shape)
        odd_mask = (rows + cols) % 2 == 1
        line_mask = np.zeros_like(interior, dtype=bool)
        if w > h:
            line_col = marker_c - (c0 + 1)
            line_mask[:, line_col] = True
            assert np.any(odd_mask[:, :line_col]), "left side must contain etchable cells"
            assert np.any(odd_mask[:, line_col + 1 :]), "right side must contain etchable cells"
        else:
            line_row = marker_r - (r0 + 1)
            line_mask[line_row, :] = True
            assert np.any(odd_mask[:line_row, :]), "upper side must contain etchable cells"
            assert np.any(odd_mask[line_row + 1 :, :]), "lower side must contain etchable cells"

        assert np.any(odd_mask & line_mask), "the protected grey line must cover odd-parity cells"
        assert np.any(odd_mask & ~line_mask), "there must be odd-parity cells available for etching"
        assert np.any((~odd_mask) & ~line_mask), "there must be even-parity colored cells left inside"

        expected = np.full_like(interior, color)
        expected[odd_mask & ~line_mask] = BG
        expected[line_mask] = LINE
        assert np.array_equal(interior, expected), "output interior must follow line-then-etch semantics"

        marker_local = (marker_r - r0, marker_c - c0)
        assert in_patch[marker_local] == BG and out_patch[marker_local] == LINE, "marker must become part of the grey line"

    assert np.all(out[~covered] == BG), "background outside rectangles must remain black"


def _gen_attempt() -> np.ndarray | None:
    orientations = _choose_orientations()
    rect_specs = [_sample_rect_spec(orientation) for orientation in orientations]
    canvas_shape = _sample_canvas(rect_specs)
    placed = _place_rectangles(canvas_shape, rect_specs)
    if placed is None:
        return None

    inp = _render_input(placed, canvas_shape)
    try:
        out = main(inp)
        _validate_generated_example(inp, out)
    except AssertionError:
        return None

    input_hash = _hash_grid(inp)
    output_hash = _hash_grid(out)
    if input_hash in _SEEN_INPUT_HASHES or output_hash in _SEEN_OUTPUT_HASHES:
        return None

    _SEEN_INPUT_HASHES.add(input_hash)
    _SEEN_OUTPUT_HASHES.add(output_hash)
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
