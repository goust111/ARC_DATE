from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if _ARC_UTILS_PATH.exists():
    _ARC_UTILS = SourceFileLoader("_arc_utils_fable_8_004_1", str(_ARC_UTILS_PATH)).load_module()
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
else:
    def get_bg(grid: np.ndarray) -> int:
        colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(colors[int(np.argmax(counts))])

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        min_area: int = 1,
    ) -> list[dict]:
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        bg = int(bg)
        seen = np.zeros(g.shape, dtype=bool)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)] if int(connectivity) == 4 else [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        ]
        out = []
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if seen[r, c] or int(g[r, c]) == bg:
                    continue
                color = int(g[r, c])
                stack = [(r, c)]
                coords = []
                seen[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in neighbors:
                        nr = rr + dr
                        nc = cc + dc
                        if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                            continue
                        if seen[nr, nc] or int(g[nr, nc]) == bg:
                            continue
                        if monochromatic and int(g[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
                if len(coords) < int(min_area):
                    continue
                rows = [p[0] for p in coords]
                cols = [p[1] for p in coords]
                out.append(
                    {
                        "color": color if monochromatic else None,
                        "coords": np.array(coords, dtype=int),
                        "bbox": (min(rows), min(cols), max(rows), max(cols)),
                        "area": len(coords),
                    }
                )
        return out


GRID_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 500
ORIENTATIONS = ("H", "V")
RELAY_MODES = ("none", "single", "mixed")
RELAY_MODE_WEIGHTS = (2, 4, 5)
PADDING_RANGE = (1, 3)
SHARED_SPAN_RANGE = (2, 6)
AXIAL_SPAN_RANGE = (2, 6)
NO_RELAY_GAPS = (3, 5, 7, 9, 11)
RELAY_GAPS = tuple(range(6, 12))


def _neighbors4(coord: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = coord
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_solid_rect(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return int(comp["area"]) == h * w and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"]))


def _parse_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert comps, "expected foreground objects"

    rects = [comp for comp in comps if int(comp["area"]) > 1 and _is_solid_rect(g, comp)]
    rect_ids = {id(comp) for comp in rects}
    relays = [comp for comp in comps if id(comp) not in rect_ids]
    assert len(rects) == 2, "expected exactly two nontrivial solid rectangles"
    assert all(int(comp["area"]) == 1 for comp in relays), "every non-rectangle object must be a single relay pixel"

    a, b = rects
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    assert (ar1 - ar0, ac1 - ac0) == (br1 - br0, bc1 - bc0), "rectangles must have the same size"

    same_rows = (ar0, ar1) == (br0, br1)
    same_cols = (ac0, ac1) == (bc0, bc1)
    assert same_rows != same_cols, "rectangles must align horizontally or vertically, but not both"

    if same_rows:
        left, right = sorted(rects, key=lambda comp: comp["bbox"][1])
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        gap = int(rc0 - lc1 - 1)
        assert gap >= 1, "rectangles must have a positive gap"
        center_col = lc1 + 1 + gap // 2 if gap % 2 == 1 else None
        relay_map = {}
        for comp in relays:
            r, c = map(int, comp["coords"][0])
            color = int(comp["color"])
            assert color not in {bg, int(left["color"]), int(right["color"])}
            assert lr0 <= r <= lr1 and lc1 < c < rc0, "relay must lie strictly inside the corridor"
            if center_col is not None:
                assert c != center_col, "odd-gap center column must stay background"
            relay_map[(r, c)] = color
        return {
            "bg": bg,
            "orientation": "H",
            "a_bbox": left["bbox"],
            "a_color": int(left["color"]),
            "b_bbox": right["bbox"],
            "b_color": int(right["color"]),
            "relay_map": relay_map,
            "gap": gap,
        }

    top, bottom = sorted(rects, key=lambda comp: comp["bbox"][0])
    tr0, tc0, tr1, tc1 = top["bbox"]
    br0, bc0, br1, bc1 = bottom["bbox"]
    gap = int(br0 - tr1 - 1)
    assert gap >= 1, "rectangles must have a positive gap"
    center_row = tr1 + 1 + gap // 2 if gap % 2 == 1 else None
    relay_map = {}
    for comp in relays:
        r, c = map(int, comp["coords"][0])
        color = int(comp["color"])
        assert color not in {bg, int(top["color"]), int(bottom["color"])}
        assert tc0 <= c <= tc1 and tr1 < r < br0, "relay must lie strictly inside the corridor"
        if center_row is not None:
            assert r != center_row, "odd-gap center row must stay background"
        relay_map[(r, c)] = color
    return {
        "bg": bg,
        "orientation": "V",
        "a_bbox": top["bbox"],
        "a_color": int(top["color"]),
        "b_bbox": bottom["bbox"],
        "b_color": int(bottom["color"]),
        "relay_map": relay_map,
        "gap": gap,
    }


def _paint_sequence(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    base_color: int,
    relay_map: dict[tuple[int, int], int],
    *,
    mode: str,
) -> None:
    if not coords:
        return

    if mode == "plain":
        for r, c in coords:
            out[r, c] = int(base_color)
        return

    if mode == "stop":
        for r, c in coords:
            if (r, c) in relay_map:
                break
            out[r, c] = int(base_color)
        return

    if mode == "outward":
        current = int(base_color)
        start = 0
        relay_positions = [(idx, relay_map[coord]) for idx, coord in enumerate(coords) if coord in relay_map]
        if not relay_positions:
            for r, c in coords:
                out[r, c] = current
            return
        for idx, color in relay_positions:
            current = int(color)
            for r, c in coords[start : idx + 1]:
                out[r, c] = current
            start = idx + 1
        for r, c in coords[start:]:
            out[r, c] = int(base_color)
        return

    assert mode == "correct"
    current = int(base_color)
    for r, c in coords:
        if (r, c) in relay_map:
            current = int(relay_map[(r, c)])
        out[r, c] = current


def _transform_with_layout(grid: np.ndarray, layout: dict, *, mode: str = "correct") -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    relay_map = dict(layout["relay_map"])
    gap = int(layout["gap"])
    span = gap // 2

    if layout["orientation"] == "H":
        ar0, ac0, ar1, ac1 = map(int, layout["a_bbox"])
        br0, bc0, br1, bc1 = map(int, layout["b_bbox"])
        for r in range(ar0, ar1 + 1):
            left_coords = [(r, c) for c in range(ac1 + 1, ac1 + 1 + span)]
            right_coords = [(r, c) for c in range(bc0 - 1, bc0 - span - 1, -1)]
            _paint_sequence(out, left_coords, int(layout["a_color"]), relay_map, mode=mode)
            _paint_sequence(out, right_coords, int(layout["b_color"]), relay_map, mode=mode)
        return out

    ar0, ac0, ar1, ac1 = map(int, layout["a_bbox"])
    br0, bc0, br1, bc1 = map(int, layout["b_bbox"])
    for c in range(ac0, ac1 + 1):
        top_coords = [(r, c) for r in range(ar1 + 1, ar1 + 1 + span)]
        bottom_coords = [(r, c) for r in range(br0 - 1, br0 - span - 1, -1)]
        _paint_sequence(out, top_coords, int(layout["a_color"]), relay_map, mode=mode)
        _paint_sequence(out, bottom_coords, int(layout["b_color"]), relay_map, mode=mode)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Half-fill the corridor between two aligned solid rectangles, letting relay pixels recolor each side toward the center.

    Input structure:
    - The background contains exactly two solid same-sized single-color rectangles, placed side-by-side or top-to-bottom with matching spans on the perpendicular axis.
    - The corridor between their facing edges may contain scattered single-pixel relay markers whose colors differ from the background and from both rectangle colors.
    - Relay pixels never lie on the odd-gap center line.

    Rule:
    - Process the corridor row-by-row for horizontal layouts, or column-by-column for vertical layouts.
    - Each side fills exactly its own half of the gap from the rectangle edge toward the center; if the gap is odd, the center column or row stays background.
    - Along one row or column, a side starts with its rectangle color; whenever its advancing fill encounters a relay pixel, that cell and all later cells on the same side switch to the relay color until another relay changes the color again.

    Key insight:
    - Relay pixels are not blockers: every side still claims exactly half of the gap, but the color is handed inward from relay to relay along that one line.
    """
    layout = _parse_layout(np.asarray(grid, dtype=int))
    return _transform_with_layout(np.asarray(grid, dtype=int), layout, mode="correct")


def _line_count(spec: dict) -> int:
    return int(spec["rect_h"] if spec["orientation"] == "H" else spec["rect_w"])


def _side_coord(spec: dict, side: str, line_idx: int, offset: int) -> tuple[int, int]:
    if spec["orientation"] == "H":
        row = int(spec["rect_r0"] + line_idx)
        if side == "a":
            col = int(spec["rect_a_c0"] + spec["rect_w"] + offset)
        else:
            col = int(spec["rect_b_c0"] - 1 - offset)
        return row, col

    col = int(spec["rect_c0"] + line_idx)
    if side == "a":
        row = int(spec["rect_a_r0"] + spec["rect_h"] + offset)
    else:
        row = int(spec["rect_b_r0"] - 1 - offset)
    return row, col


def _choose_gap(mode: str) -> int:
    if mode == "none":
        return int(random.choice(NO_RELAY_GAPS))
    return int(random.choice(RELAY_GAPS))


def _sample_offsets(span: int, mode: str) -> tuple[int, ...]:
    if mode == "none" or span == 0:
        return ()

    if mode == "single":
        k = random.choices([0, 1], weights=[3, 7], k=1)[0]
    else:
        max_k = 2 if span >= 2 else 1
        options = [0, 1, max_k]
        weights = [2, 4, 3]
        k = random.choices(options, weights=weights, k=1)[0]
        k = min(int(k), span)

    if k == 0:
        return ()
    if k == 1:
        return (int(random.randrange(span)),)
    return tuple(sorted(random.sample(range(span), k=k)))


def _build_relay_offsets(spec: dict, mode: str) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]] | None:
    span = int(spec["gap"] // 2)
    count = _line_count(spec)
    left_like = [_sample_offsets(span, mode) for _ in range(count)]
    right_like = [_sample_offsets(span, mode) for _ in range(count)]

    if mode == "none":
        return left_like, right_like

    total = sum(len(v) for v in left_like) + sum(len(v) for v in right_like)
    if total == 0:
        return None

    signatures = {(left_like[i], right_like[i]) for i in range(count)}
    if count > 1 and len(signatures) < 2:
        return None

    if span >= 3:
        has_interior = any(
            any(0 < offset < span - 1 for offset in offsets)
            for offsets in [*left_like, *right_like]
        )
        if not has_interior:
            return None

    if mode == "mixed" and span >= 4:
        has_multi = any(len(offsets) >= 2 for offsets in [*left_like, *right_like])
        if not has_multi:
            return None

    return left_like, right_like


def _assign_relay_colors(coords: list[tuple[int, int]], forbidden: set[int]) -> dict[tuple[int, int], int] | None:
    palette = [color for color in range(10) if color not in forbidden]
    if not palette:
        return None

    order = coords[:]
    random.shuffle(order)
    out: dict[tuple[int, int], int] = {}
    for coord in order:
        blocked = {out[nbr] for nbr in _neighbors4(coord) if nbr in out}
        choices = [color for color in palette if color not in blocked]
        if not choices:
            return None
        out[coord] = int(random.choice(choices))
    return out


def _render_input(spec: dict) -> np.ndarray:
    grid = np.full(spec["shape"], int(spec["bg"]), dtype=int)
    if spec["orientation"] == "H":
        r0 = int(spec["rect_r0"])
        a_c0 = int(spec["rect_a_c0"])
        b_c0 = int(spec["rect_b_c0"])
        h = int(spec["rect_h"])
        w = int(spec["rect_w"])
        grid[r0 : r0 + h, a_c0 : a_c0 + w] = int(spec["a_color"])
        grid[r0 : r0 + h, b_c0 : b_c0 + w] = int(spec["b_color"])
    else:
        c0 = int(spec["rect_c0"])
        a_r0 = int(spec["rect_a_r0"])
        b_r0 = int(spec["rect_b_r0"])
        h = int(spec["rect_h"])
        w = int(spec["rect_w"])
        grid[a_r0 : a_r0 + h, c0 : c0 + w] = int(spec["a_color"])
        grid[b_r0 : b_r0 + h, c0 : c0 + w] = int(spec["b_color"])

    for (r, c), color in spec["relay_map"].items():
        grid[int(r), int(c)] = int(color)
    return grid


def _spec_to_layout(spec: dict) -> dict:
    if spec["orientation"] == "H":
        return {
            "bg": int(spec["bg"]),
            "orientation": "H",
            "a_bbox": (
                int(spec["rect_r0"]),
                int(spec["rect_a_c0"]),
                int(spec["rect_r0"] + spec["rect_h"] - 1),
                int(spec["rect_a_c0"] + spec["rect_w"] - 1),
            ),
            "a_color": int(spec["a_color"]),
            "b_bbox": (
                int(spec["rect_r0"]),
                int(spec["rect_b_c0"]),
                int(spec["rect_r0"] + spec["rect_h"] - 1),
                int(spec["rect_b_c0"] + spec["rect_w"] - 1),
            ),
            "b_color": int(spec["b_color"]),
            "relay_map": dict(spec["relay_map"]),
            "gap": int(spec["gap"]),
        }

    return {
        "bg": int(spec["bg"]),
        "orientation": "V",
        "a_bbox": (
            int(spec["rect_a_r0"]),
            int(spec["rect_c0"]),
            int(spec["rect_a_r0"] + spec["rect_h"] - 1),
            int(spec["rect_c0"] + spec["rect_w"] - 1),
        ),
        "a_color": int(spec["a_color"]),
        "b_bbox": (
            int(spec["rect_b_r0"]),
            int(spec["rect_c0"]),
            int(spec["rect_b_r0"] + spec["rect_h"] - 1),
            int(spec["rect_c0"] + spec["rect_w"] - 1),
        ),
        "b_color": int(spec["b_color"]),
        "relay_map": dict(spec["relay_map"]),
        "gap": int(spec["gap"]),
    }


def _output_from_spec(spec: dict, *, mode: str = "correct") -> np.ndarray:
    inp = _render_input(spec)
    return _transform_with_layout(inp, _spec_to_layout(spec), mode=mode)


def _choose_spec() -> dict:
    orientation = random.choice(ORIENTATIONS)
    relay_mode = random.choices(RELAY_MODES, weights=RELAY_MODE_WEIGHTS, k=1)[0]
    rect_h = random.randint(*SHARED_SPAN_RANGE)
    rect_w = random.randint(*AXIAL_SPAN_RANGE)
    gap = _choose_gap(relay_mode)
    pad_top = random.randint(*PADDING_RANGE)
    pad_bottom = random.randint(*PADDING_RANGE)
    pad_left = random.randint(*PADDING_RANGE)
    pad_right = random.randint(*PADDING_RANGE)
    bg = random.randrange(10)
    colors = [color for color in range(10) if color != bg]
    a_color, b_color = random.sample(colors, 2)

    if orientation == "H":
        shape = (pad_top + rect_h + pad_bottom, pad_left + rect_w + gap + rect_w + pad_right)
        spec = {
            "shape": shape,
            "bg": bg,
            "orientation": "H",
            "rect_h": rect_h,
            "rect_w": rect_w,
            "gap": gap,
            "rect_r0": pad_top,
            "rect_a_c0": pad_left,
            "rect_b_c0": pad_left + rect_w + gap,
            "a_color": a_color,
            "b_color": b_color,
        }
    else:
        shape = (pad_top + rect_h + gap + rect_h + pad_bottom, pad_left + rect_w + pad_right)
        spec = {
            "shape": shape,
            "bg": bg,
            "orientation": "V",
            "rect_h": rect_h,
            "rect_w": rect_w,
            "gap": gap,
            "rect_a_r0": pad_top,
            "rect_b_r0": pad_top + rect_h + gap,
            "rect_c0": pad_left,
            "a_color": a_color,
            "b_color": b_color,
        }

    assert max(shape) <= GRID_LIMIT
    relay_offsets = _build_relay_offsets(spec, relay_mode)
    if relay_offsets is None:
        raise AssertionError("relay pattern degenerated")

    left_like, right_like = relay_offsets
    relay_coords = []
    for line_idx, offsets in enumerate(left_like):
        for offset in offsets:
            relay_coords.append(_side_coord(spec, "a", line_idx, int(offset)))
    for line_idx, offsets in enumerate(right_like):
        for offset in offsets:
            relay_coords.append(_side_coord(spec, "b", line_idx, int(offset)))

    relay_map = _assign_relay_colors(relay_coords, {bg, a_color, b_color})
    if relay_map is None:
        raise AssertionError("failed to assign non-merging relay colors")
    spec["relay_map"] = relay_map
    spec["relay_mode"] = relay_mode
    spec["left_offsets"] = left_like
    spec["right_offsets"] = right_like
    return spec


def _validate_spec(spec: dict) -> bool:
    inp = _render_input(spec)
    expected = _output_from_spec(spec, mode="correct")

    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > GRID_LIMIT or max(expected.shape) > GRID_LIMIT:
        return False

    bg = int(spec["bg"])
    counts = np.bincount(inp.ravel(), minlength=10)
    non_bg_max = max(int(counts[i]) for i in range(len(counts)) if i != bg)
    if int(counts[bg]) <= non_bg_max:
        return False

    try:
        parsed = _parse_layout(inp)
    except AssertionError:
        return False
    if not np.array_equal(_transform_with_layout(inp, parsed, mode="correct"), expected):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    relay_count = len(spec["relay_map"])
    if relay_count > 0:
        plain = _output_from_spec(spec, mode="plain")
        stop = _output_from_spec(spec, mode="stop")
        outward = _output_from_spec(spec, mode="outward")
        if np.array_equal(expected, plain):
            return False
        if np.array_equal(expected, stop) and np.array_equal(expected, outward):
            return False

    if spec["relay_mode"] == "none":
        if relay_count != 0 or spec["gap"] % 2 != 1:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            spec = _choose_spec()
        except AssertionError:
            continue
        if _validate_spec(spec):
            return _render_input(spec)
    raise RuntimeError("failed to generate a valid relay half-fill input")
