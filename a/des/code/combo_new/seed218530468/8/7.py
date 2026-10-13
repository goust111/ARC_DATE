from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
COLORS = tuple(c for c in range(10) if c not in {BG, FRAME})
GEN_RETRIES = 400
PLACEMENT_TRIES = 240


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _find_onions(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    by_color: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in colors:
        frames: list[tuple[int, int, int, int]] = []
        for comp in _components(grid == color):
            bbox = _rect_frame_bbox(comp)
            if bbox is not None:
                frames.append(bbox)
        if frames:
            by_color[int(color)] = frames

    assert by_color, "expected at least one rectangular frame color"
    scored = sorted(((len(v), k) for k, v in by_color.items()), reverse=True)
    assert len(scored) == 1 or scored[0][0] > scored[1][0], "frame color must be unambiguous"
    frame_color = int(scored[0][1])

    grouped: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for bbox in by_color[frame_color]:
        r0, c0, r1, c1 = bbox
        grouped.setdefault((r0 + r1, c0 + c1), []).append(bbox)

    onions: list[dict[str, object]] = []
    for center_key, bboxes in grouped.items():
        layers = sorted(
            bboxes,
            key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1), b[0], b[1]),
        )
        assert 2 <= len(layers) <= 4, "each onion must have between 2 and 4 layers"
        for inner, outer in zip(layers, layers[1:]):
            assert outer[0] == inner[0] - 2
            assert outer[1] == inner[1] - 2
            assert outer[2] == inner[2] + 2
            assert outer[3] == inner[3] + 2

        cr = center_key[0] // 2
        cc = center_key[1] // 2
        core_color = int(grid[cr, cc])
        assert core_color not in {BG, frame_color}, "core must be a colored pixel"

        ir0, ic0, ir1, ic1 = layers[0]
        inner = grid[ir0 + 1 : ir1, ic0 + 1 : ic1]
        non_bg = np.argwhere(inner != BG)
        assert non_bg.shape[0] == 1, "innermost region must contain exactly one core pixel"
        rr, cc_local = map(int, non_bg[0])
        assert (ir0 + 1 + rr, ic0 + 1 + cc_local) == (cr, cc), "core must sit at the shared center"
        assert int(inner[rr, cc_local]) == core_color

        onions.append(
            {
                "core_color": core_color,
                "core": (int(cr), int(cc)),
                "layers": layers,
            }
        )

    assert onions, "expected at least one onion structure"
    return frame_color, onions


def _parse_palette(
    grid: np.ndarray,
    frame_color: int,
    onions: list[dict[str, object]],
) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
    mask = (grid != BG) & (grid != frame_color)
    core_colors = {int(onion["core_color"]) for onion in onions}
    for onion in onions:
        r, c = onion["core"]
        mask[int(r), int(c)] = False

    palette_cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    assert palette_cells, "expected a corner palette"

    rows = [r for r, _c in palette_cells]
    cols = [c for _r, c in palette_cells]
    h, w = grid.shape
    touches_left = min(cols) == 0
    touches_right = max(cols) == w - 1
    touches_top = min(rows) == 0
    touches_bottom = max(rows) == h - 1
    assert (touches_left or touches_right) and (touches_top or touches_bottom), "palette must sit in a corner"
    assert touches_left != touches_right, "palette must occupy one horizontal side"
    reverse = touches_right

    mapping: dict[int, list[int]] = {}
    for comp in _components(mask):
        row_ids = {r for r, _c in comp}
        assert len(row_ids) == 1, "each palette entry must be one horizontal row"
        row = next(iter(row_ids))
        comp_cols = sorted(c for _r, c in comp)
        assert comp_cols == list(range(comp_cols[0], comp_cols[-1] + 1)), "palette rows must be contiguous"
        ordered_cols = list(reversed(comp_cols)) if reverse else comp_cols
        values = [int(grid[row, c]) for c in ordered_cols]
        key = int(values[0])
        seq = [int(v) for v in values[1:]]
        assert key in core_colors, "palette key must be a core color"
        assert 2 <= len(seq) <= 4, "palette rows must encode 2-4 layer colors"
        assert len(seq) == len(set(seq)), "palette sequence colors must be distinct within a row"
        assert key not in mapping, "palette keys must be unique"
        mapping[key] = seq

    assert set(mapping) == core_colors, "every core color must appear exactly once in the palette"
    return mapping, palette_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey nested rectangle onion from the palette row selected by its core pixel, then erase the palette.

    Input structure:
    - The background is black.
    - Several non-overlapping onions appear, each made of 2-4 concentric grey rectangular frame layers.
    - Every frame layer is one cell thick, and adjacent layers are separated by one background cell.
    - The center of each onion contains exactly one colored core pixel.
    - A palette occupies one corner as separated horizontal rows: each row starts with a core color and continues with that onion's inner-to-outer repaint colors.

    Rule:
    - Detect the shared grey frame color and group its rectangular frame components by common center to recover each onion's ordered layers.
    - Remove the core pixels from consideration, then read the remaining corner rows as the palette; the first pixel of a row is the lookup key and the rest are the recolor sequence.
    - For every onion, match its core color to the palette row and repaint the innermost frame with the first sequence color, the next frame with the second color, and so on outwards.
    - Leave each core pixel unchanged and erase the entire palette region to black.

    Key insight:
    - The grey rings themselves do not reveal their target colors; the only way to color them correctly is to recover the onion center, use that core as a palette key, and then apply the retrieved sequence from the inside out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, onions = _find_onions(g)
    palette, palette_cells = _parse_palette(g, frame_color, onions)

    out = g.copy()
    for r, c in palette_cells:
        out[int(r), int(c)] = BG

    for onion in onions:
        core_color = int(onion["core_color"])
        layers = list(onion["layers"])
        seq = palette[core_color]
        assert len(seq) == len(layers), "palette length must match the onion depth"
        for color, bbox in zip(seq, layers):
            _paint_frame(out, bbox, int(color))

    return out


def _inner_choices(n_onions: int, layers: int) -> tuple[int, ...]:
    if n_onions == 1:
        if layers == 4:
            return (3, 5)
        return (3, 5, 7)
    if n_onions == 2:
        if layers == 2:
            return (3, 5, 7)
        if layers == 3:
            return (3, 5)
        return (3,)
    if layers == 2:
        return (3, 5)
    return (3,)


def _make_specs() -> list[dict[str, object]]:
    n_onions = random.choice((1, 2, 3))
    core_colors = random.sample(list(COLORS), n_onions)
    seq_pool = [c for c in COLORS if c not in core_colors]

    specs: list[dict[str, object]] = []
    for core_color in core_colors:
        layers = random.randint(2, 4)
        inner_h = random.choice(_inner_choices(n_onions, layers))
        inner_w = random.choice(_inner_choices(n_onions, layers))
        outer_h = inner_h + 4 * (layers - 1)
        outer_w = inner_w + 4 * (layers - 1)
        seq = random.sample(seq_pool, layers)
        specs.append(
            {
                "core_color": int(core_color),
                "layers": int(layers),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "sequence": [int(x) for x in seq],
            }
        )
    return specs


def _palette_layout(
    h: int,
    w: int,
    corner: str,
    specs: list[dict[str, object]],
) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]], tuple[int, int, int, int]]:
    ordered = specs[:]
    random.shuffle(ordered)

    rows = len(ordered)
    row_lengths = [1 + int(spec["layers"]) for spec in ordered]
    palette_h = 2 * rows - 1
    palette_w = max(row_lengths)
    if palette_h > h or palette_w > w:
        raise ValueError("palette does not fit")

    top = corner in {"tl", "tr"}
    left = corner in {"tl", "bl"}
    r_start = 0 if top else h - palette_h
    c_start = 0 if left else w - palette_w
    bbox = (r_start, c_start, r_start + palette_h - 1, c_start + palette_w - 1)

    entries: list[tuple[int, int, int]] = []
    cells: set[tuple[int, int]] = set()
    for idx, spec in enumerate(ordered):
        values = [int(spec["core_color"]), *[int(x) for x in spec["sequence"]]]
        row = r_start + 2 * idx
        if left:
            cols = [c_start + j for j in range(len(values))]
        else:
            cols = [w - 1 - j for j in range(len(values))]
        for col, value in zip(cols, values):
            entries.append((int(row), int(col), int(value)))
            cells.add((int(row), int(col)))
    return entries, cells, bbox


def _place_onions(
    h: int,
    w: int,
    specs: list[dict[str, object]],
    reserved: np.ndarray,
) -> bool:
    order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["outer_h"]) * int(specs[idx]["outer_w"]),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if outer_h > h or outer_w > w:
            return False

        min_r = 1 if h - outer_h >= 2 else 0
        max_r = h - outer_h - 1 if h - outer_h >= 2 else h - outer_h
        min_c = 1 if w - outer_w >= 2 else 0
        max_c = w - outer_w - 1 if w - outer_w >= 2 else w - outer_w
        if min_r > max_r or min_c > max_c:
            return False

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(min_r, max_r)
            c0 = random.randint(min_c, max_c)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h - 1, r1 + 1)
            pc1 = min(w - 1, c1 + 1)
            if reserved[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            bboxes: list[tuple[int, int, int, int]] = []
            layers = int(spec["layers"])
            for layer_idx in range(layers):
                inset = 2 * (layers - 1 - layer_idx)
                bboxes.append((r0 + inset, c0 + inset, r1 - inset, c1 - inset))
            spec["top_left"] = (int(r0), int(c0))
            spec["core_pos"] = (int(r0 + outer_h // 2), int(c0 + outer_w // 2))
            spec["bboxes_inner_to_outer"] = bboxes
            placed = True
            break
        if not placed:
            return False

    return True


def _render_input(
    h: int,
    w: int,
    specs: list[dict[str, object]],
    palette_entries: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c, value in palette_entries:
        grid[int(r), int(c)] = int(value)
    for spec in specs:
        for bbox in spec["bboxes_inner_to_outer"]:
            _paint_frame(grid, bbox, FRAME)
        cr, cc = spec["core_pos"]
        grid[int(cr), int(cc)] = int(spec["core_color"])
    return grid


def _expected_output(
    inp: np.ndarray,
    specs: list[dict[str, object]],
    palette_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in palette_cells:
        out[int(r), int(c)] = BG
    for spec in specs:
        seq = [int(x) for x in spec["sequence"]]
        bboxes = list(spec["bboxes_inner_to_outer"])
        assert len(seq) == len(bboxes)
        for color, bbox in zip(seq, bboxes):
            _paint_frame(out, bbox, int(color))
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _make_specs()
    corner = random.choice(("tl", "tr", "bl", "br"))

    palette_h = 2 * len(specs) - 1
    palette_w = max(1 + int(spec["layers"]) for spec in specs)
    onion_area = sum((int(spec["outer_h"]) + 2) * (int(spec["outer_w"]) + 2) for spec in specs)
    side_lo = max(palette_h + 1, palette_w + 1, int(np.ceil(np.sqrt(onion_area))) + 2)
    side_lo = min(max(side_lo, 12 + 3 * len(specs)), 30)

    h = random.randint(side_lo, 30)
    w = random.randint(side_lo, 30)

    try:
        palette_entries, palette_cells, palette_bbox = _palette_layout(h, w, corner, specs)
    except ValueError:
        return None

    reserved = np.zeros((h, w), dtype=bool)
    pr0, pc0, pr1, pc1 = palette_bbox
    reserved[max(0, pr0 - 1) : min(h, pr1 + 2), max(0, pc0 - 1) : min(w, pc1 + 2)] = True

    if not _place_onions(h, w, specs, reserved):
        return None

    inp = _render_input(h, w, specs, palette_entries)
    expected = _expected_output(inp, specs, palette_cells)

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid onion-palette puzzle")
