import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6
FRAGMENT_COLORS = (1, 2, 3, 4, 7, 8, 9)
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _component_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(bool).tobytes()


def _intervals(non_separator: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for i, keep in enumerate(non_separator):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(non_separator) - 1))
    return intervals


def _split_panels(grid: np.ndarray) -> list[list[np.ndarray]]:
    sep_rows = np.all(grid == MAGENTA, axis=1)
    sep_cols = np.all(grid == MAGENTA, axis=0)
    row_spans = _intervals(~sep_rows)
    col_spans = _intervals(~sep_cols)
    assert row_spans and col_spans, "expected at least one non-separator panel"
    panels: list[list[np.ndarray]] = []
    for r0, r1 in row_spans:
        row: list[np.ndarray] = []
        for c0, c1 in col_spans:
            panel = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(panel == MAGENTA), "magenta is reserved for separator lines"
            row.append(panel)
        panels.append(row)
    return panels


def _complete_panel(panel: np.ndarray) -> np.ndarray:
    grey = panel == GREY
    assert np.any(grey), "each panel must contain one grey template"
    tr0, tc0, tr1, tc1 = _bbox(np.argwhere(grey))

    template = panel[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert np.all((template == GREY) | (template == BLACK)), "template bbox must contain only grey and black gaps"

    gap_mask = template != GREY
    gap_items: list[dict[str, object]] = []
    covered_gaps = np.zeros_like(gap_mask, dtype=bool)
    for coords in _components(gap_mask):
        bbox, mask = _component_mask(coords)
        r0, c0, r1, c1 = bbox
        touches_edge = r0 == 0 or c0 == 0 or r1 == gap_mask.shape[0] - 1 or c1 == gap_mask.shape[1] - 1
        assert touches_edge, "template gaps must touch the template boundary"
        covered_gaps[coords[:, 0], coords[:, 1]] = True
        gap_items.append({"bbox": bbox, "mask": mask, "key": _mask_key(mask)})
    assert gap_items, "expected at least one boundary gap"
    assert np.array_equal(covered_gaps, gap_mask), "all non-grey cells in the template bbox must be gaps"

    colored = (panel != BLACK) & (panel != GREY) & (panel != MAGENTA)
    fragments_by_key: dict[tuple[tuple[int, int], bytes], list[dict[str, object]]] = {}
    for coords in _components(colored):
        bbox, mask = _component_mask(coords)
        r0, c0, r1, c1 = bbox
        assert r1 < tr0 or tr1 < r0 or c1 < tc0 or tc1 < c0, "fragments must start outside the template bbox"
        patch = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
        item = {"bbox": bbox, "mask": mask, "patch": patch, "key": _mask_key(mask)}
        fragments_by_key.setdefault(item["key"], []).append(item)

    assert sum(len(v) for v in fragments_by_key.values()) == len(gap_items), "each gap needs exactly one fragment"

    used: set[int] = set()
    out = template.copy()
    for gap in gap_items:
        matches = fragments_by_key.get(gap["key"], [])
        available = [frag for frag in matches if id(frag) not in used]
        assert len(available) == 1, "gap shape must identify one unique fragment"
        frag = available[0]
        used.add(id(frag))

        gr0, gc0, gr1, gc1 = gap["bbox"]  # type: ignore[misc]
        gmask = gap["mask"]  # type: ignore[assignment]
        fmask = frag["mask"]  # type: ignore[assignment]
        patch = frag["patch"]  # type: ignore[assignment]
        assert isinstance(gmask, np.ndarray) and isinstance(fmask, np.ndarray) and isinstance(patch, np.ndarray)
        assert np.array_equal(gmask, fmask), "fragment mask must be exactly complementary to the gap"
        dest = out[gr0 : gr1 + 1, gc0 : gc1 + 1]
        assert np.all(dest[gmask] == BLACK), "gap destination must be empty before filling"
        dest[gmask] = patch[fmask]

    assert len(used) == len(gap_items), "all fragments must be consumed"
    return out


def _tile_outputs(outputs: list[list[np.ndarray]]) -> np.ndarray:
    if len(outputs) == 1 and len(outputs[0]) == 1:
        return outputs[0][0]

    row_heights = [max(int(tile.shape[0]) for tile in row) for row in outputs]
    col_widths = [max(int(outputs[r][c].shape[1]) for r in range(len(outputs))) for c in range(len(outputs[0]))]
    out_h = sum(row_heights) + len(row_heights) - 1
    out_w = sum(col_widths) + len(col_widths) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)

    rr = 0
    for r, row in enumerate(outputs):
        cc = 0
        for c, tile in enumerate(row):
            h, w = tile.shape
            out[rr : rr + h, cc : cc + w] = tile
            cc += col_widths[c] + 1
        rr += row_heights[r] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each panel's grey boundary gaps with shape-matched colored fragments and pack the completed templates.

    Input structure:
    - The grid has a black background and full magenta separator rows or columns dividing it into panels.
    - Each panel contains one large grey template whose bounding box contains rectangular black gaps touching its boundary.
    - Outside the grey template in the same panel are small colored fragment objects, one per gap.

    Rule:
    - Split the grid into panels by the magenta separator lines and process panels in row-major order.
    - In each panel, crop the grey template bbox, find every black boundary gap, and match it to the unique colored fragment with the same translated shape.
    - Paste the matched fragment into the gap without changing its colors, leaving the original grey template pixels unchanged.
    - Return only the completed template crops, arranged in the same panel order with black spacing between adjacent crops.

    Key insight:
    - The colored fragments are shape keys for the template's missing boundary pieces; their positions are irrelevant except for revealing which colored patch fills each gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    panels = _split_panels(g)
    completed = [[_complete_panel(panel) for panel in row] for row in panels]
    out = _tile_outputs(completed)
    assert out.ndim == 2 and max(out.shape) <= 30, "output must fit ARC dimensions"
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    return len(_components(mask)) == 1


def _gap_candidates(th: int, tw: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []
    for side in ("top", "bottom"):
        max_depth = min(2, th - 3)
        max_width = min(3, tw - 2)
        for depth in range(1, max_depth + 1):
            for width in range(1, max_width + 1):
                for start in range(1, tw - width):
                    if side == "top":
                        candidates.append((side, (0, start, depth - 1, start + width - 1)))
                    else:
                        candidates.append((side, (th - depth, start, th - 1, start + width - 1)))
    for side in ("left", "right"):
        max_depth = min(2, tw - 3)
        max_height = min(3, th - 2)
        for depth in range(1, max_depth + 1):
            for height in range(1, max_height + 1):
                for start in range(1, th - height):
                    if side == "left":
                        candidates.append((side, (start, 0, start + height - 1, depth - 1)))
                    else:
                        candidates.append((side, (start, tw - depth, start + height - 1, tw - 1)))
    _RNG.shuffle(candidates)
    return candidates


def _build_gap_mask(th: int, tw: int, count: int) -> np.ndarray | None:
    gap = np.zeros((th, tw), dtype=bool)
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    placed = 0
    for _side, (r0, c0, r1, c1) in _gap_candidates(th, tw):
        rect = np.zeros((th, tw), dtype=bool)
        rect[r0 : r1 + 1, c0 : c1 + 1] = True
        shape_key = _mask_key(rect[r0 : r1 + 1, c0 : c1 + 1])
        if shape_key in used_shapes or np.any(gap & rect):
            continue

        trial = gap | rect
        if len(_components(trial)) != placed + 1:
            continue

        grey = ~trial
        grey_coords = np.argwhere(grey)
        if not _connected(grey):
            continue
        if _bbox(grey_coords) != (0, 0, th - 1, tw - 1):
            continue

        gap = trial
        used_shapes.add(shape_key)
        placed += 1
        if placed == count:
            return gap
    return None


def _can_place(panel: np.ndarray, mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    ph, pw = panel.shape
    if r < 0 or c < 0 or r + h > ph or c + w > pw:
        return False
    if np.any(panel[r : r + h, c : c + w][mask] != BLACK):
        return False
    pr0 = max(0, r - 1)
    pc0 = max(0, c - 1)
    pr1 = min(ph, r + h + 1)
    pc1 = min(pw, c + w + 1)
    return bool(np.all(panel[pr0:pr1, pc0:pc1] == BLACK))


def _place_fragment(panel: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = mask.shape
    ph, pw = panel.shape
    for _ in range(100):
        r = int(_RNG.integers(0, ph - h + 1))
        c = int(_RNG.integers(0, pw - w + 1))
        if _can_place(panel, mask, r, c):
            panel[r : r + h, c : c + w][mask] = color
            return True
    return False


def _make_panel(ph: int, pw: int) -> np.ndarray | None:
    max_th = min(8, ph - 4)
    max_tw = min(8, pw - 4)
    if max_th < 5 or max_tw < 5:
        return None
    th = int(_RNG.integers(5, max_th + 1))
    tw = int(_RNG.integers(5, max_tw + 1))
    gap_count = int(_RNG.integers(2, 5 if min(th, tw) >= 7 else 4))

    gap = _build_gap_mask(th, tw, gap_count)
    if gap is None:
        return None

    panel = np.full((ph, pw), BLACK, dtype=int)
    top = int(_RNG.integers(1, ph - th))
    left = int(_RNG.integers(1, pw - tw))
    template = np.full((th, tw), GREY, dtype=int)
    template[gap] = BLACK
    panel[top : top + th, left : left + tw] = template

    gap_items: list[np.ndarray] = []
    for coords in _components(gap):
        _bbox_tuple, mask = _component_mask(coords)
        gap_items.append(mask)
    _RNG.shuffle(gap_items)

    colors = list(FRAGMENT_COLORS)
    _RNG.shuffle(colors)
    for i, mask in enumerate(gap_items):
        color = int(colors[i % len(colors)])
        if not _place_fragment(panel, mask, color):
            return None
    return panel


def _layout() -> tuple[int, int, int, int]:
    options = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
    rows, cols = options[int(_RNG.integers(0, len(options)))]
    max_ph = (30 - (rows - 1)) // rows
    max_pw = (30 - (cols - 1)) // cols
    min_ph = 9 if rows == 3 else 11
    min_pw = 9 if cols == 3 else 11
    ph = int(_RNG.integers(min_ph, min(16, max_ph) + 1))
    pw = int(_RNG.integers(min_pw, min(16, max_pw) + 1))
    return rows, cols, ph, pw


def _assemble(panels: list[list[np.ndarray]], ph: int, pw: int) -> np.ndarray:
    rows = len(panels)
    cols = len(panels[0])
    h = rows * ph + rows - 1
    w = cols * pw + cols - 1
    grid = np.full((h, w), MAGENTA, dtype=int)
    for r in range(rows):
        for c in range(cols):
            rr = r * (ph + 1)
            cc = c * (pw + 1)
            grid[rr : rr + ph, cc : cc + pw] = panels[r][c]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        rows, cols, ph, pw = _layout()
        panels: list[list[np.ndarray]] = []
        ok = True
        for _r in range(rows):
            row: list[np.ndarray] = []
            for _c in range(cols):
                panel = _make_panel(ph, pw)
                if panel is None:
                    ok = False
                    break
                row.append(panel)
            if not ok:
                break
            panels.append(row)
        if not ok:
            continue

        grid = _assemble(panels, ph, pw)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid panel template puzzle")
