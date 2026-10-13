import random

import numpy as np


BLACK = 0
GREY = 5
PANEL_SIZE = 7
PANEL_COUNT = 3
MID = PANEL_SIZE // 2
SEPARATOR_ROWS = [PANEL_SIZE, PANEL_SIZE * 2 + 1]
SEPARATOR_COLS = [PANEL_SIZE, PANEL_SIZE * 2 + 1]
GRID_SIZE = PANEL_COUNT * PANEL_SIZE + (PANEL_COUNT - 1)
NON_GREY_NON_BLACK = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 300


def _mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


ODD_BASE = _mask(
    (
        "#####.",
        "#...##",
        "#####.",
    )
)
EVEN_BASE = _mask(
    (
        "#####",
        "#.#.#",
        "#####",
    )
)
ZERO_BASES = (
    _mask(
        (
            "####.",
            "#....",
            "#####",
        )
    ),
    _mask(
        (
            ".####",
            "....#",
            "#####",
        )
    ),
    _mask(
        (
            "###..",
            "..#..",
            "..###",
        )
    ),
)


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    seen = set()
    out = []
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            key = (candidate.shape, candidate.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(candidate.copy())
    return out


ODD_VARIANTS = _variants(ODD_BASE)
EVEN_VARIANTS = _variants(EVEN_BASE)
ZERO_VARIANTS = []
for _base in ZERO_BASES:
    for _variant in _variants(_base):
        key = (_variant.shape, _variant.tobytes())
        if key not in {(m.shape, m.tobytes()) for m in ZERO_VARIANTS}:
            ZERO_VARIANTS.append(_variant.copy())


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_guide(panel: np.ndarray) -> tuple[str, int | None]:
    row_color = int(panel[MID, 0])
    col_color = int(panel[0, MID])
    has_h = row_color not in (BLACK, GREY) and np.all(panel[MID, :] == row_color)
    has_v = col_color not in (BLACK, GREY) and np.all(panel[:, MID] == col_color)
    assert not (has_h and has_v), "panel cannot have both guide lines"
    if has_h:
        return "h", row_color
    if has_v:
        return "v", col_color
    return "none", None


def _line_color(panel: np.ndarray, guide_color: int | None) -> int:
    colors = set(map(int, panel.ravel()))
    colors.discard(BLACK)
    colors.discard(GREY)
    if guide_color is not None:
        colors.discard(int(guide_color))
    assert len(colors) == 1, f"expected one line color, got {colors}"
    return int(next(iter(colors)))


def _closed_regions(panel: np.ndarray, line_color: int) -> list[list[tuple[int, int]]]:
    h, w = panel.shape
    open_mask = panel != int(line_color)
    closed: list[list[tuple[int, int]]] = []
    for comp in _components(open_mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        if any(int(panel[r, c]) == BLACK for r, c in comp):
            closed.append(comp)
    return closed


def _fill_region_black_cells(out: np.ndarray, panel: np.ndarray, regions: list[list[tuple[int, int]]], color: int) -> None:
    for comp in regions:
        for r, c in comp:
            if int(panel[r, c]) == BLACK:
                out[r, c] = int(color)


def _analyze_panel(panel: np.ndarray) -> dict:
    guide_kind, guide_color = _detect_guide(panel)
    line_color = _line_color(panel, guide_color)
    regions = _closed_regions(panel, line_color)
    n = len(regions)
    if n == 0:
        action = "keep"
        kind = f"zero_{guide_kind}"
    elif guide_kind == "h" and n % 2 == 1:
        action = "mirror_h"
        kind = "odd_h"
    elif guide_kind == "v" and n % 2 == 1:
        action = "mirror_v"
        kind = "odd_v"
    elif guide_kind != "none" and n % 2 == 0:
        action = "reverse"
        kind = f"even_{guide_kind}"
    elif guide_kind == "none" and n % 2 == 1:
        action = "rotate"
        kind = "odd_none"
    else:
        action = "fill"
        kind = "even_none"
    return {
        "guide_kind": guide_kind,
        "guide_color": guide_color,
        "line_color": line_color,
        "regions": regions,
        "n": n,
        "action": action,
        "kind": kind,
    }


def _transform_panel(panel: np.ndarray) -> np.ndarray:
    info = _analyze_panel(panel)
    action = info["action"]
    if action == "keep":
        return panel.copy()
    if action == "mirror_h":
        return np.flipud(panel)
    if action == "mirror_v":
        return np.fliplr(panel)
    if action == "rotate":
        return np.rot90(panel, 2)
    out = panel.copy()
    if action == "reverse":
        out[panel == int(info["line_color"])] = BLACK
        _fill_region_black_cells(out, panel, info["regions"], int(info["guide_color"]))
        return out
    if action == "fill":
        _fill_region_black_cells(out, panel, info["regions"], int(info["line_color"]))
        return out
    raise AssertionError(f"unknown action: {action}")


def _split_bounds(separators: list[int], size: int) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for cut in separators + [size]:
        if cut > start:
            bounds.append((start, cut))
        start = cut + 1
    return bounds


def _panel_bounds(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_bounds = _split_bounds(sep_rows, grid.shape[0])
    col_bounds = _split_bounds(sep_cols, grid.shape[1])
    assert len(row_bounds) == PANEL_COUNT and len(col_bounds) == PANEL_COUNT, "expected a 3x3 panel grid"
    assert all((r1 - r0) == PANEL_SIZE for r0, r1 in row_bounds)
    assert all((c1 - c0) == PANEL_SIZE for c0, c1 in col_bounds)
    return row_bounds, col_bounds


def _iter_panels(grid: np.ndarray):
    row_bounds, col_bounds = _panel_bounds(grid)
    for pr, (r0, r1) in enumerate(row_bounds):
        for pc, (c0, c1) in enumerate(col_bounds):
            yield pr, pc, r0, c0, grid[r0:r1, c0:c1]


def _paint_shape(panel: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = panel[top : top + h, left : left + w]
    view[mask] = int(color)


def _pick_variant(pool: list[np.ndarray], *, min_h: int | None = None, max_h: int | None = None, min_w: int | None = None, max_w: int | None = None) -> np.ndarray:
    choices = []
    for variant in pool:
        h, w = variant.shape
        if min_h is not None and h < min_h:
            continue
        if max_h is not None and h > max_h:
            continue
        if min_w is not None and w < min_w:
            continue
        if max_w is not None and w > max_w:
            continue
        choices.append(variant)
    assert choices, "no fitting variants"
    return random.choice(choices).copy()


def _build_panel(kind: str, line_color: int, guide_color: int) -> np.ndarray:
    panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)

    if kind == "odd_h":
        panel[MID, :] = guide_color
        mask = _pick_variant(ODD_VARIANTS, max_h=MID, min_w=6)
        top = random.choice([0, PANEL_SIZE - mask.shape[0]])
        left = random.randint(0, PANEL_SIZE - mask.shape[1])
        _paint_shape(panel, mask, top, left, line_color)
        return panel

    if kind == "odd_v":
        panel[:, MID] = guide_color
        mask = _pick_variant(ODD_VARIANTS, min_h=6, max_w=MID)
        top = random.randint(0, PANEL_SIZE - mask.shape[0])
        left = random.choice([0, PANEL_SIZE - mask.shape[1]])
        _paint_shape(panel, mask, top, left, line_color)
        return panel

    if kind in {"even_h", "zero_h"}:
        panel[MID, :] = guide_color
        pool = EVEN_VARIANTS if kind == "even_h" else ZERO_VARIANTS
        mask = _pick_variant(pool, max_h=MID)
        top = random.choice([0, PANEL_SIZE - mask.shape[0]])
        left = random.randint(0, PANEL_SIZE - mask.shape[1])
        _paint_shape(panel, mask, top, left, line_color)
        return panel

    if kind in {"even_v", "zero_v"}:
        panel[:, MID] = guide_color
        pool = EVEN_VARIANTS if kind == "even_v" else ZERO_VARIANTS
        mask = _pick_variant(pool, max_w=MID)
        top = random.randint(0, PANEL_SIZE - mask.shape[0])
        left = random.choice([0, PANEL_SIZE - mask.shape[1]])
        _paint_shape(panel, mask, top, left, line_color)
        return panel

    if kind == "odd_none":
        mask = random.choice(ODD_VARIANTS).copy()
    elif kind == "even_none":
        mask = random.choice(EVEN_VARIANTS).copy()
    elif kind == "zero_none":
        mask = random.choice(ZERO_VARIANTS).copy()
    else:
        raise ValueError(f"unknown kind: {kind}")

    top = random.randint(0, PANEL_SIZE - mask.shape[0])
    left = random.randint(0, PANEL_SIZE - mask.shape[1])
    _paint_shape(panel, mask, top, left, line_color)
    return panel


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _panel_signature(panel: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return panel.shape, panel.tobytes()


def _kinds_for_grid() -> list[str]:
    zero_kind = random.choice(["zero_h", "zero_v", "zero_none"])
    base = ["odd_h", "odd_v", "even_h", "even_v", "odd_none", "even_none", zero_kind]
    extras = random.choices(
        ["odd_h", "odd_v", "even_h", "even_v", "odd_none", "even_none", "zero_h", "zero_v", "zero_none"],
        k=2,
    )
    kinds = base + extras
    random.shuffle(kinds)
    return kinds


def _semantic_checks(grid: np.ndarray, expected_kinds: list[str]) -> bool:
    observed_infos = []
    observed_kinds = []
    input_panels = []
    for index, (_pr, _pc, _r0, _c0, panel) in enumerate(_iter_panels(grid)):
        info = _analyze_panel(panel)
        observed_infos.append(info)
        observed_kinds.append(info["kind"])
        input_panels.append(panel.copy())
        if info["kind"] != expected_kinds[index]:
            return False

    if "odd_h" not in observed_kinds or "odd_v" not in observed_kinds:
        return False
    if not any(kind in observed_kinds for kind in ("even_h", "even_v")):
        return False
    if "odd_none" not in observed_kinds or "even_none" not in observed_kinds:
        return False

    horizontal_actions = {info["action"] for info in observed_infos if info["guide_kind"] == "h"}
    if not {"mirror_h", "reverse"}.issubset(horizontal_actions):
        return False
    if "rotate" not in {info["action"] for info in observed_infos if info["guide_kind"] == "none"}:
        return False
    if "fill" not in {info["action"] for info in observed_infos if info["guide_kind"] == "none"}:
        return False
    if not any(info["action"] == "keep" for info in observed_infos):
        return False

    out = main(grid)
    output_panels = [panel.copy() for *_rest, panel in _iter_panels(out)]
    if len({_panel_signature(panel) for panel in input_panels}) < 6:
        return False
    if len({_panel_signature(panel) for panel in output_panels}) < 5:
        return False
    if not any(not np.array_equal(inp, outp) for inp, outp in zip(input_panels, output_panels)):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each panel by choosing a mirror, fill, rotation, or no-op from the parity of its enclosed regions and the presence of a center guide line.

    Input structure:
    - The grid is divided into a 3x3 lattice of equal panels by one-cell-wide grey separators.
    - Each panel contains a single-color line drawing on black, plus optionally a full middle row or full middle column guide line in another color.
    - Closed regions are counted only from the line drawing; the guide line is not part of the enclosing obstacle set.

    Rule:
    - For each panel, count the 4-connected black background regions fully enclosed by the line drawing.
    - Odd count + horizontal guide -> mirror the whole panel top-to-bottom across the guide line.
    - Odd count + vertical guide -> mirror the whole panel left-to-right across the guide line.
    - Positive even count + any guide -> turn the line drawing black and fill each enclosed region with the guide color.
    - Odd count + no guide -> rotate the whole panel 180 degrees.
    - Positive even count + no guide -> fill each enclosed region with the line-drawing color.
    - Zero enclosed regions -> leave the panel unchanged.

    Key insight:
    - Neither the guide line nor the region count alone determines the output; the correct operation comes from the joint decision table of guide presence/orientation and enclosed-region parity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for _pr, _pc, r0, c0, panel in _iter_panels(g):
        _paste(out, _transform_panel(panel), r0, c0)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        guide_color = random.choice(NON_GREY_NON_BLACK)
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        grid[SEPARATOR_ROWS, :] = GREY
        grid[:, SEPARATOR_COLS] = GREY

        kinds = _kinds_for_grid()
        row_bounds, col_bounds = _panel_bounds(grid)
        expected_kinds = []
        for idx, kind in enumerate(kinds):
            pr, pc = divmod(idx, PANEL_COUNT)
            r0, r1 = row_bounds[pr]
            c0, c1 = col_bounds[pc]
            line_color = random.choice([c for c in NON_GREY_NON_BLACK if c != guide_color])
            panel = _build_panel(kind, line_color, guide_color)
            _paste(grid, panel, r0, c0)
            expected_kinds.append(kind)

        if _semantic_checks(grid, expected_kinds):
            return grid
    raise RuntimeError("failed to generate a valid panel-parity ARC input")
