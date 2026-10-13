from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
MARKER = 1
GREY = 5

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ROTATIONAL = "rotational"
KINDS = (HORIZONTAL, VERTICAL, ROTATIONAL)

LEGEND_ENTRY_H = 4
LEGEND_ENTRY_W = 4
LEGEND_GAP = 1
GEN_RETRIES = 500
COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            parts.append(coords)
    return parts


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        want = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(coords) != want:
            continue
        frames.append({"bbox": (r0, c0, r1, c1), "coords": coords})
    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one grey frame"
    return frames


def _column_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    cols = [int(c) for c in range(mask.shape[1]) if bool(mask[:, c].any())]
    assert cols, "expected non-empty column groups"
    groups: list[tuple[int, int]] = []
    start = prev = cols[0]
    for c in cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        groups.append((start, prev))
        start = prev = c
    groups.append((start, prev))
    return groups


def _classify_marker(mask: np.ndarray) -> str:
    pts = np.argwhere(mask)
    count = int(len(pts))
    if count == 1:
        return HORIZONTAL
    if count == 2:
        (r0, c0), (r1, c1) = pts.tolist()
        assert r0 == r1 and abs(c0 - c1) == 1
        return VERTICAL
    if count == 4:
        rr0 = int(pts[:, 0].min())
        rr1 = int(pts[:, 0].max())
        cc0 = int(pts[:, 1].min())
        cc1 = int(pts[:, 1].max())
        assert rr1 - rr0 == 1 and cc1 - cc0 == 1
        return ROTATIONAL
    raise AssertionError("unknown legend marker shape")


def _parse_legend(grid: np.ndarray, frames: list[dict]) -> dict[str, int]:
    covered = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    legend_mask = (grid != BLACK) & (~covered)
    pts = np.argwhere(legend_mask)
    assert pts.size > 0, "expected a legend outside the frames"
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    groups = _column_groups(legend != BLACK)
    assert len(groups) == 3, "legend must contain three entries"

    split = legend.shape[0] // 2
    mapping: dict[str, int] = {}
    for gc0, gc1 in groups:
        sub = legend[:, gc0 : gc1 + 1]
        top = sub[:split, :]
        bottom = sub[split:, :]
        kind = _classify_marker(top != BLACK)
        colors = {int(v) for v in np.unique(bottom) if int(v) != BLACK}
        assert len(colors) == 1, "repair swatch must use one color"
        mapping[kind] = colors.pop()
    assert set(mapping) == set(KINDS), "legend must define all three symmetry kinds"
    return mapping


def _symmetry_pairs(shape: tuple[int, int], kind: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if kind == HORIZONTAL:
        for r in range(h // 2):
            rr = h - 1 - r
            for c in range(w):
                pairs.append(((int(r), int(c)), (int(rr), int(c))))
        return pairs
    if kind == VERTICAL:
        for c in range(w // 2):
            cc = w - 1 - c
            for r in range(h):
                pairs.append(((int(r), int(c)), (int(r), int(cc))))
        return pairs
    assert kind == ROTATIONAL
    for r in range(h):
        for c in range(w):
            rr = h - 1 - r
            cc = w - 1 - c
            if (r, c) >= (rr, cc):
                continue
            pairs.append(((int(r), int(c)), (int(rr), int(cc))))
    return pairs


def _repair_plan(mask: np.ndarray, kind: str) -> tuple[int, list[tuple[int, int, bool]]]:
    defects = 0
    actions: list[tuple[int, int, bool]] = []
    for (sr, sc), (tr, tc) in _symmetry_pairs(mask.shape, kind):
        src = bool(mask[sr, sc])
        dst = bool(mask[tr, tc])
        if src == dst:
            continue
        defects += 1
        actions.append((int(tr), int(tc), src))
    return defects, actions


def _best_kind(mask: np.ndarray) -> tuple[str, int, list[tuple[int, int, bool]]]:
    plans = []
    for kind in KINDS:
        defects, actions = _repair_plan(mask, kind)
        plans.append((int(defects), kind, actions))
    plans.sort(key=lambda item: (item[0], KINDS.index(item[1])))
    defects, kind, actions = plans[0]
    return kind, defects, actions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the closest symmetry class inside each grey frame, then copy a canonical half onto the
    opposite half. Missing target pixels are filled with the legend's repair color; extra target
    pixels are erased to black.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    legend = _parse_legend(g, frames)
    out = g.copy()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        mask = inner != BLACK
        kind, _defects, actions = _best_kind(mask)
        repair_color = legend[kind]
        for rr, cc, want_fill in actions:
            out[r0 + 1 + rr, c0 + 1 + cc] = repair_color if want_fill else BLACK

    return out


def _legend_entry(kind: str, color: int) -> np.ndarray:
    entry = np.zeros((LEGEND_ENTRY_H, LEGEND_ENTRY_W), dtype=int)
    if kind == HORIZONTAL:
        entry[0, 1] = MARKER
    elif kind == VERTICAL:
        entry[0, 1:3] = MARKER
    else:
        entry[0:2, 1:3] = MARKER
    entry[2:, :] = color
    return entry


def _build_legend(mapping: dict[str, int], rng: random.Random) -> np.ndarray:
    order = list(KINDS)
    rng.shuffle(order)
    width = 3 * LEGEND_ENTRY_W + 2 * LEGEND_GAP
    legend = np.zeros((LEGEND_ENTRY_H, width), dtype=int)
    col = 0
    for idx, kind in enumerate(order):
        legend[:, col : col + LEGEND_ENTRY_W] = _legend_entry(kind, mapping[kind])
        col += LEGEND_ENTRY_W
        if idx != len(order) - 1:
            col += LEGEND_GAP
    return legend


def _apply_symmetry(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == HORIZONTAL:
        return mask | np.flipud(mask)
    if kind == VERTICAL:
        return mask | np.fliplr(mask)
    return mask | np.rot90(mask, 2)


def _symmetry_kinds(mask: np.ndarray) -> list[str]:
    kinds = []
    if bool(np.array_equal(mask, np.flipud(mask))):
        kinds.append(HORIZONTAL)
    if bool(np.array_equal(mask, np.fliplr(mask))):
        kinds.append(VERTICAL)
    if bool(np.array_equal(mask, np.rot90(mask, 2))):
        kinds.append(ROTATIONAL)
    return kinds


def _target_cells(mask: np.ndarray, kind: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    filled: list[tuple[int, int]] = []
    empty: list[tuple[int, int]] = []
    for _src, dst in _symmetry_pairs(mask.shape, kind):
        if bool(mask[dst]):
            filled.append(dst)
        else:
            empty.append(dst)
    return filled, empty


def _generate_base_mask(kind: str, h: int, w: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(250):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(rng.randint(2, 4)):
            rect = np.zeros((h, w), dtype=bool)
            rh = rng.randint(1, max(1, h - 1))
            rw = rng.randint(1, max(1, w - 1))
            r0 = rng.randint(0, h - rh)
            c0 = rng.randint(0, w - rw)
            rect[r0 : r0 + rh, c0 : c0 + rw] = True
            mask |= _apply_symmetry(rect, kind)
        for _ in range(rng.randint(0, 1)):
            hole = np.zeros((h, w), dtype=bool)
            rh = rng.randint(1, max(1, h // 2))
            rw = rng.randint(1, max(1, w // 2))
            r0 = rng.randint(0, h - rh)
            c0 = rng.randint(0, w - rw)
            hole[r0 : r0 + rh, c0 : c0 + rw] = True
            mask &= ~_apply_symmetry(hole, kind)

        area = int(mask.sum())
        if area < max(4, (h * w) // 5):
            continue
        if area > h * w - max(4, (h * w) // 6):
            continue
        if _symmetry_kinds(mask) != [kind]:
            continue
        filled_targets, empty_targets = _target_cells(mask, kind)
        if len(filled_targets) < 2 or len(empty_targets) < 2:
            continue
        return mask
    return None


def _inject_defects(
    base: np.ndarray,
    kind: str,
    rng: random.Random,
    required_first: str | None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    filled_targets, empty_targets = _target_cells(base, kind)
    if required_first == "missing" and not filled_targets:
        return None
    if required_first == "extra" and not empty_targets:
        return None

    n_defects = rng.randint(1, 3)
    types: list[str] = []
    if required_first is not None:
        types.append(required_first)
    while len(types) < n_defects:
        types.append(rng.choice(("missing", "extra")))

    if types.count("missing") > len(filled_targets):
        return None
    if types.count("extra") > len(empty_targets):
        return None

    chosen_missing = rng.sample(filled_targets, types.count("missing"))
    chosen_extra = rng.sample(empty_targets, types.count("extra"))

    broken = base.copy()
    for r, c in chosen_missing:
        broken[r, c] = False
    for r, c in chosen_extra:
        broken[r, c] = True

    best_kind, best_defects, _actions = _best_kind(broken)
    counts = [_repair_plan(broken, cur)[0] for cur in KINDS]
    if best_kind != kind or counts.count(best_defects) != 1:
        return None
    return broken, chosen_missing, chosen_extra


def _render_container(
    input_mask: np.ndarray,
    output_mask: np.ndarray,
    source_color: int,
    repair_color: int,
    missing_cells: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = input_mask.shape
    inp = np.zeros((h + 2, w + 2), dtype=int)
    out = np.zeros((h + 2, w + 2), dtype=int)
    inp[0, :] = GREY
    inp[-1, :] = GREY
    inp[:, 0] = GREY
    inp[:, -1] = GREY
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY

    inp[1 : h + 1, 1 : w + 1][input_mask] = source_color
    out[1 : h + 1, 1 : w + 1][output_mask] = source_color
    for r, c in missing_cells:
        out[1 + r, 1 + c] = repair_color
    return inp, out


def _slot_rects(h: int, w: int) -> list[tuple[int, int, int, int]]:
    content_bottom = h - LEGEND_ENTRY_H - 2
    top = 1
    left = 1
    right = w - 2
    mid_r = (top + content_bottom) // 2
    mid_c = (left + right) // 2
    return [
        (top, left, mid_r - 1, mid_c - 1),
        (top, mid_c + 1, mid_r - 1, right),
        (mid_r + 1, left, content_bottom, mid_c - 1),
        (mid_r + 1, mid_c + 1, content_bottom, right),
    ]


def _sample_tile_origin(
    tile: np.ndarray,
    slot: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[int, int]:
    r0, c0, r1, c1 = slot
    sh = r1 - r0 + 1
    sw = c1 - c0 + 1
    th, tw = tile.shape
    assert th <= sh and tw <= sw
    rr = rng.randint(r0, r1 - th + 1)
    cc = rng.randint(c0, c1 - tw + 1)
    return rr, cc


def _place_tile(board: np.ndarray, tile: np.ndarray, rr: int, cc: int) -> None:
    th, tw = tile.shape
    board[rr : rr + th, cc : cc + tw] = tile


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_regions = rng.choice((3, 4))
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    slots = _slot_rects(h, w)
    use_slots = rng.sample(range(4), n_regions)

    palette = list(COLOR_POOL)
    rng.shuffle(palette)
    repair_colors = {
        HORIZONTAL: palette[0],
        VERTICAL: palette[1],
        ROTATIONAL: palette[2],
    }
    source_colors = palette[3 : 3 + n_regions]
    assert len(source_colors) == n_regions

    kinds = [HORIZONTAL, VERTICAL, ROTATIONAL]
    if n_regions == 4:
        kinds.append(rng.choice(KINDS))
    rng.shuffle(kinds)

    grid_in = np.zeros((h, w), dtype=int)
    grid_out = np.zeros((h, w), dtype=int)

    specs: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    forced = ["missing", "extra"] + [None] * max(0, n_regions - 2)
    rng.shuffle(forced)

    for idx in range(n_regions):
        slot = slots[use_slots[idx]]
        sr0, sc0, sr1, sc1 = slot
        max_inner_h = min(7, sr1 - sr0 - 1)
        max_inner_w = min(7, sc1 - sc0 - 1)
        if max_inner_h < 5 or max_inner_w < 5:
            return None
        kind = kinds[idx]
        source_color = source_colors[idx]
        repair_color = repair_colors[kind]

        built = None
        for _ in range(200):
            inner_h = rng.randint(5, max_inner_h)
            inner_w = rng.randint(5, max_inner_w)
            base = _generate_base_mask(kind, inner_h, inner_w, rng)
            if base is None:
                continue
            defects = _inject_defects(base, kind, rng, forced[idx])
            if defects is None:
                continue
            broken, missing_cells, _extra_cells = defects
            inp_tile, out_tile = _render_container(
                broken,
                base,
                source_color,
                repair_color,
                missing_cells,
            )
            rr, cc = _sample_tile_origin(inp_tile, slot, rng)
            built = (inp_tile, out_tile, rr, cc)
            break
        if built is None:
            return None
        specs.append(built)

    for inp_tile, out_tile, rr, cc in specs:
        _place_tile(grid_in, inp_tile, rr, cc)
        _place_tile(grid_out, out_tile, rr, cc)

    legend = _build_legend(repair_colors, rng)
    lx = rng.randint(1, w - legend.shape[1] - 1)
    ly = h - LEGEND_ENTRY_H
    grid_in[ly : ly + legend.shape[0], lx : lx + legend.shape[1]] = legend
    grid_out[ly : ly + legend.shape[0], lx : lx + legend.shape[1]] = legend

    solved = main(grid_in)
    if not bool(np.array_equal(solved, grid_out)):
        return None
    return grid_in, grid_out


_RNG = random.Random()


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case(_RNG)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC instance")
