from __future__ import annotations

import numpy as np


BG = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RNG = np.random.default_rng()

MIN_TEMPLATE_RECT = 3
MAX_TEMPLATE_RECT = 7
MIN_FRAGMENT_AREA = 3
MIN_GRID_SIDE = 16
MAX_GRID_SIDE = 30
MIN_PAIRS = 2
MAX_PAIRS = 4
MAX_GENERATION_ATTEMPTS = 500


def _normalize_coords(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    norm = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))
    h = int(rows.max() - r0 + 1)
    w = int(cols.max() - c0 + 1)
    return h, w, norm


def _piece_from_coords(coords: np.ndarray, color: int) -> dict:
    coords = np.asarray(coords, dtype=int)
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    h, w, norm = _normalize_coords(coords)
    return {
        "color": int(color),
        "coords": coords.copy(),
        "area": int(coords.shape[0]),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": h,
        "w": w,
        "norm": norm,
    }


def _piece_from_mask(mask: np.ndarray, color: int) -> dict:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _piece_from_coords(coords, color)


def _piece_signature(piece: dict) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    return int(piece["h"]), int(piece["w"]), tuple(piece["norm"])


def _piece_mask(piece: dict) -> np.ndarray:
    mask = np.zeros((int(piece["h"]), int(piece["w"])), dtype=bool)
    for r, c in piece["norm"]:
        mask[int(r), int(c)] = True
    return mask


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            pieces.append(_piece_from_coords(np.array(coords, dtype=int), color))

    pieces.sort(key=lambda p: (int(p["r0"]), int(p["c0"]), int(p["color"])))
    return pieces


def _find_layouts(anchor: dict, other: dict) -> list[dict]:
    total_area = int(anchor["area"]) + int(other["area"])
    anchor_norm = tuple(anchor["norm"])
    other_sig = _piece_signature(other)
    layouts: set[tuple[int, int, int, int, int, int]] = set()

    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        if rect_h < int(anchor["h"]) or rect_w < int(anchor["w"]):
            continue
        if rect_h < int(other["h"]) or rect_w < int(other["w"]):
            continue
        for anchor_r in range(rect_h - int(anchor["h"]) + 1):
            for anchor_c in range(rect_w - int(anchor["w"]) + 1):
                occ = np.zeros((rect_h, rect_w), dtype=bool)
                for dr, dc in anchor_norm:
                    occ[anchor_r + int(dr), anchor_c + int(dc)] = True
                comp = np.argwhere(~occ)
                other_h, other_w, other_norm = _normalize_coords(comp)
                if (other_h, other_w, other_norm) != other_sig:
                    continue
                other_r = int(comp[:, 0].min())
                other_c = int(comp[:, 1].min())
                layouts.add((rect_h, rect_w, anchor_r, anchor_c, other_r, other_c))

    out = []
    for rect_h, rect_w, anchor_r, anchor_c, other_r, other_c in sorted(layouts):
        out.append(
            {
                "rect_h": rect_h,
                "rect_w": rect_w,
                "anchor_r": anchor_r,
                "anchor_c": anchor_c,
                "other_r": other_r,
                "other_c": other_c,
            }
        )
    return out


def _pair_pieces(pieces: list[dict]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[dict]]]:
    n = len(pieces)
    neighbors: list[set[int]] = [set() for _ in range(n)]
    layout_cache: dict[tuple[int, int], list[dict]] = {}

    for i in range(n):
        for j in range(i + 1, n):
            ij_layouts = _find_layouts(pieces[i], pieces[j])
            if not ij_layouts:
                continue
            ji_layouts = _find_layouts(pieces[j], pieces[i])
            if not ji_layouts:
                continue
            neighbors[i].add(j)
            neighbors[j].add(i)
            layout_cache[(i, j)] = ij_layouts
            layout_cache[(j, i)] = ji_layouts

    for i, opts in enumerate(neighbors):
        assert len(opts) == 1, f"fragment {i} has {len(opts)} complement matches"

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(n):
        if i in used:
            continue
        j = next(iter(neighbors[i]))
        assert i in neighbors[j]
        used.add(i)
        used.add(j)
        pairs.append((min(i, j), max(i, j)))

    pairs.sort()
    return pairs, layout_cache


def _anchor_first(piece_a: dict, piece_b: dict) -> bool:
    if int(piece_a["area"]) != int(piece_b["area"]):
        return int(piece_a["area"]) > int(piece_b["area"])
    if int(piece_a["r0"]) != int(piece_b["r0"]):
        return int(piece_a["r0"]) < int(piece_b["r0"])
    return (int(piece_a["c0"]), int(piece_a["color"])) <= (int(piece_b["c0"]), int(piece_b["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered complementary fragments into solid rectangles colored by the dominant half.

    Input structure:
    - Black (0) is the background.
    - Several monochrome 4-connected fragments are scattered across the grid.
    - Every rectangle is split into exactly two differently colored fragments, and each pair fits together by translation only, with no rotation or reflection.

    Rule:
    - Find the unique partner of each fragment by checking which other fragment can join it to tile one solid axis-aligned rectangle without overlap.
    - For every matched pair, keep the larger fragment fixed as the anchor; if the areas tie, use the higher fragment as both anchor and color source.
    - Translate the other fragment into the missing complementary cells, then repaint the whole completed rectangle with the anchor/color-source fragment's color.

    Key insight:
    - Pairing depends only on geometric complementarity: two fragments belong together exactly when their untranslated shapes can fill a rectangle, and only after that pairing do area and row order decide the final color.
    """
    g = np.asarray(grid, dtype=int)
    pieces = _find_components(g, bg=BG)
    assert pieces and len(pieces) % 2 == 0

    pairs, layout_cache = _pair_pieces(pieces)
    out = np.full_like(g, BG)
    occupied = np.zeros_like(g, dtype=bool)

    for i, j in pairs:
        if _anchor_first(pieces[i], pieces[j]):
            anchor_idx, other_idx = i, j
        else:
            anchor_idx, other_idx = j, i

        layouts = layout_cache[(anchor_idx, other_idx)]
        assert len(layouts) == 1, "pair layout must be unique"
        layout = layouts[0]

        top = int(pieces[anchor_idx]["r0"]) - int(layout["anchor_r"])
        left = int(pieces[anchor_idx]["c0"]) - int(layout["anchor_c"])
        rect_h = int(layout["rect_h"])
        rect_w = int(layout["rect_w"])

        assert 0 <= top and 0 <= left
        assert top + rect_h <= g.shape[0] and left + rect_w <= g.shape[1]
        region = occupied[top : top + rect_h, left : left + rect_w]
        assert not region.any(), "completed rectangles must stay disjoint"

        occupied[top : top + rect_h, left : left + rect_w] = True
        out[top : top + rect_h, left : left + rect_w] = int(pieces[anchor_idx]["color"])

    return out


def _linear_sequence(length: int, start: int, end: int) -> tuple[int, ...]:
    if length == 1:
        return (int(start),)
    vals = []
    for i in range(length):
        t = i / (length - 1)
        vals.append(int(np.floor(start + (end - start) * t + 0.5)))
    return tuple(vals)


def _mask_from_template(rect_h: int, rect_w: int, orientation: str, seq: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((rect_h, rect_w), dtype=bool)
    if orientation == "vertical":
        for r, width in enumerate(seq):
            mask[r, : int(width)] = True
    else:
        for c, height in enumerate(seq):
            mask[: int(height), c] = True
    return mask


def _build_template_bank() -> tuple[list[dict], list[dict]]:
    equal_templates: list[dict] = []
    unequal_templates: list[dict] = []
    seen: set[tuple] = set()

    for rect_h in range(MIN_TEMPLATE_RECT, MAX_TEMPLATE_RECT + 1):
        for rect_w in range(MIN_TEMPLATE_RECT, MAX_TEMPLATE_RECT + 1):
            for orientation in ("vertical", "horizontal"):
                length = rect_h if orientation == "vertical" else rect_w
                limit = rect_w - 1 if orientation == "vertical" else rect_h - 1
                for start in range(1, limit + 1):
                    for end in range(1, limit + 1):
                        seq = _linear_sequence(length, start, end)
                        mask_a = _mask_from_template(rect_h, rect_w, orientation, seq)
                        area_a = int(mask_a.sum())
                        area_b = rect_h * rect_w - area_a
                        if min(area_a, area_b) < MIN_FRAGMENT_AREA:
                            continue

                        piece_a = _piece_from_mask(mask_a, 1)
                        piece_b = _piece_from_mask(~mask_a, 2)
                        if len(_find_layouts(piece_a, piece_b)) != 1:
                            continue
                        if len(_find_layouts(piece_b, piece_a)) != 1:
                            continue

                        pair_key = (
                            rect_h,
                            rect_w,
                            tuple(sorted((_piece_signature(piece_a), _piece_signature(piece_b)))),
                        )
                        if pair_key in seen:
                            continue
                        seen.add(pair_key)

                        smaller = min(area_a, area_b) / float(rect_h * rect_w)
                        if area_a != area_b and smaller < 0.28:
                            continue

                        template = {
                            "rect_h": rect_h,
                            "rect_w": rect_w,
                            "mask_a": mask_a.copy(),
                            "mask_b": (~mask_a).copy(),
                            "area_a": area_a,
                            "area_b": area_b,
                            "orientation": orientation,
                            "seq": seq,
                            "pair_key": pair_key,
                        }
                        if area_a == area_b:
                            equal_templates.append(template)
                        else:
                            unequal_templates.append(template)

    assert equal_templates and unequal_templates
    return equal_templates, unequal_templates


EQUAL_TEMPLATES, UNEQUAL_TEMPLATES = _build_template_bank()


def _shape_fits(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    return not bool((blocked[top : top + h, left : left + w] & mask).any())


def _mark_mask(blocked: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int = 1) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(mask):
        rr = int(top + dr)
        cc = int(left + dc)
        r0 = max(0, rr - pad)
        r1 = min(h, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(w, cc + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _mark_rect(blocked: np.ndarray, rect_h: int, rect_w: int, top: int, left: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + rect_h + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + rect_w + pad)
    blocked[r0:r1, c0:c1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for dr, dc in np.argwhere(mask):
        grid[top + int(dr), left + int(dc)] = int(color)


def _random_grid_shape(rectangles: list[tuple[int, int]]) -> tuple[int, int]:
    footprint = sum((h + 3) * (w + 3) for h, w in rectangles)
    longest_h = max(h for h, _ in rectangles)
    longest_w = max(w for _, w in rectangles)
    base = max(MIN_GRID_SIDE, longest_h + 5, longest_w + 5, int(np.ceil(np.sqrt(footprint))))

    extra_h = int(_RNG.integers(0, min(7, MAX_GRID_SIDE - base) + 1))
    extra_w = int(_RNG.integers(0, min(7, MAX_GRID_SIDE - base) + 1))
    grid_h = min(MAX_GRID_SIDE, base + extra_h)
    grid_w = min(MAX_GRID_SIDE, base + extra_w)
    return int(grid_h), int(grid_w)


def _select_templates(pair_count: int) -> list[dict] | None:
    branches = _RNG.integers(0, 2, size=pair_count)
    branches[0] = 0
    if pair_count > 1:
        branches[1] = 1

    templates: list[dict] = []
    taken_pair_keys: set[tuple] = set()

    for branch in branches:
        bank = EQUAL_TEMPLATES if int(branch) == 1 else UNEQUAL_TEMPLATES
        candidates = [tpl for tpl in bank if tpl["pair_key"] not in taken_pair_keys]
        if not candidates:
            return None
        choice = candidates[int(_RNG.integers(0, len(candidates)))]
        templates.append(choice)
        taken_pair_keys.add(choice["pair_key"])

    probe_pieces = []
    for idx, tpl in enumerate(templates):
        probe_pieces.append(_piece_from_mask(tpl["mask_a"], idx * 2 + 1))
        probe_pieces.append(_piece_from_mask(tpl["mask_b"], idx * 2 + 2))

    try:
        _pair_pieces(probe_pieces)
    except AssertionError:
        return None
    return templates


def _place_rectangles(records: list[dict], grid_h: int, grid_w: int) -> bool:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    order = sorted(range(len(records)), key=lambda idx: records[idx]["rect_h"] * records[idx]["rect_w"], reverse=True)

    for idx in order:
        record = records[idx]
        rect_h = int(record["rect_h"])
        rect_w = int(record["rect_w"])
        placed = False
        for _ in range(250):
            top = int(_RNG.integers(0, grid_h - rect_h + 1))
            left = int(_RNG.integers(0, grid_w - rect_w + 1))
            if blocked[top : top + rect_h, left : left + rect_w].any():
                continue
            record["rect_top"] = top
            record["rect_left"] = left
            _mark_rect(blocked, rect_h, rect_w, top, left, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered_piece(
    blocked: np.ndarray,
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    row_min: int = 0,
) -> tuple[int, int] | None:
    grid_h, grid_w = grid.shape
    piece_h, piece_w = mask.shape
    if row_min > grid_h - piece_h:
        return None

    candidates = []
    for top in range(max(0, row_min), grid_h - piece_h + 1):
        for left in range(grid_w - piece_w + 1):
            if _shape_fits(blocked, mask, top, left):
                candidates.append((top, left))
    if not candidates:
        return None

    top, left = candidates[int(_RNG.integers(0, len(candidates)))]
    _paint_mask(grid, mask, top, left, color)
    _mark_mask(blocked, mask, top, left, pad=1)
    return int(top), int(left)


def _generate_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(MIN_PAIRS, MAX_PAIRS + 1))
    templates = _select_templates(pair_count)
    if templates is None:
        return None

    color_pool = _RNG.permutation(np.arange(1, 10))
    records: list[dict] = []
    for pair_idx, tpl in enumerate(templates):
        color_a = int(color_pool[2 * pair_idx])
        color_b = int(color_pool[2 * pair_idx + 1])
        piece_a = _piece_from_mask(tpl["mask_a"], color_a)
        piece_b = _piece_from_mask(tpl["mask_b"], color_b)

        if int(piece_a["area"]) == int(piece_b["area"]):
            anchor_name = "a" if bool(_RNG.integers(0, 2)) else "b"
        else:
            anchor_name = "a" if int(piece_a["area"]) > int(piece_b["area"]) else "b"

        anchor_piece = piece_a if anchor_name == "a" else piece_b
        other_piece = piece_b if anchor_name == "a" else piece_a
        layout = _find_layouts(anchor_piece, other_piece)
        if len(layout) != 1:
            return None

        records.append(
            {
                "rect_h": int(tpl["rect_h"]),
                "rect_w": int(tpl["rect_w"]),
                "piece_a": piece_a,
                "piece_b": piece_b,
                "anchor_name": anchor_name,
                "layout": layout[0],
            }
        )

    grid_h, grid_w = _random_grid_shape([(rec["rect_h"], rec["rect_w"]) for rec in records])
    if not _place_rectangles(records, grid_h, grid_w):
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for rec in records:
        rect_top = int(rec["rect_top"])
        rect_left = int(rec["rect_left"])
        rect_h = int(rec["rect_h"])
        rect_w = int(rec["rect_w"])
        anchor_piece = rec["piece_a"] if rec["anchor_name"] == "a" else rec["piece_b"]
        layout = rec["layout"]
        anchor_top = rect_top + int(layout["anchor_r"])
        anchor_left = rect_left + int(layout["anchor_c"])
        rec["anchor_input_top"] = anchor_top

        _paint_mask(inp, _piece_mask(anchor_piece), anchor_top, anchor_left, int(anchor_piece["color"]))
        _mark_rect(blocked, rect_h, rect_w, rect_top, rect_left, pad=1)
        out[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = int(anchor_piece["color"])

    scatter_order = sorted(range(len(records)), key=lambda idx: min(records[idx]["piece_a"]["area"], records[idx]["piece_b"]["area"]), reverse=True)
    for idx in scatter_order:
        rec = records[idx]
        piece_a = rec["piece_a"]
        piece_b = rec["piece_b"]
        other_piece = piece_b if rec["anchor_name"] == "a" else piece_a
        row_min = 0
        if int(piece_a["area"]) == int(piece_b["area"]):
            row_min = int(rec["anchor_input_top"]) + 1
        placed = _place_scattered_piece(blocked, inp, _piece_mask(other_piece), int(other_piece["color"]), row_min=row_min)
        if placed is None:
            return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
