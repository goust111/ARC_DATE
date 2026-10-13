from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 8
BASE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
PAIR_COUNT_OPTIONS = (1, 2, 3)
SCALE_OPTIONS = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
PATTERN_MODES = ("sparse", "block", "checker", "stripes", "plus")
GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_values(patch: np.ndarray) -> np.ndarray:
    h, w = map(int, patch.shape)
    if h == 1:
        return patch[0].copy()
    if w == 1:
        return patch[:, 0].copy()
    return np.concatenate(
        [
            patch[0, :],
            patch[h - 1, :],
            patch[1 : h - 1, 0],
            patch[1 : h - 1, w - 1],
        ]
    )


def _describe_rectangle(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert bool(np.all(patch != BG)), "every object must fill its rectangle"

    border_colors = set(map(int, _perimeter_values(patch).tolist()))
    assert len(border_colors) == 1, "template/canvas border must reveal the rectangle color"
    base_color = border_colors.pop()
    assert base_color != BG

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "base_color": int(base_color),
        "area": int(patch.size),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each small template rectangle after tiling its internal pattern across the same-colored larger canvas rectangle.

    Input structure:
    - The background is Azure.
    - The non-background cells form several disjoint filled rectangles; each rectangle stays 4-connected even when some template interior pixels use accent colors.
    - For every rectangle color there are exactly two rectangles with that border color: a smaller template and a larger canvas.
    - Template borders stay in the rectangle color, while only interior cells may be replaced by other non-background colors.

    Rule:
    - Find every non-background rectangle component and read its rectangle color from the uniform border.
    - Pair rectangles by that border color and treat the smaller-area one as the template and the larger one as the canvas.
    - Partition the canvas into equal sub-rectangles matching the template size, then copy the full template patch into every sub-rectangle.
    - Replace the template rectangle itself with Azure background.

    Key insight:
    The visible border color identifies which rectangles belong together, and the template-to-canvas size multiple reveals that the template patch must be repeated as a tile.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    assert comps, "expected at least one rectangle"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        rect = _describe_rectangle(g, comp)
        by_color[int(rect["base_color"])].append(rect)

    out = g.copy()
    for color, rects in by_color.items():
        assert len(rects) == 2, f"expected exactly two rectangles for color {color}"
        rects = sorted(rects, key=lambda item: int(item["area"]))
        template, canvas = rects
        assert int(template["area"]) < int(canvas["area"]), "template must be smaller than canvas"

        template_patch = template["patch"]
        th, tw = map(int, template_patch.shape)
        ch, cw = map(int, canvas["patch"].shape)
        assert ch % th == 0 and cw % tw == 0, "canvas dimensions must be integer multiples of template dimensions"

        tiled = np.tile(template_patch, (ch // th, cw // tw))
        cr0, cc0, cr1, cc1 = canvas["bbox"]
        tr0, tc0, tr1, tc1 = template["bbox"]
        out[cr0 : cr1 + 1, cc0 : cc1 + 1] = tiled
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BG

    return out


def _template_sizes(pair_count: int) -> tuple[int, int]:
    side_max = 6 if pair_count == 1 else 5 if pair_count == 2 else 4
    return 3, side_max


def _choose_replacement_cells(h: int, w: int) -> tuple[str, list[tuple[int, int]]]:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    assert interior

    feasible: list[str] = ["sparse"]
    if len(interior) >= 2:
        feasible.append("checker")
        feasible.append("plus")
    if h >= 4 and w >= 4:
        feasible.append("block")
    if (h - 2) >= 2 or (w - 2) >= 2:
        feasible.append("stripes")

    mode = random.choice(feasible)
    cells: list[tuple[int, int]]

    if mode == "sparse":
        limit = max(1, min(len(interior), (len(interior) + 1) // 2))
        cells = random.sample(interior, random.randint(1, limit))
    elif mode == "block":
        bh = random.randint(1, h - 2)
        bw = random.randint(1, w - 2)
        r0 = random.randint(1, h - 1 - bh)
        c0 = random.randint(1, w - 1 - bw)
        cells = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]
    elif mode == "checker":
        parity = random.randint(0, 1)
        cells = [(r, c) for r, c in interior if (r + c) % 2 == parity]
        if not cells:
            cells = [random.choice(interior)]
    elif mode == "stripes":
        if random.choice([True, False]):
            parity = random.randint(0, 1)
            rows = [r for r in range(1, h - 1) if r % 2 == parity]
            if not rows:
                rows = [random.randint(1, h - 2)]
            cells = [(r, c) for r in rows for c in range(1, w - 1)]
        else:
            parity = random.randint(0, 1)
            cols = [c for c in range(1, w - 1) if c % 2 == parity]
            if not cols:
                cols = [random.randint(1, w - 2)]
            cells = [(r, c) for r in range(1, h - 1) for c in cols]
    else:
        mid_r = list({h // 2, max(1, h // 2 - 1)} & set(range(1, h - 1)))
        mid_c = list({w // 2, max(1, w // 2 - 1)} & set(range(1, w - 1)))
        cells = sorted({(r, c) for r in mid_r for c in range(1, w - 1)} | {(r, c) for r in range(1, h - 1) for c in mid_c})
        if not cells:
            cells = [random.choice(interior)]

    return mode, sorted(set(cells))


def _make_template_patch(h: int, w: int, base_color: int, accent_pool: list[int]) -> np.ndarray:
    patch = np.full((h, w), int(base_color), dtype=int)
    mode, cells = _choose_replacement_cells(h, w)
    assert cells

    accent_count = 1 if len(accent_pool) == 1 else random.choice((1, 2))
    accents = random.sample(accent_pool, accent_count)

    for idx, (r, c) in enumerate(cells):
        if len(accents) == 1:
            color = accents[0]
        elif mode in {"checker", "plus"}:
            color = accents[(r + c) % len(accents)]
        else:
            color = accents[idx % len(accents)]
        patch[int(r), int(c)] = int(color)

    assert set(map(int, _perimeter_values(patch).tolist())) == {int(base_color)}
    assert bool(np.any(patch != int(base_color)))
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    h0, w0 = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > h0 or c0 + w > w0:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h0, r0 + h + padding)
    cc1 = min(w0, c0 + w + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _crop_with_margin(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top = random.randint(0, min(3, int(r0)))
    left = random.randint(0, min(3, int(c0)))
    bottom = random.randint(0, min(3, int(inp.shape[0] - 1 - r1)))
    right = random.randint(0, min(3, int(inp.shape[1] - 1 - c1)))

    rs = slice(int(r0 - top), int(r1 + bottom + 1))
    cs = slice(int(c0 - left), int(c1 + right + 1))
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    base_colors = random.sample(BASE_COLORS, pair_count)
    accent_pool = [c for c in BASE_COLORS if c not in base_colors]
    assert accent_pool

    pairs = []
    min_side, max_side = _template_sizes(pair_count)
    for base_color in base_colors:
        th = random.randint(min_side, max_side)
        tw = random.randint(min_side, max_side)
        kh, kw = random.choice(SCALE_OPTIONS)
        template_patch = _make_template_patch(th, tw, int(base_color), accent_pool)
        canvas_patch = np.full((th * kh, tw * kw), int(base_color), dtype=int)
        pairs.append(
            {
                "base_color": int(base_color),
                "template": template_patch,
                "canvas": canvas_patch,
                "tiled": np.tile(template_patch, (kh, kw)),
            }
        )

    total_area = sum(int(pair["template"].size + pair["canvas"].size) for pair in pairs)
    max_h = max(int(max(pair["template"].shape[0], pair["canvas"].shape[0])) for pair in pairs)
    max_w = max(int(max(pair["template"].shape[1], pair["canvas"].shape[1])) for pair in pairs)

    for _ in range(PLACE_ATTEMPTS):
        min_h = max(8, max_h)
        min_w = max(8, max_w)
        grid_h = random.randint(min_h, 30)
        grid_w = random.randint(min_w, 30)
        if grid_h * grid_w < total_area * 2:
            continue

        inp = np.full((grid_h, grid_w), BG, dtype=int)
        out = np.full((grid_h, grid_w), BG, dtype=int)
        occupied = np.zeros((grid_h, grid_w), dtype=bool)

        placements = []
        items = []
        for idx, pair in enumerate(pairs):
            items.append((int(pair["canvas"].size), idx, "canvas", pair["canvas"]))
            items.append((int(pair["template"].size), idx, "template", pair["template"]))
        items.sort(reverse=True, key=lambda item: item[0])

        ok = True
        for _area, idx, kind, patch in items:
            h, w = map(int, patch.shape)
            placed = False
            for _ in range(PLACE_ATTEMPTS):
                r0 = random.randint(0, grid_h - h)
                c0 = random.randint(0, grid_w - w)
                if not _can_place(occupied, r0, c0, h, w, padding=1):
                    continue
                _mark_occupied(occupied, r0, c0, h, w)
                placements.append((idx, kind, r0, c0))
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        for idx, kind, r0, c0 in placements:
            pair = pairs[int(idx)]
            if kind == "canvas":
                _paste(inp, pair["canvas"], r0, c0)
                _paste(out, pair["tiled"], r0, c0)
            else:
                _paste(inp, pair["template"], r0, c0)

        inp, out = _crop_with_margin(inp, out)
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        return inp, out

    raise RuntimeError("failed to place rectangles without overlap")


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected = _build_scene()
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid tiled-rectangle input")
