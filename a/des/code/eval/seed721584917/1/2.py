from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _neighbors8(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neigh = _neighbors4 if connectivity == 4 else _neighbors8
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neigh(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(np.uint8).ravel())


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _extract_fragments(grid: np.ndarray) -> tuple[list[dict], int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    fg = (g != bg) & (g != BLACK)
    seen = np.zeros((h, w), dtype=bool)
    fragments: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not bool(fg[r, c]) or bool(seen[r, c]):
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc, h, w):
                    if bool(seen[nr, nc]) or not bool(fg[nr, nc]) or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = sub == color
            wins = np.argwhere(sub == BLACK)
            assert len(wins) == 1, "each fragment bbox must contain exactly one black window"
            wr, wc = map(int, wins[0])
            assert not bool(mask[wr, wc]), "window must be on an uncovered cell"
            fragments.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "top_left": (int(r0), int(c0)),
                    "mask": mask,
                    "window": (int(wr), int(wc)),
                }
            )

    assert fragments and len(fragments) % 2 == 0, "expected an even number of fragments"
    dims = {(int(f["mask"].shape[0]), int(f["mask"].shape[1])) for f in fragments}
    assert len(dims) == 1, "all fragment bboxes must have the same size"
    return fragments, bg, next(iter(dims))


def _can_pair(a: dict, b: dict) -> bool:
    if a["mask"].shape != b["mask"].shape:
        return False
    return bool(np.array_equal(a["mask"], ~b["mask"]))


def _find_matching(fragments: list[dict]) -> list[tuple[int, int]]:
    n = len(fragments)
    candidates = {
        i: [j for j in range(n) if i != j and _can_pair(fragments[i], fragments[j])]
        for i in range(n)
    }
    assert all(candidates[i] for i in range(n)), "every fragment must have a complementary mate"

    out: list[tuple[int, int]] | None = None

    def search(remaining: list[int], acc: list[tuple[int, int]]) -> bool:
        nonlocal out
        if not remaining:
            out = acc[:]
            return True
        i = min(remaining, key=lambda idx: len([j for j in candidates[idx] if j in remaining]))
        rest = [x for x in remaining if x != i]
        for j in candidates[i]:
            if j not in rest:
                continue
            next_remaining = [x for x in rest if x != j]
            acc.append((min(i, j), max(i, j)))
            if search(next_remaining, acc):
                return True
            acc.pop()
        return False

    ok = search(list(range(n)), [])
    assert ok and out is not None
    out.sort()
    return out


def _pair_border_color(a: dict, b: dict) -> int:
    wa = a["window"]
    wb = b["window"]
    assert bool(b["mask"][wa]) and bool(a["mask"][wb]), "windows must point into the mate fragment"
    colors = [int(b["color"]), int(a["color"])]
    if colors[0] == colors[1]:
        return colors[0]
    return int(colors[0] if wa <= wb else colors[1])


def _draw_frame(out: np.ndarray, top: int, left: int, height: int, width: int, color: int, bg: int) -> None:
    out[top : top + height, left : left + width] = bg
    out[top, left : left + width] = color
    out[top + height - 1, left : left + width] = color
    out[top : top + height, left] = color
    out[top : top + height, left + width - 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, read the border color through the windows, and redraw each pair
    as a hollow rectangle aligned to the top-left corner of the two input bboxes' union.
    """
    g = np.asarray(grid, dtype=int)
    fragments, bg, (box_h, box_w) = _extract_fragments(g)
    pairing = _find_matching(fragments)

    out = np.full_like(g, bg)
    for i, j in pairing:
        a = fragments[i]
        b = fragments[j]
        color = _pair_border_color(a, b)
        top = min(int(a["top_left"][0]), int(b["top_left"][0]))
        left = min(int(a["top_left"][1]), int(b["top_left"][1]))
        assert top + box_h <= g.shape[0] and left + box_w <= g.shape[1]
        assert bool(np.all(out[top : top + box_h, left : left + box_w] == bg)), "output rectangles must stay disjoint"
        _draw_frame(out, top, left, box_h, box_w, color, bg)
    return out


def _sample_mask(box_h: int, box_w: int, rng: random.Random) -> np.ndarray:
    area = box_h * box_w
    min_fill = max(6, int(area * 0.35))
    max_fill = min(area - 6, int(area * 0.65))
    assert min_fill <= max_fill

    for _ in range(800):
        fill = rng.randint(min_fill, max_fill)
        picks = rng.sample(range(area), fill)
        mask = np.zeros((box_h, box_w), dtype=bool)
        for idx in picks:
            mask[idx // box_w, idx % box_w] = True
        inv = ~mask
        if not _touches_all_sides(mask) or not _touches_all_sides(inv):
            continue
        if len(_components(mask, connectivity=8)) != 1 or len(_components(inv, connectivity=8)) != 1:
            continue
        return mask
    raise RuntimeError("failed to sample a valid complementary fragment mask")


def _pick_window(mask: np.ndarray, rng: random.Random) -> tuple[int, int]:
    box_h, box_w = mask.shape
    interior = [
        (r, c)
        for r in range(1, box_h - 1)
        for c in range(1, box_w - 1)
        if bool(mask[r, c])
    ]
    if interior:
        return rng.choice(interior)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return rng.choice(coords)


def _render_fragment(mask: np.ndarray, color: int, window: tuple[int, int]) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    patch[window] = BLACK
    return patch


def _box_from_tl(top: int, left: int, height: int, width: int) -> tuple[int, int, int, int]:
    return top, left, top + height - 1, left + width - 1


def _separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return bool(
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _build_expected(case: dict) -> np.ndarray:
    out = np.full((case["grid_h"], case["grid_w"]), BG, dtype=int)
    for pair in case["pairs"]:
        _draw_frame(out, pair["out_top"], pair["out_left"], case["box_h"], case["box_w"], pair["color"], BG)
    return out


def _place_pair(
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    used_inputs: list[tuple[int, int, int, int]],
    used_outputs: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    for _ in range(300):
        mode = rng.choice(("stacked", "corner"))

        if mode == "stacked":
            out_top = rng.randrange(grid_h - box_h + 1)
            out_left = rng.randrange(grid_w - box_w + 1)
            max_dr = grid_h - box_h - out_top
            max_dc = grid_w - box_w - out_left
            offsets: list[tuple[int, int]] = []
            for dr in range(max_dr + 1):
                for dc in range(max_dc + 1):
                    if dr == 0 and dc == 0:
                        continue
                    if dr < box_h + 1 and dc < box_w + 1:
                        continue
                    offsets.append((dr, dc))
            if not offsets:
                continue
            p1 = (out_top, out_left)
            dr, dc = rng.choice(offsets)
            p2 = (out_top + dr, out_left + dc)
        else:
            max_top = grid_h - (2 * box_h + 1)
            max_left = grid_w - (2 * box_w + 1)
            if max_top < 0 or max_left < 0:
                continue
            out_top = rng.randrange(max_top + 1)
            out_left = rng.randrange(max_left + 1)
            dy = rng.randint(box_h + 1, grid_h - box_h - out_top)
            dx = rng.randint(box_w + 1, grid_w - box_w - out_left)
            if rng.random() < 0.5:
                p1 = (out_top, out_left + dx)
                p2 = (out_top + dy, out_left)
            else:
                p1 = (out_top + dy, out_left)
                p2 = (out_top, out_left + dx)

        b1 = _box_from_tl(p1[0], p1[1], box_h, box_w)
        b2 = _box_from_tl(p2[0], p2[1], box_h, box_w)
        bo = _box_from_tl(out_top, out_left, box_h, box_w)
        if not _separate(b1, b2, gap=1):
            continue
        if any(not _separate(b1, prev, gap=1) or not _separate(b2, prev, gap=1) for prev in used_inputs):
            continue
        if any(not _separate(bo, prev, gap=1) for prev in used_outputs):
            continue
        return p1, p2, (out_top, out_left)
    return None


def _sample_case(rng: random.Random) -> dict:
    box_h = rng.randint(4, 6)
    box_w = rng.randint(4, 7)
    pair_count = rng.choice((2, 2, 3, 3, 4))
    min_h = max(2 * box_h + 4, pair_count * (box_h + 1))
    min_w = max(2 * box_w + 4, pair_count * (box_w + 1))
    grid_h = rng.randint(min_h, min_h + 8)
    grid_w = rng.randint(min_w, min_w + 8)

    palette_size = rng.randint(2, min(5, pair_count + 1))
    palette = rng.sample(PALETTE, palette_size)

    used_signatures: set[tuple[int, ...]] = set()
    pairs: list[dict] = []
    used_inputs: list[tuple[int, int, int, int]] = []
    used_outputs: list[tuple[int, int, int, int]] = []

    for idx in range(pair_count):
        mask: np.ndarray | None = None
        comp: np.ndarray | None = None
        for _ in range(200):
            cand = _sample_mask(box_h, box_w, rng)
            inv = ~cand
            sig_a = _mask_signature(cand)
            sig_b = _mask_signature(inv)
            if sig_a in used_signatures or sig_b in used_signatures:
                continue
            mask = cand
            comp = inv
            used_signatures.add(sig_a)
            used_signatures.add(sig_b)
            break
        if mask is None or comp is None:
            raise RuntimeError("failed to build unique complementary masks")

        place = _place_pair(grid_h, grid_w, box_h, box_w, used_inputs, used_outputs, rng)
        if place is None:
            raise RuntimeError("failed to place a fragment pair")
        p1, p2, (out_top, out_left) = place

        color = palette[idx % palette_size]
        if idx >= palette_size:
            color = rng.choice(palette)

        win1 = _pick_window(comp, rng)
        win2 = _pick_window(mask, rng)

        frag1 = _render_fragment(mask, color, win1)
        frag2 = _render_fragment(comp, color, win2)
        if rng.random() < 0.5:
            frag1, frag2 = frag2, frag1
            mask, comp = comp, mask
            win1, win2 = win2, win1

        used_inputs.append(_box_from_tl(p1[0], p1[1], box_h, box_w))
        used_inputs.append(_box_from_tl(p2[0], p2[1], box_h, box_w))
        used_outputs.append(_box_from_tl(out_top, out_left, box_h, box_w))

        pairs.append(
            {
                "patch1": frag1,
                "patch2": frag2,
                "pos1": p1,
                "pos2": p2,
                "out_top": out_top,
                "out_left": out_left,
                "color": color,
            }
        )

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for pair in pairs:
        r1, c1 = pair["pos1"]
        r2, c2 = pair["pos2"]
        grid[r1 : r1 + box_h, c1 : c1 + box_w] = pair["patch1"]
        grid[r2 : r2 + box_h, c2 : c2 + box_w] = pair["patch2"]
    if _get_bg(grid) != BG:
        raise RuntimeError("azure must remain the dominant background")

    case = {
        "grid": grid,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "box_h": box_h,
        "box_w": box_w,
        "pairs": pairs,
    }
    return case


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        try:
            case = _sample_case(rng)
        except RuntimeError:
            continue
        expected = _build_expected(case)
        try:
            actual = main(case["grid"])
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return case["grid"]
    raise RuntimeError("failed to generate a valid puzzle instance")
