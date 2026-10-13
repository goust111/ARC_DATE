from __future__ import annotations

import math
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
PAIR_COUNT_RANGE = (2, 4)
FULL_RECT_SIDE_RANGE = (3, 6)
MAX_GRID_SIDE = 30


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as integer coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _fragment_from_coords(grid: np.ndarray, coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    vals = np.unique(grid[rows, cols])
    assert len(vals) == 1
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return {
        "color": int(vals[0]),
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "area": int(mask.sum()),
    }


def _winner(a: dict, b: dict) -> tuple[dict, dict]:
    if int(a["area"]) > int(b["area"]):
        return a, b
    if int(b["area"]) > int(a["area"]):
        return b, a
    ar0, ac0, _ar1, _ac1 = a["bbox"]
    br0, bc0, _br1, _bc1 = b["bbox"]
    if (int(ar0), int(ac0)) <= (int(br0), int(bc0)):
        return a, b
    return b, a


def _embed(mask: np.ndarray, H: int, W: int, top: int, left: int) -> np.ndarray:
    h, w = mask.shape
    full = np.zeros((H, W), dtype=bool)
    full[top : top + h, left : left + w] = mask
    return full


def _match_options(a: dict, b: dict) -> list[tuple[int, int, int, int, int, int]]:
    """
    Find full-rectangle embeddings where a and b are disjoint and complementary.

    Each option is:
      (full_h, full_w, a_top, a_left, b_top, b_left)
    """
    mask_a = a["mask"]
    mask_b = b["mask"]
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    total_area = int(a["area"]) + int(b["area"])

    opts: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    h_lo = max(ha, hb)
    h_hi = min(total_area, ha + hb)

    for H in range(h_lo, h_hi + 1):
        if total_area % H != 0:
            continue
        W = total_area // H
        if W < max(wa, wb) or W > wa + wb:
            continue

        a_embeds = []
        for at in range(H - ha + 1):
            for al in range(W - wa + 1):
                a_embeds.append((at, al, _embed(mask_a, H, W, at, al)))

        for bt in range(H - hb + 1):
            for bl in range(W - wb + 1):
                emb_b = _embed(mask_b, H, W, bt, bl)
                for at, al, emb_a in a_embeds:
                    if bool((emb_a & emb_b).any()):
                        continue
                    if not bool((emb_a | emb_b).all()):
                        continue
                    item = (int(H), int(W), int(at), int(al), int(bt), int(bl))
                    if item not in seen:
                        seen.add(item)
                        opts.append(item)
    return opts


def _solve_pairing(fragments: list[dict]) -> list[tuple[int, int, tuple[int, int, int, int, int, int]]]:
    n = len(fragments)
    edge_opts: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            opts = _match_options(fragments[i], fragments[j])
            if not opts:
                continue
            edge_opts[(i, j)] = opts
            neighbors[i].append(j)
            neighbors[j].append(i)

    for i in range(n):
        if not neighbors[i]:
            raise AssertionError("unmatched fragment")

    used: set[int] = set()

    def rec() -> list[tuple[int, int, tuple[int, int, int, int, int, int]]] | None:
        if len(used) == n:
            return []

        cand = [i for i in range(n) if i not in used]
        idx = min(cand, key=lambda i: sum(1 for j in neighbors[i] if j not in used))
        choices = sorted(j for j in neighbors[idx] if j not in used)
        for j in choices:
            key = (idx, j) if idx < j else (j, idx)
            if key not in edge_opts:
                continue
            used.add(idx)
            used.add(j)
            tail = rec()
            if tail is not None:
                return [(key[0], key[1], edge_opts[key][0])] + tail
            used.remove(idx)
            used.remove(j)
        return None

    paired = rec()
    if paired is None:
        raise AssertionError("failed to pair all fragments")
    return paired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, align them into a common rectangle, and fill the
    completed rectangle with the larger fragment's color, or the upper fragment's
    color when both areas are equal.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    fragments = [_fragment_from_coords(g, comp) for comp in _components(g != BG)]
    for i, j, option in _solve_pairing(fragments):
        full_h, full_w, a_top, a_left, b_top, b_left = option
        frag_a = fragments[i]
        frag_b = fragments[j]
        keep, move = _winner(frag_a, frag_b)

        if keep is frag_a:
            keep_top = int(a_top)
        else:
            keep_top = int(b_top)
        if keep is frag_a:
            keep_left = int(a_left)
        else:
            keep_left = int(b_left)

        mr = move["coords"][:, 0]
        mc = move["coords"][:, 1]
        out[mr, mc] = BG

        kr0, kc0, _kr1, _kc1 = keep["bbox"]
        full_r0 = int(kr0 - keep_top)
        full_c0 = int(kc0 - keep_left)
        out[full_r0 : full_r0 + full_h, full_c0 : full_c0 + full_w] = int(keep["color"])

    return out


def _random_bounded_composition(total: int, n: int, lo: int, hi: int) -> list[int] | None:
    if total < n * lo or total > n * hi:
        return None
    vals: list[int] = []
    rem = int(total)
    for i in range(n):
        left = n - i - 1
        cur_lo = max(lo, rem - left * hi)
        cur_hi = min(hi, rem - left * lo)
        if cur_lo > cur_hi:
            return None
        vals.append(random.randint(cur_lo, cur_hi))
        rem -= vals[-1]
    random.shuffle(vals)
    return vals


def _full_masks_from_profile(h: int, w: int, equal_area: bool) -> tuple[np.ndarray, np.ndarray] | None:
    area = h * w
    for _ in range(80):
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "vertical":
            lo, hi = h, h * (w - 1)
        else:
            lo, hi = w, (h - 1) * w

        if equal_area:
            if area % 2 == 1:
                return None
            target = area // 2
            if not (lo <= target <= hi):
                continue
        else:
            candidates = [v for v in range(lo, hi + 1) if v * 2 != area]
            if not candidates:
                continue
            target = random.choice(candidates)

        if orientation == "vertical":
            widths = _random_bounded_composition(target, h, 1, w - 1)
            if widths is None:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for r, width in enumerate(widths):
                mask[int(r), : int(width)] = True
        else:
            heights = _random_bounded_composition(target, w, 1, h - 1)
            if heights is None:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for c, height in enumerate(heights):
                mask[: int(height), int(c)] = True

        comp = ~mask
        if len(_components(mask)) != 1 or len(_components(comp)) != 1:
            continue
        if random.choice([True, False]):
            return mask, comp
        return comp, mask
    return None


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = np.argwhere(mask).T
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _rect_clear(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 + 1 < rr0 or rr1 + 1 < r0 or c1 + 1 < cc0 or cc1 + 1 < c0):
            return False
    return True


def _full_rect_for_fragment(frag: dict, r0: int, c0: int, full_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    off_r, off_c = frag["offset"]
    full_h, full_w = full_shape
    top = int(r0 - off_r)
    left = int(c0 - off_c)
    return (top, left, top + full_h - 1, left + full_w - 1)


def _inside_grid(rect: tuple[int, int, int, int], H: int, W: int) -> bool:
    r0, c0, r1, c1 = rect
    return 0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W


def _pair_winner_preview(info: dict, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> tuple[int, tuple[int, int, int, int]]:
    frag_a = info["fragments"][0]
    frag_b = info["fragments"][1]
    ha, wa = frag_a["mask"].shape
    hb, wb = frag_b["mask"].shape
    live_a = {"area": int(frag_a["area"]), "bbox": (int(pos_a[0]), int(pos_a[1]), int(pos_a[0] + ha - 1), int(pos_a[1] + wa - 1))}
    live_b = {"area": int(frag_b["area"]), "bbox": (int(pos_b[0]), int(pos_b[1]), int(pos_b[0] + hb - 1), int(pos_b[1] + wb - 1))}
    keep, _move = _winner(live_a, live_b)
    if keep is live_a:
        return 0, _full_rect_for_fragment(frag_a, int(pos_a[0]), int(pos_a[1]), info["full_shape"])
    return 1, _full_rect_for_fragment(frag_b, int(pos_b[0]), int(pos_b[1]), info["full_shape"])


def _choose_pair_count() -> int:
    return random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])


def _choose_full_dims(equal_area: bool, used: set[tuple[int, int]]) -> tuple[int, int] | None:
    dims = [
        (h, w)
        for h in range(FULL_RECT_SIDE_RANGE[0], FULL_RECT_SIDE_RANGE[1] + 1)
        for w in range(FULL_RECT_SIDE_RANGE[0], FULL_RECT_SIDE_RANGE[1] + 1)
        if (h, w) not in used and (not equal_area or (h * w) % 2 == 0)
    ]
    if not dims:
        return None
    return random.choice(dims)


def _grid_shape(full_dims: list[tuple[int, int]]) -> tuple[int, int] | None:
    footprint = sum(2 * (h + 2) * (w + 2) for h, w in full_dims)
    side = int(math.ceil(math.sqrt(float(footprint))))
    max_h = max(h for h, _w in full_dims)
    max_w = max(w for _h, w in full_dims)

    H_min = max(12, max_h + 2, side - 2)
    W_min = max(12, max_w + 2, side - 2)
    H_max = min(MAX_GRID_SIDE, H_min + 8)
    W_max = min(MAX_GRID_SIDE, W_min + 8)
    if H_min > H_max or W_min > W_max:
        return None
    return random.randint(H_min, H_max), random.randint(W_min, W_max)


def _build_pair_infos() -> tuple[list[dict], list[tuple[int, int]]] | None:
    pair_count = _choose_pair_count()
    branch_flags = [True, False]
    branch_flags.extend(random.choice([True, False]) for _ in range(pair_count - 2))
    random.shuffle(branch_flags)

    full_dims: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for equal_area in branch_flags:
        dim = _choose_full_dims(equal_area, used)
        if dim is None:
            return None
        used.add(dim)
        full_dims.append(dim)

    colors = random.sample(PALETTE, 2 * pair_count)
    infos: list[dict] = []
    for idx, (equal_area, (h, w)) in enumerate(zip(branch_flags, full_dims, strict=False)):
        masks = _full_masks_from_profile(h, w, equal_area)
        if masks is None:
            return None
        crop_a, off_a = _crop_mask(masks[0])
        crop_b, off_b = _crop_mask(masks[1])
        infos.append(
            {
                "equal_area": bool(equal_area),
                "full_shape": (int(h), int(w)),
                "fragments": [
                    {
                        "mask": crop_a,
                        "offset": (int(off_a[0]), int(off_a[1])),
                        "color": int(colors[2 * idx]),
                        "area": int(crop_a.sum()),
                    },
                    {
                        "mask": crop_b,
                        "offset": (int(off_b[0]), int(off_b[1])),
                        "color": int(colors[2 * idx + 1]),
                        "area": int(crop_b.sum()),
                    },
                ],
            }
        )
    return infos, full_dims


def _position_candidates(H: int, W: int, h: int, w: int, rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    out = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1) if _rect_clear(rects, r, c, h, w)]
    random.shuffle(out)
    return out


def _rects_without(rects: list[tuple[int, int, int, int]], rect: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    return [item for item in rects if item != rect]


def _place_pairs(H: int, W: int, pair_infos: list[dict]) -> np.ndarray | None:
    grid = np.zeros((H, W), dtype=int)
    input_rects: list[tuple[int, int, int, int]] = []
    output_rects: list[tuple[int, int, int, int]] = []

    order = list(range(len(pair_infos)))
    random.shuffle(order)
    for idx in order:
        info = pair_infos[idx]
        frag_a, frag_b = info["fragments"]
        ha, wa = frag_a["mask"].shape
        hb, wb = frag_b["mask"].shape

        for pos_a in _position_candidates(H, W, ha, wa, input_rects):
            rect_a = (int(pos_a[0]), int(pos_a[1]), int(pos_a[0] + ha - 1), int(pos_a[1] + wa - 1))
            cand_bases = input_rects + [rect_a]
            candidates_b = _position_candidates(H, W, hb, wb, cand_bases)
            if bool(info["equal_area"]):
                candidates_b = [pos for pos in candidates_b if pos[0] != pos_a[0]]
            for pos_b in candidates_b:
                rect_b = (int(pos_b[0]), int(pos_b[1]), int(pos_b[0] + hb - 1), int(pos_b[1] + wb - 1))
                keep_idx, full_rect = _pair_winner_preview(info, pos_a, pos_b)
                if not _inside_grid(full_rect, H, W):
                    continue
                other_input = rect_b if keep_idx == 0 else rect_a
                if not _rect_clear(output_rects, full_rect[0], full_rect[1], full_rect[2] - full_rect[0] + 1, full_rect[3] - full_rect[1] + 1):
                    continue
                if not _rect_clear([other_input] + input_rects, full_rect[0], full_rect[1], full_rect[2] - full_rect[0] + 1, full_rect[3] - full_rect[1] + 1):
                    continue

                for frag, pos in zip((frag_a, frag_b), (pos_a, pos_b), strict=False):
                    patch = _mask_to_patch(frag["mask"], int(frag["color"]))
                    ph, pw = patch.shape
                    grid[pos[0] : pos[0] + ph, pos[1] : pos[1] + pw][patch != BG] = patch[patch != BG]
                    frag["r0"] = int(pos[0])
                    frag["c0"] = int(pos[1])
                    frag["bbox"] = (int(pos[0]), int(pos[1]), int(pos[0] + ph - 1), int(pos[1] + pw - 1))

                input_rects.extend([rect_a, rect_b])
                output_rects.append(full_rect)
                break
            else:
                continue
            break
        else:
            return None

    return grid


def _render_expected(inp: np.ndarray, pair_infos: list[dict]) -> np.ndarray:
    out = inp.copy()
    for info in pair_infos:
        frag_a = info["fragments"][0]
        frag_b = info["fragments"][1]
        live_a = {
            "area": int(frag_a["area"]),
            "bbox": frag_a["bbox"],
            "coords": np.argwhere(frag_a["mask"]) + np.array([frag_a["r0"], frag_a["c0"]], dtype=int),
            "color": int(frag_a["color"]),
        }
        live_b = {
            "area": int(frag_b["area"]),
            "bbox": frag_b["bbox"],
            "coords": np.argwhere(frag_b["mask"]) + np.array([frag_b["r0"], frag_b["c0"]], dtype=int),
            "color": int(frag_b["color"]),
        }
        keep, move = _winner(live_a, live_b)
        mr = move["coords"][:, 0]
        mc = move["coords"][:, 1]
        out[mr, mc] = BG

        if keep is live_a:
            full_rect = _full_rect_for_fragment(frag_a, int(frag_a["r0"]), int(frag_a["c0"]), info["full_shape"])
        else:
            full_rect = _full_rect_for_fragment(frag_b, int(frag_b["r0"]), int(frag_b["c0"]), info["full_shape"])
        r0, c0, r1, c1 = full_rect
        out[r0 : r1 + 1, c0 : c1 + 1] = int(keep["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    built = _build_pair_infos()
    if built is None:
        return None
    pair_infos, full_dims = built
    shape = _grid_shape(full_dims)
    if shape is None:
        return None
    H, W = shape
    inp = _place_pairs(H, W, pair_infos)
    if inp is None:
        return None

    expected = _render_expected(inp, pair_infos)
    if len(_components(inp != BG)) != 2 * len(pair_infos):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
