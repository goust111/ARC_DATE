import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _size_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _is_full_rectangle(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    return len(points) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _overlap_with_pad(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _paint_ring(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, black_only: bool = True
) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if black_only and int(grid[r, c]) != BG:
                continue
            grid[r, c] = color


def _infer_tile(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    visible = [
        (r, c, int(region[r, c]))
        for r in range(h)
        for c in range(w)
        if int(region[r, c]) != GREY
    ]
    best_key: tuple[int, int, int] | None = None
    best_tile: np.ndarray | None = None
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            key = (ph * pw, ph, pw)
            if best_key is not None and key >= best_key:
                continue
            tile = np.full((ph, pw), -1, dtype=int)
            ok = True
            for r, c, value in visible:
                rr = r % ph
                cc = c % pw
                cur = int(tile[rr, cc])
                if cur == -1:
                    tile[rr, cc] = value
                elif cur != value:
                    ok = False
                    break
            if ok and np.all(tile != -1):
                best_key = key
                best_tile = tile.copy()
    assert best_tile is not None, "failed to infer a complete periodic tile"
    return best_tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore grey occlusions in a periodic texture and mark each restored block
    with the color of the uniquely size-matched external fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    assert comps, "expected at least one non-black component"

    tex_idx = max(range(len(comps)), key=lambda idx: len(comps[idx]))
    tex_points = comps[tex_idx]
    assert _is_full_rectangle(tex_points), "texture region must be a solid rectangle"
    tr0, tc0, tr1, tc1 = _bbox(tex_points)
    texture = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert np.all(texture != BG), "texture rectangle should have no black holes"

    occs_local = _components(texture == GREY)
    assert occs_local, "expected at least one grey occlusion"

    fragments: dict[tuple[int, int], dict[str, object]] = {}
    for idx, comp in enumerate(comps):
        if idx == tex_idx:
            continue
        assert _is_full_rectangle(comp), "fragments must be solid rectangles"
        fr0, fc0, fr1, fc1 = _bbox(comp)
        patch = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
        colors = set(map(int, patch.reshape(-1)))
        assert len(colors) == 1, "each fragment must be monochrome"
        color = int(next(iter(colors)))
        assert color not in (BG, GREY), "fragment colors must differ from bg and grey"
        size = _size_from_bbox((fr0, fc0, fr1, fc1))
        assert size not in fragments, "fragment sizes must be unique"
        fragments[size] = {"color": color, "coords": comp}

    tile = _infer_tile(texture)
    ph, pw = tile.shape

    out = g.copy()
    occlusion_info: list[tuple[tuple[int, int, int, int], int]] = []
    for occ in occs_local:
        assert _is_full_rectangle(occ), "occlusions must be rectangles"
        lr0, lc0, lr1, lc1 = _bbox(occ)
        h_occ, w_occ = _size_from_bbox((lr0, lc0, lr1, lc1))
        key = (h_occ, w_occ)
        assert key in fragments, "every occlusion must have a unique size-matched fragment"
        color = int(fragments[key]["color"])
        gr0, gc0 = tr0 + lr0, tc0 + lc0
        gr1, gc1 = tr0 + lr1, tc0 + lc1
        for r in range(gr0, gr1 + 1):
            for c in range(gc0, gc1 + 1):
                out[r, c] = int(tile[(r - tr0) % ph, (c - tc0) % pw])
        occlusion_info.append(((gr0, gc0, gr1, gc1), color))

    for frag in fragments.values():
        for r, c in frag["coords"]:
            out[r, c] = BG

    for bbox, color in occlusion_info:
        _paint_ring(out, bbox, color, black_only=True)

    return out


def _axis_period(arr: np.ndarray, axis: int) -> int:
    size = arr.shape[axis]
    for period in range(1, size + 1):
        if size % period != 0:
            continue
        if axis == 0:
            base = arr[:period, :]
            tiled = np.tile(base, (size // period, 1))
        else:
            base = arr[:, :period]
            tiled = np.tile(base, (1, size // period))
        if np.array_equal(arr, tiled):
            return period
    return size


def _make_primitive_tile(
    rng: random.Random, ph: int, pw: int, colors: tuple[int, ...]
) -> np.ndarray | None:
    for _ in range(200):
        tile = np.array(
            [[rng.choice(colors) for _ in range(pw)] for _ in range(ph)], dtype=int
        )
        if len(set(map(int, tile.reshape(-1)))) < 2:
            continue
        if _axis_period(tile, axis=0) != ph:
            continue
        if _axis_period(tile, axis=1) != pw:
            continue
        return tile
    return None


def _visible_residues_ok(visible: np.ndarray, ph: int, pw: int) -> bool:
    for rr in range(ph):
        for cc in range(pw):
            if not np.any(visible[rr::ph, cc::pw]):
                return False
    return True


def _place_occlusions(
    rng: random.Random,
    th: int,
    tw: int,
    margins: dict[str, int],
    count: int,
) -> list[dict[str, object]] | None:
    occs: list[dict[str, object]] = []
    used_sizes: set[tuple[int, int]] = set()
    for side in rng.sample(SIDES, count):
        placed = False
        for _ in range(250):
            if side in ("top", "bottom"):
                h_max = min(4, th - 2, margins[side] - 1)
                w_max = min(max(2, tw - 1), 7)
                if h_max < 2 or w_max < 2:
                    return None
                h_occ = rng.randint(2, h_max)
                w_occ = rng.randint(2, w_max)
                r0 = 0 if side == "top" else th - h_occ
                c0 = rng.randint(0, tw - w_occ)
            else:
                h_max = min(max(2, th - 1), 7)
                w_max = min(4, tw - 2, margins[side] - 1)
                if h_max < 2 or w_max < 2:
                    return None
                h_occ = rng.randint(2, h_max)
                w_occ = rng.randint(2, w_max)
                r0 = rng.randint(0, th - h_occ)
                c0 = 0 if side == "left" else tw - w_occ

            size = (h_occ, w_occ)
            if size in used_sizes:
                continue

            bbox = (r0, c0, r0 + h_occ - 1, c0 + w_occ - 1)
            if any(_overlap_with_pad(bbox, occ["bbox"], pad=1) for occ in occs):
                continue

            occs.append({"side": side, "bbox": bbox, "size": size})
            used_sizes.add(size)
            placed = True
            break
        if not placed:
            return None
    return occs


def _place_fragment_bbox(
    rng: random.Random,
    canvas_shape: tuple[int, int],
    texture_bbox: tuple[int, int, int, int],
    occ: dict[str, object],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = canvas_shape
    tr0, tc0, tr1, tc1 = texture_bbox
    lr0, lc0, lr1, lc1 = occ["bbox"]
    side = str(occ["side"])
    fh, fw = occ["size"]
    gr0, gc0 = tr0 + lr0, tc0 + lc0

    for _ in range(200):
        if side == "top":
            max_r0 = tr0 - fh - 1
            if max_r0 < 0:
                return None
            r0 = rng.randint(0, max_r0)
            c0 = gc0
        elif side == "bottom":
            min_r0 = tr1 + 2
            max_r0 = h - fh
            if min_r0 > max_r0:
                return None
            r0 = rng.randint(min_r0, max_r0)
            c0 = gc0
        elif side == "left":
            max_c0 = tc0 - fw - 1
            if max_c0 < 0:
                return None
            r0 = gr0
            c0 = rng.randint(0, max_c0)
        else:
            min_c0 = tc1 + 2
            max_c0 = w - fw
            if min_c0 > max_c0:
                return None
            r0 = gr0
            c0 = rng.randint(min_c0, max_c0)

        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if _overlap_with_pad(bbox, texture_bbox, pad=1):
            continue
        if any(_overlap_with_pad(bbox, other, pad=1) for other in existing):
            continue
        return bbox
    return None


def _build_scene(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    ph = rng.randint(2, 4)
    pw = rng.randint(2, 4)
    rep_h = rng.randint(3, 5)
    rep_w = rng.randint(3, 5)
    th = ph * rep_h
    tw = pw * rep_w

    margins = {
        "top": rng.randint(4, 7),
        "bottom": rng.randint(4, 7),
        "left": rng.randint(4, 7),
        "right": rng.randint(4, 7),
    }
    h = th + margins["top"] + margins["bottom"]
    w = tw + margins["left"] + margins["right"]
    if not (18 <= h <= 30 and 18 <= w <= 30):
        return None

    color_count = rng.randint(2, 4)
    texture_colors = tuple(rng.sample(PALETTE, color_count))
    tile = _make_primitive_tile(rng, ph, pw, texture_colors)
    if tile is None:
        return None

    occ_count = rng.choice((2, 3, 4))
    occs = _place_occlusions(rng, th, tw, margins, occ_count)
    if occs is None:
        return None

    visible = np.ones((th, tw), dtype=bool)
    for occ in occs:
        r0, c0, r1, c1 = occ["bbox"]
        visible[r0 : r1 + 1, c0 : c1 + 1] = False
    if not _visible_residues_ok(visible, ph, pw):
        return None

    available_fragment_colors = [c for c in PALETTE if c not in texture_colors]
    if len(available_fragment_colors) < occ_count:
        return None
    fragment_colors = rng.sample(available_fragment_colors, occ_count)

    texture = np.tile(tile, (rep_h, rep_w))
    inp = np.full((h, w), BG, dtype=int)
    exp = np.full((h, w), BG, dtype=int)
    tr0 = margins["top"]
    tc0 = margins["left"]
    tr1 = tr0 + th - 1
    tc1 = tc0 + tw - 1
    texture_bbox = (tr0, tc0, tr1, tc1)
    inp[tr0 : tr1 + 1, tc0 : tc1 + 1] = texture
    exp[tr0 : tr1 + 1, tc0 : tc1 + 1] = texture

    fragment_bboxes: list[tuple[int, int, int, int]] = []
    for occ, color in zip(occs, fragment_colors):
        lr0, lc0, lr1, lc1 = occ["bbox"]
        gr0, gc0, gr1, gc1 = tr0 + lr0, tc0 + lc0, tr0 + lr1, tc0 + lc1
        inp[gr0 : gr1 + 1, gc0 : gc1 + 1] = GREY
        _paint_ring(exp, (gr0, gc0, gr1, gc1), color, black_only=True)

        frag_bbox = _place_fragment_bbox(rng, (h, w), texture_bbox, occ, fragment_bboxes)
        if frag_bbox is None:
            return None
        fragment_bboxes.append(frag_bbox)
        fr0, fc0, fr1, fc1 = frag_bbox
        inp[fr0 : fr1 + 1, fc0 : fc1 + 1] = color

    return inp, exp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _build_scene(rng)
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
