# Combo variant of 2c181942 (index 30)
# combo_spec: D9/0+D12/0
# Name: 空间排列+四边必全

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste


"""
Design analysis for combo D9/0 + D12/0

a) Concrete behavior trace
   - Original task: each hub side color picks the sprite of the same color; 2-4 sprites may
     exist; output attaches only the present matches.
   - This combo changes the pairing rule itself. In the input, the four detached sprites are
     stacked from top to bottom and mean [top, right, bottom, left]. All four are always
     present. Example: the highest sprite may be red while the hub top edge is green; it still
     attaches to the top because row order, not color, decides the side.
   - Output geometry stays the original flush-outside attachment, so the visible novelty is
     concentrated in the input coding and the guarantee that all four sides are filled.

b) Complexity calibration
   - If sprite colors still matched hub colors, the task would silently collapse back to the
     original rule. To prevent that, generator uses four sprite colors disjoint from the four
     hub-edge colors, so color matching is always wrong.
   - If K were still 2-4, a solver could not know whether a missing row meant "missing side"
     or just "bottom rows absent". Therefore D12/0 is implemented literally: exactly four
     sprites, one for every side, in the full top/right/bottom/left row code.
   - Sprite family is kept small and repeated so the visual difference is obvious without making
     the grid messy.

c) Hidden-assumption audit of the original solver
   - Original loop `for side_color in hub edge colors -> find same-colored sprite` is invalid.
     Replaced with `extract the 4 external components -> sort by vertical position -> assign
     to [top, right, bot, left]`.
   - Original "missing sprite is allowed" branch is invalid. The new solver assumes 4 sprites
     and the generator guarantees that.
   - Original rotation/symmetry/contact-edge logic remains valid and is reused unchanged after
     side assignment, because D9/0 modifies matching, not attachment geometry.
   - Original largest-component-per-color heuristic is no longer applicable because color is
     decoupled from side. The new solver labels all non-hub external components directly.

d) Placement legality
   - Output is built first: hub plus the four attached sprites. Every attached placement is
     bounds-checked and overlap-checked against the already-built output.
   - Input is then rebuilt as an empty-hub scene plus four detached sprites placed in four
     non-overlapping row slots on the left side, with one empty row between neighboring slots,
     so 8-connectivity never merges them.

e) Visual observability
   - Across samples, sprite colors vary independently of hub edge colors, so a human cannot
     explain the mapping by color equality. The only stable cue is the top-to-bottom ordering.
   - Because all four sides are always present, the order code is complete in every example:
     highest sprite -> top, second -> right, third -> bottom, lowest -> left.

f) Visual cleanliness
   - There are only five objects in the input: one hub and four detached sprites.
   - The sprites share a left-side column region with clear vertical spacing, making the order
     cue obvious without extra legends or overlapping marks.

g) Post-test degeneration metric
   - Metric 1: number of colors shared between detached sprites and hub edges. It should be 0
     for every generated sample; otherwise the old color-matching rule could still solve it.
   - Metric 2: difference between this solver's output and the original solver's output on the
     same input. Since the original solver sees no color-matching sprites, it should fail to
     attach the four pieces, so outputs should differ strongly on nearly all samples.

h) New solver assumptions and generator guarantees
   - Assumption: exactly 4 external sprite components exist. Generator always creates 4 and no
     distractors.
   - Assumption: their vertical ordering is strict. Generator places them into separated row
     slots.
   - Assumption: each detached sprite still satisfies the original rotation constraints.
     Generator uses a fixed sprite family with exactly one valid rotation per target side.
   - Assumption: hub is still the unique central hollow square. Input keeps the hub unchanged
     from the original task, so the original hub detector still applies.
"""


SIDE_ORDER = ["top", "right", "bot", "left"]
BG = 8
HUB_N = 4

# Canonical attached mask for the LEFT hub side:
# - 8-connected
# - flipud symmetric
# - right edge has exactly 2 contiguous cells
# - not fliplr symmetric
BASE_LEFT_ATTACH_MASK = np.array(
    [
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 0],
    ],
    dtype=bool,
)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _find_hub(
    g: np.ndarray, bg: int
) -> tuple[float, tuple[int, int, int], tuple[int, int, int, int]] | None:
    H, W = g.shape
    best = None
    for n in range(3, 7):
        for r in range(H - n + 1):
            for c in range(W - n + 1):
                if g[r, c] != bg or g[r, c + n - 1] != bg or g[r + n - 1, c] != bg or g[r + n - 1, c + n - 1] != bg:
                    continue
                inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                if not bool((inner == bg).all()):
                    continue
                top = g[r, c + 1 : c + n - 1]
                bot = g[r + n - 1, c + 1 : c + n - 1]
                left = g[r + 1 : r + n - 1, c]
                right = g[r + 1 : r + n - 1, c + n - 1]
                if bool((top == bg).any()) or bool((bot == bg).any()) or bool((left == bg).any()) or bool((right == bg).any()):
                    continue
                if len(set(map(int, top))) != 1 or len(set(map(int, bot))) != 1 or len(set(map(int, left))) != 1 or len(set(map(int, right))) != 1:
                    continue
                cr = r + (n - 1) / 2.0
                cc = c + (n - 1) / 2.0
                score = abs(cr - (H - 1) / 2.0) + abs(cc - (W - 1) / 2.0)
                cand = (
                    float(score),
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or cand[0] < best[0]:
                    best = cand
    return best


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _mask_for_side(side: str) -> np.ndarray:
    if side == "left":
        return BASE_LEFT_ATTACH_MASK.copy()
    if side == "right":
        return np.fliplr(BASE_LEFT_ATTACH_MASK)
    if side == "top":
        return np.rot90(BASE_LEFT_ATTACH_MASK, -1)
    if side == "bot":
        return np.rot90(BASE_LEFT_ATTACH_MASK, 1)
    raise ValueError(f"unknown side: {side}")


def _place_hub_frame(out: np.ndarray, hub_r: int, hub_c: int, n: int, side_colors: dict[str, int], bg: int) -> None:
    out[hub_r : hub_r + n, hub_c : hub_c + n] = int(bg)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(side_colors["top"])
        out[hub_r + n - 1, hub_c + ci] = int(side_colors["bot"])
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(side_colors["left"])
        out[hub_r + ri, hub_c + n - 1] = int(side_colors["right"])


def _attached_sprite(color: int, side: str, hub_r: int, hub_c: int, n: int, bg: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    mask = _mask_for_side(side)
    arr = np.where(mask, int(color), int(bg)).astype(int)
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        sc = hub_c + 1 - int(filled[0])
        sr = hub_r - sh
    elif side == "bot":
        filled = np.where(mask[0])[0]
        sc = hub_c + 1 - int(filled[0])
        sr = hub_r + n
    elif side == "left":
        filled = np.where(mask[:, -1])[0]
        sr = hub_r + 1 - int(filled[0])
        sc = hub_c - sw
    else:
        filled = np.where(mask[:, 0])[0]
        sr = hub_r + 1 - int(filled[0])
        sc = hub_c + n
    return arr, mask, int(sr), int(sc)


def _extract_ordered_sprites(g: np.ndarray, bg: int, hub_mask: np.ndarray) -> list[dict]:
    comp_mask = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(comp_mask, connectivity=8)
    sprites: list[dict] = []
    for cid in range(1, int(nc) + 1):
        mask = lab == int(cid)
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = np.full_like(g, int(bg))
        arr[mask] = g[mask]
        arr, mask = _crop_to_mask(arr, mask)
        sprites.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center_r": float(r0 + r1) / 2.0,
                "center_c": float(c0 + c1) / 2.0,
                "area": int(pts.shape[0]),
                "arr": arr,
                "mask": mask,
            }
        )
    sprites.sort(key=lambda obj: (obj["bbox"][0], obj["center_r"], obj["center_c"]))
    return sprites


def _solve_attachment(
    arr: np.ndarray,
    mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    n: int,
    out: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    target = int(n - 2)
    specs = {
        "top": ("bot", int(hub_r - 1), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "bot": ("top", int(hub_r + n), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "left": ("right", int(hub_c - 1), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
        "right": ("left", int(hub_c + n), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
    }
    need_edge, fixed, span = specs[side]
    H, W = out.shape

    for krot in range(4):
        a = np.rot90(arr, int(krot))
        msk = np.rot90(mask, int(krot))
        a, msk = _crop_to_mask(a, msk)

        if need_edge in ("left", "right"):
            if not bool(np.array_equal(msk, np.flipud(msk))):
                continue
        else:
            if not bool(np.array_equal(msk, np.fliplr(msk))):
                continue

        seg = _edge_segment(msk, need_edge)
        if seg is None:
            continue

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            if len(rows) != int(target) or not _is_contig(rows):
                continue
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            if len(cols) != int(target) or not _is_contig(cols):
                continue
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(msk)}
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in placed):
            continue

        ok = True
        for rr, cc in placed:
            val = int(a[int(rr - dr), int(cc - dc)])
            if val == int(bg):
                continue
            if int(out[int(rr), int(cc)]) != int(bg) and int(out[int(rr), int(cc)]) != val:
                ok = False
                break
        if not ok:
            continue

        return a, msk, int(dr), int(dc)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach all four detached sprites to the hub. Side assignment comes from the sprites'
    top-to-bottom order in the input: [top, right, bottom, left].
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None
    _score, (hub_r, hub_c, n), _hub_colors = hub

    hub_mask = np.zeros_like(g, dtype=bool)
    hub_mask[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)] = g[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)]

    sprites = _extract_ordered_sprites(g, bg, hub_mask)
    assert len(sprites) == 4

    for side, sprite in zip(SIDE_ORDER, sprites):
        found = _solve_attachment(sprite["arr"], sprite["mask"], side, int(hub_r), int(hub_c), int(n), out, int(bg))
        assert found is not None
        arr, mask, dr, dc = found
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _build_output(
    H: int,
    W: int,
    hub_r: int,
    hub_c: int,
    side_colors: dict[str, int],
    sprite_colors: dict[str, int],
    bg: int,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, int, int]]] | None:
    out = np.full((H, W), int(bg), dtype=int)
    _place_hub_frame(out, int(hub_r), int(hub_c), HUB_N, side_colors, int(bg))

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in SIDE_ORDER:
        arr, mask, sr, sc = _attached_sprite(int(sprite_colors[side]), side, int(hub_r), int(hub_c), HUB_N, int(bg))
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr, cc in np.argwhere(mask):
            if int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                return None
        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached[side] = (arr, mask, int(sr), int(sc))

    return out, attached


def _rotated_patch(arr: np.ndarray, mask: np.ndarray, rng: random.Random, bg: int) -> np.ndarray:
    k = rng.choice([1, 2, 3])
    rotated_arr = np.rot90(arr, int(k))
    rotated_mask = np.rot90(mask, int(k))
    rotated_arr, rotated_mask = _crop_to_mask(rotated_arr, rotated_mask)
    return np.where(rotated_mask, rotated_arr, int(bg)).astype(int)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG

    H = rng.randint(21, 26)
    W = rng.randint(20, 26)

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    hub_palette = palette[:4]
    sprite_palette = rng.sample(palette[4:], 4)

    side_colors = {
        "top": int(hub_palette[0]),
        "bot": int(hub_palette[1]),
        "left": int(hub_palette[2]),
        "right": int(hub_palette[3]),
    }
    sprite_colors = {side: int(col) for side, col in zip(SIDE_ORDER, sprite_palette)}

    margin = 4
    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = max(8, margin)
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    built = _build_output(H, W, int(hub_r), int(hub_c), side_colors, sprite_colors, int(bg))
    if built is None:
        return None
    out, attached = built

    inp = np.full((H, W), int(bg), dtype=int)
    _place_hub_frame(inp, int(hub_r), int(hub_c), HUB_N, side_colors, int(bg))

    patches: list[np.ndarray] = []
    for side in SIDE_ORDER:
        arr, mask, _sr, _sc = attached[side]
        patches.append(_rotated_patch(arr, mask, rng, int(bg)))

    heights = [int(p.shape[0]) for p in patches]
    widths = [int(p.shape[1]) for p in patches]
    gap = 1
    total_h = sum(heights) + gap * (len(patches) - 1)
    if total_h > H - 2:
        return None

    max_w = max(widths)
    max_x = hub_c - max_w - 2
    if max_x < 1:
        return None

    start_r = 1 + rng.randint(0, H - 2 - total_h)
    start_c = rng.randint(1, max_x)

    cursor = int(start_r)
    for patch in patches:
        ph, pw = patch.shape
        if cursor + ph > H:
            return None
        if start_c + pw > hub_c - 1:
            return None
        paste(inp, patch, int(cursor), int(start_c), transparent=int(bg))
        cursor += int(ph) + gap

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
