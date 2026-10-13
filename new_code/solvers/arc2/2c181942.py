from __future__ import annotations
import random
import numpy as np
from arc_utils import STRUCT8, bbox_coords, get_bg, label_mask, paste, random_free_location

def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach monochrome sprites to the matching sides of a central hollow square.

    Input structure:
    - bg = background. A hollow NxN square (hub) with bg corners/interior and four differently-colored edges (top, bottom, left, right).
    - 2-4 monochrome sprites scattered on the grid, each matching one hub edge color.

    Rule:
    - For each hub edge color, find the matching sprite outside the hub.
    - Rotate the sprite (0/90/180/270°) so its contact edge has exactly N-2 contiguous cells aligned with the hub side, and the rotated sprite is mirror-symmetric along the contact-edge axis (UD for left/right, LR for top/bottom).
    - Translate it to adjoin that hub side. 
    - Clear everything else to bg; output is hub + attached sprites only.

    Key insight: Each edge color uniquely pairs a sprite to a side; symmetry along the contact-edge axis disambiguates the correct rotation.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c,n), (top,bot,left,right))
        for n in range(3, 7):
            for r in range(0, int(H - n + 1)):
                for c in range(0, int(W - n + 1)):
                    # corners must be bg
                    if (
                        int(g[int(r), int(c)]) != int(bg)
                        or int(g[int(r), int(c + n - 1)]) != int(bg)
                        or int(g[int(r + n - 1), int(c)]) != int(bg)
                        or int(g[int(r + n - 1), int(c + n - 1)]) != int(bg)
                    ):
                        continue
                    inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                    if not bool((inner == int(bg)).all()):
                        continue

                    top = g[int(r), int(c + 1) : int(c + n - 1)]
                    bot = g[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                    left = g[int(r + 1) : int(r + n - 1), int(c)]
                    right = g[int(r + 1) : int(r + n - 1), int(c + n - 1)]

                    if bool((top == int(bg)).any()) or bool((bot == int(bg)).any()) or bool((left == int(bg)).any()) or bool((right == int(bg)).any()):
                        continue
                    if (
                        len(set(map(int, top))) != 1
                        or len(set(map(int, bot))) != 1
                        or len(set(map(int, left))) != 1
                        or len(set(map(int, right))) != 1
                    ):
                        continue

                    cr = float(r + (n - 1) / 2)
                    cc = float(c + (n - 1) / 2)
                    score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                    cand = (
                        float(score),
                        (int(r), int(c), int(n)),
                        (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                    )
                    if best is None or float(cand[0]) < float(best[0]):
                        best = cand
        return best

    hub = _find_hub()
    assert not (hub is None)

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    target = int(n - 2)  # length of the hub edge segment (excluding corners)

    def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(pts)
        return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    def _edge_segment(mask: np.ndarray, side: str):
        pts = np.argwhere(mask)
        if pts.size == 0:
            return None
        if side == "top":
            rr = int(pts[:, 0].min())
            cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
            return ("h", int(rr), cols)
        if side == "bot":
            rr = int(pts[:, 0].max())
            cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
            return ("h", int(rr), cols)
        if side == "left":
            cc = int(pts[:, 1].min())
            rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
            return ("v", int(cc), rows)
        if side == "right":
            cc = int(pts[:, 1].max())
            rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
            return ("v", int(cc), rows)
        return None

    def _is_contig(seq: list[int]) -> bool:
        return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))

    # desired attachment geometry: adjacent (touching) the hub side segment
    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    for _side, (col, (need_edge, fixed, span)) in segs.items():
        col = int(col)
        m = (g == int(col)) & (~hub_mask)
        if not bool(m.any()):
            continue

        # Sprites can be only diagonally connected (e.g. V-shapes), so extract with 8-connectivity.
        labc, nc = label_mask(m, connectivity=8)
        best_id = None
        best_sz = -1
        for cid in range(1, int(nc) + 1):
            sz = int((labc == int(cid)).sum())
            if sz > best_sz:
                best_sz = sz
                best_id = int(cid)
        if best_id is None or best_sz <= 0:
            continue

        spr_mask = labc == int(best_id)
        spr_arr = g.copy()
        spr_arr[~spr_mask] = int(bg)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

        found = None  # (a, msk, dr, dc)
        for krot in range(4):
            a = np.rot90(spr_arr, int(krot))
            msk = np.rot90(spr_mask, int(krot))
            a, msk = _crop_to_mask(a, msk)

            # Attached sprite must be mirror-symmetric along the contact edge axis
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

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
            if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
                continue

            # ensure no conflicts with already-drawn hub (or previously placed sprites)
            ok = True
            for R, C in placed:
                val = int(a[int(R - dr), int(C - dc)])
                if val == int(bg):
                    continue
                if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
                    ok = False
                    break
            if not ok:
                continue

            found = (a, msk, int(dr), int(dc))
            break

        if found is None:
            continue

        a, msk, dr, dc = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

from scipy.ndimage import label as _label

def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Constructively build a sprite mask in the attached orientation for `side`.
    The mask is bool, 8-connected, has the correct symmetry and contact-edge length,
    and exactly 1 rotation (out of 4) is a valid attachment for any hub side.

    Strategy: build in canonical "left-attach" frame (flipud symmetric, contact on right col),
    then rotate to the target side.
    """
    # Build in "left-attach" frame: h rows, w cols.
    # Right column = contact edge with exactly `edge_len` contiguous filled cells.
    # flipud symmetric (rows mirror top-to-bottom).
    # h > w ensures not square, breaking 90-degree rotation ambiguity.
    # Also ensure NOT fliplr symmetric to prevent the complementary side matching.
    e = edge_len  # = 2 for N=4
    h = rng.choice([3, 4])  # height: 3 or 4 (must be > e to break symmetry)
    w = rng.randint(2, 4)   # width: 2-4 (cols including the contact column)

    half_h = (h + 1) // 2  # rows to decide (rest mirrored vertically)

    for _attempt in range(100):
        grid = np.zeros((h, w), dtype=bool)

        # Contact edge: rightmost column, middle `e` rows filled, rest empty.
        # For flipud symmetry with h=3 or h=4 and e=2:
        #   h=3: rows 0,2 empty; row 1 filled -> only 1 contiguous. No good.
        #   Actually for h=4, e=2: rows 1,2 filled (symmetric pair). Contact = [F,T,T,F]
        #   For h=3, e=2: rows 0,2 mirrored. If row 0=T, row 2=T, row 1=T -> 3 contig.
        #                  If row 0=F, row 2=F, row 1=T -> 1 contig. Not enough.
        #                  Need to allow row0=T,row1=T,row2=T -> 3 != e.
        # So for h=3, e=2 the rightmost col can't have exactly 2 contiguous cells
        # with flipud symmetry. Let me think differently.

        # The contact edge doesn't have to span the full height. It's the N-2 contiguous
        # cells on the attachment side. For "left-attach", the solver checks the RIGHT edge
        # of the cropped sprite: exactly e contiguous filled cells.

        # Build each row from right to left. Rightmost col is the contact column.
        # For flipud symmetry, row[i] mirrors row[h-1-i].
        # Decide top half, mirror to bottom.

        for ri in range(half_h):
            mirror_ri = h - 1 - ri
            for ci in range(w):
                if ci == w - 1:
                    # Contact column: fill rows that form the contiguous e-block
                    # Place the e-block centered. For h=4,e=2: rows 1,2. For h=3,e=2: impossible with strict flipud.
                    # Instead: just decide freely and check later
                    val = rng.random() < 0.5
                else:
                    val = rng.random() < 0.5
                grid[ri, ci] = val
                grid[mirror_ri, ci] = val

        # Check: rightmost column has exactly e contiguous filled cells
        right_col = grid[:, -1]
        # Find contiguous runs
        runs = []
        count = 0
        for v in right_col:
            if v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        if len(runs) != 1 or runs[0] != e:
            continue

        # Trim empty rows/cols
        rows_any = grid.any(axis=1)
        cols_any = grid.any(axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        grid = grid[r0:r1+1, c0:c1+1]

        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue

        # Check flipud symmetry
        if not np.array_equal(grid, np.flipud(grid)):
            continue

        # Check 8-connectivity
        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if num != 1:
            continue

        # Must NOT be fliplr symmetric (to avoid the other side matching)
        if np.array_equal(grid, np.fliplr(grid)):
            continue

        # Pixel count 4-12
        px = int(grid.sum())
        if px < 4 or px > 12:
            continue

        # Now verify exactly 1 valid attachment across all 4 rotations x 4 sides.
        # The solver, for each side, tries rot90 x 0-3 and checks:
        #   - correct symmetry (fliplr for top/bot, flipud for left/right)
        #   - contact edge = e contiguous cells on the correct side
        # We need exactly 1 such (rotation, side) pair across all possibilities.
        # Actually, the solver matches by color, so each sprite is only tested against
        # one side. But we need: for ANY side the solver might try, only 1 rotation works.
        # Since the solver already picks the first valid rotation, we just need that
        # for the intended side, exactly 1 rotation of the sprite is valid.

        # Check: for the canonical "left" side, how many rotations are valid?
        valid_for_left = 0
        for k in range(4):
            rot = np.rot90(grid, k)
            # Trim
            ra = rot.any(axis=1)
            ca = rot.any(axis=0)
            rr0, rr1 = np.where(ra)[0][[0, -1]]
            cc0, cc1 = np.where(ca)[0][[0, -1]]
            cr = rot[rr0:rr1+1, cc0:cc1+1]
            # Left side needs flipud + right-edge = e contiguous
            if not np.array_equal(cr, np.flipud(cr)):
                continue
            rc = cr[:, -1]
            runs2 = []
            cnt2 = 0
            for v in rc:
                if v: cnt2 += 1
                else:
                    if cnt2 > 0: runs2.append(cnt2)
                    cnt2 = 0
            if cnt2 > 0: runs2.append(cnt2)
            if len(runs2) == 1 and runs2[0] == e:
                valid_for_left += 1

        if valid_for_left != 1:
            continue

        # Success! Now rotate to target side.
        # grid is in "left-attach" frame (contact on right, flipud symmetric).
        # left -> as-is
        # right -> fliplr (contact moves to left side)
        # top -> rot90 CW (contact moves to bottom)
        # bot -> rot90 CCW (contact moves to top)
        if side == "left":
            mask = grid.copy()
        elif side == "right":
            mask = np.fliplr(grid)
        elif side == "top":
            mask = np.rot90(grid, -1)  # CW: right->bottom
        else:  # bot
            mask = np.rot90(grid, 1)   # CCW: right->top
        return mask

    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    N = 4
    edge_len = N - 2  # = 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    # Choose 4 distinct non-bg colors for hub edges
    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    # Choose K sides (2-4) to get sprites
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    # Generate sprite masks for each chosen side
    sprite_masks = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    # Compute max sprite extent to ensure enough margin for hub placement
    max_sprite_h = max(m.shape[0] for m in sprite_masks.values())
    max_sprite_w = max(m.shape[1] for m in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = max_extent + 2

    # Place hub near center with margin
    hub_r_min = margin
    hub_r_max = H - N - margin
    hub_c_min = margin
    hub_c_max = W - N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    # Build output grid
    out = np.full((H, W), bg, dtype=int)

    # Place hub
    # Top edge
    for ci in range(1, N - 1):
        out[hub_r, hub_c + ci] = top_col
    # Bottom edge
    for ci in range(1, N - 1):
        out[hub_r + N - 1, hub_c + ci] = bot_col
    # Left edge
    for ri in range(1, N - 1):
        out[hub_r + ri, hub_c] = left_col
    # Right edge
    for ri in range(1, N - 1):
        out[hub_r + ri, hub_c + N - 1] = right_col

    # Attach sprites to hub in output
    attached_sprites = {}  # side -> (arr, mask, sr, sc) in output coords
    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = side_colors[side]
        arr = np.where(mask, color, bg).astype(int)

        # Compute placement so sprite's contact edge is flush with hub side.
        # The solver's segs dict defines: for each side, the contact position and span.
        #   top -> sprite's bot row at hub_r-1, cols hub_c+1..hub_c+N-2
        #   bot -> sprite's top row at hub_r+N, cols hub_c+1..hub_c+N-2
        #   left -> sprite's right col at hub_c-1, rows hub_r+1..hub_r+N-2
        #   right -> sprite's left col at hub_c+N, rows hub_r+1..hub_r+N-2
        if side == "top":
            # Find first filled col on bottom row of sprite
            filled = np.where(mask[-1])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r - sh  # bottom row of sprite at row hub_r-1
        elif side == "bot":
            filled = np.where(mask[0])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r + N  # top row of sprite at row hub_r+N
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c - sw  # right col of sprite at col hub_c-1
        else:  # right
            filled = np.where(mask[:, 0])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c + N  # left col of sprite at col hub_c+N

        # Check bounds
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        # Check no overlap with existing content
        for r in range(sh):
            for c in range(sw):
                if mask[r, c] and out[sr + r, sc + c] != bg:
                    return None

        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, sr, sc)

    # Verify output is correct by running solver later (via oracle check)

    # Build input: hub + scattered sprites in random rotations + optional distractor
    inp = np.full((H, W), bg, dtype=int)
    # Place hub on input
    inp[hub_r:hub_r+N, hub_c:hub_c+N] = out[hub_r:hub_r+N, hub_c:hub_c+N]

    # Canvas for placement tracking
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, sr, sc = attached_sprites[side]
        # Apply a random non-identity rotation
        # But we need to ensure the solver will find the right rotation
        # The solver uses 8-connectivity labeling and picks the largest component of matching color
        # Apply a random rotation (1-3 quarter turns, avoiding identity)
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, k)
        rotated_mask = np.rot90(mask, k)

        # Crop to bounding box of the mask
        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        cropped_arr = rotated_arr[r0_idx:r1_idx+1, c0_idx:c1_idx+1]
        cropped_mask = rotated_mask[r0_idx:r1_idx+1, c0_idx:c1_idx+1]
        patch = np.where(cropped_mask, cropped_arr, bg).astype(int)

        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    # Optionally add a distractor (0 or 1)
    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = rng.choice(distractor_colors)
            # Make a small distractor sprite
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, dcol, bg).astype(int)
                try:
                    pr, pc = random_free_location(canvas, dist_arr, transparent=bg, padding=2, padding_connectivity=8)
                    paste(canvas, dist_arr, pr, pc, transparent=bg)
                    paste(inp, dist_arr, pr, pc, transparent=bg)
                except ValueError:
                    pass  # skip distractor if no room

    # Oracle check
    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")


# ---------------------------------------------------------------------------
# Markdown-sourced generator override
# ---------------------------------------------------------------------------
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path
import re as _re

_GENERATOR_MARKDOWN_PATH = _Path("/home/lixinyu/snap/BARC/ceshi1/13/inputs/manual_ARC_new_00001_candidates/attempt_01.md")


@_lru_cache(maxsize=1)
def _load_markdown_generator():
    text = _GENERATOR_MARKDOWN_PATH.read_text(encoding="utf-8")
    match = _re.search(r"```python\n(.*?)\n```", text, _re.DOTALL)
    if match is None:
        raise ValueError(f"No python code block found in {_GENERATOR_MARKDOWN_PATH}")

    namespace = {"np": np}
    exec(match.group(1), namespace)
    fn = namespace.get("generate_puzzle_input")
    if not callable(fn):
        raise ValueError(f"generate_puzzle_input() not found in {_GENERATOR_MARKDOWN_PATH}")
    return fn


def generate_input(seed: int | None = None) -> np.ndarray:
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    grid = _load_markdown_generator()(int(seed))
    return np.asarray(grid, dtype=int)
