# Combo variant of 4c7dc4dd (index 16)
# combo_spec: D7/0+D20/1
# Name: 干净场景唯一规则

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


# Design analysis for the fused variant:
# - Concrete example: in the original task, a 6x6 frame could sit on a diagonal
#   stripe field, so `find_square_frames()` had to survive accidental same-color
#   borders in the background. Here the exterior is a single solid color, e.g. 9,
#   while frame interiors are still cleared to 0. A tiny fill example such as
#   corners -> rectangle therefore differs only inside the frames; nothing outside
#   can masquerade as another frame.
# - Unique-rule enforcement changes the solver/generator contract. The original
#   solver stopped at the first plausible rule and then used ad-hoc tie-breaks.
#   The fused version instead enumerates every supported rule match and expects the
#   training pair to admit exactly one directional explanation. To make that
#   visually inferable without hidden ordering conventions, the generator only
#   samples one-way rule families (`fill`, `endpoints`, `sym_fill`) and rejects any
#   scene where another rule or the reverse direction also fits.
# - Complexity is reduced but not collapsed: the scene still has four same-size
#   frames, disjoint train/test palettes, and three different rule families with
#   repeated structural evidence. What disappears is background clutter and
#   direction/rule ambiguity, not the analogy structure itself.
# - Implicit assumptions audited against `main()`:
#   1. frame detection assumes exactly four non-overlapping hollow squares of one
#      border color; generator keeps one per quadrant and uses a non-background
#      border color.
#   2. blank-frame detection assumes one all-zero interior; generator leaves one
#      slot empty and never uses background color 0 outside frames.
#   3. pair detection assumes exactly one duplicated non-zero palette signature;
#      generator uses disjoint train/test palettes and validates duplicate count.
#   4. rule inference assumes a unique directional match; generator measures the
#      full match set and rejects anything with count != 1.
#   5. application assumes the inferred rule remains non-trivial on the test
#      sprite; generator builds test sprites output-first and rejects identity or
#      out-of-range cases.
# - Placement legality and visual cleanliness: frames are sampled from separated
#   quadrants, never overlap, never leave the 30x30 canvas, and the single-color
#   exterior keeps object boundaries easy to read for a human solver.


H = W = 30
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
RULES = ["fill", "endpoints", "sym_fill"]
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
FILL_RECT_AREA_MIN, FILL_RECT_AREA_MAX = 5, 25
ENDPOINTS_LINE_LEN_MIN = 3
SYMFILL_KEEP0_PROB = 0.4
GEN_MAX_ATTEMPTS = 400


def find_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_color = int(np.bincount(g.ravel(), minlength=10).argmax())
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if col == bg_color:
                    continue
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                cands.append((size, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
        if len(picked) >= 4:
            break
    return picked


def stable_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    st = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            st.add(c)
    return st


def pick_var_color(x: np.ndarray, st: set[int]) -> int | None:
    vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
    if not vals:
        return None
    best = None
    for v in vals:
        cnt = int(np.sum(x == v))
        if best is None or cnt > best[0]:
            best = (cnt, v)
    return best[1]


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def sym_fill_base(arr: np.ndarray) -> np.ndarray | None:
    y = np.zeros_like(arr)
    for transform in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
        zz = transform(arr)
        rr, cc = np.where(zz != 0)
        for r, c in zip(rr.tolist(), cc.tolist()):
            val = int(zz[r, c])
            if y[r, c] != 0 and int(y[r, c]) != val:
                return None
            y[r, c] = val
    return y


def collect_templates(src: np.ndarray, dst: np.ndarray) -> list[tuple[str, dict]]:
    matches: list[tuple[str, dict]] = []
    st = stable_colors(src, dst)
    v = pick_var_color(dst, st) or pick_var_color(src, st)

    if np.array_equal(dst == 0, src != 0):
        nz = np.argwhere(dst != 0)
        if nz.size > 0:
            even_cols = set()
            odd_cols = set()
            for r, c in nz:
                col = int(dst[int(r), int(c)])
                if ((int(r) + int(c)) & 1) == 0:
                    even_cols.add(col)
                else:
                    odd_cols.add(col)
            if len(even_cols) == 1 and len(odd_cols) == 1:
                matches.append(("comp_checker", {"stable": st}))

    sym_src = sym_fill_base(src)
    if sym_src is not None and np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
        fill_cols = sorted(set(int(x) for x in dst[sym_src == 0].ravel() if int(x) != 0))
        if len(fill_cols) <= 1:
            fill_col = fill_cols[0] if fill_cols else None
            bad = (sym_src == 0) & (dst != 0) & (dst != fill_col) if fill_col is not None else False
            if not np.any(bad):
                keep0 = (sym_src == 0) & (dst == 0)
                matches.append(("sym_fill", {"stable": st, "keep0": keep0}))

    if v is not None:
        pred = src.copy()
        m0 = src == 0
        mv = src == v
        pred[m0] = v
        pred[mv] = 0
        for c in st:
            pred[dst == c] = c
        if np.array_equal(pred, dst):
            matches.append(("invert0", {"stable": st}))

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            matches.append(("sym", {"idx": idx}))

    if v is not None:
        coords = np.argwhere(src == v)
        cells = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in cells:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in cells:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        pred = src.copy()
        pred[src == v] = 0
        pred[keep] = v
        if np.array_equal(pred, dst):
            matches.append(("endpoints", {"stable": st}))

        pred = src.copy()
        h, w = pred.shape
        stable_mask = np.zeros_like(pred, dtype=bool)
        for c in st:
            stable_mask |= pred == c
        for r in range(h):
            nz = np.where(pred[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if pred[r, cc] == 0 and not stable_mask[r, cc]:
                        pred[r, cc] = v
        for c in range(w):
            nz = np.where(pred[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if pred[rr, c] == 0 and not stable_mask[rr, c]:
                        pred[rr, c] = v
        if np.array_equal(pred, dst):
            matches.append(("fill", {"stable": st}))

    return matches


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind == "sym":
        return symmetries(x)[params["idx"]].copy()

    st = set(params.get("stable", set()))
    v = pick_var_color(x, st)

    if kind == "comp_checker":
        nz = np.argwhere(x != 0)
        even_counts = {}
        odd_counts = {}
        for r, c in nz:
            col = int(x[int(r), int(c)])
            if ((int(r) + int(c)) & 1) == 0:
                even_counts[col] = even_counts.get(col, 0) + 1
            else:
                odd_counts[col] = odd_counts.get(col, 0) + 1
        if even_counts:
            c_even = max(even_counts.items(), key=lambda t: t[1])[0]
        elif odd_counts:
            c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
        else:
            return x.copy()
        c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
        h, w = x.shape
        y = np.zeros_like(x)
        for r in range(h):
            for c in range(w):
                if x[r, c] != 0:
                    y[r, c] = 0
                else:
                    y[r, c] = c_even if ((r + c) & 1) == 0 else c_odd
        return y

    if kind == "sym_fill":
        h, w = x.shape
        nz = np.argwhere(x != 0)
        if nz.size == 0:
            return x.copy()
        best = None
        counts = {}
        for r, c in nz:
            col = int(x[int(r), int(c)])
            d = min(int(r), int(c), h - 1 - int(r), w - 1 - int(c))
            counts[col] = counts.get(col, 0) + 1
            if best is None or d > best[0] or (d == best[0] and counts[col] > best[1]):
                best = (d, counts[col], col)
        fill_col = best[2]
        y = sym_fill_base(x)
        if y is None:
            return x.copy()
        keep0 = params.get("keep0", None)
        if keep0 is None or keep0.shape != y.shape:
            keep0 = np.zeros_like(y, dtype=bool)
        y[(y == 0) & ~keep0] = fill_col
        return y

    if v is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        m0 = x == 0
        mv = x == v
        y[m0] = v
        y[mv] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == v)
        cells = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in cells:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in cells:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        y = x.copy()
        y[x == v] = 0
        y[keep] = v
        return y

    if kind == "fill":
        y = x.copy()
        h, w = y.shape
        stable_mask = np.zeros_like(y, dtype=bool)
        for c in st:
            stable_mask |= y == c
        for r in range(h):
            nz = np.where(y[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if y[r, cc] == 0 and not stable_mask[r, cc]:
                        y[r, cc] = v
        for c in range(w):
            nz = np.where(y[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if y[rr, c] == 0 and not stable_mask[rr, c]:
                        y[rr, c] = v
        return y

    return x.copy()


def scene_roles(grid: np.ndarray):
    frames = find_square_frames(grid)
    if len(frames) < 3:
        return None
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, tuple(colors_present(inners[i], exclude={0}))) for i in nonblank]
    dup_pairs = []
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                dup_pairs.append((sigs[a][0], sigs[b][0]))
    if len(dup_pairs) == 1:
        pair = dup_pairs[0]
    elif dup_pairs:
        pair = dup_pairs[0]
    else:
        counts = [(i, int(np.count_nonzero(inners[i]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return None
    return {
        "frames": frames,
        "inners": inners,
        "blank_idx": blank_idx,
        "pair": pair,
        "test_idx": remaining[0],
        "dup_pair_count": len(dup_pairs),
    }


def all_directional_matches(src: np.ndarray, dst: np.ndarray) -> list[tuple[int, str, dict]]:
    matches = []
    for direction, (a, b) in enumerate(((src, dst), (dst, src))):
        for kind, params in collect_templates(a, b):
            matches.append((direction, kind, params))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the single intended analogy rule from a clean framed scene and apply it to the test sprite.

    Input structure:
    - The 30x30 grid contains four non-overlapping hollow square frames on a solid-color background.
    - Each frame interior is a black canvas holding a small sprite, except for one frame whose interior is fully blank.
    - Two non-blank interiors share the same non-zero palette and form the training example; the remaining non-blank interior is the test sprite.

    Rule:
    - Find the training pair by its shared palette, then evaluate every supported transformation family on both possible source→target directions.
    - In this clean variant the intended training pair is guaranteed to admit exactly one directional rule match, so ambiguity is treated as an invalid scene instead of being resolved by rule priority.
    - Apply that uniquely fitted rule to the test sprite and return only the transformed interior.

    Key insight:
    - Because the exterior background is uniform and noise-free, the hard part is no longer frame detection.
    - The task reduces to spotting the lone palette-matched pair and reading the one unambiguous structural change it demonstrates.
    """
    g = np.asarray(grid, dtype=int)
    roles = scene_roles(g)
    if roles is None or roles["dup_pair_count"] != 1:
        return g.copy()

    src_a = roles["inners"][roles["pair"][0]]
    src_b = roles["inners"][roles["pair"][1]]
    test_inner = roles["inners"][roles["test_idx"]]

    matches = all_directional_matches(src_a, src_b)
    if len(matches) == 1 and matches[0][0] == 0:
        _, kind, params = matches[0]
        return apply_template(kind, params, test_inner)
    return test_inner.copy()


def orbit_cells(n: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return sorted(cells)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def build_fill_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    dst = np.zeros((n, n), dtype=int)
    while True:
        h = rng.randint(2, n)
        w = rng.randint(2, n)
        if FILL_RECT_AREA_MIN <= h * w <= FILL_RECT_AREA_MAX:
            break
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)
    dst[r0 : r0 + h, c0 : c0 + w] = v
    src = np.zeros_like(dst)
    src[r0, c0] = v
    src[r0, c0 + w - 1] = v
    src[r0 + h - 1, c0] = v
    src[r0 + h - 1, c0 + w - 1] = v
    return src, dst


def build_endpoints_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    v = colors[0]
    src = np.zeros((n, n), dtype=int)
    dst = np.zeros_like(src)
    row_pairs = [(0, n - 1)]
    if n >= 5:
        row_pairs.extend([(0, n - 2), (1, n - 1)])
    r1, r2 = rng.choice(row_pairs)
    len1 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    len2 = rng.randint(ENDPOINTS_LINE_LEN_MIN, n)
    c1 = rng.randint(0, n - len1)
    c2 = rng.randint(0, n - len2)
    draw_line(src, r1, c1, end_r=r1, end_c=c1 + len1 - 1, color=v)
    draw_line(src, r2, c2, end_r=r2, end_c=c2 + len2 - 1, color=v)
    dst[r1, c1] = v
    dst[r1, c1 + len1 - 1] = v
    dst[r2, c2] = v
    dst[r2, c2 + len2 - 1] = v
    return src, dst


def build_sym_fill_pair(
    n: int,
    colors: list[int],
    rng: random.Random,
    keep0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fill_col, accent_col = colors
    src = np.zeros((n, n), dtype=int)
    orbits = []
    seen = set()
    for r in range(n):
        for c in range(n):
            orb = tuple(orbit_cells(n, r, c))
            if orb not in seen:
                seen.add(orb)
                orbits.append(list(orb))
    orbits.sort(
        key=lambda orb: (
            -max(min(r, c, n - 1 - r, n - 1 - c) for r, c in orb),
            -len(orb),
            orb[0],
        )
    )
    fill_orbit = next(orb for orb in orbits if not any(keep0[r, c] for r, c in orb))
    src[fill_orbit[0]] = fill_col
    accent_orbits = [orb for orb in orbits if orb != fill_orbit and not any(keep0[r, c] for r, c in orb)]
    picks = []
    for orb in accent_orbits:
        if len(orb) >= 2:
            picks.append(orb)
        if len(picks) >= 2:
            break
    if len(picks) < 2:
        raise ValueError("not enough accent orbits")
    for orb in picks:
        src[orb[0]] = accent_col
    dst = apply_template("sym_fill", {"keep0": keep0}, src)
    return src, dst


def choose_rule_setup(n: int, rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind == "sym_fill":
        color_count = 2
        keep0 = np.zeros((n, n), dtype=bool)
        if n == 6 or rng.random() < SYMFILL_KEEP0_PROB:
            for r, c in orbit_cells(n, 0, 0):
                keep0[r, c] = True
        params["keep0"] = keep0
    else:
        color_count = 1
    return kind, params, color_count


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind == "fill":
        return build_fill_pair(n, colors, rng)
    if kind == "endpoints":
        return build_endpoints_pair(n, colors, rng)
    if kind == "sym_fill":
        return build_sym_fill_pair(n, colors, rng, params["keep0"])
    raise ValueError(kind)


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    low, high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    return [
        (rng.randint(low, high), rng.randint(low, high)),
        (rng.randint(low, high), rng.randint(bottom_low, bottom_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(low, high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)),
    ]


def render_background(bg_color: int) -> np.ndarray:
    return np.full((H, W), bg_color, dtype=int)


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, int]:
    frame_pool = [c for c in ALL_COLORS if c not in used and c != 0]
    frame_color = rng.choice(frame_pool)
    bg_pool = [c for c in ALL_COLORS if c not in used and c not in {0, frame_color}]
    bg_color = rng.choice(bg_pool)
    return frame_color, bg_color


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_color: int,
) -> np.ndarray:
    grid = render_background(bg_color)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def _build_sprites(kind: str, setup_params: dict, n: int, p1: list[int], p2: list[int], rng: random.Random):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    train_matches = all_directional_matches(train_src, train_dst)
    if len(train_matches) != 1 or train_matches[0][0] != 0 or train_matches[0][1] != kind:
        return None
    inferred_kind, inferred_params = train_matches[0][1], train_matches[0][2]

    test_src, _ = build_training_pair(kind, setup_params, n, p2, rng)
    test_dst = apply_template(inferred_kind, inferred_params, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    roles = scene_roles(grid)
    if roles is None:
        return False
    frames = roles["frames"]
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    if sum(int(np.all(inner == 0)) for inner in roles["inners"]) != 1:
        return False
    if roles["dup_pair_count"] != 1:
        return False

    bg_mask = np.ones_like(grid, dtype=bool)
    for _, _, r0, c0, r1, c1 in frames:
        bg_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    bg_colors = set(int(x) for x in grid[bg_mask].ravel())
    if len(bg_colors) != 1 or 0 in bg_colors:
        return False

    src = roles["inners"][roles["pair"][0]]
    dst = roles["inners"][roles["pair"][1]]
    matches = all_directional_matches(src, dst)
    if len(matches) != 1 or matches[0][0] != 0:
        return False
    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))

    built = _build_sprites(kind, setup_params, n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_color = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(s, rng)
    train_slots, test_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, bg_color)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
