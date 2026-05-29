# Combo variant of 800d221b (index 20)
# combo_spec: D4/0+D17/0
# Name: 多hub网络+孤立路径干扰
#
# Design analysis:
# - Concrete behavior trace:
#   Original task has one 3x3 hub and treats every non-hub P component as a valid branch.
#   In this combo there are two 3x3 hubs. Each seed still has a P border and a mixed-color
#   interior, but its branch now belongs to exactly one hub, and extra isolated P segments
#   are present as decoys. The solver therefore cannot "remove one hub and recolor all
#   remaining P"; it must first find both hubs, then keep only those P components that touch
#   both a seed region and exactly one hub. Those valid components are recolored by seed
#   majority, while the isolated P segments stay P.
# - Complexity calibration:
#   The generator keeps only two seed colors plus one path color so the color rule stays
#   readable. The visible change is structural instead of numeric: two separated hub stars
#   and at least one untouched P segment remain after transformation. This makes the combo
#   easy to observe but not degenerate back to the single-hub original.
# - Hidden-assumption audit against the original solver:
#   The old "largest solid square" assumption becomes "two disjoint 3x3 hubs".
#   The old "all non-hub P components are meaningful" assumption breaks because D17 adds
#   isolated decoys, so valid branches must be filtered by hub/seed contact.
#   The old single center recolor step also breaks; each hub now computes its own center
#   color from the recolored branch cells adjacent to that hub only.
# - Placement legality and visual cleanliness:
#   The generator uses two well-separated hubs and four edge seeds (top/bottom for each hub),
#   so every branch has a clear territorial owner and the nearest-hub relationship is visible.
#   Branches are routed with thin orthogonal chains that do not overlap, and noise segments
#   are placed only in empty space with a one-cell 4-neighborhood buffer, preventing accidental
#   attachment to hubs, borders, or other noise.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import get_bg, label_mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

HUB_SIZE = 3
GRID_H_MIN = 18
GRID_H_MAX = 28
GRID_W_MIN = 22
GRID_W_MAX = 30
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
NOISE_SEGMENTS_MIN = 1
NOISE_SEGMENTS_MAX = 3
NOISE_SEGMENT_LEN_MIN = 1
NOISE_SEGMENT_LEN_MAX = 4
GEN_OUTER_RETRIES = 200


def _path_color_score(mask: np.ndarray) -> float:
    if not np.any(mask):
        return -1e9
    H, W = mask.shape
    ppad = np.pad(mask, 1)
    deg = (
        ppad[2:, 1:-1].astype(int)
        + ppad[:-2, 1:-1].astype(int)
        + ppad[1:-1, 2:].astype(int)
        + ppad[1:-1, :-2].astype(int)
    )
    score = 0.0
    for d in deg[mask]:
        if d == 1:
            score += 1.5
        elif d == 2:
            score += 3.0
        elif d == 3:
            score -= 1.0
        elif d == 4:
            score -= 4.0
    for r0 in range(H - HUB_SIZE + 1):
        for c0 in range(W - HUB_SIZE + 1):
            if np.all(mask[r0 : r0 + HUB_SIZE, c0 : c0 + HUB_SIZE]):
                score += 6.0
    return score


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors
    return int(max(colors, key=lambda c: _path_color_score(g == int(c))))


def _find_hubs(g: np.ndarray, P: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    H, W = g.shape
    cands: list[tuple[int, int]] = []
    for r0 in range(H - HUB_SIZE + 1):
        for c0 in range(W - HUB_SIZE + 1):
            if np.all(g[r0 : r0 + HUB_SIZE, c0 : c0 + HUB_SIZE] == int(P)):
                cands.append((int(r0), int(c0)))

    hubs: list[tuple[int, int]] = []
    for r0, c0 in cands:
        overlaps = False
        for rr, cc in hubs:
            if not (r0 + 2 < rr or rr + 2 < r0 or c0 + 2 < cc or cc + 2 < c0):
                overlaps = True
                break
        if not overlaps:
            hubs.append((r0, c0))
    assert len(hubs) == 2, f"expected 2 hubs, found {len(hubs)}"

    hub_id = np.zeros(g.shape, dtype=int)
    for hid, (r0, c0) in enumerate(hubs, start=1):
        hub_id[r0 : r0 + HUB_SIZE, c0 : c0 + HUB_SIZE] = int(hid)
    return hubs, hub_id


def _collect_seed_info(g: np.ndarray, bg: int, P: int) -> tuple[np.ndarray, dict[int, int]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(P) for c in range(cn, cx + 1))
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(P) for c in range(cn, cx + 1))
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(P) for r in range(rn, rx + 1))
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(P) for r in range(rn, rx + 1))
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        seed_major[int(sid)] = int(Counter(vals).most_common(1)[0][0])
    return seed_lab, seed_major


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a dual-hub path layout while ignoring detached path decoys.

    Input structure:
    - bg is the most frequent color.
    - One non-bg color P forms two separate networks, each centered on a solid 3x3 hub.
    - Colored seed rectangles (non-bg, non-P) sit on grid edges with a P border, and there
      may also be extra P line segments that touch neither hub.

    Rule:
    - Infer P, locate the two 3x3 hubs, and label all non-hub P components.
    - Keep only the P components that touch at least one seed region and exactly one hub;
      those are the real branches plus the seed borders they attach to.
    - Recolor each valid component to the majority color of the seed region it touches most,
      then recolor each hub center pixel to the most frequent color among its adjacent
      recolored branch cells. Detached P decoys remain unchanged.

    Key insight:
    The main ambiguity is not color but relevance: only hub-attached seed branches participate
    in the transform, so the solver must separate two hub territories from isolated P noise.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    P = _infer_path_color(g, bg)
    hubs, hub_id = _find_hubs(g, P)
    hub_mask = hub_id > 0
    seed_lab, seed_major = _collect_seed_info(g, bg, P)

    out = g.copy()
    branch_mask = (g == int(P)) & (~hub_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    ppad = np.pad(g == int(P), 1)
    p_deg = (
        ppad[2:, 1:-1].astype(int)
        + ppad[:-2, 1:-1].astype(int)
        + ppad[1:-1, 2:].astype(int)
        + ppad[1:-1, :-2].astype(int)
    )
    seedpad = np.pad(seed_lab > 0, 1)
    hubpad = np.pad(hub_mask, 1)
    adj_seed = seedpad[2:, 1:-1] | seedpad[:-2, 1:-1] | seedpad[1:-1, 2:] | seedpad[1:-1, :-2]
    adj_hub = hubpad[2:, 1:-1] | hubpad[:-2, 1:-1] | hubpad[1:-1, 2:] | hubpad[1:-1, :-2]
    assert not np.any((p_deg > 2) & branch_mask & ~adj_seed & ~adj_hub), "unexpected T-junction"

    hub_touch_cols: dict[int, Counter[int]] = {1: Counter(), 2: Counter()}
    valid_components = 0

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touch_hubs: set[int] = set()
        touch_seeds: Counter[int] = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    hid = int(hub_id[rr, cc])
                    sid = int(seed_lab[rr, cc])
                    if hid != 0:
                        touch_hubs.add(int(hid))
                    if sid != 0:
                        touch_seeds[int(sid)] += 1

        if len(touch_hubs) != 1 or not touch_seeds:
            continue

        valid_components += 1
        sid = int(touch_seeds.most_common(1)[0][0])
        col = int(seed_major[int(sid)])
        hid = int(next(iter(touch_hubs)))
        for r, c in pts:
            r = int(r)
            c = int(c)
            out[r, c] = int(col)
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(hub_id[rr, cc]) == hid:
                    hub_touch_cols[hid][int(col)] += 1

    assert valid_components >= 4, "expected at least four valid seed branches"

    for hid, (r0, c0) in enumerate(hubs, start=1):
        assert hub_touch_cols[hid], f"hub {hid} must touch a recolored branch"
        center_col = int(hub_touch_cols[hid].most_common(1)[0][0])
        out[r0 + 1, c0 + 1] = int(center_col)

    return out


def _fill_seed_interior(
    work: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    majority: int,
    minority: int,
) -> None:
    cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(cells)
    min_minor = max(1, int(round(n_int * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(round(n_int * MINORITY_FRAC_HI)))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(cells)
    for r, c in cells[:n_minor]:
        work[r, c] = int(minority)
    for r, c in cells[n_minor:]:
        work[r, c] = int(majority)


def _place_seed_border(
    work: np.ndarray,
    P: int,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    landing_side: str,
) -> list[tuple[int, int]]:
    H, W = work.shape
    landing_cells: list[tuple[int, int]] = []

    def _stamp(cells: list[tuple[int, int]], is_landing: bool) -> None:
        for r, c in cells:
            work[r, c] = int(P)
            if is_landing:
                landing_cells.append((int(r), int(c)))

    if ir0 > 0:
        _stamp([(ir0 - 1, c) for c in range(ic0, ic1 + 1)], landing_side == "top")
    if ir1 < H - 1:
        _stamp([(ir1 + 1, c) for c in range(ic0, ic1 + 1)], landing_side == "bottom")
    if ic0 > 0:
        _stamp([(r, ic0 - 1) for r in range(ir0, ir1 + 1)], landing_side == "left")
    if ic1 < W - 1:
        _stamp([(r, ic1 + 1) for r in range(ir0, ir1 + 1)], landing_side == "right")

    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[rc[0], rc[1]] = int(P)

    assert landing_cells
    return landing_cells


def _choose_seed_bbox(slot: str, H: int, W: int, left_hub_c0: int, right_hub_c0: int) -> tuple[int, int, int, int]:
    ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    if slot == "left_top":
        ir0 = 0
        ic_hi = max(0, left_hub_c0 - iw - 1)
        ic0 = random.randint(0, ic_hi)
    elif slot == "left_bottom":
        ir0 = H - ih
        ic_hi = max(0, left_hub_c0 - iw - 1)
        ic0 = random.randint(0, ic_hi)
    elif slot == "right_top":
        ir0 = 0
        ic_lo = right_hub_c0 + HUB_SIZE + 1
        ic0 = random.randint(ic_lo, W - iw)
    elif slot == "right_bottom":
        ir0 = H - ih
        ic_lo = right_hub_c0 + HUB_SIZE + 1
        ic0 = random.randint(ic_lo, W - iw)
    else:
        raise ValueError(slot)
    return int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1)


def _chain_cells(start: tuple[int, int], waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    cur_r, cur_c = start
    for nr, nc in waypoints:
        nr = int(nr)
        nc = int(nc)
        if cur_r != nr and cur_c != nc:
            raise ValueError("segments must be orthogonal")
        dr = 0 if cur_r == nr else (1 if nr > cur_r else -1)
        dc = 0 if cur_c == nc else (1 if nc > cur_c else -1)
        while (cur_r, cur_c) != (nr, nc):
            cur_r += dr
            cur_c += dc
            cells.append((int(cur_r), int(cur_c)))
    return cells


def _can_place_chain(
    work: np.ndarray,
    cells: list[tuple[int, int]],
    start: tuple[int, int],
    dest_set: set[tuple[int, int]],
) -> bool:
    H, W = work.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(work[r, c]) != 0:
            return False
    for i, (r, c) in enumerate(cells):
        allowed = {start}
        if i > 0:
            allowed.add(cells[i - 1])
        if i + 1 < len(cells):
            allowed.add(cells[i + 1])
        for dr, dc in _DIRS:
            rr, cc = int(r + dr), int(c + dc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in allowed or (rr, cc) in cell_set:
                continue
            v = int(work[rr, cc])
            if v == 0:
                continue
            if v != 0 and (rr, cc) in dest_set:
                continue
            return False
    return True


def _carve_seed_path(
    work: np.ndarray,
    start: tuple[int, int],
    landing_cells: list[tuple[int, int]],
    kind: str,
) -> bool:
    dest = min(landing_cells, key=lambda rc: abs(rc[1] - start[1]))
    lr, lc = dest
    if kind == "top":
        end_r = lr + 1
        stage_r = end_r + 1
        waypoints = [(stage_r, start[1]), (stage_r, lc), (end_r, lc)]
    elif kind == "bottom":
        end_r = lr - 1
        stage_r = end_r - 1
        waypoints = [(stage_r, start[1]), (stage_r, lc), (end_r, lc)]
    else:
        raise ValueError(kind)
    cells = _chain_cells(start, waypoints)
    if not cells or not _can_place_chain(work, cells, start, set(landing_cells)):
        return False
    for r, c in cells:
        work[r, c] = work[start]
    return True


def _place_noise_segments(work: np.ndarray, P: int) -> int:
    H, W = work.shape
    placed = 0
    target = random.randint(NOISE_SEGMENTS_MIN, NOISE_SEGMENTS_MAX)
    for _ in range(400):
        if placed >= target:
            break
        length = random.randint(NOISE_SEGMENT_LEN_MIN, NOISE_SEGMENT_LEN_MAX)
        dr, dc = random.choice(_DIRS[:2] + _DIRS[2:])
        r0 = random.randint(0, H - 1)
        c0 = random.randint(0, W - 1)
        cells = [(r0 + i * dr, c0 + i * dc) for i in range(length)]
        seg = set(cells)
        ok = True
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W) or int(work[r, c]) != 0:
                ok = False
                break
            for ndr, ndc in _DIRS:
                rr, cc = int(r + ndr), int(c + ndc)
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) in seg:
                    continue
                if int(work[rr, cc]) != 0:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c in cells:
            work[r, c] = int(P)
        placed += 1
    return int(placed)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_H_MIN, GRID_H_MAX)
    W = random.randint(GRID_W_MIN, GRID_W_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)
    work = np.zeros((H, W), dtype=int)

    hub_r0 = random.randint(max(4, H // 2 - 2), min(H - 7, H // 2))
    left_hub_c0 = random.randint(4, min(6, W // 3))
    right_hub_c0 = random.randint(max(left_hub_c0 + 8, W - 10), W - 8)

    work[hub_r0 : hub_r0 + HUB_SIZE, left_hub_c0 : left_hub_c0 + HUB_SIZE] = int(P)
    work[hub_r0 : hub_r0 + HUB_SIZE, right_hub_c0 : right_hub_c0 + HUB_SIZE] = int(P)

    left_major = random.choice([A, B])
    right_major = B if left_major == A else A

    seed_specs = [
        ("left_top", left_major, B if left_major == A else A, "bottom", (hub_r0, left_hub_c0 + 1), "top"),
        ("left_bottom", left_major, B if left_major == A else A, "top", (hub_r0 + 2, left_hub_c0 + 1), "bottom"),
        ("right_top", right_major, A if right_major == B else B, "bottom", (hub_r0, right_hub_c0 + 1), "top"),
        ("right_bottom", right_major, A if right_major == B else B, "top", (hub_r0 + 2, right_hub_c0 + 1), "bottom"),
    ]

    for slot, majority, minority, landing_side, _, _ in seed_specs:
        ir0, ic0, ir1, ic1 = _choose_seed_bbox(slot, H, W, left_hub_c0, right_hub_c0)
        _fill_seed_interior(work, ir0, ic0, ir1, ic1, int(majority), int(minority))
        landing_cells = _place_seed_border(work, int(P), ir0, ic0, ir1, ic1, landing_side)
        if slot == "left_top":
            left_top_landing = landing_cells
        elif slot == "left_bottom":
            left_bottom_landing = landing_cells
        elif slot == "right_top":
            right_top_landing = landing_cells
        else:
            right_bottom_landing = landing_cells

    path_plan = [
        ((hub_r0, left_hub_c0 + 1), left_top_landing, "top"),
        ((hub_r0 + 2, left_hub_c0 + 1), left_bottom_landing, "bottom"),
        ((hub_r0, right_hub_c0 + 1), right_top_landing, "top"),
        ((hub_r0 + 2, right_hub_c0 + 1), right_bottom_landing, "bottom"),
    ]
    for start, landing_cells, kind in path_plan:
        if not _carve_seed_path(work, start, landing_cells, kind):
            return None

    noise_count = _place_noise_segments(work, int(P))
    if noise_count < 1:
        return None

    inp = np.where(work == 0, int(bg), work).astype(int)
    if int(get_bg(inp)) != int(bg):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    hub_centers = [(hub_r0 + 1, left_hub_c0 + 1), (hub_r0 + 1, right_hub_c0 + 1)]
    if not all(int(out[r, c]) in (int(A), int(B)) for r, c in hub_centers):
        return None
    _, hub_id = _find_hubs(inp, int(P))
    noise_survivors = int(np.count_nonzero((inp == int(P)) & (out == int(P)) & (hub_id == 0)))
    if noise_survivors < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")
