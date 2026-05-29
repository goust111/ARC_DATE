# Combo variant of 800d221b (index 14)
# combo_spec: D2/3+D18/0
# Name: 种子数投票+单路径连接
#
# Design analysis
# a) Concrete behavior trace:
#    - Base task: if five seeds have majority colors [A, A, A, B, B], each seed may send multiple paths into the hub,
#      and the hub center follows the color with the most hub-adjacent recolored branch pixels.
#    - This combo removes multi-path ambiguity: every seed owns exactly one branch-plus-border component, so the
#      branch-to-seed relationship is one-to-one.
#    - The hub rule also changes: the center no longer counts hub-adjacent pixels. It counts connected seed regions.
#      In the generator below, the top B-major seed intentionally hugs one hub side with three adjacent branch pixels,
#      while the other four seeds touch the hub through one pixel each. Old counting would vote B by pixels (4 vs 3),
#      but the combo must vote A by seed count (3 seeds vs 2 seeds).
#
# b) Complexity calibration:
#    - The scene still contains one 3x3 hub, five mixed rectangular edge seeds, and thin orthogonal P branches, so the
#      task remains recognizable as the same object family.
#    - The only extra freedom is the contact multiplicity at the hub: one special branch touches the hub three times,
#      while all other branches touch it once. That is enough to make the new vote visibly different without adding
#      clutter or hidden metadata.
#    - Seed interiors still use exactly two visible colors with a strict majority in each seed. Repetition across five
#      seeds gives the solver repeated evidence for both "branch recolor by local seed majority" and "hub counts seeds,
#      not pixels".
#
# c) Hidden-assumption audit:
#    - Inferring P as the most network-like non-background color still holds because the input palette and object family
#      are unchanged.
#    - Detecting the hub as the largest solid P square still holds because the hub remains a clean 3x3 block.
#    - The original solver's T-junction check assumed non-hub path cells never exceed degree 2 unless they are next to
#      a seed. That breaks for the deliberate side-hugging branch: its middle cells have degree 3 only because they are
#      adjacent to the hub. The revised solver therefore allows degree > 2 on hub-adjacent non-hub P cells only.
#    - The new hub vote depends on a fresh one-to-one assumption: each seed must map to exactly one branch component.
#      The generator guarantees that by constructing exactly one path per seed and rejecting any accidental merge.
#
# d) Placement legality:
#    - Seeds are placed in fixed, separated edge slots: top-center, left-middle, right-middle, bottom-left, and
#      bottom-right. Their interiors never overlap, and their P borders remain disjoint.
#    - Paths are routed through reserved corridors: one top branch with a short horizontal run above the hub, two
#      horizontal side branches, and two bottom L-paths. The corridors are chosen so branches never cross and never
#      leave the grid.
#
# e) Visual observability:
#    - Humans can see five seeds but only two branch colors. Three seeds share one majority color and two seeds share
#      the other. The special top branch visibly touches the hub in three cells, so a pixel-counting heuristic would
#      favor its color. The output instead follows the color owned by more seeds, which is the intended contrast.
#
# f) Visual cleanliness:
#    - Only one branch is structurally unusual, and its extra hub contact is localized to a single short strip above
#      the hub. Everything else stays rectilinear and well separated, so the grid remains easy to parse.
#
# g) Inferability:
#    - No decision uses numeric color ids. Both branch recoloring and hub-center recoloring depend on visible counts:
#      local majority inside each seed, then number of seeds per branch color. The generator also rejects degenerate
#      samples where seed-vote and pixel-vote would accidentally agree.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

HUB_SIZE = 3
GRID_MIN = 18
GRID_MAX = 26
SEED_HEIGHT_CHOICES = [2, 3]
SEED_WIDTH_CHOICES = [2, 3]
TOP_SEED_HEIGHT_CHOICES = [2, 3]
TOP_SEED_WIDTH = 3
GEN_RETRIES = 200


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    H, W = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain non-background colors"

    def comp_deg_score(mask: np.ndarray) -> float:
        comps = components_mask(mask, connectivity=4)
        if not comps:
            return -1e9
        best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
        best_sz = int(best_pts.shape[0])

        m = np.zeros(mask.shape, dtype=bool)
        m[best_pts[:, 0], best_pts[:, 1]] = True

        deg2 = 0
        deg4 = 0
        for r, c in best_pts:
            r = int(r)
            c = int(c)
            d = 0
            if r > 0 and bool(m[r - 1, c]):
                d += 1
            if r + 1 < H and bool(m[r + 1, c]):
                d += 1
            if c > 0 and bool(m[r, c - 1]):
                d += 1
            if c + 1 < W and bool(m[r, c + 1]):
                d += 1
            if d == 2:
                deg2 += 1
            elif d == 4:
                deg4 += 1
        frac2 = deg2 / max(1, best_sz)
        frac4 = deg4 / max(1, best_sz)
        return float((frac2 - frac4) * best_sz)

    return int(max(colors, key=lambda c: comp_deg_score(g == int(c))))


def _find_hub(g: np.ndarray, path_color: int) -> tuple[np.ndarray, int, int, int]:
    H, W = g.shape
    mP = g == int(path_color)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = None
    for k in range(2, int(min(H, W)) + 1):
        found = None
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found

    assert best is not None, "failed to locate hub square"
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, k, r0, c0


def _extract_seed_majorities(g: np.ndarray, bg: int, path_color: int) -> tuple[np.ndarray, int, dict[int, int]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0, "empty seed component"
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())

        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(path_color) for c in range(cn, cx + 1)), "seed missing top P border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(path_color) for c in range(cn, cx + 1)), "seed missing bottom P border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(path_color) for r in range(rn, rx + 1)), "seed missing left P border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(path_color) for r in range(rn, rx + 1)), "seed missing right P border"

        counts = Counter(int(g[int(r), int(c)]) for r, c in pts)
        major_count = max(int(v) for v in counts.values())
        winners = [int(col) for col, cnt in counts.items() if int(cnt) == int(major_count)]
        assert len(winners) == 1, "seed interior must have a unique majority color"
        seed_major[int(sid)] = int(winners[0])

    return seed_lab, int(seed_n), seed_major


def _unique_max_color(counts: Counter[int]) -> int:
    assert counts, "expected a non-empty vote counter"
    best = max(int(v) for v in counts.values())
    winners = [int(col) for col, cnt in counts.items() if int(cnt) == int(best)]
    assert len(winners) == 1, "expected a unique winning color"
    return int(winners[0])


def _branch_data(
    g: np.ndarray,
    path_color: int,
    hub: np.ndarray,
    seed_lab: np.ndarray,
) -> tuple[np.ndarray, int, dict[int, int], dict[int, int]]:
    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(path_color)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]

    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]

    hpad = np.pad(hub, 1)
    adj_hub = hpad[2:, 1:-1] | hpad[:-2, 1:-1] | hpad[1:-1, 2:] | hpad[1:-1, :-2]

    bad_deg = (p_deg > 2) & pmask & ~hub & ~adj_seed & ~adj_hub
    assert not np.any(bad_deg), "path has a non-hub, non-seed T-junction"

    branch_to_seed: dict[int, int] = {}
    branch_hub_contacts: dict[int, int] = {}
    seed_seen: Counter[int] = Counter()

    H, W = g.shape
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        assert pts.size > 0, "empty branch component"
        touch_seed = Counter()
        hub_contacts = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            touch_hub_here = False
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touch_seed[int(sid)] += 1
                    if bool(hub[rr, cc]):
                        touch_hub_here = True
            if touch_hub_here:
                hub_contacts += 1

        assert touch_seed, "every branch component must touch a seed"
        assert len(touch_seed) == 1, "each branch component must belong to exactly one seed"
        sid = int(next(iter(touch_seed)))
        branch_to_seed[int(bid)] = int(sid)
        branch_hub_contacts[int(bid)] = int(hub_contacts)
        seed_seen[int(sid)] += 1

    assert branch_n == int(seed_lab.max()), "expected one branch component per seed"
    assert all(int(seed_seen[int(sid)]) == 1 for sid in range(1, int(seed_lab.max()) + 1)), "each seed must own one branch"
    assert all(int(branch_hub_contacts[int(bid)]) >= 1 for bid in range(1, int(branch_n) + 1)), "every branch must reach hub"
    return branch_lab, int(branch_n), branch_to_seed, branch_hub_contacts


def _vote_counters(g: np.ndarray) -> tuple[Counter[int], Counter[int], list[int]]:
    bg = int(get_bg(g))
    path_color = _infer_path_color(g, bg)
    hub, _k, _r0, _c0 = _find_hub(g, path_color)
    seed_lab, seed_n, seed_major = _extract_seed_majorities(g, bg, path_color)
    branch_lab, branch_n, branch_to_seed, branch_hub_contacts = _branch_data(g, path_color, hub, seed_lab)

    assert int(seed_n) == int(branch_n), "combo expects exactly one branch per seed"
    seed_votes: Counter[int] = Counter()
    pixel_votes: Counter[int] = Counter()
    contact_counts: list[int] = []
    for bid in range(1, int(branch_n) + 1):
        sid = int(branch_to_seed[int(bid)])
        col = int(seed_major[int(sid)])
        contacts = int(branch_hub_contacts[int(bid)])
        seed_votes[int(col)] += 1
        pixel_votes[int(col)] += int(contacts)
        contact_counts.append(int(contacts))
        assert contacts >= 1, "each branch must touch the hub"
    return seed_votes, pixel_votes, contact_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-seed path network where each seed owns a single branch.

    Input structure:
    - The background is the most common color.
    - One non-background color P forms a thin orthogonal network with a solid square hub.
    - Five seed regions appear as mixed-color rectangles touching grid edges, each wrapped by a P border
      on every non-edge side.

    Rule:
    - Infer P from the most network-like non-background component and locate the largest solid P square as the hub.
    - Treat every non-hub P connected component as one seed-owned branch-plus-border object, and recolor that entire
      object to the majority color of the unique seed region it surrounds or reaches.
    - Recolor only the hub center pixel, but vote by connected seed regions: each hub-reaching branch contributes one
      vote for its seed's majority color regardless of how many branch pixels touch the hub.

    Key insight:
    The changed hub rule counts seeds, not hub-adjacent pixels. A branch may touch the hub multiple times, but it still
    represents exactly one seed and therefore casts exactly one vote.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    path_color = _infer_path_color(g, bg)
    hub, k, r0, c0 = _find_hub(g, path_color)
    seed_lab, seed_n, seed_major = _extract_seed_majorities(g, bg, path_color)
    branch_lab, branch_n, branch_to_seed, _branch_hub_contacts = _branch_data(g, path_color, hub, seed_lab)

    assert int(seed_n) == int(branch_n), "combo expects exactly one branch per seed"

    out = g.copy()
    center_votes: Counter[int] = Counter()
    for bid in range(1, int(branch_n) + 1):
        sid = int(branch_to_seed[int(bid)])
        col = int(seed_major[int(sid)])
        pts = np.argwhere(branch_lab == int(bid))
        for r, c in pts:
            out[int(r), int(c)] = int(col)
        center_votes[int(col)] += 1

    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    out[cr, cc] = int(_unique_max_color(center_votes))
    return out


def _fill_seed_interior(
    g: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    major: int,
    minor: int,
) -> None:
    cells = [(r, c) for r in range(int(ir0), int(ir1) + 1) for c in range(int(ic0), int(ic1) + 1)]
    random.shuffle(cells)
    n = len(cells)
    n_minor = random.randint(max(1, n // 5), max(1, (n - 1) // 2))
    for r, c in cells[:n_minor]:
        g[int(r), int(c)] = int(minor)
    for r, c in cells[n_minor:]:
        g[int(r), int(c)] = int(major)


def _place_seed_border(
    g: np.ndarray,
    path_color: int,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
) -> None:
    H, W = g.shape
    if ir0 > 0:
        g[int(ir0 - 1), int(ic0) : int(ic1) + 1] = int(path_color)
    if ir1 < H - 1:
        g[int(ir1 + 1), int(ic0) : int(ic1) + 1] = int(path_color)
    if ic0 > 0:
        g[int(ir0) : int(ir1) + 1, int(ic0 - 1)] = int(path_color)
    if ic1 < W - 1:
        g[int(ir0) : int(ir1) + 1, int(ic1 + 1)] = int(path_color)
    if ir0 > 0 and ic0 > 0:
        g[int(ir0 - 1), int(ic0 - 1)] = int(path_color)
    if ir0 > 0 and ic1 < W - 1:
        g[int(ir0 - 1), int(ic1 + 1)] = int(path_color)
    if ir1 < H - 1 and ic0 > 0:
        g[int(ir1 + 1), int(ic0 - 1)] = int(path_color)
    if ir1 < H - 1 and ic1 < W - 1:
        g[int(ir1 + 1), int(ic1 + 1)] = int(path_color)


def _draw_line(g: np.ndarray, path_color: int, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if int(r0) == int(r1):
        lo, hi = sorted((int(c0), int(c1)))
        g[int(r0), lo : hi + 1] = int(path_color)
        return
    if int(c0) == int(c1):
        lo, hi = sorted((int(r0), int(r1)))
        g[lo : hi + 1, int(c0)] = int(path_color)
        return
    raise ValueError("draw_line requires axis-aligned endpoints")


def _seed_specs(H: int, W: int, hr0: int, hc0: int) -> list[dict]:
    top_h = random.choice(TOP_SEED_HEIGHT_CHOICES)
    left_w = random.choice(SEED_WIDTH_CHOICES)
    left_h = 3
    right_w = random.choice(SEED_WIDTH_CHOICES)
    right_h = 3
    bl_w = random.choice(SEED_WIDTH_CHOICES)
    bl_h = random.choice(SEED_HEIGHT_CHOICES)
    br_w = random.choice(SEED_WIDTH_CHOICES)
    br_h = random.choice(SEED_HEIGHT_CHOICES)

    return [
        {
            "name": "top_special",
            "bbox": (0, hc0, top_h - 1, hc0 + TOP_SEED_WIDTH - 1),
            "count_majority": "few",
        },
        {
            "name": "left_middle",
            "bbox": (hr0 - 1, 0, hr0 - 1 + left_h - 1, left_w - 1),
            "count_majority": "many",
        },
        {
            "name": "right_middle",
            "bbox": (hr0 - 1, W - right_w, hr0 - 1 + right_h - 1, W - 1),
            "count_majority": "many",
        },
        {
            "name": "bottom_left",
            "bbox": (H - bl_h, 0, H - 1, bl_w - 1),
            "count_majority": "many",
        },
        {
            "name": "bottom_right",
            "bbox": (H - br_h, W - br_w, H - 1, W - 1),
            "count_majority": "few",
        },
    ]


def _draw_paths(g: np.ndarray, path_color: int, hr0: int, hc0: int, specs: dict[str, tuple[int, int, int, int]]) -> None:
    hub_bottom = int(hr0 + HUB_SIZE)
    hub_mid_r = int(hr0 + 1)
    hub_top_strip_r = int(hr0 - 1)

    top_ir0, top_ic0, top_ir1, _top_ic1 = specs["top_special"]
    _draw_line(g, path_color, (int(top_ir1 + 1), int(top_ic0)), (int(hub_top_strip_r), int(top_ic0)))
    _draw_line(g, path_color, (int(hub_top_strip_r), int(hc0)), (int(hub_top_strip_r), int(hc0 + 2)))

    left_ir0, left_ic0, left_ir1, left_ic1 = specs["left_middle"]
    _draw_line(g, path_color, (hub_mid_r, int(left_ic1 + 1)), (hub_mid_r, int(hc0 - 1)))

    right_ir0, right_ic0, right_ir1, right_ic1 = specs["right_middle"]
    _draw_line(g, path_color, (hub_mid_r, int(hc0 + HUB_SIZE)), (hub_mid_r, int(right_ic0 - 1)))

    bl_ir0, bl_ic0, bl_ir1, bl_ic1 = specs["bottom_left"]
    bl_stem_c = int(hc0 - 2)
    bl_attach_r = int(bl_ir0 + (bl_ir1 - bl_ir0) // 2)
    _draw_line(g, path_color, (int(hub_bottom), int(bl_stem_c)), (int(hub_bottom), int(hc0)))
    _draw_line(g, path_color, (int(hub_bottom), int(bl_stem_c)), (int(bl_attach_r), int(bl_stem_c)))
    _draw_line(g, path_color, (int(bl_attach_r), int(bl_ic1 + 1)), (int(bl_attach_r), int(bl_stem_c)))

    br_ir0, br_ic0, br_ir1, br_ic1 = specs["bottom_right"]
    br_stem_c = int(hc0 + HUB_SIZE + 1)
    br_attach_r = int(br_ir0 + (br_ir1 - br_ir0) // 2)
    _draw_line(g, path_color, (int(hub_bottom), int(hc0 + 2)), (int(hub_bottom), int(br_stem_c)))
    _draw_line(g, path_color, (int(hub_bottom), int(br_stem_c)), (int(br_attach_r), int(br_stem_c)))
    _draw_line(g, path_color, (int(br_attach_r), int(br_stem_c)), (int(br_attach_r), int(br_ic0 - 1)))


def _generate_once() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    bg, path_color, color_a, color_b = random.sample(list(range(10)), 4)
    many_color = random.choice([color_a, color_b])
    few_color = int(color_b if many_color == color_a else color_a)

    hr0 = random.randint(6, H - 9)
    hc0 = random.randint(6, W - 9)

    g = np.full((H, W), int(bg), dtype=int)
    g[int(hr0) : int(hr0 + HUB_SIZE), int(hc0) : int(hc0 + HUB_SIZE)] = int(path_color)

    seed_specs = _seed_specs(H, W, hr0, hc0)
    spec_map: dict[str, tuple[int, int, int, int]] = {}
    for spec in seed_specs:
        ir0, ic0, ir1, ic1 = map(int, spec["bbox"])
        spec_map[str(spec["name"])] = (int(ir0), int(ic0), int(ir1), int(ic1))
        major = int(many_color if spec["count_majority"] == "many" else few_color)
        minor = int(few_color if major == many_color else many_color)
        _fill_seed_interior(g, ir0, ic0, ir1, ic1, major, minor)
        _place_seed_border(g, path_color, ir0, ic0, ir1, ic1)

    _draw_paths(g, path_color, hr0, hc0, spec_map)

    try:
        seed_votes, pixel_votes, contact_counts = _vote_counters(g)
        assert sorted(contact_counts) == [1, 1, 1, 1, 3], "expected one side-hugging branch and four single-contact branches"
        assert _unique_max_color(seed_votes) != _unique_max_color(pixel_votes), "degenerate sample: seed-vote matches pixel-vote"
        assert _unique_max_color(seed_votes) == int(many_color), "seed-vote winner should be the 3-seed color"
        assert _unique_max_color(pixel_votes) == int(few_color), "pixel-vote winner should be the 2-seed color"
        main(g)
    except Exception:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
