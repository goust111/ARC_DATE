from __future__ import annotations
import numpy as np
from arc_utils import components_mask, find_components, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble colored sprites into a central bar's notches, chaining successive pieces along fixed-width strips.

    Input structure:
    - bg = most common color. Largest monochrome component is a solid "bar" with boundary notches (bg holes opening to its bbox edge).
    - Several smaller colored sprites scattered around the bar.

    Rule:
    - Each bar notch is filled by a root piece whose shape exactly covers the hole.
    - The root defines a strip (column-span or row-span) for its chain.
    - Successive pieces fill the bg cells inside the previous piece's bbox, locked to the same strip width.
    - Multiple chains grow independently from different bar notches.

    Key insight: The root piece's extent perpendicular to the chain direction locks the strip width for all subsequent pieces in that chain.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = set((int(r), int(c)) for r, c in bar["coords"])
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for r, c in bar_occ:
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        rel_set = set(rel)
        return {"rel": rel, "rel_set": rel_set, "h": h, "w": w}

    pieces = []
    for p in pieces_raw:
        # For this puzzle, pieces are used in their original orientation (no rotation / flip).
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def bar_notches(bar_mask, bar_r0, bar_c0):
        empty = ~bar_mask
        h, w = map(int, empty.shape)
        notches = []
        for pts in components_mask(empty, connectivity=4):
            r = pts[:, 0]
            c = pts[:, 1]
            top = int((r == 0).sum())
            bottom = int((r == h - 1).sum())
            left = int((c == 0).sum())
            right = int((c == w - 1).sum())

            # Keep only holes that open to the bar bbox boundary.
            if top + bottom + left + right == 0:
                continue

            hole_gl = set((bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts)

            # Axis is determined by which bbox side has the widest mouth.
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(v, top, left):
        return set((top + lr, left + lc) for lr, lc in v["rel"])

    def placement_pts(v, top, left, occ, must_cover=None, strip=None):
        if top < 0 or left < 0 or top + v["h"] > H or left + v["w"] > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or v["w"] != B - A + 1:
                    return None
            else:
                if top != A or v["h"] != B - A + 1:
                    return None
        pts = pixels(v, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis, A, cover, v):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, v):
        pos = set()
        for rH, cH in hole:
            for lr, lc in v["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    # Root candidates: fill the bar hole (no strip constraint yet).
    root_cands = []
    for ni, n in enumerate(notches):
        axis = n["axis"]

        for pi, p in enumerate(pieces):
            v = p["shape"]
            for top, left in hole_positions(n["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(n["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": v},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:20]]

    best_out = out.copy()
    best_key = None

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain, remaining_set):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = pixels(head["shape"], head["top"], head["left"])
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])

            cavity = {
                (r, c)
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if out_grid[r, c] == bg
            }
            if not cavity:
                return []

            moves = []
            for pi in remaining_set:
                v = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, A, cavity, v):
                    pts = placement_pts(v, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, v))
            return moves

        # Try extending any chain.
        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            # small branching factor is enough for this puzzle
            for pi, top, left, pts, v in moves[:8]:
                # apply
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                # undo
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                # restore bar pixels (roots are always non-overlapping)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    # Try small root subsets (1..3) from top candidates.
    from itertools import combinations

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = bar_color

            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = pixels(root["shape"], root["top"], root["left"])
                if pts & occ0:
                    ok = False
                    break
                used_p.add(root["pi"])
                used_n.add(root["ni"])
                for r, c in pts:
                    out0[r, c] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append({"axis": root["axis"], "A": root["A"], "B": root["B"], "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]}})

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path
import re as _re

_GENERATOR_MARKDOWN_PATH = _Path("/home/lixinyu/snap/BARC/ceshi1/3/inputs/manual_ARC_new_00001_candidates/attempt_01.md")


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
