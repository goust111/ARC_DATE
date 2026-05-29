# Combo variant of e8686506 (index 7)
# combo_spec: D6/1+D14/2
# Name: 对称碎片同色+颜色交换

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior and flow impact:
#    In the original task, the assembled rectangle is directly the output and mirror symmetry
#    is enforced by geometry. In this combo, the scattered non-frame pieces first reconstruct a
#    pre-output rectangle where each left-right mirror pair uses one shared color. Example:
#    if the top pair is an L on the left and a mirrored L on the right, both arrive in color 2.
#    After assembly, every non-frame color is globally replaced by a different color via a fixed
#    cycle (2->3, 3->8, 8->2), so the pair stays paired spatially but changes appearance.
#    The flow is therefore input pieces -> recover paired same-color families -> recolor families.
# b) Complexity calibration:
#    If every pair got its own arbitrary color, the task would collapse into a trivial copy with
#    recoloring; if colors were reused without structure, shape matching would become ambiguous.
#    The generator therefore uses four mirror-pair families but only three non-frame colors, so
#    one color repeats across distinct families. This keeps the pairing clue visible while still
#    requiring the solver to recover all slots from geometry instead of memorizing a one-to-one
#    color-to-position map.
# c) Hidden assumptions audit:
#    The original `mirror_ok` placement rule assumed mirror cells must match during assembly.
#    That assumption is no longer the placement mechanism; frame-derived slot geometry now fixes
#    positions. The D6 reference assumed each oriented signature had its own color, but D14/2
#    breaks that by forcing mirror mates to share a color. The new solver therefore groups input
#    pieces by canonical mirror-family signature, not by oriented signature, and only afterward
#    applies the color transform.
# d) Placement legality:
#    The output is built first on a 10x5 rectangle. A single 8-connected frame occupies the
#    complement of four mirrored hole families, so all non-frame pieces fit exactly in those holes.
#    During scattering, the whole frame bbox is blocked and preserved, while the remaining patches
#    are placed outside with Chebyshev padding 1, preventing overlap and accidental touching.
# e) Visual observability:
#    In every sample, mirror partners are visibly the same color in the input, which exposes the
#    pairing rule directly. In the output, those same paired regions all change color according to
#    the same global 3-cycle. Across a few samples, a human can see both layers: same-color pairing
#    is a clue for reconstruction, and the final board is not a plain reassembly because every such
#    family changes to another recurring color.
# f) Visual cleanliness:
#    The hole template keeps the center column inside the frame, so there are no unpaired center
#    pieces competing with the mirrored-pair story. Family colors are assigned with adjacency
#    constraints, so same-color regions do not merge into larger blobs either before or after the
#    swap.
# g) Color-change inferability:
#    The swap is a fixed permutation on three visible non-frame colors rather than a per-instance
#    hidden remap. Because at least one non-frame color is reused on multiple pair families in each
#    sample, the examples provide repeated evidence that the recoloring depends on color class after
#    pairing, not on an invisible ordering of coordinates or numeric color ids.
# New solver assumptions checked against the generator:
#    - Total foreground area is always 50, giving the unique closest-to-square portrait shape 10x5.
#    - The input always contains one grouped 8-connected frame piece whose bbox is the output.
#    - Every non-frame piece belongs to one mirrored family obtained from frame holes; no center-only pieces exist.
#    - All input components from the same canonical mirror family share exactly one color.
#    - The only recolored cells are non-frame colors in the fixed swap cycle.


_OUTPUT_H = 10
_OUTPUT_W = 5
_NONFRAME_COLORS = (2, 3, 8)
_FRAME_COLOR = 1
_SWAP_MAP = {2: 3, 3: 8, 8: 2}
_MAX_PIECE_AREA = 6
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300
_MIN_FAMILIES = 4

_LEFT_BLOCKS = [
    (0, [[(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 1)]]),
    (3, [[(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 1)]]),
    (6, [[(0, 0), (0, 1)]]),
    (8, [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1)]]),
]


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _mirror_sig(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    h, w, coords = sig
    return h, w, tuple(sorted((int(r), w - 1 - int(c)) for r, c in coords))


def _canon_sig(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    msig = _mirror_sig(sig)
    return sig if sig <= msig else msig


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _chunk_component(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    chunks: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        queue = [start]
        seen = {start}
        chunk: list[tuple[int, int]] = []
        while queue and len(chunk) < _MAX_PIECE_AREA:
            r, c = queue.pop(0)
            if (r, c) not in remaining:
                continue
            remaining.remove((r, c))
            chunk.append((r, c))
            nbrs = []
            for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
                if (rr, cc) in remaining and (rr, cc) not in seen:
                    nbrs.append((rr, cc))
                    seen.add((rr, cc))
            queue.extend(sorted(nbrs))
        chunks.append(sorted(chunk))
    return chunks


def _slotize(frame_mask: np.ndarray) -> list[dict]:
    H, W = frame_mask.shape
    open_mask = ~frame_mask
    slots: list[dict] = []

    left_open = open_mask[:, : W // 2]
    for comp in components_mask(left_open, connectivity=4):
        left_cells = [(int(r), int(c)) for r, c in comp]
        for chunk in _chunk_component(left_cells):
            arr = np.array(chunk, dtype=int)
            sig = _piece_sig(arr)
            slots.append({"cells": chunk, "sig": sig, "family": _canon_sig(sig)})
            mirror = sorted((int(r), W - 1 - int(c)) for r, c in chunk)
            marr = np.array(mirror, dtype=int)
            msig = _piece_sig(marr)
            slots.append({"cells": mirror, "sig": msig, "family": _canon_sig(msig)})
    return slots


def _family_groups(slots: list[dict]) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]]:
    out: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for slot in slots:
        out.setdefault(slot["family"], []).append(slot["cells"])
    return out


def _apply_swap(pre: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    out = pre.copy()
    for col, new_col in _SWAP_MAP.items():
        out[(pre == col) & (~frame_mask)] = int(new_col)
    return out


def _is_lr_symmetric_bitmap(parts: list[tuple[int, np.ndarray]], H: int, W: int) -> bool:
    all_pts = np.concatenate([p for _, p in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((H, W), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W:
            if _is_lr_symmetric_bitmap([(c1, p1)], H, W):
                frame_parts = [(c1, p1)]
                break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_lr_symmetric_bitmap([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain grouped LR-symmetric frame part(s)"
    return frame_parts


def _infer_dims(total_area: int) -> tuple[int, int]:
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    return best_dims[0]


def _build_pre_from_input(g: np.ndarray, bg: int) -> tuple[np.ndarray, dict]:
    total_area = int((g != int(bg)).sum())
    assert total_area > 0, "grid must contain foreground pieces"
    H, W = _infer_dims(total_area)
    assert (H, W) == (_OUTPUT_H, _OUTPUT_W), "generator guarantees a 10x5 output rectangle"

    frame_parts = _find_frame_parts(g, bg, H, W)
    frame_input_mask = np.zeros_like(g, dtype=bool)
    for _, pts in frame_parts:
        for r, c in pts:
            frame_input_mask[int(r), int(c)] = True

    pre = np.full((H, W), -1, dtype=int)
    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            pre[int(r) - fr0, int(c) - fc0] = int(col)

    frame_mask = pre != -1
    slots = _slotize(frame_mask)
    families = _family_groups(slots)
    assert len(families) == _MIN_FAMILIES, "generator uses exactly four mirrored families"

    input_family_colors: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[int]] = {}
    seen = 0
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == col, connectivity=4):
            if all(frame_input_mask[int(r), int(c)] for r, c in pts):
                continue
            family = _canon_sig(_piece_sig(pts))
            input_family_colors.setdefault(family, []).append(int(col))
            seen += 1

    assert seen == len(slots), "non-frame piece count must match reconstructed slots"

    family_color: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for family, groups in families.items():
        cols = input_family_colors.get(family, [])
        assert len(cols) == len(groups), "each family must have matching multiplicity"
        uniq = set(cols)
        assert len(uniq) == 1, "mirror-family pieces must share one input color"
        col = int(next(iter(uniq)))
        assert col in _SWAP_MAP, "non-frame pieces must use the swap-palette colors"
        family_color[family] = col
        for cells in groups:
            for r, c in cells:
                pre[r, c] = col

    assert np.all(pre >= 0), "reconstructed pre-swap board must be dense"
    return pre, {"frame_mask": frame_mask, "families": families, "family_color": family_color, "dims": (H, W)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a mirror-paired rectangle from scattered fragments, then recolor each paired family.

    Input structure:
    - The background is the dominant color.
    - One grouped 8-connected frame fragment still occupies its output-relative rectangle.
    - Every other fragment is a single-color piece, and left-right mirror mates use the same input color.

    Rule:
    - Infer the output rectangle from total foreground area and recover the grouped frame.
    - Use the frame holes to recover all mirrored piece slots, then refill each mirror-family with the shared color of its scattered matching pieces.
    - After the paired same-color rectangle is reconstructed, apply the fixed non-frame color swap to obtain the final output.

    Key insight:
    - Color now serves two roles in sequence: first it identifies which scattered pieces are mirror partners, then that recovered family color is transformed by a consistent swap rule.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pre, meta = _build_pre_from_input(g, bg)
    out = _apply_swap(pre, meta["frame_mask"])
    return out.astype(int)


def _build_frame_mask() -> np.ndarray:
    open_mask = np.zeros((_OUTPUT_H, _OUTPUT_W), dtype=bool)
    for row0, options in _LEFT_BLOCKS:
        shape = random.choice(options)
        for dr, dc in shape:
            r, c = row0 + dr, dc
            open_mask[r, c] = True
            open_mask[r, _OUTPUT_W - 1 - c] = True
    return ~open_mask


def _family_adjacency(slots: list[dict], frame_mask: np.ndarray) -> dict[str, dict]:
    H, W = frame_mask.shape
    cell_to_slot: dict[tuple[int, int], int] = {}
    for i, slot in enumerate(slots):
        for cell in slot["cells"]:
            cell_to_slot[cell] = i

    adjacency: dict[tuple[int, int, tuple[tuple[int, int], ...]], set] = {}
    frame_adj: dict[tuple[int, int, tuple[tuple[int, int], ...]], set[int]] = {}
    for slot in slots:
        family = slot["family"]
        adjacency.setdefault(family, set())
        frame_adj.setdefault(family, set())

    for i, slot in enumerate(slots):
        family = slot["family"]
        for r, c in slot["cells"]:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if frame_mask[rr, cc]:
                    frame_adj[family].add(_FRAME_COLOR)
                    continue
                j = cell_to_slot.get((rr, cc))
                if j is None or j == i:
                    continue
                other = slots[j]["family"]
                if other != family:
                    adjacency[family].add(other)
    return {"adjacency": adjacency, "frame_adj": frame_adj}


def _assign_family_colors(families: list, adjacency: dict, frame_adj: dict) -> dict | None:
    assignment: dict = {}

    def domain(family) -> list[int]:
        used = {assignment[n] for n in adjacency.get(family, set()) if n in assignment}
        out = []
        for col in _NONFRAME_COLORS:
            if col in frame_adj.get(family, set()):
                continue
            if col in used:
                continue
            out.append(int(col))
        random.shuffle(out)
        return out

    def choose_unassigned():
        pending = [family for family in families if family not in assignment]
        pending.sort(key=lambda family: (len(domain(family)), -len(adjacency.get(family, set()))))
        return pending[0] if pending else None

    def dfs() -> bool:
        family = choose_unassigned()
        if family is None:
            return len(set(assignment.values())) == len(_NONFRAME_COLORS)
        for col in domain(family):
            assignment[family] = col
            if dfs():
                return True
            assignment.pop(family)
        return False

    return assignment if dfs() else None


def _build_pre_and_out() -> tuple[np.ndarray | None, np.ndarray | None, int, set[int], dict]:
    frame_mask = _build_frame_mask()
    slots = _slotize(frame_mask)
    families = _family_groups(slots)
    if len(families) != _MIN_FAMILIES:
        return None, None, 0, set(), {}
    graph = _family_adjacency(slots, frame_mask)
    family_keys = sorted(families)
    colors_by_family = _assign_family_colors(family_keys, graph["adjacency"], graph["frame_adj"])
    if colors_by_family is None:
        return None, None, 0, set(), {}

    pre = np.full((_OUTPUT_H, _OUTPUT_W), _FRAME_COLOR, dtype=int)
    for family, groups in families.items():
        col = int(colors_by_family[family])
        for cells in groups:
            for r, c in cells:
                pre[r, c] = col

    out = _apply_swap(pre, pre == _FRAME_COLOR)
    bg_choices = [c for c in range(10) if c not in {_FRAME_COLOR, *_NONFRAME_COLORS}]
    bg = random.choice(bg_choices)
    meta = {
        "family_count": len(families),
        "reused_colors": len(families) - len(set(colors_by_family.values())),
        "changed_families": sum(1 for family in families if _SWAP_MAP[colors_by_family[family]] != colors_by_family[family]),
        "frame_mask": pre == _FRAME_COLOR,
        "families": families,
        "colors_by_family": colors_by_family,
        "pre": pre,
        "out": out,
    }
    return pre, out, bg, {_FRAME_COLOR}, meta


def _extract_non_frame_patches(pre: np.ndarray, bg: int, frame_colors: set[int]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for col in sorted(map(int, np.unique(pre))):
        if col in frame_colors:
            continue
        for pts in components_mask(pre == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(pre: np.ndarray, bg: int, frame_colors: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = pre.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame bbox does not fit canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_non_frame_patches(pre, bg, frame_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.isin(pre, list(frame_colors))
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = pre[frame_mask]
    return canvas


def _piece_count(grid: np.ndarray) -> int:
    return sum(len(components_mask(grid == col, connectivity=4)) for col in np.unique(grid))


def _gen_attempt() -> tuple[np.ndarray | None, dict]:
    pre, out, bg, frame_colors, meta = _build_pre_and_out()
    if pre is None or out is None:
        return None, {}
    if _piece_count(pre) < 9:
        return None, {}
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(pre)
        if int(col) not in frame_colors
        for pts in components_mask(pre == col, connectivity=4)
    ):
        return None, {}
    if meta.get("family_count", 0) < _MIN_FAMILIES:
        return None, {}
    if meta.get("reused_colors", 0) < 1:
        return None, {}
    if meta.get("changed_families", 0) != meta.get("family_count", -1):
        return None, {}
    try:
        inp = _scatter(pre, bg, frame_colors)
        got = main(inp)
    except Exception:
        return None, {}
    if np.array_equal(got, out):
        return inp, meta
    return None, {}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid, _meta = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
