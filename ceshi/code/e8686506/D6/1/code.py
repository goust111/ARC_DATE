# Variant of e8686506
# Dimension: D6 — 变换：拼合后的颜色或形状变化 (transform)
# Change: 无变换——碎片保持原色原形直接拼合... → 碎片间颜色交换

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D6 variant
# a) Concrete behavior and flow impact:
#    The original task assembles a left-right symmetric dense rectangle and leaves every
#    fragment's color unchanged. Here the scattered pieces still encode a pre-swap
#    assembly, but after reconstruction every non-frame piece takes the color of its
#    left-right mirror mate. Concretely, a left-facing L piece and its right-facing
#    mirrored L piece keep their shapes and positions, then exchange colors in the final
#    output. The flow change happens after spatial assembly: input -> reconstruct the
#    pre-swap board -> recolor each mirror-signature family.
# b) Complexity calibration:
#    If colors were assigned independently per copy, repeated singleton/line shapes would
#    create arbitrary matchings. To keep the variant visible but still learnable, the
#    generator assigns one pre-swap color per oriented shape signature, and a different
#    color to the mirrored signature when possible. This creates repeated evidence for the
#    swap rule without exploding the number of cases.
# c) Hidden assumptions audit:
#    The original solver relied on mirrored cells having the same color (`mirror_ok`) and
#    on color-specific edge markers constraining placement. Those assumptions break once
#    mirror partners may differ in color. The new solver therefore does not place pieces
#    by color symmetry. Instead it detects the 1-2 frame parts exactly as before, derives
#    the interior slot geometry from the frame holes, fills slots by oriented shape
#    signature, and only then applies the mirror-pair color swap.
# d) Placement legality:
#    The generator still preserves the frame group inside its bounding box and scatters all
#    other pieces outside that box with padding, so nothing overlaps. Non-frame pieces are
#    derived from deterministic <=6-cell mirror-aware chunks of frame holes; this guarantees
#    the solver can reconstruct the same slots from the detected frame.
# e) Visual observability:
#    The variant is visible because several asymmetric mirror-signature families are forced
#    to use two different colors. Across examples, users can see that the geometry of the
#    assembled rectangle stays the same while opposite-side pieces exchange their colors.
# f) Visual cleanliness:
#    Signature groups are colored under adjacency constraints, so neither the pre-swap nor
#    the swapped board merges adjacent pieces or frame cells into visually confusing blobs.
# g) Color-change inferability:
#    Colors are not chosen from invisible ids or sorted numerically. The rule depends on
#    the visible left-right pairing of piece shapes. Repeated occurrences of the same
#    oriented signature use the same input color, which gives the viewer enough structure
#    to infer "mirror mate donates the output color".
# New solver assumptions checked against the generator:
#    - Total foreground area still has a unique closest-to-square portrait factorization.
#    - The input still contains 1-2 8-connected frame parts whose union bbox is the output.
#    - Non-frame pieces are exact mirror-aware chunks of frame holes, area <= 6, no rotation.
#    - Every oriented non-frame signature has a unique input color across all of its copies.


# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_MIN_SWAP_FAMILIES = 2

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 400


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _mirror_sig(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    h, w, coords = sig
    return h, w, tuple(sorted((int(r), w - 1 - int(c)) for r, c in coords))


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _valid_dims() -> list[tuple[int, int]]:
    out = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h in (8, 10):
                out.append((h, w))
    return out


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _chunk_component(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    chunks: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        queue = [start]
        queued = {start}
        chunk: list[tuple[int, int]] = []
        while queue and len(chunk) < _MAX_PIECE_AREA:
            r, c = queue.pop(0)
            if (r, c) not in remaining:
                continue
            remaining.remove((r, c))
            chunk.append((r, c))
            nbrs = []
            for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
                if (rr, cc) in remaining and (rr, cc) not in queued:
                    nbrs.append((rr, cc))
                    queued.add((rr, cc))
            queue.extend(sorted(nbrs))
        chunks.append(sorted(chunk))
    return chunks


def _slotize(frame_mask: np.ndarray) -> list[dict]:
    H, W = frame_mask.shape
    mid = W // 2
    open_mask = ~frame_mask
    slots: list[dict] = []

    left_open = open_mask[:, :mid]
    for comp in components_mask(left_open, connectivity=4):
        left_cells = [(int(r), int(c)) for r, c in comp]
        for chunk in _chunk_component(left_cells):
            arr = np.array(chunk, dtype=int)
            slots.append({"cells": chunk, "sig": _piece_sig(arr)})
            mirror = sorted((int(r), W - 1 - int(c)) for r, c in chunk)
            marr = np.array(mirror, dtype=int)
            slots.append({"cells": mirror, "sig": _piece_sig(marr)})

    center_mask = open_mask[:, mid : mid + 1]
    for comp in components_mask(center_mask, connectivity=4):
        center_cells = sorted((int(r), mid) for r, _ in comp)
        for i in range(0, len(center_cells), _MAX_PIECE_AREA):
            chunk = center_cells[i : i + _MAX_PIECE_AREA]
            arr = np.array(chunk, dtype=int)
            slots.append({"cells": chunk, "sig": _piece_sig(arr)})
    return slots


def _slot_groups(slots: list[dict]) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]]:
    out: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for slot in slots:
        out.setdefault(slot["sig"], []).append(slot["cells"])
    return out


def _swap_board(pre: np.ndarray, slot_groups: dict) -> np.ndarray:
    out = pre.copy()
    sig_color: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for sig, groups in slot_groups.items():
        r, c = groups[0][0]
        sig_color[sig] = int(pre[r, c])
    for sig, groups in slot_groups.items():
        msig = _mirror_sig(sig)
        assert msig in slot_groups, "every slot signature must have a mirror partner"
        target = sig_color[msig]
        for cells in groups:
            for r, c in cells:
                out[r, c] = target
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
    assert frame_parts, "input must contain 1-2 LR-symmetric frame parts"
    return frame_parts


def _build_pre_from_input(g: np.ndarray, bg: int) -> tuple[np.ndarray, dict]:
    total_area = int((g != int(bg)).sum())
    assert total_area > 0, "grid must contain non-background pieces"
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

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
    slot_groups = _slot_groups(slots)

    input_groups: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[int]] = {}
    seen = 0
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == col, connectivity=4):
            if all(frame_input_mask[int(r), int(c)] for r, c in pts):
                continue
            sig = _piece_sig(pts)
            input_groups.setdefault(sig, []).append(int(col))
            seen += 1

    assert seen == len(slots), "non-frame piece count must match reconstructed slots"

    sig_color: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for sig, groups in slot_groups.items():
        cols = input_groups.get(sig, [])
        assert len(cols) == len(groups), "slot multiplicity must match input multiplicity"
        uniq = set(cols)
        assert len(uniq) == 1, "each oriented signature must have one input color"
        sig_color[sig] = int(next(iter(uniq)))
        for cells in groups:
            for r, c in cells:
                pre[r, c] = sig_color[sig]

    assert np.all(pre >= 0), "reconstructed pre-swap board must be dense"
    meta = {"slot_groups": slot_groups, "frame_mask": frame_mask, "dims": (H, W)}
    return pre, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a mirror-structured rectangle from scattered fragments, then exchange mirror-pair colors.

    Input structure:
    - The background is the most frequent color.
    - One or two 8-connected frame fragments remain grouped in their output-relative positions.
    - All other single-color pieces are scattered separately and keep their original orientation.

    Rule:
    - Infer the rectangle size from total fragment area and locate the grouped frame.
    - Use the frame holes to recover the deterministic non-frame slots, then refill each slot by matching the scattered piece shape signature.
    - After the pre-swap rectangle is reconstructed, recolor every non-frame slot with the color of its left-right mirrored partner.

    Key insight:
    - The frame fixes the geometry, oriented shape signatures recover the pre-swap placement, and the final transform is a mirror-pair color exchange rather than a second spatial rearrangement.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pre, meta = _build_pre_from_input(g, bg)
    out = _swap_board(pre, meta["slot_groups"])
    return out.astype(int)


def _frame_template(H: int, W: int) -> tuple[np.ndarray, set[int], list[int]] | tuple[None, set[int], list[int]]:
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    frame1 = palette[0]
    frame2 = palette[1] if use_two_frames else frame1
    fill_pool = [c for c in palette if c not in {frame1, frame2}]
    if len(fill_pool) < 2:
        return None, set(), []

    # Build symmetric open slots first; the frame is their complement.
    left_catalog = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
    ]
    left_rows = [0, 3, 6]
    center_rows = [2, 5] if H == 8 else [2, 5, 8]
    chosen = random.sample(left_catalog, k=3)

    open_mask = np.zeros((H, W), dtype=bool)
    for r0, shape in zip(left_rows, chosen):
        for dr, dc in shape:
            r, c = r0 + dr, dc
            open_mask[r, c] = True
            open_mask[r, W - 1 - c] = True
    for r in center_rows:
        open_mask[r, W // 2] = True

    # Keep frame connected while allowing a top/bottom color split.
    split = H // 2 if frame1 != frame2 else H
    grid = np.full((H, W), -1, dtype=int)
    for r in range(H):
        for c in range(W):
            if open_mask[r, c]:
                continue
            grid[r, c] = frame1 if r < split else frame2

    frame_colors = {frame1}
    if frame2 != frame1:
        frame_colors.add(frame2)
    return grid, frame_colors, fill_pool


def _signature_graph(
    frame_grid: np.ndarray,
    slots: list[dict],
) -> tuple[dict, dict, bool]:
    H, W = frame_grid.shape
    adjacency: dict[tuple[int, int, tuple[tuple[int, int], ...]], set] = {}
    frame_adj: dict[tuple[int, int, tuple[tuple[int, int], ...]], set[int]] = {}
    cell_to_slot: dict[tuple[int, int], int] = {}
    for i, slot in enumerate(slots):
        sig = slot["sig"]
        adjacency.setdefault(sig, set())
        frame_adj.setdefault(sig, set())
        for cell in slot["cells"]:
            cell_to_slot[cell] = i

    for i, slot in enumerate(slots):
        sig = slot["sig"]
        for r, c in slot["cells"]:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                v = int(frame_grid[rr, cc])
                if v != -1:
                    frame_adj[sig].add(v)
                    continue
                j = cell_to_slot.get((rr, cc))
                if j is None or j == i:
                    continue
                other_sig = slots[j]["sig"]
                if other_sig == sig:
                    return {}, {}, False
                adjacency[sig].add(other_sig)
    return adjacency, frame_adj, True


def _assign_signature_colors(
    slot_groups: dict,
    adjacency: dict,
    frame_adj: dict,
    fill_pool: list[int],
) -> dict | None:
    signatures = list(slot_groups.keys())
    for sig in signatures:
        msig = _mirror_sig(sig)
        if msig not in slot_groups:
            return None
        if len(slot_groups[sig]) != len(slot_groups[msig]):
            return None

    seen: set = set()
    asym_families = []
    for sig in signatures:
        msig = _mirror_sig(sig)
        if sig == msig or sig in seen or msig in seen:
            continue
        seen.add(sig)
        seen.add(msig)
        asym_families.append((sig, msig))
    if len(asym_families) < _MIN_SWAP_FAMILIES:
        return None

    degrees = {sig: len(adjacency.get(sig, set())) + len(frame_adj.get(sig, set())) for sig in signatures}
    assignment: dict = {}

    def domain(sig) -> list[int]:
        used = {assignment[n] for n in adjacency.get(sig, set()) if n in assignment}
        out = []
        for col in fill_pool:
            if col in frame_adj.get(sig, set()):
                continue
            if col in used:
                continue
            msig = _mirror_sig(sig)
            if msig != sig and msig in assignment and assignment[msig] == col:
                continue
            out.append(col)
        random.shuffle(out)
        return out

    def choose_unassigned():
        cands = [sig for sig in signatures if sig not in assignment]
        cands.sort(key=lambda sig: (len(domain(sig)), -degrees[sig]))
        return cands[0] if cands else None

    def dfs() -> bool:
        sig = choose_unassigned()
        if sig is None:
            return True
        for col in domain(sig):
            assignment[sig] = int(col)
            if dfs():
                return True
            assignment.pop(sig)
        return False

    if not dfs():
        return None
    return assignment


def _build_pre_and_out() -> tuple[np.ndarray | None, np.ndarray | None, int, set[int], dict]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    frame_grid, frame_colors, fill_pool = _frame_template(H, W)
    if frame_grid is None:
        return None, None, 0, set(), {}

    frame_mask = frame_grid != -1
    slots = _slotize(frame_mask)
    slot_groups = _slot_groups(slots)
    adjacency, frame_adj, ok = _signature_graph(frame_grid, slots)
    if not ok:
        return None, None, 0, set(), {}

    colors_by_sig = _assign_signature_colors(slot_groups, adjacency, frame_adj, fill_pool)
    if colors_by_sig is None:
        return None, None, 0, set(), {}

    pre_abs = frame_grid.copy()
    for sig, groups in slot_groups.items():
        col = int(colors_by_sig[sig])
        for cells in groups:
            for r, c in cells:
                pre_abs[r, c] = col
    assert np.all(pre_abs >= 0)
    out_abs = _swap_board(pre_abs, slot_groups)

    bg = random.randint(0, 9)
    abstract_colors = sorted(int(c) for c in np.unique(pre_abs))
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    if len(avail) < len(abstract_colors):
        return None, None, 0, set(), {}
    cmap = {ac: avail[i] for i, ac in enumerate(abstract_colors)}
    pre = np.vectorize(lambda x: cmap[int(x)])(pre_abs).astype(int)
    out = np.vectorize(lambda x: cmap[int(x)])(out_abs).astype(int)
    real_frame_colors = {cmap[int(c)] for c in frame_colors}
    meta = {
        "slot_groups_abs": slot_groups,
        "colors_by_sig_abs": colors_by_sig,
        "swap_families": sum(
            1
            for sig in slot_groups
            if sig < _mirror_sig(sig) and colors_by_sig[sig] != colors_by_sig[_mirror_sig(sig)]
        ),
    }
    return pre, out, bg, real_frame_colors, meta


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
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None, {}
    pc = _piece_count(pre)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None, {}
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(pre) if col not in frame_colors for pts in components_mask(pre == col, connectivity=4)):
        return None, {}
    if meta.get("swap_families", 0) < _MIN_SWAP_FAMILIES:
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
    raise RuntimeError("failed to generate a valid e8686506 D6 input")
