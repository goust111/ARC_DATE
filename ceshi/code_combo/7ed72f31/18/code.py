# Combo variant of 7ed72f31 (index 18)
# combo_spec: D6/0+D13/0
# Name: 修剪不对称+紧凑输出

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import get_bg, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for combo D6/0 + D13/0
#
# a) Concrete behavior and trace
#    Original 7ed72f31:
#      a red row/col/point tells the solver to ADD a missing mirror copy onto the
#      same large canvas.
#    Combo behavior:
#      the payload is already almost symmetric around the red marker; a few cells
#      exist only on one side. We DELETE those unmatched cells so the sprite becomes
#      perfectly symmetric, then crop the whole repaired canvas to the minimal bbox
#      of all non-background pixels.
#    Small trace:
#      input full grid contains a horizontal red row with blue cells on both sides,
#      plus one extra blue cell only above the row. Original solver would keep that
#      extra cell and paint another blue cell below it. Combo solver instead removes
#      the unmatched upper cell, keeps the already matched pairs, then removes the
#      outer background border from the final full canvas.
#
# b) Complexity calibration
#    We keep exactly the three readable marker semantics backed by the original spec:
#      - horizontal red line
#      - vertical red line
#      - isolated red point with one adjacent 8-connected sprite
#    Each sample contains at least one of each, with 3-5 groups total and unique
#    payload colors per group. Every group has 1-2 asymmetric defects, so the
#    deletion rule repeats enough times to be visually inferable without introducing
#    multi-color ambiguity inside a single symmetry group.
#
# c) Implicit-assumption audit of the original solver
#    - Original assumption: reflected cells should be PAINTED onto background.
#      Broken by D13. We replace paint logic with "keep only pixels whose reflected
#      counterpart already exists with the same color".
#    - Original assumption: scanning all grid pixels within a bbox is safe.
#      Broken by D13 because deletion must not touch unrelated objects. We therefore
#      only inspect pixels in the marker's own 4-connected component, or the single
#      8-connected component adjacent to an isolated center.
#    - Original assumption: returning the full canvas is acceptable.
#      Broken by D6. After repair we must crop using the input/full-canvas background.
#    - New interaction between D13 and D6:
#      cropping must happen AFTER deletion, otherwise unmatched border-side defects
#      could incorrectly influence the reported output size.
#
# d) Placement legality
#    Generator constructs each group as a local perfect-symmetry output patch, then
#    injects 1-2 one-sided defects into the input patch. Patches are placed with
#    border=2 and padding=1, preventing overlap, edge clipping, accidental touching,
#    and keeping the crop visibly smaller than the input.
#
# e) Visual observability
#    Across 2-4 ARC examples, a human sees two repeated cues:
#      - input sprites are almost symmetric around red and output deletes the odd
#        one-sided pixels instead of adding anything;
#      - the output frame is tightly cropped around all active pixels.
#    Both cues are directly visible; neither depends on hidden ordering or color ids.
#
# f) Visual cleanliness
#    No free distractor sprites are added. Each group uses one payload color, sits in
#    its own padded region, and has compact geometry around the red marker. This
#    keeps the reason for every deletion legible.
#
# g) Color reasoning
#    Colors are preserved. They only separate groups; no color remapping occurs.
#    Unique colors per group prevent any ambiguity about which reflected counterpart
#    belongs to which marker.
#
# h) Review of new solver assumptions and generator guarantees
#    - Mixed row/col groups assume all payload pixels to repair lie in the same
#      4-connected component as the red line. Generator guarantees this by building
#      symmetric halves adjacent to the red axis.
#    - Isolated-point groups assume the red center has no 4-neighbor payload cells
#      and exactly one adjacent 8-connected payload component. Generator enforces
#      that by using point-symmetric templates that avoid orthogonal neighbors of the
#      red center and by only adding defects that stay in that same component.
#    - Tight crop assumes at least one non-background pixel survives. Generator uses
#      3+ groups with defects added on top of non-empty symmetric outputs, so repair
#      never empties the canvas.
# ---------------------------------------------------------------------------

MARK = 2
LAST_META: dict[str, object] = {}


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - int(r)), int(2 * pc - int(c))


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _collect_fg_components(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    lab4, n4 = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n4) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})
    return comps


def _axis_from_reds(reds: list[tuple[int, int]]) -> tuple[str, object] | None:
    if not reds:
        return None
    if len(reds) == 1:
        return "point", (int(reds[0][0]), int(reds[0][1]))
    rs = {int(r) for r, _ in reds}
    cs = {int(c) for _, c in reds}
    if len(rs) == 1:
        return "row", int(next(iter(rs)))
    if len(cs) == 1:
        return "col", int(next(iter(cs)))
    return None


def _repair_pixels(
    g: np.ndarray,
    out: np.ndarray,
    pixels: list[tuple[int, int]],
    reflect_fn,
    bg: int,
) -> None:
    keep: set[tuple[int, int]] = set()
    for r, c in pixels:
        v = int(g[r, c])
        rr, cc = reflect_fn(int(r), int(c))
        if _inb(g.shape, rr, cc) and int(g[rr, cc]) == int(v):
            keep.add((int(r), int(c)))
    for r, c in pixels:
        if (int(r), int(c)) not in keep:
            out[int(r), int(c)] = int(bg)


def _reflect_pixels_additively(
    g: np.ndarray,
    out: np.ndarray,
    pixels: list[tuple[int, int]],
    reflect_fn,
) -> None:
    for r, c in pixels:
        rr, cc = reflect_fn(int(r), int(c))
        if _inb(g.shape, rr, cc):
            out[int(rr), int(cc)] = int(g[r, c])


def _solve_repaired_full(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    comps = _collect_fg_components(g, bg)

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        if len(coords) == 1 and len(reds) == 1:
            continue

        payload = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not payload:
            continue
        axis = _axis_from_reds(reds)
        if axis is None:
            continue

        kind, ref = axis
        if kind == "row":
            _repair_pixels(g, out, payload, lambda r, c, rr=int(ref): _refl_row(rr, r, c), bg)
        elif kind == "col":
            _repair_pixels(g, out, payload, lambda r, c, cc=int(ref): _refl_col(cc, r, c), bg)
        else:
            center = (int(ref[0]), int(ref[1]))
            _repair_pixels(g, out, payload, lambda r, c, p=center: _refl_point(p, r, c), bg)

    payload_mask = (g != int(bg)) & (g != int(MARK))
    lab8, _ = label_mask(payload_mask, connectivity=8)
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) != 1 or int(vals[0]) != int(MARK):
            continue
        pr, pc = map(int, coords[0])
        seen_labels: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(g.shape, rr, cc):
                    continue
                lbl = int(lab8[rr, cc])
                if lbl != 0:
                    seen_labels.add(lbl)
        for lbl in seen_labels:
            pixels = [(int(r), int(c)) for r, c in np.argwhere(lab8 == int(lbl))]
            _repair_pixels(g, out, pixels, lambda r, c, p=(pr, pc): _refl_point(p, r, c), bg)

    return out


def _solve_additive_full(grid: np.ndarray, bg: int) -> np.ndarray:
    """
    Lightweight reference for the original "add the mirror copy" direction.

    This only reflects pixels belonging to the marker-associated component(s). That
    matches our generator's clean samples and is sufficient to detect degeneration:
    if combo repair behaved like additive mirroring, the two full-size outputs would
    coincide on generated inputs.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    comps = _collect_fg_components(g, bg)

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        if len(coords) == 1 and len(reds) == 1:
            continue

        payload = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not payload:
            continue
        axis = _axis_from_reds(reds)
        if axis is None:
            continue

        kind, ref = axis
        if kind == "row":
            _reflect_pixels_additively(g, out, payload, lambda r, c, rr=int(ref): _refl_row(rr, r, c))
        elif kind == "col":
            _reflect_pixels_additively(g, out, payload, lambda r, c, cc=int(ref): _refl_col(cc, r, c))
        else:
            center = (int(ref[0]), int(ref[1]))
            _reflect_pixels_additively(g, out, payload, lambda r, c, p=center: _refl_point(p, r, c))

    payload_mask = (g != int(bg)) & (g != int(MARK))
    lab8, _ = label_mask(payload_mask, connectivity=8)
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) != 1 or int(vals[0]) != int(MARK):
            continue
        pr, pc = map(int, coords[0])
        seen_labels: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(g.shape, rr, cc):
                    continue
                lbl = int(lab8[rr, cc])
                if lbl != 0:
                    seen_labels.add(lbl)
        for lbl in seen_labels:
            pixels = [(int(r), int(c)) for r, c in np.argwhere(lab8 == int(lbl))]
            _reflect_pixels_additively(g, out, pixels, lambda r, c, p=(pr, pc): _refl_point(p, r, c))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete unmatched near-symmetry defects around red markers, then tight-crop.

    Marker semantics are unchanged from the source task:
    - horizontal red line => reflection axis across that row
    - vertical red line   => reflection axis across that column
    - isolated red pixel  => point symmetry around that center
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    repaired = _solve_repaired_full(g, bg)
    return _tight_crop_non_bg(repaired, bg)


def _connected_shape(rng: np.random.Generator, h: int, w: int, count: int) -> set[tuple[int, int]]:
    for _ in range(100):
        cells = {(int(h - 1), int(rng.integers(w)))}
        while len(cells) < int(count):
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < int(h) and 0 <= cc < int(w) and (rr, cc) not in cells:
                        frontier.append((int(rr), int(cc)))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)
        if len(cells) == int(count):
            return cells
    raise ValueError("connected_shape_failed")


def _choice_subset(
    items: list[tuple[int, int]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    idxs = rng.choice(len(items), size=int(n), replace=False)
    return [items[int(i)] for i in np.atleast_1d(idxs)]


def _build_row_patch(
    rng: np.random.Generator,
    bg: int,
    color: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    for _ in range(120):
        half_h = int(rng.integers(2, 5))
        width = int(rng.integers(3, 7))
        area = int(half_h * width)
        max_base = max(3, min(6, area - 2))
        base_n = int(rng.integers(3, max_base + 1))

        top = _connected_shape(rng, half_h, width, base_n)
        bot = {_refl_row(half_h, r, c) for r, c in top}

        out = np.full((2 * half_h + 1, width), int(bg), dtype=int)
        out[half_h, :] = int(MARK)
        for r, c in top | bot:
            out[int(r), int(c)] = int(color)

        cand: list[tuple[int, int]] = []
        for r in range(half_h):
            for c in range(width):
                if (int(r), int(c)) in top:
                    continue
                rr, cc = _refl_row(half_h, int(r), int(c))
                if int(out[rr, cc]) != int(bg):
                    continue
                if any((int(r + dr), int(c + dc)) in top for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    cand.append((int(r), int(c)))
        if not cand:
            continue

        inp = out.copy()
        extra_n = int(rng.integers(1, min(2, len(cand)) + 1))
        for r, c in _choice_subset(cand, extra_n, rng):
            inp[int(r), int(c)] = int(color)

        if bool(rng.integers(0, 2)):
            inp = np.flipud(inp).copy()
            out = np.flipud(out).copy()

        return inp, out, {"type": "row", "extras": int(extra_n)}

    raise ValueError("row_patch_failed")


def _build_col_patch(
    rng: np.random.Generator,
    bg: int,
    color: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    inp_r, out_r, meta = _build_row_patch(rng, bg, color)
    return inp_r.T.copy(), out_r.T.copy(), {"type": "col", "extras": int(meta["extras"])}


def _transform_offsets(
    offsets: set[tuple[int, int]],
    rot: int,
    flip: bool,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in offsets:
        rr, cc = int(r), int(c)
        if bool(flip):
            cc = -cc
        for _ in range(int(rot) % 4):
            rr, cc = -cc, rr
        out.add((int(rr), int(cc)))
    return out


def _build_isolated_patch(
    rng: np.random.Generator,
    bg: int,
    color: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    templates = [
        {(-1, -1), (-2, 0), (-1, 1), (1, 1), (2, 0), (1, -1)},
        {(-1, -1), (-2, 0), (-1, 1), (0, 2), (1, 1), (2, 0), (1, -1), (0, -2)},
        {(-1, -1), (-2, -1), (-2, 0), (-2, 1), (-1, 1), (1, 1), (2, 1), (2, 0), (2, -1), (1, -1)},
    ]
    forbidden = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}

    for _ in range(120):
        base = set(templates[int(rng.integers(len(templates)))])
        base = _transform_offsets(base, int(rng.integers(4)), bool(rng.integers(0, 2)))
        if any((int(r), int(c)) in forbidden for r, c in base):
            continue

        cand: list[tuple[int, int]] = []
        for r in range(-3, 4):
            for c in range(-3, 4):
                pos = (int(r), int(c))
                if pos in base or pos in forbidden:
                    continue
                if not (int(r) < 0 or (int(r) == 0 and int(c) < 0)):
                    continue
                if (-int(r), -int(c)) in base:
                    continue
                if any(max(abs(int(r - rr)), abs(int(c - cc))) == 1 for rr, cc in base):
                    cand.append(pos)
        if not cand:
            continue

        out = np.full((9, 9), int(bg), dtype=int)
        center = 4
        out[center, center] = int(MARK)
        for r, c in base:
            out[center + int(r), center + int(c)] = int(color)

        inp = out.copy()
        extra_n = int(rng.integers(1, min(2, len(cand)) + 1))
        for r, c in _choice_subset(cand, extra_n, rng):
            inp[center + int(r), center + int(c)] = int(color)

        if np.array_equal(inp, out):
            continue
        return inp, out, {"type": "isolated", "extras": int(extra_n)}

    raise ValueError("isolated_patch_failed")


def _generate_group(
    group_type: str,
    color: int,
    bg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if group_type == "row":
        return _build_row_patch(rng, bg, color)
    if group_type == "col":
        return _build_col_patch(rng, bg, color)
    if group_type == "isolated":
        return _build_isolated_patch(rng, bg, color)
    raise ValueError(f"unknown group_type: {group_type}")


def _bbox_margin(full_out: np.ndarray, bg: int) -> int:
    fg = np.argwhere(full_out != int(bg))
    if fg.size == 0:
        return 0
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    h, w = full_out.shape
    return int(min(r0, c0, h - 1 - r1, w - 1 - c1))


def generate_input() -> np.ndarray:
    for _attempt in range(240):
        rng = np.random.default_rng()
        h = int(rng.integers(22, 29))
        w = int(rng.integers(22, 29))
        palette = [int(x) for x in rng.permutation(np.arange(10, dtype=int))]
        bg = next(c for c in palette if c != int(MARK))
        colors = [c for c in palette if c not in {int(bg), int(MARK)}]

        group_types = ["row", "col", "isolated"]
        extra_groups = int(rng.integers(0, 3))
        for _ in range(extra_groups):
            group_types.append(["row", "col", "isolated"][int(rng.integers(3))])
        rng.shuffle(group_types)

        if len(colors) < len(group_types):
            continue

        grid_in = np.full((h, w), int(bg), dtype=int)
        full_out = grid_in.copy()
        metas: list[dict[str, int]] = []
        ok = True
        for idx, group_type in enumerate(group_types):
            color = int(colors[idx])
            try:
                patch_in, patch_out, meta = _generate_group(group_type, color, bg, rng)
                loc = random_free_location(full_out, patch_out, transparent=bg, border=2, padding=1, padding_connectivity=8)
            except Exception:
                ok = False
                break
            paste(grid_in, patch_in, int(loc[0]), int(loc[1]), transparent=bg)
            paste(full_out, patch_out, int(loc[0]), int(loc[1]), transparent=bg)
            metas.append(meta)
        if not ok:
            continue

        expected = _tight_crop_non_bg(full_out, bg)
        got = main(grid_in)
        if not np.array_equal(got, expected):
            continue
        if got.shape == grid_in.shape:
            continue

        repaired_full = _solve_repaired_full(grid_in, bg)
        if not np.array_equal(repaired_full, full_out):
            continue
        additive_full = _solve_additive_full(grid_in, bg)
        repair_delta = int(np.count_nonzero(repaired_full != additive_full))
        deleted_pixels = int(np.count_nonzero(grid_in != full_out))
        if deleted_pixels <= 0 or repair_delta <= 0:
            continue
        if _bbox_margin(full_out, bg) < 2:
            continue

        global LAST_META
        LAST_META = {
            "input_shape": (int(h), int(w)),
            "output_shape": tuple(map(int, expected.shape)),
            "n_groups": int(len(group_types)),
            "groups": metas,
            "deleted_pixels": int(deleted_pixels),
            "repair_delta": int(repair_delta),
            "crop_margin": int(_bbox_margin(full_out, bg)),
        }
        return grid_in

    raise RuntimeError("failed to generate a valid combo instance")
