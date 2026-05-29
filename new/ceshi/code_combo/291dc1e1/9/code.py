# Combo variant of 291dc1e1 (index 9)
# combo_spec: D3/0+D14/0
# Name: L/T形对象+噪声过滤

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for helper_dir in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    helper_str = str(helper_dir)
    if helper_str not in sys.path:
        sys.path.append(helper_str)

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for the combo variant D3/0 + D14/0

a) Concrete behavior and trace
- Original input: two edge lines define a corner-based coordinate system, and every
  interior connected component is a solid rectangle. Sorting uses bbox distance to
  the blue line, then bbox distance to the red line.
- Combo input: legal source objects are now non-rectangular L/T sprites. They still
  sit in two bands parallel to the blue edge, but each sprite is a sparse pattern
  inside its own bbox. The grid also contains detached irregular noise blobs.
- Example local trace on one component:
  original logic:
    1. Find a connected component in the interior mask.
    2. Crop its bbox.
    3. If the crop has no background cell, accept it as a rectangle.
  combo logic:
    1. Find the same connected component in the interior mask.
    2. Build the occupancy mask of that component inside its bbox.
    3. Accept it only if the occupancy mask matches one of the legal L/T templates.
    4. Otherwise reject it as noise.
  difference:
    - Legal objects are no longer bbox-filling, so the old rectangle test would
      reject every real object.
    - Noise filtering cannot rely on "bbox contains background" anymore, because
      both legal objects and noise are non-rectangular. The solver must classify
      by specific shape families.

b) Complexity calibration
- Legal objects use only a small family of repeated templates: L or T shapes with
  canonical bbox height 3 and widths in {3,5,7}, plus their rotations/reflections.
- This is visibly different from the original rectangles but still compact enough
  that humans can recognize the recurring legal shapes across 2-4 examples.
- Noise is limited to 1-3 detached blobs from a fixed small template bank. They are
  irregular and often visually similar to "almost legal" pieces, but they never
  equal a valid L/T template. This keeps the filtering step active without causing
  clutter or combinatorial explosion.

c) Implicit-assumption audit of the original main()
- `mask = (g != bg) & (~line_mask) & (g != marker)`:
  still valid for collecting candidate components, but it no longer implies those
  components are all task objects.
- `sub = crop_bbox(g, b)` then `(sub == bg).any()`:
  broken. Legal L/T objects necessarily contain background inside their bbox.
- Using the cropped array directly for stacking:
  still valid once the component has been classified as legal, because the output
  should preserve the object's exact sparse sprite inside its bbox.
- Sorting by bbox-to-edge distances:
  still valid, but now the bbox is an external wrapper around a sparse object. The
  solver must therefore compute distances from the legal object's bbox, not from
  a filled rectangle assumption.

d) Placement legality
- Generator builds legal L/T sprites first, places them in two non-overlapping
  horizontal bands of a canonical grid, then places noise only on empty cells.
- A 4-neighbor exclusion zone is enforced between every occupied cell of every
  object/noise blob, so no components merge.
- Noise never touches the marker cell or the edge lines and never leaves the grid.
- The final transformed input is validated against an independently constructed
  expected output before being returned.

e) Visual observability
- Humans can see that the kept objects are repeated L/T-like motifs while other
  jagged blobs disappear in the output.
- The two-band distance structure remains visible because legal objects are placed
  at two distinct distances from the blue line, just as in the source task.
- Rotation remains inferable because tall-input cases still produce output sprites
  turned 90 degrees CCW before stacking.

f) Visual cleanliness
- Legal objects all share a common scale family, so they look like one object type.
- Noise count is capped and placed in background corridors above, between, or below
  the legal bands. This avoids dense overlap and keeps component boundaries clear.
- Colors stay within the familiar non-background ARC palette and no special color is
  reserved for noise, so the rule is about structure rather than arbitrary color.

g) Color-change inferability
- There is no new color mapping rule. Each object/noise blob uses ordinary object
  colors; occasional minority pixels stay on occupied cells only.
- Because legality depends on occupancy shape rather than color, the palette can be
  reused safely without forcing the solver to learn a hidden color code.

h) Variant-activity metric used after implementation
- `kept_nonrect_components`: number of accepted legal objects whose occupancy does
  not fill their bbox. This should equal the object count and confirms D3 is active.
- `rejected_noise_components`: number of interior connected components rejected by
  the legal-shape classifier. This should be positive in most generated samples and
  confirms D14 is active.

i) New assumptions introduced by the updated solver
- Every legal object must be a 4-connected component whose occupancy mask matches a
  member of the finite L/T template family under rotation/reflection.
- Noise must remain detached from legal objects by at least one 4-neighbor gap.
- The generator guarantees those assumptions by construction and re-validates the
  final grid before returning it.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = DOMINANT_COLORS.copy()
LEGAL_WIDTHS = (3, 5, 7)
BASE_BBOX_HEIGHT = 3
NOISE_TEMPLATES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
]


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    m = np.array(mask, dtype=np.uint8)
    return int(m.shape[0]), int(m.shape[1]), m.tobytes()


def _all_symmetries(mask: np.ndarray) -> list[np.ndarray]:
    out: dict[tuple[int, int, bytes], np.ndarray] = {}
    for k in range(4):
        rot = np.rot90(mask, k=k)
        for variant in (rot, np.fliplr(rot)):
            out[_mask_key(variant)] = variant.copy()
    return list(out.values())


def _make_l_mask(width: int, orient: str) -> np.ndarray:
    mask = np.zeros((BASE_BBOX_HEIGHT, int(width)), dtype=bool)
    if orient == "ul":
        mask[:, 0] = True
        mask[0, :] = True
    elif orient == "ur":
        mask[:, -1] = True
        mask[0, :] = True
    elif orient == "dl":
        mask[:, 0] = True
        mask[-1, :] = True
    elif orient == "dr":
        mask[:, -1] = True
        mask[-1, :] = True
    else:
        raise ValueError(f"unknown L orientation: {orient}")
    return mask


def _make_t_mask(width: int, orient: str) -> np.ndarray:
    mask = np.zeros((BASE_BBOX_HEIGHT, int(width)), dtype=bool)
    mid = int(width // 2)
    if orient == "up":
        mask[0, :] = True
        mask[:, mid] = True
    elif orient == "down":
        mask[-1, :] = True
        mask[:, mid] = True
    elif orient == "left":
        mask[:, 0] = True
        mask[1, :] = True
    elif orient == "right":
        mask[:, -1] = True
        mask[1, :] = True
    else:
        raise ValueError(f"unknown T orientation: {orient}")
    return mask


def _build_legal_mask_bank() -> dict[tuple[int, int, bytes], np.ndarray]:
    bank: dict[tuple[int, int, bytes], np.ndarray] = {}
    for width in LEGAL_WIDTHS:
        for orient in ("ul", "ur", "dl", "dr"):
            for variant in _all_symmetries(_make_l_mask(width, orient)):
                bank[_mask_key(variant)] = variant.copy()
        for orient in ("up", "down", "left", "right"):
            for variant in _all_symmetries(_make_t_mask(width, orient)):
                bank[_mask_key(variant)] = variant.copy()
    return bank


LEGAL_MASK_BANK = _build_legal_mask_bank()
LEGAL_MASK_KEYS = set(LEGAL_MASK_BANK)


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    H, W = g.shape
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            return int(r), int(c), int(g[r, c])
    raise AssertionError("corner not found")


def _line_info(g: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
    H, W = g.shape
    cr, cc, marker = _locate_corner(g, bg)
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
    else:
        hlen = 0
        c = cc + hstep
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += hstep
        vlen = 0
        r = cr + vstep
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += vstep
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return int(cr), int(cc), int(marker), blue_orient, red_orient


def _classify_components(
    g: np.ndarray, bg: int, cr: int, cc: int, marker: int
) -> tuple[list[dict], dict[str, int]]:
    H, W = g.shape
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    objects: list[dict] = []
    metrics = {
        "candidate_components": 0,
        "accepted_components": 0,
        "rejected_noise_components": 0,
        "kept_nonrect_components": 0,
    }

    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        metrics["candidate_components"] += 1
        r0, c0, r1, c1 = bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        occ = np.zeros((h, w), dtype=bool)
        occ[coords[:, 0] - r0, coords[:, 1] - c0] = True

        if _mask_key(occ) not in LEGAL_MASK_KEYS:
            metrics["rejected_noise_components"] += 1
            continue

        arr = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
        objects.append({"box": (int(r0), int(c0), int(r1), int(c1)), "arr": arr.copy()})
        metrics["accepted_components"] += 1
        if not bool(occ.all()):
            metrics["kept_nonrect_components"] += 1

    if not objects:
        raise AssertionError("no legal L/T objects found")
    return objects, metrics


def _sort_objects(rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str) -> list[dict]:
    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            if c0 > edge_idx:
                return int(c0 - edge_idx - 1)
            if c1 < edge_idx:
                return int(edge_idx - c1 - 1)
            return 0
        if r0 > edge_idx:
            return int(r0 - edge_idx - 1)
        if r1 < edge_idx:
            return int(edge_idx - r1 - 1)
        return 0

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["db"] = int(dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
        )
    )
    return rects


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = max(int(arr.shape[1]) for arr in seq)
    out_h = sum(int(arr.shape[0]) for arr in seq)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, out_r, out_c)
        out_r += h
    return out


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _line_info(g, bg)
    rects, metrics = _classify_components(g, bg, cr, cc, marker)
    _sort_objects(rects, cr, cc, blue_orient, red_orient)
    metrics["sorted_objects"] = int(len(rects))
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _line_info(g, bg)
    rects, _metrics = _classify_components(g, bg, cr, cc, marker)
    _sort_objects(rects, cr, cc, blue_orient, red_orient)

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return _stack_sequence(seq, bg)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid.copy()
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _sample_band_widths(count: int) -> list[int]:
    for _ in range(200):
        pool = np.array([3, 5], dtype=int) if count >= 4 else np.array([3, 5, 7], dtype=int)
        probs = np.array([0.55, 0.35, 0.10], dtype=float)[: len(pool)]
        probs = probs / probs.sum()
        widths = [int(w) for w in np.random.choice(pool, size=int(count), replace=True, p=probs)]
        used = int(sum(widths) + 2 * (count - 1))
        if used <= 22:
            return widths
    raise RuntimeError("failed to sample legal band widths")


def _make_colored_object(mask: np.ndarray, dominant: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(dominant)

    coords = np.argwhere(mask)
    if coords.shape[0] >= 6 and np.random.rand() < 0.45:
        max_minor = min(3, int(coords.shape[0] // 3))
        if max_minor >= 1:
            n_minor = int(np.random.randint(1, max_minor + 1))
            idxs = np.random.choice(coords.shape[0], size=n_minor, replace=False)
            other_colors = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
            for idx in idxs:
                r, c = coords[int(idx)]
                patch[int(r), int(c)] = int(np.random.choice(other_colors))
    return patch


def _neighbor_block(mask: np.ndarray) -> np.ndarray:
    blocked = mask.copy()
    blocked[1:, :] |= mask[:-1, :]
    blocked[:-1, :] |= mask[1:, :]
    blocked[:, 1:] |= mask[:, :-1]
    blocked[:, :-1] |= mask[:, 1:]
    return blocked


def _paint_noise_patch(shape: np.ndarray) -> np.ndarray:
    patch = np.full(shape.shape, BG, dtype=int)
    palette = np.random.choice(DOMINANT_COLORS, size=int(np.random.randint(1, 3)), replace=False)
    coords = np.argwhere(shape)
    for idx, (r, c) in enumerate(coords):
        patch[int(r), int(c)] = int(palette[idx % len(palette)])
    return patch


def _add_noise(grid: np.ndarray, occupied: np.ndarray) -> int:
    H, W = grid.shape
    count = 0
    target = int(np.random.randint(1, 4))
    templates = list(np.random.permutation(len(NOISE_TEMPLATES)))

    for _ in range(target):
        blocked = _neighbor_block(occupied)
        candidates: list[tuple[int, int, np.ndarray]] = []
        for idx in templates:
            shape = NOISE_TEMPLATES[int(idx)]
            h, w = shape.shape
            for r in range(1, H - h):
                for c in range(1, W - w):
                    cells = np.argwhere(shape) + np.array([[r, c]])
                    if np.any(blocked[cells[:, 0], cells[:, 1]]):
                        continue
                    candidates.append((int(r), int(c), shape))
        if not candidates:
            break
        r, c, shape = candidates[int(np.random.randint(len(candidates)))]
        paste(grid, _paint_noise_patch(shape), r, c, transparent=BG)
        cells = np.argwhere(shape) + np.array([[r, c]])
        occupied[cells[:, 0], cells[:, 1]] = True
        count += 1
    return count


def _sample_counts_and_widths() -> tuple[list[int], list[int]]:
    for _ in range(200):
        n0 = int(np.random.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        n1 = int(np.random.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        if n0 + n1 > 7:
            continue
        band0 = _sample_band_widths(n0)
        band1 = _sample_band_widths(n1)
        if max(sum(band0) + 2 * (n0 - 1), sum(band1) + 2 * (n1 - 1)) <= 22:
            return band0, band1
    raise RuntimeError("failed to sample band counts")


def _make_object_mask(kind: str, width: int, orient: str) -> np.ndarray:
    if kind == "L":
        return _make_l_mask(width, orient)
    if kind == "T":
        return _make_t_mask(width, orient)
    raise ValueError(f"unknown object kind: {kind}")


def _build_canonical() -> tuple[np.ndarray, list[np.ndarray], dict[str, int]]:
    band0, band1 = _sample_counts_and_widths()
    used0 = int(sum(band0) + 2 * (len(band0) - 1))
    used1 = int(sum(band1) + 2 * (len(band1) - 1))
    long_dim = int(2 + max(used0, used1) + np.random.randint(2, 5))
    gap_rows = int(np.random.randint(2, 5))
    bottom_margin = int(np.random.randint(2, 5))
    top_r = 2
    bottom_r = int(top_r + BASE_BBOX_HEIGHT + gap_rows)
    height = int(bottom_r + BASE_BBOX_HEIGHT + bottom_margin)

    grid = np.full((height, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[0, :] = True
    occupied[:, 0] = True

    total_objects = len(band0) + len(band1)
    kinds = ["L", "T"]
    while len(kinds) < total_objects:
        kinds.append(str(np.random.choice(["L", "T"])))
    kinds = [str(k) for k in np.random.permutation(kinds)]

    legal_patches: list[np.ndarray] = []
    kind_idx = 0

    for r0, widths in ((top_r, band0), (bottom_r, band1)):
        c0 = 2
        for width in widths:
            kind = kinds[kind_idx]
            kind_idx += 1
            if kind == "L":
                orient = str(np.random.choice(["ul", "ur", "dl", "dr"]))
            else:
                orient = str(np.random.choice(["up", "down", "left", "right"]))
            mask = _make_object_mask(kind, int(width), orient)
            dominant = int(np.random.choice(DOMINANT_COLORS))
            patch = _make_colored_object(mask, dominant)
            paste(grid, patch, r0, c0, transparent=BG)
            legal_patches.append(patch.copy())
            occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= (patch != BG)
            c0 += int(width) + 2

    noise_count = _add_noise(grid, occupied)
    metrics = {
        "object_count": int(total_objects),
        "noise_count": int(noise_count),
        "kept_nonrect_components": int(total_objects),
    }
    return grid, legal_patches, metrics


def _expected_sequence(
    legal_patches: list[np.ndarray], blue_orient: str, corner: str
) -> list[np.ndarray]:
    seq = []
    for patch in legal_patches:
        arr = _transform(patch, blue_orient, corner)
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return seq


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    canonical, legal_patches, build_metrics = _build_canonical()
    if build_metrics["noise_count"] < 1:
        return None

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    expected = _stack_sequence(_expected_sequence(legal_patches, blue_orient, corner), BG)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    observed_metrics = variant_metrics(grid)
    if observed_metrics["kept_nonrect_components"] != observed_metrics["accepted_components"]:
        return None
    if observed_metrics["rejected_noise_components"] < 1:
        return None
    return grid, observed_metrics


def generate_input() -> np.ndarray:
    for _ in range(500):
        attempt = _gen_attempt()
        if attempt is not None:
            grid, _metrics = attempt
            return grid
    raise RuntimeError("failed to generate input for combo variant 291dc1e1/9")
