import numpy as np


_RNG = np.random.default_rng()

BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
LABEL_STRIP = 6
REGION_GAP = 3
GLOBAL_MARGIN = 1
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200
SHAPE_CORE_RANGE = (4, 8)
LABEL_CORE = 2
LAYOUTS = ("labels_right", "labels_left", "labels_bottom", "labels_top")


def _slot_offsets(length: int) -> tuple[int, int]:
    if length <= 1:
        raise ValueError("core side length must be at least 2")
    if length == 2:
        return 0, 1
    return 1, length - 2


def _signature_catalog() -> list[tuple[int, ...]]:
    sigs = []
    for mask in range(1, 1 << 8):
        bits = tuple((mask >> i) & 1 for i in range(8))
        pop = sum(bits)
        if not (2 <= pop <= 5):
            continue
        side_counts = (
            bits[0] + bits[1],
            bits[2] + bits[3],
            bits[4] + bits[5],
            bits[6] + bits[7],
        )
        if sum(count > 0 for count in side_counts) < 2:
            continue
        sigs.append(bits)
    return sigs


SIGNATURES = _signature_catalog()


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in np.argwhere(grid == color):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and grid[rr, cc] == color:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
            }
        )
    return comps


def _recover_core_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    best = None
    best_area = -1
    for top in (0, 1):
        for bottom in (0, 1):
            for left in (0, 1):
                for right in (0, 1):
                    r0 = top
                    c0 = left
                    r1 = h - 1 - bottom
                    c1 = w - 1 - right
                    if r0 > r1 or c0 > c1:
                        continue
                    core = mask[r0 : r1 + 1, c0 : c1 + 1]
                    if not core.all():
                        continue
                    ok = True
                    for r, c in np.argwhere(mask):
                        r = int(r)
                        c = int(c)
                        if r0 <= r <= r1 and c0 <= c <= c1:
                            continue
                        inside_neighbors = 0
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            rr = r + dr
                            cc = c + dc
                            if r0 <= rr <= r1 and c0 <= cc <= c1:
                                inside_neighbors += 1
                        if inside_neighbors != 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    area = int(core.size)
                    if area > best_area:
                        best_area = area
                        best = (r0, c0, r1, c1)
    assert best is not None, "failed to recover a rectangular core"
    return best


def _extract_signature(mask: np.ndarray) -> tuple[int, ...]:
    r0, c0, r1, c1 = _recover_core_bbox(mask)
    core_h = r1 - r0 + 1
    core_w = c1 - c0 + 1
    w0, w1 = _slot_offsets(core_w)
    h0, h1 = _slot_offsets(core_h)
    slots = [
        (r0 - 1, c0 + w0),
        (r0 - 1, c0 + w1),
        (r0 + h0, c1 + 1),
        (r0 + h1, c1 + 1),
        (r1 + 1, c0 + w1),
        (r1 + 1, c0 + w0),
        (r0 + h1, c0 - 1),
        (r0 + h0, c0 - 1),
    ]
    bits = [0] * 8
    candidate_cells = np.zeros_like(mask, dtype=bool)
    for i, (r, c) in enumerate(slots):
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            candidate_cells[r, c] = True
            bits[i] = int(mask[r, c])
    outside = mask.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    assert not np.any(outside & ~candidate_cells), "unexpected protrusion layout"
    return tuple(bits)


def _build_object_mask(core_h: int, core_w: int, signature: tuple[int, ...]) -> np.ndarray:
    top = int(signature[0] or signature[1])
    right = int(signature[2] or signature[3])
    bottom = int(signature[4] or signature[5])
    left = int(signature[6] or signature[7])
    h = core_h + top + bottom
    w = core_w + left + right
    mask = np.zeros((h, w), dtype=bool)
    r0 = top
    c0 = left
    r1 = r0 + core_h - 1
    c1 = c0 + core_w - 1
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    w0, w1 = _slot_offsets(core_w)
    h0, h1 = _slot_offsets(core_h)
    if signature[0]:
        mask[r0 - 1, c0 + w0] = True
    if signature[1]:
        mask[r0 - 1, c0 + w1] = True
    if signature[2]:
        mask[r0 + h0, c1 + 1] = True
    if signature[3]:
        mask[r0 + h1, c1 + 1] = True
    if signature[4]:
        mask[r1 + 1, c0 + w1] = True
    if signature[5]:
        mask[r1 + 1, c0 + w0] = True
    if signature[6]:
        mask[r0 + h1, c0 - 1] = True
    if signature[7]:
        mask[r0 + h0, c0 - 1] = True
    return mask


def _is_safe_signature(signature: tuple[int, ...]) -> bool:
    for dims in ((2, 2), (4, 4), (5, 6)):
        mask = _build_object_mask(dims[0], dims[1], signature)
        try:
            recovered = _extract_signature(mask)
        except Exception:
            return False
        if recovered != signature:
            return False
    return True


SAFE_SIGNATURES = [sig for sig in SIGNATURES if _is_safe_signature(sig)]


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _regions(height: int, width: int, layout: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if layout == "labels_right":
        shape_region = (GLOBAL_MARGIN, GLOBAL_MARGIN, height - GLOBAL_MARGIN, width - LABEL_STRIP - REGION_GAP)
        label_region = (GLOBAL_MARGIN, width - LABEL_STRIP, height - GLOBAL_MARGIN, width - GLOBAL_MARGIN)
    elif layout == "labels_left":
        shape_region = (GLOBAL_MARGIN, LABEL_STRIP + REGION_GAP, height - GLOBAL_MARGIN, width - GLOBAL_MARGIN)
        label_region = (GLOBAL_MARGIN, GLOBAL_MARGIN, height - GLOBAL_MARGIN, LABEL_STRIP)
    elif layout == "labels_bottom":
        shape_region = (GLOBAL_MARGIN, GLOBAL_MARGIN, height - LABEL_STRIP - REGION_GAP, width - GLOBAL_MARGIN)
        label_region = (height - LABEL_STRIP, GLOBAL_MARGIN, height - GLOBAL_MARGIN, width - GLOBAL_MARGIN)
    else:
        shape_region = (LABEL_STRIP + REGION_GAP, GLOBAL_MARGIN, height - GLOBAL_MARGIN, width - GLOBAL_MARGIN)
        label_region = (GLOBAL_MARGIN, GLOBAL_MARGIN, LABEL_STRIP, width - GLOBAL_MARGIN)
    return shape_region, label_region


def _place_mask(blocked: np.ndarray, region: tuple[int, int, int, int], mask: np.ndarray) -> tuple[int, int] | None:
    h, w = mask.shape
    r0, c0, r1, c1 = region
    max_top = r1 - h
    max_left = c1 - w
    if max_top < r0 or max_left < c0:
        return None
    for _ in range(PLACEMENT_RETRIES):
        top = int(_RNG.integers(r0, max_top + 1))
        left = int(_RNG.integers(c0, max_left + 1))
        rr0 = max(0, top - 1)
        cc0 = max(0, left - 1)
        rr1 = min(blocked.shape[0], top + h + 1)
        cc1 = min(blocked.shape[1], left + w + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        blocked[rr0:rr1, cc0:cc1] = True
        return top, left
    return None


def _expected_output(shape_specs: list[dict], shape_positions: list[tuple[int, int]], shape_colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec, (top, left), color in zip(shape_specs, shape_positions, shape_colors):
        _draw_mask(out, spec["mask"], top, left, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey blobs to distant color tags by their boundary-tooth signatures, then recolor the blobs.

    Input structure:
    - A black background contains several large grey monochrome components and several smaller colored tag components.
    - Each grey component is a filled orthogonal blob whose boundary includes a distinctive set of one-cell outward teeth.
    - Each colored tag is a compact monochrome block with the same tooth signature as exactly one grey blob.

    Rule:
    - Split the non-background pixels into monochrome 4-connected components.
    - Recover each component's filled rectangular core and read the one-cell teeth around that core in clockwise order.
    - Pair every grey blob with the unique colored tag that has the same boundary signature.
    - Recolor each grey blob, including its teeth, with the paired tag color, and remove all tags from the output.

    Key insight:
    - The shape identity is not carried by the fill color or by the tag position; it is encoded by the tiny boundary teeth pattern shared by a grey blob and its matching color tag.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected at least one foreground color"
    grey_color = max(colors, key=lambda c: int(np.count_nonzero(g == c)))

    shape_components = _components_of_color(g, grey_color)
    assert shape_components, "expected at least one grey shape"

    label_by_signature: dict[tuple[int, ...], int] = {}
    for color in colors:
        if color == grey_color:
            continue
        for comp in _components_of_color(g, color):
            sig = _extract_signature(comp["mask"])
            assert sig not in label_by_signature, "label signatures must be unique"
            label_by_signature[sig] = int(color)

    out = np.full_like(g, bg)
    for comp in shape_components:
        sig = _extract_signature(comp["mask"])
        assert sig in label_by_signature, "missing matching label"
        color = label_by_signature[sig]
        for r, c in comp["coords"]:
            out[r, c] = color
    return out


def _max_pair_count(height: int, width: int, layout: str) -> int:
    if layout in ("labels_left", "labels_right"):
        if height >= 26:
            return 4
        return 3
    if width >= 26:
        return 4
    return 3


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(*GRID_H_RANGE))
    width = int(_RNG.integers(*GRID_W_RANGE))
    layout = str(_RNG.choice(LAYOUTS))
    shape_region, label_region = _regions(height, width, layout)
    pair_count = int(_RNG.integers(2, _max_pair_count(height, width, layout) + 1))
    signatures = list(_RNG.choice(np.array(SAFE_SIGNATURES, dtype=int), size=pair_count, replace=False))
    signatures = [tuple(int(x) for x in row) for row in signatures]
    label_colors = list(_RNG.choice(np.array(COLOR_POOL, dtype=int), size=pair_count, replace=False))
    label_colors = [int(c) for c in label_colors]

    shape_specs = []
    label_specs = []
    for sig in signatures:
        core_h = int(_RNG.integers(*SHAPE_CORE_RANGE))
        core_w = int(_RNG.integers(*SHAPE_CORE_RANGE))
        shape_mask = _build_object_mask(core_h, core_w, sig)
        label_mask = _build_object_mask(LABEL_CORE, LABEL_CORE, sig)
        shape_specs.append({"signature": sig, "mask": shape_mask})
        label_specs.append({"signature": sig, "mask": label_mask})

    shape_specs.sort(key=lambda item: -int(item["mask"].sum()))
    label_specs.sort(key=lambda item: -int(item["mask"].sum()))

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    shape_positions = []
    shape_colors = []
    for spec in shape_specs:
        pos = _place_mask(blocked, shape_region, spec["mask"])
        if pos is None:
            return None
        shape_positions.append(pos)
        color = label_colors[signatures.index(spec["signature"])]
        shape_colors.append(color)
        _draw_mask(grid, spec["mask"], pos[0], pos[1], GREY)

    for spec in label_specs:
        pos = _place_mask(blocked, label_region, spec["mask"])
        if pos is None:
            return None
        color = label_colors[signatures.index(spec["signature"])]
        _draw_mask(grid, spec["mask"], pos[0], pos[1], color)

    if _most_common_color(grid) != BG:
        return None

    grey_components = _components_of_color(grid, GREY)
    if len(grey_components) != pair_count:
        return None
    if len({_extract_signature(comp["mask"]) for comp in grey_components}) != pair_count:
        return None

    expected = _expected_output(shape_specs, shape_positions, shape_colors, grid.shape)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
