from __future__ import annotations

import numpy as np


BACKGROUND = 0
NONZERO_COLORS = np.arange(1, 10, dtype=int)
GENERATION_RETRIES = 400
NEIGHBOR_COPY_PROB = 0.58
PATTERN_SIDE_CHOICES = tuple(range(11, 20, 2))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _orbit_positions(r: int, c: int, center_r: int, center_c: int) -> dict[str, tuple[int, int]]:
    return {
        "V": (int(r), int(2 * center_c - c)),
        "H": (int(2 * center_r - r), int(c)),
        "P": (int(2 * center_r - r), int(2 * center_c - c)),
    }


def _apply_orbit_color(patch: np.ndarray, r: int, c: int, color: int) -> None:
    center_r = patch.shape[0] // 2
    center_c = patch.shape[1] // 2
    patch[r, c] = int(color)
    for rr, cc in _orbit_positions(r, c, center_r, center_c).values():
        patch[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair both black holes by inferring the hidden point center and vertical mirror axis.

    Input structure:
    - A black background surrounds one multicolor pattern patch whose visible bounding box has odd height and odd width.
    - The intact patch is mirror-symmetric about one hidden vertical axis and point-symmetric about the hidden center cell on that axis.
    - Two solid black rectangular holes lie strictly inside that patch and may occlude one another's one-step symmetry mates.

    Rule:
    - Find the bounding box of all visible non-black cells; its midpoint gives the hidden center row and hidden vertical-axis column, and therefore the derived horizontal axis.
    - Treat every black cell inside that bounding box as missing. For each missing cell, inspect its orbit under vertical reflection, horizontal reflection, and 180-degree rotation about the hidden center.
    - Copy the unique visible non-black color already present elsewhere in that orbit into the missing cell, leaving the outer black background unchanged.

    Key insight:
    - Some missing cells do not have a usable one-step mate: one symmetry image can stay inside the same hole while another lands in the second hole, so the only visible witness appears after composing the hidden symmetries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    box = _bbox(g != BACKGROUND)
    assert box is not None, "expected a visible pattern"
    r0, c0, r1, c1 = box
    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "pattern bbox must be odd-sized"

    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r, c] != BACKGROUND:
                continue
            values = set()
            for rr, cc in ((r, c), *tuple(_orbit_positions(r, c, center_r, center_c).values())):
                assert r0 <= rr <= r1 and c0 <= cc <= c1, "orbit escaped pattern bbox"
                val = int(g[rr, cc])
                if val != BACKGROUND:
                    values.add(val)
            assert len(values) == 1, "every missing orbit must expose exactly one visible color"
            out[r, c] = values.pop()

    return out


def _make_symmetric_patch(height: int, width: int) -> np.ndarray:
    center_r = height // 2
    center_c = width // 2
    domain = [(r, c) for r in range(center_r + 1) for c in range(center_c + 1)]

    while True:
        patch = np.zeros((height, width), dtype=int)
        palette_size = int(np.random.randint(4, 8))
        palette = np.random.choice(NONZERO_COLORS, size=palette_size, replace=False)
        chosen: dict[tuple[int, int], int] = {}

        for r, c in sorted(domain, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1])):
            neighbor_colors: list[int] = []
            for nr, nc in ((r - 1, c), (r, c - 1), (r - 1, c - 1), (r - 1, c + 1)):
                if (nr, nc) in chosen:
                    neighbor_colors.append(chosen[(nr, nc)])
            if neighbor_colors and np.random.random() < NEIGHBOR_COPY_PROB:
                color = int(neighbor_colors[int(np.random.randint(len(neighbor_colors)))])
            else:
                color = int(palette[int(np.random.randint(len(palette)))])
            chosen[(r, c)] = color
            _apply_orbit_color(patch, r, c, color)

        unique = np.unique(patch)
        center_row_colors = np.unique(patch[center_r, :])
        center_col_colors = np.unique(patch[:, center_c])
        if len(unique) >= 4 and len(center_row_colors) >= 2 and len(center_col_colors) >= 2:
            return patch


def _rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _build_holes(height: int, width: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    center_r = height // 2
    center_c = width // 2
    half_width_options = [half for half in (2, 3) if center_c - half >= 1 and center_c + half <= width - 2]
    assert half_width_options

    half_w = int(half_width_options[int(np.random.randint(len(half_width_options)))])
    hole_h = int(np.random.randint(2, min(4, center_r - 1) + 1))
    top_half = bool(np.random.randint(2))

    if top_half:
        r0 = int(np.random.randint(1, center_r - hole_h + 1))
    else:
        r0 = int(np.random.randint(center_r + 1, height - hole_h))
    r1 = r0 + hole_h - 1

    hole_a = (r0, center_c - half_w, r1, center_c + half_w)
    mirror_r0 = 2 * center_r - r1
    mirror_r1 = 2 * center_r - r0

    side = "left" if bool(np.random.randint(2)) else "right"
    hole_b_w = int(np.random.randint(2, half_w + 1))
    if side == "left":
        hole_b = (mirror_r0, center_c - hole_b_w, mirror_r1, center_c - 1)
    else:
        hole_b = (mirror_r0, center_c + 1, mirror_r1, center_c + hole_b_w)

    return hole_a, hole_b


def _hole_categories(
    shape: tuple[int, int],
    hole_a: tuple[int, int, int, int],
    hole_b: tuple[int, int, int, int],
) -> dict[str, object]:
    height, width = shape
    center_r = height // 2
    center_c = width // 2
    hole_id = np.zeros(shape, dtype=int)
    hole_id[_rect_mask(shape, hole_a)] = 1
    hole_id[_rect_mask(shape, hole_b)] = 2

    counts: dict[frozenset[str], int] = {}
    composite_ok = False
    seen_cross_from = {1: False, 2: False}
    all_have_visible = True

    for r, c in np.argwhere(hole_id > 0):
        r = int(r)
        c = int(c)
        here = int(hole_id[r, c])
        visible: set[str] = set()
        hidden_other: set[str] = set()

        for name, (rr, cc) in _orbit_positions(r, c, center_r, center_c).items():
            if (rr, cc) == (r, c):
                continue
            assert 0 <= rr < height and 0 <= cc < width
            other = int(hole_id[rr, cc])
            if other == 0:
                visible.add(name)
            else:
                if other != here:
                    hidden_other.add(name)
                    seen_cross_from[here] = True

        if not visible:
            all_have_visible = False
        key = frozenset(visible)
        counts[key] = counts.get(key, 0) + 1
        if key == frozenset({"H"}) and hidden_other:
            composite_ok = True

    return {
        "counts": counts,
        "all_have_visible": all_have_visible,
        "composite_ok": composite_ok,
        "mutual_cross": seen_cross_from[1] and seen_cross_from[2],
    }


def _embed_patch(patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = patch.shape
    while True:
        top = int(np.random.randint(1, 6))
        bottom = int(np.random.randint(1, 6))
        left = int(np.random.randint(1, 6))
        right = int(np.random.randint(1, 6))
        canvas_h = height + top + bottom
        canvas_w = width + left + right
        if canvas_h <= 30 and canvas_w <= 30 and top != bottom and left != right:
            canvas = np.zeros((canvas_h, canvas_w), dtype=int)
            canvas[top : top + height, left : left + width] = patch
            return canvas, (top, left)


def _local_to_global(
    box: tuple[int, int, int, int], offset: tuple[int, int]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    dr, dc = offset
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _zero_rect(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND


def _attempt() -> np.ndarray | None:
    patch_h = int(PATTERN_SIDE_CHOICES[int(np.random.randint(len(PATTERN_SIDE_CHOICES)))])
    patch_w = int(PATTERN_SIDE_CHOICES[int(np.random.randint(len(PATTERN_SIDE_CHOICES)))])
    patch = _make_symmetric_patch(patch_h, patch_w)
    hole_a, hole_b = _build_holes(patch_h, patch_w)

    categories = _hole_categories((patch_h, patch_w), hole_a, hole_b)
    counts = categories["counts"]
    if not categories["all_have_visible"]:
        return None
    if not categories["composite_ok"] or not categories["mutual_cross"]:
        return None
    if counts.get(frozenset({"H"}), 0) == 0:
        return None
    if counts.get(frozenset({"P"}), 0) == 0:
        return None
    if counts.get(frozenset({"V"}), 0) == 0:
        return None

    full, offset = _embed_patch(patch)
    inp = full.copy()
    _zero_rect(inp, _local_to_global(hole_a, offset))
    _zero_rect(inp, _local_to_global(hole_b, offset))

    top, left = offset
    center_r = top + patch_h // 2
    center_c = left + patch_w // 2
    if center_r == inp.shape[0] // 2 or center_c == inp.shape[1] // 2:
        return None

    repaired = main(inp)
    if not np.array_equal(repaired, full):
        return None

    support_box = _bbox(inp != BACKGROUND)
    if support_box != (top, left, top + patch_h - 1, left + patch_w - 1):
        return None

    # Semantic validation: every missing cell inside the support box must recover
    # from a unique visible color in its symmetry orbit.
    for hole_box in (_local_to_global(hole_a, offset), _local_to_global(hole_b, offset)):
        r0, c0, r1, c1 = hole_box
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                values = set()
                for rr, cc in ((r, c), *tuple(_orbit_positions(r, c, center_r, center_c).values())):
                    val = int(inp[rr, cc])
                    if val != BACKGROUND:
                        values.add(val)
                if values != {int(full[r, c])}:
                    return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-symmetry dual-hole instance")
