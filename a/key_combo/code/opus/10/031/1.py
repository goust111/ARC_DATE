from __future__ import annotations

import numpy as np


BG = 0
SEP = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
SYMMETRY_TYPES = ("vertical", "horizontal", "diagonal", "anti_diagonal")
_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _reflect_anti_diag(mask: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(mask)).T


def _panel_symmetries(panel: np.ndarray) -> set[str]:
    mask = _crop_bool(panel != BG)
    if mask.size == 0:
        return set()

    kinds: set[str] = set()
    if np.array_equal(mask, np.fliplr(mask)):
        kinds.add("vertical")
    if np.array_equal(mask, np.flipud(mask)):
        kinds.add("horizontal")
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            kinds.add("diagonal")
        if np.array_equal(mask, _reflect_anti_diag(mask)):
            kinds.add("anti_diagonal")
    return kinds


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    sep_cols = _separator_columns(grid)
    cuts = [-1] + sep_cols + [grid.shape[1]]
    panels = [grid[:, cuts[i] + 1 : cuts[i + 1]] for i in range(len(cuts) - 1)]
    widths = {panel.shape[1] for panel in panels}
    assert len(widths) == 1, "panels must be equal width"
    return panels


def _compose_survivors(panels: list[np.ndarray]) -> np.ndarray:
    survivors = []
    for panel in panels:
        symmetries = _panel_symmetries(panel)
        if symmetries:
            survivors.append((int(np.count_nonzero(panel != BG)), panel))

    assert survivors, "expected at least one symmetric panel"
    out = np.zeros_like(survivors[0][1])
    for _count, panel in sorted(survivors, key=lambda item: item[0]):
        out = np.where(panel != BG, panel, out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter symmetric panels and composite the survivors from sparse to dense.

    Input structure:
    - The grid has a black background with 4 or 5 equal-width color panels laid
      out horizontally.
    - Single-column red separators split the panels; each panel contains one
      monochrome non-red pattern on black, where black is transparent.
    - Some panel patterns are reflection-symmetric after cropping to their
      foreground bbox, while the others are not.

    Rule:
    - Split the grid at the full-height red separator columns.
    - Keep only panels whose foreground pattern has at least one reflection
      symmetry: vertical, horizontal, main-diagonal, or anti-diagonal.
    - Count the non-black pixels in each surviving panel, sort survivors from
      fewest pixels to most pixels, then transparently overlay them in that order.

    Key insight:
    - Survival depends on geometric symmetry, but depth depends on visible pixel
      count, so both properties must be combined before the final overlay.
    """
    g = np.asarray(grid, dtype=int)
    panels = _split_panels(g)
    return _compose_survivors(panels)


def _symmetrize(mask: np.ndarray, sym_type: str) -> np.ndarray:
    if sym_type == "vertical":
        return mask | np.fliplr(mask)
    if sym_type == "horizontal":
        return mask | np.flipud(mask)
    if sym_type == "diagonal":
        return mask | mask.T
    assert sym_type == "anti_diagonal"
    return mask | _reflect_anti_diag(mask)


def _place_local(local: np.ndarray, panel_h: int, panel_w: int) -> np.ndarray:
    out = np.zeros((panel_h, panel_w), dtype=int)
    h, w = local.shape
    r0 = int(_RNG.integers(0, panel_h - h + 1))
    c0 = int(_RNG.integers(0, panel_w - w + 1))
    out[r0 : r0 + h, c0 : c0 + w] = local.astype(int)
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(_RNG.integers(0, h)), int(_RNG.integers(0, w)))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = list(frontier)[int(_RNG.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return mask


def _make_symmetric_panel(panel_h: int, panel_w: int, sym_type: str) -> np.ndarray:
    for _ in range(300):
        if sym_type in ("diagonal", "anti_diagonal"):
            side = int(_RNG.integers(3, min(panel_h, panel_w) + 1))
            base = _RNG.random((side, side)) < _RNG.uniform(0.18, 0.34)
        else:
            h = int(_RNG.integers(3, panel_h + 1))
            w = int(_RNG.integers(3, panel_w + 1))
            base = _RNG.random((h, w)) < _RNG.uniform(0.18, 0.34)

        if not np.any(base):
            continue
        local = _crop_bool(_symmetrize(base, sym_type))
        area = int(local.sum())
        if area < 3 or area > min(panel_h * panel_w - 1, 12):
            continue
        if not _panel_symmetries(local.astype(int)):
            continue
        return _place_local(local, panel_h, panel_w)
    raise RuntimeError(f"failed to build symmetric panel: {sym_type}")


def _make_nonsymmetric_panel(panel_h: int, panel_w: int) -> np.ndarray:
    for _ in range(400):
        h = int(_RNG.integers(3, panel_h + 1))
        w = int(_RNG.integers(3, panel_w + 1))
        max_area = min(h * w - 1, 12)
        if max_area < 4:
            continue
        area = int(_RNG.integers(4, max_area + 1))
        local = _crop_bool(_grow_connected_mask(h, w, area))
        if local.size == 0:
            continue
        if int(local.sum()) != area:
            continue
        if _panel_symmetries(local.astype(int)):
            continue
        return _place_local(local, panel_h, panel_w)
    raise RuntimeError("failed to build non-symmetric panel")


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.zeros_like(mask, dtype=int)
    panel[mask.astype(bool)] = int(color)
    return panel


def _render_input(panels: list[np.ndarray]) -> np.ndarray:
    sep = np.full((panels[0].shape[0], 1), SEP, dtype=int)
    pieces = []
    for i, panel in enumerate(panels):
        pieces.append(panel)
        if i + 1 < len(panels):
            pieces.append(sep)
    return np.concatenate(pieces, axis=1)


def _analyze_scene(grid: np.ndarray) -> dict:
    panels = _split_panels(grid)
    info = []
    for idx, panel in enumerate(panels):
        fg = sorted(int(v) for v in np.unique(panel) if v not in (BG, SEP))
        assert len(fg) == 1, "each panel must use exactly one non-red foreground color"
        info.append(
            {
                "index": idx,
                "panel": panel,
                "color": fg[0],
                "count": int(np.count_nonzero(panel != BG)),
                "symmetries": _panel_symmetries(panel),
                "mask": panel != BG,
            }
        )

    survivors = [item for item in info if item["symmetries"]]
    nonsurvivors = [item for item in info if not item["symmetries"]]
    expected = _compose_survivors(panels)

    return {
        "panels": info,
        "survivors": survivors,
        "nonsurvivors": nonsurvivors,
        "expected": expected,
    }


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_scene(grid)
    except AssertionError:
        return False

    panels = analysis["panels"]
    survivors = analysis["survivors"]
    nonsurvivors = analysis["nonsurvivors"]
    out = main(grid)

    if not (4 <= len(panels) <= 5):
        return False
    if len(survivors) not in (2, 3):
        return False
    if len({item["color"] for item in panels}) != len(panels):
        return False
    if len({item["count"] for item in panels}) != len(panels):
        return False
    if not np.array_equal(out, analysis["expected"]):
        return False

    sym_counts = sorted(item["count"] for item in survivors)
    nonsym_counts = sorted(item["count"] for item in nonsurvivors)
    if not nonsym_counts:
        return False
    if len(nonsym_counts) == 1:
        if not (sym_counts[0] < nonsym_counts[0] < sym_counts[-1]):
            return False
    else:
        if not (min(nonsym_counts) < max(sym_counts) and max(nonsym_counts) > min(sym_counts)):
            return False

    stack = np.stack([item["mask"] for item in survivors], axis=0)
    overlap = np.sum(stack, axis=0) >= 2
    union = np.any(stack, axis=0)
    overlap_count = int(overlap.sum())
    union_count = int(union.sum())
    panel_area = int(union.size)
    if overlap_count == 0:
        return False
    if overlap_count > max(1, panel_area // 3):
        return False
    if union_count > int(panel_area * 0.75):
        return False

    for item in survivors:
        if not np.any(out == item["color"]):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    panel_count = int(_RNG.choice([4, 5]))
    panel_w = int(_RNG.choice([4, 5]))
    grid_w = panel_count * panel_w + (panel_count - 1)
    if grid_w > 30:
        return None
    panel_h = int(_RNG.integers(6, 9))

    sym_count = 2 if panel_count == 4 else int(_RNG.choice([2, 3]))
    colors = list(_RNG.choice(PALETTE, size=panel_count, replace=False))
    sym_indices = set(_RNG.choice(panel_count, size=sym_count, replace=False).tolist())

    sym_types = list(_RNG.choice(SYMMETRY_TYPES, size=sym_count, replace=False if sym_count <= 4 else True))
    _RNG.shuffle(sym_types)

    raw_masks: list[np.ndarray] = []
    sym_ptr = 0
    for idx in range(panel_count):
        if idx in sym_indices:
            sym_type = sym_types[sym_ptr]
            sym_ptr += 1
            raw_masks.append(_make_symmetric_panel(panel_h, panel_w, sym_type))
        else:
            raw_masks.append(_make_nonsymmetric_panel(panel_h, panel_w))

    counts = [int(mask.sum()) for mask in raw_masks]
    if len(set(counts)) != panel_count:
        return None

    panels = [_colorize(mask, color) for mask, color in zip(raw_masks, colors)]
    grid = _render_input(panels)
    return grid if _semantic_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-filter overlay scene")
